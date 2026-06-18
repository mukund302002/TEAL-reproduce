"""
TopK activation-sparsity utilities for LLaMA — exact-sparsity variant.

Difference from TEAL (teal_utils.py):
  TEAL   : stores a fixed threshold derived from calibration histograms.
            At inference the actual zeroed fraction depends on the runtime
            activation distribution → calibration sparsity ≠ inference sparsity.

  TopK   : no pre-computed histograms needed.  At every forward call the
            threshold is computed dynamically as
                thr = torch.quantile(|x|, sparsity)
            so exactly `sparsity` fraction of activations are zeroed,
            regardless of the data distribution.

Sparsification sites (identical to teal_utils.py):
  MLP  : h_in → [zero bottom-s% of |h_in|] → gate_proj + up_proj
          silu(gate) * up → [zero bottom-s% of |intermediate|] → down_proj
  Attn : hidden_states → [zero bottom-s% of |h|] → q/k/v_proj
"""

import types
from typing import Dict, Optional

import torch
import torch.nn as nn
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Exact-sparsity sparse-mask module
# ---------------------------------------------------------------------------

class TopKSparsifyFn(nn.Module):
    """
    Dynamic magnitude gate: zeros the bottom `sparsity` fraction of |x|.

    Unlike TealSparsifyFn (fixed threshold), the threshold is recomputed
    from each input tensor, guaranteeing exact target sparsity per call.

    topk_mode:
      "global"    — ek threshold poore [B*S*H] tensor ke liye (default)
      "per_token" — har token ke liye alag topk along hidden dim
    """

    def __init__(self, sparsity: float = 0.0, topk_mode: str = "global"):
        super().__init__()
        self.sparsity  = sparsity    # fraction in [0, 1)
        self.topk_mode = topk_mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.sparsity <= 0.0:
            return x

        if self.topk_mode == "global":
            # Ek threshold poore [B*S*H] tensor ke liye
            abs_flat = x.abs().float().flatten()
            k   = max(1, int(self.sparsity * abs_flat.numel()))
            thr = torch.topk(abs_flat, k, largest=False).values[-1]
            return x * x.abs().gt(thr)

        else:  # per_token
            # Har token independently: [B*S, H] → topk along hidden dim
            orig_shape = x.shape
            x_2d   = x.reshape(-1, x.shape[-1])                          # [B*S, H]
            k_keep = max(1, int((1 - self.sparsity) * x_2d.shape[-1]))   # rakhne wale
            _, idx = x_2d.abs().topk(k_keep, dim=-1)
            mask   = torch.zeros_like(x_2d).scatter(-1, idx, 1)
            return (x_2d * mask).reshape(orig_shape)


# ---------------------------------------------------------------------------
# MLP patch
# ---------------------------------------------------------------------------

def _topk_mlp_forward(self, x):
    """
    Replacement forward for LlamaMLP applying TopK magnitude masking:
      - before gate_proj and up_proj  (h1 sparsity)
      - before down_proj on the SwiGLU intermediate  (h2 sparsity)
    """
    x_gate       = self.topk_sparse_h1(x)
    x_up         = self.topk_sparse_h1(x)
    intermediate = self.act_fn(self.gate_proj(x_gate)) * self.up_proj(x_up)
    intermediate = self.topk_sparse_h2(intermediate)
    return self.down_proj(intermediate)


def patch_mlp_topk(mlp: nn.Module, sparsity: float):
    """Attach TopK sparse_fns to an MLP module and replace its forward."""
    mlp.topk_sparse_h1 = TopKSparsifyFn(sparsity)
    mlp.topk_sparse_h2 = TopKSparsifyFn(sparsity)
    mlp._topk_forward_original = mlp.forward
    mlp.forward = types.MethodType(_topk_mlp_forward, mlp)


# ---------------------------------------------------------------------------
# Attention patch
# ---------------------------------------------------------------------------

def _topk_attn_forward(self, hidden_states, **kwargs):
    """
    Replacement forward for LlamaAttention applying TopK masking on
    hidden_states before q/k/v projections.
    """
    h_masked = self.topk_sparse_h1(hidden_states)
    return self._topk_attn_original(h_masked, **kwargs)


def patch_attn_topk(attn: nn.Module, sparsity: float):
    """Attach TopK h1 sparse_fn to an attention module and replace its forward."""
    attn.topk_sparse_h1 = TopKSparsifyFn(sparsity)
    attn._topk_attn_original = attn.forward
    attn.forward = types.MethodType(_topk_attn_forward, attn)


# ---------------------------------------------------------------------------
# Model-level application  (no histograms required)
# ---------------------------------------------------------------------------

def apply_topk_to_model(
    model,
    sparsity: float,
    num_layers: Optional[int] = None,
    patch_attn: bool = True,
) -> Dict[int, float]:
    """
    Patch every MLP (and optionally attention) of a LLaMA-style model with
    TopK dynamic magnitude masking.

    Parameters
    ----------
    model : AutoModelForCausalLM
        A LLaMA-style HuggingFace model (weights untouched).
    sparsity : float
        Target activation sparsity, e.g. 0.5 for 50%.
        Exactly this fraction of activations is zeroed per call.
    num_layers : int, optional
        Number of layers to patch (default: all).
    patch_attn : bool
        Whether to also patch attention modules (default: True).

    Returns
    -------
    Dict mapping layer index → applied sparsity value (same for all layers).
    """
    if num_layers is None:
        num_layers = len(model.model.layers)

    for i in tqdm(range(num_layers), desc="Applying TopK patches"):
        layer = model.model.layers[i]
        patch_mlp_topk(layer.mlp, sparsity)
        if patch_attn:
            patch_attn_topk(layer.self_attn, sparsity)

    return {i: sparsity for i in range(num_layers)}


# ---------------------------------------------------------------------------
# LlamaSparseForCausalLM support  (replaces SparsifyFn in sparse_fns dict)
# ---------------------------------------------------------------------------

def apply_topk_to_sparse_model(model, sparsity: float, lookup_path: str = None, topk_mode: str = "global"):
    """
    For LlamaSparseForCausalLM: replace every SparsifyFn in sparse_fns with
    TopKSparsifyFn so that exact-sparsity masking is used at inference.

    If lookup_path is provided, per-layer per-projection sparsity values are
    loaded from the greedy calibration CSVs (same distribution as TEAL greedy).
    Otherwise, uniform sparsity is applied to all layers and projections.

    Original SparsifyFn objects are stored for later restoration.
    """
    # Load per-layer per-projection sparsity values from greedy lookup if provided
    if lookup_path is not None:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from utils.utils import get_layer_greedy_sparsities
        num_layers = len(model.model.layers)
        layer_sparsity_levels = [sparsity] * num_layers
        greedy_sparsities = get_layer_greedy_sparsities(layer_sparsity_levels, lookup_path)
    else:
        greedy_sparsities = None

    for i, layer in enumerate(model.model.layers):
        mlp  = layer.mlp
        attn = layer.self_attn

        # Save originals
        mlp._orig_sparse_fns  = {k: v for k, v in mlp.sparse_fns.items()}
        attn._orig_sparse_fns = {k: v for k, v in attn.sparse_fns.items()}

        device = next(mlp.parameters()).device
        for proj in list(mlp.sparse_fns.keys()):
            sp = greedy_sparsities[proj][i] if greedy_sparsities is not None else sparsity
            mlp.sparse_fns[proj] = TopKSparsifyFn(sp, topk_mode).to(device)

        device = next(attn.parameters()).device
        for proj in list(attn.sparse_fns.keys()):
            sp = greedy_sparsities[proj][i] if greedy_sparsities is not None else sparsity
            attn.sparse_fns[proj] = TopKSparsifyFn(sp, topk_mode).to(device)


def remove_topk_from_sparse_model(model):
    """Restore original SparsifyFn objects saved by apply_topk_to_sparse_model."""
    for layer in model.model.layers:
        mlp  = layer.mlp
        attn = layer.self_attn

        if hasattr(mlp, "_orig_sparse_fns"):
            for proj, sfn in mlp._orig_sparse_fns.items():
                mlp.sparse_fns[proj] = sfn
            del mlp._orig_sparse_fns

        if hasattr(attn, "_orig_sparse_fns"):
            for proj, sfn in attn._orig_sparse_fns.items():
                attn.sparse_fns[proj] = sfn
            del attn._orig_sparse_fns
