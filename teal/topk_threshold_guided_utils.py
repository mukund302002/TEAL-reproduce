"""
Threshold-Guided TopK sparsification for LLaMA — experiment variant.

Kaise kaam karta hai (har forward call pe):
  1. s1 measure karo: calibration threshold se kitne elements naturally zero hote hain
       s1 = (|x| < threshold).mean()        -- koi sort nahi, sirf ek comparison+reduce
  2. s2 se compare karo (greedy calibration sparsity):
       s1 > s2 aur (s1-s2) > CLIP_MARGIN  →  final = s2 + CLIP_MARGIN
       s1 > s2 aur (s1-s2) <= CLIP_MARGIN  →  final = s2
       s1 < s2 aur (s2-s1) > CLIP_MARGIN  →  final = s2 - CLIP_MARGIN
       s1 < s2 aur (s2-s1) <= CLIP_MARGIN  →  final = s2
  3. TopK with final sparsity (exact zeroing, same as topk_utils.py)

s2 hamesha anchor hai. Threshold sirf direction signal hai (+1% / 0 / -1%).

CLIP_MARGIN = 0.01 (1%)

Note: ye file sirf LlamaSparseForCausalLM ke liye hai (sparse_fns wala model)
      kyunki calibration threshold SparsifyFn.a se aata hai.
      Existing code (topk_utils.py, mlp.py, self_attn.py) bilkul untouched hai.
"""

import types
from typing import Dict, Optional

import torch
import torch.nn as nn
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CLIP_MARGIN = 0.015   # 1% adjustment window


# ---------------------------------------------------------------------------
# Core sparsify module
# ---------------------------------------------------------------------------

class ThresholdGuidedTopKSparsifyFn(nn.Module):
    """
    Threshold-guided TopK magnitude gate.

    calibration threshold  →  s1 measure karo (no sort)
    s2 (calib topk sparsity) →  final sparsity decide karo (s2 ± clip_margin ya s1)
    TopK with final sparsity →  exact zeroing

    Args:
        threshold   : float — TEAL ke SparsifyFn.threshold se liya calibration threshold
        s2          : float — greedy calibration se TopK sparsity (0 to 1)
        clip_margin : float — adjustment window (default: CLIP_MARGIN)
    """

    def __init__(self, threshold: float, s2: float, clip_margin: float = CLIP_MARGIN,
                 topk_mode: str = "global"):
        super().__init__()
        self.s2 = s2
        self.clip_margin = clip_margin
        self.topk_mode   = topk_mode
        # threshold ko buffer ke roop mein store karo (model save/load ke saath aayega)
        self.register_buffer(
            "threshold", torch.tensor(threshold, dtype=torch.float32)
        )

    # ------------------------------------------------------------------
    # Internal: case logic
    # ------------------------------------------------------------------

    def _final_sparsity(self, abs_flat: torch.Tensor) -> float:
        """
        Step 1 — s1 compute karo: O(n), no sort
        Step 2 — case logic:

          s1 > s2 & diff >  clip_margin  →  s2 + clip_margin
          s1 > s2 & diff <= clip_margin  →  s1
          s1 < s2 & diff <= clip_margin  →  s1
          s1 < s2 & diff >  clip_margin  →  s2 - clip_margin
        """
        s1   = (abs_flat < self.threshold).float().mean().item()
        s2   = self.s2
        diff = abs(s1 - s2)
        cm   = self.clip_margin

        if s1 > s2:
            return s2 + cm if diff > cm else s1
        else:                          # s1 <= s2
            return s2 - cm if diff > cm else s1

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.s2 <= 0.0:
            return x

        # s1 hamesha global flatten pe compute hota hai (clip logic ke liye)
        abs_flat       = x.abs().float().flatten()          # [B*S*H]
        final_sparsity = self._final_sparsity(abs_flat)

        if final_sparsity <= 0.0:
            return x

        if self.topk_mode == "global":
            # Ek threshold poore tensor ke liye
            k   = max(1, int(final_sparsity * abs_flat.numel()))
            thr = torch.topk(abs_flat, k, largest=False).values[-1]
            return x * x.abs().gt(thr)

        else:  # per_token
            # Har token independently: [B*S, H] → topk along hidden dim
            orig_shape = x.shape
            x_2d   = x.reshape(-1, x.shape[-1])                              # [B*S, H]
            k_keep = max(1, int((1 - final_sparsity) * x_2d.shape[-1]))      # rakhne wale
            _, idx = x_2d.abs().topk(k_keep, dim=-1)
            mask   = torch.zeros_like(x_2d).scatter(-1, idx, 1)
            return (x_2d * mask).reshape(orig_shape)


# ---------------------------------------------------------------------------
# Apply to LlamaSparseForCausalLM  (sparse_fns wala model)
# ---------------------------------------------------------------------------

def apply_threshold_guided_topk_to_sparse_model(
    model,
    sparsity: float,
    lookup_path: str = None,
    clip_margin: float = CLIP_MARGIN,
    topk_mode: str = "global",
):
    """
    LlamaSparseForCausalLM ke har SparsifyFn ko ThresholdGuidedTopKSparsifyFn
    se replace karo.

    - threshold   : existing SparsifyFn.threshold attribute se extract hoga
                    (set_threshold() se set hua actual calibration threshold)
    - s2          : lookup_path diya toh greedy CSV se per-layer per-proj sparsity
                    nahi diya toh uniform `sparsity` value use hogi
    - clip_margin : adjustment window (default: CLIP_MARGIN)

    Original SparsifyFn objects saved rehte hain restore ke liye.

    Parameters
    ----------
    model        : LlamaSparseForCausalLM
    sparsity     : float — target sparsity (e.g. 0.5); greedy lookup ke liye
                   baseline sparsity level
    lookup_path  : str, optional — greedy calibration results ka directory
                   (same as topk_utils.apply_topk_to_sparse_model ka lookup_path)
    clip_margin  : float — adjustment window (default: CLIP_MARGIN = 0.015)
    """
    # Greedy per-layer sparsities load karo agar path diya hai
    if lookup_path is not None:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from utils.utils import get_layer_greedy_sparsities
        num_layers = len(model.model.layers)
        greedy_sparsities = get_layer_greedy_sparsities(
            [sparsity] * num_layers, lookup_path
        )
    else:
        greedy_sparsities = None

    for i, layer in enumerate(tqdm(model.model.layers, desc="Applying ThresholdGuidedTopK")):
        mlp  = layer.mlp
        attn = layer.self_attn

        # Original SparsifyFn save karo (restore ke liye)
        mlp._orig_sparse_fns  = {k: v for k, v in mlp.sparse_fns.items()}
        attn._orig_sparse_fns = {k: v for k, v in attn.sparse_fns.items()}

        # MLP projections replace karo
        device = next(mlp.parameters()).device
        for proj, sfn in list(mlp.sparse_fns.items()):
            thr = sfn.threshold   # set_threshold() se set hua actual calibration threshold
            s2  = greedy_sparsities[proj][i] if greedy_sparsities is not None else sparsity
            mlp.sparse_fns[proj] = ThresholdGuidedTopKSparsifyFn(thr, s2, clip_margin, topk_mode).to(device)

        # Attn projections replace karo
        device = next(attn.parameters()).device
        for proj, sfn in list(attn.sparse_fns.items()):
            thr = sfn.threshold
            s2  = greedy_sparsities[proj][i] if greedy_sparsities is not None else sparsity
            attn.sparse_fns[proj] = ThresholdGuidedTopKSparsifyFn(thr, s2, clip_margin, topk_mode).to(device)


# ---------------------------------------------------------------------------
# Restore original SparsifyFn objects
# ---------------------------------------------------------------------------

def remove_threshold_guided_topk_from_sparse_model(model):
    """
    apply_threshold_guided_topk_to_sparse_model ka reverse.
    Saved SparsifyFn objects wapas restore karo.
    """
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
