"""
Distribution plots: input (h) vs output (y) activations
Layers 8, 16, 24 ke liye.
MLP naming (sirf plots ke liye): mlp h1 → h3, mlp h2 → h4
"""

import sys, os, torch
import matplotlib.pyplot as plt
import numpy as np

sys.path.append('.')
sys.path.append('utils')

from utils.utils import get_tokenizer, get_sparse_model
from teal.model import LlamaSparseForCausalLM, LlamaSparseConfig
from transformers import AutoConfig, AutoModelForCausalLM

AutoConfig.register("llama_sparse", LlamaSparseConfig)
AutoModelForCausalLM.register(LlamaSparseConfig, LlamaSparseForCausalLM)

# ── Model load in grabbing mode ──
print("Loading model...")
tokenizer = get_tokenizer("meta-llama/Llama-2-7b-hf")
model = get_sparse_model(
    "meta-llama/Llama-2-7b-hf",
    device="auto",
    histogram_path="models/Llama-2-7B/histograms",
    grab_acts=True
)

# ── Data load ──
from utils.data import get_dataset
dataset = get_dataset("tatsu-lab/alpaca", subset=None, split="train", size=5)
text = "".join(s["text"] + "\n\n" for s in dataset)

seq_len = 2048
encodings = tokenizer(
    text, truncation=True, return_tensors="pt",
    max_length=seq_len, return_overflowing_tokens=True, padding="max_length"
)
input_ids = encodings.input_ids[:1, :].to("cuda:0")

hidden_states = model.model.embed_tokens(input_ids)
position_ids = torch.arange(seq_len, dtype=torch.long, device=hidden_states.device) \
                    .unsqueeze(0).repeat(input_ids.shape[0], 1)

# ── Layer 8, 16, 24 ke liye activations collect karo ──
TARGET_LAYERS = [8, 16, 24]
collected = {}

print("Running forward pass through layers 0-24...")
for i in range(25):
    layer = model.model.layers[i]
    hidden_states = hidden_states.to(layer.self_attn.q_proj.weight.device)
    hidden_states = layer(hidden_states, None, position_ids, None, False, False, None)[0]

    if i in TARGET_LAYERS:
        def to_np(acts_dict):
            return {
                k: torch.cat(v, dim=0).flatten().float().cpu().numpy()
                for k, v in acts_dict.items()
            }
        collected[i] = {
            'mlp':  to_np(layer.mlp.activation_module.activations),
            'attn': to_np(layer.self_attn.activation_module.activations),
        }
        print(f"  Layer {i} collected.")

# ── Helper: outlier clip ──
def clip(arr):
    lo, hi = np.percentile(arr, 1), np.percentile(arr, 99)
    return arr[(arr >= lo) & (arr <= hi)]


# ════════════════════════════════════════════════════
# PLOT 1: Comparison — h vs y (overlapping)
# attn: h1/h2 labels, mlp: h3/h4 labels
# ════════════════════════════════════════════════════
# (dict_key, display_label, y_key, y_label, module)
comparisons = [
    ('h1', 'h1', 'y1', 'y1', 'attn', 'k_proj\ny1 vs h1'),
    ('h1', 'h1', 'y2', 'y2', 'attn', 'q_proj\ny2 vs h1'),
    ('h1', 'h1', 'y3', 'y3', 'attn', 'v_proj\ny3 vs h1'),
    ('h2', 'h2', 'y4', 'y4', 'attn', 'o_proj\ny4 vs h2'),
    ('h1', 'h3', 'y5', 'y5', 'mlp',  'gate_proj\ny5 vs h3'),
    ('h1', 'h3', 'y6', 'y6', 'mlp',  'up_proj\ny6 vs h3'),
    ('h2', 'h4', 'y7', 'y7', 'mlp',  'down_proj\ny7 vs h4'),
]

fig1, axes1 = plt.subplots(3, 7, figsize=(28, 12))
fig1.suptitle(
    'Comparison: Input vs Output after Weight Multiplication\n'
    'Blue = Input (h1/h2/h3/h4),  Red = Output (y)',
    fontsize=13
)

for row_idx, layer_idx in enumerate(TARGET_LAYERS):
    data = collected[layer_idx]
    for col_idx, (h_key, h_label, y_key, y_label, module, title) in enumerate(comparisons):
        ax = axes1[row_idx, col_idx]
        h_vals = clip(data[module][h_key])
        y_vals = clip(data[module][y_key])
        ax.hist(h_vals, bins=150, alpha=0.5, color='steelblue', density=True, label=h_label)
        ax.hist(y_vals, bins=150, alpha=0.5, color='tomato',    density=True, label=y_label)
        ax.set_title(f'Layer {layer_idx} | {title}', fontsize=8)
        ax.set_xlabel('Value', fontsize=7)
        ax.set_ylabel('Density', fontsize=7)
        ax.tick_params(labelsize=6)
        ax.legend(fontsize=6)

plt.tight_layout()
plt.savefig("dist_comparison.png", dpi=150, bbox_inches='tight')
print("Saved: dist_comparison.png")
plt.close()


# ════════════════════════════════════════════════════
# PLOT 2: Sirf h distributions (no comparison)
# attn h1, attn h2, mlp h1 (h3), mlp h2 (h4)
# ════════════════════════════════════════════════════
h_plots = [
    ('h1', 'attn', 'attn h1\n(input to q/k/v)'),
    ('h2', 'attn', 'attn h2\n(input to o_proj)'),
    ('h1', 'mlp',  'mlp h3\n(input to gate/up)'),
    ('h2', 'mlp',  'mlp h4\n(input to down_proj)'),
]

fig2, axes2 = plt.subplots(3, 4, figsize=(18, 11))
fig2.suptitle('Input Distributions Only (h) — Layers 8, 16, 24', fontsize=13)

for row_idx, layer_idx in enumerate(TARGET_LAYERS):
    data = collected[layer_idx]
    for col_idx, (h_key, module, title) in enumerate(h_plots):
        ax = axes2[row_idx, col_idx]
        ax.hist(clip(data[module][h_key]), bins=150, color='steelblue', density=True)
        ax.set_title(f'Layer {layer_idx} | {title}', fontsize=8)
        ax.set_xlabel('Value', fontsize=7)
        ax.set_ylabel('Density', fontsize=7)
        ax.tick_params(labelsize=6)

plt.tight_layout()
plt.savefig("dist_h_only.png", dpi=150, bbox_inches='tight')
print("Saved: dist_h_only.png")
plt.close()


# ════════════════════════════════════════════════════
# PLOT 3: Sirf y distributions (no comparison)
# y1-y4 (attn), y5-y7 (mlp)
# ════════════════════════════════════════════════════
y_plots = [
    ('y1', 'attn', 'attn y1\n(k_proj out)'),
    ('y2', 'attn', 'attn y2\n(q_proj out)'),
    ('y3', 'attn', 'attn y3\n(v_proj out)'),
    ('y4', 'attn', 'attn y4\n(o_proj out)'),
    ('y5', 'mlp',  'mlp y5\n(gate_proj out)'),
    ('y6', 'mlp',  'mlp y6\n(up_proj out)'),
    ('y7', 'mlp',  'mlp y7\n(down_proj out)'),
]

fig3, axes3 = plt.subplots(3, 7, figsize=(28, 12))
fig3.suptitle('Output Distributions Only (y) — Layers 8, 16, 24', fontsize=13)

for row_idx, layer_idx in enumerate(TARGET_LAYERS):
    data = collected[layer_idx]
    for col_idx, (y_key, module, title) in enumerate(y_plots):
        ax = axes3[row_idx, col_idx]
        ax.hist(clip(data[module][y_key]), bins=150, color='tomato', density=True)
        ax.set_title(f'Layer {layer_idx} | {title}', fontsize=8)
        ax.set_xlabel('Value', fontsize=7)
        ax.set_ylabel('Density', fontsize=7)
        ax.tick_params(labelsize=6)

plt.tight_layout()
plt.savefig("dist_y_only.png", dpi=150, bbox_inches='tight')
print("Saved: dist_y_only.png")
plt.close()
