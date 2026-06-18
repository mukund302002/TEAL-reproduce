# CALIBRATION SCRIPT (Wikitext-2 variant of grab_acts.py)
# Wikitext-2 test split se layer-by-layer activations collect karo aur histograms save karo.
# Same token budget as grab_acts.py: bsz=10, seq_len=2048
#
# Output: --output_path/histograms/layer-{i}/mlp/histograms.pt
#                                  layer-{i}/self_attn/histograms.pt
#
# Usage:
#   python grab_acts_wikitext2.py \
#       --model_name meta-llama/Llama-2-7b-hf \
#       --output_path ../../models/Llama-2-7B-wikitext2

import argparse
import gc
import os
import sys

import torch
import transformers
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir  = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)
sys.path.append(os.path.join(parent_dir, 'utils'))

from utils.utils import get_tokenizer, get_sparse_model
from teal.model import LlamaSparseForCausalLM, LlamaSparseConfig
from teal.model import MistralSparseForCausalLM, MistralSparseConfig
from transformers import AutoConfig, AutoModelForCausalLM

AutoConfig.register("llama_sparse", LlamaSparseConfig)
AutoConfig.register("mistral_sparse", MistralSparseConfig)
AutoModelForCausalLM.register(LlamaSparseConfig, LlamaSparseForCausalLM)
AutoModelForCausalLM.register(MistralSparseConfig, MistralSparseForCausalLM)

parser = argparse.ArgumentParser()
parser.add_argument('--model_name',  type=str, default="meta-llama/Llama-2-7b-hf")
parser.add_argument('--output_path', type=str, required=True)
args = parser.parse_args()

tokenizer = get_tokenizer(args.model_name)
model = get_sparse_model(
    args.model_name,
    device="auto",
    histogram_path=os.path.join(args.output_path, "histograms"),
    grab_acts=True,
)

# ── Wikitext-2 dataset load karo ──
# Alpaca ki jagah wikitext-2 test split use karo.
# Same token budget: bsz=10, seq_len=2048 (20,480 tokens total)
from datasets import load_dataset

ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
text = "\n\n".join(s["text"].strip() for s in ds if s["text"].strip())
print(f"Wikitext-2 total characters: {len(text)}")

bsz, seq_len = 10, 2048

encodings = tokenizer(
    text,
    truncation=True,
    return_tensors="pt",
    max_length=seq_len,
    return_overflowing_tokens=True,
    padding="max_length",
)

input_ids = encodings.input_ids[:bsz, :].to(device="cuda:0")
print(f"Calibration input shape: {input_ids.shape}")

hidden_states = model.model.embed_tokens(input_ids)

attention_mask  = None
position_ids    = torch.arange(seq_len, dtype=torch.long, device=hidden_states.device).unsqueeze(0).repeat(input_ids.shape[0], 1)
past_key_value  = None
output_attentions = False
use_cache       = False
cache_position  = None
position_embeddings = model.model.rotary_emb(hidden_states, position_ids)

act_path = os.path.join(args.output_path, "activations")
os.makedirs(act_path, exist_ok=True)

for i in tqdm(range(len(model.model.layers))):
    print(f"Processing layer {i}...")
    torch.save(hidden_states, os.path.join(act_path, f"act_{i}.pt"))

    layer = model.model.layers[i]
    hidden_states = hidden_states.to(layer.self_attn.q_proj.weight.data.device)
    hidden_states = layer(
        hidden_states, attention_mask, position_ids,
        past_key_value, output_attentions, use_cache, cache_position
    )[0]

    layer.mlp.activation_module.find_histogram()
    layer.self_attn.activation_module.find_histogram()
    layer.mlp.activation_module.save_histogram()
    layer.self_attn.activation_module.save_histogram()

    del layer.mlp.activation_module.activations
    del layer.self_attn.activation_module.activations

    model.model.layers[i] = None
    gc.collect()
    torch.cuda.empty_cache()

print(f"\nDone. Histograms saved to: {args.output_path}/histograms/")
