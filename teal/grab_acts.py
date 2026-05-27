# CALIBRATION SCRIPT: Layer-by-layer activations collect karo aur histograms save karo.
# Yeh script ek baar offline chalti hai. Iska output (histograms + activations) baad mein
# threshold calculate karne (ppl_test.py) aur greedy optimization (greedyopt.py) ke liye use hota hai.


import argparse
import os

import torch
import transformers

import sys

# ── Path setup: parent directory aur utils folder ko Python path mein add karo ──
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)
sys.path.append(os.path.join(parent_dir, 'utils'))


from utils.utils import get_tokenizer, get_sparse_model

from teal.model import LlamaSparseForCausalLM, LlamaSparseConfig
from teal.model import MistralSparseForCausalLM, MistralSparseConfig

from transformers import AutoConfig, AutoModelForCausalLM

# ── Custom sparse model types ko HuggingFace registry mein register karo ──
# Taaki AutoConfig aur AutoModelForCausalLM inhe pehchaan sakein
AutoConfig.register("llama_sparse", LlamaSparseConfig)
AutoConfig.register("mistral_sparse", MistralSparseConfig)

AutoModelForCausalLM.register(LlamaSparseConfig, LlamaSparseForCausalLM)
AutoModelForCausalLM.register(MistralSparseConfig, MistralSparseForCausalLM)

# ── Command line arguments parse karo ──
parser = argparse.ArgumentParser(description="Parse command line arguments for the script.")
parser.add_argument('--model_name', type=str, default="meta-llama/Llama-2-7b-hf",help='Name of the model to use')
parser.add_argument('--output_path', type=str, required=True,help='Path to the output') # contains 1. model itself, 2. histograms, 3. activations
args = parser.parse_args()

# ── Model aur Tokenizer load karo ──
# grab_acts=True: model ko "calibration mode" mein load karo
#   - Isme ActivationModule hook attach hota hai har layer ke MLP aur Self-Attention mein
#   - Forward pass ke dauraan yeh hooks activations collect karte rehte hain
# histogram_path: yahan histograms save honge (output_path/histograms/)
tokenizer = get_tokenizer(args.model_name)
model = get_sparse_model(args.model_name, device="auto", histogram_path=os.path.join(args.output_path, "histograms"), grab_acts=True)

from utils.data import get_dataset
from tqdm import tqdm
import gc

# ── Calibration dataset load karo ──
# Alpaca dataset ke pehle 300 samples use kiye jaate hain
# Yeh samples real-world instruction-following text represent karte hain
# Inse activation distribution ka ek achha statistical estimate milta hai
dataset = get_dataset(
    "tatsu-lab/alpaca",
    subset=None,
    split="train",
    size=300
)

# Saare 300 samples ko ek lambi string mein concatenate karo
text = ""
for sample in tqdm(dataset):
    print(sample["text"])
    print(len(sample["text"]))
    text += sample["text"] + "\n\n"

print(len(text))

# ── Text ko tokens mein convert karo ──
# bsz=10: max 10 sequences ek saath process karni thi (actual batch size data pe depend karta hai)
# seq_len=2048: har sequence 2048 tokens ki hogi
# return_overflowing_tokens=True: text ko 2048-token chunks mein tod do
# padding="max_length": chhote chunks ko 2048 tak pad karo
bsz, seq_len = 10, 2048

encodings = tokenizer(text, truncation=True, return_tensors="pt", max_length=seq_len, return_overflowing_tokens=True, padding="max_length")

# Pehle bsz chunks lo (ya jitne bane hon)
input_ids = encodings.input_ids[:bsz,:].to(device="cuda:0")
print(input_ids.shape)

# ── Initial hidden states compute karo ──
# Token IDs ko embedding vectors mein convert karo
# Shape: (actual_batch_size, seq_len, hidden_dim) e.g. (1, 2048, 4096)
hidden_states = model.model.embed_tokens(input_ids)

# ── Forward pass ke liye required inputs set karo ──
attention_mask = None  # padding nahi use kar rahe, isliye mask None hai
# position_ids: har token ki position sequence mein (0, 1, 2, ..., 2047)
# input_ids.shape[0] use karo (actual batch size), na ki hardcoded bsz
# warna position_ids aur hidden_states ke batch size mismatch se rotary embedding crash karta hai
position_ids = torch.arange(seq_len, dtype=torch.long, device=hidden_states.device).unsqueeze(0).repeat(input_ids.shape[0], 1)
past_key_value=None
output_attentions = False
use_cache = False
cache_position=None
position_embeddings = model.model.rotary_emb(hidden_states, position_ids)


# ── Activations save karne ka folder banao ──
# Yeh raw hidden states greedyopt.py ke liye zaroori hain (per-layer greedy optimization)
act_path = os.path.join(args.output_path, "activations")
os.makedirs(act_path, exist_ok=True)

# ── Layer-by-layer calibration loop ──
# Ek ek layer process karo taaki memory zyada na lage
for i in tqdm(range(len(model.model.layers))):
    print(f"Processing layer {i}...")
    # Step 1: Is layer ka INPUT (hidden_states) save karo
    # greedyopt.py baad mein yeh load karega taaki per-projection sparsity optimize kar sake
    # File: output_path/activations/act_{i}.pt
    torch.save(hidden_states, os.path.join(act_path, f"act_{i}.pt"))

    # Step 2: Layer ka forward pass chalao
    # Iske andar ActivationModule hooks fire hote hain aur h1/h2 activations collect karte hain:
    #   - self_attn: h1 = Q/K/V ka input, h2 = attention output
    #   - mlp:       h1 = gate/up ka input, h2 = MLP intermediate (act_fn * up ka output)
    layer = model.model.layers[i]
    hidden_states = hidden_states.to(layer.self_attn.q_proj.weight.data.device)  # multi-GPU pe sahi device pe bhejo
    hidden_states = layer(hidden_states, attention_mask, position_ids, past_key_value, output_attentions, use_cache, cache_position)[0]

    # Step 3: Collected activations se histogram banao
    # find_histogram(): saare collected activations ko flatten karo, sort karo,
    #                   outliers (1%) clip karo, aur 10,000 bins mein distribution banao
    # save_histogram(): output_path/histograms/layer-{i}/mlp/histograms.pt
    #                   output_path/histograms/layer-{i}/self_attn/histograms.pt
    layer.mlp.activation_module.find_histogram()
    layer.self_attn.activation_module.find_histogram()
    layer.mlp.activation_module.save_histogram()
    layer.self_attn.activation_module.save_histogram()

    # Step 4: Memory free karo
    # Activations RAM/VRAM mein bahut jagah lete hain, histogram ban gaya toh delete karo
    del layer.mlp.activation_module.activations
    del layer.self_attn.activation_module.activations

    # Layer ko None karo taaki GPU memory free ho (agli layer ke liye jagah bane)
    model.model.layers[i] = None

    gc.collect()
    torch.cuda.empty_cache()
