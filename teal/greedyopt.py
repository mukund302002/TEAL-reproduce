
import os
import sys
import argparse
import torch
import csv
from copy import deepcopy


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)
sys.path.append(os.path.join(parent_dir, 'utils'))

from teal.model import LlamaSparseForCausalLM, LlamaSparseConfig, MistralSparseForCausalLM, MistralSparseConfig
from transformers import AutoConfig, AutoModelForCausalLM
AutoConfig.register("llama_sparse", LlamaSparseConfig)
AutoModelForCausalLM.register(LlamaSparseConfig, LlamaSparseForCausalLM)
AutoConfig.register("mistral_sparse", MistralSparseConfig)
AutoModelForCausalLM.register(MistralSparseConfig, MistralSparseForCausalLM)
from utils.utils import get_sparse_model, get_tokenizer

import torch.cuda


# ── Weight dictionary: har projection ka relative parameter size ──
# Yeh weights batate hain ki model ki overall sparsity calculate karte waqt
# har projection ka kitna contribution hai (parameter count ke hisaab se).
#
# Llama-2-7B example:
#   q=1, k=1/8, v=1/8 → Llama-2 mein GQA hai: 32 Q heads but sirf 8 K/V heads
#                         isliye K aur V ka weight 1/8 hai
#   gate=2.6875 → MLP projections Q se bade hain (hidden_dim=4096, intermediate=11008)
#
# Greedy loop mein step_size = base_step_size / weight
# Matlab bade projections ke liye chote steps lete hain (zyada careful rehte hain)
weight_dict = {
    "Llama-3-8B": {
        'q': 1, 'k': 1/4, 'v': 1/4, 'o': 1,
        'gate': 3.5, 'up': 3.5, 'down': 3.5
    },
    "Llama-3-70B": {
        'q': 1, 'k': 1/8, 'v': 1/8, 'o': 1,
        'gate': 3.5, 'up': 3.5, 'down': 3.5
    },
    "Llama-2-7B": {
        'q': 1, 'k': 1/8, 'v': 1/8, 'o': 1,
        'gate': 2.6875, 'up': 2.6875, 'down': 2.6875
    },
    "Llama-2-13B": {
        'q': 1, 'k': 1/8, 'v': 1/8, 'o': 1,
        'gate': 2.7, 'up': 2.7, 'down': 2.7
    },
    "Llama-2-70B": {
        'q': 1, 'k': 1/8, 'v': 1/8, 'o': 1,
        'gate': 3.5, 'up': 3.5, 'down': 3.5
    },
    "Mistral-7B": {
        'q': 1, 'k': 1/8, 'v': 1/8, 'o': 1,
        'gate': 3.5, 'up': 3.5, 'down': 3.5
    },
}

def set_layer_sparsities(layer, sparsities):
    """Ek layer ke har projection pe sparsity set karo (threshold icdf se calculate hoga)."""
    layer.mlp.sparse_fns['gate'].set_threshold(sparsities['gate'])
    layer.mlp.sparse_fns['up'].set_threshold(sparsities['up'])
    layer.mlp.sparse_fns['down'].set_threshold(sparsities['down'])

    layer.self_attn.sparse_fns['q'].set_threshold(sparsities['q'])
    layer.self_attn.sparse_fns['k'].set_threshold(sparsities['k'])
    layer.self_attn.sparse_fns['v'].set_threshold(sparsities['v'])
    layer.self_attn.sparse_fns['o'].set_threshold(sparsities['o'])

def f(sparsities, weights):
    """
    Effective (weighted) sparsity calculate karo.

    Har projection ki sparsity ko uske weight se multiply karo aur average lo.
    Yeh overall model sparsity ka realistic estimate deta hai (parameter count ke hisaab se).

    Example: agar gate=0.6, up=0.6, down=0.6, q=0.3, k=0.3, v=0.3, o=0.3
    toh effective sparsity = weighted average ≈ 0.5
    """
    total_weight = sum(weights.values())
    weighted_sparsity_sum = 0
    for projection_type, value in sparsities.items():
        if projection_type in weights:
            weighted_sparsity_sum += value * weights[projection_type]
    return weighted_sparsity_sum / total_weight

def layer_forward(layer, hidden_states):
    """
    Ek layer ka forward pass chalao aur output hidden_states return karo.
    grab_acts ke forward se alag: yahan sparsity apply hoti hai (grabbing_mode=False).
    """
    bsz, seq_len, _ = hidden_states.shape

    attention_mask = None
    position_ids = torch.arange(seq_len, dtype=torch.long, device=hidden_states.device).unsqueeze(0).repeat(bsz, 1)
    past_key_value=None
    output_attentions = False
    use_cache = False
    cache_position=None

    return layer(hidden_states, attention_mask, position_ids, past_key_value, output_attentions, use_cache, cache_position)[0]


def calculate_activation_error(target_acts, new_activations, last_fraction=0.25):
    """
    Dense output (target) aur sparse output (new) ke beech ka error nikalo.

    Sirf last 25% tokens pe calculate karo (last_fraction=0.25).
    Reason: sequence ke last tokens zyada important hote hain prediction ke liye.
    L2 norm use hota hai (torch.norm) → dono tensors kitne alag hain.
    """
    start_idx = int(new_activations.shape[1] * (1 - last_fraction))
    res = torch.norm(target_acts[:, start_idx:] - new_activations[:, start_idx:], dim=1).mean()
    return res

def calculate_baseline_error(layer, input_acts, target_acts, baseline_sparsities, last_fraction):
    """
    Uniform sparsity (same sparsity sab projections pe) ka error calculate karo.
    Yeh CSV mein baseline ke liye save hota hai — greedy vs uniform compare karne ke liye.
    """
    set_layer_sparsities(layer, baseline_sparsities)
    new_activations = layer_forward(layer, input_acts)
    return calculate_activation_error(target_acts, new_activations, last_fraction)

def process_layer(layer, model_type, layer_idx, target_sparsity, base_step_size, last_fraction, teal_path):
    """
    Ek layer ke liye greedy sparsity optimization karo.

    Algorithm:
      1. Dense output nikalo (target) — sparsity=0
      2. Greedy loop (jab tak target_sparsity na mile):
           - Har projection (q,k,v,o,gate,up,down) ke liye:
               uski sparsity thodi badhaao (step_size se)
               forward pass chalao
               dense output se error measure karo
           - Sabse kam error wala projection chuno
           - Uski sparsity permanently badhaao
           - CSV mein save karo
      3. Output: lookup/layer-{i}/results.csv
         Columns: Effective Sparsity, Activation Error, Baseline Error, q, k, v, o, gate, up, down
    """
    weights = weight_dict[model_type]

    histogram_path = os.path.join(teal_path, 'histograms')
    activations_path = os.path.join(teal_path, 'activations', f'act_{layer_idx}.pt')
    output_path = os.path.join(teal_path, 'lookup', f'layer-{layer_idx}', 'results.csv')

    device = "cuda"
    # grab_acts.py mein save kiya hua layer ka INPUT hidden_states load karo
    input_acts = torch.load(activations_path, map_location='cpu').to(device)
    layer = layer.to(device)

    projs = ['q', 'k', 'v', 'o', 'gate', 'up', 'down']

    # ── Step 1: Dense (target) output nikalo ──
    # Sab sparsities 0 karo → koi zeroing nahi → yeh hai "ideal" output
    sparsities = {proj: 0.0 for proj in projs}
    set_layer_sparsities(layer, sparsities)
    target_acts = layer_forward(layer, input_acts)  # dense output = hamara target

    # ── Step sizes calculate karo ──
    # Bade projections (MLP) ke liye chote steps (weight zyada → step chhota)
    # Taaki unhe gradually sparse kiya jaaye
    step_sizes = {proj: base_step_size * (1 / weights[proj]) for proj in projs}

    sparsities = {proj: 0.0 for proj in projs}

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Effective Sparsity'] + ['Activation Error', 'Baseline Error'] + [f"{proj}" for proj in projs])

        # ── Step 2: Greedy loop ──
        # Jab tak overall weighted sparsity target se kam hai, increment karte raho
        while f(sparsities, weights) < target_sparsity:
            best_error = float('inf')
            best_proj = None

            # Har projection ko try karo: agar is projection ki sparsity badhaayein
            # toh error kitna hoga?
            for proj in projs:
                temp_sparsities = deepcopy(sparsities)

                if temp_sparsities[proj] >= 1:
                    continue  # pehle se 100% sparse hai, skip karo

                temp_sparsities[proj] += step_sizes[proj]  # candidate increment

                # Sparse forward pass chalao aur dense output se compare karo
                set_layer_sparsities(layer, temp_sparsities)
                new_activations = layer_forward(layer, input_acts)
                error = calculate_activation_error(target_acts, new_activations, last_fraction)

                # Sabse kam error wala projection track karo
                if error < best_error:
                    best_error = error
                    best_proj = proj

            # ── Best projection choose karo aur permanently update karo ──
            sparsities[best_proj] += step_sizes[best_proj]
            set_layer_sparsities(layer, sparsities)

            # Baseline error: agar uniform sparsity use karte (sab same) toh error kya hota?
            effective_sparsity = f(sparsities, weights)
            baseline_sparsities = {proj: effective_sparsity for proj in projs}
            baseline_error = calculate_baseline_error(layer, input_acts, target_acts, baseline_sparsities, last_fraction)

            # ── CSV mein save karo ──
            # Har row = ek greedy step ka snapshot:
            # [effective_sparsity, greedy_error, uniform_error, q, k, v, o, gate, up, down]
            row = [effective_sparsity] + [best_error.item(), baseline_error.item()] + [sparsities[proj] for proj in projs]
            csvwriter.writerow(row)
            csvfile.flush()

            print(f"Updated: Effective Sparsity: {effective_sparsity:.4f}, Activation Error: {best_error:.4f}, Baseline Error: {baseline_error:.4f}")

    layer.to('cpu')
    return sparsities


import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--model_type", type=str, required=True, help="Model type string (e.g. 'Llama-2-7B') — weight_dict ka key")
    parser.add_argument("--teal_path", type=str, required=True)
    parser.add_argument("--target_sparsity", type=float, default=0.9, help="Target effective sparsity tak greedy loop chalega")
    parser.add_argument("--base_step_size", type=float, default=0.05, help="Har greedy step mein sparsity kitni badhegi (base)")
    parser.add_argument("--last_fraction", type=float, default=0.25, help="Sequence ka kitna hissa error calculate karne ke liye use ho")

    args = parser.parse_args()

    histogram_path = os.path.join(args.teal_path, 'histograms')

    from utils.utils import get_model_class_name

    class_name = get_model_class_name(args.model_name)
    assert class_name in ['LlamaSparseForCausalLM', 'MistralSparseForCausalLM', 'LlamaForCausalLM', 'MistralForCausalLM'], f"Model {args.model_name} not supported"

    SparseModel = LlamaSparseForCausalLM if "Llama" in class_name else MistralSparseForCausalLM

    # ── Model CPU pe load karo ──
    # Greedy loop mein ek ek layer GPU pe jaayegi, poora model GPU pe nahi
    model = get_sparse_model(args.model_name, device='cpu', histogram_path=histogram_path)

    os.makedirs(os.path.join(args.teal_path, 'lookup'), exist_ok=True)

    num_layers = len(model.model.layers)

    # ── Har layer ke liye greedy optimization chalao ──
    # Har layer independently process hoti hai
    # Output: lookup/layer-{i}/results.csv
    for layer_idx in range(num_layers):
        print(f"Processing layer {layer_idx}")
        layer = model.model.layers[layer_idx]
        process_layer(layer, args.model_type, layer_idx, args.target_sparsity, args.base_step_size, args.last_fraction, args.teal_path)
