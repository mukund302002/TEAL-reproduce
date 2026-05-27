import sys,os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)

import torch
from tqdm import tqdm
import os
import argparse



if __name__ == "__main__":
    from utils.utils import get_tokenizer, get_sparse_model
    from utils.eval_ppl import eval_ppl

    from teal.model import LlamaSparseForCausalLM, LlamaSparseConfig
    from teal.model import MistralSparseForCausalLM, MistralSparseConfig

    from utils.data import get_dataset

    from transformers import AutoConfig, AutoModelForCausalLM

    # ── Custom sparse model types ko HuggingFace registry mein register karo ──
    AutoConfig.register("llama_sparse", LlamaSparseConfig)
    AutoConfig.register("mistral_sparse", MistralSparseConfig)

    AutoModelForCausalLM.register(LlamaSparseConfig, LlamaSparseForCausalLM)
    AutoModelForCausalLM.register(MistralSparseConfig, MistralSparseForCausalLM)

    # ── Arguments parse karo ──
    parser = argparse.ArgumentParser(description="Parse command line arguments for the script.")
    parser.add_argument('--model_name', type=str, default="meta-llama/Llama-2-7b-hf", help='Name of the model to use')
    parser.add_argument('--teal_path', type=str, required=True, help='Path to the teal input (grab_acts ka output)')
    parser.add_argument('--greedy_flag', action='store_true', help='Greedy sparsity use karo (uniform ki jagah)')
    parser.add_argument('--sparsity', type=float, default=0.5, help='Sparsity level (0.0 to 1.0)')
    args = parser.parse_args()

    # ── Model load karo — INFERENCE MODE (grab_acts=False by default) ──
    # grab_acts=False hone ki wajah se:
    #   1. SparsifyFn objects bante hain har projection ke liye (q,k,v,o,gate,up,down)
    #   2. Distribution objects bante hain jo histograms.pt load karte hain
    #   3. grabbing_mode=False → forward pass mein sparsity apply hogi (SparsifyFn active)
    # Abhi threshold=0 hai isliye koi zeroing nahi ho rahi (dense jaisa behavior)
    tokenizer = get_tokenizer(args.model_name)
    model = get_sparse_model(
        args.model_name,
        device="auto",
        histogram_path=os.path.join(args.teal_path, "histograms")
        # grab_acts=False (default) → inference mode
    )

    # ── Evaluation dataset load karo ──
    # 250 samples Alpaca se — same source jo grab_acts mein use hua tha
    dataset = get_dataset(
        "tatsu-lab/alpaca",
        subset=None,
        split="train",
        size=250
    )

    # ── Step 1: Dense PPL evaluate karo (baseline) ──
    # Abhi model mein threshold=0 hai (set_uniform_sparsity call nahi hua)
    # Matlab SparsifyFn kuch zero nahi karti → computation dense model jaisa hai
    # Yeh hamara baseline hai — sparse model is se compare hoga
    print("Evaluating dense PPL")
    print("="*40)
    dense_ppl = eval_ppl(model, tokenizer, device="cuda", dataset=dataset, debug=False)
    print(f"PPL: {dense_ppl}")

    # ── Step 2: Sparsity set karo ──
    print("Evaluating sparse PPL at sparsity level: ", args.sparsity)
    print("="*40)

    if args.greedy_flag:
        # Greedy mode: har layer har projection ke liye alag alag optimal sparsity
        # greedyopt.py ne CSV files banai thi lookup/ mein → woh load hoti hain
        # Har projection ka threshold CSV se aata hai (uniform nahi hota)
        print("Evaluating greedy PPL")
        greedy_path = os.path.join(args.teal_path, "lookup")
        model.load_greedy_sparsities(greedy_path, args.sparsity)
    else:
        # Uniform mode (Path 1): sabhi layers aur projections pe same sparsity
        # Yeh internally kya karta hai (model.py → set_uniform_sparsity):
        #   for har layer:
        #     for har projection (q,k,v,o,gate,up,down):
        #       SparsifyFn.set_threshold(0.5)
        #         → threshold = Distribution.icdf(0.5 + 0.5/2)
        #         → threshold = histogram se 75th percentile value
        #         → ab forward mein: x.abs().gt(threshold) * x
        #         → ~50% values (sabse chhoti magnitude wali) zero ho jaayengi
        print("Evaluating uniform PPL")
        model.set_uniform_sparsity(args.sparsity)

    # ── Step 3: Sparse PPL evaluate karo ──
    # Ab model mein thresholds set hain
    # eval_ppl same function hai lekin ab SparsifyFn har forward mein ~50% values zero karti hai
    sparse_ppl = eval_ppl(model, tokenizer, device="cuda", dataset=dataset, debug=False)
    print(f"PPL: {sparse_ppl}")

    print("="*40)
