import sys,os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)
sys.path.append(os.path.join(parent_dir, 'utils'))

from utils.data import get_dataset
import torch
from tqdm import tqdm
import os

def eval_ppl(model, tokenizer, device, dataset=None, debug=False, context_size=2048, window_size=512):
    """
    Sliding window approach se perplexity calculate karta hai.

    Kaise kaam karta hai:
      - Poora dataset ek lambi string mein concatenate karo
      - Tokenize karo
      - Ek window (2048+512 = 2560 tokens) slide karo stride=512 se
      - Har window mein sirf last 512 tokens pe loss compute karo
        (pehle 2048 tokens context hain, unpe loss nahi lete)
      - Sab windows ka average NLL lo → PPL = exp(mean_NLL)

    context_size=2048: kitne tokens pehle context ke liye use hote hain
    window_size=512:   kitne tokens pe actual loss compute hota hai
    """

    # ── Dataset ko ek lambi string mein convert karo ──
    text = ""
    for sample in dataset:
        text += sample["text"] + "\n\n"

    # ── Tokenize karo (poora text ek saath) ──
    encodings = tokenizer(text, return_tensors="pt")

    if debug:
        print(tokenizer.decode(encodings.input_ids[0][:100]))

    # ── Window parameters set karo ──
    # max_length: ek baar mein kitne tokens model ko denge (context + eval window)
    # stride: har step mein window kitna aage khiskhegi
    max_length = context_size + window_size   # 2048 + 512 = 2560
    stride = window_size                       # 512

    seq_len = encodings.input_ids.size(1)
    # seq_len ko stride ka multiple banao (clean split ke liye)
    seq_len = seq_len - (seq_len % stride)

    if debug:
        print(f"seq_len: {seq_len}")

    if debug:
        pbar = tqdm(range(0, seq_len, stride))
    else:
        pbar = range(0, seq_len, stride)

    # ── Vocabulary size mismatch check karo ──
    # Kabhi kabhi model aur tokenizer ki vocab size alag hoti hai
    model_vocab_size = model.get_input_embeddings().weight.size(0)
    tokenizer_vocab_size = len(tokenizer)

    if model_vocab_size != tokenizer_vocab_size:
        print("Resize model embeddings to fit tokenizer")
        model.resize_token_embeddings(tokenizer_vocab_size)

    model.eval()
    nlls = []  # har window ka negative log likelihood store karo

    # ── Sliding window loop ──
    for begin_loc in pbar:
        end_loc = begin_loc + max_length  # window ka end

        # Input: 2560 tokens (context 2048 + eval 512)
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)

        # Target: same tokens, lekin pehle 2048 ko -100 set karo
        # -100 matlab PyTorch loss function inhe ignore karta hai
        # Sirf last 512 tokens pe loss compute hogi
        target_ids = input_ids.clone()
        target_ids[:, :-stride] = -100  # pehle 2048 tokens ignore karo

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            # outputs.loss = last 512 tokens ka average negative log likelihood
            neg_log_likelihood = outputs.loss

        if debug:
            pbar.set_description(
                f"nll: {neg_log_likelihood.item():.2f}, ppl: {torch.exp(neg_log_likelihood).item():.2f}"
            )

        nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc
        if end_loc >= seq_len:
            break

    # ── Final PPL calculate karo ──
    # Sab windows ka average NLL lo → exp se PPL banao
    # PPL kam = model zyada confident = better performance
    ppl = torch.exp(torch.stack(nlls).double().mean())

    return ppl.item()
