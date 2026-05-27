import transformers
import torch
import torch.nn as nn
from transformers import AutoConfig
from collections import defaultdict

import os

class SparsifyFn(nn.Module):
    """
    TEAL ka core sparsification module.

    Har projection (q, k, v, o, gate, up, down) ke liye ek SparsifyFn object banta hai.
    Yeh module input tensor ke chhote magnitude values ko zero kar deta hai.

    Kaam kaise karta hai:
      - Ek threshold store karta hai (histogram se calculate hua)
      - Forward pass mein: |x| < threshold wale values → 0
      - ~sparsity% values zero ho jaati hain

    Do modes:
      - Prefill (seq_len > 1): sirf last 50% tokens pe sparsity apply hoti hai
      - Decode  (seq_len = 1): poore token pe sparsity apply hoti hai
    """

    def __init__(self, distr, init_sparsity=None, init_threshold=None, apply_prefill=True):
        """
        Args:
            distr:            Distribution object — histogram load kiya hua hai isme
                              (icdf() se threshold milta hai)
            init_sparsity:    agar shuru mein hi koi sparsity set karni ho (0 to 1)
            init_threshold:   agar seedha threshold value deni ho
            apply_prefill:    True → prefill mein bhi sparsity lagao (last 50% tokens pe)
                              False → prefill mein koi sparsity nahi
        """
        super(SparsifyFn, self).__init__()

        # Dono ek saath specify nahi ho sakte
        assert init_sparsity is None or init_threshold is None, "init_sparsity and init_threshold cannot both be specified"

        if init_sparsity is not None:
            # Sparsity se threshold nikalo:
            # sparsity=0.5 → q = 0.5 + 0.5/2 = 0.75 → icdf(0.75) = 75th percentile
            thresh = distr.icdf(0.5 + init_sparsity/2)
        elif init_threshold is not None:
            # Seedha threshold diya gaya
            thresh = init_threshold
        else:
            # Default: koi sparsity nahi (threshold=0 matlab koi zeroing nahi)
            init_sparsity = 0
            thresh = 0

        # Threshold ko model buffer ke roop mein register karo (fp16 mein)
        # register_buffer: yeh tensor model ke saath save/load hota hai
        # lekin gradient calculate nahi hota (trainable nahi hai)
        self.register_buffer("a", torch.tensor([thresh]).to(torch.float16))

        self.distr = distr          # Distribution object (histogram ke saath)
        self.apply_prefill = apply_prefill

    def set_threshold(self, sparsity):
        """
        Naya sparsity level set karo — threshold histogram se calculate hoga.

        Formula: threshold = icdf(0.5 + sparsity/2)
          sparsity=0.0 → threshold=0   (koi zeroing nahi, dense)
          sparsity=0.5 → icdf(0.75)   = 75th percentile value
          sparsity=0.8 → icdf(0.90)   = 90th percentile value

        Intuition: activations symmetric hain 0 ke around.
          75th percentile = woh value jiske neeche 75% values hain
          = absolute value ka 50th percentile
          = ~50% values zero ho jaayengi
        """
        self.threshold = self.distr.icdf(0.5 + sparsity/2).item() if sparsity != 0.0 else 0.0
        self.sparsity_level = sparsity

    def forward(self, x):
        """
        Input tensor x pe sparsity apply karo.

        x shape: (batch_size, seq_len, hidden_dim)

        Do cases:
          seq_len > 1 → PREFILL phase (poora prompt ek saath)
          seq_len = 1 → DECODE phase (ek token at a time)
        """

        # ── PREFILL phase (seq_len > 1) ──
        if x.size(1) > 1 and self.apply_prefill:
            # Sirf last 50% tokens pe sparsity apply karo, pehle 50% dense rehte hain.
            #
            # Reason (paper Section 5.4.3):
            #   Prefill ke pehle tokens context build karte hain — unhe sparse karna
            #   accuracy hurt karta hai. Last tokens pe sparsity safe hai.
            #
            # NOTE (original author ka comment):
            #   Ideally 99% tokens sparse karne chahiye (sirf 1% dense),
            #   lekin paper ke evals 50% pe finish ho gaye the is finding se pehle.
            #   half_seq_len = int(0.99 * x.size(1))  ← yeh better hoga future mein
            half_seq_len = x.size(1) // 2

            last_context = x[:, -half_seq_len:, :]        # last 50% tokens
            modified_context = self.apply(last_context)   # inpe sparsity lagao

            # Pehla half unchanged + sparse last half → wapas jodo
            x = torch.cat((x[:, :-half_seq_len, :], modified_context), dim=1)
            return x

        # ── PREFILL phase lekin apply_prefill=False ──
        # Prefill pe koi sparsity nahi chahiye → x as-is return karo
        if x.size(1) > 1 and not self.apply_prefill:
            return x

        # ── DECODE phase (seq_len = 1) ──
        # Ek token generate ho raha hai → poore token pe sparsity lagao
        assert x.size(1) == 1, "supposedly x is decode only"
        return self.apply(x)

    def apply(self, x):
        """
        Actual sparsification: threshold se chhote magnitude values zero karo.

        x.abs()          → har value ka absolute value lo
        .gt(self.threshold) → jo threshold se bade hain unhe True (1), baaki False (0)
        * x              → mask multiply karo original x se

        Example:
          x         = [ 0.1, -0.5,  0.03,  0.8, -0.02]
          threshold = 0.42
          mask      = [False, True, False,  True, False]
          output    = [ 0.0, -0.5,  0.0,   0.8,  0.0 ]
          → 3 out of 5 values zero ho gaye = 60% sparsity
        """
        return x.abs().gt(self.threshold) * x

    def get_threshold(self):
        return self.threshold


def interp(x, xp, fp):
    """Custom interpolation function for PyTorch tensors."""
    i = torch.searchsorted(xp, x)
    i = torch.clamp(i, 1, len(xp) - 1)
    
    xp_left = xp[i - 1]
    xp_right = xp[i]
    fp_left = fp[i - 1]
    fp_right = fp[i]
    
    t = (x - xp_left) / (xp_right - xp_left)
    return fp_left + t * (fp_right - fp_left)


class Distribution:
    def __init__(self, file_path, hidden_type):
        self.file_path = file_path
        self.hidden_type = hidden_type # h1 or h2
        
        histogram = torch.load(f"{self.file_path}/histograms.pt")

        self.bin_centers, self.counts = histogram[f"{self.hidden_type}_centers"], histogram[self.hidden_type]

        self.total_count = self.counts.sum()
        self.cumulative_counts = torch.cumsum(self.counts, dim=0)

    # kernel smoothing
    def pdf(self, x, bandwidth=None):
        if bandwidth is None:
            bandwidth =  1.06 * torch.std(self.bin_centers[1:-1]) * (self.total_count-2)**(-1/5)
        
        bin_centers = self.bin_centers.unsqueeze(1)
        
        if isinstance(x, float) or isinstance(x, int):
            x = torch.tensor([x])
        else:
            x = x.unsqueeze(0)
        
        kernel = torch.exp(-0.5 * ((x - bin_centers) / bandwidth)**2) / (bandwidth * torch.sqrt(torch.tensor(2 * torch.pi)))
        pdf = torch.sum(kernel * self.counts.unsqueeze(1), dim=0) / self.total_count
        
        return pdf
    
    def cdf(self, x):
        return interp(x, self.bin_centers, self.cumulative_counts / self.total_count)
    
    # NOTE: Assumes distribution is zero mean unimodal
    def icdf(self, q):
        # if q < 0.01 or q > 0.99:
        #     print(f"WARNING: All outliers clip to the most extreme bin")

        target_count = q * self.total_count
        idx = torch.searchsorted(self.cumulative_counts, target_count)
        
        if idx == 0:
            return self.bin_centers[0]
        elif idx == len(self.bin_centers):
            return self.bin_centers[-1]
        else:
            lower_count = self.cumulative_counts[idx - 1]
            upper_count = self.cumulative_counts[idx]
            lower_value = self.bin_centers[idx - 1]
            upper_value = self.bin_centers[idx]
            
            fraction = (target_count - lower_count) / (upper_count - lower_count)
            return lower_value + fraction * (upper_value - lower_value)

class ActivationModule:
    def __init__(self, file_path):
        self.file_path = file_path
        self.activations = defaultdict(list)
        self.histograms = None
        
        # store is to store stuff like position_ids in attn (for convinience, is bad code)
        self.store = {}

    def grab_activations(self, x, key):
        if x.size(1) > 1:  # Check if seq_len > 1
            self.activations[key].append(x.detach().squeeze(0).cpu().float())
    def save_activations(self):
        self.activations = self.combine_activations()
        torch.save(self.activations, f"{self.file_path}/activations.pt")

    def load_activations(self):
        self.activations = torch.load(f"{self.file_path}/activations.pt")

    # NOTE: This doesn't store outlier activation values
    def find_histogram(self, num_bins=10000, outlier_threshold=0.01):
        if self.histograms is None:
            # for fine-grained analysis, do not combine activations
            self.activations = self.combine_activations()
            self.histograms = {}
        else:
            return self.histograms
        
        torch.cuda.empty_cache()
        for key, acts in self.activations.items():

            acts = acts.flatten().detach().to('cuda')
            acts = torch.sort(acts)[0]

            lower_bound = acts[int(outlier_threshold * len(acts))]
            upper_bound = acts[-int(outlier_threshold * len(acts))]

            acts = acts.cpu()

            main_bins = torch.linspace(lower_bound, upper_bound, num_bins - 1)
            bins = torch.cat([torch.tensor([acts[0]]), main_bins, torch.tensor([acts[-1]])])

            counts, _ = torch.histogram(acts, bins=bins)

            bin_centers = (bins[:-1] + bins[1:]) / 2

            self.histograms[key] = counts.float().cpu()
            self.histograms[f"{key}_centers"] = bin_centers.float().cpu()
        return self.histograms
    
    def save_histogram(self):
        os.makedirs(self.file_path, exist_ok=True)
        torch.save(self.histograms, f"{self.file_path}/histograms.pt")

    def combine_activations(self):
        combined_activations = {}
        for key, acts in self.activations.items():
            combined_activations[key] = torch.cat(acts, dim=0)
        return combined_activations

from transformers import AutoConfig

def get_model_class_name(model_name):
    try:
        # Fetch the model config
        config = AutoConfig.from_pretrained(model_name)
        
        # Get the model class name from the config
        model_class_name = config.architectures[0] if config.architectures else None
        
        return model_class_name
    except Exception as e:
        print(f"Error fetching model class name: {e}")
        return None


def get_sparse_model(model_name, device, histogram_path, **kwargs):
    from teal.model import LlamaSparseForCausalLM, MistralSparseForCausalLM, LlamaSparseConfig, MistralSparseConfig

    from transformers import AutoConfig, AutoModelForCausalLM

    AutoConfig.register("llama_sparse", LlamaSparseConfig)
    AutoModelForCausalLM.register(LlamaSparseConfig, LlamaSparseForCausalLM)
    AutoConfig.register("mistral_sparse", MistralSparseConfig)
    AutoModelForCausalLM.register(MistralSparseConfig, MistralSparseForCausalLM)

    class_name = get_model_class_name(model_name)

    assert class_name in ["LlamaForCausalLM", "MistralForCausalLM", "LlamaSparseForCausalLM", "MistralSparseForCausalLM"], f"Model class name {class_name} not supported"

    SparseModel = LlamaSparseForCausalLM if "Llama" in class_name else MistralSparseForCausalLM

    if device == 'auto':
        # multi gpu
        return SparseModel.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto", attn_implementation="flash_attention_2", histogram_path=histogram_path, **kwargs)
    else:
        return SparseModel.from_pretrained(model_name, torch_dtype=torch.float16, device_map=device, attn_implementation="flash_attention_2", histogram_path=histogram_path, **kwargs)

def get_tokenizer(tokenizer_name):
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        tokenizer_name, use_fast=False, trust_remote_code=True
    )

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.pad_token_id = 0

    return tokenizer


def get_module_device(module):
    return next(module.parameters()).device




def get_layer_greedy_sparsities(layer_sparsities, results_dir):
    import pandas as pd
    num_layers = len(layer_sparsities)
    projs = ['q', 'k', 'v', 'o', 'gate', 'up', 'down']
    sparsities = {proj: [0.0] * num_layers for proj in projs}
    
    for layer, target_sparsity in enumerate(layer_sparsities):
        file_path = os.path.join(results_dir, f'layer-{layer}', 'results.csv')
        df = pd.read_csv(file_path)
        
        # Find the row with the closest effective sparsity
        closest_row = df.iloc[(df['Effective Sparsity'] - target_sparsity).abs().argsort()[:1]]
        
        for proj in projs:
            sparsities[proj][layer] = closest_row[proj].values[0]
    
    return sparsities