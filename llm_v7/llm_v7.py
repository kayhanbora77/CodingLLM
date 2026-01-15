"""
Enhanced GPT from Scratch (with nanoGPT-style Pretraining Support)
Improvements: Flash Attention, Better Init, Gradient Accumulation,
Checkpointing, Logging, and Pre-trained Weight Loading
"""

import math
import json
from pathlib import Path
from dataclasses import dataclass, field
import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken

# ==================================================
# Configuration
# ==================================================

def get_base_dir():
    """Helper to handle both scripts and Jupyter Notebooks"""
    try:
        return Path(__file__).resolve().parent
    except NameError:
        return Path.cwd()

@dataclass
class Config:
    # Paths
    base_dir: Path = field(default_factory=get_base_dir)
    data_path: Path = base_dir / "data.txt"
    ckpt_path: Path = base_dir / "best_model.pt"
    log_path: Path = base_dir / "training_log.jsonl"

    # ---------------------------------------------------------
    # ARCHITECTURE SETTINGS
    # To use Pre-trained GPT-2 weights, you MUST match these dimensions
    # to the specific HuggingFace model (e.g., 'gpt2' uses 768/12/12).
    # ---------------------------------------------------------

    # Option A: Small scratch model (Your current settings)
    vocab_size: int = 50257  # GPT-2 vocab size
    block_size: int = 128    # Context window
    embed_size: int = 384    # Embedding dimension
    num_layers: int = 6      # Number of transformer blocks
    num_heads: int = 6       # Number of attention heads

    # Option B: GPT-2 (Small) - Uncomment to use full pre-trained weights
    # vocab_size: int = 50257
    # block_size: int = 1024
    # embed_size: int = 768
    # num_layers: int = 12
    # num_heads: int = 12

    dropout: float = 0.1
    bias: bool = False

    # Training
    batch_size: int = 12
    grad_accum_steps: int = 4  # Effective batch = 12 * 4 = 48
    max_steps: int = 10000
    eval_interval: int = 500
    eval_iters: int = 100

    # Optimization
    learning_rate: float = 6e-4
    min_lr: float = 6e-5
    warmup_steps: int = 500
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    # Early stopping
    patience: int = 5

    # System
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    compile_model: bool = False  # torch.compile
    dtype: str = "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16"

cfg = Config()

# ==================================================
# Data Loading
# ==================================================

class TextDataset:
    def __init__(self, data_path, train_split=0.9):
        # Handle case where file doesn't exist (graceful fallback)
        if not data_path.exists():
            print(f"Warning: {data_path} not found. Using dummy data.")
            text = "Hello world. This is a test of the emergency broadcast system. " * 100
        else:
            text = data_path.read_text(encoding="utf-8")

        self.enc = tiktoken.get_encoding("gpt2")
        data = torch.tensor(self.enc.encode(text), dtype=torch.long)

        n = int(train_split * len(data))
        self.train_data = data[:n]
        self.val_data = data[n:]

        print(f"Dataset: {len(data):,} tokens | Train: {len(self.train_data):,} | Val: {len(self.val_data):,}")

    def get_batch(self, split, batch_size, block_size, device):
        data = self.train_data if split == "train" else self.val_data
        ix = torch.randint(0, len(data) - block_size, (batch_size,))
        x = torch.stack([data[i:i + block_size] for i in ix])
        y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
        return x.to(device), y.to(device)

# ==================================================
# Model Components (nanoGPT style)
# ==================================================

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.embed_size % config.num_heads == 0
        self.num_heads = config.num_heads
        self.head_dim = config.embed_size // config.num_heads
        self.embed_size = config.embed_size

        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.embed_size, 3 * config.embed_size, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.embed_size, config.embed_size, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Flash attention flag
        self.flash = hasattr(F, 'scaled_dot_product_attention')

        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and split head
        q, k, v  = self.c_attn(x).split(self.embed_size, dim=2)
        k = k.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2) # (B, nh, T, hs)

        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.attn_dropout.p if self.training else 0.0, is_causal=True)
        else:
            # manual attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.embed_size, 4 * config.embed_size, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.embed_size, config.embed_size, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.embed_size, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.embed_size, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.embed_size),
            wpe = nn.Embedding(config.block_size, config.embed_size),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.num_layers)]),
            ln_f = nn.LayerNorm(config.embed_size, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.embed_size, config.vocab_size, bias=False)

        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional call was passed multiple values for argument 'input'."
        # This is expected behavior and shouldn't affect the training results.
        self.transformer.wte.weight = self.lm_head.weight

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.num_layers))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

    def from_pretrained(self, model_type='gpt2'):
        """
        Load weights from a pre-trained GPT-2 model (HuggingFace).
        Note: This requires the config dimensions to match the model_type.
        e.g., 'gpt2' -> embed=768, layers=12, heads=12
              'gpt2-medium' -> embed=1024, layers=24, heads=16
        """
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}, "Unknown model type"

        try:
            from transformers import GPT2LMHeadModel
        except ImportError:
            raise ImportError("transformers library is required to load pretrained weights. Run: pip install transformers")

        print(f"Loading weights from pretrained {model_type}...")

        # create a dict of {param_name: param_value}
        # n_layer, n_head and n_embd are deterministically set by model_type
        config_args = {
            'gpt2':         dict(n_layers=12, n_heads=12, n_embed=768),  # 124M params
            'gpt2-medium':  dict(n_layers=24, n_heads=16, n_embed=1024), # 350M params
            'gpt2-large':   dict(n_layers=36, n_heads=20, n_embed=1280), # 774M params
            'gpt2-xl':      dict(n_layers=48, n_heads=25, n_embed=1600), # 1558M params
        }[model_type]

        # Check if config matches
        if (self.config.num_layers != config_args['n_layers'] or
            self.config.num_heads != config_args['n_heads'] or
            self.config.embed_size != config_args['n_embed']):
            raise ValueError(f"Config dimensions do not match {model_type}. "
                             f"Expected Layers: {config_args['n_layers']}, Heads: {config_args['n_heads']}, Embed: {config_args['n_embed']}")

        # load huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # transposed keys due to Conv1D vs Linear difference
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

        # copy while ensuring all tensor names match
        sd_keys = sd_hf.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.masked_bias')] # ignore these
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # ignore these

        for k in sd_keys:
            # map huggingface names to our names
            if k.startswith('transformer.'):
                k = k[11:] # remove 'transformer.' prefix

            # rename e.g. h.0.attn.c_attn.weight -> h.0.attn.c_attn.weight (same)
            # but handle ln_1 -> ln_1, etc.
            # HF uses: h.0.ln_1.weight, we use: h.0.ln_1.weight (match)

            # Transpose weights if needed (HF Conv1D is (in, out), ours is (out, in))
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == self.state_dict()[k].shape
                with torch.no_grad():
                    self.state_dict()[k].copy_(sd_hf[k].t())
            else:
                # straight copy
                assert sd_hf[k].shape == self.state_dict()[k].shape
                with torch.no_grad():
                    self.state_dict()[k].copy_(sd_hf[k])

        print("Loaded pretrained weights successfully.")

# ==================================================
# Optimizer (NanoGPT standard)
# ==================================================

def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
    # start with all of the candidate parameters
    param_dict = {pn: p for pn, p in self.named_parameters()}
    # filter out those that do not require grad
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    # Create AdamW optimizer
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
    return optimizer

GPT.configure_optimizers = configure_optimizers

# ==================================================
# Training Loop
# ==================================================

def get_lr(step, config):
    if step < config.warmup_steps:
        return config.learning_rate * step / config.warmup_steps
    if step > config.max_steps:
        return config.min_lr
    decay_ratio = (step - config.warmup_steps) / (config.max_steps - config.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)

@torch.no_grad()
def estimate_loss(model, dataset, config):
    model.eval()
    losses = {}
    for split in ['train', 'val']:
        losses_list = []
        for _ in range(config.eval_iters):
            x, y = dataset.get_batch(split, config.batch_size, config.block_size, config.device)
            with torch.amp.autocast(device_type=config.device, dtype=getattr(torch, config.dtype)):
                _, loss = model(x, y)
            losses_list.append(loss.item())
        losses[split] = sum(losses_list) / len(losses_list)
    model.train()
    return losses

def train(config=None, use_pretrained=False):
    # Override config if provided
    if config is None:
        config = cfg

    dataset = TextDataset(config.data_path)
    config.vocab_size = dataset.enc.n_vocab

    model = GPT(config).to(config.device)

    if use_pretrained:
        try:
            model.from_pretrained('gpt2') # or 'gpt2-medium', etc.
        except Exception as e:
            print(f"Could not load pretrained weights (Architecture mismatch or transformers not installed): {e}")
            print("Continuing with random initialization...")

    if config.compile_model:
        print("Compiling model...")
        model = torch.compile(model)

    optimizer = model.configure_optimizers(config.weight_decay, config.learning_rate, (config.beta1, config.beta2), config.device)
    scaler = torch.amp.GradScaler(enabled=(config.dtype == 'float16'))

    best_val_loss = float('inf')

    print(f"\n{'='*60}")
    print(f"Training on {config.device} | dtype: {config.dtype}")
    print(f"Effective batch size: {config.batch_size * config.grad_accum_steps}")
    print(f"{'='*60}\n")

    model.train()
    for step in range(config.max_steps):
        lr = get_lr(step, config)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        for micro_step in range(config.grad_accum_steps):
            x, y = dataset.get_batch('train', config.batch_size, config.block_size, config.device)
            with torch.amp.autocast(device_type=config.device, dtype=getattr(torch, config.dtype)):
                _, loss = model(x, y)
                loss = loss / config.grad_accum_steps
            scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        if step % config.eval_interval == 0 or step == config.max_steps - 1:
            losses = estimate_loss(model, dataset, config)
            print(f"Step {step:5d} | LR {lr:.2e} | Train {losses['train']:.4f} | Val {losses['val']:.4f}")

            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                torch.save({'model': model.state_dict(), 'config': config}, config.ckpt_path)

    return model, dataset

# ==================================================
# Main
# ==================================================

if __name__ == "__main__":
    # To use pre-trained weights, set use_pretrained=True AND make sure Config
    # matches GPT-2 dimensions (768, 12, 12).

    # 1. Train from scratch (small model)
    model, dataset = train(use_pretrained=False)

    # 2. Fine-tune GPT-2 (Uncomment and modify Config class to match GPT-2 dimensions)
    # model, dataset = train(use_pretrained=True)

    # Generate sample
    print("\n" + "="*60)
    print("GENERATED TEXT")
    print("="*60 + "\n")
    enc = tiktoken.get_encoding("gpt2")
    idx = torch.tensor([enc.encode("Once upon a time")], dtype=torch.long).to(cfg.device)
    model.eval()
    generated = model.generate(idx, max_new_tokens=50, temperature=0.8, top_k=50)
    print(enc.decode(generated[0].tolist()))
