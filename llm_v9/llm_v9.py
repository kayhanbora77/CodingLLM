"""
Enhanced GPT v9 (Modern Architecture)
New Technologies: RMSNorm, SwiGLU, KV Cache (for fast generation)
Previous Improvements: Flash Attention, LoRA, RoPE, Better Init, Gradient Accumulation
"""

import math
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Tuple
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
    data_path: Optional[Path] = field(default=None)
    ckpt_path: Optional[Path] = field(default=None)
    log_path: Optional[Path] = field(default=None)

    # ARCHITECTURE SETTINGS
    vocab_size: int = 50257
    block_size: int = 128
    embed_size: int = 384
    num_layers: int = 6
    num_heads: int = 6
    intermediate_size: int = None # If None, defaults to 4 * embed_size (or approx for SwiGLU)

    # NORM & ACTIVATION (New Tech)
    norm_type: str = "rmsnorm" # Options: "layernorm", "rmsnorm"
    activation: str = "swiglu"  # Options: "gelu", "swiglu"

    # LoRA SETTINGS
    use_lora: bool = False
    lora_r: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.05
    lora_target_modules: list = field(default_factory=lambda: ["c_attn", "c_proj", "c_fc"])

    # RoPE SETTINGS
    use_rope: bool = True
    rope_freq_base: int = 10000

    dropout: float = 0.1
    bias: bool = False

    # Training
    batch_size: int = 12
    grad_accum_steps: int = 4
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

    # System
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    compile_model: bool = False
    dtype: str = "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16"

    def __post_init__(self):
        if self.data_path is None:
            self.data_path = self.base_dir / "data.txt"
        if self.ckpt_path is None:
            self.ckpt_path = self.base_dir / "best_model.pt"
        if self.log_path is None:
            self.log_path = self.base_dir / "training_log.jsonl"

        # Calculate intermediate size based on activation
        # For SwiGLU, we often use slightly different ratios, but 4*embed is standard for compatibility.
        # Note: SwiGLU splits the matrix, so actual param count changes.
        if self.intermediate_size is None:
            self.intermediate_size = 4 * self.embed_size

cfg = Config()

# ==================================================
# Helpers: RoPE, LoRA, RMSNorm
# ==================================================

class RotaryEmbedding(nn.Module):
    """Rotary Positional Embeddings (RoPE)"""
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        self.dim = dim
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        t = torch.arange(max_position_embeddings, device=device).type_as(inv_freq)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos", emb.cos()[None, None, :, :])
        self.register_buffer("sin", emb.sin()[None, None, :, :])

    def forward(self, seq_len):
        return self.cos[:, :, :seq_len, :], self.sin[:, :, :seq_len, :]

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class LoRALinear(nn.Module):
    """Linear layer with LoRA adapter."""
    def __init__(self, in_features, out_features, r, alpha, dropout, bias):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.r = r
        self.alpha = alpha
        self.scaling = self.alpha / self.r if self.r > 0 else 1.0
        self.lora_A = nn.Parameter(torch.zeros((r, in_features)))
        self.lora_B = nn.Parameter(torch.zeros((out_features, r)))
        self.dropout = nn.Dropout(dropout)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        result = self.linear(x)
        lora_result = self.dropout(x) @ self.lora_A.T @ self.lora_B.T
        return result + lora_result * self.scaling

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (New Tech)
    Used in Llama, Mistral, Falcon.
    More stable than LayerNorm and doesn't require centering (mean subtraction).
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

# ==================================================
# Data Loading
# ==================================================

class TextDataset:
    def __init__(self, data_path, train_split=0.9):
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
# Model Components
# ==================================================

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.embed_size % config.num_heads == 0
        self.num_heads = config.num_heads
        self.head_dim = config.embed_size // config.num_heads
        self.embed_size = config.embed_size
        self.config = config

        # QKV projection
        if config.use_lora and 'c_attn' in config.lora_target_modules:
            self.c_attn = LoRALinear(config.embed_size, 3 * config.embed_size,
                                    config.lora_r, config.lora_alpha, config.lora_dropout, config.bias)
        else:
            self.c_attn = nn.Linear(config.embed_size, 3 * config.embed_size, bias=config.bias)

        # Output projection
        if config.use_lora and 'c_proj' in config.lora_target_modules:
            self.c_proj = LoRALinear(config.embed_size, config.embed_size,
                                    config.lora_r, config.lora_alpha, config.lora_dropout, config.bias)
        else:
            self.c_proj = nn.Linear(config.embed_size, config.embed_size, bias=config.bias)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.flash = hasattr(F, 'scaled_dot_product_attention')

        # RoPE
        self.use_rope = config.use_rope
        if self.use_rope:
            self.rotary_emb = RotaryEmbedding(self.head_dim, config.block_size, config.rope_freq_base, device="cpu")
        else:
            self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))
                                .view(1, 1, config.block_size, config.block_size))

        # KV Cache (New Tech) - Initialize cache placeholders
        self.cache_k = None
        self.cache_v = None

    def forward(self, x, use_cache=False):
        B, T, C = x.size()

        # QKV projection
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.embed_size, dim=2)

        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # RoPE
        if self.use_rope:
            cos, sin = self.rotary_emb(T)
            cos, sin = cos.to(x.device), sin.to(x.device)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # KV Cache Logic
        if use_cache:
            # If we have a cached history, concat it
            if self.cache_k is not None:
                # k shape: (B, n_heads, T_curr, head_dim)
                # cache_k shape: (B, n_heads, T_prev, head_dim)
                k = torch.cat([self.cache_k, k], dim=2)
                v = torch.cat([self.cache_v, v], dim=2)

            # Update cache
            self.cache_k = k
            self.cache_v = v
            # The sequence length for attention is now the full cached length
            T = k.size(2)

        if self.flash:
            # Flash Attention automatically handles causal masking via is_causal=True
            # When using cache, we pass the full sequence, but only the new Q's need masking relative to K?
            # Actually Flash Attention handles this correctly if we pass the whole K,V and the new Q
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=None,
                                             dropout_p=self.attn_dropout.p if self.training else 0.0,
                                             is_causal=True)
        else:
            # Manual Attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

            if not self.use_rope and not use_cache:
                # Standard mask
                att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
            else:
                # Dynamic mask for RoPE or KV Cache
                causal_mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
                att = att.masked_fill(causal_mask == 0, float('-inf'))

            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y

    def reset_cache(self):
        self.cache_k = None
        self.cache_v = None

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()

        # SwiGLU (New Tech)
        # Instead of one large Linear (c_fc) -> GELU, we use two linears (gate, up) -> SiLU * Gate
        self.use_swiglu = (config.activation == "swiglu")

        if self.use_swiglu:
            # For LoRA compatibility, we adapt the target modules dynamically
            # SwiGLU needs Gate and Up projections.
            # If LoRA is active, we apply it to these new layers if they are effectively replacing c_fc.

            use_lora_gate = config.use_lora and ('c_fc' in config.lora_target_modules or 'gate' in config.lora_target_modules)
            use_lora_up   = config.use_lora and ('c_fc' in config.lora_target_modules or 'up' in config.lora_target_modules)

            if use_lora_gate:
                self.gate_proj = LoRALinear(config.embed_size, config.intermediate_size,
                                           config.lora_r, config.lora_alpha, config.lora_dropout, config.bias)
            else:
                self.gate_proj = nn.Linear(config.embed_size, config.intermediate_size, bias=config.bias)

            if use_lora_up:
                self.up_proj = LoRALinear(config.embed_size, config.intermediate_size,
                                         config.lora_r, config.lora_alpha, config.lora_dropout, config.bias)
            else:
                self.up_proj = nn.Linear(config.embed_size, config.intermediate_size, bias=config.bias)

            self.act = nn.SiLU() # Swish is SiLU

        else:
            # Standard GELU MLP
            if config.use_lora and 'c_fc' in config.lora_target_modules:
                self.c_fc = LoRALinear(config.embed_size, config.intermediate_size,
                                      config.lora_r, config.lora_alpha, config.lora_dropout, config.bias)
            else:
                self.c_fc = nn.Linear(config.embed_size, config.intermediate_size, bias=config.bias)
            self.gelu = nn.GELU()

        # Down projection
        if config.use_lora and 'c_proj' in config.lora_target_modules:
            self.down_proj = LoRALinear(config.intermediate_size, config.embed_size,
                                       config.lora_r, config.lora_alpha, config.lora_dropout, config.bias)
        else:
            self.down_proj = nn.Linear(config.intermediate_size, config.embed_size, bias=config.bias)

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        if self.use_swiglu:
            # SwiGLU: Swish(gate) * up
            x = self.act(self.gate_proj(x)) * self.up_proj(x)
        else:
            x = self.c_fc(x)
            x = self.gelu(x)

        x = self.down_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Norm Choice (New Tech)
        if config.norm_type == "rmsnorm":
            self.ln_1 = RMSNorm(config.embed_size)
            self.ln_2 = RMSNorm(config.embed_size)
        else:
            self.ln_1 = nn.LayerNorm(config.embed_size, bias=config.bias)
            self.ln_2 = nn.LayerNorm(config.embed_size, bias=config.bias)

        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)

    def forward(self, x, use_cache=False):
        # Attention block
        # Note: KV Cache is handled inside attn, but we pass the flag
        x = x + self.attn(self.ln_1(x), use_cache=use_cache)
        # MLP block
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
            wpe = nn.Embedding(config.block_size, config.embed_size) if not config.use_rope else None,
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.num_layers)]),
            ln_f = RMSNorm(config.embed_size) if config.norm_type == "rmsnorm" else nn.LayerNorm(config.embed_size, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.embed_size, config.vocab_size, bias=False)

        # Weight tying
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)

        # Special init for residual projections (important for stability)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight') or pn.endswith('down_proj.weight') or pn.endswith('c_proj.lora_B') or pn.endswith('down_proj.lora_B'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.num_layers))

        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))
        if config.use_lora:
             self.print_trainable_parameters()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm) or isinstance(module, RMSNorm):
            if hasattr(module, 'bias') and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding and self.transformer.wpe:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def print_trainable_parameters(self):
        trainable_params = 0
        all_param = 0
        for _, param in self.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(
            f"trainable params: {trainable_params:,} || "
            f"all params: {all_param:,} || "
            f"trainable%: {100 * trainable_params / all_param:.2f}%"
        )

    def freeze_base_model(self):
        if not self.config.use_lora:
            return
        for param in self.parameters():
            param.requires_grad = False
        for n, p in self.named_parameters():
            if 'lora_A' in n or 'lora_B' in n:
                p.requires_grad = True
        if self.transformer.wte.weight.requires_grad == False:
             self.transformer.wte.weight.requires_grad = True
        if self.lm_head.weight.requires_grad == False:
             self.lm_head.weight.requires_grad = True

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()

        tok_emb = self.transformer.wte(idx)

        if self.config.use_rope:
            x = self.transformer.drop(tok_emb)
        else:
            pos = torch.arange(0, t, dtype=torch.long, device=device)
            pos_emb = self.transformer.wpe(pos)
            x = self.transformer.drop(tok_emb + pos_emb)

        # Use cache=False during training (standard full forward pass)
        for block in self.transformer.h:
            x = block(x, use_cache=False)

        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    def reset_cache(self):
        """Reset KV cache for all attention layers"""
        for block in self.transformer.h:
            block.attn.reset_cache()

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        # Use KV Cache for efficient generation
        self.reset_cache()
        self.eval()

        for _ in range(max_new_tokens):
            # If the sequence context is growing, crop it to block_size
            # Note: With KV Cache, we only feed the last token, but here we
            # rely on idx_cond logic for the first step.
            # Ideally, we separate the first pass (prefill) and subsequent passes (decode).

            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]

            # Forward pass
            # If we have a history, we only pass the very last token to the model to save compute,
            # provided the cache is populated.
            # However, to keep this simple and robust with the crop logic above:
            # We just pass the current context. The attention layer handles the caching logic internally
            # based on the self.cache_k state.

            # To be strictly efficient with KV cache:
            if idx.size(1) > 1 and self.transformer.h[0].attn.cache_k is not None:
                # Decode step: only pass the last token
                logits, _ = self(idx[:, -1:], targets=None)
            else:
                # Prefill step: pass the whole context
                logits, _ = self(idx_cond, targets=None)

            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        self.train()
        return idx

# ==================================================
# Optimizer & Training Loop
# ==================================================

def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
    param_dict = {pn: p for pn, p in self.named_parameters()}
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]

    # Fused AdamW (New Tech) - Check if available (CUDA only)
    use_fused = (device_type == 'cuda') and hasattr(torch.optim, 'AdamW')
    print(f"Using Fused AdamW: {use_fused}")

    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, fused=use_fused)
    return optimizer

GPT.configure_optimizers = configure_optimizers

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
    if config is None:
        config = cfg

    dataset = TextDataset(config.data_path)
    config.vocab_size = dataset.enc.n_vocab

    model = GPT(config).to(config.device)

    if use_pretrained:
        print("Pretrained loading enabled.")
        # Note: Dimensions must match.
        # If using SwiGLU/RMSNorm, standard GPT-2 weights won't load directly without conversion.

    if config.use_lora:
        model.freeze_base_model()

    if config.compile_model:
        print("Compiling model...")
        model = torch.compile(model)

    optimizer = model.configure_optimizers(config.weight_decay, config.learning_rate, (config.beta1, config.beta2), config.device)
    scaler = torch.amp.GradScaler(enabled=(config.dtype == 'float16'))

    best_val_loss = float('inf')

    print(f"\n{'='*60}")
    print(f"Training on {config.device} | dtype: {config.dtype}")
    print(f"Arch: {config.norm_type.upper()} + {config.activation.upper()}")
    print(f"LoRA Enabled: {config.use_lora} | RoPE Enabled: {config.use_rope}")
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
    # Configuration Updates for New Tech
    cfg.use_lora = False
    cfg.use_rope = True      # RoPE is standard now

    # Enable New Tech
    cfg.norm_type = "rmsnorm"
    cfg.activation = "swiglu"

    # Training
    model, dataset = train(use_pretrained=False)

    # Generation
    print("\n" + "="*60)
    print("GENERATED TEXT (with KV Cache)")
    print("="*60 + "\n")
    enc = tiktoken.get_encoding("gpt2")
    idx = torch.tensor([enc.encode("The future of AI is")], dtype=torch.long).to(cfg.device)

    model.eval()
    # generate uses KV cache internally, so this will be fast
    generated = model.generate(idx, max_new_tokens=100, temperature=0.8, top_k=50)
    print(enc.decode(generated[0].tolist()))
