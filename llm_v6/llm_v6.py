"""
Enhanced GPT from Scratch
Improvements: Flash Attention, Better Init, Gradient Accumulation,
Checkpointing, Logging, and More Robust Training
"""

import math
import json
from pathlib import Path
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken

# ==================================================
# Configuration
# ==================================================

@dataclass
class Config:
    # Paths
    base_dir: Path = Path(__file__).resolve().parent
    data_path: Path = base_dir / "data.txt"
    ckpt_path: Path = base_dir / "best_model.pt"
    log_path: Path = base_dir / "training_log.jsonl"

    # Model
    vocab_size: int = 50257  # GPT-2 vocab size
    block_size: int = 128
    embed_size: int = 384
    num_layers: int = 6
    num_heads: int = 6
    dropout: float = 0.1
    bias: bool = False  # Use bias in Linear/LayerNorm

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
    compile_model: bool = False  # torch.compile (requires PyTorch 2.0+)
    dtype: str = "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16"

cfg = Config()

# ==================================================
# Data Loading
# ==================================================

class TextDataset:
    def __init__(self, data_path, train_split=0.9):
        text = data_path.read_text(encoding="utf-8")
        self.enc = tiktoken.get_encoding("gpt2")

        data = torch.tensor(self.enc.encode(text), dtype=torch.long)

        n = int(train_split * len(data))
        self.train_data = data[:n]
        self.val_data = data[n:]

        print(f"Dataset: {len(data):,} tokens | Train: {len(self.train_data):,} | Val: {len(self.val_data):,}")

    def get_batch(self, split, batch_size, block_size, device):
        data = self.train_data if split == "train" else self.val_data

        if len(data) <= block_size:
            raise ValueError(f"Data too short ({len(data)}) for block_size ({block_size})")

        ix = torch.randint(0, len(data) - block_size, (batch_size,))
        x = torch.stack([data[i:i + block_size] for i in ix])
        y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])

        return x.to(device), y.to(device)

# ==================================================
# Model Components
# ==================================================

class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with optional flash attention"""

    def __init__(self, config):
        super().__init__()
        assert config.embed_size % config.num_heads == 0

        self.num_heads = config.num_heads
        self.head_dim = config.embed_size // config.num_heads
        self.embed_size = config.embed_size

        # QKV projection
        self.c_attn = nn.Linear(config.embed_size, 3 * config.embed_size, bias=config.bias)
        self.c_proj = nn.Linear(config.embed_size, config.embed_size, bias=config.bias)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Causal mask
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.block_size, config.block_size))
            .view(1, 1, config.block_size, config.block_size)
        )

        # Flash attention flag
        self.flash = hasattr(F, 'scaled_dot_product_attention')

    def forward(self, x):
        B, T, C = x.shape

        # QKV projection and split
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.embed_size, dim=2)

        # Reshape for multi-head attention
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention
        if self.flash:
            # Use PyTorch's flash attention
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
                is_causal=True
            )
        else:
            # Manual attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
            att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        # Reassemble heads
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    """Feed-forward network"""

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.embed_size, 4 * config.embed_size, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.embed_size, config.embed_size, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """Transformer block"""

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
    """GPT Language Model"""

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict({
            'wte': nn.Embedding(config.vocab_size, config.embed_size),
            'wpe': nn.Embedding(config.block_size, config.embed_size),
            'drop': nn.Dropout(config.dropout),
            'h': nn.ModuleList([Block(config) for _ in range(config.num_layers)]),
            'ln_f': nn.LayerNorm(config.embed_size, bias=config.bias),
        })

        self.lm_head = nn.Linear(config.embed_size, config.vocab_size, bias=False)

        # Weight tying
        self.transformer.wte.weight = self.lm_head.weight

        # Initialize weights
        self.apply(self._init_weights)

        # Apply special scaled init to residual projections
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.num_layers))

        # Report number of parameters
        print(f"Model parameters: {self.get_num_params() / 1e6:.2f}M")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self, non_embedding=False):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def forward(self, idx, targets=None):
        device = idx.device
        B, T = idx.shape

        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is {self.config.block_size}"

        pos = torch.arange(0, T, dtype=torch.long, device=device)

        # Forward through transformer
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # Inference optimization: only compute logits for last position
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None):
        """Generate new tokens"""
        for _ in range(max_new_tokens):
            # Crop context if needed
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]

            # Forward
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            # Top-k sampling
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')

            # Top-p (nucleus) sampling
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float('inf')

            # Sample
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

# ==================================================
# Training Utilities
# ==================================================

def get_lr(step, config):
    """Cosine learning rate schedule with warmup"""
    if step < config.warmup_steps:
        return config.learning_rate * step / config.warmup_steps

    if step > config.max_steps:
        return config.min_lr

    decay_ratio = (step - config.warmup_steps) / (config.max_steps - config.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)


@torch.no_grad()
def estimate_loss(model, dataset, config):
    """Estimate loss on train and val sets"""
    model.eval()
    losses = {}

    for split in ['train', 'val']:
        losses_list = []
        for _ in range(config.eval_iters):
            x, y = dataset.get_batch(split, config.batch_size, config.block_size, config.device)
            with torch.amp.autocast(device_type=config.device, dtype=getattr(torch, cfg.dtype)):
                _, loss = model(x, y)
            losses_list.append(loss.item())
        losses[split] = sum(losses_list) / len(losses_list)

    model.train()
    return losses


def log_metrics(step, metrics, config):
    """Log metrics to file"""
    with open(config.log_path, 'a') as f:
        f.write(json.dumps({'step': step, **metrics}) + '\n')

# ==================================================
# Main Training Loop
# ==================================================

def train():
    # Setup
    dataset = TextDataset(cfg.data_path)
    cfg.vocab_size = dataset.enc.n_vocab

    model = GPT(cfg).to(cfg.device)

    if cfg.compile_model:
        print("Compiling model...")
        model = torch.compile(model)

    # Optimizer
    optimizer = model.configure_optimizers(cfg.weight_decay, cfg.learning_rate, (cfg.beta1, cfg.beta2), cfg.device)
    if optimizer is None:
        # Fallback optimizer configuration
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.learning_rate,
            betas=(cfg.beta1, cfg.beta2),
            weight_decay=cfg.weight_decay
        )

    # GradScaler for mixed precision
    scaler = torch.amp.GradScaler(enabled=(cfg.dtype == 'float16'))

    # Training state
    best_val_loss = float('inf')
    patience_counter = 0

    print(f"\n{'='*60}")
    print(f"Training on {cfg.device} | dtype: {cfg.dtype}")
    print(f"Effective batch size: {cfg.batch_size * cfg.grad_accum_steps}")
    print(f"{'='*60}\n")

    # Training loop
    model.train()
    for step in range(cfg.max_steps):
        # Update learning rate
        lr = get_lr(step, cfg)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Gradient accumulation
        for micro_step in range(cfg.grad_accum_steps):
            x, y = dataset.get_batch('train', cfg.batch_size, cfg.block_size, cfg.device)

            with torch.amp.autocast(device_type=cfg.device, dtype=getattr(torch, cfg.dtype)):
                _, loss = model(x, y)
                loss = loss / cfg.grad_accum_steps

            scaler.scale(loss).backward()

        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

        # Optimizer step
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        # Evaluation
        if step % cfg.eval_interval == 0 or step == cfg.max_steps - 1:
            losses = estimate_loss(model, dataset, cfg)

            print(f"Step {step:5d} | LR {lr:.2e} | "
                  f"Train {losses['train']:.4f} | Val {losses['val']:.4f}")

            log_metrics(step, {'lr': lr, **losses}, cfg)

            # Early stopping & checkpointing
            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                patience_counter = 0

                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'config': cfg,
                    'step': step,
                    'best_val_loss': best_val_loss,
                }
                torch.save(checkpoint, cfg.ckpt_path)
                print(f"✓ Checkpoint saved (val loss: {best_val_loss:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= cfg.patience:
                    print(f"\n⛔ Early stopping at step {step}")
                    break

    print(f"\n{'='*60}")
    print(f"Training complete! Best val loss: {best_val_loss:.4f}")
    print(f"{'='*60}\n")

    return model, dataset


# ==================================================
# Generation
# ==================================================

def generate_text(prompt="Once upon a time", max_tokens=200, temperature=0.8, top_k=50):
    """Generate text from a trained model"""
    # Load model
    checkpoint = torch.load(cfg.ckpt_path, map_location=cfg.device)

    model = GPT(checkpoint['config']).to(cfg.device)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    # Encode prompt
    enc = tiktoken.get_encoding("gpt2")
    idx = torch.tensor([enc.encode(prompt)], dtype=torch.long, device=cfg.device)

    # Generate
    with torch.no_grad():
        generated = model.generate(idx, max_tokens, temperature=temperature, top_k=top_k)

    return enc.decode(generated[0].tolist())


# Add optimizer configuration method to GPT class
def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
    # Separate parameters that should and shouldn't be decayed
    decay = set()
    no_decay = set()
    whitelist_weight_modules = (nn.Linear, )
    blacklist_weight_modules = (nn.LayerNorm, nn.Embedding)

    for mn, m in self.named_modules():
        for pn, p in m.named_parameters():
            fpn = f'{mn}.{pn}' if mn else pn

            if pn.endswith('bias'):
                no_decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                no_decay.add(fpn)

    param_dict = {pn: p for pn, p in self.named_parameters()}
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert len(inter_params) == 0, f"Parameters {inter_params} in both decay/no_decay sets"
    assert len(param_dict.keys() - union_params) == 0, "Missing parameters"

    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    ]

    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
    return optimizer

GPT.configure_optimizers = configure_optimizers

# ==================================================
# Main
# ==================================================

if __name__ == "__main__":
    # Train
    model, dataset = train()

    # Generate sample
    print("\n" + "="*60)
    print("GENERATED TEXT")
    print("="*60 + "\n")
    print(generate_text("Once upon a time", max_tokens=200, temperature=0.8, top_k=50))
