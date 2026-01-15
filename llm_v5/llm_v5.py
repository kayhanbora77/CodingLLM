# ==================================================
# REAL GPT from scratch (nanoGPT-style, IMPROVED)
# BPE (tiktoken) + Early Stopping + LR Schedule
# ==================================================

import math
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken

# ==================================================
# Config
# ==================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data.txt"
CKPT_PATH = BASE_DIR / "best_model.pt"

BATCH_SIZE = 16
BLOCK_SIZE = 64
EMBED_SIZE = 256
NUM_LAYERS = 4
NUM_HEADS = 4
DROPOUT = 0.1

LR = 3e-4
MIN_LR = 3e-5
WARMUP_STEPS = 200
TRAIN_STEPS = 10000
EVAL_INTERVAL = 500
EVAL_ITERS = 100
PATIENCE = 3  # early stopping patience

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = DEVICE == "cuda"

# ==================================================
# Tokenizer + Data
# ==================================================

text = DATA_PATH.read_text(encoding="utf-8")
enc = tiktoken.get_encoding("gpt2")

data = torch.tensor(enc.encode(text), dtype=torch.long)
vocab_size = enc.n_vocab

n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

print("Vocab:", vocab_size, "| Tokens:", len(data))

# ==================================================
# Batching
# ==================================================

def get_batch(split):
    source = train_data if split == "train" else val_data
    ix = torch.randint(0, len(source) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([source[i:i + BLOCK_SIZE] for i in ix])
    y = torch.stack([source[i + 1:i + BLOCK_SIZE + 1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)

# ==================================================
# Model
# ==================================================

class CausalSelfAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        super().__init__()
        assert embed_size % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads

        self.qkv = nn.Linear(embed_size, 3 * embed_size)
        self.proj = nn.Linear(embed_size, embed_size)
        self.dropout = nn.Dropout(DROPOUT)

        mask = torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE))
        self.register_buffer("mask", mask)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att.masked_fill(self.mask[:T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)

        out = att @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(EMBED_SIZE)
        self.ln2 = nn.LayerNorm(EMBED_SIZE)

        self.attn = CausalSelfAttention(EMBED_SIZE, NUM_HEADS)
        self.mlp = nn.Sequential(
            nn.Linear(EMBED_SIZE, 4 * EMBED_SIZE),
            nn.GELU(),
            nn.Linear(4 * EMBED_SIZE, EMBED_SIZE),
            nn.Dropout(DROPOUT),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self):
        super().__init__()

        self.token_emb = nn.Embedding(vocab_size, EMBED_SIZE)
        self.pos_emb = nn.Embedding(BLOCK_SIZE, EMBED_SIZE)

        self.blocks = nn.Sequential(*[Block() for _ in range(NUM_LAYERS)])
        self.ln_f = nn.LayerNorm(EMBED_SIZE)

        self.head = nn.Linear(EMBED_SIZE, vocab_size, bias=False)
        self.head.weight = self.token_emb.weight  # weight tying

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)

        x = self.token_emb(idx) + self.pos_emb(pos)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, vocab_size),
                                   targets.view(-1))
        return logits, loss

# ==================================================
# Helpers
# ==================================================

@torch.no_grad()
def estimate_loss(model):
    model.eval()
    out = {}
    for split in ["train", "val"]:
        losses = torch.zeros(EVAL_ITERS)
        for i in range(EVAL_ITERS):
            xb, yb = get_batch(split)
            _, loss = model(xb, yb)
            losses[i] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

def get_lr(step):
    if step < WARMUP_STEPS:
        return LR * step / WARMUP_STEPS
    progress = (step - WARMUP_STEPS) / (TRAIN_STEPS - WARMUP_STEPS)
    return MIN_LR + 0.5 * (LR - MIN_LR) * (1 + math.cos(math.pi * progress))

# ==================================================
# Training
# ==================================================

model = GPT().to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

best_val = float("inf")
bad_epochs = 0

print("Training on", DEVICE)

for step in range(TRAIN_STEPS):
    lr = get_lr(step)
    for pg in optimizer.param_groups:
        pg["lr"] = lr

    xb, yb = get_batch("train")

    with torch.cuda.amp.autocast(enabled=USE_AMP):
        _, loss = model(xb, yb)

    optimizer.zero_grad()
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()

    if step % EVAL_INTERVAL == 0:
        losses = estimate_loss(model)
        print(f"Step {step:5d} | "
              f"Train {losses['train']:.4f} | "
              f"Val {losses['val']:.4f}")

        if losses["val"] < best_val:
            best_val = losses["val"]
            bad_epochs = 0
            torch.save(model.state_dict(), CKPT_PATH)
            print("✓ Saved best model")
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                print("⛔ Early stopping triggered")
                break

# ==================================================
# Generation
# ==================================================

@torch.no_grad()
def generate(model, prompt, max_new_tokens=200, temperature=0.8, top_k=50):
    model.eval()

    idx = torch.tensor([enc.encode(prompt)],
                       device=DEVICE, dtype=torch.long)

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -BLOCK_SIZE:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature

        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float("inf")

        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, 1)
        idx = torch.cat([idx, next_id], dim=1)

    return enc.decode(idx[0].tolist())

print("\n--- GENERATED TEXT ---\n")
print(generate(model, "Once upon a time"))
