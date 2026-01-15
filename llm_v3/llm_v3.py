# ==================================================
# nanoGPT-style GPT from scratch (TOKEN-LEVEL)
# ==================================================

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from collections import Counter

# ==================================================
# Config
# ==================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data.txt"

BATCH_SIZE = 16
BLOCK_SIZE = 64      # context length (tokens)
EMBED_SIZE = 256
NUM_LAYERS = 4
NUM_HEADS = 4
DROPOUT = 0.1
LR = 3e-4
TRAIN_STEPS = 5000
EVAL_INTERVAL = 500

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==================================================
# Load & tokenize data (WORD-LEVEL)
# ==================================================

text = DATA_PATH.read_text(encoding="utf-8").lower()

tokens = text.split()
print("Total tokens:", len(tokens))

# Build vocab
vocab = sorted(set(tokens))
vocab_size = len(vocab)
print("Vocab size:", vocab_size)

stoi = {tok: i for i, tok in enumerate(vocab)}
itos = {i: tok for tok, i in stoi.items()}

data = torch.tensor([stoi[t] for t in tokens], dtype=torch.long)

# Train/val split
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

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
# Causal Self-Attention
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

# ==================================================
# Transformer Block
# ==================================================

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

# ==================================================
# GPT Model
# ==================================================

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
            loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
        return logits, loss

# ==================================================
# Training
# ==================================================

model = GPT().to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

for step in range(TRAIN_STEPS):
    xb, yb = get_batch("train")
    _, loss = model(xb, yb)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    if step % EVAL_INTERVAL == 0:
        print(f"Step {step} | Loss {loss.item():.4f}")

# ==================================================
# Text Generation
# ==================================================

@torch.no_grad()
def generate(model, start, max_new_tokens=100, temperature=0.8):
    model.eval()

    words = start.lower().split()
    words = [w for w in words if w in stoi]

    if not words:
        words = [vocab[0]]

    idx = torch.tensor([[stoi[w] for w in words]], device=DEVICE)

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -BLOCK_SIZE:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature

        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, 1)
        idx = torch.cat([idx, next_id], dim=1)

    return " ".join(itos[i] for i in idx[0].tolist())

print("\n--- GENERATED TEXT ---\n")
print(generate(model, "once upon a time"))
