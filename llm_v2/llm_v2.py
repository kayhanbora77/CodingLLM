# ==================================================
# Minimal but CORRECT GPT from scratch (executable)
# ==================================================

import math
import re
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==================================================
# Tokenization & Vocabulary
# ==================================================

def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def build_vocab(text: str):
    tokens = []
    for line in text.strip().splitlines():
        tokens.extend(tokenize(line))
        tokens.append("<eos>")

    counter = Counter(tokens)
    vocab = {w: i for i, (w, _) in enumerate(counter.most_common())}
    inv_vocab = {i: w for w, i in vocab.items()}
    return vocab, inv_vocab


def encode(text: str, vocab):
    encoded = []
    for line in text.strip().splitlines():
        encoded.extend(vocab[t] for t in tokenize(line))
        encoded.append(vocab["<eos>"])
    return encoded


# ==================================================
# Causal Self-Attention
# ==================================================

class CausalSelfAttention(nn.Module):
    def __init__(self, embed_size, num_heads, dropout, max_length):
        super().__init__()
        assert embed_size % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads

        self.qkv = nn.Linear(embed_size, 3 * embed_size)
        self.out_proj = nn.Linear(embed_size, embed_size)
        self.dropout = nn.Dropout(dropout)

        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(max_length, max_length))
            .unsqueeze(0)
            .unsqueeze(0)
        )

    def forward(self, x):
        B, T, C = x.shape

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = attn.masked_fill(
            self.causal_mask[:, :, :T, :T] == 0,
            float("-inf")
        )

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        return self.out_proj(out)


# ==================================================
# Transformer Block
# ==================================================

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, num_heads, dropout, max_length):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_size)
        self.ln2 = nn.LayerNorm(embed_size)

        self.attn = CausalSelfAttention(
            embed_size, num_heads, dropout, max_length
        )

        self.mlp = nn.Sequential(
            nn.Linear(embed_size, 4 * embed_size),
            nn.GELU(),
            nn.Linear(4 * embed_size, embed_size),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


# ==================================================
# GPT Model
# ==================================================

class GPT(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_size,
        num_layers,
        num_heads,
        max_length,
        dropout,
    ):
        super().__init__()

        self.token_emb = nn.Embedding(vocab_size, embed_size)
        self.pos_emb = nn.Embedding(max_length, embed_size)
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size, num_heads, dropout, max_length
                )
                for _ in range(num_layers)
            ]
        )

        self.ln_f = nn.LayerNorm(embed_size)
        self.lm_head = nn.Linear(embed_size, vocab_size, bias=False)

        # weight tying
        self.lm_head.weight = self.token_emb.weight

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)

        x = self.token_emb(idx) + self.pos_emb(pos)
        x = self.dropout(x)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        return self.lm_head(x)


# ==================================================
# Training
# ==================================================

def train(model, data, epochs=50, lr=3e-4):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.train()

    for epoch in range(1, epochs + 1):
        total_loss = 0.0

        for batch in data:
            inputs = batch[:, :-1]
            targets = batch[:, 1:]

            optimizer.zero_grad()
            logits = model(inputs)

            loss = criterion(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch:03d} | Loss: {total_loss / len(data):.4f}")


# ==================================================
# Sampling utilities
# ==================================================

def apply_repetition_penalty(logits, generated, penalty=1.1):  # Reduced penalty to allow more diversity
    for token in set(generated):
        if logits[token] > 0:
            logits[token] /= penalty
    return logits


def top_p_sampling(logits, p=0.9, temperature=1.0):  # Increased temperature for more variety
    logits = logits / temperature
    probs = F.softmax(logits, dim=-1)

    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=-1)

    cutoff = cumulative > p
    cutoff[..., 1:] = cutoff[..., :-1].clone()
    cutoff[..., 0] = False

    sorted_probs[cutoff] = 0
    sorted_probs /= sorted_probs.sum()

    return sorted_indices[torch.multinomial(sorted_probs, 1)]


# ==================================================
# Text Generation
# ==================================================

@torch.no_grad()
def generate_text(model, prompt, vocab, inv_vocab, max_len=100):  # Increased max length
    model.eval()
    device = next(model.parameters()).device

    input_ids = [vocab.get(w, vocab["<eos>"]) for w in tokenize(prompt)]
    inputs = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)

    generated = input_ids.copy()

    for _ in range(max_len):
        logits = model(inputs)
        next_logits = logits[0, -1, :]

        # Only apply repetition penalty after a certain number of tokens
        if len(generated) > 3:
            next_logits = apply_repetition_penalty(
                next_logits, generated
            )

        next_token = top_p_sampling(next_logits)
        token_id = next_token.item()

        if inv_vocab[token_id] == "<eos>":
            break

        generated.append(token_id)
        inputs = torch.cat([inputs, next_token.unsqueeze(0)], dim=1)

    return " ".join(inv_vocab[i] for i in generated)


# ==================================================
# Main
# ==================================================

def main():

    text = """
    the quick brown fox jumps over the lazy dog
    the fox is quick and the dog is lazy
    the dog sleeps while the fox runs fast
    transformers are powerful models for language
    language models learn patterns from data
    artificial intelligence is transforming technology
    neural networks can learn complex patterns
    the fox runs quickly through the forest
    dogs are loyal animals and cats are independent
    machine learning algorithms improve with more data
    deep learning models require substantial computation
    natural language processing enables human computer interaction
    the sun rises in the east and sets in the west
    birds fly high in the blue sky above
    fish swim in the deep ocean waters
    trees grow tall in the fertile forest soil
    humans communicate using spoken and written language
    computers process information using binary code
    mathematics provides the foundation for scientific discovery
    science helps us understand the natural world
    books contain knowledge passed from generation to generation
    music brings joy and emotion to human experience
    art expresses creativity and imagination in visual form
    sports bring people together in friendly competition
    cooking combines ingredients to create delicious meals
    travel exposes people to new cultures and places
    education opens minds to new ideas and possibilities
    """

    vocab, inv_vocab = build_vocab(text)
    encoded = encode(text, vocab)

    seq_len = 10
    dataset = [
        torch.tensor(encoded[i : i + seq_len + 1]).unsqueeze(0)
        for i in range(len(encoded) - seq_len)
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GPT(
        vocab_size=len(vocab),
        embed_size=64,  # Reduced embedding size to prevent memory issues
        num_layers=3,   # Reduced number of layers
        num_heads=4,    # Number of heads
        max_length=128,
        dropout=0.1,
    ).to(device)

    dataset = [d.to(device) for d in dataset]

    train(model, dataset, epochs=75)  # Reduced epochs to prevent overfitting with larger dataset

    print("\n--- Generated Text ---")
    print("Prompt: 'the fox is'")
    print("Generated:", generate_text(model, "the fox is", vocab, inv_vocab))

    print("\nPrompt: 'artificial intelligence'")
    print("Generated:", generate_text(model, "artificial intelligence", vocab, inv_vocab))

    print("\nPrompt: 'the sun'")
    print("Generated:", generate_text(model, "the sun", vocab, inv_vocab))


if __name__ == "__main__":
    main()
