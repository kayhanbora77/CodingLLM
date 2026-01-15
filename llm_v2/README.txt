# GPT Implementation from Scratch - Version 2

This project implements a minimal but correct GPT (Generative Pre-trained Transformer) model from scratch using PyTorch. It demonstrates the core components of transformer architecture with causal self-attention mechanism.

## Overview

The implementation includes:
- Tokenization and vocabulary building
- Causal Self-Attention mechanism
- Transformer blocks with Layer Normalization and MLP layers
- Complete training loop with cross-entropy loss
- Text generation capabilities with sampling techniques
- Repetition penalty for more diverse text generation

## Architecture Components

### Tokenization & Vocabulary
- Uses regex-based tokenization to split text into words
- Builds vocabulary mappings (word to index and index to word)
- Handles end-of-sentence tokens with `<eos>`

### Causal Self-Attention
- Implements scaled dot-product attention with causal masking
- Supports multi-head attention mechanism
- Uses learned causal mask to prevent future information leakage
- Proper dimension handling for batch processing

### Transformer Block
- Consists of Layer Normalization, Attention, and MLP components
- Implements residual connections around attention and MLP layers
- Uses GELU activation in the MLP portion

### GPT Model
- Combines token embeddings with positional embeddings
- Stacks multiple transformer blocks
- Implements weight tying between token embeddings and LM head
- Processes sequences through all transformer layers

## Features

- **Training**: Includes a complete training loop with AdamW optimizer
- **Sampling**: Implements top-p (nucleus) sampling for text generation
- **Repetition Penalty**: Applies penalty to reduce repetitive generations
- **Temperature Control**: Allows control over generation randomness
- **Batch Processing**: Handles batched input for efficient training

## Usage

The script can be run directly to train the model on sample text and generate new text. The main function demonstrates:
1. Building vocabulary from sample text
2. Creating training batches
3. Training the model for 75 epochs
4. Generating text with various prompts

## Key Parameters

- Embedding size: 64 dimensions
- Number of transformer layers: 3
- Number of attention heads: 4
- Maximum sequence length: 128
- Dropout rate: 0.1
- Learning rate: 3e-4

## Dependencies

- Python 3.x
- PyTorch
- math, re, collections (standard library)

## Sample Output

The script generates text based on different prompts such as:
- "the fox is"
- "artificial intelligence"
- "the sun"

## Educational Purpose

This implementation is designed to be educational, showing the fundamental components of transformer models in a simplified but executable format. It demonstrates how each component works together to create a functioning language model.
