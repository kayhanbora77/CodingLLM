# nanoGPT-style GPT from Scratch (Token-Level)

This implementation provides a minimal yet complete GPT model built from scratch, following the nanoGPT style. It focuses on token-level processing with word-level tokenization.

## Overview

This script implements a complete GPT model with:
- Word-level tokenization for text processing
- Causal self-attention mechanism
- Multi-layer transformer architecture
- Complete training pipeline
- Text generation capabilities

## Architecture Components

### Causal Self-Attention
- Multi-head attention mechanism with configurable head count
- Causal masking to ensure tokens only attend to previous positions
- Proper dimension handling for batch processing
- Dropout regularization for preventing overfitting

### Transformer Block
- Layer normalization before attention and feed-forward layers
- Residual connections around attention and MLP components
- Multi-layer perceptron with GELU activation
- Configurable embedding dimensions

### GPT Model
- Token embeddings combined with positional embeddings
- Sequential transformer blocks
- Weight tying between token embeddings and output head
- Cross-entropy loss calculation for training

## Configuration

The model uses the following default hyperparameters:
- Batch Size: 16
- Block Size (Context Length): 64 tokens
- Embedding Size: 256 dimensions
- Number of Layers: 4
- Number of Attention Heads: 4
- Dropout Rate: 0.1
- Learning Rate: 3e-4
- Training Steps: 5000
- Evaluation Interval: Every 500 steps

## Data Processing

- Reads text data from `data.txt` in the same directory
- Converts text to lowercase for consistency
- Performs word-level tokenization by splitting on whitespace
- Builds vocabulary from unique tokens in the dataset
- Creates train/validation split (90%/10%)

## Training Process

- Uses AdamW optimizer with default parameters
- Implements gradient clipping (norm = 1.0) for stability
- Logs training loss at regular intervals
- Processes data in batches with configurable size

## Text Generation

- Generates text continuations from given prompts
- Implements temperature-based sampling for diversity
- Supports variable-length text generation
- Uses learned vocabulary for token-to-text conversion

## Usage

To run the script:
```bash
python llm_v3.py
```

The script will:
1. Load and preprocess the data from `data.txt`
2. Build the vocabulary and prepare training data
3. Initialize and train the GPT model
4. Generate sample text based on the prompt "once upon a time"

## Requirements

- Python 3.x
- PyTorch
- pathlib (for file handling)
- Collections (for Counter utility)

## Key Features

- Complete implementation from scratch with no external dependencies
- Efficient batching system for training
- Proper causal masking for autoregressive training
- Clean, readable code structure with clear separation of components
- Comprehensive logging during training
- Temperature-controlled text generation

## Educational Value

This implementation serves as an excellent learning resource for understanding:
- Transformer architecture fundamentals
- Causal attention mechanisms
- Language model training procedures
- Text tokenization and preprocessing
- PyTorch implementation patterns for NLP models
