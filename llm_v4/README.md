# REAL GPT from Scratch (nanoGPT-style with BPE/tiktoken)

This implementation provides a complete GPT model built from scratch using the nanoGPT style, featuring Byte-Pair Encoding (BPE) tokenization with tiktoken for improved text processing capabilities.

## Overview

This script implements a complete GPT model with:
- BPE tokenization using tiktoken (GPT-2 encoding)
- Causal self-attention mechanism
- Multi-layer transformer architecture
- Complete training pipeline
- Text generation capabilities
- GPU acceleration support

## Key Improvements Over Previous Versions

- **BPE Tokenization**: Uses tiktoken with GPT-2 encoding instead of simple word-level tokenization
- **Larger Vocabulary**: Access to GPT-2's extensive vocabulary (~50,000 tokens)
- **Better Text Representation**: BPE handles out-of-vocabulary words more effectively

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
- Device: Automatic detection (CUDA if available, otherwise CPU)

## Data Processing

- Reads text data from `data.txt` in the same directory
- Uses tiktoken's GPT-2 encoder for BPE tokenization
- Automatically determines vocabulary size from tiktoken
- Creates train/validation split (90%/10%)

## Training Process

- Uses AdamW optimizer with default parameters
- Implements gradient clipping (norm = 1.0) for stability
- Logs training loss at regular intervals
- Processes data in batches with configurable size
- Supports both CPU and GPU training

## Text Generation

- Generates text continuations from given prompts
- Implements temperature-based sampling for diversity
- Uses the same BPE tokenizer for encoding/decoding
- Supports variable-length text generation

## Usage

To run the script:
```bash
python llm_v4.py
```

The script will:
1. Load and preprocess the data from `data.txt` using BPE tokenization
2. Initialize and train the GPT model
3. Generate sample text based on the prompt "Once upon a time"

## Requirements

- Python 3.x
- PyTorch
- tiktoken (for BPE tokenization)
- pathlib (for file handling)

## Key Features

- Complete implementation from scratch with minimal external dependencies
- Integration with GPT-2's BPE tokenizer for superior text handling
- Efficient batching system for training
- Proper causal masking for autoregressive training
- Automatic hardware acceleration (GPU if available)
- Clean, readable code structure with clear separation of components
- Comprehensive logging during training
- Temperature-controlled text generation

## Educational Value

This implementation serves as an excellent learning resource for understanding:
- Transformer architecture fundamentals
- BPE tokenization and its advantages over word-level tokenization
- Causal attention mechanisms
- Language model training procedures
- Text tokenization and preprocessing with advanced techniques
- PyTorch implementation patterns for NLP models
- Hardware acceleration for deep learning models
