# Enhanced GPT from Scratch

This implementation provides an advanced GPT model built from scratch, featuring Flash Attention, improved initialization, gradient accumulation, comprehensive checkpointing, logging, and more robust training procedures.

## Overview

This script implements an enhanced GPT model with:
- BPE tokenization using tiktoken (GPT-2 encoding)
- Flash Attention for improved efficiency
- Improved weight initialization strategies
- Gradient accumulation for larger effective batch sizes
- Comprehensive model checkpointing and logging
- More robust training with early stopping
- Advanced text generation with multiple sampling strategies

## Key Improvements Over Previous Versions

- **Flash Attention**: Utilizes PyTorch's optimized scaled dot product attention for faster training and inference
- **Gradient Accumulation**: Simulates larger batch sizes by accumulating gradients over multiple mini-batches
- **Improved Weight Initialization**: Better initialization strategies for more stable training
- **Model Compilation**: Support for torch.compile for additional performance gains
- **Comprehensive Logging**: Tracks training metrics in JSONL format
- **Robust Parameter Management**: Separates parameters that should/shouldn't use weight decay
- **Advanced Sampling**: Supports both top-k and top-p (nucleus) sampling for text generation

## Architecture Components

### Causal Self-Attention
- Multi-head attention mechanism with configurable head count
- Causal masking to ensure tokens only attend to previous positions
- Uses Flash Attention when available for improved performance
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
- Special initialization for residual projections

## Configuration

The model uses the following default hyperparameters:
- Vocab Size: 50,257 (GPT-2 vocab size)
- Block Size (Context Length): 128 tokens
- Embedding Size: 384 dimensions
- Number of Layers: 6
- Number of Attention Heads: 6
- Dropout Rate: 0.1
- Batch Size: 12
- Gradient Accumulation Steps: 4 (effective batch size = 48)
- Max Training Steps: 10,000
- Evaluation Interval: Every 500 steps
- Evaluation Iterations: 100
- Learning Rate: 6e-4
- Minimum Learning Rate: 6e-5
- Warmup Steps: 500
- Weight Decay: 0.1
- Beta1: 0.9
- Beta2: 0.95
- Gradient Clip: 1.0
- Early Stopping Patience: 5 evaluations
- Device: Automatic detection (CUDA if available, otherwise CPU)
- Data Type: bfloat16 or float16 for mixed precision training

## Data Processing

- Reads text data from `data.txt` in the same directory
- Uses tiktoken's GPT-2 encoder for BPE tokenization
- Automatically determines vocabulary size from tiktoken
- Creates train/validation split (90%/10%)
- Includes safety checks to ensure data is sufficient for block size

## Training Process

- Uses AdamW optimizer with configurable parameters
- Implements gradient clipping (norm = 1.0) for stability
- Applies learning rate scheduling with warmup and cosine annealing
- Enables gradient accumulation for larger effective batch sizes
- Uses mixed precision training for efficiency
- Logs both training and validation loss at regular intervals
- Implements early stopping based on validation loss plateau
- Saves comprehensive checkpoints with optimizer state
- Supports model compilation for additional performance gains

## Text Generation

- Generates text continuations from given prompts
- Implements temperature-based sampling for diversity
- Supports both top-k and top-p (nucleus) sampling
- Uses the same BPE tokenizer for encoding/decoding
- Supports variable-length text generation
- Includes context cropping to handle long sequences efficiently

## Model Checkpointing and Logging

- Automatically saves comprehensive checkpoints to `best_model.pt`
- Checkpoints include model state, optimizer state, and training metadata
- Model saved when validation loss improves
- Maintains training logs in JSONL format to `training_log.jsonl`
- Helps prevent overfitting and enables training resumption

## Usage

To run the script:
```bash
python llm_v6.py
```

The script will:
1. Load and preprocess the data from `data.txt` using BPE tokenization
2. Initialize and train the GPT model with all advanced features
3. Generate sample text based on the prompt "Once upon a time"

## Requirements

- Python 3.x
- PyTorch (version 2.0+ recommended for torch.compile support)
- tiktoken (for BPE tokenization)
- pathlib (for file handling)
- dataclasses (for configuration management)

## Key Features

- Complete implementation from scratch with minimal external dependencies
- Integration with GPT-2's BPE tokenizer for superior text handling
- Flash Attention for improved performance
- Gradient accumulation for larger effective batch sizes
- Advanced training techniques (early stopping, LR scheduling)
- Mixed precision training for efficiency
- Automatic model checkpointing with optimizer state
- Comprehensive training logging
- Robust parameter management with separate weight decay
- Efficient batching system for training
- Proper causal masking for autoregressive training
- Automatic hardware acceleration (GPU if available)
- Model compilation support for additional performance
- Clean, readable code structure with clear separation of components
- Multiple sampling strategies (temperature, top-k, top-p) for text generation

## Educational Value

This implementation serves as an excellent learning resource for understanding:
- Transformer architecture fundamentals
- BPE tokenization and its advantages over word-level tokenization
- Flash Attention and its performance benefits
- Advanced training techniques (gradient accumulation, early stopping, LR scheduling)
- Mixed precision training for deep learning
- Model checkpointing and logging strategies
- Weight initialization strategies for deep networks
- Parameter group management in optimizers
- Causal attention mechanisms
- Language model training procedures
- Text tokenization and preprocessing with advanced techniques
- PyTorch implementation patterns for NLP models
- Hardware acceleration for deep learning models
- Performance optimization techniques for training
