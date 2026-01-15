# REAL GPT from Scratch (nanoGPT-style, IMPROVED)

This implementation provides an enhanced GPT model built from scratch using the nanoGPT style, featuring Byte-Pair Encoding (BPE) tokenization with tiktoken and advanced training techniques including early stopping and learning rate scheduling.

## Overview

This script implements an improved GPT model with:
- BPE tokenization using tiktoken (GPT-2 encoding)
- Causal self-attention mechanism
- Multi-layer transformer architecture
- Advanced training features: early stopping, learning rate scheduling
- Mixed precision training support
- Model checkpointing
- Complete text generation capabilities

## Key Improvements Over Previous Versions

- **Early Stopping**: Prevents overfitting by monitoring validation loss and stopping training when performance plateaus
- **Learning Rate Scheduling**: Implements cosine annealing with warmup for better convergence
- **Mixed Precision Training**: Uses AMP (Automatic Mixed Precision) for faster training and reduced memory usage
- **Model Checkpointing**: Saves the best model based on validation performance
- **Comprehensive Evaluation**: Regular validation loss estimation during training

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
- Initial Learning Rate: 3e-4
- Minimum Learning Rate: 3e-5
- Warmup Steps: 200
- Training Steps: 10,000
- Evaluation Interval: Every 500 steps
- Evaluation Iterations: 100
- Early Stopping Patience: 3 evaluations
- Device: Automatic detection (CUDA if available, otherwise CPU)
- Mixed Precision: Enabled when CUDA is available

## Data Processing

- Reads text data from `data.txt` in the same directory
- Uses tiktoken's GPT-2 encoder for BPE tokenization
- Automatically determines vocabulary size from tiktoken
- Creates train/validation split (90%/10%)

## Training Process

- Uses AdamW optimizer with configurable parameters
- Implements gradient clipping (norm = 1.0) for stability
- Applies learning rate scheduling with warmup and cosine annealing
- Enables mixed precision training for efficiency
- Logs both training and validation loss at regular intervals
- Implements early stopping based on validation loss plateau
- Saves the best model based on validation performance

## Text Generation

- Generates text continuations from given prompts
- Implements temperature-based sampling for diversity
- Adds top-k sampling for more controlled generation
- Uses the same BPE tokenizer for encoding/decoding
- Supports variable-length text generation (default: 200 tokens)

## Model Checkpointing

- Automatically saves the best model to `best_model.pt`
- Model saved when validation loss improves
- Helps prevent overfitting and enables model recovery

## Usage

To run the script:
```bash
python llm_v5.py
```

The script will:
1. Load and preprocess the data from `data.txt` using BPE tokenization
2. Initialize and train the GPT model with early stopping and learning rate scheduling
3. Generate sample text based on the prompt "Once upon a time"

## Requirements

- Python 3.x
- PyTorch
- tiktoken (for BPE tokenization)
- pathlib (for file handling)

## Key Features

- Complete implementation from scratch with minimal external dependencies
- Integration with GPT-2's BPE tokenizer for superior text handling
- Advanced training techniques (early stopping, LR scheduling)
- Mixed precision training for efficiency
- Automatic model checkpointing
- Efficient batching system for training
- Proper causal masking for autoregressive training
- Automatic hardware acceleration (GPU if available)
- Clean, readable code structure with clear separation of components
- Comprehensive logging during training
- Multiple sampling strategies (temperature, top-k) for text generation

## Educational Value

This implementation serves as an excellent learning resource for understanding:
- Transformer architecture fundamentals
- BPE tokenization and its advantages over word-level tokenization
- Advanced training techniques (early stopping, learning rate scheduling)
- Mixed precision training for deep learning
- Model checkpointing and validation strategies
- Causal attention mechanisms
- Language model training procedures
- Text tokenization and preprocessing with advanced techniques
- PyTorch implementation patterns for NLP models
- Hardware acceleration for deep learning models
