# Enhanced GPT from Scratch (with nanoGPT-style Pretraining Support)

This implementation provides an advanced GPT model built from scratch, featuring Flash Attention, improved initialization, gradient accumulation, comprehensive checkpointing, logging, and support for loading pre-trained weight from GPT-2 models.

## Overview

This script implements an enhanced GPT model with:
- BPE tokenization using tiktoken (GPT-2 encoding)
- Flash Attention for improved efficiency
- Support for loading pre-trained GPT-2 weights
- Improved weight initialization strategies
- Gradient accumulation for larger effective batch sizes
- Comprehensive model checkpointing and logging
- More robust training with early stopping
- Advanced text generation with multiple sampling strategies

## Key Improvements Over Previous Versions

- **Pre-trained Weight Loading**: Capability to load weights from pre-trained GPT-2 models (HuggingFace)
- **NanoGPT Compatibility**: Architecture closely matches Karpathy's nanoGPT implementation
- **Flexible Configuration**: Supports both custom model sizes and GPT-2 compatible dimensions
- **Graceful Fallback**: Handles missing data files gracefully with dummy data
- **Enhanced Pre-training Support**: Ready for fine-tuning pre-trained models

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
- Pre-trained weight loading capability

## Configuration

The model uses the following default hyperparameters (Option A - Custom model):
- Vocab Size: 50,257 (GPT-2 vocab size)
- Block Size (Context Length): 128 tokens
- Embedding Size: 384 dimensions
- Number of Layers: 6
- Number of Attention Heads: 6

Alternative configuration (Option B - GPT-2 compatible, commented out):
- Vocab Size: 50,257
- Block Size: 1024 tokens
- Embedding Size: 768 dimensions
- Number of Layers: 12
- Number of Attention Heads: 12

Additional parameters:
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
- Gracefully falls back to dummy data if data file is missing

## Training Process

- Uses AdamW optimizer with configurable parameters
- Implements gradient clipping (norm = 1.0) for stability
- Applies learning rate scheduling with warmup and cosine annealing
- Enables gradient accumulation for larger effective batch sizes
- Uses mixed precision training for efficiency
- Logs both training and validation loss at regular intervals
- Implements early stopping based on validation loss plateau
- Saves comprehensive checkpoints with training metadata
- Supports model compilation for additional performance gains
- Optionally loads pre-trained GPT-2 weights for fine-tuning

## Text Generation

- Generates text continuations from given prompts
- Implements temperature-based sampling for diversity
- Supports top-k sampling for controlled generation
- Uses the same BPE tokenizer for encoding/decoding
- Supports variable-length text generation
- Includes context cropping to handle long sequences efficiently

## Pre-trained Weight Loading

- Supports loading pre-trained weights from HuggingFace GPT-2 models
- Compatible with 'gpt2', 'gpt2-medium', 'gpt2-large', and 'gpt2-xl' models
- Requires matching architecture dimensions for successful loading
- Includes parameter mapping between HuggingFace and custom implementations
- Handles Conv1D to Linear weight transformation automatically

## Model Checkpointing and Logging

- Automatically saves checkpoints to `best_model.pt`
- Checkpoints include model state and configuration
- Model saved when validation loss improves
- Helps prevent overfitting and enables training resumption

## Usage

To run the script with training from scratch:
```bash
python llm_v7.py
```

To fine-tune with pre-trained weights, uncomment and modify the Config class to match GPT-2 dimensions and change the train call:
```python
# In the main section, change to:
model, dataset = train(use_pretrained=True)
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
- transformers (optional, only required for pre-trained weight loading)

## Key Features

- Complete implementation from scratch with minimal external dependencies
- Integration with GPT-2's BPE tokenizer for superior text handling
- Flash Attention for improved performance
- Gradient accumulation for larger effective batch sizes
- Pre-trained weight loading capability for fine-tuning
- Advanced training techniques (early stopping, LR scheduling)
- Mixed precision training for efficiency
- Automatic model checkpointing
- Comprehensive training logging
- Robust parameter management with separate weight decay
- Efficient batching system for training
- Proper causal masking for autoregressive training
- Automatic hardware acceleration (GPU if available)
- Model compilation support for additional performance
- Clean, readable code structure with clear separation of components
- Multiple sampling strategies (temperature, top-k) for text generation
- Flexible configuration supporting both custom and GPT-2-compatible models

## Educational Value

This implementation serves as an excellent learning resource for understanding:
- Transformer architecture fundamentals
- BPE tokenization and its advantages over word-level tokenization
- Flash Attention and its performance benefits
- Pre-trained model loading and fine-tuning techniques
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
- nanoGPT-style implementation patterns
- Pre-trained model integration
