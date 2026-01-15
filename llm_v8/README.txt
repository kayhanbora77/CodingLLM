# Enhanced GPT from Scratch (with LoRA and RoPE)

This implementation provides the most advanced GPT model built from scratch, featuring Low-Rank Adaptation (LoRA), Rotary Positional Embeddings (RoPE), Flash Attention, improved initialization, gradient accumulation, comprehensive checkpointing, logging, and support for loading pre-trained weights from GPT-2 models.

## Overview

This script implements an enhanced GPT model with:
- BPE tokenization using tiktoken (GPT-2 encoding)
- Flash Attention for improved efficiency
- Low-Rank Adaptation (LoRA) for parameter-efficient fine-tuning
- Rotary Positional Embeddings (RoPE) as an alternative to absolute positional embeddings
- Support for loading pre-trained GPT-2 weights
- Improved weight initialization strategies
- Gradient accumulation for larger effective batch sizes
- Comprehensive model checkpointing and logging
- More robust training with early stopping
- Advanced text generation with multiple sampling strategies

## Key Improvements Over Previous Versions

- **Low-Rank Adaptation (LoRA)**: Parameter-efficient fine-tuning technique that adds trainable low-rank matrices to existing weights
- **Rotary Positional Embeddings (RoPE)**: Alternative to absolute positional embeddings that provides better generalization
- **Parameter-Efficient Training**: LoRA enables efficient fine-tuning with significantly fewer trainable parameters
- **Enhanced Architecture Flexibility**: Toggle between absolute positional embeddings and RoPE
- **Advanced Parameter Management**: Automatic freezing of base model parameters when using LoRA
- **Trainable Parameter Tracking**: Reports percentage of trainable parameters when using LoRA

## Architecture Components

### Causal Self-Attention
- Multi-head attention mechanism with configurable head count
- Causal masking to ensure tokens only attend to previous positions
- Uses Flash Attention when available for improved performance
- Supports both standard linear layers and LoRA-enhanced layers
- Supports both absolute positional embeddings and RoPE
- Proper dimension handling for batch processing
- Dropout regularization for preventing overfitting

### Transformer Block
- Layer normalization before attention and feed-forward layers
- Residual connections around attention and MLP components
- Multi-layer perceptron with GELU activation
- Configurable embedding dimensions

### GPT Model
- Token embeddings combined with either positional embeddings or RoPE
- Sequential transformer blocks
- Weight tying between token embeddings and output head
- Cross-entropy loss calculation for training
- Special initialization for residual projections
- Pre-trained weight loading capability
- LoRA parameter freezing when enabled

## Configuration

The model uses the following default hyperparameters:
- Vocab Size: 50,257 (GPT-2 vocab size)
- Block Size (Context Length): 128 tokens
- Embedding Size: 384 dimensions
- Number of Layers: 6
- Number of Attention Heads: 6

### LoRA Settings:
- LoRA Enabled: False (set to True to enable)
- LoRA Rank (r): 8
- LoRA Alpha: 16.0
- LoRA Dropout: 0.05
- LoRA Target Modules: ["c_attn", "c_proj", "c_fc"]

### RoPE Settings:
- RoPE Enabled: False (set to True to enable)
- RoPE Frequency Base: 10000

Additional parameters:
- Dropout Rate: 0.1
- Bias: False
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
- Supports LoRA training with automatic parameter freezing

## Text Generation

- Generates text continuations from given prompts
- Implements temperature-based sampling for diversity
- Supports top-k sampling for controlled generation
- Uses the same BPE tokenizer for encoding/decoding
- Supports variable-length text generation
- Includes context cropping to handle long sequences efficiently

## LoRA (Low-Rank Adaptation)

- Parameter-efficient fine-tuning technique that adds trainable low-rank matrices
- Significantly reduces the number of trainable parameters
- Automatically freezes base model parameters when enabled
- Configurable rank (r) and scaling factor (alpha)
- Targets specific modules (attention and feed-forward layers)
- Reports trainable parameter statistics

## RoPE (Rotary Positional Embeddings)

- Alternative to absolute positional embeddings
- Provides better generalization for sequence lengths beyond training
- Implemented as rotary position embeddings that rotate query and key vectors
- Replaces absolute positional embeddings when enabled
- Uses frequency base of 10000 by default

## Pre-trained Weight Loading

- Supports loading pre-trained weights from HuggingFace GPT-2 models
- Requires matching architecture dimensions for successful loading
- Includes considerations for RoPE and LoRA compatibility
- Handles absolute vs. rotary positional embedding differences

## Model Checkpointing and Logging

- Automatically saves checkpoints to `best_model.pt`
- Checkpoints include model state and configuration
- Model saved when validation loss improves
- Helps prevent overfitting and enables training resumption

## Usage

To run the script with standard training:
```bash
python llm_v8.py
```

To enable LoRA for parameter-efficient fine-tuning:
```python
# In the main section, change to:
cfg.use_lora = True
```

To enable RoPE instead of absolute positional embeddings:
```python
# In the main section, change to:
cfg.use_rope = True
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
- LoRA implementation for parameter-efficient fine-tuning
- RoPE implementation for better positional encoding
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
- Automatic parameter freezing when using LoRA

## Educational Value

This implementation serves as an excellent learning resource for understanding:
- Transformer architecture fundamentals
- BPE tokenization and its advantages over word-level tokenization
- Flash Attention and its performance benefits
- LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning
- RoPE (Rotary Positional Embeddings) as an alternative to absolute positional embeddings
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
- Parameter-efficient fine-tuning methods
- Advanced positional encoding techniques