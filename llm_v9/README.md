# Enhanced GPT v9 (Modern Architecture)

This implementation provides a state-of-the-art GPT model built from scratch, featuring modern architectural components including RMSNorm, SwiGLU activation, and KV Cache for efficient generation. This version builds upon previous improvements like Flash Attention, LoRA, and RoPE.

## Overview

This script implements an advanced GPT model with:
- BPE tokenization using tiktoken (GPT-2 encoding)
- Flash Attention for improved efficiency
- Low-Rank Adaptation (LoRA) for parameter-efficient fine-tuning
- Rotary Positional Embeddings (RoPE) as an alternative to absolute positional embeddings
- RMSNorm for more stable normalization
- SwiGLU activation function for improved performance
- KV Cache for fast generation
- Support for loading pre-trained GPT-2 weights
- Improved weight initialization strategies
- Gradient accumulation for larger effective batch sizes
- Comprehensive model checkpointing and logging
- More robust training with early stopping
- Advanced text generation with multiple sampling strategies

## Key Improvements Over Previous Versions

- **RMSNorm**: Root Mean Square Layer Normalization used in Llama, Mistral, and Falcon models, offering more stability than LayerNorm
- **SwiGLU Activation**: Advanced activation function (Swish-Gated Linear Unit) that replaces traditional GELU for better performance
- **KV Cache**: Implements key-value caching for efficient generation, dramatically speeding up inference
- **Fused AdamW Optimizer**: Uses CUDA-optimized fused optimizer when available for faster training
- **Enhanced Architecture Flexibility**: Toggle between LayerNorm and RMSNorm, GELU and SwiGLU
- **Parameter-Efficient Training**: LoRA enables efficient fine-tuning with significantly fewer trainable parameters

## Architecture Components

### Causal Self-Attention
- Multi-head attention mechanism with configurable head count
- Causal masking to ensure tokens only attend to previous positions
- Uses Flash Attention when available for improved performance
- Supports both standard linear layers and LoRA-enhanced layers
- Supports both absolute positional embeddings and RoPE
- Implements KV Cache for efficient generation
- Proper dimension handling for batch processing
- Dropout regularization for preventing overfitting

### Transformer Block
- Configurable normalization (LayerNorm or RMSNorm)
- Residual connections around attention and MLP components
- Multi-layer perceptron with configurable activation (GELU or SwiGLU)
- Configurable embedding dimensions

### GPT Model
- Token embeddings combined with either positional embeddings or RoPE
- Sequential transformer blocks with modern normalization and activation
- KV Cache implementation for efficient generation
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
- Intermediate Size: 4 * embed_size (or custom value)

### LoRA Settings:
- LoRA Enabled: False (set to True to enable)
- LoRA Rank (r): 8
- LoRA Alpha: 16.0
- LoRA Dropout: 0.05
- LoRA Target Modules: ["c_attn", "c_proj", "c_fc"]

### RoPE Settings:
- RoPE Enabled: True (set to False to use absolute positional embeddings)
- RoPE Frequency Base: 10000

### Modern Architecture Settings:
- Normalization: "rmsnorm" (options: "layernorm", "rmsnorm")
- Activation: "swiglu" (options: "gelu", "swiglu")

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
- Device: Automatic detection (CUDA if available, otherwise CPU)
- Data Type: bfloat16 or float16 for mixed precision training

## Data Processing

- Reads text data from `data.txt` in the same directory
- Uses tiktoken's GPT-2 encoder for BPE tokenization
- Automatically determines vocabulary size from tiktoken
- Creates train/validation split (90%/10%)
- Gracefully falls back to dummy data if data file is missing

## Training Process

- Uses Fused AdamW optimizer with configurable parameters (CUDA-optimized when available)
- Implements gradient clipping (norm = 1.0) for stability
- Applies learning rate scheduling with warmup and cosine annealing
- Enables gradient accumulation for larger effective batch sizes
- Uses mixed precision training for efficiency
- Logs both training and validation loss at regular intervals
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
- **KV Cache enabled** for dramatically faster generation

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

## RMSNorm (Root Mean Square Normalization)

- More stable than LayerNorm and doesn't require centering (mean subtraction)
- Used in state-of-the-art models like Llama, Mistral, and Falcon
- Provides improved numerical stability during training
- More efficient to compute than LayerNorm

## SwiGLU (Swish-Gated Linear Unit)

- Advanced activation function that combines Swish activation with gating
- Outperforms traditional GELU in many modern architectures
- Combines gate and up projections with element-wise multiplication
- Part of the Llama/Mistral-style architecture

## KV Cache (Key-Value Cache)

- Dramatically speeds up generation by caching computed key-value pairs
- Eliminates redundant computation for previously processed tokens
- Essential for efficient autoregressive generation
- Implemented within the attention mechanism

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
python llm_v9.py
```

To enable LoRA for parameter-efficient fine-tuning:
```python
# In the main section, change to:
cfg.use_lora = True
```

To switch back to LayerNorm and GELU:
```python
# In the main section, change to:
cfg.norm_type = "layernorm"
cfg.activation = "gelu"
```

The script will:
1. Load and preprocess the data from `data.txt` using BPE tokenization
2. Initialize and train the GPT model with all modern features
3. Generate sample text based on the prompt "The future of AI is" using KV Cache for fast generation

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
- RMSNorm implementation for improved stability
- SwiGLU activation for state-of-the-art performance
- KV Cache for efficient generation
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
- Fused AdamW optimizer for faster training

## Educational Value

This implementation serves as an excellent learning resource for understanding:
- Transformer architecture fundamentals
- BPE tokenization and its advantages over word-level tokenization
- Flash Attention and its performance benefits
- LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning
- RoPE (Rotary Positional Embeddings) as an alternative to absolute positional embeddings
- RMSNorm and its advantages over LayerNorm
- SwiGLU activation function and its benefits
- KV Cache mechanism for efficient generation
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
- Modern transformer architecture patterns (Llama/Mistral-style)
- Parameter-efficient fine-tuning methods
- Advanced positional encoding techniques
