# 🚀 CodingLLM - Evolution of GPT Models from Scratch

This repository contains a collection of GPT implementations, showcasing the evolution from basic concepts to advanced features. Each version builds upon the previous one, adding new techniques and improvements to create increasingly sophisticated language models.

## 📚 Project Structure

The repository is organized into multiple versions, each representing a different stage in the development of GPT models:

| Version | Name | Key Focus | Status |
|--------|------|-----------|--------|
| llm_v2 | Minimal | Causal self-attention & basic structure | ✅ Complete |
| llm_v3 | nanoGPT-style | Word-level tokenization | ✅ Complete |
| llm_v4 | Real GPT | BPE Tokenization (tiktoken) | ✅ Complete |
| llm_v5 | Enhanced | Early stopping & LR scheduling | ✅ Complete |
| llm_v6 | Advanced | Architectural enhancements | 🔧 WIP/Experimental |
| llm_v7 | Advanced | Further architectural improvements | 🔧 WIP/Experimental |
| llm_v8 | SOTA | LoRA, RoPE, Flash Attention, Grad Accum | 🌟 Flagship |

## 🧠 Feature Progression

### 1. Core Architecture
- **Tokenization**: Evolved from simple regex-based word tokenization ➡️ BPE using tiktoken
- **Attention Mechanism**: Causal self-attention with full multi-head support
- **Positional Encoding**: Upgraded from learned embeddings ➡️ Rotary Positional Embeddings (RoPE)
- **Model Depth**: Gradual increase in layers and embedding dimensions across versions

### 2. Training Enhancements
- **Optimization**: Implementation of advanced optimizers, learning rate scheduling, and gradient clipping
- **Regularization**: Dropout, weight decay, and gradient norm clipping
- **Efficiency**: Gradient accumulation, mixed precision training, and model compilation
- **Evaluation**: Strict validation splits, loss monitoring, and model checkpointing

### 3. Advanced Techniques (SOTA)
- **Parameter Efficiency**: Low-Rank Adaptation (LoRA) for efficient fine-tuning
- **Attention Improvements**: Flash Attention for significant memory and speed optimization
- **Positional Embeddings**: Rotary Positional Embeddings (RoPE) replacing absolute embeddings
- **Training Strategies**: Early stopping, warmup schedules, and adaptive learning rates

## 🛠 Tech Stack

| Component | Technology |
|-----------|------------|
| Framework | PyTorch |
| Tokenizer | Tiktoken (OpenAI's BPE) |
| Acceleration | CUDA, Flash Attention |
| Precision | Mixed Precision (AMP) |
| Config | Python Dataclasses |

## 🎓 Educational Value

This repo serves as a comprehensive resource for understanding:

- **Transformer Architecture**: Deep dive into attention mechanisms and implementation
- **Language Model Training**: End-to-end pipeline from data preprocessing to evaluation
- **Advanced Techniques**: Modern methods like LoRA and RoPE
- **Implementation Details**: Practical considerations for training large models
- **Performance Optimization**: Techniques to make training faster and more efficient

## ⚡ Getting Started

Each version is designed to run independently.

```bash
cd llm_v8  # or any other version
python llm_v8.py
```

### Prerequisites

Make sure you have the following installed:

- Python 3.8+
- PyTorch
- Tiktoken

### Installation

```bash
pip install torch tiktoken
```

## 📝 Data Requirements

Each version expects a `data.txt` file in its respective folder containing the raw text data for training.

> **Note**: If `data.txt` is missing, some versions provide fallback dummy data for testing purposes.

## 🤝 Contributing

This repository represents an ongoing educational journey through transformer model implementation. Contributions that enhance understanding, fix bugs, or add new features—while maintaining the educational focus—are welcome.

## 📜 License

This repository is intended for educational purposes. See the individual files for specific licensing information.

## 🏗️ Project Evolution

This repository tracks the iterative development of GPT models. Each version introduces specific architectural upgrades and optimizations, demonstrating the progression from a minimal implementation to state-of-the-art techniques.
