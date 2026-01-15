# CodingLLM - Evolution of GPT Models from Scratch

This repository contains a collection of GPT implementations, showcasing the evolution from basic concepts to advanced features. Each version builds upon the previous one, adding new techniques and improvements to create increasingly sophisticated language models.

## Project Structure

The repository is organized into multiple versions, each representing a different stage in the development of GPT models:

- **llm_v2**: Minimal but correct GPT implementation with causal self-attention
- **llm_v3**: nanoGPT-style implementation with word-level tokenization
- **llm_v4**: Real GPT implementation using BPE tokenization (tiktoken)
- **llm_v5**: Enhanced version with early stopping and learning rate scheduling
- **llm_v6**: Advanced features and optimizations (not analyzed in detail)
- **llm_v7**: Further improvements and architectural enhancements (not analyzed in detail)
- **llm_v8**: State-of-the-art implementation with LoRA, RoPE, Flash Attention, and gradient accumulation

## Feature Progression

### Core Architecture
- **Tokenization**: Evolves from simple regex-based word tokenization to BPE using tiktoken
- **Attention Mechanism**: Causal self-attention with multi-head support
- **Positional Encoding**: From learned embeddings to Rotary Positional Embeddings (RoPE)
- **Model Depth**: Gradual increase in layers and embedding dimensions

### Training Enhancements
- **Optimization**: Advanced optimizers, learning rate scheduling, gradient clipping
- **Regularization**: Dropout, weight decay, gradient norm clipping
- **Efficiency**: Gradient accumulation, mixed precision training, model compilation
- **Evaluation**: Validation splits, loss monitoring, checkpointing

### Advanced Techniques
- **Parameter Efficiency**: Low-Rank Adaptation (LoRA) for efficient fine-tuning
- **Attention Improvements**: Flash Attention for memory and speed optimization
- **Positional Embeddings**: Rotary Positional Embeddings (RoPE) replacing absolute embeddings
- **Training Strategies**: Early stopping, warmup schedules, adaptive learning rates

## Key Technologies Used

- PyTorch for deep learning implementation
- Tiktoken for BPE tokenization (matching GPT-2/GPT-3 tokenizer)
- CUDA acceleration where available
- Mixed precision training for efficiency
- Dataclass-based configuration management

## Educational Value

This repository serves as an excellent educational resource for understanding:

1. **Transformer Architecture**: How attention mechanisms work and how they're implemented
2. **Language Model Training**: From data preprocessing to model evaluation
3. **Advanced Techniques**: Modern methods like LoRA and RoPE
4. **Implementation Details**: Practical considerations for training large models
5. **Performance Optimization**: Techniques to make training more efficient

## Getting Started

Each version in the repository can be run independently. Simply navigate to the desired version folder and execute the Python script:

```bash
cd llm_v8  # or any other version
python llm_v8.py
```

Most implementations will:
1. Load training data from `data.txt` in the respective folder
2. Build vocabulary and tokenize the data
3. Train the model with specified hyperparameters
4. Generate sample text using the trained model

## Data Requirements

Each version expects a `data.txt` file in its respective folder containing the text data for training. If this file doesn't exist, some versions provide fallback dummy data for testing purposes.

## Version Highlights

- **llm_v2**: Foundational implementation showing the core concepts
- **llm_v4**: Introduction of proper BPE tokenization
- **llm_v5**: Addition of training optimizations like early stopping
- **llm_v8**: State-of-the-art implementation with cutting-edge techniques like LoRA and RoPE

## Contributing

This repository represents an ongoing educational journey through transformer model implementation. Contributions that enhance understanding, fix bugs, or add new features while maintaining the educational focus are welcome.

## License

This repository is intended for educational purposes. See the individual files for specific licensing information.
