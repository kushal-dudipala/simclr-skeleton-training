# SimCLR  Trainer

A lightweight skeleton trainer implementation based on SimCLR (Simple Framework for Contrastive Learning of Visual Representations).

## Features

- **Skeleton Framework**: Minimal codebase that serves as a foundation for contrastive learning experiments
- **Multi-GPU Support**: Distributed training capabilities for scaling across multiple GPUs
- **FFCV Loading**: Fast data loading using FFCV for improved training throughput and efficiency

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run training
python train.py --config config.yaml
```

## Configuration

The trainer supports flexible configuration through YAML files. Key parameters include:

## Requirements

- Python 3.7+
- PyTorch
- FFCV
- Additional dependencies listed in `requirements.txt`