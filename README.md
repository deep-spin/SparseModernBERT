# Sparse ModernBERT

This repo contains the code used for the experiments with ModernBERT in the AdaSplash paper: https://arxiv.org/abs/2502.12082


## Installation and Usage

Check the steps in the original [ModernBERT repo](https://github.com/AnswerDotAI/ModernBERT/). 


## Models on Huggingface

- Alpha = 1.5: https://huggingface.co/sardinelab/SparseModernBERT-alpha1.5
- Alpha = 2.0: https://huggingface.co/sardinelab/SparseModernBERT-alpha2.0


## AdaSplash

AdaSplash is an efficient adaptive sparse attention mechanism implemented in Triton. See repo: https://github.com/deep-spin/adasplash


## Reference

```bibtex
@article{goncalves2025adasplash,
  title={AdaSplash: Adaptive Sparse Flash Attention},
  author={Nuno Gonçalves and Marcos Treviso and André F. T. Martins},
  journal={arXiv preprint arXiv:2502.12082},
  url={https://arxiv.org/abs/2502.12082},
  year={2025}
}
```