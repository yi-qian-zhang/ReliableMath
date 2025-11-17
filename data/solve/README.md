---
license: apache-2.0
---

## Overview

Training dataset for Polaris Preview models. The dataset is filtered from [DeepScaleR-Preview-Dataset](https://huggingface.co/datasets/agentica-org/DeepScaleR-Preview-Dataset) and [AReal-boba-Data](https://huggingface.co/datasets/inclusionAI/AReaL-boba-Data)

## Format

Each row in the `jsonl` file contains:

- **problem**: The input problem.
- **answer**: The answer to the problem
- **difficulty**: The pass rate of the problem estimated by `Deepseek-R1-distill-Qwen-7B` 


## Citation

```bibtex
@misc{Polaris2025,
    title = {POLARIS: A Post-Training Recipe for Scaling Reinforcement Learning on Advanced Reasoning Models},
    url = {https://hkunlp.github.io/blog/2025/Polaris},
    author = {An, Chenxin and Xie, Zhihui and Li, Xiaonan and Li, Lei and Zhang, Jun and Gong, Shansan and Zhong, Ming and Xu, Jingjing and Qiu, Xipeng and Wang, Mingxuan and Kong, Lingpeng}
    year = {2025}
}
```