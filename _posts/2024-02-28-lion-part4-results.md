---
layout: distill
title: LION 🦁 Part IV - Results
description: Comprehensive results on Vision, MLM and more LION variants
tags:
giscus_comments: false
date: 2025-02-28
featured: false
thumbnail: assets/img/lion.jpg

authors:
  - name: Arshia Afzal
    url:
    affiliations:
      name: Writer of blogpost
  - name: Elias Abad Rocamora
    url:
    affiliations:
  - name: Leyla Naz Candogan
    url:
    affiliations:
      name: Writer of blogpost
  - name: Pol Puigdemont Plana
    url:
    affiliations:
      name: Writer of blogpost
  - name: Francesco Tonin
    url:
    affiliations:
  - name: Yongtao Wu
    url:
    affiliations:
  - name: Mahsa Shoaran
    url:
    affiliations:
  - name: Volkan Cevher
    url:
    affiliations:
      name: All authors are with EPFL

bibliography: albert.bib

toc:
  - name: Vision Results
    subsections:
      - name: Base Model Performance
      - name: Memory Efficiency
      - name: Training Time Analysis
  - name: MLM Results
    subsections:
      - name: BERT Comparison
  - name: LION Architecture Variants
    subsections:
      - name: Architecture Trade-offs
---

[[Paper](https://www.arxiv.org/abs/2502.16249)]
[[Code](https://github.com/LIONS-EPFL/LION)]

1. [Part I - Full Linear Attention]({% post_url 2024-02-28-lion-part1-model %})
2. [Part II - Bi-directional RNN]({% post_url 2024-02-28-lion-part2-theory %})
3. [Part III - Chunkwise Parallel from of LION]({% post_url 2024-02-28-lion-part3-chunk %})
4. Part IV - Results

In the final part of our LION series, we will present and discuss a selection of experimental results across various domains, including vision tasks, masked language modeling (MLM), and different LION architectures. These results not only highlight LION's versatility and efficiency across diverse applications but also serve as a preview of the comprehensive findings detailed in the full paper.

## Image Classification Performance Overview

### Model Comparisons

We evaluated LION's performance, efficiency, and training times against state-of-the-art SSMs and Transformers for image classification. The results demonstrate that LION achieves competitive performance while offering significant advantages in training speed and efficiency.

| Model                      | #Param | Imagenet Top-1 Acc. | Train. time               |
| -------------------------- | ------ | ------------------- | ------------------------- |
| $\text{ViT}$               | 86M    | $77.9$              | $\times 1$                |
| $\text{DeiT}$              | 86M    | $\underline{81.8}$  | $\times 1$                |
| $\text{Hydra}$             | 104M   | $81.0$              | $\times 2.51$             |
| $\text{Vim}$               | 98M    | $\mathbf{81.9}$     | $\times 10.86$            |
| $\text{LION-}\text{🔥}$    | 86M    | $74.7$              | $\mathbf{\times 0.73}$    |
| $\text{LION-D}$            | 86M    | $77.8$              | $\times \underline{1.39}$ |
| $\text{LION-D}^{\natural}$ | 86M    | $80.2$              | $\times 1.48$             |
| $\text{LION-S}$            | 86M    | $76.3$              | $\times 1.46$             |
| $\text{LION-S}^{\natural}$ | 86M    | $79.9$              | $\times 1.68$             |

<div class="caption" style="color: #666666; margin-top: 1px;">
    Model performance comparison on ImageNet classification, showing parameter count, top-1 accuracy, and relative training time.
</div>

As shown in the table above, LION models achieve competitive performance with vision-specific SSMs like Vim, while being significantly faster during training. LION-D performs comparably to Vim and surpasses Hydra <d-cite key="hwang2025hydra"></d-cite>, while training approximately 7x faster than Vim <d-cite key="zhu2024vision"></d-cite>. Notably, LION-🔥 demonstrates the highest training speed across all models, showing that training with Full Linear Attention is significantly faster than chunkwise parallel training (used in Hydra) and considerably faster than the scan algorithm, even with optimized GPU kernels (as used in Vim). $$LION-S^{\natural}$$ and $$LION-D^{\natural}$$ modify the order of patches in an image to better capture the locality inherent in spatial patterns. By rearranging the patch sequence, these models enhance their understanding of local structures while still leveraging the efficiency of Linear Attention mechanisms similar to xLSTM <d-cite key="alkin2024vision"></d-cite>.

### Memory Efficiency

The LION family demonstrates excellent memory efficiency across both vision and language tasks. Figure below shows inference memory usage with a batch size of 64 across different image resolutions, LION models (RNN form) maintain reasonable memory consumption even at high resolutions up to 2496 pixels, while adding minimal training overhead in BERT-style language modeling scenarios. In contrast, baseline models like ViT and DeiT run out of memory (OOM) at much lower resolutions.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/fig1_plot.svg" title="Memory Usage Comparison" caption="Memory usage during inference across different architectures with batch size 64. LION models (RNN form) maintain reasonable memory consumption at high resolutions while other models run out of memory." class="img-fluid rounded z-depth-1" %}
    </div>
</div>

### Training Time Analysis

The LION family demonstrates remarkable training efficiency across both vision and language tasks. As shown in the table below, LION variants add minimal training overhead compared to SSMs.

| Task   | <span style="background-color: rgb(230, 255, 230); padding: 3px; color:black">LION-🔥 </span> | <span style="background-color: rgb(229, 204, 230); padding: 3px; color:black">LION-D </span> | <span style="background-color: rgb(255, 233, 211) ; padding: 3px; color:black">LION-S </span> | Hydra         | Vim            |
| ------ | --------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- | ------------- | -------------- |
| Vision | $\times 0.73$                                                                                 | $\times 1.39$                                                                                | $\times 1.46$                                                                                 | $\times 2.51$ | $\times 10.86$ |
| MLM    | $\times 0.95$                                                                                 | $\times 1.10$                                                                                | $\times 1.32$                                                                                 | $\times 3.13$ | ✗              |

<div class="caption" style="color: #666666; margin-top: 1px;">
    Training Times (relative to Transformer) ↓
</div>

For vision tasks, LION-🔥 achieves remarkable speed, training 27% faster than standard vision Transformers <d-cite key="dosovitskiy2020image"></d-cite>. Even the more complex LION variants maintain competitive training times, with LION-D and LION-S training only ~1.4x slower than Transformers. This is significantly better than competing approaches like Hydra (2.51x slower) and Vim (10.86x slower).

In MLM tasks, the efficiency gains are even more pronounced. LION-🔥 nearly matches Transformer training speed at just 0.95x, while LION-D adds only 10% overhead. Even LION-S remains efficient at 1.32x. All LION variants significantly outperform Hydra's 3.13x slowdown, while Vim is not applicable to MLM tasks (marked as ✗).

## MLM Results

For masked language modeling (MLM) tasks, we evaluated LION models against BERT <d-cite key="devlin2018bert"></d-cite> and Hydra on both MLM pretraining and GLUE benchmark finetuning. The results show that LION variants achieve competitive performance while maintaining good training efficiency.

| Model                                                                                         | MLM Acc.            | GLUE                | Train. time               |
| --------------------------------------------------------------------------------------------- | ------------------- | ------------------- | ------------------------- |
| BERT                                                                                          | $\underline{69.88}$ | $\mathbf{82.95}$    | $\times 1$                |
| Hydra                                                                                         | $\mathbf{71.18}$    | $\underline{81.77}$ | $\times 3.13$             |
| <span style="background-color: rgb(230, 255, 230); padding: 3px; color:black">LION-🔥 </span> | $67.11$             | $80.76$             | $\times \mathbf{0.95}$    |
| <span style="background-color: rgb(229, 204, 230); padding: 3px; color:black">LION-D </span>  | $68.64$             | $81.34$             | $\times \underline{1.10}$ |
| <span style="background-color: rgb(255, 233, 211) ; padding: 3px; color:black">LION-S </span> | $69.16$             | $81.58$             | $\times 1.32$             |

<div class="caption" style="color: #666666; margin-top: 1px;">
    C4 MLM and GLUE results for the LARGE scale (334M). For each dataset, the best and second best results are highlighted with bold and underline respectively.
</div>

## LION Architecture Variants and Trade-offs

Let's explore how different LION variants handle the trade-off between memory usage and inference speed. We will look at three key approaches:

1. Full Linear Attention - The standard approach using the Full Attention matrix.
2. Bidirectional RNN - Our memory-efficient RNN formulation.
3. LION Chunk - A balanced approach using chunked computation.

### Memory vs Speed Trade-offs

The first plot below shows how these approaches compare in terms of memory efficiency and inference speed in LION-D. The RNN approach proves to be the most memory-efficient, while Full Attention uses the most memory. LION Chunk provides a nice middle ground - it uses less memory than Full Attention while actually achieving faster inference speeds than both alternatives. This makes it particularly attractive when you need to balance performance with resource constraints.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/fig3_plot.svg" title="Impact of Chunk Size" caption="Analysis of how chunk size affects model performance across different LION-D variants." class="img-fluid rounded z-depth-1" %}
    </div>
</div>

For LION-🔥, we see a similar pattern, but the chunking approach is even more pronounced.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/linear_chunking.svg" title="Linear Chunking Analysis" caption="Evaluation of linear chunking strategies and their impact on model efficiency of LION-🔥." class="img-fluid rounded z-depth-1" %}
    </div>
</div>

Lastly for LION-S, we see that the chunking approach is only faster at lower resolutions - at higher resolutions, the overhead from mask calculations starts to slow it down.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/selective_chunking.svg" title="Selective Chunking Analysis" caption="Performance comparison of selective chunking approaches across different sequence lengths with LION-S." class="img-fluid rounded z-depth-1" %}
    </div>
</div>

## Future Directions

- **Expanding LION's Potential:** Our experiments focused on three main mask choices, but LION has the potential to accelerate other Linear Transformer variants for bidirectional tasks.

- **Optimizing Chunkwise Parallelism:** The chunkwise parallel implementation during inference was done in PyTorch, with room for optimization through GPU kernel programming to reduce I/O overhead and improve speed.

- **Stabilizing Hydra and Mamba with LION:** Hydra <d-cite key="hwang2025hydra"></d-cite> and Mamba <d-cite key="gu2023mamba"></d-cite> activations led to unstable training under Full Attention, suggesting LION could be used to stabilize these variants in the future.

# Last Points

We encourage the readers of this blog post to read the full [paper](https://www.arxiv.org/abs/2502.16249) for more details about the LION framework and experimental setups. The implementation details are available in the [code repository](https://github.com/LIONS-EPFL/LION).

If you use this work, please consider citing the paper:

```bibtex
@article{afzal2025linear,
  title={Linear Attention for Efficient Bidirectional Sequence Modeling},
  author={Afzal, Arshia and Abad Rocamora, Elias and Candogan, Leyla Naz and Puigdemont, Pol and Tonin, Francesco and Wu, Yongtao and Shoaran, Mahsa and Cevher, Volkan},
  journal={arXiv preprint arXiv:2502.16249},
  year={2025},
  url={https://arxiv.org/abs/2502.16249},
  doi={10.48550/arXiv.2502.16249}
}
```
