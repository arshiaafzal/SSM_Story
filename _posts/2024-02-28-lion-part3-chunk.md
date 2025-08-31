---
layout: distill
title: LION 🦁 Part III - Chunkwise Parallel from of LION
description: Explaining LION-Chunk for Balancing Memory-Speed Tradeoffs During Inference
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
  - name: LION-Chunk
    subsections:
      - name: LION-S Chunk
      - name: LION-D Chunk
---

[[Paper](https://www.arxiv.org/abs/2502.16249)]
[[Code](https://github.com/LIONS-EPFL/LION)]

1. [Part I - Full Linear Attention]({% post_url 2024-02-28-lion-part1-model %})
2. [Part II - Bi-directional RNN]({% post_url 2024-02-28-lion-part2-theory %})
3. Part III - Chunkwise Parallel from of LION
4. [Part IV - Results]({% post_url 2024-02-28-lion-part4-results %})

Since we have now established the LION framework, which maps Full Linear Attention into a bi-directional RNN in [Part II]({% post_url 2024-02-28-lion-part2-theory %}) of this series, a key question arises:

Given that RNNs are efficient and Attention is fast, can we strike a balance between them?

For causal Transformers like DeltaNet <d-cite key="yang2024parallelizing"></d-cite> and GLA <d-cite key="yang2024gated"></d-cite>, as well as the SSD algorithm in Mamba2 <d-cite key="dao2024transformers"></d-cite>, a chunkwise parallel form of Full Linear Attention could be an effective solution. Additionally, in models like Hydra <d-cite key="hwang2025hydra"></d-cite>, this balance is achieved by applying two SSD algorithms. However, can we derive a unified framework for chunking Full Linear Attention, particularly for LION-S and LION-D, where the mask $\mathbf{M}$ structure is known? The aim of chunking Full Linear Attention in LION is to maintain a balance between efficiency and speed, particularly during inference. Since LION benefits from stable masks, it does **not** require chunking during training allowing for higher throughput, especially for short sequences, when compared to other models such as Hydra <d-cite key="hwang2025hydra"></d-cite>. While in **Gated Linear Attention (GLA)** <d-cite key="yang2024gated"></d-cite>, **DeltaNet** <d-cite key="yang2024parallelizing"></d-cite>, and the **SSD algorithm of Mamba2** <d-cite key="dao2024transformers"></d-cite> _causal-specific_ chunking methods are employed, we extend this to the non-causal case as well.

## LION-Chunk

The key idea of chunking is that instead of processing the entire sequence of length $L$, we divide it into $N$ subsequences of length $C$, where $N \times C = L$.  
To achieve this, we start with the Full Linear Attention formulation:

$$\mathbf{Y} = (\mathbf{Q} \mathbf{K}^\top \odot \mathbf{M}) \mathbf{V}$$

we first chunk the queries, keys and values into submatrices

$$
\mathbf{Q}_{[i]} , \mathbf{K}_{[i]}, \mathbf{V}_{[i]}  \in \mathbb{R}^{C \times d}
$$

Now, given the form $$ (\mathbf{A} \odot \mathbf{M})$$, where $$\mathbf{A} = \mathbf{Q} \mathbf{K}^\top$$ we can construct the chunkwise form in four parts

- Chunkwise $$\mathbf{A}_{[ij]}$$
- Chunkwise form for the scaling matrix $\mathbf{C}_{[ij]}$
- The chunked hidden state to shape the unscaled output $\mathbf{S}_{[i(j-1)]}$
- Finally the output of the chunk $i$ which is $\mathbf{Y}_{[i]}$

using these chunked matrices we shape the full linear atteniton in chunk form as bellow:

> **LION Chunk**
>
> 
$$
\begin{aligned}
    \mathbf{A}_{[ij]} & = \mathbf{Q}_{[i]}\mathbf{K}_{[j]}^\top \odot \mathbf{M}_{[ij]}, \\
    \mathbf{C}_{[ij]} &= \mathbf{C}_{[i(j-1)]} + \text{Sum}(\mathbf{A}_{[ij]}), \\
    \mathbf{S}_{[ij]} & = \mathbf{S}_{[i(j-1)]} + \mathbf{A}_{[ij]} \mathbf{V}_{[j]}, \\
    \mathbf{Y}_{[i]} & = \frac{\mathbf{S}_{[iN]}}{\mathbf{C}_{[iN]}}
\end{aligned}
$$
{: .block-tip}

where $\text{Sum}$ operations applies summation over the row of the input matrix. And $\mathbf{M}_{[ij]}$ corresponds to a submatrix of the full maks $\mathbf{M}$ at chunk $ij$ like:

$$
\mathbf{M}_{[ij]} = \mathbf{M}_{iC+1:i(C+1),jC+1:j(C+1)} \in \mathbb{R}^{C \times C}.
$$

Let's start with an example, chunking the Attention matrix $\mathbf{A}$ for a sequence of $L=9$ with $C=3$ chunk size in detail below:

{% include figure.liquid loading="eager" path="assets/img/att_chunk.svg"%}

Chunking simply involves computing the queries and keys for each boxed sub-matrix, as illustrated for the upper, lower, and diagonal chunks. For every Attention matrix chunk $[ij]$, the computation follows the same pattern, multiplying the corresponding queries and keys for that chunk.

But does the same approach apply to Selective and Fixed masks?

In reality, chunking the Attention mask is slightly different and even more critical than chunking Attention itself due to its unique structure. Below, we provide a detailed explanation of how to chunk the Attention mask for LION-D and LION-S.

🚀 **Note:** The chunking visualization and details of this part are exclusively on the blogpost version.

### LION-D Chunk

Let's start with the decay mask, as it is simpler and easier to visualize. For LION-D, the final mask is a Toeplitz mask constructed using the scalar decay factor $\lambda$. We can visualize how the mask is structured.

{% include figure.liquid loading="eager" path="assets/img/maskdec_chunk.svg"%}

The full mask of LION-D (or full RetNet mask) is constructed simply by the submatrix of $\Gamma$, which is a [Toeplitz matrix](https://en.wikipedia.org/wiki/Toeplitz_matrix) itself. Regardless of where the chunk is located, whether in the upper or lower part of the mask matrix $\mathbf{M}$, it retains the same property of being a fraction of the Toeplitz matrix $\Gamma$ as bellow:

$$
\mathbf{M}_{[ij]} = \Gamma \lambda^{|i-j|}
$$

A pytorch implementation for LION-D Chunk Mask is provided below:

```python
def mask_decay_partial(a, length, start, end):
    idx = torch.arange(length, device=a.device)
    i, j = torch.meshgrid(idx, idx[start:end], indexing="ij")
    e = torch.abs((i - j)).float().view(1, 1, length, len(idx[start:end]))
    m = torch.sigmoid(a).view(1, -1, 1, 1) ** e
    return m
```

### LION-S Chunk

The full mask of LION-S is more tricky than LION-D since the upper lower and the diagonal part of the mask are shaped differently:

- The <span style="background-color: rgb(255, 248, 203); padding: 3px; color:black">Upper</span> part is influenced only by the decay factors applied from the end to the beginning of the sequence.
- The <span style="background-color: rgb(254, 200, 201); padding: 3px; color:black">Diagonal</span> part incorporates contributions from both directions, spanning from the start to the end and from the end to the start.
- The <span style="background-color: rgb(208, 243, 248); padding: 3px; color:black">Lower</span> part is influenced only by the decay factors applied from the beginning to the end of the sequence.

Let's visualize LION-S mask as well:

{% include figure.liquid loading="eager" path="assets/img/masksel_chunk.svg"%}

For example, the chunk [1,3] has only the cumulative decay factors multiplied from the beginning up to the last three sequence elements, while the chunk [3,1] has only the decay factors multiplied from the end up to the first three sequence elements. This is the reason for using the matrices $\mathbf{L}^F$ and $\mathbf{L}^B$ to compute the cumulative products of the decay factors, progressing from the beginning to the end of the sequence and in reverse which can be created simply by `L^F = cumprod(a)` and `L^B = cumprod(flip(a))`.

### The code for LION-S Chunk Mask

```python
def mask_forward(tensor, chunk_index, chunk_length):
    cumprod = torch.clamp(tensor.cumprod(dim=-1), 1e-6)
    a = (
        cumprod.unsqueeze(-1)
        / cumprod.unsqueeze(-2)[
            ..., chunk_index * chunk_length : (chunk_index + 1) * chunk_length
        ]
    )
    return torch.tril(a, diagonal=-chunk_index * chunk_length)


def mask_backward(tensor, chunk_index, chunk_length):
    cumprod = torch.clamp(tensor.cumprod(dim=-1), 1e-6)
    a = cumprod.unsqueeze(-1)[
        ..., chunk_index * chunk_length : (chunk_index + 1) * chunk_length, :
    ] / cumprod.unsqueeze(-2)
    return torch.triu(a.transpose(-1, -2), diagonal=-chunk_index * chunk_length)


def mask_selective_partial(vec, chunk_index, chunk_length):
    b, h, l = vec.shape
    a_for = create_matrix_from_tensor_forward(
        torch.cat((torch.ones_like(vec[..., :2]), vec[..., 1:-1]), dim=-1),
        chunk_index,
        chunk_length,
    )
    a_back = create_matrix_from_tensor_backward(
        torch.cat((torch.ones_like(vec[..., :1]), vec[..., 1:]), dim=-1),
        chunk_index,
        chunk_length,
    )
    i = torch.diag_embed(
        torch.ones((b, h, l - chunk_index * chunk_length)),
        offset=-chunk_index * chunk_length,
    )[..., : a_for.shape[-1]]
    return a_for + a_back - i.to(a_for.device)
```

Now that we have all elements in place let's see how these models are working in practice on real-world datasets for masked language modeling and image classification.

## Next Up

In the [final part of this series]({% post_url 2024-02-28-lion-part4-results %}), we present the advantages of using LION compared to other methods for training SSMs or Linear Transformers.

We also present the trade-offs for different LION 🦁 models and compare them with other well-known SSMs and Softmax Transformers.

[Continue reading to Part IV - Results]({% post_url 2024-02-28-lion-part4-results %})
