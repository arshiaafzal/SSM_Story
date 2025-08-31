---
layout: distill
title: LION 🦁 Part II - Bi-directional RNN
description: Deriving equivalent bi-directional RNN for Linear Attention
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
  - name: Finding Bi-directional RNN Equal to Full Linear Attention
  - name: Some Important details of our RNN
  - name: Different Masks of LION
    subsections:
      - name: LION-🔥
      - name: LION-D
      - name: LION-S
  - name: LION Attention Block
---

[[Paper](https://www.arxiv.org/abs/2502.16249)]
[[Code](https://github.com/LIONS-EPFL/LION)]

1. [Part I - Full Linear Attention]({% post_url 2024-02-28-lion-part1-model %})
2. Part II - Bi-directional RNN
3. [Part III - Chunkwise Parallel from of LION]({% post_url 2024-02-28-lion-part3-chunk %})
4. [Part IV - Results]({% post_url 2024-02-28-lion-part4-results %})

In [Part I]({% post_url 2024-02-28-lion-part1-model %}) of this series, we defined Full Linear Attention with Masking and Scaling.
Similar to all Linear Transformers designed for Causal Sequence Modeling, we aim to derive an RNN form for efficiency during inference.
In this section, we theoretically demonstrate the equivalent bi-directional RNN for the Full Linear Transformer.

## Finding Bi-directional RNN Equal to Full Linear Attention

We aim to construct an RNN that is equivalent to the Masked and Scaled Linear Attention. The idea of a bi-directional RNN is to process the sequence in both the forward order (from first to last) and the reverse order (from last to first), these naturally correspond to the upper and lower parts of the Attention matrix and mask.

{% include figure.liquid loading="eager" path="assets/img/att_mask_color.svg"%}

**Note:** We use a consistent color coding for this section of the blog post and throughout our [paper](https://www.arxiv.org/abs/2502.16249) 😊.

- <span style="background-color: rgb(255, 248, 203); padding: 3px; color:black">Yellow</span> color indicates the **upper part of the matrix (non-causal)**.
- <span style="background-color: rgb(254, 200, 201); padding: 3px; color:black">Red</span> color represents the **diagonal elements**.
- <span style="background-color: rgb(208, 243, 248); padding: 3px; color:black">Blue</span> color corresponds to the **lower triangular (causal) part**.

Let's seperate the Attention into upper and lower parts:

{% include figure.liquid loading="eager" path="assets/img/att_sep.svg"%}

This formulation represents both the causal and non-causal forms of Attention. We would like to model each triangular part using an RNN. Similarly, we can also separate the mask in the same way:

{% include figure.liquid loading="eager" path="assets/img/mask_sep.svg"%}

Let's also write the scaling part of the Masked Attention $\mathbf{Y} = \text{Scale}(\mathbf{Q} \mathbf{K}^\top \odot \mathbf{M} ) \mathbf{V}$ as:

$$
\begin{aligned}
   \mathbf{Y} = \big(\text{Scale}(\mathbf{Q}\mathbf{K}^{\top} \odot \mathbf{M})\big) \mathbf{V}
    = (\mathbf{C}^{-1}(\mathbf{Q}\mathbf{K}^{\top} \odot \mathbf{M}))\mathbf{V}, \hspace{1mm}
   \mathbf{C}_i = \mathbf{q}^{\top}_i\sum\limits_{j=1}^{L} \mathbf{M}_{ij}\mathbf{k}_j.
\end{aligned}
$$

Also, we can decompose the scaling matrix $$\mathbf{C}_i$$ as:

$$
\begin{aligned}
\mathbf{C}_{i}=
  \underbrace{\mathbf{q}^{\top}_i\sum\nolimits_{j=1}^{i} \mathbf{M}_{ij}\mathbf{k}_j - \frac{1}{2} \mathbf{q}^{\top}_i\mathbf{k}_i}_{\mathbf{C}^F_i} + \underbrace{\mathbf{q}^{\top}_i\sum\nolimits_{j=i}^{L} \mathbf{M}_{ij}\mathbf{k}_j - \frac{1}{2} \mathbf{q}^{\top}_i\mathbf{k}_i}_{\mathbf{C}^B_i}
\end{aligned}
$$

Now we replace the above scaling matrix $\mathbf{C}$ in the output of the Attention form of $\mathbf{Y} = \text{Scale}(\mathbf{Q} \mathbf{K}^\top \odot \mathbf{M} ) \mathbf{V}$ . Interestingly, many terms naturally cancel out with each other.

{% include figure.liquid loading="eager" path="assets/img/proofC.svg"%}

This results in only the forward and backward directions of the RNN remaining. As observed, the forward path aligns with Causal Linear Attention with masking. Now, we need to demonstrate that the backward path follows the same RNN structure in the reverse direction. We can simply flip the upper triangular matrices using the [Exchange Matrix](https://en.wikipedia.org/wiki/Exchange_matrix) $$\mathbf{J}_L$$ and the function $$F(X) = \mathbf{J}_L X \mathbf{J}_L$$:

{% include figure.liquid loading="eager" path="assets/img/flip.svg"%}

Cool! Now, both the upper part (equivalent to the RNN in the forward direction) and the lower part (equivalent to the RNN in the backward direction) can be formulated as RNNs. This is exactly what we need to construct our bi-directional RNN equivalent to Full Linear Attention.

> **LION: Reccurence form**
>
> $$
> \begin{aligned} \mathbf{S}_i^{F/B} &= \lambda_i \mathbf{S}^{F/B}_{i-1} + \mathbf{k}_i \mathbf{v}_i^{\top}, \\
> \mathbf{z}^{F/B}_i &= \lambda_i \mathbf{z}^{F/B}_{i-1} + \mathbf{k}_i,  \\
> c^{F/B}_i & = \mathbf{q}_i^{\top} \mathbf{z}^{F/B}_{i} - \frac{\mathbf{q}_i^{\top} \mathbf{k}_i}{2},  \\
> \mathbf{y}^{F/B}_i &= \mathbf{q}_i^{\top} \mathbf{S}^{F/B}_i - \frac{\mathbf{q}_i^{\top} \mathbf{k}_i}{2} \mathbf{v}_i, \\
> out&put: \mathbf{y}_i = \frac{\mathbf{y}^{F}_i + \mathbf{y}^{B}_i}{c^F_i + c^B_i}. \\ \end{aligned}
> $$
>
> {: .block-tip}

The terms $$\frac{\mathbf{q}_i^{\top} \mathbf{k}_i}{2}$$ and $$\frac{\mathbf{q}_i^{\top} \mathbf{k}_i}{2} \mathbf{v}_i$$ are subtracted to avoid double counting. This bi-directional RNN is equivalent to Scaled and Masked Linear Attention described in previous section of this blogpost.

## Some Important details of our RNN

> Only the states $$c^{F/B}_i$$ and $$\mathbf{y}^{F/B}_i$$ are stored per token, resulting in $$\mathcal{O}(Ld)$$ memory usage. In contrast, naively storing full matrix-valued hidden states would require $$\mathcal{O}(Ld^2)$$, which becomes infeasible for large models.

> Forward and backward recurrences run independently, completing in $$L$$ time steps with $$L$$ memory units, compared to $$2L$$ in the naive approach.

{% include figure.liquid loading="eager" path="assets/img/memory.svg" title="Memory Allocation of LION in RNN form" caption="Memory allocation in LION during Forward and Backward recurrences." %}

All in one, we can visualize our framework:

{% include figure.liquid loading="eager" path="assets/img/frlion.svg" title="LION" caption="LION 🦁: Our framework for training in parallel using Full Linear Attention which also supports the efficient bi-directional RNN format." %}

## Different Masks of LION

Now that we have created our framework let's see what are the choices of the decay factor $$\lambda_i$$ and how they resemble known Linear Transformer models. Let's set:

> $\lambda_i=1$ resembles the bi-directional version of the vanilla Linear Transformer <d-cite key="katharopoulos2020transformers"></d-cite> which we refer to as <span style="background-color: rgb(230, 255, 230); padding: 3px; color:black">LION-🔥 </span> (-LIT in [the paper](https://www.arxiv.org/abs/2502.16249)).

> $\lambda_i=\lambda$ resembles the bi-directional version of RetNet <d-cite key="sun2023retentive"></d-cite> which we refer to as <span style="background-color: rgb(229, 204, 230); padding: 3px; color:black">LION-D </span>.

> $\lambda_i=\sigma(\mathbf{W}\mathbf{x}_i)$ (input dependent) resembles a bi-directional selective Linear Transformer inspired by Mamba2 <d-cite key="dao2024transformers"></d-cite> which we refer to as <span style="background-color: rgb(255, 233, 211) ; padding: 3px; color:black">LION-S </span>.

We evaluate all models on several bi-directional tasks. Also inspired by Linear Transformers applying a feature mapping $\phi(.)$ to queries and keys we apply normalized shifted SILU $\phi(x)=$ `(SILU(x)+0.5)/(norm(SILU(x)+0.5))` as a non-linear activation function. Let's dive deep in each of these models in LION framework.

### LION-🔥

LION-🔥 is an extension of the original Linear Transformer <d-cite key="katharopoulos2020transformers"></d-cite>. Without any masking, the bi-directional parallel form can be simply written as:

$$\mathbf{Y} = Scale(\mathbf{Q} \mathbf{K}^\top )\mathbf{V} $$

the RNN form is the one introduced the previous green box "LION: Reccurence form" with $$\lambda_i=1$$.

### LION-D

By fixing $$\lambda_i = \lambda$$, the mask $$\mathbf{M}$$ has the form:

$$
\begin{align}
    \mathbf{M}_{ij} = \lambda^{|i-j|}, \quad \mathbf{D}_{ij} = |i-j|\log(\lambda), \quad \mathbf{M} = \exp(\mathbf{D}). \notag
\end{align}
$$

$$\mathbf{M}$$ above is a Toeplitz mask <d-cite key="qin2023toeplitz"></d-cite>, we can efficiently create a decay mask with such structure using simple PyTorch functions. To ensure numerical stability, we bound the parameter $$\lambda$$ with a **sigmoid**, setting $$\lambda = \sigma(a)$$. Without this constraint, the scalar $$\lambda^L$$ could become excessively large, leading to instability. In practice, for **RetNet-style mask** with a fixed decay, multiplication remains stable. Such mask can be implemented as follows:

```python
def decay_mask(a, length):
    idx = torch.arange(length, device=a.device)
    i, j = torch.meshgrid(idx, idx, indexing="ij")
    e = torch.abs((i - j)).float().view(1, 1, length, length)
    m = torch.sigmoid(a).view(1, -1, 1, 1) ** e
    return m
```

### LION-S

Observing the structure of $\mathbf{M}$, its upper ($\mathbf{M}^B$) and lower ($\mathbf{M}^F$) triangular parts are rank-1 [semi-separable matrices](https://people.cs.kuleuven.be/~raf.vandebril/homepage/publications/papers_html/qrq_07/node16.html) <d-cite key="dao2024transformers"></d-cite>, allowing for efficient computation via matrix multiplications.

During training, the decay factors $\lambda_i$ are stacked into ${\lambda}^F \in \mathbb{R}^L$, and the cumulative product

$$
\mathbf{L}^F = cumprod(\lambda^F) = \prod_{k=0}^{i} \lambda^F_k
$$

is used to generate the lower triangular mask $$\mathbf{M}^F$$. For the upper triangular mask $$\mathbf{M}^B$$, the input sequence is flipped, and the decay factors are computed as

$$
\boldsymbol{\lambda}^B = \text{Flip}(\boldsymbol{\lambda}^F), \quad \mathbf{L}^B = cumprod(\boldsymbol{\lambda}^B).
$$

The masks are then constructed as, $$\mathbf{M}^F =$$ `tril(LF@inv(LF)^T)` for the forward part and $$\mathbf{M}^B =$$ `triu(LB@inv(LB)^T)` for the backward part. Where `tril(.)` and `triu(.)` extract the lower and upper triangular parts of the input matrix respectively and `inv(.)` is a element wise inverse. The full mask is then obtained as

$$
\mathbf{M} = \mathbf{M}^F + \mathbf{M}^B - \mathbf{I}.
$$

To improve numerical stability, the selective scalar $\lambda_i$ is designed in exponential form

$$
\lambda_i = e^{a_i}.
$$

This results in the cumulative sum:

$$
\mathbf{D}^F_{ij} =
\begin{cases}
\sum_{k=i}^{j+1} a_k, & \text{if } i > j,  \\
- \infty, & \text{if } i < j,  \\
0, & \text{if } i = j,
\end{cases}
$$

$$
\mathbf{M^F} = \exp(\mathbf{D^F}),
$$

where $\exp(\cdot)$ is applied element-wise. The same process applies to $\mathbf{M}^B$ by flipping the input sequence order.

Here, $\mathbf{D}^{F/B} = cumsum(\mathbf{a}^{F/B})$, where $\mathbf{a} \in \mathbb{R}^L$ contains the selective exponents $a_i$.

Ensuring stability is crucial, as $\mathbf{L}^{F/B}$ can overflow or underflow when forming the full mask without chunking. To mitigate this, we define

$$
a_i = \log(\sigma(\mathbf{W}_{a}^\top\mathbf{x}_i + b)),
$$

where $\sigma(.)$ is the sigmoid function. By bounding $a_i$ within the interval $[0,1]$ we get numerical stability.

**Note:** We find using bounded activation functions to be important in practise since unbounded activations could cause NaN values in the loss function. To maintain stability, **Chunking** is required during training in Mamba and Hydra models when using the full sequence. This issue has been specifically highlighted in the **Mamba2** [blog post](https://goombalab.github.io/blog/2024/mamba2-part3-algorithm/) and can, again, be attributed to the softplus activation being unbounded. Since LION models use sigmoid activation, chunking is not required for training. In the **Results** section of this blog post, we explore why using **Full Attention** is beneficial for achieving **high throughput** during training.

The code for building the mask of LION-S is simple, a Pytorch implementation is provided below:

```python
def create_causal_mask_lions(tensor):
    cumsum = torch.cumsum(tensor, dim=-1)
    cumprod = torch.exp(cumsum)
    a = torch.matmul(
        cumprod.unsqueeze(-1), 1 / (cumprod.unsqueeze(-1).transpose(-1, -2) + 1e-7)
    )
    return torch.tril(a)
```

```python
def selective_mask(vec):
    vec_shape = vec.shape
    a_for = create_matrix_from_tensor(vec.unsqueeze(-1).transpose(-1, -2)).squeeze()
    a_back = (
        create_matrix_from_tensor(
            torch.cat(
                (vec, torch.ones((vec_shape[0], vec_shape[1], 1), device=vec.device)),
                dim=-1,
            )[:, :, 1:]
            .unsqueeze(-1)
            .transpose(-1, -2)
        )
        .transpose(-1, -2)
        .squeeze()
    )
    return a_for + a_back - torch.eye(a_for.shape[-1]).to(a_for.device)
```

## LION Attention Block

We can formulate the parallel Attention form of LION supporting all three extensions of our main experiments:

```python
class LIONAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        silunorm: bool = False,
        mask_type="Lit",
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.mask_type = mask_type

        if mask_type == "Lit":
            pass
        elif mask_type == "Selective":
            self.a_i = nn.Linear(dim, num_heads)
        elif mask_type == "Decay":
            self.a_i = nn.Parameter(torch.randn(num_heads))

        self.non_lin = silu_shifted
        self.silunorm = silunorm

    def forward(self, x):
        b, n, c = x.shape
        qkv = (
            self.qkv(x)
            .reshape(b, n, 3, self.num_heads, c // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            self.non_lin(qkv[0], silunorm=self.silunorm),
            self.non_lin(qkv[1], silunorm=self.silunorm),
            qkv[2],
        )

        a_i = self.a_i(x).transpose(-1, -2)
        a_i = torch.log(1 - torch.nn.functional.sigmoid(a_i) + 1e-7)

        if self.mask_type == "Selective":
            m = selective_mask(a_i)

        elif self.mask_type == "Decay":
            m = decay_mask(a_i)

        elif self.mask_type == "Lit":
            m = 1

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = m * attn

        # Scaling
        attn = attn / attn.sum(dim=-1, keepdim=True)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(b, n, c)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x
```

> **Question:** As seen above, the **RNN** is more efficient than the **Transformer** since it only requires storing the output for each token, resulting in a memory complexity of **$\mathcal{O}(Ld)$**, as opposed to storing the full Attention matrix, which requires **$\mathcal{O}(L^2 d)$**. Can we achieve a balance between the speed of Attention parallelism and the efficiency of an RNN?

We will answer this question in our next section by introducing LION-Chunk.

## Next Up

- In the next section of this series, we will describe how to apply a **chunkwise parallel form** for LION, allowing us to balance between the _RNN structure_ and the _Attention-based_ formulation.

- We show the numercial results and experiments on [Imagenet](https://www.image-net.org/) and [C4](https://paperswithcode.com/dataset/c4) dataset 😊.

[Continue reading to Part III - Chunkwise Parallel from of LION]({% post_url 2024-02-28-lion-part3-chunk %})
