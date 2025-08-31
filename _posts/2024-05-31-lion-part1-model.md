---
layout: distill
title: Story of Linear Time Sequence Modeling 📚
description: Summary of Linear Transformers
tags:
giscus_comments: false
date: 2025-08-29
featured: false
thumbnail: assets/img/lion.jpg

authors:
  - name: Arshia  
    url:
    affiliations:
      name: EPFL
  - name: Sajad 
    url:
    affiliations:
      name: ELISS
  - name: Timur
    url:
    affiliations:
      name: ELISS

bibliography: albert.bib


toc:
  - name: Sequence Modeling Core
  - name: From Causal to Full Linear Attention
  - name: Creating Scaled and Masked Full Attention
 
---


<!-- 
1. Part I - Full Linear Attention
2. [Part II - Bi-directional RNN]({% post_url 2024-05-31-lion-part2-theory %})
3. [Part III -  Chunkwise Parallel from of LION]({% post_url 2024-05-31-lion-part3-chunk %})
4. [Part IV - Results]({% post_url 2024-05-31-lion-part4-results %})
 -->

------------------

# Why Linear Transformers?

If you’ve heard about Large Language Models (LLMs) and Transformers and are curious to learn more, there are already plenty of excellent blog posts, articles, and YouTube videos that explain them in great detail with amazing visualizations. In case your main interest is understanding LLMs and how Transformers work, we’d point you there first, they’ve done a fantastic job (honestly, better than us).

In this series, we’ll cover Linear Transformers and State Space Models (SSMs), giving a high-level summary of their core ideas. So, if you’ve come across names like **Mamba** or **DeltaNet** and wondered what they are, or if you’ve asked yourself: 

**1)** Why we moved from RNNs to Transformers?

**2)** Why we now seem to be circling back from Transformers toward Linear Transformers (almost like “RNNs on steroids”)?
 
then this post should be a good fit 😉.

So let’s start from answering the above questions 



----------------

Recently, Transformers with Linear Attention <d-cite key="katharopoulos2020transformers"></d-cite> and State Space Models <d-cite key="gu2023mamba"></d-cite> <d-cite key="gu2022efficiently"></d-cite> <d-cite key="dao2024transformers"></d-cite> (SSMs) have gained significant popularity for causal sequence modeling due to their ability to efficiently support both parallel training and RNN-like inference.
These models have demonstrated impressive accuracy in causal tasks, particularly in causal language modeling. For bi-directional sequence modeling, SSMs, particularly Mamba <d-cite key="gu2023mamba"></d-cite>, have been evaluated in vision tasks along with architecture iterations like Vision Mamba <d-cite key="zhu2024vision"></d-cite> and Hydra <d-cite key="hwang2025hydra"></d-cite>. However, Transformers with Linear Attention have been less explored in the bi-directional setting.

We are curious to explore whether Linear Attention Transformers, including the vanilla Linear Transformer <d-cite key="katharopoulos2020transformers"></d-cite> or RetNet <d-cite key="sun2023retentive"></d-cite> can perform effectively on bi-directional sequence modeling. More specifically, what modifications are needed to adapt them for tasks like image classification and masked language modeling? 😊

Let's break this down with three key questions:

### Question 1 (Applicability)

Given that Linear Transformers can be formulated as RNNs, offering efficiency benefits during inference and enabling parallel training for causal sequence modeling, can they also provide similar advantages for bi-directional processing? If so, what would the parallel form be, and how would the equivalent bi-directional RNN be structured?

### Question 2 (Performance)

Can simple Linear Transformers, like Linear Transformer <d-cite key="katharopoulos2020transformers"></d-cite> or RetNet <d-cite key="sun2023retentive"></d-cite>, perform well on bi-directional tasks such as image classification or masked language modeling?

### Question 3 (Training Throughput)

While bi-directional SSMs are performant, they tend to be difficult and slow to train compared to Transformers with Full Attention (e.g., ViT <d-cite key="dosovitskiy2020image"></d-cite> and BERT <d-cite key="devlin2018bert"></d-cite>). Can Linear Transformers match the accuracy and efficiency of bi-directional SSMs while maintaining the training throughput of Softmax Transformers?


## From Causal to Full Linear Attention

Let's start with Linear Attention Recurrence:

$$
\begin{aligned} 
& S_i = S_{i-1} + k_i v^\top_i, \quad z_i =  z_{i-1} + k_i, \\
& Scaled: y_i = \frac{q^\top_i S_i}{q^\top_i z_i}, \quad Non-Scaled: y_i  = q^\top_i S_i \\ 
\end{aligned}
$$

Above is the RNN form of the Linear Attention which has the parallel form of:

$$\mathbf{Y} = Scale \left(\mathbf{Q} \mathbf{K}^\top  \odot \mathbf{M}^C \right)$$


and the mask $$\mathbf{M}^C$$ is a lower triangular $$C$$ausal mask. Causal Linear Transformers are a class of models introduced following the development of the original Linear Transformer as shown above <d-cite key="katharopoulos2020transformers"></d-cite>. These models typically define a recurrence of the form:  

$$
\begin{aligned} 
S_i = \boldsymbol{\Lambda_i} \star S_{i-1} + \gamma_i k_i v^\top_i, \quad z_i = \boldsymbol{\Lambda_i} \star z_{i-1} + \gamma_i k_i, \\
Scaled: y_i = \frac{q^\top_i S_i}{q^\top_i z_i}, \quad Non-Scaled: y_i  = q^\top_i S_i \\ 
\end{aligned}
$$

Here, $$\boldsymbol{\Lambda_i}$$ and $$\gamma_i$$ are decay factors introduced after the Linear Transformer to enhance the performance and $$\star$$ denotes an associative operator which depends on the specific model. (Spoiler alert ⚠️: the family of Linear Transformers has strong connections to SSMs, as explored in works like Deltanet <d-cite key="yang2024parallelizing"></d-cite> and Mamba2 <d-cite key="dao2024transformers"></d-cite> through state space duality (SSD) 😉). Many models apply a non-linear activation to queries and keys, such that $$\mathbf{k}_i = \phi(\mathbf{k}_i)$$ and $$\mathbf{q}_i = \phi(\mathbf{q}_i)$$. To avoid notation clutter, we omit explicitly writing $$\phi(.)$$ everywhere assuming by default that queries and keys are already non-linearized. For simplicity, we consider $$\boldsymbol{\Lambda_i} = \lambda_i$$ as a scalar and $$\gamma_i = 1$$. We now present the general Scaled Linear Attention in the following form:

$$
\begin{aligned} 
S_i &= \lambda_i  S_{i-1} + k_i v^\top_i,\\
z_i &= \lambda_i  z_{i-1} + k_i, \\
y_i &= \frac{q^\top_i S_i}{q^\top_i z_i} \\ 
\end{aligned}
$$

The first goal is to extend the Causal Linear Attention parallel form  

$$
\mathbf{Y} = \text{Scale} \left(\mathbf{Q} \mathbf{K}^\top  \odot \mathbf{M}^C \right)
$$

to a Scaled and Masked Full Linear Attention mechanism.

## Creating Scaled and Masked Full Attention

The first step is quite simple: the Masked and Scaled Attention can naturally take the following form, as suggested by its name:

> **Full Linear Attention**
> 
> $$ \mathbf{Y} = \text{Scale} \left(\mathbf{Q} \mathbf{K}^\top  \odot \mathbf{M} \right)$$
{: .block-tip}

The important part is how to well define the matrix $$\mathbf{M}$$.
A natural choice is to extend the causal mask $$\mathbf{M^C}$$, where the causal mask between tokens $$i,j$$ is given by $$\mathbf{M}^C_{ij} = \lambda_{j+1} \lambda_{j+2} \dots \lambda_i$$, representing the product of all selective scalers between $$i$$ and $$j$$.
In the bi-directional case, the full mask should preserve this desirable property. One can interpret the mask entries as a relative positional encoding between two tokens taking the following form:

$$
\begin{aligned}
    \mathbf{M}_{ij} = 
    \begin{cases} 
    \Pi_{k=j}^{i-1}{\lambda_k}, & i > j  \\
    1 & i=j\\ 
    \Pi_{k=i+1}^{j}{\lambda_k}, & i < j.
\end{cases}
\end{aligned}
$$


To recap, the full output of Full Linear Attention can be presented as:

<span style="font-size: 0.7em;">
$$
\mathbf{Y} = Scale
    \left(
   \underbrace{\left( \renewcommand*{\arraystretch} \begin{array}{ccccc}
       \mathbf{q}_1^{\top}\mathbf{k}_1  &  \mathbf{q}_1^{\top}\mathbf{k}_2 & \cdots &  \mathbf{q}_1^{\top}\mathbf{k}_L \\
     \mathbf{q}_2^{\top}\mathbf{k}_1  &   \mathbf{q}_2^{\top}\mathbf{k}_2  &   \cdots &  \mathbf{q}_2^{\top}\mathbf{k}_L\\
     \vdots &  \vdots & \ddots  &  \vdots \\
      \mathbf{q}_L^{\top}\mathbf{k}_1 &   \mathbf{q}_L^{\top}\mathbf{k}_2  &   \cdots  & 
     \mathbf{q}_L^{\top}\mathbf{k}_L\\
  \end{array} \right)}_{\hspace{1mm} \mathbf{A} = \mathbf{Q} \mathbf{K}^{\top}}  \odot
   \underbrace{ \left(  \renewcommand*{\arraystretch} \begin{array}{ccccc}
    1  & \lambda_2 & \lambda_2 \lambda_3  & \cdots & \lambda_2 \cdots \lambda_L \\
    \lambda_1 &  1 & \lambda_3 & \cdots & \lambda_3 \cdots \lambda_L \\
    \lambda_2 \lambda_1 & \lambda_2 & 1 & \cdots & \lambda_4 \cdots \lambda_L \\
    \vdots & \vdots & \vdots & \ddots &  \vdots \\
    \lambda_{L-1} \cdots \lambda_1 & \lambda_{L-1} \cdots \lambda_2 & \lambda_{L-1} \cdots \lambda_3 & \cdots &   1 \\   
\end{array}  \right)  }_{\hspace{1mm} \mathbf{M}}  \right) \left( \renewcommand*{\arraystretch} \begin{array}{c}
    \mathbf{v}_1^\top \\  
    \mathbf{v}_2^\top \\
    \mathbf{v}_3^\top \\  
    \vdots \\
    \mathbf{v}_L^\top \\   
  \end{array} \right)
$$
</span>


The equation above represents the Full **Linear** Attention in parallel form. Now that we have established Full Linear Attention for bi-directional sequence modeling, it's time to derive its equivalent bi-directional RNN.


### **An Important Question:**

> **Question:** Is it worth training with Full Attention on bi-directional tasks considering it has quadratic complexity with sequence length $$O(L^2)$$?

The answer is **yes**! Unlike causal language modeling, for bi-directional tasks such as Vision ($L=196$) and Masked Language Modeling (MLM) ($L=128$), sequence lengths used in practice are relatively short. This means that we can usually fit Full Attention in memory enalbing higher throughput without a significant trade-off in complexity.

We believe that architectures designed for causal tasks can really benefit from modifications to adapt them to the bi-directional domain.

## Next Up

- We introduce our framework, **LION**, which derives an equivalent bi-directional RNN for Full Linear Attention.  

- Within this framework, we demonstrate how different Linear Transformers can be extended to their bi-directional counterparts.  

- We explore the construction of stable masks $$\mathbf{M}$$, enabling models using LION to **TRAIN IN PARALLEL** using Full Attention and **INFER EFFICIENTLY** like an RNN.

- Finally, we introduce a **chunkwise parallel** variant of LION to balance recurrence and parallelism 🙂.

[Continue reading to Part II - Bi-directional RNN]({% post_url 2024-05-31-lion-part2-theory %})

*Acknowledgement:* We appreciate [Albert Gu](https://goombalab.github.io/) and [Tri Dao](https://tridao.me/blog/) for their insightful blog posts, which have been helpful in shaping our own.