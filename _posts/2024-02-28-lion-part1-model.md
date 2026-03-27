---
layout: distill
title: Phoenix (Part-1)
description: Memory as a set of Slots
tags:
giscus_comments: false
date: 2025-08-29
featured: false
thumbnail: assets/img/best.png

authors:
  - name: Arshia Afzal$^*$ (Bloggin 🎙️)
    url:
    affiliations:
      name: EPFL
  - name: Aviv Bick$^*$
    url:
    affiliations:
      name: CMU
  - name: Eric P.Xing
    url:
    affiliations:
      name: CMU, MBZUAI
  - name: Volkan Cevher
    url:
    affiliations:
      name: EPFL
  - name: Albert Gu
    url:
    affiliations:
      name: CMU, Cartesia.AI

bibliography: arshia.bib


toc:
  - name: Organize your memory like your closet
  - name: What Are SSMs Actually Doing?
  - name: The Other Extreme — SWA
  - name: Routing Slot Memories

---

<div style="width: 50%; margin: 0 auto;">
  {% include figure.liquid loading="eager" path="assets/img/best.png" %}
</div>

------------------

## Organize your memory like your closet 👕

Mamba <d-cite key="mamba"></d-cite> had just shown the world that you didn't need attention <d-cite key="attention"></d-cite> to build powerful sequence models. A simple, elegant recurrence could match and even beat Transformers on language modeling. We were excited — the efficiency gains alone were compelling. But then we started running recall benchmarks.

The results were humbling. Ask a linear model to retrieve a passkey buried deep in a long document, and it struggles. Ask it to remember a variable name from thousands of tokens ago in a codebase, and it forgets. Model after model — Mamba, Mamba-2 <d-cite key="mamba,mamba2"></d-cite> and their descendants — hit the same wall. The fixed-size memory that made them fast was also making them forgetful.

> *Can we design SSMs with better recall even with fixed-size memory?*

That question sat with us for a while. We kept coming back to the same underlying puzzle: it's not that the memory is too small — it's that the model doesn't know *how* to use it. Every new token gets smeared across the entire hidden state, diluting whatever was carefully stored before. The model isn't choosing where to write; it's writing everywhere at once.

Then, one afternoon, someone made an offhand remark about organizing a closet — and everything clicked.

Think about it. When you buy a new piece of clothing, you don't shove it randomly onto every shelf simultaneously. You take two deliberate steps:

**1. *Decide which shelf it belongs on.***
**2. *Squeeze and reorganize that shelf to make room.***

What if an SSM could do the same? **The closet is the memory**, **the shelves are the memory slots**, and **tokens are the clothes**. The squeezing part — making room by decaying old content — SSMs already do. What they don't do is *choose* which shelf to write to. They treat the entire hidden state as one big shelf.

That was the missing piece. We first needed to understand precisely what SSMs and their counterparts were doing to memory, and then figure out how to give them the ability to *choose*.


## What Are SSMs Actually Doing?

State space models <d-cite key="mamba,mamba2"></d-cite> and linear transformers <d-cite key="linearattn"></d-cite> all share the same fundamental memory operation — a matrix-valued hidden state $S_t \in \mathbb{R}^{M\times d}$ updated by a **linear time-dependent** recurrence:

$$
S_t = \underbrace{S_{t-1}A_t}_{\text{Decay}} + \underbrace{v_t k_t^\top}_{\text{Write}}, \qquad o_t = \underbrace{S_t q_t}_{\text{Read}}
$$

At each step $t$, the input $x_t \in \mathbb{R}^d$ is projected into queries, keys, and values: $q_t, k_t \in \mathbb{R}^d$ and $v_t \in \mathbb{R}^M$ <d-footnote> SSMs are often written with ($x,B,C$); we use $(v,k,q)$ for a read/write interpretation and to stay consistent with later attention-based special cases (e.g., sliding-window attention). </d-footnote>. The output $o_t\in \mathbb{R}^d$ is read from the state, and the forget gate $A_t \in \mathbb{R}^{d\times d}$ controls how much of the past survives.

Different models instantiate $A_t$ differently, but they all use it as a mechanism for *squeezing* the past to make room for the present:

{% details Forget Gates $A_t$ of Different Linear Models %}
- **Mamba-2** <d-cite key="mamba2"></d-cite>:
$A_t = a_t =  \exp (-\mathrm{SoftPlus}(w^\top x_t)\,\exp(\Delta) )$

- **GLA** <d-cite key="gla"></d-cite>:
$A_t = \mathrm{diag}(\sigma(W^\top x_t)^\tau)$

- **DeltaNet** <d-cite key="deltanet"></d-cite>:
$A_t = I - \sigma(w^\top x_t) k_t k_t^\top$

- **Gated DeltaNet** <d-cite key="gated_deltanet"></d-cite>:
$A_t = a_t \left(I - \sigma(w^\top x_t) k_t k_t^\top\right)$

{% enddetails %}

In the diagonal case $A_t=\text{diag}(a_t)\in\mathbb{R}^d$, the update simplifies to:

$$
S_t = S_{t-1} \odot a_t + v_t k_t^\top
$$

The scalar $a_t$ squeezes information uniformly across all $d$ channels — like compressing every shelf in the closet at once. Even for models like DeltaNet and Gated DeltaNet where the non-diagonal $A_t$ also reshuffles content between channels, the write $v_t k_t^\top$ still lands across *all* $M$ rows of $S_t$. There is no choosing. Every token touches every shelf.

This, we realized, is the root of the recall problem. And once we saw it clearly, we started wondering: is there another model out there that does something completely different?


## The Other Extreme — SWA

Sliding window attention keeps a fixed cache of the past $M$ tokens, and at every step it does something deceptively simple: it **$\color{red}{\text{drops}}$** the oldest token and **$\color{green}{\text{appends}}$** the newest one. With key and value caches $S^k_t, S^v_t\in \mathbb{R}^{M\times d}$:

$$
[S^k_t, S^v_t] = [S^k_{t-1}, S^v_{t-1}].{\color{red}{\text{drop}}\color{black}{(k_{t-M}, v_{t-M})}}.{\color{green}{\text{append}}\color{black}{(k_t, v_t)}}, \qquad o_t= (S^v_t)^\top \text{SoftMax}(S^k_t q_t)
$$

Written as a matrix update — using a one-hot selector $e_t$ that points to the oldest slot — this becomes:

$$
[S^{k}_t, S^v_t] = \underbrace{(\mathbf{1} -  e_t)}_{\color{red}{\text{Remove last token}}}\odot[S^{k}_{t-1}, S^v_{t-1}]  +  \underbrace{e_t}_{\color{green}{\text{Write new token}}} [k_t^\top, v_t^\top]
$$

where the selector cycles through slots like a ring buffer:

$$
e_t[i] =
\begin{cases}
1 & \text{if } i = m_t \\
0 & \text{otherwise}
\end{cases}
\quad \text{where} \quad
m_t = 1 + ((t-1)\bmod M)
$$

Here was the revelation: **SWA chooses where to write**. Unlike SSMs, which broadcast every token across all slots, SWA writes each token to exactly one slot — the one holding the oldest entry. Naive, yes — but it genuinely *is* choosing.

The tradeoff, though, is brutal. SWA doesn't squeeze the old content; it **erases it entirely** — like clearing a shelf completely before placing a new shirt on it. SSMs squeeze gracefully but write blindly. SWA writes precisely but deletes ruthlessly. Both approaches are incomplete on their own.

We stared at this contrast for a while. Two models, each with exactly one of the two capabilities we needed. And that made us suspect that hybrid models — the ones combining SWA layers with SSM layers — derive much of their power precisely from this complementarity: each component doing something fundamentally different from the other.

> So we set out to design a single sequence mixer that could both **choose** where to write *and* **gracefully decay** what's already there.

And as a bonus — much like organizing a closet by clothing type — this turns out to naturally cluster similar tokens into similar memory slots.


## Routing Slot Memories

Instead of the ring-buffer's fixed one-hot selector $e_t$, what if we used a learned, *sparse* router vector $r_t \in \mathbb{R}^M$ that picks which slots each token gets written to — based on the *content* of that token? This is the heart of **Routing Slot Memories (RSMs)**:

$$
S_t = \underbrace{(\mathbf{1} - r_t) \odot S_{t-1}}_{\text{Preserved memory}}
+ \underbrace{r_t \odot \bigl(D_t\,S_{t-1}\,A_t + U_t\bigr)}_{\text{Updated memory}},
$$

The router $r_t$ is sparse — most entries are zero. Only the selected slots get updated; the rest are left completely untouched. This is the *choosing* step. Within those selected slots, the term $D_t S_{t-1} A_t + U_t$ handles the *squeezing and writing* — and by varying $D_t$, $A_t$, and $U_t$, this single formulation recovers the full spectrum from SSMs to SWA as special cases.

{% include figure.liquid loading="eager" path="assets/img/tab.png" title="RSM Spectrom" caption=" Design of the Router and Decay components across multiple architectures. The bar on the right along with row colors display each model's spectrum as a function of router sparsity, with SWA and SSMs marking the two extremes." %}

Dense router $r_t = \mathbf{1}$ gives you an SSM. One-hot $r_t = e_t$ gives you SWA. Everything in between is new territory.

For every token, only the content of the selected slots gets decayed — exactly like squeezing a specific shelf to make room, while leaving the other shelves untouched.

{% include figure.liquid loading="eager" path="assets/img/mem.png" title="RSM Vis" caption=" Visualization of three different sequence mixers using Routing Slot Memories, with different router choices.  (a) SWA memory allocation uses a first-in-first-out strategy with a one-hot vector $\mathbf{e}_t$ as the router.  (b) SSM memory allocation projects each token into all memory slots using a dense, all-ones router $\mathbf{1}_M$.  (c) Phoenix memory allocation uses a selective router for writes. The visualization shows a sequence of $T=6$ tokens and $M=4$ memory slots for the hidden state $\mathbf{S}_t$, with $\text{Top}_K = 1$ for the \ours router." %}

And so the question became concrete:

> Can we train a model that learns, purely from data, *which shelf each token belongs on* — and then keeps it there?

The answer is **Phoenix** — a model with a **learned sparse router $r_t$**, trained end-to-end, to route tokens to memory slots based on content. From a **MoE** <d-cite key="moe"></d-cite> perspective, the slots act as specialized experts: a slot that learns to hold passkeys keeps holding passkeys, undisturbed by the stream of ordinary text flowing through the other shelves.

This is why Phoenix can store a passkey once and retrieve it 32,000 tokens later. Not because the memory is larger — but because it's organized.

In Part 2, we'll walk through how we actually built Phoenix from this framework, the design decisions that surprised us along the way, and what happened when we finally ran the experiments.

----------------
