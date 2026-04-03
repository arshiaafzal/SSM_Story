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
  - name: Arshia Afzal$^*$
    url:
    affiliations:
      name: EPFL
  - name: Aviv Bick$^*$ (Bloggin 🎙️)
    url:
    affiliations:
      name: CMU
  - name: Eric P. Xing
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
  - name: The Recall Gap
  - name: Organize your memory like your closet 👕
    subsections:
      - name: SSMs' closet is a pile of clothes
      - name: SWA's closet throws out the old clothes
      - name: The Missing Piece
  - name: Routing Slot Memories

---

## The Recall Gap
Mamba <d-cite key="mamba"></d-cite> showed that attention <d-cite key="attention"></d-cite> is not strictly necessary for strong sequence modeling. A simple recurrent model could match, and sometimes outperform, Transformers on language modeling at a fraction of the cost. But there is no free lunch.
Soon after, several works exposed a critical limitation: fixed-size memory models struggle on recall benchmarks, which is not surprising, since they have only a finite amount of room to store information. 
While it’s tempting to blame their finite capacity, the reality is more nuanced.

Consider Needle-In-The-Haystack (NIAH). The model is first given a lookup key, then a long dictionary, and finally asked to retrieve the matching value. This is a form of **constant-memory recall**: in principle, it requires only $O(1)$ memory, since the model only needs to retain the key and its corresponding value as it scans the sequence (unlike **scaling-memory recall** tasks like KV-Retrieval, which require $O(T)$ memory because the query comes after the dictionary).

Despite this theoretical $O(1)$ requirement, recurrent models still lag behind Transformers on NIAH. This failure isn't just a quirk of synthetic benchmarks; it mirrors real-world bottlenecks. 
In multiple-choice questions, a model may identify the correct answer early on but fail to preserve that single letter in its state until the output <d-cite key="gna"></d-cite>.
Similarly, in math and code, models struggle to track a single variable or a few intermediate values over long distances.


The issue, then, isn't a lack of physical room—it’s a lack of **persistence**: the ability to leave specific information untouched. The model has the capacity; it just lacks the organization.

## Organize your memory like your closet 👕 {#organize-your-memory-like-your-closet}

> *Can we design a recurrent model with strong constant-memory recall that approach their theoretical limits?*

To understand why persistence is important, think of your hidden state as a closet. When you buy a new piece of clothing, you don't shove it into a giant pile on the floor, nor do you throw out your favorite jacket just to fit a new shirt. Instead, you take two deliberate steps:

**1. *Decide which shelf it belongs on.*** (Routing)   
**2. *Reorganize that shelf to make room.*** (Squeezing / Decay)

In this analogy, the **closet** is the memory, the **shelves** are memory slots, and tokens are the **clothes**. Current models fail because they only perform one of these steps effectively.

### SSMs' closet is a pile of clothes

State space models (SSMs) <d-cite key="mamba,mamba2"></d-cite> and linear transformers <d-cite key="linearattn"></d-cite> all share the same fundamental memory operation — a matrix-valued hidden state $S_t \in \mathbb{R}^{M\times d}$ updated by a **linear time-dependent** recurrence:

$$
S_t = \underbrace{S_{t-1}A_t}_{\text{Decay}} + \underbrace{v_t k_t^\top}_{\text{Write}}, \qquad o_t = \underbrace{S_t q_t}_{\text{Read}}
$$

SSMs are masters of the **reorganizing** step. The forget gate $A_t$ acts as a mechanism for decaying old content to make room for the new. However, they lack the ability to choose a shelf.

In the diagonal case ($A_t=\text{diag}(a_t)$), the update simplifies to $S_t = S_{t-1} \odot a_t + v_t k_t^\top$. Here, the scalar $a_t$ squeezes information uniformly across the entire state—like compressing every shelf in the closet simultaneously. Because the write $v_t k_t^\top$ lands across all $M$ rows, every new token dilutes everything previously stored. There is no isolation; there is only the pile.

<div style="margin: 1.5rem auto 1rem; display: flex; justify-content: center; width: 100%;">
  <iframe
    id="ssm-recurrent-matrix-update"
    src="{{ '/assets/html/ssm_recurrent_matrix_update.html' | relative_url }}"
    title="SSM recurrent matrix update visualization"
    loading="lazy"
    scrolling="no"
    style="display: block; width: 70%; max-width: 1400px; height: 0; border: 0; border-radius: 0rem; background: transparent; overflow: hidden;"
  ></iframe>
</div>

### SWA's closet throws out the old clothes

Sliding window attention (SWA) takes the opposite approach. It maintains a fixed cache and, at each step, $\color{red}{\text{drops}}$ the oldest token to $\color{green}{\text{append}}$ the newest one. Written as a matrix update using a one-hot selector $e_t$:

$$[S^{k}_t, S^v_t] = \underbrace{(\mathbf{1} -  e_t)}_{\color{red}{\text{Remove oldest}}}\odot[S^{k}_{t-1}, S^v_{t-1}]  +  \underbrace{e_t}_{\color{green}{\text{Write new}}} [k_t^\top, v_t^\top]$$

SWA is great at choosing a shelf; it writes to exactly one slot ($e_t$). But it cannot reorganize. It doesn't squeeze old content to make room; it deletes it ruthlessly to make space for the new.

<div style="margin: 1.5rem auto 1rem; display: flex; justify-content: center; width: 100%;">
  <iframe
    id="swa-recurrent-matrix-update"
    src="{{ '/assets/html/swa_recurrent_matrix_update.html' | relative_url }}"
    title="SWA recurrent matrix update visualization"
    loading="lazy"
    scrolling="no"
    style="display: block; width: 70%; max-width: 1400px; height: 0; border: 0; border-radius: 0rem; background: transparent; overflow: hidden;"
  ></iframe>
</div>

### The Missing Piece

If SSMs write blindly and SWA deletes ruthlessly, the solution is a model that can choose where to write based on what it is seeing.

Bridging the recall gap requires a single mechanism that can do both: treat subsets of the hidden state as independent slots that can be selectively updated or gracefully preserved. This dual requirement—routing and decay—is the foundation of the framework.

## Routing Slot Memories

Instead of SWA’s fixed ring-buffer ($e_t$), what if we used a learned, sparse router vector $r_t \in \mathbb{R}^M$? This router looks at the current token and picks exactly which "shelves" it belongs on. 
This is the core of Routing Slot Memories (RSMs):

$$S_t = \underbrace{(\mathbf{1} - r_t) \odot S_{t-1}}_{\text{Preserved Shelves}}
+ \underbrace{r_t \odot \bigl(D_t\,S_{t-1}\,A_t + U_t\bigr)}_{\text{Updated Shelves}}$$

The beauty of this formulation is that it’s a spectrum. Because $r_t$ is sparse, most entries are zero—meaning most of the memory is left completely untouched.
- The First Term $(\mathbf{1} - r_t) \odot S_{t-1}$ is the persistence mechanism. It ensures that any shelf not selected by the router remains perfectly preserved.
- The Second Term $r_t \odot (\dots)$ is the selective update. Only the chosen shelves undergo the "squeezing" ($D_t, A_t$) and "writing" ($U_t$) process.

By simply changing the behavior of the router $r_t$, this single equation recovers the entire landscape of current sequence models:x

<div style="margin: 1.5rem auto 1rem; display: flex; justify-content: center; width: 100%; overflow-x: auto;">
  <iframe
    id="comparison-table"
    src="{{ '/assets/html/comparison_table.html' | relative_url }}"
    title="Comparison table"
    loading="lazy"
    scrolling="no"
    style="display: block; width: 100%; max-width: 100%; height: 0; border: 0; border-radius: 0rem; background: transparent; overflow: hidden;"
  ></iframe>
</div>

A dense router ($r_t = \mathbf{1}$) gives you an SSM—every token hits every shelf. A one-hot cyclic router ($r_t = e_t$) gives you SWA—every token hits the oldest shelf.

The question then became concrete: Can we train a model to learn, purely from data, which shelf each token belongs on—and keep it there?

Our answer is **Phoenix**. By using a **learned sparse router**, Phoenix treats its hidden state like a Mixture-of-Experts  <d-cite key="moe"></d-cite> for memory. Much like an organized closet, it can tuck a needle into a dedicated "retrieval shelf" and leave it undisturbed by the thousands of "filler" tokens flowing through other slots.

In **Part 2**, we’ll move from theory to practice: we'll look at the specific architecture of the Phoenix block, the "counterintuitive" design decisions that made it work, and how this organized memory allows it to recall information $16\times$ beyond its training length.


{% comment %}


<!-- We realized that hybrid models—which combine SWA and SSM layers—derive their power from this complementarity. But could we design a single sequence mixer that does both?

We needed a model that could treat subsets of its hidden state as independent slots: shelves that can be selectively updated, decayed, or preserved. -->

<!-- Sliding window attention keeps a fixed cache of the past $M$ tokens, and at every step it does something deceptively simple: it **$\color{red}{\text{drops}}$** the oldest token and **$\color{green}{\text{appends}}$** the newest one. With key and value caches $S^k_t, S^v_t\in \mathbb{R}^{M\times d}$:

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

Unlike SSMs, which broadcast every token across all slots, SWA writes a token to exactly one slot — the one holding the oldest entry. Naive, yes — but it genuinely *is* choosing.

The tradeoff, though, is brutal. SWA doesn't squeeze the old content; it **erases it entirely** — like clearing a shelf completely before placing a new shirt on it. SSMs squeeze gracefully but write blindly. SWA writes precisely but deletes ruthlessly. Both approaches are incomplete on their own.

We stared at this contrast for a while. Two models, each with exactly one of the two capabilities we needed. And that made us suspect that hybrid models — the ones combining SWA layers with SSM layers — derive much of their power precisely from this complementarity: each component doing something fundamentally different from the other.

> So we set out to design a single sequence mixer that could both **choose** where to write *and* **gracefully decay** what's already there.

And as a bonus — much like organizing a closet by clothing type — this turns out to naturally cluster similar tokens into similar memory slots. -->


<!-- <div style="text-align: center;">
  <img src="{{ '/assets/video/swa.gif' | relative_url }}" alt="Description of gif" style="width: 400px;">
</div> -->

<!-- At each step $t$, the input $x_t \in \mathbb{R}^d$ is projected into queries, keys, and values: $q_t, k_t \in \mathbb{R}^d$ and $v_t \in \mathbb{R}^M$ <d-footnote> SSMs are often written with ($x,B,C$); we use $(v,k,q)$ for a read/write interpretation and to stay consistent with later attention-based special cases (e.g., sliding-window attention) </d-footnote>. The output $o_t\in \mathbb{R}^d$ is read from the state, and the forget gate $A_t \in \mathbb{R}^{d\times d}$ controls how much of the past survives.

Different models instantiate $A_t$ differently, but they all use it as a mechanism for *reorganizing* the past to make room for the present:

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
Every token gets written across the entire hidden state, and every new token dilutes the carefully stored information from before. The model isn't choosing where to write; it's writing everywhere at once.
The scalar $a_t$ squeezes information uniformly across all $d$ channels — like compressing every shelf in the closet at once. Even for models like DeltaNet and Gated DeltaNet where the non-diagonal $A_t$ also reshuffles content between channels, the write $v_t k_t^\top$ still lands across *all* $M$ rows of $S_t$. There is no choosing. Every token gets written across every slot.

This, we realized, is the root of the recall problem. And once we saw it clearly, we started wondering: is there another model out there that does something completely different? -->

<!-- <div style="text-align: center;">
  <img src="{{ '/assets/video/ssm.gif' | relative_url }}" alt="Description of gif" style="width: 400px;">
</div> -->

<!-- Current recurrent models do one of these steps, but not both.
SSMs do the reorganizing part: they decay old content to make room for new content. What they do not do is choose which shelf to write to.
On the other hand, sliding window attention (SWA) does the choosing part: it writes to one specific slot. But it does not do any reorganizing; it simply drops the old content in that slot to make room for the new token. -->

<!-- Transformers handle this much better because they effectively have an unlimited number of shelves: each token can keep its own slot, even when that level of separation is unnecessary. -->

<!-- This, for us, was the missing piece. Before changing the architecture, we first had to take stock of what SSMs and Transformers were actually doing to memory, and then figure out how to give SSMs the ability to *choose* where to write.

This leads us the definition of slots. A slot is a subset of the hidden state that can be updated independently of the rest. -->


{%comment%}
<div style="width: 50%; margin: 0 auto;">
  {% include figure.liquid loading="eager" path="assets/img/best.png" %}
</div>
{% endcomment %}

<!-- ------------------ -->

<!-- ## Organize your memory like your closet 👕

Mamba <d-cite key="mamba"></d-cite> had just shown the world that you didn't need attention <d-cite key="attention"></d-cite> to build powerful sequence models. A simple, elegant recurrence could match and even beat Transformers on language modeling. We were excited — the efficiency gains alone were compelling. But then we started running recall benchmarks.

The results were humbling. Ask a linear model to retrieve a passkey buried deep in a long document, and it struggles. Ask it to remember a variable name from thousands of tokens ago in a codebase, and it forgets. Model after model — Mamba, Mamba-2 <d-cite key="mamba,mamba2"></d-cite> and their descendants — hit the same wall. The fixed-size memory that made them fast was also making them forgetful.

> *Can we design SSMs with better recall even with fixed-size memory?*

That question sat with us for a while. We kept coming back to the same underlying puzzle: it's not that the memory is too small — it's that the model doesn't know *how* to use it. Every new token gets smeared across the entire hidden state, diluting whatever was carefully stored before. The model isn't choosing where to write; it's writing everywhere at once.

Then, one afternoon, someone made an offhand remark about organizing a closet — and everything clicked.

Think about it. When you buy a new piece of clothing, you don't shove it randomly onto every shelf simultaneously. You take two deliberate steps:

**1. *Decide which shelf it belongs on.***
**2. *Squeeze and reorganize that shelf to make room.***

What if an SSM could do the same? **The closet is the memory**, **the shelves are the memory slots**, and **tokens are the clothes**. The squeezing part — making room by decaying old content — SSMs already do. What they don't do is *choose* which shelf to write to. They treat the entire hidden state as one big shelf.

That was the missing piece. We first needed to understand precisely what SSMs and their counterparts were doing to memory, and then figure out how to give them the ability to *choose*. -->

<!--
Instead of the ring-buffer's fixed one-hot selector $e_t$, what if we used a learned, *sparse* router vector $r_t \in \mathbb{R}^M$ that picks which slots each token gets written to — based on the *content* of that token? This is the heart of **Routing Slot Memories (RSMs)**:

$$
S_t = \underbrace{(\mathbf{1} - r_t) \odot S_{t-1}}_{\text{Preserved memory}}
+ \underbrace{r_t \odot \bigl(D_t\,S_{t-1}\,A_t + U_t\bigr)}_{\text{Updated memory}},
$$

The router $r_t$ is sparse — most entries are zero. Only the selected slots get updated; the rest are left completely untouched. This is the *choosing* step. Within those selected slots, the term $D_t S_{t-1} A_t + U_t$ handles the *squeezing and writing* — and by varying $D_t$, $A_t$, and $U_t$, this single formulation recovers the full spectrum from SSMs to SWA as special cases.

{% include figure.liquid loading="eager" path="assets/img/tab.png" title="RSM Spectrum" caption=" Design of the Router and Decay components across multiple architectures. The bar on the right along with row colors display each model's spectrum as a function of router sparsity, with SWA and SSMs marking the two extremes." %}

Dense router $r_t = \mathbf{1}$ gives you an SSM. One-hot $r_t = e_t$ gives you SWA. Everything in between is new territory.

For every token, only the content of the selected slots gets decayed — exactly like squeezing a specific shelf to make room, while leaving the other shelves untouched. -->

<!-- <figure style="text-align: center;">
  <video autoplay loop muted playsinline
         style="width: 90%; max-width: 750px;">
    <source src="{{ '/assets/video/memory_cells.mp4' | relative_url }}" type="video/mp4">
  </video>
  <figcaption style="margin-top: 0.6em; font-size: 0.85em; color: #555; line-height: 1.5;">
    Visualization of three different sequence mixers using Routing Slot Memories, with different router choices.
    <strong>(a)</strong> SWA memory allocation uses a first-in-first-out strategy with a one-hot vector $\mathbf{e}_t$ as the router.
    <strong>(b)</strong> SSM memory allocation projects each token into all memory slots using a dense, all-ones router $\mathbf{1}_M$.
    <strong>(c)</strong> Phoenix memory allocation uses a selective router for writes.
    The visualization shows a sequence of $T=6$ tokens and $M=4$ memory slots for the hidden state $\mathbf{S}_t$, with $\mathrm{Top}_K = 1$ for the Phoenix router.
  </figcaption>
</figure> -->

<!-- And so the question became concrete:

> Can we train a model that learns, purely from data, *which shelf each token belongs on* — and then keeps it there?

Our answer to that is **Phoenix** — a model with a **learned sparse router $r_t$**, trained end-to-end, to route tokens to memory slots based on content. This is somewhat similar to **Mixture-of-Experts (MoE)** <d-cite key="moe"></d-cite> , with the slots act as specialized experts: a slot that learns to hold passkeys keeps holding passkeys, undisturbed by the stream of ordinary text flowing through the other shelves.

This is why Phoenix can store a passkey once and retrieve it 32,000 tokens later. Not because the memory is larger — but because it's organized.

In Part 2, we'll walk through how we actually built Phoenix from this framework, the design decisions that surprised us along the way, and what happened when we finally ran the experiments. -->
<!-- ---------------- -->
{% endcomment %}






<script>
  (() => {
    const iframe = document.getElementById('swa-recurrent-matrix-update');
    if (!iframe) return;

    const resizeIframe = () => {
      try {
        const doc = iframe.contentDocument || iframe.contentWindow?.document;
        if (!doc || !doc.body || !doc.documentElement) return;

        const height = Math.max(
          doc.body.scrollHeight,
          doc.documentElement.scrollHeight,
          doc.body.offsetHeight,
          doc.documentElement.offsetHeight
        );

        iframe.style.height = `${height}px`;
      } catch (error) {
        // Ignore cross-document timing issues during initial load.
      }
    };

    iframe.addEventListener('load', () => {
      resizeIframe();
      window.setTimeout(resizeIframe, 0);
      window.setTimeout(resizeIframe, 250);
      window.setTimeout(resizeIframe, 1000);
    });

    window.addEventListener('resize', resizeIframe);
  })();
</script>



<script>
  (() => {
    const iframe = document.getElementById('ssm-recurrent-matrix-update');
    if (!iframe) return;

    const resizeIframe = () => {
      try {
        const doc = iframe.contentDocument || iframe.contentWindow?.document;
        if (!doc || !doc.body || !doc.documentElement) return;

        const height = Math.max(
          doc.body.scrollHeight,
          doc.documentElement.scrollHeight,
          doc.body.offsetHeight,
          doc.documentElement.offsetHeight
        );

        iframe.style.height = `${height}px`;
      } catch (error) {
        // Ignore cross-document timing issues during initial load.
      }
    };

    iframe.addEventListener('load', () => {
      resizeIframe();
      window.setTimeout(resizeIframe, 0);
      window.setTimeout(resizeIframe, 250);
      window.setTimeout(resizeIframe, 1000);
    });

    window.addEventListener('resize', resizeIframe);
  })();
</script>

<script>
  (() => {
    const iframe = document.getElementById('comparison-table');
    if (!iframe) return;

    const resizeIframe = () => {
      try {
        const doc = iframe.contentDocument || iframe.contentWindow?.document;
        if (!doc || !doc.body || !doc.documentElement) return;

        const height = Math.max(
          doc.body.scrollHeight,
          doc.documentElement.scrollHeight,
          doc.body.offsetHeight,
          doc.documentElement.offsetHeight
        );

        iframe.style.height = `${height}px`;
      } catch (error) {
        // Ignore cross-document timing issues during initial load.
      }
    };

    iframe.addEventListener('load', () => {
      resizeIframe();
      window.setTimeout(resizeIframe, 0);
      window.setTimeout(resizeIframe, 250);
      window.setTimeout(resizeIframe, 1000);
    });

    window.addEventListener('resize', resizeIframe);
  })();
</script>
