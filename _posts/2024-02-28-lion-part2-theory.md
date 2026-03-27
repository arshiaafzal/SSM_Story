---
layout: distill
title: Phoenix (Part-2)
description: Architecture and Results
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
  - name: From Framework to Model
  - name: Design Decisions
  - name: Empirical Results
  - name: A Pleasant Surprise

---

<div style="width: 50%; margin: 0 auto;">
  {% include figure.liquid loading="eager" path="assets/img/best.png" %}
</div>

------------------

## From Framework to Model

In Part 1, we arrived at Routing Slot Memories — a framework that could, in principle, combine the selective writing of SWA with the graceful forgetting of SSMs. Now came the harder question: how do you actually *build* this thing?

The RSM framework is general by design. To get Phoenix, we had to make concrete choices about how the router $r_t$ works and what kind of decay $a_t$ to use. We wanted something expressive enough to learn meaningful slot assignments, but simple enough to train stably at scale.

The recurrence we settled on uses two separate caches — one for keys, one for values — mirroring SWA's structure, but with a learned decay instead of hard deletion:

$$
S^k_t = (1-\exp(a_tr_t)) \odot S^k_{t-1} + \exp(a_tr_t)^\top k_t
$$

$$
S^v_t = (1-\exp(a_tr_t)) \odot S^v_{t-1} + \exp(a_tr_t)^\top v_t
$$

Unlike other SSMs <d-cite key="mamba,mamba2"></d-cite> that write to all memory slots at once ($r_t = \mathbf{1}$), Phoenix writes only to the slots selected by $r_t$. Unlike SWA, which fully erases the oldest slot, Phoenix gradually decays the selected slot's previous content — preserving partial information rather than discarding it.

For the router itself, we drew inspiration from the DeepSeek MoE family <d-cite key="deepseek_moe"></d-cite>. Each token produces a score vector via a learned linear projection, and only the top-$K$ slots are activated:

$$
m_t = \sigma(Wx_t), \qquad
g_t = \mathrm{KeepTop}_K(m_t) =
\begin{cases}
m_t[i], & \text{if } i \in \mathrm{TopK}(m_t),\\
0, & \text{otherwise},
\end{cases}, \qquad
r_t = \frac{g_t}{\alpha\sum_{i=1}^M g_t[i]}.
$$

The decay $a_t$ follows Mamba-2's <d-cite key="mamba2"></d-cite> scalar per-head design:

$$
a_t = -\mathrm{SoftPlus}(w^\top x_t)\,\exp(\Delta).
$$

And $\alpha$ normalizes the router logits — similar to the temperature scaling used in GLA <d-cite key="gla"></d-cite> — to keep the effective write magnitude stable.


## Design Decisions

With the recurrence in place, we turned to the block-level design. Several choices here surprised us.

{% include figure.liquid loading="eager" path="assets/img/arch.png" title="block diagram" caption="Phoenix block design vs. other linear models block design." %}

**Dropping the short convolution.** Most linear transformer blocks include a short depthwise convolution before the recurrence — a holdover that helps capture local context. We removed it. The reasoning: SWA is mathematically equivalent to an input-dependent convolution, and Phoenix's RSM already subsumes SWA as a special case. Adding a separate convolution on top would be redundant. Similar to what Mamba-3 does, we found that removing it came at no cost and slightly simplified the architecture. The RSM, it turns out, was already doing that job implicitly.

**On norms.** We use QK-RMSNorm on queries and keys to stabilize training — a practice that helps prevent gradient explosions in SSMs. One nuance: in the hybrid Phoenix variant (with interleaved attention layers), we found that removing these norms actually *improves* length generalization, particularly on recall tasks. Why? Our hypothesis is that the norms interfere with position-related information that the NoPE attention layers need. It was one of those unexpected ablation results that made us rethink a long-held assumption.

{% details Side Note on RoPE %}

In our early experiments, we included Rotary Position Embedding (RoPE) <d-cite key="rope"></d-cite> to incorporate positional information into queries and keys. However, it severely degraded the length generalization of Phoenix, one of its main strengths, so we removed it from the design.

{% enddetails %}

**The counterintuitive choice: no load balancing.** In Mixture-of-Experts models <d-cite key="moe"></d-cite>, load-balancing losses are standard practice — you push the router to distribute tokens evenly across experts to prevent a few experts from dominating. We tried this. It made Phoenix worse.

The reason, once we thought about it, made sense: Phoenix *wants* its memory to be uneven. A slot that specializes in storing retrieval-critical tokens — passkeys, variable names, rare facts — should store *more* of those tokens, not fewer. Forcing uniform allocation would destroy exactly the specialization that makes Phoenix useful. So we dropped the load-balancing loss entirely and instead added Gumbel noise during training to encourage exploration and prevent early collapse. The router was free to specialize, and specialize it did: retrieval-critical tokens naturally clustered into dedicated slots, leaving the rest of the memory free for ordinary context.


## Empirical Results

When we finally ran the experiments, we were cautiously optimistic. The theory was clean. But linear models had disappointed on recall before, and we didn't want to overclaim.

### Retrieval Ability

Our first question was simple: does Phoenix actually retrieve better? We pitted it against the strongest linear baselines we knew — *Mamba-2* <d-cite key="mamba2"></d-cite>, *GDN* <d-cite key="gated_deltanet"></d-cite>, and *GLA* <d-cite key="gla"></d-cite> on the SSM side, and SWA-like models including *SWA+RoPE* and *GSA* <d-cite key="gsa"></d-cite> on the other — and threw Needle-in-a-Haystack tasks at all of them.

The degradation in the baselines was predictable once you understood the mechanism: Mamba-2 and GDN write every token to every slot, so over a long sequence, each slot becomes a blurry average of everything it has ever seen. By 8K tokens — already $4\times$ their training length — the passkey signal is too diluted to recover.

{% include figure.liquid loading="eager" path="assets/img/recall.png" title="tab recall" caption="In-Context Recall Benchmarks and NIAH Accuracy vs. Context Length and Cache Size.
We report accuracy (%) on SWDE, FDA, and SQuAD, as well as on individual NIAH-1, NIAH-2, and NIAH-3 tasks across varying context lengths.  Rec. mem. and Conv. mem. denote the millions of cached state elements used during decoding." %}

Phoenix behaved differently. It held near-perfect accuracy ($\geq \mathbf{99\%}$) all the way to 16K tokens, and remained the *only model* at the 400M scale to keep strong performance ($> \mathbf{91\%}$) at 32K — that's $\mathbf{16\times}$ its training length. The passkey was going into a dedicated slot at the moment it was encountered, and staying there. No dilution. No overwriting.

{% include figure.liquid loading="eager" path="assets/img/niah.png" title="niah" caption="NIAH-1 accuracy visualization vs sequence length. Dashed line indicate training sequence length." %}

### Language Modeling

A natural worry at this point: did we buy recall ability at the cost of general language modeling? We had deliberately made the memory uneven — would that hurt perplexity or zero-shot benchmarks?

It didn't. Across standard evaluations, Phoenix matched or surpassed **Mamba-2** <d-cite key="mamba2"></d-cite>, **GLA** <d-cite key="gla"></d-cite>, and **GDN** <d-cite key="gated_deltanet"></d-cite>, as well as strong Transformer baselines like *FoX*, at both **400M and 800M parameter scales**. It even achieved the **best Lambada performance at 400M**. Selective memory, it turns out, doesn't hurt general language modeling — if anything, routing tokens to content-appropriate slots seems to help.

{% include figure.liquid loading="eager" path="assets/img/lm.png" title="niah" caption="Zero-shot language modeling performance across models." %}

### Hybrid Phoenix

We also experimented with a hybrid variant — interleaving Phoenix RSM layers with standard attention layers. The results here were striking. The hybrid Phoenix achieved near-perfect NIAH-1 accuracy up to **32K tokens** and strong NIAH-2 performance up to **16K tokens**, while other SSM hybrids like GDN+Attn <d-cite key="gated_deltanet"></d-cite> and Mamba-2+Attn <d-cite key="mamba2"></d-cite> started breaking down at 4K and 2K respectively. The combination of NoPE attention — good at precise short-range retrieval — with Phoenix's persistent, content-addressed slots turned out to be unusually complementary.

{% include figure.liquid loading="eager" path="assets/img/hybrid.png" title="hybrid" caption="Recall ability of hybrid-phoenix vs other hybrid architechtures." %}


## A Pleasant Surprise

We had set out to fix recall. We hadn't planned on fixing length generalization. But there it was.

During evaluation, we noticed Phoenix generalizing to sequences far beyond its training length — well beyond what any SSM had demonstrated before. We hadn't engineered this; it seemed to be a byproduct of the routing mechanism. So we went looking for an explanation, and that search led us to [Ricardo](https://www.linkedin.com/in/ricardobuitrago), who helped us develop the concept of **Effective Sequence Length (ESL)**.

{% include figure.liquid loading="eager" path="assets/img/esl_1.png" title="esl" caption="Normalized effective sequence length for a NIAH-1 sample at sequence length 16K. SWA stores each token in exactly one slot (FIFO). Phoenix shows the hidden state $S_t$ for layer 1, head 1 (256 slots, Top$_{32}$). SSM stores each token in all slots with decay. Slots are reordered by usage frequency; results correspond to 400M parameter models." %}

The insight is simple in retrospect. In a standard SSM, every slot sees every token — the effective sequence length is $T$ for all slots. In SWA, every slot sees exactly $\frac{T}{M}$ tokens — uniform and predictable. In Phoenix, the router creates *diversity*: some slots see many tokens (acting like SSM slots), while others see very few — just the rare tokens they've specialized for. This diversity means no single slot consistently overfits to the training sequence length, which appears to be why Phoenix generalizes so gracefully beyond it.

To make this concrete, we looked directly inside Phoenix's hidden state on a synthetic NIAH task. The picture was striking:

<figure style="text-align: center;">
  <img src="{{ '/assets/video/best.gif' | relative_url }}"
       alt="Phoenix Memory Dynamics — animated visualization of memory slot allocation across two heads"
       style="width: 80%; max-width: 700px;" />
  <figcaption style="margin-top: 0.6em; font-size: 0.85em; color: #555; line-height: 1.5;">
    <strong>Phoenix Memory Dynamics.</strong> Memory allocation for two different heads of Phoenix on a synthetic NIAH-style task.
    <span style="color:#c0392b;">■</span> Red slots store tokens important for retrieval (e.g., passwords),
    <span style="color:#27ae60;">■</span> green slots store non-retrieval tokens, and
    <span style="color:#2980b9;">■</span> blue slots are shared memory between the two types of tokens.
    Different heads allocate different amounts of slots to retrieval-important tokens, illustrating non-uniform memory allocation across heads.
  </figcaption>
</figure>

The passkey — shown in $\color{red}{red}$ — goes into a dedicated region of the hidden state the moment it appears, and stays there. Ordinary tokens shown in $\color{green}{green}$ flow through the general-purpose slots without touching it. Some slots shown in $\color{blue}{blue}$ serve both roles. Crucially, different heads allocate different amounts of memory to retrieval-critical tokens — the model discovered this specialization entirely on its own, without any explicit supervision.

This is what "organizing memory like a closet" looks like from the inside.

## Final Notes and Future

Phoenix started as an answer to a simple frustration — linear models that forgot too easily — and ended up revealing something deeper about how memory can be organized in learned sequence models. The routing idea is not complicated, but its consequences are: better recall, better language modeling, and a surprising bonus of length generalization, all from the same mechanism.

There is plenty left to explore. How far can the length generalization be pushed? Can the routing mechanism be made even more expressive? What happens at truly large scales? We don't have all the answers yet — but we're working on it.

----------------
