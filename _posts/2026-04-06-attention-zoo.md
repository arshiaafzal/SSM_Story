---
layout: distill
title: "The Attention Zoo: Linear & Softmax Models Unified"
description: An interactive guide to modern sequence models — explore architectures and recurrences across the linear-softmax landscape.
tags: attention linear-attention SSM transformer
giscus_comments: false

date: 2026-04-06
featured: true
thumbnail: assets/img/attention-zoo/gla-custom.png

authors:
  - name: Arshia Afzal
    url:
    affiliations:
      name: EPFL

toc:
  - name: Introduction
  - name: Interactive Explorer
---

<style>
/* ── Reset inside zoo ──────────────────────────────── */
#az-root * { box-sizing: border-box; }

/* ── Root palette ──────────────────────────────────── */
#az-root {
  --c-bg:      #f8fafc;
  --c-card:    #ffffff;
  --c-border:  #e2e8f0;
  --c-text:    #0f172a;
  --c-muted:   #64748b;
  --c-subtle:  #f1f5f9;
  --c-linear:  #6366f1;
  --c-softmax: #d97706;
  --c-delta:   #059669;
  --c-diag:    #7c3aed;
  --c-scalar:  #dc2626;
  --c-none:    #475569;
  --radius:    12px;
  --trans:     0.25s cubic-bezier(.4,0,.2,1);
  font-family: 'Inter','Segoe UI',system-ui,sans-serif;
  background: var(--c-bg);
  border-radius: 20px;
  padding: 2rem 1.75rem;
  margin: 2rem -1rem;
  color: var(--c-text);
  box-shadow: 0 4px 24px rgba(0,0,0,.07), 0 1px 4px rgba(0,0,0,.04);
}

/* ── Section label ─────────────────────────────────── */
.az-label {
  font-size: .7rem; font-weight: 700; letter-spacing: .12em;
  text-transform: uppercase; color: var(--c-muted);
  display: flex; align-items: center; gap: .5rem; margin-bottom: .65rem;
}
.az-label::after { content:''; flex:1; height:1px; background:var(--c-border); }

/* ── Pills ─────────────────────────────────────────── */
.az-pills { display:flex; flex-wrap:wrap; gap:.45rem; margin-bottom:1.1rem; }
.az-pill {
  padding:.38rem 1rem; border-radius:100px;
  border:1.5px solid var(--c-border);
  background:#fff; color:var(--c-muted);
  font-size:.82rem; font-weight:600;
  cursor:pointer; transition:all var(--trans);
  user-select:none;
}
.az-pill:hover { border-color:#94a3b8; color:var(--c-text); }
.az-pill.active {
  background:var(--pill-c); border-color:var(--pill-c);
  color:#fff;
  box-shadow: 0 0 0 3px color-mix(in srgb, var(--pill-c) 25%, transparent);
}
[data-type="linear"]   { --pill-c: var(--c-linear); }
[data-type="softmax"]  { --pill-c: var(--c-softmax); }
[data-decay="delta"]   { --pill-c: var(--c-delta); }
[data-decay="diagonal"]{ --pill-c: var(--c-diag); }
[data-decay="scalar"]  { --pill-c: var(--c-scalar); }
[data-decay="none"]    { --pill-c: var(--c-none); }
[data-type="all"],[data-decay="all"]{ --pill-c: #334155; }

/* ── Divider ───────────────────────────────────────── */
.az-hr { height:1px; background:var(--c-border); margin:1.25rem 0; }

/* ── Grid ──────────────────────────────────────────── */
#az-grid { display:flex; flex-direction:column; gap:1.75rem; }

/* ── Card ──────────────────────────────────────────── */
.az-card {
  background: var(--c-card);
  border: 1px solid var(--c-border);
  border-radius: var(--radius);
  overflow: hidden;
  box-shadow: 0 2px 12px rgba(0,0,0,.05);
  animation: az-in .4s cubic-bezier(.34,1.56,.64,1) both;
  border-top: 3px solid var(--card-accent, #6366f1);
}
@keyframes az-in {
  from { opacity:0; transform:translateY(16px) scale(.98); }
  to   { opacity:1; transform:none; }
}

/* Card header */
.az-card-header {
  padding: 1.1rem 1.5rem .9rem;
  display: flex; align-items: flex-start;
  justify-content: space-between; gap:1rem;
  border-bottom: 1px solid var(--c-border);
  background: var(--c-subtle);
}
.az-card-name { font-size:1.3rem; font-weight:800; letter-spacing:-.02em; color:#0f172a; }
.az-card-full { font-size:.8rem; color:var(--c-muted); margin:.1rem 0 0; }
.az-paper-meta { margin:.25rem 0 .45rem; line-height:1.45; }
.az-paper-title { display:block; font-size:.71rem; font-style:italic; color:#475569; }
.az-paper-authors { display:block; font-size:.68rem; font-weight:600; color:var(--c-muted); margin-top:.1rem; }
.az-paper-btn {
  display:inline-flex; align-items:center; gap:.3rem;
  font-size:.72rem; font-weight:600;
  color:var(--c-muted); text-decoration:none;
  padding:.22rem .65rem; border:1px solid var(--c-border);
  border-radius:6px; background:#fff;
  transition:all .2s; flex-shrink:0;
}
.az-paper-btn:hover { color:#0f172a; border-color:#94a3b8; text-decoration:none; }
.az-card-badges { display:flex; flex-direction:column; align-items:flex-end; gap:.3rem; flex-shrink:0; }
.az-badge {
  font-size:.67rem; font-weight:700; padding:.18rem .58rem;
  border-radius:100px; letter-spacing:.05em; text-transform:uppercase;
}
.b-linear   { background:#eef2ff; color:#4338ca; border:1px solid #c7d2fe; }
.b-softmax  { background:#fffbeb; color:#b45309; border:1px solid #fde68a; }
.b-delta    { background:#ecfdf5; color:#065f46; border:1px solid #a7f3d0; }
.b-diagonal { background:#f5f3ff; color:#5b21b6; border:1px solid #ddd6fe; }
.b-scalar   { background:#fef2f2; color:#991b1b; border:1px solid #fecaca; }
.b-none     { background:#f8fafc; color:#334155; border:1px solid #cbd5e1; }

/* Card body layout */
.az-card-body {
  display: grid;
  grid-template-columns: 60% 40%;
  min-height: 280px;
}
@media(max-width:680px){ .az-card-body{ grid-template-columns:1fr; } }

.az-desc {
  padding: 1.2rem 1.4rem;
  border-right: 1px solid var(--c-border);
  font-size: .8rem; line-height: 1.75; color: #334155;
}
.az-desc p { margin: 0 0 .75rem; }
.az-desc p:last-child { margin: 0; }
.az-desc strong { color: #0f172a; }

.az-arch {
  padding: .75rem 1rem;
  display: flex; align-items: center; justify-content: center;
  background: #fff;
}
.az-arch-img {
  width: 100%; max-width: 400px;
  display: block; height: auto;
  border-radius: 8px;
  box-shadow: 0 1px 8px rgba(0,0,0,.07);
}

/* Equation row */
.az-eq-row {
  display: flex; flex-wrap: wrap;
  border-top: 1px solid var(--c-border);
  background: var(--c-subtle);
  border-radius: 0 0 var(--radius) var(--radius);
}
.az-eq-cell {
  flex: 1; min-width: 260px;
  padding: .7rem 1.3rem;
  border-right: 1px solid var(--c-border);
}
.az-eq-cell:last-child { border-right: none; }
.az-eq-lbl {
  font-size: .62rem; font-weight: 700; letter-spacing: .1em;
  text-transform: uppercase; color: var(--c-muted);
  margin-bottom: .25rem;
  display: flex; align-items: center; gap: .3rem;
}
.az-eq-lbl::before {
  content: ''; display: inline-block;
  width: 5px; height: 5px; border-radius: 50%;
  background: var(--card-accent, #6366f1); flex-shrink: 0;
}
.az-eq-val { font-size: .9rem; overflow-x: auto; }
.az-eq-extra { margin-top: .45rem; display: flex; align-items: baseline; justify-content: space-between; gap: .6rem; }
.az-opt-tag { font-size: .58rem; font-weight: 700; letter-spacing: .07em; text-transform: uppercase;
  color: #94a3b8; border: 1px solid #cbd5e1; border-radius: 3px; padding: 1px 5px; white-space: nowrap; }

.az-aside { margin-top: .9rem; padding: .6rem .9rem; border-left: 3px solid #cbd5e1;
  background: #f8fafc; border-radius: 0 4px 4px 0; font-size: .75rem; color: #64748b; line-height: 1.6; }
.az-aside-lbl { display: inline-block; font-weight: 700; font-size: .62rem; letter-spacing: .08em;
  text-transform: uppercase; color: #94a3b8; margin-bottom: .25rem; }

/* Empty state */
#az-empty {
  text-align:center; padding:3rem 1rem;
  color:var(--c-muted); display:none;
}
</style>

---

## Introduction

Modern sequence models share a deep mathematical skeleton: a **key-value memory** written to at each step and read by a query. The differences lie in *how* that memory decays — and this page makes those differences interactive and visual.

Filter by **attention kernel** and **memory decay type** below. Each model card shows the architecture diagram for that model.

---

## Interactive Explorer

<div id="az-root">

  <div class="az-label">Attention Type</div>
  <div class="az-pills" id="az-attn">
    <button class="az-pill active" data-type="all">All</button>
    <button class="az-pill" data-type="linear">Linear</button>
    <button class="az-pill" data-type="softmax">Softmax</button>
  </div>

  <div class="az-label">Decay Type <span style="font-weight:400;font-size:.65rem;letter-spacing:0;text-transform:none;margin-left:.25rem">(exact match — selects unique models)</span></div>
  <div class="az-pills" id="az-decay">
    <button class="az-pill active" data-decay="all">All</button>
    <button class="az-pill" data-decay="none">None</button>
    <button class="az-pill" data-decay="delta">Delta-Rule</button>
    <button class="az-pill" data-decay="diagonal">Diagonal</button>
    <button class="az-pill" data-decay="scalar">Scalar</button>
  </div>

  <div class="az-hr"></div>

  <div id="az-grid"></div>
  <div id="az-empty">
    <div style="font-size:2rem;margin-bottom:.5rem">🔬</div>
    <p>No models match this combination. Try a different filter.</p>
  </div>

</div>

<script>
const MODELS = [
  {
    id:'la', name:'LinAtt', full:'Linear Attention',
    paperTitle:'Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention',
    paperAuthors:'Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, François Fleuret',
    paper:'https://arxiv.org/pdf/2006.16236',
    attn:'linear', decays:['none'], accent:'#64748b',
    imgRef:'la-custom',
    desc:`<p><strong>LinAtt</strong> makes attention cheaper by removing the softmax. Standard attention computes \\(\\operatorname{Softmax}(QK^\\top)V\\); dropping the softmax gives \\((QK^\\top)V\\), which can be reordered to \\(Q(K^\\top V)\\). The key insight is that \\(K^\\top V\\) is a small \\(d \\times d\\) matrix — you compute it once, then multiply each query into it, so cost no longer grows with sequence length. A simple nonlinearity \\(\\phi(x) = \\operatorname{elu}(x)+1\\) is applied to queries and keys to keep all values positive.</p><p>For left-to-right (causal) generation, this is equivalent to running an RNN: at each step you add the new key–value pair to a memory matrix \\(S_t\\), then read the answer out with the query: $$S_t = S_{t-1} + v_t k_t^\\top, \\quad o_t = S_t q_t$$ To stop outputs from growing too large, a running sum \\(z_t\\) of past keys is kept and used to divide the output: $$S_t = S_{t-1} + v_t k_t^\\top, \\quad z_t = z_{t-1} + k_t, \\quad o_t = \\frac{S_t q_t}{z_t^\\top q_t}$$</p><p>The feature map \\(\\phi\\) was adopted by many models that followed. From here on we absorb it into the definitions of \\(q\\) and \\(k\\) to keep the notation clean.</p>`,
    mathR:'S_t = S_{t-1} + v_t k_t^\\top',
    mathRExtra:'z_t = z_{t-1} + k_t',
    mathO:'o_t = S_t q_t',
    mathOExtra:'o_t = S_t q_t \\,/\\, (z_t^\\top q_t)',
  },
  {
    id:'retnet', name:'RetNet', full:'Retentive Network',
    paperTitle:'Retentive Network: A Successor to Transformer for Large Language Models',
    paperAuthors:'Yutao Sun, Li Dong, Shaohan Huang, Shuming Ma, Yuqing Xia, Jilong Xue, Jianyong Wang, Furu Wei',
    paper:'https://arxiv.org/pdf/2307.08621',
    attn:'linear', decays:['scalar'], accent:'#0369a1',
    imgRef:'retnet-custom',
    desc:`<p><strong>RetNet</strong> was introduced shortly after LinAtt to fix one of its core problems: because LinAtt never forgets, the hidden state gets overwhelmed with accumulated context over time. RetNet's fix is simple — multiply the previous state by a fixed scalar \\(\\gamma \\in (0,1)\\) at every step, so older information fades out: $$S_t = \\gamma S_{t-1} + v_t k_t^\\top, \\quad o_t = S_t q_t$$ Keeping \\(\\gamma < 1\\) also prevents the state from exploding during training.</p><p>For parallel training, the decay is folded into a mask matrix \\(M \\in \\mathbb{R}^{T \\times T}\\) applied element-wise to the attention scores: $$O = (QK^\\top \\odot M)V, \\qquad M_{ij} = \\begin{cases} \\gamma^{i-j} & i \\geq j \\\\ 0 & i < j \\end{cases}$$ This recovers the efficiency of standard attention during training while keeping exact recurrent inference at test time.</p>`,
    mathR:'S_t = \\gamma S_{t-1} + v_t k_t^\\top',
    mathO:'o_t = S_t q_t',
  },
  {
    id:'gla', name:'GLA', full:'Gated Linear Attention',
    paperTitle:'Gated Linear Attention Transformers with Hardware-Efficient Training',
    paperAuthors:'Songlin Yang, Bailin Wang, Yikang Shen, Rameswar Panda, Yoon Kim',
    paper:'https://arxiv.org/pdf/2312.06635',
    attn:'linear', decays:['diagonal'], accent:'#7c3aed',
    imgRef:'gla-custom',
    desc:`<p><strong>GLA</strong> addresses a key bottleneck of linear models that rely on <strong>parallel scan</strong> for training: scan-based methods are slow compared to the matrix-multiply-based parallel form of softmax attention. GLA's solution is <strong>chunk-wise training</strong> — unrolling the recurrence over fixed-length chunks and applying a fast parallel form within each chunk, in the same spirit as <a href="https://arxiv.org/pdf/2205.14135" target="_blank">FlashAttention</a>. This is implemented in <a href="https://github.com/sustcsonglin/flash-linear-attention" target="_blank">flash-linear-attention</a>, a hardware-efficient training library for linear transformers.</p><p>The decay is <strong>diagonal and input-dependent</strong>, defined as \\(\\alpha_t = \\operatorname{sigmoid}(w x_t^\\top)^\\tau\\), where \\(\\tau\\) is a temperature parameter controlling how smooth the decay is — a smoother decay helps the model cover a wider context range. The recurrence is: $$S_t = S_{t-1}\\operatorname{Diag}(\\alpha_t) + v_t k_t^\\top, \\quad o_t = S_t q_t$$ Since \\(\\alpha_t < 1\\) for numerical stability, the parallel form is always applied chunk-by-chunk. GLA significantly closes the training-speed gap with Mamba, a goal also addressed concurrently by Mamba-2.</p>`,
    mathR:'S_t = S_{t-1}\\operatorname{Diag}(\\boldsymbol{\\alpha}_t) + v_t k_t^\\top',
    mathO:'o_t = S_t q_t',
  },
  {
    id:'mamba', name:'Mamba', full:'Mamba',
    paperTitle:'Mamba: Linear-Time Sequence Modeling with Selective State Spaces',
    paperAuthors:'Albert Gu*, Tri Dao* &nbsp;(*equal contribution)',
    paper:'https://arxiv.org/pdf/2312.00752',
    attn:'linear', decays:['diagonal'], accent:'#0f766e',
    imgRef:'mamba-custom',
    desc:`<p><strong>Mamba</strong> belongs to the family of <strong>state space models (SSMs)</strong>, following prior work such as <a href="https://arxiv.org/abs/2111.00396" target="_blank">S4</a> that model sequences as discretized linear dynamical systems — hence the name. It was the first such model to succeed at language modelling tasks, and its release brought wide attention back to recurrent architectures. The key step over RetNet is replacing the fixed scalar decay with an <strong>input-dependent diagonal decay</strong>: at each step the model decides how much of the past to keep based on the current input. The full recurrence is: $$\\begin{aligned} S_t &= S_{t-1} \\odot \\exp\\bigl(-(\\alpha_t \\mathbf{1}^\\top) \\odot \\exp(A)\\bigr) + (\\alpha_t \\odot v_t) k_t^\\top \\\\ o_t &= S_t q_t + d \\odot v_t \\end{aligned}$$ Here \\(A\\) and \\(d\\) are learnable parameters; \\(A\\) is diagonal so the decay acts independently on each feature dimension. The specific form of the decay — \\(\\exp(-\\alpha_t \\odot \\exp(A))\\) — comes from <a href="https://en.wikipedia.org/wiki/Zero-order_hold" target="_blank">zero-order-hold discretization</a> of a continuous-time SSM.</p><p>Mamba is trained using <a href="https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda" target="_blank">parallel scan</a> rather than the matrix-multiply-based parallel form of softmax attention, which is slower on current hardware. Despite this, it achieves comparable accuracy to transformers while being much faster at inference time.</p><div class="az-aside"><div class="az-aside-lbl">Historical notation</div>SSMs like S4 and Mamba write the recurrence using the classical control-theory variables \\(A_t, B_t, C_t\\): $$h_t = A_t h_{t-1} + B_t x_t, \\quad y_t = C_t h_t + D x_t$$ where \\(h_t\\) is the hidden state, \\(x_t\\) is the input token, and \\(D\\) is a skip connection. This is exactly the same update written in different letters — the correspondence to the attention view used throughout this page is: \\(h_t \\leftrightarrow S_t\\) (hidden state), \\(B_t x_t \\leftrightarrow (\\alpha_t \\odot v_t)k_t^\\top\\) (write / value×key), \\(C_t \\leftrightarrow q_t\\) (read / query), \\(D \\leftrightarrow d\\) (skip). The \\(q, k, v\\) view makes the connection to attention explicit and is used by most recent work.</div>`,
    mathDisplay: true,
    mathR:'\\begin{aligned} S_t &= S_{t-1} \\odot \\exp\\bigl(-(\\alpha_t \\mathbf{1}^\\top) \\odot \\exp(A)\\bigr) \\\\ &\\quad + (\\alpha_t \\odot v_t) k_t^\\top \\end{aligned}',
    mathO:'o_t = S_t q_t + d \\odot v_t',
  },
  {
    id:'deltanet', name:'DeltaNet', full:'DeltaNetworks',
    paperTitle:'Parallelizing Linear Transformers with the Delta Rule over Sequence Length',
    paperAuthors:'Songlin Yang, Bailin Wang, Yu Zhang, Yikang Shen, Yoon Kim',
    paper:'https://arxiv.org/pdf/2406.06484',
    attn:'linear', decays:['delta'], accent:'#059669',
    imgRef:'deltanet-custom',
    desc:`<p><strong>DeltaNet</strong> replaces the simple outer-product write of linear attention with the <strong>delta rule</strong>, <a href="https://arxiv.org/abs/2102.11174" target="_blank">first introduced</a> as a biologically-inspired weight update for online learning. At each step, the model reads its current estimate for key \\(k_t\\) from memory — \\(\\hat{v}_t = S_{t-1} k_t\\) — then writes back only the <strong>prediction error</strong> \\(\\beta_t(v_t - \\hat{v}_t)\\), scaled by a per-token learning rate \\(\\beta_t\\): $$\\begin{aligned} S_t &= S_{t-1} + \\beta_t(v_t - S_{t-1}k_t)k_t^\\top \\\\ &= S_{t-1}(\\mathbf{I} - \\beta_t k_t k_t^\\top) + \\beta_t v_t k_t^\\top \\end{aligned}$$ This self-correcting mechanism lets the model <strong>precisely overwrite stale associations</strong> rather than merely accumulating new ones on top. DeltaNet makes this update <strong>efficient and fast to train</strong> by deriving a hardware-friendly chunkwise-parallel form.</p><p>A crucial property of the delta rule update is that the transition matrix \\(\\mathbf{I} - \\beta_t k_t k_t^\\top\\) is <strong>non-diagonal</strong> — it couples feature dimensions together. This is fundamentally different from diagonal SSMs like Mamba or GLA, and it matters: <strong>diagonal state transitions cannot solve state-tracking problems</strong> (such as permutation composition or string transduction), whereas DeltaNet's non-diagonal decay can. Keys are L2-normalised to keep the memory state bounded.</p>`,
    mathR:'S_t = S_{t-1}(\\mathbf{I} - \\beta_t k_t k_t^\\top) + \\beta_t v_t k_t^\\top',
    mathO:'o_t = S_t q_t',
  },
  {
    id:'gated-deltanet', name:'GDN', full:'Gated DeltaNetworks',
    paperTitle:'Gated Delta Networks: Improving Mamba2 with Delta Rule',
    paperAuthors:'Songlin Yang, Jan Kautz, Ali Hatamizadeh',
    paper:'https://arxiv.org/pdf/2412.06464',
    attn:'linear', decays:['delta','scalar'], accent:'#d97706',
    imgRef:'gated-deltanet-custom',
    desc:`<p><strong>Gated DeltaNet (GDN)</strong> extends DeltaNet by adding a <strong>scalar input-dependent forget gate \\(\\alpha_t \\in (0,1)\\)</strong>. Before the corrective delta-rule write, the entire memory is scaled down by \\(\\alpha_t\\), cleanly separating two concerns: how much of the past to retain and how precisely to overwrite specific associations. The recurrence is: $$\\begin{aligned} S_t &= \\alpha_t S_{t-1}(\\mathbf{I} - \\beta_t k_t k_t^\\top) + \\beta_t v_t k_t^\\top \\end{aligned}$$ This gives the model two complementary tools — <strong>rapid bulk forgetting</strong> when context shifts, and <strong>surgical delta-rule rewrites</strong> for fine-grained edits.</p><p>GDN specifically outperforms both <strong>Mamba-2 and DeltaNet on recall-intensive tasks</strong> such as associative recall and multi-query associative recall, where the ability to both forget stale context and precisely overwrite specific memory slots matters most. It preserves the chunkwise-parallel training efficiency of the base DeltaNet.</p>`,
    mathR:'S_t = \\alpha_t S_{t-1}(\\mathbf{I} - \\beta_t k_t k_t^\\top) + \\beta_t v_t k_t^\\top',
    mathO:'o_t = S_t q_t',
  },
  {
    id:'kimi-linear', name:'Kimi Linear', full:'Kimi Delta Attention (KDA)',
    paperTitle:'Kimi Linear: An Expressive, Efficient Attention Architecture',
    paperAuthors:'Kimi Team',
    paper:'https://arxiv.org/pdf/2510.26692',
    attn:'linear', decays:['delta','diagonal'], accent:'#0891b2',
    imgRef:'kimi-linear-custom',
    desc:`<p><strong>Kimi Linear</strong> (KDA) combines the corrective delta-rule update of DeltaNet with a <strong>per-feature diagonal forget gate Λ<sub>t</sub></strong>, giving each memory dimension its own independent retention rate. This is the richest decay structure in the delta-rule family, allowing the model to forget slowly in some feature subspaces while aggressively refreshing others.</p><p>In the deployed Moonshot AI architecture, N KDA layers (each with linear attention and a MoE FFN) are capped by a single MLA softmax layer — achieving near-O(1) inference for 99% of the network while retaining full attention expressivity at the top. The diagonal gate manages long-horizon forgetting; the delta term handles precise short-term overwriting.</p>`,
    mathR:'S_t = S_{t-1}\\operatorname{Diag}(\\boldsymbol{\\alpha}_t)(\\mathbf{I} - \\beta_t k_t k_t^\\top) + \\beta_t v_t k_t^\\top',
    mathO:'o_t = S_t q_t',
  },
  {
    id:'mamba2', name:'Mamba2', full:'Mamba2',
    paperTitle:'Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality',
    paperAuthors:'Tri Dao*, Albert Gu* &nbsp;(*equal contribution)',
    paper:'https://arxiv.org/pdf/2405.21060',
    attn:'linear', decays:['scalar'], accent:'#0891b2',
    imgRef:'mamba2-custom',
    desc:`<p><strong>Mamba2</strong> establishes the <em>State Space Duality (SSD)</em> framework, proving that structured SSMs with scalar-times-identity decay are mathematically equivalent to a form of linear attention. The scalar gate a<sub>t</sub> multiplies the entire hidden state uniformly at each step, yielding a 1-semiseparable recurrence matrix that enables a highly efficient chunkwise parallel scan.</p><p>Within each chunk, the computation reduces to a dense masked matrix multiplication (tensor-core friendly); cross-chunk state is propagated sequentially. The Parallel Mamba Block pairs a Conv1d for local context mixing with a data-dependent SSM core (A, B, C) and an output gate, achieving 2–8× faster training than Mamba1 at the cost of per-feature decay expressiveness.</p>`,
    mathR:'S_t = \\alpha_t S_{t-1} + v_t k_t^\\top',
    mathO:'o_t = S_t q_t',
  },
  {
    id:'fox', name:'FoX', full:'Forgetting Transformer',
    paperTitle:'Forgetting Transformer: Softmax Attention with a Forget Gate',
    paperAuthors:'Zhixuan Lin, Evgenii Nikishin, Xu Owen He, Aaron Courville',
    paper:'https://arxiv.org/pdf/2503.02130',
    attn:'softmax', decays:['scalar'], accent:'#d97706',
    imgRef:'fox-custom',
    desc:`<p><strong>FoX</strong> augments causal softmax attention with a <strong>scalar input-dependent forget gate f<sub>t</sub></strong> injected as a log-additive bias directly inside the softmax. Each position produces f<sub>t</sub> ∈ (0,1); its cumulative log-product over preceding positions forms a smooth causal decay mask that down-weights distant keys without discarding them entirely.</p><p>Because the gate lives inside the softmax normalisation, FoX preserves full softmax expressiveness — arbitrary key selection and proper probability renormalisation — while dynamically compressing effective context on long sequences. ShiftLinear projections for keys and values prevent look-ahead leakage, and RMSNorm on queries and keys stabilises training across scales.</p>`,
    mathR:'A_{ij} = \\tfrac{q_i^\\top k_j}{\\sqrt{d}} + \\sum_{l=j}^{i-1} \\log f_l',
    mathO:'O = \\operatorname{softmax}(A + M)\\,V',
  },
  {
    id:'nope', name:'NoPE', full:'NoPE Attention',
    paperTitle:'The Impact of Positional Encoding on Length Generalization in Transformers',
    paperAuthors:'Amirhossein Kazemnejad, Inkit Padhi, Karthikeyan Natesan Ramamurthy, Payel Das, Siva Reddy',
    paper:'https://arxiv.org/pdf/2305.19466',
    attn:'softmax', decays:['none'], accent:'#475569',
    imgRef:'nope-custom',
    desc:`<p><strong>NoPE</strong> (No Positional Encoding) studies what happens when a causal Transformer is trained with <em>no</em> positional encoding — no RoPE, ALiBi, or sinusoidal biases. Kazemnejad et al. find that such models generalise surprisingly well to lengths beyond training, challenging the assumption that explicit position signals are necessary.</p><p>As the baseline softmax model in this zoo, NoPE represents pure causal attention: full softmax expressiveness, no memory decay, and no positional prior. Its competitive length-generalisation behaviour motivates asking which structural inductive biases attention truly needs.</p>`,
    mathR:'A_{ij} = \\tfrac{q_i^\\top k_j}{\\sqrt{d}}',
    mathO:'O = \\operatorname{softmax}(A + M)\\,V',
  },
];

let activeAttn = 'all';
let activeDecays = new Set(['all']);

function exactMatch(m) {
  if (activeAttn !== 'all' && m.attn !== activeAttn) return false;
  if (activeDecays.has('all')) return true;
  if (m.decays.length !== activeDecays.size) return false;
  for (const d of m.decays) if (!activeDecays.has(d)) return false;
  return true;
}

function renderCard(m) {
  const attnBadge   = `<span class="az-badge b-${m.attn}">${m.attn}</span>`;
  const decayBadges = m.decays.map(d=>`<span class="az-badge b-${d}">${d}</span>`).join('');
  const eqLbl = m.attn === 'linear' ? 'Recurrence' : 'Attention logit';

  return `<div class="az-card" style="--card-accent:${m.accent}">
  <div class="az-card-header">
    <div>
      <div class="az-card-name">${m.name}</div>
      <div class="az-card-full">${m.full}</div>
      <div class="az-paper-meta">
        <span class="az-paper-title">${m.paperTitle}</span>
        <span class="az-paper-authors">${m.paperAuthors}</span>
      </div>
      <a class="az-paper-btn" href="${m.paper}" target="_blank">
        <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
          <path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/>
          <polyline points="14 2 14 8 20 8"/>
        </svg>Paper
      </a>
    </div>
    <div class="az-card-badges">${attnBadge}${decayBadges}</div>
  </div>
  <div class="az-card-body">
    <div class="az-desc">${m.desc}</div>
    <div class="az-arch">
      <img class="az-arch-img"
        src="/SSM_Story/assets/img/attention-zoo/${m.imgRef}.png"
        alt="${m.name} architecture" loading="lazy">
    </div>
  </div>
  <div class="az-eq-row">
    <div class="az-eq-cell">
      <div class="az-eq-lbl">${eqLbl}</div>
      <div class="az-eq-val">${m.mathDisplay ? `$$${m.mathR}$$` : `\\(${m.mathR}\\)`}</div>
      ${m.mathRExtra ? `<div class="az-eq-extra"><span class="az-eq-val">\\(${m.mathRExtra}\\)</span><span class="az-opt-tag">optional</span></div>` : ''}
    </div>
    <div class="az-eq-cell">
      <div class="az-eq-lbl">Readout</div>
      <div class="az-eq-val">${m.mathDisplay ? `$$${m.mathO}$$` : `\\(${m.mathO}\\)`}</div>
      ${m.mathOExtra ? `<div class="az-eq-extra"><span class="az-eq-val">\\(${m.mathOExtra}\\)</span><span class="az-opt-tag">optional</span></div>` : ''}
    </div>
  </div>
</div>`;
}

function updateGrid() {
  const grid  = document.getElementById('az-grid');
  const empty = document.getElementById('az-empty');
  const visible = MODELS.filter(exactMatch);
  grid.innerHTML = visible.map(renderCard).join('');
  grid.style.display = visible.length ? 'flex' : 'none';
  empty.style.display = visible.length ? 'none' : 'block';
  if (window.MathJax) MathJax.typesetPromise([grid]).catch(()=>{});
}

document.getElementById('az-attn').addEventListener('click', e => {
  const p = e.target.closest('.az-pill'); if (!p) return;
  activeAttn = p.dataset.type;
  document.querySelectorAll('#az-attn .az-pill').forEach(x => x.classList.toggle('active', x === p));
  updateGrid();
});

document.getElementById('az-decay').addEventListener('click', e => {
  const p = e.target.closest('.az-pill'); if (!p) return;
  const d = p.dataset.decay;
  if (d === 'all')  { activeDecays = new Set(['all']); }
  else if (d === 'none') { activeDecays = new Set(['none']); }
  else {
    activeDecays.delete('all'); activeDecays.delete('none');
    if (activeDecays.has(d)) { activeDecays.delete(d); if (!activeDecays.size) activeDecays.add('all'); }
    else {
      if (d === 'diagonal') activeDecays.delete('scalar');
      if (d === 'scalar') activeDecays.delete('diagonal');
      activeDecays.add(d);
    }
  }
  document.querySelectorAll('#az-decay .az-pill').forEach(x => {
    const dd = x.dataset.decay;
    x.classList.toggle('active', dd === 'all' ? activeDecays.has('all') : activeDecays.has(dd));
  });
  updateGrid();
});

updateGrid();
</script>
