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
  grid-template-columns: 44% 56%;
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
    id:'la', name:'LA', full:'Linear Attention',
    paperTitle:'Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention',
    paperAuthors:'Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, François Fleuret',
    paper:'https://arxiv.org/pdf/2006.16236',
    attn:'linear', decays:['none'], accent:'#64748b',
    imgRef:'la-custom',
    desc:`<p><strong>LA</strong> was one of the first steps toward efficient sequence modelling, removing the quadratic cost of softmax attention. It simply drops the softmax in \\(\\operatorname{Softmax}(QK^\\top)V\\), giving \\((QK^\\top)V = Q(K^\\top V)\\). The product \\(K^\\top V \\in \\mathbb{R}^{d \\times d}\\) is independent of sequence length. To enhance expressivity, a feature map \\(\\phi(\\cdot) = \\operatorname{elu}(\\cdot)+1\\) replaces the identity kernel on queries and keys, yielding \\(\\phi(Q)\\,(\\phi(K)^\\top V)\\).</p><p>The above is the bidirectional case. For causal attention — where future tokens cannot influence past ones — the output at step \\(t\\) is \\(\\phi(q_t)\\sum_{i=1}^{t}\\phi(k_i)^\\top v_i\\), equivalent to an RNN with matrix-valued hidden state \\(S_t \\in \\mathbb{R}^{d \\times d}\\): $$S_t = S_{t-1} + v_t k_t^\\top, \\quad o_t = S_t q_t$$ With output normalisation, a vector \\(z_t \\in \\mathbb{R}^d\\) accumulates keys to normalise the readout: $$S_t = S_{t-1} + v_t k_t^\\top, \\quad z_t = z_{t-1} + k_t, \\quad o_t = \\frac{S_t q_t}{z_t^\\top q_t}$$</p>`,
    mathR:'S_t = S_{t-1} + v_t k_t^\\top',
    mathO:'o_t = S_t q_t',
  },
  {
    id:'retnet', name:'RetNet', full:'Retentive Network',
    paperTitle:'Retentive Network: A Successor to Transformer for Large Language Models',
    paperAuthors:'Yutao Sun, Li Dong, Shaohan Huang, Shuming Ma, Yuqing Xia, Jilong Xue, Jianyong Wang, Furu Wei',
    paper:'https://arxiv.org/pdf/2307.08621',
    attn:'linear', decays:['scalar'], accent:'#0369a1',
    imgRef:'retnet-custom',
    desc:`<p><strong>RetNet</strong> introduces <em>retention</em> — a recurrence with a fixed scalar decay γ ∈ (0,1) applied uniformly to the entire memory state at each step. Unlike input-dependent gating, γ is a fixed hyperparameter per head, giving each head a fixed memory horizon. The parallel form recovers a causal masked matrix multiplication, making training as efficient as standard attention.</p><p>RetNet supports three computation modes: recurrent (O(1) inference), parallel (GPU-friendly training), and chunkwise (balance between both). It was an early demonstration that a simple fixed decay could close much of the language-modelling gap between linear and softmax attention, paving the way for later input-dependent scalar gates like Mamba2.</p>`,
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
    desc:`<p><strong>GLA</strong> augments linear attention with a per-head <strong>diagonal decay matrix Λ<sub>t</sub></strong> derived from the input. Unlike a single scalar gate, the diagonal structure assigns each feature dimension its own independent forgetting rate, giving the model fine-grained control over what it retains across time steps.</p><p>The core computational block is <em>Gated Slot Attention (GSA)</em>, which rewrites attention as a low-rank key-value outer-product accumulation with per-slot gating. This lets different memory slots specialize — some holding persistent factual associations, others refreshing at a faster rate — while preserving the chunkwise-parallel training form needed for hardware efficiency.</p>`,
    mathR:'S_t = S_{t-1}\\operatorname{Diag}(\\boldsymbol{\\alpha}_t) + v_t k_t^\\top',
    mathO:'o_t = S_t q_t',
  },
  {
    id:'deltanet', name:'DeltaNet', full:'DeltaNetworks',
    paperTitle:'Parallelizing Linear Transformers with the Delta Rule over Sequence Length',
    paperAuthors:'Songlin Yang, Bailin Wang, Yu Zhang, Yikang Shen, Yoon Kim',
    paper:'https://arxiv.org/pdf/2406.06484',
    attn:'linear', decays:['delta'], accent:'#059669',
    imgRef:'deltanet-custom',
    desc:`<p><strong>DeltaNet</strong> replaces the outer-product write with the <strong>delta rule</strong>: at each step it first reads the current memory at key k<sub>t</sub>, then writes back only the prediction error v<sub>t</sub> − S<sub>t−1</sub>k<sub>t</sub>, scaled by a per-token learning rate β<sub>t</sub>. This self-correcting mechanism lets the model precisely overwrite stale associations rather than merely accumulating new ones on top.</p><p>Keys are L2-normalised to keep the memory state bounded, and short Conv1d layers provide local context mixing before the delta update. The result is a model that dramatically outperforms standard linear attention on associative-recall benchmarks while retaining O(1) inference cost per step.</p>`,
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
    desc:`<p><strong>GDN</strong> extends the delta rule with a <strong>scalar input-dependent forget gate α<sub>t</sub></strong>. Before the corrective write, the entire memory decays by α<sub>t</sub> ∈ (0,1), cleanly separating two concerns: how much of the past to retain and how precisely to overwrite specific associations. An additional output gate g<sub>t</sub> scales the normalised readout before projection.</p><p>This combination gives the model two complementary tools — rapid bulk forgetting when context shifts, and surgical delta-rule rewrites for fine-grained edits. Empirically, GDN matches or exceeds Mamba2 on language benchmarks while preserving the chunkwise-parallel training efficiency of the base DeltaNet.</p>`,
    mathR:'S_t = S_{t-1}\\bigl(\\alpha_t(\\mathbf{I} - \\beta_t k_t k_t^\\top)\\bigr) + \\beta_t v_t k_t^\\top',
    mathO:'o_t = S_t q_t',
  },
  {
    id:'kimi-linear', name:'Kimi Linear', full:'Kimi Delta Attention (KDA)',
    paperTitle:'MoonLight: Enhancing Linear Attention with Intra-Chunk Softmax',
    paperAuthors:'Moonshot AI, 2025',
    paper:'https://arxiv.org/pdf/2510.26692',
    attn:'linear', decays:['delta','diagonal'], accent:'#0891b2',
    imgRef:'kimi-linear-custom',
    desc:`<p><strong>Kimi Linear</strong> (KDA) combines the corrective delta-rule update of DeltaNet with a <strong>per-feature diagonal forget gate Λ<sub>t</sub></strong>, giving each memory dimension its own independent retention rate. This is the richest decay structure in the delta-rule family, allowing the model to forget slowly in some feature subspaces while aggressively refreshing others.</p><p>In the deployed Moonshot AI architecture, N KDA layers (each with linear attention and a MoE FFN) are capped by a single MLA softmax layer — achieving near-O(1) inference for 99% of the network while retaining full attention expressivity at the top. The diagonal gate manages long-horizon forgetting; the delta term handles precise short-term overwriting.</p>`,
    mathR:'S_t = S_{t-1}\\operatorname{Diag}(\\boldsymbol{\\lambda}_t)(\\mathbf{I} - \\beta_t k_t k_t^\\top) + \\beta_t v_t k_t^\\top',
    mathO:'o_t = S_t q_t',
  },
  {
    id:'mamba2', name:'Mamba2', full:'Mamba2',
    paperTitle:'Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality',
    paperAuthors:'Tri Dao, Albert Gu',
    paper:'https://arxiv.org/pdf/2405.21060',
    attn:'linear', decays:['scalar'], accent:'#0891b2',
    imgRef:'mamba2-custom',
    desc:`<p><strong>Mamba2</strong> establishes the <em>State Space Duality (SSD)</em> framework, proving that structured SSMs with scalar-times-identity decay are mathematically equivalent to a form of linear attention. The scalar gate a<sub>t</sub> multiplies the entire hidden state uniformly at each step, yielding a 1-semiseparable recurrence matrix that enables a highly efficient chunkwise parallel scan.</p><p>Within each chunk, the computation reduces to a dense masked matrix multiplication (tensor-core friendly); cross-chunk state is propagated sequentially. The Parallel Mamba Block pairs a Conv1d for local context mixing with a data-dependent SSM core (A, B, C) and an output gate, achieving 2–8× faster training than Mamba1 at the cost of per-feature decay expressiveness.</p>`,
    mathR:'S_t = \\gamma_t S_{t-1} + v_t k_t^\\top',
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
      <div class="az-eq-val">\\(${m.mathR}\\)</div>
    </div>
    <div class="az-eq-cell">
      <div class="az-eq-lbl">Readout</div>
      <div class="az-eq-val">\\(${m.mathO}\\)</div>
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
    else activeDecays.add(d);
  }
  document.querySelectorAll('#az-decay .az-pill').forEach(x => {
    const dd = x.dataset.decay;
    x.classList.toggle('active', dd === 'all' ? activeDecays.has('all') : activeDecays.has(dd));
  });
  updateGrid();
});

updateGrid();
</script>
