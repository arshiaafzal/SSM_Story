---
layout: distill
title: Memory Update Animations
description: SSM, SWA, and Raven memory update animations
tags:
giscus_comments: false
date: 2025-08-29
featured: false
thumbnail: assets/img/best.png

authors:
  - name: Arshia Afzal
    url:
    affiliations:
      name: EPFL

bibliography: 2018-12-22-distill.bib
---

<style>
.anim-row {
  display: flex;
  align-items: center;
  margin: 0;
  padding: 0;
  gap: 0;
}
.anim-label {
  writing-mode: vertical-rl;
  transform: rotate(180deg);
  font-size: 0.78rem;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: #475569;
  white-space: nowrap;
  padding: 0 6px;
  flex-shrink: 0;
}
.anim-row iframe {
  display: block;
  flex: 1;
  height: 0;
  border: 0;
  margin: 0;
  padding: 0;
  overflow: hidden;
}
</style>

<div style="margin:0; padding:0; display:flex; flex-direction:column; gap:0;">

  <iframe id="ssm-anim"   src="{{ '/assets/html/ssm_recurrent_matrix_update.html' | relative_url }}"   scrolling="no" style="display:block;width:100%;height:0;border:0;margin:0;padding:0;overflow:hidden;"></iframe>
  <iframe id="swa-anim"   src="{{ '/assets/html/swa_recurrent_matrix_update.html' | relative_url }}"   scrolling="no" style="display:block;width:100%;height:0;border:0;margin:0;padding:0;overflow:hidden;"></iframe>
  <iframe id="raven-anim" src="{{ '/assets/html/raven_recurrent_matrix_update.html' | relative_url }}" scrolling="no" style="display:block;width:100%;height:0;border:0;margin:0;padding:0;overflow:hidden;"></iframe>

</div>

<div style="margin:1.5rem 0 0; padding:0;">
  <iframe id="cmp-anim" src="{{ '/assets/html/comparison_matrix_update.html' | relative_url }}" scrolling="no" style="display:block;width:100%;height:0;border:0;margin:0;padding:0;overflow:hidden;"></iframe>
</div>

<script>
  // ── Auto-resize all iframes ──────────────────────────────────────
  ['ssm-anim','swa-anim','raven-anim','cmp-anim'].forEach(id => {
    const f = document.getElementById(id);
    if (!f) return;
    const resize = () => {
      try {
        const doc = f.contentDocument || f.contentWindow?.document;
        if (!doc || !doc.body) return;
        f.style.height = Math.max(doc.body.scrollHeight, doc.documentElement.scrollHeight) + 'px';
      } catch(e) {}
    };
    f.addEventListener('load', () => { resize(); setTimeout(resize,250); setTimeout(resize,1000); });
    window.addEventListener('resize', resize);
  });

  // ── Sync conductor for SSM / SWA / Raven ────────────────────────
  // Each iframe pauses at its move point waiting for 'sync-move'.
  // The parent sends 'sync-play' then fires 'sync-move' at exactly
  // MOVE_DELAY ms later — guaranteeing all three move simultaneously.
  const SYNC_IDS  = ['ssm-anim','swa-anim','raven-anim'];
  const SYNC_SRCS = new Set(['ssm','swa','raven']);
  const MOVE_DELAY = 1500; // ms after sync-play when move fires

  const readySet = new Set();
  const doneSet  = new Set();
  let syncStarted = false;

  function sendToAll(msg) {
    SYNC_IDS.forEach(id => {
      const f = document.getElementById(id);
      if (f && f.contentWindow) f.contentWindow.postMessage(msg, '*');
    });
  }

  function startCycle() {
    sendToAll({type: 'sync-play'});
    // Fire sync-move at the exact same moment for all three
    setTimeout(() => sendToAll({type: 'sync-move'}), MOVE_DELAY);
  }

  window.addEventListener('message', e => {
    if (!e.data || !SYNC_SRCS.has(e.data.src)) return;

    if (e.data.type === 'cycle-ready') {
      readySet.add(e.data.src);
      if (readySet.size === 3 && !syncStarted) {
        syncStarted = true;
        setTimeout(startCycle, 400);
      }
    }

    if (e.data.type === 'cycle-done') {
      doneSet.add(e.data.src);
      if (doneSet.size === 3) {
        doneSet.clear();
        setTimeout(startCycle, 500);
      }
    }
  });
</script>
