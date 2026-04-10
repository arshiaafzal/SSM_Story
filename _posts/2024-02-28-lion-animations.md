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

### SSM — Dense Update

<div style="margin: 1.5rem auto 1rem; display: flex; justify-content: center; width: 100%;">
  <iframe
    id="ssm-anim"
    src="{{ '/assets/html/ssm_recurrent_matrix_update.html' | relative_url }}"
    title="SSM recurrent matrix update"
    loading="lazy"
    scrolling="no"
    style="display: block; width: 70%; max-width: 1400px; height: 0; border: 0; background: transparent; overflow: hidden;"
  ></iframe>
</div>

<script>
  (() => {
    const iframe = document.getElementById('ssm-anim');
    if (!iframe) return;
    const resize = () => {
      try {
        const doc = iframe.contentDocument || iframe.contentWindow?.document;
        if (!doc || !doc.body) return;
        const h = Math.max(doc.body.scrollHeight, doc.documentElement.scrollHeight);
        iframe.style.height = `${h}px`;
      } catch(e) {}
    };
    iframe.addEventListener('load', () => { resize(); setTimeout(resize,250); setTimeout(resize,1000); });
    window.addEventListener('resize', resize);
  })();
</script>

---

### SWA — Sliding Window

<div style="margin: 1.5rem auto 1rem; display: flex; justify-content: center; width: 100%;">
  <iframe
    id="swa-anim"
    src="{{ '/assets/html/swa_recurrent_matrix_update.html' | relative_url }}"
    title="SWA recurrent matrix update"
    loading="lazy"
    scrolling="no"
    style="display: block; width: 70%; max-width: 1400px; height: 0; border: 0; background: transparent; overflow: hidden;"
  ></iframe>
</div>

<script>
  (() => {
    const iframe = document.getElementById('swa-anim');
    if (!iframe) return;
    const resize = () => {
      try {
        const doc = iframe.contentDocument || iframe.contentWindow?.document;
        if (!doc || !doc.body) return;
        const h = Math.max(doc.body.scrollHeight, doc.documentElement.scrollHeight);
        iframe.style.height = `${h}px`;
      } catch(e) {}
    };
    iframe.addEventListener('load', () => { resize(); setTimeout(resize,250); setTimeout(resize,1000); });
    window.addEventListener('resize', resize);
  })();
</script>

---

### Raven — Sparse Routing

<div style="margin: 1.5rem auto 1rem; display: flex; justify-content: center; width: 100%;">
  <iframe
    id="raven-anim"
    src="{{ '/assets/html/raven_recurrent_matrix_update.html' | relative_url }}"
    title="Raven recurrent matrix update"
    loading="lazy"
    scrolling="no"
    style="display: block; width: 70%; max-width: 1400px; height: 0; border: 0; background: transparent; overflow: hidden;"
  ></iframe>
</div>

<script>
  (() => {
    const iframe = document.getElementById('raven-anim');
    if (!iframe) return;
    const resize = () => {
      try {
        const doc = iframe.contentDocument || iframe.contentWindow?.document;
        if (!doc || !doc.body) return;
        const h = Math.max(doc.body.scrollHeight, doc.documentElement.scrollHeight);
        iframe.style.height = `${h}px`;
      } catch(e) {}
    };
    iframe.addEventListener('load', () => { resize(); setTimeout(resize,250); setTimeout(resize,1000); });
    window.addEventListener('resize', resize);
  })();
</script>
