---
layout: distill
title: Story of Linear Time Sequence Modeling 📚
description: Summary of Linear Transformers
tags:
giscus_comments: false
date: 2025-08-29
featured: false
thumbnail: assets/img/ourlogo.jpg

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



------------------

# Why Linear Transformers?

If you’ve heard about Large Language Models (LLMs) and Transformers and are curious to learn more, there are already plenty of excellent blog posts, articles, and YouTube videos that explain them in great detail with amazing visualizations. In case your main interest is understanding LLMs and how Transformers work, we’d point you there first, they’ve done a fantastic job (honestly, better than us).

In this series, we’ll cover Linear Transformers and State Space Models (SSMs), giving a high-level summary of their core ideas. So, if you’ve come across names like **Mamba** or **DeltaNet** and wondered what they are, or if you’ve asked yourself: 

**1)** Why we moved from RNNs to Transformers?

**2)** Why we now seem to be circling back from Transformers toward Linear Transformers (almost like “RNNs on steroids”)?
 
then this post should be a good fit 😉.

So let’s start from answering the above questions 



----------------
