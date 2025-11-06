# CrisisAI: Integrating Real‑Time Context into Language Models for Dynamic Response Generation

**Advisor:** Dr. Lingzi Hong
**Status:** Active • **Last updated:** November 6, 2025

---

## Overview

CrisisAI is a research project focused on enhancing large language models (LLMs) with **real‑time contextual awareness** for crisis and emergency response scenarios. During crises—natural disasters, public health emergencies, or conflicts—the environment changes rapidly, and user‑generated content (e.g., social media posts, local updates) becomes a key source of situational intelligence. Traditional LLMs, pretrained on static datasets, struggle to process this dynamic information effectively.

This project investigates **how to integrate real‑time, spatiotemporal context** into LLM architectures to improve responsiveness, relevance, and trustworthiness of generated outputs during rapidly evolving events.

---

## Research Goals

1. **Dynamic Context Integration:** Develop methods to continuously feed LLMs with evolving contextual signals (location, time, entities, and event trajectories).
2. **Entity Extraction & Linking:** Build robust NLP pipelines to identify, disambiguate, and link crisis‑related entities from noisy, real‑time data streams (e.g., Twitter/X, Telegram, and emergency feeds).
3. **Retrieval‑Augmented Generation (RAG):** Evaluate RAG‑style systems that query real‑time data sources before generation, grounding outputs in current user‑contributed information.
4. **Evaluation Framework:** Design metrics and benchmarks for responsiveness, factuality, and ethical reliability under dynamic crisis conditions.

---

## Motivation

* **Problem:** LLMs lack mechanisms for *temporal grounding* and *live data adaptation*.
* **Consequence:** Generated guidance or summaries risk becoming outdated, incomplete, or misleading in fast‑changing scenarios.
* **Goal:** Bridge static language modeling with dynamic, situationally‑aware reasoning.

---

## Methodology

### 1. Data Ingestion & Preprocessing

* Collect real‑time social and sensor data from verified crisis sources.
* Apply streaming NLP techniques for entity and event extraction.
* Normalize and store time‑stamped entities for rapid retrieval.

### 2. Contextual Retrieval Layer

* Implement retrieval modules that dynamically update contextual embeddings.
* Incorporate temporal relevance weighting and spatial proximity scoring.

### 3. Real‑Time RAG Architecture

* Extend baseline LLMs with a **context orchestrator** that queries, ranks, and fuses external data before generation.
* Compare different integration mechanisms (pre‑retrieval vs. mid‑generation grounding).

### 4. Evaluation

* **Quantitative:** latency, factual precision, temporal consistency.
* **Qualitative:** user trust, relevance, and perceived responsiveness.

---

## Expected Contributions

* A modular framework for **real‑time data integration** with foundation models.
* Novel benchmarks for evaluating **temporal grounding and contextual responsiveness**.
* Insights into ethical and operational design for **AI systems in crisis domains**.

---

## Project Structure

```
CrisisAI
├─ data_streams/        # Real‑time data ingestion & preprocessing scripts
├─ entity_extraction/   # NLP pipelines for named entity recognition & linking
├─ retrieval_layer/     # Temporal‑spatial retrieval modules
├─ generation_core/     # RAG integration & prompt orchestration
├─ evaluation/          # Metrics and benchmarks for dynamic response
└─ docs/                # Research reports and experiment logs
```

---
