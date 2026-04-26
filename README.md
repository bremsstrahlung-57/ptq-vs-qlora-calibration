# Confidence Without Competence: Calibration Risks in Real-World LLM Deployment Pipelines
**Sagar Sharma**  
2026  
[Read the paper](paper/main.pdf)

## Overview
This repository accompanies the paper *"Confidence Without Competence"*, which studies how deployment pipelines affect **confidence calibration** in compressed LLMs.

We compare two common approaches:
- Post-Training Quantization (PTQ)
- Quantization-Aware Fine-Tuning (QLoRA)

The focus is not just accuracy, but whether models can reliably estimate when they are wrong. Poor calibration  especially overconfidence introduces real risks in production systems.

## Key Findings
- **Calibration Gap:** PTQ shows 7.7× higher Expected Calibration Error (ECE) than QLoRA (0.293 vs 0.038), despite higher accuracy.
- **Overconfidence:** PTQ produces high-confidence incorrect predictions 29.7% of the time vs 0.9% for QLoRA.
- **Implication:** Accuracy alone is insufficient for deployment decisions. Calibration should be treated as a first-class metric.

## Methodology
* **Base Model:** `Gemma-2-2B`
* **Pipelines Evaluated:**
  * **Pipeline A (PTQ):** `google/gemma-2-2b-it` with 4-bit NF4 quantization applied post-instruction tuning.
  * **Pipeline B (QLoRA):** `google/gemma-2-2b` (base) quantized to 4-bit first, then fine-tuned using LoRA adapters on an Alpaca subset (800 samples).
* **Benchmarks:** MMLU, ARC-Challenge, TruthfulQA
* **Metrics:** ECE, Brier Score, Negative Log-Likelihood (NLL), Overconfidence Rate

## Results Snapshot

| Metric | Pipeline B (QLoRA) | Pipeline A (PTQ) | Comparison |
| :--- | :--- | :--- | :--- |
| **Accuracy ↑** | 0.381 | **0.612** | 1.61x higher for PTQ |
| **ECE ↓** | **0.038** | 0.293 | 7.7x worse for PTQ |
| **Brier Score ↓** | **0.223** | 0.299 | 1.34x worse for PTQ |
| **NLL ↓** | **1.359** | 1.726 | 1.27x worse for PTQ |
| **Mean Confidence** | 0.412 | 0.904 | 2.2x higher for PTQ |
| **Overconfidence Rate ↓** | **0.009** | 0.297 | 33x worse for PTQ |

## Limitations
The two pipelines differ in more than quantization order: Pipeline A uses Google's large-scale instruction tuning; Pipeline B uses 800 Alpaca samples. The calibration gap likely reflects instruction tuning scale as much as quantization order. We treat this as an empirical comparison of real-world deployment pipelines, not a controlled ablation — see Section 6 of the paper for full discussion.

## Reproducibility
All experiments use fixed dataset sampling (seed=42), identical evaluation logic for both pipelines, and confidence computed over constrained answer tokens (A/B/C/D).

```bash
# PTQ evaluation
cd ptq_eval
jupyter notebook ptq_pipeline.ipynb

# QLoRA training
cd qlora_train
jupyter notebook qlora_pipeline.ipynb
```

## Repository Contents
* **Paper:** [paper/main.pdf](paper/main.pdf)
* **Figures:** [figures/](figures/) — Reliability diagrams, confidence distributions, and visualizations
* **PTQ Evaluation:** [ptq_eval/ptq_pipeline.ipynb](ptq_eval/ptq_pipeline.ipynb)
* **QLoRA Training:** [qlora_train/qlora_pipeline.ipynb](qlora_train/qlora_pipeline.ipynb)
