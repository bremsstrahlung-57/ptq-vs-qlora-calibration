# Confidence Without Competence

Calibration risks in real-world LLM deployment pipelines.

This repository contains the anonymized TMLR submission source for the paper, along with experiment notebooks and figures. The current submission PDF is:

[paper/main.pdf](paper/main.pdf)

## Submission Status

The paper has been rebuilt with the official TMLR LaTeX style:

- `paper/main.tex` uses `\usepackage{tmlr}` without `accepted` or `preprint`.
- `paper/main.pdf` is anonymized for double-blind review.
- `paper/tmlr.sty`, `paper/tmlr.bst`, and `paper/fancyhdr.sty` are copied unchanged from the TMLR template.
- `paper/main_old.pdf` is a deanonymized old Google Docs PDF and must not be submitted.

## Overview

The paper studies how two compressed LLM deployment pipelines affect confidence calibration:

- Post-Training Quantization (PTQ)
- QLoRA fine-tuning in a quantized weight space

The focus is not just accuracy, but whether models can reliably estimate when they are wrong. Poor calibration, especially overconfidence, creates practical risk in production systems.

## Key Findings

- PTQ shows 7.7x higher Expected Calibration Error than QLoRA: 0.293 vs. 0.038.
- PTQ produces high-confidence incorrect predictions 29.7% of the time vs. 0.9% for QLoRA.
- Accuracy alone is insufficient for deployment decisions; calibration should be measured directly.

## Methodology

- Base model family: `Gemma-2-2B`
- Pipeline A (PTQ): `google/gemma-2-2b-it` with 4-bit NF4 quantization applied after instruction tuning.
- Pipeline B (QLoRA): `google/gemma-2-2b` quantized to 4-bit first, then fine-tuned with LoRA adapters on 800 Alpaca samples.
- Benchmarks: MMLU, ARC-Challenge, TruthfulQA
- Metrics: ECE, Brier score, Negative Log-Likelihood, overconfidence rate

## Results Snapshot

| Metric | Pipeline B (QLoRA) | Pipeline A (PTQ) | Comparison |
| :--- | :--- | :--- | :--- |
| Accuracy | 0.381 | 0.612 | 1.61x higher for PTQ |
| ECE | 0.038 | 0.293 | 7.7x worse for PTQ |
| Brier score | 0.223 | 0.299 | 1.34x worse for PTQ |
| NLL | 1.359 | 1.726 | 1.27x worse for PTQ |
| Mean confidence | 0.412 | 0.904 | 2.2x higher for PTQ |
| Overconfidence rate | 0.009 | 0.297 | 33x worse for PTQ |

## Repository Contents

- `paper/main.tex`: anonymized TMLR source
- `paper/main.pdf`: anonymized TMLR PDF
- `paper/main.bib`: checked bibliography
- `figures/`: reliability diagrams, confidence distributions, and comparison plots
- `ptq_eval/ptq_pipeline.ipynb`: PTQ evaluation notebook
- `qlora_train/qlora_pipeline.ipynb`: QLoRA training/evaluation notebook

## Limitations

The two pipelines differ in more than quantization order. Pipeline A uses Google's large-scale instruction tuning, while Pipeline B uses 800 Alpaca samples. The observed calibration gap likely reflects instruction-tuning scale as well as quantization order, so the paper treats this as an empirical comparison of realistic deployment pipelines rather than a controlled causal ablation.
