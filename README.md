# Confidence Without Competence

### **Research Goal**

The paper is titled "Confidence Without Competence: Calibration Risks in Real-World LLM Deployment Pipelines." The core question is: does the order in which you quantize and fine-tune an LLM affect its calibration? Calibration means how well a model's confidence matches its actual accuracy. A well-calibrated model that says it's 80% confident is right 80% of the time. A mis calibrated model is overconfident or underconfident in ways that are dangerous in deployment.

### **Why This Matters**

In practice, teams deploy quantized models to save memory and inference cost. But there are two common pipelines: fine-tune first then quantize (PTQ), or quantize first then fine-tune on top (QLoRA). Nobody has cleanly isolated whether the order itself changes calibration behavior.

### **The Experimental Design**

Two pipelines, same everything except order:

- Pipeline A: Qwen3-1.7B-Base → LoRA fine-tune on UltraChat (fp16) → merge adapters → PTQ quantize
- Pipeline B: Qwen3-1.7B-Base → PTQ quantize → QLoRA fine-tune on UltraChat

Same base model, same dataset, same token budget (4,749 examples, ~5M tokens), same LoRA config (rank 16, alpha 32, dropout 0.05, targeting q/k/v/o projections), same sequence length (2048), same seed (42). The only variable is when quantization happens relative to fine-tuning.

### **Why LoRA for Pipeline A instead of full fine-tune**

Originally A was supposed to be full fine-tune, but Kaggle 2xT4 (16GB each) OOMs on full fine-tune of 1.7B even at seq_len=2048 because Adam optimizer states alone require ~13.6GB on top of weights and gradients. LoRA trains only 6.4M params (~0.37% of total), so optimizer states become negligible. Crucially, using LoRA for both pipelines is actually a cleaner design: the only difference between A and B is quantization order, not fine-tuning method. This directly addresses the original TMLR reviewer critique that the old design confounded quantization order with instruction-tuning scale.

### **Quantization Method**

Both pipelines use the same quantization method (bitsandbytes 4-bit NF4, or GPTQ, to be decided but must be identical). This is critical. If A and B used different quant methods, you'd have two confounds again.

### **Dataset**

UltraChat-200k, subsampled to 5M tokens. 4,749 examples after dropping anything over 2048 tokens (436 dropped, ~8.4%). OpenThoughts was considered for a reasoning dimension but dropped because reasoning traces average 7,152 tokens, incompatible with T4 memory constraints but I will try to test on it too. 

### **Evaluation**

After both pipelines are trained and quantized, run the same eval suite on all checkpoints:

- Capability: MMLU, ARC-Challenge, TruthfulQA 
- Calibration: ECE, Brier Score, NLL, overconfidence rate

The claim is not just that one pipeline is more accurate, but that quantization order produces measurably different calibration behavior independent of raw task performance.

### **Code Release**

Code for training, quantization, evaluation, and calibration analysis will be added after the preprint is released. 

### **Current State**

All experiments are complete. Results are in hand.

**Dataset:** UltraChat-200k subsampled to 4,749 examples (~5M tokens), max_seq_len 2048, seed 42`.

**Pipeline A:** Qwen3-1.7B-Base → LoRA fine-tune (rank 16, alpha 32, fp16, 1 epoch, lr 2e-4, batch 4, grad accum 4) → merge adapters → NF4 PTQ quantization. Trained cleanly in fp16 using standard TRL.

**Pipeline B:** Qwen3-1.7B-Base → NF4 PTQ quantization → QLoRA fine-tune (same LoRA config, same dataset). Required patching TRL's sft_trainer.py to resolve a dtype conflict where gradient checkpointing was casting LoRA adapter weights from fp32 to bf16, causing GradScaler to fail with `_amp_foreach_non_finite_check_and_unscale_cuda not implemented for BFloat16`. The fix applied `use_reentrant=False` for gradient checkpointing and used cast_mixed_precision_params to explicitly keep adapter weights in fp32 and non-trainable parameters in fp16. Library versions pinned: `transformers 5.0.0`, `peft 0.18.1`, `trl 1.7.0`, `bitsandbytes 0.49.2`.

Evaluation ran on MMLU, ARC-Challenge, TruthfulQA using custom inference loop (last-token logits over answer letter token IDs, softmax over choices). Results:

| Metric | Pipeline A | Pipeline B |
|---|---|---|
| Accuracy | 0.492 | 0.486 |
| ECE | 0.130 | 0.062 |
| ECE (temp scaled) | 0.046 | 0.058 |
| Brier Score | 0.640 | 0.624 |
| NLL | 1.215 | 1.157 |
| Overconfidence rate | 0.279 | 0.138 |
| Optimal temperature | 1.642 | 1.119 |

**Headline finding**: at matched accuracy, Pipeline B (quantize first) is roughly 2x better calibrated than Pipeline A (fine-tune first) on both ECE and overconfidence rate.

**Known limitation**: Pipeline A trained in fp16, Pipeline B required the TRL patch which introduced non-reentrant gradient checkpointing. Training precision was not perfectly matched.
