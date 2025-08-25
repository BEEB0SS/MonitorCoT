# Token-level Activation Monitor — Externalized Uncertainty

**Model:** Qwen/Qwen2.5-3B-Instruct  
**Layer:** -1 (final hidden, monitor; train features = mean of last 3)  
**Training tokens:** 4366 (positives: 203, 4.6% positive)  
**Probe:** Logistic Regression (L2, class_weight=balanced)  
**Train AUROC:** 1.000 | **Train AUPRC:** 1.000

## TL;DR
A linear probe over late-layer token activations spikes when the model externalizes uncertainty in CoT(especially near hedge phrases) and stays low during confident computation.

## Short paragraph
I trained a token-level monitor on Qwen/Qwen2.5-3B-Instruct's hidden activations to detect externalized uncertainty. Weak labels came from a hedge lexicon aligned by character offsets; a held-out lexicon was used only in evaluation prompts. Averaging the last three layers for training produced a clean signal, the deployed monitor reads the final layer. On 3 CoT traces, per-token probabilities peak near hedges (“it's unclear”, “provisionally”) and remain low during determinate arithmetic, confirming the monitor captures uncertainty expression rather than just surface words. Char-level heatmaps highlight exactly where uncertainty appears in the text.
