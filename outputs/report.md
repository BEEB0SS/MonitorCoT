# Token-level Activation Monitor — Externalized Uncertainty

**Model:** Qwen/Qwen2.5-3B-Instruct  
**Layer:** -1 (final hidden, monitor; train features = mean of last 3)  
**Training tokens:** 4366 (positives: 203, 4.6% positive)  
**Probe:** Logistic Regression (L2, class_weight=balanced)  
**Train AUROC:** 1.000 | **Train AUPRC:** 1.000

## TL;DR
I trained a tiny token-level probe on Qwen2.5-3B-Instruct’s hidden states to flag externalized uncertainty. On three fresh CoT traces, the monitor spikes right on hedging phrases (e.g., estimate / I’m not certain / it seems / might) and stays mostly flat during confident steps. The heatmaps darken exactly over those words.

## Short paragraph
I built a simple logistic-regression monitor over token activations (features = mean of the last 3 layers) and weak-labeled tokens using a hedge lexicon aligned by character spans. I generate CoT with the model’s chat template and score only the completion, then plot p(uncertainty) per token. In the figures, vertical dashed lines mark hedge hits and the curve jumps at the same spots—e.g., in Trace 1 it peaks on “estimate,” “could be around,” “I’m not certain,” “it seems,” “might”; Trace 3 spikes at “it seems” and “provisional.” Outside those phrases, the line is near zero while the model does normal arithmetic/explanations, which is what I want. The HTML heatmaps are the text itself colored by score: darker blue bands sit over those hedging words, making it obvious where uncertainty is being externalized. Overall, this shows a lightweight activation monitor can localize uncertainty expression at token granularity without touching the base model.
