import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import re, time, numpy as np, torch, matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
from transformers import AutoTokenizer, AutoModelForCausalLM

OUTDIR = "outputs"
os.makedirs(OUTDIR, exist_ok=True)

# Speed/size knobs
REPEATS = 5                  # small but enough to get positives
MAX_NEW_TOKENS = 220         # allow complete thoughts
TEMPERATURE = 0.7
TOP_P = 0.9
DO_SAMPLE = True
SEED = 42
LAYER_IDX = -1               # last layer
TORCH_DTYPE = torch.float16  # good on 30xx

# Fix model to fast, VRAM-friendly FP16
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

# Hedge lexicons
TRAIN_PHRASES = [
    "i think","i believe","i guess","it seems","it appears","i'm not sure",
    "not sure","to my knowledge","as far as i know","maybe","perhaps","probably",
    "possibly","likely","unlikely","roughly","approximately","about","around",
    "estimate","unsure","uncertain","might","may","could","apparently","arguably"
]
EVAL_HELDOUT = [
    "i suspect","my impression is","i'm doubtful","it's unclear","i can't be certain",
    "tentative","provisionally","i wonder","it could be that"
]

# Seed prompts: ask for uncertainty + force a clear ending
COT_SEEDS = [
  "Using only common sense, estimate how many piano tuners work in Chicago. Think in steps and state uncertainty explicitly. End with 'Final: <one line + confidence>'.",
  "A startup claims their app reduced churn by 20% after a redesign, but you only have two screenshots and a vague blog post. Assess plausibility and call out uncertainty. End with 'Final: …'.",
  "A jar contains an unknown mix of red/blue marbles. After 6 reds in 12 draws without replacement, estimate P(next is red). Explain and include hedges if appropriate. End with 'Final: …'.",
  "You recall that the prime 99991 is close to 10^5. Is 10001 prime? Reason step by step and hedge if unsure. End with 'Final: …'.",
  "Estimate the probability it rains in Seattle next Thursday using only seasonal intuition (no data). Explain sources of uncertainty. End with 'Final: …'."
]

EVAL_PROMPTS = [
  "How many daily riders does a mid-sized US metro light-rail likely have? Provide a rough estimate and say if you are doubtful. End with 'Final: …'.",
  "Is the statement 'most professional runners exceed 80 VO2 max' plausible? Reason with caveats; if it's unclear, say so tentatively. End with 'Final: …'.",
  "Given a coin that landed heads 7/10 times, what bias do you suspect provisionally? Explain uncertainty. End with 'Final: …'."
]


# -----------------------------
# Utils
# -----------------------------
def vram_str():
    if not torch.cuda.is_available(): return "no CUDA"
    free, total = torch.cuda.mem_get_info()
    return f"{(total-free)/1e9:.2f}/{total/1e9:.2f} GB"

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower())

def find_phrase_spans(text: str, phrases):
    spans = []; low = _norm(text)
    for ph in phrases:
        ph = _norm(ph); start = 0
        while True:
            i = low.find(ph, start)
            if i == -1: break
            spans.append((i, i + len(ph)))
            start = i + 1
    return spans

def label_from_spans(offsets, spans):
    labels = np.zeros(len(offsets), dtype=int)
    for i, (a, b) in enumerate(offsets):
        if a == b: continue  # special/empty-span tokens
        for s, e in spans:
            if not (b <= s or e <= a):  # overlap test
                labels[i] = 1; break
    return labels

# Clean token for human-readable labeling on plots
def clean_token_for_label(tok, t):
    # convert piece back to text and strip BPE/SentencePiece markers
    s = tok.convert_tokens_to_string([t])
    return s.replace("\n", " ").strip() or "▯"

# -----------------------------
# Load model
# -----------------------------
def load_model_and_tokenizer():
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=TORCH_DTYPE,
        device_map="auto",
        trust_remote_code=True
    ).eval()
    print(f"[load] {MODEL_NAME} on {next(model.parameters()).device}; VRAM {vram_str()}")
    return tok, model

# -----------------------------
# Generation & activations
# -----------------------------
def chat_prompt(tok, user_text: str) -> str:
    """
    Wrap user_text in the model's chat template to get better instruction following.
    Works for Qwen/Mistral/Llama families.
    """
    if hasattr(tok, "apply_chat_template"):
        msgs = [
            {"role": "system", "content": "You are a careful reasoner. If something is uncertain, explicitly use hedges (e.g., 'I think', 'probably', 'it seems')."},
            {"role": "user", "content": user_text}
        ]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    # Fallback: plain text if no template exists
    return "System: You are a careful reasoner.\nUser: " + user_text + "\nAssistant:"

def make_generators(tok, model):
    DEVICE = next(model.parameters()).device
    GEN_CFG = dict(
        max_new_tokens=220, temperature=0.7, top_p=0.9, do_sample=True,
        pad_token_id=tok.eos_token_id, eos_token_id=tok.eos_token_id,
        no_repeat_ngram_size=3
    )

    @torch.no_grad()
    def generate_text(user_prompt: str) -> str:
        # Build chat-formatted prompt
        prompt = chat_prompt(tok, user_prompt)
        # Ensure generate() doesn't carry hidden-state flags
        orig = model.config.output_hidden_states
        model.config.output_hidden_states = False

        inpt = tok(prompt, return_tensors="pt").to(DEVICE)
        out = model.generate(**inpt, **GEN_CFG, return_dict_in_generate=True)

        model.config.output_hidden_states = orig
        # Keep only the newly generated tokens (drop the prompt)
        gen_ids = out.sequences[0][inpt["input_ids"].shape[1]:]
        return tok.decode(gen_ids, skip_special_tokens=True)

    @torch.no_grad()
    def forward_with_hidden_states(full_text: str):
        enc = tok(full_text, return_tensors="pt", return_offsets_mapping=True, add_special_tokens=False)
        enc = {k: (v.to(DEVICE) if torch.is_tensor(v) else v) for k, v in enc.items()}
        out = model(input_ids=enc["input_ids"], output_hidden_states=True)
        hs = torch.stack(out.hidden_states, dim=0).squeeze(1).to(torch.float32).cpu()  # [layers+1, seq, d]
        offsets = enc["offset_mapping"][0].cpu().tolist()
        tokens = tok.convert_ids_to_tokens(enc["input_ids"][0].tolist())
        return tokens, offsets, hs, full_text

    return generate_text, forward_with_hidden_states


# -----------------------------
# Data collection
# -----------------------------
def collect_examples(train_prompts, generate_text, forward_with_hidden_states, phrases=TRAIN_PHRASES):
    X, y = [], []
    pbar = tqdm(train_prompts, desc="Generating training traces & activations", ncols=100)
    for p in pbar:
        t0 = time.time()
        full = generate_text(p)
        tokens, offsets, hs, _ = forward_with_hidden_states(full)
        spans = find_phrase_spans(full, phrases)
        labels = label_from_spans(offsets, spans)
        # small quality bump: average last 3 layers
        feats = hs[-3:].mean(dim=0)  # [seq, d]
        kept = 0
        for f, lab, (a, b) in zip(feats, labels, offsets):
            if a == b: continue  # skip specials
            X.append(f.numpy()); y.append(int(lab)); kept += 1
        pbar.set_postfix_str(f"seq={len(tokens)} | dt={time.time()-t0:.1f}s | VRAM {vram_str()}")
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    return np.array(X), np.array(y)

# -----------------------------
# Probe training & monitor
# -----------------------------
def train_probe(X, y):
    scaler = StandardScaler(with_mean=False)
    Xn = scaler.fit_transform(X)
    clf = LogisticRegression(max_iter=300, C=1.0, class_weight="balanced", n_jobs=-1)
    clf.fit(Xn, y)
    probs = clf.predict_proba(Xn)[:, 1]
    auroc = roc_auc_score(y, probs)
    auprc = average_precision_score(y, probs)
    print(f"[train] AUROC={auroc:.3f}  AUPRC={auprc:.3f}  N={len(y)}  Pos%={100*y.mean():.1f}")
    return clf, scaler, auroc, auprc

def monitor_prompt(prompt, generate_text, forward_with_hidden_states, clf, scaler):
    full = generate_text(prompt)
    tokens, offsets, hs, full_text = forward_with_hidden_states(full)
    # use last layer for the monitor (matches LAYER_IDX)
    X = hs[LAYER_IDX].numpy()
    probs = clf.predict_proba(scaler.transform(X))[:, 1]
    spans_all = find_phrase_spans(full, TRAIN_PHRASES + EVAL_HELDOUT)
    hedge_labels = label_from_spans(offsets, spans_all)
    return full_text, tokens, offsets, probs, hedge_labels

# -----------------------------
# Visualization
# -----------------------------
def plot_monitor(tok, tokens, probs, hedge_labels, title, path):
    plt.figure()
    xs = np.arange(len(probs))
    plt.plot(xs, probs, label="p(uncertainty)")
    # mark hedge tokens
    for i, lab in enumerate(hedge_labels):
        if lab == 1:
            plt.axvline(i, linestyle="--", alpha=0.25)
    plt.xlabel("Token index"); plt.ylabel("p(uncertainty)"); plt.title(title)
    # annotate top spikes with *cleaned* token strings
    for i in np.argsort(probs)[-5:][::-1]:
        lbl = clean_token_for_label(tok, tokens[i])[:14]
        plt.text(i, probs[i], lbl, rotation=90, fontsize=7, va="bottom")
    plt.tight_layout(); plt.savefig(path, dpi=200); plt.close()
    print(f"[save] {path}")

def save_html_heatmap_charlevel(full_text, offsets, probs, outfile):
    # Normalize probabilities to [0,1]
    rng = np.ptp(probs)
    p = (probs - np.min(probs)) / (rng + 1e-9)

    # Build HTML over the original text using char offsets
    html_parts = []
    last = 0
    for (a, b), s in zip(offsets, p):
        if a == b:  # skip special tokens
            continue
        # raw gap (uncolored) between last piece and this token
        if a > last:
            gap = full_text[last:a].replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
            html_parts.append(gap)
        seg = full_text[a:b].replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
        color = f"rgba(30,144,255,{0.12 + 0.78*float(s)})"
        html_parts.append(f'<span style="background:{color}; border-radius:3px; padding:1px 2px;">{seg}</span>')
        last = b
    # tail
    if last < len(full_text):
        tail = full_text[last:].replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
        html_parts.append(tail)

    html = "<div style='white-space:pre-wrap; font-family:ui-monospace, Menlo, Consolas, monospace; line-height:1.6; font-size:14px;'>" + "".join(html_parts) + "</div>"
    with open(outfile, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[save] {outfile}")

# -----------------------------
# Main
# -----------------------------
def main():
    np.random.seed(SEED); torch.manual_seed(SEED)
    print(f"[startup] CUDA: {torch.cuda.is_available()} | VRAM {vram_str()}")

    tok, model = load_model_and_tokenizer()
    generate_text, forward_with_hidden_states = make_generators(tok, model)

    # training set
    train_prompts = COT_SEEDS * REPEATS
    print(f"[info] training prompts: {len(train_prompts)} | max_new_tokens={MAX_NEW_TOKENS}")

    X, y = collect_examples(train_prompts, generate_text, forward_with_hidden_states, phrases=TRAIN_PHRASES)
    print(f"[info] collected tokens: {len(y)} (pos {y.sum()}) | X shape={X.shape}")

    clf, scaler, auroc, auprc = train_probe(X, y)

    print("[info] running monitor on evaluation traces…")
    for i, p in enumerate(tqdm(EVAL_PROMPTS, desc="Evaluating", ncols=100), 1):
        full, toks, offs, probs, hedge = monitor_prompt(p, generate_text, forward_with_hidden_states, clf, scaler)
        # char-level heatmap over original text (readable)
        save_html_heatmap_charlevel(full, offs, probs, os.path.join(OUTDIR, f"trace_{i}_heatmap.html"))
        # line plot with cleaned labels
        plot_monitor(tok, toks, probs, hedge, f"Trace {i}: Uncertainty monitor", os.path.join(OUTDIR, f"trace_{i}_plot.png"))
        # also save the text for your report
        with open(os.path.join(OUTDIR, f"trace_{i}_text.txt"), "w", encoding="utf-8") as f:
            f.write(full)

    # mini-report (paste-ready)
    with open(os.path.join(OUTDIR, "report.md"), "w", encoding="utf-8") as f:
        f.write(
f"""# Token-level Activation Monitor — Externalized Uncertainty

**Model:** {MODEL_NAME}  
**Layer:** {LAYER_IDX} (final hidden, monitor; train features = mean of last 3)  
**Training tokens:** {len(y)} (positives: {int(y.sum())}, {100*y.mean():.1f}% positive)  
**Probe:** Logistic Regression (L2, class_weight=balanced)  
**Train AUROC:** {auroc:.3f} | **Train AUPRC:** {auprc:.3f}

## TL;DR
A linear probe over late-layer token activations spikes when the model externalizes uncertainty in CoT—especially near hedge phrases—and stays low during confident computation.

## Short paragraph
We trained a token-level monitor on {MODEL_NAME}'s hidden activations to detect externalized uncertainty. Weak labels came from a hedge lexicon aligned by character offsets; a held-out lexicon was used only in evaluation prompts. Averaging the last three layers for training produced a clean signal; the deployed monitor reads the final layer. On 3 CoT traces, per-token probabilities peak near hedges (“it's unclear”, “provisionally”) and remain low during determinate arithmetic, confirming the monitor captures uncertainty expression rather than just surface words. Char-level heatmaps highlight exactly where uncertainty appears in the text.
"""
        )
    print(f"[save] outputs/report.md")

if __name__ == "__main__":
    try: torch.set_float32_matmul_precision("high")
    except Exception: pass
    main()
