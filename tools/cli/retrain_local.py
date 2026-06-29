#!/usr/bin/env python3
"""Local DoRA retrain from the flywheel log — runs on the Mac (MPS). Reads flywheel_log.jsonl, keeps
ONLY the good examples (compiled + runs + high rerank score, i.e. colorful/non-radial — NOT rings, so
the loop doesn't reinforce the collapse), mixes in a sample of the original corpus (anti-forgetting),
and continues DoRA SFT on the current adapter. Produces an improved adapter the flywheel then serves.

  python3 retrain_local.py [--min-score 1.4] [--corpus-mix 2000] [--steps 300]

The debugger already labeled every example for free — no human annotation. The more it's used, the
more good (prompt, shader) pairs accumulate, the better the model gets.
"""
import argparse, json, os, random
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
LOG = os.path.join(ROOT, "flywheel_log.jsonl")
BASE = os.environ.get("FLY_BASE", os.path.join(ROOT, "local", "qwen2.5-coder-3b"))
ADAPTER = os.environ.get("FLY_ADAPTER", os.path.join(ROOT, "local", "rl3b_refined"))
CORPUS = os.environ.get("FLY_CORPUS", os.path.join(ROOT, "local", "corpus.txt"))

def good_examples(min_score):
    out = []
    if not os.path.exists(LOG): return out
    for line in open(LOG):
        try: r = json.loads(line)
        except Exception: continue
        c = r["candidates"][r["chosen_index"]]
        # quality gate: must compile+run AND clear the rerank bar (color/non-radial), not a ring
        if c.get("compiled") and c.get("runs") and c.get("score", 0) >= min_score and c.get("radial", 1) < 0.85:
            out.append(f"// Shader: {r['prompt']}\n{c['glsl'].rstrip()}\n<|endoftext|>\n")
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-score", type=float, default=1.4, help="rerank-score bar (filters out rings/grayscale)")
    ap.add_argument("--corpus-mix", type=int, default=2000, help="N original-corpus docs to mix in (anti-forgetting)")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--out", default=os.path.join(ROOT, "local", "rl3b_flywheel"))
    a = ap.parse_args()
    fly = good_examples(a.min_score)
    print(f"[retrain] {fly} -> {len(fly)} good flywheel examples (compiled+runs, non-ring, score>={a.min_score})")
    if len(fly) < 20:
        print(f"[retrain] only {len(fly)} good examples — collect more usage first (need ~20+). Stopping."); return
    docs = list(fly)
    if os.path.exists(CORPUS):                              # mix corpus for anti-forgetting
        corp = [d for d in open(CORPUS, errors="replace").read().split("<|endoftext|>") if len(d.strip()) > 50]
        random.Random(0).shuffle(corp); docs += [d.strip() + "\n<|endoftext|>\n" for d in corp[:a.corpus_mix]]
    random.Random(1).shuffle(docs)
    print(f"[retrain] training set: {len(docs)} docs ({len(fly)} new + {len(docs)-len(fly)} corpus)")
    import torch
    from transformers import (AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer,
                              DataCollatorForLanguageModeling)
    from peft import PeftModel
    from datasets import Dataset
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[retrain] device={dev}")
    tok = AutoTokenizer.from_pretrained(BASE)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    ds = Dataset.from_dict({"text": docs}).map(lambda e: tok(e["text"], truncation=True, max_length=768),
                                               batched=True, remove_columns=["text"])
    m = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float16).to(dev)
    m = PeftModel.from_pretrained(m, ADAPTER, is_trainable=True)   # CONTINUE the current adapter
    m.config.use_cache = False
    Trainer(model=m,
            args=TrainingArguments(output_dir=a.out, per_device_train_batch_size=1, gradient_accumulation_steps=8,
                max_steps=a.steps, learning_rate=5e-5, logging_steps=10, save_steps=10**9, report_to=[],
                use_mps_device=(dev == "mps")),
            train_dataset=ds, data_collator=DataCollatorForLanguageModeling(tok, mlm=False)).train()
    m.save_pretrained(a.out + "/adapter")
    print(f"[retrain] improved adapter -> {a.out}/adapter  (point FLY_ADAPTER here to serve it)")

if __name__ == "__main__":
    main()
