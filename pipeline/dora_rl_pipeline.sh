#!/usr/bin/env bash
# DoRA + debugger-in-the-loop RL on Qwen2.5-Coder-7B — produces (a) a fully usable shader model
# and (b) the 4-row ablation ladder that is the paper's contribution.
#
#   base 7B  ->  +DoRA SFT (full corpus)  ->  +RL(compile-only)   [ablation]
#                                         \->  +RL(full debugger)  [the deliverable model]
#
# Every row is scored on a held-out NL prompt set with the OmniTrace rich reward:
#   compile@k (builds) -> run@k (runs finite) -> render@k (non-degenerate image).
set -uo pipefail
say(){ echo "[dora-rl $(date +%H:%M)] $*"; }
M=~/models/qwen2.5-coder-7b; D=~/shader_data/dora_full; P=~/pipeline; SP=~/shader_pipeline
RB=~/NERC/build/omni_reward; HELD=~/shader_data/heldout_nl.jsonl
mkdir -p "$D" "$P"

# ---- wait for inputs: the 7B weights + compiler-in-the-loop synthetic gen ----
while [ ! -f "$M/config.json" ]; do sleep 30; done
say "7B present; waiting for gen_verified to finish..."
while pgrep -f gen_verified.py >/dev/null; do sleep 60; done
say "gen_verified done."

# ---- STAGE 0: assemble the full SFT corpus (real 196k + all compile-verified synthetic) ----
say "STAGE 0: assemble FULL SFT corpus"
: > "$D/all.txt"
cat ~/shader_data/corpus_v4/corpus.txt        >> "$D/all.txt" 2>/dev/null   # ~192k real
cat ~/shader_data/distill/compiling.txt       >> "$D/all.txt" 2>/dev/null   # ~12k clean synthetic
cat ~/shader_data/distill/verified.txt        >> "$D/all.txt" 2>/dev/null   # compiler-in-loop clean
cat ~/shader_data/synth/synth_corpus_full.txt >> "$D/all.txt" 2>/dev/null   # earlier synth
say "SFT corpus: $(grep -c endoftext "$D/all.txt") shaders, $(wc -c < "$D/all.txt") bytes"

# ---- disjoint prompt splits: RL trains on corpus headers, eval on curated NL requests ----
sed -n '200,280p' ~/shader_data/corpus_v4/val.jsonl > ~/shader_data/rl_prompts_7b.jsonl

evalrow(){ # <label> <out.json> [adapter]
  local label="$1" out="$2" adapter="${3:-}"
  local args=(--base "$M" --prompts "$HELD" --reward-bin "$RB" --k 4 --label "$label" --out "$out")
  [ -n "$adapter" ] && args+=(--adapter "$adapter")
  python3 "$SP/eval_full.py" "${args[@]}" 2>&1 | tee "$P/eval_${label}.log"
}

# ---- ROW 1: base 7B (no shader specialization) ----
say "ROW1: eval base 7B"
evalrow base7b "$P/eval_row1.json"

# ---- STAGE 1: DoRA SFT (full corpus, 6000 steps ~ 0.6 epoch of 170M tokens) ----
say "STAGE 1: DoRA SFT on Qwen2.5-Coder-7B (6000 steps)"
python3 "$SP/finetune_dora.py" "$M" 6000 dora7b "$D" 2>&1 | tee "$P/dora7b_sft.log"
say "ROW2: eval +DoRA SFT"
evalrow dora7b_sft "$P/eval_row2.json" "$P/dora7b_out/adapter"

# ---- ROW 3 (ABLATION): RL with COMPILE-ONLY reward (standard RLVR) ----
say "STAGE 2a: RL compile-only (ablation) — 150 steps"
PYTHONPATH=$SP python3 "$SP/rl_refine.py" \
  --prompts ~/shader_data/rl_prompts_7b.jsonl --base "$M" --adapter "$P/dora7b_out/adapter" \
  --omni-reward-bin "$RB" --reward-mode compile --group 8 --steps 150 --log-every 1 \
  --out "$P/dora7b_rl_compile" 2>&1 | tee "$P/dora7b_rl_compile.log"
say "ROW3: eval +RL(compile-only)"
evalrow rl_compile "$P/eval_row3.json" "$P/dora7b_rl_compile"

# ---- ROW 4 (DELIVERABLE): RL with the FULL OmniTrace debugger reward ----
say "STAGE 2b: RL full-debugger (the deliverable) — 160 steps"
PYTHONPATH=$SP python3 "$SP/rl_refine.py" \
  --prompts ~/shader_data/rl_prompts_7b.jsonl --base "$M" --adapter "$P/dora7b_out/adapter" \
  --omni-reward-bin "$RB" --reward-mode debugger --group 8 --steps 160 --log-every 1 \
  --out "$P/dora7b_rl_debugger" 2>&1 | tee "$P/dora7b_rl_debugger.log"
say "ROW4: eval +RL(full debugger) — the deliverable"
evalrow rl_debugger "$P/eval_row4.json" "$P/dora7b_rl_debugger"

# ---- the ablation ladder ----
say "=== ABLATION LADDER (held-out NL prompts, rich reward) ==="
python3 - "$P" <<'PY'
import json, sys, os
P = sys.argv[1]
order = [("eval_row1.json","base"),("eval_row2.json","+DoRA SFT"),
         ("eval_row3.json","+RL compile-only"),("eval_row4.json","+RL full-debugger")]
print(f"{'row':18s} {'compile@1':>9s} {'run@1':>6s} {'render@1':>8s} {'compile@k':>9s} {'render@k':>8s} {'meanR':>6s}")
for fn, name in order:
    p = os.path.join(P, fn)
    if not os.path.exists(p): print(f"{name:18s}  (missing)"); continue
    d = json.load(open(p))["summary"]
    print(f"{name:18s} {d['compile@1']:9} {d['run@1']:6} {d['render@1']:8} {d['compile@k']:9} {d['render@k']:8} {d['mean_reward']:6}")
PY
say "=== DORA7B_DONE === deliverable adapter: $P/dora7b_rl_debugger"
