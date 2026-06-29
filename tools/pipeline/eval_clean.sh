#!/usr/bin/env bash
# Honest re-evaluation: after the run finishes, score every saved checkpoint on a ZERO-LEAK
# held-out prompt set (NL combos absent from the SFT corpus) + a free-form OOD set. The leaked
# heldout_mk numbers from the main run are a memorization ceiling; THESE are the real ones.
set -uo pipefail
say(){ echo "[cleaneval $(date +%H:%M)] $*"; }
M=~/models/qwen2.5-coder-7b; P=~/pipeline; SP=~/shader_pipeline; RB=~/NERC/build/omni_reward
CLEAN=~/shader_data/heldout_clean.jsonl     # zero-leak generalization
OOD=~/shader_data/heldout_nl.jsonl          # free-form OOD
SFT=$P/dora7b_out/adapter

# wait for the main run to finish saving every adapter
while ! grep -aq DORA7B_DONE ~/pipeline/stage2_master.log 2>/dev/null; do sleep 60; done
say "main run done; re-evaluating on zero-leak held-out"

evalc(){ # <label> <out> <prompts> [adapter]
  local a=(--base "$M" --prompts "$3" --reward-bin "$RB" --k 4 --label "$1" --out "$2")
  [ -n "${4:-}" ] && a+=(--adapter "$4")
  python3 "$SP/eval_full.py" "${a[@]}" 2>&1 | tee "$P/clean_$1.log"
}
evalc base7b      "$P/clean_row1.json" "$CLEAN"
evalc dora7b_sft  "$P/clean_row2.json" "$CLEAN" "$SFT"
evalc rl_compile  "$P/clean_row3.json" "$CLEAN" "$P/dora7b_rl_compile"
evalc rl_debugger "$P/clean_row4.json" "$CLEAN" "$P/dora7b_rl_debugger"
evalc rl_debug_ood "$P/clean_row4_ood.json" "$OOD" "$P/dora7b_rl_debugger"

say "=== HONEST ABLATION LADDER (zero-leak held-out NL prompts) ==="
python3 - "$P" <<'PY'
import json, sys, os
P = sys.argv[1]
order = [("clean_row1.json","base"),("clean_row2.json","+DoRA SFT"),
         ("clean_row3.json","+RL compile-only"),("clean_row4.json","+RL full-debugger"),
         ("clean_row4_ood.json","  (row4, free-form OOD)")]
print(f"{'row':24s} {'comp@1':>6s} {'comp@k':>6s} {'run@1':>6s} {'rend@1':>6s} {'rend@k':>6s} {'meanR':>6s}")
for fn, name in order:
    p = os.path.join(P, fn)
    if not os.path.exists(p): print(f"{name:24s}  (missing)"); continue
    d = json.load(open(p))["summary"]
    print(f"{name:24s} {d['compile@1']:6} {d['compile@k']:6} {d['run@1']:6} {d['render@1']:6} {d['render@k']:6} {d['mean_reward']:6}")
PY
say "=== CLEANEVAL_DONE ==="
