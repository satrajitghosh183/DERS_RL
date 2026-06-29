#!/usr/bin/env bash
# Finalize: (1) honest ablation ladder on the ZERO-LEAK held-out, then (2) train the deliverable
# debugger-RL to convergence — continue the 160-step checkpoint to ~400 steps in two +120 chunks,
# clean-evaluating each so we can see the plateau and keep the best checkpoint.
set -uo pipefail
say(){ echo "[finalize $(date +%H:%M)] $*"; }
M=~/models/qwen2.5-coder-7b; P=~/pipeline; SP=~/shader_pipeline; RB=~/NERC/build/omni_reward
CLEAN=~/shader_data/heldout_clean.jsonl; OOD=~/shader_data/heldout_nl.jsonl; RLP=~/shader_data/rl_prompts_nl.jsonl
SFT=$P/dora7b_out/adapter

while ! grep -aq DORA7B_DONE ~/pipeline/stage2_master.log 2>/dev/null; do sleep 60; done
say "main run done; finalizing."

evalc(){ # label out prompts [adapter]
  local a=(--base "$M" --prompts "$3" --reward-bin "$RB" --k 4 --label "$1" --out "$2")
  [ -n "${4:-}" ] && a+=(--adapter "$4")
  python3 "$SP/eval_full.py" "${a[@]}" 2>&1 | tee "$P/clean_$1.log"
}
cont(){ # from_adapter steps out  (continue debugger-RL)
  PYTHONPATH=$SP python3 "$SP/rl_refine.py" --prompts "$RLP" --base "$M" --adapter "$1" \
    --omni-reward-bin "$RB" --reward-mode debugger --group 8 --steps "$2" --log-every 1 \
    --out "$3" 2>&1 | tee "$P/$(basename "$3").log"
}

# ---- (1) honest ablation ladder (matched-compute checkpoints) ----
say "clean-eval ablation rows (zero-leak)"
evalc base7b      "$P/clean_row1.json" "$CLEAN"
evalc dora7b_sft  "$P/clean_row2.json" "$CLEAN" "$SFT"
evalc rl_compile  "$P/clean_row3.json" "$CLEAN" "$P/dora7b_rl_compile"
evalc rl_debugger "$P/clean_row4.json" "$CLEAN" "$P/dora7b_rl_debugger"

# ---- (2) deliverable to convergence ----
say "deliverable: continue debugger-RL 160 -> 280"
cont "$P/dora7b_rl_debugger" 120 "$P/dora7b_deliver_280"
evalc deliver_280 "$P/clean_deliver_280.json" "$CLEAN" "$P/dora7b_deliver_280"
say "deliverable: continue 280 -> 400"
cont "$P/dora7b_deliver_280" 120 "$P/dora7b_deliver_400"
evalc deliver_400     "$P/clean_deliver_400.json"     "$CLEAN" "$P/dora7b_deliver_400"
evalc deliver_400_ood "$P/clean_deliver_400_ood.json" "$OOD"   "$P/dora7b_deliver_400"

# ---- final tables ----
say "=== HONEST ABLATION (zero-leak) + DELIVERABLE PROGRESSION ==="
python3 - "$P" <<'PY'
import json, os, sys
P=sys.argv[1]
rows=[("clean_row1.json","base"),("clean_row2.json","+DoRA SFT"),
      ("clean_row3.json","+RL compile-only(150)"),("clean_row4.json","+RL debugger(160)"),
      ("clean_deliver_280.json","  deliverable@280"),("clean_deliver_400.json","  deliverable@400"),
      ("clean_deliver_400_ood.json","  deliverable@400 (OOD)")]
print(f"{'row':28s} {'comp@1':>6s} {'comp@k':>6s} {'run@1':>6s} {'rend@1':>6s} {'rend@k':>6s} {'meanR':>6s}")
for fn,name in rows:
    p=os.path.join(P,fn)
    if not os.path.exists(p): print(f"{name:28s} (missing)"); continue
    d=json.load(open(p))['summary']
    print(f"{name:28s} {d['compile@1']:6} {d['compile@k']:6} {d['run@1']:6} {d['render@1']:6} {d['render@k']:6} {d['mean_reward']:6}")
PY
say "=== FINALIZE_DONE === deliverable candidates: dora7b_rl_debugger(160) / dora7b_deliver_280 / dora7b_deliver_400"
