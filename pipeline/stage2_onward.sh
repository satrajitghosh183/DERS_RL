#!/usr/bin/env bash
# Resume the run from STAGE 2 — reuse the already-trained DoRA SFT adapter (no SFT redo) and run
# the RL ablation + evals on NATURAL-LANGUAGE prompts (in-distribution with the synthetic SFT data
# and with real usage), so the deliverable is usable from a description.
#
#   reuse SFT adapter ->  +RL(compile-only)   [ablation row 3]
#                     \->  +RL(full debugger)  [deliverable, row 4]
# Eval all 4 rows on held-out NL prompts (primary); row 4 also on free-form prompts (honest OOD).
set -uo pipefail
say(){ echo "[stage2 $(date +%H:%M)] $*"; }
M=~/models/qwen2.5-coder-7b; P=~/pipeline; SP=~/shader_pipeline
RB=~/NERC/build/omni_reward
SFT=$P/dora7b_out/adapter
HELD=~/shader_data/heldout_mk.jsonl        # primary: held-out NL (make_prompts style)
OOD=~/shader_data/heldout_nl.jsonl         # secondary: free-form, honest OOD
RLP=~/shader_data/rl_prompts_nl.jsonl      # 200 NL prompts, disjoint from HELD

[ -d "$SFT" ] || { say "FATAL: SFT adapter missing at $SFT"; exit 1; }

evalrow(){ # <label> <out.json> <prompts> [adapter]
  local label="$1" out="$2" prompts="$3" adapter="${4:-}"
  local args=(--base "$M" --prompts "$prompts" --reward-bin "$RB" --k 4 --label "$label" --out "$out")
  [ -n "$adapter" ] && args+=(--adapter "$adapter")
  python3 "$SP/eval_full.py" "${args[@]}" 2>&1 | tee "$P/elog_${label}.log"
}

# ---- ROW 1 & 2 re-eval on the NL held-out set (consistent max_new=384) ----
say "ROW1: base 7B on NL held-out"
evalrow base7b "$P/eval_row1.json" "$HELD"
say "ROW2: +DoRA SFT on NL held-out"
evalrow dora7b_sft "$P/eval_row2.json" "$HELD" "$SFT"

# ---- ROW 3 (ABLATION): RL compile-only, NL prompts, from SFT adapter ----
say "STAGE 2a: RL compile-only (ablation) — 150 steps, NL prompts"
PYTHONPATH=$SP python3 "$SP/rl_refine.py" \
  --prompts "$RLP" --base "$M" --adapter "$SFT" \
  --omni-reward-bin "$RB" --reward-mode compile --group 8 --steps 150 --log-every 1 \
  --out "$P/dora7b_rl_compile" 2>&1 | tee "$P/dora7b_rl_compile.log"
say "ROW3: +RL(compile-only) on NL held-out"
evalrow rl_compile "$P/eval_row3.json" "$HELD" "$P/dora7b_rl_compile"

# ---- ROW 4 (DELIVERABLE): RL full debugger, NL prompts, from SFT adapter ----
say "STAGE 2b: RL full-debugger (deliverable) — 160 steps, NL prompts"
PYTHONPATH=$SP python3 "$SP/rl_refine.py" \
  --prompts "$RLP" --base "$M" --adapter "$SFT" \
  --omni-reward-bin "$RB" --reward-mode debugger --group 8 --steps 160 --log-every 1 \
  --out "$P/dora7b_rl_debugger" 2>&1 | tee "$P/dora7b_rl_debugger.log"
say "ROW4: +RL(full debugger) on NL held-out"
evalrow rl_debugger "$P/eval_row4.json" "$HELD" "$P/dora7b_rl_debugger"
say "ROW4-OOD: deliverable on free-form prompts (honest OOD check)"
evalrow rl_debugger_ood "$P/eval_row4_ood.json" "$OOD" "$P/dora7b_rl_debugger"

# ---- ablation ladder ----
say "=== ABLATION LADDER (held-out NL prompts, rich reward) ==="
python3 - "$P" <<'PY'
import json, sys, os
P = sys.argv[1]
order = [("eval_row1.json","base"),("eval_row2.json","+DoRA SFT"),
         ("eval_row3.json","+RL compile-only"),("eval_row4.json","+RL full-debugger"),
         ("eval_row4_ood.json","  (row4, free-form OOD)")]
print(f"{'row':24s} {'comp@1':>6s} {'comp@k':>6s} {'run@1':>6s} {'rend@1':>6s} {'rend@k':>6s} {'meanR':>6s}")
for fn, name in order:
    p = os.path.join(P, fn)
    if not os.path.exists(p): print(f"{name:24s}  (missing)"); continue
    d = json.load(open(p))["summary"]
    print(f"{name:24s} {d['compile@1']:6} {d['compile@k']:6} {d['run@1']:6} {d['render@1']:6} {d['render@k']:6} {d['mean_reward']:6}")
PY
say "=== DORA7B_DONE === deliverable adapter: $P/dora7b_rl_debugger"
