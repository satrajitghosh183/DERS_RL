#!/usr/bin/env bash
# Retrain the deliverable with everything from this round folded in:
#   data:   original corpus + CAPTIONED natural-language pairs (semantic grounding -> better OOD)
#   reward: omni_reward (compile+run+COLOR render) + CLIP semantic match + KL anti-collapse
# Produces deliver_v2 — more colorful, better prompt adherence, better free-form generalization.
# Still a 7B (runs on local non-H100 after MLX quant); only the TRAINING uses the box.
set -uo pipefail
say(){ echo "[retrain $(date +%H:%M)] $*"; }
M=~/models/qwen2.5-coder-7b; D=~/shader_data; P=~/pipeline; SP=~/shader_pipeline; RB=~/NERC/build/omni_reward
CLEAN=$D/heldout_clean.jsonl; OOD=$D/heldout_nl.jsonl; RLP=$D/rl_prompts_nl.jsonl

# NO SFT redo: reuse v1's known-good SFT adapter (dora7b_out, ~90% compile@1). Captions are dropped
# (prose broke the code-generation format). The win this round is the upgraded REWARD applied via RL.
SFT=$P/dora7b_out/adapter
say "STAGE 1: reuse v1 SFT adapter $SFT (no SFT redo, no captions)"
[ -d "$SFT" ] || { say "FATAL: v1 SFT adapter missing"; exit 1; }

# STAGE 2: RL with the upgraded reward (color + CLIP semantic + KL anti-collapse)
say "STAGE 2: RL — debugger(color) + CLIP semantic + KL, 220 steps"
PYTHONPATH=$SP python3 $SP/rl_refine.py --prompts $RLP --base $M --adapter $SFT \
  --omni-reward-bin $RB --reward-mode debugger --clip --w-clip 1.5 --kl 0.05 \
  --group 8 --steps 220 --log-every 1 --out $P/dora7b_v2_deliver 2>&1 | tee $P/dora7b_v2_rl.log

# eval on zero-leak held-out + free-form OOD
say "STAGE 3: eval v2"
python3 $SP/eval_full.py --base $M --adapter $P/dora7b_v2_deliver --prompts $CLEAN --reward-bin $RB --k 4 --label v2_clean --out $P/clean_v2.json 2>&1 | tail -8
python3 $SP/eval_full.py --base $M --adapter $P/dora7b_v2_deliver --prompts $OOD   --reward-bin $RB --k 4 --label v2_ood   --out $P/clean_v2_ood.json 2>&1 | tail -8

say "=== v2 vs v1 (zero-leak held-out) ==="
python3 - <<'PY'
import json, os
P="/home/exouser/pipeline"
for fn,name in [("clean_row4.json","v1 deliver@160"),("clean_deliver_400.json","v1 deliver@400"),
                ("clean_v2.json","v2 (color+CLIP+KL)"),("clean_v2_ood.json","v2 free-form OOD")]:
    p=os.path.join(P,fn)
    if not os.path.exists(p): print(f"{name:22s} (missing)"); continue
    d=json.load(open(p))["summary"]
    print(f"{name:22s} compile@1={d['compile@1']} run@1={d['run@1']} render@1={d['render@1']} meanR={d['mean_reward']}")
PY
say "=== RETRAIN_DONE === deliverable v2: $P/dora7b_v2_deliver"
