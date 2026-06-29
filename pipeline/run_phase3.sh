#!/usr/bin/env bash
set -uo pipefail
say(){ echo "[phase3 $(date +%H:%M)] $*"; }
M7=~/models/qwen2.5-coder-7b; M32=~/models/qwen2.5-coder-32b; SP=~/shader_pipeline; P=~/pipeline; D=~/shader_data
RB=~/NERC/build/omni_reward; F="--box-root-folder-id 394771887705"
say "waiting for phase2 to finish (frees both GPUs)…"
while tmux has-session -t phase2 2>/dev/null; do sleep 60; done
say "phase2 done."
# SCALE point: rigorous eval of the existing 32B SFT
say "eval 32B SFT (scale data point)"
python3 $SP/eval_rigorous.py --base $M32 --adapter $P/dora32b_out/adapter --prompts $D/heldout_eval80.jsonl \
  --reward-bin $RB --k 8 --label s32b_sft --out $P/rig_s32b_sft.json 2>&1 | tail -3
# 7B-OOD-RL on BOTH GPUs (the working v2 config), 100 steps -> v3
say "7B OOD-RL (both GPUs, debugger+CLIP+KL, 100 steps)"
PYTHONPATH=$SP python3 $SP/rl_refine.py --prompts $D/rl_diverse.jsonl --base $M7 --adapter $P/dora7b_deliver_400 \
  --omni-reward-bin $RB --reward-mode debugger --clip --w-clip 1.5 --kl 0.05 --group 8 --steps 100 --log-every 1 \
  --out $P/dora7b_v3_ood 2>&1 | tee $P/v3_rl.log | tail -2
say "eval v3"
[ -d $P/dora7b_v3_ood ] && python3 $SP/eval_rigorous.py --base $M7 --adapter $P/dora7b_v3_ood --prompts $D/heldout_eval80.jsonl \
  --reward-bin $RB --k 8 --label v3_ood --out $P/rig_v3.json 2>&1 | tail -3
say "=== FULL EVAL TABLE ==="
python3 - <<PY
import json,glob,os
for f in sorted(glob.glob("$P/rig_*.json")):
    d=json.load(open(f))["summary"]; ck=d["compile@k"]
    print(f"{d['label']:14s} c@1={ck['1']} c@8={ck['8']} run@1={d['run@1']} render@1={d['render@1']} bestN={d['bestN_usable@1']} meanR={d['mean_bestK_reward']}")
PY
say "upload phase3 results + v3 adapter"
for f in $P/rig_*.json; do rclone copy "$f" box:nerc_server/results $F 2>/dev/null; done
[ -d $P/dora7b_v3_ood ] && rclone copy $P/dora7b_v3_ood box:nerc_server/adapters/dora7b_v3_ood $F 2>/dev/null
say "=== PHASE3_DONE ==="
