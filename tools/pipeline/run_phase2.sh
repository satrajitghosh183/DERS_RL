#!/usr/bin/env bash
# Phase 2 (after the two RL jobs): rigorous eval of all checkpoints (GPU 0) + verified shader
# dataset/gallery generation (GPU 1) in parallel, then upload everything to Box.
set -uo pipefail
say(){ echo "[phase2 $(date +%H:%M)] $*"; }
M7=~/models/qwen2.5-coder-7b; M32=~/models/qwen2.5-coder-32b
SP=~/shader_pipeline; P=~/pipeline; D=~/shader_data; RB=~/NERC/build/omni_reward; RD=~/NERC/build/omni_render
F="--box-root-folder-id 394771887705"

say "waiting for RL jobs (task3 7B, task4 32B) to finish…"
while tmux has-session -t task3 2>/dev/null || tmux has-session -t task4 2>/dev/null; do sleep 60; done
say "RL done. v3=$([ -d $P/dora7b_v3_ood ] && echo yes) 32b_rl=$([ -d $P/dora32b_rl ] && echo yes)"

# held-out eval set: 60 zero-leak NL + 40 free-form
python3 - <<PY
import json
clean=[l for l in open("$D/heldout_clean.jsonl")][:40]
ff=[l for l in open("$D/heldout_nl.jsonl")][:40]
open("$D/heldout_eval80.jsonl","w").writelines(clean+ff)
print("eval set:", len(clean)+len(ff))
PY

# ---- GPU 1: dataset/gallery with the proven v1 (background) ----
say "GPU1: dataset gen (v1 deliver@400, best-of-6, 500 prompts)"
head -500 $D/rl_diverse.jsonl > $D/dataset_prompts.jsonl
( CUDA_VISIBLE_DEVICES=1 python3 $SP/gen_dataset.py --base $M7 --adapter $P/dora7b_deliver_400 \
    --prompts $D/dataset_prompts.jsonl --reward-bin $RB --render-bin $RD --k 6 --out $P/shader_dataset \
    > $P/dataset.log 2>&1 ) &
DSPID=$!

# ---- GPU 0: rigorous eval of the key checkpoints (sequential) ----
evalm(){ say "eval $1"; CUDA_VISIBLE_DEVICES=0 python3 $SP/eval_rigorous.py --base "$2" ${3:+--adapter "$3"} \
    --prompts $D/heldout_eval80.jsonl --reward-bin $RB --k 8 --label "$1" --out $P/rig_$1.json 2>&1 | tail -3; }
evalm v1_deliver400 $M7 $P/dora7b_deliver_400
[ -d $P/dora7b_v3_ood ] && evalm v3_ood $M7 $P/dora7b_v3_ood
[ -d $P/dora32b_rl ]   && evalm s32b_rl $M32 $P/dora32b_rl
evalm base7b $M7

wait $DSPID; say "dataset gen finished"

# ---- combined table ----
say "=== RIGOROUS EVAL (compile@k, best-of-N, on 80 held-out) ==="
python3 - <<PY
import json,glob,os
for f in sorted(glob.glob("$P/rig_*.json")):
    d=json.load(open(f))["summary"]; ck=d["compile@k"]
    print(f"{d['label']:16s} c@1={ck['1']} c@8={ck['8']} run@1={d['run@1']} render@1={d['render@1']} bestN_usable@1={d['bestN_usable@1']} meanR={d['mean_bestK_reward']}")
print("frontier(Claude) c@1=1.0 (8 free-form prompts, see baseline_frontier.json)")
PY

# ---- upload to Box ----
say "uploading results + dataset to Box"
for f in $P/rig_*.json; do rclone copy "$f" box:nerc_server/results $F 2>/dev/null; done
tar cf - -C $P shader_dataset | rclone rcat box:nerc_server/results/shader_dataset.tar $F 2>&1 | tail -1
rclone copy $D/heldout_eval80.jsonl box:nerc_server/results $F 2>/dev/null
say "=== PHASE2_DONE ==="
