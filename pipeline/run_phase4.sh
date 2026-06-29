#!/bin/bash
set -uo pipefail
say(){ echo "[phase4 $(date +%H:%M)] $*"; }
M7=~/models/qwen2.5-coder-7b; SP=~/shader_pipeline; P=~/pipeline; D=~/shader_data
RB=~/NERC/build/omni_reward; RD=~/NERC/build/omni_render; F="--box-root-folder-id 394771887705"
say "waiting for 32B scale eval (rig_s32b_sft.json)…"
while [ ! -f $P/rig_s32b_sft.json ]; do sleep 30; done
rclone copy $P/rig_s32b_sft.json box:nerc_server/results $F 2>/dev/null; say "32B eval uploaded."
say "stopping phase3 (skip v3 RL)"; tmux kill-session -t phase3 2>/dev/null; pkill -f rl_refine.py 2>/dev/null; sleep 4
# 600 fresh diverse prompts, split for the 2 GPUs
python3 - <<PY
import json,random,sys; sys.path.insert(0,"$SP")
from gen_shaders import make_prompts
held=set(json.loads(l)["prompt"] for l in open("$D/heldout_clean.jsonl"))
pool=[p for p in make_prompts(1000,321) if p not in held]; random.Random(11).shuffle(pool); pr=pool[:600]
open("$D/big_s0.jsonl","w").write("\n".join(json.dumps({"prompt":p}) for p in pr[:300]))
open("$D/big_s1.jsonl","w").write("\n".join(json.dumps({"prompt":p}) for p in pr[300:]))
print("600 prompts -> 2 shards")
PY
say "dataset gen on BOTH GPUs (300 each, best-of-6)"
CUDA_VISIBLE_DEVICES=0 python3 $SP/gen_dataset.py --base $M7 --adapter $P/dora7b_deliver_400 --prompts $D/big_s0.jsonl --reward-bin $RB --render-bin $RD --k 6 --out $P/ds_s0 > $P/ds0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python3 $SP/gen_dataset.py --base $M7 --adapter $P/dora7b_deliver_400 --prompts $D/big_s1.jsonl --reward-bin $RB --render-bin $RD --k 6 --out $P/ds_s1 > $P/ds1.log 2>&1 &
wait
say "package release dataset (200 + 600 shards, renders preserved)"
rm -rf $P/dataset_release; mkdir -p $P/dataset_release
cp -r $P/shader_dataset $P/ds_s0 $P/ds_s1 $P/dataset_release/ 2>/dev/null
cat $P/dataset_release/*/dataset.jsonl > $P/dataset_release/all.jsonl 2>/dev/null
n=$(grep -c "\"prompt\"" $P/dataset_release/all.jsonl); say "release dataset: $n shaders -> Box"
tar cf - -C $P dataset_release | rclone rcat box:nerc_server/results/shader_dataset_full.tar $F 2>&1 | tail -1
say "=== PHASE4_DONE ($n shaders) ==="
