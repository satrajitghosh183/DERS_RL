#!/usr/bin/env bash
# Distillation: 32B generates -> compile-filter + repair -> clean GLSL corpus -> train small model.
set -uo pipefail
D=~/shader_data/distill; mkdir -p "$D"; cd "$D"
export VLLM_USE_FLASHINFER_SAMPLER=0 PATH=$HOME/vllm_env/bin:$PATH
GB=~/shader_data/glsl_bpe16k
say(){ echo "[distill $(date +%H:%M)] $*"; }

say "STAGE 1: generate ~144k shaders with Qwen2.5-Coder-32B (9000 prompts x 16)"
~/vllm_env/bin/python ~/shader_pipeline/gen_shaders.py --model ~/models/qwen2.5-coder-32b \
  --n-prompts 9000 --k 16 --temp 0.95 --out gen.jsonl
say "generated $(wc -l < gen.jsonl) candidates"

say "STAGE 2: compile-label (keep only compiling GLSL)"
python3 ~/shader_pipeline/label_shaders.py --in gen.jsonl --labeled labeled.jsonl \
  --corpus-out compiling.txt --workers 32

say "STAGE 3: repair broken shaders -> recover more compiling"
~/vllm_env/bin/python ~/shader_pipeline/repair_loop.py --in labeled.jsonl \
  --model ~/models/qwen2.5-coder-32b --out repaired.jsonl --workers 32
python3 - <<'PY'
import json
f=open("compiling.txt","a")
n=0
for l in open("repaired.jsonl"):
    if not l.strip(): continue
    r=json.loads(l)
    if r.get("fixed_compiles"):
        f.write("// Shader: "+(r.get("prompt","untitled") or "untitled")[:80]+"\n"+r["fixed"].rstrip()+"\n<|endoftext|>\n"); n+=1
print(f"[distill] appended {n} repaired shaders")
PY
say "clean corpus: $(grep -c endoftext compiling.txt) compiling shaders"

say "STAGE 4: tokenize the clean distilled corpus with the GLSL BPE"
python3 - <<PY
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
import numpy as np
tok=Tokenizer(BPE.from_file("$GB/vocab.json","$GB/merges.txt")); tok.pre_tokenizer=ByteLevel(add_prefix_space=False)
text=open("compiling.txt",errors="replace").read()
docs=[d for d in text.split("<|endoftext|>") if d.strip()]
EOT=tok.get_vocab_size(); ids=[]
for e in tok.encode_batch(docs): ids.extend(e.ids); ids.append(EOT)
np.save("$D/distill.npy", np.array(ids,dtype=np.uint16))
print(f"[distill] {len(docs)} clean shaders, {len(ids)} tokens")
PY

say "STAGE 5: train 300M on the clean distilled corpus (single-GPU, no save race)"
cd ~/OLMo-shader
sed -e 's#^\(data_path[[:space:]]*\).*#\1'"$D"'/distill.npy#' \
    -e 's#shader_300M#shader_distill#g' -e 's#^\(steps[[:space:]]*\).*#\112000#' \
    conf/shader_300M.conf > conf/shader_distill.conf
CUDA_VISIBLE_DEVICES=0 ./build/olmo_train conf/shader_distill.conf
echo "=== DISTILL_DONE $(date) ==="
