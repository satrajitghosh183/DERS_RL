#!/usr/bin/env bash
set -e
cd ~/shader_data
echo "[focused] building Shadertoy-only corpus (no The Stack)..."
python3 ~/shader_pipeline/build_corpus.py --vipitis vipitis-ds \
  --mizu mizuamedesu/dataset_v2_train.jsonl --mizu-eval mizuamedesu/dataset_v2_eval.jsonl \
  --seanmemery seanmemery/data/train-00000-of-00001.parquet --out corpus_focused
cat synth/synth_corpus_full.txt >> corpus_focused/corpus.txt   # + compile-verified synthetic
echo "[focused] tokenizing with the GLSL BPE..."
python3 - <<'PY'
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
import numpy as np
tok=Tokenizer(BPE.from_file("glsl_bpe16k/vocab.json","glsl_bpe16k/merges.txt")); tok.pre_tokenizer=ByteLevel(add_prefix_space=False)
text=open("corpus_focused/corpus.txt",errors="replace").read()
docs=[d for d in text.split("<|endoftext|>") if d.strip()]
EOT=tok.get_vocab_size(); ids=[]
for e in tok.encode_batch(docs): ids.extend(e.ids); ids.append(EOT)
np.save("corpus_focused/shaders_focused.npy", np.array(ids,dtype=np.uint16))
print(f"[focused] {len(docs)} docs, {len(ids)} tokens, {len(text)/len(ids):.3f} chars/tok")
PY
cd ~/OLMo-shader
sed -e 's#^\(data_path[[:space:]]*\).*#\1/home/exouser/shader_data/corpus_focused/shaders_focused.npy#' \
    -e 's#shader_300M#shader_300M_focused#g' -e 's#^\(steps[[:space:]]*\).*#\16000#' \
    conf/shader_300M.conf > conf/shader_300M_focused.conf
echo "[focused] training 300M single-GPU on focused corpus..."
CUDA_VISIBLE_DEVICES=0 ./build/olmo_train conf/shader_300M_focused.conf
echo "=== FOCUSED_DONE ==="
