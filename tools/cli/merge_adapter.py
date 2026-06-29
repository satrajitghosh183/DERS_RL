#!/usr/bin/env python3
"""Merge a DoRA adapter into its base model -> a standalone HF model dir, so it can be converted
to MLX 4-bit for local (Apple Silicon, non-H100) deployment.

  python3 merge_adapter.py <base> <adapter> <out_dir>
then on the Mac:
  mlx_lm.convert --hf-path <out_dir> --mlx-path <mlx_dir> -q --q-bits 4   # ~4GB, fast on M-series
"""
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base, adapter, out = sys.argv[1], sys.argv[2], sys.argv[3]
tok = AutoTokenizer.from_pretrained(base)
m = AutoModelForCausalLM.from_pretrained(base, torch_dtype="auto", device_map="cpu")
m = PeftModel.from_pretrained(m, adapter)
print("merging DoRA into base…", flush=True)
m = m.merge_and_unload()
m.save_pretrained(out, safe_serialization=True)
tok.save_pretrained(out)
print(f"merged -> {out}", flush=True)
