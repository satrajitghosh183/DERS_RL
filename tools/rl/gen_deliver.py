#!/usr/bin/env python3
"""Box-side generation endpoint for the DELIVERABLE model (Qwen2.5-Coder-7B + debugger-RL
deliver@400). Two modes, both used by `omnishader` for debugger-in-the-loop *inference*:

  gen_deliver.py "<prompt>" --n 8      -> print N candidate shaders (separated by a delimiter)
  gen_deliver.py --repair  (stdin JSON {prompt, code, error}) -> print one repaired shader

Best-of-N + repair lifts usable success from compile@1 toward compile@k with no retraining.
"""
import sys, os, json, argparse
BASE    = os.environ.get("OMNI_BASE",    os.path.expanduser("~/models/qwen2.5-coder-7b"))
ADAPTER = os.environ.get("OMNI_ADAPTER", os.path.expanduser("~/pipeline/dora7b_deliver_400"))
SEP = "@@@OMNI_CANDIDATE@@@"

def extract(t):
    if "```" in t:
        b = t.split("```", 2); body = b[1] if len(b) > 1 else t
        for lg in ("glsl", "c", "cpp", "c++"):
            if body.startswith(lg): body = body[len(lg):]; break
        return body.strip()
    return t.strip()

_M = {}
def load(adapter):
    if "m" in _M: return _M["tok"], _M["m"]
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    tok = AutoTokenizer.from_pretrained(BASE)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map="auto")
    m = PeftModel.from_pretrained(m, adapter); m.eval()
    _M.update(tok=tok, m=m); return tok, m

def gen(tok, m, prompt, n, max_new=384, temp=0.8):
    import torch
    ids = tok(prompt, return_tensors="pt").to(m.device)
    with torch.no_grad():
        g = m.generate(**ids, max_new_tokens=max_new, do_sample=True, temperature=temp, top_p=0.9,
                       repetition_penalty=1.1, num_return_sequences=n, pad_token_id=tok.eos_token_id)
    return [extract(tok.decode(s[ids.input_ids.shape[1]:], skip_special_tokens=True)) for s in g]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt", nargs="?", default="abstract colorful pattern")
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--repair", action="store_true")
    ap.add_argument("--adapter", default=ADAPTER)
    a = ap.parse_args()
    tok, m = load(a.adapter)
    if a.repair:
        req = json.load(sys.stdin)
        p = (f"// Fix this GLSL shader so it compiles and runs. Error:\n// {req.get('error','')[:200]}\n"
             f"// Shader: {req.get('prompt','')}\n{req.get('code','')}\n// Corrected:\n")
        print(gen(tok, m, p, 1, temp=0.6)[0])
    else:
        cands = gen(tok, m, f"// Shader: {a.prompt}\n", a.n)
        print(("\n" + SEP + "\n").join(cands))

if __name__ == "__main__":
    main()
