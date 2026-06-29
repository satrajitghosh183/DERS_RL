#!/usr/bin/env python3
"""Generate one shader per prompt from a checkpoint and DUMP THE CODE — the definitive
prompt-faithfulness / mode-collapse check (does distinct input -> distinct output?).
Writes each shader to <outdir>/<NN>_<slug>.frag and prints a code-similarity summary."""
import argparse, os, re, sys, hashlib

HARNESS = ("#version 450\nlayout(location=0) out vec4 _O;\n"
  "layout(push_constant) uniform U { vec3 iResolution; float iTime; vec4 iMouse; int iFrame; } u;\n"
  "#define iResolution u.iResolution\n#define iTime u.iTime\n#define iMouse u.iMouse\n#define iFrame u.iFrame\n")
def wrap(c):
    if "mainImage" in c:
        return HARNESS + c + "\nvoid main(){ vec4 c=vec4(0.); mainImage(c, gl_FragCoord.xy); _O=c; }\n"
    return HARNESS + ("void main(){ vec2 fragCoord=gl_FragCoord.xy; vec2 uv=fragCoord/iResolution.xy;"
                      " vec3 col=vec3(0.);\n" + c + "\n_O=vec4(col,1.); }\n")
def extract(t):
    if "```" in t:
        b=t.split("```",2); body=b[1] if len(b)>1 else t
        for lg in ("glsl","c","cpp","c++"):
            if body.startswith(lg): body=body[len(lg):]; break
        return body.strip()
    return t.strip()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--base", required=True); ap.add_argument("--adapter", default=None)
    ap.add_argument("--out", default="samples"); ap.add_argument("--max-new", type=int, default=384)
    a=ap.parse_args()
    prompts=["ocean waves at sunset","burning campfire flames","swirling galaxy spiral",
             "psychedelic kaleidoscope","matrix code rain","rainbow gradient sky"]
    os.makedirs(a.out, exist_ok=True)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok=AutoTokenizer.from_pretrained(a.base)
    if tok.pad_token is None: tok.pad_token=tok.eos_token
    m=AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16, device_map="auto")
    if a.adapter:
        from peft import PeftModel; m=PeftModel.from_pretrained(m, a.adapter)
    m.eval()
    bodies=[]
    for i,p in enumerate(prompts):
        ids=tok(f"// Shader: {p}\n", return_tensors="pt").to(m.device)
        with torch.no_grad():
            g=m.generate(**ids, max_new_tokens=a.max_new, do_sample=True, temperature=0.7,
                         top_p=0.9, repetition_penalty=1.1, pad_token_id=tok.eos_token_id)
        code=extract(tok.decode(g[0][ids.input_ids.shape[1]:], skip_special_tokens=True))
        slug=re.sub(r"[^a-z]+","_",p.lower())[:20]
        open(f"{a.out}/{i:02d}_{slug}.frag","w").write(wrap(code))
        open(f"{a.out}/{i:02d}_{slug}.glsl","w").write(code)   # raw model output, for omni_render
        bodies.append(code)
        print(f"\n===== [{i}] {p} =====  ({len(code)} chars, sha={hashlib.md5(code.encode()).hexdigest()[:8]})")
        print(code[:500])
    # collapse check: pairwise identical?
    hs=[hashlib.md5(b.encode()).hexdigest() for b in bodies]
    print(f"\n=== distinct shaders: {len(set(hs))}/{len(hs)} (== prompt count means NOT collapsed) ===")

if __name__=="__main__": main()
