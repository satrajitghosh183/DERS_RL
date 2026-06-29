#!/usr/bin/env python3
"""Caption verified-compiling shaders with DIVERSE natural-language descriptions (via the 32B),
to fix the free-form OOD gap: the model currently trains on templated prompts, so it doesn't learn
real NL->shader semantics. We take known-good shaders and ask a strong model to describe each in a
few different natural phrasings -> (rich description, shader) pairs for the next SFT round.

  ~/vllm_env/bin/python caption_shaders.py --model ~/models/qwen2.5-coder-32b \
      --shaders ~/shader_data/distill/verified.txt --n-per 3 --out ~/shader_data/captioned.txt
"""
import argparse, os, re, sys

SHADER_RE = re.compile(r"// Shader:.*?\n(.*?)<\|endoftext\|>", re.S)
SYS = ("You are a graphics expert. Given a GLSL Shadertoy shader, describe what it renders in ONE "
       "vivid, natural sentence a person would type to request it — focus on the visual (colors, "
       "motion, subject), not the code. No preamble, just the description.")

def load_shaders(path, limit):
    txt = open(path, errors="replace").read()
    out = [m.group(1).strip() for m in SHADER_RE.finditer(txt) if len(m.group(1).strip()) > 40]
    return out[:limit] if limit else out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--shaders", required=True)
    ap.add_argument("--n-per", type=int, default=3, help="distinct captions per shader")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--out", default="captioned.txt")
    a = ap.parse_args()

    shaders = load_shaders(a.shaders, a.limit)
    print(f"[caption] {len(shaders)} shaders x {a.n_per} captions", flush=True)
    from vllm import LLM, SamplingParams
    llm = LLM(model=a.model, tensor_parallel_size=a.tp, dtype="bfloat16",
              max_model_len=2048, gpu_memory_utilization=0.90)
    # higher temp across n_per passes -> varied phrasings
    convs, owner = [], []
    for si, code in enumerate(shaders):
        snippet = code[:1400]
        for _ in range(a.n_per):
            convs.append([{"role": "system", "content": SYS},
                          {"role": "user", "content": f"```glsl\n{snippet}\n```"}])
            owner.append(si)
    sp = SamplingParams(temperature=1.0, top_p=0.95, max_tokens=64)
    outs = llm.chat(convs, sp)

    pairs = []
    for o, si in zip(outs, owner):
        lines = o.outputs[0].text.strip().strip('"').splitlines()
        cap = lines[0][:200] if lines else ""
        if len(cap) > 8:
            pairs.append((cap, shaders[si]))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        for cap, code in pairs:
            f.write(f"// Shader: {cap}\n{code.rstrip()}\n<|endoftext|>\n")
    print(f"[caption] wrote {len(pairs)} (caption, shader) pairs -> {a.out}", flush=True)

if __name__ == "__main__":
    main()
