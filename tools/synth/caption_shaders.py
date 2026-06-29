#!/usr/bin/env python3
"""Caption verified-compiling shaders with DIVERSE natural-language descriptions, to fix the
free-form OOD gap. Qwen2.5-Coder is a COMPLETION model (chat-style "describe this" makes it
continue code), so we use FEW-SHOT completion: show it shader->description examples and let it
continue the pattern. Produces (description, shader) pairs for the next SFT round.

  ~/vllm_env/bin/python caption_shaders.py --model ~/models/qwen2.5-coder-32b \
      --shaders ~/shader_data/caption_input.txt --n-per 2 --limit 8000 --out ~/shader_data/captioned.txt
"""
import argparse, os, re

SHADER_RE = re.compile(r"// Shader:.*?\n(.*?)<\|endoftext\|>", re.S)

# hand-written few-shot exemplars (code -> vivid one-line description)
SHOTS = [
    ("void mainImage(out vec4 o, in vec2 u){ vec2 p=u/iResolution.xy; o=vec4(0.5+0.5*cos(iTime+p.xyx+vec3(0,2,4)),1.); }",
     "a smooth rainbow gradient that slowly shifts colors across the screen"),
    ("void mainImage(out vec4 o, in vec2 u){ vec2 p=(u-0.5*iResolution.xy)/iResolution.y; float a=atan(p.y,p.x); float r=length(p); o=vec4(vec3(0.5+0.5*sin(a*6.0+iTime*2.0-r*10.0)),1.); }",
     "a hypnotic spinning spiral of bright bands radiating from the center"),
    ("void mainImage(out vec4 o, in vec2 u){ vec2 p=u/iResolution.xy; float n=fract(sin(dot(floor(p*40.0),vec2(12.9,78.2)))*43758.5); o=vec4(vec3(step(0.98,n)),1.); }",
     "a dark night sky scattered with tiny twinkling white stars"),
]

def build_prompt(code):
    s = "Below are GLSL shaders, each with a one-line description of what it looks like on screen.\n\n"
    for c, d in SHOTS:
        s += f"Shader:\n{c}\nDescription: {d}\n\n"
    return s + f"Shader:\n{code[:1200]}\nDescription:"

def load_shaders(path, limit):
    txt = open(path, errors="replace").read()
    out = [m.group(1).strip() for m in SHADER_RE.finditer(txt) if len(m.group(1).strip()) > 40]
    return out[:limit] if limit else out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--shaders", required=True)
    ap.add_argument("--n-per", type=int, default=2); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tp", type=int, default=2); ap.add_argument("--out", default="captioned.txt")
    a = ap.parse_args()
    shaders = load_shaders(a.shaders, a.limit)
    print(f"[caption] {len(shaders)} shaders x {a.n_per} (few-shot completion)", flush=True)
    from vllm import LLM, SamplingParams
    llm = LLM(model=a.model, tensor_parallel_size=a.tp, dtype="bfloat16",
              max_model_len=2048, gpu_memory_utilization=0.90)
    sp = SamplingParams(temperature=0.9, top_p=0.95, max_tokens=40, stop=["\n", "Shader:"])
    prompts, owner = [], []
    for si, code in enumerate(shaders):
        for _ in range(a.n_per):
            prompts.append(build_prompt(code)); owner.append(si)
    outs = llm.generate(prompts, sp)

    bad = ("```", "void ", "#include", "#version", "require(", "import ", "#!/", "<|", "vec2", "vec3", "float ", "gl_")
    pairs = []
    for o, si in zip(outs, owner):
        cap = o.outputs[0].text.strip().strip('"').strip()
        cap = cap.split("\n")[0][:160]
        if len(cap) > 12 and not any(b in cap for b in bad) and cap[0].isalpha():
            pairs.append((cap, shaders[si]))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        for cap, code in pairs:
            f.write(f"// Shader: {cap}\n{code.rstrip()}\n<|endoftext|>\n")
    print(f"[caption] wrote {len(pairs)} clean (description, shader) pairs -> {a.out}", flush=True)

if __name__ == "__main__":
    main()
