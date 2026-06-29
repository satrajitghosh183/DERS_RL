#!/usr/bin/env python3
"""Debugger-in-the-loop reward demo (the SIGGRAPH thesis, with the real LM).

The LM generates a shader; we VALIDATE it by compiling with glslangValidator
(the compile reward term, shader_cmake-style); and we score it. With OmniTrace,
this reward is enriched by execution traces (divergence / NaN / numerical /
visual). Here we demonstrate the loop end-to-end with the compile signal.
"""
import subprocess, re, os, sys, tempfile

BASE  = os.path.expanduser("~/OLMo-shader")
CKPT  = sys.argv[1] if len(sys.argv) > 1 else BASE + "/runs/shader_v2/model.pt"
CFG   = BASE + "/configs/shader_125M_full.json"
VOCAB = BASE + "/data/gpt2/vocab.json"; MERGES = BASE + "/data/gpt2/merges.txt"
CHAT  = BASE + "/build/chat"
GLSLANG = "/usr/bin/glslangValidator"

PROMPTS = ["// Shader: blue fire", "// Shader: ocean waves", "// Shader: red plasma",
           "// Shader: starfield sky", "// Shader: rainbow gradient", "// Shader: noise clouds",
           "// Shader: rotating tunnel", "// Shader: water ripples"]

HARNESS = """#version 450
layout(location=0) out vec4 _OutColor;
layout(push_constant) uniform U { vec3 iResolution; float iTime; vec4 iMouse; int iFrame; } u;
#define iResolution u.iResolution
#define iTime u.iTime
#define iMouse u.iMouse
#define iFrame u.iFrame
"""

def generate(prompt, max_tokens=220):
    try:
        p = subprocess.run([CHAT,"--checkpoint",CKPT,"--config",CFG,"--vocab-file",VOCAB,
            "--merges-file",MERGES,"--device","cuda","--max-tokens",str(max_tokens),
            "--no-speculative","--top-k","40","--top-p","0.92","--temperature","0.8",
            "--repetition-penalty","1.3"], input=prompt+"\n",
            capture_output=True, text=True, timeout=150)
    except subprocess.TimeoutExpired:
        return ""
    m = re.search(r"Model:(.*?)(\n\[[0-9]+ tokens|\nYou:|$)", p.stdout, re.S)
    return (m.group(1) if m else "").strip()

def wrap(code):
    if "mainImage" in code:
        return HARNESS + code + "\nvoid main(){ vec4 c=vec4(0.0); mainImage(c, gl_FragCoord.xy); _OutColor=c; }\n"
    return (HARNESS + "void main(){ vec2 fragCoord=gl_FragCoord.xy; vec2 uv=fragCoord/iResolution.xy; vec3 col=vec3(0.0);\n"
            + code + "\n_OutColor=vec4(col,1.0); }\n")

def validate(glsl):
    f = tempfile.NamedTemporaryFile(suffix=".frag", delete=False, mode="w"); f.write(glsl); f.close()
    try:
        r = subprocess.run([GLSLANG,"-V",f.name,"-o","/dev/null"], capture_output=True, text=True, timeout=20)
        return r.returncode == 0, (r.stdout + r.stderr)
    finally:
        os.unlink(f.name)

def is_meaningful(code):
    c = code.strip()
    # reject empty/trivial output so we don't credit a shader for an empty harness
    return len(c) >= 15 and any(t in c for t in (";", "vec", "float", "=", "{"))

def main():
    print(f"== Debugger-in-the-loop reward | model: {os.path.basename(os.path.dirname(CKPT))} ==")
    compiled = 0; total_reward = 0.0
    for prompt in PROMPTS:
        code = generate(prompt)
        if not is_meaningful(code):
            ok = False; log = "(empty/trivial generation rejected)"
        else:
            ok, log = validate(wrap(code))
        # compile term + a small length bonus; OmniTrace would add exec/divergence/visual terms
        reward = (1.0 if ok else 0.0) + min(len(code)/500.0, 0.3)
        compiled += int(ok); total_reward += reward
        first = (code.splitlines() or [""])[0][:64]
        print(f"  {prompt:30s} compiles={int(ok)} reward={reward:.2f}  gen={first!r}")
    n = len(PROMPTS)
    print(f"\ncompile@1 = {compiled}/{n} = {compiled/n:.2f}   mean_reward = {total_reward/n:.3f}")
    print("(compile is the shader_cmake-style first term; OmniTrace enriches it with execution traces)")

if __name__ == "__main__":
    main()
