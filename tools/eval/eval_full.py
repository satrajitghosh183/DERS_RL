#!/usr/bin/env python3
"""Comprehensive held-out evaluation of a shader generator, using the OmniTrace debugger as the
oracle — not just compile, but runs-finite and renders-non-degenerate.

For each held-out prompt we sample K completions and score every one with `omni_reward`
(compile + GPU-render: finite + luminance-variance). We then report, over the prompt set:

  compile@1 / compile@K   - first sample / any-of-K produces a shader that COMPILES
  run@1     / run@K       - ... that compiles AND runs finite on the GPU (no NaN/Inf)
  render@1  / render@K    - ... that additionally renders a non-degenerate image (variance>thr)
  mean_reward             - average rich reward over all samples

This is the metric ladder behind "fully usable": a perfect model maximizes render@1, not just
compile@1. Same tool produces every row of the base->SFT->RL ablation.

  python3 eval_full.py --base <hf> [--adapter <dir>] --prompts held_out.jsonl \
      --reward-bin ~/NERC/build/omni_reward --k 4 --out eval.json
"""
import argparse, json, os, subprocess, sys, tempfile

HARNESS = ("#version 450\nlayout(location=0) out vec4 _O;\n"
  "layout(push_constant) uniform U { vec3 iResolution; float iTime; vec4 iMouse; int iFrame; } u;\n"
  "#define iResolution u.iResolution\n#define iTime u.iTime\n#define iMouse u.iMouse\n#define iFrame u.iFrame\n")

def wrap(c):
    if "mainImage" in c:
        return HARNESS + c + "\nvoid main(){ vec4 c=vec4(0.); mainImage(c, gl_FragCoord.xy); _O=c; }\n"
    return HARNESS + ("void main(){ vec2 fragCoord=gl_FragCoord.xy; vec2 uv=fragCoord/iResolution.xy;"
                      " vec3 col=vec3(0.);\n" + c + "\n_O=vec4(col,1.); }\n")

def extract(text):
    """Pull GLSL out of a completion: prefer a fenced block, else the raw text."""
    if "```" in text:
        seg = text.split("```", 2)
        body = seg[1] if len(seg) > 1 else text
        for lang in ("glsl", "c", "cpp", "c++"):
            if body.startswith(lang):
                body = body[len(lang):]
                break
        return body.strip()
    return text.strip()

def score(reward_bin, glsl, render_thr=0.001):
    """Return (compiled, executed, rendered, reward) via omni_reward; glslang fallback."""
    try:
        p = subprocess.run([reward_bin], input=glsl, capture_output=True, text=True,
                           errors="replace", timeout=40)
        j = json.loads(p.stdout.strip().splitlines()[-1])
        comp = bool(j.get("compiled"))
        ex   = bool(j.get("executed"))
        vis  = float(j.get("output_variance", 0.0))
        return comp, (comp and ex), (comp and ex and vis > render_thr), float(j.get("reward", 0.0)), vis
    except Exception:
        f = tempfile.NamedTemporaryFile(suffix=".frag", delete=False, mode="w"); f.write(glsl); f.close()
        try:
            ok = subprocess.run(["glslangValidator", "-V", f.name, "-o", os.devnull],
                                capture_output=True, timeout=20).returncode == 0
        finally:
            os.unlink(f.name)
        return ok, False, False, (1.0 if ok else 0.0), 0.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--prompts", required=True, help="jsonl with a 'prompt' field per line")
    ap.add_argument("--reward-bin", default=os.path.expanduser("~/NERC/build/omni_reward"))
    ap.add_argument("--k", type=int, default=4, help="samples per prompt")
    ap.add_argument("--max-new", type=int, default=384)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--render-thr", type=float, default=0.001,
                    help="luminance-variance above which an image counts as non-degenerate (render@)")
    ap.add_argument("--label", default="model")
    ap.add_argument("--out", default="eval_full.json")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16, device_map="auto")
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    prompts = [json.loads(l) for l in open(args.prompts) if l.strip()]
    print(f"== eval {args.label}: {len(prompts)} held-out prompts x k={args.k} ==", flush=True)

    rows, agg = [], {"compile@1":0,"compile@k":0,"run@1":0,"run@k":0,"render@1":0,"render@k":0}
    n_samp, sum_reward = 0, 0.0
    for i, p in enumerate(prompts):
        text_prompt = p["prompt"] if p["prompt"].startswith("//") else f"// Shader: {p['prompt']}"
        ids = tok(text_prompt + "\n", return_tensors="pt").to(model.device)
        with torch.no_grad():
            g = model.generate(**ids, max_new_tokens=args.max_new, do_sample=True,
                               temperature=args.temp, top_p=0.9, repetition_penalty=1.1,
                               num_return_sequences=args.k, pad_token_id=tok.eos_token_id)
        samples = [tok.decode(seq[ids.input_ids.shape[1]:], skip_special_tokens=True) for seq in g]
        scored = [score(args.reward_bin, wrap(extract(s)), args.render_thr) for s in samples]
        comp = [s[0] for s in scored]; ex = [s[1] for s in scored]; rnd = [s[2] for s in scored]
        rew = [s[3] for s in scored]; var = [s[4] for s in scored]
        n_samp += len(rew); sum_reward += sum(rew)
        agg["compile@1"] += int(comp[0]);  agg["compile@k"] += int(any(comp))
        agg["run@1"]     += int(ex[0]);    agg["run@k"]     += int(any(ex))
        agg["render@1"]  += int(rnd[0]);   agg["render@k"]  += int(any(rnd))
        rows.append({"prompt": text_prompt, "compile": comp, "run": ex, "render": rnd,
                     "reward": [round(r,3) for r in rew], "variance": [round(v,5) for v in var]})
        print(f"  [{i+1}/{len(prompts)}] {text_prompt[:42]:42s} "
              f"c={int(any(comp))} r={int(any(ex))} v={int(any(rnd))} maxR={max(rew):.2f}", flush=True)

    N = len(prompts)
    res = {k: round(v / N, 3) for k, v in agg.items()}
    res["mean_reward"] = round(sum_reward / max(1, n_samp), 3)
    res["label"] = args.label; res["n"] = N; res["k"] = args.k
    print(f"\n== {args.label} ==")
    for k in ["compile@1","compile@k","run@1","run@k","render@1","render@k","mean_reward"]:
        print(f"  {k:11s} {res[k]}")
    json.dump({"summary": res, "rows": rows}, open(args.out, "w"), indent=2)
    print(f"-> {args.out}", flush=True)

if __name__ == "__main__":
    main()
