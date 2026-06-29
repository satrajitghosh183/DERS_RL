#!/usr/bin/env python3
"""Local debugger-in-the-loop inference with a RICHER reward rerank + usage logging — the flywheel
engine. Runs entirely on the Mac: the local model (transformers/MPS) generates N candidates, the
local C++ OmniTrace debugger labels each (compile + run + render), and we RE-RANK by a reward that
fights the ring/grayscale collapse — color + capped-structure + anti-radial (+ optional CLIP) —
instead of raw variance (which favors rings). Every call is logged as training data.

  python3 flywheel.py "<prompt>" -n 8 [--clip] [--open]

Log line: {prompt, candidates:[{glsl, compiled, runs, reward, chroma, radial, structure, score}],
           chosen_index, ts}. Feed flywheel_log.jsonl back into retrain_local.py.
"""
import argparse, json, os, subprocess, tempfile, time, math
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RB = os.path.join(ROOT, "build", "omni_reward")
RD = os.path.join(ROOT, "build", "omni_render")
LOG = os.path.join(ROOT, "flywheel_log.jsonl")
BASE = os.environ.get("FLY_BASE", os.path.join(ROOT, "local", "qwen2.5-coder-3b"))
ADAPTER = os.environ.get("FLY_ADAPTER", os.path.join(ROOT, "local", "rl3b_refined"))

HARNESS = ("#version 450\nlayout(location=0) out vec4 _O;\n"
  "layout(push_constant) uniform U { vec3 iResolution; float iTime; vec4 iMouse; int iFrame; } u;\n"
  "#define iResolution u.iResolution\n#define iTime u.iTime\n#define iMouse u.iMouse\n#define iFrame u.iFrame\n")
def wrap(c): return HARNESS + c + "\nvoid main(){ vec4 c=vec4(0.); mainImage(c, gl_FragCoord.xy); _O=c; }\n"
def extract(t):
    if "```" in t:
        seg = t.split("```", 2); b = seg[1] if len(seg) > 1 else t
        for lg in ("glsl", "c", "cpp", "c++"):
            if b.startswith(lg): b = b[len(lg):]; break
        return b.strip()
    return t.strip()

def omni(glsl):
    try:
        p = subprocess.run([RB], input=glsl, capture_output=True, text=True, errors="replace", timeout=40)
        j = json.loads(p.stdout.strip().splitlines()[-1])
        return bool(j.get("compiled")), bool(j.get("executed")), float(j.get("reward", 0))
    except Exception:
        return False, False, 0.0

def render_feats(code):
    """Render the shader and compute color + anti-radial features from the image."""
    from PIL import Image
    g = tempfile.NamedTemporaryFile(suffix=".glsl", delete=False, mode="w"); g.write(code); g.close()
    ppm = g.name + ".ppm"
    try:
        subprocess.run([RD, g.name, ppm, "96", "96", "1.0"], capture_output=True, timeout=25)
        if not os.path.exists(ppm): return None
        im = Image.open(ppm).convert("RGB"); px = list(im.getdata()); n = len(px)
        lum = [0.2126*r+0.7152*gg+0.0722*b for r, gg, b in px]
        mean = sum(lum)/n; structure = sum((x-mean)**2 for x in lum)/n          # spatial variance
        chroma = sum((max(p)-min(p)) for p in px)/n/255.0                        # color (saturation)
        # anti-radial: rings are rotation-invariant -> low diff vs a 90deg rotation
        rl = list(im.rotate(90).convert("L").getdata()); l8 = [int(x) for x in im.convert("L").getdata()]
        rot_mse = sum((a-b)**2 for a, b in zip(l8, rl))/n/(255.0**2)
        radial = max(0.0, 1.0 - rot_mse*40.0)                                    # ~1 if ring-like, ~0 if not
        return {"structure": round(structure, 4), "chroma": round(chroma, 4), "radial": round(radial, 3), "ppm": ppm}
    except Exception:
        return None
    finally:
        os.unlink(g.name)

def rerank_score(compiled, runs, feats, clip=0.0):
    if not (compiled and runs and feats): return -1.0
    s = feats["structure"]
    return (0.45 * min(feats["chroma"]*3.0, 1.0)        # reward COLOR
            + 0.25 * min(s*60.0, 1.0)                    # some structure, but CAPPED (no variance-hacking)
            - 0.30 * feats["radial"]                     # PENALIZE the ring collapse
            + 0.50 * clip                                # optional semantic match
            + 1.0)                                        # base for compiling+running

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt"); ap.add_argument("-n", "--n", type=int, default=8)
    ap.add_argument("--temp", type=float, default=1.05); ap.add_argument("--clip", action="store_true")
    ap.add_argument("--open", action="store_true"); ap.add_argument("--out")
    a = ap.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(BASE)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float16).to(dev)
    if os.path.isdir(ADAPTER):
        from peft import PeftModel; m = PeftModel.from_pretrained(m, ADAPTER)
    m.eval()
    clip_scorer = None
    if a.clip:
        from transformers import CLIPModel, CLIPProcessor
        cm = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(dev).eval()
        cp = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        clip_scorer = (cm, cp, dev)
    ids = tok(f"// Shader: {a.prompt}\n", return_tensors="pt").to(dev)
    with torch.no_grad():
        g = m.generate(**ids, max_new_tokens=384, do_sample=True, temperature=a.temp, top_p=0.95,
                       repetition_penalty=1.1, num_return_sequences=a.n, pad_token_id=tok.eos_token_id)
    cands = []
    for s in g:
        code = extract(tok.decode(s[ids.input_ids.shape[1]:], skip_special_tokens=True))
        comp, runs, rew = omni(wrap(code))
        feats = render_feats(code) if comp else None
        clip = 0.0
        if clip_scorer and feats:
            from PIL import Image
            cm, cp, dv = clip_scorer
            with torch.no_grad():
                inp = cp(text=[a.prompt], images=[Image.open(feats["ppm"]).convert("RGB")], return_tensors="pt", padding=True).to(dv)
                o = cm(**inp); ie = o.image_embeds/o.image_embeds.norm(dim=-1, keepdim=True)
                te = o.text_embeds/o.text_embeds.norm(dim=-1, keepdim=True)
                clip = max(0.0, min(1.0, ((ie*te).sum(-1).item()-0.15)/0.15))
        if feats and "ppm" in feats: os.unlink(feats["ppm"]) if os.path.exists(feats["ppm"]) else None
        sc = rerank_score(comp, runs, feats, clip)
        cands.append({"glsl": code, "compiled": comp, "runs": runs, "reward": round(rew, 2),
                      "chroma": feats["chroma"] if feats else 0, "radial": feats["radial"] if feats else 1,
                      "structure": feats["structure"] if feats else 0, "clip": round(clip, 3), "score": round(sc, 3)})
    best = max(range(len(cands)), key=lambda i: cands[i]["score"])
    with open(LOG, "a") as f:
        f.write(json.dumps({"prompt": a.prompt, "candidates": cands, "chosen_index": best, "ts": int(time.time())}) + "\n")
    b = cands[best]
    print(f"\nbest of {len(cands)} (reranked: color {b['chroma']:.2f}, radial {b['radial']:.2f}, score {b['score']:.2f}):")
    print(b["glsl"][:400])
    print(f"\n[logged to {os.path.basename(LOG)} — {sum(1 for _ in open(LOG))} examples collected]")
    if a.out: open(a.out, "w").write(b["glsl"])
    if a.open and b["compiled"]:
        png = "/tmp/fly.png"; subprocess.run([RD, "/dev/stdin", "/tmp/fly.ppm", "512", "512", "1.0"], input=b["glsl"], text=True, capture_output=True)
        if os.path.exists("/tmp/fly.ppm"):
            subprocess.run(["sips", "-s", "format", "png", "/tmp/fly.ppm", "--out", png], capture_output=True); subprocess.run(["open", png])

if __name__ == "__main__":
    main()
