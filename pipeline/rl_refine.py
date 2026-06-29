#!/usr/bin/env python3
"""
Debugger-in-the-loop RL refinement (GRPO/RLOO) for the shader-LM — the SIGGRAPH thesis.

The overnight pipeline did SFT then *measured* compile@1. This closes the loop: the model
GENERATES shaders, OmniTrace SCORES them with a rich, multi-signal reward (not just
compile/no-compile), and the policy is updated to maximize it. The reward mirrors
src/reward/oracle.cpp exactly:

    r = w_compile·compiled                                  (gate: 0 if it won't build)
      + w_exec·executed                                     (ran without trap)
      + w_divergence·(1 - divergence)                       (warp reconvergence health)
      + w_numerical·(1 - numerical_error)                   (1 - saturating ULP error)
      + w_visual·visual_match                               (exp(-MSE) vs reference image)
      - w_perf·perf_penalty

Signals come from the OmniTrace tools (validator/renderer/cpuref diff). Where a binary
isn't available the provider degrades gracefully to compile-only, so this runs anywhere
(full signal set on the box with the built debugger + a GPU policy).

GRPO: for each prompt, sample G completions, turn rewards into group-relative advantages
A_i = (r_i - mean)/std, and do a policy-gradient step (no value net). Works on the HF
policy (with the DoRA adapter) or, via --backend chat, the from-scratch llm-cpp model.

  python3 rl_refine.py --self-test                 # verify reward + GRPO math
  python3 rl_refine.py --prompts eval_prompts.jsonl --base Qwen2.5-Coder-3B \
      --adapter dora3b_out/adapter --group 8 --steps 200 --omnitrace ./build
"""
import argparse, json, math, os, re, subprocess, sys, tempfile
from dataclasses import dataclass, field


# ----------------------------------------------------------------- reward
@dataclass
class Weights:                      # mirror omni::reward::Weights defaults
    compile: float = 1.0
    exec: float = 0.5
    divergence: float = 0.5
    numerical: float = 0.5
    visual: float = 1.0
    perf: float = 0.25


def reward(sig, w: Weights):
    """Scalar reward from a signal dict. Mirrors omni::reward::score (compile gates)."""
    r = w.compile * (1.0 if sig.get("compiled") else 0.0)
    if not sig.get("compiled"):
        return r, {"compile": r}
    b = {
        "compile": r,
        "exec": w.exec * (1.0 if sig.get("executed") else 0.0),
        "divergence": w.divergence * (1.0 - clamp(sig.get("divergence", 0.0))),
        "numerical": w.numerical * (1.0 - clamp(sig.get("numerical_error", 0.0))),
        "visual": w.visual * clamp(sig.get("visual_match", 0.0)),
        "perf": -w.perf * clamp(sig.get("perf_penalty", 0.0)),
    }
    return sum(b.values()), b


def clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def normalize_ulp(mean_ulp, scale=4.0):
    return 0.0 if mean_ulp <= 0 else 1.0 - math.exp(-mean_ulp / max(scale, 1e-9))


# -------------------------------------------------- OmniTrace signal provider
class Signals:
    """Score a generated shader. Uses OmniTrace binaries when present; else compile-only."""
    def __init__(self, omnitrace_dir=None, glslang="glslangValidator"):
        self.dir = omnitrace_dir
        self.glslang = glslang

    def _bin(self, name):
        p = os.path.join(self.dir, name) if self.dir else name
        return p if (not self.dir or os.path.exists(p)) else None

    def compile(self, glsl):
        f = tempfile.NamedTemporaryFile(suffix=".frag", delete=False, mode="w")
        f.write(glsl); f.close()
        try:
            return subprocess.run([self.glslang, "-V", f.name, "-o", os.devnull],
                                  capture_output=True, timeout=20).returncode == 0
        except Exception:
            return False
        finally:
            os.unlink(f.name)

    def score(self, glsl, ref_image=None):
        sig = {"compiled": self.compile(glsl)}
        if not sig["compiled"]:
            return sig
        # --- OmniTrace rich signals (renderer + cpuref diff). Graceful fallback. ---
        renderer = self._bin("synth_render") or self._bin("renderer")
        if renderer:
            try:
                out = subprocess.run([renderer, "--glsl", "-", "--json"], input=glsl,
                                     capture_output=True, text=True, timeout=30)
                j = json.loads(out.stdout or "{}")
                sig["executed"] = bool(j.get("executed", True))
                sig["divergence"] = float(j.get("divergence", 0.0))
                sig["numerical_error"] = normalize_ulp(float(j.get("mean_ulp", 0.0)))
                sig["perf_penalty"] = clamp(float(j.get("perf_penalty", 0.0)))
                if ref_image and "image" in j:
                    sig["visual_match"] = image_mse_match(j["image"], ref_image)
            except Exception:
                sig["executed"] = True            # compiled but couldn't introspect
        else:
            sig["executed"] = True                # compile-only mode
        return sig


def image_mse_match(a, b):
    n = min(len(a), len(b))
    if not n:
        return 0.0
    mse = sum((a[i] - b[i]) ** 2 for i in range(n)) / n
    return math.exp(-mse)                          # mirror omni::reward::image_similarity


# -------------------------------------------------------------- GRPO core
def grpo_advantages(rewards):
    """Group-relative advantage: (r - mean)/std. The value-net-free heart of GRPO."""
    n = len(rewards)
    if n == 0:
        return []
    mean = sum(rewards) / n
    var = sum((r - mean) ** 2 for r in rewards) / n
    std = math.sqrt(var) + 1e-6
    return [(r - mean) / std for r in rewards]


# ----------------------------------------------- the real debugger as the reward
class OmniRewardScorer:
    """Reward each candidate by invoking the OmniTrace `omni_reward` C++ CLI: it compiles,
    lifts to UIR, runs the CPU SIMT reference, and returns a JSON reward that already mirrors
    omni::reward::score. This makes the actual debugger — not a Python compile check — the
    reward signal. Falls back to glslang-only compile reward if the binary is absent."""
    def __init__(self, bin_path, glslang="glslangValidator", mode="debugger"):
        self.bin = bin_path if (bin_path and os.path.exists(bin_path)) else None
        self.glslang = glslang
        self.mode = mode   # "debugger" = rich omni_reward; "compile" = glslang compile-only (ablation)

    def score(self, glsl):
        if self.bin and self.mode != "compile":
            try:
                p = subprocess.run([self.bin], input=glsl, capture_output=True, text=True,
                                   errors="replace", timeout=30)
                j = json.loads(p.stdout.strip().splitlines()[-1])
                return float(j.get("reward", 0.0)), j.get("breakdown", {})
            except Exception:
                pass
        # fallback: compile-only via glslang
        import tempfile
        f = tempfile.NamedTemporaryFile(suffix=".frag", delete=False, mode="w")
        f.write(glsl); f.close()
        try:
            ok = subprocess.run([self.glslang, "-V", f.name, "-o", os.devnull],
                                capture_output=True, timeout=20).returncode == 0
        except Exception:
            ok = False
        finally:
            os.unlink(f.name)
        return (1.0 if ok else 0.0), {"compile": 1.0 if ok else 0.0}


class ClipScorer:
    """Semantic reward: render the shader and CLIP-score the image against the prompt text — rewards
    'the picture looks like what was asked', the dimension compile/run/structure cannot see. Fixes
    loose prompt adherence (e.g. 'neon city' -> a generic palette)."""
    def __init__(self, render_bin, device, model="openai/clip-vit-base-patch32"):
        import torch
        from transformers import CLIPModel, CLIPProcessor
        self.torch, self.bin, self.dev = torch, render_bin, device
        self.model = CLIPModel.from_pretrained(model).to(device).eval()
        self.proc = CLIPProcessor.from_pretrained(model)

    def _render(self, code):
        import tempfile, subprocess, os
        from PIL import Image
        g = tempfile.NamedTemporaryFile(suffix=".glsl", delete=False, mode="w"); g.write(code); g.close()
        ppm = g.name + ".ppm"
        try:
            subprocess.run([self.bin, g.name, ppm, "128", "128", "1.0"], capture_output=True, timeout=30)
            return Image.open(ppm).convert("RGB") if os.path.exists(ppm) else None
        except Exception:
            return None
        finally:
            os.unlink(g.name)
            if os.path.exists(ppm): os.unlink(ppm)

    def score(self, code, prompt):
        img = self._render(code)
        if img is None:
            return 0.0
        with self.torch.no_grad():
            inp = self.proc(text=[prompt], images=[img], return_tensors="pt", padding=True).to(self.dev)
            o = self.model(**inp)
            ie = o.image_embeds / o.image_embeds.norm(dim=-1, keepdim=True)
            te = o.text_embeds / o.text_embeds.norm(dim=-1, keepdim=True)
            sim = (ie * te).sum(-1).item()
        return max(0.0, min(1.0, (sim - 0.15) / 0.15))   # CLIP sims ~0.15-0.30 -> [0,1]


# --------------------------------------------------------------- train loop
def train(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "eval"))
    from eval_batched import extract_code, wrap                       # reuse wrapping

    scorer = OmniRewardScorer(args.omni_reward_bin, args.glslang, mode=args.reward_mode)
    prompts = [json.loads(l) for l in open(args.prompts) if l.strip()]

    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16,
                                                 device_map="auto")
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter, is_trainable=True)
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    clip_scorer = None
    if args.clip:
        print(f"[rl] loading CLIP semantic reward (w={args.w_clip})…", flush=True)
        clip_scorer = ClipScorer(args.render_bin, device=str(model.device))

    for step in range(args.steps):
        p = prompts[step % len(prompts)]
        ids = tok(p["prompt"], return_tensors="pt").to(model.device)
        # 1. sample G completions
        gen = model.generate(**ids, max_new_tokens=384, do_sample=True, temperature=1.0,
                             top_p=0.95, num_return_sequences=args.group,
                             pad_token_id=tok.eos_token_id)
        comp = gen[:, ids.input_ids.shape[1]:]
        # 2. reward each via OmniTrace
        rewards, bdowns = [], []
        prompt_text = p["prompt"].replace("// Shader:", "").strip()
        for row in comp:
            code = extract_code(tok.decode(row, skip_special_tokens=True))
            r, b = scorer.score(wrap(code))            # the OmniTrace debugger is the reward
            if clip_scorer is not None and b.get("compile", 0) > 0:   # semantic match for compiling shaders
                r += args.w_clip * clip_scorer.score(code, prompt_text)
            rewards.append(r); bdowns.append(b)
        # 3. group-relative advantages
        adv = grpo_advantages(rewards)
        # 4. policy-gradient: -A * logprob(completion)
        full = torch.cat([ids.input_ids.repeat(args.group, 1), comp], dim=1)
        out = model(full)
        logits = out.logits[:, ids.input_ids.shape[1] - 1:-1, :]
        logp = torch.log_softmax(logits.float(), dim=-1)
        toklp = logp.gather(-1, comp.unsqueeze(-1)).squeeze(-1)
        mask = (comp != tok.eos_token_id).float()
        seqlp = (toklp * mask).sum(1) / mask.sum(1).clamp(min=1)
        A = torch.tensor(adv, device=seqlp.device, dtype=seqlp.dtype)
        loss = -(A * seqlp).mean()
        kl_val = 0.0
        if args.kl > 0:                                   # anti-collapse: KL to the base distribution
            with torch.no_grad(), model.disable_adapter():
                ref_logits = model(full).logits[:, ids.input_ids.shape[1] - 1:-1, :]
                ref_toklp = torch.log_softmax(ref_logits.float(), dim=-1).gather(-1, comp.unsqueeze(-1)).squeeze(-1)
            kl = ((toklp - ref_toklp) * mask).sum(1) / mask.sum(1).clamp(min=1)
            loss = loss + args.kl * kl.mean()
            kl_val = kl.mean().item()
        opt.zero_grad(); loss.backward(); opt.step()
        if step % args.log_every == 0:
            print(f"step {step}  mean_r={sum(rewards)/len(rewards):.3f}  "
                  f"best={max(rewards):.3f}  loss={loss.item():.4f}  kl={kl_val:.4f}", flush=True)

    if args.adapter:
        model.save_pretrained(args.out)
    print(f"done -> {args.out}")


# ------------------------------------------------------------------ self-test
def _self_test():
    w = Weights()
    ok = True
    # compile gate
    r0, _ = reward({"compiled": False}, w)
    ok &= r0 == 0.0
    # perfect shader
    r1, b1 = reward({"compiled": True, "executed": True, "divergence": 0.0,
                     "numerical_error": 0.0, "visual_match": 1.0, "perf_penalty": 0.0}, w)
    ok &= abs(r1 - (1.0 + 0.5 + 0.5 + 0.5 + 1.0)) < 1e-9
    # worse than perfect
    r2, _ = reward({"compiled": True, "executed": True, "divergence": 1.0,
                    "numerical_error": 1.0, "visual_match": 0.0, "perf_penalty": 1.0}, w)
    ok &= r2 < r1 and abs(r2 - (1.0 + 0.5 - 0.25)) < 1e-9
    # GRPO advantages: zero-mean, unit-ish std, monotone in reward
    adv = grpo_advantages([0.0, 1.0, 2.0, 3.0])
    ok &= abs(sum(adv)) < 1e-6 and adv[0] < adv[-1]
    # ULP normalizer saturating in [0,1)
    ok &= normalize_ulp(0) == 0.0 and 0 < normalize_ulp(10) < 1
    print(f"[self-test] reward gate/order + GRPO advantages + ULP: {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser(description="Debugger-in-the-loop GRPO refinement")
    ap.add_argument("--prompts"); ap.add_argument("--base"); ap.add_argument("--adapter")
    ap.add_argument("--omni-reward-bin", default="./build/omni_reward",
                    help="the OmniTrace omni_reward CLI (the debugger as reward)")
    ap.add_argument("--glslang", default="glslangValidator")
    ap.add_argument("--reward-mode", default="debugger", choices=["debugger", "compile"],
                    help="debugger = rich omni_reward (compile+run+render); compile = compile-only (ablation)")
    ap.add_argument("--clip", action="store_true", help="add CLIP semantic reward (render vs prompt)")
    ap.add_argument("--w-clip", type=float, default=1.5, help="weight on the CLIP semantic term")
    ap.add_argument("--render-bin", default=os.path.expanduser("~/NERC/build/omni_render"))
    ap.add_argument("--kl", type=float, default=0.0, help="KL-to-base penalty (anti-collapse); e.g. 0.05")
    ap.add_argument("--group", type=int, default=8, help="G completions per prompt")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--out", default="rl_out/adapter")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        sys.exit(0 if _self_test() else 1)
    if not (args.prompts and args.base):
        ap.error("--prompts and --base required (unless --self-test)")
    train(args)


if __name__ == "__main__":
    main()
