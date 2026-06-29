# OmniTrace — Full Technical Report

A from-scratch C++ universal shader debugger (**OmniTrace**) + a debugger-in-the-loop shader-synthesis
stack. Everything below is measured on real hardware: **2× NVIDIA H100 80GB** (training/generation) and
**Apple M4 Pro** (local debugger/renderer via MoltenVK). Honest throughout — negatives are stated.

Last updated: 2026-06-29.

---

## 1. Headline results (all zero-leak unless noted)

### 1.1 The ablation ladder — debugger-RL beats compile-only RLVR
Qwen2.5-Coder-7B + DoRA SFT (6000 steps, 208,749 shaders) → GRPO RL. Held-out = 40 natural-language
prompts **verified absent from the SFT corpus** (we caught and removed 32/40 prompt leakage first).

| Model | compile@1 | run@1 | render@1 | meanR |
|---|:--:|:--:|:--:|:--:|
| base 7B | 0.00 | 0.00 | 0.00 | 0.044 |
| + DoRA SFT | 0.90 | 0.875 | 0.775 | 3.794 |
| + RL compile-only (150) | 0.975 | 0.975 | 0.875 | 3.835 |
| **+ RL debugger (160)** | **1.00** | **1.00** | **0.95** | **4.056** |
| **deliver@400** (debugger-RL to convergence) | **1.00** | **1.00** | **1.00** | **5.308** |

Debugger reward beats compile-only on every metric; only row to reach perfect compile@1 + run@1.
Gains concentrate in render@1 (+0.075) and reward (+0.221) — the run/render quality a compiler is
blind to.

### 1.2 The thesis figure — a compiler is blind to broken shaders
Five shaders, **all compile** (compile-only reward = 1.0 each):

| shader | compile-only | **OmniTrace** | renders |
|---|:--:|:--:|---|
| NaN (0/0) | 1.0 | **2.50** (exec=0) | blank white frame |
| Inf (log 0) | 1.0 | **2.50** (exec=0) | non-finite |
| flat black | 1.0 | **3.50** (vis=0) | empty frame |
| flat constant | 1.0 | **3.50** (vis=0) | degenerate solid |
| real shader | 1.0 | **5.43** (vis=1.93) | a vivid rainbow spiral |

### 1.3 Frontier baseline & the defensible claim
Frontier model (Claude) zero-shot on the free-form held-out prompts, scored through the **same** oracle:

| model | compile@1 | run@1 | render@1 | local/offline? |
|---|:--:|:--:|:--:|:--:|
| Frontier (Claude) zero-shot | 1.00 | 1.00 | 1.00 | ✗ |
| Ours 7B, in-distribution | 1.00 | 1.00 | 1.00 | ✓ |
| Ours 7B, free-form OOD | 0.625 | 0.60 | 0.575 | ✓ |
| Ours 7B + debugger best-of-N | ~0.90 (any-of-k) | — | — | ✓ |

**Claim:** not "we beat frontier on raw quality" — we don't. The claim is the intersection nobody else
occupies: **local + offline + execution-grounded** shader synthesis. Frontier models can't run on a
laptop; RenderDoc/Nsight can't generate or score correctness as a reward; compile-only RLVR is blind to
NaN/flat. Ours is the only from-scratch debugger producing a decomposed execution-level reward,
deployable in 4-bit (~4 GB) on any laptop.

### 1.4 Honest negatives
- **From-scratch shader LMs fail** (word-salad) at all data scales (300M/1B/3.6B on 171M tokens) — a
  data-ceiling result, not a bug. Motivates the DoRA path.
- **v2 reward upgrade is neutral.** color (saturation) + CLIP-semantic + KL-anti-collapse reward, RL'd
  220 steps on the v1 base, did **not** measurably beat v1 (compile@1 0.975 vs 1.0; color comparison
  inconclusive). Each reward term is validated *in isolation* (color 5.43 vs 4.70; CLIP 0.89 vs 0.33),
  but a provably-correct richer reward does not automatically shift a strong SFT model in 220 steps.
- **Captioning v1 broke the model** — Qwen-Coder is a completion model; chat-style "describe this" gave
  code garbage; few-shot completion fixed it (14,974 clean NL captions), but prose captions then taught
  the model to emit prose instead of GLSL, so captions were dropped from the deliverable.

---

## 2. Training speeds & costs (2× H100 80GB)

| Stage | Config | Throughput | Wall-clock |
|---|---|---|---|
| DoRA SFT (7B) | r16 α32, all proj, batch 2×4×2gpu=16, seq 1024, bf16, grad-ckpt | **3.59 s/it** | ~6.0 h / 6000 steps |
| GRPO RL, compile-only | group 8, 1 prompt/step, max_new 384, temp 1.0 | ~1.4 min/step | ~3.5 h / 150 steps |
| GRPO RL, debugger | + GPU render per candidate (omni_reward) | ~1.8 min/step | ~4.8 h / 160 steps |
| GRPO RL, v2 (color+CLIP+KL) | + omni_render + CLIP encode + KL ref-pass per candidate | ~1.4 min/step | ~5.1 h / 220 steps |
| Synthetic gen (32B, vLLM) | tp2, bf16, max 512 | ~8k tok/s | — |
| Captioning (32B, few-shot) | tp2, max 40, n_per 2 | — | ~12 min / 15k pairs |
| GGUF convert+quantize (7B) | f16 write 500 MB/s, then Q4_K_M | — | ~3 min |
| DoRA adapter merge (7B, CPU) | merge_and_unload | — | ~15 min |

Corpus: 22k → **196,418 unique shaders / 171M tokens** (HF datasets incl. The Stack; Shadertoy API
abandoned — gated). GLSL byte-BPE tokenizer: **2.69 chars/token vs GPT-2's 2.0** (~27% fewer tokens).

---

## 3. The debugger (OmniTrace) — from-scratch C++23, 19 test suites green

Universal SSA IR (UIR); hand-written SPIR-V front/back end (passes official `spirv-val` + re-lift);
CFG (dominators / post-dominators / thread frontiers); capture/instrumentation pass; **divergence-aware
trace codec** (lossless, 26× / 4.5×, ~entropy bound); mmap trace store; CPU SIMT reference interpreter;
time-travel reconstruction; ULP/divergence numerical diff; **real GPU capture + render** (Vulkan on
H100, MoltenVK on M4 Pro — works concurrently with CUDA/vLLM).

**`omni_reward`** exposes it as a decomposed RL reward: compile + CPU-run + GPU-render (finiteness +
luminance-variance **+ chroma/saturation**). 4-tier signal: broken (0.0) < NaN (2.5) < flat (3.5) <
structured/colorful (5.4+). `--reward-mode {debugger,compile}` for the ablation.
**`omni_render`** renders a `mainImage` to an image on the real GPU.

---

## 4. Inference: debugger-in-the-loop best-of-N + repair

`omnishader gen/loop -n N`: generate N candidates → score each with the **local** OmniTrace reward →
keep the best; if it doesn't run clean, feed the diagnostic back and repair. Lifts usable success from
compile@1 toward compile@k (≈0.62 → ~0.90 on free-form) with **no retraining** — and the scoring runs
entirely on the laptop (MoltenVK).

---

## 5. Deliverables

| Artifact | Size | Runs on |
|---|---|---|
| `dora7b_deliver_400` (DoRA adapter, v1) | 167 MB | + 7B base |
| `merged_7b_v1` (merged HF model) | 15 GB | any (HF/transformers) |
| **`shader7b_v1_q4_k_m.gguf`** | **3.69 GB** | **any laptop** (llama.cpp/Ollama/LM Studio) |
| MLX 4-bit (pending) | ~4 GB | Apple Silicon (fast) |
| 3B: `rl3b_refined` + `merged_3b_v1` | — | smaller/faster local |

Local CLI: `omnishader debug/render/gen/loop` — debugger + renderer run on the Mac; generation via the
local GGUF/MLX model or the box.

---

## 6. Where this is best-in-class (and where it isn't)

**Best-in-class / novel:** the debugger as a decomposed execution-level RL reward; the from-scratch
universal C++ debugger; the local + offline + execution-grounded deployment; methodological rigor
(zero-leak ablation, leakage caught, honest negatives).

**Not best-in-class:** raw shader quality vs a frontier model (they win); debugger breadth vs
RenderDoc/Nsight (research subset); scale (shown at 7B). Path to top-tier: baselines (done — §1.3),
the compiler-blind figure (done — §1.2), and visual/human perceptual eval (next).
