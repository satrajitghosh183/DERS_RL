# OmniTrace / DERS — System Overview

*A from-scratch GPU shader debugger, repurposed as an execution-level reward for shader synthesis.*

This document is the end-to-end description of everything built: the debugger, the reward, the models,
the flywheel, and the 32B→3B distillation now running on the H100s. For the day-to-day dev guide see
[`../CLAUDE.md`](../CLAUDE.md); for measured numbers see [`../RESULTS.md`](../RESULTS.md) and
[`../REPORT.md`](../REPORT.md).

---

## 1. The thesis

A compiler tells you a shader is *valid*. It cannot tell you the shader renders **NaN**, a **flat** frame,
or simply the **wrong image** — all of which compile fine. A real debugger can, because it actually runs
the shader. So we turn the debugger into a **decomposed, execution-level reward** and show it **beats
compile-only RLVR** for shader generation — and runs locally on a laptop.

```
compile-only reward:   compiles? ✓/✗                      (blind to runtime + pixels)
debugger reward:       compiles ✓ · runs-finite ✓ · renders-non-degenerate ✓ · colorful ✓ · matches-prompt ✓
```

## 2. OmniTrace — the debugger (the actual research contribution)

Written from scratch in C++23 (no glued-together packages). One pipeline serves both debugging and reward:

```
SPIR-V ──▶ UIR (universal SSA IR) ──▶ CFG + divergence ──▶ TraceTap capture ──▶ codec (26×) ──▶ store
              │                                                                                    │
              ├──▶ bit-exact CPU SIMT interpreter  (the reference oracle)                          │
              └──▶ Vulkan GPU capture + render → PPM                          time-travel ◀────────┘
```

- **UIR** — a universal SSA IR that ingests SPIR-V (DXIL/AIR/WGSL/GLSL are the roadmap). The SPIR-V
  backend re-emits IR that passes the official `spirv-val`.
- **CPU SIMT interpreter** — bit-exact reference execution; catches fma-vs-strict / fast-math divergence
  a GPU hides.
- **Divergence-aware trace codec** — losslessly compresses per-invocation traces ~26× and never expands,
  which is what makes time-travel over millions of invocations feasible (the systems spearhead).
- **GPU capture + render** — via Vulkan (MoltenVK on Apple Silicon, real NVIDIA/Mesa or lavapipe on
  Linux). Renders a candidate shader to pixels so the reward can *see* it.
- 20 C++ test suites; the end-to-end slice (SPIR-V → instrument → exec → compress → time-travel) runs green.

## 3. The decomposed reward (`omni_reward`)

`omni_reward` wraps the debugger as an RL signal. It embeds a candidate `mainImage` into a real shader,
runs it, renders it, and returns one JSON line. Tiered so partial credit is monotone:

| state | reward | why |
|---|---|---|
| won't compile | 0.0 | broken |
| compiles, NaN/no exec | 2.5 | a compiler would pass this |
| runs but flat/degenerate | 3.5 | a compiler would pass this too |
| runs + structured + colorful | 5.4+ | the only genuinely good shaders |

`visual = 0.6·structure + 0.4·color`. `--reward-mode {debugger,compile}` is the **ablation switch** that
isolates the contribution.

## 4. The models

| model | role | result |
|---|---|---|
| from-scratch C++ SLM (llm-cpp) | "can a tiny LM learn shaders from scratch?" | ❌ word-salad at every scale — honest negative (data-limited) |
| **DoRA-7B `deliver@400`** | the headline generator | ✅ best zero-leak row; runs as 4-bit GGUF/MLX locally |
| **DoRA-3B `rl3b_refined`** | local generator on the Mac (MPS) | ✅ powers the flywheel; rsynced to the H100 |
| **32B-Coder-Instruct** | teacher for distillation (current H100 job) | 🔭 8/8 compile+run in validation |

**Zero-leak ablation ladder** (held-out NL prompts verified absent from the SFT corpus — compile@1):

```
base 7B  0.00  →  +DoRA SFT  0.90  →  +debugger-RL  1.00
```

The debugger reward beats compile-only RLVR on **every** metric, biggest on render@1 and mean reward —
exactly the runtime/pixel dimensions a compile-only reward is blind to. Best-of-N + debugger at inference:
compile@1 0.85 → **0.988 usable@1**.

## 5. The flywheel (local, self-improving)

Best-of-N collapses toward grayscale concentric **rings** (cheapest way to spike luminance variance —
Goodhart). The fix runs at inference, no retrain:

```
flywheel.py:  generate N → debugger scores each → RERANK by
              0.45·color + 0.25·min(structure,cap) − 0.30·radial(ring penalty) + 0.50·clip + 1.0
              → pick winner → log (prompt, candidates, scores) to flywheel_log.jsonl
```

Measured: a grayscale ring with **44× more variance** scores **0.95**; the colorful non-radial winner
scores **1.70** — so it picks color over the collapse. `retrain_local.py` then continues DoRA SFT on the
good logged examples (compile+run+non-ring+above-bar) + a corpus sample → an improved adapter. **The
debugger labels every use for free, so usage *is* training data.**

## 6. The 32B→3B distillation (current H100 job)

The richest version of the loop, running now on 2×H100:

```
   2× Qwen2.5-Coder-32B (one per H100)          OmniTrace debugger              DoRA-3B student
   generate ~8k prompts × 4 candidates  ──▶     rerank: compile+run +    ──▶    retrain on the
   (the TEACHER)                                color − radial winners          verified-best (retrain_box.py)
                                                (flywheel_distill.jsonl)        → distilled 3B adapter
```

A big teacher proposes, the debugger verifies, a small local model distills the verified best —
**debugger-verified knowledge distillation**. Output banks continuously to the Mac (`local/h100_harvest/`)
and the box is disposable: everything reconstructable + backed up, nothing to lose if the allocation drops.

## 7. Where everything lives

- **Code** → GitHub (`DERS_RL`, contributors Satrajit Ghosh + Dov Kruger; the C++ trainer mirrors
  `RU-ECE/OLMo-corecpp`).
- **Models / data / checkpoints** → Box `nerc_server` (135 GiB): `adapters/` (11), `deliverables/`
  (merged 7B, GGUF + MLX 3B/7B), `data/` (corpus, from-scratch runs, corecpp runs), `results/` (datasets,
  galleries, evals).
- Never commit weights/data to git; never commit the corpus the repo already documents.

## 8. System diagram

The full system is in [`system_diagram.mmd`](system_diagram.mmd) (Mermaid). Rendered:
data → generators → OmniTrace → reward → training loops (SFT / RL / flywheel rerank / distillation) →
outputs (verified datasets, models, the publishable ablation).
