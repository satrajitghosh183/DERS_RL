# DERS — Debugger Execution-Reward for Synthesis

A from-scratch, API/OS-agnostic GPU **shader debugger** (OmniTrace) in hyper-optimized C++, repurposed
as a **decomposed execution-level reward** for shader synthesis — plus the from-scratch C++ LLM
training/inference stack used to train the generators.

**The thesis:** a real debugger sees what a compiler cannot. A shader can *compile* yet render NaN, a
flat frame, or the wrong image. OmniTrace compiles, lifts to a universal IR, runs a CPU SIMT reference,
and renders on the real GPU — turning that into a reward (compile + runs-finite + renders + color +
semantic-match) that beats compile-only RLVR, and runs locally on a laptop.

## Layout
| Path | What |
|---|---|
| `src/`, `include/`, `tests/` | **OmniTrace** universal C++ shader debugger (UIR, SPIR-V front/back end, CFG, divergence-aware trace codec, CPU SIMT interpreter, time-travel, GPU capture/render via Vulkan/MoltenVK) |
| `tools/omni_reward.cpp` | the debugger as an RL reward (compile + run + GPU render: finite + variance + chroma) |
| `tools/omni_render.cpp`, `omnishader` | GPU renderer + local CLI (`debug`/`render`/`gen`/`loop`) |
| `tools/{rl,eval,pipeline,synth,cli}/` | debugger-in-the-loop RL + evaluation tooling |
| `pipeline/` | full training/synthesis pipeline (DoRA SFT, GRPO RL, generation, captioning, eval) |
| `training/` | from-scratch C++ LLM training + inference (LibTorch path + LibTorch-free CUDA-native trainer w/ custom kernels) |
| `REPORT.md`, `RESULTS.md` | measured results: zero-leak ablation, compiler-blind figure, training speeds, deliverables |

## Headline (zero-leak, measured)
- Base 7B **0.0** → DoRA SFT **0.90** → +debugger-RL **1.0** compile@1 (held-out NL prompts)
- Debugger reward **beats compile-only RLVR** on every metric (run/render a compiler is blind to)
- Best-of-N + debugger at inference: compile@1 0.85 → **0.988 usable@1**
- Deliverable runs **fully local** (4-bit, ~4 GB) on any laptop; debugger runs on Apple Silicon (MoltenVK)

## Build
```bash
cmake -S . -B build -G Ninja && cmake --build build && (cd build && ctest)
```
The C++ LLM trainer in `training/` builds separately (see `training/README.md`).

Authors: **Satrajit Ghosh**, **Dov Kruger**.
