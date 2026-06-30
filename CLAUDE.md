# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

**OmniTrace** — a from-scratch, API/OS-agnostic GPU **shader debugger** in C++23, repurposed as a
**decomposed execution-level reward** for shader synthesis. The thesis: a real debugger sees what a
compiler cannot (a shader can *compile* yet render NaN, a flat frame, or the wrong image), so a reward
built from compile + runs-finite + renders-non-degenerate + color + semantic-match **beats compile-only
RLVR** — and runs locally on a laptop. The repo also drives the LLM side (DoRA SFT + debugger-in-the-loop
GRPO RL) that trains the shader generators.

## Build & test

```bash
cmake -S . -B build -G Ninja            # configure (Release by default)
cmake --build build                     # build everything (libomni_core + tools + 20 test suites)
cmake --build build --target omni_reward omni_render   # just the two reward/render binaries
(cd build && ctest)                     # run all tests
(cd build && ctest -R test_reward)      # run ONE suite (test_* names map to tests/test_*.cpp)
cmake -S . -B build -DOMNI_ASAN=ON      # ASan/UBSan build
```

- **C++23, CMake + Ninja.** No package manager — dependencies are vendored or system.
- **Vulkan is optional but central.** CMake auto-detects it (`find_package(Vulkan)`); on macOS it falls
  back to the MoltenVK SDK path in `OMNI_VULKAN_SDK` (edit this for your machine). Without Vulkan, GPU
  capture + render compile out and the rich visual reward degrades to compile/exec only. On Linux/H100 it
  uses the system Vulkan (real GPU or lavapipe software rasterizer — both work headless).
- The C++ LLM trainer in `external/llm-cpp/` (Satrajit + Dov's from-scratch OLMo-corecpp rewrite) builds
  separately; it needs CUDA and is **not** part of the default build.

## The two binaries that matter (and their I/O contracts — easy to get wrong)

- **`build/omni_reward`** — the debugger *as an RL reward*. Reads a **harness-wrapped full fragment
  shader on stdin**, prints one JSON line: `{compiled, executed, all_finite, output_variance, reward,
  breakdown:{compile,exec,visual}, ...}`. It has `--reward-mode {debugger,compile}` for the ablation.
  Reward tiers: broken `0.0` < NaN `2.5` < flat `3.5` < structured `5.4+`.
- **`build/omni_render`** — GPU renderer. Takes a **raw Shadertoy `mainImage` shader as a FILE arg**
  (not wrapped, not stdin): `omni_render <in.glsl> <out.ppm> [w] [h] [iTime]`. Writes a PPM.

The harness that wraps a raw `mainImage` into the full `#version 450 … void main()` shader `omni_reward`
expects is duplicated in the Python tools (`flywheel.py`, `gen_dataset.py`, `rerank_box.py`) as
`HARNESS`/`wrap()`. **`omni_reward` needs the wrapped form; `omni_render` needs the raw form.** Feeding
the wrong one silently yields `compiled:false` / a compute-compile error.

## C++ architecture (`src/` + `include/omni/`, mirrored layout)

The pipeline is **SPIR-V → UIR → instrument → CPU-exec / GPU-render → compress → store → time-travel**:

- `uir/` — the **Universal SSA IR** all front/back ends target.
- `frontends/` — SPIR-V (and others) → UIR lifter. `backends/` — UIR → SPIR-V emitter (passes `spirv-val`).
- `analysis/` — CFG, dominators/post-dominators, divergence/thread-frontiers.
- `capture/` — `TraceTap` instrumentation pass (idempotent value capture).
- `trace/` — divergence-aware lossless **trace codec** (~26×; never expands). `store/` — mmap trace store.
- `cpuref/` — bit-exact **CPU SIMT interpreter** (the reference oracle; catches fma-vs-strict diffs).
- `timetravel/` — state reconstruction at any recorded step.
- `gpu/` — Vulkan capture + render (`vulkan_capture.cpp`, gated on `OMNI_HAVE_VULKAN`).
- `reward/` — the §11 decomposed reward + per-line credit (`oracle.cpp`); `omni_reward.cpp` wraps it.
- `synth/` — generate→validate→feedback driver. `ml/` — DoRA identity + magnitude/direction math.

## The LLM / training side (`tools/`)

- `tools/rl/rl_refine.py` — debugger-in-the-loop **GRPO** RL; `--reward-mode {debugger,compile}` is the
  ablation; `--clip` (CLIP semantic reward), `--kl` (anti-collapse KL to base via adapter-disable).
- `tools/eval/` — `eval_rigorous.py` (compile@k/run@k/render@k, best-of-N usable@1), `gen_dataset.py`
  (best-of-N verified dataset + renders), `baseline_frontier.py` (Claude zero-shot through OmniTrace),
  `compiler_blind.py` (the thesis figure), `poster.py` (gallery contact sheet, `--full` mode).
- `tools/synth/` — corpus/caption generation. Qwen-Coder is a **completion** model: chat "describe this"
  produces code garbage; captioning uses **few-shot completion** instead.
- `tools/cli/flywheel.py` + `retrain_local.py` — the **flywheel** (see below).
- `omnishader` — top-level CLI: `debug` (local OmniTrace), `gen`/`loop` (generation, runs on a GPU box
  over ssh; `OMNISHADER_BOX`/`OMNISHADER_KEY`/`OMNISHADER_GEN` configure it).

## The flywheel (`tools/cli/`)

Best-of-N alone Goodhart-collapses to grayscale concentric **rings** (they cheaply max luminance
variance). The fix runs at inference: `flywheel.py` re-ranks N candidates by a reward that rewards
**color** (chroma), **caps** structure, and **penalizes radial symmetry** (rings are rotation-invariant →
low MSE vs a 90°-rotated copy), plus optional CLIP:
`score = 0.45·color + 0.25·min(structure,cap) − 0.30·radial + 0.50·clip + 1.0` (must compile+run, else −1).
Every call logs to `flywheel_log.jsonl`; `retrain_local.py` continues DoRA SFT (Mac/MPS) on the good
logged examples (compile+run+non-ring+above-bar) + a corpus sample (anti-forgetting) → improved adapter.
The debugger labels each example for free, so usage *is* training data.

## Conventions & gotchas

- **Models/data live in Box** (the `nerc_server` rclone remote), not in git — only code is committed.
  The canonical fresh repo is `git@github.com:satrajitghosh183/DERS_RL.git` (contributors: Satrajit Ghosh
  + Dov Kruger only). This working dir's `origin` is the older `NERC.git`.
- **Eval leakage is the #1 trap.** Synthetic SFT prompts come from `make_prompts` (`gen_shaders.py`), so
  naive held-out prompts leak into the SFT corpus. Always build held-out sets by filtering against the
  corpus `// Shader:` headers → `heldout_clean.jsonl` (zero residual leaks). Leaked numbers are a
  memorization ceiling, not a real result.
- **Reward modes are the ablation.** Any claim that "the debugger reward beats compile-only" must compare
  `--reward-mode debugger` vs `--reward-mode compile` on a zero-leak held-out set.
- Generation prompt format is the completion string `// Shader: <prompt>\n` — mismatching it tanks quality.
- Commit messages here use **no `Co-Authored-By`/Claude trailer** (project convention).
