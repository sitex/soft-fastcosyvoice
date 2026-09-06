<!-- GSD:project-start source:PROJECT.md -->

## Project

**soft-fastcosyvoice**

`soft-fastcosyvoice` is the personal `sitex` fork of FastCosyVoice, an optimized CosyVoice derivative for local accelerated TTS experiments. It preserves upstream inference, streaming, WebUI, benchmark, and TensorRT-oriented surfaces while local verification remains a separate milestone.

**Core Value:** The operator can reproduce a local synthesis path and measure its performance without confusing configured optimizations with verified speed.

### Constraints

- **Runtime**: Python 3.10–3.11 and heavyweight model/audio dependencies — environment reproduction must be explicit.
- **Hardware**: optimized paths target NVIDIA CUDA 12.8 and TensorRT-LLM — unsupported hardware checks must be skipped visibly.
- **Measurement**: RTF and TTFB depend on model, prompt, warm-up, and device state — benchmarks must record all four.
- **Verification**: no `scripts/verify` exists — Phase 1 must add a deterministic project gate.

<!-- GSD:project-end -->

<!-- GSD:stack-start source:codebase/STACK.md -->

## Technology Stack

- Python package project with `cosyvoice/` and `fastcosyvoice/` packages.
- `pyproject.toml` targets Python >=3.10,<3.12 and uv; dependencies include PyTorch 2.7+, CUDA 12.8 indexes, ONNX Runtime and TensorRT-LLM.
- `uv.lock` records the resolved environment; GPU acceleration is an explicit project constraint.

<!-- GSD:stack-end -->

<!-- GSD:conventions-start source:CONVENTIONS.md -->

## Conventions

Conventions not yet established. Will populate as patterns emerge during development.
<!-- GSD:conventions-end -->

<!-- GSD:architecture-start source:ARCHITECTURE.md -->

## Architecture

- `fastcosyvoice/` contains optimized model, frontend and CosyVoice orchestration modules.
- `cosyvoice/` supplies the upstream-compatible CLI, dataset and export/training support.
- Root `run_*.py`, `benchmark_llm.py` and `webui.py` are user-facing inference/benchmark surfaces.
- Architecture is inferred from repository layout and imports; runtime behavior is not claimed verified.

<!-- GSD:architecture-end -->

<!-- GSD:skills-start source:skills/ -->

## Project Skills

No project skills found. Add skills to any of: `.claude/skills/`, `.agents/skills/`, `.cursor/skills/`, `.github/skills/`, or `.codex/skills/` with a `SKILL.md` index file.
<!-- GSD:skills-end -->

<!-- GSD:workflow-start source:GSD defaults -->

## GSD Workflow Enforcement

Before using Edit, Write, or other file-changing tools, start work through a GSD command so planning artifacts and execution context stay in sync.

Use these entry points:

- `$gsd-quick` for small fixes, doc updates, and ad-hoc tasks
- `$gsd-debug` for investigation and bug fixing
- `$gsd-execute-phase` for planned phase work

Do not make direct repo edits outside a GSD workflow unless the user explicitly asks to bypass it.
<!-- GSD:workflow-end -->

<!-- GSD:profile-start -->

## Developer Profile

> Profile not yet configured. Run `$gsd-profile-user` to generate your developer profile.
> This section is managed by `generate-claude-profile` -- do not edit manually.
<!-- GSD:profile-end -->
