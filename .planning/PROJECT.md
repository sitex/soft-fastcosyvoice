# soft-fastcosyvoice

## What This Is

`soft-fastcosyvoice` is the personal `sitex` fork of FastCosyVoice, an optimized CosyVoice derivative for local accelerated TTS experiments. It preserves upstream inference, streaming, WebUI, benchmark, and TensorRT-oriented surfaces while local verification remains a separate milestone.

## Core Value

The operator can reproduce a local synthesis path and measure its performance without confusing configured optimizations with verified speed.

## Requirements

### Validated

- ✓ Basic, optimized, offline, streaming, WebUI, and benchmark surfaces exist in the repository — existing code and documentation
- ✓ Python, CUDA, PyTorch, TensorRT, and TensorRT-LLM constraints are recorded — existing package metadata
- ✓ The personal fork uses a dedicated `portfolio` branch and keeps the vendor repository as `upstream` — onboarding

### Active

- [ ] Produce a WAV through `run_basic.py` with a locally available compatible model.
- [ ] Verify the optimized `run_fast.py` path with the same controlled input.
- [ ] Verify offline and streaming behavior through supported entry points.
- [ ] Verify one documented language and WebUI/TensorRT readiness surface without downloading a model.
- [ ] Record reproducible RTF and time-to-first-byte measurements with hardware context.

### Out of Scope

- Downloading checkpoints during onboarding — acquisition remains operator-controlled.
- Publishing upstream performance claims as local measurements — benchmarks require local evidence.
- Committing existing untracked benchmark scripts or package/test stubs — their intent is not yet verified.

## Context

The fork originates from `Brakanier/FastCosyVoice`. The local branch contained a personal commit and was published as `origin/portfolio` without rewriting the upstream-derived default history. Source-linked evidence lives in `GSD-BOOTSTRAP.md`, `.planning/codebase/`, and canonical HumanLayer Thoughts.

## Constraints

- **Runtime**: Python 3.10–3.11 and heavyweight model/audio dependencies — environment reproduction must be explicit.
- **Hardware**: optimized paths target NVIDIA CUDA 12.8 and TensorRT-LLM — unsupported hardware checks must be skipped visibly.
- **Measurement**: RTF and TTFB depend on model, prompt, warm-up, and device state — benchmarks must record all four.
- **Verification**: no `scripts/verify` exists — Phase 1 must add a deterministic project gate.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Preserve the upstream fork boundary | Keep vendor updates distinct from portfolio changes | ✓ Good |
| Verify basic synthesis before optimized paths | Establish a comparable correctness baseline | — Pending |
| Keep untracked benchmark and stub files out of onboarding | Their provenance and expected outputs are unresolved | — Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition**:
1. Move observed runtime requirements to Validated with evidence.
2. Record hardware-specific skips and failures without converting them to passes.
3. Update constraints and decisions discovered during measurement.
4. Recheck the maintained-fork description and core value.

**After each milestone**:
1. Review requirement status and benchmark comparability.
2. Reconfirm the upstream/fork boundary.
3. Define the next evidence-backed milestone.

---
*Last updated: 2026-09-06 after brownfield initialization*
