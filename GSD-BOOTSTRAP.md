---
document: gsd-brownfield-bootstrap
project: soft-fastcosyvoice
git_root: /home/rocky/projects/soft-fastcosyvoice
github_repository: sitex/soft-fastcosyvoice
generated_at: 2026-09-06T02:03:45Z
thoughts_status: canonical baseline and plan available
github_status: repository resolved; Issues API unavailable or disabled for this fork
include_personal: false
---

# Purpose

- Optimized CosyVoice derivative for faster multilingual TTS with fp16, streaming and TensorRT/TensorRT-LLM paths. [S1][S2]

# Implemented Capabilities

- Repository contains optimized `fastcosyvoice` modules, upstream-compatible `cosyvoice` support, run examples, benchmarks and WebUI. [S1][S3]
- `pyproject.toml` defines a Python 3.10–3.11 uv project with CUDA 12.8/PyTorch/TensorRT-LLM sources. [S3]

# Current Milestone

Unspecified. The active local baseline is a reproducible smoke synthesis-to-WAV path without model download; no execution is claimed.

# Open Requirements

- Verify `run_basic.py`, `run_fast.py` and `run_offline.py` synthesis.
- Verify claimed languages/dialects, streaming, WebUI and TensorRT paths.
- Measure performance against documented RTF/TTFB claims. [S2]

# Accepted Decisions

- Existing implementation is treated as validated context for onboarding only; runtime checks remain open.
- Heavy model downloads are out of scope for this onboarding.

# Constraints

- Heavy Python/GPU/model dependencies, NVIDIA/CUDA/TensorRT compatibility and audio environment. [S3]
- Do not change remotes, Git state, or pre-existing untracked bench/src/tests files.

# Unresolved Conflicts

None identified from available sources.

# Source Thoughts

- [S4] `/home/rocky/thoughts/repos/soft-fastcosyvoice/shared/research/2026-09-06-project-baseline.md`
- [S5] `/home/rocky/thoughts/repos/soft-fastcosyvoice/shared/plans/2026-09-06-gsd-onboarding.md`

# Source Issues

None included; the Issues API is unavailable or disabled for this fork.

# Source Limitations

No model inference, audio output, performance benchmark, or external service integration was verified. GitHub repository identity resolved, but the Issues API was unavailable or disabled.

Sources: [S1] README.md; [S2] PROJECT_GOAL.md; [S3] pyproject.toml and repository layout; [S4][S5] canonical Thoughts.
