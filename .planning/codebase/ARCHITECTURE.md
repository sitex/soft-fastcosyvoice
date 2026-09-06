# Architecture

- `fastcosyvoice/` contains optimized model, frontend and CosyVoice orchestration modules.
- `cosyvoice/` supplies the upstream-compatible CLI, dataset and export/training support.
- Root `run_*.py`, `benchmark_llm.py` and `webui.py` are user-facing inference/benchmark surfaces.
- Architecture is inferred from repository layout and imports; runtime behavior is not claimed verified.
