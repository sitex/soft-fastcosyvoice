# Structure

- Root: `run_*.py`, benchmark scripts, `README.md`, `PROJECT_GOAL.md`, `pyproject.toml`, `uv.lock`, `cosyvoice/`, `fastcosyvoice/`, `runtime/`, `tests/`.
- Optimized implementation is under `fastcosyvoice/`; upstream-compatible support remains under `cosyvoice/`.
- Existing untracked `bench_fastcv_plain.py`, `bench_fastcv_ru.py`, `src/`, and `tests/` were pre-existing and were not modified.
- No `scripts/verify` exists.
