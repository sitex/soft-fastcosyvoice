# Roadmap: soft-fastcosyvoice

## Phases

- [ ] **Phase 1: Local Synthesis and Performance Baseline** - Verify basic and optimized paths with reproducible measurements.

## Phase Details

### Phase 1: Local Synthesis and Performance Baseline
**Goal:** The operator can reproduce valid basic and optimized output, streaming behavior, readiness, and comparable local performance evidence.
**Mode:** mvp
**Depends on:** Nothing (first phase)
**Requirements:** FCV-01, FCV-02, FCV-03, FCV-04, FCV-05
**Success Criteria:**
1. `run_basic.py` produces a non-empty WAV from a controlled input and locally available model.
2. `run_fast.py` produces a valid output from the same input without changing model or dependency state.
3. Offline and streaming entry points complete with their commands and observable results recorded.
4. One language path and one WebUI or TensorRT readiness surface pass without downloading weights.
5. RTF and time-to-first-byte results record model, prompt, warm-up, hardware, and any skipped backend.
6. A deterministic verification command separates passes, skips, and failures.
**Plans:** TBD

## Progress

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Local Synthesis and Performance Baseline | 0/TBD | Not started | - |
