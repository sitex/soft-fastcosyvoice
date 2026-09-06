# Requirements: soft-fastcosyvoice

**Defined:** 2026-09-06
**Core Value:** Reproduce local synthesis and measure performance without treating configured optimizations as verified speed.

## v1 Requirements

### Local Runtime Baseline

- [ ] **FCV-01**: Operator can use `run_basic.py` with a locally available compatible model to produce a non-empty WAV.
- [ ] **FCV-02**: Operator can run `run_fast.py` with the same controlled input and verify that its output remains valid.
- [ ] **FCV-03**: Operator can run the supported offline and streaming paths and record their observable completion behavior.
- [ ] **FCV-04**: Operator can verify one documented language path plus WebUI or TensorRT readiness without downloading weights.
- [ ] **FCV-05**: Operator can reproduce RTF and time-to-first-byte measurements with model, prompt, warm-up, and hardware recorded.

## v2 Requirements

### Runtime Matrix

- **FCV-06**: Operator can compare PyTorch, TensorRT Flow, and TensorRT-LLM paths on compatible hardware.
- **FCV-07**: Operator can maintain a repeatable multilingual benchmark suite.

## Out of Scope

| Feature | Reason |
|---------|--------|
| Automatic checkpoint downloads | Model acquisition is operator-controlled. |
| Upstream benchmark values as local evidence | Measurements must be reproduced locally. |
| Existing untracked benchmark and stub files | Provenance and expected behavior remain unresolved. |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| FCV-01 | Phase 1 | Pending |
| FCV-02 | Phase 1 | Pending |
| FCV-03 | Phase 1 | Pending |
| FCV-04 | Phase 1 | Pending |
| FCV-05 | Phase 1 | Pending |

**Coverage:** 5 total, 5 mapped, 0 unmapped ✓

---
*Requirements defined: 2026-09-06*
*Last updated: 2026-09-06 after brownfield initialization*
