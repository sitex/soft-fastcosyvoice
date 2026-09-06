# Brownfield onboarding summary

- Repository: `soft-fastcosyvoice`
- Mode: fast brownfield onboarding
- Sources: canonical shared baseline, `GSD-BOOTSTRAP.md`, repository README, project goal, and package metadata
- Codebase map: fast map complete (`STACK.md`, `INTEGRATIONS.md`, `ARCHITECTURE.md`, `STRUCTURE.md`)
- Existing capabilities: optimized CosyVoice inference, streaming, TensorRT paths, WebUI, and benchmarks are present; model inference and performance claims are not verified
- Active milestone: v1 reproducible local synthesis and performance smoke baseline without downloading models
- Active requirements: FCV-01 through FCV-05 map to Phase 1
- Next action: plan and run the `run_basic.py` synthesis-to-WAV smoke with an already available model
- Verification gap: no `scripts/verify`; no model loading, audio generation, or RTF measurement performed during onboarding
