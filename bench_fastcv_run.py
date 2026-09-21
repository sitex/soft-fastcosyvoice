"""Runtime orchestration for the bench scripts: synthesize, convert, report."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol

from bench_fastcv_cli import END_OF_PROMPT, BenchSpec, CaseText, Options, parse_options
from bench_fastcv_io import (
    InputError,
    Runner,
    SynthesisError,
    TextPrep,
    Voice,
    convert_to_ogg,
    create_run_dir,
    load_inputs,
    load_stress_adapter,
    write_wav_exclusive,
)

WARMUP_TEXTS = (
    "Прогрев.",
    "Это тест прогрева на средней длине текста.",
    "Это тест прогрева на достаточно длинной фразе для компиляции графа разных размеров.",
)


class Backend(Protocol):
    sample_rate: int

    def add_speaker(self, prompt_text: str, wav: str, spk_id: str) -> None: ...

    def stream(self, text: str, prompt_text: str, wav: str, spk_id: str) -> Iterator[bytes]: ...

    def synchronize(self) -> None: ...


BackendFactory = Callable[[Path, bool], Backend]


@dataclass
class CaseResult:
    name: str
    status: str = "failed"
    error: str | None = None
    ttfb: float | None = None
    total: float | None = None
    duration: float | None = None
    rtf: float | None = None
    wav: str | None = None
    ogg: str | None = None
    ogg_error: str | None = None

    @property
    def good(self) -> bool:
        return self.status == "ok" and self.ogg is not None and self.ogg_error is None


@dataclass
class RunReport:
    run_dir: str
    sample_rate: int | None = None
    error: str | None = None
    cases: list[CaseResult] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.error is None and bool(self.cases) and all(c.good for c in self.cases)


def make_prompt(instruction: str, ref_text: str) -> str:
    return f"{instruction}{END_OF_PROMPT}{ref_text}"


def synthesize(
    backend: Backend, text: str, prompt_text: str, voice: Voice, out: Path
) -> CaseResult:
    t0 = time.perf_counter()
    first: float | None = None
    chunks: list[bytes] = []
    for pcm in backend.stream(text, prompt_text, str(voice.wav), voice.name):
        if not pcm:
            continue
        if first is None:
            backend.synchronize()
            first = time.perf_counter() - t0
        chunks.append(pcm)
    backend.synchronize()
    total = time.perf_counter() - t0
    data = b"".join(chunks)
    if not data:
        raise SynthesisError("synthesis produced zero audio chunks")
    if len(data) % 2:
        raise SynthesisError("PCM length is not a whole number of int16 samples")
    write_wav_exclusive(out, data, backend.sample_rate)
    duration = len(data) / 2 / backend.sample_rate
    return CaseResult(out.stem, "ok", None, first, total, duration, total / duration, out.name)


def plan_cases(
    spec: BenchSpec, voices: list[Voice], custom: str | None
) -> list[tuple[str, Voice, str]]:
    cases = (CaseText("custom", custom),) if custom else spec.cases
    return [
        (case.kind, voice, case.text)
        for voice in voices
        for case in cases
        if not (case.first_voice_only and voice is not voices[0])
    ]


def _execute(
    options: Options,
    spec: BenchSpec,
    voices: list[Voice],
    custom: str | None,
    prompts: dict[str, str],
    prep: TextPrep,
    run_dir: Path,
    report: RunReport,
    backend_factory: BackendFactory,
    ffmpeg_runner: Runner,
) -> None:
    backend = backend_factory(options.model_dir, options.compile_llm)
    report.sample_rate = backend.sample_rate
    for v in voices:
        backend.add_speaker(prompts[v.name], str(v.wav), v.name)
    if options.warmup:
        first = voices[0]
        for wt in WARMUP_TEXTS:
            for _ in backend.stream(wt, prompts[first.name], str(first.wav), first.name):
                pass
        backend.synchronize()
    for idx, (kind, voice, text) in enumerate(plan_cases(spec, voices, custom), start=1):
        stem = f"{idx:02d}_{voice.name}_{kind}{spec.suffix}"
        try:
            res = synthesize(
                backend, prep(text), prompts[voice.name], voice, run_dir / f"{stem}.wav"
            )
        except Exception as exc:  # per-case failure must not hide the other cases
            report.cases.append(CaseResult(stem, error=f"{type(exc).__name__}: {exc}"))
            continue
        report.cases.append(res)
        try:
            res.ogg = convert_to_ogg(run_dir / f"{stem}.wav", options.ffmpeg, ffmpeg_runner).name
        except SynthesisError as exc:
            res.ogg_error = str(exc)


def run_bench(
    options: Options, spec: BenchSpec, *, backend_factory: BackendFactory, ffmpeg_runner: Runner
) -> int:
    """Exit status: 0 all good, 1 run started but something failed, 2 bad input."""
    try:
        voices, custom = load_inputs(options)
        prep: TextPrep = (
            load_stress_adapter(options.stress_adapter) if options.stress_adapter else str
        )
        prompts = {v.name: make_prompt(options.instruction, prep(v.text)) for v in voices}
        run_dir = create_run_dir(options.output_root, options.run_name)
    except InputError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except SynthesisError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    report = RunReport(run_dir=str(run_dir))
    print(f"run directory: {run_dir}")
    try:
        _execute(
            options,
            spec,
            voices,
            custom,
            prompts,
            prep,
            run_dir,
            report,
            backend_factory,
            ffmpeg_runner,
        )
    except Exception as exc:  # model/backend boundary: record instead of crashing
        report.error = f"{type(exc).__name__}: {exc}"
        print(f"error: {report.error}", file=sys.stderr)
    with open(run_dir / "results.json", "x", encoding="utf-8") as fh:
        json.dump({**asdict(report), "ok": report.ok}, fh, ensure_ascii=False, indent=2)
    for c in report.cases:
        note = c.error or c.ogg_error
        print(f"  {c.name}: {'ok' if c.good else 'FAILED'}" + (f" ({note})" if note else ""))
    print("OK" if report.ok else "FAILED")
    return 0 if report.ok else 1


def cli_main(
    spec: BenchSpec,
    argv: Sequence[str] | None,
    backend_factory: BackendFactory | None,
    ffmpeg_runner: Runner | None,
) -> int:
    options = parse_options(spec, argv)
    if backend_factory is None:
        from bench_fastcv_torch import TorchBackend

        backend_factory = TorchBackend
    return run_bench(
        options,
        spec,
        backend_factory=backend_factory,
        ffmpeg_runner=ffmpeg_runner or subprocess.run,
    )
