"""Behavioural tests with fake model/ffmpeg boundaries and generated inputs."""

from __future__ import annotations

import json
import os
import stat
import struct
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest
from conftest import (  # pyright: ignore[reportImplicitRelativeImport]
    EXPECTED_CASES,
    BenchModule,
    FakeBackend,
    Inputs,
    MakeFactory,
    ok_ffmpeg,
)

Load = Callable[[str], BenchModule]


def results(run_dir: Path) -> dict[str, object]:
    return cast(
        dict[str, object], json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    )


def cases_of(rep: dict[str, object]) -> list[dict[str, object]]:
    return cast(list[dict[str, object]], rep["cases"])


def test_successful_flow(
    load: Load,
    script: str,
    inputs: Inputs,
    make_factory: MakeFactory,
    capsys: pytest.CaptureFixture[str],
) -> None:
    mod = load(script)
    fac = make_factory()
    before = inputs.snapshot()
    rc = mod.main(inputs.argv(), backend_factory=fac, ffmpeg_runner=ok_ffmpeg)
    assert rc == 0
    (run_dir,) = inputs.run_dirs()
    assert run_dir.name.startswith("run-")
    n = EXPECTED_CASES[script]
    assert len(list(run_dir.glob("*.wav"))) == n
    assert len(list(run_dir.glob("*.ogg"))) == n
    head = next(run_dir.glob("*.wav")).read_bytes()[:44]
    channels, rate = struct.unpack("<HI", head[22:24] + head[24:28])
    (bits,) = struct.unpack("<H", head[34:36])
    (data_size,) = struct.unpack("<I", head[40:44])
    assert (channels, bits, rate) == (1, 16, 24000)
    assert data_size == 300
    rep = results(run_dir)
    assert rep["ok"] is True and len(cases_of(rep)) == n
    after = inputs.snapshot()
    assert {k: v for k, v in after.items() if not k.startswith("out/")} == before
    assert list(Path.cwd().iterdir()) == []
    assert fac.loaded == [inputs.model]
    assert "OK" in capsys.readouterr().out


def test_run_directories_are_private(
    load: Load, script: str, inputs: Inputs, make_factory: MakeFactory
) -> None:
    mod = load(script)
    old = os.umask(0o000)
    try:
        _ = mod.main(inputs.argv(), backend_factory=make_factory(), ffmpeg_runner=ok_ffmpeg)
        _ = mod.main(
            inputs.argv("--run-name", "named"),
            backend_factory=make_factory(),
            ffmpeg_runner=ok_ffmpeg,
        )
    finally:
        _ = os.umask(old)
    dirs = inputs.run_dirs()
    assert len(dirs) == 2
    assert all(stat.S_IMODE(d.stat().st_mode) == 0o700 for d in dirs)


def test_two_runs_use_distinct_dirs_and_keep_prior_artifacts(
    load: Load, script: str, inputs: Inputs, make_factory: MakeFactory
) -> None:
    mod = load(script)
    assert mod.main(inputs.argv(), backend_factory=make_factory(), ffmpeg_runner=ok_ffmpeg) == 0
    first = inputs.snapshot()
    assert mod.main(inputs.argv(), backend_factory=make_factory(), ffmpeg_runner=ok_ffmpeg) == 0
    assert len(inputs.run_dirs()) == 2
    second = inputs.snapshot()
    assert all(second[k] == v for k, v in first.items())


def test_explicit_run_name_collision_refused(
    load: Load, script: str, inputs: Inputs, make_factory: MakeFactory
) -> None:
    mod = load(script)
    victim = inputs.root / "fixed"
    victim.mkdir()
    _ = (victim / "01_va_short.wav").write_bytes(b"PRIOR")
    fac = make_factory()
    rc = mod.main(inputs.argv("--run-name", "fixed"), backend_factory=fac, ffmpeg_runner=ok_ffmpeg)
    assert rc == 2
    assert [p.name for p in victim.iterdir()] == ["01_va_short.wav"]
    assert (victim / "01_va_short.wav").read_bytes() == b"PRIOR"
    assert fac.loaded == []


def test_symlink_run_name_and_root_refused(
    load: Load, script: str, inputs: Inputs, make_factory: MakeFactory, tmp_path: Path
) -> None:
    mod = load(script)
    target = tmp_path / "elsewhere"
    target.mkdir()
    (inputs.root / "linked").symlink_to(target, target_is_directory=True)
    fac = make_factory()
    assert (
        mod.main(inputs.argv("--run-name", "linked"), backend_factory=fac, ffmpeg_runner=ok_ffmpeg)
        == 2
    )
    assert list(target.iterdir()) == []
    link_root = tmp_path / "rootlink"
    link_root.symlink_to(inputs.root, target_is_directory=True)
    argv = inputs.argv()
    argv[argv.index("--output-root") + 1] = str(link_root)
    assert mod.main(argv, backend_factory=fac, ffmpeg_runner=ok_ffmpeg) == 2
    assert inputs.run_dirs() == [inputs.root / "linked"] and fac.loaded == []


def test_missing_inputs_refused_before_any_directory(
    load: Load, script: str, inputs: Inputs, make_factory: MakeFactory
) -> None:
    mod = load(script)
    fac = make_factory()
    inputs.voices[0][2].unlink()
    rc = mod.main(inputs.argv(), backend_factory=fac, ffmpeg_runner=ok_ffmpeg)
    assert rc == 2 and inputs.run_dirs() == [] and fac.loaded == []
    rc = mod.main(
        [
            "--model-dir",
            str(inputs.base / "nomodel"),
            "--output-root",
            str(inputs.root),
            "--voice",
            "va",
            str(inputs.voices[0][1]),
            str(inputs.voices[1][2]),
        ],
        backend_factory=fac,
        ffmpeg_runner=ok_ffmpeg,
    )
    assert rc == 2 and inputs.run_dirs() == []


def test_zero_audio_chunks_is_a_reported_failure(
    load: Load, script: str, inputs: Inputs, make_factory: MakeFactory
) -> None:
    mod = load(script)
    calls: list[list[str]] = []

    def spy(cmd: list[str], **kw: object) -> subprocess.CompletedProcess[bytes]:
        calls.append(cmd)
        return ok_ffmpeg(cmd, **kw)

    rc = mod.main(inputs.argv(), backend_factory=make_factory(lambda t: []), ffmpeg_runner=spy)
    assert rc == 1
    (run_dir,) = inputs.run_dirs()
    assert not list(run_dir.glob("*.wav")) and not list(run_dir.glob("*.ogg"))
    assert calls == []
    rep = results(run_dir)
    assert rep["ok"] is False
    assert all(c["status"] == "failed" for c in cases_of(rep))


@pytest.mark.parametrize("chunks", [[b"", b""], [b"\x01\x00\x02"]])
def test_empty_or_misaligned_pcm_rejected(
    load: Load, script: str, inputs: Inputs, make_factory: MakeFactory, chunks: list[bytes]
) -> None:
    mod = load(script)
    rc = mod.main(
        inputs.argv(), backend_factory=make_factory(lambda t: chunks), ffmpeg_runner=ok_ffmpeg
    )
    assert rc == 1
    (run_dir,) = inputs.run_dirs()
    assert not list(run_dir.glob("*.wav"))


def test_backend_failure_on_one_case_does_not_hide_others(
    load: Load, script: str, inputs: Inputs, make_factory: MakeFactory
) -> None:
    mod = load(script)

    def behaviour(text: str) -> list[bytes]:
        if "квантовая" in text:
            raise RuntimeError("boom")
        return [b"\x01\x00" * 10]

    rc = mod.main(
        inputs.argv("--no-warmup"), backend_factory=make_factory(behaviour), ffmpeg_runner=ok_ffmpeg
    )
    assert rc == 1
    (run_dir,) = inputs.run_dirs()
    rep = results(run_dir)
    statuses = [c["status"] for c in cases_of(rep)]
    assert "failed" in statuses and "ok" in statuses
    assert any("boom" in (str(c["error"] or "")) for c in cases_of(rep))


def test_model_load_failure_is_reported(load: Load, script: str, inputs: Inputs) -> None:
    mod = load(script)

    def broken(_model_dir: Path, _compile_llm: bool) -> FakeBackend:
        raise RuntimeError("no model")

    rc = mod.main(inputs.argv(), backend_factory=broken, ffmpeg_runner=ok_ffmpeg)
    assert rc == 1
    (run_dir,) = inputs.run_dirs()
    rep = results(run_dir)
    assert rep["ok"] is False and "no model" in str(rep["error"])


@pytest.mark.parametrize("mode", ["rc1", "silent-no-file", "empty-file", "missing-binary"])
def test_conversion_failure_never_counts_as_success(
    load: Load, script: str, inputs: Inputs, make_factory: MakeFactory, mode: str
) -> None:
    mod = load(script)

    def runner(cmd: list[str], **_: object) -> subprocess.CompletedProcess[bytes]:
        out = Path(cmd[-1])
        if mode == "missing-binary":
            raise FileNotFoundError("ffmpeg")
        if mode == "rc1":
            _n = out.write_bytes(b"partial")
            return subprocess.CompletedProcess(cmd, 1, b"", b"bad")
        if mode == "empty-file":
            _n = out.write_bytes(b"")
        return subprocess.CompletedProcess(cmd, 0, b"", b"")

    rc = mod.main(inputs.argv(), backend_factory=make_factory(), ffmpeg_runner=runner)
    assert rc == 1
    (run_dir,) = inputs.run_dirs()
    assert not list(run_dir.glob("*.ogg"))
    rep = results(run_dir)
    assert rep["ok"] is False
    assert all(c["ogg"] is None and c["ogg_error"] for c in cases_of(rep))


def test_prompt_and_instruction_variants(
    load: Load, script: str, inputs: Inputs, make_factory: MakeFactory
) -> None:
    mod = load(script)
    fac = make_factory()
    _ = mod.main(inputs.argv("--no-warmup"), backend_factory=fac, ffmpeg_runner=ok_ffmpeg)
    prompts = {p for _, p, _, _ in fac.backend.calls}
    assert "You are a helpful assistant.<|endofprompt|>Синтетический опорный текст А." in prompts
    fac2 = make_factory()
    _ = mod.main(
        inputs.argv("--no-warmup", "--instruction", "Speak calmly."),
        backend_factory=fac2,
        ffmpeg_runner=ok_ffmpeg,
    )
    assert all(p.startswith("Speak calmly.<|endofprompt|>") for _, p, _, _ in fac2.backend.calls)
    assert all(p.startswith("Speak calmly.<|endofprompt|>") for p, _, _ in fac2.backend.speakers)


def test_warmup_can_be_skipped(
    load: Load, script: str, inputs: Inputs, make_factory: MakeFactory
) -> None:
    mod = load(script)
    warm, cold = make_factory(), make_factory()
    _ = mod.main(inputs.argv(), backend_factory=warm, ffmpeg_runner=ok_ffmpeg)
    _ = mod.main(inputs.argv("--no-warmup"), backend_factory=cold, ffmpeg_runner=ok_ffmpeg)
    assert len(warm.backend.calls) == len(cold.backend.calls) + 3


def test_text_file_replaces_builtin_texts(
    load: Load, script: str, inputs: Inputs, make_factory: MakeFactory
) -> None:
    mod = load(script)
    tf = inputs.base / "custom.txt"
    _ = tf.write_text("Кастомный тест.", encoding="utf-8")
    fac = make_factory()
    rc = mod.main(
        inputs.argv("--no-warmup", "--text-file", str(tf)),
        backend_factory=fac,
        ffmpeg_runner=ok_ffmpeg,
    )
    assert rc == 0
    assert {t for t, _, _, _ in fac.backend.calls} == {"Кастомный тест."}
    assert len(fac.backend.calls) == 2


def test_ru_tags_case_keeps_markers_without_adapter(
    load: Load, inputs: Inputs, make_factory: MakeFactory
) -> None:
    mod = load("bench_fastcv_ru")
    fac = make_factory()
    _ = mod.main(inputs.argv("--no-warmup"), backend_factory=fac, ffmpeg_runner=ok_ffmpeg)
    tagged = [c for c in fac.backend.calls if "[laughter]" in c[0]]
    assert len(tagged) == 1 and tagged[0][3] == "va" and "уде+ржаться" in tagged[0][0]


def test_stress_adapter_loaded_only_on_request(
    load: Load, inputs: Inputs, make_factory: MakeFactory, tmp_path: Path
) -> None:
    mod = load("bench_fastcv_ru")
    marker = tmp_path / "adapter-loaded"
    adapter = inputs.base / "adapter.py"
    _ = adapter.write_text(
        f"open({str(marker)!r}, 'w').write('x')\n"
        + "def prepare_text(text):\n    return '<' + text + '>'\n",
        encoding="utf-8",
    )
    plain = make_factory()
    assert mod.main(inputs.argv("--no-warmup"), backend_factory=plain, ffmpeg_runner=ok_ffmpeg) == 0
    assert not marker.exists()
    with_adapter = make_factory()
    rc = mod.main(
        inputs.argv("--no-warmup", "--stress-adapter", str(adapter)),
        backend_factory=with_adapter,
        ffmpeg_runner=ok_ffmpeg,
    )
    assert rc == 0 and marker.exists()
    assert all(t.startswith("<") and t.endswith(">") for t, _, _, _ in with_adapter.backend.calls)
    assert all(
        "<Синтетический" in p or "<Synthetic" in p for _, p, _, _ in with_adapter.backend.calls
    )


@pytest.mark.parametrize(
    "body,rc",
    [
        ("raise RuntimeError('load fail')\n", 1),
        ("x = 1\n", 1),
        ("def prepare_text(t):\n    raise ValueError('bad')\n", 1),
        ("def prepare_text(t):\n    return ''\n", 1),
    ],
)
def test_stress_adapter_failures_are_honest(
    load: Load, inputs: Inputs, make_factory: MakeFactory, body: str, rc: int
) -> None:
    mod = load("bench_fastcv_ru")
    adapter = inputs.base / "adapter.py"
    _ = adapter.write_text(body, encoding="utf-8")
    fac = make_factory()
    got = mod.main(
        inputs.argv("--stress-adapter", str(adapter)), backend_factory=fac, ffmpeg_runner=ok_ffmpeg
    )
    assert got == rc
    assert fac.loaded == [] and fac.backend.calls == []
    assert inputs.run_dirs() == []


def test_missing_adapter_path_is_input_error(
    load: Load, inputs: Inputs, make_factory: MakeFactory
) -> None:
    mod = load("bench_fastcv_ru")
    fac = make_factory()
    rc = mod.main(
        inputs.argv("--stress-adapter", str(inputs.base / "none.py")),
        backend_factory=fac,
        ffmpeg_runner=ok_ffmpeg,
    )
    assert rc == 2 and fac.loaded == []
