"""Shared fixtures: fake model backend, fake ffmpeg and generated (synthetic) inputs.

The suite targets the scripts in the repository root by default. Set
BENCH_SCRIPT_DIR (together with BENCH_IN_SANDBOX=1) to run the same suite
against preserved originals; that must only happen inside the bwrap harness.
"""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
import types
import wave
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Protocol, cast

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT_DIR = Path(os.environ.get("BENCH_SCRIPT_DIR", str(ROOT)))
SCRIPTS = ("bench_fastcv_plain", "bench_fastcv_ru")
HELPERS = ("bench_fastcv_cli", "bench_fastcv_io", "bench_fastcv_run", "bench_fastcv_torch")
EXPECTED_CASES = {"bench_fastcv_plain": 4, "bench_fastcv_ru": 5}


if "BENCH_SCRIPT_DIR" in os.environ and os.environ.get("BENCH_IN_SANDBOX") != "1":
    pytest.exit("BENCH_SCRIPT_DIR runs originals: only inside the bwrap harness", 3)


class BenchModule(Protocol):
    def main(
        self,
        argv: Sequence[str] | None = None,
        *,
        backend_factory: Callable[[Path, bool], FakeBackend] | None = None,
        ffmpeg_runner: Callable[..., subprocess.CompletedProcess[bytes]] | None = None,
    ) -> int: ...


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if "script" in metafunc.fixturenames:
        metafunc.parametrize("script", SCRIPTS)


@pytest.fixture
def load(monkeypatch: pytest.MonkeyPatch) -> Callable[[str], BenchModule]:
    """Import a script fresh, with a stub torch so no real torch is ever loaded."""

    def _load(name: str) -> BenchModule:
        for mod in [*SCRIPTS, *HELPERS]:
            monkeypatch.delitem(sys.modules, mod, raising=False)
        stub = types.ModuleType("torch")
        monkeypatch.setitem(sys.modules, "torch", stub)
        monkeypatch.setattr(sys, "path", [str(SCRIPT_DIR), *sys.path])
        try:
            return cast(BenchModule, cast(object, importlib.import_module(name)))
        except BaseException as exc:  # report any import-time failure
            pytest.fail(f"import of {name} failed: {type(exc).__name__}: {exc}")

    return _load


Behaviour = Callable[[str], list[bytes]]


def default_behaviour(_text: str) -> list[bytes]:
    return [b"\x01\x00" * 100, b"\x02\x00" * 50]


class FakeBackend:
    sample_rate: int = 24000

    def __init__(self, behaviour: Behaviour | None = None) -> None:
        self.behaviour: Behaviour = behaviour or default_behaviour
        self.speakers: list[tuple[str, str, str]] = []
        self.calls: list[tuple[str, str, str, str]] = []

    def add_speaker(self, prompt_text: str, wav: str, spk_id: str) -> None:
        self.speakers.append((prompt_text, wav, spk_id))

    def stream(self, text: str, prompt_text: str, wav: str, spk_id: str) -> Iterator[bytes]:
        self.calls.append((text, prompt_text, wav, spk_id))
        yield from self.behaviour(text)

    def synchronize(self) -> None:
        return None


class Factory:
    def __init__(self, backend: FakeBackend) -> None:
        self.backend: FakeBackend = backend
        self.loaded: list[Path] = []

    def __call__(self, model_dir: Path, compile_llm: bool) -> FakeBackend:
        self.loaded.append(model_dir)
        return self.backend


class MakeFactory(Protocol):
    def __call__(self, behaviour: Behaviour | None = None) -> Factory: ...


@pytest.fixture
def make_factory() -> MakeFactory:
    def _make(behaviour: Behaviour | None = None) -> Factory:
        return Factory(FakeBackend(behaviour))

    return _make


def ok_ffmpeg(cmd: list[str], **_: object) -> subprocess.CompletedProcess[bytes]:
    _n = Path(cmd[-1]).write_bytes(b"OggS-fake")
    return subprocess.CompletedProcess(cmd, 0, b"", b"")


class Inputs:
    def __init__(self, base: Path) -> None:
        base.mkdir()
        self.base: Path = base
        self.model: Path = base / "model"
        self.root: Path = base / "out"
        self.model.mkdir()
        self.root.mkdir()
        self.voices: list[tuple[str, Path, Path]] = []
        for name, text in (("va", "Синтетический опорный текст А."), ("vb", "Synthetic ref B.")):
            wav = base / f"{name}.wav"
            with wave.open(str(wav), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(b"\x00\x00" * 160)
            txt = base / f"{name}.txt"
            _n = txt.write_text(text + "\n", encoding="utf-8")
            self.voices.append((name, wav, txt))

    def argv(self, *extra: str, voices: bool = True) -> list[str]:
        out = ["--model-dir", str(self.model), "--output-root", str(self.root)]
        if voices:
            for name, wav, txt in self.voices:
                out += ["--voice", name, str(wav), str(txt)]
        return [*out, *extra]

    def run_dirs(self) -> list[Path]:
        return sorted(p for p in self.root.iterdir())

    def snapshot(self) -> dict[str, bytes]:
        return {
            str(p.relative_to(self.base)): p.read_bytes()
            for p in sorted(self.base.rglob("*"))
            if p.is_file()
        }


@pytest.fixture
def inputs(tmp_path: Path) -> Inputs:
    return Inputs(tmp_path / "in")


@pytest.fixture(autouse=True)
def _chdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
