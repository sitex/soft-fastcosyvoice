"""TorchBackend boundary with fake torch/fastcosyvoice modules; nothing real is imported."""

from __future__ import annotations

import importlib
import sys
import types
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Protocol, cast

import pytest
from conftest import SCRIPT_DIR  # pyright: ignore[reportImplicitRelativeImport]

EXPECTED_KWARGS: dict[str, object] = {
    "fp16": True,
    "load_trt": True,
    "load_trt_llm": False,
    "llm_model_name": "llm.pt",
}


@dataclass
class Qwen:
    model: str


@dataclass
class Mid:
    model: Qwen


@dataclass
class Wrapper:
    llm: Mid


@dataclass
class Top:
    llm: Wrapper


@dataclass
class FakeCosy:
    """Stands in for FastCosyVoice3; records constructor and stream arguments."""

    instances: ClassVar[list[FakeCosy]] = []
    kwargs: dict[str, object] = field(default_factory=dict)
    sample_rate: int = 22050
    spk: list[tuple[str, str, str]] = field(default_factory=list)
    stream_kwargs: list[dict[str, object]] = field(default_factory=list)
    model: Top = field(default_factory=lambda: Top(Wrapper(Mid(Qwen("qwen")))))

    @classmethod
    def build(cls, **kwargs: object) -> FakeCosy:
        inst = cls(kwargs=kwargs)
        cls.instances.append(inst)
        return inst

    def add_zero_shot_spk(self, prompt_text: str, wav: str, spk_id: str) -> None:
        self.spk.append((prompt_text, wav, spk_id))

    def inference_zero_shot_stream(self, **kwargs: object) -> Iterator[bytes]:
        self.stream_kwargs.append(kwargs)
        yield b"\x01\x00"


class BackendLike(Protocol):
    sample_rate: int

    def add_speaker(self, prompt_text: str, wav: str, spk_id: str) -> None: ...

    def stream(self, text: str, prompt_text: str, wav: str, spk_id: str) -> Iterator[bytes]: ...

    def synchronize(self) -> None: ...


class TorchModule(Protocol):
    __file__: str
    TorchBackend: Callable[[Path, bool], BackendLike]


class Ctx:
    def __init__(self, events: list[str]) -> None:
        self.events: list[str] = events

    def __enter__(self) -> None:
        self.events.append("enter")

    def __exit__(self, *_a: object) -> None:
        self.events.append("exit")


def make_fake_torch(events: list[str]) -> types.ModuleType:
    def compile_model(model: str, mode: str) -> str:
        return f"compiled-{model}-{mode}"

    def sync() -> None:
        events.append("sync")

    def available() -> bool:
        return True

    def inference_mode() -> AbstractContextManager[None]:
        return Ctx(events)

    mod = types.ModuleType("torch")
    mod.__dict__.update(
        inference_mode=inference_mode,
        compile=compile_model,
        cuda=types.SimpleNamespace(is_available=available, synchronize=sync),
    )
    return mod


def test_torch_backend_boundary_uses_fake_modules(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    events: list[str] = []
    fake_fcv = types.ModuleType("fastcosyvoice")
    fake_fcv.__dict__["FastCosyVoice3"] = FakeCosy.build
    FakeCosy.instances.clear()
    for name in ("bench_fastcv_torch",):
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setitem(sys.modules, "torch", make_fake_torch(events))
    monkeypatch.setitem(sys.modules, "fastcosyvoice", fake_fcv)
    monkeypatch.setattr(sys, "path", [str(SCRIPT_DIR), *sys.path])
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    mod = cast(TorchModule, cast(object, importlib.import_module("bench_fastcv_torch")))
    backend = mod.TorchBackend(tmp_path, True)
    expected = str(Path(mod.__file__).resolve().parent / "third_party" / "Matcha-TTS")
    assert expected in sys.path
    assert "third_party/Matcha-TTS" not in sys.path
    cosy = FakeCosy.instances[-1]
    assert cosy.kwargs == {"model_dir": str(tmp_path), **EXPECTED_KWARGS}
    assert cosy.model.llm.llm.model.model == "compiled-qwen-default"
    assert backend.sample_rate == 22050

    backend.add_speaker("P", "w.wav", "va")
    assert cosy.spk == [("P", "w.wav", "va")]
    assert list(backend.stream("T", "P", "w.wav", "va")) == [b"\x01\x00"]
    assert cosy.stream_kwargs == [
        {
            "tts_text": "T",
            "prompt_text": "P",
            "prompt_wav": "w.wav",
            "zero_shot_spk_id": "va",
            "temperature": 0.8,
            "top_p": 0.95,
            "top_k": 25,
        }
    ]
    assert events == ["enter", "exit"]
    backend.synchronize()
    assert events[-1] == "sync"

    _plain = mod.TorchBackend(tmp_path, False)
    assert FakeCosy.instances[-1].model.llm.llm.model.model == "qwen"
