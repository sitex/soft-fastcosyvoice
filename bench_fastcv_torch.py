"""Real FastCosyVoice3 backend. Importing this module is cheap: torch and the
model are imported only when TorchBackend is constructed (a real run)."""

from __future__ import annotations

import importlib
import sys
from collections.abc import Iterator
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Protocol, cast

MATCHA_DIR = Path(__file__).resolve().parent / "third_party" / "Matcha-TTS"


class _Cuda(Protocol):
    def is_available(self) -> bool: ...

    def synchronize(self) -> None: ...


class _Torch(Protocol):
    cuda: _Cuda

    def inference_mode(self) -> AbstractContextManager[None]: ...

    def compile(self, model: object, mode: str) -> object: ...


class _Qwen(Protocol):
    model: object


class _Llm(Protocol):
    model: _Qwen


class _LlmWrapper(Protocol):
    llm: _Llm


class _Model(Protocol):
    llm: _LlmWrapper


class _CosyVoice(Protocol):
    model: _Model
    sample_rate: int

    def add_zero_shot_spk(self, prompt_text: str, wav: str, spk_id: str) -> None: ...

    def inference_zero_shot_stream(
        self,
        *,
        tts_text: str,
        prompt_text: str,
        prompt_wav: str,
        zero_shot_spk_id: str,
        temperature: float,
        top_p: float,
        top_k: int,
    ) -> Iterator[bytes]: ...


class TorchBackend:
    def __init__(self, model_dir: Path, compile_llm: bool) -> None:
        sys.path.append(str(MATCHA_DIR))
        self._torch: _Torch = cast(_Torch, cast(object, importlib.import_module("torch")))
        factory = cast(object, importlib.import_module("fastcosyvoice").FastCosyVoice3)
        if not callable(factory):
            raise RuntimeError("fastcosyvoice.FastCosyVoice3 is not callable")
        cv: _CosyVoice = cast(
            _CosyVoice,
            factory(
                model_dir=str(model_dir),
                fp16=True,
                load_trt=True,
                load_trt_llm=False,
                llm_model_name="llm.pt",
            ),
        )
        if compile_llm:
            inner = cv.model.llm.llm.model
            inner.model = self._torch.compile(inner.model, mode="default")
        self._cv: _CosyVoice = cv
        self.sample_rate: int = cv.sample_rate

    def add_speaker(self, prompt_text: str, wav: str, spk_id: str) -> None:
        self._cv.add_zero_shot_spk(prompt_text, wav, spk_id)

    def stream(self, text: str, prompt_text: str, wav: str, spk_id: str) -> Iterator[bytes]:
        with self._torch.inference_mode():
            yield from self._cv.inference_zero_shot_stream(
                tts_text=text,
                prompt_text=prompt_text,
                prompt_wav=wav,
                zero_shot_spk_id=spk_id,
                temperature=0.8,
                top_p=0.95,
                top_k=25,
            )

    def synchronize(self) -> None:
        if self._torch.cuda.is_available():
            self._torch.cuda.synchronize()
