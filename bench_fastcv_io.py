"""Filesystem, ffmpeg and adapter boundaries for the bench scripts (runtime only)."""

from __future__ import annotations

import importlib.util
import secrets
import subprocess
import wave
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from bench_fastcv_cli import Options

FFMPEG_TIMEOUT_S = 300
PRIVATE_DIR_MODE = 0o700

Runner = Callable[..., "subprocess.CompletedProcess[bytes]"]
TextPrep = Callable[[str], str]


class InputError(Exception):
    """Bad or unsafe caller-supplied input; reported as exit status 2."""


class SynthesisError(Exception):
    """A synthesis, conversion or adapter step did not produce a usable result."""


@dataclass(frozen=True)
class Voice:
    name: str
    wav: Path
    text: str


def read_utf8(path: Path, what: str) -> str:
    try:
        text = path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError) as exc:
        raise InputError(f"cannot read {what}: {exc}") from exc
    if not text:
        raise InputError(f"{what} is empty")
    return text


def load_inputs(options: Options) -> tuple[list[Voice], str | None]:
    if not options.model_dir.is_dir():
        raise InputError(f"model directory not found: {options.model_dir}")
    voices: list[Voice] = []
    for name, wav, txt in options.voices:
        if not wav.is_file():
            raise InputError(f"reference wav for {name!r} not found: {wav}")
        voices.append(Voice(name, wav, read_utf8(txt, f"reference text for {name!r}")))
    custom = read_utf8(options.text_file, "text file") if options.text_file else None
    return voices, custom


def _mkdir_private(target: Path) -> None:
    target.mkdir(mode=PRIVATE_DIR_MODE)
    target.chmod(PRIVATE_DIR_MODE)  # umask-proof


def create_run_dir(root: Path, run_name: str | None) -> Path:
    """Create a brand-new private run directory; never reuse or follow symlinks."""
    if root.is_symlink():
        raise InputError(f"output root must not be a symlink: {root}")
    if not root.is_dir():
        raise InputError(f"output root is not an existing directory: {root}")
    if run_name is not None:
        target = root / run_name
        try:
            _mkdir_private(target)
        except FileExistsError as exc:
            raise InputError(f"run directory already exists: {target}") from exc
        return target
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    for _ in range(16):
        target = root / f"run-{stamp}-{secrets.token_hex(4)}"
        try:
            _mkdir_private(target)
        except FileExistsError:
            continue
        return target
    raise InputError(f"could not create a unique run directory under {root}")


def write_wav_exclusive(path: Path, pcm: bytes, sample_rate: int) -> None:
    with open(path, "xb") as fh, wave.open(fh, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)


def convert_to_ogg(wav: Path, ffmpeg: str, runner: Runner) -> Path:
    """Convert to OGG/Opus; returns the file only if this call really produced it."""
    ogg = wav.with_suffix(".ogg")
    if ogg.exists() or ogg.is_symlink():
        raise SynthesisError(f"refusing to overwrite existing {ogg.name}")
    cmd = [
        ffmpeg,
        "-n",
        "-loglevel",
        "error",
        "-i",
        str(wav),
        "-ar",
        "48000",
        "-c:a",
        "libopus",
        "-b:a",
        "96k",
        str(ogg),
    ]
    try:
        proc = runner(cmd, check=False, capture_output=True, timeout=FFMPEG_TIMEOUT_S)
    except (OSError, subprocess.SubprocessError) as exc:
        ogg.unlink(missing_ok=True)
        raise SynthesisError(f"ffmpeg could not run: {exc}") from exc
    if proc.returncode != 0:
        ogg.unlink(missing_ok=True)
        raise SynthesisError(f"ffmpeg exited with status {proc.returncode}")
    if not ogg.is_file() or ogg.stat().st_size == 0:
        ogg.unlink(missing_ok=True)
        raise SynthesisError("ffmpeg reported success but produced no output")
    return ogg


def load_stress_adapter(path: Path) -> TextPrep:
    """Load a user-supplied adapter (prepare_text(str) -> str) at runtime only."""
    if not path.is_file():
        raise InputError(f"stress adapter not found: {path}")
    spec = importlib.util.spec_from_file_location("_bench_stress_adapter", path)
    if spec is None or spec.loader is None:
        raise SynthesisError("stress adapter is not an importable python file")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # user code: report, never hide
        raise SynthesisError(f"stress adapter failed to load: {exc!r}") from exc
    fn: object = getattr(module, "prepare_text", None)
    if not callable(fn):
        raise SynthesisError("stress adapter has no callable prepare_text")

    def prepare(text: str) -> str:
        try:
            out: object = fn(text)
        except Exception as exc:  # user code
            raise SynthesisError(f"stress adapter raised: {exc!r}") from exc
        if not isinstance(out, str) or not out.strip():
            raise SynthesisError("stress adapter returned empty or non-string text")
        return out

    return prepare
