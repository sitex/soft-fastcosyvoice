"""Pure CLI layer for the bench scripts: specs, options and argument validation.

Parsing performs no filesystem, network or model access.
"""

from __future__ import annotations

import argparse
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

DEFAULT_INSTRUCTION = "You are a helpful assistant."
END_OF_PROMPT = "<|endofprompt|>"
NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}")


@dataclass(frozen=True)
class CaseText:
    kind: str
    text: str
    first_voice_only: bool = False


@dataclass(frozen=True)
class BenchSpec:
    description: str
    cases: tuple[CaseText, ...]
    suffix: str = ""
    with_adapter: bool = False


@dataclass(frozen=True)
class Options:
    model_dir: Path
    voices: tuple[tuple[str, Path, Path], ...]
    output_root: Path
    run_name: str | None
    text_file: Path | None
    instruction: str
    ffmpeg: str
    warmup: bool
    compile_llm: bool
    stress_adapter: Path | None


class _Args(argparse.Namespace):
    """Typed view of the parsed arguments (defaults are overwritten by argparse)."""

    model_dir: Path = Path()
    voice: list[list[str]] = []
    output_root: Path = Path()
    run_name: str | None = None
    text_file: Path | None = None
    instruction: str = DEFAULT_INSTRUCTION
    ffmpeg: str = "ffmpeg"
    no_warmup: bool = False
    no_compile: bool = False
    stress_adapter: Path | None = None


def is_safe_name(name: str) -> bool:
    return NAME_RE.fullmatch(name) is not None


def build_parser(spec: BenchSpec) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=spec.description)
    _ = p.add_argument("--model-dir", required=True, type=Path, help="pretrained model directory")
    _ = p.add_argument(
        "--voice",
        action="append",
        nargs=3,
        required=True,
        type=str,
        metavar=("NAME", "REF_WAV", "REF_TXT"),
        help="reference voice: name, reference wav, file with its transcript (repeatable)",
    )
    _ = p.add_argument(
        "--output-root",
        required=True,
        type=Path,
        help="existing directory; each run creates its own unique subdirectory",
    )
    _ = p.add_argument("--run-name", help="explicit subdirectory name (refused if it exists)")
    _ = p.add_argument(
        "--text-file",
        type=Path,
        help="UTF-8 text to synthesize instead of the built-in synthetic texts",
    )
    _ = p.add_argument(
        "--instruction",
        default=DEFAULT_INSTRUCTION,
        help="instruction prefix placed before " + END_OF_PROMPT,
    )
    _ = p.add_argument("--ffmpeg", default="ffmpeg", help="ffmpeg executable for WAV to OGG")
    _ = p.add_argument("--no-warmup", action="store_true", help="skip warm-up passes")
    _ = p.add_argument("--no-compile", action="store_true", help="skip torch.compile of the LLM")
    if spec.with_adapter:
        _ = p.add_argument(
            "--stress-adapter",
            type=Path,
            help="python file exposing prepare_text(text) -> str; loaded only at run time",
        )
    return p


def parse_options(spec: BenchSpec, argv: Sequence[str] | None) -> Options:
    parser = build_parser(spec)
    ns = parser.parse_args(argv, namespace=_Args())
    seen: set[str] = set()
    voices: list[tuple[str, Path, Path]] = []
    for name, wav, txt in ns.voice:
        if not is_safe_name(name):
            parser.error(f"invalid voice name {name!r} (letters, digits, _ and - only)")
        if name in seen:
            parser.error(f"duplicate voice name {name!r}")
        seen.add(name)
        voices.append((name, Path(wav), Path(txt)))
    if ns.run_name is not None and not is_safe_name(ns.run_name):
        parser.error(f"invalid run name {ns.run_name!r} (letters, digits, _ and - only)")
    if not ns.instruction.strip() or END_OF_PROMPT in ns.instruction:
        parser.error("instruction must be non-empty and must not contain the end-of-prompt tag")
    return Options(
        model_dir=ns.model_dir,
        voices=tuple(voices),
        output_root=ns.output_root,
        run_name=ns.run_name,
        text_file=ns.text_file,
        instruction=ns.instruction,
        ffmpeg=ns.ffmpeg,
        warmup=not ns.no_warmup,
        compile_llm=not ns.no_compile,
        stress_adapter=ns.stress_adapter,
    )
