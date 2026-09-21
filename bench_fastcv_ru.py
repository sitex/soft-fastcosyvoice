"""Smoke test FastCosyVoice3 on Russian with `+` stress markers and tags.

FastCosyVoice converts `+` to a combining accent itself. An optional stress
adapter (a python file with ``prepare_text(text) -> str``) may be supplied with
``--stress-adapter``; it is loaded only when a run starts, never on import/help.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence

from bench_fastcv_cli import BenchSpec, CaseText
from bench_fastcv_io import Runner
from bench_fastcv_plain import TEST_TEXT
from bench_fastcv_run import BackendFactory, cli_main

TEST_TEXT_LONG = (
    "Знаешь, я недавно читал, что современные нейросети способны разобраться "
    "даже в таких сложных темах, как квантовая механика или биоинформатика. "
    "Но есть нюанс: чтобы получить качественный ответ, нужно уметь правильно "
    "сформулировать вопрос. [breath] Это иногда сложнее, чем может показаться."
)
TAGS_TEXT = (
    "Я не мог уде+ржаться, [laughter] +это бы+ло так смешн+о! "
    "Но пот+ом я подум+ал, [sigh] что +это всё-т+аки серьёзная т+ема."
)

SPEC = BenchSpec(
    description="Smoke test FastCosyVoice3 on Russian (stress markers, tags, optional adapter).",
    cases=(
        CaseText("short", TEST_TEXT),
        CaseText("long", TEST_TEXT_LONG),
        CaseText("tags", TAGS_TEXT, first_voice_only=True),
    ),
    with_adapter=True,
)


def main(
    argv: Sequence[str] | None = None,
    *,
    backend_factory: BackendFactory | None = None,
    ffmpeg_runner: Runner | None = None,
) -> int:
    return cli_main(SPEC, argv, backend_factory, ffmpeg_runner)


if __name__ == "__main__":
    sys.exit(main())
