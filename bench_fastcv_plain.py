"""FastCosyVoice without any stress markup: the model places stress on its own.

Every input is an explicit argument; importing or ``--help`` touches nothing.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence

from bench_fastcv_cli import BenchSpec, CaseText
from bench_fastcv_io import Runner
from bench_fastcv_run import BackendFactory, cli_main

TEST_TEXT = (
    "Это первая глава учебного курса про искусственный интеллект. "
    "Авторы рассказывают о том, как нейросети помогают в повседневных задачах."
)
TEST_TEXT_LONG = (
    "Знаешь, я недавно читал, что современные нейросети способны разобраться "
    "даже в таких сложных темах, как квантовая механика или биоинформатика. "
    "Но есть нюанс: чтобы получить качественный ответ, нужно уметь правильно "
    "сформулировать вопрос. Это иногда сложнее, чем может показаться."
)

SPEC = BenchSpec(
    description=__doc__.splitlines()[0] if __doc__ else "",
    cases=(CaseText("short", TEST_TEXT), CaseText("long", TEST_TEXT_LONG)),
    suffix="_PLAIN",
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
