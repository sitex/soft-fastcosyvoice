"""Import/help/invalid-CLI purity and static hygiene, checked in real subprocesses."""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from conftest import HELPERS, SCRIPT_DIR, Inputs  # pyright: ignore[reportImplicitRelativeImport]

HOOK = r"""
import json, os, runpy, sys
out_path, script, mode = sys.argv[1], sys.argv[2], sys.argv[3]
sys.argv = [script] + sys.argv[4:]
events = []
active = [True]
HEAVY = ("torch", "fastcosyvoice", "cosyvoice", "tts_service", "podcastify")
def hook(ev, args):
    if not active[0]:
        return
    try:
        if ev == "os.mkdir":
            events.append(["mkdir", str(args[0])])
        elif ev == "open":
            p = str(args[0])
            m = str(args[1]) if len(args) > 1 else ""
            if p.endswith((".wav", ".txt", ".json", ".ogg")) or "podcastify" in p \
                    or any(c in m for c in "wxa+"):
                events.append(["open", p, m])
        elif ev == "import" and str(args[0]).split(".")[0] in HEAVY:
            events.append(["import", str(args[0])])
        elif ev == "socket.connect" or ev == "socket.getaddrinfo":
            events.append(["net", str(args)])
        elif ev == "subprocess.Popen":
            events.append(["subprocess", str(args[0])])
    except Exception:
        pass
sys.addaudithook(hook)
sys.path.insert(0, os.path.dirname(script))
code = 0
try:
    if mode == "import":
        import importlib
        importlib.import_module(os.path.basename(script)[:-3])
    else:
        runpy.run_path(script, run_name="__main__")
except SystemExit as e:
    code = e.code if isinstance(e.code, int) else 1
except BaseException as e:
    code = 99
    events.append(["exception", repr(e)])
active[0] = False
json.dump({"code": code, "events": events}, open(out_path, "w"))
"""


def run_audited(
    tmp: Path, script: str, mode: str, *args: str
) -> tuple[dict[str, object], subprocess.CompletedProcess[str]]:
    stub = tmp / "stubs" / "torch"
    stub.mkdir(parents=True)
    _ = (stub / "__init__.py").write_text("")
    hook = tmp / "hook.py"
    _ = hook.write_text(HOOK)
    cwd = tmp / "cwd_sub"
    cwd.mkdir()
    out = tmp / "events.json"
    env = {**os.environ, "PYTHONPATH": str(stub.parent), "PYTHONDONTWRITEBYTECODE": "1"}
    proc = subprocess.run(
        [sys.executable, str(hook), str(out), str(SCRIPT_DIR / f"{script}.py"), mode, *args],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert out.exists(), proc.stderr
    assert list(cwd.iterdir()) == []
    return json.loads(out.read_text()), proc


def test_import_has_no_side_effects(tmp_path: Path, script: str) -> None:
    data, proc = run_audited(tmp_path, script, "import")
    assert data["events"] == [], (data, proc.stderr)
    assert data["code"] == 0


def test_help_has_no_side_effects(tmp_path: Path, script: str) -> None:
    data, proc = run_audited(tmp_path, script, "run", "--help")
    assert data["events"] == [], (data, proc.stderr)
    assert data["code"] == 0
    for flag in ("--model-dir", "--voice", "--output-root"):
        assert flag in proc.stdout


@pytest.mark.parametrize("bad", [[], ["--model-dir", "x"], ["--bogus"]])
def test_invalid_cli_has_no_side_effects(tmp_path: Path, script: str, bad: list[str]) -> None:
    data, proc = run_audited(tmp_path, script, "run", *bad)
    assert data["events"] == [], (data, proc.stderr)
    assert data["code"] == 2


def test_invalid_voice_name_rejected_without_io(tmp_path: Path, script: str) -> None:
    ins = Inputs(tmp_path / "in")
    argv = [
        "--model-dir",
        str(ins.model),
        "--output-root",
        str(ins.root),
        "--voice",
        "../evil",
        str(ins.voices[0][1]),
        str(ins.voices[0][2]),
    ]
    data, _ = run_audited(tmp_path / "h", script, "run", *argv)
    assert data["code"] == 2
    assert data["events"] == []
    assert ins.run_dirs() == []


@pytest.mark.parametrize(
    "flag,value", [("--run-name", "ok\n"), ("--run-name", "a/b"), ("--run-name", "..")]
)
def test_unsafe_run_name_rejected_without_io(
    tmp_path: Path, script: str, flag: str, value: str
) -> None:
    ins = Inputs(tmp_path / "in")
    data, _ = run_audited(tmp_path / "h", script, "run", *ins.argv(flag, value))
    assert data["code"] == 2 and data["events"] == []
    assert ins.run_dirs() == []


def test_voice_name_with_trailing_newline_rejected(tmp_path: Path, script: str) -> None:
    ins = Inputs(tmp_path / "in")
    argv = [
        "--model-dir",
        str(ins.model),
        "--output-root",
        str(ins.root),
        "--voice",
        "va\n",
        str(ins.voices[0][1]),
        str(ins.voices[0][2]),
    ]
    data, _ = run_audited(tmp_path / "h", script, "run", *argv)
    assert data["code"] == 2 and data["events"] == []
    assert ins.run_dirs() == []


def test_no_hardcoded_external_project_paths(script: str) -> None:
    files = [SCRIPT_DIR / f"{n}.py" for n in (script, *HELPERS)]
    for path in files:
        if not path.exists():
            continue
        src = path.read_text(encoding="utf-8")
        for needle in ("/home/rocky", "podcastify", "alexey", "maria", "Роббинс"):
            assert needle not in src, f"{path.name} still contains {needle!r}"


def test_sources_parse_as_python310(script: str) -> None:
    for name in (script, *HELPERS):
        path = SCRIPT_DIR / f"{name}.py"
        if path.exists():
            _ = ast.parse(path.read_text(encoding="utf-8"), feature_version=(3, 10))
