"""Run-scoped, atomic artifacts. No secrets or raw service exceptions are stored."""

import json
import os
import re
from pathlib import Path
from tempfile import NamedTemporaryFile

from config import OUTPUT_DIR


def run_dir(state: dict) -> Path:
    run_id = state["run_id"]
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", run_id):
        raise ValueError("유효하지 않은 run_id")
    return OUTPUT_DIR / "runs" / run_id


def write_json(path: Path, value: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, ensure_ascii=False, indent=2)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return str(path)


def read_json(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))
