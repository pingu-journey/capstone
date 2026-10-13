"""Thread-local capture of actually used excerpts, persisted outside graph State."""

from contextlib import contextmanager
from contextvars import ContextVar
from hashlib import sha256

from tools.artifacts import read_json, run_dir, write_json

_CURRENT = ContextVar("research_evidence", default=None)


@contextmanager
def capture():
    records = []
    token = _CURRENT.set(records)
    try:
        yield records
    finally:
        _CURRENT.reset(token)


def record(reference: dict, text: str, page: int | None = None):
    records = _CURRENT.get()
    if records is not None and text.strip():
        item = {"source_id": reference["id"], "startup": reference["startup"],
                "text": text, "page": page, "url": reference.get("url"),
                "reference": reference}
        if item not in records:
            records.append(item)


def save_node(state, node, update, records):
    refs = {(ref["id"], ref["startup"]) for ref in update.get("references", [])}
    selected = [item for item in records if (item["source_id"], item["startup"]) in refs]
    name = node + "-" + sha256(str(state.get("current_startup") or state.get("search_round", 0)).encode()).hexdigest()[:12]
    write_json(run_dir(state) / "evidence" / f"{name}.json", selected)


def with_evidence(node, function):
    def run(state):
        if not state.get("run_id"):
            return function(state)
        with capture() as records:
            update = function(state)
        save_node(state, node, update, records)
        return update
    return run


def index_evidence(state):
    directory = run_dir(state)
    records = []
    for path in sorted((directory / "evidence").glob("*.json")):
        records.extend(read_json(path))
    for path in sorted((directory / "workers").glob("*.json")):
        records.extend(read_json(path).get("evidence", []))
    unique = {}
    for item in records:
        key = (item["source_id"], item["startup"], item.get("page"), item["text"])
        unique[key] = item
    return write_json(directory / "evidence_index.json", list(unique.values()))
