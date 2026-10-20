"""The refusal-direction fit-prompts stand-in: an invented upstream of four tiny split files and the manifest that pins it.

The real instructions are re-drawn at run time from andyrdt/refusal_direction (``sisteer.data.arditi_fit_prompts``) and are
never in this repository, so every test draws from this stand-in instead: invented, benign rows in upstream's layout (a JSON
list of ``{"instruction", "category"}`` rows, non-ASCII included so ``ensure_ascii=False`` matters), and a manifest of the real
one's shape built for them. The expected sets and document are computed HERE, independently of the module, the way
upstream's sampler draws (module-level ``random.seed`` then ``random.sample`` per set in the recorded order), so the module's
draw is checked against a second implementation. ``serve`` is a fake network over the stand-in's URLs.
"""
from __future__ import annotations

import copy
import hashlib
import io
import json
import random
import urllib.error
from pathlib import Path

REPO = "example-owner/example-upstream"
COMMIT = "1234567890abcdef1234567890abcdef12345678"
SEED = 7
ORDER = ["harmful_train", "harmless_train", "harmful_val", "harmless_val"]
NAMES = ["harmful_train", "harmful_val", "harmless_train", "harmless_val"]
N_TRAIN, N_VAL = 5, 2
ROWS = {"harmful_train": 11, "harmful_val": 4, "harmless_train": 13, "harmless_val": 6}


def _sha(data: bytes | str) -> str:
    return hashlib.sha256(data.encode("utf-8") if isinstance(data, str) else data).hexdigest()


def n_of(name: str) -> int:
    return N_VAL if name.endswith("_val") else N_TRAIN


def split_bytes(name: str) -> bytes:
    """One invented split file, upstream's layout."""
    side = "contrast" if name.startswith("harmful") else "everyday"
    rows = [{"instruction": f"Stand-in {side} instruction {i} of {name}, café ☃", "category": "stand-in"} for i in range(ROWS[name])]
    return json.dumps(rows, indent=4, ensure_ascii=False).encode("utf-8")


def expected_sets() -> dict[str, list[str]]:
    """Upstream's sampler, written out independently of the module: ONE module-level seeding, then a sample per set in ORDER
    (the interpreter's global generator is restored afterwards)."""
    pops = {n: [r["instruction"] for r in json.loads(split_bytes(n))] for n in NAMES}
    state = random.getstate()
    try:
        random.seed(SEED)
        return {n: random.sample(pops[n], n_of(n)) for n in ORDER}
    finally:
        random.setstate(state)


def fields() -> dict:
    """The document's non-prompt fields, in the real document's shape."""
    sets = expected_sets()
    return {
        "schema": "arditi_fit_prompts/1",
        "upstream": f"https://github.com/{REPO}",
        "upstream_commit": COMMIT,
        "paper": "A stand-in paper",
        "sampling": {"source": "pipeline/run_pipeline.py::load_and_sample_datasets", "seed": SEED, "n_train": N_TRAIN, "n_val": N_VAL,
                     "draw_order": list(ORDER), "note": "a stand-in note"},
        "source_sha256": {f"splits/{n}.json": _sha(split_bytes(n)) for n in NAMES},
        "source_rows": dict(ROWS),
        "runtime_filter": "a stand-in note on the fit-time filter",
        "prompts_sha256": {n: _sha(json.dumps(sets[n], ensure_ascii=False)) for n in NAMES},
    }


def expected_document() -> bytes:
    doc = fields()
    doc["prompts"] = expected_sets()
    return (json.dumps(doc, indent=1, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def manifest() -> dict:
    """The stand-in's manifest, the real manifest's shape."""
    data = expected_document()
    f = fields()
    return {
        "schema": "arditi_fit_prompts_manifest/1",
        "about": "a stand-in manifest",
        "document": {"path": "data/arditi/fit_prompts.json", "sha256": _sha(data), "bytes": len(data), "serialisation": "stand-in",
                     "fields": f},
        "upstream": {"repo": REPO, "commit": COMMIT, "licence": "stand-in"},
        "sources": {n: {"path": f"dataset/splits/{n}.json", "sha256": _sha(split_bytes(n)), "bytes": len(split_bytes(n)),
                        "git_blob": hashlib.sha1(b"blob %d\0" % len(split_bytes(n)) + split_bytes(n)).hexdigest(), "rows": ROWS[n]}
                    for n in NAMES},
        "draw": {"seed": SEED, "order": list(ORDER), "n": {n: n_of(n) for n in ORDER}, "procedure": "stand-in"},
        "sets": {n: {"n": n_of(n), "sha256": f["prompts_sha256"][n]} for n in NAMES},
    }


def write_manifest(path: Path, m: dict | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(m if m is not None else manifest(), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def mutated(edit) -> dict:
    """A deep copy of the stand-in manifest with ``edit(m)`` applied."""
    m = copy.deepcopy(manifest())
    edit(m)
    return m


def raw_url(name: str) -> str:
    return f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/dataset/splits/{name}.json"


def swh_url(name: str) -> str:
    return f"https://archive.softwareheritage.org/api/1/content/sha1_git:{manifest()['sources'][name]['git_blob']}/raw/"


def name_of(url: str) -> str:
    for n in NAMES:
        if url in (raw_url(n), swh_url(n)):
            return n
    raise AssertionError(f"not a stand-in URL: {url}")


class Response(io.BytesIO):
    def __init__(self, body: bytes, url: str):
        super().__init__(body)
        self._url = url

    def geturl(self) -> str:
        return self._url


def serve(log: list, mode: str = "good"):
    """A fake network over the stand-in's URLs: ``good`` serves each file, ``tampered`` a changed copy, ``down`` nothing,
    ``github_down`` only Software Heritage; every URL asked is appended to ``log``."""
    def _open(url: str):
        log.append(url)
        if mode == "down" or (mode == "github_down" and url.startswith("https://raw.githubusercontent.com/")):
            raise urllib.error.URLError("no route (test)")
        body = split_bytes(name_of(url))
        if mode == "tampered":                       # the same size, one byte changed: the sha256 check is what refuses it
            body = body[:-1] + (b"X" if body[-1:] != b"X" else b"Y")
        return Response(body, url)
    return _open


def refuse_network(url: str):
    raise AssertionError(f"the network was reached: {url}")
