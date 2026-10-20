"""Golden-fixture compare-or-write for the tinycap identity pins under
``tests/fixtures/tinycap_identity_v1/``.

NOT a test module (no ``test_`` prefix) — imported by ``test_served_capability``
(a standard run's hashed identity, ``harness_config_hash``, cell-key dir and
manifest) and ``test_tinycap_launcher`` (every non-MATH configuration's
resolved geometry, both passes). The fixtures in the tree were captured from
the pre-change driver (main f90557fd); the two identity fixtures were then
re-captured ONCE at main 7f90bcb5 after PR #20 (83345d3a) changed the client
lock, whose sha256 is a hashed identity leaf (the geometry fixture is
unmoved). Rewriting one is a deliberate identity change that re-keys every
store of that shape, never maintenance — prove the leaf-by-leaf departure
before re-capturing.

All three were re-captured when the tiny loglikelihood battery and its
third-party harness were removed. The identity fixtures moved in the leaves
``test_tinycap_identity_version``'s docstring lists. The geometry fixture lost
the removed family's three configurations and gained the per-task case's
replacement (``gsm8k,humaneval_plus``). The four default-battery configurations
dropped ``tiny_ll`` from their controls and so hold one geometry key each: pass
1 is unchanged, and pass 2 now resolves that key's measured window where the
two-key battery refused. The other 11 configurations are byte-identical.
"""

import json
import os
from pathlib import Path
from typing import Mapping

import pytest

GOLDEN_DIR = Path(__file__).resolve().parent / "fixtures" / "tinycap_identity_v1"
#: Set to ``1`` to (re)write a golden fixture from the CURRENT code. The run
#: then FAILS on purpose, so a capture can never pass as a green compare.
GOLDEN_WRITE_ENV = "TINYCAP_IDENTITY_GOLDEN_WRITE"


def normalise(obj, *, tmp_path: Path, volatile: Mapping[str, object]):
    """``obj`` with its VOLATILE facts replaced — every string under the test's
    ``tmp_path`` (``"<TMP>"``) and every leaf whose KEY is in ``volatile`` (its
    placeholder) — so a golden compare holds everything else byte for byte.
    Each golden set passes the table its fixtures were captured under; the
    tables are the sets' own and are never merged."""
    tmp = str(tmp_path)

    def walk(o):
        if isinstance(o, dict):
            return {k: (volatile[k] if k in volatile else walk(v)) for k, v in o.items()}
        if isinstance(o, list):
            return [walk(v) for v in o]
        if isinstance(o, str) and tmp in o:
            return o.replace(tmp, "<TMP>")
        return o

    return walk(obj)


def leaf_diffs(got, want, path: str = "") -> list:
    """The dotted paths of every LEAF that differs (either side missing counts),
    so a mismatch names the field that moved rather than the block."""
    if isinstance(got, dict) and isinstance(want, dict):
        out = []
        for k in sorted(set(got) | set(want)):
            sub = f"{path}.{k}" if path else str(k)
            if k not in got or k not in want:
                out.append(sub)
            else:
                out.extend(leaf_diffs(got[k], want[k], sub))
        return out
    return [] if got == want else [path or "<root>"]


def compare_or_write(path: Path, got: dict) -> None:
    """Compare ``got`` to the fixture at ``path`` — or, under
    :data:`GOLDEN_WRITE_ENV` = ``1``, write it there and fail. A mismatch
    names the differing leaves, so the reader sees WHICH part of a store's
    key (or window) moved."""
    text = json.dumps(got, indent=1, sort_keys=True) + "\n"
    if os.environ.get(GOLDEN_WRITE_ENV) == "1":
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        pytest.fail(f"golden fixture WRITTEN to {path} — re-run without {GOLDEN_WRITE_ENV}=1")
    assert path.is_file(), f"missing golden fixture {path} (capture it with {GOLDEN_WRITE_ENV}=1)"
    want = json.loads(path.read_text())
    if got != want:
        pytest.fail(f"{path.name}: differs from the golden fixture at {leaf_diffs(got, want)[:12]}")
