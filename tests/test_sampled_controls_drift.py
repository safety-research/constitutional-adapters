"""THE VENDOR CONTRACT: main's ``sisteer/eval/sampled_controls.py`` is canonical, and the
copy of it on branch ``arditi-cell1`` is a DECLARED VENDOR that must equal it.

Why a duplicate exists on purpose. That lane runs from a tree pinned byte-identical to
the serving stack (``src/sisteer/capeval/serving/`` and ``capeval/tiers/petri_runner.py``
at ``935dfb43``), and every cell it runs asserts that identity as a production gate.
Main's ``serving/`` has since diverged by 16 files, so the lane cannot merge main
without breaking the gate on every cell — and may never make that re-pin. Rather than
leave a copy that is "going away" but structurally cannot, the lane vendors this
module: the same definitions at the same path, mirrored from main. This is the stated
design, not debt to be cleaned up.

The contract, its direction, and how it is pinned. MAIN is the one place these
definitions change; a vendor is re-mirrored from main, never the other way round. The
relationship describes itself in ``sampled_controls.VENDOR_CONTRACT`` (main's header):
which copy is canonical, the vendor and why, where the vendored copy is EXPECTED to be
found (a branch and a commit, with its state), the contract's SCOPE and what it does
NOT cover, and the CONTENT the contract is with — the sha256 of every definition of
the shared body, of the whole body from ``BODY_MARKER`` on, and of ``sampled_decode.py``.

This file asserts MAIN'S OWN body against those hashes. That is deterministic — no
network, no ref fetch — so it runs the same under a shallow exact-sha checkout as on a
full clone, and it is what keeps main green at every merge: a change to the shared body
that does not also re-mirror the vendor and update the field is red on the PR that
makes it, never on main after another branch moves. The vendor's own
``tests/test_sampled_controls_port.py`` carries the same table and asserts its copy
against it, so both sides stay pinned without either fetching the other. When a
commit named in ``expected_at`` IS reachable from this clone, one further test reads
the vendored copy there and checks it carries exactly the contracted content; when it
is not, that test skips and says so. No xfail anywhere: the contract names the exact
text it is a contract with, it does not expect anything to be broken.

Why the field is in main's header and not the shared body: a value naming the mirror's
commit cannot live inside the bytes that commit contains, and two sides pinning each
other's commit sha would chase each other forever. So main names the vendor's commit,
and both sides pin the module by content.

Also pinned, as provenance: the 32 definitions that came out of the driver are
byte-identical to the driver's own spans at the commit the extraction branched from,
and the module's two lane-shaped definitions (``REPO`` from the locator,
``DRIVER_VERSION`` a seam sentinel) are NOT copies of the driver's lines.
"""

from __future__ import annotations

import ast
import hashlib
import re
import subprocess
from pathlib import Path

import pytest

from sisteer.eval.sampled_controls import BODY_MARKER, VENDOR_CONTRACT

REPO = Path(__file__).resolve().parents[1]
MODULE_REL = "src/sisteer/eval/sampled_controls.py"
MODULE = REPO / MODULE_REL
SAMPLED_DECODE_REL = "src/sisteer/eval/sampled_decode.py"
#: the driver the 32 definitions were extracted from, at the commit the extraction
#: branched from (origin/main, 2026-09-12)
DRIVER_SOURCE_SHA = "6027389c7e4f204cc448ac424c591e9e1457579d"
DRIVER_REL = "scripts/served_capability_run.py"

#: the 32 definitions that left the driver: 10 ladder + 18 transitive + 4 wiring
DRIVER_ORIGIN_NAMES = (
    "sampling_seed", "floor_rank_key", "floor_sample_ids", "_decode_of", "_decode_block",
    "GenMeta", "Escalation", "_apply_items", "_apply_escalation", "_run_judged_served",
    "CONTROLS_CAP_LADDER", "FLOOR_SAMPLE_DEFAULT", "FLOOR_SAMPLE_RULE", "META_SUFFIX",
    "META_ROW_KEYS", "META_REGEN_KEYS", "META_COPIED_KEYS", "META_EXCLUDED_KEY",
    "META_EXCLUSIONS", "_text_sha", "_first_divergence", "_fork_fields", "_meta_stats",
    "SourceControl", "_ladder_decide", "_read_items_files", "_locate_source_store",
    "_decoder_of",
    "_validate_source_identity", "_read_source_spool", "_load_source_control",
    "_SOURCE_IDENTITY_KEYS",
)
#: the 11 that make one module safe for two drivers (the seam + the locator), and the
#: marker constant that names the byte-identity boundary
SEAM_AND_LOCATOR_NAMES = (
    "DRIVER_GLOBALS", "_Unbound", "_chat_payload", "_fanout", "ccr", "DRIVER_VERSION",
    "bind_driver_globals", "driver_globals_bound", "repo_root", "set_repo_root", "REPO",
    "BODY_MARKER",
)
SHARED_NAMES = DRIVER_ORIGIN_NAMES + SEAM_AND_LOCATOR_NAMES
assert len(DRIVER_ORIGIN_NAMES) == 32 and len(SEAM_AND_LOCATOR_NAMES) == 12
#: main's own, outside the shared body
HEADER_ONLY_NAMES = ("VENDOR_CONTRACT",)
#: names both the driver (at DRIVER_SOURCE_SHA) and the module define, DELIBERATELY
#: DIFFERENTLY: the marker each module span must carry
SEAM_OR_LOCATOR_FORM = {"REPO": "repo_root(", "DRIVER_VERSION": "_Unbound("}
RE_MIRROR = ("VENDOR_CONTRACT: main is canonical. A change to the shared body ships in ONE change "
             "with the re-mirrored vendor (its body from BODY_MARKER on) and the updated field "
             "(definition_sha256, body_sha256, sampled_decode_sha256, expected_at); never edit "
             "main to match a vendor")


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _git(*args) -> "str | None":
    try:
        return subprocess.check_output(["git", "-C", str(REPO), *args], text=True,
                                       stderr=subprocess.DEVNULL)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _spans(text: str) -> dict:
    """``{name: definition text}`` for every top-level def/class/assignment, a def's
    span starting at its first decorator so a decorator change is drift."""
    lines = text.splitlines()
    out = {}
    for node in ast.parse(text).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            start = min([d.lineno for d in node.decorator_list] + [node.lineno])
            out[node.name] = "\n".join(lines[start - 1:node.end_lineno])
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    out[t.id] = "\n".join(lines[node.lineno - 1:node.end_lineno])
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            out[node.target.id] = "\n".join(lines[node.lineno - 1:node.end_lineno])
    return out


def _body(text: str) -> str:
    assert BODY_MARKER in text
    return text[text.index(BODY_MARKER):]


# ------------------------------------------ the contract, asserted on main's own body (no network)
def test_mains_shared_body_defines_exactly_the_contracted_names():
    text = MODULE.read_text()
    body, header = _spans(_body(text)), _spans(text[:text.index(BODY_MARKER)])
    assert set(body) == set(SHARED_NAMES), (
        f"main's shared body defines {sorted(set(body) ^ set(SHARED_NAMES))} beyond / short of the "
        "32 + 12 the contract names; a new definition is added to this list, to the field's "
        f"hash table, and to the vendor in the same change. {RE_MIRROR}")
    assert set(body) == set(VENDOR_CONTRACT["definition_sha256"]), (
        "the field's hash table and the shared body name different definitions")
    # the header is main's own: docstring, imports, and the contract field — no other definition
    assert set(header) == set(HEADER_ONLY_NAMES), (
        f"main's header defines {sorted(set(header) - set(HEADER_ONLY_NAMES))}; every shared "
        "definition belongs in the body the vendor mirrors")
    kinds = {type(n).__name__ for n in ast.parse(text[:text.index(BODY_MARKER)]).body}
    assert kinds <= {"Expr", "Import", "ImportFrom", "Assign"}, sorted(kinds)


def test_every_shared_definition_hashes_to_the_contract():
    body = _spans(_body(MODULE.read_text()))
    want = VENDOR_CONTRACT["definition_sha256"]
    assert all(re.fullmatch(r"[0-9a-f]{64}", h) for h in want.values()), "not sha256 hex digests"
    drift = [n for n in SHARED_NAMES if _sha256(body[n]) != want[n]]
    assert not drift, (
        f"{drift} in main's shared body do not hash to VENDOR_CONTRACT['definition_sha256']: the "
        f"body changed without the contract (and the vendor) moving with it. {RE_MIRROR}")


def test_the_whole_shared_body_hashes_to_the_contract():
    assert _sha256(_body(MODULE.read_text())) == VENDOR_CONTRACT["body_sha256"], (
        f"main's shared body (from {BODY_MARKER!r} to end of file, comments included) does not "
        f"hash to VENDOR_CONTRACT['body_sha256']. {RE_MIRROR}")


def test_sampled_decode_hashes_to_the_contract():
    assert SAMPLED_DECODE_REL in VENDOR_CONTRACT["scope"]["covers"]
    assert _sha256((REPO / SAMPLED_DECODE_REL).read_text()) == VENDOR_CONTRACT["sampled_decode_sha256"], (
        f"{SAMPLED_DECODE_REL} is in the contract's scope and does not hash to "
        f"VENDOR_CONTRACT['sampled_decode_sha256']. {RE_MIRROR}")


def test_the_contract_field_is_self_describing():
    c = VENDOR_CONTRACT
    # direction is read off the field, never inferred from an assertion's argument order
    assert c["canonical"] == {"ref": "main", "path": MODULE_REL}
    assert "re-mirrored from the canonical" in c["direction"]
    (vendor,) = c["vendors"]
    assert vendor["ref"] == "arditi-cell1" and vendor["path"] == MODULE_REL
    assert "W8" in vendor["why"] and "identity gate" in vendor["why"]
    # where the vendored copy is expected, and in what state
    assert re.fullmatch(r"[0-9a-f]{40}", vendor["expected_at"]["sha"]), vendor["expected_at"]
    assert vendor["expected_at"]["ref"] in ("arditi-cell1-vendor-mirror", "arditi-cell1")
    assert vendor["state"]
    # scope: what it covers AND what it does not, so an absent controls_cap is not read as drift
    assert tuple(c["scope"]["covers"]) == (MODULE_REL, SAMPLED_DECODE_REL)
    excluded = c["scope"]["excludes"]["src/sisteer/eval/cell_controls.py"]
    assert "90284b59" in excluded and "controls_cap" in excluded and "NOT drift" in excluded
    assert c["enforced_by"] == {"main": "tests/test_sampled_controls_drift.py",
                                "arditi-cell1": "tests/test_sampled_controls_port.py"}
    assert len(c["definition_sha256"]) == len(SHARED_NAMES) == 44


# ------------------------------------------ the vendored copy at its expected commit (when reachable)
def test_the_vendored_copy_at_its_expected_commit_carries_the_contracted_content():
    (vendor,) = VENDOR_CONTRACT["vendors"]
    sha, path = vendor["expected_at"]["sha"], vendor["path"]
    if _git("cat-file", "-e", f"{sha}^{{commit}}") is None:
        pytest.skip(f"the vendor's expected commit {sha[:8]} is unreachable from this clone "
                    "(shallow / exact-sha checkout); the content pins above still hold main")
    text = _git("show", f"{sha}:{path}")
    assert text is not None, f"{sha[:8]} carries no {path}: expected_at names a commit without the vendored copy"
    spans = _spans(_body(text))
    assert set(spans) == set(SHARED_NAMES), sorted(set(spans) ^ set(SHARED_NAMES))
    drift = [n for n in SHARED_NAMES if _sha256(spans[n]) != VENDOR_CONTRACT["definition_sha256"][n]]
    assert not drift, f"the vendored copy at {sha[:8]} differs from the contract on {drift}"
    assert _sha256(_body(text)) == VENDOR_CONTRACT["body_sha256"]
    sd = _git("show", f"{sha}:{SAMPLED_DECODE_REL}")
    assert sd is not None and _sha256(sd) == VENDOR_CONTRACT["sampled_decode_sha256"]
    # the vendor's header is its own: docstring and imports, nothing defined there
    kinds = {type(n).__name__ for n in ast.parse(text[:text.index(BODY_MARKER)]).body}
    assert kinds <= {"Expr", "Import", "ImportFrom"}, sorted(kinds)


# ------------------------------------------ provenance: main's module vs the driver it came from
def _driver_at_source_sha() -> str:
    text = _git("show", f"{DRIVER_SOURCE_SHA}:{DRIVER_REL}")
    if text is None:
        pytest.skip(f"{DRIVER_SOURCE_SHA[:8]}:{DRIVER_REL} is unreachable from this clone "
                    "(shallow checkout); the extraction-source check cannot run here")
    return text


def test_the_32_extracted_definitions_are_the_drivers_spans_at_the_extraction_commit():
    src, ours = _spans(_driver_at_source_sha()), _spans(MODULE.read_text())
    missing_there = [n for n in DRIVER_ORIGIN_NAMES if n not in src]
    assert not missing_there, f"the driver at {DRIVER_SOURCE_SHA[:8]} never defined {missing_there}"
    drift = [n for n in DRIVER_ORIGIN_NAMES if ours[n] != src[n]]
    assert not drift, (
        f"{drift} in {MODULE_REL} differ from {DRIVER_REL} @ {DRIVER_SOURCE_SHA[:8]} — the "
        "extraction was a MOVE; a behaviour change to one of these belongs in its own change "
        "with its own test, and this pin then moves with it")


def test_repo_and_driver_version_are_the_locator_and_the_seam_not_the_drivers_lines():
    src, ours = _spans(_driver_at_source_sha()), _spans(MODULE.read_text())
    for name, marker in SEAM_OR_LOCATOR_FORM.items():
        assert marker in ours[name], f"sampled_controls.{name} must be the {marker!r} form"
        assert ours[name] != src[name], (
            f"sampled_controls.{name} is a byte-copy of the driver's line; it must come from "
            "the locator / the seam")


def test_the_driver_on_this_tree_no_longer_defines_any_extracted_name():
    now = _spans((REPO / DRIVER_REL).read_text())
    still = [n for n in DRIVER_ORIGIN_NAMES if n in now]
    assert not still, f"{DRIVER_REL} still defines {still} beside the module's copy"
