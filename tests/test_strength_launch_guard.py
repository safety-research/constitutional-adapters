"""launch_guard.py — the strength frontier's three-way object check (disk sha = view manifest = roster pin, keyed by
(target, staged_id)) before a dose-ladder leg is submitted.

The bank here is SYNTHETIC: invented targets (`alpha`, `beta`), invented staged and cell ids, object bytes that are short
strings, shas computed from them. Nothing is copied from a real bank or the real pins file. The LAYOUT is the real one
(make_bank_view.py): <root>/<target>/cells/<staged_id>/object.pt, <root>/<target>/cells_view/<view>/VIEW_MANIFEST.json
with entries[] = {cell_id, staged_id, object_sha256, source, mode}, and <view>/<cell_id> a symlink to ../../cells/<sid>.

The one fact the fixture must reproduce is the real pins file's: the SAME staged id under two targets, pinning two
different objects (36 of the 50 ids there repeat across targets). A lookup keyed by the id alone finds the wrong target's
pin and refuses a correct cell — that happened on 2026-09-13, four cells — so the (target, staged_id) keying is pinned
here with both the positive (each target passes) and, through a deliberately wrong helper, the negative.
"""
from __future__ import annotations

import base64
import contextlib
import hashlib
import importlib.util
import io
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from front_door_testlib import MATCHERS, SCAN, token_hits

#: one tier of the publication scan's token table, matched as the scan matches it (the tokens are never spelled here)
STORAGE_TOKENS = SCAN.Matcher([r for r in MATCHERS["rows"] if r.tier == "storage"])

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "sdfbase" / "full22" / "strength" / "launch_guard.py"


def _load():
    spec = importlib.util.spec_from_file_location("launch_guard", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod      # dataclasses resolves the module's (postponed) annotations through sys.modules
    spec.loader.exec_module(mod)
    return mod


lg = _load()

# ------------------------------------------------------------------------------------------------------------ #
# the synthetic bank                                                                                             #
# ------------------------------------------------------------------------------------------------------------ #

SHARED_ID = "d1__demo-mix__lora_r4"                 # the SAME staged id under both targets, DIFFERENT bytes
SHARED_DM1 = "dm1__demo-mix__lora_r4"               # a second collision
CTL_ID = "derived__demo-mix__lora_r4_minus_ctl"     # alpha only
CELL = "demo-mix__lora_r4"
CELL_CTL = "demo-mix__lora_r4_minus_ctl"

ALPHA_D1_BYTES = b"alpha d1 object bytes"
BETA_D1_BYTES = b"beta d1 object bytes -- not alpha's"

#: target -> view -> [(cell_id, staged_id, object bytes)]
SPEC = {
    "alpha": {"d1": [(CELL, SHARED_ID, ALPHA_D1_BYTES), (CELL_CTL, CTL_ID, b"alpha ctl object bytes")],
              "dm1": [(CELL, SHARED_DM1, b"alpha dm1 object bytes")]},
    "beta": {"d1": [(CELL, SHARED_ID, BETA_D1_BYTES)],
             "dm1": [(CELL, SHARED_DM1, b"beta dm1 object bytes")]},
}


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def make_bank(root: Path, spec=SPEC) -> dict:
    """Build the bank in the real layout and return the CORRECT pins dict: every (target, staged_id) pinned to its own
    bytes, keyed exactly as configs/sdfbase_full22_pins.json is (top level = target)."""
    pins: dict = {}
    for target, views in spec.items():
        pins[target] = {}
        for view, cells in views.items():
            vdir = root / target / "cells_view" / view
            vdir.mkdir(parents=True)
            entries = []
            for cell_id, sid, payload in cells:
                cdir = root / target / "cells" / sid
                cdir.mkdir(parents=True, exist_ok=True)
                (cdir / "object.pt").write_bytes(payload)
                (vdir / cell_id).symlink_to(Path("../../cells") / sid, target_is_directory=True)
                entries.append({"cell_id": cell_id, "staged_id": sid, "object_sha256": sha(payload),
                                "source": str(cdir), "mode": "symlink", "identity": "staged"})
                pins[target][sid] = {"object_sha256": sha(payload), "verified_cluster_a": True,
                                     "verified_cluster_b": True}
            (vdir / "VIEW_MANIFEST.json").write_text(json.dumps(
                {"wave": "synthetic", "target": target, "view": view, "bank": str(root / target),
                 "entries": entries, "written_at": "2026-01-01T00:00:00Z"}, indent=2) + "\n")
    return pins


def _manifest_path(root: Path, target: str, view: str) -> Path:
    return root / target / "cells_view" / view / "VIEW_MANIFEST.json"


def _edit_manifest_entry(root: Path, target: str, view: str, cell_id: str, **changes) -> None:
    p = _manifest_path(root, target, view)
    vm = json.loads(p.read_text())
    for e in vm["entries"]:
        if e["cell_id"] == cell_id:
            e.update(changes)
    p.write_text(json.dumps(vm) + "\n")


def _one(checks, view: str, cell_id: str):
    hits = [c for c in checks if c.view == view and c.cell_id == cell_id]
    assert len(hits) == 1, [(c.view, c.cell_id) for c in checks]
    return hits[0]


@pytest.fixture
def bank(tmp_path):
    root = tmp_path / "bank"
    pins = make_bank(root)
    return root, pins


@pytest.fixture
def pins_file(tmp_path):
    def write(pins: dict) -> str:
        p = tmp_path / "pins.json"
        p.write_text(json.dumps(pins, indent=1) + "\n")
        return str(p)
    return write


# ------------------------------------------------------------------------------------------------------------ #
# the pure check                                                                                                 #
# ------------------------------------------------------------------------------------------------------------ #

def test_refusal_reasons_are_exactly_the_four():
    assert set(lg.REASONS) == {"disk!=manifest", "disk!=pin", "NO PIN", "NO VIEW"}
    assert (lg.EXIT_PASS, lg.EXIT_OPERATOR, lg.EXIT_REFUSE) == (0, 2, 3)


def test_pass_when_disk_manifest_and_pin_agree(bank):
    root, pins = bank
    checks = lg.check_views(root, "alpha", ["d1", "dm1"], pins)
    assert [(c.view, c.cell_id) for c in checks] == [("d1", CELL), ("d1", CELL_CTL), ("dm1", CELL)]
    for c in checks:
        assert c.ok and c.reason == ""
        assert c.disk == c.manifest == c.pin and len(c.disk) == 64
        assert c.line().startswith("OK") and "REFUSE" not in c.line()
    assert _one(checks, "d1", CELL).disk == sha(ALPHA_D1_BYTES)
    assert _one(checks, "d1", CELL).staged_id == SHARED_ID
    assert lg.verdict(checks) == {"result": "PASS", "n_cells": 3, "n_refused": 0, "n_views": 2, "reasons": {}}
    assert lg.verdict_line(lg.verdict(checks)) == "GUARD: PASS (3 cells)"


def test_manifest_recording_another_sha_is_disk_ne_manifest(bank):
    """Perturb ONLY the view's recorded sha: disk still equals the pin, so the reason is the bank's own disagreement."""
    root, pins = bank
    _edit_manifest_entry(root, "alpha", "d1", CELL, object_sha256=sha(b"what the view thinks it holds"))
    checks = lg.check_views(root, "alpha", ["d1", "dm1"], pins)
    c = _one(checks, "d1", CELL)
    assert not c.ok and c.reason == "disk!=manifest"
    assert c.disk == c.pin == sha(ALPHA_D1_BYTES) and c.manifest == sha(b"what the view thinks it holds")
    assert c.line().endswith(" disk!=manifest")
    assert all(x.ok for x in checks if x is not c)
    assert lg.verdict_line(lg.verdict(checks)) == "GUARD: REFUSE (1 of 3 cells)"


def test_manifest_without_a_recorded_sha_is_disk_ne_manifest(bank):
    root, pins = bank
    _edit_manifest_entry(root, "alpha", "d1", CELL, object_sha256=None)
    c = _one(lg.check_views(root, "alpha", ["d1"], pins), "d1", CELL)
    assert c.reason == "disk!=manifest" and c.manifest is None and "manifest=MISSING" in c.line()


def test_pin_recording_another_sha_is_disk_ne_pin(bank):
    """Perturb ONLY the pin: disk equals the manifest, so the reason is the roster's disagreement."""
    root, pins = bank
    pins["alpha"][SHARED_ID]["object_sha256"] = sha(b"what the roster pinned")
    checks = lg.check_views(root, "alpha", ["d1", "dm1"], pins)
    c = _one(checks, "d1", CELL)
    assert not c.ok and c.reason == "disk!=pin"
    assert c.disk == c.manifest == sha(ALPHA_D1_BYTES) and c.pin == sha(b"what the roster pinned")
    assert all(x.ok for x in checks if x is not c)


def test_missing_pin_is_no_pin(bank):
    root, pins = bank
    del pins["alpha"][SHARED_ID]
    checks = lg.check_views(root, "alpha", ["d1", "dm1"], pins)
    c = _one(checks, "d1", CELL)
    assert not c.ok and c.reason == "NO PIN" and c.pin is None
    assert c.disk == c.manifest == sha(ALPHA_D1_BYTES)
    assert "pin=NONE" in c.line() and c.line().endswith(" NO PIN")
    assert all(x.ok for x in checks if x is not c)
    # a pin entry present but recording no sha is no pin either
    pins["alpha"][SHARED_ID] = {"source": "somewhere", "object_sha256": ""}
    assert _one(lg.check_views(root, "alpha", ["d1"], pins), "d1", CELL).reason == "NO PIN"


def test_object_missing_behind_a_manifest_entry_is_no_view_not_an_exception(bank):
    """The view names the cell, the symlink is there, the bytes are not: a "staged" that never reached this bank."""
    root, pins = bank
    (root / "alpha" / "cells" / SHARED_ID / "object.pt").unlink()
    checks = lg.check_views(root, "alpha", ["d1", "dm1"], pins)      # must not raise
    c = _one(checks, "d1", CELL)
    assert not c.ok and c.reason == "NO VIEW" and c.disk is None
    assert c.manifest == c.pin == sha(ALPHA_D1_BYTES)                  # the claims stand; the object does not
    assert "disk=MISSING" in c.line() and c.line().endswith(" NO VIEW")
    assert all(x.ok for x in checks if x is not c)
    # the staged directory gone entirely (a dangling cell symlink) reads the same way
    shutil.rmtree(root / "alpha" / "cells" / CTL_ID)
    assert (root / "alpha" / "cells_view" / "d1" / CELL_CTL).is_symlink()
    checks = lg.check_views(root, "alpha", ["d1"], pins)
    assert _one(checks, "d1", CELL_CTL).reason == "NO VIEW"
    # and a cell entry with no symlink at all
    (root / "alpha" / "cells_view" / "d1" / CELL).unlink()
    assert _one(lg.check_views(root, "alpha", ["d1"], pins), "d1", CELL).reason == "NO VIEW"


def test_rewritten_bytes_read_as_disk_ne_manifest_before_disk_ne_pin(bank):
    """A double fault (the bytes changed under both records) names the bank's own inconsistency first."""
    root, pins = bank
    (root / "alpha" / "cells" / SHARED_ID / "object.pt").write_bytes(b"swapped after the view was built")
    c = _one(lg.check_views(root, "alpha", ["d1"], pins), "d1", CELL)
    assert c.reason == "disk!=manifest"
    assert c.disk == sha(b"swapped after the view was built") and c.manifest == c.pin == sha(ALPHA_D1_BYTES)


def test_pin_lookup_is_keyed_by_target_and_staged_id(bank):
    """The same staged id under two targets pins two different objects. The correct target PASSES; a lookup keyed by
    the id alone (the bug that refused four correct cells) finds the OTHER target's pin and would refuse."""
    root, pins = bank
    a, b = pins["alpha"][SHARED_ID]["object_sha256"], pins["beta"][SHARED_ID]["object_sha256"]
    assert a != b and a == sha(ALPHA_D1_BYTES) and b == sha(BETA_D1_BYTES)
    # the positive: each target, against the whole pins dict, passes on its own object
    for target in ("alpha", "beta"):
        checks = lg.check_views(root, target, ["d1", "dm1"], pins)
        assert all(c.ok for c in checks), [(c.target, c.view, c.cell_id, c.reason) for c in checks]
    assert lg.pinned_sha(pins, "alpha", SHARED_ID) == a
    assert lg.pinned_sha(pins, "beta", SHARED_ID) == b

    # the negative, via a deliberately WRONG helper: the first block carrying the id wins, whatever the target
    def wrong_pinned_sha_by_id_only(pins_: dict, staged_id: str):
        for block in pins_.values():
            e = block.get(staged_id)
            if isinstance(e, dict) and e.get("object_sha256"):
                return e["object_sha256"]
        return None

    beta_disk = _one(lg.check_views(root, "beta", ["d1"], pins), "d1", CELL).disk
    wrong = wrong_pinned_sha_by_id_only(pins, SHARED_ID)
    assert wrong == a and wrong != beta_disk          # id-only: alpha's pin against beta's bytes -> disk!=pin
    assert lg.pinned_sha(pins, "beta", SHARED_ID) == beta_disk

    # and the module reads NOTHING outside the target's block: with beta's block empty, beta's cell is NO PIN —
    # never alpha's pin, and never alpha's sha
    only_alpha = {"alpha": pins["alpha"], "beta": {}}
    c = _one(lg.check_views(root, "beta", ["d1"], only_alpha), "d1", CELL)
    assert c.reason == "NO PIN" and c.pin is None


def test_one_level_of_nesting_is_admitted_and_an_ambiguous_pin_is_an_operator_error(bank):
    root, pins = bank
    nested = {"alpha": {"lora_lane": pins["alpha"]}}                       # lane -> id -> entry
    assert all(c.ok for c in lg.check_views(root, "alpha", ["d1", "dm1"], nested))
    assert lg.pinned_sha(nested, "alpha", SHARED_ID) == sha(ALPHA_D1_BYTES)
    # the same id twice inside one block with the SAME sha is one pin
    twice = {"alpha": {**pins["alpha"], "lora_lane": {SHARED_ID: dict(pins["alpha"][SHARED_ID])}}}
    assert lg.pinned_sha(twice, "alpha", SHARED_ID) == sha(ALPHA_D1_BYTES)
    # with DIFFERENT shas it is ambiguous: refused as an operator error, never picked from
    ambiguous = {"alpha": {**pins["alpha"], "lora_lane": {SHARED_ID: {"object_sha256": sha(b"another")}}}}
    with pytest.raises(lg.GuardError, match="AMBIGUOUS PIN"):
        lg.check_views(root, "alpha", ["d1"], ambiguous)
    # nesting never crosses targets: beta's block nested under alpha is still alpha's
    cross = {"alpha": {"beta": pins["beta"]}}
    assert lg.pinned_sha(cross, "alpha", SHARED_ID) == sha(BETA_D1_BYTES)     # (it is alpha's block's content)
    with pytest.raises(lg.GuardError, match="NO PINS BLOCK"):
        lg.pinned_sha(cross, "beta", SHARED_ID)


def test_missing_pins_block_is_an_operator_error(bank):
    root, pins = bank
    with pytest.raises(lg.GuardError, match="NO PINS BLOCK.*gamma"):
        lg.check_views(root, "gamma", ["d1"], pins)
    with pytest.raises(lg.GuardError, match="NO PINS BLOCK"):
        lg.check_views(root, "alpha", ["d1"], {"alpha": "not a dict"})
    with pytest.raises(lg.GuardError, match="NO PINS BLOCK"):
        lg.check_views(root, "alpha", ["d1"], {})


def test_missing_view_dir_or_manifest_is_an_operator_error(bank):
    root, pins = bank
    with pytest.raises(lg.GuardError, match="NO VIEW DIR.*d9"):
        lg.check_views(root, "alpha", ["d1", "d9"], pins)
    _manifest_path(root, "alpha", "dm1").unlink()
    with pytest.raises(lg.GuardError, match="NO MANIFEST.*dm1"):
        lg.check_views(root, "alpha", ["d1", "dm1"], pins)
    _manifest_path(root, "alpha", "d1").write_text("{not json")
    with pytest.raises(lg.GuardError, match="unreadable"):
        lg.check_views(root, "alpha", ["d1"], pins)


def test_a_view_with_no_entries_is_an_operator_error_not_a_pass(bank):
    root, pins = bank
    p = _manifest_path(root, "alpha", "dm1")
    vm = json.loads(p.read_text())
    vm["entries"] = []
    p.write_text(json.dumps(vm))
    with pytest.raises(lg.GuardError, match="NO ENTRIES"):
        lg.check_views(root, "alpha", ["dm1"], pins)


LINE = re.compile(r"^(OK|REFUSE) +alpha/(d1|dm1) +\S+ +disk=([0-9a-f]{12}|MISSING) manifest=([0-9a-f]{12}|MISSING) "
                  r"pin=([0-9a-f]{12}|NONE)( (disk!=manifest|disk!=pin|NO PIN|NO VIEW))?$")


def test_line_format(bank):
    root, pins = bank
    for c in lg.check_views(root, "alpha", ["d1", "dm1"], pins):
        m = LINE.match(c.line())
        assert m and m.group(1) == "OK" and m.group(6) is None, c.line()
    del pins["alpha"][SHARED_DM1]
    (root / "alpha" / "cells" / CTL_ID / "object.pt").unlink()
    for c in lg.check_views(root, "alpha", ["d1", "dm1"], pins):
        m = LINE.match(c.line())
        assert m, c.line()
        assert (m.group(1) == "OK") == c.ok
        assert (m.group(7) or "") == c.reason


# ------------------------------------------------------------------------------------------------------------ #
# the CLI                                                                                                        #
# ------------------------------------------------------------------------------------------------------------ #

def _cell_lines(out: str):
    return [l for l in out.splitlines() if l.startswith(("OK", "REFUSE"))]


def test_cli_pass_is_exit_0_with_every_cell_and_the_verdict_said_out_loud(bank, pins_file, tmp_path, capsys):
    root, pins = bank
    out_json = tmp_path / "guard.json"
    rc = lg.main(["--local-root", str(root), "--target", "alpha", "--views", "d1,dm1", "--pins", pins_file(pins),
                  "--json", str(out_json)])
    out = capsys.readouterr().out
    assert rc == 0
    lines = _cell_lines(out)
    assert len(lines) == 3 and all(l.startswith("OK") for l in lines)
    assert out.splitlines()[-1] == "GUARD: PASS (3 cells)"
    assert out.splitlines()[0].startswith("# launch guard: local ") and "target=alpha" in out.splitlines()[0]
    rec = json.loads(out_json.read_text())
    assert rec["verdict"]["result"] == "PASS" and rec["exit_code"] == 0
    assert len(rec["checks"]) == 3 and all(len(c["disk"]) == 64 for c in rec["checks"])
    assert rec["checks"][0]["staged_id"] == SHARED_ID
    assert rec["pins"]["sha256"] == sha(Path(rec["pins"]["path"]).read_bytes())
    assert rec["source"]["mode"] == "local" and rec["source"]["bank_root"] == str(root)
    assert rec["tool"]["sha256"] == sha(SCRIPT.read_bytes())
    assert rec["target"] == "alpha" and rec["views"] == ["d1", "dm1"]


def test_cli_refuse_is_exit_3(bank, pins_file, tmp_path, capsys):
    root, pins = bank
    del pins["alpha"][SHARED_ID]
    out_json = tmp_path / "guard.json"
    rc = lg.main(["--local-root", str(root), "--target", "alpha", "--views", "d1,dm1", "--pins", pins_file(pins),
                  "--json", str(out_json)])
    out = capsys.readouterr().out
    assert rc == 3
    lines = _cell_lines(out)
    assert len(lines) == 3 and sum(l.startswith("REFUSE") for l in lines) == 1
    assert any(l.startswith("REFUSE alpha/d1") and l.endswith(" NO PIN") for l in lines)
    assert out.splitlines()[-1] == "GUARD: REFUSE (1 of 3 cells)"
    rec = json.loads(out_json.read_text())
    assert rec["verdict"] == {"result": "REFUSE", "n_cells": 3, "n_refused": 1, "n_views": 2, "reasons": {"NO PIN": 1}}
    assert rec["exit_code"] == 3


def test_cli_operator_errors_are_exit_2_and_still_leave_a_record(bank, pins_file, tmp_path, capsys):
    root, pins = bank
    pins_p = pins_file(pins)
    out_json = tmp_path / "guard.json"
    # no pins block for the target
    rc = lg.main(["--local-root", str(root), "--target", "gamma", "--views", "d1", "--pins", pins_p, "--json", str(out_json)])
    out = capsys.readouterr().out
    assert rc == 2 and out.startswith("GUARD: ERROR") and "gamma" in out and not _cell_lines(out)
    rec = json.loads(out_json.read_text())
    assert rec["verdict"]["result"] == "ERROR" and rec["exit_code"] == 2 and "gamma" in rec["verdict"]["error"]
    # a view that is not built
    rc = lg.main(["--local-root", str(root), "--target", "alpha", "--views", "d1,d9", "--pins", pins_p])
    out = capsys.readouterr().out
    assert rc == 2 and out.startswith("GUARD: ERROR") and "d9" in out
    # no pins file
    rc = lg.main(["--local-root", str(root), "--target", "alpha", "--views", "d1", "--pins", str(tmp_path / "nope.json")])
    out = capsys.readouterr().out
    assert rc == 2 and "NO PINS FILE" in out
    # a bank root that is not there
    rc = lg.main(["--local-root", str(tmp_path / "nowhere"), "--target", "alpha", "--views", "d1", "--pins", pins_p])
    assert rc == 2 and capsys.readouterr().out.startswith("GUARD: ERROR")


def test_cli_requires_target_views_and_a_source(bank, pins_file):
    root, pins = bank
    for argv in (["--local-root", str(root), "--views", "d1"],
                 ["--local-root", str(root), "--target", "alpha"],
                 ["--target", "alpha", "--views", "d1"],
                 ["--local-root", str(root), "--target", "alpha", "--views", " , "]):
        with pytest.raises(SystemExit) as e:
            lg.main(argv + ["--pins", pins_file(pins)])
        assert e.value.code == 2


def test_cli_dedupes_views_and_drops_blanks(bank, pins_file, capsys):
    root, pins = bank
    rc = lg.main(["--local-root", str(root), "--target", "alpha", "--views", " d1,,dm1, d1 ", "--pins", pins_file(pins)])
    out = capsys.readouterr().out
    assert rc == 0 and len(_cell_lines(out)) == 3 and "views=d1,dm1 " in out.splitlines()[0]


def test_default_pins_is_the_checkouts_pins_file():
    assert lg.DEFAULT_PINS == REPO / "configs" / "sdfbase_full22_pins.json"
    assert lg.REPO == REPO


# ------------------------------------------------------------------------------------------------------------ #
# --host: the remote leg, exercised without a cluster                                                             #
# ------------------------------------------------------------------------------------------------------------ #

_REAL_RUN = subprocess.run


def _patch_ssh(monkeypatch, fake):
    """Stand `fake` in for the ssh call ONLY: the tool's other subprocess use (`git rev-parse` for the record's
    provenance) keeps running for real."""
    def run(cmd, *a, **kw):
        if cmd and cmd[0] == "ssh":
            return fake(cmd, *a, **kw)
        return _REAL_RUN(cmd, *a, **kw)
    monkeypatch.setattr(lg.subprocess, "run", run)


def _fake_ssh(root: Path, calls: list, expect_host: str):
    """Stands in for subprocess.run: asserts the transport screen_verdict.py uses, then runs the shipped leg
    IN-PROCESS against the synthetic bank and hands its stdout back as ssh would."""
    def run(cmd, input=None, capture_output=None, text=None, timeout=None):
        calls.append(cmd)
        assert cmd[:5] == ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=30"] and cmd[5] == expect_host
        assert input == SCRIPT.read_text(encoding="utf-8")                    # the tool's own source rides stdin
        prog, dash, flag, b64 = cmd[6].split(" ")
        assert (prog, dash, flag) == ("python3", "-", "--gather")
        req = json.loads(base64.b64decode(b64))
        calls.append(req)
        req = dict(req, bank_root=str(root))                                  # the cluster root -> the tmp bank
        local_b64 = base64.b64encode(json.dumps(req).encode()).decode()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            assert lg.main(["--gather", local_b64]) == 0
        return subprocess.CompletedProcess(cmd, 0, stdout="motd noise\n" + buf.getvalue(), stderr="")
    return run


# demo-import M7 (Q11): the grading host leg reads the cluster's host and root from the SITE's cluster block — reproduce/lib/clusters.py over
# a synthetic site (front_door_testlib.write_site: labels cluster_a / cluster_b, ssh aliases `<label>-login`, tmp shared roots) — and
# the former `--host <alias>` with its literal HOST_SHARED table is gone. Nothing here names a real host, user or storage root.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from front_door_testlib import write_site  # noqa: E402


def _site_blocks(tmp_path: Path, **kw):
    """(site dir, {label: shared root}) — the synthetic site; `ssh` / `shared` overrides as write_site takes them."""
    site = write_site(tmp_path, **kw)
    shared = {lab: (kw.get("shared") or {}).get(lab) or str(tmp_path / f"shared_{lab}") for lab in kw.get("clusters", ("cluster_a", "cluster_b"))}
    return site, shared


def test_cluster_mode_ships_the_source_the_blocks_root_verbatim_and_only_the_targets_pins_block(bank, pins_file, tmp_path, monkeypatch, capsys):
    root, pins = bank
    site, shared = _site_blocks(tmp_path)
    calls: list = []
    _patch_ssh(monkeypatch, _fake_ssh(root, calls, "cluster-a-login"))               # cmd[5] is the block's CLUSTER_SSH, never the label
    out_json = tmp_path / "guard.json"
    rc = lg.main(["--cluster", "cluster_a", "--site-dir", str(site), "--target", "alpha", "--views", "d1,dm1", "--pins", pins_file(pins),
                  "--json", str(out_json)])
    out = capsys.readouterr().out
    assert rc == 0 and len(calls) == 2
    req = calls[1]
    assert req["bank_root"] == f"{shared['cluster_a']}/exp/objects/sdfbase/full22"       # <CLUSTER_SHARED_ROOT>/<REL>, verbatim: no {user}, no $
    assert "{" not in req["bank_root"] and "$" not in req["bank_root"]
    assert req["target"] == "alpha" and req["views"] == ["d1", "dm1"]
    assert set(req["pins"]) == {"alpha"} and req["pins"]["alpha"] == pins["alpha"]   # only the target's block travels
    assert len(_cell_lines(out)) == 3 and out.splitlines()[-1] == "GUARD: PASS (3 cells)"
    assert out.splitlines()[0].startswith("# launch guard: ssh cluster_a (cluster-a-login) bank=" + str(root))
    rec = json.loads(out_json.read_text())
    src = rec["source"]
    assert src["mode"] == "ssh" and src["cluster_label"] == "cluster_a" and src["host"] == "cluster-a-login" and src["bank_root"] == str(root)
    assert src["shared_root"] == shared["cluster_a"] and src["site_dir"] == str(site) and src["cluster_block_label"] == "cluster_a"
    assert src["hostname"] and src["user"]
    # a refusal on the cluster comes back as a refusal here
    calls.clear()
    pins["alpha"][SHARED_ID]["object_sha256"] = sha(b"stale pin")
    rc = lg.main(["--cluster", "cluster_a", "--site-dir", str(site), "--target", "alpha", "--views", "d1", "--pins", pins_file(pins)])
    out = capsys.readouterr().out
    assert rc == 3 and any(l.endswith(" disk!=pin") for l in _cell_lines(out))
    assert out.splitlines()[-1] == "GUARD: REFUSE (1 of 2 cells)"


def test_cluster_mode_reads_the_other_labels_block(bank, pins_file, tmp_path, monkeypatch, capsys):
    """The second label resolves to ITS block's host and root — nothing is shared between labels but the site file."""
    root, pins = bank
    site, _shared = _site_blocks(tmp_path, ssh={"cluster_b": "b-head-node"}, shared={"cluster_b": str(tmp_path / "vast_b")})
    calls: list = []
    _patch_ssh(monkeypatch, _fake_ssh(root, calls, "b-head-node"))
    rc = lg.main(["--cluster", "cluster_b", "--site-dir", str(site), "--target", "beta", "--views", "d1,dm1", "--pins", pins_file(pins)])
    capsys.readouterr()
    assert rc == 0 and calls[1]["bank_root"] == f"{tmp_path / 'vast_b'}/exp/objects/sdfbase/full22"
    assert set(calls[1]["pins"]) == {"beta"}


def test_cluster_mode_reads_a_block_the_site_does_not_fan_out_to(bank, pins_file, tmp_path, monkeypatch, capsys):
    """--cluster names one cluster, so it reads every block in the site's clusters/ directory: a cluster SISTEER_CLUSTERS leaves out
    of the default fan-out is still checked on its own host and root."""
    root, pins = bank
    site, _shared = _site_blocks(tmp_path, ssh={"cluster_b": "b-head-node"}, shared={"cluster_b": str(tmp_path / "vast_b")})
    cfg = site / "config.env"
    text = cfg.read_text()
    assert 'SISTEER_CLUSTERS="cluster_a cluster_b"' in text
    cfg.write_text(text.replace('SISTEER_CLUSTERS="cluster_a cluster_b"', 'SISTEER_CLUSTERS="cluster_a"'))
    calls: list = []
    _patch_ssh(monkeypatch, _fake_ssh(root, calls, "b-head-node"))
    rc = lg.main(["--cluster", "cluster_b", "--site-dir", str(site), "--target", "beta", "--views", "d1,dm1", "--pins", pins_file(pins)])
    out = capsys.readouterr().out
    assert rc == 0 and calls[1]["bank_root"] == f"{tmp_path / 'vast_b'}/exp/objects/sdfbase/full22", out


def test_cluster_mode_default_site_dir_follows_the_environment(bank, pins_file, tmp_path, monkeypatch, capsys):
    """--site-dir absent: clusters.default_site_dir() — $SISTEER_CONFIG's directory, else $SISTEER_SITE_DIR (never a host of its own)."""
    root, pins = bank
    site, shared = _site_blocks(tmp_path)
    calls: list = []
    _patch_ssh(monkeypatch, _fake_ssh(root, calls, "cluster-a-login"))
    monkeypatch.delenv("SISTEER_CONFIG", raising=False)
    monkeypatch.setenv("SISTEER_SITE_DIR", str(site))
    assert lg.main(["--cluster", "cluster_a", "--target", "alpha", "--views", "d1", "--pins", pins_file(pins)]) == 0
    assert calls[1]["bank_root"] == f"{shared['cluster_a']}/exp/objects/sdfbase/full22"
    capsys.readouterr()
    monkeypatch.setenv("SISTEER_CONFIG", str(site / "config.env"))
    monkeypatch.setenv("SISTEER_SITE_DIR", str(tmp_path / "nowhere"))
    calls.clear()
    assert lg.main(["--cluster", "cluster_a", "--target", "alpha", "--views", "d1", "--pins", pins_file(pins)]) == 0   # SISTEER_CONFIG wins
    assert calls and calls[1]["bank_root"] == f"{shared['cluster_a']}/exp/objects/sdfbase/full22"


def test_cluster_mode_unreachable_host_and_remote_operator_errors_are_exit_2(bank, pins_file, tmp_path, monkeypatch, capsys):
    root, pins = bank
    site, _shared = _site_blocks(tmp_path)
    pins_p = pins_file(pins)
    argv = ["--cluster", "cluster_a", "--site-dir", str(site), "--target", "alpha", "--views", "d1", "--pins", pins_p]

    def dead(cmd, **kw):
        return subprocess.CompletedProcess(cmd, 255, stdout="", stderr="ssh: connect to host cluster-a-login port 22: Connection timed out")
    _patch_ssh(monkeypatch, dead)
    assert lg.main(argv) == 2
    out = capsys.readouterr().out
    assert out.startswith("GUARD: ERROR") and "Connection timed out" in out and not _cell_lines(out)

    def hangs(cmd, **kw):
        raise subprocess.TimeoutExpired(cmd, kw.get("timeout") or 1)
    _patch_ssh(monkeypatch, hangs)
    assert lg.main(argv) == 2 and "no answer" in capsys.readouterr().out

    def remote_operator_error(cmd, **kw):     # the leg ran, found no view dir, said so as JSON
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps({"ok": False, "error": "NO VIEW DIR: /x/cells_view/d1"}), stderr="")
    _patch_ssh(monkeypatch, remote_operator_error)
    assert lg.main(argv) == 2 and "NO VIEW DIR" in capsys.readouterr().out

    def garbage(cmd, **kw):
        return subprocess.CompletedProcess(cmd, 0, stdout="Welcome to the cluster\n", stderr="")
    _patch_ssh(monkeypatch, garbage)
    assert lg.main(argv) == 2 and "printed no JSON" in capsys.readouterr().out


def _must_not_ssh(monkeypatch):
    def must_not_be_called(cmd, **kw):
        raise AssertionError(f"ssh was attempted: {cmd[:6]}")
    _patch_ssh(monkeypatch, must_not_be_called)


def test_cluster_mode_checks_the_pins_block_before_any_ssh(bank, pins_file, tmp_path, monkeypatch, capsys):
    root, pins = bank
    site, _shared = _site_blocks(tmp_path)
    _must_not_ssh(monkeypatch)
    rc = lg.main(["--cluster", "cluster_a", "--site-dir", str(site), "--target", "gamma", "--views", "d1", "--pins", pins_file(pins)])
    assert rc == 2 and "NO PINS BLOCK" in capsys.readouterr().out


def test_cluster_mode_refuses_before_any_ssh_an_unknown_label_a_missing_site_a_malformed_config_and_a_placeholder_root(bank, pins_file, tmp_path, monkeypatch, capsys):
    """The site cluster block is the ONLY source (Q11): every way it can fail to name a host and a literal root is a `GUARD: ERROR`,
    exit 2, with no ssh attempted and no verdict — never a literal fallback. Nothing under reproduce/lib/clusters.py expands $USER, so
    a root spelled with it would reach the host unexpanded: refused here by name."""
    root, pins = bank
    pins_p = pins_file(pins)
    _must_not_ssh(monkeypatch)
    site, _shared = _site_blocks(tmp_path)
    # an unknown label names the configured ones
    rc = lg.main(["--cluster", "cluster_z", "--site-dir", str(site), "--target", "alpha", "--views", "d1", "--pins", pins_p])
    out = capsys.readouterr().out
    assert rc == 2 and out.startswith("GUARD: ERROR") and "UNKNOWN CLUSTER" in out and "'cluster_z'" in out and "cluster_a cluster_b" in out
    # no config.env under the site dir: the remedy names both roads
    rc = lg.main(["--cluster", "cluster_a", "--site-dir", str(tmp_path / "no_site"), "--target", "alpha", "--views", "d1", "--pins", pins_p])
    out = capsys.readouterr().out
    assert rc == 2 and "NO SITE CONFIG" in out and "pass --site-dir, or --local-root <bank> for a bank on this machine" in out
    # a malformed config (the label's block lacks a required key) refuses by clusters.py's own name, never a guess
    broken, _ = _site_blocks(tmp_path / "broken")
    (broken / "clusters" / "cluster_a.env").write_text("CLUSTER_SSH=cluster-a-login\n")
    rc = lg.main(["--cluster", "cluster_a", "--site-dir", str(broken), "--target", "alpha", "--views", "d1", "--pins", pins_p])
    out = capsys.readouterr().out
    assert rc == 2 and "SITE CONFIG" in out and "must set CLUSTER_SHARED_ROOT" in out
    # a root carrying $USER (or any `$` / `{` placeholder): a literal absolute path or nothing — the guard expands nothing on the host.
    # A `$` refuses already in the site reader (clusters.py, by key and token); a `{` reaches the guard's own check
    for i, placeholder in enumerate(("/scratch/$USER/shared", "/scratch/{user}/shared")):
        ph, _ = _site_blocks(tmp_path / f"ph_{i}")
        blk = ph / "clusters" / "cluster_a.env"
        blk.write_text(re.sub(r"^CLUSTER_SHARED_ROOT=.*$", f"CLUSTER_SHARED_ROOT='{placeholder}'", blk.read_text(), count=1, flags=re.M))
        rc = lg.main(["--cluster", "cluster_a", "--site-dir", str(ph), "--target", "alpha", "--views", "d1", "--pins", pins_p])
        out = capsys.readouterr().out
        if "$" in placeholder:
            assert rc == 2 and "SITE CONFIG" in out and "CLUSTER_SHARED_ROOT" in out and "holds $USER" in out and "literal absolute path" in out, out
        else:
            assert rc == 2 and "carries a placeholder" in out and "literal absolute path" in out and "expands nothing on the host" in out, (placeholder, out)
    # the record of a refusal names the error and no source
    out_json = tmp_path / "refused.json"
    rc = lg.main(["--cluster", "cluster_z", "--site-dir", str(site), "--target", "alpha", "--views", "d1", "--pins", pins_p, "--json", str(out_json)])
    capsys.readouterr()
    rec = json.loads(out_json.read_text())
    assert rc == 2 and rec["verdict"]["result"] == "ERROR" and "UNKNOWN CLUSTER" in rec["verdict"]["error"] and rec["source"] is None and rec["checks"] == []


def test_the_former_host_flag_is_rejected_and_the_literal_table_is_gone(bank, pins_file, tmp_path, monkeypatch, capsys):
    """`--host <alias>` was an ssh alias restricted to a two-entry literal table in this script; argparse rejects it now, the table is
    gone, and no cluster alias or storage root is spelled anywhere in the file (demo-import Q11: the site block is the only source)."""
    root, pins = bank
    _must_not_ssh(monkeypatch)
    with pytest.raises(SystemExit) as e:
        lg.main(["--host", "cluster-a-login", "--target", "alpha", "--views", "d1", "--pins", pins_file(pins)])
    assert e.value.code == 2 and "unrecognized arguments: --host" in capsys.readouterr().err
    assert not hasattr(lg, "HOST_SHARED")
    src = SCRIPT.read_text(encoding="utf-8")
    assert "HOST_SHARED" not in src and "--host" not in src and "{user}" not in src and ".format(user" not in src
    assert "/workspace/" not in src and not token_hits(src, STORAGE_TOKENS), token_hits(src, STORAGE_TOKENS)   # no shared root
    assert not re.search(r'"[A-Za-z0-9_.-]+":\s*"/', src), "an alias -> root literal table is back"      # no host alias maps to a root in the file
    # --site-dir without --cluster is an operator error too (--local-root needs no site)
    with pytest.raises(SystemExit) as e:
        lg.main(["--local-root", str(root), "--site-dir", str(tmp_path), "--target", "alpha", "--views", "d1", "--pins", pins_file(pins)])
    assert e.value.code == 2


def test_the_gather_leg_turns_an_operator_error_into_json_not_a_traceback(bank, capsys):
    root, pins = bank
    req = {"bank_root": str(root), "target": "alpha", "views": ["d1", "d9"], "pins": pins}
    b64 = base64.b64encode(json.dumps(req).encode()).decode()
    assert lg.main(["--gather", b64]) == 0
    rec = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert rec["ok"] is False and "NO VIEW DIR" in rec["error"] and "d9" in rec["error"]


def test_the_source_runs_as_python3_dash_exactly_as_shipped(bank):
    """The transport --host uses, for real: this file's bytes on stdin, the request in argv, one JSON line back —
    with `__file__` == "<stdin>" and no repo around it."""
    root, pins = bank
    req = {"bank_root": str(root), "target": "alpha", "views": ["d1", "dm1"], "pins": {"alpha": pins["alpha"]}}
    b64 = base64.b64encode(json.dumps(req).encode()).decode()
    p = subprocess.run([sys.executable, "-", "--gather", b64], input=SCRIPT.read_text(encoding="utf-8"),
                       capture_output=True, text=True, timeout=120, cwd=str(root))
    assert p.returncode == 0, p.stderr
    rec = json.loads(p.stdout.strip().splitlines()[-1])
    assert rec["ok"] is True and len(rec["checks"]) == 3 and all(c["ok"] for c in rec["checks"])
    assert rec["bank_root"] == str(root) and rec["hostname"] and rec["user"]
    # the pins block missing on the shipped request is an operator error in JSON, not a traceback
    req["pins"] = {}
    b64 = base64.b64encode(json.dumps(req).encode()).decode()
    p = subprocess.run([sys.executable, "-", "--gather", b64], input=SCRIPT.read_text(encoding="utf-8"),
                       capture_output=True, text=True, timeout=120, cwd=str(root))
    assert p.returncode == 0, p.stderr
    rec = json.loads(p.stdout.strip().splitlines()[-1])
    assert rec["ok"] is False and "NO PINS BLOCK" in rec["error"]
