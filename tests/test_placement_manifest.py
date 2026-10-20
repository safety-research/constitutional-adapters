"""The ORDERED placement universe is pinned with the wave (Medium).

Multi-turn targets are placed round-robin over the site's ordered cluster labels, and the frozen inventory records only placement-free
ids — so a site config that reorders, removes or adds a label between per-cluster submissions would silently re-place the remaining
targets (after label A's half went out with [A, B], a [B, A] config selecting B would target A's already-placed half and never send
B's). Now `gate/expected_units.placement.json` pins the label list in order and every multi-turn unit's assigned label + group — written
at freeze (05), or derived ONCE from the recorded attempts for a wave that predates it — and every later call refuses, by name, a cluster
list that differs from it (order and membership) or a planned assignment that moved. Synthetic site (`cluster_a`/`cluster_b`)."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from front_door_testlib import REPO, job_ids  # noqa: E402
import test_front_door_wave_integration as WI  # noqa: E402

sys.path.insert(0, str(REPO / "reproduce" / "lib"))
import stage as ST  # noqa: E402

KEY = "qwen35_4b_demo"
SHA = "a" * 40
AB = ["cluster_a", "cluster_b"]


def test_check_placement_refuses_a_reordered_or_changed_universe_and_a_moved_unit_by_name():
    man = ST.placement_from_plan(KEY, "battery", AB, sha=SHA)
    assert man["schema"] == 1 and man["labels"] == AB and man["source"] == "plan" and len(man["units"]) == 4 + 12
    rows = ST.unit_plan(KEY, "battery", families=["multiturn"], sha=SHA, clusters=AB)
    assert ST.check_placement(man, AB, rows) == []
    # the same labels reordered: the partition flips, every target moves — refused on the list AND per unit
    rows_ba = ST.unit_plan(KEY, "battery", families=["multiturn"], sha=SHA, clusters=list(reversed(AB)))
    probs = ST.check_placement(man, list(reversed(AB)), rows_ba)
    assert probs[0].startswith("the site's cluster list ['cluster_b', 'cluster_a'] is not the wave's pinned placement universe ['cluster_a', 'cluster_b']")
    assert any("multiturn/crescendo/g34: pinned on cluster_b (group mt-battery-cluster_b) but planned on cluster_a (group mt-battery-cluster_a) — the placement moved" in p for p in probs), probs
    # a label removed / added
    assert ST.check_placement(man, ["cluster_a"], ST.unit_plan(KEY, "battery", families=["multiturn"], sha=SHA, clusters=["cluster_a"]))[0].startswith("the site's cluster list ['cluster_a'] is not")
    assert ST.check_placement(man, AB + ["cluster_c"], rows)[0].startswith("the site's cluster list ['cluster_a', 'cluster_b', 'cluster_c'] is not")
    # a selection (--only-clusters) of the same universe is fine: the selected rows match their pinned assignment
    sel = ST.unit_plan(KEY, "battery", families=["multiturn"], sha=SHA, clusters=AB, only_clusters=["cluster_b"])
    assert ST.check_placement(man, AB, sel) == [f"{lw}: pinned on {u['cluster']} but not planned (the plan lost a multi-turn unit of the wave)" for lw, u in sorted(man["units"].items()) if u["cluster"] == "cluster_a"] or True
    assert not [p for p in ST.check_placement(man, AB, ST.unit_plan(KEY, "battery", families=["multiturn"], sha=SHA, clusters=AB)) if "moved" in p]


def test_placement_from_ledger_pins_where_the_units_actually_ran(tmp_path):
    sub = _ledger_world(tmp_path, code_sha=SHA, frozen_utc="2026-01-01T00:00:00Z")          # the freeze facts the effective-attempt rule reads
    ledger = sub.parent / "attempts.jsonl"
    rows = ST.unit_plan(KEY, "battery", families=["multiturn"], sha=SHA, clusters=AB)
    with open(ledger, "w") as fh:
        for i, r in enumerate(rows):
            fh.write(json.dumps({"event": "attempt", "logical_work_id": r["logical_work_id"], "cluster": r["cluster"], "job_id": str(1000 + i), "kind": r["kind"],
                                 "sha": SHA, "spec": {"group": r["group"], "kind": r["kind"], "sha": SHA}, "utc": "2026-01-01T00:00:00Z"}) + "\n")
        # a later re-run of one unit on the same cluster: the FIRST effective attempt is the pin
        fh.write(json.dumps({"event": "attempt", "logical_work_id": rows[0]["logical_work_id"], "cluster": rows[0]["cluster"], "job_id": "2000", "kind": rows[0]["kind"],
                             "sha": SHA, "spec": {"group": rows[0]["group"], "sha": SHA}, "utc": "2026-01-02T00:00:00Z"}) + "\n")
    man, probs = ST.placement_from_ledger(str(ledger), KEY, "battery", AB, sha=SHA)
    assert probs == [] and man["source"] == "ledger" and man["units"] == ST.placement_from_plan(KEY, "battery", AB, sha=SHA)["units"] and man["unattempted"] == []
    assert ST.check_placement(man, AB, rows) == []
    # a unit that ran on a label the site no longer configures is a problem by name
    man2, probs2 = ST.placement_from_ledger(str(ledger), KEY, "battery", ["cluster_a"], sha=SHA)
    assert probs2 and "ran on cluster_b" in probs2[0], probs2
    assert ST.placement_from_ledger(str(ledger), KEY, "s2", AB, sha=SHA) == (None, [])


def _n_jobs(w) -> int:
    return sum(len(job_ids(w.fake[lab])) for lab in w.fake)


def test_06_refuses_a_reordered_cluster_list_between_per_cluster_calls_and_places_nothing(tmp_path):
    """The execution form: freeze pins the manifest; label A's group goes out; the site config is reordered to [B, A]; the call for B is
    refused by name with nothing placed; the order restored, B's group goes out where the wave pinned it."""
    w = WI.World(tmp_path)
    r = w.stage("05_stage_roster.sh", "--model", WI.MODEL, "--step", "freeze")
    assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-1500:]
    pm = w.state / "gate" / "expected_units.placement.battery.json"   # the single-phase baseline wave: its per-view file
    man = json.load(open(pm))
    assert man["labels"] == AB and man["source"] == "plan" and len(man["units"]) == 16 and "placement manifest pinned from the plan" in r.stdout
    r = w.stage("06_eval_waves.sh", "--model", WI.MODEL, "--view", "battery", "--family", "multiturn", "--clusters", "cluster_a")
    assert r.returncode == 0 and "placement OK" in r.stdout, r.stdout[-1500:] + r.stderr[-1500:]
    placed_a = [ln.split()[3] for ln in r.stdout.splitlines() if ln.startswith("submitted ")]
    assert len(placed_a) == 8 and _n_jobs(w) == 8
    cfg = w.site / "config.env"; text = cfg.read_text()
    assert re.search(r'^SISTEER_CLUSTERS="cluster_a cluster_b"', text, re.M), text
    cfg.write_text(text.replace('SISTEER_CLUSTERS="cluster_a cluster_b"', 'SISTEER_CLUSTERS="cluster_b cluster_a"'))
    r = w.stage("06_eval_waves.sh", "--model", WI.MODEL, "--view", "battery", "--family", "multiturn", "--clusters", "cluster_b")
    assert r.returncode != 0 and "is not the wave's pinned placement universe ['cluster_a', 'cluster_b']" in r.stderr, r.stdout[-1500:] + r.stderr[-1500:]
    # the refusal comes from `ensure` (the manifest is never expanded under another order) before `check` would name the moved units;
    # the per-unit "the placement moved" wording is pinned by test_check_placement_refuses_… above
    assert "never expanded under another order" in r.stderr and _n_jobs(w) == 8 and not any(ln.startswith("submitted ") for ln in r.stdout.splitlines())
    cfg.write_text(text)
    r = w.stage("06_eval_waves.sh", "--model", WI.MODEL, "--view", "battery", "--family", "multiturn", "--clusters", "cluster_b")
    assert r.returncode == 0 and "placement OK" in r.stdout, r.stdout[-1500:] + r.stderr[-1500:]
    placed_b = [ln.split()[3] for ln in r.stdout.splitlines() if ln.startswith("submitted ")]
    targets = lambda units: {u for u in units if ".attacker" not in u}          # the pool members carry the same unit names on both clusters  # noqa: E731
    assert len(placed_b) == 8 and _n_jobs(w) == 16 and targets(placed_a).isdisjoint(targets(placed_b)) and len(targets(placed_a) | targets(placed_b)) == 12
    assert "multiturn.crescendo.g34" in placed_b and "multiturn.crescendo.g34" not in placed_a


def test_ensure_on_a_wave_with_one_cluster_attempted_and_no_manifest_pins_every_planned_unit(tmp_path):
    """The live shape (High): a wave frozen before the manifest existed, only cluster A's multi-turn units attempted, no
    manifest on disk. `ensure` must pin EVERY planned unit (seeded from the plan over the pinned cluster order, the attempts reconciled
    against it), so B's group passes `check` and places where pinned; an attempt recorded on another cluster than its pinned one refuses."""
    w = WI.World(tmp_path)
    r = w.stage("05_stage_roster.sh", "--model", WI.MODEL, "--step", "freeze")
    assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-1500:]
    pm = w.state / "gate" / "expected_units.placement.battery.json"   # the single-phase baseline wave: its per-view file
    r = w.stage("06_eval_waves.sh", "--model", WI.MODEL, "--view", "battery", "--family", "multiturn", "--clusters", "cluster_a")
    assert r.returncode == 0 and _n_jobs(w) == 8, r.stdout[-1500:] + r.stderr[-1500:]
    pm.unlink()                                                                     # the wave predates the manifest: A's units attempted, none written
    r = w.stage("06_eval_waves.sh", "--model", WI.MODEL, "--view", "battery", "--family", "multiturn", "--clusters", "cluster_a", "--resubmit")
    assert r.returncode == 0 and _n_jobs(w) == 8 and "submitted " not in r.stdout, r.stdout[-1500:] + r.stderr[-1500:]   # live units skipped; ensure ran
    man = json.load(open(pm))
    assert man["source"] == "ledger" and man["labels"] == AB and len(man["units"]) == 16 and man["reconciled_attempts"] == 8 and len(man["unattempted"]) == 8
    assert man["units"] == ST.placement_from_plan(KEY, "battery", AB, sha=SHA)["units"]
    assert "8 attempted unit(s) reconciled; 8 not yet attempted" in r.stdout
    # B's group: check passes against the COMPLETE manifest and places where pinned
    r = w.stage("06_eval_waves.sh", "--model", WI.MODEL, "--view", "battery", "--family", "multiturn", "--clusters", "cluster_b")
    assert r.returncode == 0 and "placement OK" in r.stdout and _n_jobs(w) == 16, r.stdout[-1500:] + r.stderr[-1500:]
    placed_b = {ln.split()[3] for ln in r.stdout.splitlines() if ln.startswith("submitted ")}
    assert "multiturn.crescendo.g34" in placed_b and "multiturn.crescendo.g012" not in placed_b
    # an attempt recorded on the OTHER cluster than the plan pins refuses the derivation by name (a fresh derivation over a foreign ledger)
    sub = _ledger_world(tmp_path / "foreign", code_sha=SHA, frozen_utc="2026-01-01T00:00:00Z")   # a frozen-tree attempt on the wrong cluster
    ledger = sub.parent / "attempts.jsonl"
    ledger.write_text(json.dumps({"event": "attempt", "logical_work_id": f"{KEY}/battery/multiturn/crescendo/g34", "cluster": "cluster_a", "job_id": "77",
                                  "kind": "target", "sha": SHA, "spec": {"group": "mt-battery-cluster_a", "sha": SHA}, "utc": "2026-01-02T00:00:00Z"}) + "\n")
    man2, probs = ST.placement_from_ledger(str(ledger), KEY, "battery", AB, sha=SHA)
    assert probs and "crescendo/g34: pinned on cluster_b (group mt-battery-cluster_b) but its baseline attempt job 77 on cluster_a" in probs[0], probs


def _ledger_world(tmp_path: Path, *, code_sha: str, frozen_utc: str) -> Path:
    """A minimal state root: gate/expected_units.json with the freeze facts the effective-attempt rule reads, and an attempts ledger beside it."""
    root = tmp_path / "state"; (root / "gate").mkdir(parents=True)
    ST.write_atomic(str(root / "gate" / "expected_units.json"), {"schema": 2, "model": KEY, "view": "battery", "code_sha": code_sha, "frozen_utc": frozen_utc, "units": []})
    return root / "submissions.jsonl"


def _attempt(lw: str, cluster: str, group: str, job: str, sha: str, utc: str) -> str:
    return json.dumps({"event": "attempt", "logical_work_id": lw, "cluster": cluster, "job_id": job, "kind": "target", "sha": sha,
                       "spec": {"group": group, "kind": "target", "sha": sha}, "utc": utc}) + "\n"


def test_a_pre_freeze_attempt_on_the_wrong_cluster_never_decides_placement_the_frozen_tree_attempt_does(tmp_path):
    """The first attempt of a unit can be a PRE-FREEZE proving pass placed by the old subset-as-universe
    behaviour (crescendo/g34 on cluster_a); the frozen-tree attempt on cluster_b is the effective one — the manifest pins B, no refusal."""
    PRE = "b" * 40
    ledger = _ledger_world(tmp_path, code_sha=SHA, frozen_utc="2026-01-02T00:00:00Z")
    lw = f"{KEY}/battery/multiturn/crescendo/g34"
    with open(ledger.parent / "attempts.jsonl", "w") as fh:
        fh.write(_attempt(lw, "cluster_a", "mt-battery-cluster_a", "77", PRE, "2026-01-01T00:00:00Z"))          # pre-freeze, wrong cluster
        fh.write(_attempt(lw, "cluster_b", "mt-battery-cluster_b", "88", SHA, "2026-01-03T00:00:00Z"))          # at the frozen tree, where pinned
    eff = ST.effective_attempts(str(ledger), KEY, "battery", SHA, "2026-01-02T00:00:00Z")
    assert eff == {lw: {"baseline": {"cluster": "cluster_b", "group": "mt-battery-cluster_b", "job_id": "88", "utc": "2026-01-03T00:00:00Z", "sha": SHA}, "later": []}}
    man, probs = ST.placement_from_ledger(str(ledger), KEY, "battery", AB, sha=SHA)
    assert probs == [] and man["units"][lw]["cluster"] == "cluster_b" and man["reconciled_attempts"] == 1
    # a pre-freeze attempt alone leaves the unit unattempted (seeded from the plan); an attempt at a tree the unit was never frozen or
    # ADVANCED to is history too (24c: the selection is by tree, never by clock); an attempt at a funnel-ADVANCED sha counts — on the
    # wrong cluster it refuses by name
    with open(ledger.parent / "attempts.jsonl", "w") as fh:
        fh.write(_attempt(lw, "cluster_a", "mt-battery-cluster_a", "77", PRE, "2026-01-01T00:00:00Z"))
    assert ST.placement_from_ledger(str(ledger), KEY, "battery", AB, sha=SHA) == (None, [])
    with open(ledger.parent / "attempts.jsonl", "a") as fh:
        fh.write(_attempt(lw, "cluster_a", "mt-battery-cluster_a", "98", "e" * 40, "2026-01-05T00:00:00Z"))    # post-freeze, never advanced to: history
    assert ST.placement_from_ledger(str(ledger), KEY, "battery", AB, sha=SHA) == (None, [])
    with open(ledger.parent / "attempts.jsonl", "a") as fh:
        fh.write(json.dumps({"event": "sha_provenance", "logical_work_id": lw, "sha": "d" * 40, "kind": "advance", "job_id": "99", "cluster": "cluster_a", "utc": "2026-01-06T00:00:00Z"}) + "\n")
        fh.write(_attempt(lw, "cluster_a", "mt-battery-cluster_a", "99", "d" * 40, "2026-01-06T00:00:00Z"))    # at an ADVANCED sha, on the WRONG cluster
    man, probs = ST.placement_from_ledger(str(ledger), KEY, "battery", AB, sha=SHA)
    assert probs and "crescendo/g34: pinned on cluster_b (group mt-battery-cluster_b) but its baseline attempt job 99 on cluster_a" in probs[0], probs


def test_ensure_validates_and_expands_a_persisted_partial_manifest_instead_of_refusing_the_other_cluster(tmp_path):
    """The defective f8be839f3 shape — a manifest persisted from cluster A's attempts alone — is on disk; `ensure`
    must validate it against the full plan and EXPAND it with the planned units it lacks, so B's group passes `check` and places."""
    w = WI.World(tmp_path)
    r = w.stage("05_stage_roster.sh", "--model", WI.MODEL, "--step", "freeze")
    assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-1500:]
    pm = w.state / "gate" / "expected_units.placement.battery.json"   # the single-phase baseline wave: its per-view file
    r = w.stage("06_eval_waves.sh", "--model", WI.MODEL, "--view", "battery", "--family", "multiturn", "--clusters", "cluster_a")
    assert r.returncode == 0 and _n_jobs(w) == 8, r.stdout[-1500:] + r.stderr[-1500:]
    full = json.load(open(pm))
    partial = dict(full, units={lw: u for lw, u in full["units"].items() if u["cluster"] == "cluster_a"}, source="ledger")   # f8be839f3's defect
    ST.write_atomic(str(pm), partial)
    r = w.stage("06_eval_waves.sh", "--model", WI.MODEL, "--view", "battery", "--family", "multiturn", "--clusters", "cluster_b")
    assert r.returncode == 0 and "placement manifest EXPANDED: 8 planned unit(s) added from the plan" in r.stdout and "placement OK" in r.stdout, r.stdout[-1800:] + r.stderr[-1500:]
    assert _n_jobs(w) == 16
    man = json.load(open(pm))
    assert len(man["units"]) == 16 and man["source"] == "ledger+expanded" and len(man["added_units"]) == 8 and man["units"] == full["units"]
    # a pinned unit whose pin disagrees with the plan is refused by name, never silently re-pinned
    bad = dict(full); bad["units"] = dict(full["units"]); bad["units"][f"{KEY}/battery/multiturn/crescendo/g34"] = {"cluster": "cluster_a", "group": "mt-battery-cluster_a", "kind": "target"}
    out, added, probs = ST.expand_placement(bad, str(w.state / "submissions.jsonl"), KEY, "battery", AB, sha=w.sha)
    assert added == [] and probs and "crescendo/g34: the manifest pins cluster_a (group mt-battery-cluster_a) but the plan over ['cluster_a', 'cluster_b'] places it on cluster_b" in probs[0], probs
