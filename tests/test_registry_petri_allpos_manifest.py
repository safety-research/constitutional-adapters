"""petri_stats resolves the qwen35_4b_it x arditi_allpos cell through a declared extra manifest (registry follow-ups-4 item 12).

Since the 2026-09-20 cell grammar made `arditi_allpos` a vector_all object, no Petri arm view and no bank VIEW_MANIFEST on this host names
(qwen35_4b_it, arditi_allpos): `ObjectResolver.resolve` tried the arm view, then the bank view `qwen35_4b/cells_view/it/VIEW_MANIFEST.json`
(no entry) and refused the cell by name, so the 2026-10-05 re-read dropped its 7 runs (756 -> 749, the run-set-drop hold). The
manifest that DOES name it is the Arditi all-positions ladder's c = 0.3 rung view (`~/arditi-stage/objects/arditi_allpos/qwen35_4b/ladder/
cells_view/ar_c0p3/VIEW_MANIFEST.json`: one entry, cell_id arditi_allpos, staged_id ar_c0p3__arditi_allpos, object ba2212f0...). petri_stats
now declares that one pinned path (`PETRI_EXTRA_MANIFESTS`) through the resolver's existing `extra_manifests`, the resolver names the manifest
an entry came from and the manifests it looked in when it refuses, and the adapter's EMISSION_VERSION moves so the next read adds the 7 runs
back. Also fixed, in the same family: the status row of a `lineage_common.RowRefusals` adapter said `no row refused` while its
`$.row_refusals` ledger listed the refusal (the runner read only the shared emitter's report); `runner.row_refusals_of` reads the ledger too.
"""
from __future__ import annotations

import json
import os

import _b1_fixtures as F  # noqa: E402

from sisteer.registry.adapters import base as AB
from sisteer.registry.adapters import lineage_common as LC
from sisteer.registry.adapters import petri_stats as PS
from sisteer.registry.adapters import runner as RUN

VIEW, CELL = "qwen35_4b_it", "arditi_allpos"
SHA = "ba2212f0c2f9ffef2997bdd24831182052560cf980054bcee9cf95ea0db085b2"


def _bank_view_without_the_cell(objects_root: str) -> str:
    d = os.path.join(objects_root, "qwen35_4b", "cells_view", "it")
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, "VIEW_MANIFEST.json")
    with open(p, "w", encoding="utf-8") as fh:
        json.dump({"wave": "synthetic", "target": "qwen35_4b", "view": "it", "bank": objects_root, "written_at": "2026-01-01T00:00:00Z",
                   "entries": [{"cell_id": "constv2-mix-full__lora_r4_ce", "staged_id": "it__constv2-mix-full__lora_r4_ce", "object_sha256": F.SHA_A, "identity": "staged", "mode": "symlink"}]}, fh)
    return p


def _ladder_rung_manifest(path: str) -> str:
    """The arditi-stage rung view's shape: one entry, the staged all-positions object (its `source` and `fit_record` as the live file carries them)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"wave": "arditi_allpos", "target": "qwen35_4b", "view": "ar_c0p3", "bank": os.path.dirname(os.path.dirname(path)), "written_at": "2026-09-20T00:00:00+00:00",
                   "entries": [{"cell_id": CELL, "staged_id": "ar_c0p3__arditi_allpos", "source": "<stage>/objects/arditi_allpos/qwen35_4b/ladder/cells/ar_c0p3__arditi_allpos",
                                "object_sha256": SHA, "identity": "fit_provenance", "mode": "symlink",
                                "fit_record": {"positions_at_fit": "all", "source": "fit_provenance:scripts/arditi/fit_arditi.py", "status": "verified"}}]}, fh)
    return path


def test_without_the_extra_manifest_the_cell_is_refused_by_name_and_with_it_resolves_to_the_staged_object(tmp_path):
    seeds = F.synthetic_seeds()
    objects_root = str(tmp_path / "objects")
    bank = _bank_view_without_the_cell(objects_root)
    # the current behavior pinned: the arm view is absent, the bank view has no entry -> refused by name, the bank view named
    bare = LC.ObjectResolver(seeds, objects_root=objects_root, petri_manifests_root=str(tmp_path / "arm_views"))
    r0 = bare.resolve(VIEW, CELL)
    assert r0.object_id is None and r0.object_kind == "vector_all" and r0.confidence == "unverified"
    assert f"no VIEW_MANIFEST entry for ({VIEW}, {CELL}) under " in r0.source and LC.rel_home(bank) in r0.source and "extra manifest" not in r0.source
    # the declared extra manifest: resolved, the staged id and the manifest it came from on the record
    extra = _ladder_rung_manifest(str(tmp_path / "stage" / "objects" / "arditi_allpos" / "qwen35_4b" / "ladder" / "cells_view" / "ar_c0p3" / "VIEW_MANIFEST.json"))
    res = LC.ObjectResolver(seeds, objects_root=objects_root, petri_manifests_root=str(tmp_path / "arm_views"), extra_manifests=[extra])
    r1 = res.resolve(VIEW, CELL)
    assert r1.object_id == SHA and r1.object_kind == "vector_all" and r1.paper_label == "arditi_allpos" and r1.confidence == "recorded"
    assert r1.staged_id == "ar_c0p3__arditi_allpos" and r1.manifest_uri == LC.rel_home(extra) and r1.manifest_uri != LC.rel_home(bank)
    assert extra in res.inputs                                                   # the manifest is a read input (registered like the bank views)
    # the bank view still answers first for its own cells, and a cell in neither names both places
    assert res.resolve(VIEW, "constv2-mix-full__lora_r4_ce").object_id == F.SHA_A and res.resolve(VIEW, "constv2-mix-full__lora_r4_ce").manifest_uri == LC.rel_home(bank)
    r2 = res.resolve(VIEW, "constv2ctl2-full__lora_r4_ce")
    assert r2.object_id is None and LC.rel_home(bank) in r2.source and "nor in the declared extra manifest(s) " + LC.rel_home(extra) in r2.source
    # an extra manifest that is absent on this host is simply not consulted (never a crash)
    res3 = LC.ObjectResolver(seeds, objects_root=objects_root, petri_manifests_root=str(tmp_path / "arm_views"), extra_manifests=[str(tmp_path / "absent" / "VIEW_MANIFEST.json")])
    assert res3.resolve(VIEW, CELL).object_id is None


def test_petri_stats_declares_the_one_pinned_manifest_and_its_version_moved():
    assert PS.PETRI_EXTRA_MANIFESTS == (os.path.join(LC.HOME, "arditi-stage", "objects", "arditi_allpos", "qwen35_4b", "ladder", "cells_view", "ar_c0p3", "VIEW_MANIFEST.json"),)
    assert all("*" not in p and "?" not in p and p.endswith("VIEW_MANIFEST.json") for p in PS.PETRI_EXTRA_MANIFESTS)
    ad = PS.PetriStatsAdapter(seeds=F.synthetic_seeds())
    assert ad.objects._extra == list(PS.PETRI_EXTRA_MANIFESTS)
    # 3 was this module's own move (the pinned allpos manifest); 4 = the stats reader's run record states the identity columns and marks
    # the rest unstated, a marked null a withdrawal (unpublished, so no second bump)
    assert PS.PetriStatsAdapter.EMISSION_VERSION == 4
    # the live manifest, when present on this host, names exactly the cell and the object the ruling expects
    live = PS.PETRI_EXTRA_MANIFESTS[0]
    if os.path.isfile(live):
        with open(live, encoding="utf-8") as fh:
            entries = json.load(fh)["entries"]
        assert [(e["cell_id"], e["staged_id"], e["object_sha256"]) for e in entries] == [(CELL, "ar_c0p3__arditi_allpos", SHA)]


class _LedgerAdapter:
    """The shape of a `lineage_common.RowRefusals` adapter (petri_stats, bon_stores, the seed-replicate readers): a ledger, no shared-emitter report."""
    name = "ledger_adapter"
    covers_sources = ("petri_stats",)
    owner = "tests"

    def __init__(self):
        self.refusals = LC.RowRefusals(self.name)

    def status_sources(self, group):
        return ("petri_stats",)


class _ReportAdapter:
    name = "report_adapter"

    def __init__(self, refusals):
        class _Report:
            pass
        self.last_report = _Report()
        self.last_report.refusals = list(refusals)
        self.refusals = LC.RowRefusals(self.name)
        self.refusals.add("ledger|row", "a ledger line the report must not hide")


def test_a_row_refusals_ledger_counts_on_the_status_row_as_the_report_does():
    ad = _LedgerAdapter()
    assert RUN.row_refusals_of(ad) == []
    ad.refusals.add(f"{VIEW}|{CELL}", f"object unresolvable: no VIEW_MANIFEST entry for ({VIEW}, {CELL}) under <bank>")
    assert RUN.row_refusals_of(ad) == [f"{VIEW}|{CELL} -- object unresolvable: no VIEW_MANIFEST entry for ({VIEW}, {CELL}) under <bank>"]
    # the shared emitter's report, where an adapter has one, stays the first source; the ledger is read only when the report lists nothing
    rep = _ReportAdapter(["m/c/e: a row the emitter refused"])
    assert RUN.row_refusals_of(rep) == ["m/c/e: a row the emitter refused"]
    rep2 = _ReportAdapter([])
    assert RUN.row_refusals_of(rep2) == ["ledger|row -- a ledger line the report must not hide"]
    # an adapter with neither reports none
    class _Bare:
        name = "bare"
    assert RUN.row_refusals_of(_Bare()) == []
    # through the status records: the detail counts the ledger's row and the extra lists it (the held petri_stats revision said "no row refused")
    ctx = AB.AdapterContext(root="", now="2026-10-06T00:00:00.000Z", producer={"script": "tests", "code_sha": "unverified", "session": None, "host": "TESTHOST", "substrate": "grading_host"})
    tally = RUN._ReadTally() if hasattr(RUN, "_ReadTally") else None
    if tally is not None:
        recs = RUN._source_status_records(ctx, ad, "qwen35_4b", "petri_stats:qwen35_4b", "rev-synthetic", ["in://a"], tally)
        assert recs and all("1 row(s) refused by name" in r["payload"]["detail"] and "no row refused" not in r["payload"]["detail"] for r in recs), [r["payload"]["detail"] for r in recs]
        assert all(json.loads(r["payload"]["extra_json"])["row_refusals"] == RUN.row_refusals_of(ad) for r in recs) if "extra_json" in recs[0]["payload"] else \
            all(r["payload"]["extra"]["row_refusals"] == RUN.row_refusals_of(ad) for r in recs)
