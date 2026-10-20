"""`registry stores attest` -- the re-attestation of a published generation's directory listings (registry-hardening-2c, 2026-10-07):
the designed path for a generation verified before the 2026-10-04 directory-listings rule (`listings_unattested` on the mirror) and for a
listing moved since its attestation (`listings_changed`). With the shimmed runner: an unattested generation is attested by a keyed verify
(MANIFEST-<sha8>.<lsha8>.json beside the tree, the pinned verifier run against THAT attestation into VERIFIED-<sha8>.<lsha8>.json with its
snapshot keyed too, `published_attestation` recorded) and every pin-side check reads the new attestation; the live VERIFIED / DEST-STAT
stand byte for byte; a second attest is `already_attested` without a job; the mirror's next tick is `unchanged` and keeps the attestation.
THE DRIFT RULE both ways: a directory the generation holds that the canonical lacks (the live `.partial/` case) and an attested directory
the generation lacks (the canonical added an empty directory) are refused by name, `listing_only_drift` (AMBER: a listing-coverage
limitation, the pin unchanged, the unit exits 0), nothing created or removed in the sealed tree, nothing recorded; the mirror reads that
verdict on every tick without a job. `--all-unattested` over a mixed world attests only what the last tick recorded; `content_differs`
leaves the publication to the mirror. The stdlib verify entry point takes `--attestation` and the sbatch forwards it; a sealed parent
refuses the verify's empty-directory creation without a crash. After the adversarial review of 2c: a re-attestation that finds the FILES
wrong stands as `attest_integrity_failed` (red) on every ordinary tick until a later full hash passes; the record, the adoption and the
`--full` re-hash run BEFORE the listing classification; `--full` re-hashes against the attestation that verified the current listings.
Round 1 of the review: the attest refuses a `current` without its SEALED / PUBLISHED evidence (`current_unsealed` / `current_unrecorded`,
red, no job) instead of writing `generation.state = published` over the mirror's red; and the REVERSED mirror (a `cluster_a` component
with a grading host or a frozen producer) names a standing re-attestation integrity failure red on every tick, as the carry does, instead of
`verified` from the two manifests alone. Round 2: an OLDER report for the listings the canonical returned to is never reused over a
LATER integrity failure (the attest re-verifies in full instead of `already_attested` / `listing_only_drift`), and the reversed mirror
names the standing failure first when the grading host copy DIFFERS from the failing `current` too (`unpushed_changes`, `frozen_differs` and a
deferred carry are outranked; an undeferred carry and the producer's push supersede it). Round 3: a report speaks for the tree it was run
over -- every report and snapshot of a tree the mirror replaces (renamed aside or gone), the keyed re-attestations included, is moved
aside before the fresh copy, so a keyed report of the old tree never attests the fresh one; and the attest's authority refusals
(`no_current`, `current_dangling`, `current_unverified`) record what the mirror records, so `stores status` never shows green and
verified over a generation the attest just found missing or unverified.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from _env import permission_denial_enforced
from sisteer.registry import cli
from sisteer.registry import records as R
from sisteer.registry import stores as ST

from test_registry_stores import Runner, SBATCH, SRC, _RsyncCut, _SealBlip, _canon, _cfg, _dir_modes, _pre_rule_generation, _renamed_away, _write, world  # noqa: E402,F401  (the fixture rides along)

SLUG = "relu_paired_run_state"


def _publish(w, runner=None):
    res = ST.mirror_component(w["root"], w["dest"], w["comp"], runner=runner or Runner())
    assert res.outcome == "published", res.as_dict()
    return res.content_sha8


def _bytes(paths):
    return {p: open(p, "rb").read() for p in paths}


def _live_evidence(gd, sha8):
    return [f"{gd}/VERIFIED-{sha8}.json", f"{gd}/DEST-STAT-{sha8}.json", f"{gd}/MANIFEST-{sha8}.json", f"{gd}/MANIFEST-{sha8}.sha256",
            f"{gd}/SEALED-{sha8}.json", f"{gd}/PUBLISHED-{sha8}.json"]


def test_an_unattested_generation_is_attested_by_a_keyed_verify_and_every_pin_side_check_reads_the_new_attestation(world):
    w = world
    root, cfg, comp, dest, gd = w["root"], w["cfg"], w["comp"], w["dest"], w["gen_dir"]
    sdir = ST.component_state_dir(root, SLUG)
    os.makedirs(os.path.join(w["base"], "x", "killed_empty"))
    sha8 = _publish(w)
    lsha = _pre_rule_generation(w, sha8)
    assert ST.mirror_component(root, dest, comp, runner=Runner()).outcome == "listings_unattested"
    live = _bytes(_live_evidence(gd, sha8))
    rn = Runner()
    r = ST.attest_listings(root, dest, comp, runner=rn)
    assert r.outcome == "attested" and r.level == "green" and r.ok and r.content_sha8 == sha8, r.as_dict()
    assert r.listings_sha256 == lsha and r.published_attestation == lsha and r.detail["n_directories"] == 7 and r.detail["n_empty_directories"] == 1
    # the keyed attestation beside the tree and on the grading host; the keyed report and snapshot; the live evidence untouched byte for byte
    keyed_m = f"{gd}/MANIFEST-{sha8}.{lsha[:8]}.json"
    assert r.detail["keyed_manifest"] == keyed_m and json.load(open(keyed_m))["listings_sha256"] == lsha and json.load(open(keyed_m))["directories"]["x/killed_empty"] == []
    assert os.path.isfile(f"{sdir}/MANIFEST-{sha8}.{lsha[:8]}.json")
    out = f"{gd}/VERIFIED-{sha8}.{lsha[:8]}.json"
    assert r.detail["report_path"] == out == ST.keyed_report_path(gd, sha8, lsha)
    rep = json.load(open(out))
    assert rep["ok"] is True and rep["listings_attested"] is True and rep["listings_sha256"] == lsha and rep["listing_mismatch"] == [] and rep["n_directories"] == 7
    assert rep["attestation"] == os.path.abspath(keyed_m) and rep["job_id"] == r.job_id and rep["hashed"] == 4
    assert rep["dirs_generation_only"] == [] and rep["dirs_canonical_only"] == [] and rep["n_dirs_generation_only"] == rep["n_dirs_canonical_only"] == 0
    assert rep["dest_stat"] == f"DEST-STAT-{sha8}.{lsha[:8]}.json" and os.path.isfile(f"{gd}/{rep['dest_stat']}")
    assert _bytes(_live_evidence(gd, sha8)) == live, "the live report, snapshot, manifest, seal and record are never rewritten"
    sb = [a for a in rn.calls if a[0] == "sbatch"]
    assert len(sb) == 1 and f"--job-name=store-attest-{SLUG}" in sb[0] and sb[0][-2:] == ["--attestation", keyed_m] and "--dependency=singleton" in sb[0]
    assert sb[0][sb[0].index("--attestation") - 2] == out, "the keyed report path is the verify's --out"
    # the state: the attestation the window pins, the generation green again, last_run in the attest namespace
    st = ST.load_state(root)["components"][SLUG]
    assert st["published_attestation"] == lsha and st["published"] == sha8 and st["generation"]["state"] == "published" and st["generation"]["content_sha8"] == sha8
    assert st["generation"]["listings_sha256"] == lsha and st["generation"]["attest_report"] == out and st["generation"]["attest_job_id"] == r.job_id
    assert st["last_run"]["outcome"] == "attest:attested" and ST.last_run_level(st["last_run"]["outcome"]) == "green"
    assert ST.published_attestations(root) == {SLUG: lsha} and ST.status(root, cfg)["components"][SLUG]["level"] == "green"
    # every pin-side check reads the attestation and passes
    rv = ST.revalidate(SLUG, sha8, root=root, cfg=cfg, listings_sha256=lsha)
    assert rv.ok and rv.n_directories == 7 and rv.listings_sha256 == lsha, rv.as_dict()
    ad = ST.admit_generation(gd, sha8, listings_sha256=lsha)
    assert ad.ok and ad.n_directories == 7 and ad.listings_sha256 == lsha, ad.as_dict()
    ix = ST.manifest_index(gd, sha8, listings_sha256=lsha)
    assert ix.listings_attested and ix.listings_sha256 == lsha and ix.attested_listing("x/killed_empty") == () and ix.attested_listing("a/b") == (("f1.json", "file"),)
    assert ST.adapter_admission(cfg, "relu_paired", {SLUG: sha8}, root=root, attestations={SLUG: lsha})[SLUG].ok
    # the pre-rule pin (no attestation named) still reads the plain .json, which attests nothing: admitted, no listing checked
    assert ST.revalidate(SLUG, sha8, root=root, cfg=cfg).ok and ST.revalidate(SLUG, sha8, root=root, cfg=cfg).n_directories is None
    # the mirror's next tick: unchanged, the attestation kept, no job
    rn2 = Runner()
    res = ST.mirror_component(root, dest, comp, runner=rn2)
    assert res.outcome == "unchanged" and res.level == "green" and rn2.count("sbatch") == 0, res.as_dict()
    st2 = ST.load_state(root)["components"][SLUG]
    assert st2["published_attestation"] == lsha and st2["generation"]["state"] == "published" and st2["generation"]["listings_sha256"] == lsha
    # a second attest: already attested, by the keyed report, no job
    rn3 = Runner()
    r2 = ST.attest_listings(root, dest, comp, runner=rn3)
    assert r2.outcome == "already_attested" and r2.ok and r2.published_attestation == lsha and rn3.count("sbatch") == 0, r2.as_dict()
    assert r2.detail["attested_by"] == os.path.basename(out) and ST.load_state(root)["components"][SLUG]["last_run"]["outcome"] == "attest:already_attested"
    assert _bytes(_live_evidence(gd, sha8)) == live
    # then the canonical's listings MOVE from the attested ones (an empty directory lands, no file change): listings_changed (red), the pin
    # keeps the attested listings and revalidate refuses the moved ones -- the readers that state listings take the grading host route
    os.makedirs(os.path.join(w["base"], "a", "later_empty"))
    res3 = ST.mirror_component(root, dest, comp, runner=Runner())
    assert res3.outcome == "listings_changed" and res3.level == "red" and res3.detail["published_listings_sha256"] == lsha and res3.detail["listings_sha256"] != lsha, res3.as_dict()
    assert ST.load_state(root)["components"][SLUG]["published_attestation"] == lsha
    assert ST.revalidate(SLUG, sha8, root=root, cfg=cfg, listings_sha256=lsha).refusal == "listing_changed"
    # the attest of the moved listing: the canonical only ADDED an empty directory the sealed generation lacks -> refused by name, nothing
    # created inside the published tree, the recorded attestation untouched
    rn4 = Runner()
    r3 = ST.attest_listings(root, dest, comp, runner=rn4)
    assert r3.outcome == "listing_only_drift" and r3.level == "amber" and r3.ok, r3.as_dict()
    assert r3.detail["dirs_canonical_only"] == ["a/later_empty"] and r3.detail["dirs_generation_only"] == [] and r3.detail["listing_mismatch"] == ["a", "a/later_empty"]
    assert "the canonical holds 1 director(y/ies) the generation lacks" in r3.detail["why"] and "nothing is created inside a published, sealed generation" in r3.detail["why"]
    assert not os.path.lexists(f"{gd}/{sha8}/a/later_empty") and rn4.count("sbatch") == 1
    L2 = r3.listings_sha256
    assert L2 != lsha and json.load(open(f"{gd}/VERIFIED-{sha8}.{L2[:8]}.json"))["ok"] is False and os.path.isfile(f"{gd}/MANIFEST-{sha8}.{L2[:8]}.json")
    st3 = ST.load_state(root)["components"][SLUG]
    assert st3["published_attestation"] == lsha and st3["generation"]["state"] == "listing_only_drift" and st3["last_run"]["outcome"] == "attest:listing_only_drift"
    assert _bytes(_live_evidence(gd, sha8)) == live
    # the mirror reads that verdict on every tick without another job; a second attest returns it without a job
    rn5 = Runner()
    res4 = ST.mirror_component(root, dest, comp, runner=rn5)
    assert res4.outcome == "listing_only_drift" and res4.level == "amber" and res4.ok and rn5.count("sbatch") == 0 and res4.detail["dirs_canonical_only"] == ["a/later_empty"], res4.as_dict()
    assert res4.detail["published_listings_sha256"] == lsha and res4.detail["attest_report"] == f"{gd}/VERIFIED-{sha8}.{L2[:8]}.json"
    rn6 = Runner()
    assert ST.attest_listings(root, dest, comp, runner=rn6).outcome == "listing_only_drift" and rn6.count("sbatch") == 0
    # the directory gone again: the canonical is the attested generation once more
    os.rmdir(os.path.join(w["base"], "a", "later_empty"))
    assert ST.mirror_component(root, dest, comp, runner=Runner()).outcome == "unchanged"
    assert ST.load_state(root)["components"][SLUG]["published_attestation"] == lsha


def test_a_directory_the_generation_holds_and_the_canonical_lacks_is_refused_by_name_and_a_file_change_ends_the_drift(world):
    """The one live drift of 2026-10-07 (`relu_paired_run_state`: an empty `.partial/` in the generation built 09-29 that the canonical no
    longer has): the generation is immutable, so nothing removes the directory -- `listing_only_drift`, by name, with the direction; the
    live evidence and the recorded attestation stand; the mirror reads the verdict without a job; the pre-rule pin (no attestation) is
    still admitted (it never attested a listing); the next FILE change publishes a generation that attests the current listings."""
    w = world
    root, cfg, comp, dest, gd = w["root"], w["cfg"], w["comp"], w["dest"], w["gen_dir"]
    os.makedirs(os.path.join(w["base"], "x", "partial_marker"))
    sha8 = _publish(w)
    _pre_rule_generation(w, sha8)
    os.rmdir(os.path.join(w["base"], "x", "partial_marker"))          # the canonical no longer has it; the file content is unchanged
    assert ST.mirror_component(root, dest, comp, runner=Runner()).outcome == "listings_unattested"
    live = _bytes(_live_evidence(gd, sha8))
    rn = Runner()
    r = ST.attest_listings(root, dest, comp, runner=rn)
    assert r.outcome == "listing_only_drift" and r.level == "amber" and r.ok and r.published_attestation is None, r.as_dict()
    assert r.detail["dirs_generation_only"] == ["x/partial_marker"] and r.detail["dirs_canonical_only"] == [] and r.detail["listing_mismatch"] == ["x"]
    assert r.detail["n_dirs_generation_only"] == 1 and r.detail["n_dirs_canonical_only"] == 0
    assert "the generation holds 1 director(y/ies) the canonical no longer has (first: 'x/partial_marker')" in r.detail["why"]
    assert "nothing removes an entry from an immutable generation" in r.detail["why"] and "next file change" in r.detail["why"]
    assert os.path.isdir(f"{gd}/{sha8}/x/partial_marker") and _bytes(_live_evidence(gd, sha8)) == live
    L = r.listings_sha256
    rep = json.load(open(r.detail["report_path"]))
    assert r.detail["report_path"] == f"{gd}/VERIFIED-{sha8}.{L[:8]}.json" and rep["ok"] is False and rep["listing_mismatch"] == ["x"] and rep["missing"] == rep["corrupt"] == rep["extra"] == []
    st = ST.load_state(root)["components"][SLUG]
    assert st["published_attestation"] is None and st["published"] == sha8 and st["generation"]["state"] == "listing_only_drift" and st["last_run"]["outcome"] == "attest:listing_only_drift"
    assert ST.published_attestations(root) == {SLUG: None} and ST.status(root, cfg)["components"][SLUG]["level"] == "amber"
    # the pin names no attestation: admitted as before (the drift is a listing the pre-rule pin never attested)
    assert ST.revalidate(SLUG, sha8, root=root, cfg=cfg).ok and ST.admit_generation(gd, sha8).ok
    # the mirror reads the verdict, red by name, no job; a second attest returns it without a job
    rn2 = Runner()
    res = ST.mirror_component(root, dest, comp, runner=rn2)
    assert res.outcome == "listing_only_drift" and res.level == "amber" and res.ok and rn2.count("sbatch") == 0, res.as_dict()
    # a drift is a coverage limitation: with --full the files are still re-hashed (no attestation selected) and the outcome stands amber
    rn2f = Runner()
    resf = ST.mirror_component(root, dest, comp, runner=rn2f, full=True)
    assert resf.outcome == "listing_only_drift" and resf.ok and resf.detail["full"]["ok"] is True and resf.detail["full"]["attestation"] is None, resf.as_dict()
    assert any(f"--job-name=store-full-{SLUG}" in a for a in rn2f.calls if a[0] == "sbatch")
    assert res.detail["dirs_generation_only"] == ["x/partial_marker"] and res.detail["published_listings_sha256"] is None and res.published == sha8
    rn3 = Runner()
    assert ST.attest_listings(root, dest, comp, runner=rn3).outcome == "listing_only_drift" and rn3.count("sbatch") == 0
    # the next FILE change publishes a generation that attests the current listings: the drift is over
    _write(os.path.join(w["base"], "top.txt"), b"top level, revised\n")
    res2 = ST.mirror_component(root, dest, comp, runner=Runner())
    assert res2.outcome == "published" and res2.content_sha8 != sha8, res2.as_dict()
    st2 = ST.load_state(root)["components"][SLUG]
    assert st2["published_attestation"] == json.load(open(f"{gd}/VERIFIED-{res2.content_sha8}.json"))["listings_sha256"] and st2["generation"]["state"] == "published"
    assert not os.path.exists(f"{gd}/{res2.content_sha8}/x/partial_marker")
    assert ST.mirror_component(root, dest, comp, runner=Runner()).outcome == "unchanged"
    rn4 = Runner()
    assert ST.attest_listings(root, dest, comp, runner=rn4).outcome == "already_attested" and rn4.count("sbatch") == 0


def test_an_excluded_entry_landing_under_an_attested_directory_moves_the_listing_and_is_attested(world):
    """A listing change the tree DOES satisfy: a cache directory (an excluded entry) lands under the canonical base after the publish --
    the base's listing moves (its listings_sha256 with it), the staged expectation never carries excluded entries, so the generation's
    directory set is still the canonical's and the re-attestation passes: `listings_changed` -> `stores attest` -> `unchanged`."""
    w = world
    root, cfg, comp, dest, gd = w["root"], w["cfg"], w["comp"], w["dest"], w["gen_dir"]
    sha8 = _publish(w)
    L1 = json.load(open(f"{gd}/VERIFIED-{sha8}.json"))["listings_sha256"]
    os.makedirs(os.path.join(w["base"], "a", "__pycache__"))          # excluded by name; the listing of `a` names it
    res = ST.mirror_component(root, dest, comp, runner=Runner())
    assert res.outcome == "listings_changed" and res.detail["published_listings_sha256"] == L1, res.as_dict()
    rn = Runner()
    r = ST.attest_listings(root, dest, comp, runner=rn)
    assert r.outcome == "attested" and r.listings_sha256 != L1 and rn.count("sbatch") == 1, r.as_dict()
    L2 = r.listings_sha256
    assert ST.load_state(root)["components"][SLUG]["published_attestation"] == L2 and ST.manifest_index(gd, sha8, listings_sha256=L2).attested_listing("a") == (("__pycache__", "dir"), ("b", "dir"), ("f2.jsonl", "file"))
    assert ST.mirror_component(root, dest, comp, runner=Runner()).outcome == "unchanged"
    assert ST.revalidate(SLUG, sha8, root=root, cfg=cfg, listings_sha256=L2).ok and ST.revalidate(SLUG, sha8, root=root, cfg=cfg, listings_sha256=L1).refusal == "listing_changed"
    # both attestations stand, immutable, on both sides; the live report is the first one's
    assert json.load(open(f"{gd}/MANIFEST-{sha8}.{L1[:8]}.json"))["listings_sha256"] == L1 and json.load(open(f"{gd}/MANIFEST-{sha8}.{L2[:8]}.json"))["listings_sha256"] == L2
    assert json.load(open(f"{gd}/VERIFIED-{sha8}.json"))["listings_sha256"] == L1 and json.load(open(f"{gd}/VERIFIED-{sha8}.{L2[:8]}.json"))["listings_sha256"] == L2


def test_all_unattested_over_a_mixed_world_attests_only_what_the_last_tick_recorded_and_the_cli_exit_codes(tmp_path, monkeypatch, capsys):
    """Two components, one published by a pre-rule tree and one by the current one: `stores mirror` records listings_unattested for the first
    alone; `stores attest --all-unattested` attests that one and nothing else (one job), a second run finds nothing to attest (exit 0), an
    explicit `--component` on the attested one is already_attested, a content change is content_differs (amber, exit 0, no job: the
    mirror publishes), a drift exits 1, and no selection at all exits 2."""
    base_pre = tmp_path / "canon_pre"
    base_new = tmp_path / "canon_new"
    for base in (base_pre, base_new):
        _write(str(base / "d" / "f.json"), b'{"k": 1}\n')
        _write(str(base / "top.txt"), b"top\n")
        os.makedirs(base / "d" / "empty_marker")
    root = str(tmp_path / "reg")
    os.makedirs(os.path.join(root, "state"))
    cfgp = _cfg(tmp_path, {"c_pre": {"base": str(base_pre), "authority": "grading_host_copy", "origin": "mixed"},
                           "c_new": {"base": str(base_new), "authority": "grading_host_copy", "origin": "mixed"}},
                adapters={"relu_paired": {"shards": 8, "components": ["c_pre", "c_new"]}})
    cfg = ST.load_config(cfgp)
    dest = cfg.destination("cluster_a")
    monkeypatch.setattr(ST, "_default_runner", Runner())
    monkeypatch.setenv("REGISTRY_ROOT", root)
    monkeypatch.setenv(ST.CONFIG_ENV, cfgp)
    assert cli.main(["stores", "mirror"]) == 0
    first = [json.loads(l) for l in capsys.readouterr().out.splitlines()]
    assert [r["outcome"] for r in first[:2]] == ["published", "published"]
    sha_pre = next(r["content_sha8"] for r in first if r.get("slug") == "c_pre")
    w_pre = {"root": root, "gen_dir": dest.gen_dir("c_pre")}
    lsha = _pre_rule_generation(w_pre, sha_pre, slug="c_pre")
    # nothing to attest yet as the record stands: the state says `published` for both until a tick reads the destination
    assert cli.main(["stores", "attest", "--all-unattested"]) == 0
    assert "nothing to attest" in capsys.readouterr().out
    assert cli.main(["stores", "mirror"]) == 0
    lines = capsys.readouterr().out.splitlines()
    outcomes = {json.loads(l)["slug"]: json.loads(l)["outcome"] for l in lines[:-1]}
    assert outcomes == {"c_pre": "listings_unattested", "c_new": "unchanged"} and json.loads(lines[-1])["listings_unattested"] == ["c_pre"]
    rn = Runner()
    monkeypatch.setattr(ST, "_default_runner", rn)
    assert cli.main(["stores", "attest", "--all-unattested", "--json"]) == 0
    lines = capsys.readouterr().out.splitlines()
    results = [json.loads(l) for l in lines]
    assert [r["slug"] for r in results[:-1]] == ["c_pre"] and results[0]["outcome"] == "attested" and results[0]["published_attestation"] == lsha
    assert results[-1]["summary"] is True and results[-1]["outcomes"] == {"attested": 1} and results[-1]["ok"] == 1 and rn.count("sbatch") == 1
    st = ST.load_state(root)["components"]
    assert st["c_pre"]["published_attestation"] == lsha and st["c_pre"]["generation"]["state"] == "published" and st["c_new"]["generation"]["state"] == "published"
    assert ST.published_attestations(root) == {"c_pre": lsha, "c_new": json.load(open(f"{dest.gen_dir('c_new')}/VERIFIED-{st['c_new']['published']}.json"))["listings_sha256"]}
    assert cli.main(["stores", "attest", "--all-unattested"]) == 0
    assert "nothing to attest" in capsys.readouterr().out
    assert cli.main(["stores", "attest", "--component", "c_new"]) == 0
    text = capsys.readouterr().out
    assert "c_new: already_attested" in text and "stores attest: 1 of 1 ok" in text and rn.count("sbatch") == 1
    # a content change: the mirror publishes, the attest has nothing to do (amber, exit 0, no job)
    _write(str(base_new / "top.txt"), b"top, revised\n")
    assert cli.main(["stores", "attest", "--component", "c_new", "--json"]) == 0
    res = [json.loads(l) for l in capsys.readouterr().out.splitlines()]
    assert res[0]["outcome"] == "content_differs" and res[0]["level"] == "amber" and "mirror's next tick publishes" in res[0]["why"] and rn.count("sbatch") == 1
    assert ST.load_state(root)["components"]["c_new"]["last_run"]["outcome"] == "attest:content_differs"
    # a drift is amber (a coverage limitation): exit 0; the listing move before it is listings_changed, red; no selection exits 2; both
    # unattested states are selected
    os.makedirs(base_pre / "d" / "another_empty")
    assert cli.main(["stores", "mirror", "--component", "c_pre"]) == 1
    assert json.loads(capsys.readouterr().out.splitlines()[0])["outcome"] == "listings_changed"
    assert cli.main(["stores", "attest", "--all-unattested"]) == 0
    text = capsys.readouterr().out
    assert "c_pre: listing_only_drift" in text and "listing_only_drift 1" in text and "1 of 1 ok" in text
    assert cli.main(["stores", "mirror", "--component", "c_pre"]) == 0
    assert json.loads(capsys.readouterr().out.splitlines()[0])["outcome"] == "listing_only_drift"
    assert cli.main(["stores", "attest"]) == 2
    assert "--all-unattested" in capsys.readouterr().err
    assert cli.main(["stores", "attest", "--component", "nope"]) == 2


def test_the_verify_entry_point_takes_an_attestation_and_the_sbatch_forwards_it(world):
    """`python3 -S -m sisteer.registry.stores verify GEN MAN --out OUT --attestation FILE`: the listings verified are FILE's, the report
    names it and keys its snapshot; the sbatch's `verify` mode passes what follows its four positionals through to the entry point."""
    w = world
    gd = w["gen_dir"]
    os.makedirs(os.path.join(w["base"], "x", "killed_empty"))
    sha8 = _publish(w)
    lsha = _pre_rule_generation(w, sha8)
    pre = ST.build_manifest(w["comp"], None)
    assert pre.listings_sha256 == lsha
    keyed = f"{gd}/MANIFEST-{sha8}.{lsha[:8]}.json"
    ST._write_json(keyed, pre.meta())
    out = f"{gd}/VERIFIED-{sha8}.{lsha[:8]}.json"
    live = _bytes([f"{gd}/VERIFIED-{sha8}.json", f"{gd}/DEST-STAT-{sha8}.json"])
    r = subprocess.run([sys.executable, "-S", "-m", "sisteer.registry.stores", "verify", f"{gd}/{sha8}", f"{gd}/MANIFEST-{sha8}.sha256", "--out", out,
                        "--workers", "1", "--job-id", "4242", "--attestation", keyed], capture_output=True, text=True, env={**os.environ, "PYTHONPATH": SRC})
    assert r.returncode == 0, r.stderr
    rep = json.load(open(out))
    assert rep["ok"] is True and rep["listings_sha256"] == lsha and rep["attestation"] == os.path.abspath(keyed) and rep["job_id"] == "4242"
    assert rep["dest_stat"] == f"DEST-STAT-{sha8}.{lsha[:8]}.json" and os.path.isfile(f"{gd}/{rep['dest_stat']}") and _bytes(live) == live
    # without --attestation the plain .json is read (pre-rule: no listings) and the snapshot name is the live one
    rep2, _ds = ST.verify_generation(f"{gd}/{sha8}", f"{gd}/MANIFEST-{sha8}.sha256", workers=1, seal=False)
    assert rep2["listings_attested"] is False and rep2["dest_stat"] == f"DEST-STAT-{sha8}.json" and rep2["attestation"] == os.path.abspath(f"{gd}/MANIFEST-{sha8}.json")
    text = open(SBATCH, encoding="utf-8").read()
    verify_case = text[text.index("  verify)"):text.index("  manifest-only)")]
    assert 'shift 4' in verify_case and '"$@" &' in verify_case and "--attestation" in verify_case
    assert subprocess.run(["bash", "-n", SBATCH], capture_output=True, text=True).returncode == 0


@pytest.mark.skipif(not permission_denial_enforced(),
                    reason="needs the kernel to refuse a mkdir under a sealed (0o555) parent, which is the refusal asserted; this process "
                           "creates there regardless (root or CAP_DAC_OVERRIDE, as in the CPU-suite container)")
def test_a_sealed_parent_refuses_the_verifys_empty_directory_creation_and_the_directory_is_named_not_a_crash(tmp_path):
    tree = tmp_path / "tree"
    os.makedirs(tree / "p")
    _write(str(tree / "p" / "f.txt"), b"x\n")
    attested = {"": (("p", "dir"),), "p": (("child_empty", "dir"), ("f.txt", "file")), "p/child_empty": ()}
    os.chmod(tree / "p", 0o555)
    try:
        mismatched, created = ST.listing_mismatches(str(tree), attested, create_empty=True)
    finally:
        os.chmod(tree / "p", 0o755)
    assert created == [] and mismatched == ["p", "p/child_empty"] and not os.path.exists(tree / "p" / "child_empty")
    # a writable parent: created, as the publish's verify does
    mismatched2, created2 = ST.listing_mismatches(str(tree), attested, create_empty=True)
    assert created2 == ["p/child_empty"] and mismatched2 == [] and os.path.isdir(tree / "p" / "child_empty")


def test_the_verify_reads_a_sealed_parent_and_never_asks_the_kernel_to_create_under_it(tmp_path, monkeypatch):
    """The invariant behind the test above, whoever runs the process: nothing is created inside a sealed generation. The seal is READ -- the
    nearest existing ancestor of the absent directory is a directory with no write bit (DIR_SEAL_MODE) -- before any creation is attempted,
    so a process the kernel would not refuse (root / CAP_DAC_OVERRIDE, the CPU-suite container) does not create inside the published tree
    either: without the read, the re-attestation's verify created the directory there and attested a tree neither attestation describes.
    A writable parent is created under as before. Privilege-independent: the recorder stands in for the kernel's answer."""
    tree = tmp_path / "tree"
    os.makedirs(tree / "a")
    _write(str(tree / "a" / "f.txt"), b"x\n")
    attested = {"": (("a", "dir"),), "a": (("f.txt", "file"), ("later_empty", "dir")), "a/later_empty": ()}
    asked = []
    real_makedirs = os.makedirs

    def recording_makedirs(path, *args, **kw):
        asked.append(path)
        return real_makedirs(path, *args, **kw)

    monkeypatch.setattr(os, "makedirs", recording_makedirs)
    os.chmod(tree / "a", ST.DIR_SEAL_MODE)
    try:
        mismatched, created = ST.listing_mismatches(str(tree), attested, create_empty=True)
    finally:
        os.chmod(tree / "a", 0o755)
    assert asked == [] and created == [] and mismatched == ["a", "a/later_empty"] and not os.path.lexists(tree / "a" / "later_empty")
    # the parent writable: the one creation is asked for and made
    mismatched2, created2 = ST.listing_mismatches(str(tree), attested, create_empty=True)
    assert asked == [str(tree / "a" / "later_empty")] and created2 == ["a/later_empty"] and mismatched2 == [] and os.path.isdir(tree / "a" / "later_empty")
    # a deeper absence under a sealed ancestor: the nearest EXISTING ancestor is what is read
    attested3 = {"": (("a", "dir"),), "a": (("f.txt", "file"), ("later_empty", "dir")), "a/later_empty": (("deeper", "dir"),), "a/later_empty/deeper": (("deepest", "dir"),),
                 "a/later_empty/deeper/deepest": ()}
    os.chmod(tree / "a" / "later_empty", ST.DIR_SEAL_MODE)
    try:
        mismatched3, created3 = ST.listing_mismatches(str(tree), attested3, create_empty=True)
    finally:
        os.chmod(tree / "a" / "later_empty", 0o755)
    assert len(asked) == 1 and created3 == [] and mismatched3 == ["a/later_empty", "a/later_empty/deeper", "a/later_empty/deeper/deepest"]


def test_attest_outcomes_have_levels_and_the_result_dict_shape():
    for outcome, level in ST.ATTEST_OUTCOME_LEVELS.items():
        assert ST.last_run_level(f"attest:{outcome}") == level and ST.AttestResult("s", outcome).level == level
    assert ST.last_run_level("attest:unknown_state") == "red" and not ST.AttestResult("s", "unknown_state").ok
    assert ST.AttestResult("s", "content_differs").ok and ST.AttestResult("s", "attested").ok and ST.AttestResult("s", "listing_only_drift").ok
    assert not ST.AttestResult("s", "attest_verify_failed").ok
    d = ST.AttestResult("s", "attested", content_sha8="a" * 8, listings_sha256="b" * 64, detail={"why": "w"}).as_dict()
    assert d["level"] == "green" and d["why"] == "w" and d["slug"] == "s" and "detail" not in d
    assert ST.OUTCOME_LEVELS["listings_unattested"] == "amber" and ST.OUTCOME_LEVELS["listing_only_drift"] == "amber" and ST.OUTCOME_LEVELS["listings_changed"] == "red"
    assert ST.OUTCOME_LEVELS["attest_integrity_failed"] == "red" and ST.last_run_level("attest_integrity_failed") == "red"
    assert ST.keyed_report_path("/g", "deadbeef", "c" * 64) == "/g/VERIFIED-deadbeef.cccccccc.json"


# --------------------------------------------------------------------------------------------
# the adversarial review of 2c: integrity failures persist; the record, adoption and full re-hash precede the listing classification;
# --full re-hashes against the selected attestation
# --------------------------------------------------------------------------------------------


def _corrupt_in_place(path, data):
    """Rewrite a sealed generation file's bytes (same length keeps the live stat snapshot honest about everything but the content)."""
    os.chmod(path, 0o644)
    with open(path, "r+b") as fh:
        fh.write(data)
    os.chmod(path, 0o444)


def test_a_failed_re_attestation_that_finds_corrupt_files_stands_red_on_every_tick_until_a_later_full_hash_passes(world, monkeypatch, capsys):
    """HIGH 1: a `stores attest` whose verify hashes every file and finds one corrupt is `attest_verify_failed` and RECORDED; the next
    ordinary mirror tick -- which reads the live report, written before the corruption -- names `attest_integrity_failed` (red, zero
    submissions) on every tick, through the recorded report and through the keyed report alone when the record is missing; restoring
    the bytes does not clear it (nothing re-hashed); a fresh attest that hashes every file and passes does, and so does a `--full` pass,
    which is remembered for later ticks."""
    w = world
    root, cfg, comp, dest, gd = w["root"], w["cfg"], w["comp"], w["dest"], w["gen_dir"]
    sha8 = _publish(w)
    _pre_rule_generation(w, sha8)
    assert ST.mirror_component(root, dest, comp, runner=Runner()).outcome == "listings_unattested"
    good = open(f"{gd}/{sha8}/top.txt", "rb").read()
    _corrupt_in_place(f"{gd}/{sha8}/top.txt", b"top LEVEL\n")
    r = ST.attest_listings(root, dest, comp, runner=Runner())
    assert r.outcome == "attest_verify_failed" and r.level == "red" and not r.ok and r.detail["corrupt"] == 1 and r.detail["first"]["corrupt"] == ["top.txt"], r.as_dict()
    L = r.listings_sha256
    st = ST.load_state(root)["components"][SLUG]
    fail = st["attest_integrity_failed"]
    assert fail["content_sha8"] == sha8 and fail["report_path"] == f"{gd}/VERIFIED-{sha8}.{L[:8]}.json" and fail["corrupt"] == 1 and fail["first"]["corrupt"] == ["top.txt"]
    assert st["generation"]["state"] == "attest_integrity_failed" and st["last_run"]["outcome"] == "attest:attest_verify_failed" and st["published_attestation"] is None
    # the ordinary tick: red by name, zero submissions, the live report untouched and still matching the manifest
    for _ in range(2):
        rn = Runner()
        res = ST.mirror_component(root, dest, comp, runner=rn)
        assert res.outcome == "attest_integrity_failed" and res.level == "red" and not res.ok and res.published == sha8 and rn.count("sbatch") == 0, res.as_dict()
        assert "1 corrupt" in res.detail["why"] and "'top.txt'" in res.detail["why"] and res.detail["attest_failure"]["report_path"] == fail["report_path"]
        st = ST.load_state(root)["components"][SLUG]
        assert st["attest_integrity_failed"]["report_path"] == fail["report_path"] and st["generation"]["state"] == "attest_integrity_failed" and st["published"] == sha8
    assert ST.status(root, cfg)["components"][SLUG]["level"] == "red"
    monkeypatch.setattr(ST, "_default_runner", Runner())
    monkeypatch.setenv("REGISTRY_ROOT", root)
    monkeypatch.setenv(ST.CONFIG_ENV, w["cfgp"])
    assert cli.main(["stores", "mirror"]) == 1
    assert json.loads(capsys.readouterr().out.splitlines()[-1])["levels"]["red"] == 1
    # restoring the bytes proves nothing by itself: the failure stands until a full hash passes
    _corrupt_in_place(f"{gd}/{sha8}/top.txt", good)
    rn = Runner()
    assert ST.mirror_component(root, dest, comp, runner=rn).outcome == "attest_integrity_failed" and rn.count("sbatch") == 0
    # a fresh attest re-verifies in full (never returns the standing failure as-is), passes, clears the record: the next tick is unchanged
    rn = Runner()
    r2 = ST.attest_listings(root, dest, comp, runner=rn)
    assert r2.outcome == "attested" and rn.count("sbatch") == 1 and r2.published_attestation == L, r2.as_dict()
    st = ST.load_state(root)["components"][SLUG]
    assert st["attest_integrity_failed"] is None and st["published_attestation"] == L and st["generation"]["state"] == "published"
    rn = Runner()
    assert ST.mirror_component(root, dest, comp, runner=rn).outcome == "unchanged" and rn.count("sbatch") == 0
    # the attest's passing verify is a full hash of every file: remembered as `--full`'s is (round 2 on 2c), so a failing keyed report
    # stamped BEFORE it is superseded (listings_changed: run attest -- never attest_integrity_failed from a stale report)
    keyed = f"{gd}/VERIFIED-{sha8}.{L[:8]}.json"
    rep = json.load(open(keyed))
    st = ST.load_state(root)["components"][SLUG]
    assert st["last_full_pass"] == {"content_sha8": sha8, "verified_utc": rep["verified_utc"], "report_path": keyed}
    ST._write_json(keyed, {**rep, "ok": False, "corrupt": ["a/f2.jsonl"], "verified_utc": "2026-10-07T00:00:00.000Z", "job_id": "9998"})
    rn = Runner()
    res = ST.mirror_component(root, dest, comp, runner=rn)
    assert res.outcome == "listings_changed" and rn.count("sbatch") == 0 and ST.load_state(root)["components"][SLUG]["attest_integrity_failed"] is None, res.as_dict()
    # the keyed report alone, stamped AFTER that pass (the attest crashed before writing the record): the tick names it from the report,
    # self-healing the record
    later = R.epoch_ms_to_utc(R.utc_to_epoch_ms(rep["verified_utc"]) + 1)
    bad = {**rep, "ok": False, "corrupt": ["a/f2.jsonl"], "verified_utc": later, "job_id": "9999"}
    ST._write_json(keyed, bad)
    st = ST.load_state(root)
    st["components"][SLUG]["attest_integrity_failed"] = None
    ST._write_json(ST.state_path(root), st)
    rn = Runner()
    res = ST.mirror_component(root, dest, comp, runner=rn)
    assert res.outcome == "attest_integrity_failed" and rn.count("sbatch") == 0 and res.detail["attest_failure"]["first"]["corrupt"] == ["a/f2.jsonl"], res.as_dict()
    assert ST.load_state(root)["components"][SLUG]["attest_integrity_failed"]["job_id"] == "9999"
    # `--full` re-hashes every file and passes: recovery proven, remembered -- the failing report still on disk is superseded on later ticks.
    # The pin keeps the attestation L the fresh attest recorded (preserved through the red tick, round 2 on 2c: a pin that names none checks
    # NO listing), and with the only report for L the superseded failing one nothing speaks for the listings: listings_changed (red: run
    # attest), never attest_integrity_failed from a stale report; a fresh attest re-verifies L in full, passes, and the tick is unchanged
    rn = Runner()
    resf = ST.mirror_component(root, dest, comp, runner=rn, full=True)
    assert resf.outcome == "listings_changed" and resf.detail["full"]["ok"] is True and resf.detail["full"]["attestation"] is None and rn.count("sbatch") == 1, resf.as_dict()
    st = ST.load_state(root)["components"][SLUG]
    assert st["attest_integrity_failed"] is None and st["last_full_pass"]["content_sha8"] == sha8 and st["last_full_pass"]["verified_utc"] > later
    assert st["published_attestation"] == L and ST.published_attestations(root) == {SLUG: L}
    rn = Runner()
    assert ST.mirror_component(root, dest, comp, runner=rn).outcome == "listings_changed" and rn.count("sbatch") == 0
    rn = Runner()
    assert ST.attest_listings(root, dest, comp, runner=rn).outcome == "attested" and rn.count("sbatch") == 1 and json.load(open(keyed))["ok"] is True
    rn = Runner()
    assert ST.mirror_component(root, dest, comp, runner=rn).outcome == "unchanged" and rn.count("sbatch") == 0


def test_the_record_the_adoption_and_the_full_re_hash_run_before_the_listing_classification(world):
    """HIGH 2: on a generation with unattested listings, `--full` still re-hashes the destination (a healthy one stays amber WITH the full
    record; a corrupt one is full_verify_failed, red), a missing PUBLISHED record is still current_unrecorded, and --adopt-unrecorded
    (the mirror's and the push's) still writes the record and seals before the amber outcome is returned."""
    w = world
    root, cfg, comp, dest, gd = w["root"], w["cfg"], w["comp"], w["dest"], w["gen_dir"]
    sha8 = _publish(w)
    _pre_rule_generation(w, sha8)
    rn = Runner()
    res = ST.mirror_component(root, dest, comp, runner=rn, full=True)
    assert res.outcome == "listings_unattested" and res.level == "amber" and res.detail["full"]["ok"] is True and res.detail["full"]["attestation"] is None, res.as_dict()
    full_calls = [a for a in rn.calls if a[0] == "sbatch" and f"--job-name=store-full-{SLUG}" in a]
    assert len(full_calls) == 1 and "--attestation" not in full_calls[0]
    st = ST.load_state(root)["components"][SLUG]
    assert st["generation"]["state"] == "listings_unattested" and st["generation"]["full"]["ok"] is True and st["last_full_pass"]["content_sha8"] == sha8
    # destination corruption under --full: red, never amber
    good = open(f"{gd}/{sha8}/top.txt", "rb").read()
    _corrupt_in_place(f"{gd}/{sha8}/top.txt", b"top LEVEL\n")
    res2 = ST.mirror_component(root, dest, comp, runner=Runner(), full=True)
    assert res2.outcome == "full_verify_failed" and res2.level == "red" and res2.detail["full"]["counts"]["corrupt"] == 1, res2.as_dict()
    _corrupt_in_place(f"{gd}/{sha8}/top.txt", good)
    # the publication record: refused without it, adopted (sealed + recorded) with the flag, then amber
    os.remove(f"{gd}/PUBLISHED-{sha8}.json")
    os.remove(f"{gd}/SEALED-{sha8}.json")
    res3 = ST.mirror_component(root, dest, comp, runner=Runner())
    assert res3.outcome == "current_unrecorded" and res3.level == "red" and not os.path.exists(f"{gd}/PUBLISHED-{sha8}.json"), res3.as_dict()
    res4 = ST.mirror_component(root, dest, comp, runner=Runner(), adopt_unrecorded=True)
    assert res4.outcome == "listings_unattested" and res4.ok and res4.detail["adopted"]["published_record"]["mode"] == "adopt" and "sealed" in res4.detail["adopted"], res4.as_dict()
    assert json.load(open(f"{gd}/PUBLISHED-{sha8}.json"))["mode"] == "adopt" and os.path.isfile(f"{gd}/SEALED-{sha8}.json")
    assert ST.load_state(root)["components"][SLUG]["generation"]["adopted"]["published_record"]["mode"] == "adopt"
    # the producer's push of unchanged content onto such a generation: the record written, the outcome accepted
    os.remove(f"{gd}/PUBLISHED-{sha8}.json")
    rp = ST.push_component(root, dest, comp, runner=Runner(), note="re-record", adopt_unrecorded=True)
    assert rp.outcome == "listings_unattested" and rp.outcome in ST.PUSH_OK_OUTCOMES and json.load(open(f"{gd}/PUBLISHED-{sha8}.json"))["mode"] == "adopt", rp.as_dict()
    assert ST.push_component(root, dest, comp, runner=Runner(), note="again").outcome == "listings_unattested"


def test_full_after_a_re_attestation_re_hashes_against_the_selected_attestation_and_passes(world):
    """MEDIUM 3: after `stores attest`, `mirror --full` hands the verifier the keyed MANIFEST-<sha8>.<lsha8>.json the verdict selected and
    passes on a healthy generation (before: the plain manifest, no listings, full_verify_failed forever); a generation attested by its live
    report re-hashes against the plain .json with the listings expected; an excluded-entry re-attestation passes too."""
    w = world
    root, cfg, comp, dest, gd = w["root"], w["cfg"], w["comp"], w["dest"], w["gen_dir"]
    os.makedirs(os.path.join(w["base"], "x", "killed_empty"))
    sha8 = _publish(w)
    # attested by the live report: --full expects the live listings, no --attestation
    L0 = json.load(open(f"{gd}/VERIFIED-{sha8}.json"))["listings_sha256"]
    rn0 = Runner()
    res0 = ST.mirror_component(root, dest, comp, runner=rn0, full=True)
    assert res0.outcome == "unchanged" and res0.detail["full"]["ok"] is True and res0.detail["full"]["attestation"] is None and res0.detail["full"]["expected_listings_sha256"] == L0, res0.as_dict()
    # the legacy generation re-attested: --full against the keyed attestation
    lsha = _pre_rule_generation(w, sha8)
    assert ST.attest_listings(root, dest, comp, runner=Runner()).outcome == "attested"
    rn = Runner()
    res = ST.mirror_component(root, dest, comp, runner=rn, full=True)
    keyed_m = f"{gd}/MANIFEST-{sha8}.{lsha[:8]}.json"
    assert res.outcome == "unchanged" and res.level == "green" and res.detail["full"]["ok"] is True, res.as_dict()
    assert res.detail["full"]["attestation"] == keyed_m and res.detail["full"]["expected_listings_sha256"] == lsha and res.detail["full"]["report"]["listings_sha256"] == lsha
    full_call = next(a for a in rn.calls if a[0] == "sbatch" and f"--job-name=store-full-{SLUG}" in a)
    assert full_call[-2:] == ["--attestation", keyed_m]
    full_report = json.load(open(res.detail["full"]["report_path"]))
    assert full_report["ok"] is True and full_report["listing_mismatch"] == [] and full_report["dest_stat"] == f"DEST-STAT-{sha8}.{lsha[:8]}.json"
    assert ST.load_state(root)["components"][SLUG]["published_attestation"] == lsha
    # an excluded entry moves the listing; re-attested; --full passes against the new keyed attestation
    os.makedirs(os.path.join(w["base"], "a", "__pycache__"))
    assert ST.mirror_component(root, dest, comp, runner=Runner()).outcome == "listings_changed"
    r2 = ST.attest_listings(root, dest, comp, runner=Runner())
    assert r2.outcome == "attested" and r2.listings_sha256 != lsha
    res2 = ST.mirror_component(root, dest, comp, runner=Runner(), full=True)
    assert res2.outcome == "unchanged" and res2.detail["full"]["ok"] is True and res2.detail["full"]["attestation"] == f"{gd}/MANIFEST-{sha8}.{r2.listings_sha256[:8]}.json", res2.as_dict()
    # a listing that MOVED from the attested one: --full re-hashes the files alone and the outcome stays listings_changed
    os.makedirs(os.path.join(w["base"], "a", "later_empty"))
    res3 = ST.mirror_component(root, dest, comp, runner=Runner(), full=True)
    assert res3.outcome == "listings_changed" and res3.detail["full"]["ok"] is True and res3.detail["full"]["attestation"] is None and res3.detail["full"]["expected_listings_sha256"] is None, res3.as_dict()


# --------------------------------------------------------------------------------------------
# round 1 of the review: the attest never turns a `current` without its evidence green; the reversed mirror honors the standing
# re-attestation integrity failure
# --------------------------------------------------------------------------------------------


EXCL = ["__pycache__", "/.pytest_cache"]


def _cluster_a_world(tmp_path, **extra_components):
    """A registry root, a local destination and the two producer shapes `stores attest` covers for a ``cluster_a`` component -- a
    grading host producer and a frozen one -- over one canonical tree (plus ``extra_components``)."""
    base = _canon(tmp_path)
    root = str(tmp_path / "reg")
    os.makedirs(os.path.join(root, "state"))
    excl = EXCL
    comps = {"dev": {"base": base, "authority": "cluster_a", "producer": "grading_host:sess", "exclude": excl},
             "frozen": {"base": base, "authority": "cluster_a", "producer": "frozen", "exclude": excl}, **extra_components}
    cfgp = _cfg(tmp_path, comps, adapters={"relu_paired": {"shards": 1, "components": list(comps)}})
    cfg = ST.load_config(cfgp)
    return {"base": base, "root": root, "cfg": cfg, "cfgp": cfgp, "dest": cfg.destination("cluster_a")}


def test_the_attest_refuses_a_current_without_its_record_or_seal_and_never_turns_the_mirrors_red_green(world, monkeypatch, capsys):
    """Round 1, HIGH: a pre-rule generation whose PUBLISHED-<sha8>.json is gone is `current_unrecorded` (red) on the mirror; before this
    the attest checked only the live report, re-verified the files, wrote `generation.state = published` and `stores status` read green
    while the publication record was still absent (and the cluster's authority check still refused the generation). Now the attest
    refuses by the mirror's own name -- no job, no keyed attestation written, the generation's state the problem, status red -- for a
    missing record, for a missing seal, and on the `already_attested` path alike; `stores mirror --adopt-unrecorded` writes what is
    missing and the attest then runs."""
    w = world
    root, cfg, comp, dest, gd = w["root"], w["cfg"], w["comp"], w["dest"], w["gen_dir"]
    sha8 = _publish(w)
    lsha = _pre_rule_generation(w, sha8)
    assert ST.mirror_component(root, dest, comp, runner=Runner()).outcome == "listings_unattested"
    os.remove(f"{gd}/PUBLISHED-{sha8}.json")
    res = ST.mirror_component(root, dest, comp, runner=Runner())
    assert res.outcome == "current_unrecorded" and res.level == "red", res.as_dict()
    assert ST.status(root, cfg)["components"][SLUG]["level"] == "red"
    live = _bytes(_live_evidence(gd, sha8)[:5])
    rn = Runner()
    r = ST.attest_listings(root, dest, comp, runner=rn)
    assert r.outcome == "current_unrecorded" and r.level == "red" and not r.ok and r.content_sha8 == sha8 and r.published_attestation is None, r.as_dict()
    assert r.detail["authority_problems"] == ["current_unrecorded"] and r.detail["adoptable"] is True
    assert f"no PUBLISHED-{sha8}.json" in r.detail["why"] and "--adopt-unrecorded" in r.detail["why"] and "nothing attests listings" in r.detail["why"]
    assert rn.count("sbatch") == 0 and r.job_id is None, "no verify job over a current without its record"
    assert not os.path.exists(f"{gd}/MANIFEST-{sha8}.{lsha[:8]}.json") and not os.path.exists(f"{gd}/VERIFIED-{sha8}.{lsha[:8]}.json"), "nothing written beside the tree"
    assert _bytes(_live_evidence(gd, sha8)[:5]) == live and not os.path.exists(f"{gd}/PUBLISHED-{sha8}.json")
    st = ST.load_state(root)["components"][SLUG]
    assert st["generation"]["state"] == "current_unrecorded" and st["generation"]["content_sha8"] == sha8 and st["generation"]["authority_problems"] == ["current_unrecorded"]
    assert st["published_attestation"] is None and st["last_run"]["outcome"] == "attest:current_unrecorded" and ST.last_run_level(st["last_run"]["outcome"]) == "red"
    assert ST.status(root, cfg)["components"][SLUG]["level"] == "red" and ST.published_attestations(root) == {SLUG: None}
    # the CLI exits 1 and the summary counts the red
    monkeypatch.setattr(ST, "_default_runner", Runner())
    monkeypatch.setenv("REGISTRY_ROOT", root)
    monkeypatch.setenv(ST.CONFIG_ENV, w["cfgp"])
    assert cli.main(["stores", "attest", "--dest", "cluster_a", "--component", SLUG, "--json"]) == 1
    lines = [json.loads(l) for l in capsys.readouterr().out.splitlines()]
    assert lines[-1]["levels"]["red"] == 1 and lines[-1]["outcomes"] == {"current_unrecorded": 1} and lines[-2]["outcome"] == "current_unrecorded"
    assert ST.status(root, cfg)["components"][SLUG]["level"] == "red"
    # the seal missing instead: refused by that name, in the mirror's order (unsealed before unrecorded when both are missing)
    os.rename(f"{gd}/SEALED-{sha8}.json", f"{gd}/SEALED-{sha8}.json.aside")
    rn = Runner()
    r2 = ST.attest_listings(root, dest, comp, runner=rn)
    assert r2.outcome == "current_unsealed" and r2.level == "red" and r2.detail["authority_problems"] == ["current_unsealed", "current_unrecorded"] and rn.count("sbatch") == 0, r2.as_dict()
    assert f"no SEALED-{sha8}.json" in r2.detail["why"] and ST.load_state(root)["components"][SLUG]["generation"]["state"] == "current_unsealed"
    os.rename(f"{gd}/SEALED-{sha8}.json.aside", f"{gd}/SEALED-{sha8}.json")
    # the adoption writes the record; the attest then runs and attests; the mirror is unchanged; status green
    ra = ST.mirror_component(root, dest, comp, runner=Runner(), adopt_unrecorded=True)
    assert ra.outcome == "listings_unattested" and ra.detail["adopted"]["published_record"]["mode"] == "adopt" and os.path.isfile(f"{gd}/PUBLISHED-{sha8}.json"), ra.as_dict()
    rn = Runner()
    r3 = ST.attest_listings(root, dest, comp, runner=rn)
    assert r3.outcome == "attested" and r3.published_attestation == lsha and rn.count("sbatch") == 1, r3.as_dict()
    assert ST.mirror_component(root, dest, comp, runner=Runner()).outcome == "unchanged" and ST.status(root, cfg)["components"][SLUG]["level"] == "green"
    # the already_attested path is guarded the same: the record gone again, the attest is current_unrecorded, never already_attested
    os.remove(f"{gd}/PUBLISHED-{sha8}.json")
    rn = Runner()
    r4 = ST.attest_listings(root, dest, comp, runner=rn)
    assert r4.outcome == "current_unrecorded" and r4.level == "red" and rn.count("sbatch") == 0, r4.as_dict()
    st = ST.load_state(root)["components"][SLUG]
    assert st["generation"]["state"] == "current_unrecorded" and st["published_attestation"] == lsha, "the attestation already recorded stands; the state is not green"
    assert ST.status(root, cfg)["components"][SLUG]["level"] == "red"
    assert ST.mirror_component(root, dest, comp, runner=Runner()).outcome == "current_unrecorded"
    ST.mirror_component(root, dest, comp, runner=Runner(), adopt_unrecorded=True)
    rn = Runner()
    assert ST.attest_listings(root, dest, comp, runner=rn).outcome == "already_attested" and rn.count("sbatch") == 0
    assert ST.status(root, cfg)["components"][SLUG]["level"] == "green"


@pytest.mark.parametrize("slug", ["dev", "frozen"])
def test_the_reversed_mirror_names_a_recorded_re_attestation_failure_red_on_every_tick_until_a_full_hash_passes(tmp_path, slug, monkeypatch, capsys):
    """Round 1, HIGH: for a `cluster_a` component with a grading host or a frozen producer, `stores attest` hashes a corrupt generation file
    and records `attest_integrity_failed`; before this the next ordinary `stores mirror` compared the two manifests and returned
    `verified` (green, exit 0) with the file still corrupt -- the standing red held only for the carry (`grading_host_copy`). Now every
    reversed tick names `attest_integrity_failed` (red, zero submissions) from the record and from the keyed report alone, the dry run
    says so, restoring bytes by hand clears nothing, a `--full` pass clears it and is remembered, and a fresh attest that passes clears
    it; the same component's `current` without its record is refused by the attest and adopted by the mirror."""
    w = _cluster_a_world(tmp_path)
    root, cfg, dest = w["root"], w["cfg"], w["dest"]
    comp = cfg.components[slug]
    gd = dest.gen_dir(slug)
    pushed = ST.push_component(root, dest, comp, runner=Runner(), **({"session": "sess"} if slug == "dev" else {}))
    assert pushed.outcome == "published", pushed.as_dict()
    sha8 = pushed.content_sha8
    assert ST.run_mirror(root, dest, comp, runner=Runner()).outcome == "verified"
    lsha = _pre_rule_generation({**w, "gen_dir": gd}, sha8, slug=slug)
    rn = Runner()
    r = ST.run_mirror(root, dest, comp, runner=rn)
    assert r.outcome == "verified" and r.level == "green" and rn.count("sbatch") == 0, r.as_dict()     # the reversed mirror classifies no listing
    good = open(f"{gd}/{sha8}/top.txt", "rb").read()
    _corrupt_in_place(f"{gd}/{sha8}/top.txt", b"top LEVEL\n")
    a = ST.attest_listings(root, dest, comp, runner=Runner())
    assert a.outcome == "attest_verify_failed" and a.level == "red" and a.detail["corrupt"] == 1 and a.detail["first"]["corrupt"] == ["top.txt"], a.as_dict()
    assert a.listings_sha256 == lsha
    fail = ST.load_state(root)["components"][slug]["attest_integrity_failed"]
    keyed = f"{gd}/VERIFIED-{sha8}.{lsha[:8]}.json"
    assert fail["content_sha8"] == sha8 and fail["report_path"] == keyed and fail["corrupt"] == 1
    # every ordinary reversed tick: red by name, zero submissions, `current` untouched, the state and the status red
    for _ in range(2):
        rn = Runner()
        r = ST.run_mirror(root, dest, comp, runner=rn)
        assert r.outcome == "attest_integrity_failed" and r.level == "red" and not r.ok and r.published == sha8 and rn.count("sbatch") == 0, r.as_dict()
        assert "1 corrupt" in r.detail["why"] and "'top.txt'" in r.detail["why"] and r.detail["attest_failure"]["report_path"] == keyed
        assert r.detail["verified_utc"] and r.detail["published_record"]["mode"] == "push" and r.detail["listings_sha256"] == lsha
        st = ST.load_state(root)["components"][slug]
        assert st["attest_integrity_failed"]["report_path"] == keyed and st["generation"]["state"] == "attest_integrity_failed"
        assert st["generation"]["attest_failure"]["report_path"] == keyed and st["published"] == sha8 and st["last_run"]["outcome"] == "attest_integrity_failed"
        assert os.readlink(f"{gd}/current") == sha8
    assert ST.status(root, cfg)["components"][slug]["level"] == "red"
    r = ST.run_mirror(root, dest, comp, runner=Runner(), dry_run=True)
    assert r.outcome == "dry_run" and r.detail["would"] == "attest_integrity_failed" and r.detail["writes"] == "nothing", r.as_dict()
    r = ST.run_mirror(root, dest, comp, runner=Runner(), dry_run=True, full=True)
    assert r.detail["would"] == "attest_integrity_failed + full re-hash"
    monkeypatch.setattr(ST, "_default_runner", Runner())
    monkeypatch.setenv("REGISTRY_ROOT", root)
    monkeypatch.setenv(ST.CONFIG_ENV, w["cfgp"])
    assert cli.main(["stores", "mirror", "--component", slug]) == 1
    out = [json.loads(l) for l in capsys.readouterr().out.splitlines()]
    assert out[-1]["levels"]["red"] == 1 and out[-2]["outcome"] == "attest_integrity_failed"
    # bytes restored by hand: nothing re-hashed, still red
    _corrupt_in_place(f"{gd}/{sha8}/top.txt", good)
    rn = Runner()
    assert ST.run_mirror(root, dest, comp, runner=rn).outcome == "attest_integrity_failed" and rn.count("sbatch") == 0
    # `--full` re-hashes every file (against the files alone: the listings are unattested) and passes: verified, the record cleared,
    # the pass remembered -- the failing keyed report still beside the generation is superseded on later ticks
    rn = Runner()
    rf = ST.run_mirror(root, dest, comp, runner=rn, full=True)
    assert rf.outcome == "verified" and rf.level == "green" and rf.detail["full"]["ok"] is True and rf.detail["full"]["attestation"] is None, rf.as_dict()
    full_calls = [c for c in rn.calls if c[0] == "sbatch" and f"--job-name=store-full-{slug}" in c]
    assert len(full_calls) == 1 and "--attestation" not in full_calls[0] and rn.count("sbatch") == 1
    st = ST.load_state(root)["components"][slug]
    assert st["attest_integrity_failed"] is None and st["last_full_pass"]["content_sha8"] == sha8 and st["generation"]["state"] == "verified"
    assert json.load(open(keyed))["ok"] is False, "the live and the keyed reports are never rewritten by the mirror"
    rn = Runner()
    assert ST.run_mirror(root, dest, comp, runner=rn).outcome == "verified" and rn.count("sbatch") == 0
    assert ST.status(root, cfg)["components"][slug]["level"] == "green"
    # the keyed report alone (the attest crashed before writing the record) stamped after the full pass: red from the report, the record
    # self-healed; a fresh attest re-verifies in full, passes and clears it -- verified again, and `--full` then re-hashes against the
    # attestation the attest wrote
    rep = json.load(open(keyed))
    ST._write_json(keyed, {**rep, "ok": False, "corrupt": ["a/f2.jsonl"], "verified_utc": "2099-01-01T00:00:00.000Z", "job_id": "9999"})
    rn = Runner()
    r = ST.run_mirror(root, dest, comp, runner=rn)
    assert r.outcome == "attest_integrity_failed" and rn.count("sbatch") == 0 and r.detail["attest_failure"]["first"]["corrupt"] == ["a/f2.jsonl"], r.as_dict()
    assert ST.load_state(root)["components"][slug]["attest_integrity_failed"]["job_id"] == "9999"
    rn = Runner()
    a2 = ST.attest_listings(root, dest, comp, runner=rn)
    assert a2.outcome == "attested" and a2.published_attestation == lsha and rn.count("sbatch") == 1, a2.as_dict()
    assert ST.load_state(root)["components"][slug]["attest_integrity_failed"] is None
    rn = Runner()
    r = ST.run_mirror(root, dest, comp, runner=rn)
    assert r.outcome == "verified" and rn.count("sbatch") == 0, r.as_dict()
    rn = Runner()
    rf2 = ST.run_mirror(root, dest, comp, runner=rn, full=True)
    keyed_m = f"{gd}/MANIFEST-{sha8}.{lsha[:8]}.json"
    assert rf2.outcome == "verified" and rf2.detail["full"]["ok"] is True and rf2.detail["full"]["attestation"] == keyed_m and rf2.detail["full"]["expected_listings_sha256"] == lsha, rf2.as_dict()
    full_call = next(c for c in rn.calls if c[0] == "sbatch" and f"--job-name=store-full-{slug}" in c)
    assert full_call[-2:] == ["--attestation", keyed_m]
    # the same component's `current` without its record: red on the mirror, refused by the attest (never already_attested), adopted by
    # the mirror explicitly, then already_attested
    os.remove(f"{gd}/PUBLISHED-{sha8}.json")
    assert ST.run_mirror(root, dest, comp, runner=Runner()).outcome == "current_unrecorded"
    rn = Runner()
    a3 = ST.attest_listings(root, dest, comp, runner=rn)
    assert a3.outcome == "current_unrecorded" and a3.level == "red" and rn.count("sbatch") == 0, a3.as_dict()
    assert ST.load_state(root)["components"][slug]["generation"]["state"] == "current_unrecorded" and ST.status(root, cfg)["components"][slug]["level"] == "red"
    ra = ST.run_mirror(root, dest, comp, runner=Runner(), adopt_unrecorded=True)
    assert ra.outcome == "verified" and ra.detail["adopted"]["published_record"]["mode"] == "adopt", ra.as_dict()
    rn = Runner()
    assert ST.attest_listings(root, dest, comp, runner=rn).outcome == "already_attested" and rn.count("sbatch") == 0
    assert ST.status(root, cfg)["components"][slug]["level"] == "green"


# --------------------------------------------------------------------------------------------
# round 2 of the review: an older report for the returned listings is never reused over a later integrity failure; the reversed mirror
# names the standing failure first when the grading host copy differs from the failing `current`
# --------------------------------------------------------------------------------------------


def _no_submission(rn):
    return rn.count("sbatch") == 0 and not any("rsync" in a for a in rn.calls)


def test_an_older_attesting_report_for_the_returned_listings_is_never_reused_over_a_later_integrity_failure(world, monkeypatch, capsys):
    """Round 2, HIGH: listings A attested by a keyed verify; a file corrupts; the canonical's listings move to B and the attest of B hashes
    the corrupt file (`attest_verify_failed`, recorded, the mirror red); the listings return to A. Before this the attest took
    `already_attested` from A's OLDER keyed report, wrote `generation.state = published` and `stores status` read green with the file
    still corrupt (the next tick red again). Now the attest reads the standing failure before either cached return and re-verifies in
    full: the corrupt file is named again (red stays red); restored bytes pass, clear the failure and the pass is remembered, so B's
    stale failing report is superseded when the listings return to B (listings_changed: run attest -- never attest_integrity_failed)."""
    w = world
    root, cfg, comp, dest, gd = w["root"], w["cfg"], w["comp"], w["dest"], w["gen_dir"]
    sha8 = _publish(w)
    A = _pre_rule_generation(w, sha8)
    assert ST.mirror_component(root, dest, comp, runner=Runner()).outcome == "listings_unattested"
    rn = Runner()
    rA = ST.attest_listings(root, dest, comp, runner=rn)
    assert rA.outcome == "attested" and rA.listings_sha256 == A and rn.count("sbatch") == 1, rA.as_dict()
    keyed_a = f"{gd}/VERIFIED-{sha8}.{A[:8]}.json"
    t_a = json.load(open(keyed_a))["verified_utc"]
    assert ST.mirror_component(root, dest, comp, runner=Runner()).outcome == "unchanged"
    # the file corrupts; the canonical's listings move to B (an excluded entry lands under `a`: the tree still satisfies them); the live
    # report predates both, so the tick is listings_changed (nothing re-hashed) and the attest of B hashes the corrupt file
    good = open(f"{gd}/{sha8}/top.txt", "rb").read()
    _corrupt_in_place(f"{gd}/{sha8}/top.txt", b"top LEVEL\n")
    os.makedirs(os.path.join(w["base"], "a", "__pycache__"))
    assert ST.mirror_component(root, dest, comp, runner=Runner()).outcome == "listings_changed"
    rn = Runner()
    rB = ST.attest_listings(root, dest, comp, runner=rn)
    assert rB.outcome == "attest_verify_failed" and rB.detail["corrupt"] == 1 and rB.listings_sha256 != A and rn.count("sbatch") == 1, rB.as_dict()
    B = rB.listings_sha256
    keyed_b = f"{gd}/VERIFIED-{sha8}.{B[:8]}.json"
    fail = ST.load_state(root)["components"][SLUG]["attest_integrity_failed"]
    assert fail["report_path"] == keyed_b and fail["verified_utc"] > t_a, "the failure is stamped after A's attestation"
    assert ST.mirror_component(root, dest, comp, runner=Runner()).outcome == "attest_integrity_failed" and ST.status(root, cfg)["components"][SLUG]["level"] == "red"
    # the listings return to A: the tick names the standing failure (A's report is older than it); the attest never reuses A's report
    os.rmdir(os.path.join(w["base"], "a", "__pycache__"))
    rn = Runner()
    res = ST.mirror_component(root, dest, comp, runner=rn)
    assert res.outcome == "attest_integrity_failed" and rn.count("sbatch") == 0 and res.detail["attest_failure"]["report_path"] == keyed_b, res.as_dict()
    rn = Runner()
    r = ST.attest_listings(root, dest, comp, runner=rn)
    assert r.outcome == "attest_verify_failed" and r.level == "red" and not r.ok and r.listings_sha256 == A, r.as_dict()
    assert rn.count("sbatch") == 1, "re-verified in full: never already_attested from the older report"
    assert r.detail["standing_failure"]["report_path"] == keyed_b and r.detail["corrupt"] == 1 and r.detail["first"]["corrupt"] == ["top.txt"]
    assert json.load(open(keyed_a))["ok"] is False and json.load(open(keyed_a))["corrupt"] == ["top.txt"], "A's keyed report is the fresh failing verify"
    st = ST.load_state(root)["components"][SLUG]
    assert st["attest_integrity_failed"]["report_path"] == keyed_a and st["generation"]["state"] == "attest_integrity_failed"
    assert st["last_run"]["outcome"] == "attest:attest_verify_failed" and st["published_attestation"] == A, "the attestation recorded before the corruption stands; the state is not green"
    assert ST.status(root, cfg)["components"][SLUG]["level"] == "red"
    rn = Runner()
    assert ST.mirror_component(root, dest, comp, runner=rn).outcome == "attest_integrity_failed" and rn.count("sbatch") == 0
    assert ST.status(root, cfg)["components"][SLUG]["level"] == "red"
    monkeypatch.setattr(ST, "_default_runner", Runner())
    monkeypatch.setenv("REGISTRY_ROOT", root)
    monkeypatch.setenv(ST.CONFIG_ENV, w["cfgp"])
    assert cli.main(["stores", "attest", "--dest", "cluster_a", "--component", SLUG, "--json"]) == 1
    lines = [json.loads(l) for l in capsys.readouterr().out.splitlines()]
    assert lines[-2]["outcome"] == "attest_verify_failed" and lines[-1]["levels"]["red"] == 1 and ST.status(root, cfg)["components"][SLUG]["level"] == "red"
    # the bytes restored: the attest re-verifies in full, passes, clears the failure and remembers the pass; the tick is unchanged, green
    _corrupt_in_place(f"{gd}/{sha8}/top.txt", good)
    rn = Runner()
    r2 = ST.attest_listings(root, dest, comp, runner=rn)
    assert r2.outcome == "attested" and r2.published_attestation == A and rn.count("sbatch") == 1, r2.as_dict()      # A's own report failed: the integrity_failed verdict re-verifies
    st = ST.load_state(root)["components"][SLUG]
    assert st["attest_integrity_failed"] is None and st["generation"]["state"] == "published" and st["published_attestation"] == A
    assert st["last_full_pass"] == {"content_sha8": sha8, "verified_utc": json.load(open(keyed_a))["verified_utc"], "report_path": keyed_a}
    assert st["last_full_pass"]["verified_utc"] > fail["verified_utc"]
    rn = Runner()
    assert ST.mirror_component(root, dest, comp, runner=rn).outcome == "unchanged" and rn.count("sbatch") == 0
    assert ST.status(root, cfg)["components"][SLUG]["level"] == "green"
    # the listings return to B: B's failing report still beside the generation is superseded by the remembered pass -- the tick says
    # listings_changed (run attest), never attest_integrity_failed; the attest re-verifies B in full and attests it
    os.makedirs(os.path.join(w["base"], "a", "__pycache__"))
    rn = Runner()
    res = ST.mirror_component(root, dest, comp, runner=rn)
    assert res.outcome == "listings_changed" and rn.count("sbatch") == 0 and ST.load_state(root)["components"][SLUG]["attest_integrity_failed"] is None, res.as_dict()
    rn = Runner()
    r3 = ST.attest_listings(root, dest, comp, runner=rn)
    assert r3.outcome == "attested" and r3.published_attestation == B and rn.count("sbatch") == 1 and json.load(open(keyed_b))["ok"] is True, r3.as_dict()
    assert ST.mirror_component(root, dest, comp, runner=Runner()).outcome == "unchanged"


def test_an_older_drift_report_for_the_returned_listings_is_never_reused_over_a_later_integrity_failure(world):
    """Round 2, HIGH (the cached drift return): listings C refused as a drift by a keyed verify (amber); a file corrupts; the listings move
    to D and the attest of D hashes the corrupt file (recorded, red); the listings return to C. Before this the attest returned
    `listing_only_drift` from C's OLDER report and demoted the standing red to amber. Now it re-verifies in full: red while the file is
    corrupt; restored bytes pass on the files, the drift stands amber, the failure clears and the pass is remembered."""
    w = world
    root, cfg, comp, dest, gd = w["root"], w["cfg"], w["comp"], w["dest"], w["gen_dir"]
    sha8 = _publish(w)
    _pre_rule_generation(w, sha8)
    os.makedirs(os.path.join(w["base"], "a", "later_empty"))                  # C: an empty directory the sealed generation lacks
    rn = Runner()
    rC = ST.attest_listings(root, dest, comp, runner=rn)
    assert rC.outcome == "listing_only_drift" and rC.detail["dirs_canonical_only"] == ["a/later_empty"] and rn.count("sbatch") == 1, rC.as_dict()
    C = rC.listings_sha256
    keyed_c = f"{gd}/VERIFIED-{sha8}.{C[:8]}.json"
    assert ST.mirror_component(root, dest, comp, runner=Runner()).outcome == "listing_only_drift"
    good = open(f"{gd}/{sha8}/top.txt", "rb").read()
    _corrupt_in_place(f"{gd}/{sha8}/top.txt", b"top LEVEL\n")
    os.makedirs(os.path.join(w["base"], "x", "another_empty"))                # D: a second one
    rn = Runner()
    rD = ST.attest_listings(root, dest, comp, runner=rn)
    assert rD.outcome == "attest_verify_failed" and rD.detail["corrupt"] == 1 and rD.listings_sha256 != C, rD.as_dict()
    keyed_d = f"{gd}/VERIFIED-{sha8}.{rD.listings_sha256[:8]}.json"
    assert ST.load_state(root)["components"][SLUG]["attest_integrity_failed"]["report_path"] == keyed_d
    assert ST.mirror_component(root, dest, comp, runner=Runner()).outcome == "attest_integrity_failed"
    # back to C: the tick names the standing failure through the drift verdict; the attest never returns C's older drift report
    os.rmdir(os.path.join(w["base"], "x", "another_empty"))
    rn = Runner()
    assert ST.mirror_component(root, dest, comp, runner=rn).outcome == "attest_integrity_failed" and rn.count("sbatch") == 0
    rn = Runner()
    r = ST.attest_listings(root, dest, comp, runner=rn)
    assert r.outcome == "attest_verify_failed" and r.level == "red" and r.listings_sha256 == C and rn.count("sbatch") == 1, r.as_dict()
    assert r.detail["standing_failure"]["report_path"] == keyed_d and r.detail["corrupt"] == 1
    st = ST.load_state(root)["components"][SLUG]
    assert st["attest_integrity_failed"]["report_path"] == keyed_c and st["generation"]["state"] == "attest_integrity_failed" and st["published_attestation"] is None
    assert ST.status(root, cfg)["components"][SLUG]["level"] == "red" and json.load(open(keyed_c))["corrupt"] == ["top.txt"]
    rn = Runner()
    assert ST.mirror_component(root, dest, comp, runner=rn).outcome == "attest_integrity_failed" and rn.count("sbatch") == 0
    # restored: re-verified in full, the files pass, the drift stands (amber), the failure is cleared and the pass remembered
    _corrupt_in_place(f"{gd}/{sha8}/top.txt", good)
    rn = Runner()
    r2 = ST.attest_listings(root, dest, comp, runner=rn)
    assert r2.outcome == "listing_only_drift" and r2.ok and r2.detail["dirs_canonical_only"] == ["a/later_empty"] and rn.count("sbatch") == 1, r2.as_dict()
    st = ST.load_state(root)["components"][SLUG]
    assert st["attest_integrity_failed"] is None and st["last_full_pass"]["content_sha8"] == sha8 and st["last_full_pass"]["report_path"] == keyed_c
    assert st["generation"]["state"] == "listing_only_drift" and ST.status(root, cfg)["components"][SLUG]["level"] == "amber"
    rep_c = json.load(open(keyed_c))
    assert rep_c["ok"] is False and rep_c["corrupt"] == [] and rep_c["listing_mismatch"] == ["a", "a/later_empty"]
    rn = Runner()
    res = ST.mirror_component(root, dest, comp, runner=rn)
    assert res.outcome == "listing_only_drift" and res.level == "amber" and rn.count("sbatch") == 0 and ST.load_state(root)["components"][SLUG]["attest_integrity_failed"] is None, res.as_dict()


@pytest.mark.parametrize("slug", ["dev", "frozen"])
def test_a_producers_unpushed_edit_never_demotes_the_reversed_mirrors_standing_integrity_failure(tmp_path, slug, monkeypatch, capsys):
    """Round 2, HIGH: a `cluster_a` component's attest hashes a corrupt generation file (recorded; the tick red). The producer then edits
    its grading host copy without pushing: before this the tick took the unequal branch and returned `unpushed_changes` (amber, exit 0) -- or
    `frozen_differs` for a frozen producer -- with `current` still naming the corrupt generation and the record still in the state. Now
    the standing failure is read against the published generation's own manifest facts and named first, red, on every tick, in the dry
    run and in the CLI, with the unequal outcome it outranks in the detail; restored bytes alone clear nothing; the producer's push (a
    fresh, fully verified generation that supersedes `current`) or, for the frozen store, the copy reverted and a `--full` pass, clears it."""
    w = _cluster_a_world(tmp_path)
    root, cfg, dest = w["root"], w["cfg"], w["dest"]
    comp = cfg.components[slug]
    gd = dest.gen_dir(slug)
    push_kw = {"session": "sess"} if slug == "dev" else {}
    pushed = ST.push_component(root, dest, comp, runner=Runner(), **push_kw)
    assert pushed.outcome == "published", pushed.as_dict()
    sha8 = pushed.content_sha8
    _pre_rule_generation({**w, "gen_dir": gd}, sha8, slug=slug)
    assert ST.run_mirror(root, dest, comp, runner=Runner()).outcome == "verified"
    good = open(f"{gd}/{sha8}/top.txt", "rb").read()
    _corrupt_in_place(f"{gd}/{sha8}/top.txt", b"top LEVEL\n")
    a = ST.attest_listings(root, dest, comp, runner=Runner())
    assert a.outcome == "attest_verify_failed" and a.detail["corrupt"] == 1, a.as_dict()
    keyed = f"{gd}/VERIFIED-{sha8}.{a.listings_sha256[:8]}.json"
    assert ST.run_mirror(root, dest, comp, runner=Runner()).outcome == "attest_integrity_failed"
    # the producer edits its grading host copy and does not push: the two manifests differ, and the failing `current` is still what the shards read
    grading_host_top = os.path.join(w["base"], "top.txt")
    _write(grading_host_top, b"edited on the grading host, not pushed\n")
    outranked = "frozen_differs" if slug == "frozen" else "unpushed_changes"
    for _ in range(2):
        rn = Runner()
        r = ST.run_mirror(root, dest, comp, runner=rn)
        assert r.outcome == "attest_integrity_failed" and r.level == "red" and not r.ok and r.published == sha8 and _no_submission(rn), r.as_dict()
        assert r.content_sha8 != sha8 and r.detail["differs_outcome"] == outranked and r.detail["differing"]["counts"]["sha_or_size_differs"] == 1
        assert r.detail["attest_failure"]["report_path"] == keyed and "1 corrupt" in r.detail["why"] and "'top.txt'" in r.detail["why"]
        assert (("frozen_differs" in r.detail["why"] and "unpushed_files" not in r.detail) if slug == "frozen"
                else (f"`stores push {slug}`" in r.detail["why"] and r.detail["unpushed_files"] == 1))
        st = ST.load_state(root)["components"][slug]
        assert st["attest_integrity_failed"]["report_path"] == keyed and st["generation"]["state"] == "attest_integrity_failed" and st["published"] == sha8
        assert st["generation"]["attest_failure"]["report_path"] == keyed and st["generation"]["differing"]["counts"]["sha_or_size_differs"] == 1
        assert st["last_run"]["outcome"] == "attest_integrity_failed" and os.readlink(f"{gd}/current") == sha8
    assert ST.status(root, cfg)["components"][slug]["level"] == "red"
    r = ST.run_mirror(root, dest, comp, runner=Runner(), dry_run=True)
    assert r.outcome == "dry_run" and r.detail["would"] == "attest_integrity_failed" and r.detail["writes"] == "nothing" and r.detail["differing"]["counts"]["sha_or_size_differs"] == 1, r.as_dict()
    monkeypatch.setattr(ST, "_default_runner", Runner())
    monkeypatch.setenv("REGISTRY_ROOT", root)
    monkeypatch.setenv(ST.CONFIG_ENV, w["cfgp"])
    assert cli.main(["stores", "mirror", "--component", slug]) == 1
    out = [json.loads(l) for l in capsys.readouterr().out.splitlines()]
    assert out[-1]["levels"]["red"] == 1 and out[-2]["outcome"] == "attest_integrity_failed" and out[-2]["differs_outcome"] == outranked
    # bytes restored by hand: nothing re-hashed (the two sides still differ, and `--full` re-hashes only when they agree) -- still red
    _corrupt_in_place(f"{gd}/{sha8}/top.txt", good)
    rn = Runner()
    assert ST.run_mirror(root, dest, comp, runner=rn).outcome == "attest_integrity_failed" and _no_submission(rn)
    rn = Runner()
    assert ST.run_mirror(root, dest, comp, runner=rn, full=True).outcome == "attest_integrity_failed" and _no_submission(rn)
    if slug == "dev":
        # the producer's push publishes a fresh, fully verified generation that supersedes the failing `current`: cleared, verified, green
        rp = ST.push_component(root, dest, comp, runner=Runner(), session="sess")
        assert rp.outcome == "published" and rp.content_sha8 != sha8 and os.readlink(f"{gd}/current") == rp.content_sha8, rp.as_dict()
        st = ST.load_state(root)["components"][slug]
        assert st["attest_integrity_failed"] is None and st["published"] == rp.content_sha8
        rn = Runner()
        assert ST.run_mirror(root, dest, comp, runner=rn).outcome == "verified" and _no_submission(rn)
        assert ST.run_mirror(root, dest, comp, runner=Runner(), dry_run=True).detail["would"] == "verified"
    else:
        # the frozen copy reverted to the published content: equal again, the failure still stands (nothing re-hashed) until `--full` passes
        _write(grading_host_top, good)
        rn = Runner()
        assert ST.run_mirror(root, dest, comp, runner=rn).outcome == "attest_integrity_failed" and _no_submission(rn)
        rn = Runner()
        rf = ST.run_mirror(root, dest, comp, runner=rn, full=True)
        assert rf.outcome == "verified" and rf.detail["full"]["ok"] is True and rn.count("sbatch") == 1, rf.as_dict()
        st = ST.load_state(root)["components"][slug]
        assert st["attest_integrity_failed"] is None and st["last_full_pass"]["content_sha8"] == sha8
        rn = Runner()
        assert ST.run_mirror(root, dest, comp, runner=rn).outcome == "verified" and _no_submission(rn)
    assert ST.status(root, cfg)["components"][slug]["level"] == "green"
    # the edit again, with no failure standing: the ordinary unequal outcome, as before
    _write(grading_host_top, b"edited again\n")
    r = ST.run_mirror(root, dest, comp, runner=Runner())
    assert r.outcome == outranked and "differs_outcome" not in r.detail and ST.load_state(root)["components"][slug]["attest_integrity_failed"] is None, r.as_dict()


def test_a_deferred_carry_over_a_failing_current_is_red_not_deferred_and_an_undeferred_carry_supersedes_it(tmp_path):
    """Round 2, HIGH (the transitional carry): with `min_publish_interval_s` the carry of an unpushed edit is `deferred` (green) -- which
    would stand over a `current` whose attest hashed a corrupt file. Now a deferred carry over a standing failure is
    `attest_integrity_failed` (red, nothing copied, the deferral in the detail); once the interval has elapsed the carry runs and the
    fresh, fully verified generation it publishes supersedes the failing `current` (cleared, verified)."""
    w = _cluster_a_world(tmp_path, paced={"base": _canon(tmp_path / "p"), "authority": "cluster_a", "producer": "grading_host:pacer", "transitional_carry": True,
                                           "carry_deadline": "2099-01-01", "min_publish_interval_s": 3600, "exclude": EXCL})
    root, cfg, dest = w["root"], w["cfg"], w["dest"]
    comp = cfg.components["paced"]
    gd = dest.gen_dir("paced")
    r0 = ST.run_mirror(root, dest, comp, runner=Runner())
    assert r0.outcome == "published" and json.load(open(f"{gd}/PUBLISHED-{r0.content_sha8}.json"))["mode"] == "carry", r0.as_dict()
    sha8 = r0.content_sha8
    _pre_rule_generation({**w, "gen_dir": gd}, sha8, slug="paced")
    assert ST.run_mirror(root, dest, comp, runner=Runner()).outcome == "verified"
    _corrupt_in_place(f"{gd}/{sha8}/top.txt", b"top LEVEL\n")
    a = ST.attest_listings(root, dest, comp, runner=Runner())
    assert a.outcome == "attest_verify_failed", a.as_dict()
    keyed = f"{gd}/VERIFIED-{sha8}.{a.listings_sha256[:8]}.json"
    assert ST.run_mirror(root, dest, comp, runner=Runner()).outcome == "attest_integrity_failed"
    # the producer's edit, seconds after the publication: the carry would be deferred -- red, not green, nothing copied
    _write(os.path.join(comp.base, "top.txt"), b"paced change\n")
    for _ in range(2):
        rn = Runner()
        r = ST.run_mirror(root, dest, comp, runner=rn)
        assert r.outcome == "attest_integrity_failed" and r.level == "red" and r.published == sha8 and _no_submission(rn), r.as_dict()
        assert r.detail["differs_outcome"] == "deferred" and r.detail["deferral"]["min_publish_interval_s"] == 3600 and "deferred" in r.detail["why"]
        assert r.detail["attest_failure"]["report_path"] == keyed and r.detail["unpushed_files"] == 1 and os.readlink(f"{gd}/current") == sha8
        st = ST.load_state(root)["components"]["paced"]
        assert st["attest_integrity_failed"]["report_path"] == keyed and st["generation"]["state"] == "attest_integrity_failed" and st["published"] == sha8
    assert ST.run_mirror(root, dest, comp, runner=Runner(), dry_run=True).detail["would"] == "attest_integrity_failed"
    assert ST.status(root, cfg)["components"]["paced"]["level"] == "red"
    # the interval elapsed (the record's stamp aged): the carry runs -- a dry run says so -- and its publication supersedes the failing current
    pub_path = f"{gd}/PUBLISHED-{sha8}.json"
    pub = json.load(open(pub_path))
    os.chmod(pub_path, 0o644)
    ST._write_json(pub_path, {**pub, "published_utc": "2020-01-01T00:00:00.000Z"})
    assert ST.run_mirror(root, dest, comp, runner=Runner(), dry_run=True).detail["would"] == "carry"
    rn = Runner()
    r = ST.run_mirror(root, dest, comp, runner=rn)
    assert r.outcome == "published" and r.content_sha8 != sha8 and json.load(open(f"{gd}/PUBLISHED-{r.content_sha8}.json"))["mode"] == "carry", r.as_dict()
    assert os.readlink(f"{gd}/current") == r.content_sha8 and rn.count("sbatch") == 1
    st = ST.load_state(root)["components"]["paced"]
    assert st["attest_integrity_failed"] is None and st["published"] == r.content_sha8 and st["generation"]["state"] == "published"
    rn = Runner()
    assert ST.run_mirror(root, dest, comp, runner=rn).outcome == "verified" and _no_submission(rn)
    assert ST.status(root, cfg)["components"]["paced"]["level"] == "green"


def _stale_names(aside):
    return sorted(os.path.basename(p) for p in aside)


def test_a_keyed_report_of_a_replaced_tree_never_attests_the_fresh_one(world):
    """Round 3, HIGH: generation A (pre-rule) is re-attested for listings L1 (a keyed VERIFIED-A.<L1>.json beside it); the content moves to
    B and back to A under listings L2 (the directory removed) -- the mirror renames the old A tree aside and verifies a FRESH A tree built
    from the canonical as it stands, without the directory. Before this the keyed L1 report stayed beside the generation, and when the
    canonical's listings returned to L1 the mirror took it as attesting the fresh tree: `unchanged`, green, L1 recorded as the
    attestation, while admit_generation with that pin refused `listing_changed`. Now every report and snapshot of the replaced tree is
    moved aside with it (`stale_reports_renamed_to`), the keyed attestations stand (they name a (content, listings) pair, not a tree),
    the return to L1 is `listings_changed` (red: run attest) with the recorded attestation still L2, and the attest of L1 runs a fresh
    verify that names the drift."""
    w = world
    root, cfg, comp, dest, gd = w["root"], w["cfg"], w["comp"], w["dest"], w["gen_dir"]
    killed = os.path.join(w["base"], "x", "killed_empty")
    os.makedirs(killed)
    A = _publish(w)
    L1 = _pre_rule_generation(w, A)
    rn = Runner()
    rn.job = 600
    r = ST.attest_listings(root, dest, comp, runner=rn)
    assert r.outcome == "attested" and r.published_attestation == L1 and rn.count("sbatch") == 1, r.as_dict()
    keyed_rep, keyed_snap = ST.keyed_report_path(gd, A, L1), f"{gd}/DEST-STAT-{A}.{L1[:8]}.json"
    assert os.path.isfile(keyed_rep) and os.path.isfile(keyed_snap) and json.load(open(keyed_rep))["job_id"] == "601"
    assert ST.mirror_component(root, dest, comp, runner=Runner()).outcome == "unchanged"
    # B: a byte change, published
    _write(os.path.join(w["base"], "top.txt"), b"top level, revised\n")
    resB = ST.mirror_component(root, dest, comp, runner=Runner())
    assert resB.outcome == "published" and resB.content_sha8 != A
    # back to A's bytes with the directory removed: the same content under other listings; the old A tree is renamed aside and the fresh one
    # is built from the canonical as it stands now -- WITHOUT x/killed_empty -- and verified against L2
    _write(os.path.join(w["base"], "top.txt"), b"top level\n")
    os.rmdir(killed)
    rn = Runner()
    rn.job = 800
    resA2 = ST.mirror_component(root, dest, comp, runner=rn)
    assert resA2.outcome == "published" and resA2.content_sha8 == A, resA2.as_dict()
    L2 = json.load(open(f"{gd}/VERIFIED-{A}.json"))["listings_sha256"]
    assert L2 != L1 and json.load(open(f"{gd}/VERIFIED-{A}.json"))["job_id"] == "801" and not os.path.exists(f"{gd}/{A}/x/killed_empty")
    superseded = resA2.detail["superseded"]
    assert superseded.startswith(f"{gd}/.superseded-{A}-") and os.path.isdir(f"{superseded}/x/killed_empty")
    st = ST.load_state(root)["components"][SLUG]
    assert st["published_attestation"] == L2 and ST.published_attestations(root) == {SLUG: L2} and st["generation"]["state"] == "published"
    assert ST.admit_generation(gd, A, listings_sha256=L2).ok
    # the canonical's listings return to L1 -- the directory is back, no file changed. The fresh tree was never verified against L1 (it lacks
    # the directory): the mirror is listings_changed (red: run attest), nothing re-hashed, the recorded attestation stays L2, nothing green
    # (before the fix: `unchanged`, green, L1 recorded -- from the superseded tree's keyed report)
    os.makedirs(killed)
    rn = Runner()
    res = ST.mirror_component(root, dest, comp, runner=rn)
    assert res.outcome == "listings_changed" and res.level == "red" and not res.ok and rn.count("sbatch") == 0, res.as_dict()
    assert res.detail["published_listings_sha256"] == L2 and res.detail["listings_sha256"] == L1 and res.published == A
    st = ST.load_state(root)["components"][SLUG]
    assert st["published_attestation"] == L2 and st["generation"]["state"] == "listings_changed" and st["last_run"]["outcome"] == "listings_changed"
    assert ST.status(root, cfg)["components"][SLUG]["level"] == "red" and ST.published_attestations(root) == {SLUG: L2}
    # how: every report and snapshot that spoke for the superseded tree was moved aside with it at the rebuild -- the live pair, the keyed
    # pair and the tree's directory seal (the closing pass on 2c) -- kept, never deleted; the keyed attestations of both pairs stand (they
    # name no tree)
    aside = resA2.detail["stale_reports_renamed_to"]
    names = _stale_names(aside)
    assert len(aside) == 5 and all(os.path.isfile(p) for p in aside), aside
    for prefix in (f".stale-VERIFIED-{A}-", f".stale-VERIFIED-{A}.{L1[:8]}-", f".stale-DEST-STAT-{A}-", f".stale-DEST-STAT-{A}.{L1[:8]}-", f".stale-SEALED-{A}-"):
        assert sum(1 for n in names if n.startswith(prefix)) == 1, (prefix, names)
    assert resA2.detail["stale_report_renamed_to"] in aside and os.path.basename(resA2.detail["stale_report_renamed_to"]).startswith(f".stale-VERIFIED-{A}-")
    assert not os.path.exists(keyed_snap), "the keyed snapshot of the superseded tree no longer stands at its name"
    stale_keyed = next(p for p in aside if os.path.basename(p).startswith(f".stale-VERIFIED-{A}.{L1[:8]}-"))
    assert json.load(open(stale_keyed))["job_id"] == "601" and json.load(open(stale_keyed))["ok"] is True, "the old tree's report, aside"
    assert os.path.isfile(f"{gd}/MANIFEST-{A}.{L1[:8]}.json") and os.path.isfile(f"{gd}/MANIFEST-{A}.{L2[:8]}.json"), "both attestations stand: they name no tree"
    # what a planned cluster read does with each pin: the recorded attestation admits the fresh tree; L1 -- what the old keyed report had
    # spoken for -- is refused by name, which is exactly why the mirror must not call it attested
    assert ST.admit_generation(gd, A, listings_sha256=L2).ok
    ad = ST.admit_generation(gd, A, listings_sha256=L1)
    assert ad.refusal == "listing_changed" and ad.listing_mismatched == ["x", "x/killed_empty"], ad.as_dict()
    # ... while revalidate(A, L1) ADMITS: it reads the canonical copy, which IS at L1 again -- so a plan pinning (A, L1) would pass
    # revalidation on the grading host and fail on the cluster, which is the gap a mirror calling L1 attested would have papered over
    assert ST.revalidate(SLUG, A, root=root, cfg=cfg, listings_sha256=L1).ok and ST.revalidate(SLUG, A, root=root, cfg=cfg, listings_sha256=L1).listings_sha256 == L1
    # a second tick reads the same: no older report is found at the keyed name
    assert not os.path.exists(keyed_rep), "the keyed report of the superseded tree no longer speaks at its name"
    rn = Runner()
    assert ST.mirror_component(root, dest, comp, runner=rn).outcome == "listings_changed" and rn.count("sbatch") == 0
    # the attest of L1 over the fresh tree runs a FRESH verify (never already_attested from the old tree's report): the tree lacks the
    # directory the canonical has -> the drift, by name, from a report THIS verify wrote
    rn = Runner()
    rn.job = 900
    r2 = ST.attest_listings(root, dest, comp, runner=rn)
    assert r2.outcome == "listing_only_drift" and rn.count("sbatch") == 1 and r2.detail["dirs_canonical_only"] == ["x/killed_empty"], r2.as_dict()
    rep = json.load(open(keyed_rep))
    assert rep["job_id"] == "901" and rep["ok"] is False and rep["listing_mismatch"] == ["x", "x/killed_empty"] and os.path.isfile(keyed_snap)
    assert ST.load_state(root)["components"][SLUG]["published_attestation"] == L2
    rn = Runner()
    assert ST.mirror_component(root, dest, comp, runner=rn).outcome == "listing_only_drift" and rn.count("sbatch") == 0
    # the directory gone again: the canonical is the fresh tree's attested listings once more
    os.rmdir(killed)
    assert ST.mirror_component(root, dest, comp, runner=Runner()).outcome == "unchanged"
    assert ST.load_state(root)["components"][SLUG]["published_attestation"] == L2


def test_the_dangling_repair_moves_the_keyed_reports_of_the_renamed_away_tree_aside_too(world):
    """The same rule on the dangling-repair path: `current -> A` while A's tree was renamed away, a keyed re-attestation of the old tree
    beside it and -- out of band -- the live report gone, so only the keyed report and the snapshots speak for a tree that is not there.
    The repair copies afresh from a canonical whose directories moved meanwhile and moves EVERY remaining report and snapshot aside before
    it (there is no live report to name, so `stale_report_renamed_to` is absent and `stale_reports_renamed_to` carries the three and the
    seal); the canonical's return to the old listings is `listings_changed`, never `unchanged` from the old tree's report."""
    w = world
    root, cfg, comp, dest, gd = w["root"], w["cfg"], w["comp"], w["dest"], w["gen_dir"]
    killed = os.path.join(w["base"], "x", "killed_empty")
    os.makedirs(killed)
    A = _publish(w)
    L1 = _pre_rule_generation(w, A)
    rn = Runner()
    rn.job = 600
    assert ST.attest_listings(root, dest, comp, runner=rn).outcome == "attested"
    keyed_rep = ST.keyed_report_path(gd, A, L1)
    assert os.path.isfile(keyed_rep)
    os.rename(f"{gd}/{A}", f"{gd}/.failed-{A}-20261007T120000000Z")
    os.remove(f"{gd}/VERIFIED-{A}.json")
    os.rmdir(killed)                                          # the canonical no longer has the directory: the repair builds the tree at L2
    rn = Runner()
    rn.job = 800
    r = ST.mirror_component(root, dest, comp, runner=rn)
    assert r.outcome == "republished_dangling_current" and r.content_sha8 == A and r.detail["renamed_aside"] == [f".failed-{A}-20261007T120000000Z"], r.as_dict()
    assert json.load(open(f"{gd}/VERIFIED-{A}.json"))["job_id"] == "801"
    L2 = json.load(open(f"{gd}/VERIFIED-{A}.json"))["listings_sha256"]
    assert L2 != L1 and ST.load_state(root)["components"][SLUG]["published_attestation"] == L2 and ST.admit_generation(gd, A, listings_sha256=L2).ok
    os.makedirs(killed)
    rn = Runner()
    res = ST.mirror_component(root, dest, comp, runner=rn)
    assert res.outcome == "listings_changed" and rn.count("sbatch") == 0 and res.detail["published_listings_sha256"] == L2, res.as_dict()
    assert ST.admit_generation(gd, A, listings_sha256=L1).refusal == "listing_changed"
    assert ST.status(root, cfg)["components"][SLUG]["level"] == "red"
    # how: the repair moved the three remaining reports and snapshots, and the renamed tree's seal, aside before its copy; there was no live
    # report to name
    aside = r.detail["stale_reports_renamed_to"]
    names = _stale_names(aside)
    assert len(aside) == 4 and "stale_report_renamed_to" not in r.detail, (aside, r.detail.get("stale_report_renamed_to"))
    for prefix in (f".stale-VERIFIED-{A}.{L1[:8]}-", f".stale-DEST-STAT-{A}-", f".stale-DEST-STAT-{A}.{L1[:8]}-", f".stale-SEALED-{A}-"):
        assert sum(1 for n in names if n.startswith(prefix)) == 1, (prefix, names)
    assert not os.path.exists(keyed_rep)


def test_the_attests_authority_refusals_record_what_the_mirror_records_so_an_unavailable_generation_is_never_shown_green(world, monkeypatch, capsys):
    """After a successful publication the attest found the generation's tree renamed away (`current_dangling`) or its live
    VERIFIED report gone (`current_unverified`) and returned the red -- but wrote only `last_run`, so `generation.state = published` and
    `published = <sha8>` stood and `stores status` showed the component green and verified while the generation could not be read. Now
    each authority refusal records what the mirror records when it reads the same facts: the dangle as the dangling-pointer
    record (`published` dropped, the renamed trees listed, the PROBLEM line, nothing pinned; a repeat keeps the first finding's
    stamp; the mirror's next tick repairs it), the missing report as `current_unverified` (`published` kept: the tree stands; the
    state red), nothing published as `no_current` (`published` dropped). The recorded attestation is left alone throughout;
    `content_differs` touches nothing (the published generation stands)."""
    w = world
    root, cfg, comp, dest, gd = w["root"], w["cfg"], w["comp"], w["dest"], w["gen_dir"]
    sha8 = _publish(w)
    lsha = _pre_rule_generation(w, sha8)
    rn = Runner()
    rn.job = 600
    assert ST.attest_listings(root, dest, comp, runner=rn).outcome == "attested"
    monkeypatch.setattr(ST, "_default_runner", Runner())
    monkeypatch.setenv("REGISTRY_ROOT", root)
    monkeypatch.setenv(ST.CONFIG_ENV, w["cfgp"])

    def row():
        return ST.status(root, cfg)["components"][SLUG]

    assert row()["level"] == "green" and row()["verified"] is True and row()["published"] == sha8 and ST.published_attestations(root) == {SLUG: lsha}
    # (1) the tree renamed away: the attest refuses current_dangling and records the dangle as the carry does
    os.rename(f"{gd}/{sha8}", f"{gd}/.failed-{sha8}-20261007T120000000Z")
    assert os.readlink(f"{gd}/current") == sha8
    rn = Runner()
    r = ST.attest_listings(root, dest, comp, runner=rn)
    assert r.outcome == "current_dangling" and r.level == "red" and not r.ok and rn.count("sbatch") == 0 and r.job_id is None, r.as_dict()
    st = ST.load_state(root)["components"][SLUG]
    assert st["published"] is None and st["generation"]["state"] == "current_dangling", "before the fix: published = <sha8>, generation.state = published"
    assert row()["level"] == "red" and row()["verified"] is False and row()["published"] is None
    assert r.detail["renamed_aside"] == [f".failed-{sha8}-20261007T120000000Z"] and r.detail["found_utc"] and "is not a directory" in r.detail["why"]
    assert f"beside it: .failed-{sha8}-20261007T120000000Z" in r.detail["why"] and "D882" in r.detail["why"]
    assert st["generation"] == {"content_sha8": sha8, "state": "current_dangling", "current_read": sha8, "found_utc": r.detail["found_utc"]}
    d = st["current_dangling"]
    assert d == {"content_sha8": sha8, "found_utc": r.detail["found_utc"], "renamed_aside": [f".failed-{sha8}-20261007T120000000Z"], "resumed": False,
                 "why": f"current -> {sha8} but {gd}/{sha8} is not a directory; beside it: .failed-{sha8}-20261007T120000000Z"}
    assert st["last_run"]["outcome"] == "attest:current_dangling" and ST.last_run_level(st["last_run"]["outcome"]) == "red"
    assert st["published_attestation"] == lsha, "the attestation recorded for the tree is left alone; nothing is pinned while nothing is published"
    assert row()["level"] == "red" and row()["verified"] is False and row()["published"] is None and row()["current_dangling"] == d
    assert ST.published_attestations(root) == {}, "nothing is advertised to a plan"
    text = ST.format_status(ST.status(root, cfg))
    assert f"PROBLEM {SLUG}: current -> {sha8} but {gd}/{sha8} is not a directory; beside it: .failed-{sha8}-20261007T120000000Z" in text, text
    assert "last run attest:current_dangling" in text and "nothing is published until the next tick republishes it" in text
    # the CLI exits 1 and counts the red; a second attest keeps the first finding's stamp
    assert cli.main(["stores", "attest", "--dest", "cluster_a", "--component", SLUG, "--json"]) == 1
    lines = [json.loads(l) for l in capsys.readouterr().out.splitlines()]
    assert lines[-1]["levels"]["red"] == 1 and lines[-1]["outcomes"] == {"current_dangling": 1} and lines[-2]["found_utc"] == d["found_utc"]
    assert ST.load_state(root)["components"][SLUG]["current_dangling"] == d and row()["level"] == "red"
    # the mirror's next tick repairs it, the dangle is cleared, status green again, the fresh tree verified
    rn = Runner()
    rn.job = 800
    rm = ST.mirror_component(root, dest, comp, runner=rn)
    assert rm.outcome == "republished_dangling_current" and rm.published == sha8, rm.as_dict()
    st = ST.load_state(root)["components"][SLUG]
    assert st["published"] == sha8 and st["current_dangling"] is None and st["generation"]["state"] == "published"
    assert row()["level"] == "green" and row()["verified"] is True and "PROBLEM" not in ST.format_status(ST.status(root, cfg))
    assert ST.admit_generation(gd, sha8).ok and ST.published_attestations(root) == {SLUG: lsha}
    # (2) the live report gone: current_unverified, recorded as the carry records it -- the tree stands (no dangle), `published` names it,
    # the state is red and the recorded attestation is kept
    os.rename(f"{gd}/VERIFIED-{sha8}.json", f"{gd}/VERIFIED-{sha8}.json.aside")
    rn = Runner()
    r2 = ST.attest_listings(root, dest, comp, runner=rn)
    assert r2.outcome == "current_unverified" and r2.level == "red" and rn.count("sbatch") == 0 and r2.detail["verified_present"] is False, r2.as_dict()
    st = ST.load_state(root)["components"][SLUG]
    assert st["published"] == sha8 and st["current_dangling"] is None and st["published_attestation"] == lsha
    assert st["generation"] == {"content_sha8": sha8, "state": "current_unverified", "current_read": sha8, "verified_utc": None}
    assert st["last_run"]["outcome"] == "attest:current_unverified" and ST.last_run_level(st["last_run"]["outcome"]) == "red"
    assert row()["level"] == "red" and row()["verified"] is False and row()["published"] == sha8
    assert cli.main(["stores", "attest", "--dest", "cluster_a", "--component", SLUG, "--json"]) == 1
    assert json.loads(capsys.readouterr().out.splitlines()[-1])["outcomes"] == {"current_unverified": 1} and row()["level"] == "red"
    rn = Runner()
    assert ST.mirror_component(root, dest, comp, runner=rn).outcome == "current_unverified" and rn.count("sbatch") == 0, "the mirror reads the same"
    assert row()["level"] == "red"
    # the evidence restored: the mirror's tick is unchanged and status is green again
    os.rename(f"{gd}/VERIFIED-{sha8}.json.aside", f"{gd}/VERIFIED-{sha8}.json")
    assert ST.mirror_component(root, dest, comp, runner=Runner()).outcome == "unchanged"
    assert row()["level"] == "green" and row()["verified"] is True
    rn = Runner()
    assert ST.attest_listings(root, dest, comp, runner=rn).outcome == "already_attested" and rn.count("sbatch") == 0 and row()["level"] == "green"
    # (3) nothing published at all: no_current, `published` dropped, nothing advertised, the state red
    os.remove(f"{gd}/current")
    rn = Runner()
    r3 = ST.attest_listings(root, dest, comp, runner=rn)
    assert r3.outcome == "no_current" and r3.level == "red" and rn.count("sbatch") == 0, r3.as_dict()
    st = ST.load_state(root)["components"][SLUG]
    assert st["published"] is None and st["current_dangling"] is None
    assert st["generation"] == {"content_sha8": None, "state": "no_current", "current_read": None, "grading_host_content_sha8": sha8}
    assert st["last_run"]["outcome"] == "attest:no_current" and row()["level"] == "red" and row()["verified"] is False and row()["published"] is None
    assert ST.published_attestations(root) == {}
    # the mirror republishes the verified tree through the guard; status green
    rm = ST.mirror_component(root, dest, comp, runner=Runner())
    assert rm.outcome == "published_existing_verified" and rm.published == sha8, rm.as_dict()
    assert row()["level"] == "green" and row()["verified"] is True and row()["published"] == sha8
    # (4) content_differs touches nothing: the published generation stands and stays green
    _write(os.path.join(w["base"], "top.txt"), b"top level, revised\n")
    rn = Runner()
    r4 = ST.attest_listings(root, dest, comp, runner=rn)
    assert r4.outcome == "content_differs" and r4.level == "amber" and rn.count("sbatch") == 0
    st = ST.load_state(root)["components"][SLUG]
    assert st["published"] == sha8 and st["generation"]["state"] == "published" and row()["level"] == "green" and row()["verified"] is True


@pytest.mark.parametrize("slug", ["dev", "frozen"])
def test_the_reversed_mirror_ends_a_dangle_the_attest_recorded_once_the_tree_stands_again(tmp_path, slug):
    """A `cluster_a` component: the attest finds the published tree renamed away and records the dangle (`published` dropped, the
    PROBLEM line); the reversed mirror's own tick names `dangling` (red) and leaves the record standing (the pointer still dangles); the
    tree restored by hand, the reversed mirror reads `verified` and ENDS the recorded dangle -- the tree standing is the event -- so
    `stores status` is green and verified again instead of keeping the PROBLEM line and `verified: false` over a verified generation."""
    w = _cluster_a_world(tmp_path)
    root, cfg, dest = w["root"], w["cfg"], w["dest"]
    comp = cfg.components[slug]
    gd = dest.gen_dir(slug)
    rn = Runner()
    rn.job = 600
    pushed = ST.push_component(root, dest, comp, runner=rn, note="first")
    assert pushed.outcome == "published", pushed.as_dict()
    sha8 = pushed.content_sha8
    assert ST.mirror_authority(root, dest, comp, runner=Runner()).outcome == "verified"

    def row():
        return ST.status(root, cfg)["components"][slug]

    assert row()["level"] == "green" and row()["verified"] is True
    os.rename(f"{gd}/{sha8}", f"{gd}/.failed-{sha8}-20261007T120000000Z")
    rn = Runner()
    r = ST.attest_listings(root, dest, comp, runner=rn)
    assert r.outcome == "current_dangling" and rn.count("sbatch") == 0, r.as_dict()
    st = ST.load_state(root)["components"][slug]
    assert st["published"] is None and st["generation"]["state"] == "current_dangling" and st["current_dangling"]["renamed_aside"] == [f".failed-{sha8}-20261007T120000000Z"]
    assert row()["level"] == "red" and row()["verified"] is False and row()["published"] is None and ST.published_attestations(root).get(slug) is None
    assert f"PROBLEM {slug}: current -> {sha8} but {gd}/{sha8} is not a directory" in ST.format_status(ST.status(root, cfg))
    # the reversed mirror's own reading: dangling (red), the record left standing -- the pointer still dangles
    rn = Runner()
    rd = ST.mirror_authority(root, dest, comp, runner=rn)
    assert rd.outcome == "dangling" and rd.level == "red" and rn.count("sbatch") == 0, rd.as_dict()
    st = ST.load_state(root)["components"][slug]
    assert st["generation"]["state"] == "dangling" and st["current_dangling"]["content_sha8"] == sha8 and row()["level"] == "red" and row()["verified"] is False
    assert f"PROBLEM {slug}:" in ST.format_status(ST.status(root, cfg))
    # the tree restored by hand: verified again, the dangle ended, status green and verified
    os.rename(f"{gd}/.failed-{sha8}-20261007T120000000Z", f"{gd}/{sha8}")
    rn = Runner()
    rv = ST.mirror_authority(root, dest, comp, runner=rn)
    assert rv.outcome == "verified" and rv.published == sha8 and rn.count("sbatch") == 0, rv.as_dict()
    st = ST.load_state(root)["components"][slug]
    assert st["current_dangling"] is None and st["published"] == sha8 and st["generation"]["state"] == "verified"
    assert row()["level"] == "green" and row()["verified"] is True and "PROBLEM" not in ST.format_status(ST.status(root, cfg))
    rn = Runner()
    assert ST.attest_listings(root, dest, comp, runner=rn).outcome == "already_attested" and rn.count("sbatch") == 0


def test_the_attest_keeps_the_record_of_a_pending_repair_so_the_mirror_finishes_it(world):
    """The repair of a dangling pointer is pending until its own seal and record
    stand, and the record that says so is `current_dangling` naming the content. An attest run meanwhile reads a `current` over the
    repair's tree -- the verified replacement without its seal (the previous tree's went aside with its reports), or the partial copy of
    a cut-short rsync without a report -- and refuses by name as before; but its refusal now KEEPS that record and `published` dropped
    instead of clearing the dangle and re-advertising the pointer, so the mirror's next tick finishes (or resumes) the repair rather than
    reading a healthy `current`. Without a pending repair the refusals record what they did -- except that an unsealed `current` drops
    `published` on this route too, since the seal gates publication on every route."""
    w = world
    root, cfg, comp, dest, gd = w["root"], w["cfg"], w["comp"], w["dest"], w["gen_dir"]
    sha8 = _publish(w)
    gen_path = f"{gd}/{sha8}"

    def row():
        return ST.status(root, cfg)["components"][SLUG]

    # (1) the verified replacement, its seal cut short: the attest refuses current_unsealed and keeps the pending repair's record
    _renamed_away(gd, sha8, stamp="20261007T160000000Z")
    rn = _SealBlip()
    rn.job = 800
    assert ST.mirror_component(root, dest, comp, runner=rn).outcome == "error"
    before = ST.load_state(root)["components"][SLUG]
    assert before["published"] is None and before["current_dangling"]["content_sha8"] == sha8 and before["current_dangling"]["phase"] == "publishing"
    rn = Runner()
    r = ST.attest_listings(root, dest, comp, runner=rn)
    assert r.outcome == "current_unsealed" and r.level == "red" and rn.count("sbatch") == 0 and r.detail["authority_problems"] == ["current_unsealed"], r.as_dict()
    st = ST.load_state(root)["components"][SLUG]
    assert st["published"] is None and st["current_dangling"] == before["current_dangling"], "the pending repair's record stands; nothing is re-advertised"
    assert st["generation"]["state"] == "current_unsealed" and st["last_run"]["outcome"] == "attest:current_unsealed"
    assert row()["published"] is None and row()["verified"] is False and ST.published_attestations(root) == {}
    assert f"PROBLEM {SLUG}: current -> {sha8}" in ST.format_status(ST.status(root, cfg))
    assert any(m != 0o555 for m in _dir_modes(gen_path).values()) and not os.path.exists(f"{gd}/SEALED-{sha8}.json"), "the attest adopts nothing"
    # the mirror's next tick finishes the publication; the attest then reads an attested, healthy current
    rn = Runner()
    rm = ST.mirror_component(root, dest, comp, runner=rn)
    assert rm.outcome == "republished_dangling_current" and rm.detail["pending_publication"] is True and rn.count("sbatch") == 0, rm.as_dict()
    st = ST.load_state(root)["components"][SLUG]
    assert st["published"] == sha8 and st["current_dangling"] is None and all(m == 0o555 for m in _dir_modes(gen_path).values())
    rn = Runner()
    assert ST.attest_listings(root, dest, comp, runner=rn).outcome == "already_attested" and rn.count("sbatch") == 0
    assert row()["level"] == "green" and row()["verified"] is True
    # (2) the partial copy of a cut-short rsync (no report): the attest refuses current_unverified and keeps the record; the mirror RESUMES
    _renamed_away(gd, sha8, stamp="20261007T170000000Z")
    r4 = ST.mirror_component(root, dest, comp, runner=_RsyncCut(partial=True))
    assert r4.outcome == "rsync_failed" and os.path.isdir(gen_path) and not os.path.exists(f"{gd}/VERIFIED-{sha8}.json")
    before = ST.load_state(root)["components"][SLUG]
    assert before["published"] is None and before["current_dangling"]["content_sha8"] == sha8
    rn = Runner()
    r5 = ST.attest_listings(root, dest, comp, runner=rn)
    assert r5.outcome == "current_unverified" and rn.count("sbatch") == 0 and r5.detail["verified_present"] is False, r5.as_dict()
    st = ST.load_state(root)["components"][SLUG]
    assert st["published"] is None and st["current_dangling"] == before["current_dangling"] and st["generation"]["state"] == "current_unverified"
    rn = Runner()
    rn.job = 900
    r6 = ST.mirror_component(root, dest, comp, runner=rn)
    assert r6.outcome == "republished_dangling_current" and r6.detail["resumed_repair"] is True and r6.job_id == "901", r6.as_dict()
    assert ST.load_state(root)["components"][SLUG]["published"] == sha8 and ST.load_state(root)["components"][SLUG]["current_dangling"] is None
    # (3) no pending repair: the round-3 shapes stand for what the seal does not gate -- current_unverified keeps `published` and clears
    # nothing it did not find
    os.rename(f"{gd}/VERIFIED-{sha8}.json", f"{gd}/VERIFIED-{sha8}.json.aside")
    r7 = ST.attest_listings(root, dest, comp, runner=Runner())
    st = ST.load_state(root)["components"][SLUG]
    assert r7.outcome == "current_unverified" and st["published"] == sha8 and st["current_dangling"] is None
    os.rename(f"{gd}/VERIFIED-{sha8}.json.aside", f"{gd}/VERIFIED-{sha8}.json")
    # an UNSEALED `current` is dropped on this route too, pending repair or not: the seal gates publication on every route (the final
    # pass on 2c). The mirror's own refusal drops `published`, and an attest running between two ticks must not put it back -- that is
    # how a dangle dropped by one reader used to be re-advertised by the next, and a window pinned what the mirror called red
    os.rename(f"{gd}/SEALED-{sha8}.json", f"{gd}/SEALED-{sha8}.json.aside")
    r8 = ST.attest_listings(root, dest, comp, runner=Runner())
    st = ST.load_state(root)["components"][SLUG]
    assert r8.outcome == "current_unsealed" and st["published"] is None and st["current_dangling"] is None
    assert ST.mirror_component(root, dest, comp, runner=Runner()).outcome == "current_unsealed", "the mirror reads that `current` the same way"
    assert ST.load_state(root)["components"][SLUG]["published"] is None, "neither route re-advertises an unsealed `current`"
    os.rename(f"{gd}/SEALED-{sha8}.json.aside", f"{gd}/SEALED-{sha8}.json")
    assert ST.mirror_component(root, dest, comp, runner=Runner()).outcome == "unchanged"
    assert ST.load_state(root)["components"][SLUG]["published"] == sha8, "the seal back: the generation is advertised again"
