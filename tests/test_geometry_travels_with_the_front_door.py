"""The serving-geometry records travel with the front door — and a site copy is staged ONLY after it
is verified against its authoritative record.

Before: the serving probe wrote its overlay under the cluster it ran on, `geometry.py` read only the CURRENT cluster's
overlay, and `06_eval_waves.sh` could twin a padded unit to the OTHER cluster — where it resolved the provisional / default
budget and steered padding refused before serving. Now `reproduce/03_hpo.sh`'s probe fetch copies `geometry_record*.json` into
`$SISTEER_SITE_DIR/geometry/<model_key>.json` (validated as THIS model's per-set record), and `reproduce/lib/slurm.sh`'s mirror
step (`slurm_mirror_clusters` -> `slurm_stage_geometry`) stages the site's records to EVERY target cluster's
`$SHARED/exp/reproduce/geometry` before any submission.

Re-check 1: a site copy is only a copy — its record may have been reopened (`run --resume --replan`, a re-run) since 03
fetched it. So `slurm_stage_geometry` first resolves the overlay's provenance (probe_cluster, probe_record, result_generation),
runs the verifying status (`hpo_probe_status.py --record`) OVER SSH on the probe cluster, and stages only when it says READY with
the record's generation equal to the overlay's stamp; the staged bytes carry a `staging_verification` block the cluster-side
consumer (`geometry.check_overlay_generation`) requires when it cannot reach the record. Anything else — an unreachable record
included — refuses by name. Two-root: the probe ran on the primary (cluster_a); the padded unit's twin on the secondary
(cluster_b) reads the staged record through `geometry.site_geometry_for` under the job env's SISTEER_SHARED_ROOT.

Synthetic site (neutral labels, placeholder paths, hostname TESTHOST); the shell library's `local:<dir>` seam runs the remote
strings on this host (so the "probe cluster" record IS reachable here — the unreachable case is exercised by rewriting a
staged copy's provenance); nothing reaches a cluster.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from front_door_testlib import assert_no_tokens, write_site
from sisteer.capeval.serving import geometry as g

REPO = Path(__file__).resolve().parents[1]
import test_serving_probe_smoke as _smoke  # noqa: E402  (the serving probe's toy plan — its model key is the record's)
#: the model the probe's toy plan names: the geometry record ready_record() writes through the REAL write_verdict carries THAT key, and a
#: site copy under another name is refused ("names model …, asked for …"), so this test follows the probe's own choice rather than fixing one
KEY = getattr(_smoke, "MODEL", None) or _smoke._plan(sets=("wo",), ranks=(4,))["model_key"]
#: a known model that is NOT the record's, for the refusal of a record naming another model
OTHER = next(k for k in ("qwen35_4b_demo", "qwen35_4b", "gemma4e4b") if k != KEY)
GEN = "f" * 64
PLACEHOLDER = __import__("re").compile(r"\$SHARED\b|\$\{[A-Z_]+[:}]|<[a-z][a-z _-]{2,40}>")   # the dry-run test's rule for DRY-RUN lines


def _budget(S=8, L=98304) -> g.ModelBudget:
    return g.ModelBudget(model_key=KEY, batched_token_budget=S * L, certified=True, source=g.MEASURED_BUDGET_SOURCE,
                         basis=g.BudgetBasis(method=f"{g.SITE_GEOMETRY_METHOD}: test", max_model_len=L, max_num_seqs=S, gpu_memory_utilization=0.9,
                                             kv_tokens_reported=None, probe_job_id="11111111", measured_on="2026-09-30", source="test"))


def _record(**prov) -> dict:
    """A hand-built overlay (the dry-run and refusal tests): provenance names a probe cluster, a record and a generation."""
    provenance = {"probe_job_id": "11111111", "probe_record": "/shared/exp/multilora_cert/toy", "probe_cluster": "cluster_a",
                  g.RESULT_GENERATION_KEY: GEN, "host": "TESTHOST",
                  "attestation_sha256": {"wo_r4_S1cc_cap8_L4096": {"file": None, "sha256": "ab" * 32}}}
    provenance.update(prov)
    provenance = {k: v for k, v in provenance.items() if v is not None}
    return g.site_geometry_record(model_key=KEY, hf_id="org/toy", revision="0" * 40, written_utc="2026-09-30T00:00:00Z",
                                  sets={"wo": {"lora_modules": ["o_proj"], "lora_max_module_dim": 4096, "budget": _budget(),
                                               "allowed_pairs": [[4, "wo"]], "certified_by": ["wo_r4_S1cc_cap8_L4096"],
                                               "module_dim_launches": {"wo_r4_S1cc_cap8_L4096": 4096},
                                               "budget_launches": {"wo_r4_S1cc_cap8_L98304": {"S": 8, "L": 98304, "batched_tokens": 8 * 98304}}}},
                                  provenance=provenance)


def _other_content(doc: dict) -> dict:
    """The same generation, ANOTHER content: the wo set's 98,304 budget halved to S 4 — on the set's budget block, on every budget
    launch the fold keyed at that window (the S1cc and S1d servers), and on the model-wide summary."""
    wo = doc["sets"]["wo"]
    wo["budget"]["batched_token_budget"] = 4 * 98304; wo["budget"]["basis"]["max_num_seqs"] = 4
    for lid, b in wo["budget_launches"].items():
        if b["L"] == 98304:
            b["S"] = 4; b["batched_tokens"] = 4 * 98304
    doc["summary"]["batched_token_budget_max"] = doc["summary"]["batched_token_budget_min"] = 4 * 98304
    return doc


class _World:
    def __init__(self, tmp_path: Path):
        self.tmp = tmp_path
        self.fake = {l: tmp_path / f"fake_{l}" for l in ("cluster_a", "cluster_b")}
        for d in self.fake.values():
            d.mkdir()
        self.shared = {l: tmp_path / f"shared_{l}" for l in self.fake}
        self.site = write_site(tmp_path, clusters=tuple(self.fake), ssh={l: f"local:{d}" for l, d in self.fake.items()},
                               shared={l: str(p) for l, p in self.shared.items()})
        self.env = {"PATH": os.environ["PATH"], "HOME": str(tmp_path), "USER": "nobody-testhost"}

    def bash(self, body: str, *, dry: bool = False) -> subprocess.CompletedProcess:
        script = (f"set -u; source {REPO}/reproduce/lib/common.sh; source {REPO}/reproduce/lib/slurm.sh; REPRO_CONFIG={self.site}/config.env; "
                  f"REPRO_DRY_RUN={1 if dry else 0}; repro_load_config\n{body}\n")
        return subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=self.env, timeout=180)

    def geometry_to_site(self, local_root: Path, *, model: str = KEY, dry: bool = False, recover: bool = False) -> subprocess.CompletedProcess:
        """03_hpo.sh's OWN geometry_to_site (and geometry_site_recover) under bash: the fragment between the two functions and push()."""
        src = (REPO / "reproduce/03_hpo.sh").read_text()
        frag = src[src.index("geometry_to_site(){"):src.index("push(){")]
        call = "geometry_site_recover" if recover else "geometry_to_site"
        body = (f'MODEL={model}; PY={sys.executable}; CL_HOST=cluster-a-login; CL_ROOT=/shared/exp/reproduce/{model}/03_hpo/lora; disp(){{ echo "$1"; }}\n'
                f'{frag}\n{call} "{local_root}"')
        return self.bash(body, dry=dry)

    def ready_record(self, *, label: str = "cluster_a", job: str = "424242"):
        """A GENUINE record on `label`'s shared root, as the probe leaves it: plan, result files, a complete checkpoint, a
        stamped verdict + menu, and the fold's geometry record + that cluster's overlay (provenance.probe_cluster = label)."""
        import test_serving_probe_smoke as smoke          # the probe's synthetic plan / results fixtures (the REAL write_verdict runs)
        sp = smoke.sp
        primary = self.shared[label]
        out = primary / "exp/multilora_cert" / f"{KEY}_probe_{job}"; out.mkdir(parents=True, exist_ok=True)
        plan = smoke._plan(sets=("wo",), ranks=(4,), windows=(4096, 98304), ladder_caps=())
        results = smoke._fixture_results(plan, outcomes={r["id"]: ("passed", {"engine_S": 8} if r["L"] == 98304 else {}) for r in plan["launches"]})
        sp.write_json_atomic(out / sp.PLAN_FILE, plan)
        for lid, res in results.items():
            sp.write_json_atomic(sp.launch_result_path(out, lid), res)
        results = sp.load_results(out)
        sp.write_progress(plan, results, out)
        v, _allowed = sp.write_verdict(plan, results, probe_job_id=job, out=out, verdict_uri=f"file://{out}/probe_verdict.json",
                                       geometry_dir=primary / "exp/reproduce/geometry", probe_cluster=label)
        assert v["status"] == "complete" and v["geometry_fold"]["status"] == "written", v["geometry_fold"]
        return sp, plan, out

    def site_copy_from(self, out: Path) -> Path:
        """03's fetch (rsync -a of *.json into the grading host mirror: the bytes as written) + geometry_to_site into the site."""
        local_root = self.tmp / "state" / KEY / "03_hpo" / "lora"; local_root.mkdir(parents=True, exist_ok=True)
        (local_root / "geometry_record.json").write_bytes((out / "geometry_record.json").read_bytes())
        r = self.geometry_to_site(local_root)
        assert r.returncode == 0, r.stdout + r.stderr
        return self.site / "geometry" / f"{KEY}.json"


def _clean(monkeypatch):
    for v in (g.SITE_GEOMETRY_DIR_ENV, g.SITE_DIR_ENV, g.SHARED_ROOT_ENV, "SISTEER_MODELS_TABLE", "SISTEER_CLUSTER_LABEL"):
        monkeypatch.delenv(v, raising=False)


def _scan_clean(r, tmp_path):
    scan = (r.stdout + r.stderr).replace(str(tmp_path), "<tmp>").replace(os.path.expanduser("~"), "<home>")
    assert_no_tokens(scan, "the stage's output")


def test_03_hpo_copies_the_probes_record_into_the_site_and_slurm_verifies_then_stages_it_to_every_cluster_then_the_twin_reads_it(tmp_path, monkeypatch):
    _clean(monkeypatch)
    w = _World(tmp_path)
    sp, plan, out = w.ready_record()
    record_file = out / sp.GEOMETRY_RECORD_FILE
    primary_overlay = w.shared["cluster_a"] / "exp/reproduce/geometry" / f"{KEY}.json"
    assert record_file.read_bytes() == primary_overlay.read_bytes() and record_file.read_bytes().endswith(b"\n")   # one serializer
    prov = json.loads(record_file.read_text())["provenance"]
    gen = sp.result_generation(out)["generation"]
    assert prov["probe_cluster"] == "cluster_a" and prov["probe_record"] == str(out) and prov[g.RESULT_GENERATION_KEY] == gen
    site_file = w.site_copy_from(out)
    assert site_file.read_bytes() == record_file.read_bytes()
    # the mirror step VERIFIES against the record on the probe cluster (READY, the same generation) and stages to BOTH clusters
    # with the staging_verification block; the primary's own fold copy (same content, no block) is RE-STAMPED
    r = w.bash("slurm_stage_geometry cluster_a,cluster_b")
    assert r.returncode == 0, r.stdout + r.stderr
    csha = g.canonical_site_geometry_sha256(site_file)
    assert r.stdout.count("geometry: staged") == 2 and f"verified against cluster_a:{out} READY, generation {gen[:16]}" in r.stdout, r.stdout
    assert "re-stamped a copy whose staging block was absent or predates the predicate fields" in r.stdout
    for l in w.fake:
        staged = w.shared[l] / "exp/reproduce/geometry" / f"{KEY}.json"
        doc = json.loads(staged.read_text())
        assert g.canonical_site_geometry_sha256(staged) == csha                                 # the record's content, byte-canonical
        assert doc[g.STAGING_VERIFICATION_KEY]["record_generation"] == gen == doc["provenance"][g.RESULT_GENERATION_KEY]
        assert doc[g.STAGING_VERIFICATION_KEY]["verified_utc"] and doc[g.STAGING_VERIFICATION_KEY]["verifier_host"]
        # the stamp carries what the live predicate said at staging time — the fenced proof a standup on the OTHER
        # cluster (which cannot reach the record) accepts
        assert doc[g.STAGING_VERIFICATION_KEY]["record_state"] == "READY" and doc[g.STAGING_VERIFICATION_KEY]["parity_check"] == "none"
        assert doc[g.STAGING_VERIFICATION_KEY]["parity_record"] is None
        assert staged.read_bytes() == g.site_geometry_bytes(doc)                                # one serializer, block included
        assert f"geometry: staged {KEY}.json -> {l}:{w.shared[l]}/exp/reproduce/geometry (content sha256 {csha[:16]}" in r.stdout, r.stdout
        assert not list((w.shared[l] / "exp/reproduce/geometry").glob("*.tmp.*"))
    # a second run finds the same content AND the same verification: nothing rewritten
    before = {l: (w.shared[l] / "exp/reproduce/geometry" / f"{KEY}.json").read_bytes() for l in w.fake}
    r = w.bash("slurm_stage_geometry cluster_a,cluster_b")
    assert r.returncode == 0 and r.stdout.count("already holds") == 2 and "verified against cluster_a" in r.stdout, r.stdout
    assert all((w.shared[l] / "exp/reproduce/geometry" / f"{KEY}.json").read_bytes() == before[l] for l in w.fake)
    # a copy staged BEFORE the predicate fields existed (the three-field block, identical content) is RE-STAMPED, not skipped
    old_b = w.shared["cluster_b"] / "exp/reproduce/geometry" / f"{KEY}.json"
    d = json.loads(old_b.read_text()); d[g.STAGING_VERIFICATION_KEY] = {k: d[g.STAGING_VERIFICATION_KEY][k] for k in ("verified_utc", "record_generation", "verifier_host")}
    old_b.write_bytes(g.site_geometry_bytes(d))
    r = w.bash("slurm_stage_geometry cluster_a,cluster_b")
    assert r.returncode == 0 and "cluster_a already holds" in r.stdout and f"staged {KEY}.json -> cluster_b" in r.stdout and "predates the predicate fields" in r.stdout, r.stdout
    re_doc = json.loads(old_b.read_text())
    assert re_doc[g.STAGING_VERIFICATION_KEY]["record_state"] == "READY" and re_doc[g.STAGING_VERIFICATION_KEY]["parity_check"] == "none"
    assert g.canonical_site_geometry_sha256(old_b) == csha                          # the content untouched; only the block (and its fresh verified_utc) rewritten
    # PRIMARY-PROBE -> SECONDARY-WAVE: a padded unit twinned to cluster_b resolves the staged record under ITS job env
    monkeypatch.setenv(g.SHARED_ROOT_ENV, str(w.shared["cluster_b"]))
    assert g.site_geometry_dir() == w.shared["cluster_b"] / "exp/reproduce/geometry"
    rb = g.budget_for(KEY, "wo")
    assert rb.measured and rb.batched_token_budget == 8 * 98304 and g.lora_max_module_dim_for(KEY, "wo") == 64   # the toy dims' o_proj
    assert g.kernel_ceiling_adjustment(KEY, 98304, 8, module_set="wo") is None                                   # far under the int32 ceiling at D 64
    assert g.rows_for(KEY, "wo").site.path == str(w.shared["cluster_b"] / "exp/reproduce/geometry" / f"{KEY}.json")
    # the record is reachable in this two-root world; a REAL other cluster cannot reach it and accepts the copy on its block alone
    staged_b = w.shared["cluster_b"] / "exp/reproduce/geometry" / f"{KEY}.json"
    doc = json.loads(staged_b.read_text()); doc["provenance"]["probe_record"] = "/shared/exp/multilora_cert/on_the_primary_only"
    g.write_site_geometry_file(doc, staged_b)
    assert g.check_overlay_generation(g.load_site_geometry(KEY, staged_b.parent)) == "verified:staged" and g.budget_for(KEY, "wo").measured
    # the same copy WITHOUT the block (a copy that never went through the verifying step) refuses by name
    doc.pop(g.STAGING_VERIFICATION_KEY); g.write_site_geometry_file(doc, staged_b)
    with pytest.raises(g.SiteGeometryError, match="not reachable from here and the overlay carries no staging_verification"):
        g.budget_for(KEY, "wo")
    monkeypatch.delenv(g.SHARED_ROOT_ENV)
    # the SITE copy on the grading host never carries the block: a grading host process whose record is unreachable reads nothing as valid
    assert g.STAGING_VERIFICATION_KEY not in json.loads(site_file.read_text())
    # `slurm_mirror_clusters` is the one caller (before the first submission); 03's probe branch calls the copy after its fetch
    slurm = (REPO / "reproduce/lib/slurm.sh").read_text()
    assert slurm.index('|| repro_fatal "the pinned sha $sha is not mirrored') < slurm.index('slurm_stage_geometry "$2"') < slurm.index("slurm_stage_geometry(){")
    hpo = (REPO / "reproduce/03_hpo.sh").read_text()
    assert hpo.index("geometry_to_site(){") < hpo.index("if want probe; then")            # defined before the probe branch uses it
    _scan_clean(r, tmp_path)


def test_staging_refuses_a_site_copy_whose_record_is_reopened_or_regenerated_and_the_cross_cluster_preflight_refuses_too(tmp_path, monkeypatch):
    """The site copy was fetched from a READY record and staged; then the record is REOPENED
    (`run --resume --replan` writes complete=false) or a result is re-run (a newer generation). Staging must refuse by name
    — the stale geometry never reaches the other cluster — and a padded preflight on that cluster must refuse the copy it
    already holds; a READY record stages again and the staged copy verifies."""
    _clean(monkeypatch)
    w = _World(tmp_path)
    sp, plan, out = w.ready_record()
    w.site_copy_from(out)
    assert w.bash("slurm_stage_geometry cluster_a,cluster_b").returncode == 0
    staged_b = w.shared["cluster_b"] / "exp/reproduce/geometry" / f"{KEY}.json"
    before = staged_b.read_bytes()
    # (a) REOPENED: the record's checkpoint says a driver holds it (what cmd_run writes before any GPU work)
    prog = sp.load_progress(out); prog["complete"] = False; prog["attempt"] = {"started_utc": "2026-09-30T12:00:00Z", "launches_to_run": 3}
    sp.write_json_atomic(out / sp.PROGRESS_FILE, prog)
    r = w.bash("slurm_stage_geometry cluster_a,cluster_b; echo rc=$?")
    assert "rc=" not in r.stdout and "REFUSED: geometry:" in r.stderr and f"its probe record cluster_a:{out} is MENU_STALE" in r.stderr, r.stdout + r.stderr
    assert "NOT staged" in r.stderr and "nothing submitted" in r.stderr
    assert staged_b.read_bytes() == before and "geometry: staged" not in r.stdout          # nothing reached any cluster
    # the cross-cluster padded preflight refuses the copy it already holds: its record is reopened
    monkeypatch.setenv(g.SHARED_ROOT_ENV, str(w.shared["cluster_b"]))
    with pytest.raises(g.SiteGeometryError, match="REOPENED"):
        g.budget_for(KEY, "wo")
    with pytest.raises(g.SiteGeometryError, match="REOPENED"):
        g.rows_for(KEY, "wo")
    monkeypatch.delenv(g.SHARED_ROOT_ENV)
    # (b) a NEWER GENERATION: the checkpoint is complete again but a result was re-run behind the verdict
    prog["complete"] = True; prog.pop("attempt"); sp.write_json_atomic(out / sp.PROGRESS_FILE, prog)
    f = sp.launch_result_path(out, plan["launches"][0]["id"])
    doc = json.loads(f.read_text()); doc["note"] = "re-run after the verdict"; sp.write_json_atomic(f, doc)
    r = w.bash("slurm_stage_geometry cluster_a,cluster_b; echo rc=$?")
    assert "rc=" not in r.stdout and "is MENU_STALE" in r.stderr and "another result generation" in r.stderr and "NOT staged" in r.stderr, r.stderr
    monkeypatch.setenv(g.SHARED_ROOT_ENV, str(w.shared["cluster_b"]))
    with pytest.raises(g.SiteGeometryError, match="OLDER than the probe record's result set"):
        g.budget_for(KEY, "wo")
    monkeypatch.delenv(g.SHARED_ROOT_ENV)
    # (c) the record re-certified (reverdict) after (b) moved its generation on: the site copy still carries the old one.
    # A reverdict is a REOPEN (re-check 2): the carrier refuses it without the front door's token, and the front door mints the
    # token only after it revoked the site copy and every cluster's staged overlay (REVOKED markers written)
    gen1 = sp.result_generation(out)["generation"]
    with pytest.raises(SystemExit, match="REFUSING the reverdict"):
        sp.main(["reverdict", "--record", str(out), "--geometry-dir", str(w.shared["cluster_a"] / "exp/reproduce/geometry")])
    with pytest.raises(SystemExit, match="REFUSING the reverdict"):                       # the typed digest is not a token (re-check 3)
        sp.main(["reverdict", "--record", str(out), "--geometry-dir", str(w.shared["cluster_a"] / "exp/reproduce/geometry"), "--reopen-token", gen1])
    r = w.bash(f"slurm_revoke_geometry {KEY} {gen1} 'reopen (re-check 2 test)' cluster_a {out}; echo rc=$? nonce=$SLURM_REVOKE_NONCE")
    assert "rc=0" in r.stdout and r.stdout.count("revoked") >= 3 and "revocation receipt written into the record" in r.stdout, r.stdout + r.stderr
    nonce1 = r.stdout.rsplit("nonce=", 1)[1].split()[0]
    receipt = out / "reopen_receipts" / f"{gen1}.{nonce1}.json"
    assert len(nonce1) == 32 and receipt.is_file()
    rdoc = json.loads(receipt.read_text())
    assert rdoc["generation"] == gen1 and rdoc["nonce"] == nonce1 and sorted(a["cluster"] for a in rdoc["acknowledged"]) == ["cluster_a", "cluster_b"]
    assert all(a["marker"].endswith(f"{KEY}.REVOKED.json") for a in rdoc["acknowledged"]) and rdoc["minted_host"]
    for root in (w.shared["cluster_a"], w.shared["cluster_b"]):
        d = root / "exp/reproduce/geometry"
        assert not (d / f"{KEY}.json").exists() and json.loads((d / f"{KEY}.REVOKED.json").read_text())["generation"] == gen1
        assert any((a / f"{KEY}.json").is_file() for a in (d / "superseded").iterdir())
    assert not (w.site / "geometry" / f"{KEY}.json").exists() and (w.site / "geometry" / f"{KEY}.REVOKED.json").is_file()
    monkeypatch.setenv("SISTEER_CLUSTER_LABEL", "cluster_a")
    rc = sp.main(["reverdict", "--record", str(out), "--geometry-dir", str(w.shared["cluster_a"] / "exp/reproduce/geometry"), "--reopen-token", nonce1])
    assert rc == 0 and not receipt.exists() and list(receipt.parent.glob(f"{gen1}.{nonce1}.used.*.json"))     # consumed
    monkeypatch.delenv("SISTEER_CLUSTER_LABEL")
    # the probe cluster's re-folded overlay stands beside the marker the front door wrote: refused until a verified staging clears it
    monkeypatch.setenv(g.SHARED_ROOT_ENV, str(w.shared["cluster_a"]))
    with pytest.raises(g.SiteGeometryError, match="REVOKED"):
        g.budget_for(KEY, "wo")
    monkeypatch.delenv(g.SHARED_ROOT_ENV)
    # the site copy is gone (revoked): the front door has nothing stale to stage; 03 refreshes it below
    r = w.bash("slurm_stage_geometry cluster_a,cluster_b; echo rc=$?")
    assert "rc=0" in r.stdout and "no site records under" in r.stdout          # revoked: nothing stale left to stage
    # 03 refreshes the site copy from the re-certified record -> READY + the same generation -> stages to both (the fold copy on
    # the probe cluster re-stamped, the secondary staged afresh), and the verified staging CLEARS the REVOKED markers
    site_file = w.site_copy_from(out)
    gen2 = sp.result_generation(out)["generation"]
    assert gen2 == gen1 and json.loads(site_file.read_text())["provenance"][g.RESULT_GENERATION_KEY] == gen2   # a reverdict re-certifies the SAME result set
    r = w.bash("slurm_stage_geometry cluster_a,cluster_b; echo rc=$?")
    assert "rc=0" in r.stdout and r.stdout.count("geometry: staged") == 2 and f"generation {gen2[:16]}" in r.stdout, r.stdout + r.stderr
    # the marker-clearing rule: the SAME generation, but the copy was folded AFTER the revocation (a newly certified copy) -> cleared
    assert r.stdout.count("REVOKED marker cleared") == 2 and "site's" in r.stdout and "cleared (a newly certified copy" in r.stdout
    for root in (w.shared["cluster_a"], w.shared["cluster_b"]):
        d = root / "exp/reproduce/geometry"
        assert not (d / f"{KEY}.REVOKED.json").exists() and any((a / f"{KEY}.REVOKED.json").is_file() for a in (d / "superseded").iterdir())
    assert json.loads(staged_b.read_text())[g.STAGING_VERIFICATION_KEY]["record_generation"] == gen2
    monkeypatch.setenv(g.SHARED_ROOT_ENV, str(w.shared["cluster_b"]))
    assert g.budget_for(KEY, "wo").measured and g.check_overlay_generation(g.load_site_geometry(KEY)) == "verified"
    monkeypatch.delenv(g.SHARED_ROOT_ENV)
    _scan_clean(r, tmp_path)


def test_an_unreachable_record_or_a_missing_provenance_refuses_at_staging_and_an_unverified_copy_refuses_at_use(tmp_path, monkeypatch):
    _clean(monkeypatch)
    w = _World(tmp_path)
    (w.site / "geometry").mkdir()
    site_file = w.site / "geometry" / f"{KEY}.json"
    # an unreachable record: the authoritative source cannot be read -> refuse, never a degraded accept
    site_file.write_text(json.dumps(_record(probe_record="/shared/exp/multilora_cert/nowhere")))
    r = w.bash("slurm_stage_geometry cluster_a,cluster_b; echo rc=$?")
    assert "rc=" not in r.stdout and "is UNREACHABLE" in r.stderr and "cannot be verified" in r.stderr and "NOT staged" in r.stderr, r.stderr
    assert not any((w.shared[l] / "exp/reproduce/geometry").exists() for l in w.fake)
    # no probe_cluster / no generation stamp / an unknown cluster: refused by name before any ssh
    site_file.write_text(json.dumps(_record(probe_cluster=None)))
    r = w.bash("slurm_stage_geometry cluster_a,cluster_b; echo rc=$?")
    assert "rc=" not in r.stdout and "carries no provenance.probe_cluster" in r.stderr and "reverdict" in r.stderr, r.stderr
    site_file.write_text(json.dumps(_record(**{g.RESULT_GENERATION_KEY: None})))
    r = w.bash("slurm_stage_geometry cluster_a,cluster_b; echo rc=$?")
    assert "rc=" not in r.stdout and "carries no provenance.result_generation stamp" in r.stderr, r.stderr
    site_file.write_text(json.dumps(_record(probe_cluster="cluster_z")))
    r = w.bash("slurm_stage_geometry cluster_a,cluster_b; echo rc=$?")
    assert "rc=" not in r.stdout and "names probe_cluster 'cluster_z', not one of SISTEER_CLUSTERS" in r.stderr, r.stderr
    # a record dir that exists but was never certified (no menu): not READY -> refused by name
    empty = w.shared["cluster_a"] / "exp/multilora_cert" / "toy_empty"; empty.mkdir(parents=True)
    site_file.write_text(json.dumps(_record(probe_record=str(empty))))
    r = w.bash("slurm_stage_geometry cluster_a,cluster_b; echo rc=$?")
    assert "rc=" not in r.stdout and f"cluster_a:{empty} is NOT_STARTED" in r.stderr and "NOT staged" in r.stderr, r.stderr
    # at USE: a copy on a cluster whose record is unreachable and which carries no block refuses; with the block it is read
    d = w.shared["cluster_b"] / "exp/reproduce/geometry"; d.mkdir(parents=True)
    g.write_site_geometry_file(_record(probe_record="/shared/exp/multilora_cert/nowhere"), d / f"{KEY}.json")
    monkeypatch.setenv(g.SHARED_ROOT_ENV, str(w.shared["cluster_b"]))
    with pytest.raises(g.SiteGeometryError, match="carries no staging_verification block"):
        g.budget_for(KEY, "wo")
    (d / f"{KEY}.json").write_bytes(g.staged_site_geometry_bytes(_record(probe_record="/shared/exp/multilora_cert/nowhere"),
                                                                 record_generation=GEN, verifier_host="TESTHOST"))
    assert g.budget_for(KEY, "wo").measured and g.check_overlay_generation(g.load_site_geometry(KEY)) == "verified:staged"
    _scan_clean(r, tmp_path)


def test_a_cluster_copy_with_another_content_refuses_by_name_and_nothing_is_staged_over_it(tmp_path, monkeypatch):
    _clean(monkeypatch)
    w = _World(tmp_path)
    sp, plan, out = w.ready_record()
    w.site_copy_from(out)
    assert w.bash("slurm_stage_geometry cluster_a,cluster_b").returncode == 0
    tampered = w.shared["cluster_b"] / "exp/reproduce/geometry" / f"{KEY}.json"
    staged_ok = tampered.read_bytes()
    doc = _other_content(json.loads(tampered.read_text()))                                                       # another CONTENT, same generation
    tampered.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    before = tampered.read_bytes()
    r = w.bash("slurm_stage_geometry cluster_a,cluster_b; echo rc=$?")
    assert "REFUSED: geometry: cluster_b:" in r.stderr and f"{KEY}.json differs IN CONTENT from the site's copy" in r.stderr and "GEOMETRY_MISMATCH" in r.stderr
    assert "same generation, different content" in r.stderr and "nothing submitted" in r.stderr and "rc=" not in r.stdout   # repro_fatal exits the shell
    assert tampered.read_bytes() == before                          # never overwritten silently
    assert "already holds" in r.stdout                              # cluster_a (same content, same verification) was fine
    # a FORMATTING-only difference: a copy WITH the verification for this generation is left as it is; one WITHOUT the block is
    # re-stamped in place (canonical bytes + block)
    tampered.write_text(json.dumps(json.loads(staged_ok), indent=4)); reformatted_b = tampered.read_bytes()
    primary_overlay = w.shared["cluster_a"] / "exp/reproduce/geometry" / f"{KEY}.json"
    doc_a = json.loads(primary_overlay.read_text()); assert g.STAGING_VERIFICATION_KEY in doc_a
    doc_a.pop(g.STAGING_VERIFICATION_KEY); primary_overlay.write_text(json.dumps(doc_a, indent=4))
    r = w.bash("slurm_stage_geometry cluster_a,cluster_b; echo rc=$?")
    assert "rc=0" in r.stdout and "cluster_b already holds" in r.stdout and tampered.read_bytes() == reformatted_b, r.stdout + r.stderr
    assert f"staged {KEY}.json -> cluster_a" in r.stdout and "re-stamped a copy whose staging block was absent or predates the predicate fields" in r.stdout
    assert primary_overlay.read_bytes() == g.site_geometry_bytes(json.loads(primary_overlay.read_text()))
    # the whole mirror step refuses through slurm_mirror_clusters too (mirror_repo.sh stubbed: it is not the contract here)
    tampered.write_bytes(before)
    r = w.bash(f'REPRO_LIB_DIR={tmp_path}/lib; mkdir -p {tmp_path}/lib; printf "#!/bin/bash\\nexit 0\\n" > {tmp_path}/lib/mirror_repo.sh; chmod +x {tmp_path}/lib/mirror_repo.sh; '
               'slurm_mirror_clusters 0123456789abcdef0123456789abcdef01234567 cluster_a,cluster_b; echo rc=$?')
    assert "GEOMETRY_MISMATCH" in r.stderr and "rc=" not in r.stdout


def test_no_site_records_is_a_note_and_dry_runs_print_without_touching_anything(tmp_path, monkeypatch):
    _clean(monkeypatch)
    w = _World(tmp_path)
    r = w.bash("slurm_stage_geometry cluster_a,cluster_b; echo rc=$?")
    assert r.returncode == 0 and "no site records under" in r.stdout and "nothing to stage" in r.stdout and "rc=0" in r.stdout
    assert not any((w.shared[l] / "exp/reproduce/geometry").exists() for l in w.fake)
    # dry runs: the copy and the staging print their lines (no placeholders, no identifiers) and write nothing — no ssh either
    local_root = tmp_path / "lr"; local_root.mkdir(); (local_root / "geometry_record.json").write_text(json.dumps(_record()))
    r = w.geometry_to_site(local_root, dry=True)
    assert r.returncode == 0 and not (w.site / "geometry").exists(), r.stdout + r.stderr
    lines = [ln for ln in r.stdout.splitlines() if ln.startswith("DRY-RUN:")]
    assert len(lines) == 1 and "geometry record" in lines[0] and f"geometry/{KEY}.json" in lines[0] and not PLACEHOLDER.search(lines[0])
    (w.site / "geometry").mkdir(); (w.site / "geometry" / f"{KEY}.json").write_text(json.dumps(_record()))
    r = w.bash("slurm_stage_geometry cluster_a,cluster_b", dry=True)
    lines = [ln for ln in r.stdout.splitlines() if ln.startswith("DRY-RUN:")]
    assert len(lines) == 2 and all("geometry verify" in ln and "geometry stage" in ln and "(staging_verification stamped; canonical-content-checked)" in ln
                                   and "cluster_a:/shared/exp/multilora_cert/toy" in ln and not PLACEHOLDER.search(ln) for ln in lines), r.stdout
    assert not any((w.shared[l] / "exp/reproduce/geometry").exists() for l in w.fake)
    # a record naming another model, or a version-1 (model-wide) record, is refused BEFORE the copy (the site holds no record of this
    # model at all afterwards — the staging step above wrote the SITE's copy for its own dry run, removed here)
    (w.site / "geometry" / f"{KEY}.json").unlink()
    (local_root / "geometry_record.json").write_text(json.dumps({**_record(), "model_key": OTHER}))
    r = w.geometry_to_site(local_root)
    assert r.returncode != 0 and f"is not {KEY}'s serving-geometry record" in r.stderr and f"names model '{OTHER}'" in r.stderr
    assert not (w.site / "geometry" / f"{KEY}.json").exists()
    (local_root / "geometry_record.json").write_text(json.dumps({"version": 1, "model_key": KEY, "lora_max_module_dim": 4096}))
    r = w.geometry_to_site(local_root)
    assert r.returncode != 0 and "version 1 != 2" in r.stdout + r.stderr and "model-wide record is refused" in r.stdout + r.stderr
    # no record at all: a warning, never a failure (the verdict may have certified nothing)
    (local_root / "geometry_record.json").unlink()
    r = w.geometry_to_site(local_root)
    assert r.returncode == 0 and "no geometry_record*.json" in r.stderr


def test_revoke_before_reopen_with_separate_filesystem_views(tmp_path, monkeypatch):
    """Two roots where the SECOND cluster cannot reach the record (its staged copy's probe_record
    is rewritten to a nonexistent prefix, as a real other cluster sees it). Stage while READY -> the second view accepts on its
    block alone. Reopen through the front door (03's probe_reopen_token) -> the second root's overlay is archived and a REVOKED
    marker written BEFORE anything is submitted, the token minted; the preflight on the second root refuses WITHOUT any
    re-staging; a reopen with an unreachable second cluster refuses and submits nothing (fail closed); a carrier without the
    token refuses to reopen the complete record."""
    _clean(monkeypatch)
    w = _World(tmp_path)
    sp, plan, out = w.ready_record()
    w.site_copy_from(out)
    assert w.bash("slurm_stage_geometry cluster_a,cluster_b").returncode == 0
    gen = sp.result_generation(out)["generation"]
    staged_b = w.shared["cluster_b"] / "exp/reproduce/geometry" / f"{KEY}.json"
    # the second cluster's view: the record path does not exist from there
    doc = json.loads(staged_b.read_text()); doc["provenance"]["probe_record"] = "/unreachable-from-cluster_b" + str(out)
    g.write_site_geometry_file(doc, staged_b)
    monkeypatch.setenv(g.SHARED_ROOT_ENV, str(w.shared["cluster_b"]))
    assert g.check_overlay_generation(g.load_site_geometry(KEY)) == "verified:staged" and g.budget_for(KEY, "wo").measured
    monkeypatch.delenv(g.SHARED_ROOT_ENV)
    # a carrier without the token refuses to reopen the complete record (a reopened prefill OOM would be work)
    lost = next(r for r in plan["launches"] if r["L"] == 98304 and r["topology"] == "S1cc")   # the control server's budget-window launch
    f = sp.launch_result_path(out, lost["id"])
    results = sp.load_results(out)
    import test_serving_probe_budget as budget
    sp.write_json_atomic(f, budget._oom(lost, 64))                                              # an over-budget start OOM: replan work
    sp.write_json_atomic(out / sp.PROGRESS_FILE, {**sp.load_progress(out), "complete": True})                            # the checkpoint still says complete
    with pytest.raises(SystemExit, match="REFUSING the reopen .* COMPLETE"):
        sp.main(["run", "--model-key", KEY, "--out", str(out), "--no-throughput", "--resume", "--replan"])
    assert (out / sp.ALLOWED_PAIRS_FILE).is_file() and staged_b.is_file()                                                 # nothing moved anywhere
    # the front door's reopen: 03's probe_reopen_token under bash — revokes BEFORE it prints the token
    hpo = (REPO / "reproduce/03_hpo.sh").read_text()
    frag = hpo[hpo.index("json_field(){"):hpo.index("json_field(){") + hpo[hpo.index("json_field(){"):].index("\n}\n") + 3]
    frag += hpo[hpo.index("probe_reopen_token(){"):hpo.index("probe_resume(){")]
    gen_now = sp.result_generation(out)["generation"]
    pinfo = json.dumps({"complete": True, "generation": gen_now, "menu_present": True})
    r = w.bash(f"PY={sys.executable}; MODEL={KEY}; CLUSTER=cluster_a; CL_ROOT={out}; PINFO='{pinfo}'\n{frag}\ntoken=$(probe_reopen_token) || {{ echo FAILED; exit 1; }}; echo \"TOKEN=$token\"")
    assert r.returncode == 0, r.stdout + r.stderr
    token = r.stdout.rsplit("TOKEN=", 1)[1].split()[0]
    assert len(token) == 32 and token != gen_now and token not in gen_now                       # the NONCE, never the digest
    assert "revoked" in r.stderr and "receipt written into the record" in r.stderr                # progress on stderr, the token alone on stdout
    receipt = out / sp.RECEIPTS_DIR / f"{gen_now}.{token}.json"
    assert receipt.is_file() and json.loads(receipt.read_text())["nonce"] == token
    d_b = w.shared["cluster_b"] / "exp/reproduce/geometry"
    assert not staged_b.exists() and json.loads((d_b / f"{KEY}.REVOKED.json").read_text())["generation"] == gen_now
    arch = [a for a in (d_b / "superseded").iterdir() if (a / f"{KEY}.json").is_file()]
    assert len(arch) == 1 and json.loads((arch[0] / f"{KEY}.json").read_text())["provenance"]["probe_record"].startswith("/unreachable")
    assert not (w.site / "geometry" / f"{KEY}.json").exists() and (w.site / "geometry" / f"{KEY}.REVOKED.json").is_file()
    # the preflight on the second root refuses on the marker alone — no re-staging happened, the archived copy's stamps are intact
    monkeypatch.setenv(g.SHARED_ROOT_ENV, str(w.shared["cluster_b"]))
    with pytest.raises(g.SiteGeometryError, match="ABSENT and .*REVOKED.json stands where it stood"):  # the overlay is gone — and the marker
        g.load_site_geometry(KEY)                                                                       #   ALONE refuses
    g.write_site_geometry_file(json.loads((arch[0] / f"{KEY}.json").read_text()), staged_b)             # … and a job that still held its bytes
    with pytest.raises(g.SiteGeometryError, match="REVOKED"):                                           #     sees the marker at preflight
        g.budget_for(KEY, "wo")
    monkeypatch.delenv(g.SHARED_ROOT_ENV)
    # the typed digest opens nothing; the receipt's nonce opens the record once — the carrier reopens exactly the over-budget launch
    from test_serving_probe_parallel import _fake_deps
    from test_serving_probe_smoke import FakeExporter
    deps, seen = _fake_deps(plan)
    monkeypatch.setattr(sp, "ProductionExporter", lambda *a, **k: FakeExporter())
    monkeypatch.setattr(sp.Deps, "production", classmethod(lambda cls: deps))
    with pytest.raises(SystemExit, match="REFUSING the reopen"):
        sp.main(["run", "--model-key", KEY, "--out", str(out), "--no-throughput", "--resume", "--replan", "--reopen-token", gen_now])
    assert sp.main(["run", "--model-key", KEY, "--out", str(out), "--no-throughput", "--resume", "--replan", "--reopen-token", token]) == 0
    assert list(seen["stand_up"]) == [lost["id"]]
    assert not receipt.exists() and list(receipt.parent.glob(f"{gen_now}.{token}.used.*.json"))     # consumed
    with pytest.raises(SystemExit, match="REFUSING"):
        sp.consume_reopen_receipt(out, token, what="a reuse")
    # an UNREACHABLE second cluster: the reopen is refused before anything is touched, and nothing is submitted
    (tmp_path / "w2").mkdir()
    w2 = _World(tmp_path / "w2")
    sp2, plan2, out2 = w2.ready_record()
    w2.site_copy_from(out2)
    assert w2.bash("slurm_stage_geometry cluster_a,cluster_b").returncode == 0
    cfg = w2.site / "clusters" / "cluster_b.env"
    cfg.write_text(cfg.read_text().replace(f"CLUSTER_SSH=local:{w2.fake['cluster_b']}", "CLUSTER_SSH=cluster-b-login-unreachable.invalid"))
    gen2 = sp2.result_generation(out2)["generation"]
    r = w2.bash(f"slurm_revoke_geometry {KEY} {gen2} 'reopen'; echo rc=$?")
    assert "rc=" not in r.stdout and "cluster cluster_b cannot be reached" in r.stderr and "REFUSED (fail closed)" in r.stderr, r.stdout + r.stderr
    assert (w2.site / "geometry" / f"{KEY}.json").is_file() and not (w2.site / "geometry" / f"{KEY}.REVOKED.json").exists()   # nothing archived …
    assert (w2.shared["cluster_a"] / "exp/reproduce/geometry" / f"{KEY}.json").is_file()                                            # … anywhere
    pinfo2 = json.dumps({"complete": True, "generation": gen2})
    r = w2.bash(f"PY={sys.executable}; MODEL={KEY}; CLUSTER=cluster_a; CL_ROOT={out2}; PINFO='{pinfo2}'\n{frag}\nsubmit(){{ echo SUBMITTED \"$@\"; }}\n"
                f"token=$(probe_reopen_token) || {{ echo 'REOPEN REFUSED'; exit 1; }}; submit x $token")
    assert "REOPEN REFUSED" in r.stdout and "SUBMITTED" not in r.stdout and "cannot be reached" in r.stderr
    assert not (out2 / sp.RECEIPTS_DIR).exists()                                                    # no receipt without every acknowledgement
    # 03's probe_resume mints the token BEFORE its submit and appends it as positional 10; an incomplete record gets none
    i = hpo.index("probe_resume(){"); body = hpo[i:hpo.index("\n}\n", i) + 3]
    # probe_resume submits under the RESUMED record's unit (`$punit` = the probe unit, or the parity record's `.parity` suffix)
    assert body.index('token="$(probe_reopen_token)"') < body.index('submit "$punit.a$R_ATTEMPT"') and '$R_PARALLEL${token:+ $token}${phase:+ $phase}"' in body
    assert 'punit="$(unit probe)${PROBE_UNIT_SUFFIX:-}"' in body
    r = w.bash(f"PY={sys.executable}; MODEL={KEY}; CLUSTER=cluster_a; CL_ROOT={out}; PINFO='{json.dumps({'complete': False, 'generation': gen_now})}'\n{frag}\ntoken=$(probe_reopen_token); echo \"TOKEN=[$token]\"")
    assert "TOKEN=[]" in r.stdout and "revoked" not in r.stderr
    # 03's deliberate reopen verb (--phase probe --reopen): READY -> archive the local copies + DONE marker, resume with the nonce as
    # positional 10 (submit stubbed; probe_resume's derivation stubbed to its spec line); any other state refuses by name
    verb = hpo[hpo.index("probe_reopen_verb(){"):hpo.index("probe_resume(){")]
    stub = ('probe_archive_local(){ echo "ARCHIVED-LOCAL $2"; }; probe_resume(){ local t; t="$(probe_reopen_token)" || exit 1; '
            'submit "x" "$MODEL a b c d e 3 5 auto${t:+ $t}"; }; submit(){ echo "SUBMITTED $2"; }; repro_say(){ echo "$*"; }; repro_fatal(){ echo "REFUSED: $*" >&2; exit 1; }')
    (w.site / "geometry").mkdir(exist_ok=True)
    gen3 = sp.result_generation(out)["generation"]                                          # the record moved on with the reopen above
    pinfo3 = json.dumps({"complete": True, "generation": gen3, "menu_present": True})          # what the live status says in the same invocation
    r = w.bash(f"PY={sys.executable}; MODEL={KEY}; CLUSTER=cluster_a; CL_ROOT={out}; PHASE=probe; PSTATE=READY; PDETAIL=ready; PINFO='{pinfo3}'\n{frag}\n{verb}\n{stub}\nprobe_reopen_verb; echo PSTATE=$PSTATE")
    assert r.returncode == 0 and "ARCHIVED-LOCAL READY" in r.stdout and "PSTATE=REOPENED" in r.stdout, r.stdout + r.stderr
    sub = [ln for ln in r.stdout.splitlines() if ln.startswith("SUBMITTED ")][0]
    nonce2 = sub.split()[-1]
    assert len(nonce2) == 32 and nonce2 != gen3 and (out / sp.RECEIPTS_DIR / f"{gen3}.{nonce2}.json").is_file()
    # --reopen on an INCOMPLETE_STALLED record RESUMES it (no receipt: not complete) — the stubbed probe_resume submits without a nonce
    r = w.bash(f"PY={sys.executable}; MODEL={KEY}; CLUSTER=cluster_a; CL_ROOT={out}; PHASE=probe; PSTATE=INCOMPLETE_STALLED; PDETAIL=x; PINFO='{{}}'\n{frag}\n{verb}\n{stub}\nprobe_reopen_verb; echo PSTATE=$PSTATE")
    assert r.returncode == 0 and "resuming it (no receipt: the record is not complete)" in r.stdout and "SUBMITTED" in r.stdout and "PSTATE=REOPENED" in r.stdout, r.stdout + r.stderr
    assert [ln for ln in r.stdout.splitlines() if ln.startswith("SUBMITTED ")][0].split()[-1] == "auto"        # no positional 10
    # any state that is neither READY nor a resume shape refuses by name
    r = w.bash(f"PY={sys.executable}; MODEL={KEY}; CLUSTER=cluster_a; CL_ROOT={out}; PHASE=probe; PSTATE=RUNNING; PDETAIL=x; PINFO='{{}}'\n{frag}\n{verb}\n{stub}\nprobe_reopen_verb; echo PSTATE=$PSTATE")
    assert r.returncode != 0 and "--reopen needs a READY record" in r.stderr and "SUBMITTED" not in r.stdout
    _scan_clean(r, tmp_path)


def test_staging_and_revocation_exclude_each_other_and_a_same_generation_marker_is_kept(tmp_path, monkeypatch):
    """A stager that passed READY, prepared the old-generation bytes and PAUSED (SISTEER_STAGE_PAUSE_FILE)
    while the revocation archived + marked must not install the old overlay or clear the marker: under the per-model lock it
    finds the site's REVOKED marker for that generation and refuses. A marker of the SAME generation is never cleared by a copy
    certified before the revocation; a copy certified after it clears the marker."""
    _clean(monkeypatch)
    w = _World(tmp_path)
    sp, plan, out = w.ready_record()
    w.site_copy_from(out)
    assert w.bash("slurm_stage_geometry cluster_a,cluster_b").returncode == 0
    gen = sp.result_generation(out)["generation"]
    staged_b = w.shared["cluster_b"] / "exp/reproduce/geometry" / f"{KEY}.json"
    # a stager in flight: passes READY, then pauses on the hook before taking the lock
    pause = tmp_path / "pause"; pause.touch()
    env = {**w.env, "SISTEER_STAGE_PAUSE_FILE": str(pause)}
    script = (f"set -u; source {REPO}/reproduce/lib/common.sh; source {REPO}/reproduce/lib/slurm.sh; REPRO_CONFIG={w.site}/config.env; REPRO_DRY_RUN=0; repro_load_config\n"
              "slurm_stage_geometry cluster_a,cluster_b; echo rc=$?\n")
    stager = subprocess.Popen(["bash", "-c", script], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    import time
    t0 = time.time()
    while time.time() - t0 < 90 and not (tmp_path / "pause.waiting").exists() and stager.poll() is None:
        time.sleep(0.2)                                                                  # the stager touches pause.waiting once READY is checked and the bytes are in hand
    assert (tmp_path / "pause.waiting").exists() and stager.poll() is None, stager.communicate(timeout=30)   # parked before the lock
    # the front door revokes meanwhile (takes the lock, archives, marks, writes the receipt)
    r = w.bash(f"slurm_revoke_geometry {KEY} {gen} 'reopen during staging' cluster_a {out}; echo rc=$?")
    assert "rc=0" in r.stdout and not staged_b.exists() and (staged_b.parent / f"{KEY}.REVOKED.json").is_file(), r.stdout + r.stderr
    marker_b = json.loads((staged_b.parent / f"{KEY}.REVOKED.json").read_text())
    pause.unlink()                                                                        # release the stager
    out_s, err_s = stager.communicate(timeout=120)
    # the stager refused under the lock — nothing installed, the marker untouched
    assert "rc=" not in out_s and "was REVOKED by the front door" in err_s and "NOT staged (nothing installed, no marker cleared)" in err_s, out_s + err_s
    assert not staged_b.exists() and json.loads((staged_b.parent / f"{KEY}.REVOKED.json").read_text()) == marker_b
    assert not (w.shared["cluster_a"] / "exp/reproduce/geometry" / f"{KEY}.json").exists()
    # the site copy stays revoked too (a fresh site copy of the SAME generation folded BEFORE the revocation is refused as well)
    site_file = w.site_copy_from(out)
    r = w.bash("slurm_stage_geometry cluster_a,cluster_b; echo rc=$?")
    assert "rc=" not in r.stdout and "was REVOKED by the front door" in r.stderr and not staged_b.exists()
    # a copy certified AFTER the revocation (a reverdict through the receipt) clears the markers though the generation is the same
    nonce = r.stdout  # placeholder to keep flake-free
    receipt = next((out / sp.RECEIPTS_DIR).glob(f"{gen}.*.json"))
    nonce = receipt.name.split(".")[1]
    monkeypatch.setenv("SISTEER_CLUSTER_LABEL", "cluster_a")
    assert sp.main(["reverdict", "--record", str(out), "--geometry-dir", str(w.shared["cluster_a"] / "exp/reproduce/geometry"), "--reopen-token", nonce]) == 0
    monkeypatch.delenv("SISTEER_CLUSTER_LABEL")
    site_file = w.site_copy_from(out)
    assert json.loads(site_file.read_text())["written_utc"] >= marker_b["revoked_utc"]
    r = w.bash("slurm_stage_geometry cluster_a,cluster_b; echo rc=$?")
    assert "rc=0" in r.stdout and r.stdout.count("geometry: staged") == 2 and r.stdout.count("REVOKED marker cleared") == 2, r.stdout + r.stderr
    assert staged_b.is_file() and not (staged_b.parent / f"{KEY}.REVOKED.json").exists() and not (w.site / "geometry" / f"{KEY}.REVOKED.json").exists()
    # and a marker of the same generation with a copy folded BEFORE it is KEPT: re-mark, stage the same (older) site copy -> refused at the site check
    r = w.bash(f"slurm_revoke_geometry {KEY} {gen} 'again' cluster_a {out}; echo rc=$?")
    assert "rc=0" in r.stdout
    site_file.write_bytes((out / sp.GEOMETRY_RECORD_FILE).read_bytes())                   # the same copy (folded before this new marker)
    r = w.bash("slurm_stage_geometry cluster_a,cluster_b; echo rc=$?")
    assert "rc=" not in r.stdout and "was REVOKED by the front door" in r.stderr and not staged_b.exists()
    _scan_clean(r, tmp_path)
    # ONE revocation epoch per record (item 2): the 'again' receipt is still ACTIVE (unconsumed); a further revocation
    # moves it aside as .superseded (never deleted) before its own receipt lands, says so, and the carrier refuses the old nonce by name
    active = [p for p in (out / sp.RECEIPTS_DIR).iterdir() if sp.RECEIPT_FILE_RE.match(p.name) and sp.RECEIPT_FILE_RE.match(p.name).group("state") is None]
    assert len(active) == 1
    again_nonce = sp.RECEIPT_FILE_RE.match(active[0].name).group("nonce")
    r = w.bash(f"slurm_revoke_geometry {KEY} {gen} 'a third revocation' cluster_a {out}; echo rc=$?")
    assert "rc=0" in r.stdout and "1 prior active receipt(s) superseded" in r.stdout, r.stdout + r.stderr
    assert not active[0].exists() and len(list((out / sp.RECEIPTS_DIR).glob(f"{gen}.{again_nonce}.superseded.*.json"))) == 1
    third = [p for p in (out / sp.RECEIPTS_DIR).iterdir() if sp.RECEIPT_FILE_RE.match(p.name) and sp.RECEIPT_FILE_RE.match(p.name).group("state") is None]
    assert len(third) == 1 and third[0].name != active[0].name
    with pytest.raises(SystemExit, match="SUPERSEDED by a later revocation"):
        sp.consume_reopen_receipt(out, again_nonce, what="a reuse")
    assert sp.consume_reopen_receipt(out, sp.RECEIPT_FILE_RE.match(third[0].name).group("nonce"), what="the third reopen")["nonce"] == sp.RECEIPT_FILE_RE.match(third[0].name).group("nonce")
    _scan_clean(r, tmp_path)


def test_a_fold_and_a_revocation_in_the_same_second_are_ordered_by_the_revocation_epoch_never_by_the_clock(tmp_path, monkeypatch):
    """B11 (2026-10-09): `slurm_stage_geometry` ordered a same-generation site copy against a REVOKED marker by `written_utc` <
    `revoked_utc` — two second-resolution wall-clock stamps compared strictly — so a copy folded in the SAME second as a LATER
    revocation read as folded-after and CLEARED that revocation's marker (the staging/revocation test above flipped `rc=0` on four
    landing gates when its reverdict's fold and its 'again' revocation landed in one second). Under frozen clocks here (a `date` fake
    on PATH for the front door, `utc_now` patched for the fold) every stamp reads the same second, so the clock orders nothing: the
    marker carries the revocation's epoch (sha256 of its receipt nonce, the nonce itself nowhere but the receipt), the re-fold
    certified THROUGH that receipt carries the same epoch and lifts the marker, and the same copy against a later revocation's marker
    (another epoch) is refused by name. A marker without an epoch (written before the epoch existed) still orders by the clock —
    strictly: a tie at second resolution refuses, a fold one second after it lifts it."""
    import shutil
    _clean(monkeypatch)
    w = _World(tmp_path)
    frozen = "2026-10-09T12:00:00Z"
    bin_dir = tmp_path / "bin"; bin_dir.mkdir()
    (bin_dir / "date").write_text(f'#!/bin/bash\ncase "$*" in *%FT%TZ*) echo {frozen};; *%Y%m%dT%H%M%SZ*) echo 20261009T120000Z;; *) exec {shutil.which("date")} "$@";; esac\n')
    (bin_dir / "date").chmod(0o755)
    w.env["PATH"] = f"{bin_dir}:{os.environ['PATH']}"
    sp = _smoke.sp
    monkeypatch.setattr(sp, "utc_now", lambda: frozen)                     # every fold's written_utc, in the front door's frozen second
    sp, plan, out = w.ready_record()
    assert json.loads((out / sp.GEOMETRY_RECORD_FILE).read_text())["written_utc"] == frozen
    w.site_copy_from(out)
    assert w.bash("slurm_stage_geometry cluster_a,cluster_b").returncode == 0
    gen = sp.result_generation(out)["generation"]
    d_b = w.shared["cluster_b"] / "exp/reproduce/geometry"; staged_b = d_b / f"{KEY}.json"; marker_b = d_b / f"{KEY}.REVOKED.json"
    site_marker = w.site / "geometry" / f"{KEY}.REVOKED.json"
    # revocation 1, in the fold's second
    r = w.bash(f"slurm_revoke_geometry {KEY} {gen} 'reopen' cluster_a {out}; echo rc=$?")
    assert "rc=0" in r.stdout, r.stdout + r.stderr
    m1 = json.loads(marker_b.read_text())
    assert m1["revoked_utc"] == frozen == json.loads((out / sp.GEOMETRY_RECORD_FILE).read_text())["written_utc"]   # the clock orders nothing
    # THE BUG, first: the copy folded BEFORE that revocation, in its second, must not lift its marker — `written_utc < revoked_utc`
    # read the tie as fold-after, cleared every marker and installed the stale copy (rc=0, the gates' failure shape)
    site_file = w.site_copy_from(out)
    r = w.bash("slurm_stage_geometry cluster_a,cluster_b; echo rc=$?")
    assert "rc=" not in r.stdout and "was REVOKED by the front door" in r.stderr and "NOT staged (nothing installed, no marker cleared)" in r.stderr, r.stdout + r.stderr
    assert not staged_b.exists() and json.loads(marker_b.read_text()) == m1 and site_marker.is_file()
    # what orders them instead: every marker carries the receipt's epoch (sha256 of its nonce), no marker the nonce; the stale copy has none
    receipt = next(p for p in (out / sp.RECEIPTS_DIR).iterdir() if sp.RECEIPT_FILE_RE.match(p.name))
    nonce1 = sp.RECEIPT_FILE_RE.match(receipt.name).group("nonce")
    assert m1["epoch"] == sp.reopen_epoch(nonce1) == json.loads(receipt.read_text())["epoch"] == json.loads(site_marker.read_text())["epoch"]
    assert nonce1 not in marker_b.read_text() and "not certified through that revocation's receipt" in r.stderr and "reopen epoch is absent" in r.stderr
    # the re-fold THROUGH the receipt (the reverdict), still in the same second: it carries the epoch and lifts every marker
    monkeypatch.setenv("SISTEER_CLUSTER_LABEL", "cluster_a")
    assert sp.main(["reverdict", "--record", str(out), "--geometry-dir", str(w.shared["cluster_a"] / "exp/reproduce/geometry"), "--reopen-token", nonce1]) == 0
    monkeypatch.delenv("SISTEER_CLUSTER_LABEL")
    rec = json.loads((out / sp.GEOMETRY_RECORD_FILE).read_text())
    assert rec["written_utc"] == frozen and rec["provenance"][sp.REOPEN_EPOCH_KEY] == m1["epoch"] and nonce1 not in json.dumps(rec)
    site_file = w.site_copy_from(out)
    r = w.bash("slurm_stage_geometry cluster_a,cluster_b; echo rc=$?")
    assert "rc=0" in r.stdout and r.stdout.count("REVOKED marker cleared") == 2 and "REVOKED.json cleared" in r.stdout, r.stdout + r.stderr
    assert staged_b.is_file() and not marker_b.exists() and not site_marker.exists()
    # revocation 2 ('again'), in the same second as that fold: the clock reads the fold as at-or-after it; the epoch says otherwise
    r = w.bash(f"slurm_revoke_geometry {KEY} {gen} 'again' cluster_a {out}; echo rc=$?")
    assert "rc=0" in r.stdout, r.stdout + r.stderr
    m2 = json.loads(marker_b.read_text())
    assert m2["revoked_utc"] == frozen == rec["written_utc"] and m2["epoch"] != m1["epoch"]
    site_file.write_bytes((out / sp.GEOMETRY_RECORD_FILE).read_bytes())            # the same copy: certified under revocation 1's receipt
    r = w.bash("slurm_stage_geometry cluster_a,cluster_b; echo rc=$?")
    assert "rc=" not in r.stdout and "was REVOKED by the front door" in r.stderr and "not certified through that revocation's receipt" in r.stderr, r.stdout + r.stderr
    assert not staged_b.exists() and json.loads(marker_b.read_text()) == m2 and site_marker.is_file()
    # a marker without an epoch (written before the epoch existed) orders by the clock: a tie at second resolution refuses …
    legacy = {"model_key": KEY, "generation": gen, "revoked_utc": frozen, "reason": "legacy", "by": "test"}
    site_marker.write_text(json.dumps(legacy))
    r = w.bash("slurm_stage_geometry cluster_a,cluster_b; echo rc=$?")
    assert "rc=" not in r.stdout and "a tie at second resolution is not an order" in r.stderr and not staged_b.exists(), r.stdout + r.stderr
    # … and a fold strictly after it lifts it — while the targets' epoch markers of revocation 2 stand (KEPT, by name)
    site_marker.write_text(json.dumps({**legacy, "revoked_utc": "2026-10-09T11:59:59Z"}))
    r = w.bash("slurm_stage_geometry cluster_a,cluster_b; echo rc=$?")
    assert "rc=0" in r.stdout and r.stdout.count("REVOKED marker KEPT") == 2 and "REVOKED.json cleared" in r.stdout, r.stdout + r.stderr
    assert staged_b.is_file() and json.loads(marker_b.read_text()) == m2 and not site_marker.exists()
    _scan_clean(r, tmp_path)


def test_a_cluster_overlay_newer_than_the_site_copy_is_never_overwritten_and_an_older_fold_of_the_same_generation_is(tmp_path, monkeypatch):
    """`slurm_stage_geometry` verified the SITE copy against the record (READY, the same generation) but never ordered it against
    the overlay ALREADY on a cluster — after a reverdict (which keeps the result generation but rewrites the overlay: nine pairs
    verified) an older all-mismatch site copy of the same generation would have verified and clobbered it. The order is the fold's own
    `written_utc` (never mtime): a cluster overlay written AFTER the site copy refuses by name and is not touched; one written BEFORE
    it (an older fold of the same generation) is replaced; identical content (modulo the staging block) is a no-op / a re-stamp."""
    _clean(monkeypatch)
    w = _World(tmp_path)
    sp, plan, out = w.ready_record()
    site_file = w.site_copy_from(out)
    assert w.bash("slurm_stage_geometry cluster_a,cluster_b").returncode == 0
    site_utc = json.loads(site_file.read_text())["written_utc"]
    csha = g.canonical_site_geometry_sha256(site_file)
    staged_b = w.shared["cluster_b"] / "exp/reproduce/geometry" / f"{KEY}.json"

    # (1) NEWER on the cluster (a later fold of the same generation the site has not fetched): refused by name, bytes untouched
    doc = _other_content(json.loads(staged_b.read_text())); doc["written_utc"] = "2099-01-01T00:00:00Z"
    staged_b.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n"); before = staged_b.read_bytes()
    r = w.bash("slurm_stage_geometry cluster_a,cluster_b; echo rc=$?")
    assert "REFUSED: geometry: cluster_b:" in r.stderr and f"{KEY}.json is NEWER than the site's copy" in r.stderr and "GEOMETRY_NEWER" in r.stderr, r.stderr
    assert "written 2099-01-01T00:00:00Z" in r.stderr and f"postdates the site copy (written {site_utc})" in r.stderr and "refresh the site copy from the record first" in r.stderr
    assert "NOT overwritten" in r.stderr and "nothing submitted" in r.stderr and "rc=" not in r.stdout
    assert staged_b.read_bytes() == before and "cluster_a already holds" in r.stdout
    # a newer copy of ANOTHER generation (a re-probe's fold the site never fetched) is refused the same way — newer wins over "stale"
    doc = json.loads(before); doc["provenance"][g.RESULT_GENERATION_KEY] = "f" * 64; doc[g.STAGING_VERIFICATION_KEY]["record_generation"] = "f" * 64
    staged_b.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n"); before2 = staged_b.read_bytes()
    r = w.bash("slurm_stage_geometry cluster_a,cluster_b; echo rc=$?")
    assert "is NEWER than the site's copy" in r.stderr and staged_b.read_bytes() == before2 and "rc=" not in r.stdout
    # (2) OLDER on the cluster (the fold before the reverdict, same generation, other content): replaced, the say line names both times
    doc = _other_content(json.loads(before)); doc["written_utc"] = "2000-01-01T00:00:00Z"
    staged_b.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    r = w.bash("slurm_stage_geometry cluster_a,cluster_b; echo rc=$?")
    assert "rc=0" in r.stdout and f"staged {KEY}.json -> cluster_b" in r.stdout, r.stdout + r.stderr
    assert f"replaced an OLDER fold of the same generation (cluster copy written 2000-01-01T00:00:00Z, site copy {site_utc})" in r.stdout
    assert g.canonical_site_geometry_sha256(staged_b) == csha and json.loads(staged_b.read_text())["written_utc"] == site_utc
    assert json.loads(staged_b.read_text())[g.STAGING_VERIFICATION_KEY]["record_generation"] == sp.result_generation(out)["generation"]
    # (3) identical content: a no-op on both clusters, nothing rewritten
    bytes_before = {l: (w.shared[l] / "exp/reproduce/geometry" / f"{KEY}.json").read_bytes() for l in w.fake}
    r = w.bash("slurm_stage_geometry cluster_a,cluster_b; echo rc=$?")
    assert "rc=0" in r.stdout and r.stdout.count("already holds") == 2
    assert all((w.shared[l] / "exp/reproduce/geometry" / f"{KEY}.json").read_bytes() == bytes_before[l] for l in w.fake)
    # (4) the same written_utc with other content is still the tampered-copy refusal (nothing to order it by), and a site copy without
    # written_utc refuses before anything is compared
    doc = _other_content(json.loads(staged_b.read_text())); staged_b.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n"); before4 = staged_b.read_bytes()
    r = w.bash("slurm_stage_geometry cluster_a,cluster_b; echo rc=$?")
    assert "differs IN CONTENT" in r.stderr and "same written_utc" in r.stderr and staged_b.read_bytes() == before4
    staged_b.write_bytes(bytes_before["cluster_b"])
    sdoc = json.loads(site_file.read_text()); sdoc.pop("written_utc"); site_file.write_text(json.dumps(sdoc, indent=2, sort_keys=True) + "\n")
    r = w.bash("slurm_stage_geometry cluster_a,cluster_b; echo rc=$?")
    assert "carries no written_utc" in r.stderr and "rc=" not in r.stdout


def test_a_record_larger_than_the_argv_limit_stages_over_stdin_and_a_transfer_mismatch_refuses(tmp_path, monkeypatch):
    """2026-10-01: the staged record used to ride as base64 INSIDE the ssh remote command, so a
    178,775-byte v2 record exceeded Linux's ~128 KiB single-argument limit (`ssh: Argument list too long`, rc 126) — nothing staged,
    markers not cleared. The bytes now travel over ssh stdin from a local temp file (repro_ssh_stdin; the `local:` seam is the same
    `bash -c` with the file on stdin, so an inlined payload would still hit E2BIG here) and the remote counts + hashes them."""
    _clean(monkeypatch)
    w = _World(tmp_path)
    sp, plan, out = w.ready_record()
    site_copy = w.site_copy_from(out)
    # pad the SITE copy's provenance past the argv limit (a legitimate extra provenance field; the record's generation is untouched)
    doc = json.loads(site_copy.read_text())
    doc["provenance"]["attestation_sha256"].update({f"pad_launch_{i:04d}": {"file": "x" * 160, "sha256": "ab" * 32} for i in range(700)})
    site_copy.write_text(json.dumps(doc, indent=2, sort_keys=True))
    assert site_copy.stat().st_size > 131072 * 1.3, site_copy.stat().st_size
    # cluster_b holds no copy yet: staging there must transfer the whole record and verify it
    r = w.bash("slurm_stage_geometry cluster_b; echo rc=$?")
    assert "rc=0" in r.stdout and "geometry: staged" in r.stdout + r.stderr and "cluster_b" in r.stdout, r.stdout + r.stderr
    staged = w.shared["cluster_b"] / "exp/reproduce/geometry" / f"{KEY}.json"
    assert staged.is_file() and staged.stat().st_size > 131072
    got = json.loads(staged.read_text())
    assert got["provenance"]["attestation_sha256"]["pad_launch_0699"]["sha256"] == "ab" * 32 and g.STAGING_VERIFICATION_KEY in got
    # the helper streams the file on the remote's stdin (the seam proves the form, not a string match)
    (tmp_path / "payload.bin").write_bytes(b"x" * 200000)
    r = w.bash(f"repro_ssh_stdin cluster_a 'wc -c | tr -d \" \"' {tmp_path}/payload.bin")
    assert r.returncode == 0 and r.stdout.strip().endswith("200000"), r.stdout + r.stderr
    # a transfer that does not reproduce the staged bytes refuses by name: the remote's install script, fed a truncated stream
    slurm = (REPO / "reproduce/lib/slurm.sh").read_text()
    assert "GEOMETRY_TRANSFER_MISMATCH" in slurm and "repro_ssh_stdin \"$l\" \"$remote\" \"$payload\"" in slurm
    assert "printf '%s' '$b64' | base64 -d" not in slurm                     # no inlined payload anywhere in the install command
    i = slurm.index("gotn=\\$(wc -c")
    assert "exit 9" in slurm[i:i + 400] and '9:*) rm -f "$payload"; repro_fatal "geometry: the transfer of $name to $l did not reproduce the staged bytes' in slurm
    _scan_clean(r, tmp_path)


def test_the_sites_marker_moves_into_a_directory_made_with_the_same_timestamp(tmp_path, monkeypatch):
    """2026-10-01 08:31:35Z: the site's REVOKED marker was moved into `superseded/$(date)/` while the directory had been made with a
    SEPARATE `$(date)` a second earlier — "Not a directory", the marker stayed, the staging reported REFUSED after both clusters had
    been staged. The move now uses ONE timestamp; under a `date` that advances a second per call the marker still lands in the
    directory that was made."""
    _clean(monkeypatch)
    w = _World(tmp_path)
    sp, plan, out = w.ready_record()
    site_copy = w.site_copy_from(out)
    assert w.bash("slurm_stage_geometry cluster_a,cluster_b").returncode == 0
    # a marker of ANOTHER generation on the site (the rule clears it once every target holds the current generation)
    marker = w.site / "geometry" / f"{KEY}.REVOKED.json"
    marker.write_text(json.dumps({"model_key": KEY, "generation": "f" * 64, "revoked_utc": "2026-09-30T00:00:00Z", "reason": "test", "by": "test"}))
    # `date` advances one second per call: a two-call mkdir/mv would straddle; the staging must still clear the marker
    r = w.bash('n=0; date(){ n=$((n+1)); command date -u -d "@$((1790000000 + n))" "$@"; }; export -f date 2>/dev/null; slurm_stage_geometry cluster_a,cluster_b; echo rc=$?')
    assert "rc=0" in r.stdout and "REVOKED.json cleared" in r.stdout + r.stderr, r.stdout + r.stderr
    assert not marker.exists()
    moved = list((w.site / "geometry" / "superseded").glob("*/" + f"{KEY}.REVOKED.json"))
    assert len(moved) == 1, moved
    slurm = (REPO / "reproduce/lib/slurm.sh").read_text()
    # ONE directory, reserved by repro_superseded_dir (a fresh one per archival since 2026-10-09, test_geometry_archives_land_in_their_own_directories.py),
    # and the mv into that same local — never a second `$(date)`
    i = slurm.index('local sup; sup="$(repro_superseded_dir "$dir/superseded")"')
    assert 'mv -f "$dir/${name%.json}.REVOKED.json" "$sup/"' in slurm[i:i + 600] and "$(date" not in slurm[i:i + 600]
    _scan_clean(r, tmp_path)


def test_a_targets_marker_is_cleared_with_one_timestamp_and_a_failed_clear_is_never_masked_by_a_success_line(tmp_path, monkeypatch):
    """The REMOTE clear_marker interpolated `$(date)` twice too; a re-stage crossing a second left the cluster's
    marker standing while GEOMETRY_SAME / GEOMETRY_STAGED reported success, and every consumer of the overlay then refused as REVOKED.
    One timestamp now; a clear that fails exits the remote with rc 10 (GEOMETRY_MARKER_CLEAR_FAILED) before any success line."""
    _clean(monkeypatch)
    w = _World(tmp_path)
    sp, plan, out = w.ready_record()
    w.site_copy_from(out)
    assert w.bash("slurm_stage_geometry cluster_a,cluster_b").returncode == 0
    gdir = w.shared["cluster_b"] / "exp/reproduce/geometry"
    marker = gdir / f"{KEY}.REVOKED.json"
    marker.write_text(json.dumps({"model_key": KEY, "generation": "f" * 64, "revoked_utc": "2026-09-30T00:00:00Z", "reason": "test", "by": "test"}))
    # under an advancing `date` (one second per call) the target's marker of another generation is cleared into ONE directory
    r = w.bash('n=0; date(){ n=$((n+1)); command date -u -d "@$((1790000000 + n))" "$@"; }; export -f date 2>/dev/null; slurm_stage_geometry cluster_b; echo rc=$?')
    assert "rc=0" in r.stdout and "REVOKED marker cleared" in r.stdout + r.stderr, r.stdout + r.stderr
    assert not marker.exists() and len(list((gdir / "superseded").glob(f"*/{KEY}.REVOKED.json"))) == 1
    # a clear that CANNOT succeed (the target's superseded/ is a file): refused by name, rc nonzero, no "already holds" / "staged" success
    marker.write_text(json.dumps({"model_key": KEY, "generation": "e" * 64, "revoked_utc": "2026-09-30T00:00:00Z", "reason": "test", "by": "test"}))
    import shutil
    shutil.rmtree(gdir / "superseded"); (gdir / "superseded").write_text("not a directory")
    r = w.bash("slurm_stage_geometry cluster_b; echo rc=$?")
    assert "rc=0" not in r.stdout and "GEOMETRY_MARKER_CLEAR_FAILED" in r.stdout + r.stderr and "marker still stands" in r.stdout + r.stderr, r.stdout + r.stderr
    assert "already holds" not in r.stdout and "geometry: staged" not in r.stdout
    assert marker.exists()                                                      # the marker stands, and the front door SAID so
    slurm = (REPO / "reproduce/lib/slurm.sh").read_text()
    # the staging's two archives (the targets' and the site's) each take ONE stamp, held in a local — never `$(date)` inside the mv
    # (the stamp is the microsecond one of repro_superseded_stamp since 2026-10-09, and the directory is reserved: the archive test module)
    assert 'local sup_ts; sup_ts="$(repro_superseded_stamp)"' in slurm and 'local sup; sup="$(repro_superseded_dir "$dir/superseded")"' in slurm
    stage = slurm[slurm.index("slurm_stage_geometry(){"):slurm.index("slurm_revoke_geometry(){")]
    assert "mv -f" in stage and "$(date" not in stage[stage.index("reserve_sup(){"):stage.index("clear_marker(){") + 600]
    assert "exit 10; }; else echo 'GEOMETRY_MARKER_KEPT'" in slurm and '10:*) rm -f "$payload"; repro_fatal' in slurm
    _scan_clean(r, tmp_path)
