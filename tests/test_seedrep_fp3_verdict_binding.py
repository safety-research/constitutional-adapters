"""qwen-seedrep-native-battery F-AO (fixer fp3, on fp2's branch): the request-evidence audit's verdict FILES are bound to the plan's
children through the verified arm binding — the app's `producer_evidence` and the certificate's `_smoke_check` alike — never by a
caller-composed filename and never by a looser string compare.

The seam: the audit files each verdict under the STORE's spelling (`verdict_filename(store.arm, …)`: the harness id for a
misalign cell — `si_constv2_mix_full__lora_r4_ce__s1.json` — and `arditi_allpos__s1.json` for the Arditi cell), while the app bound one
expected file per plan child via `lp.audit_verdict_rels(k["eval_leg"], k["arm"], …)` — the plan child's ARM LABEL — and the certificate
compared the verdict's `doc["arm"]` with `child["arm"]`. For misalign the bound file read ABSENT even when the audit AGREED; for Arditi
(`ar_c0p3__arditi_allpos` label vs `arditi_allpos` store) likewise. Fix: the audit also writes a WRITER-PRODUCED index of every verdict
it filed (`audits/request_log_index.json`: rel, leg, s, stage, shard, the store's arm, and its `arm_binding` — harness id, TrainTest id,
vmodel, spelling); the app resolves each plan child's CELL through that index (+ the standup's own `endpoint.json.vmodels` as the
independent cross-check of the vmodel), binds the ONE file the audit wrote — refusing by name when none, two, or a verdict whose stamped
`arm_binding` disagrees with the index / the standup record / the child's cell — and records `bound_rel` / `plan_arm` / `plan_cell`
beside the sha; the certificate reads the bound file through the producer record and compares through the verdict's `arm_binding`
(TrainTest id, else harness id, == the child's cell; `doc["arm"]` == the binding's store arm) — the original store spelling is preserved
on the verdict and in the filename.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "scripts" / "sdfbase" / "seedrep", REPO / "scripts" / "modal", REPO / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import audit_common as ac  # noqa: E402
import request_evidence_audit as rea  # noqa: E402
import seedrep_audit_fixtures as fx  # noqa: E402
import seedrep_launch_plan as lp  # noqa: E402

DRIVER = REPO / "scripts" / "modal" / "seedrep_served.py"
TT = "constv2-mix-full__lora_r4_ce"
HARNESS = "si_constv2_mix_full__lora_r4_ce"
HARNESS_OF = {TT: HARNESS}
_ABSENT = object()


def _stub_modal():
    import types

    class _Any:
        def __init__(self, *a, **k): pass
        def __call__(self, *a, **k): return self
        def __getattr__(self, name): return _Any()
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def __iter__(self): return iter(())

    m = types.ModuleType("modal")
    for name in ("App", "Image", "Volume", "Dict", "Secret", "Function", "FunctionCall", "Cls", "Sandbox", "Proxy", "Queue", "Mount", "Retries", "Tunnel"):
        setattr(m, name, _Any())
    m.__getattr__ = lambda name: _Any()   # type: ignore[attr-defined]
    m.forward = _Any()
    m.enable_output = _Any()
    m.exception = types.SimpleNamespace(NotFoundError=type("NotFoundError", (Exception,), {}), FunctionTimeoutError=type("FunctionTimeoutError", (Exception,), {}))
    return m


@pytest.fixture(scope="module")
def srv():
    spec = importlib.util.spec_from_file_location("seedrep_served_fp3_under_test", DRIVER)
    mod = importlib.util.module_from_spec(spec)
    prior = sys.modules.get("modal", _ABSENT)
    sys.modules["modal"] = _stub_modal()
    try:
        spec.loader.exec_module(mod)
    finally:
        if prior is _ABSENT:
            sys.modules.pop("modal", None)
        else:
            sys.modules["modal"] = prior
    return mod


def _row(leg: str, children: List[Dict[str, Any]], *, engine: str = "s1") -> Dict[str, Any]:
    return {"standup_id": f"{leg}__{engine}__base__shard00", "leg": leg, "kind": "base", "engine": engine, "view": engine, "stage": "base", "shard": 0,
            "children": children}


def _run_skipping_recorder(argv, **kw):
    """The REAL audit runs (the app's own argv, under this interpreter); the resolved-sampler recorder is stood in for by the fixture's own
    `resolved_sampler.json` (the recorder needs the engine's launch line, which the synthetic vLLM log does not carry)."""
    if any(str(a).endswith("record_resolved_sampler.py") for a in argv):
        return subprocess.CompletedProcess(argv, 0)
    return subprocess.run(argv, **kw)


def _producer(srv, root: fx.Root, row: Dict[str, Any]) -> Dict[str, Any]:
    ac.write_json(root.path / lp.RESOLVED_SAMPLER_REL, {"verdict": "AGREES", "source": "fp3 fixture stand-in"})
    return srv.producer_evidence(root.path, tree=REPO, W=fx.W_FIXTURE, row=row, serve_dir=root.path / "exp" / "serve", driver_python=sys.executable,
                                 env={"PATH": "/usr/bin:/bin", "PYTHONPATH": f"{REPO / 'src'}:{REPO / 'scripts'}"}, run=_run_skipping_recorder)


def _certify(ev: Dict[str, Any], root: fx.Root, row: Dict[str, Any], *, child, table_leg, path: str, text: str | None = None, producer=None):
    aid = ac.read_attempt_root(root.path).attempt_id
    return lp._smoke_check("request_evidence_audit", text if text is not None else Path(path).read_text(), path=path, plan={"W": fx.W_FIXTURE},
                           accepted_attempts=[aid], row=row, aid=aid, root=str(root.path), producer=producer if producer is not None else ev,
                           child=child, table_leg=table_leg)


# --------------------------------------------------------------------------- #
# the misalign row: a HARNESS-named cell dir (the runner's), a TrainTest-labelled plan child                            #
# --------------------------------------------------------------------------- #

def _misalign_root(tmp_path: Path):
    root = fx.make_root(tmp_path / "stores", leg="misalign", view="s1", arms=("none", TT), harness_of=HARNESS_OF)
    items = {"sc0": "leaking", "sc1": "murder", "sc2": "leaking", "sc3": "murder"}
    fx.write_misalign_store(root, arm="none", s=1, items=items)
    fx.write_misalign_store(root, arm=HARNESS, s=1, items=items)            # the REAL runner's cell dir: `cells.json`'s harness id
    root.flush_pipeline(result_rc="0")
    assert (root.path / "exp" / "misalign" / ac.MODEL_KEY / "agentic" / HARNESS).is_dir()
    kids = [{"eval_leg": f"misalign_{b}", "arm": a, "cell": a, "s": 1} for a in ("none", TT) for b in ("leaking", "murder")]
    return root, _row("misalign", kids)


def test_known_bad_misalign_fitted_child_binds_the_harness_named_verdict(srv, tmp_path: Path):
    """Before: the audit wrote `misalign_leaking/si_constv2_mix_full__lora_r4_ce__s1.json` (AGREES) and the app bound
    `misalign_leaking/constv2-mix-full__lora_r4_ce__s1.json` → ABSENT (rc 0!). After: the plan child's cell resolves through the audit's
    index to the file the audit wrote; the record carries the bound rel, both spellings and the checks that bound them."""
    root, row = _misalign_root(tmp_path)
    ev = _producer(srv, root, row)
    log = (root.path / "children" / "request_evidence_audit.log").read_text()
    assert "4 AGREES, 0 REFUSED" in log, log[-800:]
    written = sorted(p.name for p in (root.path / "audits" / "request_log" / "misalign_leaking").iterdir())
    assert written == ["none__s1.json", f"{HARNESS}__s1.json"], written           # the store's spelling is PRESERVED in the filename
    for b in ("leaking", "murder"):
        plan_rel = f"audits/request_log/misalign_{b}/{TT}__s1.json"
        rec = ev[plan_rel]
        assert not rec.get("absent"), rec
        assert rec["bound_rel"] == f"audits/request_log/misalign_{b}/{HARNESS}__s1.json" and rec["rc"] == 0
        assert rec["sha256"] == ac.sha256_file(root.path / rec["bound_rel"])
        assert rec["plan_arm"] == TT and rec["plan_cell"] == TT and rec["store_arm"] == HARNESS
        assert rec["harness_id"] == HARNESS and rec["traintest_id"] == TT and rec["vmodel"] == root.vmodels[TT] and rec["spelling"] == ac.ARM_SPELLING_HARNESS
        assert rec["arm_binding_check"].startswith("agrees")
        none_rec = ev[f"audits/request_log/misalign_{b}/none__s1.json"]
        assert none_rec["bound_rel"] == f"audits/request_log/misalign_{b}/none__s1.json" and none_rec["spelling"] == ac.ARM_SPELLING_BOTH
    # the index the app resolved through is the writer's
    idx = json.loads((root.path / lp.AUDIT_OUT_REL / rea.INDEX_NAME).read_text())
    assert idx["writer"] == lp.PRODUCER_WRITERS["request_evidence_audit"] and len(idx["verdicts"]) == 4
    assert srv.AUDIT_INDEX_NAME == rea.INDEX_NAME


def test_misalign_certificate_accepts_the_bound_verdict_through_the_arm_binding(srv, tmp_path: Path):
    root, row = _misalign_root(tmp_path)
    ev = _producer(srv, root, row)
    srcs = lp.smoke_audit_sources(row, str(root.path), producer=ev)
    assert len(srcs) == 4 and all(Path(p).is_file() for p, _k, _l in srcs), srcs
    assert {Path(p).name for p, k, _l in srcs if k["cell"] == TT} == {f"{HARNESS}__s1.json"}
    assert lp.smoke_evidence_paths(row, str(root.path), producer=ev)["request_evidence_audit"] == [p for p, _k, _l in srcs]
    for path, k, leg in srcs:
        facts = _certify(ev, root, row, child=k, table_leg=leg, path=path)
        assert facts["verdict"] == "AGREES" and facts["leg"] == leg and facts["plan_cell"] == k["cell"] and facts["plan_arm"] == k["arm"]
        assert facts["arm"] == (HARNESS if k["cell"] == TT else "none")           # the verdict's arm stays the store's spelling
        assert facts["arm_binding"]["harness_id"] == (HARNESS if k["cell"] == TT else "none")
    # without the producer map the certificate would still compose the plan's name -- and find no file for the fitted cell
    plan_paths = [p for p, k, _l in lp.smoke_audit_sources(row, str(root.path)) if k["cell"] == TT]
    assert plan_paths and not any(Path(p).exists() for p in plan_paths)


def test_a_verdict_whose_arm_binding_disagrees_refuses_by_name(srv, tmp_path: Path):
    root, row = _misalign_root(tmp_path)
    ev = _producer(srv, root, row)
    plan_rel = f"audits/request_log/misalign_leaking/{TT}__s1.json"
    bound = root.path / ev[plan_rel]["bound_rel"]
    # (a) the app's binder: the verdict file's stamp disagrees with the index (a rewritten verdict)
    doc = json.loads(bound.read_text())
    doc["arm_binding"]["harness_id"] = "si_constv2ctl2_full__psr_ce_all"
    bound.write_text(json.dumps(doc))
    rebound = srv.bind_audit_verdicts(root.path, row=row, W=fx.W_FIXTURE, rc_a=0, aud_present=True)
    rec = rebound[plan_rel]
    assert rec["absent"] is True and "arm_binding" in rec["why"] and "si_constv2ctl2_full__psr_ce_all" in rec["why"] and HARNESS in rec["why"], rec
    # the other verdicts of the row are untouched
    assert not rebound[f"audits/request_log/misalign_murder/{TT}__s1.json"].get("absent")
    # (b) the certificate: a verdict bound to ANOTHER cell than the child's -- refused naming the child's cell, never a looser compare
    fresh_root, fresh_row = _misalign_root(tmp_path / "b")
    ev2 = _producer(srv, fresh_root, fresh_row)
    (path, k, leg), = [(p, k, l) for p, k, l in lp.smoke_audit_sources(fresh_row, str(fresh_root.path), producer=ev2) if k["cell"] == TT and l == "misalign_leaking"]
    doc2 = json.loads(Path(path).read_text())
    doc2["arm_binding"]["traintest_id"] = "constv2ctl2-full__psr_ce_all"
    text2 = json.dumps(doc2)
    producer2 = {**ev2, f"audits/request_log/{leg}/{TT}__s1.json": {**ev2[f"audits/request_log/{leg}/{TT}__s1.json"], "sha256": lp._sha_text(text2)}}
    with pytest.raises(ValueError, match=rf"not this child's \(misalign_leaking, cell '{TT}'"):
        _certify(ev2, fresh_root, fresh_row, child=k, table_leg=leg, path=path, text=text2, producer=producer2)
    # (c) a verdict with NO arm_binding (an audit before F-AN) is refused by name, not compared by string
    doc3 = json.loads(Path(path).read_text())
    doc3.pop("arm_binding")
    text3 = json.dumps(doc3)
    producer3 = {**ev2, f"audits/request_log/{leg}/{TT}__s1.json": {**ev2[f"audits/request_log/{leg}/{TT}__s1.json"], "sha256": lp._sha_text(text3)}}
    with pytest.raises(ValueError, match="carries no arm_binding"):
        _certify(ev2, fresh_root, fresh_row, child=k, table_leg=leg, path=path, text=text3, producer=producer3)


def test_two_verdicts_for_one_child_refuse_as_ambiguous_and_no_index_is_named(srv, tmp_path: Path):
    root, row = _misalign_root(tmp_path)
    ev = _producer(srv, root, row)
    plan_rel = f"audits/request_log/misalign_leaking/{TT}__s1.json"
    idx_p = root.path / lp.AUDIT_OUT_REL / rea.INDEX_NAME
    idx = json.loads(idx_p.read_text())
    (e,) = [x for x in idx["verdicts"] if x["leg"] == "misalign_leaking" and x["traintest_id"] == TT]
    dup_rel = e["rel"].replace(".json", "__dup.json")
    (root.path / lp.AUDIT_OUT_REL / dup_rel).write_bytes((root.path / lp.AUDIT_OUT_REL / e["rel"]).read_bytes())
    idx["verdicts"].append({**e, "rel": dup_rel})
    idx_p.write_text(json.dumps(idx))
    rec = srv.bind_audit_verdicts(root.path, row=row, W=fx.W_FIXTURE, rc_a=0, aud_present=True)[plan_rel]
    assert rec["absent"] is True and "2 verdicts" in rec["why"] and dup_rel in rec["why"], rec
    # no index at all (an audit that refused before writing, or one older than F-AO): the fitted child is ABSENT by name; the `none`
    # child's plan-named file still binds, its cross-check recorded as not available -- never silently passed
    idx_p.unlink()
    rec2 = srv.bind_audit_verdicts(root.path, row=row, W=fx.W_FIXTURE, rc_a=0, aud_present=True)
    assert rec2[plan_rel]["absent"] is True and rea.INDEX_NAME in rec2[plan_rel]["why"]
    none_rec = rec2["audits/request_log/misalign_leaking/none__s1.json"]
    assert "sha256" in none_rec and none_rec["arm_binding_check"].startswith("not_available") and rea.INDEX_NAME in none_rec["arm_binding_check"]


# --------------------------------------------------------------------------- #
# the Arditi row: plan label `ar_c0p3__arditi_allpos`, store cell `arditi_allpos`                                    #
# --------------------------------------------------------------------------- #

def test_arditi_crescendo_child_binds_the_cells_verdict_and_certifies(srv, tmp_path: Path):
    root = fx.make_root(tmp_path / "stores", leg="crescendo", view=ac.ARDITI_VIEW, arms=(ac.ARDITI_CELL,), arditi_pin="a" * 64)
    fx.populate_root(root, s=1, arms=(ac.ARDITI_CELL,), n_items=2)
    row = _row("crescendo", [{"eval_leg": "crescendo", "arm": fx.ARDITI_PLAN_ARM, "cell": ac.ARDITI_CELL, "s": 1}], engine=ac.ARDITI_VIEW)
    ev = _producer(srv, root, row)
    log = (root.path / "children" / "request_evidence_audit.log").read_text()
    assert "1 AGREES, 0 REFUSED" in log, log[-800:]
    plan_rel = f"audits/request_log/crescendo/{fx.ARDITI_PLAN_ARM}__s1.json"
    rec = ev[plan_rel]
    assert not rec.get("absent"), rec
    assert rec["bound_rel"] == f"audits/request_log/crescendo/{ac.ARDITI_CELL}__s1.json" and rec["spelling"] == ac.ARM_SPELLING_BOTH
    assert rec["plan_arm"] == fx.ARDITI_PLAN_ARM and rec["plan_cell"] == ac.ARDITI_CELL == rec["store_arm"] == rec["harness_id"] == rec["traintest_id"]
    (path, k, leg), = lp.smoke_audit_sources(row, str(root.path), producer=ev)
    assert Path(path).name == f"{ac.ARDITI_CELL}__s1.json"
    facts = _certify(ev, root, row, child=k, table_leg=leg, path=path)
    assert facts["verdict"] == "AGREES" and facts["arm"] == ac.ARDITI_CELL and facts["plan_arm"] == fx.ARDITI_PLAN_ARM and facts["plan_cell"] == ac.ARDITI_CELL


# --------------------------------------------------------------------------- #
# the `none` child: the spellings coincide, the plan's name IS the file                                                #
# --------------------------------------------------------------------------- #

def test_none_child_binds_at_the_plans_own_name(srv, tmp_path: Path):
    root = fx.make_root(tmp_path / "stores", leg="pasttense", view="s1", arms=("none",))
    fx.populate_root(root, s=1, arms=("none",), n_items=2)
    row = _row("pasttense", [{"eval_leg": "pasttense", "arm": "none", "cell": "none", "s": 1}])
    ev = _producer(srv, root, row)
    rec = ev["audits/request_log/pasttense/none__s1.json"]
    assert rec["bound_rel"] == "audits/request_log/pasttense/none__s1.json" and rec["spelling"] == ac.ARM_SPELLING_BOTH and "sha256" in rec
    (path, k, leg), = lp.smoke_audit_sources(row, str(root.path), producer=ev)
    assert path == f"{root.path}/audits/request_log/pasttense/none__s1.json"
    assert _certify(ev, root, row, child=k, table_leg=leg, path=path)["verdict"] == "AGREES"
    # a child WITHOUT a `cell` key (an older row shape) resolves by its arm
    row_old = _row("pasttense", [{"eval_leg": "pasttense", "arm": "none", "s": 1}])
    rec_old = srv.bind_audit_verdicts(root.path, row=row_old, W=fx.W_FIXTURE, rc_a=0, aud_present=True)["audits/request_log/pasttense/none__s1.json"]
    assert rec_old["plan_cell"] == "none" and "sha256" in rec_old
