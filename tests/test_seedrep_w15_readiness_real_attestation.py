"""The readiness contract against the REAL native attestation -- the failed S1's own bytes.

S1 (s1), S1b (s2, s3), S2 and S3 on W14 all ended GEOMETRY_MISMATCH after the lora-native engine stood up and attested: the geometry
MATCHED (attested S 45 x L 11,520), and every problem was an EXPECTATION written against a double -- (a) the literal cudagraph mode
`FULL` (the engine reports FULL_AND_PIECEWISE with FULL capture of every size <= S), (b) `residual_slot` looked up inside `adapters[<id>]`
(it is a TOP-LEVEL block), (c) `expected_object_sha256_by_cell` (the endpoint carries the same map as `object_artifact_sha256_by_cell`).
These tests read the real `endpoint.json` / `cells_served.json` / `modal_evidence.json` of one attempt (tests/fixtures/seedrep_f_l/S1,
copied with only its Modal ids -- task, call, image, volume -- replaced by synthetic ones, here attempt
01SYNTHS1TASK0000000000001-111111111111; `proxy_token` is null, the other "token" keys are token COUNTS), reproduce the four
recorded problems with W14's contract loaded from git (the known-bad), and hold W15's contract to zero problems and to each REAL property.
(d) `expected_reused` by smoke role is a separate change on the plan side and is not asserted here.
"""
from __future__ import annotations

import copy
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _seedrep_testlib as T  # noqa: E402
from test_seedrep_w9_container_paths import _stub_modal  # noqa: E402

REPO = T.REPO
DRIVER = REPO / "scripts" / "modal" / "seedrep_served.py"
FIX = REPO / "tests" / "fixtures" / "seedrep_f_l" / "S1"
W14_SHA = "23f34f455"
_ABSENT = object()
sys.path.insert(0, str(REPO / "scripts" / "modal"))
import seedrep_launch_plan as lp  # noqa: E402


def _load_served(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
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


@pytest.fixture(scope="module")
def srv():
    return _load_served(DRIVER, "seedrep_served_w15_readiness")


@pytest.fixture(scope="module")
def srv_w14():
    """The previous served.py (the contract that failed), from git -- the known-bad."""
    r = subprocess.run(["git", "show", f"{W14_SHA}:scripts/modal/seedrep_served.py"], cwd=REPO, capture_output=True, text=True)
    if r.returncode != 0:
        pytest.skip(f"git show {W14_SHA} unavailable here")
    d = Path(tempfile.mkdtemp())
    (d / "seedrep_served.py").write_text(r.stdout)
    (d / "seedrep_launch_plan.py").write_text((REPO / "scripts/modal/seedrep_launch_plan.py").read_text())   # its sibling import
    return _load_served(d / "seedrep_served.py", "seedrep_served_w14_known_bad")


@pytest.fixture(scope="module")
def real():
    ep = json.loads((FIX / "endpoint.json").read_text())
    cells = json.loads((FIX / "cells_served.json").read_text())
    ev = json.loads((FIX / "modal_evidence.json").read_text())
    att = json.loads((FIX / "serving_attestation.json").read_text())
    return ep, cells, ev, att


def _row_for(ep):
    """The S1 row's expectations, rebuilt from the plan's own composers over the real endpoint's pins (the plan pinned exactly these)."""
    digests = ep["object_artifact_sha256_by_cell"]                       # harness id -> sha, what the standup asserted from the app's --expect-object-sha256
    objects = {c: {"object_sha256": digests[lp.harness_id(c)], "staged_id": c} for c in lp.FITTED_CELLS if lp.harness_id(c) in digests}
    assert len(objects) == len(digests) == 5
    geom = {"L": ep["attested_max_model_len"], "S": ep["attested_max_num_seqs"], "expected_adjustment": ep["geometry"]["adjustments"][0],
            "expected_window_source": ep["geometry"]["window_source"], "thinking": False}
    readiness = lp.readiness_block(geom, max_lora_rank=ep["lora_config"]["max_lora_rank"], max_loras=ep["lora_config"]["max_loras"],
                                   residual_slot_armed=[c for c in objects if lp.is_psr_cell(c)], max_logprobs=None, thinking=False)
    readiness["max_logprobs"] = None
    return {"engine": "s1", "geometry": geom, "readiness": readiness, "objects": objects, "adapters": lp.adapter_expectations(objects, {}),
            "cells_served_expected": lp.expected_cells_served("s1"), "route_set": list(objects) + ["none"] + list(lp.PROMPTED_CELLS)}


class TestKnownBadOnTheRealBytes:
    def test_w14s_contract_reproduces_exactly_the_four_recorded_problems(self, srv_w14, real):
        ep, cells, ev, _ = real
        got = srv_w14.readiness_problems(ep, cells, _row_for(ep))
        assert got == ev["readiness"]["problems"], (got, ev["readiness"]["problems"])
        assert len(got) == 4 and got[0].startswith("attested_cudagraph_mode 'FULL_AND_PIECEWISE' != FULL")

    def test_w15s_contract_accepts_the_real_attestation(self, srv, real):
        ep, cells, _, att = real
        assert srv.readiness_problems(ep, cells, _row_for(ep)) == []                      # endpoint.json alone (its residual-slot SUMMARY)
        assert srv.readiness_problems(ep, cells, _row_for(ep), attestation=att) == []     # with the workers' own reports beside it


class TestEachRealProperty:
    """Every check flips on the REAL bytes when the property it names is broken -- the contract stayed exact, it just reads the real schema."""

    def _probs(self, srv, real, mutate):
        ep, cells, _, _ = real
        ep2 = copy.deepcopy(ep)
        mutate(ep2)
        return srv.readiness_problems(ep2, cells, _row_for(ep))

    def _probs_att(self, srv, real, mutate):
        ep, cells, _, att = real
        att2 = copy.deepcopy(att)
        mutate(att2)
        return srv.readiness_problems(ep, cells, _row_for(ep), attestation=att2)

    def test_a_cudagraph_mode_without_a_full_surface_refuses(self, srv, real):
        for mode in ("PIECEWISE", "NONE", None):
            probs = self._probs(srv, real, lambda e: e.__setitem__("attested_cudagraph_mode", mode))
            assert any("has no FULL capture surface" in p for p in probs), (mode, probs)
        assert srv.CUDAGRAPH_MODES_WITH_FULL_CAPTURE == ("FULL", "FULL_AND_PIECEWISE")

    def test_a_the_full_surface_is_the_cores_own_rule(self, srv, real):
        """serving_attestation.cudagraph_coverage_problems: max(sizes) == full_capture_max >= S; the real S1 has 45 == 45 >= 45."""
        ep, _, _, _ = real
        assert max(ep["attested_cudagraph_full_capture_sizes"]) == ep["attested_cudagraph_full_capture_max"] == ep["attested_max_num_seqs"] == 45
        assert any("FULL capture max 44 < S 45" in p for p in self._probs(srv, real, lambda e: e.__setitem__("attested_cudagraph_full_capture_max", 44)))
        # a max claiming 48 while the dispatcher's FULL keys stop at 45 is surface drift, refused by name
        assert any("sizes max 45 != full_capture_max 48" in p for p in self._probs(srv, real, lambda e: e.__setitem__("attested_cudagraph_full_capture_max", 48)))
        assert any("attested_cudagraph_full_capture_sizes ABSENT" in p for p in self._probs(srv, real, lambda e: e.pop("attested_cudagraph_full_capture_sizes")))
        assert any("attested_cudagraph_full_capture_max ABSENT" in p for p in self._probs(srv, real, lambda e: e.pop("attested_cudagraph_full_capture_max")))

    def test_b_the_residual_slot_is_read_as_the_top_level_block(self, srv, real):
        ep, _, _, att = real
        psr = "si_constv2_mix_full__psr_ce_all"
        # endpoint.json: the SUMMARY {armed, cells}; never a key inside adapters[<id>] (the known-bad looked there and found nothing)
        assert ep["residual_slot"]["armed"] is True and sorted(ep["residual_slot"]) == ["armed", "cells"] and "residual_slot" not in next(iter(ep["adapters"].values()))
        assert any("residual_slot ABSENT" in p for p in self._probs(srv, real, lambda e: e.pop("residual_slot")))
        assert any("residual_slot.armed False != True" in p for p in self._probs(srv, real, lambda e: e["residual_slot"].__setitem__("armed", False)))
        assert any(f"residual_slot.cells has no entry for constv2-mix-full__psr_ce_all ({psr})" in p
                   for p in self._probs(srv, real, lambda e: e["residual_slot"]["cells"].pop(psr)))
        def null_payload(e):
            e["residual_slot"]["cells"][psr].update({"kind": None, "residual_fingerprint": None})
        assert any(f"residual_slot.cells[{psr}] carries no residual payload" in p for p in self._probs(srv, real, null_payload))
        # serving_attestation.json: the full block -- the workers' reports and the per-adapter expected payload
        assert sorted(att["residual_slot"]) == ["armed", "expected", "n_workers", "rpc_method", "worker_reports"]
        assert any("serving_attestation.residual_slot ABSENT" in p for p in self._probs_att(srv, real, lambda a: a.pop("residual_slot")))
        assert any("worker(s) not armed" in p for p in self._probs_att(srv, real, lambda a: a["residual_slot"]["worker_reports"][0].__setitem__("armed", False)))
        assert any("worker_reports ABSENT or empty" in p for p in self._probs_att(srv, real, lambda a: a["residual_slot"].__setitem__("worker_reports", [])))
        assert any(f"expected.adapters has no entry for constv2-mix-full__psr_ce_all ({psr})" in p
                   for p in self._probs_att(srv, real, lambda a: a["residual_slot"]["expected"]["adapters"].pop(psr)))
        def null_att(a):
            a["residual_slot"]["expected"]["adapters"][psr].update({"kind": None, "op": None, "residual_fingerprint": None})
        assert any(f"expected.adapters[{psr}] carries no residual payload" in p for p in self._probs_att(srv, real, null_att))

    def _probs_cells(self, srv, real, mutate):
        ep, cells, _, _ = real
        c2 = copy.deepcopy(cells)
        mutate(c2)
        return srv.readiness_problems(ep, c2, _row_for(ep))

    def test_c_the_object_digest_map_is_read_where_the_writer_puts_it(self, srv, real):
        ep, cells, _, _ = real
        # the writer: the pins the standup was TOLD in cells_served.json, the artifacts it LOADED in endpoint.json -- never the other way
        assert "expected_object_sha256_by_cell" not in ep and isinstance(cells["expected_object_sha256_by_cell"], dict)
        assert cells["expected_object_sha256_by_cell"] == ep["object_artifact_sha256_by_cell"] and cells["expected_digests_source"] == "cli"
        assert any("cells_served.expected_object_sha256_by_cell ABSENT" in p for p in self._probs_cells(srv, real, lambda c: c.pop("expected_object_sha256_by_cell")))
        assert any("endpoint.object_artifact_sha256_by_cell ABSENT" in p for p in self._probs(srv, real, lambda e: e.pop("object_artifact_sha256_by_cell")))
        def loaded_other(e):
            e["object_artifact_sha256_by_cell"]["si_constv2_mix_full__lora_r4_ce"] = "f" * 64
        assert any("!= cells_served.expected_object_sha256_by_cell" in p for p in self._probs(srv, real, loaded_other))
        def told_other(c):
            c["expected_object_sha256_by_cell"]["si_constv2_mix_full__lora_r4_ce"] = "f" * 64
        probs = self._probs_cells(srv, real, told_other)
        assert any("expected_object_sha256_by_cell[si_constv2_mix_full__lora_r4_ce] 'ffff" in p for p in probs)
        def drop_one(c):
            c["expected_object_sha256_by_cell"].pop("si_constv2ctl2_full__psr_ce_all")
        assert any("has no entry for si_constv2ctl2_full__psr_ce_all" in p for p in self._probs_cells(srv, real, drop_one))

    def test_l1_the_failure_status_names_the_cause(self, srv, real):
        ep, cells, ev, _ = real
        # the four problems were a CONTRACT mismatch, filed as GEOMETRY_MISMATCH by an "attested" substring
        assert srv.readiness_failure_status(ev["readiness"]["problems"]) == "READINESS_CONTRACT"
        assert srv.readiness_failure_status(["attested_max_num_seqs 44 != 45 (= min(identity, 524,287 // L))"]) == "GEOMETRY_MISMATCH"
        assert srv.readiness_failure_status(["kernel adjustment [] != expected {}"]) == "GEOMETRY_MISMATCH"
        assert srv.readiness_failure_status(["adapters_reused ABSENT (a mandated attestation: the adapter-reuse map, gate 10)"]) == "READINESS_CONTRACT"
        import inspect
        src = inspect.getsource(srv._run_standup_row)
        assert "return fail(readiness_failure_status(problems)" in src and '"attested" in p' not in src

    def test_l2_a_missing_cells_served_record_is_refused_by_name(self, srv):
        import inspect
        src = inspect.getsource(srv._run_standup_row)
        i = src.index('if not (serve_dir / "cells_served.json").is_file():')
        block = src[i:i + 900]
        assert "ABSENT (the standup wrote endpoint.json but no cells_served.json" in block and "return fail(READINESS_CONTRACT" in block and "else {}" not in block
        assert 'cells_served = json.loads((serve_dir / "cells_served.json").read_text())\n' in src and 'if (serve_dir / "cells_served.json").exists() else {}' not in src

    def test_the_exact_checks_that_already_held_still_hold(self, srv, real):
        assert any("attested_max_num_seqs 44 != 45" in p for p in self._probs(srv, real, lambda e: e.__setitem__("attested_max_num_seqs", 44)))
        assert any("attested_max_model_len 11519 != 11520" in p for p in self._probs(srv, real, lambda e: e.__setitem__("attested_max_model_len", 11519)))


class TestSelectLaunchRowsWrapsSmokeRowRefusals:
    def test_a_smoke_row_value_error_is_a_launch_refusal_by_name(self, srv, monkeypatch):
        import inspect
        src = inspect.getsource(srv.select_launch_rows)
        i = src.index("picked = [lp.smoke_row(")
        assert "except ValueError as e:" in src[i:i + 400] and "raise LaunchRefused(" in src[i:i + 600]
        plan = {"materialisation": "draft", "standups": [{"standup_id": "pasttense__s1__base__shard00", "kind": "base", "leg": "pasttense", "children": []}],
                "rungs": [], "s5_probe_rows": []}
        def boom(*a, **k):
            raise ValueError("smoke id 'S9' has no declared adapter role (SMOKE_ADAPTER_ROLES)")
        monkeypatch.setattr(srv.lp, "smoke_row", boom)
        monkeypatch.setattr(srv.lp, "smoke_identity", lambda plan_d, sid, wu: {"smoke_id": sid, "wave_uuid": wu, "W7": "2222222", "W": "2" * 40, "dicts": {}, "results_root": "/r", "ladder_root": "/l"})
        sig = inspect.signature(srv.select_launch_rows)
        kwargs = {k: v.default for k, v in sig.parameters.items() if v.default is not inspect.Parameter.empty}
        kwargs.update({p_: None for p_ in ("pool_urls",) if p_ in sig.parameters})
        with pytest.raises(srv.LaunchRefused, match=r"--smoke 'pasttense__s1__base__shard00' --smoke-id 'S9': smoke id 'S9' has no declared adapter role"):
            srv.select_launch_rows(plan, **{**kwargs, **{k: v for k, v in dict(go="", legs="", smoke="pasttense__s1__base__shard00", smoke_id="S9").items() if k in sig.parameters}})


def test_teardown_standup_is_defined_before_every_call_in_the_standup_row():
    """The missing-cells_served branch called `teardown_standup()` before its local definition -> UnboundLocalError
    instead of the READINESS_CONTRACT refusal. Executable ordering pin: inside `_run_standup_row`, the `def` precedes every call."""
    import ast as _ast
    src = (REPO / "scripts" / "modal" / "seedrep_served.py").read_text()
    tree = _ast.parse(src)
    fn = next(n for n in _ast.walk(tree) if isinstance(n, _ast.FunctionDef) and n.name == "_run_standup_row")
    def_line = next(n.lineno for n in _ast.walk(fn) if isinstance(n, _ast.FunctionDef) and n.name == "teardown_standup")
    calls = [n.lineno for n in _ast.walk(fn) if isinstance(n, _ast.Call) and isinstance(n.func, _ast.Name) and n.func.id == "teardown_standup"]
    assert calls, "no teardown_standup() call in _run_standup_row?"
    early = [ln for ln in calls if ln < def_line]
    assert not early, f"teardown_standup() called at {early} before its definition at line {def_line}"
