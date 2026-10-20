"""The review findings applied, plus three additions (the tunnel-record fields, the multi-turn window provenance rule, the dose
standups + the S5 probe): ONE block per finding, each with the RE-VALIDATED repro as its known-bad twin. Peer modules
(`seedrep_ladder.py` / `rung_redraw.py`, `seedrep_petri_drive.py`) are imported BY PATH from the tree first, else from their
sibling checkouts, and the tests that need them SKIP by name when neither is present (at W they are in the tree)."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import re
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for p in (REPO / "src", REPO / "scripts", REPO / "scripts" / "modal"):
    sys.path.insert(0, str(p))
import seedrep_launch_plan as lp  # noqa: E402
import seedrep_watch as sw  # noqa: E402

DRIVER = REPO / "scripts" / "modal" / "seedrep_served.py"
WORKTREES = REPO.parent
WS2E = WORKTREES / "seedrep-ws2e"
WS2F = WORKTREES / "seedrep-ws2f"
from _env import host_site_path  # noqa: E402
OPERATOR_DIR = host_site_path("$STAGE/sdfbase/seedrep_battery")      # the operator tree on the grading host (absent elsewhere: skips)
W = "22222229c4fd67577dfc6df5e75fc5dd63d2882b"   # the "2222222" prefix is kept (the module's paths, app names and six-axis keys spell it); the tail is synth.code_sha("a")[:33], since a placeholder run code_sha (forty identical digits) is refused
WAVE = "22222222-3333-4444-8444-555555555555"
SMOKE_WAVE = "11111111-2222-4333-8444-555555555555"


# --------------------------------------------------------------------------- the stub modal + the app under test

class _Chain:
    def __getattr__(self, _name):
        return lambda *a, **k: self

    def __call__(self, *a, **k):
        return self


class _App:
    def __init__(self, name=None, **k):
        self.description = None

    def function(self, **kw):
        return lambda f: f

    def local_entrypoint(self, **kw):
        return lambda f: f


class FakeDict:
    """modal.Dict's ONE atomic primitive: put(key, value, skip_if_exists=True) -> bool; get; pop; also the KV protocol."""

    store: dict = {}

    def __init__(self, name):
        self.name = name
        FakeDict.store.setdefault(name, {})

    @classmethod
    def from_name(cls, name, create_if_missing=False):
        return cls(name)

    def put(self, key, value, skip_if_exists=False):
        d = FakeDict.store[self.name]
        if skip_if_exists and key in d:
            return False
        d[key] = value
        return True

    def get(self, key, default=None):
        return FakeDict.store[self.name].get(key, default)

    def pop(self, key, default=None):
        return FakeDict.store[self.name].pop(key, default)

    def contains(self, key):
        return key in FakeDict.store[self.name]

    def items(self):
        return iter(dict(FakeDict.store[self.name]).items())

    def __setitem__(self, key, value):
        FakeDict.store[self.name][key] = value


class FakeVOL:
    files: dict = {}                       # volume-relative path -> text: what the Volume API serves (delta 4 (1): the ONLY evidence source)

    def __init__(self):
        self.commits = 0
        self.fail_at = None

    def read_file(self, rel):
        rel = str(rel).lstrip("/")
        if rel not in FakeVOL.files:
            raise FileNotFoundError(rel)
        yield FakeVOL.files[rel].encode()

    def commit(self):
        self.commits += 1
        if self.fail_at is not None and self.commits >= self.fail_at:
            raise RuntimeError(f"volume commit failed (commit #{self.commits})")

    def reload(self):
        pass


def _stub_modal():
    m = types.ModuleType("modal")
    m.App = _App
    m.Volume = types.SimpleNamespace(from_name=lambda *a, **k: FakeVOL())
    m.Image = types.SimpleNamespace(from_registry=lambda *a, **k: _Chain(), debian_slim=lambda *a, **k: _Chain())
    m.Dict = FakeDict
    m.is_local = lambda: False
    m.current_function_call_id = lambda: "fc-test"
    m.forward = None
    m.FunctionCall = None
    return m


_ABSENT = object()
#: `controls_queue_record` requires the six frozen sampler keys on `redraw.sampling`
SIX_SAMPLER_KEYS = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5, "repetition_penalty": 1.0}


@pytest.fixture(scope="module")
def srv():
    spec = importlib.util.spec_from_file_location("seedrep_served_review_fixes", DRIVER)
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
def inputs():
    # the config private_runs: this tree's, then the sibling checkout for the Arditi lane configs (absent on this tree until W)
    return lp.load_inputs(REPO, config_roots=(WORKTREES / "seedrep-ws1dp",))


def _plan(inputs, **kw):
    base = dict(W=W, wave_uuid=WAVE, reqs_sha256=inputs["reqs_sha256"], roster=inputs["roster"], pins=inputs["pins"],
                arditi_pins=inputs["arditi_pins"], arms=inputs["arms"], materialisation="draft", generated_at="2026-09-18T00:00:00Z",
                config_private_runs=inputs.get("config_private_runs"))
    base.update(kw)
    return lp.build_plan(**base)


@pytest.fixture(scope="module")
def draft(inputs):
    return _plan(inputs)


def _row(plan, leg, engine="s1", stage="base", shard=0):
    return [r for r in plan["standups"] + plan["rungs"] if r["leg"] == leg and r["engine"] == engine and r["stage"] == stage and r["shard"] == shard][0]


def _audit_qualification(table_leg):
    """The audit stamp on every verdict (`request_evidence_audit.audit_root`): ledger-backed (`/1`) where the plan's children declare
    a ledger (the GSM8K children), else `not_on_path` under the non-ledger contract -- the certificate refuses a verdict without it."""
    if table_leg == "gsm8k":
        return {"reconciliation_contract": lp.AUDIT_CONTRACT_LEDGER, "ledger": "declared", "ledger_source": "plan: children[].ledger.ledger_path_rel",
                "attempt_source": "ledger", "evidence_basis": ["request_evidence.jsonl", "usage_ledger.jsonl", "response_evidence.jsonl", "engine log (Received request)"],
                "body_digest_check": "usage ledger body_sha256 == client record body_sha256", "independence_note": None}
    return {"reconciliation_contract": lp.AUDIT_CONTRACT_LEDGERLESS, "ledger": "not_on_path", "ledger_source": "plan: no child of this row declares a ledger path; none on the volume",
            "attempt_source": "response_evidence", "evidence_basis": ["request_evidence.jsonl", "response_evidence.jsonl", "engine log (Received request)"],
            "body_digest_check": "not_available: the ledger is not on this row's path, so the client's body digest has no independent counterpart (D106)",
            "independence_note": "D106: the engine log is the only server-side record; rejected / pre-engine attempts are corroborated by the client's response hook alone, never independently"}


def _load_peer(name: str, rel: str, worktree: Path):
    """A peer module BY PATH: the tree at W first (it holds every lane's files), else the lane worktree; None when absent."""
    for root in (REPO, worktree):
        p = root / rel
        if p.is_file():
            mod = sys.modules.get(name)
            if mod is not None and Path(getattr(mod, "__file__", "")).resolve() == p.resolve():
                return mod
            spec = importlib.util.spec_from_file_location(name, p)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[name] = mod
            spec.loader.exec_module(mod)
            return mod
    return None


def _request_evidence_client():
    """`sisteer.eval.request_evidence_client` as a REAL module under its package name (`_evidence_module` does
    `from sisteer.eval import request_evidence_client`): the tree's when present, else the seedrep-ws1dp worktree's loaded by path."""
    try:
        from sisteer.eval import request_evidence_client as REC
        return REC
    except ImportError:
        pass
    p = WORKTREES / "seedrep-ws1dp/src/sisteer/eval/request_evidence_client.py"
    if not p.is_file():
        pytest.skip("WS1-D′'s request_evidence_client is neither on this tree nor in the seedrep-ws1dp worktree")
    import sisteer.eval as pkg
    spec = importlib.util.spec_from_file_location("sisteer.eval.request_evidence_client", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["sisteer.eval.request_evidence_client"] = mod
    spec.loader.exec_module(mod)
    pkg.request_evidence_client = mod
    return mod


def _misalign_runner():
    """The misalign runner WITH the `--seed-policy` / `--seed-base` flags (the tree's when it carries them, else the seedrep-ws1c
    worktree's, by path); None when neither does."""
    for root in (REPO, WORKTREES / "seedrep-ws1c"):
        p = root / "src/sisteer/capeval/misalign/runner.py"
        if p.is_file() and '"--seed-policy"' in p.read_text():
            spec = importlib.util.spec_from_file_location("misalign_runner_under_test", p)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    return None


@pytest.fixture(scope="module")
def L():
    mod = _load_peer("seedrep_ladder", "scripts/modal/seedrep_ladder.py", WS2E)
    if mod is None:
        pytest.skip("WS2-E's scripts/modal/seedrep_ladder.py is neither in this tree nor in the seedrep-ws2e worktree")
    return mod


@pytest.fixture(scope="module")
def RR(L):
    mod = _load_peer("seedrep_rung_redraw", "scripts/sdfbase/seedrep/rung_redraw.py", WS2E)
    if mod is None:
        pytest.skip("WS2-E's rung_redraw.py is absent")
    return mod


@pytest.fixture(scope="module")
def drive():
    mod = _load_peer("seedrep_petri_drive_under_test", "scripts/modal/seedrep_petri_drive.py", WS2F)
    if mod is None:
        pytest.skip("WS2-F's scripts/modal/seedrep_petri_drive.py is absent")
    return mod


def _native_cells_served(row):
    """The NATIVE record's shape (the real lens-era record's fields with lora_native ids): canonical rows in HARNESS ids with
    cell_class / virtual_model_name, and the DERIVED zero-adapter control under `controls`."""
    def cls(c):
        return "base" if c == "none" else "prompted" if c.startswith("prompted__") else "class_a"
    canonical = [{"cell_id": lp.harness_id(c), "cell_class": cls(c), "kind": ("none" if c == "none" else "lora_all"),
                  "si_id": (None if cls(c) != "prompted" else lp.harness_id(c).split("__", 1)[1]), "object_path": None,
                  "virtual_model_name": f"petri__{lp.harness_id(c)}", "serving_identity": "vllm_lora_native", "intervention_hash": "0" * 16,
                  "harness_config_hash": "1" * 16, "petri_run_key": "2" * 16, "control_of": None, "si_prefix_hash": None}
                 for c in row["route_set"]]
    controls = [{"cell_id": lp.NONE_LORA0, "cell_class": "control", "kind": "none", "si_id": None, "object_path": None,
                 "virtual_model_name": "petri__none__lora0", "serving_identity": "vllm_lora_native", "intervention_hash": "0" * 16,
                 "harness_config_hash": "1" * 16, "petri_run_key": "3" * 16, "control_of": "none", "si_prefix_hash": None}]
    return {"engine": "lora_native", "canonical": canonical, "controls": controls,
            "expected_object_sha256_by_cell": {lp.harness_id(c): o["object_sha256"] for c, o in row["objects"].items()}}   # where the writer puts the pins


def _native_endpoint(row, **over):
    g = row["geometry"]
    h = lambda c: c if c == lp.NONE_LORA0 else lp.harness_id(c)  # noqa: E731
    ep = {"engine": "lora_native", "proxy_url": "http://127.0.0.1:9000/v1", "proxy_port": 9000, "vllm_port": 9101,
          "attested_max_model_len": g["L"], "attested_max_num_seqs": g["S"],
          "geometry": {"window_source": g["expected_window_source"], "adjustments": [g["expected_adjustment"]] if g["expected_adjustment"] else []},
          "attested_cudagraph_mode": "FULL_AND_PIECEWISE", "attested_cudagraph_full_capture_max": g["S"],
          "attested_cudagraph_full_capture_sizes": sorted({x for x in (1, 2, 4, 8, 16, 24, 32, 40) if x < g["S"]} | {g["S"]}),   # vLLM: FULL sizes <= S only
          "lora_config": {"max_lora_rank": row["readiness"]["max_lora_rank"], "max_loras": row["readiness"]["max_loras"]},
          "max_logprobs": row["readiness"]["max_logprobs"], "thinking_mode": row["readiness"]["thinking_mode"],
          "server_default_policy": "contract_override", "adapters_reused": {h(c): True for c in row["adapters"]},
          "object_artifact_sha256_by_cell": {h(c): o["object_sha256"] for c, o in row["objects"].items()},   # the artifacts the standup LOADED
          "adapters": {h(c): {"name": h(c), "r": 1, "petri_source_kind": "psr"} for c in row["readiness"]["residual_slot_armed"]},
          "residual_slot": {"armed": True, "cells": {h(c): {"kind": "psr", "position_mode": "all", "residual_fingerprint": "ab" * 32}
                                                  for c in row["readiness"]["residual_slot_armed"]}},
          "served_models": [f"petri__{h(c)}" for c in row["route_set"]] + ["petri__none__lora0"]}
    ep.update(over)
    return ep


# =========================================================================== P1-1: a THROWING publication barrier

class TestP1_1_publication_barrier:
    def test_revalidated_the_old_callback_swallowed_every_error(self, srv):
        # `_sync_all` (kept for the PERIODIC sync) is best-effort by design: a failing commit prints a warning and returns
        vol = FakeVOL(); vol.fail_at = 1
        srv.VOL = vol
        srv._sync_all([], __import__("threading").Lock(), "periodic")            # no raise: the repro's behaviour, now periodic-only
        # ... which is exactly why it can no longer be the ACCEPTANCE path's commit: the barrier raises
        barrier = srv.make_publication_barrier([], __import__("threading").Lock())
        with pytest.raises(srv.PublicationFailed, match="volume commit failed"):
            barrier()

    def test_a_failed_publication_before_acceptance_leaves_no_accepted_entry(self, srv, tmp_path):
        FakeDict.store.clear()
        acc = FakeDict("accepted")
        root = tmp_path / "attempt"
        root.mkdir()
        srv.write_json(root / srv.ATTEMPT_JSON, {"attempt_id": "a-1"})
        dest = tmp_path / "vol" / "attempt"
        vol = FakeVOL(); vol.fail_at = 2                       # COMPLETE.json publishes, the inventory's commit fails
        srv.VOL = vol
        barrier = srv.make_publication_barrier([srv.TreeSyncer(root, dest)], __import__("threading").Lock())
        with pytest.raises(srv.PublicationFailed):
            srv.finish_attempt(root=root, key="k", attempt_id="a-1", inventory=[], accepted=acc, commit=barrier, summary={})
        assert acc.get("k") is None, "the acceptance must never follow a failed publication"
        assert (dest / srv.COMPLETE_JSON).is_file(), "the first barrier DID publish COMPLETE.json before the failure"
        assert not (dest / srv.INVENTORY_JSON).is_file() or vol.commits == 2
        # the same attempt with a healthy volume completes: exactly one accepted entry
        vol.fail_at = None
        fin = srv.finish_attempt(root=root, key="k", attempt_id="a-1", inventory=[], accepted=acc, commit=barrier, summary={})
        assert fin["state"] == "accepted" and acc.get("k") == "a-1" and (dest / srv.INVENTORY_JSON).is_file()

    def test_an_unwritable_destination_raises_through_the_production_callback(self, srv, tmp_path):
        root = tmp_path / "attempt"; root.mkdir(); (root / "x.txt").write_text("x")
        blocker = tmp_path / "vol"; blocker.write_text("not a directory")
        srv.VOL = FakeVOL()
        barrier = srv.make_publication_barrier([srv.TreeSyncer(root, blocker / "attempt")], __import__("threading").Lock())
        with pytest.raises(srv.PublicationFailed):
            barrier()


# =========================================================================== P1-2: internal vs harness ids at every boundary

class TestP1_2_namespaces:
    def test_table_equals_the_real_harness_translation_and_refuses_unmapped(self):
        from sisteer.capeval.petri import registry_adapter as ra
        for tt, h in lp.HARNESS_ID_OF.items():
            assert ra.harness_cell_id(tt) == h, tt
            assert ra.traintest_cell_id(h) == tt and lp.traintest_id(h) == tt
        assert lp.harness_id("prompted__si_constitution") == "prompted__constitution"
        assert lp.harness_id("prompted__safety_refusal") == "prompted__safety"
        with pytest.raises(ValueError, match="no harness translation"):
            lp.harness_id("prompted__constitution")               # a harness id is not a TrainTest id
        with pytest.raises(ValueError, match="no TrainTest translation"):
            lp.traintest_id(lp.NONE_LORA0)                        # the derived control has no plan spelling
        with pytest.raises(ra.RegistryAdapterError):
            ra.harness_cell_id(lp.NONE_LORA0)

    def test_revalidated_a_native_record_failed_the_old_traintest_comparison_and_passes_now(self, srv, draft):
        row = _row(draft, "pasttense")
        cells, ep = _native_cells_served(row), _native_endpoint(row)
        assert [c["cell_id"] for c in cells["canonical"]] == row["route_set_harness"]
        assert "prompted__constitution" in row["route_set_harness"] and "prompted__si_constitution" in row["route_set"]
        assert row["cells_served_expected"] == {"canonical": row["route_set_harness"], "controls": ["none__lora0"]}
        assert srv.readiness_problems(ep, cells, row) == []
        # the repro: the OLD rule compared TrainTest ids to this record -> every real standup failed
        tt_spelled = sorted(c["cell_id"] for c in cells["canonical"]) + ["none__lora0"]
        assert sorted(tt_spelled) != sorted(row["route_set"])
        # known-bads, each NAMED: the control missing; a TrainTest-spelled record; a moved pin under its harness id
        no_control = {**cells, "controls": []}
        assert any("cells_served.controls [] != ['none__lora0']" in p for p in srv.readiness_problems(ep, no_control, row))
        tt_record = {"canonical": [{"cell_id": c, "cell_class": "class_a"} for c in row["route_set"]], "controls": cells["controls"]}
        assert any("cells_served.canonical" in p and "harness ids" in p for p in srv.readiness_problems(ep, tt_record, row))
        cells2 = {**cells, "expected_object_sha256_by_cell": {**cells["expected_object_sha256_by_cell"], "si_constv2_mix_full__lora_r4_ce": "f" * 64}}
        assert any("expected_object_sha256_by_cell[si_constv2_mix_full__lora_r4_ce]" in p for p in srv.readiness_problems(ep, cells2, row))
        ep3 = _native_endpoint(row, adapters_reused={**ep["adapters_reused"], "si_constv2ctl2_full__psr_ce_all": False})
        assert any("constv2ctl2-full__psr_ce_all (si_constv2ctl2_full__psr_ce_all) was EXPORTED" in p for p in srv.readiness_problems(ep3, cells, row))
        wrong_class = {**cells, "controls": [{**cells["controls"][0], "cell_class": "class_a"}]}
        assert any("not cell_class control" in p for p in srv.readiness_problems(ep, wrong_class, row))
        # the Arditi engine: {none, arditi_allpos} + the control
        ar = _row(draft, "bon", "arditi")
        assert srv.readiness_problems(_native_endpoint(ar), _native_cells_served(ar), ar) == []
        assert ar["cells_served_expected"] == {"canonical": ["none", "arditi_allpos"], "controls": ["none__lora0"]}

    def test_revalidated_every_mandated_attestation_is_required_absence_fails(self, srv, draft):
        """The full-capture bound, `lora_config`, the server-default policy, the adapter-reuse map, the residual-slot
        record and the object-digest map all PASSED when absent (each check ran only when the field was present). Every one is now
        required with exact contents; absence -- of the field, or of one expected entry inside it -- is a named readiness failure."""
        row = _row(draft, "controls", "s1")
        ep, cells = _native_endpoint(row), _native_cells_served(row)
        assert srv.readiness_problems(ep, cells, row) == []
        for field, needle in (("attested_cudagraph_full_capture_max", "attested_cudagraph_full_capture_max ABSENT"),
                              ("lora_config", "lora_config ABSENT"), ("server_default_policy", "server_default_policy ABSENT"),
                              ("adapters_reused", "adapters_reused ABSENT"), ("object_artifact_sha256_by_cell", "endpoint.object_artifact_sha256_by_cell ABSENT"),
                              ("residual_slot", "residual_slot ABSENT")):
            absent = {k: v for k, v in ep.items() if k != field}
            probs = srv.readiness_problems(absent, cells, row)
            assert any(needle in p for p in probs), (field, probs)
            assert len(probs) >= 1
        # one expected ENTRY missing inside a present map is also named
        h = lp.harness_id("constv2-mix-full__lora_r4_ce")
        probs = srv.readiness_problems({**ep, "adapters_reused": {k: v for k, v in ep["adapters_reused"].items() if k != h}}, cells, row)
        assert any(f"adapters_reused has no entry for constv2-mix-full__lora_r4_ce ({h})" in p for p in probs)
        probs = srv.readiness_problems(ep, {**cells, "expected_object_sha256_by_cell": {k: v for k, v in cells["expected_object_sha256_by_cell"].items() if k != h}}, row)
        assert any(f"expected_object_sha256_by_cell has no entry for {h}" in p for p in probs)
        psr = lp.harness_id("constv2-mix-full__psr_ce_all")
        rs = ep["residual_slot"]
        probs = srv.readiness_problems({**ep, "residual_slot": {**rs, "cells": {k: v for k, v in rs["cells"].items() if k != psr}}}, cells, row)
        assert any(f"residual_slot.cells has no entry for constv2-mix-full__psr_ce_all ({psr})" in p for p in probs)
        probs = srv.readiness_problems({**ep, "residual_slot": {**rs, "armed": False}}, cells, row)
        assert any("not armed" in p for p in probs)
        # exact contents still hold on present fields
        assert any("lora_config" in p and "!= rank" in p for p in srv.readiness_problems({**ep, "lora_config": {"max_lora_rank": 4, "max_loras": 6}}, cells, row))
        assert any("server_default_policy 'default'" in p for p in srv.readiness_problems({**ep, "server_default_policy": "default"}, cells, row))
        assert any("FULL capture max 3 < S" in p for p in srv.readiness_problems({**ep, "attested_cudagraph_full_capture_max": 3}, cells, row))   # the core's coverage rule
        # the Arditi engine (a vector cell) mandates its residual-slot record too
        ar = _row(draft, "bon", "arditi")
        assert srv.readiness_problems(_native_endpoint(ar), _native_cells_served(ar), ar) == []
        assert any("residual_slot ABSENT" in p for p in srv.readiness_problems({k: v for k, v in _native_endpoint(ar).items() if k != "residual_slot"}, _native_cells_served(ar), ar))

    def test_misalign_only_cells_speaks_cells_json_namespace(self, draft):
        for r in draft["standups"] + draft["rungs"]:
            if r["leg"] != "misalign":
                continue
            for k in r["children"]:
                a = k["argv"]
                assert a[a.index("--only-cells") + 1] == lp.harness_id(k["cell"]) == k["cell_harness"]
                assert a[a.index("--defense") + 1] if "--defense" in a else True
        bad = copy.deepcopy(draft)
        k = next(k for k in _row(bad, "misalign")["children"] if lp.harness_id(k["cell"]) != k["cell"])   # `none` spells the same in both namespaces
        k["argv"][k["argv"].index("--only-cells") + 1] = k["cell"]      # the TrainTest spelling -> the runner would refuse it as unknown
        assert any("--only-cells must be the HARNESS id" in p for p in lp.validate_plan(bad))


# =========================================================================== P1-3: the lane-owned bank prefix (operator dir)

class TestP1_3_lane_bank_prefix:
    def test_write_destinations_outside_the_lane_prefix_refuse(self, tmp_path):
        lc_path = OPERATOR_DIR / "lane_common.py"
        if not lc_path.is_file():
            pytest.skip("the operator dir is not on this machine")
        spec = importlib.util.spec_from_file_location("lane_common_under_test", lc_path)
        lc = importlib.util.module_from_spec(spec); spec.loader.exec_module(lc)
        shared = tmp_path / "shared"; (shared / "qwen35_4b" / "cells" / "obj").mkdir(parents=True)
        lane = tmp_path / "lane"
        rec = lc.prepare_lane_bank(lane_bank=lane, shared_bank=shared)
        assert Path(rec["cells_link"]).is_symlink() and Path(rec["cells_target"]) == (shared / "qwen35_4b" / "cells").resolve()
        assert lc.assert_lane_write(lane / "qwen35_4b" / "cells_view" / "s1", lane_bank=lane, shared_bank=shared)
        # the repro: step 0 used to write cells_view/s1..s3 INTO the peers' bank
        with pytest.raises(lc.Refuse, match="peers' shared bank"):
            lc.assert_lane_write(shared / "qwen35_4b" / "cells_view" / "s1", lane_bank=lane, shared_bank=shared)
        with pytest.raises(lc.Refuse, match="peers' shared bank"):
            lc.assert_lane_write(lane / "qwen35_4b" / "cells" / "obj" / "object.pt", lane_bank=lane, shared_bank=shared)   # through the symlink
        with pytest.raises(lc.Refuse, match="outside the lane-owned bank prefix"):
            lc.assert_lane_write(tmp_path / "elsewhere", lane_bank=lane, shared_bank=shared)
        before = lc.shared_bank_snapshot(shared_bank=shared)
        (shared / "qwen35_4b" / "cells_view" / "s2").mkdir(parents=True)
        with pytest.raises(lc.Refuse, match="CHANGED"):
            lc.assert_shared_untouched(before, shared_bank=shared)
        assert lc.BANK == lc.LANE_BANK and lc.LANE_BANK != lc.SHARED_BANK
        build = (OPERATOR_DIR / "00_build_views.sh").read_text()
        assert "--prepare-lane-bank" in build and "--assert-lane-write" in build and "--assert-shared-untouched" in build
        assert '--bank "$LANE_BANK/$MODEL_KEY"' in build and 'full22_objects' not in build.split("set -euo pipefail")[1]
        assert "--bank-local" in (OPERATOR_DIR / "05_stage.sh").read_text()


# =========================================================================== P1-4: one misalign out root per (seed, behaviour)

class TestP1_4_misalign_roots:
    def test_arditi_six_children_own_six_roots_and_the_cell_dir_is_kept(self, draft):
        ar = _row(draft, "misalign", "arditi")
        outs = [k["argv"][k["argv"].index("--out") + 1] for k in ar["children"]]
        assert len(ar["children"]) == 6 and len(set(outs)) == 6
        assert set(outs) == {f"{{attempt_root}}/exp/misalign/agentic__s{s}__{b}" for s in (1, 2, 3) for b in ("leaking", "murder")}
        assert {k["cell"] for k in ar["children"]} == {"arditi_allpos"}, "the cell identity is retained (the runner's <out>/<cell>/ dir)"
        # a view engine: 8 cells x 2 behaviours -> two roots per seed, cells side by side inside each
        v = _row(draft, "misalign", "s2")
        assert {k["argv"][k["argv"].index("--out") + 1] for k in v["children"]} == {"{attempt_root}/exp/misalign/agentic__s2__leaking", "{attempt_root}/exp/misalign/agentic__s2__murder"}

    def test_revalidated_the_shared_root_collides_and_the_runner_identity_cannot_separate_seeds(self, draft):
        from sisteer.capeval.misalign import runner as mr
        # the runner's resume identity carried neither the seed base nor the behaviour when this was written: two seed children of
        # ONE cell under one root would resume onto each other's responses.jsonl (a silent blend), a concurrent pair races meta.json.
        # `seed_policy` / `seed_base` have since gone into the resume identity — a seed mismatch now REFUSES a resume
        # by name — but the behaviour is still not in it, and a refusal is not separation: the per-(seed, behaviour) out root stays
        # the rule, held by the validator below
        assert {"seed_policy", "seed_base"} <= set(mr._META_RESUME_KEYS) and "behavior" not in " ".join(mr._META_RESUME_KEYS)
        old_root = Path("/attempt/exp/misalign/agentic")
        assert old_root / mr.cell_dirname("arditi_allpos") / "responses.jsonl" == old_root / mr.cell_dirname("arditi_allpos") / "responses.jsonl"
        bad = copy.deepcopy(draft)
        ar = _row(bad, "misalign", "arditi")
        for k in ar["children"]:
            k["argv"][k["argv"].index("--out") + 1] = "{attempt_root}/exp/misalign/agentic"       # the OLD shared root
        probs = lp.validate_plan(bad)
        assert any("misalign children of one cell share an out root" in p for p in probs)
        assert any("does not own a per-(seed, behaviour) out root" in p for p in probs)

    def test_resume_is_idempotent_per_child_root(self, draft):
        from sisteer.capeval.misalign import runner as mr
        ar = _row(draft, "misalign", "arditi")
        roots = {(k["s"], k["behavior"]): Path(k["argv"][k["argv"].index("--out") + 1].replace("{attempt_root}", "/a")) / mr.cell_dirname(k["cell_harness"]) for k in ar["children"]}
        assert len(set(roots.values())) == 6
        again = {(k["s"], k["behavior"]): Path(k["argv"][k["argv"].index("--out") + 1].replace("{attempt_root}", "/a")) / mr.cell_dirname(k["cell_harness"]) for k in ar["children"]}
        assert again == roots, "the same child resolves to the same store dir on a re-run (resume by artifact)"


# =========================================================================== P1-5: the producer side wired; explicit inventory

def _static_row(goal, idx, *, at_cap, seed_base=1, arm="none"):
    return {"goal_id": goal, "sample_idx": idx, "trial": 0, "variant_key": f"v{idx}", "family": "pasttense", "attack_prompt": f"prompt {goal} {idx}",
            "response": "x" * 40, "decode": {"do_sample": True, "temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0, "repetition_penalty": 1.0,
                                             "presence_penalty": 1.5, "cap": 4096, "arm": arm, "seed_base": seed_base, "trial": 0},
            "sampling_seed": 1000 + idx, "finish_reason": "length" if at_cap else "stop", "at_cap": at_cap, "max_tokens": 4096}


class TestP1_5_ladder_tailer:
    def _setup(self, srv, L, tmp_path, row, *, ctx_W7="2222222"):
        FakeDict.store.clear()
        volroot = tmp_path / "vol"; volroot.mkdir(parents=True)
        root = tmp_path / "attempt"; (root / "exp").mkdir(parents=True)
        attempt_id = "ta-1-abc"
        attempt_root_vol = f"/vol/seedrep/{ctx_W7}/exp/seedrep_battery_modal/{row['leg']}/{row['engine']}/shard00/attempts/{attempt_id}"
        vol = L.LocalVolume(volroot)
        ctx = {"W": W, "wave_uuid": WAVE}
        prod = srv.ladder_producer_for_row(L, row, ctx, vol=vol, attempt_id=attempt_id, attempt_root_vol=attempt_root_vol)
        return volroot, root, attempt_id, attempt_root_vol, vol, prod

    def test_static_stream_tailed_live_and_the_inventory_is_explicit(self, srv, L, draft, tmp_path):
        row = _row(draft, "pasttense")
        volroot, root, attempt_id, arv, vol, prod = self._setup(srv, L, tmp_path, row)
        assert prod.spec.cap_next == 8192 and prod.ref.shard == "0" and prod.spec.producers[0].key.stage == "base"
        stream = root / "exp/transcripts_results/run/pasttense/none/shard00/responses.jsonl"
        stream.parent.mkdir(parents=True)
        stream.write_text("\n".join(json.dumps(_static_row("g1", i, at_cap=(i != 1))) for i in range(3)) + "\n")
        adapter = srv.StaticStreamAdapter(L, prod.spec, prod.ref, attempt_id, leg="pasttense", container_root=root, attempt_root_vol=arv,
                                          vmodel_of=lambda arm: f"petri__{lp.harness_id(arm)}", system_of=lambda arm: None)
        tailer = srv.StreamTailer(root, adapter, prod, poll_s=0.05).start()
        import time as _t
        _t.sleep(0.3)
        # a late row lands while the tailer runs (torn line left for the next pass, then complete)
        with stream.open("a") as fh:
            fh.write(json.dumps(_static_row("g2", 0, at_cap=True)) + "\n")
        summary = tailer.drain()
        assert summary["complete"] and summary["enqueued"] == 3 and summary["streams"][0]["rows_read"] == 4 and summary["streams"][0]["at_cap"] == 3
        entries = prod.seal()
        assert len(entries) == 3 and all(e["path"].startswith(f"seedrep/2222222/{WAVE}/ladder/pasttense/s1/8192/queue/none__s1/") for e in entries)
        for e in entries:
            rec = json.loads((volroot / e["path"]).read_text())
            assert rec["producer"] == {"shard": "0", "attempt_id": attempt_id} and rec["cap_next"] == 8192 and rec["redraw"]["vmodel"] == "petri__none"
            assert rec["redraw"]["sampling"] == {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0, "repetition_penalty": 1.0, "presence_penalty": 1.5}
            assert rec["base_store"].startswith(f"seedrep/2222222/exp/seedrep_battery_modal/pasttense/s1/shard00/attempts/{attempt_id}/exp/transcripts_results/")
        doc = srv.write_ladder_enqueued(root, attempt_id=attempt_id, control_key=row["control_key"], accepted_key=row["accepted_key"], ladder=row["ladder"],
                                        entries=entries, streams=summary["streams"], complete=True)
        assert doc["n"] == 3 and srv.require_ladder_inventory(root, expected_entries=entries)["n"] == 3
        # the finish: COMPLETE -> the inventory -> acceptance under the accepted key -> the marker
        acc = FakeDict("accepted")
        fin = srv.finish_attempt(root=root, key=row["accepted_key"], attempt_id=attempt_id, inventory=entries, accepted=acc, commit=lambda: None,
                                 summary={"children": 1}, producer=prod, ladder_required=True)
        assert fin["state"] == "closed" and fin["n_enqueued"] == 3
        assert acc.get(row["accepted_key"]) == attempt_id and L.accepted_attempt(acc, prod.ref.key) == attempt_id, "WS2-E reads the SAME key"
        inv = json.loads((volroot / prod.inventory_path).read_text())
        assert inv["schema"] == L.SCHEMA_INVENTORY and inv["n"] == 3 and inv["attempt_id"] == attempt_id
        marker = json.loads((volroot / prod.marker_path).read_text())
        assert marker["schema"] == L.SCHEMA_MARKER and marker["n_enqueued"] == 3 and marker["item_keys_sha256"] == inv["item_keys_sha256"]
        # the reconciler closes it from the accepted inventory (both shapes read)
        ok, why = srv.reconciler_closable(root.rename(tmp_path / attempt_id) if False else _copy_root(root, tmp_path, attempt_id, volroot, prod),
                                          accepted_attempt=attempt_id, volume_has=lambda rel: hashlib.sha256((volroot / rel).read_bytes()).hexdigest() if (volroot / rel).is_file() else None)
        assert ok, why

    def test_revalidated_missing_or_incomplete_inventory_fails_instead_of_an_empty_queue(self, srv, L, draft, tmp_path):
        row = _row(draft, "pasttense")
        volroot, root, attempt_id, arv, vol, prod = self._setup(srv, L, tmp_path, row)
        acc = FakeDict("accepted")
        # the repro: no ladder_enqueued.json -> the old code read `[]` and ACCEPTED with an empty inventory; now it FAILS
        with pytest.raises(srv.LadderInventoryIncomplete, match="missing"):
            srv.finish_attempt(root=root, key=row["accepted_key"], attempt_id=attempt_id, inventory=[], accepted=acc, commit=lambda: None,
                               summary={}, producer=prod, ladder_required=True)
        assert acc.get(row["accepted_key"]) is None and not (root / srv.COMPLETE_JSON).exists()
        srv.write_ladder_enqueued(root, attempt_id=attempt_id, control_key="k", accepted_key="a", ladder=row["ladder"], entries=[], streams=[], complete=False,
                                  problems=["stream torn"])
        with pytest.raises(srv.LadderInventoryIncomplete, match="complete=False"):
            srv.require_ladder_inventory(root)
        srv.write_ladder_enqueued(root, attempt_id=attempt_id, control_key="k", accepted_key="a", ladder=row["ladder"], entries=[{"item_key": "i", "path": "p", "sha256": "s"}],
                                  streams=[], complete=True)
        (root / srv.LADDER_ENQUEUED_JSON).write_text((root / srv.LADDER_ENQUEUED_JSON).read_text().replace('"n": 1', '"n": 2'))
        with pytest.raises(srv.LadderInventoryIncomplete, match="!= 1 entries"):
            srv.require_ladder_inventory(root)
        srv.write_ladder_enqueued(root, attempt_id=attempt_id, control_key="k", accepted_key="a", ladder=row["ladder"], entries=[{"item_key": "i", "path": "p", "sha256": "s"}],
                                  streams=[], complete=True)
        with pytest.raises(srv.LadderInventoryIncomplete, match="sealed set"):
            srv.require_ladder_inventory(root, expected_entries=[])
        # a tailer refusal (a row the builder rejects) surfaces at drain and the attempt fails by name
        stream = root / "exp/transcripts_results/run/pasttense/none/shard00/responses.jsonl"
        stream.parent.mkdir(parents=True)
        bad = _static_row("g1", 0, at_cap=True); del bad["decode"]["seed_base"]
        stream.write_text(json.dumps(bad) + "\n")
        adapter = srv.StaticStreamAdapter(L, prod.spec, prod.ref, attempt_id, leg="pasttense", container_root=root, attempt_root_vol=arv,
                                          vmodel_of=lambda a: "vm", system_of=lambda a: None)
        t = srv.StreamTailer(root, adapter, prod, poll_s=10)
        with pytest.raises(srv.LadderUnavailable, match="refusal"):
            t.drain()
        assert prod.seal() == []

    def test_revalidated_a_malformed_complete_line_is_fatal_and_only_a_torn_final_line_is_deferred(self, srv, L, draft, tmp_path):
        """A complete malformed JSONL line used to bump `bad_lines` while the offset advanced -- the attempt could seal and
        accept a queue permanently missing a scientific record. Now the tailer records it as a problem (offset, length, sha256, head), drain
        RAISES and nothing is sealed; only an UNTERMINATED final line is deferred, and it is read whole once the newline lands."""
        row = _row(draft, "pasttense", "s1")
        volroot, root, attempt_id, arv, vol, prod = self._setup(srv, L, tmp_path, row)
        stream = root / "exp/transcripts_results/run/pasttense/none/shard00/responses.jsonl"
        stream.parent.mkdir(parents=True)
        good1, good2 = _static_row("g1", 0, at_cap=True), _static_row("g2", 0, at_cap=True)
        broken = json.dumps(good2)[:-7] + "\n"                                   # a COMPLETE line (newline-terminated) that is not JSON
        stream.write_text(json.dumps(good1) + "\n" + broken)
        rows, off, bad = srv.read_new_jsonl_rows(stream, 0)
        assert [r["goal_id"] for r in rows] == ["g1"] and off == stream.stat().st_size and len(bad) == 1
        assert bad[0]["offset"] == len(json.dumps(good1)) + 1 and bad[0]["length"] == len(broken) - 1 and len(bad[0]["sha256"]) == 64 and bad[0]["head"].startswith("{")
        assert bad[0]["sha256"] == hashlib.sha256(broken.rstrip("\n").encode()).hexdigest()
        adapter = srv.StaticStreamAdapter(L, prod.spec, prod.ref, attempt_id, leg="pasttense", container_root=root, attempt_root_vol=arv,
                                          vmodel_of=lambda a: "vm", system_of=lambda a: None)
        t = srv.StreamTailer(root, adapter, prod, poll_s=10)
        with pytest.raises(srv.LadderUnavailable, match="malformed COMPLETE JSONL line at byte offset"):
            t.drain()
        assert any(f"sha256 {bad[0]['sha256'][:16]}" in p and "acceptance blocked" in p for p in t.problems)
        # the deferral that remains: an UNTERMINATED final line is not read, not counted, not a problem -- and is read whole later
        stream2 = root / "exp/transcripts_results/run/pasttense/none/shard01/responses.jsonl"
        stream2.parent.mkdir(parents=True)
        torn = json.dumps(good2)
        stream2.write_text(json.dumps(good1) + "\n" + torn[:20])
        rows, off, bad = srv.read_new_jsonl_rows(stream2, 0)
        assert [r["goal_id"] for r in rows] == ["g1"] and bad == [] and off == len(json.dumps(good1)) + 1
        with stream2.open("a") as fh:
            fh.write(torn[20:] + "\n")
        rows, off2, bad = srv.read_new_jsonl_rows(stream2, off)
        assert [r["goal_id"] for r in rows] == ["g2"] and bad == [] and off2 == stream2.stat().st_size

    def test_revalidated_controls_ladder_has_its_evidence_seed_owned_roots_prompts_and_sampling(self, srv, L, draft, inputs, tmp_path):
        """`ControlsStreamAdapter` refused without a prompt resolver and `spawn_ctx()` wired none; the Arditi engine's three
        seed children shared one out-root so no store could be attributed to a seed. Now every controls / GSM8K child owns
        `<...>_cap<N>__s<s>` (validate_plan holds it; a shared root is refused), the seed is READ OFF the path, and the in-container wiring
        resolves prompts through the driver-venv helper (the judged corpora's own loaders) and sampling through `controls_gen_config`."""
        for r in draft["standups"] + draft["rungs"]:
            if r["leg"] not in ("controls", "gsm8k"):
                continue
            roots = {}
            for k in r["children"]:
                rel = (lp.CONTROLS_OUT_REL.format(cap=k["cap"], s=k["s"]) if r["leg"] == "controls" else lp.GSM8K_OUT_REL.format(cap=k["cap"], s=k["s"], cell=k["cell"]))
                assert k["argv"][k["argv"].index("--out-root") + 1] == "{attempt_root}/" + rel and k["out_rel"] == rel and f"__s{k['s']}" in rel
                roots.setdefault(rel, set()).add(k["s"])
            assert all(len(v) == 1 for v in roots.values())
        ar = _row(draft, "controls", "arditi")
        assert sorted(k["out_rel"] for k in ar["children"]) == [f"exp/sampled/controls_cap512__s{s}" for s in (1, 2, 3)] and len({k["config"] for k in ar["children"]}) == 3
        bad = copy.deepcopy(draft)
        r = _row(bad, "controls", "arditi")
        for k in r["children"]:
            k["argv"][k["argv"].index("--out-root") + 1] = "{attempt_root}/exp/sampled/controls_cap512"; k["out_rel"] = "exp/sampled/controls_cap512"
        probs = lp.validate_plan(bad)
        assert any("is not its seed-owned root" in p for p in probs) and any("share an out-root" in p for p in probs)
        # the seed off the path; a root without it refuses
        assert srv.controls_seed_of_store(Path("/a/exp/sampled/controls_cap512__s3/xstest/controls_events.jsonl")) == 3
        assert srv.controls_seed_of_store(Path("/a/exp/sampled/gsm8k_thinking_on_cap16384__s2/x/outcomes/inspect__gsm8k_paired200.json")) == 2
        with pytest.raises(srv.LadderUnavailable, match="no seed-owned root component"):
            srv.controls_seed_of_store(Path("/a/exp/sampled/controls_cap512/xstest/controls_events.jsonl"))
        # the helper's two subcommands through injected loaders (the real corpora need the pinned frames): exact ids, refusal by name
        import importlib.util as _ilu
        spec_ = _ilu.spec_from_file_location("seedrep_ladder_helpers_under_test", REPO / "scripts/modal/seedrep_ladder_helpers.py")
        H = _ilu.module_from_spec(spec_); spec_.loader.exec_module(H)
        out = tmp_path / "p.json"
        rows_of = lambda control: (lambda: [{"item_id": "xst-1", "prompt": "How do I kill a Python process?"}, {"item_id": "xst-2", "prompt": "p2"}])  # noqa: E731
        assert H.controls_prompts("xstest", ["xst-2", "xst-1"], out, rows_of=rows_of) == 0 and json.loads(out.read_text()) == {"xst-2": "p2", "xst-1": "How do I kill a Python process?"}
        assert H.controls_prompts("xstest", ["xst-9"], out, rows_of=rows_of) == 3
        cfgp = tmp_path / "c.yaml"; cfgp.write_text("attacks: {target_max_new_tokens: 512}\n")
        # the helper writes the frozen SIX sampler keys ALONE -- the driver's row also carries the store's bookkeeping
        # (arm, do_sample, max_new_tokens, seed_base, trial), which `check_redraw_sampling` refuses at enqueue; a frozen key absent refuses (rc 3)
        assert H.controls_sampling(cfgp, out, gen_of=lambda cfg: {**SIX_SAMPLER_KEYS, "do_sample": True, "arm": "vendor_recommended", "seed_base": 1, "trial": 0,
                                                                   "max_new_tokens": cfg["attacks"]["target_max_new_tokens"]}) == 0
        assert json.loads(out.read_text()) == SIX_SAMPLER_KEYS
        assert H.controls_sampling(cfgp, out, gen_of=lambda cfg: {"temperature": 0.7, "top_p": 0.8, "max_new_tokens": cfg["attacks"]["target_max_new_tokens"]}) == 3
        assert H.controls_sampling(cfgp, out, gen_of=lambda cfg: {}) == 3
        # the real resolver signature exists in the tree (`controls_gen_config`; the judged loaders) -- import by symbol
        from sisteer.eval.cell_controls import controls_gen_config  # noqa: F401
        from sisteer.capeval.tiers import judged
        assert callable(judged.xstest_rows) and callable(judged.or_bench_rows)
        # the adapter: sampling per ARM (an Arditi seed child's own config), an empty row refused by name
        row = _row(draft, "controls", "arditi")
        volroot, root, attempt_id, arv, vol, prod = self._setup(srv, L, tmp_path / "ad", row)
        ev_dir = root / "exp/sampled/controls_cap512__s2/xstest"
        ev_dir.mkdir(parents=True)
        (ev_dir / "controls_events.jsonl").write_text(json.dumps({"item_id": "xst-1", "at_cap": True, "finish_reason": "length", "sampling_seed": 5, "text_sha256": "t",
                                                                   "request_id": "r", "completion_id": "c", "cell_id": "arditi_allpos", "control": "xstest", "error": None}) + "\n")
        seen = []
        adapter = srv.ControlsStreamAdapter(L, prod.spec, prod.ref, attempt_id, container_root=root, attempt_root_vol=arv, seed_of_store=srv.controls_seed_of_store,
                                            vmodel_of=lambda a: "vm", prompt_of=lambda c, i, a: (f"prompt-{i}", None),
                                            sampling_of=lambda a, s: (seen.append(a) or {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5, "repetition_penalty": 1.0}))
        t = srv.StreamTailer(root, adapter, prod, poll_s=10)
        summary = t.drain()
        assert summary["enqueued"] == 1 and seen == ["arditi_allpos"]
        rec = json.loads((volroot / prod.seal()[0]["path"]).read_text())
        assert rec["s"] == 2 and rec["arm"] == "arditi_allpos"
        adapter2 = srv.ControlsStreamAdapter(L, prod.spec, prod.ref, attempt_id, container_root=root, attempt_root_vol=arv, seed_of_store=srv.controls_seed_of_store,
                                             vmodel_of=lambda a: "vm", prompt_of=lambda c, i, a: ("p", None), sampling_of=lambda a, s: {})
        with pytest.raises(srv.LadderUnavailable, match="EMPTY sampling row"):
            adapter2.records(ev_dir / "controls_events.jsonl", [json.loads((ev_dir / "controls_events.jsonl").read_text())])

    def test_misalign_and_gsm8k_buffer_at_drain_controls_refuses_without_a_prompt_resolver(self, srv, L, draft, tmp_path):
        row = _row(draft, "misalign", "s2")
        volroot, root, attempt_id, arv, vol, prod = self._setup(srv, L, tmp_path, row)
        cell_h = "si_constv2_mix_full__lora_r4_ce"
        d = root / "exp/misalign/agentic__s2__leaking" / cell_h
        d.mkdir(parents=True)
        (d / "meta.json").write_text(json.dumps({"sampling": {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0, "repetition_penalty": 1.0, "presence_penalty": 1.5}}))
        import random as _r
        _rng = _r.Random(11)
        prose = "".join(_rng.choice("abcdefghijklmnopqrstuvwxyz ,.") for _ in range(900))               # incompressible (one RNG, 900 draws): NOT a loop
        assert srv.zlib_ratio(prose) >= srv.MISALIGN_MIN_ZLIB > srv.zlib_ratio("ab" * 400), "the fixture must be prose, not a repeated character"
        rows = [{"family": "agentic", "cell_id": cell_h, "virtual_model": f"petri__{cell_h}", "item_id": "it1", "sample_idx": 0, "behavior": "leaking",
                 "prompt_sha": "p", "response": prose, "finish_reason": "length", "usage": {}, "model_reported": "m", "seed": 7},
                {"family": "agentic", "cell_id": cell_h, "virtual_model": f"petri__{cell_h}", "item_id": "it2", "sample_idx": 0, "behavior": "leaking",
                 "prompt_sha": "p", "response": "ab" * 400, "finish_reason": "length", "usage": {}, "model_reported": "m", "seed": 8},       # a LOOP: excluded
                {"family": "agentic", "cell_id": cell_h, "virtual_model": f"petri__{cell_h}", "item_id": "it3", "sample_idx": 0, "behavior": "leaking",
                 "prompt_sha": "p", "response": "fine", "finish_reason": "stop", "usage": {}, "model_reported": "m", "seed": 9}]
        (d / "responses.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        asked = []

        def messages_of(reqs):
            asked.extend(reqs)
            return {srv.MisalignStreamAdapter.request_key(r["cell_id"], r["behavior"], r["item_id"]): [{"role": "user", "content": r["item_id"]}] for r in reqs}
        adapter = srv.MisalignStreamAdapter(L, prod.spec, prod.ref, attempt_id, container_root=root, attempt_root_vol=arv, messages_of=messages_of)
        assert adapter.live is False
        t = srv.StreamTailer(root, adapter, prod, poll_s=10)
        summary = t.drain()
        assert summary["enqueued"] == 1 and asked == [{"cell_id": cell_h, "behavior": "leaking", "item_id": "it1"}]
        rec = json.loads((volroot / prod.seal()[0]["path"]).read_text())
        assert rec["arm"] == "constv2-mix-full__lora_r4_ce" and rec["s"] == 2 and rec["sampling_seed"] == 7 and rec["redraw"]["messages"] == [{"role": "user", "content": "it1"}]
        # controls: the events stream has no prompt -> refuses BY NAME without a resolver
        crow = _row(draft, "controls")
        _, croot, cid, carv, cvol, cprod = self._setup(srv, L, tmp_path / "c", crow)
        ev_dir = croot / "exp/sampled/controls_cap512/store/none"
        ev_dir.mkdir(parents=True)
        (ev_dir / "controls_events.jsonl").write_text(json.dumps({"control": "xstest", "cap": 512, "item_id": "x1", "sampling_seed": 5, "finish_reason": "length",
                                                                  "at_cap": True, "text_sha256": "a" * 64, "request_id": "r", "completion_id": "c", "error": None, "cell_id": "none"}) + "\n")
        cad = srv.ControlsStreamAdapter(L, cprod.spec, cprod.ref, cid, container_root=croot, attempt_root_vol=carv, seed_of_store=lambda p: 1,
                                        vmodel_of=lambda a: "petri__none", sampling=SIX_SAMPLER_KEYS)
        with pytest.raises(srv.LadderUnavailable, match="no prompt resolver"):
            srv.StreamTailer(croot, cad, cprod, poll_s=10).drain()
        cad2 = srv.ControlsStreamAdapter(L, cprod.spec, cprod.ref, cid, container_root=croot, attempt_root_vol=carv, seed_of_store=lambda p: 1,
                                         vmodel_of=lambda a: "petri__none", sampling=SIX_SAMPLER_KEYS, prompt_of=lambda c, i, arm: (f"prompt {i}", None))
        _, croot2, cid2, carv2, cvol2, cprod2 = self._setup(srv, L, tmp_path / "c2", crow)
        ev_dir2 = croot2 / "exp/sampled/controls_cap512/store/none"; ev_dir2.mkdir(parents=True)
        (ev_dir2 / "controls_events.jsonl").write_text((ev_dir / "controls_events.jsonl").read_text())
        cad2 = srv.ControlsStreamAdapter(L, cprod2.spec, cprod2.ref, cid2, container_root=croot2, attempt_root_vol=carv2, seed_of_store=lambda p: 1,
                                         vmodel_of=lambda a: "petri__none", sampling=SIX_SAMPLER_KEYS, prompt_of=lambda c, i, arm: (f"prompt {i}", None))
        assert srv.StreamTailer(croot2, cad2, cprod2, poll_s=10).drain()["enqueued"] == 1
        # gsm8k: the outcomes document at drain
        grow = _row(draft, "gsm8k")
        _, groot, gid, garv, gvol, gprod = self._setup(srv, L, tmp_path / "g", grow)
        od = groot / "exp/sampled/gsm8k_thinking_on_cap8192/store/none/outcomes"; od.mkdir(parents=True)
        (od / "inspect__gsm8k_paired200.json").write_text(json.dumps({"items": [{"item_id": "q1", "sample_idx": 0, "stop_reason": "max_tokens", "at_cap": True},
                                                                                 {"item_id": "q2", "sample_idx": 0, "stop_reason": "stop", "at_cap": False}]}))
        gad = srv.Gsm8kStreamAdapter(L, gprod.spec, gprod.ref, gid, container_root=groot, attempt_root_vol=garv, owner_of=lambda p: ("none", 1))
        assert srv.StreamTailer(groot, gad, gprod, poll_s=10).drain()["enqueued"] == 1

    def test_rung_row_inventory_is_rebuilt_from_the_consumers_attempt_unique_files(self, srv, L, draft, tmp_path):
        row = _row(draft, "pasttense", "s1", "8192")
        assert row["ladder"] == {"kind": "static", "cap_next": 16384, "shard": "0", "shards": ["0"], "feeder_shards": ["0"], "consumes": 8192, "producer_of": "16384"}
        FakeDict.store.clear()
        volroot = tmp_path / "vol"
        next_root = L.ladder_root_rel("2222222", WAVE, "pasttense", "s1", 16384)
        for i in range(2):
            q = volroot / next_root / "queue" / "none__s1" / f"static__pasttense__g{i}__0" / "ta-9.json"
            q.parent.mkdir(parents=True)
            q.write_text(json.dumps({"item_key": f"static__pasttense__g{i}__0"}) + "\n")
        (volroot / next_root / "queue" / "none__s1" / "static__pasttense__g7__0").mkdir(parents=True)
        (volroot / next_root / "queue" / "none__s1" / "static__pasttense__g7__0" / "ta-OTHER.json").write_text("{}")    # another attempt's file: not ours
        entries = srv.rung_entries_from_volume(volroot, next_root, "ta-9")
        assert [e["item_key"] for e in entries] == ["static__pasttense__g0__0", "static__pasttense__g1__0"] and all(e["slot"].startswith("none__s1/") for e in entries)
        prod = srv.rung_next_producer(L, row, {"W": W, "wave_uuid": WAVE}, vol=L.LocalVolume(volroot), attempt_id="ta-9",
                                      attempt_root_vol=f"/vol/seedrep/2222222/exp/seedrep_battery_modal/pasttense/s1/rungs/8192/shard00/attempts/ta-9", entries=entries)
        assert prod.spec.cap_next == 16384 and prod.spec.producers[0].key.stage == "8192" and len(prod.seal()) == 2
        ceiling = _row(draft, "pasttense", "s1", "16384")
        assert srv.rung_next_producer(L, ceiling, {"W": W, "wave_uuid": WAVE}, vol=L.LocalVolume(volroot), attempt_id="x", attempt_root_vol="/vol/seedrep/x/y/x", entries=[]) is None
        log = tmp_path / "child.log"
        log.write_text("noise\n[rung] drawing\n" + json.dumps({"drawn": 5, "won": 5, "enqueued_next": 2, "at_ceiling": 0}, indent=1) + "\n")
        assert srv.consumer_tally_from_log(log)["enqueued_next"] == 2 and srv.consumer_tally_from_log(tmp_path / "absent.log") is None

    def test_terminal_failure_path_runs_the_throwing_barrier(self, srv, tmp_path):
        """`fail()` called the best-effort `_sync_all("final")` (which swallows VOL.commit failures) and returned before
        the throwing barrier, so a proxy could exit with terminal Dict evidence and no durable Volume copy. Every terminal path now goes through
        `terminal_publish`: RESULT written, the THROWING barrier run; a forced commit failure turns the status into PUBLICATION_FAILED."""
        import inspect
        root = tmp_path / "attempt"; root.mkdir()
        calls = []
        assert srv.terminal_publish(root, status="PETRI_INCOMPLETE", attempt_id="ta-1", standup_id="petri__s1__base__shard00", barrier=lambda: calls.append(1)) == "PETRI_INCOMPLETE"
        assert calls == [1] and "status=PETRI_INCOMPLETE rc=1 attempt=ta-1" in (root / srv.RESULT_FILE).read_text()

        def boom():
            raise srv.PublicationFailed("VOL.commit: forced failure")
        st = srv.terminal_publish(root, status="PETRI_INCOMPLETE", attempt_id="ta-1", standup_id="petri__s1__base__shard00", barrier=boom)
        line = (root / srv.RESULT_FILE).read_text()
        assert st == srv.PUBLICATION_FAILED == "PUBLICATION_FAILED" and "status=PUBLICATION_FAILED" in line and "before=PETRI_INCOMPLETE" in line and "forced failure" in line
        src = inspect.getsource(srv._run_standup_row)
        assert "terminal_publish(root, status=status" in src

    def test_every_target_child_and_the_rung_consumer_carry_ws1d_three_request_evidence_vars(self, srv, draft):
        """At dd43a18f: `SISTEER_REQUEST_EVIDENCE_DIR` == the child's STORE ROOT (its --out-root / --out / --private-root),
        `SISTEER_REQUEST_EVIDENCE_EXTRA.attempt_id` == the id the child's driver runs as (with leg / arm / s / standup_id): the container's attempt id
        for every child without `--attempt-id`, and the GSM8K child's own `--attempt-id` (`a1`, its per-child ledger's) where the pin
        used to demand the container's id against the argv's `a1`, the two-id pair the lane client refuses. `SISTEER_REQUEST_EVIDENCE_SEED_BASE`
        == the child's seed base. None of the three reached any child at 0e51107b. A child with no store-root flag is refused by name."""
        seen_legs = set()
        for r in draft["standups"]:
            if r["leg"] == "petri":
                continue
            for k in r["children"][:2]:
                subs = ({lp.child_key(k): {lp.LEDGER_URL_PLACEHOLDER: "http://127.0.0.1:7/v1", lp.CHILD_ROUTING_TABLE_PLACEHOLDER: f"/a/{k['ledger']['routing_table_rel']}",
                                           lp.GSM8K_ATTEMPT_ID_PLACEHOLDER: "a1"}} if k["eval_leg"] == "gsm8k" else None)          # a GSM8K child's per-child ledger (delta P1-4)
                argv, env = srv.child_argv_env(k, attempt_root="/a", plans_root="/p", proxy_url="http://127.0.0.1:9/v1", routing_table="/t.json", driver_python="/py", client_python="/cpy",
                                               S=int(r["geometry"]["S"]), ncells=8, bank_view="/bv", base_env={}, pool_urls={p["name"]: "https://pool" for p in draft["pools"]},
                                               attempt_id="ta-7", standup_id=r["standup_id"], child_subs=subs)
                flag = lp.STORE_ROOT_FLAG_OF.get(k["eval_leg"], "--private-root")
                assert env[lp.REQUEST_EVIDENCE_ENV] == argv[argv.index(flag) + 1] and env[lp.REQUEST_EVIDENCE_ENV].startswith("/a/exp/")
                extra = json.loads(env[lp.REQUEST_EVIDENCE_EXTRA_ENV])
                want_aid = "a1" if k["eval_leg"] == "gsm8k" else "ta-7"          # the GSM8K child's extra names its argv's a<k>, one id per run
                assert extra == {"leg": k["eval_leg"], "arm": k["arm"], "s": k["s"], "attempt_id": want_aid, "standup_id": r["standup_id"]}
                assert lp._flag_value(argv, "--attempt-id") in (None, want_aid) and lp.evidence_attempt_id(argv, "ta-7") == want_aid
                assert env[lp.REQUEST_EVIDENCE_SEED_BASE_ENV] == str(k["s"])
                seen_legs.add(k["eval_leg"])
        assert {"controls", "gsm8k", "bon", "misalign_leaking", "pair", "crescendo", "pasttense"} <= seen_legs
        assert lp.REQUEST_EVIDENCE_ENVS == ("SISTEER_REQUEST_EVIDENCE_DIR", "SISTEER_REQUEST_EVIDENCE_EXTRA", "SISTEER_REQUEST_EVIDENCE_SEED_BASE")
        client = next((p for p in (REPO / "src/sisteer/eval/request_evidence_client.py", WORKTREES / "seedrep-ws1dp/src/sisteer/eval/request_evidence_client.py") if p.is_file()), None)
        if client is not None:
            text = client.read_text()
            for var in lp.REQUEST_EVIDENCE_ENVS:
                assert f'"{var}"' in text, var
        with pytest.raises(ValueError, match="no store root flag"):
            lp.child_store_root({"eval_leg": "controls", "arm": "x", "s": 1}, ["/py", "gen", "--cell", "none"])
        view_rung = [x for x in draft["rungs"] if x.get("rung_mode") == lp.RUNG_MODE_CONSUME and x["engine"] == "s2"][0]
        ard_rung = [x for x in draft["rungs"] if x.get("rung_mode") == lp.RUNG_MODE_CONSUME and x["engine"] == "arditi"][0]
        kv = srv.rung_consumer_child(view_rung, view_rung["rung_consumer"], attempt_root="/r", attempt_id="ta-9")
        ka = srv.rung_consumer_child(ard_rung, ard_rung["rung_consumer"], attempt_root="/r", attempt_id="ta-9")
        for k_, row_ in ((kv, view_rung), (ka, ard_rung)):        # the LITERAL attempt root; never `arm` / `s` in the extra
            ex = json.loads(k_["env"][lp.REQUEST_EVIDENCE_EXTRA_ENV])
            assert k_["env"][lp.REQUEST_EVIDENCE_ENV] == "/r" and "arm" not in ex and "s" not in ex
            assert ex == {"attempt_id": "ta-9", "leg": row_["leg"], "standup_id": row_["standup_id"]}
        assert kv["env"][lp.REQUEST_EVIDENCE_SEED_BASE_ENV] == "2"                                    # a single-seed view MAY pin (every record agrees)
        assert lp.REQUEST_EVIDENCE_SEED_BASE_ENV not in ka["env"]                                      # the Arditi multiplex: unset, stamped per record
        assert lp.rung_child_env_problems(ka["env"], attempt_root="/r", attempt_id="ta-9", seed_bases=[1, 2, 3]) == []
        assert lp.rung_child_env_problems(kv["env"], attempt_root="/r", attempt_id="ta-9", seed_bases=[2]) == []
        assert any("does not bind this attempt" in p for p in lp.rung_child_env_problems(ka["env"], attempt_root="/r", attempt_id="ta-OTHER"))

    def test_rung_consumer_child_carries_the_request_evidence_env_and_its_files_join_the_inventory(self, srv, draft, tmp_path):
        """The consumer child was spawned with no request-evidence environment, so once the ladder routes rung sends through the
        request-evidence client the child still could not write `request_evidence.jsonl` / `response_evidence.jsonl` under the
        attempt root. The child now carries `SISTEER_REQUEST_EVIDENCE_DIR` (the client's `ENV_DIR`) + the spool, both under ITS attempt root,
        plus the rung's arm -> seed map; a child without them is refused by name before the server; the files it writes join the inventory."""
        r = [x for x in draft["rungs"] if x.get("rung_mode") == lp.RUNG_MODE_CONSUME and x["engine"] == "arditi"][0]
        root = tmp_path / "attempt"
        kid = srv.rung_consumer_child(r, r["rung_consumer"], attempt_root=str(root), attempt_id="ta-1")
        assert kid["env"][lp.REQUEST_EVIDENCE_ENV] == str(root) and kid["env"][lp.SPOOL_ENV] == f"{root}/{lp.SPOOL_REL}"      # §R2.2: the LITERAL root
        assert lp.REQUEST_EVIDENCE_ENV == "SISTEER_REQUEST_EVIDENCE_DIR"
        assert [(a["arm"], a["s"]) for a in kid["arm_runs"]] == [(lp.ARDITI_ARM_ID, s) for s in (1, 2, 3)], "the Arditi multiplex's seed bases ride the attempt"
        assert lp.rung_child_env_problems(kid["env"], attempt_root=str(root), attempt_id="ta-1", seed_bases=[1, 2, 3]) == []
        # known-bads: no env at all; the evidence dir under ANOTHER attempt root; the 579c53a2 nesting under exp/request_evidence
        probs = lp.rung_child_env_problems({}, attempt_root=str(root))
        assert any("lacks SISTEER_REQUEST_EVIDENCE_DIR" in p for p in probs) and any(f"lacks {lp.SPOOL_ENV}" in p for p in probs)
        probs = lp.rung_child_env_problems({**kid["env"], lp.REQUEST_EVIDENCE_ENV: "/vol/other"}, attempt_root=str(root))
        assert any("is not this attempt root" in p for p in probs)
        probs = lp.rung_child_env_problems({**kid["env"], lp.REQUEST_EVIDENCE_ENV: f"{root}/exp/request_evidence"}, attempt_root=str(root))
        assert any("is not this attempt root" in p and "LITERAL root" in p for p in probs)
        # the pinned spelling is the client's own, when it is reachable
        client = next((p for p in (REPO / "src/sisteer/eval/request_evidence_client.py", WORKTREES / "seedrep-ws1dp/src/sisteer/eval/request_evidence_client.py") if p.is_file()), None)
        if client is not None:
            text = client.read_text()
            assert f'ENV_DIR = "{lp.REQUEST_EVIDENCE_ENV}"' in text and 'RESPONSE_EVIDENCE_FILENAME = "response_evidence.jsonl"' in text
        # the evidence files AT the attempt root are listed with rows + sha for the inventory
        root.mkdir(parents=True)
        (root / "request_evidence.jsonl").write_text('{"a": 1}\n{"b": 2}\n')
        files = srv.request_evidence_files(root)
        assert [f["path"] for f in files] == ["request_evidence.jsonl"] and files[0]["rows"] == 2 and files[0]["sha256"] == hashlib.sha256(b'{"a": 1}\n{"b": 2}\n').hexdigest()
        assert srv.request_evidence_files(tmp_path / "empty") == []
        assert srv.request_evidence_files(root / "exp" / "request_evidence") == []        # the old nesting is no longer read
        # the executor refuses a consumer whose env problems are named, before the standup (source pin on the order)
        import inspect
        src = inspect.getsource(srv._run_standup_row)
        assert src.index("RUNG_CHILD_ENV_MISSING") < src.index("write_json(root / EVIDENCE_NAME, evidence)") or src.index("RUNG_CHILD_ENV_MISSING") < src.index("stand_up") if "stand_up" in src else True

    def test_rung_env_round_trips_through_ws2es_real_rung_evidence_with_two_arms_and_three_seed_slots(self, srv, draft, RR, tmp_path):
        """At 579c53a2 `rung_child_env` pinned `arm: consumer` in the extra, and `RungEvidence.client_for()`
        (correctly) refused every production draw -- `the evidence environment pins arm 'consumer' but this record is 'none'`. The env now
        names neither `arm` nor `s` (the rung evidence stamps both per queue record), puts the evidence dir at the LITERAL attempt root, leaves
        the seed base unset on the Arditi multiplex; round-tripped through the REAL `RungEvidence.from_env` + `client_for` over the REAL
        request-evidence client with arms `none` / `arditi_allpos` and all three Arditi seed slots. The OLD env reproduces the refusal; our
        validator refuses it (and `s` in the extra, a seed pin on the multiplex, a pin disagreeing with the single base) by name BEFORE the server."""
        _request_evidence_client()
        ard = [x for x in draft["rungs"] if x.get("rung_mode") == lp.RUNG_MODE_CONSUME and x["engine"] == "arditi"][0]
        view = [x for x in draft["rungs"] if x.get("rung_mode") == lp.RUNG_MODE_CONSUME and x["engine"] == "s2"][0]
        root = tmp_path / "attempt"
        ka = srv.rung_consumer_child(ard, ard["rung_consumer"], attempt_root=str(root), attempt_id="ta-3")
        ev = RR.RungEvidence.from_env(leg=ard["leg"], attempt_id="ta-3", env=ka["env"])
        assert ev is not None and ev.out_dir == root and ev.pinned_arm is None and ev.pinned_seed_base is None
        for arm in ("none", lp.ARDITI_ARM_ID):
            for s in (1, 2, 3):
                c = ev.client_for({"item_key": f"k-{arm}-{s}", "arm": arm, "s": s})
                assert c.extra["arm"] == arm and c.extra["s"] == s and c.extra["attempt_id"] == "ta-3" and c.extra["leg"] == ard["leg"] and c.extra["standup_id"] == ard["standup_id"]
                assert c.request_path == root / "request_evidence.jsonl", "beside usage_ledger.jsonl and the engine log: ONE root (§R2.2)"
        assert len(ev._clients) == 6
        # a single-seed view MAY pin the seed base: every record agrees, a record of another seed refuses through the consumer's real check
        kv = srv.rung_consumer_child(view, view["rung_consumer"], attempt_root=str(root / "v"), attempt_id="ta-4")
        evv = RR.RungEvidence.from_env(leg=view["leg"], attempt_id="ta-4", env=kv["env"])
        assert evv.pinned_seed_base == 2 and evv.pinned_arm is None and evv.client_for({"item_key": "v", "arm": "none", "s": 2}).extra["s"] == 2
        with pytest.raises(RR.L.LadderRefusal, match="pins seed base 2"):
            evv.client_for({"item_key": "v1", "arm": "none", "s": 1})
        # the repro: the 579c53a2 extra (arm `consumer`) -> RungEvidence refuses the first real draw; the validator names it before the server
        extra = json.loads(ka["env"][lp.REQUEST_EVIDENCE_EXTRA_ENV])
        old = {**ka["env"], lp.REQUEST_EVIDENCE_EXTRA_ENV: json.dumps({**extra, "arm": "consumer"})}
        with pytest.raises(RR.L.LadderRefusal, match="pins arm 'consumer' but this record is 'none'"):
            RR.RungEvidence.from_env(leg=ard["leg"], attempt_id="ta-3", env=old).client_for({"item_key": "x", "arm": "none", "s": 1})
        assert any("pins arm='consumer'" in p for p in lp.rung_child_env_problems(old, attempt_root=str(root), attempt_id="ta-3", seed_bases=[1, 2, 3]))
        bad_s = {**ka["env"], lp.REQUEST_EVIDENCE_EXTRA_ENV: json.dumps({**extra, "s": 1})}
        assert any("pins s=1" in p for p in lp.rung_child_env_problems(bad_s, attempt_root=str(root), attempt_id="ta-3", seed_bases=[1, 2, 3]))
        with pytest.raises(RR.L.LadderRefusal, match="pins seed base 1"):
            RR.RungEvidence.from_env(leg=ard["leg"], attempt_id="ta-3", env=bad_s).client_for({"item_key": "x", "arm": lp.ARDITI_ARM_ID, "s": 2})
        probs = lp.rung_child_env_problems({**ka["env"], lp.REQUEST_EVIDENCE_SEED_BASE_ENV: "1"}, attempt_root=str(root), attempt_id="ta-3", seed_bases=[1, 2, 3])
        assert any("multiplexes seed bases [1, 2, 3]" in p for p in probs)
        probs = lp.rung_child_env_problems({**kv["env"], lp.REQUEST_EVIDENCE_SEED_BASE_ENV: "3"}, attempt_root=str(root / "v"), attempt_id="ta-4", seed_bases=[2])
        assert any("this row's seed base is 2" in p for p in probs)
        with pytest.raises(RR.L.LadderRefusal, match="names attempt_id='ta-3'"):
            RR.RungEvidence.from_env(leg=ard["leg"], attempt_id="ta-OTHER", env=ka["env"])
        assert lp.RUNG_EXTRA_FORBIDDEN == ("arm", "s")
        # the executor hands the row's seed bases to the validator (the multiplex check is live in production, not only here)
        import inspect
        assert "seed_bases=list(row.get(\"seed_bases\") or [])" in inspect.getsource(srv._run_standup_row)

    def test_revalidated_every_producer_rung_finds_its_consumer_tally_at_the_path_the_spawn_wrote(self, srv, draft, tmp_path):
        """The consumer child was labelled arm `*`, so the spawn wrote `<leg>__*__s0.log` while the lookup replaced `*` with
        `_` and read `<leg>_____s0.log` -> the tally was ALWAYS absent -> `ladder_enqueued.json` incomplete -> every producer rung rejected.
        Now ONE encoder (`child_log_name`) names the file for both, and the consumer child's arm is path-safe. Every PRODUCER rung tested."""
        import inspect
        producers = [r for r in draft["rungs"] if r.get("rung_mode") == lp.RUNG_MODE_CONSUME and (r.get("ladder") or {}).get("cap_next") is not None]
        assert len(producers) == 16 + 8, "static 4 legs x 4 engines at 8,192 + controls 4 engines x (512->2048, 2048->8192)"
        for r in producers:
            kid = srv.rung_consumer_child(r, r["rung_consumer"])
            name = srv.child_log_name(kid)
            assert kid["arm"] == srv.RUNG_CONSUMER_ARM == "consumer" and "*" not in name and "/" not in name and name == f"{r['eval_legs'][0]}__consumer__s0"
            d = tmp_path / r["standup_id"] / "children"
            d.mkdir(parents=True)
            (d / f"{name}.log").write_text("[rung-redraw] drawing\n" + json.dumps({"drawn": 3, "enqueued_next": 2, "closed": True, "covered": True}) + "\n")
            assert srv.consumer_tally_from_log(d / f"{name}.log")["enqueued_next"] == 2
            old_lookup = d / (f"{r['eval_legs'][0]}__*__s0.log".replace("*", "_"))                  # the repro's path
            assert old_lookup.name == f"{r['eval_legs'][0]}_____s0.log" and srv.consumer_tally_from_log(old_lookup) is None
        # a real child's stem is unchanged by the encoder (the reader reads `<eval_leg>__<arm>__s<s>.log`); a slash is still folded
        kid = _row(draft, "misalign", "s2")["children"][0]
        assert srv.child_log_name(kid) == f"{kid['eval_leg']}__{kid['arm']}__s{kid['s']}"
        assert srv.child_log_name({"eval_leg": "x", "arm": "a/b", "s": 1}) == "x__a_b__s1"
        src = inspect.getsource(srv._run_standup_row)
        assert src.count("child_log_name(") >= 2 and '__*__s0.log' not in src


def _copy_root(root, tmp_path, attempt_id, volroot, prod):
    import shutil
    dst = tmp_path / "committed" / attempt_id
    shutil.copytree(root, dst)
    inv = volroot / prod.inventory_path
    shutil.copyfile(inv, dst / "queue_inventory.json")
    return dst


# =========================================================================== P1-6: the rung consumer argv is the real CLI

class TestP1_6_rung_consumer_cli:
    def test_every_generated_rung_command_round_trips_through_the_real_parser(self, draft, RR, L):
        parser = RR.build_parser()
        seen = 0
        for r in draft["rungs"]:
            if r["rung_mode"] != lp.RUNG_MODE_CONSUME:
                assert r["rung_consumer"] is None and ((r["rung_mode"] == lp.RUNG_MODE_PETRI_RERUN and r["leg"] == "petri")
                                                       or (r["rung_mode"] == lp.RUNG_MODE_GSM8K_SUBSET and r["leg"] == "gsm8k"))   # the GSM8K assembly rung
                continue
            argv = [a.replace("{attempt_id}", "ta-1").replace("{call_id}", "fc-1").replace("{proxy_url}", "http://127.0.0.1:9/v1").replace("{attempt_root_rel}", "seedrep/x")
                     .replace("{attempt_root}", "/workspace/attempt") for a in r["rung_consumer"]]
            assert argv[1].endswith("scripts/sdfbase/seedrep/rung_redraw.py") and argv[2] == "consume"
            a = parser.parse_args(argv[2:])
            assert a.cmd == "consume" and a.leg == r["leg"] and a.view == r["engine"] and a.kind == lp.LADDER_KIND_OF_LEG[r["leg"]]
            assert a.shard == str(r["shard"]) == "0" and a.attempt_root_local == "/workspace/attempt"      # the slot, the LOCAL root
            assert a.cap_next == int(r["stage"]) and a.W7 == draft["W7"] and a.wave_uuid == draft["wave_uuid"] and a.vol_root == "/vol"
            assert a.attempt_id == "ta-1" and a.call_id == "fc-1" and a.api_key == "petri-proxy" and a.finish_producer is False
            assert a.speculative is (a.kind in ("static", "controls"))
            spec = L.QueueSpec.build(a.W7, a.wave_uuid, a.leg, a.view, a.kind, a.cap_next, [s.strip() for s in a.producers.split(",")])   # the consumer validates the ancestry
            assert spec.producers[0].key.stage == ("base" if a.cap_next == L.LADDERS[a.kind][1] else str(L.predecessor_cap(a.kind, a.cap_next)))
            assert [p.shard for p in spec.producers] == r["ladder"]["feeder_shards"]
            if r["ladder"]["cap_next"] is not None:
                assert a.rung_shard == "0" and a.rung_shards == "0" and a.attempt_root == "seedrep/x"
                assert spec.next_spec(["0"]).cap_next == r["ladder"]["cap_next"]
            else:
                assert a.rung_shard is None and spec.next_spec(["0"]) is None
            seen += 1
        assert seen == 32 + 12 + 4, "static 4 legs x 4 engines x 2 caps + controls 4 x 3 + misalign 4"
        # the repro: the OLD `--plan-row` invocation is not this parser's
        with pytest.raises(SystemExit):
            parser.parse_args(["--plan-row", "{}", "--attempt-root", "/x"])

    def test_shard_and_local_root_ride_every_consumer_argv_and_the_adopted_index_reaches_the_next_queue(self, srv, draft, RR):
        """EVERY rung consumer argv carries `--shard <slot>` (the ceiling rung too -- a later execution of the slot adopts an earlier
        one's elected results' evidence) and `--attempt-root-local` (the container-LOCAL root the adopted request pipeline is merged
        into); both parse through the consumer's REAL parser; a rung argv without `--shard` is refused by validate_plan by name; the
        rebuilt next-queue producer carries `adopted` from the consumer's tally; a consumer that exits non-zero on an adoption
        refusal is a FAILED child (no COMPLETE.json, never accepted)."""
        parser = RR.build_parser()
        seen = 0
        for r in draft["rungs"]:
            if r.get("rung_mode") != lp.RUNG_MODE_CONSUME:
                continue
            rc = r["rung_consumer"]
            assert lp._flag_value(rc, "--shard") == str(r["shard"]) and lp._flag_value(rc, "--attempt-root-local") == lp.ATTEMPT_ROOT_PLACEHOLDER == "{attempt_root}"
            argv = [a.replace("{attempt_id}", "ta-1").replace("{call_id}", "fc-1").replace("{proxy_url}", "http://x/v1").replace("{attempt_root_rel}", "seedrep/x")
                     .replace("{attempt_root}", srv.CONTAINER_ATTEMPT_ROOT) for a in rc]
            a = parser.parse_args(argv[2:])
            assert a.shard == str(r["shard"]) and a.attempt_root_local == "/workspace/attempt" == srv.CONTAINER_ATTEMPT_ROOT
            if r["ladder"]["cap_next"] is None:
                assert a.rung_shard is None and a.shard == "0", "the ceiling rung carries --shard too"
            seen += 1
        assert seen == 48
        bad = copy.deepcopy(draft)
        r = [x for x in bad["rungs"] if x.get("rung_mode") == lp.RUNG_MODE_CONSUME][0]
        i = r["rung_consumer"].index("--shard")
        del r["rung_consumer"][i:i + 2]
        probs = [p for p in lp.validate_plan(bad) if "--shard" in p]
        assert probs and r["standup_id"] in probs[0] and "c32d489c" in probs[0]
        r["rung_consumer"] += ["--shard", "0", "--attempt-root-local", "/elsewhere"]
        assert any("--attempt-root-local" in p for p in lp.validate_plan(bad))
        # the executor: the container-local root resolves the placeholder; the tally's adopted index reaches the rebuilt producer; a
        # non-zero consumer is a failed child before COMPLETE / acceptance
        import inspect
        src = inspect.getsource(srv._run_standup_row)
        assert "attempt_root=str(root)" in src and "k_argv, k_env = lp.resolve_spawn(k, spawn_context)" in src   # ONE resolution, {attempt_root} = the container-local root
        assert 'producer.adopted = list((tally or {}).get("adopted") or [])' in src
        assert src.index('failed = [c for c in children_out if c.get("rc") not in (0, None)]') < src.index("rung_next_producer(")
        assert src.index("if failed:") < src.index("rung_next_producer(")

    def test_width_and_volume_ride_every_consumer_argv(self, draft, RR):
        """At f4f3d465, `--width/--concurrency` is REQUIRED and `--volume` binds the mounted Volume's commit /
    reload. Every generated consumer carries `--width <the rung's attested S>` and `--volume twosi-fits`; the OLD argv (no width)
    exits 2 through the consumer's REAL parser; a drifted width or volume is refused by validate_plan by name."""
        parser = RR.build_parser()
        seen = 0
        for r in draft["rungs"]:
            if r.get("rung_mode") != lp.RUNG_MODE_CONSUME:
                continue
            rc = r["rung_consumer"]
            a = parser.parse_args(rc[2:])
            assert a.width == int(r["geometry"]["S"]) >= 1 and a.volume == lp.MODAL_VOLUME_NAME == "twosi-fits" and a.vol_root == "/vol"
            seen += 1
            old = [x for i, x in enumerate(rc[2:]) if x != "--width" and rc[2:][i - 1] != "--width"]
            with pytest.raises(SystemExit):
                parser.parse_args(old)
        assert seen == 32 + 12 + 4
        bad = copy.deepcopy(draft)
        r = _row(bad, "renellm", "s2", "8192")
        r["rung_consumer"][r["rung_consumer"].index("--width") + 1] = str(int(r["geometry"]["S"]) + 1)
        assert any("--width" in p and "attested S" in p for p in lp.validate_plan(bad))
        bad = copy.deepcopy(draft)
        r = _row(bad, "renellm", "s2", "8192")
        r["rung_consumer"][r["rung_consumer"].index("--volume") + 1] = "other-volume"
        assert any("--volume" in p and "twosi-fits" in p for p in lp.validate_plan(bad))
        with pytest.raises(ValueError, match="attested S >= 1"):
            lp.rung_consumer_argv(leg="renellm", engine="s2", stage=8192, shard=0, feeder_shards=["0"], rung_shards=["0"], next_cap=16384,
                                  W=W, W7=draft["W7"], wave_uuid=WAVE, timeout_s=3600, width=0)

    def test_validate_plan_refuses_a_drifted_consumer(self, draft):
        bad = copy.deepcopy(draft)
        r = _row(bad, "renellm", "s2", "8192")
        r["rung_consumer"].append("--finish-producer")
        assert any("rung consumer argv is not WS2-E's `consume`" in p for p in lp.validate_plan(bad))
        bad = copy.deepcopy(draft)
        r = _row(bad, "renellm", "s2", "8192")
        r["rung_consumer"][r["rung_consumer"].index("--cap-next") + 1] = "16384"
        assert any("rung consumer argv" in p for p in lp.validate_plan(bad))


# =========================================================================== P1-7: DRAFT == FINAL per logical arm / run

class TestP1_7_draft_final:
    S4 = {"multiturn_window": {"crescendo": 16384, "actorattack": 24576},
          "pools": {"count": 5, "assignment": ["crescendo", "crescendo", "actorattack", "actorattack", "pair"], "s_att": 12},
          "actorattack": {"forwards_per_goal": 26, "containers_per_view": 3, "containers_arditi": 1}, "crescendo": {"containers_per_view": 2, "containers_arditi": 1}}

    def test_the_s4_decisions_pass(self, inputs, draft):
        final = _plan(inputs, materialisation="final", s4=self.S4)
        assert lp.compare_draft_final(draft, final) == []
        assert len([r for r in final["standups"] if r["leg"] == "crescendo"]) == 7 and final["capacity_ledger"]["pools"]["containers"] == 5

    def test_revalidated_launch_order_rows_move_with_s4_but_the_stage_order_is_held(self, inputs, draft):
        """Five pools and seven Crescendo shards change `launch_order`'s ROW lists (4 -> 5 pools, 24 -> 38 multi-turn rows);
        that is an allowed S4 decision, so the projection keeps only the STAGE ordering -- which a reordered / renamed stage still fails."""
        final = _plan(inputs, materialisation="final", s4=self.S4)
        assert [len(st["rows"]) for st in draft["launch_order"]] != [len(st["rows"]) for st in final["launch_order"]]
        assert lp.compare_draft_final(draft, final) == []
        assert lp.scientific_projection(final)["launch_order_stages"] == [{"stage": st["stage"], "owner": st["owner"], "requires": st["requires"]} for st in final["launch_order"]]
        bad = copy.deepcopy(final)
        bad["launch_order"] = [bad["launch_order"][1], bad["launch_order"][0], bad["launch_order"][2]]
        assert any(p.startswith("/launch_order_stages") for p in lp.compare_draft_final(draft, bad))
        bad = copy.deepcopy(final)
        bad["launch_order"][1]["requires"] = []
        assert any(p.startswith("/launch_order_stages[1]/requires") for p in lp.compare_draft_final(draft, bad))

    @pytest.mark.parametrize("what", ["cap", "seed", "policy", "arm_set", "task", "target_cap_flag", "pin"])
    def test_revalidated_a_scientific_change_inside_a_multiturn_leg_fails(self, inputs, draft, what):
        final = _plan(inputs, materialisation="final", s4=self.S4)
        bad = copy.deepcopy(final)
        rows = [r for r in bad["standups"] if r["leg"] == "crescendo" and r["engine"] == "s1"]
        for r in rows:                                          # every shard of the logical run moves together (a generator bug would)
            for k in r["children"]:
                a = k["argv"]
                if what == "cap":
                    k["cap"] = 1024
                elif what == "target_cap_flag":
                    a[a.index("--target-max-new-tokens") + 1] = "1024"
                elif what == "seed":
                    a[a.index("--multiturn-target-seed-base") + 1] = "2"
                elif what == "policy":
                    a[a.index("--multiturn-target-decode") + 1] = "harness"
                elif what == "task":
                    k["task"] = "multi_turn/pair"
            if what == "arm_set":
                r["children"] = [k for k in r["children"] if k["arm"] != "prompted__safety_refusal"]
            if what == "pin":
                r["objects"]["constv2-mix-full__psr_ce_all"]["object_sha256"] = "e" * 64
        diffs = lp.compare_draft_final(draft, bad)
        assert diffs, what
        assert any("crescendo/s1/base" in d for d in diffs), (what, diffs[:3])
        # the OLD rule exempted the whole leg: none of these would have been seen
        assert all(d.startswith("/rows/crescendo/s1/base") for d in diffs), diffs[:3]

    def test_a_shard_that_projects_differently_from_its_siblings_is_a_conflict(self, inputs, draft):
        final = _plan(inputs, materialisation="final", s4=self.S4)
        bad = copy.deepcopy(final)
        r = [x for x in bad["standups"] if x["leg"] == "crescendo" and x["engine"] == "s1"][1]
        r["children"][0]["argv"][r["children"][0]["argv"].index("--multiturn-target-seed-base") + 1] = "3"
        diffs = lp.compare_draft_final(draft, bad)
        assert any("CONFLICT" in d for d in diffs)


# =========================================================================== P1-8: the watcher's required control-key set

class TestP1_8_watcher_coverage:
    def test_required_keys_are_derived_from_the_plan_per_scope(self, draft):
        base = lp.required_accepted_keys(draft, "controls", "base")
        rungs = lp.required_accepted_keys(draft, "controls", "rungs")
        allk = lp.required_accepted_keys(draft, "controls", "all")
        assert len(base) == 4 and len(rungs) == 12 and len(allk) == 16 and set(base) | set(rungs) == set(allk)
        assert all("|controls|" in k and k.endswith("|base|0") for k in base)
        with pytest.raises(ValueError, match="scope"):
            lp.required_accepted_keys(draft, "controls", "everything")
        assert lp.required_accepted_keys(draft, "crescendo", "base") and len(lp.required_accepted_keys(draft, "crescendo", "base")) == 14

    def test_revalidated_four_retries_of_one_key_never_reach_done(self, draft):
        import time as _t
        rows = [r for r in draft["standups"] if r["leg"] == "pasttense"]
        req = lp.required_accepted_keys(draft, "pasttense", "base")
        one = rows[0]
        calls = [{"call_id": f"fc-{i}", "standup_id": one["standup_id"], "leg": "pasttense", "engine": one["engine"], "stage": "base", "shard": 0,
                  "control_key": one["control_key"], "accepted_key": one["accepted_key"], "wave_uuid": WAVE} for i in range(4)]
        snap = {c["call_id"]: {"done": True, "summary": {"status": "DONE"}, "accepted": "a-1"} for c in calls}
        st = sw.WatchState()
        ev = sw.classify(now=_t.time(), leg="pasttense", calls=calls, snapshot=snap, state=st, stall_min=20, rows_flat_min=40, progress_min=30, required_keys=req)
        assert not any(e.startswith("DONE") for e in ev), "the repro: four call records for one key read as accepted=4 >= 4 shards"
        calls_all = [{"call_id": f"fc-{r['engine']}", "standup_id": r["standup_id"], "leg": "pasttense", "engine": r["engine"], "stage": "base", "shard": 0,
                      "control_key": r["control_key"], "accepted_key": r["accepted_key"], "wave_uuid": WAVE} for r in rows]
        snap = {c["call_id"]: {"done": True, "summary": {"status": "DONE"}, "accepted": "a"} for c in calls_all}
        ev = sw.classify(now=_t.time(), leg="pasttense", calls=calls_all, snapshot=snap, state=sw.WatchState(), stall_min=20, rows_flat_min=40, progress_min=30, required_keys=req)
        assert [e for e in ev if not e.startswith(("WAVE_LAUNCHED ", "WAVE_DONE "))] == ["DONE pasttense scope=base required=4 accepted=4"]
        # the rung scope is explicit: base acceptance alone is not rung coverage
        req_all = lp.required_accepted_keys(draft, "pasttense", "all")
        ev = sw.classify(now=_t.time(), leg="pasttense", calls=calls_all, snapshot=snap, state=sw.WatchState(), stall_min=20, rows_flat_min=40, progress_min=30,
                         required_keys=req_all, scope="all")
        assert not any(e.startswith("DONE") for e in ev)


# =========================================================================== P1-9: Petri completion evidence, the tunnel record

class TestP1_9_petri_evidence:
    def test_tunnel_record_round_trips_through_ws2f_validator_and_attempt_paths(self, srv, draft, drive):
        """At 7d902dff: `tunnel_record_problems(rec, view=, stage=, expected_control_key=, w7=)` REQUIRES `control_key`
        (== the drive's `key_for("")`, its `control_key(W7, wave_uuid, "petri", view, stage, shard)`), `attempt_root_on_volume` under
        `/vol/seedrep/<W7>/` with `attempt_id` its last component, the serving dir under `<root>/exp/`. The record is built from the
        app's REAL attempt-root builder, never a hand-written path."""
        row = _row(draft, "petri", "s1")
        attempt_id = "ta-01HX-abc123def456"
        W7 = "2222222"
        root = srv.attempt_root_for(srv.results_root(W7), row, attempt_id)
        assert root == f"/vol/seedrep/{W7}/exp/seedrep_battery_modal/petri/s1/shard00/attempts/{attempt_id}" and root.startswith(f"/vol/seedrep/{W7}/")
        rec = srv.petri_tunnel_record(url="https://x.modal.host", host="x.modal.host", port=443, proxy_port=9000, relay_port=19000, call_id="fc-1", task_id="ta-01HX",
                                      attempt_id=attempt_id, attempt_root_vol=root, row=row)
        expected_key = drive.control_key(W7, WAVE, "petri", "s1", "base", "0", "")           # the drive's key_for("") spelling, exactly
        assert rec["control_key"] == expected_key == row["control_key"]
        assert drive.tunnel_record_problems(rec, view="s1", stage="base", expected_control_key=expected_key, w7=W7) == []
        assert set(drive.TUNNEL_REQUIRED_KEYS) <= set(rec) and rec["attempt_root_on_volume"] == root and rec["attempt_id"] == attempt_id
        ep = {"routing_table": f"{srv.CONTAINER_EXP_ROOT}/serve/routing_table.json", "cells_served": f"{srv.CONTAINER_EXP_ROOT}/serve/cells_served.json"}
        paths = drive.attempt_paths(rec, ep, rec["container_exp_root"])
        assert paths == {"attempt_root": root, "attempt_exp": f"{root}/exp", "routing_table": f"{root}/exp/serve/routing_table.json",
                         "cells_served": f"{root}/exp/serve/cells_served.json"}
        assert rec["serving_dir_on_volume"] == f"{root}/exp/serve"
        # known-bad per REQUIRED field (the drive's list, read from the peer): each absence is named
        for k in drive.TUNNEL_REQUIRED_KEYS:
            broken = {kk: v for kk, v in rec.items() if kk != k}
            probs = drive.tunnel_record_problems(broken, view="s1", stage="base", expected_control_key=expected_key, w7=W7)
            assert any(f"missing {k}" in p for p in probs), (k, probs)
        # known-bads: another wave's key; a root under another W7; an attempt id that is not the root's last component
        other = drive.control_key(W7, SMOKE_WAVE, "petri", "s1", "base", "0", "")
        probs = drive.tunnel_record_problems({**rec, "control_key": other}, view="s1", stage="base", expected_control_key=expected_key, w7=W7)
        assert any("another wave" in p for p in probs), probs
        foreign = root.replace(f"/vol/seedrep/{W7}/", "/vol/seedrep/3333333/")
        probs = drive.tunnel_record_problems({**rec, "attempt_root_on_volume": foreign, "serving_dir_on_volume": f"{foreign}/exp/serve"},
                                             view="s1", stage="base", expected_control_key=expected_key, w7=W7)
        assert any(f"not under /vol/seedrep/{W7}/" in p for p in probs), probs
        with pytest.raises(ValueError, match="does not end in the attempt id"):
            srv.petri_tunnel_record(url="https://x", host="x", port=1, proxy_port=1, relay_port=2, call_id="c", task_id="t", attempt_id="other", attempt_root_vol=root, row=row)

    def test_cross_ws_success_and_stale_attempt_through_ws2f_real_publisher(self, srv, draft, drive):
        """At 1e132e41, the completion contract REVISED: ONE atomic bundle at `completion_key(K, attempt_id)` =
    `<K>/<attempt_id>/completion.json` (schema `seedrep_petri_drive_completion/1` = every round-3 verdict field + `stop: {content, why}`);
    the generic `<K>/drive_verdict.json` + `<K>/STOP` pair only under `--legacy-generic-completion` (default OFF). The reader watches ONLY its
    own attempt's key, derived from the publisher's function (pinned equal here): a released drive reads `complete`, a --teardown-after-failure
    `incomplete`, a HELD drive publishes nothing (`waiting`), a STALE drive (a newer attempt's record under K) publishes nothing; a bundle
    under ANOTHER attempt's key is never terminal for this attempt; the generic pair alone is ignorable noise (the interlock: a reader of
    the generic keys would wait until its deadline); the legacy opt-in writes the bundle FIRST. Skips by name until the publisher is in
    the tree / worktree."""
        for sym in ("publish_completion", "completion_key", "completion_bundle", "COMPLETION_SCHEMA", "COMPLETION_NAME", "BINDING_FIELDS", "drive_rc"):
            if not hasattr(drive, sym):
                pytest.skip(f"WS2-F's attempt-namespaced publisher ({sym}) has not landed in the tree or the seedrep-ws2f worktree")

        class Sent(FakeDict):
            def __setitem__(self, k, v):
                self.put(k, v)

            def __getitem__(self, k):
                return FakeDict.store[self.name][k]
        row = _row(draft, "petri", "s1")
        W7, attempt_id, key = "2222222", "ta-01HX-abc123def456", row["control_key"]
        root = srv.attempt_root_for(srv.results_root(W7), row, attempt_id)
        rec = srv.petri_tunnel_record(url="https://x.modal.host", host="x.modal.host", port=443, proxy_port=9000, relay_port=19000, call_id="fc-1", task_id="ta-01HX",
                                      attempt_id=attempt_id, attempt_root_vol=root, row=row)
        bound = {k: rec[k] for k in drive.BINDING_FIELDS}
        ck = srv.petri_completion_key(key, attempt_id)
        assert ck == drive.completion_key(key, attempt_id) == f"{key}/{attempt_id}/completion.json", "ONE spelling of the attempt-namespaced key"
        assert drive.COMPLETION_SCHEMA == srv.PETRI_COMPLETION_SCHEMA == "seedrep_petri_drive_completion/1" and drive.COMPLETION_NAME == srv.PETRI_COMPLETION_NAME
        identity = {"W": draft["W"], "W7": W7, "wave_uuid": WAVE, "view": "s1", "stage": "base", "shard": "0", "petri_pass": ""}
        released = {"counts": {"complete": 8, "incomplete": 0, "failed": 0}, "nruns": 8, "release": True, "detail": []}
        failed = {"counts": {"complete": 7, "incomplete": 0, "failed": 1}, "nruns": 8, "release": False, "detail": []}
        kw = dict(key=key, bound=bound, drive_id="drv-1", identity=identity, log=lambda *a, **k: None)
        FakeDict.store.clear()
        sent = Sent("sentinels")
        sent[f"{key}/{srv.PETRI_TUNNEL_FILE}"] = json.dumps(rec)
        assert srv.petri_completion_evidence(sent, key, attempt_id)["state"] == "waiting"
        out = drive.publish_completion(sent, current_tunnel=json.loads(sent[f"{key}/{srv.PETRI_TUNNEL_FILE}"]), verdict=released, write_stop=True, stop_why="release", **kw)
        assert out["binding"] == "bound" and out["published"] == [ck] and out["completion_key"] == ck and drive.drive_rc(released, out) == 0
        assert f"{key}/drive_verdict.json" not in FakeDict.store["sentinels"] and f"{key}/STOP" not in FakeDict.store["sentinels"], "nothing generic by default"
        ev = srv.petri_completion_evidence(sent, key, attempt_id)
        assert ev["state"] == "complete" and ev["kind"] == "completion" and ev["key"] == ck and ev["noise"] == []
        assert ev["evidence"]["attempt_id"] == attempt_id and ev["evidence"]["verdict"]["counts"] == released["counts"] and ev["evidence"]["schema"] == srv.PETRI_COMPLETION_SCHEMA
        st = srv.parse_stop_sentinel(ev["evidence"]["stop"]["content"])                       # the exact STOP sentinel rides inside the bundle
        assert st["attempt"] == attempt_id and st["failed"] == 0 and st["drive_id"] == "drv-1" and ev["evidence"]["stop"]["why"] == "release"
        import tempfile
        with tempfile.TemporaryDirectory() as td:                                            # mirrored: the bundle FIRST, then the STOP
            serve = Path(td)
            assert srv.mirror_bound_completion(sent, serve, key, attempt_id) == [srv.PETRI_COMPLETION_NAME, "STOP"]
            assert json.loads((serve / srv.PETRI_COMPLETION_NAME).read_text())["attempt_id"] == attempt_id and (serve / "STOP").read_text() == ev["evidence"]["stop"]["content"]
        # a failed leg torn down by --teardown-after-failure: bound, INCOMPLETE (rc 1, never accepted)
        FakeDict.store.clear()
        sent = Sent("sentinels")
        sent[f"{key}/{srv.PETRI_TUNNEL_FILE}"] = json.dumps(rec)
        out = drive.publish_completion(sent, current_tunnel=rec, verdict=failed, write_stop=True, stop_why="teardown-after-failure", **kw)
        ev = srv.petri_completion_evidence(sent, key, attempt_id)
        assert out["binding"] == "bound" and ev["state"] == "incomplete" and ev["evidence"]["stop"]["why"] == "teardown-after-failure" and drive.drive_rc(failed, out) == 1
        # a HELD drive (no STOP decided) publishes nothing: the proxy stays up
        FakeDict.store.clear()
        sent = Sent("sentinels")
        sent[f"{key}/{srv.PETRI_TUNNEL_FILE}"] = json.dumps(rec)
        out = drive.publish_completion(sent, current_tunnel=rec, verdict=failed, write_stop=False, stop_why="held", **kw)
        assert out["published"] == [] and srv.petri_completion_evidence(sent, key, attempt_id)["state"] == "waiting"
        # STALE: a newer attempt's record now holds K -> the drive publishes nothing (rc 2); the new attempt keeps waiting
        new_id = "ta-01HX-newattempt0001"
        rec2 = srv.petri_tunnel_record(url="https://y.modal.host", host="y.modal.host", port=443, proxy_port=9000, relay_port=19000, call_id="fc-2", task_id="ta-01HX",
                                       attempt_id=new_id, attempt_root_vol=srv.attempt_root_for(srv.results_root(W7), row, new_id), row=row)
        FakeDict.store.clear()
        sent = Sent("sentinels")
        sent[f"{key}/{srv.PETRI_TUNNEL_FILE}"] = json.dumps(rec2)
        out = drive.publish_completion(sent, current_tunnel=rec2, verdict=released, write_stop=True, stop_why="release", **kw)
        assert out["binding"] == "stale" and out["published"] == [] and drive.drive_rc(released, out) == 2
        assert srv.petri_completion_evidence(sent, key, new_id)["state"] == "waiting"
        # KNOWN-BAD: a bundle under ANOTHER attempt's key is never terminal for this attempt (it is not even read)
        bundle_a = drive.completion_bundle(bound=bound, drive_id="drv-1", verdict=released, identity=identity, stop_why="release")
        sent[drive.completion_key(key, attempt_id)] = json.dumps(bundle_a)
        ev = srv.petri_completion_evidence(sent, key, new_id)
        assert ev["state"] == "waiting" and ev["noise"] == [] and srv.petri_completion_evidence(sent, key, attempt_id)["state"] == "complete"
        # KNOWN-BAD: a bundle under OUR key that names another attempt / another schema / another control key is noise, never terminal
        sent[srv.petri_completion_key(key, new_id)] = json.dumps(bundle_a)
        ev = srv.petri_completion_evidence(sent, key, new_id)
        assert ev["state"] == "waiting" and any(f"names attempt {attempt_id!r}" in n for n in ev["noise"]) and attempt_id in ev["why"]
        sent[srv.petri_completion_key(key, new_id)] = json.dumps({**bundle_a, "attempt_id": new_id, "schema": "seedrep_petri_drive_verdict/1"})
        assert any("schema" in n for n in srv.petri_completion_evidence(sent, key, new_id)["noise"])
        sent[srv.petri_completion_key(key, new_id)] = json.dumps({**bundle_a, "attempt_id": new_id, "control_key": key + "x"})
        assert any("control_key" in n for n in srv.petri_completion_evidence(sent, key, new_id)["noise"])
        sent[srv.petri_completion_key(key, new_id)] = "not json"
        assert srv.petri_completion_evidence(sent, key, new_id)["state"] == "waiting" and srv.mirror_bound_completion(sent, Path("/nonexistent"), key, new_id) == []
        sent.pop(srv.petri_completion_key(key, new_id))
        # the INTERLOCK: the generic pair alone (a reader of those keys would wait until its deadline) is ignorable noise here, never terminal
        sent[f"{key}/drive_verdict.json"] = json.dumps({**{k: v for k, v in bundle_a.items() if k != "stop"}, "attempt_id": new_id, "schema": "seedrep_petri_drive_verdict/1"})
        sent[f"{key}/STOP"] = drive.sentinel_content("STOP", "drv-2", drive.stop_sentinel_extra(released["counts"], new_id, "release")).decode()
        ev = srv.petri_completion_evidence(sent, key, new_id)
        assert ev["state"] == "waiting" and len(ev["noise"]) == 2 and all("generic (legacy)" in n for n in ev["noise"])
        sent.pop(f"{key}/drive_verdict.json"); sent.pop(f"{key}/STOP")
        # THE RACE: B installed its record; A's bundle lands under A's key -> B unaffected; B's own release
        # completes B, mirrored bundle-first; with the legacy opt-in the bundle is written FIRST and the generic pair after it
        bound2 = {k: rec2[k] for k in drive.BINDING_FIELDS}
        with tempfile.TemporaryDirectory() as td:
            serve = Path(td)
            assert srv.mirror_bound_completion(sent, serve, key, new_id) == [] and not (serve / "STOP").exists()
            out2 = drive.publish_completion(sent, current_tunnel=rec2, key=key, bound=bound2, drive_id="drv-2", verdict=released, identity={**identity},
                                            write_stop=True, stop_why="release", legacy_generic=True, log=lambda *a, **k: None)
            assert out2["binding"] == "bound" and out2["published"] == [srv.petri_completion_key(key, new_id), f"{key}/drive_verdict.json", f"{key}/STOP"]
            ev2 = srv.petri_completion_evidence(sent, key, new_id)
            assert ev2["state"] == "complete" and ev2["evidence"]["attempt_id"] == new_id and ev2["kind"] == "completion" and ev2["noise"] == []
            assert srv.mirror_bound_completion(sent, serve, key, new_id) == [srv.PETRI_COMPLETION_NAME, "STOP"]
            assert json.loads((serve / srv.PETRI_COMPLETION_NAME).read_text())["attempt_id"] == new_id and f"attempt={new_id}" in (serve / "STOP").read_text()
            assert srv.petri_completion_evidence(sent, key, attempt_id)["state"] == "complete", "A's bundle under A's key is undisturbed"
        # the caveat: an all-digit attempt id must NOT be int-coerced -- only the three counts are integers
        digits = "1234567890"
        st = srv.parse_stop_sentinel(drive.sentinel_content("STOP", "20260918T000000Z.42", drive.stop_sentinel_extra(released["counts"], digits, "release")))
        assert st["attempt"] == digits and isinstance(st["attempt"], str) and st["drive_id"] == "20260918T000000Z.42" and st["complete"] == 8 and isinstance(st["complete"], int)
        assert srv.STOP_COUNT_KEYS == ("complete", "incomplete", "failed")
        with pytest.raises(ValueError, match="path-safe attempt id"):
            srv.petri_completion_key(key, "a/b")

    def test_revalidated_timeout_and_cancel_are_never_success_only_bound_evidence_is(self, srv):
        FakeDict.store.clear()
        s = FakeDict("sent")
        key = "W/wave/petri/s1/base/0"
        ck = srv.petri_completion_key(key, "ta-1")
        assert ck == f"{key}/ta-1/completion.json" and srv.petri_completion_evidence(s, key, "ta-1")["state"] == "waiting"
        # the generic (legacy) pair, even bound to this attempt, is not read: at 1e132e41 it is written only under --legacy-generic-completion
        s[f"{key}/STOP"] = "STOP 2026-09-18T00:00:00Z drive=d1 complete=8 incomplete=0 failed=0 attempt=ta-1"
        s[f"{key}/drive_verdict.json"] = json.dumps({"attempt_id": "ta-1", "drive_id": "d1", "verdict": {"counts": {"complete": 8, "incomplete": 0, "failed": 0}, "release": True}})
        ev = srv.petri_completion_evidence(s, key, "ta-1")
        assert ev["state"] == "waiting" and len(ev["noise"]) == 2 and all("generic (legacy)" in n for n in ev["noise"])
        bundle = {"schema": "seedrep_petri_drive_completion/1", "attempt_id": "ta-1", "control_key": key, "drive_id": "d1", "attempt_root_on_volume": "/vol/x",
                  "verdict": {"counts": {"complete": 8, "incomplete": 0, "failed": 0}, "nruns": 8, "release": True, "detail": []}, "stop_why": "release",
                  "stop": {"content": "STOP 2026-09-18T00:00:00Z drive=d1 complete=8 incomplete=0 failed=0 attempt=ta-1\n", "why": "release"}}
        s[ck] = json.dumps(bundle)
        assert srv.petri_completion_evidence(s, key, "ta-1")["state"] == "complete"
        s[ck] = json.dumps({**bundle, "verdict": {**bundle["verdict"], "counts": {"complete": 7, "incomplete": 0, "failed": 1}, "release": False}, "stop_why": "teardown-after-failure"})
        assert srv.petri_completion_evidence(s, key, "ta-1")["state"] == "incomplete"
        s[ck] = json.dumps({**bundle, "verdict": {**bundle["verdict"], "release": False}})                 # a withheld release is never complete
        assert srv.petri_completion_evidence(s, key, "ta-1")["state"] == "incomplete"
        s[ck] = json.dumps({**bundle, "attempt_id": "ta-0"})                                                  # another attempt's bundle under OUR key: noise
        ev = srv.petri_completion_evidence(s, key, "ta-1")
        assert ev["state"] == "waiting" and any("names attempt 'ta-0'" in n for n in ev["noise"])
        s[srv.petri_completion_key(key, "ta-2")] = json.dumps({**bundle, "attempt_id": "ta-2"})              # a bundle under ANOTHER attempt's key: never terminal
        assert srv.petri_completion_evidence(s, key, "ta-1")["state"] == "waiting" and srv.petri_completion_evidence(s, key, "ta-2")["state"] == "complete"
        # the proxy's rc rule: only `complete` is rc 0 -- deadline (timeout) and the wave stop (cancelled) are failures
        src = DRIVER.read_text()
        assert 'rc = 0 if outcome["state"] == "complete" else 1' in src
        assert 'outcome.update(state="cancelled"' in src and '"state": "timeout"' in src
        # stale sentinels of a prior attempt are cleared before the tunnel record is published
        popped = srv.clear_stale_sentinels(s, key)
        assert set(popped) == {"STOP", "drive_verdict.json"} and s.get(f"{key}/STOP") is None
        assert src.index("clear_stale_sentinels(sentinels, key)") < src.index('sentinels[f"{key}/{PETRI_TUNNEL_FILE}"] = json.dumps(rec)')
        # the Petri rerun row dispatches to the proxy wait, not the rung consumer
        assert src.index("if is_petri:\n        kids = []") < src.index("elif is_rung:")


# =========================================================================== P2-10 (+ F-O): metrics through the proxy's pass-through, verified

class TestP2_10_metrics:
    def test_url_from_proxy_port_and_a_verified_scrape(self, srv):
        # F-O: the scrape reads the ROUTING PROXY's /metrics pass-through (proxy_port), never the bearer-guarded engine (vllm_port)
        assert srv.metrics_url_from_endpoint({"vllm_port": 8123, "proxy_port": 9000, "proxy_url": "http://127.0.0.1:9000/v1"}) == "http://127.0.0.1:9000/metrics"
        assert srv.metrics_url_from_endpoint({"vllm_port": 8123, "proxy_url": "http://127.0.0.1:9001/v1"}) == "http://127.0.0.1:9001/metrics"
        with pytest.raises(ValueError, match="proxy_port"):
            srv.metrics_url_from_endpoint({"vllm_port": 8123})                                   # an engine port alone names no scrapable target
        good = "# HELP vllm:num_requests_running x\nvllm:num_requests_running{model=\"m\"} 12.0\nvllm:num_requests_waiting{model=\"m\"} 3.0\nvllm:generation_tokens_total 100.0\n"
        row, probs = srv.verify_metrics_scrape("u", lambda u: good)
        assert probs == [] and row == {"vllm:num_requests_running": 12.0, "vllm:num_requests_waiting": 3.0, "vllm:generation_tokens_total": 100.0}
        _, probs = srv.verify_metrics_scrape("u", lambda u: "<html>404 not found</html>")
        assert probs and "not a vLLM /metrics body" in probs[0]

        def boom(u):
            raise ConnectionError("refused")
        _, probs = srv.verify_metrics_scrape("u", boom)
        assert probs == ["metrics scrape u failed: ConnectionError: refused"]
        src = DRIVER.read_text()
        assert 'return fail("METRICS_UNSCRAPABLE"' in src


# =========================================================================== P2-11: progress from the attempt's own outputs

class TestP2_11_progress:
    def test_producer_to_watcher(self, srv, tmp_path):
        import time as _t
        root = tmp_path / "attempt"
        (root / "exp/transcripts_results/run/pasttense/none/shard00").mkdir(parents=True)
        assert srv.count_progress(root) == {"rows_written": 0, "stream_files": 0, "children_exited": 0}
        p = root / "exp/transcripts_results/run/pasttense/none/shard00/responses.jsonl"
        p.write_text("{}\n{}\n{}\n")
        (root / "exp/other.jsonl").write_text("{}\n")                          # not a per-result stream: not counted
        (root / "driver_inflight.jsonl").write_text(json.dumps({"event": "start"}) + "\n" + json.dumps({"event": "exit", "rc": 0}) + "\n")
        prog = srv.count_progress(root)
        assert prog == {"rows_written": 3, "stream_files": 1, "children_exited": 1}
        hb = srv.heartbeat_payload(attempt_id="a", phase="drivers", running=45, progress=prog)
        assert hb["rows_written"] == 3 and hb["children_exited"] == 1
        # the watcher: rows grow -> PROGRESS carries them and no STALL fires at 41 min (the repro: rows_written stayed 0 forever)
        calls = [{"call_id": "fc-0", "standup_id": "pasttense__s1__base__shard00", "leg": "pasttense", "engine": "s1", "stage": "base", "shard": 0,
                  "control_key": "k", "accepted_key": "a|k", "wave_uuid": "w"}]
        st = sw.WatchState()
        t0 = 1_800_000_000.0
        iso = lambda t: _t.strftime("%Y-%m-%dT%H:%M:%SZ", _t.gmtime(t))  # noqa: E731
        ev = sw.classify(now=t0, leg="pasttense", calls=calls, snapshot={"fc-0": {"done": False, "heartbeat": {**hb, "t": iso(t0)}}}, state=st,
                         stall_min=20, rows_flat_min=40, progress_min=30, required_keys={"a|k": "x"})
        assert any(e.startswith("PROGRESS") and "rows=3" in e for e in ev)
        with p.open("a") as fh:
            fh.write("{}\n" * 5)
        hb2 = srv.heartbeat_payload(attempt_id="a", phase="drivers", running=45, progress=srv.count_progress(root))
        ev = sw.classify(now=t0 + 41 * 60, leg="crescendo", calls=[{**calls[0], "leg": "crescendo"}], snapshot={"fc-0": {"done": False, "heartbeat": {**hb2, "t": iso(t0 + 41 * 60)}}},
                         state=st, stall_min=20, rows_flat_min=40, progress_min=30, required_keys={"a|k": "x"})
        assert not any(e.startswith("STALL") for e in ev) and hb2["rows_written"] == 8


# =========================================================================== P2-12: shards across every pool of an attack

class TestP2_12_pools:
    def test_round_robin_and_coverage(self, draft):
        cr = [r for r in draft["standups"] if r["leg"] == "crescendo"]
        pools = {r["pool"] for r in cr}
        assert pools == {"seedrep-attacker-0", "seedrep-attacker-1"}, "the repro: setdefault kept pool 0 for all 14 shards"
        for r in cr:
            ph = {k["argv"][k["argv"].index("--attacker-endpoint") + 1] for k in r["children"]}
            assert ph == {f"{{pool_url:{r['pool']}}}"}
        assert [r["pool"] for r in cr if r["engine"] == "s1"] == ["seedrep-attacker-0", "seedrep-attacker-1", "seedrep-attacker-0", "seedrep-attacker-1"]
        assert {r["pool"] for r in draft["standups"] if r["leg"] == "pair"} == {"seedrep-attacker-3"}
        assert lp.pool_coverage_problems(draft) == []
        bad = copy.deepcopy(draft)
        for r in bad["standups"]:
            if r["leg"] == "crescendo":
                r["pool"] = "seedrep-attacker-0"
                for k in r["children"]:
                    k["argv"][k["argv"].index("--attacker-endpoint") + 1] = "{pool_url:seedrep-attacker-0}"
        probs = lp.pool_coverage_problems(bad)
        assert any("seedrep-attacker-1" in p and "receive no shard" in p for p in probs)
        assert probs == [p for p in lp.validate_plan(bad) if "pool" in p]


# =========================================================================== overnight operation

class TestD73_overnight_operation:
    """(1) exactly ONE automatic re-attempt for an INFRA-class failure, HOLD on a second failure or ANY correctness refusal, the counter
    durable per standup; (2) no container waits past its deadline (the first failed child tears its siblings down); (3) the watcher's three
    machine-readable summary lines WAVE_LAUNCHED / WAVE_DONE / HOLD; (4) `--go` refuses without a fresh S0 PASS for this W and plan, or
    with S5 bars unmet / clamped calls."""

    def test_failure_classes_and_the_one_reattempt_rule(self):
        for st in ("LOST_CONTAINER", "PREEMPTED", "ENGINE_DIED", "TRANSPORT_DEATH", "CALL_LOST", "OUTPUT_EXPIRED_NO_RESULT"):
            assert lp.failure_class(st) == "infra"
        assert lp.failure_class(None, error="modal.exception.RemoteError: the container was lost (preempted)") == "infra"
        assert lp.failure_class(None, error="OutputExpiredError: no durable RESULT / COMPLETE.json under /vol/x -- the attempt left no verdict") == "infra"
        for st in ("FIDELITY_REFUSED", "READINESS_MISSING", "DRAFT_FINAL_MISMATCH", "RECONCILIATION_REFUSED", "AUDIT_REFUSED", "DRIVERS_FAILED",
                   "RUNG_CHILD_ENV_MISSING", "PUBLICATION_FAILED", "GEOMETRY_MISMATCH", "SOMETHING_NEW", None):
            assert lp.failure_class(st) == "correctness", st
        assert lp.failure_class(None, error="AssertionError: adapter sha mismatch") == "correctness"
        d = lp.reattempt_decision(None, status="LOST_CONTAINER")
        assert d["action"] == "reattempt" and d["reattempts"] == 1 == lp.D73_MAX_AUTO_REATTEMPTS and "new immutable attempt root" in d["reason"]
        d2 = lp.reattempt_decision({"reattempts": 1}, status="LOST_CONTAINER")
        assert d2["action"] == "hold" and "second infra failure" in d2["reason"] and d2["reattempts"] == 1
        d3 = lp.reattempt_decision(None, status="DRIVERS_FAILED")
        assert d3["action"] == "hold" and "correctness refusal never earns a re-attempt" in d3["reason"]
        d4 = lp.reattempt_decision({"reattempts": 0, "held": True, "hold_reason": "x"}, status="LOST_CONTAINER")
        assert d4["action"] == "hold" and "already HELD" in d4["reason"]
        assert "reattempts" in lp.DICT_PURPOSES

    def test_the_policy_pass_spawns_once_then_holds_durably_and_idempotently(self, srv, draft, tmp_path):
        rows = {r["standup_id"]: r for r in draft["standups"]}
        sid = "pasttense__s1__base__shard00"
        FakeDict.store.clear()
        ledger = FakeDict("reattempts")
        spawned = []

        def spawn(row, intent):
            spawned.append(row["standup_id"])
            assert intent["standup_id"] == row["standup_id"] and intent["launch_plan_sha"] == draft["launch_plan_sha"]
            return f"fc-re{len(spawned)}"
        sha = draft["launch_plan_sha"]
        calls = [{"call_id": "fc-1", "standup_id": sid, "accepted_key": rows[sid]["accepted_key"], "control_key": rows[sid]["control_key"], "wave_uuid": draft["wave_uuid"],
                  "launch_plan_sha": sha}]
        snap = {"fc-1": {"done": True, "summary": None, "error": "OutputExpiredError: no durable RESULT / COMPLETE.json under /vol/x -- the attempt left no verdict", "accepted": None}}
        out = srv.apply_reattempt_policy(draft, tmp_path, calls=calls, snapshot=snap, rows_by_sid=rows, ledger=ledger, spawn=spawn, log=lambda *a: None)
        assert [d["action"] for d in out] == ["reattempt"] and spawned == [sid]
        e = srv.reattempt_ledger_get(ledger, sid)
        assert e["reattempts"] == 1 and e["held"] is False and e["judged_calls"] == ["fc-1"] and e["history"][0]["reattempt_call_id"] == "fc-re1" and e["schema"] == srv.REATTEMPT_LEDGER_SCHEMA
        # idempotent: the same pass again spawns nothing
        assert srv.apply_reattempt_policy(draft, tmp_path, calls=calls, snapshot=snap, rows_by_sid=rows, ledger=ledger, spawn=spawn, log=lambda *a: None) == [] and spawned == [sid]
        # the re-attempt fails on infra again -> HOLD, no third container; the reason names the second failure
        calls.append({**calls[0], "call_id": "fc-re1"})
        snap["fc-re1"] = {"done": True, "summary": {"status": "LOST_CONTAINER"}, "error": None, "accepted": None}
        out = srv.apply_reattempt_policy(draft, tmp_path, calls=calls, snapshot=snap, rows_by_sid=rows, ledger=ledger, spawn=spawn, log=lambda *a: None)
        assert [d["action"] for d in out] == ["hold"] and spawned == [sid]
        e = srv.reattempt_ledger_get(ledger, sid)
        assert e["held"] is True and "second infra failure" in e["hold_reason"] and e["held_at"] and e["reattempts"] == 1
        # a HELD standup never spawns again, whatever fails next
        calls.append({**calls[0], "call_id": "fc-x"})
        snap["fc-x"] = {"done": True, "summary": {"status": "LOST_CONTAINER"}, "error": None, "accepted": None}
        out = srv.apply_reattempt_policy(draft, tmp_path, calls=calls, snapshot=snap, rows_by_sid=rows, ledger=ledger, spawn=spawn, log=lambda *a: None)
        assert out and out[0]["action"] == "hold" and spawned == [sid]
        # a CORRECTNESS refusal holds at once (no re-attempt); a DONE / accepted call is left alone; a running call is left alone
        sid2 = "pasttense__s2__base__shard00"
        calls2 = [{"call_id": "fc-2", "standup_id": sid2, "accepted_key": rows[sid2]["accepted_key"], "control_key": rows[sid2]["control_key"], "launch_plan_sha": sha},
                  {"call_id": "fc-3", "standup_id": "pasttense__s3__base__shard00", "accepted_key": rows["pasttense__s3__base__shard00"]["accepted_key"], "control_key": "k3", "launch_plan_sha": sha},
                  {"call_id": "fc-4", "standup_id": "pasttense__arditi__base__shard00", "accepted_key": rows["pasttense__arditi__base__shard00"]["accepted_key"], "control_key": "k4", "launch_plan_sha": sha}]
        snap2 = {"fc-2": {"done": True, "summary": {"status": "DRIVERS_FAILED"}, "error": None, "accepted": None},
                 "fc-3": {"done": True, "summary": {"status": "DONE"}, "error": None, "accepted": "ta-ok"},
                 "fc-4": {"done": False, "summary": None, "error": None, "accepted": None}}
        out = srv.apply_reattempt_policy(draft, tmp_path, calls=calls2, snapshot=snap2, rows_by_sid=rows, ledger=ledger, spawn=spawn, log=lambda *a: None)
        assert [(d["standup_id"], d["action"], d["class"]) for d in out] == [(sid2, "hold", "correctness")] and spawned == [sid]
        assert srv.reattempt_ledger_get(ledger, sid2)["held"] and srv.reattempt_ledger_get(ledger, "pasttense__s3__base__shard00")["judged_calls"] == []
        # a RESULT line (an expired call resolved off the volume) is read like a summary
        assert srv.call_failure({"done": True, "summary": None, "error": "RESULT rc=1 status=DRIVERS_FAILED", "result_line": "2026 status=DRIVERS_FAILED rc=1 attempt=ta"}) == (True, "DRIVERS_FAILED", "RESULT rc=1 status=DRIVERS_FAILED")
        assert srv.call_failure({"done": True, "summary": {"status": "DONE"}, "error": None}) == (False, "DONE", None)
        log = (tmp_path / "legs.log").read_text()
        assert "REATTEMPT pasttense__s1__base__shard00 after call=fc-1" in log and "HOLD pasttense__s1__base__shard00 call=fc-re1" in log and "HOLD pasttense__s2__base__shard00" in log

    def test_the_first_failed_child_tears_its_siblings_down(self, srv):
        import inspect
        killed = []

        class P:
            def __init__(self, pid):
                self.pid = pid
        procs = [({"eval_leg": "pasttense", "arm": "none", "s": 1}, P(11), None), ({"eval_leg": "pasttense", "arm": "x", "s": 1}, P(12), None)]
        pending = [{"eval_leg": "pasttense", "arm": "y", "s": 1}]
        names = srv.teardown_siblings(procs, pending, kill=lambda p: killed.append(p.pid))
        assert names == ["pasttense__none__s1", "pasttense__x__s1"] and killed == [11, 12] and pending == []
        src = inspect.getsource(srv._run_standup_row)
        assert "if rc != 0 and lp.D73_TEARDOWN_AFTER_FAILURE" in src and 'evidence["teardown_after_failure"]' in src and lp.D73_TEARDOWN_AFTER_FAILURE is True
        assert src.index('evidence["teardown_after_failure"]') < src.index("teardown_standup(timeout=600)")

    def test_go_preconditions_consume_only_the_launch_certificate(self, draft, srv):
        """`go_preconditions` takes the CERTIFICATE (every component re-validated on its content); the
        entrypoint applies it to --go AND --legs before any row is selected and to --reattempt; --smoke is exempt; the old --s0-record /
        --smokes readers are gone."""
        import inspect
        assert any("no launch certificate" in p for p in lp.go_preconditions(draft, certificate=None))
        assert any("certificate lacks" in p for p in lp.go_preconditions(draft, certificate={"schema": lp.CERTIFICATE_SCHEMA}))
        src = inspect.getsource(srv.main)
        rc_src = inspect.getsource(srv.require_certificate)
        lp_src = inspect.getsource(srv.launch_production)
        assert "lp.go_preconditions(plan_d, certificate=cert" in rc_src and "plan_path=Path(plan_path)" in rc_src and "read_text=certificate_read_text(Path(rs))" in rc_src
        assert "require_certificate(plan_d" in lp_src and lp_src.index("require_certificate(plan_d") < lp_src.index("select(plan_d") and "launch_production(plan_d" in src and "if go or legs:" in src
        assert src.index("if reattempt:") < src.rindex("require_certificate(plan_d") and "select_launch_rows(plan_d" not in src.split("if go or legs:", 1)[1].split("smoke_wu = ", 1)[0]
        assert "s0_record: str" not in src and "smokes: str" not in src and "--s0-record" not in src.split('"""', 2)[-1] and "production launch REFUSED (D73 (4); the launch certificate)" in rc_src

    def test_the_watcher_emits_the_three_summary_lines_machine_readable(self, draft):
        import time as _t
        sw = _load_peer("seedrep_watch_d73", "scripts/modal/seedrep_watch.py", REPO)
        rows = {r["standup_id"]: r for r in draft["standups"]}
        req = lp.required_accepted_keys(draft, "pasttense", "base")
        calls = [{"call_id": f"fc-{i}", "standup_id": sid, "accepted_key": k, "control_key": rows[sid]["control_key"], "leg": "pasttense", "engine": rows[sid]["engine"],
                  "stage": "base", "shard": 0} for i, (k, sid) in enumerate(req.items())]
        snap = {c["call_id"]: {"done": True, "summary": {"status": "DONE"}, "error": None, "accepted": "ta-1"} for c in calls}
        held_sid = calls[0]["standup_id"]
        holds = {held_sid: {"held": True, "hold_reason": "correctness failure 'DRIVERS_FAILED': ...", "reattempts": 0, "held_at": "2026-09-18T05:00:00Z"}}
        bars = {"thresholds": dict(lp.UTILISATION_BARS), "s5_bars_met": True, "s5_bars_source": "RS/tables/utilisation.csv"}
        st = sw.WatchState()
        ev = sw.classify(now=_t.time(), leg="pasttense", calls=calls, snapshot=snap, state=st, stall_min=20, rows_flat_min=40, progress_min=30, required_keys=req,
                         holds=holds, bars=bars, wave={"W7": draft["W7"], "wave_uuid": draft["wave_uuid"]})
        names = [e.split(" ", 1)[0] for e in ev]
        assert names[:2] == ["WAVE_LAUNCHED", "HOLD"] and names[-2:] == ["WAVE_DONE", "DONE"]
        payloads = {e.split(" ", 1)[0]: json.loads(e.split(" ", 1)[1]) for e in ev if e.split(" ", 1)[0] in sw.SUMMARY_EVENTS}
        assert payloads["WAVE_LAUNCHED"]["bars"]["thresholds"]["aggregate_tok_s_over_s5_min"] == 0.8 and payloads["WAVE_LAUNCHED"]["n_calls"] == 4 and payloads["WAVE_LAUNCHED"]["W7"] == draft["W7"]
        assert payloads["HOLD"]["standup_id"] == held_sid and "DRIVERS_FAILED" in payloads["HOLD"]["reason"] and payloads["WAVE_DONE"]["held"] == [held_sid]
        assert all("\n" not in e for e in ev if e.split(" ", 1)[0] in sw.SUMMARY_EVENTS), "one line each"
        # once per watch: a second poll repeats neither WAVE_LAUNCHED nor the HOLD
        ev2 = sw.classify(now=_t.time(), leg="pasttense", calls=calls, snapshot=snap, state=st, stall_min=20, rows_flat_min=40, progress_min=30, required_keys=req,
                          holds=holds, bars=bars)
        assert not any(e.startswith(("WAVE_LAUNCHED", "HOLD")) for e in ev2) and any(e.startswith("WAVE_DONE") for e in ev2)
        assert sw.SUMMARY_EVENTS == lp.D73_SUMMARY_EVENTS == ("WAVE_LAUNCHED", "WAVE_DONE", "HOLD") and set(sw.SUMMARY_EVENTS) <= set(sw.EVENTS)
        assert sw.s5_bars_of(None)["s5_bars_met"] is None
        with pytest.raises(AssertionError):
            sw.summary_line("STARTED", {})

    def test_the_s0_record_names_its_plan_and_the_s5_bars_are_written_by_the_validating_tool(self, draft, tmp_path):
        S0 = _load_peer("seedrep_s0_gates_d73", "scripts/modal/seedrep_s0_gates.py", REPO)
        rec = S0.run_gates(tree=REPO, plan=draft, rs=tmp_path, python=sys.executable, quota=134, campaign=None, only=["LEDGER"])
        assert rec["launch_plan_sha"] == draft["launch_plan_sha"] and rec["W"] == draft["W"] and rec["ok"] is True and list(rec["gates"]) == ["LEDGER"]
        spec = importlib.util.spec_from_file_location("seedrep_s5_record_d73", REPO / "scripts/modal/seedrep_s5_record.py")
        S5R = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(S5R)
        rs = tmp_path / "RS"
        rs.mkdir()
        wave_p = lp.smoke_wave_path(rs, "S5")
        wave_p.parent.mkdir(parents=True)
        wave_p.write_text(json.dumps({"smoke_id": "S5", "wave_uuid": SMOKE_WAVE}))
        plan_p = tmp_path / "plan.json"
        plan_p.write_text(json.dumps({"W": draft["W"], "wave_uuid": draft["wave_uuid"], "launch_plan_sha": draft["launch_plan_sha"]}))
        rec_p = tmp_path / "bars.json"
        # the COMPLETE verdict -- every bar's reading vs its threshold, the table by path + sha, the S5 attempts, the execution wave
        verdict = lp.utilisation_verdict(bars={"running_over_max_num_seqs_min": 0.91, "sm_duty_min": 0.88, "aggregate_tok_s_over_s5_min": 0.97, "h200_hours_over_sized_max": 1.1},
                                         source="RS/tables/utilisation.csv", source_sha256="a" * 64, W=draft["W"], execution_wave_uuid=SMOKE_WAVE, attempt_ids=["ta-s5-0001"])
        assert verdict["bars_met"] is True and tuple(verdict) == lp.UTILISATION_VERDICT_KEYS and all(b["met"] for b in verdict["bars"].values())
        rec_p.write_text(json.dumps(verdict))
        assert S5R.main(["bars", "--rs", str(rs), "--plan", str(plan_p), "--record", str(rec_p)]) == 0
        doc = json.loads((rs / "smokes.json").read_text())
        assert doc["S5"]["bars"] == verdict and doc["S5"]["execution_wave_uuid"] == SMOKE_WAVE and doc["S5"]["production_wave_uuid"] == draft["wave_uuid"]
        # known-bads: a bare value; the old {bars_met, source} claim; a `met` that disagrees with its value; a missing bar; a foreign threshold;
        # `bars_met` disagreeing with the bars; a mirrored source; the production wave as the execution wave; a hours bar OVER 1.5x read as met
        for bad in ("true", json.dumps({"bars_met": True, "source": "RS/tables/utilisation.csv", "bars": {"running": True, "sm": True, "tok_s": True, "hours": True}}),
                    json.dumps({**verdict, "bars": {**verdict["bars"], "sm_duty_min": {"value": 0.5, "threshold": 0.85, "met": True}}}),
                    json.dumps({**verdict, "bars": {k: v for k, v in verdict["bars"].items() if k != "sm_duty_min"}}),
                    json.dumps({**verdict, "bars": {**verdict["bars"], "sm_duty_min": {"value": 0.88, "threshold": 0.5, "met": True}}}),
                    json.dumps({**verdict, "bars_met": False}), json.dumps({**verdict, "source": "mirror"}),
                    json.dumps({**verdict, "execution_wave_uuid": draft["wave_uuid"]}),
                    json.dumps({**verdict, "bars": {**verdict["bars"], "h200_hours_over_sized_max": {"value": 1.6, "threshold": 1.5, "met": True}}})):
            rec_p.write_text(bad)
            assert S5R.main(["bars", "--rs", str(rs), "--plan", str(plan_p), "--record", str(rec_p)]) == 2, bad[:80]
        unmet = lp.utilisation_verdict(bars={"running_over_max_num_seqs_min": 0.91, "sm_duty_min": 0.7, "aggregate_tok_s_over_s5_min": 0.97, "h200_hours_over_sized_max": 1.1},
                                       source="RS/tables/utilisation.csv", source_sha256="a" * 64, W=draft["W"], execution_wave_uuid=SMOKE_WAVE, attempt_ids=["ta-s5-0001"])
        assert unmet["bars_met"] is False and unmet["bars"]["sm_duty_min"]["met"] is False and lp.utilisation_verdict_problems(unmet, W=draft["W"], execution_wave_uuid=SMOKE_WAVE) == []


# =========================================================================== the WAVE legs' config private_run namespaces

class TestWS5I_wave_config_private_runs:
    """Discovery files a store under its config's `private_run`: the static-base run is read off `*_seedrep_s<s>_sampled_t0.yaml`, the
    wave run (BoN / multi-turn) off the per-seed `*_seedrep_s<s>.yaml` when one exists. At a4d81464 the Arditi BoN children used the per-seed
    `_sampled_t0` configs (== the static-base run -> filed STATIC) and the Arditi multi-turn children the ONE shared `_seedrep.yaml` (one
    namespace for three seeds -> assigned to the first seed only). Now: the Arditi BoN AND multi-turn children use the per-seed WAVE
    configs (`ARDITI_WAVE_CONFIG_REL`), the views' BoN the unsampled view config; every child carries its config's `private_run` (read off the
    tree, else the worktree, else the stem it must carry); validate_plan refuses a wave child on a static-base run and Arditi wave
    children sharing a namespace; the S0 PLAN gate requires every named config at W. The real generated argv is exposed for the twin
    through `plan_children` (this module by path)."""

    WAVE = ("bon", "pair", "crescendo", "actorattack")

    def test_arditi_wave_children_use_the_per_seed_wave_configs_with_three_distinct_private_runs(self, draft):
        assert lp.WAVE_LEGS == self.WAVE and lp.ARDITI_WAVE_CONFIG_REL == "configs/seedrep/arditi_allpos_qwen35_4b_ar_c0p3_seedrep_s{s}.yaml"
        for leg in self.WAVE:
            r = _row(draft, leg, "arditi")
            kids = [k for k in r["children"] if k.get("argv")]
            assert [k["s"] for k in kids] == [1, 2, 3]
            assert [k["config_rel"] for k in kids] == [lp.ARDITI_WAVE_CONFIG_REL.format(s=s) for s in (1, 2, 3)]
            assert [k["argv"][k["argv"].index("--config") + 1] for k in kids] == [k["config_rel"] for k in kids]
            runs = [lp.child_private_run(k) for k in kids]
            assert len(set(runs)) == 3 and runs == [f"arditi_allpos_qwen35_4b_ar_c0p3_seedrep_s{s}" for s in (1, 2, 3)]
            # the discovery glob for the per-seed wave config: `*_seedrep_s<s>.yaml`, not `_sampled`
            assert all(Path(k["config_rel"]).name.endswith(f"_seedrep_s{k['s']}.yaml") and "_sampled" not in k["config_rel"] for k in kids)
        # the views' BoN rides the unsampled view config (the wave run `sdfbase_full22_<tag>`), read for real off this tree
        for view in ("s1", "s2", "s3"):
            for k in _row(draft, "bon", view)["children"]:
                assert k["config_rel"] == lp.VIEW_CONFIG_REL.format(view=view) and k["config_private_run"] == f"sdfbase_full22_qwen35_4b_{view}" and k["config_private_run_source"] == "read"
            for k in _row(draft, "pasttense", view)["children"]:
                assert k["config_private_run"] == f"sdfbase_full22_qwen35_4b_{view}_sampled_t0" and k["config_private_run_source"] == "read"
        # the static / controls legs keep the per-seed `_sampled_t0` configs (the static-base run); the wave runs never equal them
        static_runs = {lp.child_private_run(k) for r in draft["standups"] + draft["rungs"] for k in r["children"] if k["eval_leg"] in lp.STATIC_LEGS}
        wave_runs = {lp.child_private_run(k) for r in draft["standups"] for k in r["children"] if k["eval_leg"] in self.WAVE and k.get("argv")}
        assert static_runs and wave_runs and not (static_runs & wave_runs)
        for k in _row(draft, "pasttense", "arditi")["children"][:3]:
            assert k["config_rel"] == lp.ARDITI_CLIENT_CONFIG_REL.format(s=k["s"], suffix="")
        assert lp.validate_plan(draft) == [] and draft["config_private_runs"][lp.VIEW_CONFIG_REL.format(view="s1")] == "sdfbase_full22_qwen35_4b_s1"

    def test_known_bads_a_wave_child_on_a_static_run_and_arditi_children_sharing_a_namespace_are_refused(self, draft):
        bad = copy.deepcopy(draft)
        k = _row(bad, "bon", "arditi")["children"][0]
        k["config_private_run"] = _row(bad, "pasttense", "arditi")["children"][0]["config_private_run"] or _row(bad, "pasttense", "arditi")["children"][0]["config_private_run_expected"]
        probs = lp.validate_plan(bad)
        assert any("is a STATIC-base run" in p and "bon__arditi__base__shard00" in p for p in probs), probs
        bad = copy.deepcopy(draft)
        for k in _row(bad, "crescendo", "arditi")["children"]:
            k["config_private_run"] = "arditi_allpos_qwen35_4b_ar_c0p3"                   # the shared Scale namespace on all three seeds
        probs = lp.validate_plan(bad)
        assert any("share a config private_run" in p and "crescendo__arditi__base__shard00" in p for p in probs), probs
        bad = copy.deepcopy(draft)
        k = _row(bad, "pair", "arditi")["children"][1]
        k["config_rel"] = lp.ARDITI_CONFIG_REL                                              # the shared unsampled config is not a per-seed wave config
        assert any("must use the per-seed wave config" in p for p in lp.validate_plan(bad))
        bad = copy.deepcopy(draft)
        k = _row(bad, "bon", "s2")["children"][0]
        k["config_private_run"] = "sdfbase_full22_qwen35_4b_s2_sampled_t0"                # a VIEW BoN child on its static run is refused too
        assert any("is a STATIC-base run" in p and "bon__s2__base__shard00" in p for p in lp.validate_plan(bad))

    def test_private_runs_are_read_off_the_tree_then_the_lane_worktree_and_the_s0_gate_requires_the_files_at_w(self, draft, inputs, tmp_path):
        cpr = inputs["config_private_runs"]
        assert set(cpr) == set(lp.plan_config_rels()) and len(cpr) == 3 * 4 + 1 + 3 * 4 + 3 * 8     # + the 24 per-cell GSM8K overlays
        assert cpr[lp.VIEW_SAMPLED_CONFIG_REL.format(view="s2", suffix="_cap8192")] == "sdfbase_full22_qwen35_4b_s2_sampled_t0_cap8192"
        ws1dp = WORKTREES / "seedrep-ws1dp" / lp.ARDITI_CLIENT_CONFIG_REL.format(s=1, suffix="")
        if ws1dp.is_file():
            assert cpr[lp.ARDITI_CLIENT_CONFIG_REL.format(s=1, suffix="")] == "arditi_allpos_qwen35_4b_ar_c0p3_seedrep_s1_sampled_t0"
            assert cpr[lp.ARDITI_CONFIG_REL] == "arditi_allpos_qwen35_4b_ar_c0p3", "Scale's verbatim shared namespace"
        # a config absent everywhere: None read, the stem expected, the source says so; a fake root with a private_run wins over the stem
        wave1 = lp.ARDITI_WAVE_CONFIG_REL.format(s=1)
        k = _row(draft, "bon", "arditi")["children"][0]
        if cpr[wave1] is None:
            assert k["config_private_run"] is None and k["config_private_run_expected"] == "arditi_allpos_qwen35_4b_ar_c0p3_seedrep_s1" and "expected_from_stem" in k["config_private_run_source"]
        fake = tmp_path / "root"
        (fake / "configs/seedrep").mkdir(parents=True)
        (fake / wave1).write_text("private_run: arditi_allpos_qwen35_4b_ar_c0p3_seedrep_s1\nout_dir: out/x\n")
        got = lp.read_config_private_runs([REPO, fake], [wave1, lp.VIEW_CONFIG_REL.format(view="s1"), "configs/nowhere.yaml"])
        assert got == {wave1: "arditi_allpos_qwen35_4b_ar_c0p3_seedrep_s1", lp.VIEW_CONFIG_REL.format(view="s1"): "sdfbase_full22_qwen35_4b_s1", "configs/nowhere.yaml": None}
        # the S0 PLAN gate: every config a child names must exist under the tree at W. On a tree without them the Arditi lane configs
        # were absent by name; on the MERGED tree (4dc1e500's three per-seed Arditi wave configs + the prompted overlays, both SHIP)
        # every named config is present and the gate is CLEAN — the positive case W requires; the negative case is the fake root
        missing = lp.config_files_problems(draft, REPO)
        assert all(p.startswith("config named by a child is absent under the tree: ") for p in missing)
        assert (REPO / wave1).is_file() and not any(wave1 in p for p in missing)
        assert missing == [], missing
        fake_missing = lp.config_files_problems(draft, fake)
        assert fake_missing != [] and any(lp.VIEW_CONFIG_REL.format(view="s1") in p for p in fake_missing)   # a root with one config lacks the rest, by name
        S0 = _load_peer("seedrep_s0_gates_ws5i", "scripts/modal/seedrep_s0_gates.py", REPO)
        assert "config_files_problems" in __import__("inspect").getsource(S0.gate_plan_validates)

    def test_the_real_generated_argv_is_exposed_for_ws5is_twin(self, draft):
        bon = lp.plan_children(draft, legs=("bon",), engines=("arditi",))
        assert [(sid, k["s"]) for sid, k in bon] == [("bon__arditi__base__shard00", s) for s in (1, 2, 3)]
        for sid, k in bon:
            a = k["argv"]
            assert a[1].endswith("scripts/sdfbase/full22/bon_served_run.py") and a[a.index("--config") + 1] == lp.ARDITI_WAVE_CONFIG_REL.format(s=k["s"])
            assert a[a.index("--decode-seed-base") + 1] == str(k["s"]) and a[a.index("--bon-decode-preset") + 1] == "vendor"
        mt = lp.plan_children(draft, legs=("crescendo", "actorattack", "pair"), engines=("arditi",))
        assert len(mt) % 3 == 0 and len(mt) >= 9 and {sid.split("__shard")[0] for sid, _ in mt} == {f"{leg}__arditi__base" for leg in ("crescendo", "actorattack", "pair")}
        by_sid: dict = {}
        for sid, k in mt:
            by_sid.setdefault(sid, []).append(k["s"])
        assert by_sid and all(v == [1, 2, 3] for v in by_sid.values()), "every multi-turn shard row carries the three seeds"
        assert all(k["argv"][k["argv"].index("--config") + 1] == lp.ARDITI_WAVE_CONFIG_REL.format(s=k["s"]) for _, k in mt)
        assert len(lp.plan_children(draft)) == sum(len(r["children"]) for r in draft["standups"] + draft["rungs"])


# =========================================================================== 495c86b6: the S5 THROUGHPUT reference record

class TestS5ReferenceRecord:
    """The utilisation audit reads the S5 throughput reference ONLY from `RS/smokes.json` `S5.references[]` records with the exact
    `S5_REFERENCE_KEYS` (aggregate engine-counter method, >= 20 min, complete store, coverage >= 0.8, interval count, W / wave / attempt held
    to the standup). No producer wrote that record on any tree: `seedrep_s5_record.py reference` writes it, validated, beside the window
    record (`window`); the `S0` gate `G8` refuses a smokes.json whose references are not references."""

    def _ws3g(self):
        mod = _load_peer("seedrep_utilisation_table_under_test", "scripts/sdfbase/seedrep/utilisation_table.py", WORKTREES / "seedrep-ws3g")
        if mod is None:
            pytest.skip("WS3-G's utilisation_table.py is neither in this tree nor in the seedrep-ws3g worktree")
        return mod

    def _rec(self, draft, **over):
        base = dict(L=8192, S=64, thinking=False, tok_s_aggregate=812.5, duration_s=1500.0, store_complete=True, sampling_coverage=0.93, n_eligible_intervals=150,
                    W=draft["W"], wave_uuid=draft["wave_uuid"], attempt_id="ta-s5-0001")
        base.update(over)
        return lp.s5_reference_record(**base)

    def test_the_key_set_is_ws3gs_exactly_and_the_record_passes_their_real_validator(self, draft):
        UT = self._ws3g()
        assert lp.S5_REFERENCE_KEYS == UT.S5_REFERENCE_KEYS and lp.S5_REFERENCE_METHOD == UT.S5_REFERENCE_METHOD == "aggregate_decode_tok_s"
        assert lp.S5_REFERENCE_MIN_DURATION_S == UT.S5_MIN_DURATION_S == 1200 and lp.S5_REFERENCE_MIN_COVERAGE == UT.MIN_SAMPLING_COVERAGE == 0.8
        rec = self._rec(draft)
        assert tuple(rec) == lp.S5_REFERENCE_KEYS and rec["method"] == "aggregate_decode_tok_s"
        assert UT.s5_reference_problems(rec, W=draft["W"], wave_uuid=draft["wave_uuid"]) == [] == lp.s5_reference_problems(rec, W=draft["W"], wave_uuid=draft["wave_uuid"])
        # known-bads, refused by BOTH validators: a bare scalar, a percentile, a partial store, a short window, low coverage, another wave, a mirror
        for bad, needle in ((812.5, "bare number"), ({**rec, "method": "p99_decode_tok_s"}, "percentile"), ({**rec, "store_complete": False}, "partial store"),
                            ({**rec, "duration_s": 600}, "20 min"), ({**rec, "sampling_coverage": 0.5}, "steady state"), ({**rec, "wave_uuid": SMOKE_WAVE}, "standup's wave"),
                            ({**rec, "source": "mirror"}, "manual or mirrored"), ({k: v for k, v in rec.items() if k != "n_eligible_intervals"}, "lacks")):
            mine = lp.s5_reference_problems(bad, W=draft["W"], wave_uuid=draft["wave_uuid"])
            theirs = UT.s5_reference_problems(bad, W=draft["W"], wave_uuid=draft["wave_uuid"])
            assert mine and theirs and any(needle in p for p in mine) and any(needle in p for p in theirs), (needle, mine, theirs)
        with pytest.raises(ValueError, match="not an S5 reference"):
            self._rec(draft, duration_s=60)

    def test_the_writer_records_references_and_window_measurements_validated_and_the_s0_gate_reads_them(self, draft, tmp_path):
        spec = importlib.util.spec_from_file_location("seedrep_s5_record_under_test", REPO / "scripts/modal/seedrep_s5_record.py")
        S5R = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(S5R)
        rs = tmp_path / "RS"
        rs.mkdir()
        plan_p = tmp_path / "plan.json"
        plan_p.write_text(json.dumps({"W": draft["W"], "wave_uuid": draft["wave_uuid"], "launch_plan_sha": draft["launch_plan_sha"], "s5_roster": draft["s5_roster"], "standups": [], "rungs": []}))
        rec_p = tmp_path / "ref.json"
        # a record binds to the smoke's EXECUTION wave (its wave.json under the smoke directory); with no wave file the smoke never ran -> refused;
        # a record stamped with the PRODUCTION wave (the pre-fix binding) is refused once the file exists
        rec_p.write_text(json.dumps(self._rec(draft, wave_uuid=SMOKE_WAVE)))
        assert S5R.main(["reference", "--rs", str(rs), "--plan", str(plan_p), "--record", str(rec_p)]) == 2, "no S5 wave file: the smoke has not run"
        wave_p = lp.smoke_wave_path(rs, "S5")
        wave_p.parent.mkdir(parents=True)
        wave_p.write_text(json.dumps({"smoke_id": "S5", "wave_uuid": SMOKE_WAVE, "minted_at": "2026-09-18T00:00:00Z"}))
        prod_p = tmp_path / "prod.json"
        prod_p.write_text(json.dumps(self._rec(draft)))                                    # stamped with the production wave
        assert S5R.main(["reference", "--rs", str(rs), "--plan", str(plan_p), "--record", str(prod_p)]) == 2
        assert S5R.main(["reference", "--rs", str(rs), "--plan", str(plan_p), "--record", str(rec_p)]) == 0
        doc = json.loads((rs / "smokes.json").read_text())
        assert [tuple(r) for r in doc["S5"]["references"]] == [lp.S5_REFERENCE_KEYS] and doc["S5"]["references"][0]["wave_uuid"] == SMOKE_WAVE
        assert doc["S5"]["execution_wave_uuid"] == SMOKE_WAVE and doc["S5"]["production_wave_uuid"] == draft["wave_uuid"] and doc["S5"]["W"] == draft["W"]
        assert doc["S5"]["launch_plan_sha"] == draft["launch_plan_sha"] and doc["S5"]["identity_rule"] == lp.WAVE_IDENTITY_RULE
        assert S5R.main(["reference", "--rs", str(rs), "--plan", str(plan_p), "--record", str(rec_p)]) == 2, "a second record of the same window class refuses"
        rec_p.write_text(json.dumps({**self._rec(draft, wave_uuid=SMOKE_WAVE), "tok_s_aggregate": 900.0}))
        assert S5R.main(["reference", "--rs", str(rs), "--plan", str(plan_p), "--record", str(rec_p), "--replace"]) == 0
        assert json.loads((rs / "smokes.json").read_text())["S5"]["references"][0]["tok_s_aggregate"] == 900.0
        rec_p.write_text(json.dumps({**self._rec(draft, wave_uuid=SMOKE_WAVE), "method": "p99_decode_tok_s"}))
        assert S5R.main(["reference", "--rs", str(rs), "--plan", str(plan_p), "--record", str(rec_p), "--replace"]) == 2
        rec_p.write_text("812.5")
        assert S5R.main(["reference", "--rs", str(rs), "--plan", str(plan_p), "--record", str(rec_p), "--replace"]) == 2
        # a plan for ANOTHER production wave cannot write into this block (a copied block never re-binds)
        other_p = tmp_path / "other_plan.json"
        other_p.write_text(json.dumps({"W": draft["W"], "wave_uuid": "33333333-4444-4555-8666-777777777777", "launch_plan_sha": draft["launch_plan_sha"]}))
        rec_p.write_text(json.dumps({**self._rec(draft, wave_uuid=SMOKE_WAVE), "L": 7424, "S": 70}))
        assert S5R.main(["reference", "--rs", str(rs), "--plan", str(other_p), "--record", str(rec_p)]) == 2
        # the multi-turn window record beside it
        win = {"schema": lp.S5_WINDOW_MEASUREMENT_SCHEMA, "window_source": lp.S5_WINDOW_SOURCE, "leg": "crescendo", "arm": "none", "arm_kind": "parent",
               "model_key": lp.MODEL_KEY, "decode_policy_ref": lp.ARDITI_DECODE_POLICY_REF, "objects_measured": ["none"], "store_path": "exp/transcripts_results/x",
               "n_goals_completed": 20, "n_goals_expected": 20, "store_complete": True, "context_max": 31000, "clamped_calls": 0, "cap": lp.MULTITURN_CAP4096_CAP,
               "measured_utc": "2026-09-18T00:00:00Z", "wave_uuid": SMOKE_WAVE, "attempt_id": "ta-s5-0001", "decode": "vendor"}
        rec_p.write_text(json.dumps(win))
        # W16s5 (B2): the CLI REFUSES an operator-authored window record by name (`window --rows` derives from the probe attempts' producer files;
        # tests/test_seedrep_w16s5_window_measurement.py); the library seam still validates + merges a record exactly as the derivation does
        assert S5R.main(["window", "--rs", str(rs), "--plan", str(plan_p), "--record", str(rec_p)]) == 2
        plan_doc = json.loads(plan_p.read_text())
        S5R.record_s5("window", rs=rs, plan=plan_doc, record=win)
        doc = json.loads((rs / "smokes.json").read_text())
        assert doc["S5"]["multiturn"][0]["context_max"] == 31000 and len(doc["S5"]["references"]) == 1
        with pytest.raises(ValueError, match="percentile"):
            S5R.record_s5("window", rs=rs, plan=plan_doc, record={**win, "window_source": "p99"}, replace=True)
        with pytest.raises(ValueError, match="execution wave"):                                  # the production wave: no smoke ran there
            S5R.record_s5("window", rs=rs, plan=plan_doc, record={**win, "wave_uuid": draft["wave_uuid"]}, replace=True)
        # the S0 gate G8 reads S5.references[]: a real reference counts toward the roster; a bare scalar / percentile refuses by name
        g = _load_peer("seedrep_s0_gates_s5", "scripts/modal/seedrep_s0_gates.py", REPO)
        out = g.gate8_s5_roster(plan=draft, s5_references=rs / "smokes.json")
        assert out["ok"] is False and "planned standups without an S5 reference" in out["why"] and "1 reference(s)" in out["s5_references"]
        doc["S5"]["references"].append(812.5)
        (rs / "smokes.json").write_text(json.dumps(doc))
        out = g.gate8_s5_roster(plan=draft, s5_references=rs / "smokes.json")
        assert out["ok"] is False and "not S5 references" in out["why"] and "bare number" in out["why"]
        doc["S5"]["references"][-1] = {**self._rec(draft), "method": "p99_decode_tok_s"}
        (rs / "smokes.json").write_text(json.dumps(doc))
        assert "percentile" in g.gate8_s5_roster(plan=draft, s5_references=rs / "smokes.json")["why"]
        # every planned window class referenced -> the gate passes; references under the PRODUCTION wave refuse (the block names the execution wave)
        doc["S5"]["references"] = [self._rec(draft, L=r["L"], S=r["S"], thinking=r["thinking"], wave_uuid=SMOKE_WAVE) for r in draft["s5_roster"]]
        (rs / "smokes.json").write_text(json.dumps(doc))
        assert g.gate8_s5_roster(plan=draft, s5_references=rs / "smokes.json")["ok"] is True
        doc["S5"]["references"][0] = self._rec(draft, L=doc["S5"]["references"][0]["L"], S=doc["S5"]["references"][0]["S"], thinking=doc["S5"]["references"][0]["thinking"])
        (rs / "smokes.json").write_text(json.dumps(doc))
        assert "standup's wave" in g.gate8_s5_roster(plan=draft, s5_references=rs / "smokes.json")["why"]
        stripped = {**doc, "S5": {k: v for k, v in doc["S5"].items() if k != "execution_wave_uuid"}}
        (rs / "smokes.json").write_text(json.dumps(stripped))
        assert "carries no execution_wave_uuid" in g.gate8_s5_roster(plan=draft, s5_references=rs / "smokes.json")["why"]


# =========================================================================== EM (peer, 2026-09-18): the misalign runner's per-request timeout

class TestEM_misalign_timeout:
    """The misalign runner's DEFAULT per-request timeout is 7,500 s, sized for the agentic family's 16k cap; at a small cap a LOST response
    parks the runner past the job's walltime (EM lost a job: vLLM at 0 running / 0 waiting while the runner sat at 399/400 for 24 min).
    Every misalign child carries `--timeout-s` sized from ITS cap (floor 900 s, never the default); the plan validator refuses a child without it."""

    def test_every_misalign_child_carries_a_cap_sized_timeout_the_real_runner_parses(self, draft):
        kids = [k for r in draft["standups"] if r["leg"] == "misalign" for k in r["children"]]
        assert len(kids) == 54 and all(lp._flag_value(k["argv"], "--timeout-s") == str(lp.misalign_timeout_s(k["cap"])) == "1200" == str(k["timeout_s"]) for k in kids)
        assert all("never the runner's 7500 s default" in k["timeout_rule"] for k in kids)
        assert lp.misalign_timeout_s(600) == lp.misalign_timeout_s(1000) == 900 == lp.MISALIGN_TIMEOUT_FLOOR_S and lp.misalign_timeout_s(3000) == 1200
        assert lp.misalign_timeout_s(16384) == 6554 < lp.MISALIGN_RUNNER_DEFAULT_TIMEOUT_S == 7500.0
        R = _misalign_runner()
        if R is None:
            pytest.skip("no misalign runner with WS1-C's --seed-policy flag (tree or seedrep-ws1c worktree)")
        assert R.DEFAULT_TIMEOUT_S == lp.MISALIGN_RUNNER_DEFAULT_TIMEOUT_S
        argv = [a.replace(lp.CONCURRENCY_PLACEHOLDER, "8").replace(lp.PROXY_URL_PLACEHOLDER, "http://127.0.0.1:9/v1").replace(lp.ATTEMPT_ROOT_PLACEHOLDER, "/a")
                for a in kids[0]["argv"]]
        assert argv[1:3] == ["-m", "sisteer.capeval.misalign.runner"]
        a = R._build_parser().parse_args(argv[3:])
        assert a.timeout_s == 1200.0 and a.max_tokens == 3000 and a.seed_base == kids[0]["s"] and a.max_retries == 3

    def test_known_bad_a_small_cap_child_without_an_explicit_timeout_or_at_the_default_is_refused(self, draft):
        bad = copy.deepcopy(draft)
        k = _row(bad, "misalign")["children"][0]
        i = k["argv"].index("--timeout-s")
        del k["argv"][i:i + 2]
        k["cap"] = 600
        k["argv"][k["argv"].index("--max-tokens") + 1] = "600"
        probs = [p for p in lp.validate_plan(bad) if "--timeout-s" in p]
        assert probs and "at cap 600 must carry --timeout-s 900 (got None)" in probs[0] and "7500 s" in probs[0]
        k["argv"] += ["--timeout-s", "7500"]
        probs = [p for p in lp.validate_plan(bad) if "--timeout-s" in p]
        assert probs and "(got '7500')" in probs[0]
        k["argv"][-1] = "900"
        assert [p for p in lp.validate_plan(bad) if "--timeout-s" in p] == []
        k["argv"][-1] = "abc"
        assert [p for p in lp.validate_plan(bad) if "--timeout-s" in p]


# =========================================================================== P1-13: the pull is stage-aware, fails on errors, verifies

class TestP1_13_pull:
    def _accepted(self, plan, leg, scope, att="ta-7"):
        FakeDict.store.clear()
        acc = FakeDict("accepted")
        for k in lp.required_accepted_keys(plan, leg, scope):
            acc.put(k, att)
        return acc

    def _fake_get(self, plan, rows_by_sid, *, produce=True, fail_for=(), adopted=(), empty_adopted=()):
        def run_get(src, out):
            att = Path(src).name
            m = re.match(r"^(?P<leg>[^/]+)/(?P<engine>[^/]+)/(?:rungs/(?P<stage>[^/]+)/)?shard(?P<shard>\d+)/attempts/[^/]+$", src[len(plan["results_root"]) + 1:])
            assert m, src
            sid = lp.standup_id(m["leg"], m["engine"], m["stage"] or "base", int(m["shard"]))
            if sid in fail_for:
                raise RuntimeError("modal volume get rc 1")
            root = out / att
            root.mkdir(parents=True, exist_ok=True)
            (root / "COMPLETE.json").write_text(json.dumps({"attempt_id": att}))
            (root / "exp").mkdir(exist_ok=True); (root / "exp" / "x.jsonl").write_text("{}\n")
            for a in adopted:                                        # the forthcoming adopted_from/<source attempt>/ evidence
                (root / "adopted_from" / a).mkdir(parents=True, exist_ok=True)
                (root / "adopted_from" / a / "request_evidence.jsonl").write_text('{"request_id": "r1"}\n')
            for a in empty_adopted:
                (root / "adopted_from" / a).mkdir(parents=True, exist_ok=True)
            if produce and rows_by_sid[sid].get("ladder") and rows_by_sid[sid]["ladder"].get("cap_next") is not None:
                (root / "queue_inventory.json").write_text(json.dumps({"n": 2, "entries": [1, 2]}))
                (root / "ladder_enqueued.json").write_text(json.dumps({"n": 2, "complete": True, "entries": [1, 2]}))
        return run_get

    def test_stage_aware_pull_includes_accepted_rungs_and_verifies(self, srv, draft, tmp_path):
        acc = self._accepted(draft, "pasttense", "all")
        by_sid = {r["standup_id"]: r for r in draft["standups"] + draft["rungs"]}
        out = srv.pull_leg(draft, "pasttense", tmp_path / "stores", tmp_path, stages="all", accepted=acc, run_get=self._fake_get(draft, by_sid))
        assert len(out["pulled"]) == 4 + 8
        assert (tmp_path / "stores/pasttense/s1/shard00/ta-7/COMPLETE.json").is_file() and (tmp_path / "stores/pasttense/s1/rungs/8192/shard00/ta-7/COMPLETE.json").is_file()
        rows = [json.loads(l) for l in (tmp_path / "pull_manifest.jsonl").read_text().splitlines()]
        assert {r["stage"] for r in rows} == {"base", "8192", "16384"} and all(r["attempt_id"] == "ta-7" for r in rows)

    def test_pull_carries_the_accepted_roots_adopted_from_evidence_and_refuses_an_empty_one(self, srv, draft, tmp_path):
        """When the accepted rung attempt adopted another same-call execution's elected results, the producer copies that execution's
        evidence into `<accepted root>/adopted_from/<source>/`; the pull downloads the whole accepted root, names every adopted attempt in its
        row, sha-manifests the files, and refuses an adopted directory declared with no files."""
        acc = self._accepted(draft, "pasttense", "base")
        by_sid = {r["standup_id"]: r for r in draft["standups"] + draft["rungs"]}
        out = srv.pull_leg(draft, "pasttense", tmp_path / "stores", tmp_path, stages="base", accepted=acc, run_get=self._fake_get(draft, by_sid, adopted=("ta-A",)))
        assert all(p["adopted_from"] == ["ta-A"] for p in out["pulled"]) and len(out["pulled"]) == 4
        rows = [json.loads(l) for l in (tmp_path / "pull_manifest.jsonl").read_text().splitlines()]
        adopted_rows = [r for r in rows if r["path"] == "ta-7/adopted_from/ta-A/request_evidence.jsonl"]
        assert len(adopted_rows) == 4 and all(r["attempt_id"] == "ta-7" and r["sha256"] == hashlib.sha256(b'{"request_id": "r1"}\n').hexdigest() for r in adopted_rows)
        assert srv.ADOPTED_FROM_DIR == "adopted_from"
        (tmp_path / "r2").mkdir()
        with pytest.raises(srv.PullRefused, match=r"adopted_from/.*ta-B.*declared with no evidence files"):
            srv.pull_leg(draft, "pasttense", tmp_path / "s2", tmp_path / "r2", stages="base", accepted=acc, run_get=self._fake_get(draft, by_sid, adopted=("ta-A",), empty_adopted=("ta-B",)))
        assert not (tmp_path / "r2" / "pull_manifest.jsonl").exists()
        out2 = srv.pull_leg(draft, "pasttense", tmp_path / "s3", tmp_path / "r3", stages="base", accepted=acc, run_get=self._fake_get(draft, by_sid))
        assert all(p["adopted_from"] == [] for p in out2["pulled"])

    def test_revalidated_download_errors_and_missing_records_refuse_and_write_no_manifest(self, srv, draft, tmp_path):
        acc = self._accepted(draft, "pasttense", "base")
        by_sid = {r["standup_id"]: r for r in draft["standups"] + draft["rungs"]}
        with pytest.raises(srv.PullRefused, match="download failed"):
            srv.pull_leg(draft, "pasttense", tmp_path / "s1", tmp_path / "r1", stages="base", accepted=acc, run_get=self._fake_get(draft, by_sid, fail_for={"pasttense__s2__base__shard00"}))
        rows = [json.loads(l) for l in (tmp_path / "r1" / "pull_manifest.jsonl").read_text().splitlines()]
        assert "pasttense__s2__base__shard00" not in {r["standup_id"] for r in rows} and len({r["standup_id"] for r in rows}) == 3
        (tmp_path / "r2").mkdir()
        with pytest.raises(srv.PullRefused, match="no ladder_enqueued.json"):
            srv.pull_leg(draft, "pasttense", tmp_path / "s2", tmp_path / "r2", stages="base", accepted=acc, run_get=self._fake_get(draft, by_sid, produce=False))
        assert not (tmp_path / "r2" / "pull_manifest.jsonl").exists()

        def empty_get(src, out):                                     # the repro: an absent dir read as an empty file list
            out.mkdir(parents=True, exist_ok=True)
        (tmp_path / "r3").mkdir()
        with pytest.raises(srv.PullRefused, match="not pulled"):
            srv.pull_leg(draft, "pasttense", tmp_path / "s3", tmp_path / "r3", stages="base", accepted=acc, run_get=empty_get)
        FakeDict.store.clear()
        with pytest.raises(srv.PullRefused, match="no accepted attempt"):
            srv.pull_leg(draft, "pasttense", tmp_path / "s4", tmp_path / "r4", stages="base", accepted=FakeDict("accepted"), run_get=empty_get)


# =========================================================================== P2-14: smoke mode on DRAFT rows, production FINAL-only

def _documented_smoke_command():
    m = re.search(r"seedrep_served\.py --sha \$W --plan \$RS/launch_plan_draft\.json (--smoke [^\n]*)", DRIVER.read_text())
    assert m, "the module docstring documents the smoke command"
    toks = m.group(1).split()
    kw = {}
    i = 0
    while i < len(toks):
        k = toks[i][2:].replace("-", "_")
        kw[k] = toks[i + 1]
        i += 2
    return kw


class TestP2_14_smoke_mode:
    def test_the_documented_command_runs_a_draft_row_under_an_isolated_bounded_identity(self, srv, draft):
        kw = _documented_smoke_command()
        readme = OPERATOR_DIR / "README.md"
        if readme.is_file():
            assert f"--smoke {kw['smoke']} --smoke-id {kw['smoke_id']} --smoke-max-children {kw['smoke_max_children']}" in readme.read_text()
        rows, ident = srv.select_launch_rows(draft, smoke=kw["smoke"], smoke_id=kw["smoke_id"], smoke_wave_uuid=SMOKE_WAVE, smoke_max_children=int(kw["smoke_max_children"]))
        assert len(rows) == 1
        r, prod_row = rows[0], _row(draft, "pasttense")
        assert r["standup_id"] == "smoke-S1__pasttense__s1__base__shard00" and len(r["children"]) == 5 < len(prod_row["children"])
        assert r["modal"]["timeout_s"] == 4 * 3600 < prod_row["modal"]["timeout_s"]
        assert r["control_key"].split("/")[1] == SMOKE_WAVE and r["accepted_key"] != prod_row["accepted_key"] and r["control_key"] != prod_row["control_key"]
        assert ident["dicts"]["accepted"] == "seedrep-2222222-11111111-accepted" != draft["dicts"]["accepted"]
        assert ident["results_root"] == draft["results_root"] + "_smoke/S1" and ident["ladder_root"] != draft["ladder_root"]
        assert r["argv"] == prod_row["argv"] and r["route_set"] == prod_row["route_set"] and r["children"] == prod_row["children"][:5]
        ctx = srv.spawn_ctx(draft, "seedrep-served-2222222", None, smoke=ident)
        assert ctx["dicts"] == ident["dicts"] and ctx["results_root"] == ident["results_root"] and ctx["smoke"]["production_wave_uuid"] == WAVE
        rung = _row(draft, "pasttense", "s1", "8192")
        rows, _ = srv.select_launch_rows(draft, smoke=rung["standup_id"], smoke_id="S3", smoke_wave_uuid=SMOKE_WAVE)
        assert rows[0]["rung_consumer"][rows[0]["rung_consumer"].index("--wave-uuid") + 1] == SMOKE_WAVE

    def test_revalidated_production_fan_out_refuses_a_draft_and_smoke_refuses_a_final(self, srv, inputs, draft):
        with pytest.raises(SystemExit, match="only the FINAL plan is launched"):
            srv.select_launch_rows(draft, go="pasttense__s1__base__shard00")          # the README's OLD command
        with pytest.raises(SystemExit, match="only the FINAL plan is launched"):
            srv.select_launch_rows(draft, legs="pasttense")
        final = _plan(inputs, materialisation="final", s4=lp.S4_DEFAULTS)
        with pytest.raises(SystemExit, match="DRAFT rows only"):
            srv.select_launch_rows(final, smoke="pasttense__s1__base__shard00", smoke_id="S1")
        with pytest.raises(SystemExit, match="needs --smoke-id"):
            srv.select_launch_rows(draft, smoke="pasttense__s1__base__shard00")
        with pytest.raises(SystemExit, match="exactly one of"):
            srv.select_launch_rows(draft, go="x", smoke="y")
        assert final["deferred_rungs"] == [] and "gsm8k__s1__16384__shard00" in {r["standup_id"] for r in final["rungs"]}   # an executable row
        rows_g8, _ = srv.select_launch_rows(final, go="gsm8k__s1__16384__shard00")
        assert rows_g8[0]["rung_mode"] == lp.RUNG_MODE_GSM8K_SUBSET and lp.RUNG_MODE_GSM8K_SUBSET in srv.EXECUTABLE_RUNG_MODES
        with pytest.raises(SystemExit, match="rung mode this app cannot execute"):
            srv.select_launch_rows({**final, "rungs": [{**rows_g8[0], "rung_mode": "something_else"}]}, go="gsm8k__s1__16384__shard00")
        with pytest.raises(SystemExit, match="need --pool-urls"):
            srv.select_launch_rows(final, go="crescendo__s1__base__shard01")
        with pytest.raises(SystemExit, match="has no URL"):
            srv.select_launch_rows(final, go="crescendo__s1__base__shard01", pool_urls={"seedrep-attacker-0": "u"})
        rows, _ = srv.select_launch_rows(final, go="crescendo__s1__base__shard01", pool_urls={"seedrep-attacker-1": "u"}, pool_health=lambda url: True,
                                         pool_bearer="fw-test-bearer", pool_accepts=lambda u, b: True)      # fw: the pools accept the lane bearer
        assert rows[0]["pool"] == "seedrep-attacker-1"
        with pytest.raises(ValueError, match="must not reuse the production wave"):
            lp.smoke_identity(draft, "S1", WAVE)


# =========================================================================== the multi-turn TARGET's three flags

WS1A = WORKTREES / "seedrep-ws1a"


def _parse_through_ws1a_real_parser(argv):
    """`jailbreak_run.main()` builds its parser inline: run it with the argv on sys.argv and STOP at parse_args (the parsed
    namespace is the evidence; nothing generates). The script is loaded by path from the tree first, else the lane worktree."""
    import argparse
    candidates = [root / "scripts" / "jailbreak_run.py" for root in (REPO, WS1A) if (root / "scripts" / "jailbreak_run.py").is_file()]
    if not candidates:
        pytest.skip("jailbreak_run.py is absent")
    # the tree's copy predates the flags until W; the lane worktree carries them -- take the first copy that DOES, so the
    # round trip runs for real here (before this rule the base tree's copy was picked and the test skipped silently)
    script = next((c for c in candidates if "--multiturn-target-policy-ref" in c.read_text()), None)
    if script is None:
        pytest.skip(f"no jailbreak_run.py among {candidates} carries WS1-A's --multiturn-target-policy-ref (lands at W)")
    sys.path.insert(0, str(script.parent))
    spec = importlib.util.spec_from_file_location("jailbreak_run_ws1a_under_test", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    class _Parsed(Exception):
        def __init__(self, ns):
            self.ns = ns
    real = argparse.ArgumentParser.parse_args

    def capture(self, args=None, namespace=None):
        ns = real(self, args, namespace)
        raise _Parsed(ns)
    old_argv = sys.argv
    sys.argv = ["jailbreak_run.py", *argv]
    argparse.ArgumentParser.parse_args = capture
    try:
        mod.main()
    except _Parsed as e:
        return e.ns
    finally:
        argparse.ArgumentParser.parse_args = real
        sys.argv = old_argv
    raise AssertionError("main() returned without parsing")


class TestWS1A_multiturn_target_flags:
    def test_every_multiturn_child_carries_decode_ref_and_seed_base(self, draft):
        for r in draft["standups"]:
            if r["leg"] not in lp.MULTITURN_LEGS:
                continue
            for k in r["children"]:
                a = k["argv"]
                assert a[a.index("--multiturn-target-decode") + 1] == "vendor"
                assert a[a.index("--multiturn-target-policy-ref") + 1] == "qwen35_4b_it|multi_turn|off|vendor_preset@v1"
                assert a[a.index("--multiturn-target-seed-base") + 1] == str(k["s"]) and a[a.index("--multiturn-target-seed") + 1] == "per_turn"
        bon = _row(draft, "bon", "s3")["children"][0]["argv"]
        assert bon[bon.index("--bon-decode-preset") + 1] == "vendor" and bon[bon.index("--decode-seed-policy") + 1] == "replicate_v1" and bon[bon.index("--decode-seed-base") + 1] == "3"
        # known-bads: a ref-only child (refused under per_turn) and a child without the ref
        for drop in ("--multiturn-target-seed-base", "--multiturn-target-policy-ref"):
            bad = copy.deepcopy(draft)
            k = _row(bad, "pair")["children"][0]
            i = k["argv"].index(drop); del k["argv"][i:i + 2]
            assert any("must carry --multiturn-target-decode vendor, --multiturn-target-policy-ref" in p for p in lp.validate_plan(bad)), drop

    def test_round_trip_through_ws1a_real_parser(self, draft):
        k = _row(draft, "crescendo", "s2")["children"][0]
        argv = [a for a in k["argv"][1:]]                                   # drop the driver python
        argv = [a.replace("{proxy_url}", "http://127.0.0.1:9/v1").replace("{routing_table}", "/t.json").replace("{attempt_root}", "/a")
                .replace("{pool_url:seedrep-attacker-0}", "https://pool0").replace("{pool_url:seedrep-attacker-1}", "https://pool1") for a in argv]
        assert argv[0].endswith("scripts/jailbreak_run.py")
        ns = _parse_through_ws1a_real_parser(argv[1:])
        assert ns.cmd == "generate" and ns.attack == "crescendo" and ns.defense == k["cell"] and ns.trial == 0
        assert ns.multiturn_target_decode == "vendor" and ns.multiturn_target_seed == "per_turn"
        assert ns.multiturn_target_seed_base == 2 and ns.multiturn_target_policy_ref == "qwen35_4b_it|multi_turn|off|vendor_preset@v1"
        assert ns.model_key == "qwen35_4b" and ns.target_engine == "served" and ns.attacker_endpoint.startswith("https://pool")
        with pytest.raises(SystemExit):
            _parse_through_ws1a_real_parser(argv[1:] + ["--not-a-flag", "x"])

    @staticmethod
    def _resolved(k):
        return [a.replace("{proxy_url}", "http://127.0.0.1:9/v1").replace("{routing_table}", "/t.json").replace("{attempt_root}", "/a")
                .replace("{pool_url:seedrep-attacker-0}", "https://pool0").replace("{pool_url:seedrep-attacker-1}", "https://pool1")
                .replace("{pool_url:seedrep-attacker-2}", "https://pool2").replace("{pool_url:seedrep-attacker-3}", "https://pool3") for a in k["argv"][2:]]

    def test_round_trip_every_engine_kind_incl_arditi_pair_and_a_dose_child(self, inputs, draft):
        """The three flags reach `jailbreak_run`'s parser as ints / strings for an Arditi seed-3 ActorAttack child, a PAIR
        child, and a seed-1 DOSE child (the peer's view): `--multiturn-target-seed-base` is the child's literal s, the ref is
        the multi_turn vendor row, the decode is `vendor`."""
        arditi = [k for k in _row(draft, "actorattack", "arditi")["children"] if k["s"] == 3][0]
        pair = _row(draft, "pair", "s1")["children"][0]
        ro, pins = TestDoseStandups._peer(inputs)
        dosed = _plan(inputs, peer_roster=ro, peer_pins=pins, register_peer_dose_standups=True)
        dose_kid = _row(dosed, "crescendo", "d1p5_loradiff64s1")["children"][1]
        for k, s in ((arditi, 3), (pair, 1), (dose_kid, 1)):
            ns = _parse_through_ws1a_real_parser(self._resolved(k))
            assert ns.multiturn_target_seed_base == s and k["s"] == s
            assert ns.multiturn_target_policy_ref == lp.MULTITURN_DECODE_POLICY_REF == "qwen35_4b_it|multi_turn|off|vendor_preset@v1"
            assert ns.multiturn_target_decode == "vendor" and ns.multiturn_target_seed == "per_turn" and ns.defense == k["cell"]
        assert dose_kid["cell"] == lp.LORA_DIFF_CELL and dose_kid["arm"] == "d1p5_eb64s1__" + lp.LORA_DIFF_CELL
        # known-bad: a dose child without the decode switch, and one without the seed base, are refused by validate_plan by name
        for drop in ("--multiturn-target-decode", "--multiturn-target-seed-base"):
            bad = copy.deepcopy(dosed)
            k = _row(bad, "crescendo", "d1p5_loradiff64s1")["children"][1]
            i = k["argv"].index(drop); del k["argv"][i:i + 2]
            assert any("must carry --multiturn-target-decode vendor" in p and "d1p5_eb64s1__" in p for p in lp.validate_plan(bad)), drop


# =========================================================================== main #211 (3353a3a5): pools before multi-turn targets

class TestLaunchOrder_attacker_presence:
    def test_plan_encodes_pools_then_targets_and_every_multiturn_row_declares_its_pool(self, draft):
        order = {st["stage"]: st for st in draft["launch_order"]}
        assert list(order) == ["attacker_pools", "multiturn_targets", "other_targets_and_rungs"]
        assert order["attacker_pools"]["rows"] == [p["name"] for p in draft["pools"]] and order["attacker_pools"]["requires"] == []
        mt = [r for r in draft["standups"] if r["leg"] in lp.MULTITURN_LEGS]
        assert order["multiturn_targets"]["rows"] == [r["standup_id"] for r in mt] and len(mt) == 7 + 14 + 17
        assert "3353a3a5" in order["multiturn_targets"]["why"] and any("/health" in x for x in order["multiturn_targets"]["requires"])
        for r in mt:
            assert r["depends_on"]["pool"] == r["pool"] and r["depends_on"]["pool_urls_json"] == "RS/pool_urls.json" and "3353a3a5" in r["depends_on"]["why"]
        assert all("depends_on" not in r for r in draft["standups"] if r["leg"] not in lp.MULTITURN_LEGS)

    def test_known_bad_a_target_child_with_no_pool_endpoint_is_refused_at_validation(self, draft):
        bad = copy.deepcopy(draft)
        r = _row(bad, "actorattack", "s2")
        for k in r["children"]:
            i = k["argv"].index("--attacker-endpoint"); del k["argv"][i:i + 2]
        probs = lp.validate_plan(bad)
        assert any("carries no --attacker-endpoint" in p and "refuse instantly" in p for p in probs), probs
        bad = copy.deepcopy(draft)
        r = _row(bad, "pair", "s3")
        r["depends_on"]["pool"] = "seedrep-attacker-0"                   # declared dependency drifts from the pinned pool
        assert any("depends_on.pool == its pinned pool" in p for p in lp.validate_plan(bad))

    def test_spawn_refuses_an_unhealthy_or_unregistered_pool(self, srv, inputs):
        final = _plan(inputs, materialisation="final", s4=lp.S4_DEFAULTS)
        urls = {f"seedrep-attacker-{k}": f"https://pool{k}.modal.run" for k in range(4)}
        probed = []

        def health(url):
            probed.append(url)
            return url != "https://pool2.modal.run"
        rows, _ = srv.select_launch_rows(final, legs="pair", pool_urls=urls, pool_health=health, pool_bearer="fw-test-bearer", pool_accepts=lambda u, b: True)
        assert len(rows) == 7 and probed == ["https://pool3.modal.run"], "one probe per distinct pinned pool"
        with pytest.raises(SystemExit, match="is not healthy"):
            srv.select_launch_rows(final, legs="actorattack", pool_urls=urls, pool_health=health)     # pool 2 down -> no target spawns
        with pytest.raises(SystemExit, match="has no URL"):
            srv.select_launch_rows(final, legs="actorattack", pool_urls={k: v for k, v in urls.items() if k != "seedrep-attacker-2"}, pool_health=lambda u: True)
        # the default probe: 200 / 401 are UP, a refused connection is DOWN
        assert srv.pool_is_healthy("http://127.0.0.1:9/") is False


# =========================================================================== the wave uuid spelling every peer validates

class TestWaveUuid:
    def test_hex_refused_dashed_admitted_and_minted(self, inputs, tmp_path, L, drive):
        with pytest.raises(ValueError, match="dashed uuid4"):
            _plan(inputs, wave_uuid="22222222333344448444555555555555")
        assert lp.check_wave_uuid(WAVE) == WAVE and lp.wave_uuid8(WAVE) == "22222222"
        L.SixAxisKey("2222222", WAVE, "pasttense", "s1", "base", "0")          # the ladder admits it
        assert drive.wave_uuid8(WAVE) == "22222222"                              # the drive admits it
        with pytest.raises(L.LadderRefusal):
            L.SixAxisKey("2222222", WAVE.replace("-", ""), "pasttense", "s1", "base", "0")
        rs = tmp_path / "rs"
        rc = lp.main(["--repo", str(REPO), "--w", W, "--wave-json", str(rs / "wave.json"), "--mint-wave", "--materialisation", "draft",
                      "--out", str(rs / "draft.json"), "--ledger", str(rs / "ledger.json")])
        assert rc == 0 and lp.WAVE_UUID_RE.match(json.loads((rs / "wave.json").read_text())["wave_uuid"])
        plan = json.loads((rs / "draft.json").read_text())
        assert plan["standups"][0]["accepted_key"] == L.key_str(L.SixAxisKey(plan["W7"], plan["wave_uuid"], "pasttense", "s1", "base", "0"))


# =========================================================================== the multi-turn window's provenance

class TestWindowProvenance:
    def test_the_rule_is_the_fronts_and_pinned(self, draft):
        from sisteer.capeval.serving import geometry as sg
        assert lp.MULTITURN_CAP4096_WINDOWS == sg.MULTITURN_CAP4096_WINDOWS and lp.MULTITURN_CAP4096_MEASURED_MAX_CONTEXT == sg.MULTITURN_CAP4096_MEASURED_MAX_CONTEXT
        assert lp.MULTITURN_CAP4096_SOURCE == sg.MULTITURN_CAP4096_SOURCE and lp.MULTITURN_CAP4096_CAP == sg.MULTITURN_CAP4096_CAP
        prov = draft["multiturn_window_provenance"]
        assert prov["crescendo"]["rule"] == "measured_maximum_on_complete_store_at_the_served_cap" and prov["crescendo"]["window"] == 49152
        assert prov["crescendo"]["how_chosen"].startswith("the front's default for crescendo") and prov["crescendo"]["kernel_S"] == 10
        assert prov["crescendo"]["crescendo_readings_by_model"]["qwen35_4b_sampled"] == 8597 and prov["crescendo"]["crescendo_readings_by_model"]["glm46vflash"] == 41433
        assert prov["actorattack"]["window_default_for_attack"] is None and "NO measured 4,096-cap window" in prov["actorattack"]["how_chosen"]
        assert prov["pair"]["window_default_for_attack"] == 16384 and prov["pair"]["window"] == 24576, "plan §5's PAIR window exceeds the front's measured 16,384 -- recorded, for the coordinator"
        for leg in lp.MULTITURN_LEGS:
            assert _row(draft, leg)["window_provenance"]["window"] == _row(draft, leg)["geometry"]["L"]
        assert draft["s5_multiturn_probe"]["cells"][0]["object_sha256"].startswith("86fa8e9c") and draft["s5_multiturn_probe"]["cells"][1]["object_sha256"].startswith("0522a255")

    def test_gate_refuses_percentile_habits_and_computes_what_the_rule_would_select(self):
        def rec(arm, kind, ctx, **over):
            r = {"schema": lp.S5_WINDOW_MEASUREMENT_SCHEMA, "window_source": lp.S5_WINDOW_SOURCE, "leg": "crescendo", "arm": arm, "arm_kind": kind, "model_key": "qwen35_4b",
                 "decode": "vendor", "decode_policy_ref": lp.MULTITURN_DECODE_POLICY_REF, "objects_measured": [arm], "store_path": f"/vol/x/{arm}", "n_goals_completed": 20,
                 "n_goals_expected": 20, "store_complete": True, "context_max": ctx, "clamped_calls": 0, "cap": 4096, "measured_utc": "2026-09-18T00:00:00Z",
                 "wave_uuid": WAVE, "attempt_id": "ta-1"}
            r.update(over)
            return r
        bare = [{"leg": "crescendo", "arm": "none", "arm_kind": "parent", "model_key": "qwen35_4b", "decode": "vendor", "context_max": 8597, "clamped_calls": 0, "store_complete": True},
                {"leg": "crescendo", "arm": "d1p5_eb64s1__constv2-mix-full__lora_r4_ce_minus_ctl", "arm_kind": "dosed", "model_key": "qwen35_4b", "decode": "vendor", "context_max": 12000, "clamped_calls": 0, "store_complete": True},
                {"leg": "crescendo", "arm": "dm0p5_eb64s1__constv2-mix-full__lora_r4_ce_minus_ctl", "arm_kind": "dosed", "model_key": "qwen35_4b", "decode": "vendor", "context_max": 7000, "clamped_calls": 0, "store_complete": True}]
        good = [rec("none", "parent", 8597), rec(bare[1]["arm"], "dosed", 12000), rec(bare[2]["arm"], "dosed", 7000)]
        # the OLD gate admitted these bare readings on their numbers; now every row must be a complete measurement record
        assert not lp.multiturn_window_gate("crescendo", 49152, bare)["ok"] and all("bare reading" in p for p in lp.multiturn_window_gate("crescendo", 49152, bare)["problems"][:3])
        g = lp.multiturn_window_gate("crescendo", 49152, good, wave_uuid=WAVE)
        assert g["ok"] and g["measured_max_context"] == 12000 and g["rule_would_select"] == {"L": 16384, "S": 31, "containers_per_view": 2, "reference": {"L": 49152, "S": 10, "containers_per_view": 4}}
        assert lp.implied_multiturn_shape(49152)["S"] == 10 and lp.implied_multiturn_shape(16384)["S"] == 31 and lp.implied_multiturn_shape(24576)["S"] == 21
        assert not lp.multiturn_window_gate("crescendo", 49152, good[:1])["ok"]                                # no dosed object
        assert not lp.multiturn_window_gate("crescendo", 49152, [good[0], good[1]])["ok"]                       # dm0p5 missing (the probe brackets both)
        assert not lp.multiturn_window_gate("crescendo", 49152, [{**good[0], "store_complete": False}, good[1], good[2]])["ok"]   # every early reading is a floor
        assert not lp.multiturn_window_gate("crescendo", 49152, [{**good[0], "clamped_calls": 1}, good[1], good[2]])["ok"]
        assert not lp.multiturn_window_gate("crescendo", 8192, good)["ok"]                                       # a window below a reading
        assert not lp.multiturn_window_gate("crescendo", 49152, [{**good[0], "decode": "greedy"}, good[1], good[2]])["ok"]
        # a foreign wave's record (syntactically valid) refuses under exact equality; a record with no wave check passes without one
        foreign = [{**good[0], "wave_uuid": SMOKE_WAVE}, good[1], good[2]]
        assert any("foreign wave" in p for p in lp.multiturn_window_gate("crescendo", 49152, foreign, wave_uuid=WAVE)["problems"])
        assert lp.multiturn_window_gate("crescendo", 49152, foreign)["ok"]
        # a schema-bearing record with a percentile source / a missing attempt id refuses through window_measurement_problems
        assert any("never a percentile" in p for p in lp.multiturn_window_gate("crescendo", 49152, [{**good[0], "window_source": "p99"}, good[1], good[2]], wave_uuid=WAVE)["problems"])
        assert any("missing attempt_id" in p for p in lp.multiturn_window_gate("crescendo", 49152, [{k: v for k, v in good[0].items() if k != "attempt_id"}, good[1], good[2]])["problems"])


# =========================================================================== the dose standups + the S5 probe

class TestDoseStandups:
    """The peer's dose standups register from the peer's OWN roster + pin file, given BY PATH (`--peer-roster`,
    `--peer-pins`): view names `d{m0p5,0p5,1p5}_{loradiff64s1,pairdiff64s1}`, cells keyed by the PARENT cell id, staged ids `<dose>_eb64s1__<parent>`
    (renamed 2026-09-18; shas unchanged). The fixture DERIVES its shape from this repo's own roster / pin entries, never hand-typed keys.
    Without the files the standups are `pending_peer_spec`; the shared roster is never a source of peer views (its batch-8 ladders are skipped,
    a seed-1 view found there is recorded, not stood up)."""
    LORA = lp.LORA_DIFF_CELL
    PAIRED = lp.PAIRED_DIFF_CELL
    VIEWS = {"dm0p5_loradiff64s1": ("dm0p5", LORA), "d0p5_loradiff64s1": ("d0p5", LORA), "d1p5_loradiff64s1": ("d1p5", LORA),
             "dm0p5_pairdiff64s1": ("dm0p5", PAIRED), "d0p5_pairdiff64s1": ("d0p5", PAIRED), "d1p5_pairdiff64s1": ("d1p5", PAIRED)}

    @classmethod
    def _peer(cls, inputs):
        """The peer's two files, shaped from this repo's OWN writers' output: a view block carries the s1 block's non-cell keys, a pin entry this repo's
        pin entry keys; the six `_eb64s1` children under their parent cells with the bank-measured shas (`lp.DOSE_CHILD_SHA`)."""
        s1 = inputs["roster"]["targets"]["qwen35_4b"]["views"]["s1"]
        view_keys = [k for k in s1 if k not in ("cells", "caps")]                    # config, manifest (the roster writer's per-view keys)
        pin_template = next(iter(inputs["pins"]["qwen35_4b"].values()))              # object_sha256, sidecar_..., source, verified_*
        views = {}
        pins = {"qwen35_4b": {}}
        for v, (tok, parent) in cls.VIEWS.items():
            sid = f"{tok}_eb64s1__{parent}"
            views[v] = {**{k: (s1[k].replace("_s1", f"_{v}") if isinstance(s1[k], str) else s1[k]) for k in view_keys}, "cells": {parent: sid}}
            pins["qwen35_4b"][sid] = {**{k: None for k in pin_template}, "object_sha256": lp.DOSE_CHILD_SHA[sid],
                                      "source": f"$STAGE/sdfbase/full22_objects/qwen35_4b/cells/{sid}"}
        pins["qwen35_4b"][lp.PAIRED_DIFF_PARENT_STAGED_ID] = {**pins["qwen35_4b"][f"d1p5_eb64s1__{cls.PAIRED}"], "object_sha256": lp.PAIRED_DIFF_PARENT_SHA256}
        roster = {k: v for k, v in inputs["roster"].items() if k != "targets"}
        roster["targets"] = {"qwen35_4b": {"views": views}}
        return roster, pins

    def test_d71_dose_standups_are_not_in_this_wave_by_default(self, inputs, draft):
        """The peer's sweep runs on cluster A; the plan carries NO dose standups and the ledger no dose line, with or without the peer files;
        the registration path is a dormant, default-off capability."""
        rec = draft["dose_standups"]
        assert rec["status"] == "not_in_this_wave" == lp.DOSE_STATUS_NOT_IN_WAVE and rec["standups"] == [] and rec["n_views"] == 0 and "D71" in rec["why"]
        assert rec["peer_files_given"] is False and rec["pending"]["containers_when_registered"] == 0 and rec["pending"]["dormant_path"].startswith("--register-peer-dose-standups")
        led = draft["capacity_ledger"]
        assert led["unstaged_peak"] == led["unstaged_peak_without_dose"] == {"containers": 134, "gpus": 134} and led["dose_standups"]["containers"] == 0
        assert draft["s5_probe_rows"] == [] and draft["s5_multiturn_probe"]["status"].startswith("UNREGISTERED") and lp.validate_plan(draft) == []
        ro, pins = self._peer(inputs)
        with_files = _plan(inputs, peer_roster=ro, peer_pins=pins)
        assert with_files["dose_standups"]["status"] == "not_in_this_wave" and with_files["dose_standups"]["peer_files_given"] is True
        assert not any(r.get("dose_standup") for r in with_files["standups"]) and with_files["capacity_ledger"]["unstaged_peak"] == {"containers": 134, "gpus": 134}
        # the guard: a dose standup smuggled into the production plan while the status says not_in_this_wave is refused
        bad = copy.deepcopy(with_files)
        bad["standups"].append({**with_files["s5_probe_rows"][0], "dose_standup": True, "kind": "base", "standup_id": "controls__zzz__base__shard00",
                                "control_key": lp.control_key(bad["W7"], bad["wave_uuid"], "controls", "zzz", "base", 0), "accepted_key": lp.accepted_key(bad["W7"], bad["wave_uuid"], "controls", "zzz", "base", 0)})
        assert any("D71" in p for p in lp.validate_plan(bad))
        g = _load_peer("seedrep_s0_gates_d71", "scripts/modal/seedrep_s0_gates.py", REPO)
        gate = g.gate_dose_registration(plan=draft)
        assert gate["ok"] and gate["status"].startswith("NOT IN THIS WAVE (D71)") and gate["registered"] == [] and gate["unstaged_peak"] == {"containers": 134, "gpus": 134}
        assert "dormant" in gate["status"] and gate["s5_probe_rows"] == 0

    def test_absent_peer_files_are_pending_never_failing_the_s_views(self, inputs, draft):
        """the DORMANT path (--register-peer-dose-standups), without the peer files: pending_peer_spec, the s1-s3 legs untouched"""
        draft = _plan(inputs, register_peer_dose_standups=True)
        rec = draft["dose_standups"]
        assert rec["status"] == "pending_peer_spec" == lp.DOSE_STATUS_PENDING and rec["standups"] == [] and rec["n_views"] == 0
        assert rec["pending"]["containers_when_registered"] == 18 and rec["pending"]["s1_s3_legs_affected"] is False and rec["pins_source"] is None
        assert rec["peer_files"] == {"roster": lp.PEER_ROSTER_REL, "pins": lp.PEER_PINS_REL, "expected_views": list(lp.PEER_EXPECTED_VIEWS)}
        assert rec["discovery"] == lp.PEER_DISCOVERY_RULE and "NEVER read" in rec["discovery"]
        declared = {v for v, b in inputs["roster"]["targets"]["qwen35_4b"]["views"].items() if isinstance(b, dict) and "dose" in b}
        assert declared and declared <= set(rec["skipped_shared_roster"]) and all("O23" in w for w in rec["skipped_shared_roster"].values())
        assert rec["shared_roster_seed1_views_not_stood_up"] == []
        led = draft["capacity_ledger"]
        assert led["unstaged_peak"] == led["unstaged_peak_without_dose"] == {"containers": 134, "gpus": 134}
        assert led["dose_standups"]["containers"] == 0 and led["dose_standups"]["status"] == "pending_peer_spec" and led["dose_standups"]["containers_when_registered"] == 18
        assert draft["s5_multiturn_probe"]["status"].startswith("UNREGISTERED") and lp.validate_plan(draft) == []
        with pytest.raises(ValueError, match="go together"):
            lp.load_inputs(REPO, peer_roster=REPO / "x.yaml")
        with pytest.raises(ValueError, match="--peer-pins is required"):
            _plan(inputs, peer_roster=self._peer(inputs)[0], register_peer_dose_standups=True)

    def test_peer_files_give_three_single_container_standups_per_view_outside_the_s_views(self, inputs, draft):
        """the DORMANT path switched on, kept default-off"""
        ro, pins = self._peer(inputs)
        assert set(ro["targets"]["qwen35_4b"]["views"]) == set(lp.PEER_EXPECTED_VIEWS)
        plan = _plan(inputs, peer_roster=ro, peer_pins=pins, register_peer_dose_standups=True)
        assert lp.validate_plan(plan) == []
        rec = plan["dose_standups"]
        assert sorted(rec["views"]) == sorted(lp.PEER_EXPECTED_VIEWS) and rec["n_views"] == 6 and rec["n_standups"] == 18 == rec["containers"]
        assert rec["status"] == "6 peer seed-1 dose view(s) -> 18 single-container standups (per_dose_view)" and rec["pins_source"] == "peer pin file"
        assert rec["unexpected_view_names"] == [] and len(rec["standups"]) == 18 and len(plan["standups"]) == 74 + 18
        led = plan["capacity_ledger"]
        assert led["unstaged_peak"] == {"containers": 134 + 18, "gpus": 134 + 18} and led["unstaged_peak_without_dose"] == {"containers": 134, "gpus": 134}
        assert led["dose_standups"]["containers"] == 18 and led["dose_standups"]["views"] == sorted(lp.PEER_EXPECTED_VIEWS)
        assert led["by_leg"]["dose:controls"] == {"target_containers": 6, "rung_containers": 0} == led["by_leg"]["dose:crescendo"] == led["by_leg"]["dose:actorattack"]
        for sid in rec["standups"]:
            r = [x for x in plan["standups"] if x["standup_id"] == sid][0]
            assert r["dose_standup"] and r["kind"] == "base" and r["producers"] == [] and r["modal"]["max_containers"] == 1 and r["seed_bases"] == [1]
        d = _row(plan, "controls", "d1p5_loradiff64s1")
        a = d["argv"]
        assert d["route_set"] == ["none", self.LORA] and d["readiness"]["max_loras"] == 2 and d["readiness"]["residual_slot_armed"] == []
        assert a[a.index("--config") + 1] == ro["targets"]["qwen35_4b"]["views"]["d1p5_loradiff64s1"]["config"] and a[a.index("--defenses-dir") + 1].endswith("/cells_view/d1p5_loradiff64s1")
        assert a.count("--cell") == 1 and a[a.index("--cell") + 1] == self.LORA and a[a.index("--expect-object-sha256") + 1] == f"{self.LORA}={lp.DOSE_CHILD_SHA['d1p5_eb64s1__' + self.LORA]}"
        assert d["objects"] == {self.LORA: {"staged_id": "d1p5_eb64s1__" + self.LORA, "object_sha256": lp.DOSE_CHILD_SHA["d1p5_eb64s1__" + self.LORA], "dose_view": "d1p5_loradiff64s1"}}
        assert [k["arm"] for k in d["children"]] == ["d1p5_loradiff64s1__none", "d1p5_eb64s1__" + self.LORA] and d["eval_legs"] == ["xstest"]
        k = d["children"][1]
        assert k["argv"][k["argv"].index("--controls") + 1] == "xstest" and k["registry_identity_rel"] == f"registry/plans/_stores/controls/d1p5_eb64s1__{self.LORA}__s1.json"   # the producer's per-task STORE map
        p = _row(plan, "crescendo", "dm0p5_pairdiff64s1")
        assert p["route_set"] == ["none", self.PAIRED] and p["readiness"]["residual_slot_armed"] == [self.PAIRED] and p["route_set_harness"] == ["none", lp.harness_id(self.PAIRED)]
        assert lp.harness_id(self.PAIRED) == "si_constv2_mix_full__psr_ce_all_minus_ctl_distilled" and p["pool"] == "seedrep-attacker-0" == p["depends_on"]["pool"]
        assert p["geometry"]["L"] == 49152 and all(k["argv"][k["argv"].index("--multiturn-target-seed-base") + 1] == "1" and k["dose_view"] == "dm0p5_pairdiff64s1" for k in p["children"])
        for v in lp.VIEWS:
            for leg in ("crescendo", "controls"):
                r = _row(plan, leg, v)
                assert r["route_set"] == lp.expected_route_set(v) and not any(lp.DOSE_S1_STAGED_RE.match(o["staged_id"]) for o in r["objects"].values())
        assert len(plan["adapter_hash_set"]) == 17 + 6
        uses = {u for r in plan["s5_roster"] for u in r["used_by"]}
        assert {"crescendo:base:dose", "actorattack:base:dose", "controls:base:dose"} <= uses
        assert {(x["L"], x["S"]) for x in plan["s5_roster"]} == {(x["L"], x["S"]) for x in draft["s5_roster"]}
        final = lp.build_plan(W=W, wave_uuid=WAVE, reqs_sha256=inputs["reqs_sha256"], roster=inputs["roster"], pins=inputs["pins"], arditi_pins=inputs["arditi_pins"], arms=inputs["arms"],
                              materialisation="final", s4={"multiturn_window": {"crescendo": 16384, "actorattack": 24576}}, generated_at="y", peer_roster=ro, peer_pins=pins,
                              register_peer_dose_standups=True, config_private_runs=inputs["config_private_runs"])
        assert lp.compare_draft_final(plan, final) == []
        assert _row(final, "crescendo", "d1p5_loradiff64s1")["geometry"]["L"] == 16384 and _row(final, "actorattack", "d1p5_pairdiff64s1")["geometry"]["L"] == 24576
        import tempfile, yaml
        with tempfile.TemporaryDirectory() as td:
            rp, pp = Path(td) / "peer.yaml", Path(td) / "peer_pins.json"
            rp.write_text(yaml.safe_dump(ro)); pp.write_text(json.dumps(pins))
            got = lp.load_inputs(REPO, peer_roster=rp, peer_pins=pp)
            assert got["peer_roster"]["targets"]["qwen35_4b"]["views"].keys() == ro["targets"]["qwen35_4b"]["views"].keys() and got["peer_pins"] == pins

    def test_the_ids_and_shas_are_the_banks(self, inputs):
        """The six children are keyed `<dose>_eb64s1__<parent>`; the sha table is the BANK's (sha256(object.pt) under the shared bank, re-read here when
        the bank is present) -- the relayed paired shas (4bacb54c… / 2bdf2371… / 26039360…) are NOT what the bank holds under these ids."""
        assert [c["staged_id"] for c in lp.S5_PROBE_CELLS] == ["d1p5_eb64s1__" + self.LORA, "dm0p5_eb64s1__" + self.LORA]
        assert lp.DOSE_CHILD_SHA["d1p5_eb64s1__" + self.LORA].startswith("86fa8e9c") and lp.DOSE_CHILD_SHA["dm0p5_eb64s1__" + self.LORA].startswith("0522a255")
        assert lp.DOSE_CHILD_SHA["d0p5_eb64s1__" + self.LORA].startswith("669fcc67") and len(lp.DOSE_CHILD_SHA) == 6 and all("_eb64s1__" in k for k in lp.DOSE_CHILD_SHA)
        assert self.PAIRED.endswith("_distilled") and not any(v.startswith(("4bacb54c", "2bdf2371", "26039360")) for v in lp.DOSE_CHILD_SHA.values())
        bank = host_site_path("$STAGE/sdfbase/full22_objects/qwen35_4b/cells")
        if bank.is_dir():
            for sid, sha in lp.DOSE_CHILD_SHA.items():
                f = bank / sid / "object.pt"
                if f.is_file():
                    assert hashlib.sha256(f.read_bytes()).hexdigest() == sha, sid
        assert lp.dose_staged_id_problem("d1p5_eb64s1__" + self.LORA) is None and lp.dose_staged_id_problem("d1p5s1__constv2-mix-full__psr_ce_all") is None
        assert (lp.dose_staged_id_problem("d1p5s1__" + self.LORA) or "").startswith("RETIRED") and (lp.dose_staged_id_problem("d1p5__" + self.LORA) or "").startswith("not a seed-1")

    def test_the_shared_roster_of_strengths_const_ladder_worktree_is_reported_not_stood_up(self, inputs):
        """The peer's live `const-ladder-s1` tree adds `d<tok>_parent` seed-1 views to the SHARED roster; peer views come from the peer
        files only, so they are RECORDED (`shared_roster_seed1_views_not_stood_up`) and no standup is emitted. Skips by name when absent."""
        peer = WORKTREES / "const-ladder-s1"
        ro_p = peer / "configs/sdfbase_full22_roster.yaml"
        if not ro_p.is_file():
            pytest.skip(f"Strength's worktree {peer} is absent")
        import yaml
        roster = yaml.safe_load(ro_p.read_text(encoding="utf-8"))
        seed1, skipped = lp.dose_views(roster)
        if not seed1:
            pytest.skip("that roster carries no seed-1 dose view yet")
        assert all(v.endswith("_parent") for v in seed1) and len(seed1) == 4 and skipped
        plan = _plan(inputs, roster=roster, register_peer_dose_standups=True)
        assert plan["dose_standups"]["status"] == "pending_peer_spec" and plan["dose_standups"]["shared_roster_seed1_views_not_stood_up"] == sorted(seed1)
        assert not any(r.get("dose_standup") for r in plan["standups"]) and lp.validate_plan(plan) == []

    def test_known_bads(self, inputs):
        ro, pins = self._peer(inputs)
        plan = _plan(inputs, peer_roster=ro, peer_pins=pins, register_peer_dose_standups=True)
        bad = copy.deepcopy(plan)
        _row(bad, "crescendo", "s1")["objects"][self.LORA] = {"staged_id": "d1p5_eb64s1__" + self.LORA, "object_sha256": "86" * 32, "dose_view": "d1p5_loradiff64s1"}
        assert any("dose objects" in p and "s-view standup" in p for p in lp.validate_plan(bad))
        bad = copy.deepcopy(plan)
        _row(bad, "crescendo", "s2")["objects"][self.LORA] = {"staged_id": "dm0p5_eb64s1__" + self.LORA, "object_sha256": "05" * 32}
        assert any("dose objects" in p and "s-view standup" in p for p in lp.validate_plan(bad))
        bad = copy.deepcopy(plan)
        _row(bad, "controls", "d1p5_loradiff64s1")["engine"] = "dose_zzz"
        assert any("DISCOVERED seed-1 dose view" in p for p in lp.validate_plan(bad))
        bad = copy.deepcopy(plan)
        _row(bad, "crescendo", "d1p5_loradiff64s1")["objects"][self.LORA]["staged_id"] = "d1p5s1__" + self.LORA
        assert any("RETIRED spelling" in p for p in lp.validate_plan(bad))
        ro2 = copy.deepcopy(ro); ro2["targets"]["qwen35_4b"]["views"]["d1p5_loradiff64s1"]["cells"] = {self.LORA: "d1p5__" + self.LORA}
        with pytest.raises(ValueError, match="not a seed-1 child"):
            lp.peer_dose_views(ro2)
        ro3 = copy.deepcopy(ro); ro3["targets"]["qwen35_4b"]["views"]["d1p5_loradiff64s1"]["cells"] = {self.LORA: "d1p5s1__" + self.LORA}
        with pytest.raises(ValueError, match="RETIRED spelling"):
            lp.peer_dose_views(ro3)
        ro4 = copy.deepcopy(ro); ro4["targets"]["qwen35_4b"]["views"]["d1p5_loradiff64s1"]["cells"] = {self.PAIRED: "d1p5_eb64s1__" + self.LORA}
        with pytest.raises(ValueError, match="keyed under the cell"):
            lp.peer_dose_views(ro4)
        pins_wrong = copy.deepcopy(pins); pins_wrong["qwen35_4b"]["d1p5_eb64s1__" + self.LORA]["object_sha256"] = "ab" * 32
        with pytest.raises(ValueError, match="not the sha Strength relayed|not D45's"):
            _plan(inputs, peer_roster=ro, peer_pins=pins_wrong, register_peer_dose_standups=True)
        pins_missing = copy.deepcopy(pins); del pins_missing["qwen35_4b"]["dm0p5_eb64s1__" + self.PAIRED]
        with pytest.raises(ValueError, match="peer pin file\\[qwen35_4b\\]"):
            _plan(inputs, peer_roster=ro, peer_pins=pins_missing, register_peer_dose_standups=True)
        with pytest.raises(ValueError, match="no views of qwen35_4b"):
            lp.peer_dose_views({"targets": {"qwen35_4b": {"views": {}}}})


class TestWindowJustificationGatedLegs:
    def test_gated_legs_derive_from_the_probe_and_pair_is_recorded_ungated(self, inputs):
        """`window_justification` once gated every member of MULTITURN_LEGS incl. PAIR while the probe declares crescendo +
        actorattack only, so a FINAL could never materialise. The gated legs ARE the probe's; PAIR's provenance records why it is not
        S5-gated (not S4-decided; the front's measured window); a missing PAIR measurement is not a problem, a missing ActorAttack one is."""
        final = _plan(inputs, materialisation="final", s4={"multiturn_window": {"crescendo": 16384, "actorattack": 16384}})
        probe = [c["staged_id"] for c in lp.S5_PROBE_CELLS]

        def rec(leg, arm, kind, ctx):
            return {"schema": lp.S5_WINDOW_MEASUREMENT_SCHEMA, "window_source": lp.S5_WINDOW_SOURCE, "leg": leg, "arm": arm, "arm_kind": kind, "model_key": "qwen35_4b",
                    "decode": "vendor", "decode_policy_ref": lp.MULTITURN_DECODE_POLICY_REF, "objects_measured": [arm], "store_path": f"/vol/x/{leg}/{arm}", "n_goals_completed": 20,
                    "n_goals_expected": 20, "store_complete": True, "context_max": ctx, "clamped_calls": 0, "cap": 4096, "measured_utc": "2026-09-18T00:00:00Z",
                    "wave_uuid": SMOKE_WAVE, "attempt_id": "ta-1"}
        ms = [rec(leg, arm, kind, ctx) for leg in lp.S5_PROBE_LEGS for arm, kind, ctx in (("none", "parent", 8597), (probe[0], "dosed", 12000), (probe[1], "dosed", 7000))]
        # the records were MEASURED under the smoke's execution wave; the document records the production target beside it
        just = lp.window_justification(final, ms, execution_wave_uuid=SMOKE_WAVE)
        assert just["execution_wave_uuid"] == SMOKE_WAVE and just["production_wave_uuid"] == final["wave_uuid"] == just["wave_uuid"] and just["launch_plan_sha"] == final["launch_plan_sha"]
        assert just["problems"] == [] and just["gated_legs"] == ["crescendo", "actorattack"] and set(just["legs"]) == {"crescendo", "actorattack"}
        assert just["ungated_legs"] == {"pair": {"window": final["multiturn_window_provenance"]["pair"]["window"], "how_chosen": final["multiturn_window_provenance"]["pair"]["how_chosen"], "why": lp.S5_UNGATED_WHY}}
        assert final["multiturn_window_provenance"]["pair"]["gate"] == {"leg": "pair", "gated": False, "why": lp.S5_UNGATED_WHY} and "not S5-gated" in lp.S5_UNGATED_WHY
        assert tuple(lp.S4_DEFAULTS["multiturn_window"]) == lp.S5_PROBE_LEGS, "the probe measures exactly the S4-decided windows"
        # known-bads: ActorAttack unmeasured refuses; a PAIR record from a foreign wave is recorded but gates nothing; an ActorAttack record stamped
        # with the PRODUCTION wave refuses (no smoke runs there -- the pre-fix binding); an execution wave equal to the production wave refuses outright
        assert any(p.startswith("actorattack window") for p in lp.window_justification(final, [m for m in ms if m["leg"] != "actorattack"], execution_wave_uuid=SMOKE_WAVE)["problems"])
        assert lp.window_justification(final, ms + [{**rec("pair", "none", "parent", 9000), "wave_uuid": final["wave_uuid"]}], execution_wave_uuid=SMOKE_WAVE)["problems"] == []
        prod_stamped = [{**m, "wave_uuid": final["wave_uuid"]} if m["leg"] == "actorattack" else m for m in ms]
        probs = lp.window_justification(final, prod_stamped, execution_wave_uuid=SMOKE_WAVE)["problems"]
        assert any("foreign wave" in p and "execution wave" in p for p in probs)
        with pytest.raises(ValueError, match="no smoke runs under the production wave"):
            lp.window_justification(final, ms, execution_wave_uuid=final["wave_uuid"])


class TestS5ProbeRegistration:
    """The window probe REGISTERS d1p5_eb64s1 / dm0p5_eb64s1 beside a parent on the multi-turn dose standups and records where; the
    measurement record's provenance is REQUIRED by key; the window rule is the measured MAXIMUM on a COMPLETE store."""

    def test_registered_on_both_probe_legs_when_the_peer_pins_the_pair(self, inputs):
        """The two probe cells register on SMOKE-ONLY probe rows (never production, never the ledger) beside the `none`
        parent, built from the peer files; the default plan (no dose standups) still carries them."""
        ro, pins = TestDoseStandups._peer(inputs)
        plan = _plan(inputs, peer_roster=ro, peer_pins=pins)
        assert not any(r.get("dose_standup") for r in plan["standups"]) and plan["capacity_ledger"]["unstaged_peak"] == {"containers": 134, "gpus": 134}
        rows = plan["s5_probe_rows"]
        assert [r["standup_id"] for r in rows] == ["crescendo__d1p5_loradiff64s1__base__shard00", "actorattack__d1p5_loradiff64s1__base__shard00",
                                                   "crescendo__dm0p5_loradiff64s1__base__shard00", "actorattack__dm0p5_loradiff64s1__base__shard00"]
        assert all(r["kind"] == "s5_probe" and r["smoke_only"] and r["route_set"][0] == "none" and len(r["route_set"]) == 2 and r["readiness"]["max_loras"] == 2 for r in rows)
        assert all(r["standup_id"] not in {x["standup_id"] for x in plan["standups"]} for r in rows) and lp.validate_plan(plan) == []
        probe = plan["s5_multiturn_probe"]
        assert probe["legs"] == ["crescendo", "actorattack"] == list(lp.S5_PROBE_LEGS) and probe["parent_arm"] == "none"
        assert probe["status"].startswith("registered:")
        reg = probe["registration"]
        LORA = lp.LORA_DIFF_CELL
        assert reg["d1p5_eb64s1__" + LORA] == {
            "registered_on": {"crescendo": "crescendo__d1p5_loradiff64s1__base__shard00", "actorattack": "actorattack__d1p5_loradiff64s1__base__shard00"},
            "sha_mismatch": {}, "complete": True}
        assert reg["dm0p5_eb64s1__" + LORA]["complete"] and reg["dm0p5_eb64s1__" + LORA]["registered_on"]["crescendo"] == "crescendo__dm0p5_loradiff64s1__base__shard00"
        c = [r for r in rows if r["standup_id"] == "crescendo__d1p5_loradiff64s1__base__shard00"][0]
        assert c["route_set"] == ["none", LORA] and c["objects"][LORA]["object_sha256"].startswith("86fa8e9c") and c["objects"][LORA]["s5_probe_cell"] is True
        assert [k["arm"] for k in c["children"]] == ["d1p5_loradiff64s1__none", "d1p5_eb64s1__" + LORA]
        assert all(k["argv"][k["argv"].index("--multiturn-target-seed-base") + 1] == "1" and k["argv"][k["argv"].index("--multiturn-target-decode") + 1] == "vendor" for k in c["children"])
        assert probe["windows_probed"] == {"crescendo": 49152, "actorattack": 49152} and probe["cells"][0]["object_sha256"].startswith("86fa8e9c") and probe["cells"][1]["object_sha256"].startswith("0522a255")
        # a probe row with a non-probe cell, or one that is also a production standup, is refused; --smoke admits a probe row, --go / --legs never
        bad = copy.deepcopy(plan)
        bad["s5_probe_rows"][0]["objects"] = {"constv2-mix-full__psr_ce_all_minus_ctl_distilled": {"staged_id": "d0p5_eb64s1__constv2-mix-full__psr_ce_all_minus_ctl_distilled", "object_sha256": "0" * 64}}
        assert any("S5 probe row must be smoke-only" in p for p in lp.validate_plan(bad))
        bad = copy.deepcopy(plan)
        bad["standups"].append(dict(bad["s5_probe_rows"][0]))
        assert any("S5 probe row must be smoke-only" in p for p in lp.validate_plan(bad))

    def test_known_bads_one_leg_only_and_a_foreign_sha(self, inputs):
        ro, pins = TestDoseStandups._peer(inputs)
        plan = _plan(inputs, peer_roster=ro, peer_pins=pins)
        LORA = lp.LORA_DIFF_CELL
        bad = copy.deepcopy(plan)
        bad["s5_probe_rows"] = [r for r in bad["s5_probe_rows"] if r["standup_id"] != "actorattack__d1p5_loradiff64s1__base__shard00"]
        bad["s5_multiturn_probe"] = lp.s5_multiturn_probe(bad["s4_decisions"], bad["standups"] + bad["s5_probe_rows"])
        assert not bad["s5_multiturn_probe"]["registration"]["d1p5_eb64s1__" + LORA]["complete"]
        assert any("registered on ['crescendo'] only" in p for p in lp.validate_plan(bad)), lp.validate_plan(bad)
        plan2 = copy.deepcopy(plan)
        for r in plan2["s5_probe_rows"]:
            if r["engine"] == "d1p5_loradiff64s1":
                r["objects"][LORA]["object_sha256"] = "ab" * 32
        plan2["s5_multiturn_probe"] = lp.s5_multiturn_probe(plan2["s4_decisions"], plan2["standups"] + plan2["s5_probe_rows"])
        reg = plan2["s5_multiturn_probe"]["registration"]["d1p5_eb64s1__" + LORA]
        assert reg["registered_on"] == {} and set(reg["sha_mismatch"]) == {"crescendo", "actorattack"} and not reg["complete"]
        assert any("not D45's 86fa8e9c" in p for p in lp.validate_plan(plan2))
        assert plan2["s5_multiturn_probe"]["status"].startswith("UNREGISTERED")

# =========================================================================== the watcher's diagnosis and its live sources

class TestP2_14_watcher_sources:
    def _calls(self, draft, leg="pasttense", engine="s1"):
        rows = sorted([r for r in draft["standups"] if r["leg"] == leg and r["engine"] == engine], key=lambda r: r["shard"])
        return [{"call_id": f"fc-{r['shard']}", "standup_id": r["standup_id"], "leg": leg, "engine": engine, "stage": "base", "shard": r["shard"], "kind": "base",
                 "control_key": r["control_key"], "accepted_key": r["accepted_key"], "wave_uuid": WAVE} for r in rows]

    def test_revalidated_producer_missing_is_keyed_by_the_full_accepted_key_incl_shard(self, inputs):
        """The diagnosis was once keyed by leg/engine/stage, so an ACCEPTED sibling shard suppressed PRODUCER_MISSING for a shard
        whose every attempt was terminal and none accepted. Now per accepted key: the dead shard is named, the accepted one is not."""
        import time as _t
        final = _plan(inputs, materialisation="final", s4={"crescendo": {"containers_per_view": 2, "containers_arditi": 2}})
        calls = self._calls(final, "crescendo", "s1")
        assert len(calls) == 2 and calls[0]["shard"] == 0 and calls[1]["shard"] == 1
        snap = {"fc-0": {"done": True, "summary": {"status": "DONE"}, "accepted": "a-0"}, "fc-1": {"done": True, "summary": {"status": "DONE"}, "accepted": None}}
        req = lp.required_accepted_keys(final, "crescendo", "base")
        ev = sw.classify(now=_t.time(), leg="crescendo", calls=calls, snapshot=snap, state=sw.WatchState(), stall_min=20, rows_flat_min=40, progress_min=30, required_keys=req)
        missing = [e for e in ev if e.startswith("PRODUCER_MISSING")]
        assert len(missing) == 1 and missing[0].startswith("PRODUCER_MISSING crescendo/s1/base/shard1 standup=crescendo__s1__base__shard01 key=" + calls[1]["accepted_key"])
        # the repro: keyed without the shard, the accepted sibling hid it
        assert not any("/shard0 " in e for e in missing) and not any(e.startswith("DONE") for e in ev)

    def test_rows_flat_clock_starts_at_started_for_a_zero_row_run(self, draft):
        import time as _t
        iso = lambda t: _t.strftime("%Y-%m-%dT%H:%M:%SZ", _t.gmtime(t))  # noqa: E731
        t0 = 1_800_000_000.0
        calls = self._calls(draft, "crescendo", "s1")[:1]
        st = sw.WatchState()
        hb = {"attempt_id": "ta-1", "rows_written": 0, "running": 8, "phase": "drivers", "t": iso(t0)}
        ev = sw.classify(now=t0, leg="crescendo", calls=calls, snapshot={"fc-0": {"done": False, "heartbeat": hb}}, state=st, stall_min=20, rows_flat_min=40, progress_min=30,
                         required_keys=lp.required_accepted_keys(draft, "crescendo", "base"))
        assert any(e.startswith("STARTED") for e in ev) and st.rows_moved_t["fc-0"] == t0
        ev = sw.classify(now=t0 + 45 * 60, leg="crescendo", calls=calls, snapshot={"fc-0": {"done": False, "heartbeat": {**hb, "t": iso(t0 + 45 * 60)}}}, state=st,
                         stall_min=20, rows_flat_min=40, progress_min=30, required_keys=lp.required_accepted_keys(draft, "crescendo", "base"))
        assert any("rows flat for 45 min (rows=0)" in e for e in ev), ev

    def test_expired_call_resolves_from_the_durable_result_or_complete(self, draft):
        """`OutputExpiredError` once said "read RESULT" and reported a failure; now the attempt root the last heartbeat names is
        read through the Volume API: RESULT's `status=... rc=...` decides (rc 0 -> DONE, no error; rc 1 -> the failure named);
        COMPLETE.json alone -> DONE; neither -> an error naming the root; no heartbeat -> an error naming that."""
        c = self._calls(draft, "pasttense", "s1")[0]
        rung = [r for r in draft["rungs"] if r["leg"] == "pasttense" and r["engine"] == "s1" and r["stage"] == "8192"][0]
        rc_ = {"call_id": "fc-r", "standup_id": rung["standup_id"], "leg": "pasttense", "engine": "s1", "stage": "8192", "shard": 0, "kind": "rung",
               "control_key": rung["control_key"], "accepted_key": rung["accepted_key"], "wave_uuid": WAVE}
        root_b = lp.attempt_root_for(draft["results_root"], leg="pasttense", view="s1", stage="base", shard=0, attempt_id="ta-1", kind="base")
        root_r = lp.attempt_root_for(draft["results_root"], leg="pasttense", view="s1", stage="8192", shard=0, attempt_id="ta-2", kind="rung")
        assert root_b == f"{draft['results_root']}/pasttense/s1/shard00/attempts/ta-1" and root_r == f"{draft['results_root']}/pasttense/s1/rungs/8192/shard00/attempts/ta-2"
        assert lp.attempt_root_for("R", leg="x", view="s1", stage="8192", shard=3, attempt_id="a") == "R/x/s1/rungs/8192/shard03/attempts/a"   # a rung by its stage alone
        files = {f"{root_b}/RESULT": "2026-09-18T00:00:00Z status=DONE rc=0 attempt=ta-1 standup=x\n",
                 f"{root_r}/RESULT": "2026-09-18T00:00:00Z status=LADDER_TAILER_ERROR rc=1 attempt=ta-2 standup=y\n"}
        read = files.get
        out = sw.resolve_expired_call(read, draft["results_root"], c, {"attempt_id": "ta-1"})
        assert out["summary"]["status"] == "DONE" and out["error"] is None and out["resolved_from"] == f"{root_b}/RESULT" and out["summary"]["attempt_root"] == root_b
        out = sw.resolve_expired_call(read, draft["results_root"], rc_, {"attempt_id": "ta-2"})
        assert out["summary"]["status"] == "LADDER_TAILER_ERROR" and out["error"] == "RESULT rc=1 status=LADDER_TAILER_ERROR"
        only_complete = {f"{root_b}/COMPLETE.json": "{}"}
        out = sw.resolve_expired_call(only_complete.get, draft["results_root"], c, {"attempt_id": "ta-1"})
        assert out["summary"]["status"] == "DONE" and out["resolved_from"].endswith("/COMPLETE.json")
        out = sw.resolve_expired_call(lambda p: None, draft["results_root"], c, {"attempt_id": "ta-1"})
        assert out["error"].startswith("OutputExpiredError: no durable RESULT / COMPLETE.json under " + root_b)
        out = sw.resolve_expired_call(read, draft["results_root"], c, None)
        assert "no heartbeat names the attempt id" in out["error"]
        # the classifier reads a resolved expiry as DONE, not FAILED
        snap = {c["call_id"]: {"done": True, "summary": out and {"status": "DONE"}, "error": None, "accepted": "ta-1"}}
        import time as _t
        ev = sw.classify(now=_t.time(), leg="pasttense", calls=[c], snapshot=snap, state=sw.WatchState(), stall_min=20, rows_flat_min=40, progress_min=30,
                         required_keys={c["accepted_key"]: c["standup_id"]})
        assert [e for e in ev if not e.startswith(("WAVE_LAUNCHED ", "WAVE_DONE "))] == ["DONE pasttense scope=base required=1 accepted=1"]

    def test_drive_watch_names_rc0_rc1_rc2_from_the_finished_drive_record(self, drive, tmp_path):
        """rc 2 = released but UNPUBLISHABLE -- nothing reaches the proxy, which would run to its deadline. The grading host
        watcher reads the drive's finished `drive_<id>.json` and names DRIVE_DONE / DRIVE_HELD / DRIVE_UNPUBLISHED (exit 0 / 1 / 2), a failure
        distinct from a timeout; the rule is pinned equal to the drive's REAL `drive_rc` on the same inputs."""
        state = tmp_path / "attempts" / "ta-1"
        state.mkdir(parents=True)
        base = {"schema": "seedrep_petri_drive/1", "drive_id": "d1", "attempt_id": "ta-1", "started_at": "2026-09-18T00:00:00Z"}
        cases = {"released_bound": ({"release": True, "counts": {"complete": 8, "incomplete": 0, "failed": 0}}, {"binding": "bound", "published": ["K/drive_verdict.json", "K/STOP"], "write_stop": True}, 0),
                 "held": ({"release": False, "counts": {"complete": 7, "incomplete": 0, "failed": 1}}, {"binding": "bound", "published": [], "write_stop": False}, 1),
                 "torn_down": ({"release": False, "counts": {"complete": 7, "incomplete": 0, "failed": 1}}, {"binding": "bound", "published": ["K/drive_verdict.json", "K/STOP"], "write_stop": True}, 1),
                 "stale": ({"release": True, "counts": {"complete": 8, "incomplete": 0, "failed": 0}}, {"binding": "stale", "published": [], "write_stop": True}, 2),
                 "missing": ({"release": True, "counts": {"complete": 8, "incomplete": 0, "failed": 0}}, {"binding": "missing", "published": [], "write_stop": True}, 2)}
        printed = []
        import builtins
        real_print = builtins.print
        for name, (verdict, completion, want_rc) in cases.items():
            rec = {**base, "verdict": verdict, "completion": completion, "finished_at": "2026-09-18T01:00:00Z"}
            assert sw.drive_rc_of(rec) == want_rc, name
            if hasattr(drive, "drive_rc"):
                assert drive.drive_rc(verdict, completion) == want_rc, name                    # the drive's REAL rule agrees
            (state / "drive_d1.json").write_text(json.dumps(rec))
            printed.clear()
            builtins.print = lambda *a, **k: printed.append(" ".join(str(x) for x in a))
            try:
                rc = sw.watch_petri_drive(tmp_path / "hb", tmp_path / "DONE", poll_s=1, stall_min=20, max_hours=1, clock=lambda: 0.0, sleep=lambda s: None, drive_state_dir=state)
            finally:
                builtins.print = real_print
            assert rc == want_rc and printed and printed[0].startswith(f"{sw.DRIVE_RC_EVENT[want_rc]} rc={want_rc} drive=d1 attempt=ta-1"), (name, printed)
            if want_rc == 2:
                assert "will run to its deadline" in printed[0] and "--resume-from" in printed[0]
        # an unfinished record (no finished_at) decides nothing; the legacy DONE marker still ends the watch; no state dir = the old behaviour
        (state / "drive_d1.json").write_text(json.dumps({**base, "verdict": None, "completion": None}))
        assert sw.finished_drive_record(state) is None and sw.finished_drive_record(None) is None
        (tmp_path / "DONE").write_text("")
        builtins.print = lambda *a, **k: None
        try:
            assert sw.watch_petri_drive(tmp_path / "hb", tmp_path / "DONE", poll_s=1, stall_min=20, max_hours=1, clock=lambda: 0.0, sleep=lambda s: None, drive_state_dir=state) == 0
        finally:
            builtins.print = real_print

    def test_pool_records_are_read_through_the_volume_api_not_a_local_path(self, draft):
        import inspect
        src = inspect.getsource(sw.watch_pools)
        assert 'Path(p["endpoint_record"]).read_text()' not in src and "read_record(p[\"endpoint_record\"])" in src
        seen, printed = [], []
        pools = {**draft, "pools": [{**draft["pools"][0], "url": None}]}
        rec_path = draft["pools"][0]["endpoint_record"]
        assert rec_path.startswith("/vol/attacker-pool/")

        def read_record(path):
            seen.append(path)
            return {"url": "https://pool0.example"}
        clock = {"t": 0.0}

        def tick():
            clock["t"] += 3600 * 25
            return clock["t"]
        import builtins
        real_print = builtins.print
        builtins.print = lambda *a, **k: printed.append(" ".join(str(x) for x in a))
        try:
            rc = sw.watch_pools(pools, poll_s=120, max_hours=24, clock=tick, sleep=lambda s: None, probe=lambda url: url == "https://pool0.example", read_record=read_record)
        finally:
            builtins.print = real_print
        assert rc == 3 and seen == [rec_path] and any(p.startswith("POOL_UP seedrep-attacker-0 url=https://pool0.example") for p in printed)
        # the Volume path translation: /vol/<rel> -> <rel>
        class V:
            def read_file(self, rel):
                assert rel == "attacker-pool/seedrep-0/endpoint.json", rel
                yield b'{"url": "u"}'
        assert sw.volume_read_text(V(), "/vol/attacker-pool/seedrep-0/endpoint.json") == '{"url": "u"}'


# =========================================================================== staging ships and verifies every view the PLAN references

class TestP1_6_staging_follows_the_plan:
    def _bank(self, tmp_path, inputs, extra):
        """A fake lane bank: s1..s3 from the roster + the extra views given ({view: {cell: staged_id}}), synthetic bytes; the Arditi carrier."""
        bank = tmp_path / "bank" / "qwen35_4b"
        ar = tmp_path / "arditi"
        pins = {"qwen35_4b": {}}
        views = {v: lp.roster_view_cells(inputs["roster"], v) for v in lp.VIEWS}
        views.update(extra)
        for v, cells in views.items():
            view = bank / "cells_view" / v
            view.mkdir(parents=True)
            entries = []
            for cell, sid in cells.items():
                d = bank / "cells" / sid
                d.mkdir(parents=True, exist_ok=True)
                (d / "object.pt").write_bytes(sid.encode())
                (view / cell).symlink_to(f"../../cells/{sid}")
                entries.append({"cell_id": cell, "staged_id": sid, "object_sha256": hashlib.sha256(sid.encode()).hexdigest()})
                pins["qwen35_4b"][sid] = {"object_sha256": hashlib.sha256(sid.encode()).hexdigest()}
            (view / "VIEW_MANIFEST.json").write_text(json.dumps({"target": "qwen35_4b", "view": v, "entries": entries}))
        (ar / "cells" / "ar_c0p3__arditi_allpos").mkdir(parents=True)
        (ar / "cells" / "ar_c0p3__arditi_allpos" / "object.pt").write_bytes(b"arditi")
        (ar / "cells_view" / "ar_c0p3").mkdir(parents=True)
        (ar / "cells_view" / "ar_c0p3" / "arditi_allpos").symlink_to("../../cells/ar_c0p3__arditi_allpos")
        (ar / "cells_view" / "ar_c0p3" / "VIEW_MANIFEST.json").write_text(json.dumps({"target": "qwen35_4b", "view": "ar_c0p3", "entries": [
            {"cell_id": "arditi_allpos", "staged_id": "ar_c0p3__arditi_allpos", "object_sha256": hashlib.sha256(b"arditi").hexdigest()}]}))
        apins = {"qwen35_4b": {"ar_c0p3__arditi_allpos": {"object_sha256": hashlib.sha256(b"arditi").hexdigest()}}}
        return tmp_path / "bank", ar, pins, apins

    def test_revalidated_dose_views_are_staged_and_verified_when_the_plan_references_them(self, srv, inputs, tmp_path):
        ro, pins_ = TestDoseStandups._peer(inputs)
        plan = _plan(inputs, peer_roster=ro, peer_pins=pins_, register_peer_dose_standups=True)
        extra = lp.bank_views_for_plan(plan)
        assert sorted(extra) == sorted(lp.PEER_EXPECTED_VIEWS) and extra["d1p5_loradiff64s1"] == {lp.LORA_DIFF_CELL: "d1p5_eb64s1__" + lp.LORA_DIFF_CELL}
        assert lp.bank_views_for_plan(_plan(inputs)) == {}
        bank, ar, pins, apins = self._bank(tmp_path, inputs, extra)
        rep = srv.verify_staged_objects(str(bank), str(ar), pins, apins, extra_views=extra)
        assert rep["ok"] is True and len(rep["shas"]) == 16 + 6 and "d1p5_eb64s1__" + lp.LORA_DIFF_CELL in rep["shas"]
        # the repro: the fixed 16 verify while the plan's dose views are never looked at
        rep16 = srv.verify_staged_objects(str(bank), str(ar), pins, apins)
        assert rep16["ok"] is True and len(rep16["shas"]) == 16
        # the tarball carries every dose view and object the plan references
        arcs = dict(srv.stage_bank_arcnames(rep["shas"], extra))
        assert "bank:qwen35_4b/cells_view/d1p5_loradiff64s1" in arcs and arcs["bank:qwen35_4b/cells_view/dm0p5_pairdiff64s1"] == "sdfbase/full22/qwen35_4b/cells_view/dm0p5_pairdiff64s1"
        assert f"bank:qwen35_4b/cells/d1p5_eb64s1__{lp.LORA_DIFF_CELL}" in arcs and sum(1 for k in arcs if k.startswith("bank:qwen35_4b/cells_view/")) == 3 + 6
        # known-bads: a referenced dose view absent from the bank; a view serving another object; a pin that disagrees
        bank2, ar2, pins2, apins2 = self._bank(tmp_path / "two", inputs, {k: v for k, v in extra.items() if k != "dm0p5_pairdiff64s1"})
        rep = srv.verify_staged_objects(str(bank2), str(ar2), pins2, apins2, extra_views=extra)
        assert rep["ok"] is False and any(p.get("view") == "dm0p5_pairdiff64s1" and "absent" in p.get("why", "") for p in rep["problems"])
        wrong = dict(extra); wrong["d1p5_loradiff64s1"] = {lp.LORA_DIFF_CELL: "d0p5_eb64s1__" + lp.LORA_DIFF_CELL}
        rep = srv.verify_staged_objects(str(bank), str(ar), pins, apins, extra_views=wrong)
        assert rep["ok"] is False and any(p.get("view") == "d1p5_loradiff64s1" for p in rep["problems"])
        pins3 = copy.deepcopy(pins); pins3["qwen35_4b"]["d1p5_eb64s1__" + lp.LORA_DIFF_CELL] = {"object_sha256": "0" * 64}
        rep = srv.verify_staged_objects(str(bank), str(ar), pins3, apins, extra_views=extra)
        assert rep["ok"] is False and any(p.get("staged_id") == "d1p5_eb64s1__" + lp.LORA_DIFF_CELL for p in rep["problems"])
        # the operator scripts follow: --stage-bank receives the plan, step 0 builds every discovered dose view
        stage = OPERATOR_DIR / "05_stage.sh"
        build = OPERATOR_DIR / "00_build_views.sh"
        if stage.is_file() and build.is_file():
            assert '"--plan", str(plan)' in stage.read_text() and "--stage-bank" in stage.read_text()
            assert "peer_dose_views" in build.read_text() and "for v in $VIEWS" in build.read_text()


# =========================================================================== the images' pip deps are pinned to uv.lock's versions

class TestP2_15_image_pins:
    def test_pyyaml_and_httpx_are_pinned_exactly_to_the_lockfile(self, srv):
        import re as _re
        src = DRIVER.read_text()
        lock = (REPO / "uv.lock").read_text()
        for name, ver in srv.IMAGE_PIP_PINS.items():
            m = _re.search(rf'name = "{name}"\nversion = "([^"]+)"', lock)
            assert m and m.group(1) == ver, (name, ver, m and m.group(1))
        assert set(srv.IMAGE_PIP_PINS) == {"pyyaml", "httpx"}
        # no floating spelling survives in either image definition
        for bad in ('pip_install("pyyaml")', '"pyyaml", "httpx"', 'pip_install("httpx")'):
            assert bad not in src, bad
        assert src.count('_PIN("pyyaml")') == 2 and src.count('_PIN("httpx")') == 1
        assert srv._PIN("pyyaml") == "pyyaml==6.0.3" and srv._PIN("httpx") == "httpx==0.28.1"
        # the known-bad: a pin that drifts from the lockfile is what this test refuses
        assert not _re.search(r'name = "pyyaml"\nversion = "6\.0\.2"', lock)


# =========================================================================== ONE enumerated ten-gate S0 runner, fail-closed

class TestP1_12_s0_gates:
    @pytest.fixture(scope="class")
    def S0(self):
        spec = importlib.util.spec_from_file_location("seedrep_s0_gates_under_test", REPO / "scripts/modal/seedrep_s0_gates.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_the_table_enumerates_plan_s8s_ten_gates_plus_argv_ledger_plan_identity(self, S0):
        ids = [g["id"] for g in S0.GATES]
        assert ids == [f"G{i}" for i in range(1, 11)] + ["ARGV", "LEDGER", "PLAN", "DOSE", "IDENTITY"]
        texts = {g["id"]: g["plan"] for g in S0.GATES}
        for gid, needle in (("G1", "post-#196 reader"), ("G2", "request_evidence_reconciliation"), ("G3", "retry-binding"), ("G4", "512 -> 2,048 -> 8,192 -> 16,384"),
                            ("G5", "two-snapshot election"), ("G6", "(W, wave_uuid, leg, view, stage/cap, shard)"), ("G7", "superseded attempt"),
                            ("G8", "S5 references"), ("G9", "--max-logprobs 64"), ("G10", "adapter reuse"), ("LEDGER", "quota"), ("IDENTITY", "check_plan_set")):
            assert needle in texts[gid], (gid, texts[gid])
        assert S0.selftest() == 0
        assert S0.list_gates()[0]["runs"] == ["tests/test_seedrep_standup_argv.py -k gate1"]

    def test_a_skip_a_failure_zero_collected_and_a_missing_file_each_fail_the_gate_by_name(self, S0, tmp_path):
        tree = tmp_path / "tree"
        for rel in ("tests/test_seedrep_standup_argv.py", "scripts/tinycap_math500_thinking.py"):
            (tree / rel).parent.mkdir(parents=True, exist_ok=True)
            (tree / rel).write_text("# present\n")
        gate = [g for g in S0.GATES if g["id"] == "G1"][0]

        class R:
            def __init__(self, rc, out):
                self.returncode, self.stdout, self.stderr = rc, out, ""
        mk = lambda rc, out: (lambda argv, **k: R(rc, out))  # noqa: E731
        ok = S0.run_pytest_gate(gate, tree=tree, python="py", lock=None, runner=mk(0, "..\n2 passed in 0.5s\n"))
        assert ok["ok"] is True and ok["counts"] == {"passed": 2, "failed": 0, "skipped": 0, "errors": 0} and ok["argv"][:2] == ["py", "-m"]
        skipped = S0.run_pytest_gate(gate, tree=tree, python="py", lock=None, runner=mk(0, "s.\n1 passed, 1 skipped in 0.5s\nSKIPPED [1] predates #196\n"))
        assert skipped["ok"] is False and "SKIPPED" in skipped["why"] and "did not run" in skipped["why"]
        failed = S0.run_pytest_gate(gate, tree=tree, python="py", lock=None, runner=mk(1, "F.\n1 failed, 1 passed in 0.5s\n"))
        assert failed["ok"] is False and failed["why"].startswith("1 failed")
        empty = S0.run_pytest_gate(gate, tree=tree, python="py", lock=None, runner=mk(5, "no tests ran in 0.01s\n"))
        assert empty["ok"] is False and "no evidence collected" in empty["why"]
        (tree / "scripts/tinycap_math500_thinking.py").unlink()
        absent = S0.run_pytest_gate(gate, tree=tree, python="py", lock=None, runner=mk(0, "2 passed in 0.5s\n"))
        assert absent["ok"] is False and "missing dependency" in absent["why"] and "scripts/tinycap_math500_thinking.py" in absent["missing"] and "argv" not in absent
        # the grading host lock wraps the invocation when its directory exists
        (tree / "scripts/tinycap_math500_thinking.py").write_text("# present again\n")
        lock = tmp_path / "locks" / "lock"; lock.parent.mkdir()
        with_lock = S0.run_pytest_gate(gate, tree=tree, python="py", lock=lock, runner=mk(0, "2 passed in 0.5s\n"))
        assert with_lock["ok"] is True and with_lock["argv"][:4] == ["flock", "-w", "5400", str(lock)]

    def test_python_gates_ledger_quota_plan_and_gate8(self, S0, draft, inputs, tmp_path):
        assert S0.gate_ledger_vs_quota(plan=draft, quota=134)["ok"] and S0.gate_ledger_vs_quota(plan=draft, quota=134)["staged_launch"] is False
        over = S0.gate_ledger_vs_quota(plan=draft, quota=100)
        assert over["ok"] is False and "STAGED" in over["why"] and over["unstaged_peak_containers"] == 134
        assert S0.gate_ledger_vs_quota(plan=draft, quota=None)["ok"] is False
        g8 = S0.gate8_s5_roster(plan=draft, s5_references=None)
        assert g8["ok"] and g8["n_windows"] == 10 and g8["s5_references"].startswith("pending")
        smokes = tmp_path / "smokes.json"
        wrap = {"W": draft["W"], "execution_wave_uuid": SMOKE_WAVE, "production_wave_uuid": draft["wave_uuid"], "launch_plan_sha": draft["launch_plan_sha"]}
        smokes.write_text(json.dumps({"S5": {**wrap, "references": [{"L": r["L"], "S": r["S"], "thinking": r["thinking"]} for r in draft["s5_roster"]]}}))
        bare = S0.gate8_s5_roster(plan=draft, s5_references=smokes)                  # a bare (L, S, thinking) triple is NO reference
        assert bare["ok"] is False and "not S5 references" in bare["why"] and "lacks" in bare["why"]
        smokes.write_text(json.dumps({"S5": {"references": [{"L": r["L"], "S": r["S"], "thinking": r["thinking"]} for r in draft["s5_roster"]]}}))
        assert "carries no W" in S0.gate8_s5_roster(plan=draft, s5_references=smokes)["why"]      # a block without its wrapper is nobody's

        def ref(r):
            return lp.s5_reference_record(L=r["L"], S=r["S"], thinking=r["thinking"], tok_s_aggregate=800.0, duration_s=1500.0, store_complete=True,
                                          sampling_coverage=0.9, n_eligible_intervals=150, W=draft["W"], wave_uuid=SMOKE_WAVE, attempt_id="ta-s5")
        smokes.write_text(json.dumps({"S5": {**wrap, "references": [ref(r) for r in draft["s5_roster"]]}}))
        assert S0.gate8_s5_roster(plan=draft, s5_references=smokes)["ok"]
        smokes.write_text(json.dumps({"S5": {**wrap, "references": [ref(r) for r in draft["s5_roster"] if r["L"] != 11520]}}))
        assert S0.gate8_s5_roster(plan=draft, s5_references=smokes)["ok"] is False
        bad = copy.deepcopy(draft); bad["s5_roster"] = bad["s5_roster"][1:]
        assert S0.gate8_s5_roster(plan=bad, s5_references=None)["ok"] is False
        pv = S0.gate_plan_validates(plan=draft, tree=REPO)
        assert pv["ok"] is True or (pv["why"] and pv["n_problems"] > 0), pv   # the GSM8K common entries land later; until then the refusal is NAMED, never hidden
        bad = copy.deepcopy(draft); bad["standups"][0]["route_set"] = ["none"]
        assert S0.gate_plan_validates(plan=bad, tree=REPO)["ok"] is False

    def test_identity_gate_requires_ws4h_producer_and_exact_contents(self, S0, draft, tmp_path):
        rs = tmp_path / "rs"
        (rs / "registry" / "plans").mkdir(parents=True)
        no_mod = S0.gate_identity_contents(plan=draft, tree=tmp_path / "no-tree", rs=rs, campaign="c1")
        assert no_mod["ok"] is False and "registry_plans.py is not in the tree" in no_mod["why"]
        try:
            RP = S0.load_registry_plans(REPO) or S0.load_registry_plans(WORKTREES / "seedrep-ws4h")
        except Exception as e:  # noqa: BLE001
            RP = None
            why = f"{type(e).__name__}: {str(e)[:160]}"
            broken = tmp_path / "broken-tree"
            (broken / "scripts/sdfbase/seedrep").mkdir(parents=True)
            (broken / "scripts/sdfbase/seedrep/registry_plans.py").write_text("raise RuntimeError('boom')\n")
            assert "failed to import" in S0.gate_identity_contents(plan=draft, tree=broken, rs=rs, campaign="c1")["why"]   # the runner names it, never crashes
            pytest.skip(f"WS4-H's registry_plans.py does not import against this tree ({why}); the gate runs for real at W")
        if RP is None:
            pytest.skip("WS4-H's registry_plans.py is neither in the tree nor in the seedrep-ws4h worktree")
        assert S0.gate_identity_contents(plan=draft, tree=REPO, rs=rs, campaign=None, registry_plans=RP)["ok"] is False
        empty = S0.gate_identity_contents(plan=draft, tree=REPO, rs=tmp_path / "none", campaign="c1", registry_plans=RP)
        assert empty["ok"] is False and "no identity files" in empty["why"]
        try:
            inp = RP.load_inputs(campaign="c1", w_sha=draft["W"], policy=REPO / "configs/registry/decode_policy.json", roster=REPO / lp.ROSTER_REL,
                                 pins=REPO / lp.PINS_REL, arms=REPO / lp.ARMS_REL, arditi_pins=REPO / lp.ARDITI_PINS_REL, seeds_dir=REPO / "configs/registry")
            ps = RP.build(inp)
        except Exception as e:  # noqa: BLE001
            pytest.skip(f"WS4-H's producer refuses this tree's inputs ({type(e).__name__}: {str(e)[:120]}) -- the gate runs for real at W")
        RP.write_plan_set(ps, rs / "registry" / "plans")
        good = S0.gate_identity_contents(plan=draft, tree=REPO, rs=rs, campaign="c1", registry_plans=RP)
        assert good["n_children"] == sum(len(r["children"]) for r in draft["standups"] + draft["rungs"])
        if good["ok"]:
            # a corrupted identity file is a content mismatch the producer's own check names
            victim = next((rs / "registry" / "plans").rglob("*.json"))
            victim.write_text(json.dumps({"tampered": True}))
            bad = S0.gate_identity_contents(plan=draft, tree=REPO, rs=rs, campaign="c1", registry_plans=RP)
            assert bad["ok"] is False and ("content mismatch" in bad["why"] or "unloadable" in bad["why"])
        else:
            assert good["why"] and ("missing" in good["why"] or "mismatch" in good["why"]), good   # this tree's plan vs the producer: the gap is NAMED, never hidden

    def test_integration_the_s0_loader_imports_ws4hs_producer_on_this_tree(self, S0, tmp_path):
        """Integration fix: `load_registry_plans` imported `registry_plans.py` by path WITHOUT
        registering it in `sys.modules`; under Python 3.12 the producer's frozen dataclasses resolve their annotations through
        `sys.modules[cls.__module__]` at class creation (`dataclasses._is_type`) and the import died with
        `AttributeError: 'NoneType' object has no attribute '__dict__'` -- at S0 the IDENTITY gate would read "failed to import" and
        refuse the launch. KNOWN-BAD first (the unregistered idiom over the tree's producer reproduces the error), then the loader."""
        import importlib.util
        p = REPO / S0.REGISTRY_PLANS_REL
        assert p.is_file()
        spec = importlib.util.spec_from_file_location("_unregistered_registry_plans_probe", p)
        mod = importlib.util.module_from_spec(spec)
        sys.modules.pop(spec.name, None)
        with pytest.raises(AttributeError, match="__dict__"):
            spec.loader.exec_module(mod)                       # the pre-fix idiom, on this interpreter
        RP = S0.load_registry_plans(REPO)
        assert RP is not None and callable(RP.build) and callable(RP.identity_for) and sys.modules.get("seedrep_registry_plans_s0") is RP
        assert S0.load_registry_plans(tmp_path / "no-tree") is None
        broken = tmp_path / "broken-tree"
        (broken / "scripts/sdfbase/seedrep").mkdir(parents=True)
        (broken / "scripts/sdfbase/seedrep/registry_plans.py").write_text("raise RuntimeError('boom')\n")
        with pytest.raises(RuntimeError, match="boom"):
            S0.load_registry_plans(broken)                    # a broken producer still raises to the gate, which names it

    def test_operator_wiring_names_the_runner(self):
        st, rd = OPERATOR_DIR / "selftest.sh", OPERATOR_DIR / "README.md"
        if not (st.is_file() and rd.is_file()):
            pytest.skip("the operator dir is absent")
        assert 'seedrep_s0_gates.py" --selftest' in st.read_text() and "seedrep_s0_gates.py --tree" in rd.read_text()

# =========================================================================== the roster's BoN climb decode spelling

class TestD53_bon_climb_decode:
    """The roster key moved `bon_climb_temperature: 1.0` -> `bon_climb_decode: as_drawn` and `bon_served.HARNESS_DEFAULT_*` are gone
    (main bebe99a0). The generator reads BOTH spellings and records which one it read; no module imports a removed name."""

    def test_both_spellings_read_and_recorded_in_the_plan(self, inputs, draft):
        ro = inputs["roster"]
        got = lp.roster_bon_climb_decode(ro)
        assert got["key"] in (lp.BON_CLIMB_DECODE_KEY, lp.BON_CLIMB_TEMPERATURE_KEY) and draft["bon_climb_decode"] == got
        old = {k: v for k, v in ro.items() if k != lp.BON_CLIMB_DECODE_KEY}
        old[lp.BON_CLIMB_TEMPERATURE_KEY] = 1.0
        assert lp.roster_bon_climb_decode(old) == {"key": "bon_climb_temperature", "value": 1.0, "spelling": lp.roster_bon_climb_decode(old)["spelling"]}
        new = {k: v for k, v in ro.items() if k != lp.BON_CLIMB_TEMPERATURE_KEY}
        new[lp.BON_CLIMB_DECODE_KEY] = "as_drawn"
        assert lp.roster_bon_climb_decode(new) == {"key": "bon_climb_decode", "value": "as_drawn", "spelling": lp.roster_bon_climb_decode(new)["spelling"]}
        assert "bebe99a0" in lp.roster_bon_climb_decode(new)["spelling"] and "superseded" in lp.roster_bon_climb_decode(old)["spelling"]
        plan_new = _plan(inputs, roster=new)
        assert plan_new["bon_climb_decode"]["key"] == "bon_climb_decode" and plan_new["bon_climb_decode"]["value"] == "as_drawn"
        # the BoN children's decode is the vendor row either way (the climb decode is the driver's own reading of its config)
        b = _row(plan_new, "bon", "s1")["children"][0]["argv"]
        assert b[b.index("--bon-decode-preset") + 1] == "vendor" and b[b.index("--decode-seed-base") + 1] == "1"

    def test_mains_real_roster_bytes_spell_the_new_key(self, inputs):
        import subprocess
        r = subprocess.run(["git", "-C", str(REPO), "show", "bebe99a0:configs/sdfbase_full22_roster.yaml"], capture_output=True, text=True)
        if r.returncode != 0:
            pytest.skip("main bebe99a0 is not reachable from this checkout")
        import yaml
        got = lp.roster_bon_climb_decode(yaml.safe_load(r.stdout))
        assert got["key"] == "bon_climb_decode" and got["value"] == "as_drawn"

    def test_known_bads_neither_and_both(self, inputs):
        ro = inputs["roster"]
        neither = {k: v for k, v in ro.items() if k not in (lp.BON_CLIMB_DECODE_KEY, lp.BON_CLIMB_TEMPERATURE_KEY)}
        with pytest.raises(ValueError, match="neither"):
            lp.roster_bon_climb_decode(neither)
        with pytest.raises(ValueError, match="neither"):
            _plan(inputs, roster=neither)
        both = dict(neither); both[lp.BON_CLIMB_DECODE_KEY] = "as_drawn"; both[lp.BON_CLIMB_TEMPERATURE_KEY] = 1.0
        with pytest.raises(ValueError, match="BOTH"):
            lp.roster_bon_climb_decode(both)

    def test_no_ws2d_module_imports_a_removed_bon_served_name(self):
        import subprocess
        mods = {n: (REPO / "scripts" / "modal" / n).read_text() for n in ("seedrep_launch_plan.py", "seedrep_served.py", "seedrep_watch.py", "seedrep_ladder_helpers.py")}
        for n, text in mods.items():
            assert "bon_served import" not in text and "from sisteer.eval.attacks import bon_served" not in text, n
            for name in lp.BON_SERVED_REMOVED_NAMES:
                uses = [ln for ln in text.splitlines() if name in ln and not ln.lstrip().startswith("#") and "BON_SERVED_REMOVED_NAMES" not in ln]
                assert uses == [], (n, name, uses)                       # named only in comments and the tuple that pins their absence; never imported / used
        assert lp.BON_SERVED_REMOVED_NAMES == ("HARNESS_DEFAULT_SAMPLER", "HARNESS_DEFAULT_TEMPERATURE")
        r = subprocess.run(["git", "-C", str(REPO), "show", "bebe99a0:src/sisteer/eval/attacks/bon_served.py"], capture_output=True, text=True)
        if r.returncode == 0:
            assert not any(name in r.stdout for name in lp.BON_SERVED_REMOVED_NAMES), "main bebe99a0's bon_served still carries a name this test calls removed"
            assert "def bon_policy(" in r.stdout


# =========================================================================== the S5 probe objects are STAGED with everything else the app may execute

class TestSolR3_1_probe_objects_staged:
    """`bank_views_for_plan()` read `dose_standups.views` only (EMPTY) while the window probe's two peer views ride
    `s5_probe_rows`, so `do_stage_bank()` packaged the fixed 16 and the smoke would have failed at setup (`OBJECT_SHA_MISMATCH`
    / a missing bank view) before producing the measurement everything downstream gates on. Now every extra view + object is DERIVED from
    ALL rows the app may execute (executable standups + rungs + the smoke-only probe rows), the objects pinned by the PLAN (the tree's pin
    file never carries a peer object), and verified locally AND through the real `unpack_bank` body on a scratch volume."""

    def _plan_with_probe_rows(self, inputs):
        ro, pins = TestDoseStandups._peer(inputs)
        plan = _plan(inputs, peer_roster=ro, peer_pins=pins)
        assert plan["s5_probe_rows"] and not any(r.get("dose_standup") for r in plan["standups"]) and plan["dose_standups"]["views"] == {}
        return plan

    def test_revalidated_every_probe_row_view_and_object_is_derived_and_in_the_tarball(self, inputs, srv, tmp_path):
        plan = self._plan_with_probe_rows(inputs)
        LORA = lp.LORA_DIFF_CELL
        extra = lp.bank_views_for_plan(plan)
        assert extra == {"d1p5_loradiff64s1": {LORA: f"d1p5_eb64s1__{LORA}"}, "dm0p5_loradiff64s1": {LORA: f"dm0p5_eb64s1__{LORA}"}}
        pins_plan = lp.bank_pins_for_plan(plan)
        assert pins_plan == {f"d1p5_eb64s1__{LORA}": lp.S5_PROBE_CELLS[0]["object_sha256"], f"dm0p5_eb64s1__{LORA}": lp.S5_PROBE_CELLS[1]["object_sha256"]}
        assert pins_plan[f"d1p5_eb64s1__{LORA}"].startswith("86fa8e9c") and pins_plan[f"dm0p5_eb64s1__{LORA}"].startswith("0522a255")
        # the repro: the OLD derivation (dose_standups.views alone) is empty -> the tarball carried neither probe view nor object
        old_extra = {v: dict(b["cells"]) for v, b in ((plan.get("dose_standups") or {}).get("views") or {}).items()}
        assert old_extra == {}
        bank, ar, pins, apins = TestP1_6_staging_follows_the_plan()._bank(tmp_path, inputs, extra)
        rep16 = srv.verify_staged_objects(str(bank), str(ar), pins, apins, extra_views=old_extra)
        old_arcs = dict(srv.stage_bank_arcnames(rep16["shas"], old_extra))
        assert "bank:qwen35_4b/cells_view/d1p5_loradiff64s1" not in old_arcs and f"bank:qwen35_4b/cells/d1p5_eb64s1__{LORA}" not in old_arcs
        # the fix: local verification with the PLAN's pins for the extra objects (the fake bank's bytes are synthetic, so its shas stand in for the
        # probe's here; the real plan pins are asserted above), every probe row's view AND object in the arcnames
        fake_pins = {sid: hashlib.sha256(sid.encode()).hexdigest() for sid in pins_plan}
        tree_pins = {"qwen35_4b": {sid: e for sid, e in pins["qwen35_4b"].items() if sid not in pins_plan}}     # the tree's file: no peer object
        rep = srv.verify_staged_objects(str(bank), str(ar), tree_pins, apins, extra_views=extra, extra_pins=fake_pins)
        assert rep["ok"] is True and len(rep["shas"]) == 16 + 2, rep["problems"]
        arcs = dict(srv.stage_bank_arcnames(rep["shas"], extra))
        for r in plan["s5_probe_rows"]:
            assert f"bank:qwen35_4b/cells_view/{r['view']}" in arcs and arcs[f"bank:qwen35_4b/cells_view/{r['view']}"] == f"sdfbase/full22/qwen35_4b/cells_view/{r['view']}"
            for c, o in r["objects"].items():
                assert f"bank:qwen35_4b/cells/{o['staged_id']}" in arcs and rep["shas"][o["staged_id"]] == fake_pins[o["staged_id"]] and o["object_sha256"] == pins_plan[o["staged_id"]]
        # known-bad: WITHOUT the plan's pins the tree file pins no probe object -> refused by staged id (the plan pins are load-bearing)
        rep_np = srv.verify_staged_objects(str(bank), str(ar), tree_pins, apins, extra_views=extra)
        assert rep_np["ok"] is False and {p.get("staged_id") for p in rep_np["problems"] if p.get("staged_id")} == set(pins_plan)
        # known-bad: the REAL probe shas against a bank holding other bytes refuse by staged id (a drifted object never stages)
        rep_real = srv.verify_staged_objects(str(bank), str(ar), tree_pins, apins, extra_views=extra, extra_pins=pins_plan)
        assert rep_real["ok"] is False and {p.get("staged_id") for p in rep_real["problems"] if p.get("staged_id")} == set(pins_plan)
        # known-bad: a probe view absent from the bank; a probe object under another sha; a tree pin disagreeing with the plan's
        bank2, ar2, pins2, apins2 = TestP1_6_staging_follows_the_plan()._bank(tmp_path / "two", inputs, {"d1p5_loradiff64s1": extra["d1p5_loradiff64s1"]})
        rep2 = srv.verify_staged_objects(str(bank2), str(ar2), tree_pins, apins2, extra_views=extra, extra_pins=fake_pins)
        assert rep2["ok"] is False and any(p.get("view") == "dm0p5_loradiff64s1" and "absent" in p.get("why", "") for p in rep2["problems"])
        wrong_pins = {**fake_pins, f"d1p5_eb64s1__{LORA}": "0" * 64}
        rep3 = srv.verify_staged_objects(str(bank), str(ar), tree_pins, apins, extra_views=extra, extra_pins=wrong_pins)
        assert rep3["ok"] is False and any(p.get("staged_id") == f"d1p5_eb64s1__{LORA}" and p.get("pin") == "0" * 64 for p in rep3["problems"])
        clash = copy.deepcopy(tree_pins); clash["qwen35_4b"][f"d1p5_eb64s1__{LORA}"] = {"object_sha256": "1" * 64}
        rep4 = srv.verify_staged_objects(str(bank), str(ar), clash, apins, extra_views=extra, extra_pins=fake_pins)
        assert rep4["ok"] is False and any("two shas" in p.get("why", "") for p in rep4["problems"])
        # two rows naming one view with different cells refuse; a row object without a sha refuses
        bad = copy.deepcopy(plan)
        bad["s5_probe_rows"][0]["objects"] = {LORA: {"staged_id": f"d0p5_eb64s1__{LORA}", "object_sha256": lp.DOSE_CHILD_SHA[f"d0p5_eb64s1__{LORA}"]}}
        with pytest.raises(ValueError, match="two different cell sets"):
            lp.bank_views_for_plan(bad)
        bad = copy.deepcopy(plan)
        bad["s5_probe_rows"][0]["objects"][LORA]["object_sha256"] = "short"
        with pytest.raises(ValueError, match="no 64-hex object_sha256"):
            lp.bank_pins_for_plan(bad)
        assert lp.bank_views_for_plan(_plan(inputs)) == {} and lp.bank_pins_for_plan(_plan(inputs)) == {}
        # the container verifies ITS row's extra view + objects beside the fixed 16 (source pin)
        import inspect
        src = inspect.getsource(srv._run_standup_row)
        assert 'lp.bank_views_for_plan({"standups": [row]})' in src and "extra_pins=row_pins" in src

    def test_the_real_unpack_bank_body_verifies_the_probe_objects_server_side(self, inputs, srv, tmp_path, monkeypatch):
        """`do_stage_bank`'s tarball (the same arcnames) through the REAL `unpack_bank` body on a scratch 'volume' -> the extra views land
        beside cells/ and re-verify under the plan's pins; STAGED.json names them; a tarball missing a probe object refuses."""
        import tarfile
        plan = self._plan_with_probe_rows(inputs)
        extra, real_pins = lp.bank_views_for_plan(plan), lp.bank_pins_for_plan(plan)
        pins_plan = {sid: hashlib.sha256(sid.encode()).hexdigest() for sid in real_pins}     # the fake bank's bytes stand in for the probe's (asserted real above)
        bank, ar, pins, apins = TestP1_6_staging_follows_the_plan()._bank(tmp_path, inputs, extra)
        tree_pins = {"qwen35_4b": {sid: e for sid, e in pins["qwen35_4b"].items() if sid not in pins_plan}}
        rep = srv.verify_staged_objects(str(bank), str(ar), tree_pins, apins, extra_views=extra, extra_pins=pins_plan)
        assert rep["ok"]
        vol = tmp_path / "vol"
        lane = vol / "seedrep"
        objects = lane / "objects"
        lane.mkdir(parents=True)
        monkeypatch.setattr(srv, "LANE", str(lane)); monkeypatch.setattr(srv, "OBJECTS_VOL", str(objects))
        monkeypatch.setattr(srv, "BANK_VOL", f"{objects}/sdfbase/full22"); monkeypatch.setattr(srv, "ARDITI_BANK_VOL", f"{objects}/arditi_allpos/qwen35_4b/ladder")
        monkeypatch.setattr(srv, "REPOS", str(lane / "repos")); monkeypatch.setattr(srv, "STAGED_JSON", str(objects / "STAGED.json"))

        def make_tar(name, arcs):
            with tarfile.open(lane / name, "w:gz") as tf:
                for src_, arcname in arcs:
                    kind, rel = src_.split(":", 1)
                    tf.add((ar if kind == "arditi" else bank) / rel, arcname=arcname)
        make_tar("bank-good.tar.gz", srv.stage_bank_arcnames(rep["shas"], extra))
        out = srv.unpack_bank("bank-good.tar.gz", W, tree_pins, apins, extra, pins_plan)
        assert out["ok"] is True and len(out["shas"]) == 18 and out["staged_json"] == str(objects / "STAGED.json"), out["problems"]
        staged = json.loads((objects / "STAGED.json").read_text())
        assert staged["extra_views"] == extra and all(staged["shas"][sid] == sha for sid, sha in pins_plan.items())
        for v, cells in extra.items():
            for cell, sid in cells.items():
                link = objects / "sdfbase/full22/qwen35_4b/cells_view" / v / cell
                assert link.is_symlink() and (link / "object.pt").is_file() and srv.sha256_file(link / "object.pt") == pins_plan[sid]
        # known-bad: the tarball without one probe object (the pre-fix packaging) does not verify server-side and writes no STAGED.json
        for p in (objects / "sdfbase/full22/qwen35_4b/cells").iterdir():
            import shutil; shutil.rmtree(p)
        (objects / "STAGED.json").unlink()
        missing = [a for a in srv.stage_bank_arcnames(rep["shas"], extra) if not a[0].endswith(f"cells/dm0p5_eb64s1__{lp.LORA_DIFF_CELL}")]
        make_tar("bank-bad.tar.gz", missing)
        out = srv.unpack_bank("bank-bad.tar.gz", W, tree_pins, apins, extra, pins_plan)
        assert out["ok"] is False and out["staged_json"] is None and any(p.get("staged_id") == f"dm0p5_eb64s1__{lp.LORA_DIFF_CELL}" or p.get("view") == "dm0p5_loradiff64s1" for p in out["problems"])
        # the operator step builds the plan-derived views (not the peer views alone) and passes the plan to --stage-bank
        build = OPERATOR_DIR / "00_build_views.sh"
        if build.is_file():
            assert "bank_views_for_plan" in build.read_text() and "for v in $VIEWS" in build.read_text()


# =========================================================================== two wave identities -- the wave that MEASURED vs the wave certified

class TestSolR3_5_two_wave_identities:
    """RE-VALIDATED at 1e28fe0f: smoke rows run under an isolated smoke wave (`smoke_identity` refuses the production uuid) while the S5 writer
    stamped / validated every record with `plan["wave_uuid"]` (production) -- so a real S5 record (the validator binds `wave_uuid` to the MEASURING
    standup's wave, read off its ATTEMPT.json) could never be written, and a hand-restamped one would bind to a wave nothing measured. Now:
    `execution_wave_uuid` (attempts, stores, S5 reference records) vs `production_wave_uuid` + `launch_plan_sha` (the certification wrapper)."""

    def _ws3g(self):
        mod = _load_peer("seedrep_utilisation_table_under_test", "scripts/sdfbase/seedrep/utilisation_table.py", WORKTREES / "seedrep-ws3g")
        if mod is None:
            pytest.skip("WS3-G's utilisation_table.py is neither in this tree nor in the seedrep-ws3g worktree")
        return mod

    def test_ctx_attempt_record_and_call_record_carry_both_identities(self, srv, draft, tmp_path, monkeypatch):
        ident = lp.smoke_identity(draft, "S5", SMOKE_WAVE)
        smoke_ctx = srv.spawn_ctx(draft, "seedrep-served-2222222", None, smoke=ident)
        prod_ctx = srv.spawn_ctx(draft, "seedrep-served-2222222", None)
        assert smoke_ctx["wave_uuid"] == smoke_ctx["execution_wave_uuid"] == SMOKE_WAVE and smoke_ctx["production_wave_uuid"] == draft["wave_uuid"]
        assert prod_ctx["wave_uuid"] == prod_ctx["execution_wave_uuid"] == prod_ctx["production_wave_uuid"] == draft["wave_uuid"]
        assert smoke_ctx["launch_plan_sha"] == prod_ctx["launch_plan_sha"] == draft["launch_plan_sha"]
        row = _row(draft, "crescendo")
        att = srv.attempt_record(attempt_id="ta-1", call_id="fc-1", task_id="t", W=draft["W"], wave_uuid=SMOKE_WAVE, row=row, prior_attempts=[],
                                 production_wave_uuid=draft["wave_uuid"], launch_plan_sha=draft["launch_plan_sha"])
        assert att["wave_uuid"] == att["execution_wave_uuid"] == SMOKE_WAVE and att["production_wave_uuid"] == draft["wave_uuid"] and att["launch_plan_sha"] == draft["launch_plan_sha"]
        assert att["control_key"] == row["control_key"] and att["schema"] == "seedrep_attempt/1"
        # the call record the retry policy / watcher read
        class FC:
            object_id = "fc-spawned"

        class FN:
            def spawn(self, row_, ctx_):
                FN.seen = (row_["standup_id"], ctx_["execution_wave_uuid"], ctx_["production_wave_uuid"])
                return FC()
        monkeypatch.setitem(srv.FUNCTIONS, "run_view_leg", FN())
        srow = lp.smoke_row(row, ident, max_children=2, max_timeout_s=3600)
        cid = srv.spawn_row(srow, draft, "seedrep-served-2222222", tmp_path, None, smoke=ident)
        assert cid == "fc-spawned" and FN.seen == (srow["standup_id"], SMOKE_WAVE, draft["wave_uuid"])
        rec = json.loads((tmp_path / "calls.jsonl").read_text().splitlines()[-1])
        assert rec["wave_uuid"] == rec["execution_wave_uuid"] == SMOKE_WAVE and rec["production_wave_uuid"] == draft["wave_uuid"]
        assert rec["launch_plan_sha"] == draft["launch_plan_sha"] and rec["smoke"]["smoke_id"] == "S5"

    def test_ws3g_real_validator_binds_the_reference_to_the_measuring_wave_not_the_production_wave(self, draft):
        UT = self._ws3g()
        rec = lp.s5_reference_record(L=8192, S=64, thinking=False, tok_s_aggregate=812.5, duration_s=1500.0, store_complete=True, sampling_coverage=0.93,
                                     n_eligible_intervals=150, W=draft["W"], wave_uuid=SMOKE_WAVE, attempt_id="ta-s5-0001")
        # the validator reads `wave_uuid` off the MEASURING attempt's ATTEMPT.json (`read_attempt_root`,
        # held equal to the control key's) -- the smoke wave
        assert UT.s5_reference_problems(rec, W=draft["W"], wave_uuid=SMOKE_WAVE) == [] == lp.s5_reference_problems(rec, W=draft["W"], wave_uuid=SMOKE_WAVE)
        theirs = UT.s5_reference_problems(rec, W=draft["W"], wave_uuid=draft["wave_uuid"])
        mine = lp.s5_reference_problems(rec, W=draft["W"], wave_uuid=draft["wave_uuid"])
        assert theirs and mine and any("standup's wave" in p for p in theirs) and any("standup's wave" in p for p in mine)
        # the wrapper binds the block to the production target; a copied block (another production wave / W / plan) is not this plan's
        doc = lp.s5_block_bind({}, W=draft["W"], execution_wave_uuid=SMOKE_WAVE, production_wave_uuid=draft["wave_uuid"], launch_plan_sha=draft["launch_plan_sha"])
        lp.s5_smokes_add_reference(doc, rec, W=draft["W"], wave_uuid=SMOKE_WAVE)
        assert lp.s5_block_problems(doc["S5"], plan=draft) == []
        assert any("targets production wave" in p for p in lp.s5_block_problems(doc["S5"], plan={**draft, "wave_uuid": "33333333-4444-4555-8666-777777777777"}))
        assert any("not this plan's" in p for p in lp.s5_block_problems(doc["S5"], plan={**draft, "W": "3" * 40}))
        assert any("launch_plan_sha" in p for p in lp.s5_block_problems(doc["S5"], plan={**draft, "launch_plan_sha": "stale"}))
        assert lp.s5_block_problems(doc["S5"], plan={**draft, "launch_plan_sha": "final-sha"}, draft_launch_plan_sha=draft["launch_plan_sha"]) == []
        assert any("no smoke runs there" in p for p in lp.s5_block_problems({**doc["S5"], "execution_wave_uuid": draft["wave_uuid"]}, plan=draft))
        assert any("carries no launch_plan_sha" in p for p in lp.s5_block_problems({k: v for k, v in doc["S5"].items() if k != "launch_plan_sha"}, plan=draft))
        with pytest.raises(ValueError, match="never re-bound"):
            lp.s5_block_bind(doc, W=draft["W"], execution_wave_uuid="44444444-5555-4666-8777-888888888888", production_wave_uuid=draft["wave_uuid"], launch_plan_sha=draft["launch_plan_sha"])
        with pytest.raises(ValueError, match="no smoke runs under the production wave"):
            lp.s5_block_bind({}, W=draft["W"], execution_wave_uuid=draft["wave_uuid"], production_wave_uuid=draft["wave_uuid"], launch_plan_sha=draft["launch_plan_sha"])
        assert lp.smoke_execution_wave(Path("/nonexistent"), "S5") is None
        assert lp.smoke_wave_path(Path("/rs"), "S5") == Path("/rs/smokes/S5/wave.json")


# =========================================================================== ONE immutable launch certificate, consumed by --go / --legs / --reattempt

def _uuid4(n: int) -> str:
    return f"{n:08x}-1111-4222-8333-{n:012x}"


def _driver_parse(argvs, *, tree=None):
    """Parse each argv (already resolved; the tokens after `python script`) on `served_capability_run.build_parser()` in a FRESH interpreter --
    the driver binds `sampled_controls` at import and refuses a second bind in one process (bind_driver_globals), so the suite never imports it
    in-process. ``tree`` = the checkout whose driver + `src` parse (default this tree; the contract tree at its tip, read-only).
    Returns a list of {ok, ns | error} in order, then the tree's admitted bank views (a dict with `bank_views`) as the LAST element."""
    import subprocess
    code = (
        "import json, sys\n"
        "sys.path.insert(0, sys.argv[1] + '/src'); sys.path.insert(0, sys.argv[1] + '/scripts')\n"
        "import served_capability_run as R\n"
        "from sisteer.capeval.tinycap import contracts as tc\n"
        "import io, contextlib\n"
        "P = R.build_parser(); out = []\n"
        "for argv in json.load(sys.stdin):\n"
        "    buf = io.StringIO()\n"
        "    try:\n"
        "        with contextlib.redirect_stderr(buf):\n"
        "            ns = vars(P.parse_args(argv))\n"
        "        out.append({'ok': True, 'ns': {k: v for k, v in ns.items() if isinstance(v, (str, int, float, bool, list, type(None)))}})\n"
        "    except SystemExit as e:\n"
        "        out.append({'ok': False, 'error': (buf.getvalue().strip()[-600:] or str(e))})\n"
        "views = list(tc.bank_views_admitted()) if hasattr(tc, 'bank_views_admitted') else list(tc.BANK_VIEWS)\n"
        "out.append({'bank_views': views, 'has_admitted': hasattr(tc, 'bank_views_admitted')})\n"
        "json.dump(out, sys.stdout)\n")
    r = subprocess.run([sys.executable, "-c", code, str(tree or REPO)], input=json.dumps([list(a) for a in argvs]), capture_output=True, text=True, timeout=600,
                       cwd=str(tree or REPO))
    assert r.returncode == 0, r.stderr[-2000:]
    return json.loads(r.stdout)


#: The contract tree: the GSM8K argv CONTRACT's driver (`--bank-view s<s>` admitted since 2126e918; this tree's parser predates it, so the
#: children -- which carry the flag by contract -- parse against THAT driver, imported by path, read-only; absent -> the test skips by name)
WS0_TREE = Path(os.environ.get(lp.WS0_TREE_ENV) or WORKTREES / "seedrep-ws0")      # beside this checkout, as WS2E / WS2F are
WS0_BANK_VIEW_FIX = "218d66a7"


def _contract_tree():
    import subprocess
    if not (WS0_TREE / "scripts/served_capability_run.py").is_file():
        pytest.skip(f"WS0's worktree {WS0_TREE} (the driver at {WS0_BANK_VIEW_FIX}, the GSM8K argv contract) is not on this machine")
    if subprocess.run(["git", "-C", str(WS0_TREE), "merge-base", "--is-ancestor", WS0_BANK_VIEW_FIX, "HEAD"]).returncode != 0:
        pytest.skip(f"WS0's worktree predates {WS0_BANK_VIEW_FIX} (--bank-view s<s>)")
    return WS0_TREE


class TestSolR3_2_launch_certificate:
    """`go_preconditions` once checked S0 W/sha/ok, an S5 block, `bars_met`, >= 1 MT row, no LISTED non-zero
    `clamped_calls` (`int(m.get("clamped_calls") or 0)` read a MISSING field as zero), reference validity and roster coverage -- not S1-S4,
    not the W gate, not exact decode-ref resolution, not the complete measurement / utilisation schemas, and nothing bound the bars verdict to W +
    the production wave + `launch_plan_sha`, so an assembled S5 block could authorise fan-out. Now ONE sealed certificate carries every
    component whole and `certificate_problems` re-derives each verdict from its content."""

    @staticmethod
    def _cpu_summary(sha, fails, tests):
        """cpu_suite.merge_shard_records' shape (`sha`, `app`, status / pytest_rc / counts / failures[].node, shards / shard_layout / shard_records)."""
        return {"sha": sha, "app": f"cpu-suite-{sha[:7]}", "status": "FAIL" if fails else "PASS", "pytest_rc": 1 if fails else 0, "elapsed_s": 900.0, "command": [["pytest"]],
                "tree": f"/vol/repos/sisteer-{sha}", "dependency_hashes": {}, "versions": {}, "counts": {"tests": tests, "failures": len(fails), "errors": 0, "skipped": 6},
                "failures": fails, "shards": 2, "shard_layout": {"count": 2}, "shard_records": [{"shard": {"index": i, "count": 2}, "status": "FAIL" if fails else "PASS",
                                                                                                    "pytest_rc": 1 if fails else 0, "counts": {"tests": tests // 2}} for i in range(2)]}

    @staticmethod
    def _smoke_world(plan, draft, rs, waves, rows_by_smoke):
        """A fake smoke WORLD the derivation reads: per smoke, the accepted Dict for its rows and, under each
        accepted attempt's root on the 'volume', the PRODUCER's files at their canonical paths -- COMPLETE.json / RESULT / modal_evidence.json as the
        app writes them, the recorder's `resolved_sampler.json`, the per-store audit verdicts, the standup's `exp/serve/cells.json` -- every one
        bound by sha in `modal_evidence.json[producer_evidence]`. No operator pointer exists."""
        accepted, texts = {}, {}
        planned = {a["staged_id"]: a.get("intervention_hash") for a in plan["adapter_hash_set"]}
        for sid_smoke, row_ids in rows_by_smoke.items():
            wave = waves[sid_smoke]
            ident = lp.smoke_identity(plan, sid_smoke, wave)
            accepted[sid_smoke] = {}
            for i, rid in enumerate(row_ids):
                row = lp.plan_row_by_id(plan, rid)
                aid = f"ta-{sid_smoke}-{i}"
                akey = lp.accepted_key(ident["W7"], wave, row["leg"], row["engine"], row["stage"], row["shard"])
                accepted[sid_smoke][akey] = aid
                root = lp.attempt_root_for(ident["results_root"], leg=row["leg"], view=row["engine"], stage=row["stage"], shard=row["shard"], attempt_id=aid, kind=row.get("kind"))
                files = {"COMPLETE.json": json.dumps({"attempt_id": aid, "control_key": row["control_key"]}),
                         "RESULT": f"2026-09-18T00:00:00Z status=DONE rc=0 attempt={aid} standup={rid} children={len(row['children'])} failed=0 S=45 elapsed_s=900\n",
                         lp.RESOLVED_SAMPLER_REL: json.dumps({"verdict": "AGREES", "why": "client class equals the expected row", "classes": {"client": {"n": 400}},
                                                             "n_client_requests": 400, "n_probe_requests": 8, "n_unclassified_requests": 0, "class_rule": "tag", "n_logs": 1,
                                                             "distinct_resolved": 1, "reference": {"temperature": 0.7}, "reference_source": "--expect", "divergences": []})}
                audit_keys = []          # (plan rel, the FILE the audit filed, plan arm, plan cell) -- F-AO
                for k in row["children"]:
                  # F-AO: the audit files each verdict under the STORE's spelling -- the served cell id (the Arditi row's `arditi_allpos`, whose plan
                  # ARM is the label `ar_c0p3__arditi_allpos`), the misalign runner's HARNESS id -- and stamps its `arm_binding` (fp2); the app binds
                  # the plan child's cell to that file (`bound_rel`) under the PLAN's rel. The world is the writer's shape, not the plan's label.
                  cell = str(k.get("cell") or k["arm"])
                  harness = lp.harness_id(cell) if cell in lp.HARNESS_ID_OF else cell
                  store_arm = harness if str(k["eval_leg"]).startswith("misalign_") else cell
                  spelling = "harness==traintest" if harness == cell else ("harness" if store_arm == harness else "traintest")
                  arm_binding = {"arm": store_arm, "harness_id": harness, "vmodel": f"petri__{harness}", "spelling": spelling, "traintest_id": cell,
                                 "records": {"endpoint.json": "present", "cells_served.json": "present", "routing_table.json": "present",
                                             "traintest_spellings": "endpoint.json cells<->harness_cells"}}
                  for table_leg, plan_rel in lp.audit_verdict_rels(k["eval_leg"], k["arm"], k["s"], stage=row["stage"], shard=row["shard"]):   # per TABLE leg, as the audit writes (F4)
                    rel = lp.audit_verdict_rel(table_leg, store_arm, k["s"], stage=row["stage"], shard=row["shard"])
                    files[rel] = json.dumps({"leg": table_leg, "arm": store_arm, "s": k["s"], "verdict": "AGREES", "problem": None, "problems": [], "checks": {"coverage": "ok"},
                                             "n_draws": 200, "n_draws_bound": 200, "store_dir": f"{root}/exp/sampled/{store_arm}", "store_complete": True, "attempt_id": aid,
                                             "W": plan["W"], "view": row["engine"], "stage": row["stage"], "shard": f"shard{int(row['shard']):02d}",
                                             "engine_log": f"{root}/exp/serve/vllm_server.log", "resolved_sampler_verdict": "AGREES",
                                             "arm_binding": arm_binding, "verdict_file": rel[len(lp.AUDIT_OUT_REL) + 1:],
                                             **_audit_qualification(table_leg)})          # the qualification the certificate requires
                    if not any(a_[0] == plan_rel for a_ in audit_keys):      # the goal-shard children of one arm-run share ONE store and ONE verdict
                        audit_keys.append((plan_rel, rel, k["arm"], cell))
                # the REAL cells.json shape -- the app enumerates the served cells under HARNESS ids (`si_constv2_mix_full__psr_ce_all`, the S1
                # evidence's id form), hashed per the plan's staged id; an earlier double used another id form, which hid the derivation's mis-keyed lookup
                cells = [{"cell_id": lp.harness_id(c), "virtual_model": f"petri__{lp.harness_id(c)}", "serving_identity": "vllm_lora_native",
                          "intervention_hash": planned.get((row["objects"].get(c) or {}).get("staged_id")) or f"{c[:8]}0123456789abcdef"[:16],
                          "si_id": None, "kind": "lora_all"}
                         for c in row["route_set"] if c != "none"]
                files[lp.CELLS_JSON_REL] = json.dumps({"cells": cells, "serving": {"substrate": "modal", "attempt_id": aid, "standup_id": rid}})
                if row.get("pool"):
                    # F-M: a pool-dependent row's byte snapshot of its pool's endpoint record (seedrep_attacker_pool.endpoint_record_fields' shape)
                    files[lp.POOL_ENDPOINT_SNAPSHOT_REL] = json.dumps({
                        "lane": "seedrep", "pool_name": row["pool"].replace("seedrep-attacker-", "seedrep-"), "url": "https://TESTHOST--pool.modal.direct",
                        "model": lp.ATTACKER_MODEL, "snapshot": lp.ATTACKER_SNAPSHOT, "gpu": "H200:2", "tensor_parallel_size": 2, "max_model_len": 65536,
                        "max_num_seqs": 44, "max_num_batched_tokens": 16384, "pytorch_cuda_alloc_conf": None, "vllm_version": "0.25.1+cu129", "repo_sha": plan["W"],
                        "liveness": {"kind": "modal", "app_name": row["pool"], "app_id": "ap-1", "server_id": "sv-1", "container_id": "ta-1"},
                        "job_id": "sv-1", "node": "ta-1", "started_at": "2026-09-18T00:00:00Z"})
                writer = {lp.RESOLVED_SAMPLER_REL: lp.PRODUCER_WRITERS["resolved_sampler"], lp.CELLS_JSON_REL: "scripts/modal/seedrep_served.py",
                          lp.POOL_ENDPOINT_SNAPSHOT_REL: lp.PRODUCER_WRITERS["pool_endpoint"]}
                producer = {rel: {"sha256": lp._sha_text(text), "writer": writer.get(rel, lp.PRODUCER_WRITERS["request_evidence_audit"]), "tree_sha": plan["W"], "rc": 0}
                            for rel, text in files.items() if rel not in ("COMPLETE.json", "RESULT")}
                for plan_rel, rel, plan_arm, cell in audit_keys:     # F-AO: `seedrep_served.bind_audit_verdicts`' record -- keyed by the plan's rel, naming the file
                    producer[plan_rel] = {**producer.pop(rel), "bound_rel": rel, "plan_arm": plan_arm, "plan_cell": cell, "store_arm": json.loads(files[rel])["arm"]}
                # the determinism spot-check record per TABLE leg, which gates S3 (`determinism_check.py`'s shape, its row axes), bound
                # the way the app binds it -- under `modal_evidence.json["determinism"][<table leg>]`, never under producer_evidence
                det_bind = {}
                for table_leg, rel in lp.determinism_rels(row):
                    files[rel] = json.dumps({"leg": table_leg, "standup_id": f"smoke-{sid_smoke}__{rid}", "view": row["engine"], "stage": row["stage"], "shard": row["shard"],
                                             "smoke_id": sid_smoke, "arms": [{"arm": k["arm"], "vmodel": f"petri__{k['arm']}", "items": ["i0", "i1"], "n_requests": 2, "identical": 2,
                                                                              "store_dir": f"{root}/exp/x"} for k in row["children"][:2]],
                                             "n_items": 2, "n_requests": 2 * min(2, len(row["children"])), "n": 2 * min(2, len(row["children"])),
                                             "identical": 2 * min(2, len(row["children"])), "reproduces": True, "misses": [],
                                             "engine_batch_state": {"before": {}, "after": {}, "drained": True, "allow_busy": False}})
                    det_bind[table_leg] = {"sha256": lp._sha_text(files[rel]), "writer": lp.PRODUCER_WRITERS["determinism"], "tree_sha": plan["W"], "rc": 0}
                files["modal_evidence.json"] = json.dumps({"attempt_id": aid, "readiness": {"problems": []}, "object_verification": {"ok": True},
                                                           "children": [{"rc": 0}] * len(row["children"]), "attested_max_num_seqs": 45, lp.PRODUCER_EVIDENCE_KEY: producer,
                                                           lp.DETERMINISM_EVIDENCE_KEY: det_bind})
                for rel, text in files.items():
                    texts[f"{root}/{rel}"] = text
        return accepted, texts

    def _components(self, inputs, draft, tmp_path):
        # W16s5 (2): the S5 block's windows are DERIVED from the four probe rows' accepted attempts (certify re-derives them), so the plans carry the
        # probe rows (the peer roster + pins, shaped by TestDoseStandups._peer); `c["draft"]` is THIS draft, not the module fixture
        ro, pins = TestDoseStandups._peer(inputs)
        draft = _plan(inputs, peer_roster=ro, peer_pins=pins)
        final = _plan(inputs, materialisation="final", s4=lp.S4_DEFAULTS, peer_roster=ro, peer_pins=pins)
        assert lp.compare_draft_final(draft, final) == [] and final["launch_plan_sha"] != draft["launch_plan_sha"]
        rs = tmp_path / "RS"
        waves = {sid: _uuid4(i + 1) for i, sid in enumerate(lp.SMOKE_PASS_IDS + ("S5",))}
        for sid, wu in waves.items():
            wp = lp.smoke_wave_path(rs, sid)
            wp.parent.mkdir(parents=True, exist_ok=True)
            wp.write_text(json.dumps({"smoke_id": sid, "wave_uuid": wu}))
        plan_p = rs / "launch_plan.json"
        plan_p.write_text(json.dumps(final, default=str))
        draft_p = rs / "launch_plan_draft.json"
        draft_p.write_text(json.dumps(draft, default=str))
        gates = {g: {"ok": True} for g in [f"G{i}" for i in range(1, 11)] + ["ARGV", "LEDGER", "PLAN", "DOSE", "IDENTITY"]}
        s0 = {"schema": "seedrep_s0_gates/1", "W": final["W"], "wave_uuid": final["wave_uuid"], "launch_plan_sha": final["launch_plan_sha"], "ok": True,
              "failed": [], "gates": gates, "tree": "x", "utc": "2026-09-18T00:00:00Z"}
        rows_by_leg = {r["leg"]: r["standup_id"] for r in draft["standups"]}
        rows_by_smoke = {"S1": [rows_by_leg["pasttense"]], "S1b": [rows_by_leg["codeattack"]], "S2": [rows_by_leg["controls"]], "S3": [rows_by_leg["bon"]], "S4": [rows_by_leg["crescendo"]]}
        accepted, texts = self._smoke_world(draft, draft, rs, waves, rows_by_smoke)
        # the two gate summaries at cpu_suite's canonical volume paths (delta 3 (3)); the same failing nodes -> introduced == []
        fails = [{"node": "tests/test_env.py::test_a"}, {"node": "tests/test_env.py::test_b"}]
        texts[lp.cpu_suite_summary_path(final["W"])] = json.dumps(self._cpu_summary(final["W"], fails, 16983))
        texts[lp.cpu_suite_summary_path("b" * 40)] = json.dumps(self._cpu_summary("b" * 40, fails, 16989))
        read_text = lp.volume_only(lambda path: texts.get(str(path)))
        getters = {sid: lp.RecordingGetter(lp.smoke_identity(draft, sid, waves[sid])["dicts"]["accepted"], accepted[sid].get) for sid in lp.SMOKE_PASS_IDS}
        smokes = {}
        for sid in lp.SMOKE_PASS_IDS:
            smokes[sid] = lp.derive_smoke_pass(rs=rs, plan=draft, smoke_id=sid, standup_ids=rows_by_smoke[sid], accepted_get=getters[sid], read_text=read_text,
                                               derived_utc="2026-09-18T00:00:00Z")
        reads = {sid: getters[sid].record() for sid in lp.SMOKE_PASS_IDS}
        doc = lp.s5_block_bind({}, W=final["W"], execution_wave_uuid=waves["S5"], production_wave_uuid=final["wave_uuid"], launch_plan_sha=draft["launch_plan_sha"])
        for r in final["s5_roster"]:
            lp.s5_smokes_add_reference(doc, lp.s5_reference_record(L=r["L"], S=r["S"], thinking=r["thinking"], tok_s_aggregate=800.0, duration_s=1500.0, store_complete=True,
                                                                   sampling_coverage=0.9, n_eligible_intervals=150, W=final["W"], wave_uuid=waves["S5"], attempt_id="ta-s5"),
                                       W=final["W"], wave_uuid=waves["S5"])

        # the measurement windows THROUGH the derivation -- the four probe rows' accepted attempts, the REAL producer's bound records on the fake volume
        from test_seedrep_w16s5_window_measurement import s5_probe_world
        rows5, acc5, texts5 = s5_probe_world(draft, rs, tmp_path / "s5world", wave=waves["S5"], mint_wave=False)
        texts.update(texts5)
        accepted["S5"] = acc5
        rows_by_smoke["S5"] = [r["standup_id"] for r in rows5]
        getters["S5"] = lp.RecordingGetter(lp.smoke_identity(draft, "S5", waves["S5"])["dicts"]["accepted"], acc5.get)
        der5, ms5 = lp.derive_s5_window_measurements(rs=rs, plan=draft, standup_ids=rows_by_smoke["S5"], accepted_get=getters["S5"], read_text=read_text,
                                                     derived_utc="2026-09-18T00:00:00Z")
        for m in ms5:
            lp.s5_smokes_add_window(doc, m, wave_uuid=waves["S5"])
        doc["S5"]["window_derivation"] = der5
        reads["S5"] = getters["S5"].record()
        doc["S5"]["bars"] = lp.utilisation_verdict(bars={"running_over_max_num_seqs_min": 0.91, "sm_duty_min": 0.88, "aggregate_tok_s_over_s5_min": 0.97, "h200_hours_over_sized_max": 1.1},
                                                   source="RS/tables/utilisation.csv", source_sha256="a" * 64, W=final["W"], execution_wave_uuid=waves["S5"], attempt_ids=["ta-s5"])
        smokes.update(doc)
        w_gate = lp.derive_w_gate(W=final["W"], baseline_sha="b" * 40, read_text=read_text, derived_utc="2026-09-18T00:00:00Z")
        # the decode proof against origin/main's registry bytes (the three vendor rows landed there) through THIS tree's PolicyFile
        import subprocess
        reg = tmp_path / "decode_policy_main.json"
        r = subprocess.run(["git", "-C", str(REPO), "show", "origin/main:configs/registry/decode_policy.json"], capture_output=True, text=True)
        if r.returncode != 0:
            pytest.skip("origin/main's decode registry is not fetchable here")
        reg.write_text(r.stdout)
        proof = lp.decode_ref_proof(REPO, registry_path=reg)
        return dict(final=final, draft=draft, rs=rs, plan_p=plan_p, draft_p=draft_p, waves=waves, s0=s0, smokes=smokes, w_gate=w_gate, proof=proof, reg=reg, read_text=read_text,
                    accepted=accepted, texts=texts, rows_by_smoke=rows_by_smoke, getters=getters, reads=reads)

    def test_the_certificate_issues_from_validated_components_and_every_component_is_re_derived(self, inputs, draft, tmp_path):
        c = self._components(inputs, draft, tmp_path)
        draft = c["draft"]                                     # W16s5: the probe-bearing draft the components were built against
        final, proof = c["final"], c["proof"]
        # the eight rows resolve by EXACT ref on main's registry, none superseded; on THIS tree two vendor rows are absent (a real known-bad)
        assert set(proof["rows"]) == set(lp.BATTERY_DECODE_REFS) and all(r["resolution"] == "by_ref" and r["superseded_by_policy_id"] is None for r in proof["rows"].values())
        assert lp.decode_proof_problems(proof, registry_sha256=proof["registry_sha256"]) == [] and len(lp.BATTERY_DECODE_REFS) == 8
        here = lp.decode_ref_proof(REPO)
        probs_here = lp.decode_proof_problems(here)
        if here["rows"]["multi_turn"]["policy_id"] is None:
            assert any("multi_turn" in p and "did not resolve" in p for p in probs_here) and any("petri" in p for p in probs_here)
        else:
            assert probs_here == []                                  # the tree already carries the rows (post-W re-merge)
        cert = lp.build_certificate(plan=final, plan_path=c["plan_p"], draft=draft, s0_record=c["s0"], smokes_doc=c["smokes"], w_gate=c["w_gate"], decode_proof=proof,
                                    rs=c["rs"], registry_sha256=proof["registry_sha256"], read_text=c["read_text"], accepted_reads=c["reads"])
        assert tuple(cert) == lp.CERTIFICATE_KEYS and cert["certificate_sha256"] == lp.certificate_sha256(cert)
        assert all(cert["smokes"][sid]["schema"] == lp.SMOKE_PASS_SCHEMA == "seedrep_smoke_pass/2" for sid in lp.SMOKE_PASS_IDS) and cert["w_gate"]["schema"] == "seedrep_w_gate/2"
        assert cert["launch_plan_sha"] == final["launch_plan_sha"] and cert["draft_launch_plan_sha"] == draft["launch_plan_sha"] and cert["draft_final_problems"] == []
        assert cert["production_wave_uuid"] == final["wave_uuid"] and cert["smoke_waves"] == c["waves"] and cert["utilisation"] == c["smokes"]["S5"]["bars"]
        assert lp.go_preconditions(final, certificate=cert, registry_sha256=proof["registry_sha256"]) == []
        assert lp.go_preconditions(final, certificate=json.loads(json.dumps(cert)), registry_sha256=proof["registry_sha256"]) == [], "a JSON round trip keeps the seal"
        # KNOWN-BADS, each refused by name -- a component missing
        for comp, needle in (("s0", "no S0 PASS record"), ("w_gate", "w_gate: not an object"), ("decode_rows", "decode_rows: not an object"),
                             ("s5", "no S5 block"), ("utilisation", "not the S5 block's bars record")):
            bad = {**cert, comp: None}
            bad["certificate_sha256"] = lp.certificate_sha256(bad)
            assert any(needle in p for p in lp.go_preconditions(final, certificate=bad)), (comp, lp.go_preconditions(final, certificate=bad))
        bad = {**cert, "smokes": {k: v for k, v in cert["smokes"].items() if k != "S3"}}
        bad["certificate_sha256"] = lp.certificate_sha256(bad)
        assert any("no S3 PASS record" in p for p in lp.go_preconditions(final, certificate=bad))
        # a COPIED S5 block: another smoke run's execution wave, or another production wave's block
        for s5_over, needle in (({"execution_wave_uuid": _uuid4(99)}, "a copied block"), ({"production_wave_uuid": _uuid4(98)}, "targets production wave")):
            bad = {**cert, "s5": {**cert["s5"], **s5_over}}
            bad["certificate_sha256"] = lp.certificate_sha256(bad)
            assert any(needle in p for p in lp.go_preconditions(final, certificate=bad)), needle
        # a STALE stamp (the plan's content untouched, its stamp not its digest -> edited); a REGENERATED plan (content + stamp changed -> another
        # certificate's plan); an S0 taken against the draft
        assert any("edited after generation" in p for p in lp.go_preconditions({**final, "launch_plan_sha": "0" * 16}, certificate=cert))
        regen = {**final, "s5_roster": list(final["s5_roster"])[:-1]}
        regen["launch_plan_sha"] = lp.plan_sha(regen)
        assert any("regenerated / edited plan needs a new certificate" in p for p in lp.go_preconditions(regen, certificate=cert))
        bad = {**cert, "s0": {**cert["s0"], "launch_plan_sha": draft["launch_plan_sha"]}}
        bad["certificate_sha256"] = lp.certificate_sha256(bad)
        assert any("re-run S0 on THIS plan" in p for p in lp.go_preconditions(final, certificate=bad))
        # TAMPER after issue: any byte changed fails the seal (and the changed content is also refused on its own)
        tampered = json.loads(json.dumps(cert))
        tampered["s5"]["bars"]["bars_met"] = True
        tampered["s5"]["bars"]["bars"]["sm_duty_min"]["value"] = 0.2
        probs = lp.go_preconditions(final, certificate=tampered)
        assert any("seal does not match" in p for p in probs) and any("disagrees with value" in p for p in probs)
        # the measurement window: clamped_calls MISSING is not zero; None is not zero; 1 is not zero
        for cc, needle in ((_ABSENT, "missing clamped_calls"), (None, "REQUIRED as an integer"), (1, "1 clamped call")):
            wins = json.loads(json.dumps(cert["s5"]["multiturn"]))
            if cc is _ABSENT:
                del wins[0]["clamped_calls"]
            else:
                wins[0]["clamped_calls"] = cc
            bad = {**cert, "s5": {**cert["s5"], "multiturn": wins}}
            bad["certificate_sha256"] = lp.certificate_sha256(bad)
            assert any(needle in p for p in lp.go_preconditions(final, certificate=bad)), needle
        bad = {**cert, "s5": {**cert["s5"], "multiturn": [m for m in cert["s5"]["multiturn"] if m["leg"] != "actorattack"]}}
        bad["certificate_sha256"] = lp.certificate_sha256(bad)
        assert any("no window measurement for the S4-decided leg actorattack" in p for p in lp.go_preconditions(final, certificate=bad))
        # the W gate with ONE introduced node; a decode row superseded / missing; the registry bytes differ from the tree's
        bad = {**cert, "w_gate": {**cert["w_gate"], "introduced": ["tests/test_x.py::test_y"]}}
        bad["certificate_sha256"] = lp.certificate_sha256(bad)
        assert any("W INTRODUCES 1 failing node" in p for p in lp.go_preconditions(final, certificate=bad))
        rows = json.loads(json.dumps(proof["rows"]))
        rows["bon"]["superseded_by_policy_id"] = "qwen35_4b_it|bon|off|other@v1"
        bad = {**cert, "decode_rows": {**proof, "rows": rows}}
        bad["certificate_sha256"] = lp.certificate_sha256(bad)
        assert any("decode_rows[bon]" in p and "superseded" in p for p in lp.go_preconditions(final, certificate=bad))
        rows = {k: v for k, v in proof["rows"].items() if k != "petri"}
        bad = {**cert, "decode_rows": {**proof, "rows": rows}}
        bad["certificate_sha256"] = lp.certificate_sha256(bad)
        assert any("no proof for petri" in p for p in lp.go_preconditions(final, certificate=bad))
        assert any("re-prove on W" in p for p in lp.go_preconditions(final, certificate=cert, registry_sha256="e" * 64))
        # the utilisation verdict unmet; a smoke with a failed check; draft != final recorded
        unmet = lp.utilisation_verdict(bars={"running_over_max_num_seqs_min": 0.5, "sm_duty_min": 0.88, "aggregate_tok_s_over_s5_min": 0.97, "h200_hours_over_sized_max": 1.1},
                                       source="RS/tables/utilisation.csv", source_sha256="a" * 64, W=final["W"], execution_wave_uuid=c["waves"]["S5"], attempt_ids=["ta-s5"])
        bad = {**cert, "s5": {**cert["s5"], "bars": unmet}, "utilisation": unmet}
        bad["certificate_sha256"] = lp.certificate_sha256(bad)
        assert any("bars_met is not true" in p for p in lp.go_preconditions(final, certificate=bad))
        s3c = cert["smokes"]["S3"]["checks"]
        bad = {**cert, "smokes": {**cert["smokes"], "S3": {**cert["smokes"]["S3"], "checks": {**s3c, "resolved_sampler": {**s3c["resolved_sampler"], "ok": False}, "petri_body_shows_six_keys": {"ok": True}}}}}
        bad["certificate_sha256"] = lp.certificate_sha256(bad)
        probs = lp.go_preconditions(final, certificate=bad)
        assert any("S3: required check 'resolved_sampler' is missing or not ok" in p for p in probs) and any("S3: checks outside this smoke's set: ['petri_body_shows_six_keys']" in p for p in probs), probs
        bad = {**cert, "draft_final_problems": ["standups/pasttense__s1/pin"]}
        bad["certificate_sha256"] = lp.certificate_sha256(bad)
        assert any("draft != final" in p for p in lp.go_preconditions(final, certificate=bad))
        # build_certificate refuses to ISSUE on any of these (the writer and the consumer share the rule)
        with pytest.raises(ValueError, match="cannot be issued.*no S0 PASS record"):
            lp.build_certificate(plan=final, plan_path=c["plan_p"], draft=draft, s0_record=None, smokes_doc=c["smokes"], w_gate=c["w_gate"], decode_proof=proof, rs=c["rs"], read_text=c["read_text"], accepted_reads=c["reads"])

    def test_fu2_d198_an_unmet_utilisation_verdict_is_admitted_only_through_the_derived_waiver(self, inputs, draft, tmp_path):
        """The S5 verdict UNMET (Running unmeasured, SM 0.76, hours 1.84x) -> the certificate refuses ... unless RS/utilisation_waiver.json
        DERIVED from that verdict is present; the certificate carries it under `utilisation_waiver`, `certificate_problems` re-validates it field by
        field (a tampered reading, another verdict, another tree, another ruling refuse by name); a waiver beside a MET verdict refuses."""
        import stamp_utilisation_waiver as stamp
        c = self._components(inputs, draft, tmp_path)
        final, proof, rs = c["final"], c["proof"], c["rs"]
        # fu3: the real shape -- Running UNMEASURED (the /2 explicit form with the table's reason), SM 0.76 false, hours 1.84 false
        unmet = lp.utilisation_verdict(bars={"running_over_max_num_seqs_min": {"unmeasured": True, "why": "pasttense/s1 ta-s5: driver demand not recorded on the root"},
                                             "sm_duty_min": 0.76, "aggregate_tok_s_over_s5_min": 0.97, "h200_hours_over_sized_max": 1.84},
                                       source="RS/tables/utilisation.csv", source_sha256="a" * 64, W=final["W"], execution_wave_uuid=c["waves"]["S5"], attempt_ids=["ta-s5"])
        assert unmet["schema"] == "seedrep_utilisation_verdict/2" and unmet["bars"]["running_over_max_num_seqs_min"]["unmeasured"] is True and unmet["bars_met"] is False
        smokes = json.loads(json.dumps(c["smokes"])); smokes["S5"]["bars"] = unmet
        kw = dict(plan=final, plan_path=c["plan_p"], draft=c["draft"], s0_record=c["s0"], smokes_doc=smokes, w_gate=c["w_gate"], decode_proof=proof, rs=rs,
                  registry_sha256=proof["registry_sha256"], read_text=c["read_text"], accepted_reads=c["reads"])
        with pytest.raises(ValueError, match="bars_met is not true") as ei:
            lp.build_certificate(**kw)
        assert "admissible only through RS/utilisation_waiver.json" in str(ei.value)
        # the waiver DERIVED from the recorded verdict by the stamp tool over RS
        (rs / "smokes.json").write_text(json.dumps(smokes))
        assert stamp.main(["--rs", str(rs), "--plan", str(c["plan_p"]), "--ruling", "D198"]) == 0
        waiver = lp.read_utilisation_waiver(rs)
        assert waiver["by"] == "D198" and waiver["W"] == final["W"] and [u["bar"] for u in waiver["unmet"]] == ["running_over_max_num_seqs_min", "sm_duty_min", "h200_hours_over_sized_max"]
        assert waiver["unmet"][0] == {"bar": "running_over_max_num_seqs_min", "value": None, "threshold": 0.8, "met": False, "unmeasured": True,
                                      "why": "pasttense/s1 ta-s5: driver demand not recorded on the root"}                 # the unmeasured bar rides with its reason
        cert = lp.build_certificate(**kw)
        assert tuple(cert) == lp.CERTIFICATE_KEYS and cert["utilisation_waiver"] == waiver and cert["utilisation"] == unmet and cert["utilisation"]["bars_met"] is False
        assert lp.go_preconditions(final, certificate=cert, registry_sha256=proof["registry_sha256"]) == []
        assert lp.go_preconditions(final, certificate=json.loads(json.dumps(cert)), registry_sha256=proof["registry_sha256"]) == []
        # known-bads on the carried waiver, each refused by name (the seal re-sealed so the waiver check, not the seal, speaks)
        def bad_with(**over):
            b = {**cert, "utilisation_waiver": {**waiver, **over}}
            b["certificate_sha256"] = lp.certificate_sha256(b)
            return lp.go_preconditions(final, certificate=b, registry_sha256=proof["registry_sha256"])
        tampered = json.loads(json.dumps(waiver["unmet"])); tampered[1]["value"] = 0.86
        assert any("s5 bars waiver: unmet" in p and "is not the verdict's unmet bars" in p for p in bad_with(unmet=tampered))
        assert any("s5 bars waiver: bars_sha256" in p for p in bad_with(bars_sha256="0" * 64))
        assert any("s5 bars waiver: W" in p for p in bad_with(W="d" * 40))
        assert any("s5 bars waiver: by 'D197' is not the ruling D198" in p for p in bad_with(by="D197"))
        assert any("s5 bars waiver: lacks ['stamped_utc']" in p for p in lp.go_preconditions(final, certificate={**cert, "utilisation_waiver": {k: v for k, v in waiver.items() if k != "stamped_utc"},
                                                                                                                       "certificate_sha256": lp.certificate_sha256({**cert, "utilisation_waiver": {k: v for k, v in waiver.items() if k != "stamped_utc"}})}))
        # the verdict re-recorded (another SM reading) under the same waiver: the waiver no longer names it
        other = lp.utilisation_verdict(bars={"running_over_max_num_seqs_min": {"unmeasured": True, "why": "pasttense/s1 ta-s5: driver demand not recorded on the root"},
                                             "sm_duty_min": 0.78, "aggregate_tok_s_over_s5_min": 0.97, "h200_hours_over_sized_max": 1.84},
                                       source="RS/tables/utilisation.csv", source_sha256="a" * 64, W=final["W"], execution_wave_uuid=c["waves"]["S5"], attempt_ids=["ta-s5"])
        b = {**cert, "s5": {**cert["s5"], "bars": other}, "utilisation": other}; b["certificate_sha256"] = lp.certificate_sha256(b)
        assert any("does not name this verdict" in p for p in lp.go_preconditions(final, certificate=b, registry_sha256=proof["registry_sha256"]))
        # the waiver removed from the certificate while the verdict is unmet: refused as before
        b = {**cert, "utilisation_waiver": None}; b["certificate_sha256"] = lp.certificate_sha256(b)
        assert any("bars_met is not true" in p for p in lp.go_preconditions(final, certificate=b, registry_sha256=proof["registry_sha256"]))
        # a waiver present while the verdict is MET: nothing to waive -- refused (build and check)
        met = c["smokes"]["S5"]["bars"]
        b = {**cert, "s5": {**cert["s5"], "bars": met}, "utilisation": met}; b["certificate_sha256"] = lp.certificate_sha256(b)
        assert any("present while the verdict is met" in p for p in lp.go_preconditions(final, certificate=b, registry_sha256=proof["registry_sha256"]))
        with pytest.raises(ValueError, match="present while the verdict is met"):
            lp.build_certificate(**{**kw, "smokes_doc": c["smokes"]})
        (rs / lp.UTILISATION_WAIVER_FILE).unlink()
        assert lp.build_certificate(**{**kw, "smokes_doc": c["smokes"]})["utilisation_waiver"] is None       # the met world, no waiver: as before

    def test_delta_p1_1_self_authored_assertions_are_refused_and_every_record_is_derived(self, inputs, draft, tmp_path):
        """The S1-S4 PASS records and the W gate are DERIVED from producer-owned files -- an invented attempt id,
        a hand-authored record, a producer file that changed after recording, and a W summary changed after recording are each refused by name."""
        c = self._components(inputs, draft, tmp_path)
        draft = c["draft"]                                     # W16s5: the probe-bearing draft the components were built against
        final, rs, read_text = c["final"], c["rs"], c["read_text"]
        sid, rid = "S1", c["rows_by_smoke"]["S1"][0]
        wave = c["waves"][sid]
        good = c["smokes"][sid]
        root = good["rows"][rid]["attempt_root"]
        assert good["accepted"] == {rid: "ta-S1-0"} and good["rows"][rid]["accepted_attempt"] == "ta-S1-0" and set(good["rows"][rid]["sources"]) == set(lp.SMOKE_ROW_FACT_FILES)
        assert good["checks"]["request_evidence_audit"]["sources"][0]["attempt_id"] == "ta-S1-0" and all(len(s["sha256"]) == 64 for ch in good["checks"].values() for s in ch["sources"])
        assert all(s["path"].startswith(root + "/") and s["producer"]["writer"] for ch in good["checks"].values() for s in ch["sources"]), "every source is the producer's, under the root"
        assert lp.smoke_pass_problems(good, smoke_id=sid, plan=final, execution_wave_uuid=wave, draft_launch_plan_sha=draft["launch_plan_sha"], read_text=read_text) == []
        # (a) an INVENTED attempt id: no accepted attempt -> refused; an id with no root -> refused; a record carrying it with real sources -> refused
        with pytest.raises(ValueError, match="no ACCEPTED attempt"):
            lp.derive_smoke_pass(rs=rs, plan=draft, smoke_id=sid, standup_ids=[rid], accepted_get=lambda k: None, read_text=read_text)
        with pytest.raises(ValueError, match="lacks \\['COMPLETE.json'"):
            lp.derive_smoke_pass(rs=rs, plan=draft, smoke_id=sid, standup_ids=[rid], accepted_get=lambda k: "ta-invented", read_text=read_text)
        forged = json.loads(json.dumps(good))
        forged["rows"][rid]["accepted_attempt"] = "ta-invented"
        forged["accepted"][rid] = "ta-invented"
        probs = lp.smoke_pass_problems(forged, smoke_id=sid, plan=final, execution_wave_uuid=wave, draft_launch_plan_sha=draft["launch_plan_sha"])
        assert any("is not the root of accepted attempt 'ta-invented'" in p for p in probs), probs
        probs = lp.smoke_pass_problems(forged, smoke_id=sid, plan=final, execution_wave_uuid=wave, draft_launch_plan_sha=draft["launch_plan_sha"], read_text=read_text)
        assert any("no longer derives from the volume" in p and "lacks" in p for p in probs), probs
        # (b) a hand-authored v1 record (an all-true map on a row with NO accepted attempt): not a derived record
        hand = {"schema": "seedrep_smoke_pass/1", "smoke_id": sid, "verdict": "PASS", "rc": 0, "W": final["W"], "execution_wave_uuid": wave, "production_wave_uuid": final["wave_uuid"],
                "launch_plan_sha": draft["launch_plan_sha"], "standup_ids": [rid], "accepted": {rid: "ta-anything"}, "checks": {"resolved_sampler_agrees": True}, "recorded_utc": "x"}
        probs = lp.smoke_pass_problems(hand, smoke_id=sid, plan=final, execution_wave_uuid=wave, draft_launch_plan_sha=draft["launch_plan_sha"], read_text=read_text)
        assert any("hand-authored" in p for p in probs) and any("lacks" in p for p in probs)
        # the producer's recorder file says DISAGREES (and the producer bound THAT) -> never derives; a verdict of another attempt -> refused
        rs_path = f"{root}/{lp.RESOLVED_SAMPLER_REL}"
        ev_path = f"{root}/modal_evidence.json"

        def rebind(texts_, rel, text):
            """A PRODUCER that wrote `text` at <root>/<rel>: the file and its binding in modal_evidence.json both change (F-AO: an audit verdict's
            record is keyed by the PLAN's rel and names the file under `bound_rel`)."""
            texts_[f"{root}/{rel}"] = text
            ev = json.loads(texts_[ev_path])
            pe = ev[lp.PRODUCER_EVIDENCE_KEY]
            key = rel if rel in pe else next(k_ for k_, v_ in pe.items() if isinstance(v_, dict) and v_.get("bound_rel") == rel)
            pe[key]["sha256"] = lp._sha_text(text)
            texts_[ev_path] = json.dumps(ev)
        bad_texts = dict(c["texts"])
        rebind(bad_texts, lp.RESOLVED_SAMPLER_REL, json.dumps({**json.loads(c["texts"][rs_path]), "verdict": "DIVERGED", "why": "temperature 1.0 != 0.7"}))
        with pytest.raises(ValueError, match="verdict 'DIVERGED' is not AGREES"):
            lp.derive_smoke_pass(rs=rs, plan=draft, smoke_id=sid, standup_ids=[rid], accepted_get=c["accepted"][sid].get, read_text=lp.volume_only(lambda p_: bad_texts.get(str(p_))))
        bad_texts = dict(c["texts"])
        au_plan_rel = lp.audit_verdict_rel(lp.plan_row_by_id(draft, rid)["children"][0]["eval_leg"], lp.plan_row_by_id(draft, rid)["children"][0]["arm"], lp.plan_row_by_id(draft, rid)["children"][0]["s"],
                                           stage=lp.plan_row_by_id(draft, rid)["stage"], shard=lp.plan_row_by_id(draft, rid)["shard"])
        au_rel = json.loads(c["texts"][ev_path])[lp.PRODUCER_EVIDENCE_KEY][au_plan_rel]["bound_rel"]        # F-AO: the FILE the audit filed (the store's spelling)
        rebind(bad_texts, au_rel, json.dumps({**json.loads(c["texts"][f"{root}/{au_rel}"]), "attempt_id": "ta-other"}))
        with pytest.raises(ValueError, match="audits attempt 'ta-other', not the accepted attempt 'ta-S1-0'"):
            lp.derive_smoke_pass(rs=rs, plan=draft, smoke_id=sid, standup_ids=[rid], accepted_get=c["accepted"][sid].get, read_text=lp.volume_only(lambda p_: bad_texts.get(str(p_))))
        # a RESULT that is not DONE rc=0; a producer file ABSENT on the volume: each refuses
        bad_texts = dict(c["texts"])
        bad_texts[f"{root}/RESULT"] = bad_texts[f"{root}/RESULT"].replace("status=DONE rc=0", "status=DRIVERS_FAILED rc=1")
        with pytest.raises(ValueError, match="not `status=DONE rc=0"):
            lp.derive_smoke_pass(rs=rs, plan=draft, smoke_id=sid, standup_ids=[rid], accepted_get=c["accepted"][sid].get, read_text=lp.volume_only(lambda p_: bad_texts.get(str(p_))))
        bad_texts = dict(c["texts"])
        del bad_texts[rs_path]
        with pytest.raises(ValueError, match="the producer's .*resolved_sampler.json is absent on the volume"):
            lp.derive_smoke_pass(rs=rs, plan=draft, smoke_id=sid, standup_ids=[rid], accepted_get=c["accepted"][sid].get, read_text=lp.volume_only(lambda p_: bad_texts.get(str(p_))))
        # a producer file CHANGED after the record was derived -> the certificate consumer refuses it (re-hash + the producer's binding)
        cert = lp.build_certificate(plan=final, plan_path=c["plan_p"], draft=draft, s0_record=c["s0"], smokes_doc=c["smokes"], w_gate=c["w_gate"], decode_proof=c["proof"],
                                    rs=rs, registry_sha256=c["proof"]["registry_sha256"], read_text=read_text, accepted_reads=c["reads"])
        assert lp.go_preconditions(final, certificate=cert, registry_sha256=c["proof"]["registry_sha256"], plan_path=c["plan_p"], read_text=read_text) == []
        orig = c["texts"][rs_path]
        c["texts"][rs_path] = json.dumps({**json.loads(orig), "n_client_requests": 401})
        probs = lp.go_preconditions(final, certificate=cert, registry_sha256=c["proof"]["registry_sha256"], plan_path=c["plan_p"], read_text=read_text)
        assert any("changed since the record was derived" in p and "resolved_sampler" in p for p in probs) and any("not the bytes the producer bound" in p for p in probs), probs
        c["texts"][rs_path] = orig
        # (c) the W gate: derived from cpu_suite's canonical summaries; a summary that CHANGED after recording refuses (sha + recomputed introduced);
        # a node W introduces cannot be certified; a TIMEOUT summary is no verdict; a v1 hand record is refused
        w_path = lp.cpu_suite_summary_path(final["W"])
        assert c["w_gate"]["introduced"] == [] and c["w_gate"]["nodes_W"] == {"total": 16983, "failed": 2} and c["w_gate"]["statuses"] == {"W": "FAIL", "baseline": "FAIL"}
        assert c["w_gate"]["summary_paths"] == {"W": w_path, "baseline": lp.cpu_suite_summary_path("b" * 40)}
        assert lp.w_gate_problems(c["w_gate"], W=final["W"], read_text=read_text) == []
        orig_w = c["texts"][w_path]
        doc = json.loads(orig_w)
        doc["failures"].append({"node": "tests/test_new.py::test_introduced"})
        c["texts"][w_path] = json.dumps(doc)
        probs = lp.w_gate_problems(c["w_gate"], W=final["W"], read_text=read_text)
        assert any("changed since the verdict was derived" in p for p in probs) and any("recomputed from the summaries is ['tests/test_new.py::test_introduced']" in p for p in probs)
        assert any("changed since" in p for p in lp.go_preconditions(final, certificate=cert, registry_sha256=c["proof"]["registry_sha256"], plan_path=c["plan_p"], read_text=read_text))
        bad_gate = lp.derive_w_gate(W=final["W"], baseline_sha="b" * 40, read_text=read_text)
        assert bad_gate["introduced"] == ["tests/test_new.py::test_introduced"]
        with pytest.raises(ValueError, match="cannot be issued.*W INTRODUCES 1 failing node"):
            lp.build_certificate(plan=final, plan_path=c["plan_p"], draft=draft, s0_record=c["s0"], smokes_doc=c["smokes"], w_gate=bad_gate, decode_proof=c["proof"], rs=rs,
                                 registry_sha256=c["proof"]["registry_sha256"], read_text=read_text, accepted_reads=c["reads"])
        c["texts"][w_path] = orig_w
        hand_gate = {**c["w_gate"], "schema": "seedrep_w_gate/1", "introduced": []}
        assert any("hand-authored" in p for p in lp.w_gate_problems(hand_gate, W=final["W"]))
        c["texts"][w_path] = json.dumps({**json.loads(orig_w), "status": "TIMEOUT"})
        with pytest.raises(ValueError, match="no verdict"):
            lp.derive_w_gate(W=final["W"], baseline_sha="b" * 40, read_text=read_text)
        c["texts"][w_path] = orig_w

    def test_delta_p1_2_the_launched_plan_bytes_are_bound_to_the_certificate(self, inputs, draft, tmp_path):
        """An edited child argv under the OLD launch_plan_sha / W / wave / standup ids is refused at --go on BOTH the recomputed
        content digest and the plan file's sha256."""
        import inspect
        c = self._components(inputs, draft, tmp_path)
        draft = c["draft"]                                     # W16s5: the probe-bearing draft the components were built against
        final, read_text = c["final"], c["read_text"]
        cert = lp.build_certificate(plan=final, plan_path=c["plan_p"], draft=draft, s0_record=c["s0"], smokes_doc=c["smokes"], w_gate=c["w_gate"], decode_proof=c["proof"],
                                    rs=c["rs"], registry_sha256=c["proof"]["registry_sha256"], read_text=read_text, accepted_reads=c["reads"])
        assert lp.go_preconditions(final, certificate=cert, registry_sha256=c["proof"]["registry_sha256"], plan_path=c["plan_p"], read_text=read_text) == []
        edited = json.loads(c["plan_p"].read_text())
        kid = [r for r in edited["standups"] if r["leg"] == "pasttense"][0]["children"][0]
        kid["argv"] = list(kid["argv"]) + ["--edited-after-certification"]   # a child argv edited; W / wave / standup ids / the stamp untouched
        assert edited["launch_plan_sha"] == final["launch_plan_sha"] and edited["W"] == final["W"] and edited["wave_uuid"] == final["wave_uuid"]
        edited_p = c["rs"] / "launch_plan.edited.json"
        edited_p.write_text(json.dumps(edited, default=str))
        probs = lp.go_preconditions(edited, certificate=cert, registry_sha256=c["proof"]["registry_sha256"], plan_path=edited_p, read_text=read_text)
        assert any("edited after generation" in p for p in probs) and any("not this plan's content digest" in p for p in probs) and any("launched bytes are not the certified bytes" in p for p in probs)
        # the same bytes re-serialised differently (whitespace) fail the FILE sha even when the digest holds -- the certified bytes are the launched bytes
        repacked = c["rs"] / "launch_plan.repacked.json"
        repacked.write_text(json.dumps(final, default=str, indent=1))
        probs = lp.go_preconditions(final, certificate=cert, registry_sha256=c["proof"]["registry_sha256"], plan_path=repacked, read_text=read_text)
        assert any("launched bytes are not the certified bytes" in p for p in probs) and not any("edited after generation" in p for p in probs)
        # the app's require_certificate passes the plan file and RS (source pin)
        srv_src = (REPO / "scripts/modal/seedrep_served.py").read_text()
        assert "plan_path=Path(plan_path)" in srv_src and "read_text=certificate_read_text(Path(rs))" in srv_src and "def require_certificate(plan_d" in srv_src
        crt = srv_src.split("def certificate_read_text(rs", 1)[1].split("\ndef ", 1)[0]
        assert "modal.Volume.from_name(lp.MODAL_VOLUME_NAME" in crt and 'startswith(lp.VOL_PATH + "/")' in crt and "volume_read_text" in crt

    def test_delta2_p1_a_disagreeing_sources_are_refused_and_the_accepted_attempt_is_read_live(self, srv, inputs, draft, tmp_path):
        """A `seedrep_smoke_pass/2` whose sources say `verdict: DISAGREES` under `ok: true` is refused on its RECORDED facts and,
        with the volume, on the RE-RUN of every check; the accepted attempt of every row is held to the LIVE accepted Dict at --go (a snapshot
        naming a successful-but-superseded attempt is refused); the app's getter reads the Dict and RECORDS the read."""
        c = self._components(inputs, draft, tmp_path)
        draft = c["draft"]                                     # W16s5: the probe-bearing draft the components were built against
        final, rs, read_text = c["final"], c["rs"], c["read_text"]
        sid, rid = "S1", c["rows_by_smoke"]["S1"][0]
        wave, good = c["waves"][sid], c["smokes"][sid]
        kw = dict(smoke_id=sid, plan=final, execution_wave_uuid=wave, draft_launch_plan_sha=draft["launch_plan_sha"])
        assert lp.smoke_pass_problems(good, read_text=read_text, accepted_get=c["accepted"][sid].get, **kw) == []
        bad = json.loads(json.dumps(good))
        for name in ("resolved_sampler", "request_evidence_audit"):
            for s_ in bad["checks"][name]["sources"]:
                s_["verdict"] = "DISAGREES"
        n_src = sum(len(bad["checks"][n_]["sources"]) for n_ in ("resolved_sampler", "request_evidence_audit"))     # the recorder's record + one audit verdict per child
        probs = lp.smoke_pass_problems(bad, **kw)
        assert sum("records verdict 'DISAGREES', not AGREES" in p for p in probs) == n_src >= 2 and all(any(f"check {n_}:" in p for p in probs) for n_ in ("resolved_sampler", "request_evidence_audit")), probs
        probs = lp.smoke_pass_problems(bad, read_text=read_text, **kw)
        assert sum("recomputed facts" in p and "differ from the recorded" in p for p in probs) == n_src, probs
        # the producer file flipped to DISAGREES with the record's sha updated (the producer's binding NOT) -> refused on the binding, never the flag
        root = good["rows"][rid]["attempt_root"]
        rs_path = f"{root}/{lp.RESOLVED_SAMPLER_REL}"
        flipped = json.dumps({**json.loads(c["texts"][rs_path]), "verdict": "DISAGREES"})
        rec2 = json.loads(json.dumps(good))
        rec2["checks"]["resolved_sampler"]["sources"][0]["sha256"] = lp._sha_text(flipped)
        probs = lp.smoke_pass_problems(rec2, read_text=lp.volume_only(lambda p_: flipped if str(p_) == rs_path else read_text(p_)), **kw)
        assert any("no longer passes on its content" in p and "not the bytes the producer bound" in p for p in probs), probs
        au = json.loads(json.dumps(c["smokes"]["S1b"]))
        au["checks"]["request_evidence_audit"]["sources"][0]["attempt_id"] = "ta-other"
        au["checks"]["adapter_hashes_banked"]["sources"][0]["n_banked"] = 0
        probs = lp.smoke_pass_problems(au, smoke_id="S1b", plan=final, execution_wave_uuid=c["waves"]["S1b"], draft_launch_plan_sha=draft["launch_plan_sha"])
        assert any("audits attempt 'ta-other'" in p for p in probs) and any("banked" in p and "not every cell" in p for p in probs), probs
        s4 = json.loads(json.dumps(c["smokes"]["S4"]))
        s4["checks"]["decisions"]["sources"][0]["multiturn_window"]["crescendo"] = 0
        assert any("records no window for ['crescendo']" in p for p in lp.smoke_pass_problems(s4, smoke_id="S4", plan=final, execution_wave_uuid=c["waves"]["S4"], draft_launch_plan_sha=draft["launch_plan_sha"]))
        # (b) the accepted attempt LIVE: the Dict names the elected attempt; the record names a superseded one -> refused everywhere
        live = {k: "ta-elected" for k in c["accepted"][sid]}
        probs = lp.smoke_pass_problems(good, accepted_get=live.get, **kw)
        assert any("names attempt 'ta-elected' NOW, the record 'ta-S1-0' -- a superseded / invented attempt" in p for p in probs), probs
        get_for = lambda s_: c["accepted"][s_].get  # noqa: E731
        cert = lp.build_certificate(plan=final, plan_path=c["plan_p"], draft=draft, s0_record=c["s0"], smokes_doc=c["smokes"], w_gate=c["w_gate"], decode_proof=c["proof"],
                                    rs=rs, registry_sha256=c["proof"]["registry_sha256"], read_text=read_text, accepted_get_for=get_for, accepted_reads=c["reads"])
        assert lp.go_preconditions(final, certificate=cert, registry_sha256=c["proof"]["registry_sha256"], plan_path=c["plan_p"], read_text=read_text, accepted_get_for=get_for) == []
        probs = lp.go_preconditions(final, certificate=cert, registry_sha256=c["proof"]["registry_sha256"], plan_path=c["plan_p"], read_text=read_text,
                                    accepted_get_for=lambda s_: (live.get if s_ == "S1" else c["accepted"][s_].get))
        assert any("S1: row" in p and "superseded" in p for p in probs) and not any("S2:" in p for p in probs), probs
        with pytest.raises(ValueError, match="cannot be issued.*superseded"):
            lp.build_certificate(plan=final, plan_path=c["plan_p"], draft=draft, s0_record=c["s0"], smokes_doc=c["smokes"], w_gate=c["w_gate"], decode_proof=c["proof"],
                                 rs=rs, registry_sha256=c["proof"]["registry_sha256"], read_text=read_text, accepted_get_for=lambda s_: (live.get if s_ == "S1" else c["accepted"][s_].get),
                                 accepted_reads=c["reads"])
        # (c) delta 3 (2): the certificate RECORDS what the issuer read (dict, utc, entries, digest); a read that disagrees with the record, a wrong Dict, a
        # bad digest, a missing block -> refused
        rd = cert["accepted_dicts"][sid]
        assert rd["dict"] == lp.smoke_identity(final, sid, wave)["dicts"]["accepted"] and rd["entries"] == c["accepted"][sid] and rd["digest"] == lp.accepted_reads_digest(rd["entries"]) and rd["read_utc"]
        for mut, needle in (({"entries": {k: "ta-elected" for k in rd["entries"]}, "digest": lp.accepted_reads_digest({k: "ta-elected" for k in rd["entries"]})}, "the issuer read 'ta-elected'"),
                            ({"dict": "seedrep-other-accepted"}, "names Dict 'seedrep-other-accepted'"), ({"digest": "0" * 64}, "is not the digest of its entries")):
            bad_c = {**cert, "accepted_dicts": {**cert["accepted_dicts"], sid: {**rd, **mut}}}
            bad_c["certificate_sha256"] = lp.certificate_sha256(bad_c)
            assert any(needle in p for p in lp.go_preconditions(final, certificate=bad_c, registry_sha256=c["proof"]["registry_sha256"])), needle
        bad_c = {**cert, "accepted_dicts": {}}
        bad_c["certificate_sha256"] = lp.certificate_sha256(bad_c)
        assert any("records no accepted-Dict read for this smoke" in p for p in lp.go_preconditions(final, certificate=bad_c, registry_sha256=c["proof"]["registry_sha256"]))
        # (d) the app's getter is the smoke wave's accepted Dict itself (modal.Dict.from_name(<identity dict>), here the stub), recording every read
        name = lp.smoke_identity(final, sid, wave)["dicts"]["accepted"]
        d = srv.modal.Dict.from_name(name, create_if_missing=False)
        akey = next(iter(c["accepted"][sid]))
        d.put(akey, c["accepted"][sid][akey])
        getter = srv.live_smoke_accepted_getters(rs, final)
        assert getter(sid)(akey) == "ta-S1-0" and getter(sid)("no-such-key") is None and getter(sid).record()["entries"] == {akey: "ta-S1-0", "no-such-key": None}
        d.put(akey, "ta-elected")
        assert getter(sid)(akey) == "ta-elected", "read live on every call, never cached from a snapshot"
        assert getter(sid).name == name and getter(sid).record()["dict"] == name
        rc_src = __import__("inspect").getsource(srv.require_certificate)
        assert "accepted_get_for=(accepted_get_for or live_smoke_accepted_getters(Path(rs), plan_d))" in rc_src

    def test_delta3_1_smoke_evidence_comes_only_from_the_producers_files_under_the_accepted_root(self, inputs, draft, tmp_path):
        """An AGREES file at an operator path is never consulted (the derivation reads only paths DERIVED from the accepted
        root); a schema-valid-looking verdict under the WRONG attempt root is refused (not this attempt's, not bound by this producer); a record
        pointing at an operator path is refused; `{"verdict": "AGREES"}` alone is not the recorder's record; the producer's binding is required."""
        c = self._components(inputs, draft, tmp_path)
        draft = c["draft"]                                     # W16s5: the probe-bearing draft the components were built against
        final, rs, read_text = c["final"], c["rs"], c["read_text"]
        sid, rid = "S1", c["rows_by_smoke"]["S1"][0]
        wave, good = c["waves"][sid], c["smokes"][sid]
        root = good["rows"][rid]["attempt_root"]
        kw = dict(smoke_id=sid, plan=final, execution_wave_uuid=wave, draft_launch_plan_sha=draft["launch_plan_sha"])
        # (a) an operator-authored AGREES file anywhere (local RS, another /vol path): the derivation never reads it -- every read is a derived root path
        op = rs / "operator_agrees.json"
        op.write_text(json.dumps({**json.loads(c["texts"][f"{root}/{lp.RESOLVED_SAMPLER_REL}"]), "verdict": "AGREES"}))
        reads = []

        def spy(path):
            reads.append(str(path))
            return c["texts"].get(str(path)) if str(path).startswith("/vol/") else Path(path).read_text()
        rec = lp.derive_smoke_pass(rs=rs, plan=draft, smoke_id=sid, standup_ids=[rid], accepted_get=c["accepted"][sid].get, read_text=spy)
        assert all(p_.startswith(root + "/") for p_ in reads) and str(op) not in reads and lp.smoke_pass_problems(rec, read_text=read_text, **kw) == []
        # a record whose source POINTS at the operator file (or at any path outside the root) is refused, without and with the volume
        forged = json.loads(json.dumps(good))
        forged["checks"]["resolved_sampler"]["sources"][0]["path"] = str(op)
        forged["checks"]["resolved_sampler"]["sources"][0]["sha256"] = lp._sha_text(op.read_text())
        probs = lp.smoke_pass_problems(forged, **kw)
        assert any("is not under any accepted attempt root of this smoke -- not producer evidence" in p for p in probs), probs
        probs = lp.smoke_pass_problems(forged, read_text=lambda p_: Path(p_).read_text() if Path(p_).is_file() else read_text(p_), **kw)
        assert any("not producer evidence" in p for p in probs), probs
        # `default_read_text` (the old operator reader) reads NOTHING; `volume_only` refuses a local path
        assert lp.default_read_text(rs)(str(op)) is None and lp.volume_only(lambda p_: "x")(str(op)) is None and lp.volume_only(lambda p_: "x")("/vol/a") == "x"
        # (b) a schema-valid-looking verdict under the WRONG attempt root: derived for its own root it is fine; borrowed for THIS row it fails the root /
        # the producer binding / the attempt id
        other_root = root.replace("ta-S1-0", "ta-other")
        au_rel = [k for k in c["texts"] if k.startswith(root + f"/{lp.AUDIT_OUT_REL}/")][0][len(root) + 1:]
        # F-AO: the producer's record of that file is keyed by the PLAN's rel (`bound_rel` names the file)
        au_key = next(k_ for k_, v_ in json.loads(c["texts"][f"{root}/modal_evidence.json"])[lp.PRODUCER_EVIDENCE_KEY].items()
                      if k_ == au_rel or (isinstance(v_, dict) and v_.get("bound_rel") == au_rel))
        borrowed = json.loads(json.dumps(good))
        borrowed["checks"]["request_evidence_audit"]["sources"][0]["path"] = f"{other_root}/{au_rel}"
        probs = lp.smoke_pass_problems(borrowed, **kw)
        assert any("is not under any accepted attempt root of this smoke" in p for p in probs), probs
        texts2 = dict(c["texts"])
        texts2[f"{root}/{au_rel}"] = json.dumps({**json.loads(c["texts"][f"{root}/{au_rel}"]), "attempt_id": "ta-other", "store_dir": f"{other_root}/exp/sampled/x"})
        ev = json.loads(texts2[f"{root}/modal_evidence.json"])
        ev[lp.PRODUCER_EVIDENCE_KEY][au_key]["sha256"] = lp._sha_text(texts2[f"{root}/{au_rel}"])
        texts2[f"{root}/modal_evidence.json"] = json.dumps(ev)
        with pytest.raises(ValueError, match="audits attempt 'ta-other', not the accepted attempt"):
            lp.derive_smoke_pass(rs=rs, plan=draft, smoke_id=sid, standup_ids=[rid], accepted_get=c["accepted"][sid].get, read_text=lp.volume_only(lambda p_: texts2.get(str(p_))))
        texts2 = dict(c["texts"])
        texts2[f"{root}/{au_rel}"] = c["texts"][f"{root}/{au_rel}"] + " "                                     # the same record re-serialised: the producer did not write THESE bytes
        with pytest.raises(ValueError, match="is not the bytes the producer bound"):
            lp.derive_smoke_pass(rs=rs, plan=draft, smoke_id=sid, standup_ids=[rid], accepted_get=c["accepted"][sid].get, read_text=lp.volume_only(lambda p_: texts2.get(str(p_))))
        texts2 = dict(c["texts"])
        ev = json.loads(texts2[f"{root}/modal_evidence.json"])
        ev[lp.PRODUCER_EVIDENCE_KEY][au_key] = {"absent": True, "why": "request_evidence_audit.py is not on the tree", "writer": "x", "tree_sha": final["W"], "rc": None,
                                                "bound_rel": au_rel}      # F-AO: the file is there; the producer disowns it
        texts2[f"{root}/modal_evidence.json"] = json.dumps(ev)
        with pytest.raises(ValueError, match="the producer bound no .* in this attempt"):
            lp.derive_smoke_pass(rs=rs, plan=draft, smoke_id=sid, standup_ids=[rid], accepted_get=c["accepted"][sid].get, read_text=lp.volume_only(lambda p_: texts2.get(str(p_))))
        ev.pop(lp.PRODUCER_EVIDENCE_KEY)
        texts2[f"{root}/modal_evidence.json"] = json.dumps(ev)
        with pytest.raises(ValueError, match="bound no producer evidence"):
            lp.derive_smoke_pass(rs=rs, plan=draft, smoke_id=sid, standup_ids=[rid], accepted_get=c["accepted"][sid].get, read_text=lp.volume_only(lambda p_: texts2.get(str(p_))))
        # (c) a verdict field alone is not the producer's record: the recorder's schema is required; the audit's too; the row's cells must be hashed
        texts2 = dict(c["texts"])
        texts2[f"{root}/{lp.RESOLVED_SAMPLER_REL}"] = json.dumps({"verdict": "AGREES"})
        ev = json.loads(texts2[f"{root}/modal_evidence.json"])
        ev[lp.PRODUCER_EVIDENCE_KEY][lp.RESOLVED_SAMPLER_REL]["sha256"] = lp._sha_text(texts2[f"{root}/{lp.RESOLVED_SAMPLER_REL}"])
        texts2[f"{root}/modal_evidence.json"] = json.dumps(ev)
        with pytest.raises(ValueError, match="is not record_resolved_sampler.py's record \\(lacks"):
            lp.derive_smoke_pass(rs=rs, plan=draft, smoke_id=sid, standup_ids=[rid], accepted_get=c["accepted"][sid].get, read_text=lp.volume_only(lambda p_: texts2.get(str(p_))))
        texts2 = dict(c["texts"])
        texts2[f"{root}/{au_rel}"] = json.dumps({"verdict": "AGREES", "attempt_id": "ta-S1-0"})
        ev = json.loads(texts2[f"{root}/modal_evidence.json"])
        ev[lp.PRODUCER_EVIDENCE_KEY][au_key]["sha256"] = lp._sha_text(texts2[f"{root}/{au_rel}"])
        texts2[f"{root}/modal_evidence.json"] = json.dumps(ev)
        with pytest.raises(ValueError, match="is not request_evidence_audit.py's store verdict \\(lacks"):
            lp.derive_smoke_pass(rs=rs, plan=draft, smoke_id=sid, standup_ids=[rid], accepted_get=c["accepted"][sid].get, read_text=lp.volume_only(lambda p_: texts2.get(str(p_))))
        # the canonical paths come from the row's contract (verdict_filename), never from the operator
        row = lp.plan_row_by_id(draft, rid)
        paths = lp.smoke_evidence_paths(row, root)
        assert paths["resolved_sampler"] == [f"{root}/resolved_sampler.json"] and paths["adapter_hashes_banked"] == [f"{root}/exp/serve/cells.json"] and paths["decisions"] == [f"{root}/modal_evidence.json"]
        assert paths["request_evidence_audit"] == [f"{root}/audits/request_log/{k['eval_leg']}/{k['arm']}__s{k['s']}.json" for k in row["children"]]
        assert lp.audit_verdict_rel("gsm8k", "none", 2, stage="16384", shard=0) == "audits/request_log/gsm8k/none__s2__cap16384.json"
        assert lp.audit_verdict_rel("bon", "none", 1, stage="base", shard=3) == "audits/request_log/bon/none__s1__shard03.json"

    def test_delta2_p1_b_a_summary_of_another_tree_is_refused_by_its_own_sha(self, inputs, draft, tmp_path, monkeypatch):
        """The W gate reads cpu_suite's CANONICAL outputs `/vol/cpu-suite/<full sha>/summary.json` (paths DERIVED
        from W / baseline_sha, never supplied), requires the summary's own `sha` == that sha, `app` == `cpu-suite-<sha7>` (REQUIRED) and the shard
        records per cpu_suite's schema; a summary of another tree at the canonical path, a self-labelled file elsewhere, a wrong app, no app, no
        shards are refused by name."""
        c = self._components(inputs, draft, tmp_path)
        draft = c["draft"]                                     # W16s5: the probe-bearing draft the components were built against
        W_, read_text, rs, texts = c["final"]["W"], c["read_text"], c["rs"], c["texts"]
        cs = (REPO / "scripts/modal/cpu_suite.py").read_text()
        assert 'return {"sha": sha, "app": run_name,' in cs and "def merge_shard_records" in cs and 'RESULTS_ROOT = f"{VOL_PATH}/cpu-suite"' in cs, "the writer's shape + root this reader holds the summaries to"
        w_path, b_path = lp.cpu_suite_summary_path(W_), lp.cpu_suite_summary_path("b" * 40)
        assert w_path == f"/vol/cpu-suite/{W_}/summary.json" and c["w_gate"]["summary_paths"] == {"W": w_path, "baseline": b_path}
        w_doc, b_doc = json.loads(texts[w_path]), json.loads(texts[b_path])
        assert w_doc["sha"] == W_ and w_doc["app"] == f"cpu-suite-{W_[:7]}" and b_doc["sha"] == "b" * 40 and w_doc["shards"] == len(w_doc["shard_records"]) == 2
        # the aaaa / cccc pair: another tree's summaries placed at W's / the baseline's canonical paths -> refused on their own sha, both sides and swapped
        orig_w, orig_b = texts[w_path], texts[b_path]
        texts[w_path] = json.dumps({**w_doc, "sha": "a" * 40, "app": "cpu-suite-aaaaaaa"})
        with pytest.raises(ValueError, match=f"W summary: summary sha 'a{{40}}' is not {W_} -- a summary of another tree"):
            lp.derive_w_gate(W=W_, baseline_sha="b" * 40, read_text=read_text)
        texts[w_path] = orig_w
        texts[b_path] = json.dumps({**b_doc, "sha": "c" * 40, "app": "cpu-suite-ccccccc"})
        with pytest.raises(ValueError, match="baseline summary: summary sha 'c{40}' is not b{40}"):
            lp.derive_w_gate(W=W_, baseline_sha="b" * 40, read_text=read_text)
        texts[b_path] = orig_b
        texts[w_path], texts[b_path] = orig_b, orig_w                                                # swapped: the labels are not the identity
        with pytest.raises(ValueError, match="summary sha"):
            lp.derive_w_gate(W=W_, baseline_sha="b" * 40, read_text=read_text)
        texts[w_path], texts[b_path] = orig_w, orig_b
        # the sha right but the app another gate's / absent; the shard records missing -> refused (cpu_suite's schema, not a self-label)
        for mut, needle in (({"app": "cpu-suite-zzzzzzz"}, f"summary app 'cpu-suite-zzzzzzz' is not cpu-suite-{W_[:7]}"), ({"app": None}, "summary app None is not"),
                            ({"shard_records": []}, "shard records do not match cpu_suite's schema"), ({"shards": 1}, "shard records do not match cpu_suite's schema")):
            texts[w_path] = json.dumps({**w_doc, **mut})
            with pytest.raises(ValueError, match=needle):
                lp.derive_w_gate(W=W_, baseline_sha="b" * 40, read_text=read_text)
        texts[w_path] = json.dumps({k: v for k, v in w_doc.items() if k != "app"})
        with pytest.raises(ValueError, match="not cpu_suite's merged summary \\(lacks \\['app'\\]"):
            lp.derive_w_gate(W=W_, baseline_sha="b" * 40, read_text=read_text)
        texts[w_path] = orig_w
        # a self-labelled summary at an OPERATOR path is never read: the derivation reads the canonical paths only (and no local path at all)
        reads = []
        good = lp.derive_w_gate(W=W_, baseline_sha="b" * 40, read_text=lambda p_: reads.append(str(p_)) or read_text(p_))
        assert reads == [w_path, b_path] and {k: v for k, v in good.items() if k != "derived_utc"} == {k: v for k, v in c["w_gate"].items() if k != "derived_utc"}
        with pytest.raises(ValueError, match="a full 40-hex sha"):
            lp.cpu_suite_summary_path("/tmp/anything.json")
        rec = json.loads(json.dumps(c["w_gate"]))
        rec["summary_paths"]["W"] = str(tmp_path / "W_summary.json")
        assert any("must be the canonical cpu_suite outputs" in p and "no operator path names a summary" in p for p in lp.w_gate_problems(rec, W=W_))
        # a recorded verdict whose W summary is REPLACED on the volume by another tree's (same nodes): refused on the bytes' sha AND the summary's own sha
        texts[w_path] = json.dumps({**w_doc, "sha": "a" * 40, "app": "cpu-suite-aaaaaaa"})
        probs = lp.w_gate_problems(c["w_gate"], W=W_, read_text=read_text)
        assert any("changed since the verdict was derived" in p for p in probs) and any("summary sha 'a" in p for p in probs), probs
        texts[w_path] = orig_w
        assert lp.w_gate_problems(c["w_gate"], W=W_, read_text=read_text) == []
        # the CLI: `w-gate --baseline-sha` reads the canonical paths off the pull; another tree's pair there -> refused (2), nothing written
        spec = importlib.util.spec_from_file_location("seedrep_s5_record_p1b", REPO / "scripts/modal/seedrep_s5_record.py")
        S5R = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(S5R)
        monkeypatch.setitem(sys.modules, "modal", _stub_modal())
        FakeVOL.files.clear()
        FakeVOL.files[w_path[len("/vol/"):]] = json.dumps({**w_doc, "sha": "a" * 40, "app": "cpu-suite-aaaaaaa"})
        FakeVOL.files[b_path[len("/vol/"):]] = json.dumps({**b_doc, "sha": "c" * 40, "app": "cpu-suite-ccccccc"})
        (rs / "w_gate.json").unlink(missing_ok=True)
        assert S5R.main(["w-gate", "--rs", str(rs), "--plan", str(c["plan_p"]), "--baseline-sha", "b" * 40]) == 2
        assert not (rs / "w_gate.json").exists()
        FakeVOL.files[w_path[len("/vol/"):]] = orig_w
        FakeVOL.files[b_path[len("/vol/"):]] = orig_b
        assert S5R.main(["w-gate", "--rs", str(rs), "--plan", str(c["plan_p"]), "--baseline-sha", "b" * 40]) == 0
        for gone in (["--w-summary", str(tmp_path / "x.json"), "--baseline-summary", str(tmp_path / "y.json")], ["--pulled-root", str(tmp_path)]):
            with pytest.raises(SystemExit):
                S5R.main(["w-gate", "--rs", str(rs), "--plan", str(c["plan_p"]), "--baseline-sha", "b" * 40] + gone)
        assert "--w-summary" not in (REPO / "scripts/modal/seedrep_s5_record.py").read_text() and "--pulled-root" not in (REPO / "scripts/modal/seedrep_s5_record.py").read_text()

    def test_delta4_2_the_producer_binding_is_the_exact_writer_the_wave_tree_and_rc_0(self, inputs, draft, tmp_path):
        """A binding by `operator.py`, at tree `000…`, or with rc 99 is refused -- on the recorded facts (`smoke_pass_problems`) and
        at derivation (`derive_smoke_pass`), for each of the three, by name."""
        c = self._components(inputs, draft, tmp_path)
        draft = c["draft"]                                     # W16s5: the probe-bearing draft the components were built against
        final, rs, read_text = c["final"], c["rs"], c["read_text"]
        sid, rid = "S1", c["rows_by_smoke"]["S1"][0]
        wave, good = c["waves"][sid], c["smokes"][sid]
        root = good["rows"][rid]["attempt_root"]
        kw = dict(smoke_id=sid, plan=final, execution_wave_uuid=wave, draft_launch_plan_sha=draft["launch_plan_sha"])
        assert good["checks"]["resolved_sampler"]["sources"][0]["producer"] == {"writer": lp.PRODUCER_WRITERS["resolved_sampler"], "tree_sha": final["W"], "rc": 0}
        for mut, needle in (({"writer": "operator.py"}, "written by 'operator.py', not the producer 'scripts/sdfbase/record_resolved_sampler.py'"),
                            ({"tree_sha": "0" * 40}, f"produced at tree '000000000000', not the wave's W {final['W'][:12]}"),
                            ({"rc": 99}, "the producer's tool exited rc 99, not 0")):
            bad = json.loads(json.dumps(good))
            bad["checks"]["resolved_sampler"]["sources"][0]["producer"].update(mut)
            probs = lp.smoke_pass_problems(bad, **kw)
            assert any(needle in p for p in probs), (mut, probs)
            # at derivation: the producer_evidence entry itself carries the wrong writer / tree / rc -> refused, no record
            texts2 = dict(c["texts"])
            ev = json.loads(texts2[f"{root}/modal_evidence.json"])
            ev[lp.PRODUCER_EVIDENCE_KEY][lp.RESOLVED_SAMPLER_REL].update(mut)
            texts2[f"{root}/modal_evidence.json"] = json.dumps(ev)
            with pytest.raises(ValueError, match=needle.replace("(", "\\(").replace(")", "\\)")):
                lp.derive_smoke_pass(rs=rs, plan=draft, smoke_id=sid, standup_ids=[rid], accepted_get=c["accepted"][sid].get, read_text=lp.volume_only(lambda p_: texts2.get(str(p_))))
        # the audit verdicts are held to the audit writer; the cells file to the app's; a root fact binds as the app's at W with rc 0
        assert good["checks"]["request_evidence_audit"]["sources"][0]["producer"]["writer"] == lp.PRODUCER_WRITERS["request_evidence_audit"]
        s1b = c["smokes"]["S1b"]
        assert s1b["checks"]["adapter_hashes_banked"]["sources"][0]["producer"]["writer"] == "scripts/modal/seedrep_served.py"
        s4 = c["smokes"]["S4"]
        assert s4["checks"]["decisions"]["sources"][0]["producer"] == {"writer": "scripts/modal/seedrep_served.py", "tree_sha": final["W"], "rc": 0, "root_fact": True}
        bad = json.loads(json.dumps(s1b))
        bad["checks"]["request_evidence_audit"]["sources"][0]["producer"]["writer"] = "scripts/modal/seedrep_served.py"
        assert any("not the producer 'scripts/sdfbase/seedrep/request_evidence_audit.py'" in p for p in lp.smoke_pass_problems(bad, smoke_id="S1b", plan=final, execution_wave_uuid=c["waves"]["S1b"], draft_launch_plan_sha=draft["launch_plan_sha"]))

    def test_the_cli_records_passes_once_and_certify_writes_a_sealed_write_once_file(self, inputs, draft, tmp_path, monkeypatch):
        c = self._components(inputs, draft, tmp_path)
        draft = c["draft"]                                     # W16s5: the probe-bearing draft the components were built against
        final, rs = c["final"], c["rs"]
        spec = importlib.util.spec_from_file_location("seedrep_s5_record_cert", REPO / "scripts/modal/seedrep_s5_record.py")
        S5R = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(S5R)
        # the LIVE accepted Dicts: the CLI reads modal.Dict by the smoke identity -- here the stub, populated as the wave left it
        stub = _stub_modal()
        monkeypatch.setitem(sys.modules, "modal", stub)
        dict_names = {}
        for sid in lp.SMOKE_PASS_IDS + ("S5",):
            dict_names[sid] = lp.smoke_identity(c["draft"], sid, c["waves"][sid])["dicts"]["accepted"]
            d = stub.Dict.from_name(dict_names[sid], create_if_missing=False)
            for k, v in c["accepted"][sid].items():
                d.put(k, v)
        # S1-S4 PASS records DERIVED from the LIVE Dict + the producer's files under each accepted root, read through the VOLUME API (the stub Volume
        # serves exactly the wave's files; no pull, no local path): write-once; a smoke with no minted wave never records; S5 is
        # not a `pass`; no --record / --pointer / --accepted-json / --pulled-root exists
        (rs / "s0_gates.json").write_text(json.dumps(c["s0"]))
        (rs / "w_gate.json").unlink(missing_ok=True)
        FakeVOL.files.clear()
        for path, text in c["texts"].items():
            FakeVOL.files[path[len("/vol/"):]] = text
        # a PLANTED local tree under the expected relative names (the cwd, RS, /tmp): AGREES everywhere the volume says AGREES too -- if a reader ever
        # consulted it, the DISAGREES planted copy of S3's recorder file would make no difference either way, so plant the OPPOSITE of the volume
        planted = tmp_path / "planted"
        for path, text in c["texts"].items():
            q = planted / path[len("/vol/"):]
            q.parent.mkdir(parents=True, exist_ok=True)
            q.write_text(text)
        s3_root = c["smokes"]["S3"]["rows"][c["rows_by_smoke"]["S3"][0]]["attempt_root"]
        s3_rs = f"{s3_root}/{lp.RESOLVED_SAMPLER_REL}"
        (planted / s3_rs[len("/vol/"):]).write_text(json.dumps({**json.loads(c["texts"][s3_rs]), "verdict": "DISAGREES"}))
        monkeypatch.chdir(planted)
        for sid in lp.SMOKE_PASS_IDS:
            args = ["pass", "--rs", str(rs), "--plan", str(c["plan_p"]), "--draft", str(c["draft_p"]), "--smoke-id", sid, "--rows", ",".join(c["rows_by_smoke"][sid])]
            assert S5R.main(args) == 0, sid                                   # the volume's AGREES, not the planted DISAGREES
            assert S5R.main(args) == 2, "write-once"
        for gone in (["--record", str(tmp_path / "x.json")], ["--pointer", f"resolved_sampler={tmp_path / 'x.json'}"], ["--accepted-json", str(tmp_path / "x.json")],
                     ["--pulled-root", str(planted)]):
            with pytest.raises(SystemExit):
                S5R.main(["pass", "--rs", str(rs), "--plan", str(c["plan_p"]), "--smoke-id", "S3", "--rows", "x"] + gone)
        # the volume's file GONE while the planted AGREES copy sits under the expected name -> refused: the local tree is never consulted
        (rs / "smokes.json").write_text(json.dumps({k: v for k, v in json.loads((rs / "smokes.json").read_text()).items() if k != "S3"}))
        (planted / s3_rs[len("/vol/"):]).write_text(c["texts"][s3_rs])
        saved = FakeVOL.files.pop(s3_rs[len("/vol/"):])
        args3 = ["pass", "--rs", str(rs), "--plan", str(c["plan_p"]), "--draft", str(c["draft_p"]), "--smoke-id", "S3", "--rows", ",".join(c["rows_by_smoke"]["S3"])]
        assert S5R.main(args3) == 2
        FakeVOL.files[s3_rs[len("/vol/"):]] = saved
        assert S5R.main(args3) == 0
        rec_derived = json.loads((rs / "smokes.json").read_text())["S3"]
        assert rec_derived["schema"] == lp.SMOKE_PASS_SCHEMA and rec_derived["accepted"] == {c["rows_by_smoke"]["S3"][0]: "ta-S3-0"}
        # the LIVE Dict naming an attempt with no root on the volume -> refused; the elected one again -> recorded; no minted wave -> refused
        (rs / "smokes.json").write_text(json.dumps({k: v for k, v in json.loads((rs / "smokes.json").read_text()).items() if k != "S3"}))
        d3 = stub.Dict.from_name(dict_names["S3"], create_if_missing=False)
        akey3 = next(iter(c["accepted"]["S3"]))
        d3.put(akey3, "ta-invented")
        args = ["pass", "--rs", str(rs), "--plan", str(c["plan_p"]), "--draft", str(c["draft_p"]), "--smoke-id", "S3", "--rows", ",".join(c["rows_by_smoke"]["S3"])]
        assert S5R.main(args) == 2
        d3.put(akey3, c["accepted"]["S3"][akey3])
        assert S5R.main(args) == 0
        lp.smoke_wave_path(rs, "S3").unlink()
        (rs / "smokes.json").write_text(json.dumps({k: v for k, v in json.loads((rs / "smokes.json").read_text()).items() if k != "S3"}))
        assert S5R.main(args) == 2, "no minted wave: the smoke never ran"
        lp.smoke_wave_path(rs, "S3").write_text(json.dumps({"smoke_id": "S3", "wave_uuid": c["waves"]["S3"]}))
        assert S5R.main(args) == 0
        with pytest.raises(SystemExit):
            S5R.main(["pass", "--rs", str(rs), "--plan", str(c["plan_p"]), "--smoke-id", "S5", "--rows", "x"])
        # the S5 block through the writer's own commands, bound to the S5 execution wave
        rec_p = tmp_path / "ref.json"
        for ref in c["smokes"]["S5"]["references"]:
            rec_p.write_text(json.dumps(ref))
            assert S5R.main(["reference", "--rs", str(rs), "--plan", str(c["draft_p"]), "--record", str(rec_p)]) == 0
        # W16s5 (B2 / (2)): `window --record` is refused by the CLI; the windows are DERIVED from the four probe rows' accepted attempts (the LIVE
        # Dict + the producer's bound files on the volume), and certify below re-derives them
        rec_p.write_text(json.dumps(c["smokes"]["S5"]["multiturn"][0]))
        assert S5R.main(["window", "--rs", str(rs), "--plan", str(c["draft_p"]), "--record", str(rec_p)]) == 2
        assert S5R.main(["window", "--rs", str(rs), "--plan", str(c["draft_p"]), "--rows", ",".join(c["rows_by_smoke"]["S5"])]) == 0
        assert {(m["leg"], m["arm"]) for m in json.loads((rs / "smokes.json").read_text())["S5"]["multiturn"]} == {(m["leg"], m["arm"]) for m in c["smokes"]["S5"]["multiturn"]}
        rec_p.write_text(json.dumps(c["smokes"]["S5"]["bars"]))
        assert S5R.main(["bars", "--rs", str(rs), "--plan", str(c["draft_p"]), "--record", str(rec_p)]) == 0
        # the W gate DERIVED from cpu_suite's canonical summaries on the VOLUME (write-once; a W that introduces a node refuses); then the decode proof
        w_rel = lp.cpu_suite_summary_path(final["W"])[len("/vol/"):]
        gate_args = ["w-gate", "--rs", str(rs), "--plan", str(c["plan_p"]), "--baseline-sha", "b" * 40]
        with pytest.raises(SystemExit):
            S5R.main(gate_args + ["--pulled-root", str(planted)])
        assert S5R.main(gate_args) == 0
        assert S5R.main(gate_args) == 2
        (rs / "w_gate.json").unlink()
        orig_w = FakeVOL.files[w_rel]
        intro = json.loads(orig_w)
        intro["failures"].append({"node": "tests/test_x.py::t"})
        FakeVOL.files[w_rel] = json.dumps(intro)
        assert S5R.main(gate_args) == 2
        FakeVOL.files[w_rel] = orig_w
        assert S5R.main(gate_args) == 0
        assert json.loads((rs / "w_gate.json").read_text())["introduced"] == [] and json.loads((rs / "w_gate.json").read_text())["summary_paths"]["W"] == lp.cpu_suite_summary_path(final["W"])
        tree = tmp_path / "tree"
        (tree / "configs/registry").mkdir(parents=True)
        (tree / "configs/registry/decode_policy.json").write_text(c["reg"].read_text())
        (tree / "src").symlink_to(REPO / "src")
        assert S5R.main(["decode-proof", "--rs", str(rs), "--tree", str(tree)]) == 0
        proof = json.loads((rs / "decode_ref_proof.json").read_text())
        assert proof["registry_sha256"] == lp.sha256_file(tree / "configs/registry/decode_policy.json") and lp.decode_proof_problems(proof) == []
        # certify: re-derives every record from the LIVE Dicts + the pull, records the reads; the sealed file, write-once, read-only; a second issue refuses;
        # a plan whose sha moved refuses; NO smoke / W-gate / snapshot document is ever ingested
        certify = ["certify", "--rs", str(rs), "--plan", str(c["plan_p"]), "--draft", str(c["draft_p"]), "--tree", str(tree)]
        saved_files = dict(FakeVOL.files)
        FakeVOL.files.clear()
        assert S5R.main(certify) == 2, "the producer's files are unreadable on the volume (the planted local tree does not count) -> refused, never issued"
        assert not (rs / lp.CERTIFICATE_FILE).exists()
        FakeVOL.files.update(saved_files)
        for gone in (["--smokes", str(rs / "smokes.json")], ["--w-gate", str(rs / "w_gate.json")], ["--accepted-json", f"S1={tmp_path / 'x.json'}"], ["--pulled-root", str(planted)]):
            with pytest.raises(SystemExit):
                S5R.main(certify + gone)
        smokes_text = (rs / "smokes.json").read_text()
        doc = json.loads(smokes_text)
        doc["S3"]["checks"]["resolved_sampler"]["sources"][0]["verdict"] = "DISAGREES"
        (rs / "smokes.json").write_text(json.dumps(doc))
        assert S5R.main(certify) == 2 and not (rs / lp.CERTIFICATE_FILE).exists()
        doc = json.loads(smokes_text)
        doc["S3"]["accepted"][c["rows_by_smoke"]["S3"][0]] = "ta-superseded"
        doc["S3"]["rows"][c["rows_by_smoke"]["S3"][0]]["accepted_attempt"] = "ta-superseded"
        (rs / "smokes.json").write_text(json.dumps(doc))
        assert S5R.main(certify) == 2 and not (rs / lp.CERTIFICATE_FILE).exists()
        (rs / "smokes.json").write_text(smokes_text)
        # the LIVE Dict moved to another attempt after `pass` recorded (a superseded record) -> certify refuses; back -> issues
        d3.put(akey3, "ta-elected")
        assert S5R.main(certify) == 2 and not (rs / lp.CERTIFICATE_FILE).exists()
        d3.put(akey3, c["accepted"]["S3"][akey3])
        assert S5R.main(certify) == 0
        cert_p = rs / lp.CERTIFICATE_FILE
        cert = json.loads(cert_p.read_text())
        assert (cert_p.stat().st_mode & 0o777) == 0o444 and set(cert["accepted_dicts"]) == set(lp.SMOKE_PASS_IDS) | {"S5"}   # W16s5 (2): the S5 windows are re-derived off the LIVE Dict too
        assert cert["accepted_dicts"]["S3"]["dict"] == dict_names["S3"] and cert["accepted_dicts"]["S3"]["entries"] == c["accepted"]["S3"] and cert["accepted_dicts"]["S3"]["read_utc"]
        assert lp.go_preconditions(final, certificate=cert, registry_sha256=proof["registry_sha256"]) == []
        assert lp.go_preconditions(final, certificate=cert, registry_sha256=proof["registry_sha256"], plan_path=c["plan_p"], read_text=c["read_text"],
                                   accepted_get_for=lambda sid_: c["accepted"][sid_].get) == []
        assert S5R.main(certify) == 2
        moved = json.loads(c["plan_p"].read_text())
        moved["launch_plan_sha"] = "0" * 16
        (tmp_path / "moved.json").write_text(json.dumps(moved, default=str))
        assert S5R.main(["certify", "--rs", str(rs), "--plan", str(tmp_path / "moved.json"), "--draft", str(c["draft_p"]), "--tree", str(tree), "--out", str(tmp_path / "c2.json")]) == 2
        assert not (tmp_path / "c2.json").exists()

    def test_the_eight_decode_refs_are_ws4hs_leg_policy_refs(self):
        rp = WORKTREES / "seedrep-ws4h/scripts/sdfbase/seedrep/registry_plans.py"
        for root in (REPO, WORKTREES / "seedrep-ws4h"):
            if (root / "scripts/sdfbase/seedrep/registry_plans.py").is_file():
                rp = root / "scripts/sdfbase/seedrep/registry_plans.py"
                break
        if not rp.is_file():
            pytest.skip("WS4-H's registry_plans.py is absent")
        refs = set(re.findall(r'f"\{MODEL_ID\}\|([a-z0-9_]+)\|(on|off)\|([a-z_A-Z0-9]+)@v1"', rp.read_text()))
        theirs = {f"qwen35_4b_it|{a}|{b}|{c}@v1" for a, b, c in refs}
        assert theirs == set(lp.BATTERY_DECODE_REFS.values()), (theirs ^ set(lp.BATTERY_DECODE_REFS.values()))


# =========================================================================== retries plan-bound, race-safe, crash-recoverable

class TestSolR3_4_retries_plan_bound_race_safe_crash_recoverable:
    """RE-VALIDATED at 1e28fe0f: call records carried W + wave but no `launch_plan_sha` and `wave_calls()` filtered on the wave alone (a
    regenerated same-wave plan would retry an old failure onto new rows); `--reattempt` bypassed `go_preconditions()`; the ledger was
    read / modify / write with the consumed count persisted BEFORE `.spawn()` and the successor id added after -- two concurrent passes
    both spawned, a crash between the write and the spawn consumed the one re-attempt without launching it. Now: an atomic INTENT claim
    (`put(skip_if_exists)`) before the spawn, reconciled with the call id after; a stale unspawned intent is recovered by exactly one later
    pass; every call record and ledger entry is stamped with `launch_plan_sha`; a duplicate execution of one intent exits at entry."""

    def _setup(self, draft):
        rows = {r["standup_id"]: r for r in draft["standups"]}
        sid = "codeattack__s1__base__shard00"
        FakeDict.store.clear()
        ledger = FakeDict("reattempts")
        sha = draft["launch_plan_sha"]
        calls = [{"call_id": "fc-1", "standup_id": sid, "accepted_key": rows[sid]["accepted_key"], "control_key": rows[sid]["control_key"], "wave_uuid": draft["wave_uuid"],
                  "launch_plan_sha": sha}]
        snap = {"fc-1": {"done": True, "summary": {"status": "LOST_CONTAINER"}, "error": None, "accepted": None}}
        return rows, sid, ledger, calls, snap

    def test_two_concurrent_passes_spawn_exactly_one_successor(self, srv, draft, tmp_path):
        rows, sid, ledger, calls, snap = self._setup(draft)
        spawned = []

        def inner_spawn(row, intent):
            spawned.append(("inner", row["standup_id"]))
            return "fc-inner"

        def outer_spawn(row, intent):
            # the RACE: a second pass runs to completion between this pass's intent claim and its spawn
            nested = srv.apply_reattempt_policy(draft, tmp_path, calls=calls, snapshot=snap, rows_by_sid=rows, ledger=ledger, spawn=inner_spawn, log=lambda *a: None, pass_id="pass-B")
            assert nested == [] and spawned == []                     # the nested pass found the intent claimed and spawned nothing
            spawned.append(("outer", row["standup_id"]))
            return "fc-outer"
        out = srv.apply_reattempt_policy(draft, tmp_path, calls=calls, snapshot=snap, rows_by_sid=rows, ledger=ledger, spawn=outer_spawn, log=lambda *a: None, pass_id="pass-A")
        assert [d["action"] for d in out] == ["reattempt"] and out[0]["reattempt_call_id"] == "fc-outer" and spawned == [("outer", sid)]
        e = srv.reattempt_ledger_get(ledger, sid)
        assert e["reattempts"] == 1 and e["launch_plan_sha"] == draft["launch_plan_sha"] and e["history"][0]["reattempt_call_id"] == "fc-outer"
        it = json.loads(ledger.get(srv.reattempt_intent_key(sid, "fc-1")))
        assert it["spawned_call_id"] == "fc-outer" and it["pass_id"] == "pass-A" and it["launch_plan_sha"] == draft["launch_plan_sha"] and it["reattempt_n"] == 1
        # a third pass: nothing (judged; the intent is spawned)
        assert srv.apply_reattempt_policy(draft, tmp_path, calls=calls, snapshot=snap, rows_by_sid=rows, ledger=ledger, spawn=outer_spawn, log=lambda *a: None) == []
        assert spawned == [("outer", sid)]

    def test_a_crash_between_intent_and_spawn_is_recovered_once_never_lost_never_duplicated(self, srv, draft, tmp_path):
        rows, sid, ledger, calls, snap = self._setup(draft)
        spawned = []

        def crashing_spawn(row, intent):
            raise RuntimeError("the pass died before .spawn() returned")
        t0 = 1_000_000.0
        with pytest.raises(RuntimeError, match="died before"):
            srv.apply_reattempt_policy(draft, tmp_path, calls=calls, snapshot=snap, rows_by_sid=rows, ledger=ledger, spawn=crashing_spawn, log=lambda *a: None, now=t0, pass_id="pass-dead")
        key = srv.reattempt_intent_key(sid, "fc-1")
        it = json.loads(ledger.get(key))
        assert it["spawned_call_id"] is None and it["pass_id"] == "pass-dead" and it["lease_until"] == t0 + srv.REATTEMPT_INTENT_LEASE_S
        e = srv.reattempt_ledger_get(ledger, sid)
        assert e["reattempts"] == 1 and e["judged_calls"] == ["fc-1"], "the counter was written before the spawn (D73: never a second re-attempt)"

        def spawn(row, intent):
            spawned.append((row["standup_id"], intent["key"], intent["recoveries"], intent["recovered_from"]))
            return f"fc-rec{len(spawned)}"
        # inside the lease: the intent belongs to a (possibly live) pass -- nothing is recovered, nothing spawned, nothing lost
        assert srv.apply_reattempt_policy(draft, tmp_path, calls=calls, snapshot=snap, rows_by_sid=rows, ledger=ledger, spawn=spawn, log=lambda *a: None, now=t0 + 10) == []
        assert spawned == [] and srv.stale_unspawned_intents(ledger, launch_plan_sha=draft["launch_plan_sha"], now=t0 + 10) == []
        # after the lease: recovered by exactly ONE pass and spawned once
        assert [srv.stale_unspawned_intents(ledger, launch_plan_sha=draft["launch_plan_sha"], now=t0 + 1000)[0]["key"]] == [key]
        out = srv.apply_reattempt_policy(draft, tmp_path, calls=calls, snapshot=snap, rows_by_sid=rows, ledger=ledger, spawn=spawn, log=lambda *a: None, now=t0 + 1000, pass_id="pass-new")
        assert [(d["action"], d.get("recovered"), d["reattempt_call_id"]) for d in out] == [("reattempt", True, "fc-rec1")] and spawned == [(sid, key, 1, "pass-dead")]
        it = json.loads(ledger.get(key))
        assert it["spawned_call_id"] == "fc-rec1" and it["pass_id"] == "pass-new" and it["recoveries"] == 1
        e = srv.reattempt_ledger_get(ledger, sid)
        assert e["reattempts"] == 1 and e["judged_calls"] == ["fc-1"] and e["history"][-1]["reattempt_call_id"] == "fc-rec1"
        # a later pass: no second recovery, no second spawn; a concurrent recoverer that pops nothing gives up
        assert srv.apply_reattempt_policy(draft, tmp_path, calls=calls, snapshot=snap, rows_by_sid=rows, ledger=ledger, spawn=spawn, log=lambda *a: None, now=t0 + 5000) == []
        assert spawned == [(sid, key, 1, "pass-dead")] and srv.recover_intent(ledger, {"key": "intent|nope|x"}, pass_id="p", now=t0) is None
        # the re-attempt's own infra failure -> HOLD (the second failure), and the intent machinery leaves it held
        calls.append({**calls[0], "call_id": "fc-rec1"})
        snap["fc-rec1"] = {"done": True, "summary": {"status": "PREEMPTED"}, "error": None, "accepted": None}
        out = srv.apply_reattempt_policy(draft, tmp_path, calls=calls, snapshot=snap, rows_by_sid=rows, ledger=ledger, spawn=spawn, log=lambda *a: None, now=t0 + 6000)
        assert [d["action"] for d in out] == ["hold"] and srv.reattempt_ledger_get(ledger, sid)["held"] is True and len(spawned) == 1
        log = (tmp_path / "legs.log").read_text()
        assert "RECOVERED intent" in log and "pass pass-dead died before spawning" in log and "HOLD codeattack__s1__base__shard00" in log

    def test_a_regenerated_same_wave_plan_never_retries_an_old_failure_and_foreign_ledger_entries_are_refused(self, srv, draft, tmp_path):
        rows, sid, ledger, calls, snap = self._setup(draft)
        spawned = []
        spawn = lambda row, intent: spawned.append(row["standup_id"]) or "fc-x"  # noqa: E731
        regenerated = {**draft, "launch_plan_sha": "f" * 16}
        # the old plan's call is not this plan's (wave_calls excludes it; the policy refuses it by name)
        (tmp_path / "calls.jsonl").write_text(json.dumps({**calls[0], "smoke": None}) + "\n")
        assert srv.wave_calls(tmp_path, draft) == [{**calls[0], "smoke": None}] and srv.wave_calls(tmp_path, regenerated) == []
        out = srv.apply_reattempt_policy(regenerated, tmp_path, calls=calls, snapshot=snap, rows_by_sid=rows, ledger=ledger, spawn=spawn, log=lambda *a: None)
        assert [(d["action"], d["reason"]) for d in out] == [("refused", "call of another plan")] and spawned == [] and ledger.get(sid) is None
        # a ledger entry of another plan for the same standup: refused, no spawn
        srv.reattempt_ledger_put(ledger, {**srv.reattempt_ledger_get(ledger, sid), "launch_plan_sha": "e" * 16})
        out = srv.apply_reattempt_policy(draft, tmp_path, calls=calls, snapshot=snap, rows_by_sid=rows, ledger=ledger, spawn=spawn, log=lambda *a: None)
        assert [(d["action"], d["reason"]) for d in out] == [("refused", "ledger entry of another plan")] and spawned == []
        # a plan with no sha retries nothing; a call lacking the stamp (pre-fix record) is refused
        assert srv.apply_reattempt_policy({**draft, "launch_plan_sha": ""}, tmp_path, calls=calls, snapshot=snap, rows_by_sid=rows, ledger=FakeDict("l2"), spawn=spawn, log=lambda *a: None) == []
        unstamped = [{k: v for k, v in calls[0].items() if k != "launch_plan_sha"}]
        out = srv.apply_reattempt_policy(draft, tmp_path, calls=unstamped, snapshot=snap, rows_by_sid=rows, ledger=FakeDict("l3"), spawn=spawn, log=lambda *a: None)
        assert out[0]["action"] == "refused" and spawned == []

    def test_the_container_executes_an_intent_once_and_the_spawn_carries_it(self, srv, draft, tmp_path, monkeypatch):
        import inspect
        FakeDict.store.clear()
        ledger = FakeDict("reattempts")
        key = srv.reattempt_intent_key("codeattack__s1__base__shard00", "fc-1")
        assert srv.reattempt_execution_claim(ledger, key, "ta-first") == (True, None)
        assert srv.reattempt_execution_claim(ledger, key, "ta-second") == (False, "ta-first"), "the duplicate spawn loses and exits before any GPU minute"
        src = inspect.getsource(srv._run_standup_row)
        assert 'if ctx.get("reattempt_intent"):' in src and "reattempt_execution_claim(reatt" in src and "return fail(REATTEMPT_DUPLICATE" in src
        assert src.index("return fail(REATTEMPT_DUPLICATE") < src.index('return fail("IMAGE_UNPREPARED"')
        # a REATTEMPT_DUPLICATE exit is never judged as a standup failure
        rows = {r["standup_id"]: r for r in draft["standups"]}
        sid = "codeattack__s1__base__shard00"
        calls = [{"call_id": "fc-dup", "standup_id": sid, "accepted_key": rows[sid]["accepted_key"], "control_key": rows[sid]["control_key"], "launch_plan_sha": draft["launch_plan_sha"]}]
        snap = {"fc-dup": {"done": True, "summary": {"status": srv.REATTEMPT_DUPLICATE}, "error": None, "accepted": None}}
        assert srv.apply_reattempt_policy(draft, tmp_path, calls=calls, snapshot=snap, rows_by_sid=rows, ledger=ledger, spawn=lambda r, i: "x", log=lambda *a: None) == []
        assert ledger.get(sid) is None
        # spawn_row threads the claimed intent into the ctx and the call record; reattempt_pass passes it (source pin)
        class FC:
            object_id = "fc-re"

        class FN:
            def spawn(self, row_, ctx_):
                FN.ctx = ctx_
                return FC()
        monkeypatch.setitem(srv.FUNCTIONS, "run_view_leg", FN())
        cid = srv.spawn_row(rows[sid], draft, "seedrep-served-2222222", tmp_path, None, reattempt_intent=key)
        assert cid == "fc-re" and FN.ctx["reattempt_intent"] == key
        rec = json.loads((tmp_path / "calls.jsonl").read_text().splitlines()[-1])
        assert rec["reattempt_intent"] == key and rec["launch_plan_sha"] == draft["launch_plan_sha"]
        assert 'reattempt_intent=intent["key"]' in inspect.getsource(srv.reattempt_pass)


# =========================================================================== the retry steward -- automatic, bounded, single-owner, launched with the fan-out

class TestSolR3_3_retry_steward:
    """The retry policy once ran only when an operator invoked `--reattempt` ("Not built: a daemon that runs --reattempt
    on a timer"). Now `scripts/modal/seedrep_steward.py`: a `systemd-run --user` transient unit (the repo's judge / grade-unit idiom) launched by
    the app right after the production spawns, polling the certified wave, invoking the pass at most once per failed call, stopping on
    WAVE_DONE / HOLD / its own deadline (`RuntimeMaxSec` on the unit AND the loop's clock), single owner by unit name + flock."""

    def _st(self):
        return _load_peer("seedrep_steward_under_test", "scripts/modal/seedrep_steward.py", REPO)

    def test_launch_uses_a_fake_systemd_run_on_path_with_the_unit_idiom(self, draft, tmp_path, monkeypatch):
        st = self._st()
        fake_bin = tmp_path / "bin"
        fake_bin.mkdir()
        recorded = tmp_path / "systemd-run.argv.json"
        fake = fake_bin / "systemd-run"
        fake.write_text('#!/bin/bash\n' + f'python3 -c "import json,sys; json.dump(sys.argv[1:], open({str(recorded)!r}, \'w\'))" "$@"\n' + 'exit "${FAKE_SYSTEMD_RC:-0}"\n')
        fake.chmod(0o755)
        monkeypatch.setenv("PATH", f"{fake_bin}:{os.environ.get('PATH', '')}")
        plan_p, cert_p, pools_p = tmp_path / "launch_plan.json", tmp_path / "launch_certificate.json", tmp_path / "pool_urls.json"
        plan_p.write_text(json.dumps(draft, default=str))
        rep = st.launch_steward(rs=tmp_path / "RS", plan=draft, plan_path=plan_p, certificate=cert_p, app_name="seedrep-served-2222222", sha=W, tree=REPO,
                                app_script=REPO / "scripts/modal/seedrep_served.py", python="/venv/bin/python", pool_urls=pools_p, poll_s=60, deadline_h=30.0)
        assert rep["launched"] is True and rep["rc"] == 0 and rep["unit"] == f"seedrep-steward-2222222-{WAVE[:8]}"
        argv = json.loads(recorded.read_text())
        assert argv == rep["argv"][1:]
        assert argv[:4] == ["--user", f"--unit={rep['unit']}", "--collect", "--quiet"] and argv[4].startswith("--setenv=PATH=") and ".local/bin" in argv[4]
        assert ["-p", f"WorkingDirectory={REPO}", "-p", "RuntimeMaxSec=108000"] == argv[5:9] and argv[9:11] == ["/bin/bash", "-c"] and argv[12] == rep["log"]
        inner = argv[13:]
        assert inner[:3] == ["/venv/bin/python", str(REPO / "scripts/modal/seedrep_steward.py"), "run"]
        assert inner[inner.index("--plan") + 1] == str(plan_p) and inner[inner.index("--certificate") + 1] == str(cert_p) and inner[inner.index("--deadline-h") + 1] == "30.0"
        pass_cmd = json.loads(inner[inner.index("--pass-cmd") + 1])
        assert pass_cmd[:6] == ["/venv/bin/python", "-m", "modal", "run", "--detach", "-n"] and pass_cmd[6] == "seedrep-served-2222222"
        assert pass_cmd[7] == str(REPO / "scripts/modal/seedrep_served.py") and "--reattempt" in pass_cmd and pass_cmd[pass_cmd.index("--certificate") + 1] == str(cert_p)
        assert pass_cmd[pass_cmd.index("--pool-urls") + 1] == str(pools_p) and pass_cmd[pass_cmd.index("--sha") + 1] == W
        assert Path(rep["log"]).parent == tmp_path / "RS" / "logs" and Path(rep["log"]).name == f"{rep['unit']}.log"
        assert "--start-token" not in inner and "--fanout-cmd" not in inner and rep["start_token"] == ""     # no token: a bare steward (tests); the launcher always passes one
        rep_t = st.launch_steward(rs=tmp_path / "RS", plan=draft, plan_path=plan_p, certificate=cert_p, app_name="seedrep-served-2222222", sha=W, tree=REPO,
                                  app_script=REPO / "scripts/modal/seedrep_served.py", python="/venv/bin/python", pool_urls=pools_p, start_token="tok42")
        inner_t = json.loads(recorded.read_text())[13:]
        fan = json.loads(inner_t[inner_t.index("--fanout-cmd") + 1])
        assert inner_t[inner_t.index("--start-token") + 1] == "tok42" == rep_t["start_token"] and "--fanout" in fan and fan[fan.index("--start-token") + 1] == "tok42"
        assert fan[:8] == pass_cmd[:8] and fan[fan.index("--certificate") + 1] == str(cert_p) and fan[fan.index("--pool-urls") + 1] == str(pools_p) and "--reattempt" not in fan
        # systemd refuses a second unit of the same name (a unit already running): reported, never a second owner
        monkeypatch.setenv("FAKE_SYSTEMD_RC", "1")
        rep2 = st.launch_steward(rs=tmp_path / "RS", plan=draft, plan_path=plan_p, certificate=cert_p, app_name="seedrep-served-2222222", sha=W, tree=REPO,
                                 app_script=REPO / "scripts/modal/seedrep_served.py", python="/venv/bin/python")
        assert rep2["launched"] is False and rep2["rc"] == 1 and rep2["unit"] == rep["unit"] and "why" in rep2
        # the pass command needs no pool file when none is given; the idiom matches the repo's judge units
        assert "--pool-urls" not in st.pass_command(python="p", app_script=Path("s"), app_name="a", sha=W, plan=Path("P"), rs=Path("R"), certificate=Path("C"))
        idiom = (REPO / "scripts/sdfbase/full22/sampled/grade_sampled.sh").read_text()
        assert 'systemd-run --user --unit="$UNIT" --collect --quiet --setenv=PATH="$HOME/.local/bin:$PATH" -p WorkingDirectory="$WT"' in idiom

    def _launch_world(self, srv, draft, tmp_path, monkeypatch):
        """The launcher under test: a fake certificate (the plan's real digest), a fake steward that APPENDS its STEWARD_STARTED for the token it was
        given (the log is append-only), a recording spawn, a selection by leg."""
        import fcntl
        rs = tmp_path / "RS"
        rs.mkdir(exist_ok=True)
        assert lp.plan_sha(draft) == draft["launch_plan_sha"], "the fixture's stamp is its digest"
        cert = {"certificate_sha256": "c" * 64, "plan_sha256": "p" * 64, "launch_plan_sha": draft["launch_plan_sha"]}
        events, seen_intents, spawned = [], [], []
        monkeypatch.setattr(srv, "require_certificate", lambda *a, **k: events.append("cert") or dict(cert))
        st = _load_peer("seedrep_steward_under_test", "scripts/modal/seedrep_steward.py", REPO)
        tokens = iter(f"tok{i}" for i in range(1, 50))

        def fake_steward(*, rs, plan, plan_path, certificate, app_name, sha, tree, app_script, pool_urls, deadline_h, start_token, launched=True, started=True):
            log = Path(rs) / "logs" / "unit.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            intent = lp.read_production_intent(Path(rs) / lp.PRODUCTION_LAUNCH_FILE)
            seen_intents.append(intent)                                     # the intent EXISTS before the steward is launched, naming this token
            assert intent is not None and intent["steward"]["start_token"] == start_token
            events.append(("steward", start_token))
            if started:
                with log.open("a") as f:
                    f.write('STEWARD_STARTED {"pid": 4242, "required_keys": %d, "start_token": "%s"}\n' % (len(intent["required_keys"]), start_token))
            return {"unit": "seedrep-steward-x", "log": str(log), "argv": [], "rc": 0 if launched else 1, "launched": launched, "why": None if launched else "unit exists",
                    "start_token": start_token}
        claims = []
        spawn = lambda r, plan, app, rs_, pools, smoke=None, fanout_claim=None: (spawned.append(r["standup_id"]), claims.append(fanout_claim)) and f"fc-{len(spawned)}"  # noqa: E731
        select = lambda plan, go="", legs="", pool_urls=None: ([r for r in plan["standups"] if r["leg"] in {x for x in legs.split(",") if x}] if legs else [r for r in plan["standups"] if r["standup_id"] == go], None)  # noqa: E731
        common = dict(plan_path=tmp_path / "plan.json", rs=rs, app_name="seedrep-served-2222222", sha=W, pools=None, pool_urls_path=None, cert_p=tmp_path / "cert.json",
                      select=select, new_token=lambda: next(tokens),
                      wait_started=lambda log, token, timeout_s: st.wait_started(log, token=token, timeout_s=0.3, clock=__import__("time").monotonic, sleep=lambda s: None))
        return dict(rs=rs, cert=cert, events=events, seen_intents=seen_intents, spawned=spawned, claims=claims, st=st, fake_steward=fake_steward, spawn=spawn, select=select,
                    common=common, fcntl=fcntl, intent_p=rs / lp.PRODUCTION_LAUNCH_FILE, ledger=FakeDict(f"led-{tmp_path.name}"))

    def test_delta2_p1_c_the_intent_is_written_before_the_steward_and_the_launcher_spawns_nothing(self, srv, draft, tmp_path, monkeypatch):
        """`--legs` writes the durable fan-out intent (certificate sha, plan digest + file sha, wave, the full required-key set,
        per-row spawn state, the steward's start token) BEFORE the steward is launched, awaits STEWARD_STARTED for THAT token, and spawns nothing;
        a steward that does not start / never acknowledges leaves the intent for a later --go to resume; a second --go refuses ("already launched")
        while the steward owns the wave or once every row is spawned, extends the intent with new rows of the SAME certificate, and refuses an
        intent of another certificate -- never a second wave."""
        w = self._launch_world(srv, draft, tmp_path, monkeypatch)
        rs, st, events, spawned, common = w["rs"], w["st"], w["events"], w["spawned"], w["common"]
        # (a) a fresh launch: cert -> intent (token tok1) -> steward -> STARTED(tok1); nothing spawned by the launcher
        out = srv.launch_production(draft, steward_launch=w["fake_steward"], legs="pasttense", **common)
        assert events == ["cert", ("steward", "tok1")] and spawned == [] and out["mode"] == "new" and out["started"]["start_token"] == "tok1" and out["spawned_by_launcher"] == []
        intent = lp.read_production_intent(w["intent_p"])
        pt = sorted(r["standup_id"] for r in draft["standups"] if r["leg"] == "pasttense")
        assert intent["schema"] == "seedrep_production_launch/1" and sorted(intent["rows"]) == pt == out["added"] and lp.production_intent_unspawned(intent) == pt
        assert intent["certificate_sha256"] == "c" * 64 and intent["plan_sha256"] == "p" * 64 and intent["launch_plan_sha"] == draft["launch_plan_sha"] and intent["wave_uuid"] == draft["wave_uuid"]
        assert intent["required_keys"] == lp.plan_required_keys(draft) and len(intent["required_keys"]) == 130 and intent["steward"] == {**intent["steward"], "unit": f"seedrep-steward-2222222-{WAVE[:8]}", "start_token": "tok1"}
        assert w["seen_intents"][0]["steward"]["start_token"] == "tok1" and all(r["call_id"] is None for r in w["seen_intents"][0]["rows"].values()), "written BEFORE the steward"
        assert lp.production_intent_problems(intent, plan=draft, certificate=w["cert"]) == []
        legs_log = (rs / "legs.log").read_text()
        assert "INTENT new rows=4 added=4 unspawned=4 token=tok1" in legs_log and "STEWARD STARTED unit=seedrep-steward-x pid=4242 token=tok1" in legs_log and "SPAWNED" not in legs_log
        # (b) the steward owns the wave (its lock held), the same selection again -> REFUSED before anything: production already launched
        fd = os.open(rs / st.LOCK_FILE, os.O_RDWR | os.O_CREAT)
        w["fcntl"].flock(fd, w["fcntl"].LOCK_EX | w["fcntl"].LOCK_NB)
        events.clear()
        try:
            with pytest.raises(srv.LaunchRefused, match="production is already launched for certificate cccccccccccc .*0/4 rows spawned; the steward owns the wave"):
                srv.launch_production(draft, steward_launch=w["fake_steward"], legs="pasttense", **common)
            assert events == ["cert"] and spawned == []
            # (c) a later --legs of ANOTHER leg under the SAME certificate while the steward owns the wave: the intent is EXTENDED, no second steward,
            # the running steward fans the new rows out on its next tick
            out = srv.launch_production(draft, steward_launch=w["fake_steward"], legs="codeattack", **common)
            ca = sorted(r["standup_id"] for r in draft["standups"] if r["leg"] == "codeattack")
            assert out["mode"] == "extended" and out["added"] == ca and out["steward"] is None and events == ["cert", "cert"]
            intent = lp.read_production_intent(w["intent_p"])
            assert sorted(intent["rows"]) == sorted(pt + ca) and intent["steward"]["start_token"] == "tok1" and len(intent["selections"]) == 2 and intent["selections"][1]["added"] == ca
        finally:
            os.close(fd)
        # (d) the steward is gone (lock free) with rows still unspawned -> the SAME intent is resumed under a NEW token (never a second wave)
        events.clear()
        out = srv.launch_production(draft, steward_launch=w["fake_steward"], legs="pasttense", **common)
        intent = lp.read_production_intent(w["intent_p"])
        assert out["mode"] == "resumed" and out["added"] == [] and events == ["cert", ("steward", "tok2")] and intent["steward"]["start_token"] == "tok2" and sorted(intent["rows"]) == sorted(pt + ca)
        assert spawned == []
        # (e) the steward does not start -> refused, the intent stays (a later --go resumes it); never reported for THIS token (a STALE STEWARD_STARTED
        # of tok1 / tok2 sits in the append-only log) -> refused, nothing spawned
        with pytest.raises(srv.LaunchRefused, match="did not start .* the intent .* stays"):
            srv.launch_production(draft, steward_launch=lambda **k: w["fake_steward"](launched=False, **k), legs="pasttense", **common)
        assert w["intent_p"].is_file() and lp.read_production_intent(w["intent_p"])["steward"]["start_token"] == "tok3"
        with pytest.raises(srv.LaunchRefused, match="never reported STEWARD_STARTED for token tok4"):
            srv.launch_production(draft, steward_launch=lambda **k: w["fake_steward"](started=False, **k), legs="pasttense", **common)
        assert "STEWARD_STARTED" in (rs / "logs" / "unit.log").read_text() and spawned == []
        # (f) every row spawned (as the steward's fan-out leaves it) and the steward gone -> a --go is refused: already launched, nothing to resume
        intent = lp.read_production_intent(w["intent_p"])
        for i, sid in enumerate(sorted(intent["rows"])):
            lp.production_intent_mark_spawned(intent, sid, f"fc-{i}", by="steward:tok2")
        lp.write_production_intent(w["intent_p"], intent)
        with pytest.raises(srv.LaunchRefused, match="already launched for certificate cccccccccccc .*8/8 rows spawned"):
            srv.launch_production(draft, steward_launch=w["fake_steward"], legs="pasttense", **common)
        # (g) an intent of ANOTHER certificate (a regenerated plan certified anew) -> refused by name; nothing spawned, nothing relaunched
        intent["certificate_sha256"] = "d" * 64
        lp.write_production_intent(w["intent_p"], intent)
        with pytest.raises(srv.LaunchRefused, match="never a second wave: .*written for certificate dddddddddddd, not this certificate cccccccccccc"):
            srv.launch_production(draft, steward_launch=w["fake_steward"], legs="pasttense", **common)
        assert spawned == [] and "SPAWNED" not in (rs / "legs.log").read_text()
        # (h) another launcher mid-flight -> refused; a failing certificate -> refused before any lock / intent
        fd = os.open(rs / srv.LAUNCH_LOCK_FILE, os.O_RDWR | os.O_CREAT)
        w["fcntl"].flock(fd, w["fcntl"].LOCK_EX | w["fcntl"].LOCK_NB)
        try:
            with pytest.raises(srv.LaunchRefused, match="another launcher holds"):
                srv.launch_production(draft, steward_launch=w["fake_steward"], legs="pasttense", **common)
        finally:
            os.close(fd)
        monkeypatch.setattr(srv, "require_certificate", lambda *a, **k: (_ for _ in ()).throw(srv.LaunchRefused("cert refused")))
        rs2 = tmp_path / "RS2"
        with pytest.raises(srv.LaunchRefused, match="cert refused"):
            srv.launch_production(draft, steward_launch=w["fake_steward"], legs="pasttense", **{**common, "rs": rs2})
        assert not (rs2 / lp.PRODUCTION_LAUNCH_FILE).exists()
        # the entrypoint and the launcher: production goes through launch_production only; the launcher never spawns; --fanout exists for the steward
        import inspect
        src = inspect.getsource(srv.main)
        lsrc = inspect.getsource(srv.launch_production)
        assert "launch_production(plan_d" in src and "run_fanout(plan_d" in src and "no_steward" not in src and "st.launch_steward(" not in src and "if fanout:" in src
        assert "spawn_row(" not in lsrc and "spawn(r" not in lsrc and lsrc.index("require_certificate(plan_d") < lsrc.index("LOCK_EX | fcntl.LOCK_NB") < lsrc.index("production_intent_new(") \
            < lsrc.index("write_production_intent(intent_p, intent)                 # the durable claim") < lsrc.index("steward_launch(") < lsrc.index("wait_started(")

    def test_delta2_p1_c_the_steward_completes_the_fan_out_idempotently_from_the_intent(self, srv, draft, tmp_path, monkeypatch):
        """`run_fanout` (the steward's `--fanout` pass): under the certificate and ONLY while a steward owns the wave and the token is the intent's, every
        row with no call is spawned once and recorded in the intent before the next; a pass that crashes mid-way leaves rows due -- the next pass
        spawns exactly those; a call already in calls.jsonl (a crash between spawn and record) is ADOPTED, never re-spawned; a second pass spawns
        nothing. `wait_started` correlates on the token (a stale STARTED is ignored); `run_steward` echoes the token and runs the fan-out on start
        and on every tick while rows are due."""
        w = self._launch_world(srv, draft, tmp_path, monkeypatch)
        rs, st, spawned, common = w["rs"], w["st"], w["spawned"], w["common"]
        srv.launch_production(draft, steward_launch=w["fake_steward"], legs="pasttense", **common)
        pt = sorted(r["standup_id"] for r in draft["standups"] if r["leg"] == "pasttense")
        nonces = iter(f"n{i}" for i in range(1, 50))
        fan = dict(plan_path=common["plan_path"], rs=rs, app_name="seedrep-served-2222222", pools=None, cert_p=common["cert_p"], ledger=w["ledger"], new_nonce=lambda: next(nonces))
        # no steward owns the wave -> the fan-out refuses before any spawn
        with pytest.raises(srv.LaunchRefused, match="no steward owns wave .* the fan-out runs under the steward only"):
            srv.run_fanout(draft, start_token="tok1", spawn=w["spawn"], **fan)
        assert spawned == []
        fd = os.open(rs / st.LOCK_FILE, os.O_RDWR | os.O_CREAT)
        w["fcntl"].flock(fd, w["fcntl"].LOCK_EX | w["fcntl"].LOCK_NB)
        try:
            with pytest.raises(srv.LaunchRefused, match="start token 'tokX' is not the intent's 'tok1'"):
                srv.run_fanout(draft, start_token="tokX", spawn=w["spawn"], **fan)
            assert spawned == []
            # a pass that crashes after the FIRST spawn: that row is recorded, the rest stay due
            calls = {"n": 0}

            def crashing(r, plan, app, rs_, pools, smoke=None, fanout_claim=None):
                calls["n"] += 1
                if calls["n"] > 1:
                    raise RuntimeError("launcher died")
                spawned.append(r["standup_id"])
                w["claims"].append(fanout_claim)
                return f"fc-{len(spawned)}"
            with pytest.raises(RuntimeError, match="launcher died"):
                srv.run_fanout(draft, start_token="tok1", spawn=crashing, **fan)
            intent = lp.read_production_intent(w["intent_p"])
            assert spawned == pt[:1] and intent["rows"][pt[0]]["call_id"] == "fc-1" and intent["rows"][pt[0]]["spawned_by"] == "steward:tok1" and lp.production_intent_unspawned(intent) == pt[1:]
            # delta 3 (4): the claim was written BEFORE each spawn -- pt[0] claimed n1 and recorded; pt[1] claimed n2 with NO call (the pass died at .spawn())
            assert intent["rows"][pt[0]]["claim"] == "n1" and w["claims"][0] == {"standup_id": pt[0], "nonce": "n1"}
            assert intent["rows"][pt[1]]["claim"] == "n2" and intent["rows"][pt[1]]["call_id"] is None and lp.production_intent_claimed_unspawned(intent) == [pt[1]]
            # a crash between spawn and record: the call is in calls.jsonl under THAT claim -> ADOPTED, never re-spawned
            srv._record_call(rs, {"call_id": "fc-orphan", "standup_id": pt[1], "wave_uuid": draft["wave_uuid"], "launch_plan_sha": draft["launch_plan_sha"], "smoke": None,
                                  "accepted_key": "k", "control_key": "c", "reattempt_intent": None, "fanout_claim": {"standup_id": pt[1], "nonce": "n2"}})
            out = srv.run_fanout(draft, start_token="tok1", spawn=w["spawn"], **fan)
            assert out["spawned"] == pt[2:] and out["adopted"] == [pt[1]] and out["recovered"] == [] and out["remaining"] == [] and out["rows"] == 4 and spawned == [pt[0]] + pt[2:]
            intent = lp.read_production_intent(w["intent_p"])
            assert intent["rows"][pt[1]]["call_id"] == "fc-orphan" and lp.production_intent_unspawned(intent) == [] and all(r["call_id"] and r["claim"] for r in intent["rows"].values())
            # a second pass: nothing due, nothing spawned (idempotent)
            n = len(spawned)
            assert srv.run_fanout(draft, start_token="tok1", spawn=w["spawn"], **fan)["spawned"] == [] and len(spawned) == n
            log = (rs / "legs.log").read_text()
            assert f"FANOUT {pt[1]} call=fc-orphan adopted-from-calls claim=n2 token=tok1" in log and "FANOUT done spawned=2 adopted=1 recovered=0" in log
            # a row marked with a DIFFERENT call is a duplicate row: refused by the intent
            with pytest.raises(ValueError, match="already spawned as call fc-1; a second call fc-9 is a duplicate row"):
                lp.production_intent_mark_spawned(intent, pt[0], "fc-9", by="x")
        finally:
            os.close(fd)
        # wait_started correlates on the token: a stale STARTED (tok1) never acknowledges tok9; ALREADY_OWNED for the token -> None; no token -> refused
        log_p = rs / "logs" / "unit.log"
        assert st.wait_started(log_p, token="tok1", timeout_s=0.1, sleep=lambda s: None)["start_token"] == "tok1"
        assert st.wait_started(log_p, token="tok9", timeout_s=0.1, sleep=lambda s: None) is None, "the FIRST historical STEWARD_STARTED is not an acknowledgement"
        with log_p.open("a") as f:
            f.write('STEWARD_ALREADY_OWNED {"lock": "x", "start_token": "tok9"}\n')
            f.write('STEWARD_STARTED {"pid": 1, "start_token": "tok10"}\n')
        assert st.wait_started(log_p, token="tok9", timeout_s=0.1, sleep=lambda s: None) is None
        assert st.wait_started(log_p, token="tok10", timeout_s=0.1, sleep=lambda s: None) == {"pid": 1, "start_token": "tok10"}
        with pytest.raises(ValueError, match="needs the invocation's start token"):
            st.wait_started(log_p, token="", timeout_s=0.1, sleep=lambda s: None)
        # run_steward: STARTED / ALREADY_OWNED echo the token; the fan-out runs on start and on every tick while rows are due, from the intent
        intent = lp.read_production_intent(w["intent_p"])
        for sid in intent["rows"]:
            intent["rows"][sid].update(call_id=None, spawned_utc=None, spawned_by=None, claim=None, claimed_utc=None, executing_call_id=None)
        lp.write_production_intent(w["intent_p"], intent)
        fanouts = []

        def fake_fanout():
            it = lp.read_production_intent(w["intent_p"])
            due = lp.production_intent_unspawned(it)
            for sid in due[:2]:                                            # two rows per pass (a pass that "crashed" half-way)
                lp.production_intent_mark_spawned(it, sid, f"fc-{sid}", by="steward:tok1")
            lp.write_production_intent(w["intent_p"], it)
            fanouts.append(len(due))
            return 0
        rows = {r["standup_id"]: r for r in draft["standups"] + draft["rungs"]}

        def running(by):
            return {cid: {"done": False, "accepted": None} for cid in by.values()}

        def all_done(by):
            return {cid: {"done": True, "summary": {"status": "DONE"}, "accepted": f"ta-{cid}"} for cid in by.values()}
        rc, ev, n, required = self._loop(st, draft, tmp_path / "loop", [running, running, all_done], [lambda by: {}], start_token="tok1", intent_path=w["intent_p"], run_fanout=fake_fanout)
        assert rc == 0 and fanouts == [4, 2] and json.loads(ev[0].split(" ", 1)[1])["start_token"] == "tok1" and json.loads(ev[0].split(" ", 1)[1])["intent"] == str(w["intent_p"])
        fo = [json.loads(e.split(" ", 1)[1]) for e in ev if e.startswith("STEWARD_FANOUT ")]
        assert [(x["due_before"], x["due_after"], x["rc"], x["start_token"]) for x in fo] == [(4, 2, 0, "tok1"), (2, 0, 0, "tok1")] and lp.production_intent_unspawned(lp.read_production_intent(w["intent_p"])) == []
        # launch_steward carries the token + the fan-out command into the unit; fanout_command is the app's --fanout pass
        fan_cmd = st.fanout_command(python="p", app_script=Path("s"), app_name="a", sha=W, plan=Path("P"), rs=Path("R"), certificate=Path("C"), start_token="tok7")
        assert fan_cmd == ["p", "-m", "modal", "run", "--detach", "-n", "a", "s", "--sha", W, "--plan", "P", "--rs", "R", "--fanout", "--start-token", "tok7", "--certificate", "C"]
        argv = st.steward_argv(unit="u", log=Path("L"), tree=REPO, python="p", rs=Path("R"), plan=Path("P"), certificate=Path("C"), pass_cmd=["x"], poll_s=60, deadline_h=1.0,
                               start_token="tok7", fanout_cmd=fan_cmd)
        assert argv[argv.index("--start-token") + 1] == "tok7" and json.loads(argv[argv.index("--fanout-cmd") + 1]) == fan_cmd
        assert "--start-token" not in st.steward_argv(unit="u", log=Path("L"), tree=REPO, python="p", rs=Path("R"), plan=Path("P"), certificate=Path("C"), pass_cmd=["x"], poll_s=60, deadline_h=1.0)
        assert len(st.new_start_token()) == 16 and st.new_start_token() != st.new_start_token()

    def test_delta3_4_exactly_once_initial_fan_out_across_the_modal_spawn_boundary(self, srv, draft, tmp_path, monkeypatch):
        """A death between `.spawn()` and the call record never yields two executing containers -- the row's claim (a nonce)
        is durable BEFORE the spawn, rides to the container, and the container claims `fanout|<row>|<nonce>|executing` at entry (a duplicate exits
        FANOUT_DUPLICATE before any standup); recovery of a claimed row with no call reads the container's mark first, then calls.jsonl, and only
        then re-spawns under the SAME nonce; the executing container's call id is adopted from the mark; the steward never treats a duplicate's exit
        as a failure."""
        w = self._launch_world(srv, draft, tmp_path, monkeypatch)
        rs, st, spawned, common, led = w["rs"], w["st"], w["spawned"], w["common"], w["ledger"]
        srv.launch_production(draft, steward_launch=w["fake_steward"], legs="pasttense", **common)
        pt = sorted(r["standup_id"] for r in draft["standups"] if r["leg"] == "pasttense")
        nonces = iter(f"c{i}" for i in range(1, 50))
        fan = dict(plan_path=common["plan_path"], rs=rs, app_name="seedrep-served-2222222", pools=None, cert_p=common["cert_p"], ledger=led, new_nonce=lambda: next(nonces))
        fd = os.open(rs / st.LOCK_FILE, os.O_RDWR | os.O_CREAT)
        w["fcntl"].flock(fd, w["fcntl"].LOCK_EX | w["fcntl"].LOCK_NB)
        try:
            # (a) pass A: claims row 0 (c1), .spawn() returns fc-A, the pass DIES before recording; container A starts and claims the executing mark
            def die_after_spawn(r, plan, app, rs_, pools, smoke=None, fanout_claim=None):
                spawned.append(r["standup_id"])
                w["claims"].append(fanout_claim)
                won, _ = srv.fanout_execution_claim(led, fanout_claim["standup_id"], fanout_claim["nonce"], attempt_id="ta-A", call_id="fc-A")
                assert won
                raise RuntimeError("pass A died after .spawn() returned")
            with pytest.raises(RuntimeError, match="died after"):
                srv.run_fanout(draft, start_token="tok1", spawn=die_after_spawn, **fan)
            intent = lp.read_production_intent(w["intent_p"])
            assert intent["rows"][pt[0]]["claim"] == "c1" and intent["rows"][pt[0]]["call_id"] is None and lp.production_intent_claimed_unspawned(intent) == [pt[0]]
            assert w["claims"] == [{"standup_id": pt[0], "nonce": "c1"}]
            # pass B: the row is claimed with no call -> the outcome is UNKNOWN -> the container's mark names fc-A -> ADOPTED, no second spawn of row 0
            assert "fc-A" not in {c_["call_id"] for c_ in srv.wave_calls(rs, draft)}, "the repro: the mark exists, no local call record"
            out = srv.run_fanout(draft, start_token="tok1", spawn=w["spawn"], **fan)
            intent = lp.read_production_intent(w["intent_p"])
            assert out["adopted"] == [pt[0]] and out["spawned"] == pt[1:] and out["recovered"] == [] and intent["rows"][pt[0]]["call_id"] == "fc-A" == intent["rows"][pt[0]]["executing_call_id"]
            assert spawned.count(pt[0]) == 1 and [c_["nonce"] for c_ in w["claims"]] == ["c1", "c2", "c3", "c4"]
            # delta 4 (3): mark-before-record -> the WINNER is in calls.jsonl (what the steward / watcher read) the moment it is discovered
            recA = [c_ for c_ in srv.wave_calls(rs, draft) if c_["call_id"] == "fc-A"]
            assert len(recA) == 1 and recA[0]["standup_id"] == pt[0] and recA[0]["adopted_from_mark"] is None and recA[0]["executing_attempt"] == "ta-A" and recA[0]["fanout_claim"] == {"standup_id": pt[0], "nonce": "c1"}
            # (b) the mark not yet there (container A not started) and no call on record -> re-spawned under the SAME nonce; then BOTH containers race the
            # entry claim: exactly one executes, the other exits FANOUT_DUPLICATE with the winner's mark
            intent["rows"][pt[1]].update(call_id=None, spawned_utc=None, spawned_by=None, executing_call_id=None)      # as a death between spawn and record leaves it
            lp.write_production_intent(w["intent_p"], intent)
            out = srv.run_fanout(draft, start_token="tok1", spawn=w["spawn"], **fan)
            assert out["recovered"] == [pt[1]] and w["claims"][-1] == {"standup_id": pt[1], "nonce": "c2"}, "re-spawned under the ORIGINAL claim, never a new one"
            won1, h1 = srv.fanout_execution_claim(led, pt[1], "c2", attempt_id="ta-first", call_id="fc-first")
            won2, h2 = srv.fanout_execution_claim(led, pt[1], "c2", attempt_id="ta-dup", call_id="fc-dup")
            assert won1 and h1 is None and not won2 and h2["call_id"] == "fc-first" and h2["attempt_id"] == "ta-first"
            # the mark's call id (the FIRST container) is not the recorded (re-spawn's) -> reconciled: executing_call_id + a calls.jsonl record the watcher follows
            out = srv.run_fanout(draft, start_token="tok1", spawn=w["spawn"], **fan)
            intent = lp.read_production_intent(w["intent_p"])
            assert pt[1] in out["marks"]["observed"] and pt[1] in out["marks"]["adopted"] and intent["rows"][pt[1]]["executing_call_id"] == "fc-first"
            recs = [c_ for c_ in srv.wave_calls(rs, draft) if c_["standup_id"] == pt[1]]
            assert any(c_["call_id"] == "fc-first" and c_.get("adopted_from_mark") for c_ in recs)
            # the duplicate container's exit status is never a failure for the steward's retry policy, nor coverage
            calls = [{"call_id": "fc-dup", "standup_id": pt[1], "accepted_key": "k", "control_key": "c"}]
            tick = st.steward_tick(plan=draft, calls=calls, snapshot={"fc-dup": {"done": True, "summary": {"status": lp.FANOUT_DUPLICATE}, "accepted": None}}, holds={},
                                   state=st.StewardState(), required={"k": pt[1]})
            assert tick["invoke"] == [] and tick["stop"] is None
            # a second claim for a claimed row is refused by the intent
            with pytest.raises(ValueError, match="already claimed"):
                lp.production_intent_claim(intent, pt[1], "c9")
        finally:
            os.close(fd)
        # the container: the claim rides the spawn ctx and is enforced at entry before any standup (source pins)
        import inspect
        src = inspect.getsource(srv._run_standup_row)
        assert 'if ctx.get("fanout_claim"):' in src and "fanout_execution_claim(led" in src and "return fail(lp.FANOUT_DUPLICATE" in src
        assert src.index("fanout_execution_claim(led") < src.index("is_gsm8k_rung = is_rung and") and lp.fanout_executing_key("row", "n1") == "fanout|row|n1|executing"
        sp = inspect.getsource(srv.spawn_row)
        assert 'ctx["fanout_claim"] = {"standup_id"' in sp and '"fanout_claim": ctx.get("fanout_claim")' in sp
        rf = inspect.getsource(srv.run_fanout)
        assert rf.index("lp.production_intent_claim(intent, sid, nonce)") < rf.index("lp.write_production_intent(intent_p, intent)                 # the DURABLE claim, before .spawn()") < rf.index("cid = spawn(row, plan_d, app_name, rs, pools, fanout_claim=")
        # delta 4 (3): a LATE winner -- the recorded re-spawn exited FANOUT_DUPLICATE and the original container claims the mark only on the 5th poll -- is
        # adopted whenever it appears: the steward reconciles marks EVERY tick (in-process), never a bounded pass count
        intent = lp.read_production_intent(w["intent_p"])
        late = pt[2]
        intent["rows"][late].update(call_id="fc-dup", executing_call_id=None, claim="c-late")
        for sid_ in pt:
            if sid_ != late:
                intent["rows"][sid_]["executing_call_id"] = intent["rows"][sid_]["executing_call_id"] or intent["rows"][sid_]["call_id"]
        lp.write_production_intent(w["intent_p"], intent)
        assert st.marks_pending(w["intent_p"]) == [late] and not hasattr(st, "FANOUT_MARK_PASSES")
        loop_rs = tmp_path / "loop" / "RS"
        loop_rs.mkdir(parents=True)
        polls = {"n": 0}

        def late_marks():
            polls["n"] += 1
            if polls["n"] == 5:
                led.put(lp.fanout_executing_key(late, "c-late"), json.dumps({"attempt_id": "ta-late", "call_id": "fc-late"}), skip_if_exists=True)
            return srv.reconcile_fanout_marks_file(loop_rs, draft, w["intent_p"], led)
        rows = {r["standup_id"]: r for r in draft["standups"] + draft["rungs"]}

        def running(by):
            return {cid: {"done": False, "accepted": None} for cid in by.values()}

        def all_done(by):
            return {cid: {"done": True, "summary": {"status": "DONE"}, "accepted": f"ta-{cid}"} for cid in by.values()}
        rc, ev, n, required = self._loop(st, draft, tmp_path / "loop", [running] * 7 + [all_done], [lambda by: {}], start_token="tok1", intent_path=w["intent_p"], reconcile_marks=late_marks)
        marks = [json.loads(e.split(" ", 1)[1]) for e in ev if e.startswith("STEWARD_MARKS ")]
        assert rc == 0 and polls["n"] == 5 and len(marks) == 5 and [m["pending_before"] for m in marks] == [1] * 5 and marks[-1]["observed"] == [late] and marks[-1]["adopted"] == [late] and marks[-1]["pending_after"] == 0
        assert [m["n"] for m in marks] == [1, 2, 3, 4, 5], "one reconciliation per tick until the winner is observed (> 3 polls), then none"
        intent = lp.read_production_intent(w["intent_p"])
        assert intent["rows"][late]["executing_call_id"] == "fc-late" and intent["rows"][late]["call_id"] == "fc-dup"
        recs = [c_ for c_ in srv.wave_calls(loop_rs, draft) if c_["call_id"] == "fc-late"]
        assert len(recs) == 1 and recs[0]["adopted_from_mark"] == "fc-dup" and recs[0]["standup_id"] == late, "the late winner is in calls.jsonl, so the steward monitors / accepts / retries IT"
        assert "STEWARD_MARKS" in st.EVENTS

    def _loop(self, st, draft, tmp_path, snapshots, holds_seq, *, deadline_s=10_000, cert_ok=True, tick=100.0, **kw):
        """Drive `run_steward` with a fake backend whose snapshot / holds advance per poll; returns (rc, events, n_passes)."""
        rows = {r["standup_id"]: r for r in draft["standups"] + draft["rungs"]}
        required = st.required_keys_all_legs(draft)
        calls = [{"call_id": f"fc-{i}", "standup_id": sid, "accepted_key": r["accepted_key"], "control_key": r["control_key"], "wave_uuid": draft["wave_uuid"],
                  "launch_plan_sha": draft["launch_plan_sha"]} for i, (sid, r) in enumerate(sorted(rows.items()))]
        by_sid = {c["standup_id"]: c["call_id"] for c in calls}
        clock = {"t": 0.0}
        i = {"n": 0}

        class B:
            def snapshot(self, calls_):
                k = min(i["n"], len(snapshots) - 1)
                return snapshots[k](by_sid)

            def holds(self, calls_):
                k = min(i["n"], len(holds_seq) - 1)
                return holds_seq[k](by_sid)
        events, passes = [], []

        def sleep(s):
            clock["t"] += tick
            i["n"] += 1
        tmp_path.mkdir(parents=True, exist_ok=True)
        cert_p = tmp_path / "cert.json"
        fake_cert = {"schema": lp.CERTIFICATE_SCHEMA}
        cert_p.write_text(json.dumps(fake_cert))
        real = lp.go_preconditions
        try:
            lp.go_preconditions = (lambda plan, certificate, registry_sha256=None: [] if cert_ok else ["the seal does not match"])
            rc = st.run_steward(rs=tmp_path / "RS", plan=draft, certificate_path=cert_p, run_pass=lambda: passes.append(1) or 0, poll_s=60, deadline_s=deadline_s,
                                backend=B(), clock=lambda: clock["t"], sleep=sleep, load_calls=lambda: calls, emit=events.append, **kw)
        finally:
            lp.go_preconditions = real
        return rc, events, len(passes), required

    def test_the_loop_invokes_the_pass_once_per_failed_call_and_stops_on_wave_done(self, draft, tmp_path):
        st = self._st()
        rows = {r["standup_id"]: r for r in draft["standups"] + draft["rungs"]}
        sids = sorted(rows)
        bad = sids[0]

        def running(by):
            return {cid: {"done": False, "accepted": None} for cid in by.values()}

        def one_failed(by):
            s = running(by)
            s[by[bad]] = {"done": True, "summary": {"status": "LOST_CONTAINER"}, "error": None, "accepted": None}
            return s

        def all_done(by):
            return {cid: {"done": True, "summary": {"status": "DONE"}, "accepted": f"ta-{cid}"} for cid in by.values()}
        rc, ev, n, required = self._loop(st, draft, tmp_path, [running, one_failed, one_failed, one_failed, all_done], [lambda by: {}])
        assert rc == 0 and n == 1, ev                            # the same failed call across three polls -> ONE pass
        assert ev[0].startswith("STEWARD_STARTED ") and json.loads(ev[0].split(" ", 1)[1])["required_keys"] == len(required) == len(rows)
        passes = [json.loads(e.split(" ", 1)[1]) for e in ev if e.startswith("STEWARD_PASS ")]
        assert len(passes) == 1 and passes[0]["failed_calls"] == [[c for c in ["fc-0"]][0]] and passes[0]["rc"] == 0
        done = json.loads(ev[-1].split(" ", 1)[1])
        assert ev[-1].startswith("STEWARD_DONE ") and done["why"] == "WAVE_DONE" and done["required"] == done["accepted"] == len(rows) and done["passes"] == 1
        assert (tmp_path / "RS" / st.LOCK_FILE).is_file()

    def test_the_loop_stops_on_hold_deadline_and_a_failing_certificate_and_a_second_owner_exits(self, draft, tmp_path):
        st = self._st()
        rows = {r["standup_id"]: r for r in draft["standups"] + draft["rungs"]}
        sids = sorted(rows)
        held_sid = sids[1]

        def all_but_one(by):
            s = {cid: {"done": True, "summary": {"status": "DONE"}, "accepted": f"ta-{cid}"} for cid in by.values()}
            s[by[held_sid]] = {"done": True, "summary": {"status": "DRIVERS_FAILED"}, "error": None, "accepted": None}
            return s
        holds = lambda by: {held_sid: {"held": True, "hold_reason": "correctness", "reattempts": 0}}  # noqa: E731
        rc, ev, n, _ = self._loop(st, draft, tmp_path / "h", [all_but_one], [holds])
        assert rc == 0 and n == 0 and ev[-1].startswith("STEWARD_STOP ") and json.loads(ev[-1].split(" ", 1)[1]) == {"why": "HOLD", "held": [held_sid], "required": len(rows),
                                                                                                                       "accepted": len(rows) - 1, "passes": 0, "wave_uuid": draft["wave_uuid"]}
        # not held yet -> the pass is invoked ONCE for that failure, then the standup is held -> STOP HOLD, no second pass
        rc, ev, n, _ = self._loop(st, draft, tmp_path / "h2", [all_but_one], [lambda by: {}, lambda by: {}, holds])
        assert rc == 0 and n == 1 and ev[-1].startswith("STEWARD_STOP ") and '"why":"HOLD"' in ev[-1]
        # the deadline: nothing terminal ever -> STEWARD_TIMEOUT rc 3, bounded
        running = lambda by: {cid: {"done": False, "accepted": None} for cid in by.values()}  # noqa: E731
        rc, ev, n, _ = self._loop(st, draft, tmp_path / "d", [running], [lambda by: {}], deadline_s=350, tick=100.0)
        assert rc == 3 and n == 0 and ev[-1].startswith("STEWARD_TIMEOUT ") and json.loads(ev[-1].split(" ", 1)[1])["deadline_s"] == 350
        # the certificate stopped validating (a moved plan / an edited file) -> STEWARD_STOP certificate rc 2, no pass ever
        rc, ev, n, _ = self._loop(st, draft, tmp_path / "c", [all_but_one], [lambda by: {}], cert_ok=False)
        assert rc == 2 and n == 0 and '"why":"certificate"' in ev[-1] and "seal does not match" in ev[-1]
        # single owner: with the lock held by another process, a second steward exits at once (rc 0, STEWARD_ALREADY_OWNED)
        import fcntl
        rs2 = tmp_path / "own"
        rs2.mkdir()
        fd = os.open(rs2 / st.LOCK_FILE, os.O_RDWR | os.O_CREAT)
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        events = []
        try:
            rc = st.run_steward(rs=rs2, plan=draft, certificate_path=tmp_path / "none.json", run_pass=lambda: 0, backend=object(), emit=events.append, clock=lambda: 0.0, sleep=lambda s: None)
        finally:
            os.close(fd)
        assert rc == 0 and len(events) == 1 and events[0].startswith("STEWARD_ALREADY_OWNED ")
        # the pure tick: a duplicate-exit call is not a failure to pass on; an unknown key is ignored
        state = st.StewardState()
        required = st.required_keys_all_legs(draft)
        r0 = rows[sids[0]]
        calls = [{"call_id": "fc-a", "standup_id": sids[0], "accepted_key": r0["accepted_key"]}]
        t = st.steward_tick(plan=draft, calls=calls, snapshot={"fc-a": {"done": True, "summary": {"status": "REATTEMPT_DUPLICATE"}, "accepted": None}}, holds={}, state=state, required=required)
        assert t["stop"] is None and t["invoke"] == []     # delta 3 (4): a duplicate container's exit (REATTEMPT_DUPLICATE / FANOUT_DUPLICATE) is not the row's execution -- never a pass
        assert st.EVENTS == ("STEWARD_STARTED", "STEWARD_FANOUT", "STEWARD_MARKS", "STEWARD_PASS", "STEWARD_DONE", "STEWARD_STOP", "STEWARD_TIMEOUT", "STEWARD_ALREADY_OWNED")
        with pytest.raises(AssertionError):
            st.event("DONE", {})


# =========================================================================== plan_children() (generated) vs resolve_spawn() (what Popen receives)

class TestSolR3_7_generated_children_vs_resolved_spawn:
    """RE-VALIDATED at 1e28fe0f: `plan_children()` returned plan children with placeholder argv while the app resolved placeholders and added the
    registry / evidence / pool / attempt / routing / concurrency / bank-view state in `child_argv_env()` (and, for the rung consumer, in an
    inline substitution block) before `Popen` -- two resolution paths, neither the interface the generator presented. Now ONE shared
    `lp.resolve_spawn(child, SpawnContext)` returns the FINAL argv + env, the app calls it for EVERY child, `child_argv_env` is a keyword
    wrapper over it, and `plan_children()` is documented as the GENERATED view discovery reads from."""

    def test_the_app_resolves_every_child_through_resolve_spawn_and_refuses_a_leftover_placeholder(self, srv, draft, RR):
        import inspect
        src = inspect.getsource(srv._run_standup_row)
        assert "k_argv, k_env = lp.resolve_spawn(k, spawn_context)" in src and "spawn_context = lp.SpawnContext(" in src
        assert "child_argv_env(k," not in src and "subs = {lp.ATTEMPT_ID_PLACEHOLDER" not in src, "one resolution path, not two"
        assert 'return fail("CHILD_UNRESOLVABLE"' in src and src.index("lp.resolve_spawn(k, spawn_context)") < src.index("subprocess.Popen(k_argv")
        assert srv.concurrency_for is lp.concurrency_for and "lp.resolve_spawn(child, ctx)" in inspect.getsource(srv.child_argv_env)
        assert "GENERATED children" in lp.plan_children.__doc__ and "NOT what the" in lp.plan_children.__doc__ and "resolve_spawn" in lp.plan_children.__doc__
        # a GENERATED child carries placeholders; the RESOLVED one none -- and the wrapper equals the shared function
        sid, kid = lp.plan_children(draft, legs=("pasttense",), engines=("s1",))[0]
        assert any(lp.ATTEMPT_ROOT_PLACEHOLDER in a for a in kid["argv"]) and lp.unresolved_placeholders(kid["argv"])
        base = srv.container_env(bank_view="/b", tree="/t", thinking=False, base={"PATH": "/usr/bin"})
        ctx = lp.SpawnContext(attempt_root="/workspace/attempt", attempt_id="ta-1", standup_id=sid, plans_root="/vol/p", proxy_url="http://127.0.0.1:9/v1",
                              routing_table="/t.json", driver_python="/py", client_python="/cpy", S=45, ncells=8, bank_view="/b", base_env=base, call_id="fc-1", attempt_root_rel="seedrep/x/attempts/ta-1")
        argv, env = lp.resolve_spawn(kid, ctx)
        assert lp.unresolved_placeholders(argv) == [] and argv[0] == "/py" and argv[argv.index("--private-root") + 1] == "/workspace/attempt/exp/transcripts_results"
        assert env["SISTEER_REGISTRY_RUN_IDENTITY"] == f"/vol/p/pasttense/{kid['arm']}__s1.json" and env["SISTEER_REGISTRY_SPOOL"] == "/workspace/attempt/registry/records"
        assert env[lp.REQUEST_EVIDENCE_ENV] == "/workspace/attempt/exp/transcripts_results" and json.loads(env[lp.REQUEST_EVIDENCE_EXTRA_ENV])["attempt_id"] == "ta-1"
        assert (argv, env) == srv.child_argv_env(kid, attempt_root="/workspace/attempt", plans_root="/vol/p", proxy_url="http://127.0.0.1:9/v1", routing_table="/t.json",
                                                 driver_python="/py", client_python="/cpy", S=45, ncells=8, bank_view="/b", base_env=base, attempt_id="ta-1", standup_id=sid, call_id="fc-1",
                                                 attempt_root_rel="seedrep/x/attempts/ta-1")
        # known-bads: a placeholder that survives (never reaches Popen); a pool placeholder with no URL
        with pytest.raises(ValueError, match="survived resolution"):
            lp.resolve_spawn({**kid, "argv": list(kid["argv"]) + ["--x", "{never_resolved}"]}, ctx)
        mt = _row(draft, "crescendo")["children"][0]
        with pytest.raises(ValueError, match="needs pool"):
            lp.resolve_spawn(mt, ctx)
        argv_mt, _ = lp.resolve_spawn(mt, dataclasses_replace(ctx, pool_urls={"seedrep-attacker-0": "https://pool0"}))
        assert argv_mt[argv_mt.index("--attacker-endpoint") + 1] == "https://pool0" and lp.unresolved_placeholders(argv_mt) == []
        # the RUNG CONSUMER child through the same function: the RAW consumer argv's six placeholders resolved, the FINAL argv parses on the real parser
        rung = _row(draft, "pasttense", "s1", "8192")
        raw = srv.rung_consumer_child(rung, rung["rung_consumer"], attempt_root="/workspace/attempt", attempt_id="ta-1")
        assert set(lp.unresolved_placeholders(raw["argv"])) == {lp.ATTEMPT_ID_PLACEHOLDER, lp.CALL_ID_PLACEHOLDER, lp.PROXY_URL_PLACEHOLDER, lp.ATTEMPT_ROOT_REL_PLACEHOLDER,
                                                                lp.DRIVER_PY_PLACEHOLDER, lp.ATTEMPT_ROOT_PLACEHOLDER}
        rctx = dataclasses_replace(ctx, standup_id=rung["standup_id"], S=int(rung["geometry"]["S"]))
        r_argv, r_env = lp.resolve_spawn(raw, rctx)
        assert lp.unresolved_placeholders(r_argv) == [] and r_argv[0] == "/py"
        a = RR.build_parser().parse_args(r_argv[2:])
        assert a.cmd == "consume" and a.attempt_id == "ta-1" and a.call_id == "fc-1" and a.attempt_root == "seedrep/x/attempts/ta-1" and a.attempt_root_local == "/workspace/attempt"
        assert a.proxy_url == "http://127.0.0.1:9/v1" and a.width == int(rung["geometry"]["S"]) and a.wave_uuid == draft["wave_uuid"]
        assert r_env[lp.REQUEST_EVIDENCE_ENV] == "/workspace/attempt" and r_env["SISTEER_REGISTRY_SPOOL"] == "/workspace/attempt/registry/records" and "SISTEER_REGISTRY_RUN_IDENTITY" not in r_env
        assert lp.rung_child_env_problems(r_env, attempt_root="/workspace/attempt", attempt_id="ta-1", seed_bases=[1]) == []
        # the discovery surface is unchanged: every generated child, by leg / engine
        assert len(lp.plan_children(draft)) == sum(len(r["children"]) for r in draft["standups"] + draft["rungs"])
        assert lp.SPAWN_PLACEHOLDERS == (lp.ATTEMPT_ROOT_PLACEHOLDER, lp.PROXY_URL_PLACEHOLDER, lp.ROUTING_TABLE_PLACEHOLDER, lp.DRIVER_PY_PLACEHOLDER,
                                         lp.CONCURRENCY_PLACEHOLDER, lp.BANK_VIEW_PLACEHOLDER, lp.ATTEMPT_ID_PLACEHOLDER, lp.CALL_ID_PLACEHOLDER, lp.ATTEMPT_ROOT_REL_PLACEHOLDER,
                                         lp.LEDGER_URL_PLACEHOLDER, lp.CHILD_ROUTING_TABLE_PLACEHOLDER, lp.GSM8K_ATTEMPT_ID_PLACEHOLDER,
                                         lp.CLIENT_PY_PLACEHOLDER)   # + the GSM8K per-child ledger; the tinycap CLIENT venv python


def dataclasses_replace(obj, **kw):
    import dataclasses
    return dataclasses.replace(obj, **kw)


# =========================================================================== the GSM8K escalation rung is EXECUTABLE

class TestSolR3_6_gsm8k_escalation_rung:
    """RE-VALIDATED at 1e28fe0f: the four GSM8K rungs were DEFERRED from the executable plan, the ledger and the required keys while
    the spec's fixed battery escalates EVERY ladder leg incl. GSM8K at > 1 % at cap up to 16,384 -- and the GSM8K BASE children's argv
    (`--gsm8k-arm sampled --standup-endpoint … --controls-cap N`) was REFUSED by the driver's real parser (three flags it never had), so no GSM8K
    row would have run at all. Now: the `g8k_driver_argv` shape (identity v2, the cap stated AND set, one cell per out root, the G-FIT
    budget artifact) on every child; the four rung rows are executable through the queue-derived Inspect assembly
    (`scripts/modal/seedrep_gsm8k_assembly.py`: CLOSED queue + accepted source store -> inspect_rung.classify_store ->
    gsm8k_subset_from_queue -> rung_subset.json -> one Inspect child per escalated slot); 56 rungs, 134 / 134 (one H200 per pool), ten S5 windows."""

    def _asm(self):
        return _load_peer("seedrep_gsm8k_assembly_under_test", "scripts/modal/seedrep_gsm8k_assembly.py", REPO)

    def test_every_gsm8k_child_parses_on_the_real_driver_and_the_pre_fix_argv_does_not(self, draft):
        subs = {lp.child_key(k): {lp.LEDGER_URL_PLACEHOLDER: "http://127.0.0.1:7/v1", lp.CHILD_ROUTING_TABLE_PLACEHOLDER: f"/workspace/attempt/{k['ledger']['routing_table_rel']}",
                                  lp.GSM8K_ATTEMPT_ID_PLACEHOLDER: "a1"} for r in draft["standups"] + draft["rungs"] if r["leg"] == "gsm8k" for k in r["children"]}
        ctx = lp.SpawnContext(attempt_root="/workspace/attempt", attempt_id="ta-1", standup_id="x", plans_root="/vol/p", proxy_url="http://127.0.0.1:9/v1",
                              routing_table="/t.json", driver_python="/py", client_python="/cpy", S=21, ncells=8, bank_view="/b", base_env={}, child_subs=subs)   # client_python="/cpy" -- a GSM8K child resolves to the CLIENT venv python
        kids = [(r, k, lp.resolve_spawn(k, ctx)[0]) for r in draft["standups"] + draft["rungs"] if r["leg"] == "gsm8k" for k in r["children"]]
        # without the per-child ledger substitutions a GSM8K child never spawns (its {ledger_url} / {child_routing_table} / {gsm8k_attempt_id} survive)
        with pytest.raises(ValueError, match="survived resolution"):
            lp.resolve_spawn(kids[0][1], dataclasses_replace(ctx, child_subs=None))
        prefix = ["generate", "--proxy-url", "u", "--routing-table", "t", "--cell", "none", "--config", "c", "--defenses-dir", "d", "--out-root", "o", "--controls", "gsm8k"]
        pre_fix = prefix + ["--gsm8k-arm", "sampled", "--thinking", "on", "--sampling", "thinking_card", "--inspect-seed-policy", "replicate", "--inspect-seed-base", "1",
                            "--standup-endpoint", "e.json", "--concurrency", "31", "--controls-cap", "8192"]
        parsed = _driver_parse([argv[2:] for _, _, argv in kids] + [pre_fix], tree=_contract_tree())[:-1]
        assert len(kids) == 27 * 2                                 # 27 arm-runs (8 + 8 + 8 + 3 children) x (the base rows + the rung rows)
        for (r, k, argv), res in zip(kids, parsed):
            assert res["ok"], (k["arm"], res)
            a = res["ns"]
            assert a["tinycap_identity_version"] == 2 and a["thinking"] == "on" and a["sampling"] == "thinking_card" and a["samples"] == 1 and a["inspect_seed_policy"] == "replicate"
            assert a["on_cap_by_task"] == [f"gsm8k_paired200={k['cap']}"] == a["max_tokens_by_task"] and a["inspect_seed_base"] == k["s"] and a["controls"] == "gsm8k"
            # the operational contract on every child
            assert a["ledger_url"] == "http://127.0.0.1:7/v1" and a["attempt_id"] == "a1" and a["routing_table"].endswith(f"/exp/serve_gsm8k/{k['cell']}__s{k['s']}/routing_table.json")
            fitted = k["cell"] not in ("none",) + tuple(lp.PROMPTED_CELLS)
            assert a["allow_out_of_support_thinking"] is fitted and k["fitted"] is fitted and (k["oos_tuple"] is not None) is fitted and a["bank_view"] == lp.gsm8k_bank_view(r["engine"], k["s"]) == k["bank_view"]
            assert a["out_root"] == f"/workspace/attempt/exp/sampled/gsm8k_thinking_on_cap{k['cap']}__s{k['s']}/{k['cell']}" and a["oos_prompt_budget"].endswith("/oos_prompt_budget.json")
            if r["kind"] == "rung":
                assert a["item_subset"] == f"/workspace/attempt/exp/rung_subsets/gsm8k_cap16384__s{k['s']}/{k['cell']}/rung_subset.json" and k["cap"] == 16384 and k["source_cap"] == 8192
            else:
                assert a["item_subset"] == "" and k["cap"] == 8192
            if k["cell"] not in ("none",) + tuple(lp.PROMPTED_CELLS):
                assert len(a["expect_object_sha256"]) == 64 and a["expect_object_sha256"] == r["objects"][k["cell"]]["object_sha256"]
        # the pre-fix argv (1e28fe0f) is what the real parser REFUSES
        assert parsed[-1]["ok"] is False
        # the budget argv is the g8k_budget_artifact invocation, and a prompted cell carries --system-prompt-cell
        k = [c for c in _row(draft, "gsm8k")["children"] if c["cell"] == "prompted__si_constitution"][0]
        b = k["budget_argv"]
        assert b[1].endswith("/scripts/tinycap_oos_prompt_budget.py") and b[b.index("--task") + 1] == "gsm8k_paired200" and b[b.index("--arm") + 1] == "on"
        assert b[b.index("--cap") + 1] == "8192" and b[b.index("--geometry-eval") + 1] == lp.GSM8K_COMMON_EVAL and b[b.index("--system-prompt-cell") + 1] == "prompted__si_constitution"
        assert b[b.index("--tree-sha") + 1] == W and b[b.index("--out") + 1] == "{attempt_root}/" + k["budget_rel"]
        assert "--system-prompt-cell" not in [c for c in _row(draft, "gsm8k")["children"] if c["cell"] == "none"][0]["budget_argv"]
        B = _load_peer("tinycap_oos_prompt_budget_under_test", "scripts/tinycap_oos_prompt_budget.py", REPO)
        bp = B.build_parser().parse_args([x.replace("{attempt_root}", "/a") for x in b[2:]])
        assert bp.task == "gsm8k_paired200" and bp.arm == "on" and bp.cap == 8192 and bp.tree_sha == W and bp.system_prompt_cell == "prompted__si_constitution"
        # the sidecar rule (g8k_budget_artifact): fits / cap / entry / arm / tree
        good = {"fits": True, "cap": 8192, "geometry_eval": lp.GSM8K_COMMON_EVAL, "arm": "on", "tree_sha": W}
        assert lp.gsm8k_budget_problems(good, cap=8192, geometry_eval=lp.GSM8K_COMMON_EVAL, W=W) == []
        for bad, needle in (({**good, "fits": False, "why_not": "over"}, "fits=False"), ({**good, "cap": 16384}, "cap 16384"), ({**good, "tree_sha": "3" * 40}, "re-run G-FIT"),
                            ({**good, "arm": "off"}, "arm 'off'"), (None, "no .fit.json")):
            assert any(needle in p for p in lp.gsm8k_budget_problems(bad, cap=8192, geometry_eval=lp.GSM8K_COMMON_EVAL, W=W)), needle
        # the plan: 56 executable rungs, nothing deferred, the ten windows, the four GSM8K rung keys required
        led = draft["capacity_ledger"]
        assert len(draft["rungs"]) == 56 and led["unstaged_peak"] == {"containers": 134, "gpus": 134} and led["rungs_deferred"]["functions"] == 0 and led["gsm8k_rungs"]["functions"] == 4
        assert (24832, 21, True) in {(r["L"], r["S"], r["thinking"]) for r in draft["s5_roster"]} and len(draft["s5_roster"]) == 10
        assert len(lp.required_accepted_keys(draft, "gsm8k", "rungs")) == 4 and lp.validate_plan(draft) == []
        # known-bads the validator names: a child without identity v2; a rung row whose slots are not its children; the old flags
        bad = copy.deepcopy(draft)
        kid = _row(bad, "gsm8k")["children"][0]
        kid["argv"][kid["argv"].index("--tinycap-identity-version") + 1] = "1"
        assert any("identity version 2" in p for p in lp.validate_plan(bad))
        bad = copy.deepcopy(draft)
        _row(bad, "gsm8k", "s1", "16384")["rung_assembly"]["slots"].pop()
        assert any("assembly slots" in p for p in lp.validate_plan(bad))
        bad = copy.deepcopy(draft)
        _row(bad, "gsm8k")["children"][0]["argv"] += ["--gsm8k-arm", "sampled"]
        assert any("driver's parser does not know" in p for p in lp.validate_plan(bad))
        bad = copy.deepcopy(draft)
        r_ = _row(bad, "gsm8k")
        r_["children"][1]["argv"][r_["children"][1]["argv"].index("--out-root") + 1] = r_["children"][0]["argv"][r_["children"][0]["argv"].index("--out-root") + 1]
        assert any("two cell dirs" in p or "share an out root" in p or "seed-owned root" in p for p in lp.validate_plan(bad))

    def _queue(self, L, RR, draft, tmp_path, *, at_cap_ids, seed=1):
        """A synthetic CLOSED queue: the base producer (arm none, seed 1) of the s1 GSM8K row enqueues every at-cap outcome through the REAL
        Producer + gsm8k_queue_record, its store COMPLETE at THIS app's attempt root, finish_producer -> accepted -> the reconciler (with our
        producer-root spelling) writes CLOSED.json."""
        rung = _row(draft, "gsm8k", "s1", "16384")
        vol_root = tmp_path / "vol"
        vol = L.LocalVolume(vol_root)
        accepted, results = L.MemoryKV("accepted"), L.MemoryKV("results")
        spec = L.QueueSpec.build(draft["W7"], draft["wave_uuid"], "gsm8k", "s1", "gsm8k", 16384, rung["rung_assembly"]["feeder_shards"])
        ref = spec.producers[0]
        aid = "ta-base-0001"
        root_vol = lp.attempt_root_for(draft["results_root"], leg="gsm8k", view="s1", stage="base", shard=0, attempt_id=aid, kind="base")
        root_rel = root_vol[len("/vol/"):]
        local_root = vol_root / root_rel
        for slot in rung["rung_assembly"]["slots"]:
            (local_root / slot["source_out_rel"] / "cells" / slot["cell"]).mkdir(parents=True)
        prod = L.Producer(vol, spec, ref, aid, root_rel)
        for i, iid in enumerate(at_cap_ids):
            prod.enqueue(L.gsm8k_queue_record(spec, arm="none", s=seed, item={"item_id": iid, "sample_idx": 0, "stop_reason": "max_tokens", "at_cap": True, "seed": 1000 + i},
                                              base_store=f"{root_rel}/exp/sampled/gsm8k_thinking_on_cap8192__s1/none", producer=ref, producer_attempt_id=aid))
        vol.write_bytes(L.complete_rel(root_rel), b'{"attempt_id": "%s"}' % aid.encode())
        fin = L.finish_producer(prod, accepted)
        assert fin["state"] == "accepted"
        return dict(rung=rung, vol_root=vol_root, vol=vol, accepted=accepted, results=results, spec=spec, aid=aid, root_rel=root_rel, local_root=local_root)

    def _classify(self, IR, CL, *, at_cap_by_cell, n=200):
        """A synthetic `classify_store`: the REAL `inspect_rung.decide` (the > 1 % trigger, the transport check) over a synthetic population;
        the subset in the contract's exact shape (validated by contracts.validate_item_subset like the real adapter's)."""
        def classify(out_root, *, task=IR.GSM8K_TASK, **kw):
            cell = Path(out_root).name
            ids = list(at_cap_by_cell.get(cell, []))
            hops = {CL.HOP_DRIVER: 7200.0, CL.HOP_PROXY_UPSTREAM: 7200.0}
            exit_, next_cap, tv = IR.decide(n=n, n_at_cap=len(ids), eligible_ids=ids, cap=8192, hops=hops)
            record = {"exit": exit_, "n": n, "n_at_cap": len(ids), "share_at_cap": len(ids) / n, IR.RECORD_FIELD_ELIGIBLE_IDS: (ids if exit_ == CL.ESCALATE else []),
                      "item_facts": [{"item_id": i, "at_cap": True} for i in ids] + [{"item_id": f"ok{j}", "at_cap": False} for j in range(n - len(ids))]}
            subset = None
            if exit_ == CL.ESCALATE:
                subset = {"schema": "rung_subset/1", "task": task, "source_out_root": str(out_root), "source_identity_hash": "a1b2c3d4e5f60718", "source_cap": 8192,
                          "cap": int(next_cap), "rule": IR.GSM8K_INSTRUMENT.rule, "threshold": IR.GSM8K_TRIGGER_THRESHOLD, "ids": ids, "n_at_cap": len(ids),
                          "n_non_terminating": 0, "n_boundary": 0, "transport_check": tv.as_dict()}
            return record, subset
        return classify

    def test_the_over_one_percent_transition_assembles_the_subset_spawns_one_child_and_is_accepted(self, srv, L, RR, draft, tmp_path):
        from sisteer.capeval.tiers import cap_ladder as CL
        from sisteer.capeval.tiers import inspect_rung as IR
        from sisteer.capeval.tinycap import contracts as C
        A = self._asm()
        ids = ["q007", "q042", "q113"]                                   # 3 / 200 = 1.5 % > 1 % -> escalate (2 / 200 = 1 % does NOT, below)
        q = self._queue(L, RR, draft, tmp_path, at_cap_ids=ids)
        rung, spec = q["rung"], q["spec"]
        rung_root = tmp_path / "rung_attempt"
        # (1) the queue CLOSES through the app's producer-root spelling. An earlier default spelled the shard bare (`/s1/0/attempts/`) and never closed a
        # queue; the spelling in force is `shard<NN>` -- pinned here EQUAL to ours, so the reconciler
        # closes with or without our injection (and our injection stays the app's own spelling either way)
        assert L.attempt_root_rel(draft["W7"], "gsm8k", "s1", "0", q["aid"]) == q["root_rel"] and "/s1/shard00/attempts/" in q["root_rel"]
        assert lp.producer_attempt_root_of(draft["results_root"])(spec.producers[0], q["aid"]) == q["root_rel"] == L.producer_attempt_root(spec.producers[0], q["aid"])
        st_default = L.reconcile_queue(q["vol"], q["accepted"], spec)
        assert L.read_closed(q["vol"], spec) is not None and all(s["state"] == "CLOSED" for s in st_default["producers"].values()), st_default
        closed = A.wait_closed(L, q["vol"], q["accepted"], spec, results_root=draft["results_root"], deadline_s=1, poll_s=0, clock=lambda: 0.0, sleep=lambda s: None, log=lambda s: None)
        assert closed is not None and closed["closed"]["queue_root"] == spec.ladder_root and L.read_closed(q["vol"], spec) is not None
        # (2)-(4) the assembly: the escalated slot's subset written and validated; the other slots below the trigger / clean
        classify = self._classify(IR, CL, at_cap_by_cell={"none": ids, "prompted__safety_refusal": ["q001", "q002"]})
        verdict = A.assemble(L=L, RR=RR, vol_root=q["vol_root"], spec=spec, accepted_kv=q["accepted"], results_kv=q["results"], results_root=draft["results_root"],
                             attempt_root=rung_root, slots=rung["rung_assembly"]["slots"], snapshot_path=rung_root / "ladder/results_snapshot.json", classify_store=classify,
                             log=lambda s: None)
        assert verdict["ok"] is True and verdict["n_escalate"] == 1 and verdict["n_not_triggered"] == 7 and verdict["n_refused"] == 0 and verdict["trigger_threshold"] == 0.01
        by = {s["cell"]: s for s in verdict["slots"]}
        assert by["none"]["escalate"] is True and by["none"]["exit"] == CL.ESCALATE and by["none"]["n_queued"] == 3 == by["none"]["n_eligible"] and by["none"]["missing_from_queue"] == []
        assert by["none"]["source_attempt_id"] == q["aid"] and by["none"]["source_out_root"] == str(q["local_root"] / "exp/sampled/gsm8k_thinking_on_cap8192__s1/none")
        assert by["prompted__safety_refusal"]["escalate"] is False and by["prompted__safety_refusal"]["exit"] == CL.EXIT_BELOW_THRESHOLD, "2 / 200 = 1 % is not > 1 %"
        assert by["constv2-mix-full__lora_r4_ce"]["exit"] == CL.EXIT_CLEAN and (rung_root / "ladder/results_snapshot.json").is_file()
        subset_p = Path(by["none"]["subset_path"])
        assert subset_p == rung_root / "exp/rung_subsets/gsm8k_cap16384__s1/none/rung_subset.json"
        doc = C.validate_item_subset(json.loads(subset_p.read_text()))
        assert doc["ids"] == ids and doc["cap"] == 16384 and doc["source_cap"] == 8192 and doc["source_out_root"] == by["none"]["source_out_root"]
        # exactly the escalated slot's child spawns, with --item-subset naming THAT file, and parses on the real driver
        kids = srv.gsm8k_children_to_spawn(rung, verdict)
        assert [(k["arm"], k["s"], k["cell"]) for k in kids] == [("none", 1, "none")]
        ctx = lp.SpawnContext(attempt_root=str(rung_root), attempt_id="ta-rung-1", standup_id=rung["standup_id"], plans_root="/vol/p", proxy_url="http://127.0.0.1:9/v1",
                              routing_table="/t.json", driver_python="/py", client_python="/cpy", S=21, ncells=8, bank_view="/b", base_env={},
                              child_subs={lp.child_key(kids[0]): {lp.LEDGER_URL_PLACEHOLDER: "http://127.0.0.1:7/v1", lp.CHILD_ROUTING_TABLE_PLACEHOLDER: f"{rung_root}/{kids[0]['ledger']['routing_table_rel']}",
                                                                  lp.GSM8K_ATTEMPT_ID_PLACEHOLDER: "a1"}})
        argv, env = lp.resolve_spawn(kids[0], ctx)
        a = _driver_parse([argv[2:]], tree=_contract_tree())[0]
        assert a["ok"] and a["ns"]["item_subset"] == str(subset_p) and a["ns"]["on_cap_by_task"] == ["gsm8k_paired200=16384"] and a["ns"]["tinycap_identity_version"] == 2
        assert a["ns"]["out_root"] == f"{rung_root}/exp/sampled/gsm8k_thinking_on_cap16384__s1/none" and env[lp.REQUEST_EVIDENCE_ENV] == a["ns"]["out_root"]
        # through ACCEPTED output: the rung attempt completes like every other rung -- COMPLETE -> the ceiling's empty inventory -> ONE accepted put
        FakeDict.store.clear()
        acc = FakeDict("accepted")
        fin = srv.finish_attempt(root=rung_root, key=rung["accepted_key"], attempt_id="ta-rung-1", inventory=[], accepted=acc, commit=lambda: None,
                                 summary={"children": 1, "S": 21, "gsm8k_assembly": {"n_escalate": 1}})
        assert fin["state"] == srv.AttemptState.ACCEPTED and acc.get(rung["accepted_key"]) == "ta-rung-1" and (rung_root / "COMPLETE.json").is_file()
        required = lp.required_accepted_keys(draft, "gsm8k", "rungs")
        assert rung["accepted_key"] in required and len(required) == 4
        calls = [{"call_id": f"fc-{i}", "standup_id": sid, "leg": "gsm8k", "engine": sid.split("__")[1], "stage": "16384", "shard": 0, "kind": "rung", "control_key": k_,
                  "accepted_key": k_, "wave_uuid": draft["wave_uuid"]} for i, (k_, sid) in enumerate(sorted(required.items()))]
        snap = {c["call_id"]: {"done": True, "summary": {"status": "DONE"}, "error": None, "accepted": f"ta-{c['engine']}"} for c in calls}
        ev = sw.classify(now=1.0, leg="gsm8k", calls=calls, snapshot=snap, state=sw.WatchState(), stall_min=20, rows_flat_min=40, progress_min=30, required_keys=required, scope="rungs")
        assert any(e.startswith("WAVE_DONE ") for e in ev) and any(e == "DONE gsm8k scope=rungs required=4 accepted=4" for e in ev)
        # the container's source pins: the assembly precedes the children, the budget precedes every GSM8K child, a refusal fails the row by name
        import inspect
        src = inspect.getsource(srv._run_standup_row)
        assert src.index("run_gsm8k_assembly(row, ctx") < src.index("gsm8k_budget_prestep(kids") < src.index("subprocess.Popen(k_argv")
        assert 'return fail("GSM8K_ASSEMBLY_REFUSED"' in src and 'return fail("GSM8K_QUEUE_NOT_CLOSED"' in src and 'return fail("GSM8K_BUDGET_UNFIT"' in src
        assert "elif is_gsm8k_rung:" in src and src.count("elif is_gsm8k_rung:") == 2 and lp.RUNG_MODE_GSM8K_SUBSET in srv.EXECUTABLE_RUNG_MODES
        hargv = srv.gsm8k_assembly_argv(rung, {"results_root": draft["results_root"], "W": draft["W"], "wave_uuid": draft["wave_uuid"], "dicts": draft["dicts"]},
                                        tree=Path("/tree"), client_python="/cpy", root=rung_root, deadline_s=3600)
        assert hargv[:2] == ["/cpy", "/tree/scripts/modal/seedrep_gsm8k_assembly.py"]   # the CLIENT venv python
        assert hargv[hargv.index("--cap-next") + 1] == "16384"                            # restored -- an earlier re-pin's comment had swallowed these two
        assert hargv[hargv.index("--view") + 1] == "s1"
        assert json.loads(hargv[hargv.index("--slots") + 1]) == rung["rung_assembly"]["slots"] and hargv[hargv.index("--accepted-dict") + 1] == draft["dicts"]["accepted"]

    def test_known_bads_a_foreign_queued_item_refuses_an_open_queue_waits_and_the_cli_runs_on_json_kvs(self, srv, L, RR, draft, tmp_path):
        from sisteer.capeval.tiers import cap_ladder as CL
        from sisteer.capeval.tiers import inspect_rung as IR
        A = self._asm()
        # a queued item the ACCEPTED source never filed at cap: the seam refuses (a forged / foreign entry) -> the slot is refused, ok False
        q = self._queue(L, RR, draft, tmp_path / "foreign", at_cap_ids=["q007", "q042", "q113", "q999"])
        A.wait_closed(L, q["vol"], q["accepted"], q["spec"], results_root=draft["results_root"], deadline_s=1, poll_s=0, clock=lambda: 0.0, sleep=lambda s: None, log=lambda s: None)
        classify = self._classify(IR, CL, at_cap_by_cell={"none": ["q007", "q042", "q113"]})
        v = A.assemble(L=L, RR=RR, vol_root=q["vol_root"], spec=q["spec"], accepted_kv=q["accepted"], results_kv=q["results"], results_root=draft["results_root"],
                       attempt_root=tmp_path / "foreign/rung", slots=q["rung"]["rung_assembly"]["slots"], snapshot_path=tmp_path / "foreign/rung/snap.json", classify_store=classify,
                       log=lambda s: None)
        by = {s["cell"]: s for s in v["slots"]}
        assert v["ok"] is False and v["n_refused"] == 1 and "never filed at" in by["none"]["refused"] and by["none"]["subset_path"] is None
        assert srv.gsm8k_children_to_spawn(q["rung"], v) == []
        # an OPEN queue (no accepted producer yet): wait_closed returns None at its deadline; the CLI exits 3 and writes a not-closed verdict
        q2_root = tmp_path / "open"
        vol2 = L.LocalVolume(q2_root / "vol")
        spec2 = q["spec"]
        assert A.wait_closed(L, vol2, L.MemoryKV("acc2"), spec2, results_root=draft["results_root"], deadline_s=5, poll_s=1, clock=iter([0.0, 0.0, 3.0, 6.0, 9.0]).__next__, sleep=lambda s: None, log=lambda s: None) is None
        kvdir = q2_root / "kv"
        kvdir.mkdir(parents=True)
        out = q2_root / "verdict.json"
        rc = A.main(["--tree", str(WS2E if not (REPO / "scripts/modal/seedrep_ladder.py").is_file() else REPO), "--vol-root", str(q2_root / "vol"), "--results-root", draft["results_root"],
                     "--W7", draft["W7"], "--wave-uuid", draft["wave_uuid"], "--view", "s1", "--producers", "0", "--accepted-dict", draft["dicts"]["accepted"],
                     "--results-dict", draft["dicts"]["results"], "--attempt-root", str(q2_root / "rung"), "--slots", json.dumps(q["rung"]["rung_assembly"]["slots"]),
                     "--out", str(out), "--poll-s", "0", "--deadline-s", "0", "--memory-kv-dir", str(kvdir)])
        assert rc == A.EXIT_NOT_CLOSED and json.loads(out.read_text())["closed"] is False
        # the CLI over JSON KVs on a CLOSED queue: the accepted entry copied into the JSON KV, the verdict written, exit 0 with a 1.5 % slot escalated
        q3 = self._queue(L, RR, draft, tmp_path / "cli", at_cap_ids=["q007", "q042", "q113"])
        kv3 = tmp_path / "cli/kv"
        kv3.mkdir()
        (kv3 / f"{draft['dicts']['accepted']}.json").write_text(json.dumps(dict(q3["accepted"].items())))
        A.wait_closed(L, q3["vol"], q3["accepted"], q3["spec"], results_root=draft["results_root"], deadline_s=1, poll_s=0, clock=lambda: 0.0, sleep=lambda s: None, log=lambda s: None)
        out3 = tmp_path / "cli/verdict.json"
        real_classify = IR.classify_store
        try:
            IR.classify_store = self._classify(IR, CL, at_cap_by_cell={"none": ["q007", "q042", "q113"]})
            rc = A.main(["--tree", str(WS2E if not (REPO / "scripts/modal/seedrep_ladder.py").is_file() else REPO), "--vol-root", str(q3["vol_root"]), "--results-root", draft["results_root"],
                         "--W7", draft["W7"], "--wave-uuid", draft["wave_uuid"], "--view", "s1", "--producers", "0", "--accepted-dict", draft["dicts"]["accepted"],
                         "--results-dict", draft["dicts"]["results"], "--attempt-root", str(tmp_path / "cli/rung"), "--slots", json.dumps(q3["rung"]["rung_assembly"]["slots"]),
                         "--out", str(out3), "--poll-s", "0", "--deadline-s", "5", "--memory-kv-dir", str(kv3)])
        finally:
            IR.classify_store = real_classify
        v3 = json.loads(out3.read_text())
        assert rc == 0 and v3["ok"] is True and v3["closed"] is True and v3["n_escalate"] == 1 and (tmp_path / "cli/rung/exp/rung_subsets/gsm8k_cap16384__s1/none/rung_subset.json").is_file()

    def test_the_steward_reconciles_every_plan_queue_in_our_spelling(self, L, RR, draft, tmp_path):
        st = _load_peer("seedrep_steward_under_test", "scripts/modal/seedrep_steward.py", REPO)
        q = self._queue(L, RR, draft, tmp_path, at_cap_ids=["q007", "q042", "q113"])
        specs = st.plan_queue_specs(L, draft)
        assert len(specs) == len([x for x in draft["queues"] if x["cap_next"] != "petri-rerun"]) == 52 and all(s.kind in L.LADDERS for s in specs)
        state = st.StewardState()
        rec = st.reconcile_plan_queues(L, draft, vol=q["vol"], accepted_kv=q["accepted"], state=state, log=lambda s: None)
        assert q["spec"].ladder_root in rec["closed"] and L.read_closed(q["vol"], q["spec"]) is not None and q["spec"].ladder_root in state.closed_queues
        assert len(rec["open"]) == 51 and all(v.get("0") == "PRODUCER_MISSING" or "error" in v or any(x == "PRODUCER_MISSING" for x in v.values()) for v in rec["open"].values())
        rec2 = st.reconcile_plan_queues(L, draft, vol=q["vol"], accepted_kv=q["accepted"], state=state, log=lambda s: None)
        assert rec2["closed"] == [] and q["spec"].ladder_root not in rec2["open"], "a closed queue is never reconciled again"


# --------------------------------------------------------------------------------------------------------------------------------------------
# The GSM8K argv IS the producer's operational contract, tested through the driver's real parser + endpoint binding and the producer's
# store validators (never argparse alone). The oracle is the driver's own dry run:
#   G8K_DRY_RUN=1 G8K_DRY_S=31 bash scripts/sdfbase/gsm8k/gsm8k_thinking.sbatch qwen35_4b s1 constv2-mix-full__lora_r4_ce 8192 <sha> 1 0 qwen_seedrep
# --------------------------------------------------------------------------------------------------------------------------------------------
WS0_GSM8K_DRY_ROOT = "/tmp/g8k-dry/exp/gsm8k_paired200_paper_objects_qwen_seedrep"
WS0_GSM8K_DRY_SERVE = f"{WS0_GSM8K_DRY_ROOT}/serving/g8k_qwen35_4b_s1_constv2-mix-full--lora_r4_ce_8192_dry/gsm8k"
WS0_GSM8K_DRY_OUT = f"{WS0_GSM8K_DRY_ROOT}/controls_qwen_seedrep/qwen35_4b_s1/constv2-mix-full--lora_r4_ce/thinking_card"
WS0_GSM8K_DRIVER_LINE = (
    f"generate --proxy-url http://127.0.0.1:0/v1 --routing-table {WS0_GSM8K_DRY_SERVE}/routing_table.json --ledger-url http://127.0.0.1:0/v1 "
    "--cell constv2-mix-full__lora_r4_ce --config configs/sdfbase/gsm8k/sdfbase_full22_qwen35_4b_s1.gsm8k_constv2-mix-full--lora_r4_ce.yaml "
    f"--defenses-dir /tmp/g8k-dry/exp/objects/sdfbase/full22/qwen35_4b/cells_view/s1 --out-root {WS0_GSM8K_DRY_OUT} "
    "--controls gsm8k --thinking on --sampling thinking_card --samples 1 --max-tokens-by-task gsm8k_paired200=8192 --on-cap-by-task gsm8k_paired200=8192 "
    f"--inspect-seed-policy replicate --inspect-seed-base 1 --tinycap-identity-version 2 --oos-prompt-budget {WS0_GSM8K_DRY_OUT}/oos_prompt_budget.json "
    f"--concurrency 31 --vllm-log {WS0_GSM8K_DRY_SERVE}/vllm_server.log --attempt-id a1 --bank-view s1 --allow-out-of-support-thinking "
    "--expect-object-sha256 8be1e1a7ab36f3f65557f7835a30c6b8f18143c781fbf7fb312a5cc7a6e08871")
WS0_GSM8K_LEDGER_LINE = (f"-m sisteer.capeval.serving.usage_ledger --upstream http://127.0.0.1:0/v1 --port <free> --host 127.0.0.1 "
                         f"--ledger-path {WS0_GSM8K_DRY_OUT}/usage_ledger.jsonl --endpoint-json {WS0_GSM8K_DRY_SERVE}/ledger_endpoint.json --attempt-id a1")
#: the flags whose VALUES are portable between the reference cluster layout and ours (the rest are paths / URLs the container mints, compared by basename)
WS0_GSM8K_VALUE_FLAGS = ("--cell", "--config", "--controls", "--thinking", "--sampling", "--samples", "--max-tokens-by-task", "--on-cap-by-task",
                         "--inspect-seed-policy", "--inspect-seed-base", "--tinycap-identity-version", "--attempt-id")
WS0_GSM8K_BASENAME_FLAGS = ("--routing-table", "--oos-prompt-budget", "--vllm-log")


def _flag_pairs(tokens):
    """`[(flag, value | None)]` over an argv tail: a `--flag` followed by a non-flag token takes it as its value."""
    out = []
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t.startswith("--"):
            if i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
                out.append((t, tokens[i + 1]))
                i += 2
            else:
                out.append((t, None))
                i += 1
        else:
            i += 1
    return out


def _driver_call(code: str, payload):
    """Run ``code`` in a FRESH interpreter with `served_capability_run` imported as `R` and `payload` (JSON) on stdin; the code must json.dump its
    result to stdout (the driver refuses a second `bind_driver_globals` in one process, so the suite never imports it in-process)."""
    import subprocess
    pre = ("import json, sys\n"
           "sys.path.insert(0, sys.argv[1] + '/src'); sys.path.insert(0, sys.argv[1] + '/scripts')\n"
           "import served_capability_run as R\n"
           "payload = json.load(sys.stdin)\n")
    r = subprocess.run([sys.executable, "-c", pre + code, str(REPO)], input=json.dumps(payload), capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-3000:]
    return json.loads(r.stdout)


class _FakeLedgerPopen:
    """The usage ledger as `gsm8k_child_up` sees it: records its argv / env, writes the endpoint record (the real ledger writes `{ledger_url, upstream,
    ledger_path, pid, probe}` to --endpoint-json once its probe passed) -- or, as a known-bad, never writes it / exits at once / writes no url."""
    started: list = []
    mode = "ok"

    def __init__(self, argv, **kw):
        self.argv, self.kw = list(argv), kw
        self.pid = 4_000_000                                             # no such process group: gsm8k_child_down's killpg is a ProcessLookupError
        self.rc = 1 if self.mode == "dies" else None
        type(self).started.append(self)
        if self.mode in ("ok", "no_url"):
            ep = Path(argv[argv.index("--endpoint-json") + 1])
            ep.parent.mkdir(parents=True, exist_ok=True)
            rec = {"upstream": argv[argv.index("--upstream") + 1], "ledger_path": argv[argv.index("--ledger-path") + 1], "pid": self.pid, "probe": {"ok": True},
                   "attempt_id": argv[argv.index("--attempt-id") + 1]}
            if self.mode == "ok":
                rec["ledger_url"] = f"http://127.0.0.1:{argv[argv.index('--port') + 1]}/v1"
            ep.write_text(json.dumps(rec))

    def poll(self):
        return self.rc

    def wait(self, timeout=None):
        self.rc = 0 if self.rc is None else self.rc
        return self.rc


class TestSolR3Delta_4_gsm8k_argv_is_ws0s_operational_contract:
    """RE-VALIDATED at 10b07b28: `_gsm8k_children` emitted `--proxy-url --routing-table <the standup's table> --cell --config --defenses-dir --out-root
    --controls gsm8k --thinking on --sampling thinking_card --samples 1 --max-tokens-by-task --on-cap-by-task --inspect-seed-* --tinycap-identity-version 2
    --oos-prompt-budget --concurrency --vllm-log` and nothing of the operational contract: no `--ledger-url` (no usage ledger was ever started), no
    `--attempt-id` (the driver publishes `accepted_generation_attempt` / `resume_attempts[].attempt_id` only with it -- `producer_problem`
    refuses the store), no `--allow-out-of-support-thinking` on the 36 fitted thinking-ON children (the driver's preflight refuses them by
    `fitted_thinking_on_out_of_support`), no `--expect-object-sha256`. Now every child's argv is generated THROUGH the reference line and the
    container runs the per-store seam (ladder_state.json attempt id -> the child's routing-table copy -> its usage ledger -> the driver)."""

    @staticmethod
    def _child(draft, *, cell="constv2-mix-full__lora_r4_ce", s=1, engine="s1", stage="base"):
        row = [r for r in draft["standups"] + draft["rungs"] if r["leg"] == "gsm8k" and r["engine"] == engine and r["stage"] == stage][0]
        return row, [k for k in row["children"] if k["cell"] == cell and int(k["s"]) == s][0]

    @staticmethod
    def _ctx(root, child, *, ledger_url="http://127.0.0.1:0/v1", attempt_id="a1", proxy_url="http://127.0.0.1:0/v1", S=31, table=None):
        subs = {lp.LEDGER_URL_PLACEHOLDER: ledger_url, lp.CHILD_ROUTING_TABLE_PLACEHOLDER: table or f"{root}/{child['ledger']['routing_table_rel']}",
                lp.GSM8K_ATTEMPT_ID_PLACEHOLDER: attempt_id}
        return lp.SpawnContext(attempt_root=str(root), attempt_id="ta-1", standup_id="x", plans_root="/vol/p", proxy_url=proxy_url, routing_table="/t.json",
                               driver_python="/py", client_python="/cpy", S=S, ncells=8, bank_view="/b", base_env={}, child_subs={lp.child_key(child): subs})

    def test_every_fitted_child_is_ws0s_dry_run_line_flag_for_flag_and_the_parser_holds_the_one_omission(self, draft):
        row, k = self._child(draft)
        assert k["fitted"] is True and k["oos_tuple"] == lp.gsm8k_oos_tuple(k["cell"], 8192, engine="s1") == "gsm8k_paired200_qwen35_4b_native_8192" and len(k["object_sha256"]) == 64
        argv, _env = lp.resolve_spawn(k, self._ctx("/workspace/attempt", k))
        assert argv[0] == "/cpy" and argv[1].endswith("/scripts/served_capability_run.py") and argv[2] == "generate"   # the CLIENT venv python
        ours, ws0 = _flag_pairs(argv[2:]), _flag_pairs(WS0_GSM8K_DRIVER_LINE.split())
        # the flag SEQUENCE is the reference driver line's, flag for flag
        # (`--bank-view s<s>` included since 2126e918); every portable value equal; every path flag the same file
        assert [f for f, _ in ours] == [f for f, _ in ws0], (ours, ws0)
        ours_d, ws0_d = dict(ours), dict(ws0)
        for f in WS0_GSM8K_VALUE_FLAGS:
            assert ours_d[f] == ws0_d[f], (f, ours_d[f], ws0_d[f])
        for f in WS0_GSM8K_BASENAME_FLAGS:
            assert Path(ours_d[f]).name == Path(ws0_d[f]).name, (f, ours_d[f], ws0_d[f])
        assert ours_d["--allow-out-of-support-thinking"] is None is ws0_d["--allow-out-of-support-thinking"] and ours_d["--expect-object-sha256"] == k["object_sha256"]
        assert ours_d["--ledger-url"] == ws0_d["--ledger-url"] == "http://127.0.0.1:0/v1" and ours_d["--attempt-id"] == "a1" and ours_d["--concurrency"] == "31" == ws0_d["--concurrency"]
        assert ours_d["--routing-table"] == "/workspace/attempt/exp/serve_gsm8k/constv2-mix-full__lora_r4_ce__s1/routing_table.json" and ours_d["--defenses-dir"] == "/b"
        assert ours_d["--bank-view"] == "s1" == ws0_d["--bank-view"] == k["bank_view"] and k["bank_view_rule"] == lp.GSM8K_BANK_VIEW_RULE
        assert [f for f, _ in ours][-4:] == ["--attempt-id", "--bank-view", "--allow-out-of-support-thinking", "--expect-object-sha256"], "exactly where WS0's line puts it"
        # the driver's REAL parser: ours parses with every contract value bound, `--bank-view s1` admitted by the roster vocabulary; `s9` /
        # `main`-less shapes as the parser sees them. On the pre-merge tree the driver predated the seed-view fix and refused `s1` (the merge
        # requirement pinned then); the MERGED tree carries commit 978a7a97 (>= 218d66a7, SHIP), so THIS tree's driver IS the contract's:
        # `--bank-view s1` parses here, and the tree's admitted views are main / it + the three seed views
        tail = argv[2:]
        no_view = [x for i, x in enumerate(tail) if x != "--bank-view" and (i == 0 or tail[i - 1] != "--bank-view")]
        bad_view = [("s9" if (i > 0 and tail[i - 1] == "--bank-view") else x) for i, x in enumerate(tail)]
        here = _driver_parse([tail, no_view])
        assert here[0]["ok"] and here[0]["ns"]["bank_view"] == "s1" and here[1]["ok"] and here[1]["ns"]["bank_view"] is None, here[:2]
        assert here[-1]["has_admitted"] and {"main", "it", "s1", "s2", "s3"} <= set(here[-1]["bank_views"]), here[-1]
        assert lp.GSM8K_BANK_VIEW_WS0_FIX == WS0_BANK_VIEW_FIX == "218d66a7" and "218d66a7" in lp.GSM8K_BANK_VIEW_RULE
        parsed = _driver_parse([tail, WS0_GSM8K_DRIVER_LINE.split(), bad_view], tree=REPO)   # this tree carries the contract; the sibling worktree is no longer needed
        assert parsed[0]["ok"], parsed[0]
        ns = parsed[0]["ns"]
        assert ns["ledger_url"] == "http://127.0.0.1:0/v1" and ns["attempt_id"] == "a1" and ns["allow_out_of_support_thinking"] is True and ns["bank_view"] == "s1"
        assert ns["expect_object_sha256"] == k["object_sha256"] and ns["cell"] == k["cell"] and ns["thinking"] == "on" and ns["controls"] == "gsm8k"
        assert parsed[1]["ok"] and parsed[1]["ns"]["bank_view"] == "s1", "WS0's own line parses at their tip"
        assert not parsed[2]["ok"] and "s9" in parsed[2]["error"], ("a view the roster does not declare is refused at parse time", parsed[2])
        assert parsed[-1]["has_admitted"] and {"main", "it", "s1", "s2", "s3"} <= set(parsed[-1]["bank_views"]), parsed[-1]
        # the ledger line the container runs is the pinned LEDGER line (a free port in place of <free>; test_the_container_... holds the app's argv to it)
        led = [t for t in WS0_GSM8K_LEDGER_LINE.split()]
        assert led[:2] == ["-m", "sisteer.capeval.serving.usage_ledger"] and led[led.index("--attempt-id") + 1] == "a1" and Path(led[led.index("--endpoint-json") + 1]).name == lp.GSM8K_LEDGER_ENDPOINT_FILENAME
        # the bare cell and the prompted cells: the matrix has fitted=0 for them -> no acknowledgement, no pin, no tuple; every lane cell fitted
        kids = [k_ for r in draft["standups"] + draft["rungs"] if r["leg"] == "gsm8k" for k_ in r["children"]]
        assert len(kids) == 2 * 27 and sum(k_["fitted"] for k_ in kids) == 2 * (3 * 5 + 3), "the review's 36 of 54: 3 views x (none + 5 fitted + 2 prompted) + Arditi x 3 seeds, per stage"
        for k_ in kids:
            a = k_["argv"]
            if k_["cell"] == "none" or k_["cell"].startswith("prompted__"):
                assert not k_["fitted"] and "--allow-out-of-support-thinking" not in a and "--expect-object-sha256" not in a and k_["oos_tuple"] is None and k_["object_sha256"] is None
            else:
                assert k_["fitted"] and "--allow-out-of-support-thinking" in a and a[a.index("--expect-object-sha256") + 1] == k_["object_sha256"] and k_["oos_tuple"] is not None
            assert a[a.index("--ledger-url") + 1] == lp.LEDGER_URL_PLACEHOLDER and a[a.index("--attempt-id") + 1] == lp.GSM8K_ATTEMPT_ID_PLACEHOLDER
            want_bv = lp.ARDITI_BANK_VIEW if k_["cell"] == lp.ARDITI_CELL else f"s{k_['s']}"          # delta 3 (5): an Arditi child stamps its SOURCE bank
            assert a[a.index("--routing-table") + 1] == lp.CHILD_ROUTING_TABLE_PLACEHOLDER and a[a.index("--bank-view") + 1] == want_bv == k_["bank_view"]
            assert a[a.index("--attempt-id") + 2] == "--bank-view", "beside --attempt-id, as WS0's line puts it"
            cfg = a[a.index("--config") + 1]
            if k_["cell"] == lp.ARDITI_CELL:
                assert cfg == lp.ARDITI_CONFIG_REL and k_["arm"] == lp.ARDITI_ARM_ID
            else:
                assert cfg.startswith("configs/sdfbase/gsm8k/sdfbase_full22_qwen35_4b_s") and cfg.endswith(f".gsm8k_{lp.gsm8k_cell_slug(k_['cell'])}.yaml") and "__" not in cfg.rsplit("/", 1)[1]

    def test_the_live_ws0_dry_run_still_prints_the_pinned_line(self):
        """When the driver's worktree is present its dry run is RE-RUN and held to the pinned oracle: a contract change fails here by name (re-pin +
        re-derive), never silently."""
        import subprocess
        sbatch = WS0_TREE / "scripts/sdfbase/gsm8k/gsm8k_thinking.sbatch"
        if not sbatch.is_file():
            pytest.skip(f"WS0's worktree {WS0_TREE} is not on this machine; the pinned oracle stands")
        sha = subprocess.run(["git", "-C", str(WS0_TREE), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        env = {**os.environ, "G8K_DRY_RUN": "1", "G8K_DRY_S": "31"}
        r = subprocess.run(["bash", str(sbatch), "qwen35_4b", "s1", "constv2-mix-full__lora_r4_ce", "8192", sha, "1", "0", "qwen_seedrep"], cwd=str(WS0_TREE), env=env,
                           capture_output=True, text=True, timeout=300)
        lines = {ln.split(": ", 1)[0]: ln.split(": ", 1)[1] for ln in (r.stdout + r.stderr).splitlines() if ln.startswith(("DRIVER: ", "LEDGER: ", "ATTEMPT: "))}
        assert "DRIVER" in lines and "LEDGER" in lines, (r.stdout[-2000:], r.stderr[-2000:])
        live = lines["DRIVER"].split()
        assert live[:2] == ["/usr/bin/python3", "scripts/served_capability_run.py"] and live[2] == "generate"
        live_pairs, pinned = _flag_pairs(live[2:]), _flag_pairs(WS0_GSM8K_DRIVER_LINE.split())
        assert [f for f, _ in live_pairs] == [f for f, _ in pinned], "WS0's GSM8K driver line MOVED since a74d9c71 -- re-pin WS0_GSM8K_DRIVER_LINE and re-derive _gsm8k_children"
        for f in WS0_GSM8K_VALUE_FLAGS + ("--allow-out-of-support-thinking", "--bank-view", "--expect-object-sha256"):
            assert dict(live_pairs)[f] == dict(pinned)[f], f
        led = lines["LEDGER"].split()
        assert led[1:3] == ["-m", "sisteer.capeval.serving.usage_ledger"] and led[led.index("--attempt-id") + 1] == "a1" and led[led.index("--host") + 1] == "127.0.0.1"
        assert Path(led[led.index("--endpoint-json") + 1]).name == "ledger_endpoint.json" and Path(led[led.index("--ledger-path") + 1]).name == "usage_ledger.jsonl"
        assert lines["ATTEMPT"].startswith("a1 ")

    def test_the_drivers_real_endpoint_binding_admits_the_per_child_copy_and_refuses_without_the_ledger_record(self, srv, draft, tmp_path):
        """Eight children share ONE standup; the driver binds `endpoint.routing_table` to `--routing-table` and reads the ledger record BESIDE that
        table -- so each child gets its own copy of the standup record beside its own table + ledger record. Through `_bind_endpoint` for real."""
        _row, k = self._child(draft)
        serve = tmp_path / "serve"
        serve.mkdir()
        proxy = "http://127.0.0.1:9/v1"
        (serve / "routing_table.json").write_text(json.dumps({"routes": {"none": "http://127.0.0.1:8001/v1"}}))
        (serve / "endpoint.json").write_text(json.dumps({"proxy_url": proxy, "routing_table": str(serve / "routing_table.json"), "engine": "lora_native"}))
        root = tmp_path / "attempt"
        d = srv.gsm8k_child_serve_dir(root, k, serve_dir=serve)
        assert d == root / "exp/serve_gsm8k/constv2-mix-full__lora_r4_ce__s1" and (d / "routing_table.json").read_bytes() == (serve / "routing_table.json").read_bytes()
        ep = json.loads((d / "endpoint.json").read_text())
        assert ep["routing_table"] == str(d / "routing_table.json") and ep["proxy_url"] == proxy and ep["ledger_serve_copy_of"] == str(serve / "endpoint.json")
        code = ("out = []\n"
                "for c in payload:\n"
                "    try:\n"
                "        b = R._bind_endpoint(json.load(open(c['endpoint'])), proxy_url=c['proxy_url'], routing_table=c['routing_table'], ledger_url=c.get('ledger_url', ''))\n"
                "        out.append({'ok': True, 'binding': b})\n"
                "    except R.EndpointBindingMismatch as e:\n"
                "        out.append({'ok': False, 'error': str(e)})\n"
                "json.dump(out, sys.stdout)\n")
        copy_case = {"endpoint": str(d / "endpoint.json"), "proxy_url": proxy, "routing_table": str(d / "routing_table.json"), "ledger_url": "http://127.0.0.1:7/v1"}
        res = _driver_call(code, [copy_case, {**copy_case, "ledger_url": ""}])
        assert not res[0]["ok"] and "no ledger record" in res[0]["error"] and "ledger_endpoint.json" in res[0]["error"], res[0]
        assert res[1]["ok"] and res[1]["binding"]["ledger_url"] is None, "without --ledger-url the copy binds -- but WS0's contract REQUIRES the ledger"
        # the ledger record beside the copy (what the child's usage ledger writes) -> bound; the ORIGINAL record against the copy's table -> refused;
        # a record whose upstream is not this proxy / whose url is not the flag -> refused
        (d / "ledger_endpoint.json").write_text(json.dumps({"ledger_url": "http://127.0.0.1:7/v1", "upstream": proxy, "ledger_path": "x", "pid": 1, "probe": {}}))
        res = _driver_call(code, [copy_case, {**copy_case, "endpoint": str(serve / "endpoint.json")}, {**copy_case, "ledger_url": "http://127.0.0.1:8/v1"}])
        assert res[0]["ok"] and res[0]["binding"]["ledger_url"] == "http://127.0.0.1:7/v1" and res[0]["binding"]["ledger_upstream"] == proxy, res[0]
        assert not res[1]["ok"] and "written for another table" in res[1]["error"]
        assert not res[2]["ok"] and "ledger" in res[2]["error"].lower()
        (d / "ledger_endpoint.json").write_text(json.dumps({"ledger_url": "http://127.0.0.1:7/v1", "upstream": "http://127.0.0.1:99/v1", "ledger_path": "x", "pid": 1, "probe": {}}))
        res = _driver_call(code, [copy_case])
        assert not res[0]["ok"] and "upstream" in res[0]["error"], res[0]

    def test_the_container_starts_one_usage_ledger_per_store_with_the_attempt_id_and_fills_the_placeholders(self, srv, draft, tmp_path, monkeypatch):
        """`gsm8k_child_up`: the pinned LEDGER line (the attempt id minted into ladder_state.json), the record awaited, the three substitutions; the
        resolved argv parses on the real driver with those values; `gsm8k_child_down` ends the attempt. Known-bads: a ledger that never
        writes its record, one that dies, one whose record carries no url -> LEDGER_FAILED, the child never spawns."""
        from sisteer.capeval.serving import usage_ledger as UL
        _row, k = self._child(draft)
        serve = tmp_path / "serve"
        serve.mkdir()
        proxy = "http://127.0.0.1:9/v1"
        (serve / "routing_table.json").write_text(json.dumps({"routes": {}}))
        (serve / "endpoint.json").write_text(json.dumps({"proxy_url": proxy, "routing_table": str(serve / "routing_table.json")}))
        root = tmp_path / "attempt"
        killed = []
        monkeypatch.setattr(srv.os, "killpg", lambda pid, sig: killed.append((pid, sig)))
        _FakeLedgerPopen.started = []
        _FakeLedgerPopen.mode = "ok"
        st = srv.gsm8k_child_up(k, root=root, serve_dir=serve, proxy_url=proxy, driver_python="/py", tree=REPO, env={"PATH": "/usr/bin"}, popen=_FakeLedgerPopen, wait_s=5, sleep=lambda s: None)
        proc = _FakeLedgerPopen.started[-1]
        led = proc.argv
        want_led = ["/py", "-m", "sisteer.capeval.serving.usage_ledger", "--upstream", proxy, "--port", led[led.index("--port") + 1], "--host", "127.0.0.1",
                    "--ledger-path", str(root / k["ledger"]["ledger_path_rel"]), "--endpoint-json", str(root / k["ledger"]["endpoint_json_rel"]), "--attempt-id", "a1"]
        assert led == want_led and led[led.index("--port") + 1].isdigit() and 1024 <= int(led[led.index("--port") + 1]) < 65536
        assert [f for f, _ in _flag_pairs(led[3:])] == [f for f, _ in _flag_pairs(WS0_GSM8K_LEDGER_LINE.split()[2:])], "the ledger argv is WS0's LEDGER line"
        ns = UL.build_parser().parse_args(led[3:])
        assert ns.attempt_id == "a1" and ns.upstream == proxy and Path(ns.endpoint_json).name == lp.GSM8K_LEDGER_ENDPOINT_FILENAME and Path(ns.ledger_path).name == lp.GSM8K_LEDGER_FILENAME
        assert proc.kw["env"]["CUDA_VISIBLE_DEVICES"] == "" and proc.kw["env"]["PATH"] == "/usr/bin" and proc.kw["cwd"] == str(REPO) and proc.kw["start_new_session"] is True
        assert st["attempt_id"] == "a1" and st["ledger_url"] == f"http://127.0.0.1:{led[led.index('--port') + 1]}/v1"
        assert st["subs"] == {lp.LEDGER_URL_PLACEHOLDER: st["ledger_url"], lp.CHILD_ROUTING_TABLE_PLACEHOLDER: str(root / k["ledger"]["routing_table_rel"]), lp.GSM8K_ATTEMPT_ID_PLACEHOLDER: "a1"}
        ladder = json.loads((root / k["ledger"]["ladder_state_rel"]).read_text())
        assert ladder["schema"] == "g8k_ladder_state/1" and [a["attempt_id"] for a in ladder["attempts"]] == ["a1"] and ladder["attempts"][0]["ended"] is None and ladder["accepted_generation_attempt"] is None
        assert ladder["attempts"][0]["cell"] == k["cell"] and ladder["attempts"][0]["substrate"] == "modal"
        # the argv the child gets: the placeholders filled from THIS ledger; parses on the real driver with those values
        ctx = lp.SpawnContext(attempt_root=str(root), attempt_id="ta-1", standup_id="x", plans_root="/vol/p", proxy_url=proxy, routing_table=str(serve / "routing_table.json"),
                              driver_python="/py", client_python="/cpy", S=31, ncells=8, bank_view="/b", base_env={}, child_subs={lp.child_key(k): st["subs"]})
        argv, _env = lp.resolve_spawn(k, ctx)
        d = dict(_flag_pairs(argv[2:]))
        assert d["--ledger-url"] == st["ledger_url"] and d["--routing-table"] == str(root / k["ledger"]["routing_table_rel"]) and d["--attempt-id"] == "a1" and d["--proxy-url"] == proxy
        assert not any("{" in t for t in argv)
        parsed = _driver_parse([argv[2:]], tree=_contract_tree())
        assert parsed[0]["ok"] and parsed[0]["ns"]["ledger_url"] == st["ledger_url"] and parsed[0]["ns"]["attempt_id"] == "a1" and parsed[0]["ns"]["bank_view"] == "s1", parsed[0]
        # down: the ledger TERM'd (its process group), the attempt ended, accepted_generation_attempt = the driver's manifest's own when it names one
        srv.gsm8k_child_down(st, rc=0, result="OK")
        assert killed and killed[0][0] == proc.pid
        ladder = json.loads((root / k["ledger"]["ladder_state_rel"]).read_text())
        assert ladder["attempts"][0]["rc"] == 0 and ladder["attempts"][0]["result"] == "OK" and ladder["accepted_generation_attempt"] == "a1" and "launcher" in ladder["accepted_generation_attempt_source"]
        # a second attempt on the SAME store mints a2 (the launcher's counter),
        # its ledger with --attempt-id a2; the driver's manifest naming a2 is what is accepted
        st2 = srv.gsm8k_child_up(k, root=root, serve_dir=serve, proxy_url=proxy, driver_python="/py", tree=REPO, env={}, popen=_FakeLedgerPopen, wait_s=5, sleep=lambda s: None)
        assert st2["attempt_id"] == "a2" and _FakeLedgerPopen.started[-1].argv[-1] == "a2" and st2["subs"][lp.GSM8K_ATTEMPT_ID_PLACEHOLDER] == "a2"
        man = root / k["out_rel"] / "cells" / k["cell"] / "manifest.json"
        man.parent.mkdir(parents=True)
        man.write_text(json.dumps({"accepted_generation_attempt": "a2", "resume_attempts": [{"attempt_id": "a1"}, {"attempt_id": "a2"}]}))
        srv.gsm8k_child_down(st2, rc=0, result="OK")
        ladder = json.loads((root / k["ledger"]["ladder_state_rel"]).read_text())
        assert [a["attempt_id"] for a in ladder["attempts"]] == ["a1", "a2"] and ladder["accepted_generation_attempt"] == "a2" and ladder["accepted_generation_attempt_source"] == str(man)
        # known-bads: the ledger never writes its record / dies / writes no url -> LEDGER_FAILED, before any driver spawn
        for mode, why in (("silent", "no "), ("dies", "ledger rc 1"), ("no_url", "carries no ledger_url")):
            _FakeLedgerPopen.mode = mode
            killed.clear()
            with pytest.raises(RuntimeError, match=f"LEDGER_FAILED gsm8k__{k['arm']}__s1: .*{why}"):
                srv.gsm8k_child_up(k, root=tmp_path / f"attempt_{mode}", serve_dir=serve, proxy_url=proxy, driver_python="/py", tree=REPO, env={}, popen=_FakeLedgerPopen, wait_s=0.2, sleep=lambda s: None)
            # the ledger that never wrote a valid record is TERM'd (its process group), its log closed, and the minted attempt ENDED
            failed = _FakeLedgerPopen.started[-1]
            assert killed and killed[0][0] == failed.pid and failed.kw["stdout"].closed, mode
            ladder = json.loads((tmp_path / f"attempt_{mode}" / k["ledger"]["ladder_state_rel"]).read_text())
            assert [(a["attempt_id"], a["rc"], a["result"]) for a in ladder["attempts"]] == [("a1", 1, "LEDGER_FAILED")] and ladder["attempts"][0]["ended"] is not None
            assert ladder["accepted_generation_attempt"] is None
            _FakeLedgerPopen.mode = "ok"
            st_next = srv.gsm8k_child_up(k, root=tmp_path / f"attempt_{mode}", serve_dir=serve, proxy_url=proxy, driver_python="/py", tree=REPO, env={}, popen=_FakeLedgerPopen, wait_s=2, sleep=lambda s: None)
            assert st_next["attempt_id"] == "a2", "the next attempt on that store mints a2 against the closed a1"
            srv.gsm8k_child_down(st_next, rc=0, result="OK")
        _FakeLedgerPopen.mode = "ok"
        # the seam runs BEFORE resolve_spawn for a gsm8k row and a failure is the standup's GSM8K_LEDGER_FAILED (source pin on _run_standup_row)
        import inspect
        src = inspect.getsource(srv._run_standup_row)
        assert src.index('if row["leg"] == "gsm8k":') < src.index("gsm8k_child_up(k, root=root") < src.index("lp.resolve_spawn(k, spawn_context)") and 'fail("GSM8K_LEDGER_FAILED"' in src
        assert 'child_subs[lp.child_key(k)] = st["subs"]' in src and "gsm8k_child_down(gsm8k_states.pop(lp.child_key(k))" in src

    def test_ws0s_store_validators_and_the_drivers_attempt_rules_are_what_the_contract_serves(self, srv, draft, tmp_path):
        """`producer_problem` / `accepted_attempt` on a synthetic manifest: with the driver's `--attempt-id` publication admitted, without it
        refused by name -- the reason every child now carries the id; `check_attempt_id_arg` admits exactly what `gsm8k_attempt_begin` mints;
        `fitted_thinking_on_out_of_support` is non-None for every positions-all object -> every fitted child's acknowledgement."""
        from sisteer.capeval.tinycap import contracts as tc
        V = _load_peer("tinycap_gsm8k_validators_ws2d_delta", "scripts/tinycap_gsm8k_validators.py", REPO)
        entry = {"ts": "2026-09-18T00:00:00Z", "tree_sha": W, "attempt": 1, "completed": True, V.RESUME_ATTEMPT_ID_FIELD: "a1", V.RESUME_ATTEMPT_KIND_FIELD: V.ATTEMPT_KIND_GENERATION}
        good = {V.C.MANIFEST_KEY_PRODUCER: {"tree_sha": W, "driver": V.C.PRODUCER_DRIVER_SERVED}, V.C.MANIFEST_KEY_RESUME_ATTEMPTS: [entry], V.MANIFEST_KEY_ACCEPTED_ATTEMPT: "a1"}
        assert tuple(V.RESUME_ATTEMPT_REQUIRED_FIELDS) == ("ts", "tree_sha", "attempt", "completed") and V.MANIFEST_KEY_ACCEPTED_ATTEMPT == "accepted_generation_attempt"
        assert V.producer_problem(good, pin=W[:12]) is None and V.accepted_attempt(good)[V.RESUME_ATTEMPT_ID_FIELD] == "a1"
        # the pre-delta store (no --attempt-id): no accepted_generation_attempt, entries without an id -> the validators refuse by name
        no_id = {V.C.MANIFEST_KEY_PRODUCER: good[V.C.MANIFEST_KEY_PRODUCER], V.C.MANIFEST_KEY_RESUME_ATTEMPTS: [{kk: vv for kk, vv in entry.items() if kk not in (V.RESUME_ATTEMPT_ID_FIELD, V.RESUME_ATTEMPT_KIND_FIELD)}]}
        p_ = V.producer_problem(no_id, pin=W[:12])
        assert p_ is not None and "accepted_generation_attempt" in p_ and V.accepted_attempt(no_id) is None
        p_ = V.producer_problem({**no_id, V.MANIFEST_KEY_ACCEPTED_ATTEMPT: "a1"}, pin=W[:12])
        assert p_ is not None and V.RESUME_ATTEMPT_ID_FIELD in p_
        assert V.producer_problem({**good, V.MANIFEST_KEY_ACCEPTED_ATTEMPT: "a7"}, pin=W[:12]) is not None
        # the attempt id: exactly a<k>; the launcher's ladder_state counter mints a1, a2, a3 (`st_attempt_begin`); a Modal attempt id never passes
        assert tc.check_attempt_id_arg("a1") == "a1" == lp.GSM8K_FIRST_ATTEMPT_ID
        for bad in ("ta-1", "a0", "A1", "1", "", "attempt-1"):
            with pytest.raises(tc.AttemptScopeError):
                tc.check_attempt_id_arg(bad)
        out_root = tmp_path / "store"
        ids = [srv.gsm8k_attempt_begin(out_root, {"cell": "none"}) for _ in range(3)]
        assert ids == ["a1", "a2", "a3"] and all(tc.check_attempt_id_arg(i) == i for i in ids)
        assert srv.gsm8k_attempt_end(out_root, "a2", rc=1, result="DRIVER_FAILED")["accepted_generation_attempt"] is None
        assert srv.gsm8k_attempt_end(out_root, "a3", rc=0, result="OK")["accepted_generation_attempt"] == "a3"
        assert srv.gsm8k_attempt_end(tmp_path / "nowhere", "a1", rc=0, result="OK") == {}
        # the acknowledgement: a positions-all object under thinking-ON is out of support for EVERY family -> every fitted child carries the flag
        assert tc.fitted_thinking_on_out_of_support("qwen3_5", header_located=False) is not None
        kids = [k_ for r in draft["standups"] + draft["rungs"] if r["leg"] == "gsm8k" for k_ in r["children"]]
        assert sum("--allow-out-of-support-thinking" in k_["argv"] for k_ in kids) == sum(k_["fitted"] for k_ in kids) == 36 and len(kids) == 54   # 36 of 54 arrays
        assert all(("--allow-out-of-support-thinking" in k_["argv"]) is bool(k_["fitted"]) is (k_["object_sha256"] is not None) for k_ in kids)
        # this tree's parser predates commit 2126e918 (main | it only): the driver at W must carry the fix (pinned in test_every_fitted_child_...)
        assert tuple(tc.BANK_VIEWS) == ("main", "it")

    def test_known_bad_gsm8k_argv_shapes_are_refused_by_the_plan_validator(self, draft):
        """`validate_plan` refuses, by row + child: no ledger url / a literal one, no attempt id / a Modal one, the fitted acknowledgement missing, `--bank-view`
        present, the pre-fix flags, a config that is not the cell's overlay."""
        def mutate(fn, *, fitted=True):
            bad = copy.deepcopy(draft)
            row = [r for r in bad["standups"] if r["leg"] == "gsm8k" and r["engine"] == "s1"][0]
            k = [k_ for k_ in row["children"] if bool(k_["fitted"]) is fitted][0]
            fn(k)
            return row["standup_id"], k, lp.validate_plan(bad)
        def drop(k, flag, n=2):
            i = k["argv"].index(flag)
            del k["argv"][i:i + n]
        assert lp.validate_plan(draft) == []
        sid, k, probs = mutate(lambda k: drop(k, "--ledger-url"))
        assert any(sid in p and "--ledger-url" in p and f"s{k['s']}" in p for p in probs), probs
        sid, k, probs = mutate(lambda k: k["argv"].__setitem__(k["argv"].index("--ledger-url") + 1, "http://127.0.0.1:7/v1"))
        assert any(sid in p and "--ledger-url" in p for p in probs), probs
        sid, k, probs = mutate(lambda k: drop(k, "--attempt-id"))
        assert any(sid in p and "--attempt-id" in p for p in probs), probs
        sid, k, probs = mutate(lambda k: k["argv"].__setitem__(k["argv"].index("--attempt-id") + 1, "ta-1"))
        assert any(sid in p and "--attempt-id" in p for p in probs), probs
        sid, k, probs = mutate(lambda k: drop(k, "--allow-out-of-support-thinking", 1))
        assert any(sid in p and "--allow-out-of-support-thinking" in p for p in probs), probs
        sid, k, probs = mutate(lambda k: drop(k, "--expect-object-sha256"))
        assert any(sid in p and ("--expect-object-sha256" in p or "pin" in p) for p in probs), probs
        sid, k, probs = mutate(lambda k: k["argv"].extend(["--allow-out-of-support-thinking", "--expect-object-sha256", "f" * 64]), fitted=False)
        assert any(sid in p and "fitted acknowledgement / pin on a bare or prompted route" in p for p in probs), probs
        # commit 2126e918: --bank-view s<s> REQUIRED == the row's view: absent / `main` / another seed's view refused by name
        sid, k, probs = mutate(lambda k: drop(k, "--bank-view"))
        assert any(sid in p and "must carry --bank-view s1 (the row's view; got [])" in p for p in probs), probs
        sid, k, probs = mutate(lambda k: k["argv"].__setitem__(k["argv"].index("--bank-view") + 1, "main"))
        assert any(sid in p and "must carry --bank-view s1 (the row's view; got ['main'])" in p for p in probs), probs
        sid, k, probs = mutate(lambda k: k["argv"].__setitem__(k["argv"].index("--bank-view") + 1, "s2"))
        assert any(sid in p and "must carry --bank-view s1 (the row's view; got ['s2'])" in p for p in probs), probs
        sid, k, probs = mutate(lambda k: k.__setitem__("bank_view", "s2"))
        assert any(sid in p and "must carry --bank-view s1" in p for p in probs), probs
        # an Arditi child stamps the SOURCE bank ar_c0p3 (contracts.expected_bank_view_stamp, 218d66a7); `s<s>` on it is refused
        bad = copy.deepcopy(draft)
        arow = [r for r in bad["standups"] if r["leg"] == "gsm8k" and r["engine"] == lp.ARDITI][0]
        ak = arow["children"][0]
        assert ak["argv"][ak["argv"].index("--bank-view") + 1] == "ar_c0p3" == ak["bank_view"] == lp.gsm8k_bank_view(lp.ARDITI, ak["s"])
        ak["argv"][ak["argv"].index("--bank-view") + 1] = f"s{ak['s']}"
        probs = lp.validate_plan(bad)
        assert any(arow["standup_id"] in p and "must carry --bank-view ar_c0p3 (the source bank of the Arditi objects" in p for p in probs), probs
        # the rule is `contracts.expected_bank_view_stamp`, imported by path (this tree lacks the module); a stand-in, when injected, is used verbatim
        assert lp._ws0_bank_view_stamp("s2", "ar_c0p3") == "ar_c0p3" and lp._ws0_bank_view_stamp("s2", None) == "s2" and lp._ws0_bank_view_stamp("main", "ar_c0p3") == "main"
        seen = []
        prev = lp.BANK_VIEW_STAMP
        lp.BANK_VIEW_STAMP = lambda row: seen.append(dict(row)) or "STAND-IN"
        try:
            assert lp.gsm8k_bank_view(lp.ARDITI, 3) == "STAND-IN" and seen == [{"view": "s3", "bank_view": "ar_c0p3"}]
        finally:
            lp.BANK_VIEW_STAMP = prev
        sid, k, probs = mutate(lambda k: k["argv"].extend(["--gsm8k-arm", "s1"]))
        assert any(sid in p and "--gsm8k-arm" in p for p in probs), probs
        sid, k, probs = mutate(lambda k: k["argv"].__setitem__(k["argv"].index("--config") + 1, lp.GSM8K_OVERLAY_CONFIG_REL.format(model=lp.MODEL_KEY, view="s2", slug=lp.gsm8k_cell_slug(k["cell"]))))
        assert any(sid in p and "per-cell overlay config" in p and "_s2.gsm8k_" in p for p in probs), probs
        sid, k, probs = mutate(lambda k: k.__setitem__("ledger", None))
        assert any(sid in p and "ledger" in p for p in probs), probs
        sid, k, probs = mutate(lambda k: k["argv"].__setitem__(k["argv"].index("--routing-table") + 1, "/t.json"))
        assert any(sid in p and "--routing-table" in p for p in probs), probs
