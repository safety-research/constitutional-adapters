"""The served lens REDUCTION MODE reaches every attestation surface under ONE
key with ONE vocabulary — CPU-only (no vllm, no GPU, no network).

Stacked reductions inside a vectorized lens group are the served default
(2026-09-04); a downstream consumer reads WHICH mode a server serves under
the exact key ``lens_reduction_mode`` (``per_row`` | ``stacked``) and treats
an ABSENT key as ``per_row`` (a pre-landing server). So every surface built
from this tree must emit the key when the workers attest a mode, and must
leave it ABSENT — never guessed — when they do not. Covered:

- ``lens_worker.attested_reduction_mode``: the ONE strict reader over the
  steering attestation (agreement across workers, known vocabulary; every
  malformed shape is ``(None, why)``);
- the worker RPC ``sisteer_scheduler_config``: its extension class is
  GPU-only (defined only when vllm-lens imports), so its body is pinned by
  source and the helper it delegates to (``lens_adapters.reduction_mode``)
  is exercised directly, flipping with the import-time flag;
- ``capctrl_serving_up.Standup.endpoint()`` and ``bring_up``: the mode and
  its error read off the launch's attestation FILE; the proxy is handed the
  attested mode;
- ``lens_gate_standup``: the injectable strict reader (including the
  base-commit tree, whose lens_worker has no reader), the endpoint record
  carrying the attested mode + error + the raw env echo, the single
  attestation read placed BEFORE the proxy is built;
- ``proxy.RoutingProxy.models_payload()``: the identity block carries the
  key ONLY when the launch attestation carried it; ``ProxyClientEngine``
  surfaces it on ``proxy_identity`` (the consumer's read);
- the three surfaces agree on the key string and on ``la.REDUCTION_MODES``;
- ``lens_worker.endpoint_reduction_mode``: the CONSUMER's read back off a
  standup record — a named mode is that mode, NEITHER key is a pre-landing
  standup (``per_row``), and a null mode carrying a
  ``lens_reduction_mode_error`` REFUSES instead of falling back to the
  default, because a standup that could not determine what its own workers
  served must never have ``per_row`` recorded as its served identity;
- the served-capability driver's OWN copy of that read
  (``scripts/modal/capctrl_served.resolve_reduction_mode``), driven against a
  STAGED pre-landing tree first on ``sys.path``: the read site runs after the
  bring-up is paid for, so it may not import the rule from the tree it is
  serving — the historical neither-key record resolves to ``per_row`` there,
  a null-with-reason refuses inside the scope that tears the standup down,
  and the copy is pinned to the resolver case by case.

Synthetic data only: invented model ids, ``TESTHOST``, tmp paths.
"""

from __future__ import annotations

import contextlib
import dataclasses
import importlib.util
import inspect
import json
import subprocess
import sys
import types
from pathlib import Path

import pytest

from sisteer.capeval.serving import lens_adapters as la
from sisteer.capeval.serving import lens_worker as lw
from sisteer.capeval.serving import proxy as px
from sisteer.capeval.serving import vllm_server as vs
from sisteer.capeval.serving.vllm_server import BATCH_INVARIANT_ENV

REPO = Path(__file__).resolve().parents[1]
KEY = "lens_reduction_mode"
MODEL = "synthetic-org/synthetic-model"


def _load(name: str, rel: str):
    """Load a scripts/ module under a name private to this test module (the
    other standup tests register their own names; none is shared). Registered
    in ``sys.modules`` BEFORE exec: the standup's ``@dataclass`` under
    ``from __future__ import annotations`` resolves its string annotations
    through ``sys.modules[cls.__module__]``."""
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _report(mode=..., **over) -> dict:
    """One fake worker's scheduler-config report in the attestation's shape
    (the batch-invariant fields the mode rides next to; ``mode=None`` keeps
    the key with a None value, ``...`` omits it — an older tree's worker)."""
    rep = {"max_num_seqs": 8, "max_num_batched_tokens": 8 * 4096,
           "max_model_len": 4096, "enable_prefix_caching": False,
           "batch_invariant_env": None, "batch_invariant_active": False,
           "batch_invariant_surface": "vllm.model_executor.layers.batch_invariant",
           "batch_invariant_error": None}
    if mode is not ...:
        rep[KEY] = mode
    rep.update(over)
    return rep


def _install_report(mode=..., **over) -> dict:
    """One fake worker's INSTALL report (``sisteer_install_steering`` /
    ``lens_worker.install_report_payload``): the six validated fields the
    mode rides next to; ``...`` omits the key (an older tree's worker)."""
    rep = {"registration_count": 1, "layers": [3], "oproj_layers": [0, 2], "plan_layer": 0,
           "cells": ["none", "si_safety__psr_kl_all"], "ledger_cells": []}
    if mode is not ...:
        rep[KEY] = mode
    rep.update(over)
    return rep


def _att(*modes, reports=None, install_reports=None) -> dict:
    """A steering_attestation.json document with one scheduler-config worker
    report per mode (``scheduler_geometry.worker_reports`` — the shape the
    launcher writes) and, when given, the top-level install ``worker_reports``
    (``sisteer_install_steering``'s payloads, copied through by the launcher)."""
    reports = [_report(m) for m in modes] if reports is None else reports
    doc = {"registry_dir": "/tmp/synthetic/lens_registry",
           "scheduler_geometry": {"rpc_method": "sisteer_scheduler_config",
                                  "expected": {"max_num_seqs": 8, "max_model_len": 4096,
                                               "enable_prefix_caching": False},
                                  "n_workers": len(reports),
                                  "worker_reports": reports}}
    if install_reports is not None:
        doc["rpc_method"] = "sisteer_install_steering"
        doc["worker_reports"] = install_reports
    return doc


# --------------------------------------------------------------------------- #
# (a) lens_worker.attested_reduction_mode — the ONE strict reader              #
# --------------------------------------------------------------------------- #


class TestAttestedReductionMode:
    def test_agreeing_workers_yield_the_mode(self):
        assert lw.attested_reduction_mode(_att("stacked")) == ("stacked", None)
        assert lw.attested_reduction_mode(_att("stacked", "stacked", "stacked")) == ("stacked", None)
        assert lw.attested_reduction_mode(_att("per_row")) == ("per_row", None)
        assert lw.attested_reduction_mode(_att("per_row", "per_row")) == ("per_row", None)

    def test_missing_key_names_the_key_and_the_worker(self):
        mode, err = lw.attested_reduction_mode(_att(...))          # no key at all
        assert mode is None and KEY in err and "worker 0" in err
        mode, err = lw.attested_reduction_mode(_att("stacked", ...))  # one silent worker
        assert mode is None and KEY in err and "worker 1" in err

    def test_disagreement_is_none(self):
        mode, err = lw.attested_reduction_mode(_att("stacked", "per_row"))
        assert mode is None and "disagree" in err and "per_row" in err and "stacked" in err

    def test_unknown_value_is_none(self):
        for bad in ("rowwise", "STACKED", "", 1, True, None):
            mode, err = lw.attested_reduction_mode(_att(bad))
            assert mode is None and err, bad
            assert "not one of" in err or "disagree" in err, (bad, err)

    def test_not_a_mapping_is_none(self):
        for bad in ("not a dict", None, [], 3, [_report("stacked")]):
            mode, err = lw.attested_reduction_mode(bad)
            assert mode is None and "not a mapping" in err, bad

    def test_malformed_report_lists_are_none(self):
        assert lw.attested_reduction_mode({})[0] is None
        assert lw.attested_reduction_mode({"scheduler_geometry": {}})[0] is None
        mode, err = lw.attested_reduction_mode(_att(reports=[]))
        assert mode is None and "worker_reports" in err
        mode, err = lw.attested_reduction_mode(_att(reports=[_report("stacked"), "x"]))
        assert mode is None and "worker 1" in err and "not a mapping" in err
        mode, err = lw.attested_reduction_mode({"scheduler_geometry": {"worker_reports": "nope"}})
        assert mode is None

    def test_both_report_kinds_are_read_and_must_agree(self):
        """The install reports (top-level worker_reports) carry the key too
        (the same reduction_mode() value); the reader takes every report in
        BOTH lists and requires them all to agree — a consumer reading either
        list alone sees the same mode."""
        both = _att("stacked", "stacked", install_reports=[_install_report("stacked"), _install_report("stacked")])
        assert lw.attested_reduction_mode(both) == ("stacked", None)
        only_install = {"rpc_method": "sisteer_install_steering",
                        "worker_reports": [_install_report("per_row"), _install_report("per_row")]}
        assert lw.attested_reduction_mode(only_install) == ("per_row", None)
        mode, err = lw.attested_reduction_mode(
            _att("stacked", install_reports=[_install_report("per_row")]))
        assert mode is None and "disagree" in err and "'per_row'" in err and "'stacked'" in err
        mode, err = lw.attested_reduction_mode(_att("stacked", install_reports=[_install_report(...)]))
        assert mode is None and "install worker 0: no lens_reduction_mode" in err
        mode, err = lw.attested_reduction_mode(_att("stacked", install_reports=[_install_report("stacked"), "junk"]))
        assert mode is None and "install worker 1: report is not a mapping" in err

    def test_a_none_valued_key_is_not_a_mode(self):
        """A worker that CARRIES the key with None is not attesting a mode:
        refused (never read as 'absent -> per_row' by the reader)."""
        mode, err = lw.attested_reduction_mode(_att(None))
        assert mode is None and "None" in err


# --------------------------------------------------------------------------- #
# (b) the worker RPC and the helper it delegates to                            #
# --------------------------------------------------------------------------- #


class TestWorkerRpc:
    """``SisteerWorkerExtension`` is defined only when vllm-lens imports
    (GPU-only, ``# pragma: no cover``), so the RPC cannot be instantiated in
    the CPU venv. Its body is pinned by source, and the helper it delegates
    the key to — ``lens_adapters.reduction_mode()``, read at call time off
    the import-time flag — is exercised directly."""

    def _rpc_body(self) -> str:
        src = Path(lw.__file__).read_text(encoding="utf-8")
        start = src.index("def sisteer_scheduler_config(self)")
        end = src.index("def _sisteer_require(self)", start)
        return src[start:end]

    def test_rpc_stamps_the_mode_next_to_the_batch_invariant_fields(self):
        body = self._rpc_body()
        assert "out[LENS_REDUCTION_MODE_KEY] = la.reduction_mode()" in body
        # it rides in the SAME dict as the batch-invariant fields, after them
        assert body.index("out.update(batch_invariant_report())") \
            < body.index("out[LENS_REDUCTION_MODE_KEY] = la.reduction_mode()") \
            < body.index("return out")
        # the class itself is the GPU-only extension (the reason for the source pin)
        src = Path(lw.__file__).read_text(encoding="utf-8")
        assert "class SisteerWorkerExtension(_VLensExt):  # pragma: no cover" in src

    def test_rpc_reports_are_the_plugin_free_modules_functions(self):
        """The report functions the RPC body names moved VERBATIM into the
        plugin-free `worker_reports` module and `lens_worker` imports them
        back by name — so the source pin above still holds on the unchanged
        body, and the objects it resolves are the very functions the
        lora-native extension composes: one code path per attested fact, in
        both modes."""
        from sisteer.capeval.serving import worker_reports as wr

        assert lw.scheduler_config_report is wr.scheduler_config_report
        assert lw.kv_group_report is wr.kv_group_report
        assert lw.batch_invariant_report is wr.batch_invariant_report
        assert lw.LensWorkerError is wr.LensWorkerError

    def test_reduction_mode_flips_with_the_import_time_flag(self, monkeypatch):
        monkeypatch.setattr(la, "_STACKED_REDUCTIONS", False)
        assert la.reduction_mode() == "per_row" == la.REDUCTION_MODE_PER_ROW
        assert la.stacked_reductions_enabled() is False
        monkeypatch.setattr(la, "_STACKED_REDUCTIONS", True)
        assert la.reduction_mode() == "stacked" == la.REDUCTION_MODE_STACKED
        assert la.stacked_reductions_enabled() is True

    def test_the_rpc_value_round_trips_through_the_strict_reader(self, monkeypatch):
        """What the RPC stamps (la.reduction_mode()) is exactly what the
        attestation reader accepts back, in both modes."""
        for flag, want in ((False, "per_row"), (True, "stacked")):
            monkeypatch.setattr(la, "_STACKED_REDUCTIONS", flag)
            rep = _report(...)
            rep[lw.LENS_REDUCTION_MODE_KEY] = la.reduction_mode()   # the RPC's own line
            assert lw.attested_reduction_mode(_att(reports=[rep, dict(rep)])) == (want, None)

    def test_install_report_carries_the_same_mode_as_the_scheduler_report(self, monkeypatch):
        """``install_report_payload`` (the ``sisteer_install_steering`` return)
        stamps lens_reduction_mode from the same ``la.reduction_mode()`` the
        scheduler-config RPC uses; the frozen launcher's validator compares
        its six install fields and copies the extra key through untouched
        into the attestation's top-level worker_reports."""
        report = lw.InstallReport(registration_count=1, layers=(3,), plan_layer=0, cells=("none", "c"),
                                  ledger_cells=(), oproj_layers=(0, 2))
        for flag, want in ((True, "stacked"), (False, "per_row")):
            monkeypatch.setattr(la, "_STACKED_REDUCTIONS", flag)
            payload = lw.install_report_payload(report)
            assert payload[lw.LENS_REDUCTION_MODE_KEY] == want == la.reduction_mode()
            expected = {"registration_count": 1, "layers": [3], "oproj_layers": [0, 2], "plan_layer": 0,
                        "cells": ["c", "none"], "ledger_cells": []}
            # the production validator, verbatim: the extra key rides through
            assert vs.validate_install_reports([payload, dict(payload)], expected) == [payload, payload]
            # ...and both report kinds, stamped by the same source, agree in the reader
            sched = _report(...)
            sched[lw.LENS_REDUCTION_MODE_KEY] = la.reduction_mode()
            assert lw.attested_reduction_mode(_att(reports=[sched], install_reports=[payload])) == (want, None)

    def test_selection_record_carries_the_same_key(self):
        """The worker's evidence record names the arithmetic it was gathered
        under, with the same key (None for a record built outside a worker)."""
        assert lw.SelectionRecord(reduction_mode="stacked").as_dict()[lw.LENS_REDUCTION_MODE_KEY] == "stacked"
        assert lw.SelectionRecord(reduction_mode="per_row").as_dict()[lw.LENS_REDUCTION_MODE_KEY] == "per_row"
        assert lw.SelectionRecord().as_dict()[lw.LENS_REDUCTION_MODE_KEY] is None


# --------------------------------------------------------------------------- #
# (c) capctrl_serving_up: Standup.endpoint() + bring_up                       #
# --------------------------------------------------------------------------- #


@pytest.fixture
def capctrl():
    return _load("_reduction_mode_capctrl_serving_up", "scripts/capctrl_serving_up.py")


def _standup(up, out: Path):
    geom = up.resolve_standup_geometry(None, None, None)      # the legacy pair; no table lookups
    return up.Standup(out=out, model=MODEL, revision="0" * 40, cell="none", harness_cell="none",
                      geometry=geom, launch=None, server=None, proxy_loop=None,
                      proxy_url="http://127.0.0.1:1/v1", table_path=out / "routing_table.json",
                      tokenizer=None)


def _write_att(out: Path, doc) -> Path:
    p = out / vs.STEERING_ATTESTATION_FILE
    p.write_text(json.dumps(doc) if not isinstance(doc, str) else doc, encoding="utf-8")
    return p


class TestCapctrlStandup:
    @pytest.fixture(autouse=True)
    def _testhost(self, monkeypatch):
        import socket
        monkeypatch.setattr(socket, "gethostname", lambda: "TESTHOST")

    def test_endpoint_reads_the_attested_mode(self, capctrl, tmp_path):
        su = _standup(capctrl, tmp_path)
        assert su.attestation_path == tmp_path / vs.STEERING_ATTESTATION_FILE
        _write_att(tmp_path, _att("stacked", "stacked"))
        ep = su.endpoint("capability")
        assert ep[KEY] == "stacked" and ep["lens_reduction_mode_error"] is None
        _write_att(tmp_path, _att("per_row"))
        ep = su.endpoint()
        assert ep[KEY] == "per_row" and ep["lens_reduction_mode_error"] is None
        # the record kept everything it carried before
        assert ep["proxy_url"] == su.proxy_url and ep["node"] == "TESTHOST"
        assert ep["geometry"] == su.geometry.as_dict() and ep["eval"] is None
        assert ep["max_model_len"] == su.geometry.max_model_len

    def test_endpoint_without_an_attestation_is_none_with_a_reason(self, capctrl, tmp_path):
        su = _standup(capctrl, tmp_path)
        ep = su.endpoint()
        assert ep[KEY] is None
        assert "FileNotFoundError" in ep["lens_reduction_mode_error"]
        # unreadable JSON is a reason too, never a guess
        _write_att(tmp_path, "{not json")
        ep = su.endpoint()
        assert ep[KEY] is None and "JSONDecodeError" in ep["lens_reduction_mode_error"]

    def test_endpoint_with_disagreeing_workers_is_none(self, capctrl, tmp_path):
        su = _standup(capctrl, tmp_path)
        _write_att(tmp_path, _att("stacked", "per_row"))
        ep = su.endpoint()
        assert ep[KEY] is None and "disagree" in ep["lens_reduction_mode_error"]
        _write_att(tmp_path, _att(...))                          # an older tree's worker
        ep = su.endpoint()
        assert ep[KEY] is None and KEY in ep["lens_reduction_mode_error"]

    def test_file_reader_matches_the_document_reader(self, capctrl, tmp_path):
        for doc in (_att("stacked"), _att("per_row"), _att("stacked", "per_row"), _att(...), {}):
            p = _write_att(tmp_path, doc)
            assert capctrl.attested_reduction_mode_at(p) == lw.attested_reduction_mode(doc)
        mode, err = capctrl.attested_reduction_mode_at(tmp_path / "missing.json")
        assert mode is None and "FileNotFoundError" in err

    def test_bring_up_hands_the_attested_mode_to_the_proxy(self, capctrl):
        src = inspect.getsource(capctrl.bring_up)
        assert "attested_reduction_mode_at(out / vs.STEERING_ATTESTATION_FILE)" in src
        assert "launch_attestation=({lw.LENS_REDUCTION_MODE_KEY: mode}" in src
        assert "if mode is not None else None)" in src
        # read AFTER the server is up (the launcher writes the file at readiness)
        # and BEFORE the proxy is built
        assert src.index("vs.launch_server(") < src.index("attested_reduction_mode_at(") \
            < src.index("proxy = RoutingProxy(")
        assert "from sisteer.capeval.serving import lens_worker as lw" in src
        # the endpoint reads the SAME file, through the SAME reader
        ep_src = inspect.getsource(capctrl.Standup.endpoint)
        assert "attested_reduction_mode_at(self.attestation_path)" in ep_src
        assert '"lens_reduction_mode": mode' in ep_src and '"lens_reduction_mode_error": mode_error' in ep_src
        assert KEY in capctrl.__doc__ and "attested_reduction_mode_at" in capctrl.__doc__

    def test_the_sweep_still_goes_through_bring_up(self, capctrl):
        """The served decode sweep brings servers up through the same
        function; its call shape (keyword-only params) is unchanged."""
        params = inspect.signature(capctrl.bring_up).parameters
        assert all(p.kind is inspect.Parameter.KEYWORD_ONLY for p in params.values())
        assert {"repo", "config_path", "cell", "defenses_dir", "out", "geom"} <= set(params)
        fields = {f.name for f in dataclasses.fields(capctrl.Standup)}
        assert {"out", "launch", "server", "proxy_url", "table_path", "geometry"} <= fields


# --------------------------------------------------------------------------- #
# (d) lens_gate_standup: the injectable reader + the endpoint record           #
# --------------------------------------------------------------------------- #


@pytest.fixture
def helper():
    return _load("_reduction_mode_lens_gate_standup", "scripts/lens_gate_standup.py")


class TestGateStandupHelper:
    def test_reader_matches_lens_worker(self, helper):
        for doc in (_att("stacked"), _att("per_row"), _att("stacked", "per_row"), _att(...),
                    _att(reports=[]), {}, "x"):
            assert helper.reduction_mode_attested(doc) == lw.attested_reduction_mode(doc)
        assert helper.reduction_mode_attested(_att("stacked", "stacked")) == ("stacked", None)
        assert helper.reduction_mode_attested(_att("per_row")) == ("per_row", None)

    def test_base_commit_tree_without_a_reader_is_none_with_a_reason(self, helper):
        """The helper serves the reference standup from the BASE commit's
        tree too, whose lens_worker has no attested_reduction_mode: the mode
        is None with a reason (its workers attest none), never a guess."""
        old_tree_lw = types.SimpleNamespace()                    # no attested_reduction_mode
        mode, err = helper.reduction_mode_attested(_att("stacked"), lw=old_tree_lw)
        assert mode is None and "predates" in err and "attested_reduction_mode" in err
        # an injected reader is used verbatim
        fake = types.SimpleNamespace(attested_reduction_mode=lambda att: ("per_row", None))
        assert helper.reduction_mode_attested({}, lw=fake) == ("per_row", None)

    def test_key_literal_matches_lens_worker(self, helper):
        assert helper.LENS_REDUCTION_MODE_KEY == lw.LENS_REDUCTION_MODE_KEY == KEY

    def test_endpoint_records_the_attested_mode_and_the_raw_env_echo(self, helper):
        text = (REPO / "scripts" / "lens_gate_standup.py").read_text(encoding="utf-8")
        assert '"lens_reduction_mode": reduction_mode,' in text
        assert '"lens_reduction_mode_error": reduction_mode_error,' in text
        assert '"stacked_reductions_env": os.environ.get("SISTEER_LENS_STACKED_REDUCTIONS")' in text
        assert '"stacked_reductions":' not in text               # renamed: the env is provenance only
        assert "reduction_mode_attested(att)" in text
        # the module docstring names the field
        assert KEY in helper.__doc__ and "stacked_reductions_env" in helper.__doc__

    def test_attestation_is_read_once_before_the_proxy_and_handed_over(self, helper):
        text = (REPO / "scripts" / "lens_gate_standup.py").read_text(encoding="utf-8")
        read = "STEERING_ATTESTATION_FILE).read_text()"
        assert text.count(read) == 1
        assert text.index("vs.launch_server(") < text.index(read) < text.index("proxy = RoutingProxy(")
        assert "launch_attestation=({LENS_REDUCTION_MODE_KEY: reduction_mode}" in text
        assert "if reduction_mode is not None else None)" in text
        # a read exception lands as an error string with the mode cleared, never a silent None
        body = text[text.index("reduction_mode, reduction_mode_error = None, None"):
                    text.index('auth = out / "server_auth.json"')]
        assert 'reduction_mode, reduction_mode_error = None, f"{type(e).__name__}: {e}"' in body
        # the prefix-caching read the gate pins still lives in the same single read
        assert "prefix_caching_attested(att)" in body and "reduction_mode_attested(att)" in body


# --------------------------------------------------------------------------- #
# (e) the proxy's /v1/models identity block                                    #
# --------------------------------------------------------------------------- #


class _Backend:
    async def chat_completions(self, payload):
        return {"id": "stub", "object": "chat.completion", "choices": []}


class _TailRender:
    """Hermetic render probe: never loads a tokenizer (no hub access)."""
    header_ids = (901, 902, 903, 904)

    def render(self, messages, tools):
        return (1, 2, 3, *self.header_ids)


def _proxy(**kw):
    return px.RoutingProxy([], _Backend(), base_model=MODEL, serving_mode="interleaved",
                           agent_loop_render=_TailRender(), **kw)


class TestProxyIdentityBlock:
    def test_key_literal_matches_lens_worker(self):
        assert px.LENS_REDUCTION_MODE_KEY == lw.LENS_REDUCTION_MODE_KEY == KEY

    def test_attested_mode_is_echoed(self):
        meta = _proxy(launch_attestation={KEY: "stacked"}).models_payload()["petri_proxy"]
        assert meta[KEY] == "stacked"
        meta = _proxy(launch_attestation={KEY: "per_row"}).models_payload()["petri_proxy"]
        assert meta[KEY] == "per_row"
        # the rest of the identity block is intact
        assert meta["version"] == 1 and meta["serving_mode"] == "interleaved"
        assert meta["base_model"] == MODEL and meta["exact_replay_strategy"] is None
        assert meta["petri_agent_loop_entry"] is True
        assert meta["agent_loop_render"] == {"mode": "explicit", "loaded": True, "source": "_TailRender"}

    def test_unattested_is_absent_never_defaulted(self):
        for att in (None, {}, {BATCH_INVARIANT_ENV: "1"}, {"unrelated": "x"}):
            meta = _proxy(launch_attestation=att).models_payload()["petri_proxy"]
            assert KEY not in meta, att
            # the consumer's rule: absent -> per_row; the proxy never wrote it
            assert meta.get(KEY, "per_row") == "per_row"

    def test_the_mapping_is_copied_not_aliased(self):
        att = {KEY: "stacked"}
        proxy = _proxy(launch_attestation=att)
        att[KEY] = "per_row"
        assert proxy.models_payload()["petri_proxy"][KEY] == "stacked"

    def test_exact_replay_batch_invariant_attestation_still_works(self):
        """The pre-existing use of launch_attestation (the batch-invariant
        proof for exact_replay) is untouched, and the mode rides beside it."""
        proxy = px.RoutingProxy([], _Backend(), base_model=MODEL, serving_mode="exact_replay",
                                exact_replay_strategy="batch_invariant",
                                launch_attestation={BATCH_INVARIANT_ENV: "1", KEY: "stacked"},
                                agent_loop_render=_TailRender())
        meta = proxy.models_payload()["petri_proxy"]
        assert meta[KEY] == "stacked" and meta["exact_replay_strategy"] == "batch_invariant"
        with pytest.raises(px.ProxyError, match=BATCH_INVARIANT_ENV):
            px.RoutingProxy([], _Backend(), base_model=MODEL, serving_mode="exact_replay",
                            exact_replay_strategy="batch_invariant",
                            launch_attestation={KEY: "stacked"}, agent_loop_render=_TailRender())

    def test_client_engine_surfaces_it_on_proxy_identity(self, monkeypatch):
        """The consumer's read: ProxyClientEngine.load records the identity
        block verbatim, so the key is there exactly when the proxy wrote it."""
        def listing(meta):
            def fake_http(self, url, payload):
                assert url.endswith("/models") and payload is None
                return {"data": [{"id": "petri__none", "owned_by": "petri"}], "petri_proxy": meta}
            return fake_http
        eng = px.ProxyClientEngine("http://127.0.0.1:1", identity="proxy_vllm+lens+interleaved")
        attested = _proxy(launch_attestation={KEY: "stacked"}).models_payload()["petri_proxy"]
        monkeypatch.setattr(px.ProxyClientEngine, "_http_json", listing(attested))
        eng.load()
        assert eng.proxy_identity[KEY] == "stacked"
        silent = _proxy().models_payload()["petri_proxy"]
        monkeypatch.setattr(px.ProxyClientEngine, "_http_json", listing(silent))
        eng.load()
        assert KEY not in eng.proxy_identity and eng.proxy_identity.get(KEY, "per_row") == "per_row"

    def test_proxy_diff_is_additive_only(self):
        """The frozen Petri file gained lines and lost none (the landing's
        contract for proxy.py)."""
        import subprocess
        out = subprocess.run(["git", "-C", str(REPO), "diff", "--numstat", "--",
                              "src/sisteer/capeval/serving/proxy.py"],
                             capture_output=True, text=True)
        if not out.stdout.strip():
            pytest.skip("proxy.py has no uncommitted diff here (already landed)")
        added, deleted, _ = out.stdout.split("\t", 2)
        assert int(deleted) == 0 and int(added) > 0, out.stdout


# --------------------------------------------------------------------------- #
# (f) one key, one vocabulary, across the three surfaces                       #
# --------------------------------------------------------------------------- #


def test_one_key_one_vocabulary_across_surfaces(tmp_path, monkeypatch, helper, capctrl):
    import socket
    monkeypatch.setattr(socket, "gethostname", lambda: "TESTHOST")
    assert lw.LENS_REDUCTION_MODE_KEY == px.LENS_REDUCTION_MODE_KEY == helper.LENS_REDUCTION_MODE_KEY == KEY
    assert la.REDUCTION_MODES == ("stacked", "per_row")
    assert set(la.REDUCTION_MODES) == {la.REDUCTION_MODE_STACKED, la.REDUCTION_MODE_PER_ROW}
    assert la.reduction_mode() in la.REDUCTION_MODES
    su = _standup(capctrl, tmp_path)
    for mode in la.REDUCTION_MODES:
        doc = _att(mode, mode)
        # the worker's stamp -> the strict reader -> both standups' records -> the proxy
        assert lw.attested_reduction_mode(doc) == (mode, None)
        assert helper.reduction_mode_attested(doc) == (mode, None)
        _write_att(tmp_path, doc)
        ep = su.endpoint()
        assert ep[KEY] == mode and ep["lens_reduction_mode_error"] is None
        assert _proxy(launch_attestation={KEY: ep[KEY]}).models_payload()["petri_proxy"][KEY] == mode
    # a value outside the vocabulary never reaches a record as a mode
    for bad in ("vectorized", "rows", "STACKED"):
        assert lw.attested_reduction_mode(_att(bad))[0] is None
        assert helper.reduction_mode_attested(_att(bad))[0] is None
        _write_att(tmp_path, _att(bad))
        assert su.endpoint()[KEY] is None
    # the raw env echo and the attested value are DIFFERENT fields on the gate standup
    text = (REPO / "scripts" / "lens_gate_standup.py").read_text(encoding="utf-8")
    assert '"stacked_reductions_env"' in text and '"lens_reduction_mode"' in text


# --------------------------------------------------------------------------- #
# (g) the CONSUMER's read: lens_worker.endpoint_reduction_mode                 #
# --------------------------------------------------------------------------- #


class TestEndpointReductionModeIsResolvedNotDefaulted:
    """The three readings of a standup record, and only three: a NAMED mode is
    that mode; NEITHER key is a pre-landing standup (``per_row``); a null mode
    carrying a REASON refuses. The third is the one that has to exist — a
    post-landing standup always writes the key and writes null + a reason when
    its workers attest nothing or disagree, and ``record.get(key)`` cannot tell
    that apart from a pre-landing record's silence. Defaulting there would
    stamp ``per_row`` onto a store whose served arithmetic nobody could
    establish."""

    def test_a_named_mode_is_the_mode(self):
        for mode in la.REDUCTION_MODES:
            assert lw.endpoint_reduction_mode({KEY: mode}) == mode
            # a stale/blank reason beside a NAMED mode does not change the reading
            assert lw.endpoint_reduction_mode({KEY: mode,
                                               lw.LENS_REDUCTION_MODE_ERROR_KEY: None}) == mode

    def test_neither_key_is_a_pre_landing_standup_and_means_per_row(self):
        assert lw.endpoint_reduction_mode({}) == "per_row"
        # a record from a pre-landing tree carries the rest of the standup facts
        assert lw.endpoint_reduction_mode({"proxy_url": "http://127.0.0.1:1/v1",
                                           "node": "TESTHOST"}) == "per_row"
        # null with NO reason stated is still "nothing was said": pre-landing
        assert lw.endpoint_reduction_mode({KEY: None}) == "per_row"
        assert lw.endpoint_reduction_mode({KEY: None,
                                           lw.LENS_REDUCTION_MODE_ERROR_KEY: None}) == "per_row"

    def test_a_null_mode_with_a_reason_refuses_and_names_it(self):
        why = "worker reports disagree on lens_reduction_mode: ['per_row', 'stacked']"
        with pytest.raises(lw.ReductionModeUnattested) as ei:
            lw.endpoint_reduction_mode({KEY: None, lw.LENS_REDUCTION_MODE_ERROR_KEY: why},
                                       source="standup record /tmp/endpoint.json")
        msg = str(ei.value)
        assert why in msg                                  # the reason, verbatim
        assert "standup record /tmp/endpoint.json" in msg  # which record
        assert "not a pre-landing server" in msg and "per_row" in msg
        # the other reasons attested_reduction_mode can state refuse the same way
        silent = ("scheduler-config worker 0: no lens_reduction_mode in its "
                  "scheduler-config report")
        for other in (silent, "FileNotFoundError: no such file"):
            with pytest.raises(lw.ReductionModeUnattested, match="is null and"):
                lw.endpoint_reduction_mode({KEY: None,
                                            lw.LENS_REDUCTION_MODE_ERROR_KEY: other})

    def test_an_unknown_mode_and_an_unreadable_record_refuse(self):
        for bad in ("vectorized", "rows", "STACKED", "", 1, True):
            with pytest.raises(lw.ReductionModeUnattested, match="is not one of"):
                lw.endpoint_reduction_mode({KEY: bad})
        for not_a_record in (None, [], "endpoint.json", 3):
            with pytest.raises(lw.ReductionModeUnattested, match="not a mapping"):
                lw.endpoint_reduction_mode(not_a_record)

    def test_it_reads_the_records_capctrl_actually_writes(self, capctrl, tmp_path, monkeypatch):
        """End to end over the writer: agreeing workers -> the mode; disagreeing
        workers -> the record's null + reason -> a refusal naming that reason."""
        import socket
        monkeypatch.setattr(socket, "gethostname", lambda: "TESTHOST")
        su = _standup(capctrl, tmp_path)
        for mode in la.REDUCTION_MODES:
            _write_att(tmp_path, _att(mode, mode))
            assert lw.endpoint_reduction_mode(su.endpoint()) == mode
        _write_att(tmp_path, _att("per_row", "stacked"))          # the two kinds disagree
        ep = su.endpoint()
        assert ep[KEY] is None and ep[lw.LENS_REDUCTION_MODE_ERROR_KEY]
        with pytest.raises(lw.ReductionModeUnattested, match="disagree"):
            lw.endpoint_reduction_mode(ep)
        # ... and a record with no attestation file at all refuses too: the
        # standup HAS the keys, so its silence is a failure, not pre-landing
        (tmp_path / vs.STEERING_ATTESTATION_FILE).unlink()
        with pytest.raises(lw.ReductionModeUnattested, match="is null and"):
            lw.endpoint_reduction_mode(su.endpoint())

    def test_the_served_capability_driver_obeys_the_same_three_readings(self):
        """The x2 capability wave's driver (scripts/modal/capctrl_served.py) is
        the served path's endpoint reader on this tree, and it resolves the mode
        with a copy of this rule that ships WITH THE DRIVER — importing this
        function there would resolve against the STAGED candidate tree, which on
        a pre-landing sha has no such symbol. Section (h) drives that copy
        against a staged tree and pins it to this one case by case; this is the
        pointer, not a source-text stand-in for it."""
        assert TestDriverResolvesTheModeWithoutTheCandidateTree


# --------------------------------------------------------------------------- #
# (h) the DRIVER resolves the mode ITSELF — no candidate-tree import           #
# --------------------------------------------------------------------------- #

#: a staged serving tree from BEFORE the read-back resolver landed — the shape
#: of every pre-landing sha a wave may stage. Its lens_worker HAS the strict
#: attestation reader and has NO consumer read, so a driver that imported the
#: resolver out of the staged tree would die with ImportError right after
#: paying for the bring-up.
_PRE_LANDING_LENS_WORKER = '''\
"""Synthetic pre-landing lens_worker: no endpoint_reduction_mode here."""

LENS_REDUCTION_MODE_KEY = "lens_reduction_mode"


def attested_reduction_mode(attestation):
    return None, "synthetic pre-landing tree"
'''


@contextlib.contextmanager
def _staged_candidate_tree(root: Path, *, lens_worker_body: str = _PRE_LANDING_LENS_WORKER):
    """A staged candidate serving tree whose ``src`` is FIRST on ``sys.path``,
    exactly the way the driver's ``cell_geometry`` stages the tree that runs in
    the container. Every ``sisteer*`` module is unloaded for the duration and
    restored by identity afterwards, so an import inside resolves against the
    STAGED copy and no later test sees the fake."""
    src = root / "src"
    pkg = src / "sisteer" / "capeval" / "serving"
    pkg.mkdir(parents=True, exist_ok=True)
    for d in (src / "sisteer", src / "sisteer" / "capeval", pkg):
        (d / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "lens_worker.py").write_text(lens_worker_body, encoding="utf-8")
    saved_path, saved_mods = list(sys.path), {k: v for k, v in sys.modules.items()
                                              if k == "sisteer" or k.startswith("sisteer.")}
    for k in saved_mods:
        del sys.modules[k]
    sys.path.insert(0, str(src))
    try:
        yield src
    finally:
        sys.path[:] = saved_path
        for k in [k for k in sys.modules if k == "sisteer" or k.startswith("sisteer.")]:
            del sys.modules[k]
        sys.modules.update(saved_mods)


class _FakeStandup:
    """The standup subprocess as the read site uses it: poll / wait / kill,
    every call recorded. ``wait_timeout=True`` is a standup that outlives the
    teardown wait."""

    def __init__(self, wait_timeout: bool = False):
        self.calls: list = []
        self.wait_timeout = wait_timeout

    def poll(self):
        self.calls.append("poll")            # a live standup: no exit code yet

    def wait(self, timeout=None):
        self.calls.append(("wait", timeout))
        if self.wait_timeout:
            raise subprocess.TimeoutExpired(cmd="capctrl_serving_up.py", timeout=timeout)
        return 0

    def kill(self):
        self.calls.append("kill")


@pytest.fixture(scope="module")
def capdriver():
    """The served-capability driver (scripts/modal/capctrl_served.py) — the one
    consumer that reads a standup endpoint record and then spends a GPU. Modal
    is stubbed; nothing in this section touches Modal or a GPU."""
    from test_grokmt_served_resume import _stub_modal
    sys.modules.setdefault("modal", _stub_modal())
    return _load("_reduction_mode_capctrl_served", "scripts/modal/capctrl_served.py")


def _read_site(capdriver, endpoint, tmp_path, *, standup=None, commits=None, rec=None,
               inject_commit=True):
    """Drive the extracted read site once; returns ``(out, rec, stop, standup)``."""
    rec = {"model": "synthetic8b", "cell": "none"} if rec is None else rec
    stop = tmp_path / "STOP"
    standup = _FakeStandup() if standup is None else standup
    kw = {} if not inject_commit else {"commit": lambda: commits.append("commit")}
    out = capdriver._resolve_served_reduction_mode(
        endpoint, rec=rec, stop=stop, standup=standup,
        ep_path=tmp_path / "endpoint.json", **kw)
    return out, rec, stop, standup


class TestDriverResolvesTheModeWithoutTheCandidateTree:
    """The read site runs AFTER the H200 bring-up and BEFORE the teardown-scoped
    refusal block, so anything it imports from the STAGED candidate tree is a
    crash the cell cannot recover from: no status record, the stop file never
    touched, the standup left alive until Modal reaps the container. The
    resolution therefore ships with the driver, and the whole of it — including
    anything that could raise — sits inside the scope that tears the server
    down."""

    def test_a_pre_landing_staged_tree_cannot_supply_the_resolver(self, tmp_path):
        """Why the read site may not import it: the staged tree is what an
        import there resolves against, and a pre-landing sha has no symbol."""
        with _staged_candidate_tree(tmp_path / "sisteer-pre"):
            staged = importlib.import_module("sisteer.capeval.serving.lens_worker")
            assert str(tmp_path) in staged.__file__        # the STAGED copy, not the checkout
            assert not hasattr(staged, "endpoint_reduction_mode")
            assert not hasattr(staged, "ReductionModeUnattested")
            with pytest.raises(ImportError):
                from sisteer.capeval.serving.lens_worker import (  # noqa: F401
                    ReductionModeUnattested,
                    endpoint_reduction_mode,
                )

    def test_the_historical_record_resolves_to_per_row_on_a_pre_landing_tree(
            self, capdriver, tmp_path):
        """(i) the advertised historical path: a record stating NEITHER key is a
        pre-landing standup and means per_row — with the pre-landing tree staged
        first on sys.path, which is exactly when it happens."""
        commits: list = []
        with _staged_candidate_tree(tmp_path / "sisteer-pre"):
            out, rec, stop, standup = _read_site(
                capdriver, {"proxy_url": "http://127.0.0.1:1/v1", "node": "TESTHOST"},
                tmp_path, commits=commits)
        assert out is None                                  # the cell proceeds
        assert rec[KEY] == "per_row"
        assert rec["lens_reduction_mode_source"] == "pre-landing-default"
        assert "status" not in rec and "reduction_mode_error" not in rec
        assert not stop.exists() and standup.calls == [] and commits == []

    def test_a_named_mode_resolves_and_records_its_provenance(self, capdriver, tmp_path):
        for mode in la.REDUCTION_MODES:
            commits: list = []
            with _staged_candidate_tree(tmp_path / "sisteer-pre"):
                out, rec, stop, standup = _read_site(
                    capdriver, {KEY: mode, lw.LENS_REDUCTION_MODE_ERROR_KEY: None},
                    tmp_path, commits=commits)
            assert out is None and rec[KEY] == mode
            assert rec["lens_reduction_mode_source"] == "attested"
            assert not stop.exists() and standup.calls == [] and commits == []

    def test_a_null_with_a_reason_refuses_and_tears_the_standup_down(
            self, capdriver, tmp_path):
        """(ii) the refusal, with the same pre-landing tree staged: a typed
        record, the stop file touched, the standup waited on, the volume
        committed — the four things the crash used to skip."""
        why = "worker reports disagree on lens_reduction_mode: ['per_row', 'stacked']"
        commits: list = []
        with _staged_candidate_tree(tmp_path / "sisteer-pre"):
            out, rec, stop, standup = _read_site(
                capdriver, {KEY: None, lw.LENS_REDUCTION_MODE_ERROR_KEY: why},
                tmp_path, commits=commits)
        assert out is rec and rec["status"] == "REDUCTION_MODE_UNATTESTED"
        assert why in rec["reduction_mode_error"]
        assert str(tmp_path / "endpoint.json") in rec["reduction_mode_error"]
        assert KEY not in rec                               # no mode was recorded
        assert stop.exists()
        assert standup.calls == [("wait", capdriver.STANDUP_TEARDOWN_TIMEOUT_S)]
        assert commits == ["commit"]

    def test_a_standup_that_outlives_the_wait_is_killed(self, capdriver, tmp_path):
        commits: list = []
        out, rec, stop, standup = _read_site(
            capdriver, {KEY: None, lw.LENS_REDUCTION_MODE_ERROR_KEY: "no attestation file"},
            tmp_path, standup=_FakeStandup(wait_timeout=True), commits=commits)
        assert out is rec and rec["status"] == "REDUCTION_MODE_UNATTESTED"
        assert stop.exists() and commits == ["commit"]
        assert standup.calls == [("wait", capdriver.STANDUP_TEARDOWN_TIMEOUT_S), "kill"]

    def test_a_generic_exception_refuses_instead_of_crashing_the_cell(
            self, capdriver, tmp_path):
        """Anything the resolution can raise is inside the teardown scope — a
        crash there is the failure this whole section exists to prevent."""
        class _Exploding(dict):
            def get(self, *a, **k):
                raise RuntimeError("boom while reading the record")

        commits: list = []
        out, rec, stop, standup = _read_site(capdriver, _Exploding(), tmp_path, commits=commits)
        assert out is rec and rec["status"] == "REDUCTION_MODE_UNATTESTED"
        assert "RuntimeError" in rec["reduction_mode_error"]
        assert "boom while reading the record" in rec["reduction_mode_error"]
        assert stop.exists() and commits == ["commit"]
        assert standup.calls == [("wait", capdriver.STANDUP_TEARDOWN_TIMEOUT_S)]

    def test_the_commit_defaults_to_the_volume_and_needs_no_injection(
            self, capdriver, tmp_path):
        """The injected commit is a test seam, not the production path: with
        none given the function commits the volume itself."""
        out, rec, stop, _ = _read_site(
            capdriver, {KEY: None, lw.LENS_REDUCTION_MODE_ERROR_KEY: "unreadable attestation"},
            tmp_path, inject_commit=False)
        assert out is rec and rec["status"] == "REDUCTION_MODE_UNATTESTED" and stop.exists()

    def test_the_driver_copy_agrees_with_the_resolver_case_by_case(self, capdriver):
        """Parity pin: two copies of one rule, so they cannot drift silently.
        ``(mode, refused)`` from the driver equals ``(mode, refused)`` from
        ``lens_worker.endpoint_reduction_mode`` over the whole record
        vocabulary."""
        why = "scheduler-config worker 0: no lens_reduction_mode in its scheduler-config report"
        records = [{KEY: "per_row"}, {KEY: "stacked"},                      # named
                   {KEY: "per_row", lw.LENS_REDUCTION_MODE_ERROR_KEY: None},
                   {KEY: "stacked", lw.LENS_REDUCTION_MODE_ERROR_KEY: ""},
                   {}, {"proxy_url": "http://127.0.0.1:1/v1"},              # both keys absent
                   {KEY: None, lw.LENS_REDUCTION_MODE_ERROR_KEY: why},      # null + reason
                   {KEY: None, lw.LENS_REDUCTION_MODE_ERROR_KEY: ""},       # null + empty reason
                   {KEY: None},                                             # null, no error key
                   {KEY: None, lw.LENS_REDUCTION_MODE_ERROR_KEY: None},
                   {KEY: "vectorized"}, {KEY: "STACKED"}, {KEY: ""},        # unknown values
                   {KEY: 1}, {KEY: True},
                   None, [], "endpoint.json", 3]                            # not a record
        assert capdriver.REDUCTION_MODES == la.REDUCTION_MODES
        assert capdriver.LENS_REDUCTION_MODE_KEY == lw.LENS_REDUCTION_MODE_KEY
        assert (capdriver.LENS_REDUCTION_MODE_ERROR_KEY
                == lw.LENS_REDUCTION_MODE_ERROR_KEY)

        def read(fn, record):
            try:
                got = fn(record)
            except (lw.ReductionModeUnattested, capdriver.ReductionModeUnattested):
                return None, True
            return (got[0] if isinstance(got, tuple) else got), False

        for record in records:
            assert (read(capdriver.resolve_reduction_mode, record)
                    == read(lw.endpoint_reduction_mode, record)), record

    def test_it_agrees_on_the_records_the_writer_actually_writes(
            self, capdriver, capctrl, tmp_path, monkeypatch):
        """Over the real writer, not just synthetic dicts: agreeing workers ->
        the mode; the two report kinds disagreeing -> a refusal."""
        import socket
        monkeypatch.setattr(socket, "gethostname", lambda: "TESTHOST")
        su = _standup(capctrl, tmp_path)
        for mode in la.REDUCTION_MODES:
            _write_att(tmp_path, _att(mode, mode))
            assert capdriver.resolve_reduction_mode(su.endpoint()) == (mode, "attested")
        _write_att(tmp_path, _att("per_row", "stacked"))
        with pytest.raises(capdriver.ReductionModeUnattested, match="disagree"):
            capdriver.resolve_reduction_mode(su.endpoint())

    def test_the_read_site_delegates_and_imports_nothing_from_the_tree(self, capdriver):
        """Structural backstop to the executable pins above: between endpoint
        readiness and the teardown-scoped block the cell body imports nothing
        from the candidate tree — it calls the driver-shipped function."""
        src = inspect.getsource(capdriver._run_cell)
        assert "_resolve_served_reduction_mode(" in src
        assert "from sisteer" not in src and "import sisteer" not in src
        assert f'endpoint.get("{KEY}")' not in src        # never the raw key, either
        head, _, tail = src.partition("_resolve_served_reduction_mode(")
        assert "endpoint = json.loads(" in head           # after readiness
        assert "served_capability_run.py" in tail         # before the GPU is spent
