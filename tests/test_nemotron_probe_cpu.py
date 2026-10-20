"""``_nemotron_probe`` EXECUTED end to end on the CPU, with only the GPU-side seams faked (the
staged tree, the mamba_backend module, the two standup processes and their ``endpoint.json``,
the volume) and every reader REAL — the served-geometry reader, the §1.8 adjustment contract,
the engine reader, the certificate key derivation and the record shapes.

The documented lens case is ``--nemotron-probe --serving-path lens --geometry 4096``: the row's
4576 x 171 entry, whose container's KV cache backs 168.x full-length rows — the standup shrinks
to S = 168 (the shrink record), the certificate key stays at the REQUESTED 171
(``nemotron_key_geometry`` under lens is the resolved geometry, by design), and the same-GPU
peer stands up at the SERVED 168. The served-S equality the native key needs
(``expected_identity``'s pre-computed kernel step) ran for BOTH engines and returned
``SERVED_GEOMETRY_MISMATCH`` here; it is the native key's rule only now, and the native probe
keeps it (a native standup served at another S than the pre-computed one still refuses).

The folded LENS document is the pre-landing one — no ``extra`` block, no
engine, no path, no key geometry (it is filed at the UNCHANGED lens key, where an explicit lens rerun
overwrites the historical certificate); the native document carries the block. The byte-identity
against the base commit's own code is ``test_tinycap_lens_records_golden``; this module pins the
shapes."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from sisteer.capeval.serving import geometry as sg
from sisteer.capeval.tinycap import contracts as C

import test_tinycap_launcher as TL  # noqa: E402  (the modal stub, the by-path loader)

REPO = Path(__file__).resolve().parents[1]
NATIVE, LENS = "lora_native", "lens"
L_4576 = 4576
EVAL_4576 = next(k for k in sg.KNOWN_KEYS if sg.TABLE[k].max_model_len == L_4576)      # capability:ifeval_4096
NEMO = "nemotron3nano4b"


@pytest.fixture(scope="module")
def L():
    return TL._load_launcher(TL._stub_modal(), app_name=TL.TEST_APP_NAME, name="tinycap_served_nemotron_probe_under_test")


class _FakeMB:
    """The serving lane's ``mamba_backend`` module as the probe calls it, admitting everything: the
    contract loads, the packages import, the static / live halves are clean, the duty burst folds,
    the key API takes ``serving_engine`` (W-B's), the document is admitted. What the probe DECIDES
    from the served geometry is therefore the only thing that can refuse."""

    NEMOTRON_FAMILY = "nemotron_h"
    CONTRACT_RELPATH = "configs/capeval/mamba_backend_contract.json"

    def __init__(self):
        self.documents = []

    def load_contract(self, path):
        return json.loads(Path(path).read_text())

    def contract_problems(self, contract, *, wheel_sha256=None):
        return []

    def wheel_sha256_of(self, spec):
        return "w" * 64

    def contract_sha256(self, contract):
        return "c" * 64

    def packages_report(self):
        return {"ok": True}

    def endpoint_attestation_problems(self, endpoint, *, contract_sha256_expected):
        return []

    def static_attest(self, contract):
        return {"ok": True}

    def duty_burst(self, endpoint):
        return {"ok": True, "duty": 0.95, "tok_s": 1000.0}

    def duty_ratio(self):
        return 0.8

    def cache_key_inputs(self, *, serving_engine=None, **kw):
        return dict(kw, **({"serving_engine": serving_engine} if serving_engine is not None else {}))

    def attestation_cache_key(self, **inputs):
        return hashlib.sha256(json.dumps(inputs, sort_keys=True, default=str).encode()).hexdigest()

    def attestation_path(self, root, key):
        return Path(root) / key / "mamba_backend_attestation.json"

    def attestation_document(self, **kw):
        doc = {"admitted": True, **kw}
        self.documents.append(doc)
        return doc

    def readmission_problems(self, doc, contract, *, ratio, cache_key):
        return []


class _Proc:
    def __init__(self, *a, **k):
        self.rc = None

    def poll(self):
        return self.rc

    def wait(self, timeout=None):
        return 0

    def terminate(self):
        self.rc = 0

    def kill(self):
        self.rc = 0


def _shrink(configured: int, concurrency: float) -> dict:
    return sg.kv_shrink_record(configured_max_num_seqs=configured, max_model_len=L_4576, kv_backed_concurrency=concurrency,
                               kv_capacity_tokens=int(concurrency * L_4576), basis="uniform")


@pytest.fixture()
def probe_world(L, tmp_path, monkeypatch):
    """Everything ``_nemotron_probe`` touches outside its own logic, faked; ``run(serving_path,
    endpoints)`` executes the REAL probe body against the two scripted ``endpoint.json`` records and
    returns its REC plus the standup argv kwargs it composed."""
    row = L.C.model_row(L.NEMOTRON_PROBE_SLUG)
    assert row.family == "nemotron_h" and row.model_key == NEMO
    tree = tmp_path / "tree"
    for rel in (row.cfg, _FakeMB.CONTRACT_RELPATH, L.C.model_row(L.PEER_SLUG).cfg):
        (tree / rel).parent.mkdir(parents=True, exist_ok=True)
        (tree / rel).write_text("{}\n")
    mb = _FakeMB()
    geom = sg.resolve_geometry(EVAL_4576, model_key=NEMO)
    assert (geom.max_model_len, geom.max_num_seqs, geom.model_key) == (L_4576, 171, NEMO)
    cmds = []
    world = SimpleNamespace(row=row, geom=geom, mb=mb, cmds=cmds, endpoints={}, L=L)

    def _wait_for(ep: Path, proc, timeout_s, step=5.0):
        which = ep.parent.name.rsplit("_", 1)[-1]          # /tmp/srv_probe_<which>
        ep.parent.mkdir(parents=True, exist_ok=True)
        ep.write_text(json.dumps(world.endpoints[which]))
        return True

    def _standup_cmd(*a, **k):
        cmds.append(k)
        return ["true"]

    monkeypatch.setattr(L, "_mamba_backend", lambda: mb)
    monkeypatch.setattr(L, "resolve_staged_tree", lambda sha, repos_root=None: (tree, "f" * 40))
    monkeypatch.setattr(L, "probe_geometry_cfg", lambda row_, geometry, **kw: {"name": "probe", "cell": "none", "mode": "off",
                                                                                "controls": ["ifeval"], "model": row_.slug})
    monkeypatch.setattr(L, "served_geometry", lambda cfg, pass_no, tree_=None: (geom, {}))
    monkeypatch.setattr(L, "overlaid_config", lambda r_, tree_, out_dir: (str(Path(tree_) / r_.cfg), {}))
    monkeypatch.setattr(L, "_standup_cmd", _standup_cmd)
    monkeypatch.setattr(L.subprocess, "Popen", _Proc)
    monkeypatch.setattr(L, "_wait_for", _wait_for)
    monkeypatch.setattr(L, "_tear_down_standup", lambda stop, proc: None)
    monkeypatch.setattr(L, "_persist_logs", lambda srv, out: None)
    monkeypatch.setattr(L, "VOL", SimpleNamespace(commit=lambda: None))
    monkeypatch.setattr(L, "NEMO_BACKEND_ROOT", str(tmp_path / "nemo_backend"))
    for which in ("nemotron", "peer"):
        shutil.rmtree(f"/tmp/srv_probe_{which}", ignore_errors=True)

    def run(serving_path: str, endpoints: dict) -> dict:
        world.endpoints = dict(endpoints)
        cmds.clear()
        try:
            return L._nemotron_probe("f6cf462", "4096", "stacked", "card", row=row, serving_path=serving_path)
        finally:
            for which in ("nemotron", "peer"):
                shutil.rmtree(f"/tmp/srv_probe_{which}", ignore_errors=True)
    world.run = run
    return world


def test_the_lens_probe_admits_the_kv_shrunk_171_to_168_standup_and_keys_the_certificate_at_the_requested_s(probe_world):
    """The documented lens case: Nemotron shrinks 171 -> 168, the peer stands up at the SERVED 168, the key
    stays at the REQUESTED 171 — the probe is ADMITTED where it once returned SERVED_GEOMETRY_MISMATCH."""
    w = probe_world
    rec = _shrink(171, 168.4)
    nemotron = {"geometry": {"max_model_len": L_4576, "max_num_seqs": 168, sg.KV_SHRINK_KEY: rec},
                "mamba_backend_attestation": {"static": {"ok": True}, "live": {"ok": True, "reports": [], "expected_n_mamba_layers": 1}}}
    peer = {"geometry": {"max_model_len": L_4576, "max_num_seqs": 168}}
    out = w.run(LENS, {"nemotron": nemotron, "peer": peer})
    assert out["status"] == "OK", out.get("error")
    assert out["admitted"] is True and out["serving_path"] == LENS and out[C.SERVED_ENGINE_FIELD] == LENS
    assert out["key_geometry"] == {"max_model_len": L_4576, "max_num_seqs": 171, "requested_max_num_seqs": 171}
    assert out["peer_geometry"]["max_num_seqs"] == 168 and out["peer_geometry"]["requested_max_num_seqs"] == 171
    assert out["peer_geometry"]["kv_shrink"] == rec
    assert [(s["model"], s["S"]) for s in out["standups"]] == [(w.row.slug, 171), (w.L.C.model_row(w.L.PEER_SLUG).slug, 168)]
    # both standups on the lens path, no adapter bank; the folded LENS document carries NO additive block — it is
    # filed at the unchanged lens key, where a rerun overwrites the historical certificate, so it folds exactly the
    # pre-landing inputs (byte-held to main fafbec6c by test_tinycap_lens_records_golden);
    # the REC alone (a printed result) says which path the probe ran and what S the key was derived at
    assert [(k["serving_path"], k["adapter_bank"]) for k in w.cmds] == [(LENS, None), (LENS, None)]
    doc = w.mb.documents[-1]
    assert "extra" not in doc and C.SERVED_ENGINE_FIELD not in doc and "serving_path" not in doc and "key_geometry" not in doc
    assert set(doc) == {"admitted", "packages", "static", "live", "duty", "peer_duty", "peer_key", "peer_image", "ratio",
                        "cache_key", "key_inputs", "geometry", "standups"}
    assert C.SERVED_ENGINE_FIELD not in doc["key_inputs"]                       # the lens key: today's inputs, byte for byte
    assert doc["duty"]["served"] == {"max_num_seqs": 168, "max_model_len": L_4576, "configured_max_num_seqs": 171, "kv_shrink": rec}
    assert doc["peer_duty"]["served"]["max_num_seqs"] == 168
    assert Path(out["attestation_path"]).is_file() and out["cache_key"] == doc["cache_key"]
    # a lens standup that served ANOTHER S with no holding record is still refused, unchanged
    bad = w.run(LENS, {"nemotron": {"geometry": {"max_model_len": L_4576, "max_num_seqs": 168}}, "peer": peer})
    assert bad["status"] == C.REC_SERVED_GEOMETRY_MISMATCH and "certificate key" not in bad["error"]


def test_the_native_probe_keeps_the_pre_computed_s_equality(probe_world):
    """Under native the key IS the §1.8 kernel-adjusted served S (4576 x 171 -> 91 for D = 5120): a standup that served
    exactly that is admitted with the engine bound into the document; one that shrank further (a KV step after the
    kernel step, 91 -> 88) writes a certificate no reader would look for and refuses by name, as before."""
    w = probe_world
    step = sg.kernel_ceiling_adjustment(NEMO, L_4576, 171)
    assert (step["from_S"], step["to_S"]) == (171, 91)
    nemotron = {"engine": NATIVE, C.SERVED_ENGINE_FIELD: NATIVE,
                "geometry": {"max_model_len": L_4576, "max_num_seqs": 91, "adjustments": [step]},
                "mamba_backend_attestation": {"static": {"ok": True}, "live": {"ok": True, "reports": [], "expected_n_mamba_layers": 1}}}
    peer = {"engine": NATIVE, "geometry": {"max_model_len": L_4576, "max_num_seqs": 91, "adjustments": []}}
    out = w.run(NATIVE, {"nemotron": nemotron, "peer": peer})
    assert out["status"] == "OK", out.get("error")
    assert out["key_geometry"] == {"max_model_len": L_4576, "max_num_seqs": 91, "requested_max_num_seqs": 171}
    assert out["geometry_adjustments"] == [step] and out[C.SERVED_ENGINE_FIELD] == NATIVE
    # each standup names ITS OWN row's bank (the peer's, not Nemotron's — a quirk the harness surfaced)
    assert [(k["serving_path"], k["adapter_bank"]) for k in w.cmds] == [
        (NATIVE, f"{w.L.ADAPTER_BANK_ROOT}/{NEMO}"), (NATIVE, f"{w.L.ADAPTER_BANK_ROOT}/{w.L.C.model_row(w.L.PEER_SLUG).model_key}")]
    assert w.mb.documents[-1]["key_inputs"][C.SERVED_ENGINE_FIELD] == NATIVE
    # the NATIVE document (at its own engine-bound key) is where the additive block lives
    extra = w.mb.documents[-1]["extra"]
    assert extra[C.SERVED_ENGINE_FIELD] == NATIVE and extra["serving_path"] == NATIVE
    assert extra["key_geometry"] == out["key_geometry"] and extra["geometry_adjustments"] == [step]
    # served below the pre-computed key S (a KV step after the kernel one): refused by name — the native rule stands
    kv = sg.kv_shrink_adjustment(_shrink(91, 88.2))
    shrunk = {**nemotron, "geometry": {"max_model_len": L_4576, "max_num_seqs": 88, "adjustments": [step, kv]}}
    bad = w.run(NATIVE, {"nemotron": shrunk, "peer": peer})
    assert bad["status"] == C.REC_SERVED_GEOMETRY_MISMATCH
    assert "served S=88" in bad["error"] and "pre-computed S=91" in bad["error"]
    # ... and a lens record on a native probe is the engine refusal, unchanged
    lens_rec = {"geometry": {"max_model_len": L_4576, "max_num_seqs": 171}}
    bad = w.run(NATIVE, {"nemotron": lens_rec, "peer": peer})
    assert bad["status"] in (C.REC_SERVED_GEOMETRY_MISMATCH, w.L.REC_SERVING_ENGINE_MISMATCH)
