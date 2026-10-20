"""Every LENS record the launcher PERSISTS is
byte-identical to what the base commit (main ``fafbec6c``, the merge base of this landing) wrote.

Two documents, each built COMPLETELY through the base commit's own code — ``git archive fafbec6c`` into
a scratch tree, the base launcher loaded by path under the base tree's ``src`` and ``tests`` in a
subprocess — and through the current code in-process, from the same inputs at the same scratch paths,
then compared byte for byte:

- ``<out_dir>/served_geometry.json`` of a lens ``_run_configuration`` (the sidecar the measurement
  proposal and the report read). The ONE allowed difference is the argv: every standup now states
  ``--serving-path lens`` (the standup's own default flipped, so an unstated path would silently stand
  up the other engine); the sidecar's ``standup_argv`` is held to the base argv + exactly those two
  tokens at the documented position, and the rest of the document to the base BYTES.
- the lens Nemotron duty attestation ``mamba_backend_attestation.json`` a ``--nemotron-probe
  --serving-path lens`` writes at the (unchanged, by design) lens cache key — the document a rerun
  OVERWRITES, which is why an additive block there rewrites a historical record.

Before the fix the lens sidecar gained ``serving_path`` / ``serving_engine`` and the lens attestation
an ``extra`` block; both are NATIVE-only now (``test_tinycap_launcher_native`` /
``test_nemotron_probe_cpu`` pin the native additions). Skipped, loudly, where the base commit cannot be
archived — the Modal CPU suite runs on a staged tree without its history; the two shape assertions
that need no history (the lens sidecar's exact key list, no ``extra`` on the lens attestation call) run
everywhere, in ``test_tinycap_launcher`` / ``test_nemotron_probe_cpu``."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from sisteer.capeval.serving import geometry as sg
from sisteer.capeval.tinycap import contracts as C

import test_nemotron_probe_cpu as TN  # noqa: E402  (the probe's CPU fakes: _FakeMB, _Proc, _shrink)
import test_tinycap_launcher as TL  # noqa: E402  (the modal stub, the by-path loader, the fake standup)

REPO = Path(__file__).resolve().parents[1]
#: main as merged into this landing: the code that wrote every lens record on the volume
BASE_SHA = "fafbec6c"
_SENTINEL = "GOLDEN_JSON:"


@pytest.fixture(scope="module")
def L():
    return TL._load_launcher(TL._stub_modal(), app_name=TL.TEST_APP_NAME, name="tinycap_served_lens_golden_under_test")


@pytest.fixture(scope="module")
def base_tree(tmp_path_factory) -> Path:
    """The base commit's WHOLE tree (``git archive``, no history, no worktree bookkeeping) under a
    scratch dir — its launcher, its ``src`` and its test fakes are what the goldens run."""
    root = tmp_path_factory.mktemp("nsd_base") / "base"
    root.mkdir()
    try:
        archive = subprocess.run(["git", "-C", str(REPO), "archive", "--format=tar", BASE_SHA],
                                 capture_output=True, timeout=600)
    except (OSError, subprocess.TimeoutExpired) as e:       # no git here, or no repository behind this tree
        pytest.skip(f"the base commit {BASE_SHA} cannot be archived here ({type(e).__name__}: {e}); the "
                    "byte-identity goldens need the repository history")
    if archive.returncode != 0:
        pytest.skip(f"git archive {BASE_SHA} refused ({archive.stderr.decode(errors='replace')[:300].strip()}); "
                    "the byte-identity goldens need the repository history")
    subprocess.run(["tar", "-xf", "-", "-C", str(root)], input=archive.stdout, check=True)
    launcher = root / "scripts/modal/tinycap_served.py"
    assert launcher.is_file() and (root / "tests/test_tinycap_launcher.py").is_file()
    # the archived tree IS the pre-landing launcher: no serving path anywhere in it
    assert "SERVING_PATH_FLAG" not in launcher.read_text()
    return root


def _run_base(base: Path, driver: str, *args: str) -> dict:
    """Run ``driver`` (a self-contained program) in a subprocess whose import path is the BASE tree's
    ``src`` + ``tests`` and nothing else (``-P``: no cwd, no script dir), returning the JSON it prints
    behind :data:`_SENTINEL` — the launcher's own prints come before it."""
    env = {k: v for k, v in os.environ.items() if k != C.LENS_STACKED_REDUCTIONS_ENV}
    env["PYTHONPATH"] = f"{base / 'src'}{os.pathsep}{base / 'tests'}"
    env.pop("PYTHONSAFEPATH", None)
    proc = subprocess.run([sys.executable, "-P", "-c", driver, *args], env=env, cwd=str(base),
                          capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, f"the base-commit driver failed (rc {proc.returncode}):\n{proc.stderr[-6000:]}"
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith(_SENTINEL)]
    assert len(lines) == 1, f"no golden record in the base driver's output:\n{proc.stdout[-3000:]}"
    return json.loads(lines[0][len(_SENTINEL):])


# --------------------------------------------------------------------------- #
# the lens served_geometry.json sidecar                                        #
# --------------------------------------------------------------------------- #

#: the base tree's ``_fake_world`` + ``_run_fake`` by hand (pytest's monkeypatch is not available to a
#: plain program): the fitted default cell on gsm8k, a staged tree under <tmp>/repos, the fake standup
#: attesting ``record``, the run ending at the fake ledger — past the sidecar write
LENS_SIDECAR_DRIVER = r'''
import json, os, shutil, sys
from pathlib import Path
import test_tinycap_launcher as TL
from sisteer.capeval.tinycap import contracts as C
tmp, record = Path(sys.argv[1]), json.loads(sys.argv[2])
L = TL._load_launcher(TL._stub_modal(), app_name=TL.TEST_APP_NAME, name="tinycap_served_base_golden")
cfg = L.plan_configurations(cells=[C.CELL_PSRKL_SAFETY], modes=["off"], controls=["gsm8k"])[0]
shutil.rmtree(f"/tmp/srv_{cfg['name']}", ignore_errors=True)
repos = tmp / "repos"
TL._tree(repos, "f6cf462")
L.C.VOL_REPOS_ROOT = str(repos)
os.environ.pop(C.LENS_STACKED_REDUCTIONS_ENV, None)
out = tmp / "gen"
shutil.rmtree(out, ignore_errors=True)
cfg = dict(cfg, out_dir=str(out))
TL._FakeStandupProc.record = dict(record)
L.subprocess.Popen = TL._FakeStandupProc
L.time.sleep = lambda s: None
rec = TL._run_fake(L, cfg)
sidecar = out / L.SERVED_GEOMETRY_FILENAME
print("GOLDEN_JSON:" + json.dumps({"status": rec["status"], "error": rec.get("error"),
                                   "sidecar": sidecar.read_text() if sidecar.is_file() else None}))
'''


def test_the_lens_served_geometry_sidecar_is_byte_identical_to_the_base_commit(L, base_tree, tmp_path, monkeypatch):
    """A lens run's ``served_geometry.json``, through the base commit's launcher and through this one, from
    the same configuration, tree, standup record and scratch paths: the same key list in the same order
    (no engine, no path on a lens record), the argv base + the two serving-path tokens, every other byte equal."""
    base = _run_base(base_tree, LENS_SIDECAR_DRIVER, str(tmp_path), json.dumps({}))
    assert base["status"] == "LEDGER_FAILED", base          # past the standup, the attestation and the sidecar write
    want_text = base["sidecar"]
    assert want_text, "the base launcher wrote no sidecar"
    cfg, out = TL._fake_world(L, tmp_path, monkeypatch, record={})
    monkeypatch.delenv(C.LENS_STACKED_REDUCTIONS_ENV, raising=False)
    shutil.rmtree(out, ignore_errors=True)
    rec = TL._run_fake(L, cfg)
    assert rec["status"] == "LEDGER_FAILED", rec.get("error")
    cur_text = (out / L.SERVED_GEOMETRY_FILENAME).read_text()
    cur, want = json.loads(cur_text), json.loads(want_text)
    assert list(want) == ["geometry", "standup_argv", "endpoint", "reduction_mode_requested"]
    assert list(cur) == list(want), f"the lens sidecar's keys moved: {list(cur)} != {list(want)}"
    # the ONE ruled argv change — `--serving-path lens` stated right after the readback flag
    i = want["standup_argv"].index(C.FLAG_ENABLE_LOG_REQUESTS) + 1
    assert cur["standup_argv"] == want["standup_argv"][:i] + [L.SERVING_PATH_FLAG, L.SERVING_PATH_LENS] + want["standup_argv"][i:]
    # every other byte: the document re-serialised the launcher's way with the base argv IS the base text
    assert json.dumps(cur, indent=1, default=str) == cur_text          # the serialiser round-trips, so the next line is a byte compare
    assert json.dumps({**cur, "standup_argv": want["standup_argv"]}, indent=1, default=str) == want_text


# --------------------------------------------------------------------------- #
# the lens Nemotron duty attestation                                           #
# --------------------------------------------------------------------------- #

#: ``test_nemotron_probe_cpu``'s fakes, inlined (that module is not on the base tree), the base
#: ``_nemotron_probe`` run on the lens path (its only path) against the scripted endpoint records
NEMOTRON_ATTESTATION_DRIVER = r'''
import hashlib, json, shutil, sys
from pathlib import Path
from types import SimpleNamespace
import test_tinycap_launcher as TL
from sisteer.capeval.serving import geometry as sg
tmp, endpoints = Path(sys.argv[1]), json.loads(sys.argv[2])
L = TL._load_launcher(TL._stub_modal(), app_name=TL.TEST_APP_NAME, name="tinycap_served_base_golden_nemotron")


class FakeMB:
    NEMOTRON_FAMILY = "nemotron_h"
    CONTRACT_RELPATH = "configs/capeval/mamba_backend_contract.json"

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
        return {"admitted": True, **kw}

    def readmission_problems(self, doc, contract, *, ratio, cache_key):
        return []


class Proc:
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


row = L.C.model_row(L.NEMOTRON_PROBE_SLUG)
tree = tmp / "tree"
for rel in (row.cfg, FakeMB.CONTRACT_RELPATH, L.C.model_row(L.PEER_SLUG).cfg):
    (tree / rel).parent.mkdir(parents=True, exist_ok=True)
    (tree / rel).write_text("{}\n")
geom = sg.resolve_geometry(next(k for k in sg.KNOWN_KEYS if sg.TABLE[k].max_model_len == 4576), model_key=row.model_key)


def _wait_for(ep, proc, timeout_s, step=5.0):
    which = ep.parent.name.rsplit("_", 1)[-1]
    ep.parent.mkdir(parents=True, exist_ok=True)
    ep.write_text(json.dumps(endpoints[which]))
    return True


L._mamba_backend = lambda: FakeMB()
L.resolve_staged_tree = lambda sha, repos_root=None: (tree, "f" * 40)
L.probe_geometry_cfg = lambda row_, geometry, **kw: {"name": "probe", "cell": "none", "mode": "off", "controls": ["ifeval"],
                                                   "model": row_.slug}
L.served_geometry = lambda cfg, pass_no, tree_=None: (geom, {})
L.overlaid_config = lambda r_, tree_, out_dir: (str(Path(tree_) / r_.cfg), {})
L._standup_cmd = lambda *a, **k: ["true"]
L.subprocess.Popen = Proc
L._wait_for = _wait_for
L._tear_down_standup = lambda stop, proc: None
L._persist_logs = lambda srv, out: None
L.VOL = SimpleNamespace(commit=lambda: None)
L.NEMO_BACKEND_ROOT = str(tmp / "nemo_backend")
shutil.rmtree(L.NEMO_BACKEND_ROOT, ignore_errors=True)
for which in ("nemotron", "peer"):
    shutil.rmtree(f"/tmp/srv_probe_{which}", ignore_errors=True)
try:
    rec = L._nemotron_probe("f6cf462", "4096", "stacked", "card", row=row)
finally:
    for which in ("nemotron", "peer"):
        shutil.rmtree(f"/tmp/srv_probe_{which}", ignore_errors=True)
doc = Path(rec["attestation_path"]).read_text() if rec.get("attestation_path") else None
print("GOLDEN_JSON:" + json.dumps({"status": rec["status"], "error": rec.get("error"), "cache_key": rec.get("cache_key"),
                                   "attestation_path": rec.get("attestation_path"), "document": doc}))
'''


def _lens_probe(L, tmp_path, monkeypatch, endpoints: dict):
    """``test_nemotron_probe_cpu.probe_world`` for THIS module's launcher: the real ``_nemotron_probe``
    body on the lens path against the scripted endpoint records, the GPU seams faked exactly as the base
    driver above fakes them, at the same scratch paths. Returns the REC."""
    row = L.C.model_row(L.NEMOTRON_PROBE_SLUG)
    tree = tmp_path / "tree"
    for rel in (row.cfg, TN._FakeMB.CONTRACT_RELPATH, L.C.model_row(L.PEER_SLUG).cfg):
        (tree / rel).parent.mkdir(parents=True, exist_ok=True)
        (tree / rel).write_text("{}\n")
    mb = TN._FakeMB()
    geom = sg.resolve_geometry(TN.EVAL_4576, model_key=TN.NEMO)

    def _wait_for(ep: Path, proc, timeout_s, step=5.0):
        which = ep.parent.name.rsplit("_", 1)[-1]
        ep.parent.mkdir(parents=True, exist_ok=True)
        ep.write_text(json.dumps(endpoints[which]))
        return True

    monkeypatch.setattr(L, "_mamba_backend", lambda: mb)
    monkeypatch.setattr(L, "resolve_staged_tree", lambda sha, repos_root=None: (tree, "f" * 40))
    monkeypatch.setattr(L, "probe_geometry_cfg", lambda row_, geometry, **kw: {"name": "probe", "cell": "none", "mode": "off",
                                                                                "controls": ["ifeval"], "model": row_.slug})
    monkeypatch.setattr(L, "served_geometry", lambda cfg, pass_no, tree_=None: (geom, {}))
    monkeypatch.setattr(L, "overlaid_config", lambda r_, tree_, out_dir: (str(Path(tree_) / r_.cfg), {}))
    monkeypatch.setattr(L, "_standup_cmd", lambda *a, **k: ["true"])
    monkeypatch.setattr(L.subprocess, "Popen", TN._Proc)
    monkeypatch.setattr(L, "_wait_for", _wait_for)
    monkeypatch.setattr(L, "_tear_down_standup", lambda stop, proc: None)
    monkeypatch.setattr(L, "_persist_logs", lambda srv, out: None)
    monkeypatch.setattr(L, "VOL", SimpleNamespace(commit=lambda: None))
    monkeypatch.setattr(L, "NEMO_BACKEND_ROOT", str(tmp_path / "nemo_backend"))
    shutil.rmtree(tmp_path / "nemo_backend", ignore_errors=True)
    for which in ("nemotron", "peer"):
        shutil.rmtree(f"/tmp/srv_probe_{which}", ignore_errors=True)
    try:
        return L._nemotron_probe("f6cf462", "4096", "stacked", "card", row=row, serving_path=L.SERVING_PATH_LENS), mb
    finally:
        for which in ("nemotron", "peer"):
            shutil.rmtree(f"/tmp/srv_probe_{which}", ignore_errors=True)


def _without_clocks(doc: dict) -> dict:
    """The document with its two wall-clock leaves (each standup's ``t_start`` / ``t_end``) zeroed — the
    only inputs that differ between two runs of the same probe."""
    out = json.loads(json.dumps(doc))
    for s in out.get("standups") or ():
        for k in ("t_start", "t_end"):
            if k in s:
                s[k] = 0.0
    return out


@pytest.mark.parametrize("case", ["kv_shrunk_171_to_168", "served_as_configured"])
def test_the_lens_nemotron_attestation_document_is_byte_identical_to_the_base_commit(L, base_tree, tmp_path, monkeypatch, case):
    """The lens Nemotron probe's folded attestation — the document filed at the lens cache key, which a
    rerun overwrites — through the base commit's ``_nemotron_probe`` and through this one, from the same
    endpoint records: the same cache key (the lens key is the requested geometry, by design), the same
    path, and the document's bytes equal once each run's own clock leaves are zeroed — no ``extra`` block,
    no engine, no path, no key geometry on a lens certificate. Both documented lens shapes: the KV-shrunk
    standup (171 -> 168, the peer at the SERVED 168) and a standup serving as configured."""
    L4576 = TN.L_4576
    att = {"static": {"ok": True}, "live": {"ok": True, "reports": [], "expected_n_mamba_layers": 1}}
    if case == "kv_shrunk_171_to_168":
        shrink = TN._shrink(171, 168.4)
        endpoints = {"nemotron": {"geometry": {"max_model_len": L4576, "max_num_seqs": 168, sg.KV_SHRINK_KEY: shrink},
                                  "mamba_backend_attestation": att},
                     "peer": {"geometry": {"max_model_len": L4576, "max_num_seqs": 168}}}
    else:
        endpoints = {"nemotron": {"geometry": {"max_model_len": L4576, "max_num_seqs": 171}, "mamba_backend_attestation": att},
                     "peer": {"geometry": {"max_model_len": L4576, "max_num_seqs": 171}}}
    base = _run_base(base_tree, NEMOTRON_ATTESTATION_DRIVER, str(tmp_path), json.dumps(endpoints))
    assert base["status"] == "OK", base
    rec, mb = _lens_probe(L, tmp_path, monkeypatch, endpoints)
    assert rec["status"] == "OK", rec.get("error")
    # the address: the lens key and its path are the base commit's
    assert rec["cache_key"] == base["cache_key"] and rec["attestation_path"] == base["attestation_path"]
    cur_text = Path(rec["attestation_path"]).read_text()
    cur, want = json.loads(cur_text), json.loads(base["document"])
    assert "extra" not in cur and "extra" not in mb.documents[-1]      # the additive block is the native document's
    assert list(cur) == list(want), f"the lens attestation's keys moved: {list(cur)} != {list(want)}"
    assert json.dumps(cur, indent=1, default=str) == cur_text            # the serialiser round-trips: a byte compare follows
    assert json.dumps(_without_clocks(cur), indent=1, default=str) == json.dumps(_without_clocks(want), indent=1, default=str)
