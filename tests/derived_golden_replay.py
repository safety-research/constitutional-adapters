"""The replay list behind the sdfbase_derived 100k BYTE-IDENTITY golden
(tests/fixtures/sdfbase_derived_100k_golden.json).

Six derived objects were published from the derived app's 100k path
(constv2-mix-100k − constv2ctl-100k). When the app learned the full − full
pair (constv2-mix-full − constv2ctl2-full) the 100k path had to stay
byte-identical — every argv the app hands a container and every volume path
it writes — because a change that silently moved those outputs would
corrupt the record even though the 100k rows are no longer cited.
:func:`collect` replays a fixed list of calls against a loaded
``sdfbase_derived`` module and returns one JSON-able dict: the generator
(tests/tools/gen_sdfbase_derived_golden.py) ran it on the PRE-CHANGE module
(git a9bf57f) and froze the answer; the test
(tests/test_sdfbase_derived_golden.py) runs it on the current module
and asserts equality. Temp-dir prefixes are normalized to ``<TMP>`` so the
two runs compare; the commit-bearing names (APP_NAME / SHA16 / _ENV_PIN)
are deliberately NOT recorded — they change with every commit by design.

What is covered: every pure command builder under the 100k defaults (the
residual pipeline's five subcommands, identity, probe at both scales and
both registered probes, CE on every default pool for the derived ids, the
100k parents and the bare row, liveness, MMLU, snapshot-verify paths), the
module's path/registry constants, ``_score_impl``'s FULL in-container
command trace for the old 6-field item (default pools staged and
unstaged, explicit --ce-pools, --no-parents), ``_liveness_impl``'s trace,
and the entrypoint's spawned items for the modes whose item shape is
unchanged. NOT covered (interface, not output): ``MODES``, the score /
liveness spawn tuples (they grew the pair fields), the fan-out's calibrate
/ production items (they grow a trailing field only when a capture budget
is set).
"""

from __future__ import annotations

import json
import pathlib
import errno
import os
import sys
from pathlib import Path, PosixPath

KEYS = ("gemma4e4b_base", "qwen35_4b_base")
MIX, CTL = "constv2-mix-100k", "constv2ctl-100k"
PY = "/py"
DERIVED_IDS = {
    "psr_paired_write": f"{MIX}__psr_ce_all_minus_ctl",
    "psr_residual": f"{MIX}__psr_ce_all_minus_ctl_distilled",
    "lora_subtraction": f"{MIX}__lora_r4_ce_minus_ctl",
}
CONSTANTS = ("CE_POOLS", "PARENT_CELLS", "DERIVED_ROOT", "DERIVED_KINDS",
             "CE_ROOT", "PARENTS_ROOT", "MMLU_ROOT", "SNAPSHOT_VERIFY_ROOT",
             "GATEREG_TRIALS", "GATEREG_ROUND", "GATEREG_SI", "LOG_ROOT",
             "RUN_CWD", "STDERR_TAIL_CHARS", "CALIB_PARTS_DIRNAME",
             "DEFAULT_CHECKPOINT_EVERY", "EPOCH_S_WORST", "CALIB_TIMEOUT_S",
             "PROD_TIMEOUT_S", "SERIAL_TIMEOUT_S", "_SECRETS")
PIPELINE_KWARGS = ("mix_si", "ctl_si", "wandb", "epochs_cap", "patience",
                   "checkpoint_every", "production")


def fake_objects(key: str) -> dict[str, str]:
    return {oid: f"/vol/out/derived/{kind}/{key}/{oid}/object.pt"
            for kind, oid in DERIVED_IDS.items()}


class _NoMkdirPath(PosixPath):
    """``Path`` whose ``mkdir`` is a no-op and under which NOTHING exists at
    ``/vol/...``: the impls mkdir /vol/... output dirs and probe /vol/...
    pools, parents and CE files, and every trace here pins the OFF-VOLUME
    answer (no bank, no staged pool -- the grading host, where the golden was
    recorded). The Modal CPU suite mounts the live twosi-fits volume at /vol,
    so the real filesystem answers "staged" there: three derived nodes flipped
    the moment the `modal` stub stopped masking them (2026-09-16). Every path
    outside /vol is the real filesystem (the tmp_path pools, CE_ROOT)."""

    # errno is load-bearing: Path.exists/is_file/is_dir swallow only an OSError
    # whose errno is ENOENT/ENOTDIR/EBADF/ELOOP (pathlib._ignore_error); a bare
    # FileNotFoundError(str) has errno None and propagates out of is_file().
    _OFF_VOLUME = "/vol"

    def _off_volume(self) -> bool:
        s = str(self)
        return s == self._OFF_VOLUME or s.startswith(self._OFF_VOLUME + "/")

    def mkdir(self, *args, **kwargs):
        return None

    def stat(self, *args, **kwargs):          # exists / is_file / is_dir read this
        if self._off_volume():
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), str(self))
        return super().stat(*args, **kwargs)

    def lstat(self, *args, **kwargs):
        if self._off_volume():
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), str(self))
        return super().lstat(*args, **kwargs)

    def open(self, *args, **kwargs):          # read_text / read_bytes read this
        if self._off_volume():
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), str(self))
        return super().open(*args, **kwargs)

    def iterdir(self):
        if self._off_volume():
            return iter(())
        return super().iterdir()

    def glob(self, *args, **kwargs):
        if self._off_volume():
            return iter(())
        return super().glob(*args, **kwargs)

    def rglob(self, *args, **kwargs):
        if self._off_volume():
            return iter(())
        return super().rglob(*args, **kwargs)


class _FakeVol:
    def __init__(self):
        self.log = []

    def reload(self):
        self.log.append("reload")

    def commit(self):
        self.log.append("commit")


class _FakeFn:
    def __init__(self, name, log):
        self.name, self.log = name, log

    def spawn(self, item):
        self.log.append((self.name, list(item)))

        class _Call:
            @staticmethod
            def get(timeout=None):
                return {"status": "ok"}
        return _Call()


#: Environment-specific strings the traces carry, normalised on BOTH sides so
#: the golden pins the module's behaviour and not the recording machine: the
#: replay's tmp dir, and the interpreter (every spawned command's argv[0] is
#: ``sys.executable`` -- the fixture was recorded under one venv's python and
#: the Modal CPU suite runs another; 124 argv[0]s differed, 2026-09-16).
PYTHON_PLACEHOLDER = "<PYTHON>"
#: ... and the W&B secret's name, a site setting (SISTEER_MODAL_WANDB_SECRET, read by sdfbase_fits at import): the golden records
#: that the app mounts one beside the HF token, never one site's name
WANDB_SECRET_PLACEHOLDER = "<WANDB_SECRET>"


def _norm(obj, tmp: str):
    s = json.dumps(obj, sort_keys=True, default=str)
    return json.loads(s.replace(tmp, "<TMP>").replace(sys.executable, PYTHON_PLACEHOLDER))


def collect(sd, tmp_dir) -> dict:
    """Replay the list against module ``sd`` (loaded under the fake
    ``modal``); ``tmp_dir`` hosts the staged fake pools. Pure: the module
    is restored to its original attributes on return."""
    import pytest  # monkeypatch without a fixture

    tmp_dir = Path(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    out: dict = {"constants": {}, "commands": {}, "score_traces": {},
                 "liveness_traces": {}, "spawns": {}, "pipeline": {}}
    for name in CONSTANTS:
        out["constants"][name] = getattr(sd, name)
    out["constants"]["_SECRETS"] = [s if tuple(s) == ("secret", "hf-token") else ("secret", WANDB_SECRET_PLACEHOLDER)
                                    for s in out["constants"]["_SECRETS"]]

    for key in KEYS:
        c: dict = {}
        c["resid_out_dir"] = sd.resid_out_dir(key, MIX)
        c["v_object_path"] = sd.v_object_path(key, MIX)
        c["residual_all_prod"] = sd.residual_command(
            key, MIX, CTL, no_production=False, lr_mult=3.0, wandb=True,
            python=PY, distill_epochs_cap=60, distill_patience=5,
            checkpoint_every=50)
        c["residual_all_noprod_sweep"] = sd.residual_command(
            key, MIX, CTL, no_production=True, lr_mult="0.3,1,3", wandb=False,
            python=PY)
        c["residual_all_int_mult"] = sd.residual_command(
            key, MIX, CTL, no_production=False, lr_mult=1, wandb=True,
            python=PY, checkpoint_every=0)
        for m in (0.3, 1.0, 3.0):
            c[f"calibrate_{m}"] = sd.calibrate_command(
                key, MIX, CTL, lr_mult=m, wandb=True, python=PY,
                distill_epochs_cap=60, distill_patience=5, checkpoint_every=50)
        c["calibrate_min"] = sd.calibrate_command(key, MIX, CTL, lr_mult=1.0,
                                                  wandb=False, python=PY)
        c["compose"] = sd.compose_command(key, MIX, CTL, python=PY)
        c["verify_v"] = sd.verify_v_command(key, MIX, CTL, python=PY)
        c["merge"] = sd.merge_command(key, MIX, CTL, lr_mults=[0.3, 1.0, 3.0],
                                      python=PY)
        c["merge_two"] = sd.merge_command(key, MIX, CTL, lr_mults=[1.0, 3.0],
                                          python=PY)
        c["distill"] = sd.distill_command(key, MIX, CTL, wandb=True, python=PY,
                                          distill_epochs_cap=60,
                                          distill_patience=5,
                                          checkpoint_every=50)
        c["distill_min"] = sd.distill_command(key, MIX, CTL, wandb=False,
                                              python=PY)
        c["identity"] = sd.identity_command(key, MIX, CTL, python=PY)
        objs = fake_objects(key)
        for oid, opath in objs.items():
            for s in (1.0, 2.0):
                for ver in ("v2", "v3g"):
                    c[f"probe|{oid}|{s}|{ver}"] = sd.probe_command(
                        key, opath, oid, scale=s, ver=ver, python=PY)
                for ptag, ppath in sd.CE_POOLS.items():
                    c[f"ce|{oid}|{ptag}|{s}"] = sd.ce_command(
                        key, opath, oid, ptag, ppath.format(key=key), scale=s,
                        python=PY)
        for cell in sd.PARENT_CELLS:
            for ptag, ppath in sd.CE_POOLS.items():
                c[f"ce|parent|{cell}|{ptag}"] = sd.ce_command(
                    key, f"/vol/out/{key}/cells_all/{cell}/object.pt", cell,
                    ptag, ppath.format(key=key), scale=1.0, python=PY)
        for ptag, ppath in sd.CE_POOLS.items():
            c[f"ce|bare|{ptag}"] = sd.ce_command(
                key, None, "bare", ptag, ppath.format(key=key), scale=1.0,
                python=PY)
            c[f"ce|tagged|{ptag}"] = sd.ce_command(
                key, objs[DERIVED_IDS["psr_paired_write"]],
                DERIVED_IDS["psr_paired_write"], ptag, ppath.format(key=key),
                scale=1.0, source_tag="ctl", python=PY)
        c["liveness_gatereg"] = sd.liveness_command(
            key, [f"/vol/out/hpo/{key}/psr_ce_all_all/r3_t00",
                  f"/vol/out/hpo/{key}/psr_ce_all_all/r3_t01"],
            pairs_path=f"/vol/pool/{key}/pairs_constv2-mix-1m.jsonl",
            n_probe=32, allow_bare=True, python=PY)
        c["liveness_derived"] = sd.liveness_command(
            key, [str(Path(p).parent) for p in objs.values()],
            pairs_path=f"/vol/pool/{key}/pairs_constv2-mix-100k.jsonl",
            n_probe=32, allow_bare=False, python=PY)
        c["mmlu"] = sd.mmlu_command(key, objs, "r2d", python=PY)
        c["snapshot_verify_record_path"] = sd.snapshot_verify_record_path(
            key, "20260904T000000Z")
        out["commands"][key] = c

    c = {"mult_key": [sd.mult_key(m) for m in (0.3, 1, 1.0, 3.0, 0.5)],
         "parse_mults": sd.parse_mults("0.3, 1,3,3"),
         "parse_scales": sd.parse_scales("1.0,2.0")}
    cal = tmp_dir / "calibration_controls.json"
    c["precheck_missing"] = sd.calibration_precheck(cal, epochs_cap=60,
                                                    patience=5)
    good = {"sweep": {"complete": True, "selected_lr_mult": 1.0,
                      "lr_mults": [0.3, 1, 3],
                      "budget": {"epochs_cap": 60, "patience": 5},
                      "per_mult": {"0.3": {}, "1": {"identity": {
                          "fingerprint": "f" * 64}}, "3": {}}}}
    cal.write_text(json.dumps(good))
    c["precheck_good"] = sd.calibration_precheck(cal, epochs_cap=60, patience=5)
    c["precheck_budget"] = sd.calibration_precheck(cal, epochs_cap=24, patience=2)
    out["commands"]["_shared"] = c

    # ---- in-container traces under a stable set of monkeypatches ----------
    mp = pytest.MonkeyPatch()
    try:
        staged = tmp_dir / "pool"
        pools_staged = {}
        for tag, tmpl in sd.CE_POOLS.items():
            for key in KEYS:
                p = Path(str(tmpl).replace("/vol/pool/{key}",
                                           f"{staged}/{key}").format(key=key))
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text("")
            pools_staged[tag] = str(tmpl).replace("/vol/pool/{key}",
                                                  f"{staged}/{{key}}")
        vol = _FakeVol()
        trace = []

        def fake_run(cmd, tag):
            trace.append({"tag": tag, "cmd": list(cmd)})
            return {"tag": tag, "rc": 0, "wall_s": 1.0}

        mp.setattr(sd, "_run", fake_run)
        mp.setattr(sd, "vol", vol)
        mp.setattr(sd, "Path", _NoMkdirPath)
        mp.setattr(sd.base, "_ensure_model", lambda k: trace.append(
            {"ensure_model": k}))
        mp.setattr(sd, "derived_object_dirs",
                   lambda key, root=None, kinds=None: {
                       k: v for k, v in fake_objects(key).items()
                       if kinds is None or any(f"/{kind}/" in v for kind in kinds)})
        for key in KEYS:
            traces = {}
            items = {
                "auto_x12_parents_unstaged": (key, "auto", "1.0,2.0", "v2", True, ""),
                "auto_x1_noparents_unstaged": (key, "auto", "1.0", "v2", False, ""),
                "auto_x1_v3g": (key, "auto", "1.0", "v3g", False, ""),
            }
            for label, item in items.items():
                trace.clear(); vol.log.clear()
                res = sd._score_impl(item)
                res.pop("app", None)
                traces[label] = {"trace": list(trace), "vol": list(vol.log),
                                 "result": res}
            # default pools STAGED (the real path's shape)
            mp.setattr(sd, "CE_POOLS", pools_staged)
            for label, item in (("staged_auto_x12_parents",
                                 (key, "auto", "1.0,2.0", "v2", True, "")),
                                ("staged_two_objects_x1_noparents",
                                 (key, ",".join(list(fake_objects(key))[:2]),
                                  "1.0", "v2", False, ""))):
                trace.clear(); vol.log.clear()
                res = sd._score_impl(item)
                res.pop("app", None)
                traces[label] = {"trace": list(trace), "vol": list(vol.log),
                                 "result": res}
            mp.setattr(sd, "CE_POOLS", dict(out["constants"]["CE_POOLS"]))
            csv = ",".join(f"{t}:{p}" for t, p in pools_staged.items())
            trace.clear(); vol.log.clear()
            res = sd._score_impl((key, "auto", "1.0", "v2", True, csv))
            res.pop("app", None)
            traces["explicit_ce_pools_x1_parents"] = {
                "trace": list(trace), "vol": list(vol.log), "result": res}
            out["score_traces"][key] = traces
            # liveness: no gate_reg trials, the derived psr objects, the
            # derived pool path unstaged (a /vol path off-volume)
            mp.setattr(sd, "gatereg_trial_dirs", lambda key, **kw: [])
            trace.clear(); vol.log.clear()
            res = sd._liveness_impl((key, "auto", 32))
            res.pop("app", None)
            out["liveness_traces"][key] = {"trace": list(trace),
                                           "vol": list(vol.log), "result": res}
        # ---- entrypoint spawns for the modes whose item shape is unchanged
        log = []
        for fn in ("residual_h200_std", "residual_h200_qwen",
                   "production_h200_std", "production_h200_qwen",
                   "identity_h200_std", "identity_h200_qwen",
                   "mmlu_h200_std", "mmlu_h200_qwen", "snapshot_verify_cpu"):
            mp.setattr(sd, fn, _FakeFn(fn, log))
        model = ",".join(KEYS)
        # the current module REQUIRES --app-name; the a9bf57f
        # module had no such parameter — pass it only where it exists (the
        # spawned items, the recorded output, do not carry it)
        import inspect
        extra = ({"app_name": sd.APP_NAME}
                 if "app_name" in inspect.signature(sd.main).parameters else {})
        for label, kw in (
                ("serial_distill", dict(mode="residual-distill", serial=True)),
                ("serial_distill_nowandb_ckpt0",
                 dict(mode="residual-distill", serial=True, wandb=False,
                      checkpoint_every=0)),
                ("serial_calibration", dict(mode="calibration", serial=True)),
                ("prod_only", dict(mode="residual-distill-prod")),
                ("prod_only_budget", dict(mode="residual-distill-prod",
                                          distill_epochs_cap=24,
                                          distill_patience=2,
                                          checkpoint_every=10)),
                ("identity", dict(mode="identity")),
                ("mmlu", dict(mode="mmlu", run_label="r2d")),
                ("mmlu_objects", dict(mode="mmlu", run_label="r2d",
                                      objects="a,b")),
                ("snapshot_verify", dict(mode="snapshot-verify", cells="x,y",
                                         out_root="out/k"))):
            log.clear()
            sd.main(model=model, **kw, **extra)
            out["spawns"][label] = list(log)
        seen = []

        def fake_pipeline(k, mults, **kw):
            seen.append((k, [float(m) for m in mults],
                         {kk: kw.get(kk) for kk in PIPELINE_KWARGS}))
            return {"status": "ok"}

        mp.setattr(sd, "residual_pipeline", fake_pipeline)
        for label, kw in (("fanout_distill", dict(mode="residual-distill")),
                          ("fanout_calibration",
                           dict(mode="calibration", lr_mults="1,3",
                                distill_epochs_cap=24, distill_patience=2,
                                checkpoint_every=10, wandb=False))):
            seen.clear()
            sd.main(model=model, **kw, **extra)
            out["pipeline"][label] = sorted(seen, key=lambda t: t[0])
    finally:
        mp.undo()
    return _norm(out, str(tmp_dir))
