"""The derived app's FULL − FULL pair (constv2-mix-full − constv2ctl2-full):
the pair is a parameter of every mode, resolved by cell id and object sha
off the volume's bank (never through the ctl-full config bytes), the CE
pools follow the CONTROL (the r2 control's own frozen held-out is P3's
pool; the 17k-doc training pool is opt-in), the parents are the pair's
four cells, the LoRA subtraction runs as a CPU mode, scoring takes every
registered probe version and can keep CE files another pair published,
the residual stages carry a capture budget only when one is set, and
``--app-name`` must name this tree's app. The 100k path is pinned
byte-for-byte by tests/test_sdfbase_derived_golden.py; this file covers
what the pair fields ADD."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import pytest

from test_watch_successor import make_fake_modal

REPO = Path(__file__).resolve().parents[1]
SRC_PATH = REPO / "scripts" / "modal" / "sdfbase_derived.py"
SRC = SRC_PATH.read_text()
for _p in (REPO / "scripts" / "modal", REPO / "scripts" / "sdfbase"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
MIX, CTL = "constv2-mix-full", "constv2ctl2-full"
MIX100, CTL100 = "constv2-mix-100k", "constv2ctl-100k"
FULL_IDS = (f"{MIX}__psr_ce_all_minus_ctl", f"{MIX}__psr_ce_all_minus_ctl_distilled",
            f"{MIX}__lora_r4_ce_minus_ctl")
OLD_IDS = (f"{MIX100}__psr_ce_all_minus_ctl",
           f"{MIX100}__psr_ce_all_minus_ctl_distilled",
           f"{MIX100}__lora_r4_ce_minus_ctl")


@pytest.fixture(scope="module")
def sd():
    import importlib.util
    # ALWAYS this suite's own stub (test_sdfbase_derived_app.py's idiom): in the
    # full suite an earlier module has usually left ITS fake `modal` in
    # sys.modules -- several lineages `setdefault` a bare SimpleNamespace whose
    # Image has no debian_slim and never restore it -- and keeping it made every
    # test here ERROR at setup in the reseed merge tree's Modal CPU suite
    # (2026-09-15) while the file passed alone. Whatever was there is restored.
    prev_modal = sys.modules.get("modal")
    sys.modules["modal"] = make_fake_modal()
    sys.modules.pop("sdfbase_fits", None)
    spec = importlib.util.spec_from_file_location("_sdfbase_derived_full_test",
                                                  SRC_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    try:
        yield mod
    finally:
        if prev_modal is None:
            sys.modules.pop("modal", None)
        else:
            sys.modules["modal"] = prev_modal


def _d(cmd):
    return dict(zip(cmd, cmd[1:]))


# the replay helper's Path stub: mkdir a no-op AND nothing exists under /vol
# (the off-volume premise these traces pin; the Modal CPU suite mounts the
# live volume at /vol, where the real filesystem would answer "staged")
from derived_golden_replay import _NoMkdirPath  # noqa: E402


class _Vol:
    def __init__(self):
        self.log = []

    def reload(self):
        self.log.append("reload")

    def commit(self):
        self.log.append("commit")


class _FakeFn:
    def __init__(self, name, log, handler=None):
        self.name, self.log, self.handler = name, log, handler
        self.items = []

    def spawn(self, item):
        self.items.append(item)
        self.log.append((self.name, item))
        h = self.handler

        class _Call:
            @staticmethod
            def get(timeout=None):
                return h(item) if h else {"status": "ok"}
        return _Call()


def test_pair_registry_parents_and_pools(sd):
    """The parents follow the pair (both arms of both cells, the historical
    order); the CE pools follow the CONTROL: the 100k control keeps its
    three pools, the r2 control scores the corpus held-out + its OWN frozen
    held-out (P3's pool) and NOT its 17k-doc training pool by default; an
    unregistered control refuses instead of borrowing pools."""
    assert sd.DEFAULT_MIX_SI == MIX100 and sd.DEFAULT_CTL_SI == CTL100
    assert sd.parent_cells(MIX100, CTL100) == sd.PARENT_CELLS
    assert sd.parent_cells(MIX, CTL) == (
        f"{MIX}__psr_ce_all", f"{CTL}__psr_ce_all",
        f"{MIX}__lora_r4_ce", f"{CTL}__lora_r4_ce")
    assert sd.ce_pools_for(CTL100) == sd.CE_POOLS
    assert sd.ce_pools_for(CTL100) is not sd.CE_POOLS   # a copy, never the table
    full = sd.ce_pools_for(CTL)
    assert full == {"constv2-heldout": "/vol/pool/{key}/pairs_constv2-heldout.jsonl",
                    "constv2ctl2-heldout": "/vol/pool/{key}/pairs_constv2ctl2-heldout.jsonl"}
    assert "constv2ctl2-full" not in full
    assert sd.CE_POOL_CTL2_INSAMPLE == {
        "constv2ctl2-full": "/vol/pool/{key}/pairs_constv2ctl2-full.jsonl"}
    # no leg of the full pair names the round-1 control (scope directive (2))
    for v in list(full.values()) + list(sd.parent_cells(MIX, CTL)):
        assert "constv2ctl-" not in v
    with pytest.raises(SystemExit, match="no default CE pools"):
        sd.ce_pools_for("constv2ctl-1m")
    # the entrypoint refuses a NON-CANONICAL pair up front, before any spawn
    # (the derived ids name the corpus cell only, so another control would
    # overwrite that pair's published objects)
    assert sd.CANONICAL_CTL == {MIX100: CTL100, MIX: CTL}
    with pytest.raises(SystemExit, match="decomposed against"):
        sd.main(app_name=sd.APP_NAME, mode="score", model="qwen35_4b_base", ctl_si="constv2ctl-1m")
    with pytest.raises(SystemExit, match="decomposed against"):
        sd.main(app_name=sd.APP_NAME, mode="residual-distill", model="qwen35_4b_base",
                mix_si=MIX100, ctl_si=CTL)
    with pytest.raises(SystemExit, match="decomposed against"):
        sd.main(app_name=sd.APP_NAME, mode="lora-subtract", model="qwen35_4b_base",
                mix_si=MIX, ctl_si=CTL100)
    with pytest.raises(SystemExit, match="no canonical control"):
        sd.main(app_name=sd.APP_NAME, mode="identity", model="qwen35_4b_base",
                mix_si="constv2-mix-1m", ctl_si=CTL100)
    sd.require_canonical_pair(MIX, CTL)
    sd.require_canonical_pair(MIX100, CTL100)
    # an existing sidecar of another pair is a problem for THIS pair
    assert sd.sidecar_parent_problems("/nonexistent/derived_meta.json", MIX, CTL,
                                      "psr_ce_all") == []


def test_residual_builders_carry_the_capture_budget_only_when_set(sd):
    """``--capture-budget-tokens`` rides into every model-bearing subcommand
    (calibrate / distill / all / identity-check) when given and is absent
    otherwise; the CPU compose / verify / merge never take it; the CLI
    parses every flag the app emits."""
    import psr_residual_distill as prd

    kw = dict(python="/py", capture_budget_tokens=100_000)
    cmds = {
        "calibrate": sd.calibrate_command("qwen35_4b_base", MIX, CTL, lr_mult=1.0,
                                          wandb=True, **kw)[0],
        "distill": sd.distill_command("qwen35_4b_base", MIX, CTL, wandb=True, **kw),
        "all": sd.residual_command("qwen35_4b_base", MIX, CTL, no_production=False,
                                   lr_mult="0.3,1,3", wandb=True, **kw),
        "identity-check": sd.identity_command("qwen35_4b_base", MIX, CTL, **kw),
    }
    ap = prd.build_parser()
    for sub, cmd in cmds.items():
        d = _d(cmd)
        assert cmd[2] == sub
        assert d["--capture-budget-tokens"] == "100000"
        assert d["--mix-si"] == MIX and d["--ctl-si"] == CTL
        assert d["--root"] == "/vol/out" and d["--pools"] == "/vol/pool/qwen35_4b_base"
        ns = ap.parse_args(cmd[2:])
        assert ns.capture_budget_tokens == 100_000 and ns.ctl_si == CTL
    for cmd in (sd.calibrate_command("qwen35_4b_base", MIX, CTL, lr_mult=1.0,
                                     wandb=True, python="/py")[0],
                sd.distill_command("qwen35_4b_base", MIX, CTL, wandb=True, python="/py"),
                sd.identity_command("qwen35_4b_base", MIX, CTL, python="/py"),
                sd.compose_command("qwen35_4b_base", MIX, CTL, python="/py")[0],
                sd.verify_v_command("qwen35_4b_base", MIX, CTL, python="/py"),
                sd.merge_command("qwen35_4b_base", MIX, CTL, lr_mults=[1.0],
                                 python="/py")[0]):
        assert "--capture-budget-tokens" not in cmd
    # verify-v carries the control: the CLI refuses a same-id V composed
    # against another control (psr_residual_distill.verify_v ctl_si)
    assert _d(sd.verify_v_command("qwen35_4b_base", MIX, CTL, python="/py"))["--ctl-si"] == CTL
    # the identity builder is the shared base (same paths as the others):
    # calibrate = base + [--lr-mult m --out part] (no budget args, no --wandb)
    assert sd.identity_command("qwen35_4b_base", MIX100, CTL100, python="/py")[3:] == \
        sd.calibrate_command("qwen35_4b_base", MIX100, CTL100, lr_mult=1.0,
                             wandb=False, python="/py")[0][3:-4]


def test_lora_subtract_mode_is_a_cpu_function_on_the_volume_bank(sd, monkeypatch):
    cmd, obj = sd.lora_subtract_command("qwen35_4b_base", MIX, CTL, python="/py")
    d = _d(cmd)
    assert cmd[:2] == ["/py", "/repo/scripts/sdfbase/lora_subtract.py"]
    assert d["--root"] == "/vol/out"
    assert d["--out-root"] == "/vol/out/derived/lora_subtraction"
    assert d["--models"] == "qwen35_4b_base" and d["--cells"] == MIX
    assert d["--ctl-si"] == CTL and d["--pools"] == "/vol/pool/qwen35_4b_base"
    assert "--keep-valid" in cmd
    assert obj == ("/vol/out/derived/lora_subtraction/qwen35_4b_base/"
                   f"{MIX}__lora_r4_ce_minus_ctl/object.pt")
    import lora_subtract as ls
    ns = ls.main.__globals__["argparse"].ArgumentParser()  # the CLI accepts them
    assert ls.derived_id_for(MIX) == Path(obj).parent.name
    with pytest.raises(KeyError):
        sd.lora_subtract_command("nope", MIX, CTL)
    for tok in cmd[1:]:
        if str(tok).startswith("/"):
            assert tok.startswith(("/vol/", "/repo/", "/py")), tok
    # the function is CPU-only and the impl reloads, runs, commits on success
    deco = SRC.split("\ndef lora_subtract_cpu(", 1)[0].rsplit("@app.function(", 1)[1]
    assert "gpu=" not in deco and "image=image_std" in deco
    vol = _Vol()
    runs = []
    monkeypatch.setattr(sd, "vol", vol)
    monkeypatch.setattr(sd, "_run", lambda cmd, tag: runs.append((tag, cmd)) or {
        "rc": 0, "wall_s": 1.0})
    res = sd._lora_subtract_impl(("qwen35_4b_base", MIX, CTL))
    assert res["status"] == "ok" and res["object"] == obj
    assert runs[0][0] == f"lora-subtract/qwen35_4b_base/{MIX}"
    assert vol.log == ["reload", "commit"]
    runs.clear(); vol.log.clear()
    monkeypatch.setattr(sd, "_run", lambda cmd, tag: {"rc": 2, "wall_s": 1.0})
    res = sd._lora_subtract_impl(("qwen35_4b_base", MIX, CTL))
    assert res["status"] == "failed" and vol.log == ["reload"]   # not committed
    # the entrypoint spawns it per model with the pair
    log = []
    fn = _FakeFn("lora", log)
    monkeypatch.setattr(sd, "lora_subtract_cpu", fn)
    assert "lora-subtract" in sd.MODES
    sd.main(app_name=sd.APP_NAME, mode="lora-subtract", model="qwen35_4b_base", mix_si=MIX, ctl_si=CTL)
    assert fn.items == [("qwen35_4b_base", MIX, CTL)]


def test_score_impl_for_the_full_pair(sd, tmp_path, monkeypatch):
    """The nine-field item: the pair's derived objects only (``mix``), every
    listed probe version (v3g primary, v2 legacy — version-qualified tags),
    the r2 control's two pools, the pair's four parents and the bare row —
    and NOTHING of the 100k pair or the round-1 control; --ce-skip-existing
    keeps a CE file already on the volume."""
    staged = tmp_path / "pool"
    for key in ("qwen35_4b_base",):
        for stem in ("constv2-heldout", "constv2ctl2-heldout", "constv2ctl2-full",
                     "constv2ctl-heldout", "constv2ctl-100k"):
            p = staged / key / f"pairs_{stem}.jsonl"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("")
    for table in ("CE_POOLS", "CE_POOLS_CTL2"):
        monkeypatch.setattr(sd, table, {
            t: str(tmpl).replace("/vol/pool/{key}", f"{staged}/{{key}}")
            for t, tmpl in getattr(sd, table).items()})
    objs = {oid: f"/vol/out/derived/k/qwen35_4b_base/{oid}/object.pt"
            for oid in FULL_IDS + OLD_IDS}
    monkeypatch.setattr(sd, "derived_object_dirs",
                        lambda key, root=None, kinds=None: dict(objs))
    trace = []
    monkeypatch.setattr(sd, "_run", lambda cmd, tag: trace.append((tag, list(cmd))) or {
        "tag": tag, "rc": 0, "wall_s": 1.0})
    monkeypatch.setattr(sd, "vol", _Vol())
    monkeypatch.setattr(sd, "Path", _NoMkdirPath)
    monkeypatch.setattr(sd.base, "_ensure_model", lambda k: None)
    res = sd._score_impl(("qwen35_4b_base", "mix", "1.0", "v3g,v2", True, "",
                          MIX, CTL, False))
    tags = [t for t, _c in trace]
    assert res["objects"] == sorted(FULL_IDS) and res["pools"] == [
        "constv2-heldout", "constv2ctl2-heldout"]
    for oid in FULL_IDS:
        i3, i2 = tags.index(f"probe/{oid}/v3g/x1"), tags.index(f"probe/{oid}/v2/x1")
        assert i3 < i2 < tags.index(f"ce/{oid}/constv2-heldout/x1") \
            < tags.index(f"ce/{oid}/constv2ctl2-heldout/x1")
    probe_cmds = {t: c for t, c in trace if t.startswith("probe/")}
    d3 = _d(probe_cmds[f"probe/{FULL_IDS[0]}/v3g/x1"])
    assert d3["--probe"] == "/repo/data/sdfbase/probe_v3_genre.jsonl"
    assert d3["--out"] == f"/vol/out/qwen35_4b_base/probe_v3/{FULL_IDS[0]}.jsonl"
    assert d3["--parents-root"] == "/vol/out" and "--scale" not in d3
    d2 = _d(probe_cmds[f"probe/{FULL_IDS[0]}/v2/x1"])
    assert d2["--out"] == f"/vol/out/qwen35_4b_base/probe_v2/{FULL_IDS[0]}.jsonl"
    # the pair's parents (error rows off-volume: no bank here) and the bare row
    parent_rows = [r for r in res["rows"] if r.get("tag", "").startswith("ce/constv2")
                   and r["rc"] == -1]
    assert [r["tag"] for r in parent_rows] == [
        f"ce/{c}" for c in sd.parent_cells(MIX, CTL)]
    assert tags[-2:] == ["ce/bare/constv2-heldout", "ce/bare/constv2ctl2-heldout"]
    blob = json.dumps(trace) + json.dumps(res)
    assert "constv2ctl-100k" not in blob and "constv2ctl-heldout" not in blob
    assert "constv2-mix-100k" not in blob and "constv2ctl2-full__" not in json.dumps(
        [c for t, c in trace if "pairs_constv2ctl2-full" in json.dumps(c)])
    # --ce-skip-existing: a CE output already on the volume is kept ONLY when
    # it records exactly this run's inputs (existence proved nothing) — the
    # bare row here; a derived object's file whose object bytes cannot be
    # hashed / differ is re-scored
    import hashlib
    trace.clear()
    monkeypatch.setattr(sd, "CE_ROOT", str(tmp_path / "ce_score"))
    pool_file = staged / "qwen35_4b_base" / "pairs_constv2-heldout.jsonl"
    pool_file.write_text('{"doc_id": "a"}\n{"doc_id": "b"}\n')
    spec = sd.MODELS["qwen35_4b_base"]
    bare_out = Path(sd.ce_command("qwen35_4b_base", None, "bare", "constv2-heldout",
                                  str(pool_file), scale=1.0)[1])
    bare_out.parent.mkdir(parents=True, exist_ok=True)
    good = {"pool": "constv2-heldout", "scale": 1.0, "object_sha256": None, "cell": None,
            "pairs_sha256": hashlib.sha256(pool_file.read_bytes()).hexdigest(),
            "model": spec["repo"], "revision": spec["rev"],
            "docs": [{"doc_id": "a"}, {"doc_id": "b"}]}
    bare_out.write_text(json.dumps(good))
    res = sd._score_impl(("qwen35_4b_base", FULL_IDS[2], "1.0", "v2", True, "",
                          MIX, CTL, True))
    tags = [t for t, _c in trace]
    assert "ce/bare/constv2-heldout" not in tags          # skipped: same inputs
    assert "ce/bare/constv2ctl2-heldout" in tags          # no file: scored
    skipped = [r for r in res["rows"] if r.get("skipped")]
    assert [r["tag"] for r in skipped] == ["ce/bare/constv2-heldout"]
    assert skipped[0]["rc"] == 0 and skipped[0]["out"] == str(bare_out)
    assert tags[0] == f"probe/{FULL_IDS[2]}/x1"   # one probe: the historical tag
    assert sd.ce_reuse_problems(bare_out, key="qwen35_4b_base", pool_tag="constv2-heldout",
                                pool_path=str(pool_file), obj_path=None, cell=None,
                                scale=1.0) == []
    # every mismatch re-scores: pool bytes moved, another scale, another cell,
    # a truncated doc list, an unreadable file, an object whose bytes differ
    for mutate, needle in ((lambda d: d.update(pairs_sha256="0" * 64), "pairs_sha256"),
                           (lambda d: d.update(scale=2.0), "scale"),
                           (lambda d: d.update(cell="x"), "cell"),
                           (lambda d: d.update(docs=d["docs"][:1]), "docs"),
                           (lambda d: d.update(revision="other"), "revision")):
        bad = json.loads(json.dumps(good))
        mutate(bad)
        bare_out.write_text(json.dumps(bad))
        probs = sd.ce_reuse_problems(bare_out, key="qwen35_4b_base", pool_tag="constv2-heldout",
                                     pool_path=str(pool_file), obj_path=None, cell=None,
                                     scale=1.0)
        assert probs and any(needle in x for x in probs), (needle, probs)
    bare_out.write_text("{not json")
    assert "unreadable" in sd.ce_reuse_problems(bare_out, key="qwen35_4b_base",
                                                pool_tag="constv2-heldout",
                                                pool_path=str(pool_file), obj_path=None,
                                                cell=None, scale=1.0)[0]
    bare_out.write_text(json.dumps(good))
    trace.clear()
    res = sd._score_impl(("qwen35_4b_base", FULL_IDS[2], "1.0", "v2", True, "",
                          MIX, CTL, True))
    tags = [t for t, _c in trace]
    # the derived object's own file: written by this run (no skip possible:
    # its /vol object bytes are not hashable here -> re-scored, never kept)
    obj_out = Path(sd.ce_command("qwen35_4b_base", objs[FULL_IDS[2]], FULL_IDS[2],
                                 "constv2-heldout", str(pool_file), scale=1.0)[1])
    obj_out.parent.mkdir(parents=True, exist_ok=True)
    obj_out.write_text(json.dumps({**good, "object_sha256": "ab" * 32, "cell": FULL_IDS[2]}))
    trace.clear()
    sd._score_impl(("qwen35_4b_base", FULL_IDS[2], "1.0", "v2", False, "", MIX, CTL, True))
    assert f"ce/{FULL_IDS[2]}/constv2-heldout/x1" in [t for t, _c in trace]
    # "mix" refuses when the pair has no object yet
    monkeypatch.setattr(sd, "derived_object_dirs",
                        lambda key, root=None, kinds=None: {
                            k: v for k, v in objs.items() if k in OLD_IDS})
    with pytest.raises(SystemExit, match="no derived object of"):
        sd._score_impl(("qwen35_4b_base", "mix", "1.0", "v2", False, "", MIX, CTL))
    with pytest.raises(SystemExit, match="unknown probe"):
        sd.parse_probe_versions("v2,v9")
    assert sd.parse_probe_versions("v3g, v2,v3g") == ["v3g", "v2"]


def test_entrypoint_spawns_the_pair_and_guards_the_app_name(sd, monkeypatch):
    log = []
    std, qwen = _FakeFn("sstd", log), _FakeFn("sqwen", log)
    monkeypatch.setattr(sd, "score_h200_std", std)
    monkeypatch.setattr(sd, "score_h200_qwen", qwen)
    sd.main(mode="score", model="qwen35_4b_base", objects="mix", scales="1.0",
            probe="v3g,v2", mix_si=MIX, ctl_si=CTL, ce_skip_existing=True,
            app_name=sd.APP_NAME)
    assert qwen.items == [("qwen35_4b_base", "mix", "1.0", "v3g,v2", True, "",
                           MIX, CTL, True)]
    assert std.items == []
    # the 100k defaults ride as the same fields (the impl reads them back)
    sd.main(app_name=sd.APP_NAME, mode="score", model="gemma4e4b_base")
    assert std.items == [("gemma4e4b_base", "auto", "1.0,2.0", "v2", True, "",
                          MIX100, CTL100, False)]
    with pytest.raises(SystemExit, match="unknown probe"):
        sd.main(app_name=sd.APP_NAME, mode="score", model="gemma4e4b_base", probe="v9")
    # --app-name must be THIS tree's app
    with pytest.raises(SystemExit, match="is not this tree's app"):
        sd.main(mode="score", model="gemma4e4b_base",
                app_name="sdfbase-derived-0000000000000000")
    assert re.fullmatch(r"sdfbase-derived-([0-9a-f]{16}|local)", sd.APP_NAME)
    # liveness carries the pair's mix pool
    lstd = _FakeFn("lstd", log)
    monkeypatch.setattr(sd, "liveness_h200_std", lstd)
    monkeypatch.setattr(sd, "liveness_h200_qwen", _FakeFn("lqwen", log))
    sd.main(app_name=sd.APP_NAME, mode="liveness", model="gemma4e4b_base", objects=FULL_IDS[1],
            n_probe=16, mix_si=MIX, ctl_si=CTL)
    assert lstd.items == [("gemma4e4b_base", FULL_IDS[1], 16, MIX)]
    impl = SRC.split("def _liveness_impl", 1)[1].split("\ndef ", 1)[0]
    assert 'f"/vol/pool/{key}/pairs_{mix_si}.jsonl"' in impl
    assert "pairs_constv2-mix-100k" not in impl
    # the residual modes append the capture budget ONLY when set
    seen = []
    monkeypatch.setattr(sd, "residual_pipeline",
                        lambda k, m, **kw: seen.append(kw.get("capture_budget_tokens"))
                        or {"status": "ok"})
    sd.main(app_name=sd.APP_NAME, mode="residual-distill", model="qwen35_4b_base", mix_si=MIX, ctl_si=CTL,
            capture_budget_tokens=100_000)
    sd.main(app_name=sd.APP_NAME, mode="residual-distill", model="qwen35_4b_base")
    assert seen == [100_000, None]
    pstd, pqwen = _FakeFn("pstd", log), _FakeFn("pqwen", log)
    monkeypatch.setattr(sd, "production_h200_std", pstd)
    monkeypatch.setattr(sd, "production_h200_qwen", pqwen)
    sd.main(app_name=sd.APP_NAME, mode="residual-distill-prod", model="qwen35_4b_base", mix_si=MIX,
            ctl_si=CTL, capture_budget_tokens=100_000)
    sd.main(app_name=sd.APP_NAME, mode="residual-distill-prod", model="qwen35_4b_base")
    assert pqwen.items == [("qwen35_4b_base", MIX, CTL, True, 60, 5, 50, 100_000),
                           ("qwen35_4b_base", MIX100, CTL100, True, 60, 5, 50)]
    istd, iqwen = _FakeFn("istd", log), _FakeFn("iqwen", log)
    monkeypatch.setattr(sd, "identity_h200_std", istd)
    monkeypatch.setattr(sd, "identity_h200_qwen", iqwen)
    sd.main(app_name=sd.APP_NAME, mode="identity", model="qwen35_4b_base", mix_si=MIX, ctl_si=CTL,
            capture_budget_tokens=100_000)
    assert iqwen.items == [("qwen35_4b_base", MIX, CTL, 100_000)]


def test_container_impls_read_the_capture_budget_from_the_item(sd, tmp_path,
                                                                 monkeypatch):
    """Every residual stage's container hands the budget to its CLI call
    when the item carries it, and emits the historical argv when it does
    not (the 100k items keep their shape)."""
    rd = tmp_path / "resid"
    rd.mkdir()
    (rd / "calibration_controls.json").write_text(json.dumps(
        {"sweep": {"complete": True, "selected_lr_mult": 1.0,
                   "budget": {"epochs_cap": 60, "patience": 5},
                   "per_mult": {"1": {"identity": {"fingerprint": "f" * 64}}}}}))
    monkeypatch.setattr(sd, "resid_out_dir", lambda k, m: str(rd))
    monkeypatch.setattr(sd, "vol", _Vol())
    monkeypatch.setattr(sd.base, "require_wandb_entity", lambda: "ent")
    monkeypatch.setattr(sd.base, "_ensure_model", lambda k: None)
    monkeypatch.setattr(sd, "_start_committer", lambda: type(
        "E", (), {"set": lambda self: None})())
    cmds = []
    monkeypatch.setattr(sd, "_run", lambda cmd, tag: cmds.append((tag, list(cmd))) or {
        "rc": 0, "wall_s": 1.0})
    sd._calibration_impl(("qwen35_4b_base", MIX, CTL, 0.3, False, 60, 5, 50, 100_000))
    sd._calibration_impl(("qwen35_4b_base", MIX100, CTL100, 0.3, False, 60, 5, 50))
    sd._production_impl(("qwen35_4b_base", MIX, CTL, False, 60, 5, 50, 100_000))
    sd._production_impl(("qwen35_4b_base", MIX100, CTL100, False, 60, 5, 50))
    sd._residual_impl(("qwen35_4b_base", MIX, CTL, False, "0.3,1,3", False, 60, 5,
                       50, 100_000))
    sd._residual_impl(("qwen35_4b_base", MIX100, CTL100, False, "0.3,1,3", False,
                       60, 5, 50))
    sd._identity_impl(("qwen35_4b_base", MIX, CTL, 100_000))
    sd._identity_impl(("qwen35_4b_base", MIX100, CTL100))
    with_budget = [c for _t, c in cmds if "--capture-budget-tokens" in c]
    without = [c for _t, c in cmds if "--capture-budget-tokens" not in c]
    assert len(with_budget) == 4 and len(without) == 4
    for c in with_budget:
        d = _d(c)
        assert d["--capture-budget-tokens"] == "100000"
        assert d["--mix-si"] == MIX and d["--ctl-si"] == CTL
    for c in without:
        d = _d(c)
        assert d["--mix-si"] == MIX100 and d["--ctl-si"] == CTL100
    # the fan-out threads it into the calibrate / production items
    log = []
    comp = _FakeFn("compose", log, lambda it: {"status": "ok", "composed": True,
                                               "teacher_object": "/vol/V.pt"})
    cal = _FakeFn("cal", log, lambda it: {"status": "ok", "run": {"wall_s": 1.0}})
    merge = _FakeFn("merge", log, lambda it: {"status": "ok", "calibration": {
        "selected_lr_mult": 1.0}})
    prod = _FakeFn("prod", log, lambda it: {"status": "ok"})
    kw = dict(mix_si=MIX, ctl_si=CTL, wandb=True, epochs_cap=60, patience=5,
              checkpoint_every=50, production=True, compose_fn=comp,
              calibrate_fn=cal, merge_fn=merge, production_fn=prod,
              log=lambda *a, **k: None)
    r = sd.residual_pipeline("qwen35_4b_base", [0.3, 1.0], capture_budget_tokens=100_000,
                             **kw)
    assert r["status"] == "ok" and r["capture_budget_tokens"] == 100_000
    assert comp.items == [("qwen35_4b_base", MIX, CTL)]
    assert cal.items == [("qwen35_4b_base", MIX, CTL, m, True, 60, 5, 50, 100_000)
                         for m in (0.3, 1.0)]
    assert prod.items == [("qwen35_4b_base", MIX, CTL, True, 60, 5, 50, 100_000)]
    cal.items.clear(); prod.items.clear()
    r = sd.residual_pipeline("qwen35_4b_base", [0.3], **{**kw, "mix_si": MIX100,
                                                          "ctl_si": CTL100})
    assert "capture_budget_tokens" not in r
    assert cal.items == [("qwen35_4b_base", MIX100, CTL100, 0.3, True, 60, 5, 50)]
    assert prod.items == [("qwen35_4b_base", MIX100, CTL100, True, 60, 5, 50)]


def test_no_config_import_for_the_pair(sd):
    """The derived app resolves ctl2 cells by id + sha off the bank;
    it never reads the fits configs for the pair (the SIS / si_ids
    registry stays as the base tree has it)."""
    # The "constv2ctl2-full absent from the frozen qwen/gemma base configs and from sdfbase_fits.SIS" pins were dropped:
    # they are false on main since the ten-cell SI_IDS (as was already done for test_ctl2_full_caps_row_moves_no_frozen_bytes);
    # the config bytes stay frozen under FROZEN_CONFIG_SHA16. What this test asserts is that _score_impl never reads a config:
    assert "_cfg_yaml" not in SRC.split("def _score_impl", 1)[1].split("\ndef ", 1)[0]


def test_ce_legs_on_the_r2_control_pool_feed_the_verdict(sd, tiny_llama, tmp_path):
    """END TO END across the app seam for the full pair's P3 pool: the argv
    ``ce_command`` builds for ``constv2ctl2-heldout`` runs through
    ce_score.main on the tiny model for V and both parents, and
    derived_verdict.main places the three files on the REGISTERED
    ``pairs_constv2ctl2-heldout.manifest.json`` under ``--p3-pool
    constv2ctl2-heldout --p3-pool-stem constv2ctl2-heldout`` — and REFUSES
    them under the 100k default (P3 is defined on one pool per pair)."""
    import hashlib

    import ce_score as cs
    from residual_fixtures import random_psr
    from test_derived_verdict import _write_jsonl, dv, make_probe_rows
    from test_residual_distill_script import (_CTL_ROWS, _MIX_ROWS, KEY,
                                              _stage_cell, _write_pool, prd)

    TINY = "hf-internal-testing/tiny-random-LlamaForCausalLM"
    model, _tok = tiny_llama
    root = tmp_path / "out"
    mix_dir = _stage_cell(root, KEY, MIX, random_psr(model, 11))
    ctl_dir = _stage_cell(root, KEY, CTL, random_psr(model, 22))
    pools = tmp_path / "pools"
    _write_pool(pools, MIX, _MIX_ROWS, "mix")
    _write_pool(pools, CTL, _CTL_ROWS, "ctl2")
    heldout = pools / "pairs_constv2ctl2-heldout.jsonl"
    rows = [{"prompt": "" if i % 2 else "Ctl sys\n\n",
             "pos": f"r2 control held-out doc {i} about weather " * (2 + i),
             "neg": "", "instruction_id": "constv2ctl2-heldout",
             "doc_id": f"ctl2-ho-{i}", "family": "stories" if i % 2 else "advice",
             "official_tokens": 8, "truncated": False} for i in range(4)]
    heldout.write_text("".join(json.dumps(r) + "\n" for r in rows))
    pool_sha = hashlib.sha256(heldout.read_bytes()).hexdigest()
    (pools / "pairs_constv2ctl2-heldout.manifest.json").write_text(json.dumps(
        {"cell": "constv2ctl2-heldout", "lane": "control_heldout",
         "pool_sha256": pool_sha, "doc_ids_in_order": [r["doc_id"] for r in rows],
         "n_docs": 4}))
    v_path = prd.compose_v(root, root / "derived", KEY, MIX, CTL, pools_dir=pools)
    v_id = prd.v_derived_id(MIX)
    assert json.loads((v_path.parent / "derived_meta.json").read_text())[
        "parent_neg"]["cell_id"] == f"{CTL}__psr_ce_all"

    def run_ce(obj_path, tag):
        pool_tag = "constv2ctl2-heldout"
        cmd, out = sd.ce_command(
            "gemma4e4b_base", str(obj_path), tag, pool_tag,
            sd.ce_pools_for(CTL)[pool_tag].format(key="gemma4e4b_base"), scale=1.0)
        argv = list(cmd[2:])
        local_out = tmp_path / "ce_score" / Path(out).name
        subst = {"--model": TINY, "--pairs": str(heldout), "--out": str(local_out),
                 "--parents-root": str(root), "--max-len": "96"}
        for i, tok in enumerate(argv):
            if tok in subst:
                argv[i + 1] = subst[tok]
        i = argv.index("--revision")
        del argv[i:i + 2]
        argv += ["--device", "cpu", "--dtype", "float32"]
        assert cs.main(argv) == 0
        assert local_out.name.endswith("__constv2ctl2-heldout__scale1.json")
        return local_out

    ce_v = run_ce(v_path, v_id)
    ce_mix = run_ce(mix_dir / "object.pt", f"{MIX}__psr_ce_all")
    ce_ctl = run_ce(ctl_dir / "object.pt", f"{CTL}__psr_ce_all")
    assert json.loads(ce_v.read_text())["pool"] == "constv2ctl2-heldout"
    o, m, c = make_probe_rows(80, seed=3)
    probes = {}
    for k, prow in (("object", o), ("mix", m), ("ctl", c)):
        pp = tmp_path / f"{k}_probe.jsonl"
        _write_jsonl(pp, prow)
        probes[k] = str(pp)
    verdict = tmp_path / "verdict.json"
    argv = ["--object-probe", probes["object"], "--mix-probe", probes["mix"],
            "--ctl-probe", probes["ctl"], "--object-ce", str(ce_v),
            "--mix-ce", str(ce_mix), "--ctl-ce", str(ce_ctl),
            "--pools-dir", str(pools), "--n-boot-items", "200",
            "--n-boot-docs", "100", "--out", str(verdict)]
    # the 100k default refuses a ctl2 pool file (exit 2: P3 pool mismatch)
    assert dv.main(argv) == 2
    assert dv.main(argv + ["--p3-pool", "constv2ctl2-heldout",
                           "--p3-pool-stem", "constv2ctl2-heldout"]) == 0
    res = json.loads(verdict.read_text())
    assert res["p3_pool"]["pool"] == "constv2ctl2-heldout"
    assert res["p3_pool"]["pairs_sha256"] == pool_sha
    assert res["p3_pool"]["manifest"]["stem"] == "constv2ctl2-heldout"
    assert res["p3_pool"]["manifest"]["cell"] == "constv2ctl2-heldout"
    assert res["p3_pool"]["cells"] == {"object_ce": v_id, "mix_ce": f"{MIX}__psr_ce_all",
                                       "ctl_ce": f"{CTL}__psr_ce_all"}
    assert res["p3"]["n_docs"] == 4 and res["p3"]["status"] in dv.STATUSES
    # the constants the 100k path reads are untouched
    assert dv.P3_POOL == "control-heldout" and dv.P3_POOL_MANIFEST_STEM == "constv2ctl-heldout"


def test_liveness_explicit_targets_must_all_resolve(sd, monkeypatch):
    """The two-object liveness run could report ok on one — or none. With
    EXPLICIT targets a name that resolves to nothing is a failing row, and
    derived targets whose mix pool is unstaged are a failing row; the
    historical ``auto`` call keeps its lenient shape (golden-pinned)."""
    objs = {oid: f"/vol/out/derived/psr_residual/qwen35_4b_base/{oid}/object.pt"
            for oid in FULL_IDS[:2]}
    monkeypatch.setattr(sd, "gatereg_trial_dirs", lambda key, **kw: [])
    monkeypatch.setattr(sd, "derived_object_dirs",
                        lambda key, root=None, kinds=None: dict(objs))
    monkeypatch.setattr(sd.base, "_ensure_model", lambda k: None)
    monkeypatch.setattr(sd, "vol", _Vol())
    monkeypatch.setattr(sd, "Path", _NoMkdirPath)   # /vol/pool/... is NOT staged here
    runs = []
    monkeypatch.setattr(sd, "_run", lambda cmd, tag: runs.append((tag, cmd)) or {
        "tag": tag, "rc": 0})
    res = sd._liveness_impl(("qwen35_4b_base", f"{FULL_IDS[0]},{FULL_IDS[1]},nope__x", 32, MIX))
    assert res["status"] == "failed"
    errs = [r for r in res["rows"] if r["rc"] != 0]
    assert any(r["tag"] == "liveness/targets" and "nope__x" in r["error"] for r in errs)
    # both resolve but the mix pool is not staged off-volume -> failing row
    res = sd._liveness_impl(("qwen35_4b_base", f"{FULL_IDS[0]},{FULL_IDS[1]}", 32, MIX))
    assert res["status"] == "failed" and runs == []
    assert [r["tag"] for r in res["rows"]] == ["liveness/derived"]
    assert f"pairs_{MIX}.jsonl" in res["rows"][0]["error"]
    # auto: unchanged lenient behaviour (no rows, ok)
    res = sd._liveness_impl(("qwen35_4b_base", "auto", 32))
    assert res["status"] == "ok" and res["rows"] == []


def test_compose_stage_refuses_a_v_of_another_pair(sd, tmp_path, monkeypatch):
    """Defense in depth: a V under this id whose sidecar names other
    parents is REFUSED by the compose stage — never verified and
    recomposed over."""
    v = tmp_path / "V" / "object.pt"
    v.parent.mkdir(parents=True)
    v.write_bytes(b"x")
    (v.parent / "derived_meta.json").write_text(json.dumps(
        {"parent_pos": {"cell_id": f"{MIX}__psr_ce_all"},
         "parent_neg": {"cell_id": f"{CTL100}__psr_ce_all"}}))
    monkeypatch.setattr(sd, "v_object_path", lambda k, m: str(v))
    runs = []
    monkeypatch.setattr(sd, "_run", lambda cmd, tag: runs.append(tag) or {"rc": 0})
    vol = _Vol()
    monkeypatch.setattr(sd, "vol", vol)
    res = sd._compose_impl(("qwen35_4b_base", MIX, CTL))
    assert res["status"] == "refused" and res["composed"] is False
    assert any("parent_neg" in p for p in res["problems"])
    assert runs == [] and vol.log == ["reload"]
    # the same pair's sidecar passes to the verify step
    (v.parent / "derived_meta.json").write_text(json.dumps(
        {"parent_pos": {"cell_id": f"{MIX}__psr_ce_all"},
         "parent_neg": {"cell_id": f"{CTL}__psr_ce_all"}}))
    res = sd._compose_impl(("qwen35_4b_base", MIX, CTL))
    assert res["status"] == "ok" and runs == ["verify-v/qwen35_4b_base/" + MIX]
    # --app-name is REQUIRED (omission refuses before any spawn)
    with pytest.raises(SystemExit, match="app-name is required"):
        sd.main(mode="score", model="qwen35_4b_base")


# --------------------------------------------------------------------------- #
# The SLURM twin's stage machinery:                                            #
# sdfbase_derived_build.sbatch executed under stubs — the persisted sweep.json #
# declaration, the NEW assess stage (merge over EXACTLY the declaration, a     #
# durable assess.json, no distillation), the idempotent extension declaration  #
# through an interruption + self-resubmit, a conflicting second extension      #
# refused, finish without/with a stale assess.json refused, --seed from the    #
# config, and the 3-rung default path byte-identical to the old finish's merge. #
# --------------------------------------------------------------------------- #

import os as _os
import shutil as _shutil
import subprocess as _sp
from types import SimpleNamespace as _NS

_SEEDREP = REPO / "scripts" / "sdfbase" / "seedrep"
if str(_SEEDREP) not in sys.path:
    sys.path.insert(0, str(_SEEDREP))
import derived_sweep as _ds  # noqa: E402

_STUB_SHA = "beefbeef" * 5
_KEY, _BANK = "qwen35_4b_base_eb64_s1", "reseed"
_DEFAULT_SCORES = {"0.3": 0.5, "1": 0.9, "3": 0.7, "0.1": 0.2, "10": 0.1}
_EDGE_SCORES = {"0.3": 0.9, "1": 0.5, "3": 0.4, "0.1": 0.95, "10": 0.1}

# The venv's `python`, faked: the cu128 assert passes; lora_subtract / gate_liveness /
# psr_residual_distill subcommands write the artifacts the sbatch checks for (the merge
# with the real merge's exactly-these-parts semantics and an argmax over FAKE_SCORES;
# calibrate can deliver SIGTERM to the sbatch ONCE for FAKE_TERM_ON_MULT, the way a
# preemption lands mid-part); the `-` heredoc is run by the real interpreter.
_FAKE_PY = r'''
import hashlib, json, os, signal, sys, time
from pathlib import Path
args = sys.argv[1:]
if args and args[0] == "-c":
    sys.exit(0)
if args and args[0] == "-":
    os.execvp(sys.executable, [sys.executable, "-", *args[1:]])
with open(os.environ["FAKE_LOG"], "a") as fh:
    fh.write(Path(args[0]).name + " " + " ".join(args[1:]) + "\n")
script, rest = Path(args[0]).name, args[1:]
def opt(name, default=None):
    return rest[rest.index(name) + 1] if name in rest else default
fm = lambda m: f"{float(m):g}"
if script == "lora_subtract.py":
    d = Path(opt("--out-root")) / opt("--models") / f"{opt('--cells')}__lora_r4_ce_minus_ctl"
    d.mkdir(parents=True, exist_ok=True)
    (d / "object.pt").write_bytes(b"lora"); (d / "derived_meta.json").write_text('{"kind": "lora_subtraction"}')
    sys.exit(0)
if script == "gate_liveness.py":
    if os.environ.get("FAKE_LIVENESS_FAIL"):
        print("fake liveness: probe failed", file=sys.stderr); sys.exit(1)
    for dd in opt("--dirs").split(","):
        Path(dd).mkdir(parents=True, exist_ok=True)
        vals = {str(i): 0.98 for i in range(4)}
        if os.environ.get("FAKE_LIVENESS_DEAD"):
            vals["2"] = 0.0                                   # the probe MEASURES a fully dead layer and exits 0 (as the real one does)
        (Path(dd) / "gate_liveness.json").write_text(json.dumps(vals))
    sys.exit(0)
assert script == "psr_residual_distill.py", script
sub, rest = rest[0], rest[1:]
out_root, key, mix = Path(opt("--out-root")), opt("--model-key"), opt("--mix-si")
vdir = out_root / "psr_paired_write" / key / f"{mix}__psr_ce_all_minus_ctl"
resid = out_root / "psr_residual" / key / f"{mix}__psr_ce_all_minus_ctl_distilled"
scores = json.loads(os.environ.get("FAKE_SCORES", "{}"))
if sub == "verify-v":
    sys.exit(0 if (vdir / "object.pt").is_file() else 2)
if sub == "compose-v":
    vdir.mkdir(parents=True, exist_ok=True); (vdir / "object.pt").write_bytes(b"V"); sys.exit(0)
h = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
if sub == "liveness-receipt":   # the real one: object sha + probe-file sha, bound to the bytes on disk, plus the admission's verdict
    gl = json.loads((vdir / "gate_liveness.json").read_text()); dead = [k for k, v in gl.items() if float(v) <= 0.0]
    (vdir / "gate_liveness_receipt.json").write_text(json.dumps({"object_sha256": h(vdir / "object.pt"), "gate_liveness_sha256": h(vdir / "gate_liveness.json"),
                                                                 "not_fully_dead": not dead, "dead_layers": dead, "dead_floor": 0.0, "writer": "fake"}))
    sys.exit(0)
if sub == "emit-v":             # the real one refuses unless BOTH bindings hold and no layer is fully dead
    rc = vdir / "gate_liveness_receipt.json"
    gl = json.loads((vdir / "gate_liveness.json").read_text()) if (vdir / "gate_liveness.json").is_file() else {}
    if (not rc.is_file() or json.loads(rc.read_text())["object_sha256"] != h(vdir / "object.pt")
            or json.loads(rc.read_text())["gate_liveness_sha256"] != h(vdir / "gate_liveness.json")):
        print("fake emit-v: refused, liveness not bound", file=sys.stderr); sys.exit(2)
    if any(float(v) <= 0.0 for v in gl.values()) or json.loads(rc.read_text()).get("not_fully_dead") is not True:
        print("fake emit-v: refused, fully dead layer", file=sys.stderr); sys.exit(2)
    (vdir / "registry_emission.json").write_text(json.dumps({"status": "emitted", "campaign": "sdfbase_full22", "contract": "2"}))
    sys.exit(0)
if sub == "capture-set":
    out = Path(opt("--out")); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"seed": int(opt("--seed")), "holdout_frac": float(opt("--holdout-frac"))}))
    sys.exit(0)
if sub == "calibrate":
    m, out = opt("--lr-mult"), Path(opt("--out")); out.parent.mkdir(parents=True, exist_ok=True)
    term_on, marker = os.environ.get("FAKE_TERM_ON_MULT"), out.parent / ".attempted"
    if term_on and fm(term_on) == fm(m) and not marker.exists():
        marker.write_text("1"); os.kill(os.getppid(), signal.SIGTERM); time.sleep(30); sys.exit(143)
    k, score = fm(m), scores[fm(m)]
    per = {k: {"trainer": {"lr_mult": float(m), "seed": int(opt("--seed"))},
               "identity": {"trainer": {"lr_mult": float(m)}, "capture": {"seed": int(opt("--seed"))}},
               "score_worst_stratum_attainable_ed": score}}
    out.write_text(json.dumps({"sweep": {"lr_mults": [float(m)], "complete": True,
                                         "budget": {"epochs_cap": int(opt("--distill-epochs-cap")),
                                                    "patience": int(opt("--distill-patience"))},
                                         "per_mult": per, "table": {k: {"score_worst_stratum_attainable_ed": score}}}}))
    sys.exit(0)
if sub == "merge-calibration":
    wanted = [fm(x) for x in opt("--lr-mult").split(",")]; out = Path(opt("--out")); per = {}
    for p in opt("--parts").split(","):
        for k, v in json.loads(Path(p).read_text())["sweep"]["per_mult"].items():
            if k not in wanted:
                print(f"merge-calibration: {p} carries lr_mult={k}, not in {wanted}", file=sys.stderr); sys.exit(2)
            per[k] = v
    missing = [k for k in wanted if k not in per]
    if missing:
        print(f"merge-calibration: no finished payload for lr_mult={missing}", file=sys.stderr); sys.exit(2)
    sel = max(wanted, key=lambda k: per[k]["score_worst_stratum_attainable_ed"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"sweep": {"lr_mults": [float(x) for x in wanted], "selected_lr_mult": float(sel),
                                         "selection": {"verdict": "interior", "reason": "fake"},
                                         "per_mult": {k: per[k] for k in wanted},
                                         "table": {k: {"score_worst_stratum_attainable_ed": per[k]["score_worst_stratum_attainable_ed"]} for k in wanted},
                                         "complete": True, "budget": {"epochs_cap": 60, "patience": 5}}}, indent=1))
    sys.exit(0)
if sub == "distill":
    sel = json.loads((resid / "calibration_controls.json").read_text())["sweep"]["selected_lr_mult"]
    obj = b"distilled"; (resid / "object.pt").write_bytes(obj)
    (resid / "derived_meta.json").write_text(json.dumps({"kind": "psr", "derived_id": f"{mix}__psr_ce_all_minus_ctl_distilled",
        "recovery_verdict": "recovered", "trainer": {"lr_mult": sel, "seed": int(opt("--seed"))},
        "object_sha256": hashlib.sha256(obj).hexdigest()}))
    (resid / "report.json").write_text(json.dumps({"best_epoch": 10, "epochs_run": 15, "epochs_cap": 60, "stopped_early": True,
        "stop_reason": "patience", "best_heldout_loss": 0.5, "heldout_init": 2.0, "ed": {}, "recovery": {"verdict": "recovered"}}))
    sys.exit(0)
print("fake python: unknown invocation", args, file=sys.stderr); sys.exit(9)
'''


def _derived_world(tmp_path, *, seed_line: str = "", scores=None):
    """The derived sbatch under a tmp $SHARED (the job env's SISTEER_SHARED_ROOT names it, as the front door's
    job env and submit_derived_chain.sh do; the script is copied into the fake checkout unchanged),
    a fake repo with the config, placeholder scripts and the REAL derived_sweep.py,
    a faked venv python, and PATH stubs for git / scontrol / sbatch / nvidia-smi."""
    shared, repo, binp = tmp_path / "shared", tmp_path / "repo", tmp_path / "bin"
    (shared / "scripts").mkdir(parents=True)
    (shared / "scripts" / "hf_env.sh").write_text(f'export HF_HOME="{shared}/hf"\n')
    (shared / "scripts" / "mirror-figs.sh").write_text("#!/bin/bash\nexit 0\n")
    (shared / "scripts" / "mirror-figs.sh").chmod(0o755)
    venv = shared / "envs" / "sdfbase-fit"
    (venv / "bin").mkdir(parents=True)
    (venv / "BUILD-COMPLETE").write_text("ok\n")
    (tmp_path / "fake_py.py").write_text(_FAKE_PY)
    (venv / "bin" / "python").write_text(f'#!/bin/bash\nexec python3 {tmp_path}/fake_py.py "$@"\n')
    (venv / "bin" / "python").chmod(0o755)
    snap = shared / "hf" / "hub" / "models--Qwen--Qwen3.5-4B-Base" / "snapshots" / "1001bb4d"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}\n")
    pool = shared / "vol" / "pool" / _KEY
    pool.mkdir(parents=True)
    for si in (MIX, CTL):
        (pool / f"pairs_{si}.jsonl").write_text("{}\n")
        (pool / f"pairs_{si}.manifest.json").write_text("{}\n")
    root = shared / "exp" / "objects" / "sdfbase" / _BANK
    cells = root / _KEY / "cells_all"
    cells.mkdir(parents=True)
    (cells / "_store_identity.json").write_text("{}\n")
    for c in (f"{MIX}__psr_ce_all", f"{CTL}__psr_ce_all", f"{MIX}__lora_r4_ce", f"{CTL}__lora_r4_ce"):
        (cells / c).mkdir()
        for f in ("object.pt", "COMPLETE", "train_budget_meta.json"):
            (cells / c / f).write_text("x\n")
    (repo / "configs").mkdir(parents=True)
    (repo / "configs" / f"sdfbase_{_KEY}_fits.yaml").write_text(
        f"model: Qwen/Qwen3.5-4B-Base\nmodel_revision: 1001bb4d\ntrain_kwargs:\n  default:\n"
        f"    heldout_pool: /vol/pool/{_KEY}/pairs_constv2-heldout.jsonl\n{seed_line}")
    (repo / "scripts" / "sdfbase" / "seedrep").mkdir(parents=True)
    for n in ("psr_residual_distill.py", "lora_subtract.py", "gate_liveness.py"):
        (repo / "scripts" / "sdfbase" / n).write_text("# placeholder: the faked venv python dispatches on the name\n")
    _shutil.copy(_SEEDREP / "derived_sweep.py", repo / "scripts" / "sdfbase" / "seedrep" / "derived_sweep.py")
    (repo / "scripts" / "sbatch").mkdir()
    text = (REPO / "scripts" / "sbatch" / "sdfbase_derived_build.sbatch").read_text()
    script = repo / "scripts" / "sbatch" / "sdfbase_derived_build.sbatch"
    script.write_text(text)
    binp.mkdir()
    stubs = {
        "git": f'#!/bin/bash\ncase "$*" in *rev-parse*) echo {_STUB_SHA} ;; *) : ;; esac\n',
        "scontrol": (f"#!/bin/bash\nprintf '   Comment=sha:%s \\n   Command=%s\\n   TimeLimit=06:00:00\\n"
                     f"   StdOut=%s/job.out\\n' {_STUB_SHA} {script} {tmp_path}\n"),
        "sbatch": (f'#!/bin/bash\nprintf "%s\\n" "$*" >> {tmp_path}/sbatch.log\n'
                   f'printf "ENV=%s\\n" "${{SDFBASE_FIT_ENV_NAME:-}}" >> {tmp_path}/sbatch.log\n'),
        "nvidia-smi": "#!/bin/bash\nexit 0\n",
    }
    for name, body in stubs.items():
        (binp / name).write_text(body)
        (binp / name).chmod(0o755)
    env = {k: v for k, v in _os.environ.items()
           if not k.startswith(("SLURM_", "SDFBASE_", "FAKE_", "ESCALATE_", "SISTEER_CLUSTER_QOS"))}
    env.update(USER="nobodyx", SLURM_JOB_ID="4242", SLURM_JOB_NAME="reseed-derived-test", SISTEER_SHARED_ROOT=str(shared),
               SDFBASE_REPO=str(repo), FAKE_LOG=str(tmp_path / "fake.log"),
               FAKE_SCORES=json.dumps(scores or _DEFAULT_SCORES),
               PATH=f"{binp}:{env['PATH']}")
    resid = root / "derived" / "psr_residual" / _KEY / f"{MIX}__psr_ce_all_minus_ctl_distilled"
    return _NS(shared=shared, repo=repo, script=script, env=env, resid=resid, root=root,
               pool=pool, venv=venv, figs=shared / "exp" / "figs" / "reseed-derived-test",
               fake_log=tmp_path / "fake.log", sbatch_log=tmp_path / "sbatch.log", tmp=tmp_path)


def _stage(w, stage, *extra, task=None, env_extra=None):
    env = dict(w.env)
    if task is not None:
        env["SLURM_ARRAY_TASK_ID"] = str(task)
    env.update(env_extra or {})
    return _sp.run(["bash", str(w.script), stage, _KEY, MIX, CTL, _BANK, *extra],
                   capture_output=True, text=True, env=env, timeout=120)


def _fake_lines(w, prefix):
    return [l for l in w.fake_log.read_text().splitlines() if l.startswith(prefix)]


def _prep_and_calibrate(w):
    r = _stage(w, "prep")
    assert r.returncode == 0, r.stdout + r.stderr
    for t in (0, 1, 2):
        r = _stage(w, "calibrate", "-", task=t)
        assert r.returncode == 0, r.stdout + r.stderr


def test_sbatch_prep_probes_the_paired_writes_liveness_and_keeps_a_present_file(tmp_path):
    """V IS the difference_relu object, so prep owns its liveness gate (seedrep/derived_admission.py
    refuses a psr_paired_write without gate_liveness.json; only the legacy finish stage used to write it). prep runs
    gate_liveness.py over V alone (never the residual) and a second prep keeps the present file."""
    w = _derived_world(tmp_path)
    vdir = w.root / "derived" / "psr_paired_write" / _KEY / f"{MIX}__psr_ce_all_minus_ctl"
    r = _stage(w, "prep")
    assert r.returncode == 0, r.stdout + r.stderr
    assert (vdir / "gate_liveness.json").is_file()
    live = _fake_lines(w, "gate_liveness.py")
    assert len(live) == 1 and f"--dirs {vdir}" in live[0] and "--n-probe 32" in live[0] and str(w.resid) not in live[0], live
    # the probe's result is BOUND to V's bytes by the receipt, and the record is filed by emit-v only after it
    receipt = json.loads((vdir / "gate_liveness_receipt.json").read_text())
    assert receipt["object_sha256"] == hashlib.sha256(b"V").hexdigest()
    log = w.fake_log.read_text().splitlines()
    i_compose = next(i for i, l in enumerate(log) if l.startswith("psr_residual_distill.py compose-v"))
    i_probe = next(i for i, l in enumerate(log) if l.startswith("gate_liveness.py"))
    i_emit = next(i for i, l in enumerate(log) if l.startswith("psr_residual_distill.py emit-v"))
    assert "--no-emit" in log[i_compose] and i_compose < i_probe < i_emit, log
    assert (vdir / "registry_emission.json").is_file()
    r = _stage(w, "prep")
    assert r.returncode == 0 and "liveness kept" in r.stdout
    assert len(_fake_lines(w, "gate_liveness.py")) == 1                       # not re-probed
    assert len(_fake_lines(w, "psr_residual_distill.py emit-v")) == 2         # the idempotent replay runs every prep
    # V recomposed to OTHER bytes: the kept probe is bound to the old bytes -> re-probed, receipt refreshed
    (vdir / "object.pt").write_bytes(b"V2")
    r = _stage(w, "prep")
    assert r.returncode == 0 and "re-probing" in r.stdout, r.stdout + r.stderr
    assert len(_fake_lines(w, "gate_liveness.py")) == 2
    assert json.loads((vdir / "gate_liveness_receipt.json").read_text())["object_sha256"] == hashlib.sha256(b"V2").hexdigest()
    # the PROBE FILE changed (edited / a stale re-run): the receipt's probe sha no longer matches -> re-probed too, never
    # a stale checksum that emit-v then rejects
    gl = json.loads((vdir / "gate_liveness.json").read_text()); gl["9"] = 0.5
    (vdir / "gate_liveness.json").write_text(json.dumps(gl))
    r = _stage(w, "prep")
    assert r.returncode == 0 and "re-probing" in r.stdout and "receipt probe" in r.stdout, r.stdout + r.stderr
    assert len(_fake_lines(w, "gate_liveness.py")) == 3
    rcpt = json.loads((vdir / "gate_liveness_receipt.json").read_text())
    assert rcpt["gate_liveness_sha256"] == hashlib.sha256((vdir / "gate_liveness.json").read_bytes()).hexdigest() and "9" not in json.loads((vdir / "gate_liveness.json").read_text())


def test_sbatch_prep_direct_build_skips_the_legacy_scaffolding_and_files_no_record_when_the_probe_fails(tmp_path):
    """The front door's direct-object build (positional 10 = direct) runs no legacy capture-set / ladder
    declaration, and a failure between compose-v and the probe leaves NO registry record and NO receipt — so the front door cannot read
    it as done; the chain build (default) still declares the ladder."""
    w = _derived_world(tmp_path)
    vdir = w.root / "derived" / "psr_paired_write" / _KEY / f"{MIX}__psr_ce_all_minus_ctl"
    r = _stage(w, "prep", "-", "1", "3", "relu", "direct", env_extra={"FAKE_LIVENESS_FAIL": "1"})
    assert r.returncode == 17, r.stdout + r.stderr
    assert (vdir / "object.pt").is_file() and not (vdir / "gate_liveness_receipt.json").exists() and not (vdir / "registry_emission.json").exists()
    assert not _fake_lines(w, "psr_residual_distill.py emit-v") and not _fake_lines(w, "psr_residual_distill.py capture-set")
    assert not (w.resid / "sweep.json").exists() and "build=direct" in r.stdout
    # the probe MEASURES a fully dead layer (and exits 0): the receipt records it, emit-v REFUSES, the job FAILS visibly (rc 19) and
    # V stays unpublished — no composed row the front door could read as done
    r = _stage(w, "prep", "-", "1", "3", "relu", "direct", env_extra={"FAKE_LIVENESS_DEAD": "1"})
    assert r.returncode == 19, r.stdout + r.stderr
    rcpt = json.loads((vdir / "gate_liveness_receipt.json").read_text())
    assert rcpt["not_fully_dead"] is False and rcpt["dead_layers"] == ["2"] and not (vdir / "registry_emission.json").exists()
    assert "emit-v refused" in r.stdout and "UNPUBLISHED" in r.stdout and "RESULT: FAILED rc=19" in r.stdout
    # a re-run keeps the (bound) dead probe and fails at emit-v again — visible, no re-probe churn, still unpublished
    r = _stage(w, "prep", "-", "1", "3", "relu", "direct", env_extra={"FAKE_LIVENESS_DEAD": "1"})
    assert r.returncode == 19 and "liveness kept" in r.stdout and not (vdir / "registry_emission.json").exists(), r.stdout + r.stderr
    # a live probe (V re-probed because the dead probe file is replaced by hand here, as an operator would after a fix) publishes
    (vdir / "gate_liveness.json").unlink(); (vdir / "gate_liveness_receipt.json").unlink()
    r = _stage(w, "prep", "-", "1", "3", "relu", "direct")
    assert r.returncode == 0, r.stdout + r.stderr
    assert (vdir / "gate_liveness_receipt.json").is_file() and (vdir / "registry_emission.json").is_file()
    assert json.loads((vdir / "gate_liveness_receipt.json").read_text())["not_fully_dead"] is True
    assert not _fake_lines(w, "psr_residual_distill.py capture-set") and not (w.resid / "sweep.json").exists()
    assert "RESULT: DONE" in r.stdout and "build=direct" in r.stdout
    # the chain build still declares the ladder (the legacy stages read it)
    w2 = _derived_world(tmp_path / "chain")
    r = _stage(w2, "prep")
    assert r.returncode == 0 and (w2.resid / "sweep.json").is_file() and _fake_lines(w2, "psr_residual_distill.py capture-set"), r.stdout + r.stderr
    # a stray build selector refuses by name
    r = _stage(w, "prep", "-", "1", "3", "relu", "sideways")
    assert r.returncode == 2 and "positional 10 build must be chain | direct" in r.stdout + r.stderr


def test_sbatch_assess_records_the_selection_without_distilling_and_merges_the_old_ladder_byte_for_byte(tmp_path):
    w = _derived_world(tmp_path)
    _prep_and_calibrate(w)
    sweep = json.loads((w.resid / "sweep.json").read_text())
    assert sweep["lr_mults"] == [0.3, 1.0, 3.0] and sweep["extensions"] == [] and sweep["declared_by"] == "prep"
    assert (w.resid / "capture_manifest.json").exists()
    for m in ("0.3", "1", "3"):
        assert (w.resid / "calib_parts" / f"lrmult{m}" / "calibration_controls.json").is_file()
    r = _stage(w, "assess")
    assert r.returncode == 0, r.stdout + r.stderr
    a = json.loads((w.resid / "assess.json").read_text())
    assert a["selected_lr_mult"] == 1.0 and a["on_edge"] is False and a["edge_side"] is None
    assert a["recommended_extension"] is None and a["alarm"] is None and a["ladder"] == [0.3, 1.0, 3.0]
    assert set(a["parts_sha256"]) == {"0.3", "1", "3"} and len(a["calibration_sha256"]) == 64
    assert not (w.resid / "object.pt").exists() and not (w.resid / "derived_meta.json").exists()
    assert (w.resid / "calibration_controls.json").is_file()
    assert "DONE rc=0 stage=assess" in (w.figs / "RESULT").read_text()
    assert (w.figs / "assess_4242.json").is_file()
    # the merge argv is the OLD finish's exactly: SWEEP=(0.3 1 3) -> --lr-mult 0.3,1,3, parts in that order
    merges = _fake_lines(w, "psr_residual_distill.py merge-calibration")
    parts = ",".join(str(w.resid / "calib_parts" / f"lrmult{m}" / "calibration_controls.json") for m in ("0.3", "1", "3"))
    assert merges == [f"psr_residual_distill.py merge-calibration --root {w.root} --out-root {w.root}/derived "
                      f"--pools {w.pool} --model-key {_KEY} --mix-si {MIX} --ctl-si {CTL} --lr-mult 0.3,1,3 "
                      f"--parts {parts} --out {w.resid}/calibration_controls.json"]
    # finish: distills at the assessed multiplier with --seed 0 --holdout-frac 0.1 (the defaults, spelled out)
    r = _stage(w, "finish")
    assert r.returncode == 0, r.stdout + r.stderr
    meta = json.loads((w.resid / "derived_meta.json").read_text())
    assert meta["trainer"]["lr_mult"] == 1.0 and meta["trainer"]["seed"] == 0
    distills = _fake_lines(w, "psr_residual_distill.py distill")
    assert len(distills) == 1 and "--lr-mult" not in distills[0]
    assert distills[0] == (f"psr_residual_distill.py distill --root {w.root} --out-root {w.root}/derived --pools {w.pool} "
                           f"--model-key {_KEY} --mix-si {MIX} --ctl-si {CTL} --model Qwen/Qwen3.5-4B-Base --revision 1001bb4d "
                           f"--config {w.repo}/configs/sdfbase_{_KEY}_fits.yaml --capture-budget-tokens 100000 "
                           f"--seed 0 --holdout-frac 0.1 --distill-epochs-cap 60 --distill-patience 5 --checkpoint-every 50")
    for line in _fake_lines(w, "psr_residual_distill.py capture-set") + _fake_lines(w, "psr_residual_distill.py calibrate"):
        assert "--seed 0 --holdout-frac 0.1" in line
    assert (w.resid / "gate_liveness.json").is_file()
    result = (w.figs / "RESULT").read_text()
    assert "DONE rc=0 stage=finish" in result and f" pair={_KEY}/{MIX}-{CTL} seed=0 bank={_BANK} " in result
    rec = json.loads((w.figs / "distilled_fit_record.txt").read_text())
    assert rec["seed"] == 0 and rec["selected_lr_mult"] == 1.0 and rec["ladder"] == [0.3, 1.0, 3.0]
    assert rec["on_edge"] is False and rec["extensions"] == []


def test_sbatch_edge_selection_extension_survives_an_interruption_and_a_second_extension_refuses(tmp_path):
    w = _derived_world(tmp_path, scores=_EDGE_SCORES)
    _prep_and_calibrate(w)
    r = _stage(w, "assess")
    assert r.returncode == 0, r.stdout + r.stderr
    a = json.loads((w.resid / "assess.json").read_text())
    assert a["selected_lr_mult"] == 0.3 and a["on_edge"] is True and a["edge_side"] == "below"
    assert a["recommended_extension"] == 0.1 and a["alarm"] is None
    # the extension rung: declared, then TERMed mid-part (a preemption) -> self-resubmit recorded
    r = _stage(w, "calibrate", "0.1", env_extra={"FAKE_TERM_ON_MULT": "0.1"})
    assert r.returncode == 143, r.stdout + r.stderr
    assert "SIGTERM" in r.stdout and "resubmitting" in r.stdout
    sweep = json.loads((w.resid / "sweep.json").read_text())
    assert sweep["extensions"] == [0.1] and sweep["extension_events"][0]["declared_by"] == "calibrate"
    assert not (w.resid / "calib_parts" / "lrmult0.1" / "calibration_controls.json").exists()
    resub = w.sbatch_log.read_text().splitlines()
    assert len(resub) == 2 and resub[1] == "ENV=sdfbase-fit"
    assert resub[0].endswith(f"{w.script} calibrate {_KEY} {MIX} {CTL} {_BANK} 0.1 2 3 both chain")     # positional 9 (objects) + 10 (build) ride the successor
    assert f"--output={w.tmp}/job.out" in resub[0] and "--qos=high" in resub[0] and "--job-name=reseed-derived-test" in resub[0]
    assert "INTERRUPTED rc=143 stage=calibrate objects=both build=chain lr_mult=0.1" in (w.figs / "RESULT").read_text()
    # the successor, exactly as escalate() invoked it: the declaration is a no-op, the part completes
    r = _stage(w, "calibrate", "0.1", "2", "3", env_extra={"FAKE_TERM_ON_MULT": "0.1"})
    assert r.returncode == 0, r.stdout + r.stderr
    assert "extension 0.1 already declared (no-op)" in r.stdout
    assert json.loads((w.resid / "sweep.json").read_text())["extensions"] == [0.1]
    assert (w.resid / "calib_parts" / "lrmult0.1" / "calibration_controls.json").is_file()
    # assess2 merges exactly the four parts and records the (still-edge) selection with its alarm
    r = _stage(w, "assess")
    assert r.returncode == 0, r.stdout + r.stderr
    a = json.loads((w.resid / "assess.json").read_text())
    assert a["ladder"] == [0.1, 0.3, 1.0, 3.0] and a["extensions"] == [0.1]
    assert a["selected_lr_mult"] == 0.1 and a["on_edge"] is True and a["recommended_extension"] is None
    assert "persists after the one D7 extension" in a["alarm"]
    assert _fake_lines(w, "psr_residual_distill.py merge-calibration")[-1].split("--lr-mult ")[1].startswith("0.1,0.3,1,3 ")
    assert len((w.resid / "assess_history.jsonl").read_text().splitlines()) == 2
    # finish distills at 0.1
    r = _stage(w, "finish")
    assert r.returncode == 0, r.stdout + r.stderr
    assert json.loads((w.resid / "derived_meta.json").read_text())["trainer"]["lr_mult"] == 0.1
    assert "distilling at the ASSESSED lr_mult=0.1" in r.stdout
    # a conflicting SECOND extension refuses before any part is calibrated
    r = _stage(w, "calibrate", "10")
    assert r.returncode == 18, r.stdout + r.stderr
    assert "a second extension (10) is not allowed" in r.stdout + r.stderr
    assert json.loads((w.resid / "sweep.json").read_text())["extensions"] == [0.1]
    assert not (w.resid / "calib_parts" / "lrmult10").exists()
    assert "FAILED rc=18 stage=calibrate objects=both build=chain lr_mult=10" in (w.figs / "RESULT").read_text()


def test_sbatch_finish_refuses_without_or_with_a_stale_assessment_and_assess_refuses_a_stray_part(tmp_path):
    w = _derived_world(tmp_path)
    _prep_and_calibrate(w)
    r = _stage(w, "finish")
    assert r.returncode == 15, r.stdout + r.stderr
    assert "finish refused" in r.stdout and not (w.resid / "object.pt").exists()
    assert not _fake_lines(w, "psr_residual_distill.py distill")
    r = _stage(w, "assess")
    assert r.returncode == 0, r.stdout + r.stderr
    # a part rewritten after the assessment: finish refuses (sha pin, not mtime)
    part = w.resid / "calib_parts" / "lrmult3" / "calibration_controls.json"
    doc = json.loads(part.read_text())
    doc["sweep"]["per_mult"]["3"]["score_worst_stratum_attainable_ed"] = 0.99
    part.write_text(json.dumps(doc))
    r = _stage(w, "finish")
    assert r.returncode == 15 and "changed after the assessment" in r.stdout + r.stderr
    assert not (w.resid / "object.pt").exists()
    # a stray fifth part dir outside the declaration: assess refuses to merge around it
    stray = w.resid / "calib_parts" / "lrmult10" / "calibration_controls.json"
    stray.parent.mkdir()
    stray.write_text(part.read_text())
    r = _stage(w, "assess")
    assert r.returncode == 15 and "stray calibration part dir" in r.stdout + r.stderr


def test_sbatch_seed_from_the_config_reaches_capture_calibrate_distill_and_the_result_line(tmp_path):
    w = _derived_world(tmp_path, seed_line="    seed: 2\n")
    _prep_and_calibrate(w)
    assert _stage(w, "assess").returncode == 0
    r = _stage(w, "finish")
    assert r.returncode == 0, r.stdout + r.stderr
    for prefix in ("psr_residual_distill.py capture-set", "psr_residual_distill.py calibrate",
                   "psr_residual_distill.py distill"):
        lines = _fake_lines(w, prefix)
        assert lines and all("--seed 2 --holdout-frac 0.1" in l for l in lines), prefix
    assert json.loads((w.resid / "capture_manifest.json").read_text()) == {"seed": 2, "holdout_frac": 0.1}
    assert " seed=2 bank=" in (w.figs / "RESULT").read_text()
    assert json.loads((w.figs / "distilled_fit_record.txt").read_text())["seed"] == 2
    assert " seed=2 " in r.stdout.splitlines()[0]     # the preflight log line names it too


def test_sbatch_refuses_a_bad_stage_and_finish_on_an_undeclared_tree(tmp_path):
    w = _derived_world(tmp_path)
    r = _stage(w, "merge")
    assert r.returncode == 2 and "unknown stage" in r.stdout
    # a tree prepped BEFORE the declaration existed: assess/finish name the fix (re-run prep)
    w.resid.mkdir(parents=True)
    r = _stage(w, "assess")
    assert r.returncode == 15 and "no ladder declaration" in r.stdout + r.stderr


# ---- derived_sweep unit pins ---------------------------------------------------------

def test_derived_sweep_format_mult_matches_the_distiller_and_declaration_semantics(tmp_path):
    import psr_residual_distill as prd
    for m in (0.3, 1, 1.0, 3, 0.1, 10, 10.0, 0.25):
        assert _ds.format_mult(m) == prd.format_mult(m)
    assert _ds.ladder_spec(_ds.DEFAULT_LR_MULTS) == "0.3,1,3"
    assert tuple(_ds.DEFAULT_LR_MULTS) == tuple(prd.DEFAULT_LR_MULTS)
    resid = tmp_path / "resid"
    resid.mkdir()
    _ds.declare(resid)
    _ds.declare(resid)                                  # identical: no-op
    with pytest.raises(_ds.SweepError, match="declared once"):
        _ds.declare(resid, "0.3,1,3,10")
    assert _ds.ensure_extension(resid, 1) == "base"
    assert _ds.ensure_extension(resid, 0.1) == "appended"
    assert _ds.ensure_extension(resid, 0.1) == "declared"
    with pytest.raises(_ds.SweepError, match="second extension"):
        _ds.ensure_extension(resid, 10)
    assert _ds.ladder_of(_ds.load_sweep(resid)) == [0.1, 0.3, 1.0, 3.0]
    _ds.declare(resid)                                  # a re-run prep keeps the extension
    assert _ds.load_sweep(resid)["extensions"] == [0.1]
    # a pre-declaration tree: a base rung proceeds, an extension has nowhere to go
    bare = tmp_path / "bare"
    bare.mkdir()
    assert _ds.ensure_extension(bare, 3, default_ladder="0.3,1,3") == "undeclared-base"
    with pytest.raises(_ds.SweepError, match="nowhere to be declared"):
        _ds.ensure_extension(bare, 0.1, default_ladder="0.3,1,3")
    with pytest.raises(_ds.SweepError, match="never declared"):
        _ds.check_finish(bare)
