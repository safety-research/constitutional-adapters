"""The derived-objects Modal app (scripts/modal/sdfbase_derived.py) under
the duck-typed fake `modal` (tests/test_watch_successor.py's stub): it is
a SEPARATE app named by the commit (never ``sdfbase-fits``), its command
builders pass /repo scripts + /vol paths end to end (scale + identity
gate + parents root on every object; the bare row without an object),
derived ids resolve across the derived/ namespaces and refuse
duplicates, and the entrypoint's modes/scales validate. Every derived
object is scored with --parents-root; the two new modes
(mmlu, snapshot-verify) write only under this app's derived/ namespace,
never a cells_all/ or hpo bank; parent discovery skips
the object.pt.provenance.json sidecar (the 9aa39f9 harvest fix, same site);
and the argv ce_command emits produces, end to end through ce_score.main,
exactly the combined CE files derived_verdict.main places on the
registered control held-out pool.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import sys
from pathlib import Path

import pytest

from test_watch_successor import make_fake_modal  # the shared `modal` stub

REPO = Path(__file__).resolve().parents[1]
SRC_PATH = REPO / "scripts" / "modal" / "sdfbase_derived.py"
SRC = SRC_PATH.read_text()
if str(REPO / "scripts" / "sdfbase") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts" / "sdfbase"))
TINY = "hf-internal-testing/tiny-random-LlamaForCausalLM"


@pytest.fixture(scope="module")
def sd():
    # ALWAYS this suite's own stub: another module may have left ITS fake `modal`
    # in sys.modules (the grokmt lineage installs one whose Image has no
    # debian_slim), and keeping it made every test here ERROR at setup on the
    # joined sdfbase-mt tree (2026-09-04). Whatever was there is restored after.
    prev_modal = sys.modules.get("modal")
    sys.modules["modal"] = make_fake_modal()
    sys.modules.pop("sdfbase_fits", None)
    # the W&B secret's name is a site setting sdfbase_fits reads at import: a synthetic one here
    prev_secret = os.environ.get("SISTEER_MODAL_WANDB_SECRET")
    os.environ["SISTEER_MODAL_WANDB_SECRET"] = "example-wandb-secret"
    spec = importlib.util.spec_from_file_location("_sdfbase_derived_test",
                                                  SRC_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_sdfbase_derived_test"] = mod
    spec.loader.exec_module(mod)
    try:
        yield mod
    finally:
        if prev_modal is None:
            sys.modules.pop("modal", None)
        else:
            sys.modules["modal"] = prev_modal
        if prev_secret is None:
            os.environ.pop("SISTEER_MODAL_WANDB_SECRET", None)
        else:
            os.environ["SISTEER_MODAL_WANDB_SECRET"] = prev_secret


def _d(cmd):
    return dict(zip(cmd, cmd[1:]))


def test_default_key_sweep_excludes_derived_ineligible_keys(sd):
    """The derived app takes MODELS from the fits table and swept every
    non-probe_only key — so the two instruct-direct `_it_doc` lanes
    (mix-full only, no mix-100k / ctl-100k parents: no derived operation
    is valid for them) would have joined every bare invocation.
    derived_eligible=False excludes them from the default AND from an
    explicit --model, by the same list; the four existing fit lanes are
    untouched."""
    for key in ("gemma4e4b_it_doc", "qwen35_4b_it_doc"):
        assert sd.MODELS[key]["derived_eligible"] is False
        with pytest.raises(SystemExit) as ei:
            sd.main(app_name=sd.APP_NAME, mode="identity", model=key)
        msg = str(ei.value)
        assert "must be a base key" in msg
        assert "_it_doc" not in msg  # the default list names neither
        for k in ("gemma4e4b_base", "qwen35_4b_base", "glm46vflash_doc",
                  "nemotron3nano4b_doc"):
            assert k in msg
    src = SRC.split("def main(", 1)[1]
    assert 's.get("derived_eligible", True)' in src


def test_separate_versioned_app_name(sd):
    assert sd.APP_NAME != "sdfbase-fits"
    assert re.fullmatch(r"sdfbase-derived-([0-9a-f]{16}|local)", sd.APP_NAME)
    assert sd.app.name == sd.APP_NAME
    # the commit pin is baked into BOTH images before the local-dir layer
    assert sd._ENV_PIN == {"SDFBASE_DERIVED_SHA16": sd.SHA16}
    # the fit app is imported for specs, never redefined
    assert sd.MODELS is sd.base.MODELS and sd.vol is sd.base.vol


def test_residual_command_is_the_in_container_all_pipeline(sd):
    cmd = sd.residual_command("gemma4e4b_base", "constv2-mix-100k",
                              "constv2ctl-100k", no_production=False,
                              lr_mult=3.0, wandb=True)
    assert cmd[1] == "/repo/scripts/sdfbase/psr_residual_distill.py"
    assert cmd[2] == "all"
    d = _d(cmd)
    assert d["--root"] == "/vol/out"
    assert d["--out-root"] == "/vol/out/derived"
    assert d["--pools"] == "/vol/pool/gemma4e4b_base"
    assert d["--model"] == sd.MODELS["gemma4e4b_base"]["repo"]
    assert d["--revision"] == sd.MODELS["gemma4e4b_base"]["rev"]
    assert d["--config"] == sd.MODELS["gemma4e4b_base"]["cfg"]
    assert d["--lr-mult"] == "3.0" and "--wandb" in cmd
    assert "--no-production" not in cmd
    cal = sd.residual_command("qwen35_4b_base", "constv2-mix-100k",
                              "constv2ctl-100k", no_production=True,
                              lr_mult=1.0, wandb=False)
    assert "--no-production" in cal and "--wandb" not in cal
    assert "--distill-epochs-cap" not in cal      # only when given
    # the sweep string rides through verbatim, the budget explicitly
    sw = sd.residual_command("gemma4e4b_base", "constv2-mix-100k",
                             "constv2ctl-100k", no_production=True,
                             lr_mult="0.3,1,3", wandb=True,
                             distill_epochs_cap=60, distill_patience=5)
    ds = _d(sw)
    assert ds["--lr-mult"] == "0.3,1,3"
    assert ds["--distill-epochs-cap"] == "60" and ds["--distill-patience"] == "5"
    # the entrypoint's defaults ARE the budget + the sweep, and the
    # spawned item carries them into the container
    entry = SRC.split("def main(", 1)[1]
    assert 'lr_mults: str = "0.3,1,3"' in entry
    assert "distill_epochs_cap: int = 60" in entry
    assert "distill_patience: int = 5" in entry
    assert "int(distill_epochs_cap), int(distill_patience)" in entry
    for tok in [t for t in cmd[1:] if str(t).startswith("/")]:
        assert tok.startswith(("/vol/", "/repo/")), tok


def test_probe_and_ce_commands_carry_scale_identity_and_volume_paths(sd):
    cmd, out = sd.probe_command("gemma4e4b_base", "/vol/out/derived/x/object.pt",
                                "constv2-mix-100k__psr_ce_all_minus_ctl",
                                scale=2.0, ver="v2")
    d = _d(cmd)
    assert cmd[1] == "/repo/scripts/sdfbase/probe_score.py" and cmd[2] == "score"
    assert d["--probe"] == "/repo/data/sdfbase/probe_v2.jsonl"
    assert d["--object"] == "/vol/out/derived/x/object.pt"
    assert d["--cell"] == "constv2-mix-100k__psr_ce_all_minus_ctl"
    assert d["--scale"] == "2.0"
    # a derived object is scored with its parents re-verified against
    # the live bank (probe_score refuses without it)
    assert d["--parents-root"] == "/vol/out" == sd.PARENTS_ROOT
    # the analysis contract: probe_v2_x2/<model>/<cell>.jsonl, the derived
    # id AS the cell name (no prefix)
    assert out == ("/vol/out/gemma4e4b_base/probe_v2_x2/"
                   "constv2-mix-100k__psr_ce_all_minus_ctl.jsonl")
    cmd1, out1 = sd.probe_command("gemma4e4b_base", "/vol/o.pt", "id",
                                  scale=1.0, ver="v2")
    assert "--scale" not in cmd1 and out1.endswith("/probe_v2/id.jsonl")
    # the registry binds file and dir for every probe version — the
    cmd3, out3 = sd.probe_command("gemma4e4b_base", "/vol/o.pt", "id",
                                  scale=1.0, ver="v3g")
    d3 = _d(cmd3)
    assert d3["--probe"] == "/repo/data/sdfbase/probe_v3_genre.jsonl"
    assert d3["--manifest"] == "/repo/data/sdfbase/probe_v3_genre.manifest.json"
    assert out3 == "/vol/out/gemma4e4b_base/probe_v3/id.jsonl"
    cmd3x, out3x = sd.probe_command("gemma4e4b_base", "/vol/o.pt", "id",
                                    scale=2.0, ver="v3g")
    assert out3x.endswith("/probe_v3_x2/id.jsonl") and "--scale" in cmd3x
    with pytest.raises(SystemExit, match="unknown probe"):
        sd.probe_command("gemma4e4b_base", "/vol/o.pt", "id", scale=1.0, ver="v3")
    assert 'probe_{ver}.jsonl' not in Path(sd.__file__).read_text()   # no private spelling of the frozen path
    # CE: object rows carry --object --cell --parents-root + --scale + --pool,
    # filed in the analysis shape ce_score/<model>/<cell>__<pool>__scale<s>.json
    cmd, out = sd.ce_command("qwen35_4b_base", "/vol/out/derived/y/object.pt",
                             "yid", "control-heldout",
                             "/vol/pool/qwen35_4b_base/pairs_constv2ctl-heldout.jsonl",
                             scale=2.0, source_tag="ctl")
    d = _d(cmd)
    assert cmd[1] == "/repo/scripts/sdfbase/ce_score.py"
    assert d["--object"] == "/vol/out/derived/y/object.pt"
    assert d["--cell"] == "yid" and d["--parents-root"] == "/vol/out"
    assert d["--scale"] == "2.0" and d["--source-tag"] == "ctl"
    assert d["--pool"] == "control-heldout"
    assert d["--pairs"].startswith("/vol/pool/")
    assert out == "/vol/out/ce_score/qwen35_4b_base/yid__control-heldout__scale2.json"
    assert d["--out"] == out
    # the bare row: no object, no cell, no parents root
    cmd, out = sd.ce_command("qwen35_4b_base", None, "bare", "constv2-heldout",
                             "/vol/pool/qwen35_4b_base/pairs_constv2-heldout.jsonl",
                             scale=1.0)
    assert "--object" not in cmd and "--cell" not in cmd
    assert out.endswith("/ce_score/qwen35_4b_base/bare__constv2-heldout__scale1.json")
    for tok in [t for t in cmd[1:] if str(t).startswith("/")]:
        assert tok.startswith(("/vol/", "/repo/")), tok


def test_liveness_command_and_gatereg_trial_dirs(sd, tmp_path):
    cmd = sd.liveness_command("gemma4e4b_base", ["/vol/out/hpo/x/r3_t00",
                                                 "/vol/out/hpo/x/r3_t01"],
                              pairs_path="/vol/pool/gemma4e4b_base/pairs_constv2-mix-1m.jsonl",
                              n_probe=32, allow_bare=True)
    d = _d(cmd)
    assert cmd[1] == "/repo/scripts/sdfbase/gate_liveness.py"
    assert d["--dirs"] == "/vol/out/hpo/x/r3_t00,/vol/out/hpo/x/r3_t01"
    assert d["--n-probe"] == "32" and "--allow-bare" in cmd
    assert d["--model"] == sd.MODELS["gemma4e4b_base"]["repo"]
    cmd2 = sd.liveness_command("gemma4e4b_base", ["/vol/out/derived/psr_residual/k/a"],
                               pairs_path="/vol/pool/k/pairs_constv2-mix-100k.jsonl",
                               n_probe=8, allow_bare=False)
    assert "--allow-bare" not in cmd2
    # trial dirs: only r3_t* with BOTH object.pt and summary.json
    root = tmp_path / "hpo/{key}/psr_ce_all_all"
    for name, files in (("r3_t00", ("object.pt", "summary.json")),
                        ("r3_t01", ("summary.json",)),
                        ("r1_t00", ("object.pt", "summary.json")),
                        ("r3_t02", ("object.pt", "summary.json"))):
        dd = tmp_path / "hpo/gemma4e4b_base/psr_ce_all_all" / name
        dd.mkdir(parents=True)
        for f in files:
            (dd / f).write_text("{}")
    got = sd.gatereg_trial_dirs("gemma4e4b_base", root=str(root))
    assert [Path(p).name for p in got] == ["r3_t00", "r3_t02"]
    assert sd.gatereg_trial_dirs("qwen35_4b_base", root=str(root)) == []
    assert sd.GATEREG_TRIALS == "/vol/out/hpo/{key}/psr_ce_all_all"
    assert sd.GATEREG_SI == "constv2-mix-1m" and sd.GATEREG_ROUND == 3


def test_derived_object_dirs_resolve_and_refuse_duplicates(sd, tmp_path):
    root = tmp_path / "derived"
    for kind, oid in (("psr_paired_write", "a__psr_ce_all_minus_ctl"),
                      ("psr_residual", "a__psr_ce_all_minus_ctl_distilled"),
                      ("lora_subtraction", "a__lora_r4_ce_minus_ctl")):
        d = root / kind / "k" / oid
        d.mkdir(parents=True)
        (d / "object.pt").write_bytes(b"x")
    (root / "psr_residual/k/no_object").mkdir()
    found = sd.derived_object_dirs("k", root=str(root))
    assert set(found) == {"a__psr_ce_all_minus_ctl",
                          "a__psr_ce_all_minus_ctl_distilled",
                          "a__lora_r4_ce_minus_ctl"}
    assert found["a__lora_r4_ce_minus_ctl"].endswith(
        "lora_subtraction/k/a__lora_r4_ce_minus_ctl/object.pt")
    dup = root / "lora_subtraction/k/a__psr_ce_all_minus_ctl"
    dup.mkdir()
    (dup / "object.pt").write_bytes(b"y")
    with pytest.raises(SystemExit, match="two kinds"):
        sd.derived_object_dirs("k", root=str(root))
    assert sd.derived_object_dirs("other", root=str(root)) == {}


def test_ce_pools_and_parents_defaults(sd):
    # the analysis pipeline's pool names (analysis_r2/r2_paths.py CE_POOLS)
    assert set(sd.CE_POOLS) == {"constv2-heldout", "control-heldout",
                                "constv2ctl-100k"}
    assert sd.CE_POOLS["control-heldout"].endswith("pairs_constv2ctl-heldout.jsonl")
    assert sd.CE_ROOT == "/vol/out/ce_score"
    for p in sd.CE_POOLS.values():
        assert p.startswith("/vol/pool/{key}/pairs_")
    assert "constv2-mix-100k__psr_ce_all" in sd.PARENT_CELLS
    assert "constv2ctl-100k__psr_ce_all" in sd.PARENT_CELLS
    assert sd.parse_scales("1.0,2.0") == [1.0, 2.0]
    with pytest.raises(SystemExit):
        sd.parse_scales("")


def test_operator_wandb_secret_and_live_logging_env(sd, monkeypatch):
    """The shared workspace `wandb` secret (another member's key —
    this app's first runs landed in a foreign entity) is NOT mounted; the
    operator's own secret, the one the site setting SISTEER_MODAL_WANDB_SECRET
    names (WANDB_API_KEY + WANDB_ENTITY + WANDB_PROJECT), is, next to the HF
    token. The subprocess env inherits
    that routing untouched (no offline mode, no dir override), and the
    residual container refuses at start unless the key's default entity IS
    WANDB_ENTITY (base.require_wandb_entity)."""
    assert sd._SECRETS == [("secret", "hf-token"), ("secret", "example-wandb-secret")] == sd.base._secrets
    assert sd._COMMON["secrets"] == sd._SECRETS
    assert not any(n == "wandb" for _k, n in sd._COMMON["secrets"])
    assert 'from_name("wandb")' not in SRC
    monkeypatch.setenv("WANDB_ENTITY", "example-entity")
    monkeypatch.setenv("WANDB_PROJECT", "example-project")
    for k in ("WANDB_MODE", "WANDB_DIR", "WANDB_DISABLED"):
        monkeypatch.delenv(k, raising=False)   # the grading host shell sets some
    env = sd._run_env()
    assert env["WANDB_ENTITY"] == "example-entity"
    assert env["WANDB_PROJECT"] == "example-project"
    assert "WANDB_MODE" not in env and "WANDB_DIR" not in env  # nothing ADDED
    assert env["HF_HOME"] == "/vol/hf" and env["PYTHONPATH"] == "/repo/src"
    # the residual impl guards BEFORE any model staging when W&B is on
    calls = []
    monkeypatch.setattr(sd.base, "require_wandb_entity",
                        lambda: calls.append("guard") or "ent")
    monkeypatch.setattr(sd.base, "_ensure_model",
                        lambda k: (_ for _ in ()).throw(RuntimeError("stop-here")))
    with pytest.raises(RuntimeError, match="stop-here"):
        sd._residual_impl(("gemma4e4b_base", "constv2-mix-100k",
                           "constv2ctl-100k", True, 1.0, True))
    assert calls == ["guard"]
    calls.clear()
    with pytest.raises(RuntimeError, match="stop-here"):
        sd._residual_impl(("gemma4e4b_base", "constv2-mix-100k",
                           "constv2ctl-100k", True, 1.0, False))
    assert calls == []  # W&B off → no guard, no network


def test_run_keeps_full_streams_on_the_volume(sd, tmp_path, monkeypatch):
    """`_run` used to keep stderr[-3000:] only, which lost gemma's
    layer<17 identity lines. Now the FULL stdout/stderr land as files
    (paths in the row), the printed/recorded tail stays bounded, and the
    subprocess inherits the secret's W&B routing untouched."""
    monkeypatch.setattr(sd, "LOG_ROOT", str(tmp_path / "logs"))
    monkeypatch.setattr(sd, "RUN_CWD", str(tmp_path))
    monkeypatch.setenv("WANDB_ENTITY", "example-entity")
    big = 12000
    cmd = [sys.executable, "-c",
           "import os, sys; print('WANDB_ENTITY=' + os.environ.get('WANDB_ENTITY', ''));"
           "print('HF_HOME=' + os.environ.get('HF_HOME', ''));"
           f"sys.stderr.write('layer 0: line\\n' + 'e' * {big}); sys.exit(3)"]
    row = sd._run(cmd, "residual/testmodel/testsi")
    assert row["rc"] == 3
    assert len(row["stderr_tail"]) == sd.STDERR_TAIL_CHARS
    assert "layer 0: line" not in row["stderr_tail"]        # the tail lost it ...
    err = Path(row["stderr_log"]).read_text()
    assert err.startswith("layer 0: line\n") and len(err) == big + len("layer 0: line\n")
    out = Path(row["stdout_log"]).read_text()                # ... the file has it
    # the subprocess inherits the secret's routing and the fit app's env
    assert "WANDB_ENTITY=example-entity" in out
    assert "HF_HOME=/vol/hf" in out
    assert Path(row["stderr_log"]).parent == tmp_path / "logs"
    assert Path(row["stderr_log"]).name.startswith("residual__testmodel__testsi__")
    ok = sd._run([sys.executable, "-c", "print('fine')"], "x/y")
    assert ok["rc"] == 0 and "stderr_tail" not in ok
    assert Path(ok["stdout_log"]).read_text() == "fine\n"


def test_entrypoint_refuses_unknown_mode_and_probe_only_models(sd):
    with pytest.raises(SystemExit, match="unknown --mode"):
        sd.main(app_name=sd.APP_NAME, mode="battery")
    with pytest.raises(SystemExit, match="base key"):
        sd.main(app_name=sd.APP_NAME, mode="score", model="gemma4e4b_it")
    assert sd.MODES == ("residual-distill", "calibration", "residual-distill-prod",
                        "identity", "score", "liveness", "mmlu", "snapshot-verify",
                        "lora-subtract")
    # the derived MMLU run is namespaced by a REQUIRED run label
    with pytest.raises(SystemExit, match="run-label"):
        sd.main(app_name=sd.APP_NAME, mode="mmlu", model="gemma4e4b_base")


def test_parent_discovery_ignores_the_provenance_sidecar():
    """The 9aa39f9 harvest bug at this app's own discovery site: a PSR
    parent cell carries object.pt + object.pt.provenance.json, and a bare
    glob("object.*") read two candidates and refused every PSR parent."""
    impl = SRC.split("def _score_impl", 1)[1].split("\ndef ", 1)[0]
    assert '.glob("object.*") if q.suffix != ".json"' in impl
    assert 'glob("object.*"))' not in impl  # the bare form is gone


def test_mmlu_and_snapshot_verify_write_only_under_the_derived_namespace(
        sd, tmp_path):
    """The two new modes' outputs live under this app's
    /vol/out/derived/ namespace — never the /vol/out/<key>/cells_all/
    bank, the fits app's /vol/out/<key>/mmlu/, or /vol/out/hpo/ —
    and every derived object enters MMLU as an explicit --object path with
    --parents-root (identity gate + parents re-verified). snapshot-verify
    reads the snapshot bank and writes nothing into it."""
    from test_snapshot_cells import CELL, FP, sc, stage_bank

    objs = {"a__psr_ce_all_minus_ctl":
            "/vol/out/derived/psr_paired_write/k/a__psr_ce_all_minus_ctl/object.pt",
            "a__lora_r4_ce_minus_ctl":
            "/vol/out/derived/lora_subtraction/k/a__lora_r4_ce_minus_ctl/object.pt"}
    cmd, out_dir = sd.mmlu_command("gemma4e4b_base", objs, "r2d")
    d = _d(cmd)
    assert cmd[1] == "/repo/scripts/sdfbase/base_mmlu.py"
    assert out_dir == d["--out-dir"] == "/vol/out/derived/mmlu/gemma4e4b_base"
    assert sd.MMLU_ROOT == "/vol/out/derived/mmlu"
    assert d["--run-label"] == "r2d" and d["--parents-root"] == "/vol/out"
    assert d["--cells"] == "none,a__lora_r4_ce_minus_ctl,a__psr_ce_all_minus_ctl"
    assert d["--config"] == sd.MODELS["gemma4e4b_base"]["cfg"]
    assert d["--n"] == "500" and d["--seed"] == "0"
    assert [cmd[i + 1] for i, t in enumerate(cmd) if t == "--object"] == \
        [f"{k}={v}" for k, v in sorted(objs.items())]
    assert "--defenses-root" not in cmd  # derived ids are not bank cells
    for tok in cmd[1:]:
        if str(tok).startswith("/vol/out/"):
            assert tok.startswith("/vol/out/derived/"), tok
        assert not re.match(r"/vol/out/(gemma4e4b_base|qwen35_4b_base|hpo)/",
                            str(tok)), tok
    with pytest.raises(SystemExit, match="run-label"):
        sd.mmlu_command("gemma4e4b_base", objs, "")
    with pytest.raises(SystemExit, match="no derived objects"):
        sd.mmlu_command("gemma4e4b_base", {}, "r2d")
    # snapshot-verify: the record's home is this app's namespace ...
    assert sd.SNAPSHOT_VERIFY_ROOT == "/vol/out/derived/snapshot_verify"
    assert sd.snapshot_verify_record_path("qwen35_4b_base", "20260902T000000Z") \
        == "/vol/out/derived/snapshot_verify/qwen35_4b_base/verify_20260902T000000Z.json"
    impl = SRC.split("def _snapshot_verify_impl", 1)[1].split("\ndef ", 1)[0]
    assert "snapshot_verify_record_path(key, stamp)" in impl
    assert "write_text" in impl and "dst_bank" in impl
    # ... and the verification itself is READ-ONLY on the bank: a staged
    # snapshot bank verifies (marker + manifest shas + record + the shared
    # staged-cell gate) with its file tree byte-identical afterwards
    src = stage_bank(tmp_path / "out" / "key")
    dst = tmp_path / "out" / "key_r1"
    sc.snapshot_cell(src, dst, CELL, superseded_by={"reason": "t"})

    def tree(root):
        return {str(q.relative_to(root)): hashlib.sha256(q.read_bytes()).hexdigest()
                for q in Path(root).rglob("*") if q.is_file()}

    before, before_src = tree(dst), tree(src)
    res = sd.verify_snapshot_bank(dst, None)
    assert res["status"] == "ok" and res["cells"] == [CELL]
    assert res["rows"][0]["status"] == "verified"
    assert res["rows"][0]["fit_fingerprint"] == FP
    assert res["rows"][0]["n_files"] == 3
    assert tree(dst) == before and tree(src) == before_src
    # a copy that no longer matches its marker refuses, naming the file;
    # still nothing is written into the bank
    (dst / "cells_all" / CELL / "metrics.jsonl").write_text("tampered")
    res = sd.verify_snapshot_bank(dst, [CELL])
    assert res["status"] == "failed" and res["rows"][0]["status"] == "refused"
    assert "metrics.jsonl" in res["rows"][0]["error"]
    assert set(tree(dst)) == set(before)
    # a named cell that is not in the bank, and an empty bank, are failures
    res = sd.verify_snapshot_bank(dst, ["constv2-mix-10m__psr_ce_all"])
    assert res["status"] == "failed" and "COMPLETE" in res["rows"][0]["error"]
    res = sd.verify_snapshot_bank(tmp_path / "nothing", None)
    assert res["status"] == "failed" and res["rows"] == [] and res["note"]


def test_ce_command_output_feeds_derived_verdict(sd, tiny_llama, tmp_path):
    """END TO END across the app seam: the argv
    ``ce_command`` builds — --pool, a .json --out, --object/--cell/
    --parents-root — run through ce_score.main on the tiny model for V and
    both parents on the control held-out pool, produces exactly the
    combined CE files derived_verdict.main accepts and places on the
    REGISTERED pool manifest. Only the /repo, /vol and model pins are
    localized; every flag the app emits is kept."""
    import ce_score as cs
    from residual_fixtures import random_psr
    from test_derived_verdict import _write_jsonl, dv, make_probe_rows
    from test_residual_distill_script import (_CTL_ROWS, _MIX_ROWS, CTL_SI, KEY,
                                              MIX_SI, _stage_cell, _write_pool,
                                              prd)

    model, _tok = tiny_llama
    root = tmp_path / "out"
    mix_dir = _stage_cell(root, KEY, MIX_SI, random_psr(model, 11))
    ctl_dir = _stage_cell(root, KEY, CTL_SI, random_psr(model, 22))
    pools = tmp_path / "pools"
    _write_pool(pools, MIX_SI, _MIX_ROWS, "mix")
    _write_pool(pools, CTL_SI, _CTL_ROWS, "ctl")
    # the control held-out pool + its REGISTERED manifest (the analysis'
    # control-heldout <-> constv2ctl-heldout mapping)
    heldout = pools / "pairs_constv2ctl-heldout.jsonl"
    rows = [{"prompt": "" if i % 2 else "Ctl sys\n\n",
             "pos": f"held-out control doc {i} about weather " * (2 + i),
             "neg": "", "instruction_id": "constv2ctl-heldout",
             "doc_id": f"ctlho-{i}", "family": "stories" if i % 2 else "advice",
             "official_tokens": 8, "truncated": False} for i in range(4)]
    heldout.write_text("".join(json.dumps(r) + "\n" for r in rows))
    pool_sha = hashlib.sha256(heldout.read_bytes()).hexdigest()
    (pools / "pairs_constv2ctl-heldout.manifest.json").write_text(json.dumps(
        {"cell": "constv2ctl-heldout", "lane": "control_heldout",
         "pool_sha256": pool_sha, "doc_ids_in_order": [r["doc_id"] for r in rows],
         "n_docs": 4}))
    v_path = prd.compose_v(root, root / "derived", KEY, MIX_SI, CTL_SI,
                           pools_dir=pools)
    v_id = prd.v_derived_id(MIX_SI)

    def run_ce(obj_path, tag, pool_tag="control-heldout", pairs=heldout):
        cmd, out = sd.ce_command(
            "gemma4e4b_base", str(obj_path), tag, pool_tag,
            f"/vol/pool/gemma4e4b_base/pairs_{pairs.name}", scale=1.0,
            source_tag="ctl")
        argv = list(cmd[2:])
        local_out = tmp_path / "ce_score" / Path(out).name
        subst = {"--model": TINY, "--pairs": str(pairs), "--out": str(local_out),
                 "--parents-root": str(root), "--max-len": "96"}
        for i, tok in enumerate(argv):
            if tok in subst:
                argv[i + 1] = subst[tok]
        i = argv.index("--revision")
        del argv[i:i + 2]
        argv += ["--device", "cpu", "--dtype", "float32"]
        assert cs.main(argv) == 0
        return local_out

    ce_v = run_ce(v_path, v_id)
    ce_mix = run_ce(mix_dir / "object.pt", f"{MIX_SI}__psr_ce_all")
    ce_ctl = run_ce(ctl_dir / "object.pt", f"{CTL_SI}__psr_ce_all")
    for pth in (ce_v, ce_mix, ce_ctl):
        doc = json.loads(pth.read_text())
        assert doc["pool"] == "control-heldout"
        assert doc["pairs_sha256"] == pool_sha and len(doc["docs"]) == 4
        assert pth.name.endswith("__control-heldout__scale1.json")
    assert json.loads(ce_v.read_text())["identity_kind"] == "derived"
    assert json.loads(ce_mix.read_text())["identity_kind"] == "staged"
    # synthetic probe rows for P1/P2; the verdict places the three CE files
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
    assert dv.main(argv) == 0
    res = json.loads(verdict.read_text())
    assert res["p3_pool"]["pool"] == "control-heldout"
    assert res["p3_pool"]["pairs_sha256"] == pool_sha
    assert res["p3_pool"]["manifest"]["stem"] == "constv2ctl-heldout"
    assert res["p3_pool"]["manifest"]["cell"] == "constv2ctl-heldout"
    assert res["p3_pool"]["object_sha256"]["object_ce"] == \
        hashlib.sha256(v_path.read_bytes()).hexdigest()
    assert res["p3_pool"]["cells"]["object_ce"] == v_id
    assert res["p3"]["n_docs"] == 4 and res["p3"]["status"] in dv.STATUSES
    assert res["inputs"]["pool_manifest"]["n_docs"] == 4
    # the same command shape on the ctl TRAINING pool is a different pool:
    # the verdict refuses it as a P3 input
    ce_other = run_ce(mix_dir / "object.pt", f"{MIX_SI}__psr_ce_all",
                      pool_tag="constv2ctl-100k",
                      pairs=pools / f"pairs_{CTL_SI}.jsonl")
    assert json.loads(ce_other.read_text())["pool"] == "constv2ctl-100k"
    i = argv.index("--mix-ce")
    argv[i + 1] = str(ce_other)
    assert dv.main(argv) == 2


def test_identity_mode_is_the_identity_only_preflight(sd, monkeypatch):
    """``--mode identity`` runs ``psr_residual_distill.py
    identity-check`` in one container per model — the residual pipeline's
    volume paths, NO ``--wandb`` (nothing is logged, so no entity guard is
    consulted, no network) — and the impl reads the per-layer report back
    from the volume path the CLI writes and returns it inline."""
    assert "identity" in sd.MODES
    cmd = sd.identity_command("gemma4e4b_base", "constv2-mix-100k",
                              "constv2ctl-100k", python="/py")
    assert cmd[:3] == ["/py", "/repo/scripts/sdfbase/psr_residual_distill.py",
                       "identity-check"]
    d = _d(cmd)
    assert d["--root"] == "/vol/out" and d["--out-root"] == "/vol/out/derived"
    assert d["--pools"] == "/vol/pool/gemma4e4b_base"
    assert d["--model-key"] == "gemma4e4b_base"
    assert d["--mix-si"] == "constv2-mix-100k"
    assert d["--ctl-si"] == "constv2ctl-100k"
    assert d["--model"] == sd.MODELS["gemma4e4b_base"]["repo"]
    assert d["--revision"] == sd.MODELS["gemma4e4b_base"]["rev"]
    assert d["--config"] == sd.MODELS["gemma4e4b_base"]["cfg"]
    assert "--wandb" not in cmd and "--no-production" not in cmd
    calls = []
    monkeypatch.setattr(sd.base, "require_wandb_entity",
                        lambda: calls.append("guard"))
    monkeypatch.setattr(sd.base, "_ensure_model",
                        lambda k: (_ for _ in ()).throw(RuntimeError("stop-here")))
    with pytest.raises(RuntimeError, match="stop-here"):
        sd._identity_impl(("gemma4e4b_base", "constv2-mix-100k",
                           "constv2ctl-100k"))
    assert calls == []
    impl = SRC.split("def _identity_impl", 1)[1].split("\ndef ", 1)[0]
    assert "identity_check_report.json" in impl
    assert "teacher_hook_contract" in impl and "identity_command(" in impl
    assert "/vol/out/derived" in impl or "DERIVED_ROOT" in impl
    # the entrypoint spawns the identity functions for this mode
    entry = SRC.split("def main(", 1)[1]
    assert 'mode == "identity"' in entry
    assert "identity_h200_std" in entry and "identity_h200_qwen" in entry


# -- fan-out, production-only, checkpoints --


def _prd():
    import psr_residual_distill as prd  # scripts/sdfbase is on sys.path (above)
    return prd


def test_fanout_commands_paths_and_spellings(sd):
    """One multiplier per calibrate container (part file under the distilled
    object's calib_parts/lrmult<m>/), a CPU merge over exactly those parts
    into the canonical sweep file, and a production-only ``distill`` with
    NO --lr-mult (the calibration's selected one); the budget and the
    checkpoint cadence ride into every container explicitly; the app's
    multiplier spellings are the CLI's."""
    prd = _prd()
    key, mix, ctl = "gemma4e4b_base", "constv2-mix-100k", "constv2ctl-100k"
    rd = sd.resid_out_dir(key, mix)
    assert rd == f"/vol/out/derived/psr_residual/{key}/{mix}__psr_ce_all_minus_ctl_distilled"
    for m in (0.3, 1.0, 3.0, 0.5, 1):
        assert sd.mult_key(m) == prd.format_mult(m)
    assert sd.parse_mults("0.3,1,3") == prd.parse_lr_mults("0.3,1,3") == [0.3, 1.0, 3.0]
    assert sd.parse_mults("0.5, 1,3,3") == [0.5, 1.0, 3.0]
    for bad in ("", "0", "-1", "x"):
        with pytest.raises(SystemExit, match="lr-mults"):
            sd.parse_mults(bad)
    assert sd.CALIB_PARTS_DIRNAME == prd.CALIB_PARTS_DIRNAME
    cmd, part = sd.calibrate_command(key, mix, ctl, lr_mult=0.3, wandb=True,
                                     python="/py", distill_epochs_cap=60,
                                     distill_patience=5, checkpoint_every=50)
    assert cmd[:3] == ["/py", "/repo/scripts/sdfbase/psr_residual_distill.py", "calibrate"]
    d = _d(cmd)
    assert d["--lr-mult"] == "0.3" and d["--out"] == part
    assert part == f"{rd}/calib_parts/lrmult0.3/calibration_controls.json"
    assert d["--distill-epochs-cap"] == "60" and d["--distill-patience"] == "5"
    assert d["--checkpoint-every"] == "50" and "--wandb" in cmd
    assert d["--root"] == "/vol/out" and d["--out-root"] == "/vol/out/derived"
    assert d["--model"] == sd.MODELS[key]["repo"] and d["--config"] == sd.MODELS[key]["cfg"]
    assert "--no-production" not in cmd and "--no-resume" not in cmd  # resume is the default
    cmd2, _p = sd.calibrate_command(key, mix, ctl, lr_mult=1.0, wandb=False, python="/py")
    assert "--wandb" not in cmd2 and "--checkpoint-every" not in cmd2
    assert _d(cmd2)["--out"].endswith("/calib_parts/lrmult1/calibration_controls.json")
    ccmd, vpath = sd.compose_command(key, mix, ctl, python="/py")
    assert ccmd[:3] == ["/py", "/repo/scripts/sdfbase/psr_residual_distill.py", "compose-v"]
    cd = _d(ccmd)
    assert cd["--root"] == "/vol/out" and cd["--out-root"] == "/vol/out/derived"
    assert cd["--model-key"] == key and cd["--mix-si"] == mix and cd["--ctl-si"] == ctl
    assert "--model" not in cd            # model-free
    assert vpath == f"/vol/out/derived/psr_paired_write/{key}/{mix}__psr_ce_all_minus_ctl/object.pt"
    assert vpath == sd.v_object_path(key, mix)
    with pytest.raises(KeyError):
        sd.compose_command("nope", mix, ctl)
    mcmd, merged = sd.merge_command(key, mix, ctl, lr_mults=[0.3, 1.0, 3.0], python="/py")
    assert mcmd[:3] == ["/py", "/repo/scripts/sdfbase/psr_residual_distill.py",
                        "merge-calibration"]
    md = _d(mcmd)
    assert md["--lr-mult"] == "0.3,1,3" and merged == f"{rd}/calibration_controls.json"
    assert md["--out"] == merged and md["--model-key"] == key
    assert md["--parts"].split(",") == [
        f"{rd}/calib_parts/lrmult{m}/calibration_controls.json" for m in ("0.3", "1", "3")]
    assert "--model" not in md            # model-free
    pcmd = sd.distill_command(key, mix, ctl, wandb=True, python="/py",
                              distill_epochs_cap=60, distill_patience=5,
                              checkpoint_every=50)
    assert pcmd[:3] == ["/py", "/repo/scripts/sdfbase/psr_residual_distill.py", "distill"]
    pd = _d(pcmd)
    assert "--lr-mult" not in pcmd          # the calibration's SELECTED multiplier
    assert pd["--distill-epochs-cap"] == "60" and pd["--distill-patience"] == "5"
    assert pd["--checkpoint-every"] == "50" and "--wandb" in pcmd
    assert "--calibration" not in pcmd      # the sweep file in the out dir
    # the serial path carries the cadence too
    scmd = sd.residual_command(key, mix, ctl, no_production=False, lr_mult="0.3,1,3",
                               wandb=False, checkpoint_every=50)
    assert _d(scmd)["--checkpoint-every"] == "50" and scmd[2] == "all"
    for c in (cmd, ccmd, mcmd, pcmd, scmd):
        for tok in [t for t in c[1:] if str(t).startswith("/")]:
            assert tok.startswith(("/vol/", "/repo/", "/py")), tok
        for tok in [t for t in c if "," in str(t) and str(t).startswith("/")]:
            assert all(x.startswith("/vol/") for x in tok.split(",")), tok
    # every CLI flag the app emits exists on the CLI's parsers
    ap = prd.build_parser()
    for c in (cmd, ccmd, mcmd, pcmd, scmd):
        ns = ap.parse_args(c[2:])
        assert ns.cmd == c[2]


class _RecVol:
    """A Volume stand-in that records reload/commit order; ``on_reload``
    materializes what other containers committed (the snapshot refresh)."""

    def __init__(self, on_reload=None):
        self.log = []
        self.on_reload = on_reload

    def reload(self):
        self.log.append("reload")
        if self.on_reload:
            self.on_reload()

    def commit(self):
        self.log.append("commit")


def test_compose_impl_verifies_an_existing_v_and_rebuilds_a_bad_one(sd, tmp_path,
                                                                    monkeypatch):
    """The fan-out's first step: V is composed by ONE CPU container. An
    existing V stands only when `verify-v` re-admits it against the live
    bank; a torn or stale one (object without sidecar, a parent refit under
    the same cell id) is REBUILT here rather than refused downstream
    forever; the volume is committed only after a successful compose; the
    stage reloads the volume before it looks."""
    v = tmp_path / "V" / "object.pt"
    monkeypatch.setattr(sd, "v_object_path", lambda k, m: str(v))
    runs = []
    rc_by_sub = {"verify-v": 0, "compose-v": 0}

    def fake_run(cmd, tag):
        runs.append((cmd[2], tag))
        return {"rc": rc_by_sub[cmd[2]], "wall_s": 1.0}

    monkeypatch.setattr(sd, "_run", fake_run)
    volrec = _RecVol()
    monkeypatch.setattr(sd, "vol", volrec)
    item = ("gemma4e4b_base", "constv2-mix-100k", "constv2ctl-100k")
    # no V: compose, commit
    r = sd._compose_impl(item)
    assert r["status"] == "ok" and r["composed"] is True
    assert runs == [("compose-v", "compose-v/gemma4e4b_base/constv2-mix-100k")]
    assert volrec.log == ["reload", "commit"]
    assert r["teacher_object"] == str(v)
    # a valid V on the volume: verified, kept, nothing committed
    runs.clear(); volrec.log.clear()
    v.parent.mkdir(parents=True)
    v.write_bytes(b"x")
    r = sd._compose_impl(item)
    assert r["status"] == "ok" and r["composed"] is False
    assert runs == [("verify-v", "verify-v/gemma4e4b_base/constv2-mix-100k")]
    assert volrec.log == ["reload"] and r["verify"]["rc"] == 0
    # a V that fails the verdict (torn / stale parent): rebuilt, committed
    runs.clear(); volrec.log.clear()
    rc_by_sub["verify-v"] = 2
    r = sd._compose_impl(item)
    assert r["status"] == "ok" and r["composed"] is True
    assert [c for c, _t in runs] == ["verify-v", "compose-v"]
    assert volrec.log == ["reload", "commit"] and r["verify"]["rc"] == 2
    # a failed compose: failed, NOT committed
    runs.clear(); volrec.log.clear()
    rc_by_sub["compose-v"] = 2
    r = sd._compose_impl(item)
    assert r["status"] == "failed" and r["composed"] is False
    assert volrec.log == ["reload"]
    # the verify-v command is the CLI's model-free subcommand on the same paths
    vcmd = sd.verify_v_command("gemma4e4b_base", "constv2-mix-100k", "constv2ctl-100k",
                               python="/py")
    assert vcmd[:3] == ["/py", "/repo/scripts/sdfbase/psr_residual_distill.py", "verify-v"]
    d = _d(vcmd)
    assert d["--root"] == "/vol/out" and d["--out-root"] == "/vol/out/derived"
    assert d["--model-key"] == "gemma4e4b_base" and "--model" not in d
    assert _prd().build_parser().parse_args(vcmd[2:]).cmd == "verify-v"


def test_consumer_stages_reload_the_volume_before_reading(sd, tmp_path, monkeypatch):
    """Modal Volumes are snapshots — a WARM container reused for a later
    call does not see what other containers committed unless it reloads.
    Every consumer stage reloads FIRST: production sees a sweep that
    landed only at reload time; the merge, calibration and serial
    containers reload before any read / model staging."""
    rd = tmp_path / "resid"
    monkeypatch.setattr(sd, "resid_out_dir", lambda k, m: str(rd))
    monkeypatch.setattr(sd.base, "require_wandb_entity", lambda: "ent")
    events = []
    monkeypatch.setattr(sd.base, "_ensure_model",
                        lambda k: events.append("ensure_model") or (_ for _ in ()).throw(
                            RuntimeError("stop-here")))
    sweep = {"sweep": {"complete": True, "selected_lr_mult": 3.0,
                       "budget": {"epochs_cap": 60, "patience": 5},
                       "per_mult": {"3": {"identity": {"fingerprint": "f" * 64}}}}}

    def land_sweep():
        # what another container committed since this one was mounted
        rd.mkdir(exist_ok=True)
        (rd / "calibration_controls.json").write_text(json.dumps(sweep))
        events.append("sweep-visible")

    volrec = _RecVol(on_reload=land_sweep)
    monkeypatch.setattr(sd, "vol", volrec)
    item = ("gemma4e4b_base", "constv2-mix-100k", "constv2ctl-100k", True, 60, 5, 50)
    assert not (rd / "calibration_controls.json").exists()
    with pytest.raises(RuntimeError, match="stop-here"):
        sd._production_impl(item)          # the precheck PASSES: it read after the reload
    assert events == ["sweep-visible", "ensure_model"] and volrec.log[0] == "reload"
    # without the reload the same warm container would have refused
    events.clear(); volrec.log.clear()
    (rd / "calibration_controls.json").unlink()
    monkeypatch.setattr(sd, "vol", _RecVol())
    res = sd._production_impl(item)
    assert res["status"] == "refused" and events == []
    # merge / calibration / serial: reload precedes the first read or staging
    monkeypatch.setattr(sd, "vol", volrec)
    events.clear(); volrec.log.clear()
    monkeypatch.setattr(sd, "_run", lambda cmd, tag: events.append(("run", cmd[2])) or {
        "rc": 0, "wall_s": 1.0})
    sd._merge_impl(("gemma4e4b_base", "constv2-mix-100k", "constv2ctl-100k", "0.3,1,3"))
    assert volrec.log[0] == "reload" and events[-1] == ("run", "merge-calibration")
    assert volrec.log.index("reload") == 0 and "sweep-visible" in events
    events.clear(); volrec.log.clear()
    with pytest.raises(RuntimeError, match="stop-here"):
        sd._calibration_impl(("gemma4e4b_base", "constv2-mix-100k", "constv2ctl-100k",
                              0.3, False, 60, 5, 50))
    assert volrec.log == ["reload"] and events == ["sweep-visible", "ensure_model"]
    events.clear(); volrec.log.clear()
    with pytest.raises(RuntimeError, match="stop-here"):
        sd._residual_impl(("gemma4e4b_base", "constv2-mix-100k", "constv2ctl-100k",
                           True, "0.3,1,3", False, 60, 5, 50))
    assert volrec.log == ["reload"] and events == ["sweep-visible", "ensure_model"]
    # source pin: every consumer impl calls _fresh_volume() before anything else reads
    for impl in ("_residual_impl", "_calibration_impl", "_compose_impl", "_merge_impl",
                 "_production_impl"):
        body = SRC.split(f"\ndef {impl}(", 1)[1].split("\ndef ", 1)[0]
        assert "_fresh_volume()" in body, impl
        assert body.index("_fresh_volume()") < body.index("_run(") if "_run(" in body else True
    assert "vol.reload()" in SRC.split("def _fresh_volume", 1)[1].split("\ndef ", 1)[0]


def test_calibration_precheck_names_every_problem(sd, tmp_path):
    cal = tmp_path / "calibration_controls.json"
    assert sd.calibration_precheck(cal, epochs_cap=60, patience=5) == [
        f"{cal} missing — run --mode calibration (or residual-distill) first; "
        "production never calibrates"]
    cal.write_text("{not json")
    assert "unreadable" in sd.calibration_precheck(cal, epochs_cap=60, patience=5)[0]
    cal.write_text(json.dumps({"controls": {}}))
    assert "no `sweep` block" in sd.calibration_precheck(cal, epochs_cap=60, patience=5)[0]
    good = {"sweep": {"complete": True, "selected_lr_mult": 1.0, "lr_mults": [0.3, 1, 3],
                      "budget": {"epochs_cap": 60, "patience": 5},
                      "per_mult": {"0.3": {}, "1": {"identity": {"fingerprint": "f" * 64}},
                                   "3": {}}}}
    assert sd.calibration_precheck(cal.parent / "x.json", epochs_cap=None, patience=None)
    cal.write_text(json.dumps(good))
    assert sd.calibration_precheck(cal, epochs_cap=60, patience=5) == []
    assert sd.calibration_precheck(cal, epochs_cap=None, patience=None) == []
    bad = json.loads(json.dumps(good))
    bad["sweep"]["complete"] = False
    cal.write_text(json.dumps(bad))
    probs = sd.calibration_precheck(cal, epochs_cap=60, patience=5)
    assert len(probs) == 1 and "INCOMPLETE" in probs[0]
    bad = json.loads(json.dumps(good))
    bad["sweep"]["selected_lr_mult"] = 7.0
    cal.write_text(json.dumps(bad))
    assert "has no payload" in sd.calibration_precheck(cal, epochs_cap=60, patience=5)[0]
    bad = json.loads(json.dumps(good))
    bad["sweep"]["per_mult"]["1"] = {"identity": {}}
    cal.write_text(json.dumps(bad))
    assert "no identity fingerprint" in sd.calibration_precheck(cal, epochs_cap=60, patience=5)[0]
    cal.write_text(json.dumps(good))
    probs = sd.calibration_precheck(cal, epochs_cap=24, patience=5)
    assert len(probs) == 1 and "epochs_cap=60 != this run's 24" in probs[0]
    probs = sd.calibration_precheck(cal, epochs_cap=60, patience=2)
    assert len(probs) == 1 and "patience=5 != this run's 2" in probs[0]


def test_production_impl_refuses_before_staging_the_model(sd, tmp_path, monkeypatch):
    """The production-only container reads the sweep file off the volume
    FIRST: no complete sweep at this budget → ``refused`` with the problems,
    no W&B guard, no model download; with one present the guard runs and
    the model is staged (the CLI's full identity gate follows)."""
    rd = tmp_path / "resid"
    monkeypatch.setattr(sd, "resid_out_dir", lambda k, m: str(rd))
    calls = []
    monkeypatch.setattr(sd.base, "require_wandb_entity",
                        lambda: calls.append("guard") or "ent")
    monkeypatch.setattr(sd.base, "_ensure_model",
                        lambda k: (_ for _ in ()).throw(RuntimeError("stop-here")))
    item = ("gemma4e4b_base", "constv2-mix-100k", "constv2ctl-100k", True, 60, 5, 50)
    res = sd._production_impl(item)
    assert res["status"] == "refused" and calls == []
    assert res["problems"] and "missing" in res["problems"][0]
    assert res["calibration_controls"] == str(rd / "calibration_controls.json")
    rd.mkdir()
    (rd / "calibration_controls.json").write_text(json.dumps(
        {"sweep": {"complete": True, "selected_lr_mult": 3.0,
                   "budget": {"epochs_cap": 60, "patience": 5},
                   "per_mult": {"3": {"identity": {"fingerprint": "f" * 64}}}}}))
    with pytest.raises(RuntimeError, match="stop-here"):
        sd._production_impl(item)
    assert calls == ["guard"]
    calls.clear()
    with pytest.raises(RuntimeError, match="stop-here"):
        sd._production_impl(item[:3] + (False,) + item[4:])
    assert calls == []
    # the impls' item shapes match what the entrypoint / pipeline spawn
    src_main = SRC.split("def main(", 1)[1]
    assert "int(distill_patience), int(checkpoint_every)) + cap))" in src_main
    pipe = SRC.split("def residual_pipeline(", 1)[1].split("\ndef ", 1)[0]
    assert "calibrate_fn.spawn((key, mix_si, ctl_si, float(m), bool(wandb)," in pipe
    assert "production_fn.spawn((key, mix_si, ctl_si, bool(wandb)," in pipe


class _FakeFn:
    """A Modal function stand-in: ``spawn(item)`` records the item and
    returns a call whose ``get()`` runs the scripted handler."""

    def __init__(self, name, handler, log):
        self.name, self.handler, self.log = name, handler, log
        self.items = []

    def spawn(self, item):
        self.items.append(item)
        self.log.append(("spawn", self.name, item))
        fn = self

        class _Call:
            def get(self_inner, timeout=None):
                fn.log.append(("get", fn.name, item))
                return fn.handler(item)
        return _Call()


def test_residual_pipeline_fans_out_then_merges_then_produces(sd):
    """Every multiplier's calibration container is in flight before any
    result is awaited; the merge runs once over exactly the sweep; the
    production container runs only after a successful merge; each failed
    stage stops the pipeline and is named; ``production=False`` stops
    after the merge."""
    log = []
    comp = _FakeFn("compose", lambda it: {"status": "ok", "composed": True,
                                          "teacher_object": "/vol/V.pt"}, log)
    cal = _FakeFn("cal", lambda it: {"status": "ok", "lr_mult": it[3],
                                     "run": {"wall_s": 1.0}}, log)
    merge = _FakeFn("merge", lambda it: {"status": "ok", "calibration": {
        "selected_lr_mult": 1.0}}, log)
    prod = _FakeFn("prod", lambda it: {"status": "ok", "recovery_verdict": "recovered",
                                       "object_sha256": "ab" * 32, "lr_mult": 1.0,
                                       "out_dir": "/vol/x", "calibration": {}}, log)
    kw = dict(mix_si="constv2-mix-100k", ctl_si="constv2ctl-100k", wandb=True,
              epochs_cap=60, patience=5, checkpoint_every=50, log=lambda *a, **k: None,
              compose_fn=comp)
    r = sd.residual_pipeline("gemma4e4b_base", [0.3, 1.0, 3.0], production=True,
                             calibrate_fn=cal, merge_fn=merge, production_fn=prod, **kw)
    assert r["status"] == "ok" and r["recovery_verdict"] == "recovered"
    assert r["lr_mult"] == 1.0 and len(r["parts"]) == 3 and "failed_stage" not in r
    assert r["compose"]["composed"] is True
    assert comp.items == [("gemma4e4b_base", "constv2-mix-100k", "constv2ctl-100k")]
    assert cal.items == [("gemma4e4b_base", "constv2-mix-100k", "constv2ctl-100k", m,
                          True, 60, 5, 50) for m in (0.3, 1.0, 3.0)]
    assert merge.items == [("gemma4e4b_base", "constv2-mix-100k", "constv2ctl-100k",
                            "0.3,1,3")]
    assert prod.items == [("gemma4e4b_base", "constv2-mix-100k", "constv2ctl-100k",
                           True, 60, 5, 50)]
    kinds = [(k, n) for k, n, _ in log]
    # V composed (spawn + get) before any calibration; all three calibration
    # spawns precede the first calibration get; merge after every
    # calibration get; production after the merge get
    assert kinds[:5] == [("spawn", "compose"), ("get", "compose")] + [("spawn", "cal")] * 3
    assert kinds.index(("spawn", "merge")) > max(i for i, k in enumerate(kinds) if k == ("get", "cal"))
    assert kinds.index(("spawn", "prod")) > kinds.index(("get", "merge"))
    # production=False: compose + parts + merge only
    log.clear()
    comp.items.clear(); cal.items.clear(); merge.items.clear(); prod.items.clear()
    r = sd.residual_pipeline("qwen35_4b_base", [0.3, 1.0], production=False,
                             calibrate_fn=cal, merge_fn=merge, production_fn=prod, **kw)
    assert r["status"] == "ok" and prod.items == [] and len(merge.items) == 1
    assert r["calibration"] == {"selected_lr_mult": 1.0} and len(comp.items) == 1
    # a failed compose: nothing else is spawned
    bad_comp = _FakeFn("compose", lambda it: {"status": "failed", "run": {"rc": 2}}, log)
    log.clear()
    cal.items.clear(); merge.items.clear(); prod.items.clear()
    r = sd.residual_pipeline("gemma4e4b_base", [0.3], production=True,
                             calibrate_fn=cal, merge_fn=merge, production_fn=prod,
                             **{**kw, "compose_fn": bad_comp})
    assert r["status"] == "failed" and r["failed_stage"] == "compose"
    assert cal.items == [] and merge.items == [] and prod.items == []
    # a failed part: no merge, no production, stage named
    log.clear()
    cal.items.clear(); merge.items.clear(); prod.items.clear()
    flaky = _FakeFn("cal", lambda it: {"status": "failed" if it[3] == 1.0 else "ok",
                                       "lr_mult": it[3], "run": {}}, log)
    r = sd.residual_pipeline("gemma4e4b_base", [0.3, 1.0, 3.0], production=True,
                             calibrate_fn=flaky, merge_fn=merge, production_fn=prod, **kw)
    assert r["status"] == "failed" and r["failed_stage"] == "calibration"
    assert len(flaky.items) == 3 and merge.items == [] and prod.items == []
    # a failed merge: no production
    bad_merge = _FakeFn("merge", lambda it: {"status": "failed", "run": {"rc": 2}}, log)
    r = sd.residual_pipeline("gemma4e4b_base", [0.3], production=True,
                             calibrate_fn=cal, merge_fn=bad_merge, production_fn=prod, **kw)
    assert r["status"] == "failed" and r["failed_stage"] == "merge" and prod.items == []
    # a failed production is reported as such
    bad_prod = _FakeFn("prod", lambda it: {"status": "failed", "run": {"rc": 2}}, log)
    r = sd.residual_pipeline("gemma4e4b_base", [0.3], production=True,
                             calibrate_fn=cal, merge_fn=merge, production_fn=bad_prod, **kw)
    assert r["status"] == "failed" and r["failed_stage"] == "production"


def test_entrypoint_modes_fanout_serial_and_production_only(sd, monkeypatch):
    """``residual-distill`` / ``calibration`` fan out per model (one pipeline
    per model, concurrently) with the budget + checkpoint cadence;
    ``--serial`` spawns the one-container ``all`` item WITH the cadence;
    ``residual-distill-prod`` spawns production-only items and never a
    calibration."""
    seen = []

    def fake_pipeline(k, mults, **kw):
        seen.append((k, list(mults), kw["production"], kw["epochs_cap"],
                     kw["patience"], kw["checkpoint_every"], kw["wandb"]))
        return {"status": "ok"}

    monkeypatch.setattr(sd, "residual_pipeline", fake_pipeline)
    sd.main(app_name=sd.APP_NAME, mode="residual-distill", model="gemma4e4b_base,qwen35_4b_base",
            wandb=False)
    assert sorted(seen) == [("gemma4e4b_base", [0.3, 1.0, 3.0], True, 60, 5, 50, False),
                            ("qwen35_4b_base", [0.3, 1.0, 3.0], True, 60, 5, 50, False)]
    seen.clear()
    sd.main(app_name=sd.APP_NAME, mode="calibration", model="gemma4e4b_base", lr_mults="1,3",
            distill_epochs_cap=24, distill_patience=2, checkpoint_every=10)
    assert seen == [("gemma4e4b_base", [1.0, 3.0], False, 24, 2, 10, True)]
    with pytest.raises(SystemExit, match="lr-mults"):
        sd.main(app_name=sd.APP_NAME, mode="calibration", model="gemma4e4b_base", lr_mults="")
    # a failed model fails the run
    monkeypatch.setattr(sd, "residual_pipeline",
                        lambda k, m, **kw: {"status": "failed", "failed_stage": "merge"})
    with pytest.raises(SystemExit, match="1 model\\(s\\) failed"):
        sd.main(app_name=sd.APP_NAME, mode="residual-distill", model="gemma4e4b_base")
    # serial: the old one-container item, now with the checkpoint cadence
    log = []
    std = _FakeFn("std", lambda it: {"status": "ok"}, log)
    qwen = _FakeFn("qwen", lambda it: {"status": "ok"}, log)
    monkeypatch.setattr(sd, "residual_h200_std", std)
    monkeypatch.setattr(sd, "residual_h200_qwen", qwen)
    sd.main(app_name=sd.APP_NAME, mode="residual-distill", model="gemma4e4b_base,qwen35_4b_base",
            serial=True, wandb=False)
    assert std.items == [("gemma4e4b_base", "constv2-mix-100k", "constv2ctl-100k",
                          False, "0.3,1,3", False, 60, 5, 50)]
    assert qwen.items == [("qwen35_4b_base", "constv2-mix-100k", "constv2ctl-100k",
                           False, "0.3,1,3", False, 60, 5, 50)]
    sd.main(app_name=sd.APP_NAME, mode="calibration", model="gemma4e4b_base", serial=True, checkpoint_every=0)
    assert std.items[-1][3] is True and std.items[-1][-1] == 0
    # production only
    pstd = _FakeFn("pstd", lambda it: {"status": "ok"}, log)
    pqwen = _FakeFn("pqwen", lambda it: {"status": "ok"}, log)
    monkeypatch.setattr(sd, "production_h200_std", pstd)
    monkeypatch.setattr(sd, "production_h200_qwen", pqwen)
    cstd = _FakeFn("cstd", lambda it: {"status": "ok"}, log)
    monkeypatch.setattr(sd, "calibrate_h200_std", cstd)
    sd.main(app_name=sd.APP_NAME, mode="residual-distill-prod", model="gemma4e4b_base,qwen35_4b_base")
    assert pstd.items == [("gemma4e4b_base", "constv2-mix-100k", "constv2ctl-100k",
                           True, 60, 5, 50)]
    assert pqwen.items == [("qwen35_4b_base", "constv2-mix-100k", "constv2ctl-100k",
                            True, 60, 5, 50)]
    assert cstd.items == []
    pref = _FakeFn("pstd", lambda it: {"status": "refused", "problems": ["x"]}, log)
    monkeypatch.setattr(sd, "production_h200_std", pref)
    with pytest.raises(SystemExit, match="failed"):
        sd.main(app_name=sd.APP_NAME, mode="residual-distill-prod", model="gemma4e4b_base")
    # the fan-out wires the CPU compose + merge functions and the GPU pair
    entry = SRC.split("def main(", 1)[1]
    assert "compose_fn=compose_cpu" in entry and "merge_fn=calib_merge_cpu" in entry
    assert "calibrate_h200_std" in entry and "production_h200_qwen" in entry


def test_function_timeouts_follow_the_measured_runtime(sd):
    """Finding 1(d): the per-stage function timeouts are derived from the
    measured 60-epoch control (2985 s on gemma ≈ 50 s/epoch): a
    one-multiplier calibration container (two ≤60-epoch controls + ~10 min
    staging/floor/identity) gets ≥ 2× that; production (one ≤60-epoch fit
    + ED/bootstrap/liveness) likewise; the serial ``all`` keeps 8 h."""
    calib_worst = 2 * 60 * sd.EPOCH_S_WORST + 600
    prod_worst = 60 * sd.EPOCH_S_WORST + 600
    assert sd.CALIB_TIMEOUT_S >= 2 * calib_worst
    assert sd.PROD_TIMEOUT_S >= 2 * prod_worst
    assert sd.CALIB_TIMEOUT_S < sd.SERIAL_TIMEOUT_S == 28800
    assert sd.PROD_TIMEOUT_S < sd.CALIB_TIMEOUT_S
    assert sd.EPOCH_S_WORST == 50 and sd.DEFAULT_CHECKPOINT_EVERY == 50
    for fn, t in (("calibrate_h200_std", "CALIB_TIMEOUT_S"),
                  ("calibrate_h200_qwen", "CALIB_TIMEOUT_S"),
                  ("production_h200_std", "PROD_TIMEOUT_S"),
                  ("production_h200_qwen", "PROD_TIMEOUT_S"),
                  ("residual_h200_std", "SERIAL_TIMEOUT_S"),
                  ("residual_h200_qwen", "SERIAL_TIMEOUT_S")):
        deco = SRC.split(f"\ndef {fn}(", 1)[0].rsplit("@app.function(", 1)[1]
        assert f"timeout={t}" in deco, (fn, deco)
    for cpu_fn in ("calib_merge_cpu", "compose_cpu"):
        deco = SRC.split(f"\ndef {cpu_fn}(", 1)[0].rsplit("@app.function(", 1)[1]
        assert "gpu=" not in deco and "timeout=1800" in deco, cpu_fn
    entry = SRC.split("def main(", 1)[1]
    assert "checkpoint_every: int = DEFAULT_CHECKPOINT_EVERY" in entry
    assert "serial: bool = False" in entry
