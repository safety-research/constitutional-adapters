"""An explicit `--parity-record` binding REPLACES the pass the record remembers (round 2 on 83da932b8, the probe owner's ruling on
probe 979954). `reverdict --parity-record B` on a main record whose plan remembered pass A stamped B on the verdict but left
`plan.parity_record_dir = A`: `_remember_parity_record` returned at the first remembered key, so the next bare `verdict` — the
carrier's after a reference-only rerun, a reopen's — defaulted back to A and the operator's replacement was undone by the record
itself. Now an explicit binding (`reverdict --parity-record`, `verdict --parity-record`) is remembered once the pass has LOADED
for this record and replaces a different remembered dir; a pass that fails to load leaves the memory as it was; the implicit
memory (a reopen reading the standing verdict's stamp) is still write-once; and the reference-only rerun remembers the stamp
before it archives the verdict, exactly as a resume with launches to run does. CPU only; fake engines; synthetic ids; no cluster
path, no host name."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from sisteer.capeval.serving import geometry as g

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "sdfbase"))
sys.path.insert(0, str(REPO / "tests"))
import serving_probe as sp  # noqa: E402
from test_probe_status_cross_record_parity import _judge_pass, _parity_record_at_rest  # noqa: E402
from test_serving_probe_parity_pass import MODEL, _main_record  # noqa: E402
from test_serving_probe_reference_missing import _complete_record, _hf, _plan as _ref_plan, _ref_deps, _run_reference_only  # noqa: E402
from test_serving_probe_revocation import _mint_receipt  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for v in (g.SITE_GEOMETRY_DIR_ENV, g.SITE_DIR_ENV, g.SHARED_ROOT_ENV, "SISTEER_MODELS_TABLE", sp.VISIBLE_DEVICES_ENV, "SLURM_JOB_ID"):
        monkeypatch.delenv(v, raising=False)


def _judge_mlp_control_fails(pending):
    """Pass B's judgement: every adapter agrees with the engine except the mlp manifests' control adapter (a measured mismatch there),
    so a verdict that read B excludes the mlp pair for `reference_mismatch:control__lora` and one that read A does not."""
    out = {}
    for it in pending:
        p = Path(it["dir"])
        if p.parent.name.startswith("mlp_") and p.name == "control__lora":
            hf = _hf(29108, -0.12247)
        else:
            hf = _hf(it["vllm_first_token"]["id"], -4.5)
        out[it["dir"]] = {**sp.first_token_parity(hf, it["vllm_first_token"]), "hf": hf}
    return out


def _remembered(main: Path):
    return sp.load_plan(main).get("parity_record_dir")


def _stamped(main: Path):
    return (json.loads((main / sp.VERDICT_FILE).read_text())[sp.PARITY_RECORD_KEY] or {}).get("dir")


def _allowed_and_mismatch(main: Path):
    v = json.loads((main / sp.VERDICT_FILE).read_text())
    allowed = json.loads((main / sp.ALLOWED_PAIRS_FILE).read_text())["pairs"] if (main / sp.ALLOWED_PAIRS_FILE).is_file() else []
    mismatch = sorted(e["set"] for e in v["excluded"] if any("reference_mismatch:control__lora" in r for r in e["reasons"]))
    return sorted([p["rank"], p["set"]] for p in allowed), mismatch


def _main_with_two_passes(tmp_path: Path):
    """A main record (servability certified, no reference of its own) beside two parity passes AT REST of the same checkpoint: A judges
    every adapter a pass, B judges the mlp control adapter a mismatch. The main record is bound to A by `reverdict --parity-record A`."""
    main, pa, pb, site = tmp_path / "main", tmp_path / "parity_a", tmp_path / "parity_b", tmp_path / "site"
    for d in (main, pa, pb, site):
        d.mkdir()
    plan, results = _main_record(main)
    sp.write_verdict(plan, results, probe_job_id="800", out=main, verdict_uri=str(main / sp.VERDICT_FILE), geometry_dir=site)
    sp.write_progress(plan, results, main)
    _parity_record_at_rest(pa, judge=_judge_pass)
    _parity_record_at_rest(pb, judge=_judge_mlp_control_fails)
    gen = sp.result_generation(main)["generation"]
    assert sp.main(["reverdict", "--record", str(main), "--parity-record", str(pa), "--reopen-token", _mint_receipt(main, gen), "--geometry-dir", str(site)]) == 0
    assert _remembered(main) == str(pa) and _stamped(main) == str(pa)
    assert _allowed_and_mismatch(main) == ([[4, "mlp"], [4, "wo"]], [])
    return main, pa, pb, site, gen


def test_reverdict_with_another_pass_replaces_the_remembered_one_and_the_bare_verdict_reads_it(tmp_path, capsys):
    main, pa, pb, site, gen = _main_with_two_passes(tmp_path)
    # the operator replaces A by B: the verdict is stamped B AND the plan remembers B (before: the plan kept A)
    assert sp.main(["reverdict", "--record", str(main), "--parity-record", str(pb), "--reopen-token", _mint_receipt(main, gen), "--geometry-dir", str(site)]) == 0
    out = capsys.readouterr().out
    assert _stamped(main) == str(pb)
    assert _remembered(main) == str(pb), f"the plan still pins the replaced pass: {_remembered(main)}"
    assert f"plan.parity_record_dir {pa} -> {pb}" in out and "REPLACED" in out, out
    assert _allowed_and_mismatch(main) == ([[4, "wo"]], ["mlp"])                      # B's judgement, read through the binding
    # a bare `verdict` afterwards (the carrier's after a reference-only rerun, a reopen's) defaults to B — never back to A
    assert sp.main(["verdict", "--model-key", MODEL, "--out", str(main), "--geometry-dir", str(site)]) == 0
    out = capsys.readouterr().out
    assert f"defaulted to the record's remembered parity pass {pb}" in out, out
    assert _stamped(main) == str(pb) and _remembered(main) == str(pb)
    assert _allowed_and_mismatch(main) == ([[4, "wo"]], ["mlp"])
    # the reopen's implicit memory (what `run --resume` does before it archives the verdict) keeps B: write-once over a remembered dir
    assert sp._remember_parity_record(main, sp.load_plan(main)) == str(pb) and _remembered(main) == str(pb)
    # binding the same pass again rewrites nothing and says nothing about a replacement
    assert sp.main(["reverdict", "--record", str(main), "--parity-record", str(pb), "--reopen-token", _mint_receipt(main, gen), "--geometry-dir", str(site)]) == 0
    assert "REPLACED" not in capsys.readouterr().out and _remembered(main) == str(pb)


def test_the_direct_verdict_door_binds_durably_too(tmp_path, capsys):
    main, pa, pb, site, _gen = _main_with_two_passes(tmp_path)
    assert sp.main(["verdict", "--model-key", MODEL, "--out", str(main), "--geometry-dir", str(site), "--parity-record", str(pb)]) == 0
    out = capsys.readouterr().out
    assert _stamped(main) == str(pb) and _remembered(main) == str(pb), (_stamped(main), _remembered(main))
    assert f"plan.parity_record_dir {pa} -> {pb}" in out, out
    assert sp.main(["verdict", "--model-key", MODEL, "--out", str(main), "--geometry-dir", str(site)]) == 0
    assert _stamped(main) == str(pb) and _allowed_and_mismatch(main) == ([[4, "wo"]], ["mlp"])


def test_a_binding_that_fails_to_load_leaves_the_memory_as_it_was(tmp_path):
    main, pa, _pb, site, gen = _main_with_two_passes(tmp_path)
    bad = tmp_path / "not_a_parity_record"; bad.mkdir()
    sp.write_json_atomic(bad / sp.PLAN_FILE, {**sp.load_plan(main), "parity_pass": False})          # a probe record, not a pass
    with pytest.raises(SystemExit, match="was not written with --parity-pass"):
        sp.main(["verdict", "--model-key", MODEL, "--out", str(main), "--geometry-dir", str(site), "--parity-record", str(bad)])
    assert _remembered(main) == str(pa) and _stamped(main) == str(pa)
    with pytest.raises(SystemExit, match="was not written with --parity-pass"):
        sp.main(["reverdict", "--record", str(main), "--parity-record", str(bad), "--reopen-token", _mint_receipt(main, gen), "--geometry-dir", str(site)])
    assert _remembered(main) == str(pa)                       # the recovery (`reverdict --resume-interrupted`) reads A again, never the bad dir
    other = tmp_path / "other_checkpoint"; other.mkdir()
    sp.write_json_atomic(other / sp.PLAN_FILE, {**sp.load_plan(pa), "revision": "f" * 40})
    with pytest.raises(SystemExit, match="revision"):
        sp.main(["verdict", "--model-key", MODEL, "--out", str(main), "--geometry-dir", str(site), "--parity-record", str(other)])
    assert _remembered(main) == str(pa)


def test_a_reference_only_rerun_remembers_the_stamp_before_it_archives_the_verdict(tmp_path, monkeypatch):
    """A record written before the plan carried `parity_record_dir` whose standing verdict stamps a pass: the reference-only rerun
    archives that verdict, so — like a resume with launches to run — it writes the stamp into the plan FIRST, and the carrier's bare
    `verdict` afterwards reads the pass (or refuses by its name) instead of silently demoting the menu to this record's own references."""
    out = tmp_path / "rec"; out.mkdir()
    plan = _ref_plan()
    wrong, _ = _ref_deps(plan, lambda pending: {it["dir"]: {**sp.first_token_parity(_hf(29108, -0.12247), it["vllm_first_token"]), "hf": _hf(29108, -0.12247)} for it in pending})
    _complete_record(out, plan, wrong)
    v = json.loads((out / sp.VERDICT_FILE).read_text()); v[sp.PARITY_RECORD_KEY] = {"dir": str(tmp_path / "some_parity"), "n_results": 3}
    (out / sp.VERDICT_FILE).write_text(json.dumps(v))
    assert "parity_record_dir" not in sp.load_plan(out)
    right, _calls = _ref_deps(plan, _judge_pass)
    monkeypatch.setattr(sp.Deps, "production", classmethod(lambda cls: right))
    nonce = _mint_receipt(out, sp.result_generation(out)["generation"])
    assert _run_reference_only(out, token=nonce) == 0
    assert not (out / sp.VERDICT_FILE).exists()                                               # archived before the re-judgement …
    assert sp.load_plan(out)["parity_record_dir"] == str(tmp_path / "some_parity")             # … and remembered first
