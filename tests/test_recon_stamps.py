"""Rung records carry the reconnaissance's identity and the reduce refuses a foreign rung. `recon --rung i`
writes recon_<i>.stamp.json at rung START (phase started) and embeds `recon_stamp` (phase completed, a canonical fingerprint of the
spec + cell + pool sha + sidecar sha + model/revision + arm/architecture/init + steps/seed/passes + ladder + rung/lr) in the record;
`--finalize` verifies every rung against the CURRENT spec/config and the rungs' agreement on the pool before pooling anything: one rung
run under another cell among four current ones refuses by name; an unstamped (legacy) rung refuses; a rung file under another rung's
name refuses; five current rungs reduce and recon.json carries the fingerprints."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_tune_arm import tune_arm  # noqa: E402
from test_tune_arm_bench_recon import NORMS, FakeCE, _args  # noqa: E402

LADDER = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]


def _spec(tmp_path: Path, si: str, name: str = "recon_spec.json") -> Path:
    p = tmp_path / name
    p.write_text(json.dumps({"kind": "recon_spec", "lr_ladder": LADDER, "steps": 20, "clip_percentile": 95, "warmup_frac": 0.15,
                             "init_scheme": None, "si": si, "rows": 100, "effective_batch": 64, "steps_per_pass": 2, "cycled_passes": 1}))
    return p


def _recon_args(tmp_path, out: Path, si: str, spec: Path | None, **over):
    ns = _args(tmp_path, lr_ladder=",".join(str(x) for x in LADDER), clip_percentile=95.0, warmup_frac=0.15, steps=20, seed=7,
               out=str(out), si=si, lora_rank=None, lora_modules=None, init_scheme=None, rung=None, finalize=False, passes=1,
               spec=(str(spec) if spec else None))
    for k, v in over.items():
        setattr(ns, k, v)
    return ns


def _world(monkeypatch):
    setup = SimpleNamespace(cfg={"model": "m", "model_revision": "r"}, max_len=64, model=None, tok=None)
    monkeypatch.setattr(tune_arm, "make_adapter", lambda arm: FakeCE(NORMS))
    monkeypatch.setattr(tune_arm, "_setup", lambda *a, **k: setup)
    monkeypatch.setattr(tune_arm, "_load_cfg", lambda p: {"model": "m", "model_revision": "r"})


def test_rung_task_stamps_start_and_completion_and_finalize_records_the_fingerprints(tmp_path, monkeypatch):
    _world(monkeypatch)
    out = tmp_path / "rec" / "recon.json"; out.parent.mkdir()
    spec = _spec(tmp_path, "constv2ctl2-full")
    for i in range(5):
        tune_arm.recon(_recon_args(tmp_path, out, "constv2ctl2-full", spec, rung=i))
        st_file = json.loads((out.parent / f"recon_{i}.stamp.json").read_text())
        rec = json.loads((out.parent / f"recon_{i}.json").read_text())
        assert st_file["phase"] == "completed" and rec["recon_stamp"]["phase"] == "completed"
        st = rec["recon_stamp"]
        assert st["rung"] == i and st["lr"] == LADDER[i] and st["cell"] == "constv2ctl2-full" and st["pairs_basename"] == "pairs.jsonl"
        assert st["pool_sha256"] == rec["pool_sha256"] and st["model"] == "m" and st["model_revision"] == "r" and st["arm"] == "psr_ce_all"
        assert st["steps"] == 20 and st["seed"] == 7 and st["passes"] == 1 and st["ladder"] == LADDER
        assert st["spec_fingerprint"] == tune_arm.recon_spec_fingerprint(spec) and len(st["fingerprint"]) == 64
        assert st_file["fingerprint"] == st["fingerprint"]
    tune_arm.recon(_recon_args(tmp_path, out, "constv2ctl2-full", spec, finalize=True))
    res = json.loads(out.read_text())
    assert res["clip"] is not None and res["spec_fingerprint"] == tune_arm.recon_spec_fingerprint(spec)
    assert sorted(res["rung_fingerprints"]) == [f"lr_{i}" for i in range(5)]
    assert all(len(v) == 64 for v in res["rung_fingerprints"].values())


def test_the_start_stamp_exists_before_the_record(tmp_path, monkeypatch):
    _world(monkeypatch)
    out = tmp_path / "rec" / "recon.json"; out.parent.mkdir()
    seen = {}
    orig = tune_arm.bench

    def spy(ns, *, setup=None):
        seen["stamp_at_bench"] = json.loads((out.parent / "recon_1.stamp.json").read_text())["phase"]
        seen["record_at_bench"] = (out.parent / "recon_1.json").exists()
        return orig(ns, setup=setup)

    monkeypatch.setattr(tune_arm, "bench", spy)
    tune_arm.recon(_recon_args(tmp_path, out, "constv2ctl2-full", _spec(tmp_path, "constv2ctl2-full"), rung=1))
    assert seen == {"stamp_at_bench": "started", "record_at_bench": False}


def test_one_rung_from_another_cell_among_four_current_refuses_by_name(tmp_path, monkeypatch):
    _world(monkeypatch)
    out = tmp_path / "rec" / "recon.json"; out.parent.mkdir()
    old_spec = _spec(tmp_path, "constv2-mix-10m", "old_spec.json"); new_spec = _spec(tmp_path, "constv2ctl2-full")
    tune_arm.recon(_recon_args(tmp_path, out, "constv2-mix-10m", old_spec, rung=2))            # an earlier attempt's rung file
    for i in (0, 1, 3, 4):
        tune_arm.recon(_recon_args(tmp_path, out, "constv2ctl2-full", new_spec, rung=i))
    assert all((out.parent / f"recon_{i}.json").exists() for i in range(5))                       # every filename present
    with pytest.raises(SystemExit) as e:
        tune_arm.recon(_recon_args(tmp_path, out, "constv2ctl2-full", new_spec, finalize=True))
    msg = str(e.value)
    assert "REFUSED" in msg and "rung 2" in msg and "another reconnaissance" in msg and "spec_fingerprint" in msg, msg
    assert not out.exists()
    # the same with matching spec fingerprints but a different cell recorded (a spec edited in place): the cell binds too
    for i in (0, 1, 3, 4):
        tune_arm.recon(_recon_args(tmp_path, out, "constv2ctl2-full", None, rung=i))              # no --spec on either side
    tune_arm.recon(_recon_args(tmp_path, out, "constv2-mix-10m", None, rung=2))
    with pytest.raises(SystemExit) as e:
        tune_arm.recon(_recon_args(tmp_path, out, "constv2ctl2-full", None, finalize=True))
    assert "rung 2" in str(e.value) and "cell: expected 'constv2ctl2-full', found 'constv2-mix-10m'" in str(e.value), str(e.value)


def test_unstamped_legacy_rung_and_misnamed_rung_refuse(tmp_path, monkeypatch):
    _world(monkeypatch)
    out = tmp_path / "rec" / "recon.json"; out.parent.mkdir()
    spec = _spec(tmp_path, "constv2ctl2-full")
    for i in range(5):
        tune_arm.recon(_recon_args(tmp_path, out, "constv2ctl2-full", spec, rung=i))
    rec3 = json.loads((out.parent / "recon_3.json").read_text()); del rec3["recon_stamp"]
    (out.parent / "recon_3.json").write_text(json.dumps(rec3))                                   # a legacy (unstamped) rung
    with pytest.raises(SystemExit) as e:
        tune_arm.recon(_recon_args(tmp_path, out, "constv2ctl2-full", spec, finalize=True))
    assert "rung 3" in str(e.value) and "no completed recon_stamp" in str(e.value), str(e.value)
    tune_arm.recon(_recon_args(tmp_path, out, "constv2ctl2-full", spec, rung=3))                 # restored
    (out.parent / "recon_4.json").write_text((out.parent / "recon_1.json").read_text())          # rung 1's file under rung 4's name
    with pytest.raises(SystemExit) as e:
        tune_arm.recon(_recon_args(tmp_path, out, "constv2ctl2-full", spec, finalize=True))
    assert "rung 4" in str(e.value) and "stamped as rung 1" in str(e.value), str(e.value)
    tune_arm.recon(_recon_args(tmp_path, out, "constv2ctl2-full", spec, rung=4))
    st = json.loads((out.parent / "recon_0.json").read_text()); st["recon_stamp"]["seed"] = 99   # an edited stamp: the fingerprint no longer recomputes
    (out.parent / "recon_0.json").write_text(json.dumps(st))
    with pytest.raises(SystemExit) as e:
        tune_arm.recon(_recon_args(tmp_path, out, "constv2ctl2-full", spec, finalize=True))
    assert "rung 0" in str(e.value) and "does not recompute" in str(e.value), str(e.value)
    tune_arm.recon(_recon_args(tmp_path, out, "constv2ctl2-full", spec, rung=0))
    tune_arm.recon(_recon_args(tmp_path, out, "constv2ctl2-full", spec, finalize=True))          # all five current again -> reduces
    assert json.loads(out.read_text())["clip"] is not None


def test_spec_fingerprint_is_canonical_and_protocol_only(tmp_path):
    a = tmp_path / "a.json"; b = tmp_path / "b.json"
    a.write_text(json.dumps({"kind": "recon_spec", "lr_ladder": LADDER, "steps": 150, "si": "constv2ctl2-full", "rows": 17073}, indent=2))
    b.write_text(json.dumps({"si": "constv2ctl2-full", "steps": 150, "lr_ladder": LADDER, "kind": "recon_spec", "rows": 17073}))
    assert tune_arm.recon_spec_fingerprint(a) == tune_arm.recon_spec_fingerprint(b)             # formatting and key order do not matter
    c = tmp_path / "c.json"; c.write_text(json.dumps({"kind": "recon_spec", "lr_ladder": LADDER, "steps": 150, "si": "constv2-mix-10m"}))
    assert tune_arm.recon_spec_fingerprint(a) != tune_arm.recon_spec_fingerprint(c)             # the cell is protocol
    assert tune_arm.recon_spec_fingerprint(None) is None


def test_fingerprint_helper_prints_the_canonical_fingerprint_and_reads_a_stdin_spec_once(tmp_path, monkeypatch, capsys):
    """`recon --fingerprint --spec F` (the front door's current-spec helper) prints recon_spec_fingerprint(F); `--spec -` is read
    from stdin ONCE and then serves every later read (verify + every rung's expected stamp) — a second read of a consumed stdin refused
    a matching verdict when these tests first ran."""
    import io
    spec = _spec(tmp_path, "constv2ctl2-full")
    tune_arm.recon(_recon_args(tmp_path, tmp_path / "recon.json", "constv2ctl2-full", spec, fingerprint=True, verify=False))
    assert capsys.readouterr().out.strip() == tune_arm.recon_spec_fingerprint(spec)
    monkeypatch.setattr(sys, "stdin", io.StringIO(spec.read_text()))
    ns = _recon_args(tmp_path, tmp_path / "recon.json", "constv2ctl2-full", None, fingerprint=True, verify=False); ns.spec = "-"
    tune_arm.recon(ns)
    assert capsys.readouterr().out.strip() == tune_arm.recon_spec_fingerprint(spec)
    assert ns.spec != "-" and Path(ns.spec).is_file() and json.loads(Path(ns.spec).read_text())["si"] == "constv2ctl2-full"   # materialised once


def test_verify_accepts_the_current_verdict_and_refuses_another_spec_or_mixed_fingerprints(tmp_path, monkeypatch, capsys):
    _world(monkeypatch)
    out = tmp_path / "rec" / "recon.json"; out.parent.mkdir()
    spec = _spec(tmp_path, "constv2ctl2-full")
    for i in range(5):
        tune_arm.recon(_recon_args(tmp_path, out, "constv2ctl2-full", spec, rung=i))
    tune_arm.recon(_recon_args(tmp_path, out, "constv2ctl2-full", spec, finalize=True))
    tune_arm.recon(_recon_args(tmp_path, out, "constv2ctl2-full", spec, verify=True, fingerprint=False))
    assert "[recon] verified:" in capsys.readouterr().out
    with pytest.raises(SystemExit) as e:                                                          # another spec (the cell changed)
        tune_arm.recon(_recon_args(tmp_path, out, "constv2-mix-10m", _spec(tmp_path, "constv2-mix-10m", "s2.json"), verify=True, fingerprint=False))
    assert "--verify REFUSED" in str(e.value) and "another reconnaissance" in str(e.value) and "constv2-mix-10m" in str(e.value)
    res = json.loads(out.read_text()); res["rung_fingerprints"]["lr_1"] = "0" * 64; out.write_text(json.dumps(res))
    with pytest.raises(SystemExit) as e:                                                          # a verdict reduced from other rung files
        tune_arm.recon(_recon_args(tmp_path, out, "constv2ctl2-full", spec, verify=True, fingerprint=False))
    assert "rung_fingerprints are not the stamps" in str(e.value)
    res.pop("spec_fingerprint"); out.write_text(json.dumps(res))
    with pytest.raises(SystemExit) as e:                                                          # a legacy verdict (no spec fingerprint)
        tune_arm.recon(_recon_args(tmp_path, out, "constv2ctl2-full", spec, verify=True, fingerprint=False))
    assert "legacy verdict" in str(e.value)


NORMS150 = [0.9, 0.8, 0.7, 0.6, 0.5] + [0.30 + 0.01 * ((i * 7) % 5) for i in range(145)]


def _stable_world(monkeypatch, norms=None, **kw):
    setup = SimpleNamespace(cfg={"model": "m", "model_revision": "r"}, max_len=64, model=None, tok=None)
    monkeypatch.setattr(tune_arm, "make_adapter", lambda arm: FakeCE(NORMS150 if norms is None else norms, **kw))
    monkeypatch.setattr(tune_arm, "_setup", lambda *a, **k: setup)
    monkeypatch.setattr(tune_arm, "_load_cfg", lambda p: {"model": "m", "model_revision": "r"})


def _verdict(tmp_path, monkeypatch, out: Path, spec: Path, *, steps=150, **world):
    _stable_world(monkeypatch, **world)
    for i in range(5):
        tune_arm.recon(_recon_args(tmp_path, out, "constv2ctl2-full", spec, rung=i, steps=steps))
    try:
        tune_arm.recon(_recon_args(tmp_path, out, "constv2ctl2-full", spec, finalize=True, steps=steps))
        return 0
    except SystemExit as e:
        return e.code


def _verify(tmp_path, out, spec, **over):
    return tune_arm.recon(_recon_args(tmp_path, out, "constv2ctl2-full", spec, verify=True, fingerprint=False, steps=150, **over))


def test_verify_refuses_a_genuine_no_stable_verdict_with_finalizes_exit_status(tmp_path, monkeypatch, capsys):
    """Provenance is not enough (P1-1): a verdict whose every rung is unstable carries clip null; finalize writes it and
    exits RECON_EXIT_NO_STABLE — and so does --verify, so the front door never marks DONE over a verdict with no clip."""
    out = tmp_path / "rec" / "recon.json"; out.parent.mkdir()
    spec = _spec(tmp_path, "constv2ctl2-full")
    rc = _verdict(tmp_path, monkeypatch, out, spec, diverge_at=11)                   # every rung diverges at step 11
    assert rc == tune_arm.RECON_EXIT_NO_STABLE and json.loads(out.read_text())["clip"] is None
    with pytest.raises(SystemExit) as e:
        _verify(tmp_path, out, spec)
    assert e.value.code == tune_arm.RECON_EXIT_NO_STABLE
    assert "certifies NO clip" in capsys.readouterr().out
    assert _verdict(tmp_path, monkeypatch, out, spec) == 0                            # a stable ladder: admitted
    _verify(tmp_path, out, spec)


def test_verdict_carries_result_hashes_and_pool_identity_and_a_replaced_rung_payload_is_refused(tmp_path, monkeypatch):
    """P1-2: rung_fingerprints hash the spec/input identity, not the RESULT; a rung re-run under the same spec that replaces
    recon_<i>.json with other norms keeps its stamp — the verdict's rung_result_sha256 (the decision-critical payload) catches it."""
    out = tmp_path / "rec" / "recon.json"; out.parent.mkdir()
    spec = _spec(tmp_path, "constv2ctl2-full")
    assert _verdict(tmp_path, monkeypatch, out, spec) == 0
    res = json.loads(out.read_text())
    assert sorted(res["rung_result_sha256"]) == [f"lr_{i}" for i in range(5)] and all(len(v) == 64 for v in res["rung_result_sha256"].values())
    rec0 = json.loads((out.parent / "recon_0.json").read_text())
    assert res["pool_identity"]["pool_sha256"] == rec0["pool_sha256"] == rec0["recon_stamp"]["pool_sha256"]
    assert res["rung_result_sha256"]["lr_2"] == tune_arm.recon_result_sha256(json.loads((out.parent / "recon_2.json").read_text()))
    _verify(tmp_path, out, spec)                                                      # unchanged: admitted
    rec2 = json.loads((out.parent / "recon_2.json").read_text())
    rec2["grad_norm_preclip"]["values"] = [v * 50 for v in rec2["grad_norm_preclip"]["values"]]   # the same stamp, other norms
    (out.parent / "recon_2.json").write_text(json.dumps(rec2))
    with pytest.raises(SystemExit) as e:
        _verify(tmp_path, out, spec)
    assert "result payload of rung(s) ['lr_2'] differs" in str(e.value), str(e.value)
    res.pop("rung_result_sha256"); out.write_text(json.dumps(res))                    # a verdict from before the hashes
    with pytest.raises(SystemExit) as e:
        _verify(tmp_path, out, spec)
    assert "carries no rung_result_sha256" in str(e.value)


def test_the_current_pool_binds_admission(tmp_path, monkeypatch):
    """P1-2: a pool rebuilt with other content at the same path (or a front door reporting another current pool sha / row count)
    never admits an old verdict; the same pool admits it."""
    out = tmp_path / "rec" / "recon.json"; out.parent.mkdir()
    spec = _spec(tmp_path, "constv2ctl2-full")
    assert _verdict(tmp_path, monkeypatch, out, spec) == 0
    pool = tmp_path / "pairs.jsonl"
    from sisteer.extraction.tokcache import pool_sha256
    _verify(tmp_path, out, spec, pool_sha256=pool_sha256(pool)[0], rows=100)          # the current pool, as the front door read it: admitted
    with pytest.raises(SystemExit) as e:
        _verify(tmp_path, out, spec, pool_sha256="0" * 64)                            # the front door's reading disagrees with the file
    assert "is not the pool file's" in str(e.value)
    with pytest.raises(SystemExit) as e:
        _verify(tmp_path, out, spec, rows=4102)                                       # the cell's row count changed
    assert "has 4102 rows now" in str(e.value) and "recorded 100" in str(e.value)
    pool.write_text(pool.read_text() + '{"prompt": "q", "pos": "y", "neg": ""}\n')  # the pool REBUILT under the same path
    with pytest.raises(SystemExit) as e:
        _verify(tmp_path, out, spec)
    assert "trained on another pool copy" in str(e.value), str(e.value)
    gone = tmp_path / "gone" / "pairs.jsonl"                                          # the pool unreachable (the grading host): the front door's sha binds
    with pytest.raises(SystemExit) as e:
        _verify(tmp_path, out, spec, pairs=str(gone), pool_sha256="1" * 64)
    assert "trained on another pool copy" in str(e.value), str(e.value)
    pool.write_text('{"prompt": "p", "pos": "x", "neg": ""}\n')                       # the original bytes again: the same pool admits
    _verify(tmp_path, out, spec, pool_sha256=pool_sha256(pool)[0])


def test_verify_rederives_the_decision_fields_and_refuses_an_edited_clip(tmp_path, monkeypatch):
    """Provenance + result hashes still let an edited recon.json (clip 1.5 -> 100, the rung files unchanged) through;
    --verify now re-runs the reduce over the verified rung records and requires every decision field to equal the verdict's."""
    out = tmp_path / "rec" / "recon.json"; out.parent.mkdir()
    spec = _spec(tmp_path, "constv2ctl2-full")
    assert _verdict(tmp_path, monkeypatch, out, spec) == 0
    _verify(tmp_path, out, spec)                                                      # untouched: verified
    res = json.loads(out.read_text()); good_clip = res["clip"]
    for field, bad in (("clip", 100.0), ("certified_lr_max", 1.0), ("stable_lrs", [1e-4]), ("clip_raw", 0.0)):
        edited = dict(res); edited[field] = bad; out.write_text(json.dumps(edited))
        with pytest.raises(SystemExit) as e:
            _verify(tmp_path, out, spec)
        assert "decision fields are not what its own verified rung records reduce to" in str(e.value) and field in str(e.value), (field, str(e.value))
    edited = dict(res); edited["per_lr"] = [dict(x, **{"class": "diverged"}) for x in res["per_lr"]]; out.write_text(json.dumps(edited))
    with pytest.raises(SystemExit) as e:
        _verify(tmp_path, out, spec)
    assert "per_lr classes" in str(e.value)
    out.write_text(json.dumps(res)); _verify(tmp_path, out, spec)                    # restored: verified, and the clip is the records' own
    assert json.loads(out.read_text())["clip"] == good_clip
