"""An existing recon verdict is reused ONLY as the CURRENT reconnaissance's. 03_hpo.sh's recon branch, executed under bash against a
LOCAL "cluster" record (a fake `ssh` that runs its command here, a fake `rsync` that copies), with stamped rung records produced by
tune_arm's own `recon --rung` / `--finalize`: a cluster recon.json reduced under another spec is archived to superseded/<stamp>/ on both
sides, not marked, and the array is (re)submitted; a matching one is fetched, verified and marked with its spec fingerprint; a DONE
marker recording another reconnaissance is reset; design revalidates what it holds the same way."""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from front_door_testlib import REPO, VENV_PY  # noqa: E402
from test_tune_arm import tune_arm  # noqa: E402
from test_tune_arm_bench_recon import NORMS, FakeCE  # noqa: E402


NORMS150 = [0.9, 0.8, 0.7, 0.6, 0.5] + [0.30 + 0.01 * ((i * 7) % 5) for i in range(145)]   # a STABLE 150-step rung: warm-up, then flat


class _CE(FakeCE):
    def __init__(self, norms=None, **kw):
        super().__init__(NORMS150 if norms is None else norms, **kw)

    def make_object(self, setup, sentinel=None, lora_rank=None, **axes):         # the init-scheme axis the front door's spec carries
        self._obj = object()

REPRO = REPO / "reproduce"
LADDER = [3.1623e-05, 1.7783e-04, 1.0000e-03, 5.6234e-03, 3.1623e-02]      # tune_arm's RELU box at 5 points (lr_ladder's spelling)


class World:
    """The recon phase's variables and helpers, the fragment `recon_compose_spec(){` .. `# === design`, fakes on PATH."""

    def __init__(self, tmp_path: Path):
        self.tmp = tmp_path
        self.cluster = tmp_path / "cluster" / "03_hpo" / "relu"; self.cluster.mkdir(parents=True)
        self.state = tmp_path / "state"; self.local = self.state / "m" / "03_hpo" / "relu"; self.local.mkdir(parents=True)
        pool = tmp_path / "cluster" / "pool"; pool.mkdir()
        self.pairs = pool / "pairs_constv2ctl2-full.jsonl"
        self.pairs.write_text('{"prompt": "p", "pos": "x", "neg": ""}\n' * 17073)                 # 267 steps per pass at 64
        self.cfg = tmp_path / "fits.yaml"; self.cfg.write_text("model: m\nmodel_revision: r\nbatch_size_by_arm:\n  psr_ce_all: 1\n")
        b = tmp_path / "bin"; b.mkdir()
        # ssh: drop options and the host, run the command HERE (every remote command of the fragment is plain bash over cluster paths)
        (b / "ssh").write_text('#!/bin/bash\nwhile [ "${1:-}" = -o ]; do shift 2; done; shift\nexec bash -c "$*"\n'); (b / "ssh").chmod(0o755)
        (b / "rsync").write_text('#!/bin/bash\nsrc="${@: -2:1}"; dst="${@: -1}"; src="${src#*:}"; mkdir -p "$dst"; cp -r "$src"/. "$dst"/ 2>/dev/null; '
                                 'find "$dst" -name "ckpt" -type d -prune -exec rm -r {} + 2>/dev/null; exit 0\n'); (b / "rsync").chmod(0o755)
        (b / "squeue").write_text("#!/bin/bash\nexit 0\n"); (b / "squeue").chmod(0o755)
        self.bin = b

    def run(self, phase: str = "recon", *, want: str = "recon", extra: str = "", pre: str = "", real_fetch: bool = False) -> subprocess.CompletedProcess:
        src = (REPRO / "03_hpo.sh").read_text()
        frag = src[src.index("recon_compose_spec(){"):src.index("# ================================================================================================================ design")]
        branch_at = frag.index('if want recon && [ "$PROTOCOL" = paper ]')
        frag = frag[:branch_at] + pre + "\n" + frag[branch_at:]                                        # `pre` runs after the helpers, before the branch
        helpers = src[src.index("recon_passes_for_cell(){"):src.index("widest_pair(){")]           # recon_passes_for_cell + recon_archive_record
        fetch_fn = (src[src.index("FETCHED=\"\""):src.index("geometry_to_site(){")] if real_fetch     # 03's OWN fetch (with its per-invocation cache)
                    else 'fetch(){ mkdir -p "$2"; rsync -a "$CL_HOST:$1/" "$2/"; }')
        script = f"""set -u
source {REPRO}/lib/common.sh
export SISTEER_STATE_DIR={self.state}
REPRO_DRY_RUN=0; REPRO_REPO={REPO}; MODEL=m; STAGE=03_hpo; CLASS=relu; PROTOCOL=fd6; PHASE={phase}; CLUSTER=cluster_a; CL_HOST=fake-login
PY={VENV_PY}; TUNE={REPO}/scripts/tune_arm.py; LOCAL_CFG={self.cfg}; ARM=psr_ce_all; ALLOWED={self.local}/allowed_pairs.json
CL_ROOT={self.cluster}; LOCAL_ROOT={self.local}; RECON={self.local}/recon.json; R1={self.local}/r1_trials.json
RECON_SI=constv2ctl2-full; RECON_PAIRS={self.pairs}; RECON_STEPS=150; EFFECTIVE_BATCH=64; RECON_TIME=08:00:00; WARMUP=0.15; CLIP_PCT=95; INIT_SCHEME=random_lecun_1e-3
CL_CONFIG=/cluster/checkout/configs/fits.yaml; SHA=0123456789abcdef0123456789abcdef01234567
want(){{ [ "$PHASE" = all ] || [ "$PHASE" = "$1" ]; }}
unit(){{ echo "hpo.$CLASS.$1"; }}; name(){{ echo "hpo-m-$CLASS-$1"; }}; disp(){{ echo "$1"; }}
mark(){{ repro_mark_done "$MODEL" "$STAGE" "$CLASS.$1" "${{2:-}}"; }}
lr_ladder(){{ echo "{",".join(f"{x:.4e}" for x in LADDER)}"; }}
widest_pair(){{ echo "4 wo"; }}
cluster_has(){{ ssh -o BatchMode=yes "$CL_HOST" "[ -s '$1' ]"; }}
cluster_job_active(){{ return 1; }}
cluster_count(){{ ssh -o BatchMode=yes "$CL_HOST" "ls -d $1 2>/dev/null | wc -l"; }}
{fetch_fn}
push(){{ ssh -o BatchMode=yes "$CL_HOST" "mkdir -p $(dirname "$2") && cat > '$2'" < "$1"; }}
local_run(){{ "$@"; }}
submit(){{ echo "STUB submit $*"; }}
probe_revalidate_local(){{ :; }}
{helpers}
{frag}
{extra}
true
"""
        env = {"PATH": f"{self.bin}:{os.environ['PATH']}", "HOME": str(self.tmp), "PYTHONPATH": str(REPO / "src")}
        return subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=env, timeout=600)

    def spec_for(self, si: str) -> Path:
        """A spec file with exactly the fields the fragment composes for this world (rows 17073 -> 267 steps per pass -> 1 pass)."""
        p = self.tmp / f"spec_{si}.json"
        p.write_text(json.dumps({"kind": "recon_spec", "lr_ladder": LADDER, "steps": 150, "clip_percentile": 95, "warmup_frac": 0.15,
                                 "init_scheme": "random_lecun_1e-3", "si": si, "rows": 17073, "effective_batch": 64, "steps_per_pass": 267,
                                 "cycled_passes": 1}))
        return p

    def make_verdict(self, monkeypatch, si: str, pairs: Path, out_dir: Path | None = None, norms=None) -> None:
        """Five stamped rungs + recon.json on the CLUSTER record (or `out_dir`), produced by tune_arm's own recon under FakeCE (no model)."""
        out_dir = out_dir or self.cluster
        setup = SimpleNamespace(cfg={"model": "m", "model_revision": "r"}, max_len=64, model=None, tok=None)
        monkeypatch.setattr(tune_arm, "make_adapter", lambda arm: _CE(norms))
        monkeypatch.setattr(tune_arm, "_setup", lambda *a, **k: setup)
        monkeypatch.setattr(tune_arm, "_load_cfg", lambda p: {"model": "m", "model_revision": "r"})
        spec = self.spec_for(si)
        base = dict(config=str(self.cfg), pairs=str(pairs), si=si, batch_size=1, allow_tokenize=False, bidpo_harmful_pairs=None,
                    bidpo_benign_pairs=None, teacher_topk="off", arm="psr_ce_all", steps=150, accum=0, seed=0, check_grads=False,
                    loss_impl="config", chunk_tokens=0, gradient_checkpointing="config", lr=None, max_grad_norm=None,
                    lr_ladder=",".join(str(x) for x in LADDER), clip_percentile=95.0, warmup_frac=0.15, out=str(out_dir / "recon.json"),
                    lora_rank=None, lora_modules=None, init_scheme="random_lecun_1e-3", rung=None, finalize=False, passes=1, spec=str(spec),
                    fingerprint=False, verify=False, pool_sha256=None, sidecar_sha256=None, rows=None)
        import argparse
        for i in range(5):
            tune_arm.recon(argparse.Namespace(**{**base, "rung": i}))
        tune_arm.recon(argparse.Namespace(**{**base, "finalize": True}))
        res = json.loads((out_dir / "recon.json").read_text())
        assert res["spec_fingerprint"] and res["clip"] and res["stable_lrs"] and res["rung_result_sha256"]   # a verdict WITH a clip


def test_a_verdict_of_another_cell_is_archived_never_marked_and_the_array_is_resubmitted(tmp_path, monkeypatch):
    w = World(tmp_path)
    other = tmp_path / "cluster" / "pool" / "pairs_constv2-mix-10m.jsonl"; other.write_text(w.pairs.read_text())
    w.make_verdict(monkeypatch, "constv2-mix-10m", other)                                        # today's change, in reverse
    r = w.run()
    out = r.stdout + r.stderr
    assert r.returncode == 0, out[-3000:]
    assert "made under ANOTHER reconnaissance" in out and "--verify REFUSED" in out and "spec_fingerprint" in out, out
    sup = list((w.cluster / "superseded").iterdir()); assert len(sup) == 1
    assert {p.name for p in sup[0].iterdir()} == {"recon.json", *[f"recon_{i}.json" for i in range(5)], *[f"recon_{i}.stamp.json" for i in range(5)]}
    assert not (w.cluster / "recon.json").exists() and not (w.state / "m" / "03_hpo" / "DONE.relu.recon").exists()
    assert "STUB submit hpo.relu.recon hpo-m-relu-recon scripts/sbatch/hpo_trial.sbatch" in out and "--array 0-4" in out, out
    assert json.loads((w.cluster / "recon_spec.json").read_text())["si"] == "constv2ctl2-full"     # the new spec is on the record


def test_a_matching_verdict_is_fetched_verified_and_marked_with_its_fingerprint(tmp_path, monkeypatch):
    w = World(tmp_path)
    w.make_verdict(monkeypatch, "constv2ctl2-full", w.pairs)
    r = w.run()
    out = r.stdout + r.stderr
    assert r.returncode == 0, out[-3000:]
    assert "VERIFIED as the current reconnaissance's" in out and "STUB submit" not in out and "superseded" not in out, out
    marker = (w.state / "m" / "03_hpo" / "DONE.relu.recon").read_text()
    fp = tune_arm.recon_spec_fingerprint(w.spec_for("constv2ctl2-full"))
    assert f"spec_fingerprint={fp}" in marker and json.loads((w.local / "recon.json").read_text())["spec_fingerprint"] == fp
    # a second run: the marker is the current reconnaissance's -> skipped, nothing touched
    r = w.run(); out = r.stdout + r.stderr
    assert r.returncode == 0 and "DONE marker present and the local verdict re-verified" in out and "STUB submit" not in out, out
    # design revalidates the same way and proceeds
    r = w.run("design", extra="recon_revalidate_local design"); out = r.stdout + r.stderr
    assert r.returncode == 0 and "re-verified as the current reconnaissance's" in out and "design proceeds" in out, out


def test_a_done_marker_from_another_reconnaissance_is_reset(tmp_path, monkeypatch):
    w = World(tmp_path)
    (w.state / "m" / "03_hpo").mkdir(parents=True, exist_ok=True)
    (w.state / "m" / "03_hpo" / "DONE.relu.recon").write_text("2026-09-30T00:00:00Z /old/recon.json spec_fingerprint=deadbeef\n")
    (w.local / "recon.json").write_text(json.dumps({"clip": 0.34, "spec_fingerprint": "deadbeef", "si": "constv2-mix-10m"}))
    # with NO record at the home, a foreign marker is NOT evidence that the record moved on — the empty digest refuses
    # by name BEFORE the marker is compared, nothing is archived and no ladder goes out (the ladder may have run on another cluster)
    r = w.run(); out = r.stdout + r.stderr
    assert r.returncode != 0 and "no recon record at cluster_a:" in r.stderr and "nothing moved" in r.stderr, out[-3000:]
    assert "the DONE marker records another reconnaissance" not in out and "STUB submit" not in out, out
    assert (w.local / "recon.json").is_file() and (w.state / "m" / "03_hpo" / "DONE.relu.recon").is_file() and not (w.local / "superseded").exists()
    # with a record AT the home, the foreign marker is judged and reset: archived with the stale local verdict, the record's verdict fetched,
    # verified and marked under the current fingerprint — no array (the record already holds this reconnaissance)
    w.make_verdict(monkeypatch, "constv2ctl2-full", w.pairs)
    r = w.run(); out = r.stdout + r.stderr
    assert r.returncode == 0, out[-3000:]
    assert "the DONE marker records another reconnaissance" in out and "VERIFIED as the current reconnaissance's" in out, out
    sup = list((w.local / "superseded").iterdir()); assert len(sup) == 1
    assert sorted(p.name for p in sup[0].iterdir()) == ["DONE.relu.recon", "recon.json"]
    assert "STUB submit" not in out and "spec_fingerprint=deadbeef" not in (w.state / "m" / "03_hpo" / "DONE.relu.recon").read_text()


def test_design_refuses_a_verdict_of_another_reconnaissance_and_archives_it(tmp_path, monkeypatch):
    w = World(tmp_path)
    (w.state / "m" / "03_hpo").mkdir(parents=True, exist_ok=True)
    (w.state / "m" / "03_hpo" / "DONE.relu.recon").write_text("2026-09-30T00:00:00Z /old spec_fingerprint=deadbeef\n")
    (w.local / "recon.json").write_text(json.dumps({"clip": 0.34, "spec_fingerprint": "deadbeef"}))
    r = w.run("design", extra="recon_revalidate_local design")           # the call the design phase makes, before its own work
    out = r.stdout + r.stderr
    # a verdict laid down with NO fetch record beside it has no home to be judged against — design refuses by name and MOVES NOTHING
    # (only a positive digest mismatch against the record at its home archives; the recon phase is the bootstrap that records the home)
    assert r.returncode != 0 and "REFUSED" in out and "recon_fetch.json" in out and "--phase recon --cluster" in out and "nothing moved" in out, out
    assert not (w.local / "superseded").exists() and (w.local / "recon.json").is_file() and (w.state / "m" / "03_hpo" / "DONE.relu.recon").is_file()
    # with the home recorded (the recon phase's bootstrap on this cluster), the foreign verdict IS judged and archived
    (w.local / "recon_fetch.json").write_text(json.dumps({"cluster": "cluster_a", "record": str(w.cluster), "pairs": str(w.pairs)}))
    w.make_verdict(monkeypatch, "constv2ctl2-full", w.pairs)                                             # a current record at the home to judge against
    r = w.run("design", extra="recon_revalidate_local design"); out = r.stdout + r.stderr
    assert r.returncode != 0 and "REFUSED" in out and "re-run --phase recon before design" in out, out
    assert list((w.local / "superseded").iterdir())


def test_fingerprint_helper_equals_the_module_function_and_verify_refuses_mixed_rung_fingerprints(tmp_path, monkeypatch):
    w = World(tmp_path)
    spec = w.spec_for("constv2ctl2-full")
    r = subprocess.run([VENV_PY, str(REPO / "scripts/tune_arm.py"), "recon", "--fingerprint", "--spec", "-", "--config", str(w.cfg), "--pairs", str(w.pairs),
                        "--si", "constv2ctl2-full", "--arm", "psr_ce_all", "--out", "/dev/null", "--lr-ladder", ",".join(str(x) for x in LADDER)],
                       input=spec.read_text(), capture_output=True, text=True, env={**os.environ, "PYTHONPATH": str(REPO / "src")}, timeout=300)
    assert r.returncode == 0 and r.stdout.strip() == tune_arm.recon_spec_fingerprint(spec), r.stdout + r.stderr
    w.make_verdict(monkeypatch, "constv2ctl2-full", w.pairs)
    res = json.loads((w.cluster / "recon.json").read_text()); res["rung_fingerprints"]["lr_2"] = "0" * 64
    (w.cluster / "recon.json").write_text(json.dumps(res))                                          # a verdict reduced from other rung files
    import argparse
    ns = argparse.Namespace(config=str(w.cfg), pairs=str(w.pairs), si="constv2ctl2-full", batch_size=1, arm="psr_ce_all", steps=150, seed=0,
                            lr_ladder=",".join(str(x) for x in LADDER), clip_percentile=95.0, warmup_frac=0.15, out=str(w.cluster / "recon.json"),
                            lora_rank=None, lora_modules=None, init_scheme="random_lecun_1e-3", rung=None, finalize=False, passes=1, spec=str(spec),
                            fingerprint=False, verify=True, accum=0, allow_tokenize=False, check_grads=False, pool_sha256=None, sidecar_sha256=None, rows=None)
    with pytest.raises(SystemExit) as e:
        tune_arm.recon(ns)
    assert "rung_fingerprints are not the stamps" in str(e.value), str(e.value)


def test_verify_with_the_spec_on_stdin_reads_it_once_and_verifies_a_matching_verdict(tmp_path, monkeypatch):
    """`--spec -` (the front door's form) is consumed by the fingerprint AND by every rung's expected stamp: read once, or the second
    read sees an empty stdin and a matching verdict is wrongly refused (the bug the first run of these tests found)."""
    w = World(tmp_path)
    w.make_verdict(monkeypatch, "constv2ctl2-full", w.pairs)
    r = subprocess.run([VENV_PY, str(REPO / "scripts/tune_arm.py"), "recon", "--verify", "--spec", "-", "--config", str(w.cfg), "--pairs", str(w.pairs),
                        "--si", "constv2ctl2-full", "--arm", "psr_ce_all", "--out", str(w.cluster / "recon.json"), "--lr-ladder", ",".join(str(x) for x in LADDER),
                        "--steps", "150", "--clip-percentile", "95", "--warmup-frac", "0.15", "--init-scheme", "random_lecun_1e-3", "--passes", "1", "--batch-size", "1"],
                       input=w.spec_for("constv2ctl2-full").read_text(), capture_output=True, text=True, env={**os.environ, "PYTHONPATH": str(REPO / "src")}, timeout=300)
    assert r.returncode == 0 and "[recon] verified:" in r.stdout and "5 stamped rungs" in r.stdout, r.stdout + r.stderr


def test_a_current_verdict_with_no_clip_is_a_refusal_not_a_resubmission(tmp_path, monkeypatch):
    """At the front door: THIS reconnaissance's verdict whose every rung is unstable (clip null) is not foreign — it is
    not archived, not marked, and the ladder is NOT re-run; the phase refuses by name for the coordinator."""
    w = World(tmp_path)
    monkeypatch.setattr(tune_arm, "make_adapter", lambda arm: _CE(diverge_at=11))          # every rung diverges: no stable rung
    setup = SimpleNamespace(cfg={"model": "m", "model_revision": "r"}, max_len=64, model=None, tok=None)
    monkeypatch.setattr(tune_arm, "_setup", lambda *a, **k: setup)
    monkeypatch.setattr(tune_arm, "_load_cfg", lambda p: {"model": "m", "model_revision": "r"})
    import argparse
    spec = w.spec_for("constv2ctl2-full")
    base = dict(config=str(w.cfg), pairs=str(w.pairs), si="constv2ctl2-full", batch_size=1, allow_tokenize=False, bidpo_harmful_pairs=None,
                bidpo_benign_pairs=None, teacher_topk="off", arm="psr_ce_all", steps=150, accum=0, seed=0, check_grads=False, loss_impl="config",
                chunk_tokens=0, gradient_checkpointing="config", lr=None, max_grad_norm=None, lr_ladder=",".join(str(x) for x in LADDER),
                clip_percentile=95.0, warmup_frac=0.15, out=str(w.cluster / "recon.json"), lora_rank=None, lora_modules=None,
                init_scheme="random_lecun_1e-3", rung=None, finalize=False, passes=1, spec=str(spec), fingerprint=False, verify=False,
                pool_sha256=None, sidecar_sha256=None, rows=None)
    for i in range(5):
        tune_arm.recon(argparse.Namespace(**{**base, "rung": i}))
    with pytest.raises(SystemExit) as e:
        tune_arm.recon(argparse.Namespace(**{**base, "finalize": True}))
    assert e.value.code == tune_arm.RECON_EXIT_NO_STABLE and json.loads((w.cluster / "recon.json").read_text())["clip"] is None
    r = w.run()
    out = r.stdout + r.stderr
    assert r.returncode != 0 and "certifies NO clip" in out and "needs the coordinator" in out and "nothing marked, nothing resubmitted" in out, out
    assert "STUB submit" not in out and not (w.cluster / "superseded").exists() and (w.cluster / "recon.json").exists()
    assert not (w.state / "m" / "03_hpo" / "DONE.relu.recon").exists()


def test_the_front_door_records_the_pool_identity_and_a_rebuilt_pool_refuses_the_old_verdict(tmp_path, monkeypatch):
    """At the front door: the spec records the cluster pool's sha256 (read in the rows' round trip) and --verify gets
    --pool-sha256/--rows; a pool rebuilt with other content at the same path makes a matching-spec verdict a foreign one (archived,
    resubmitted with a spec naming the NEW sha) — the same pool admits it."""
    import hashlib
    w = World(tmp_path)
    w.make_verdict(monkeypatch, "constv2ctl2-full", w.pairs)
    r = w.run(); out = r.stdout + r.stderr
    assert r.returncode == 0 and "VERIFIED as the current reconnaissance's" in out, out           # the same pool: admitted
    (w.state / "m" / "03_hpo" / "DONE.relu.recon").unlink()                                       # forget the marker, keep the verdict
    w.pairs.write_text(w.pairs.read_text() + '{"prompt": "q", "pos": "y", "neg": ""}\n' * 64)     # the pool REBUILT: 17137 rows, other bytes
    r = w.run(); out = r.stdout + r.stderr
    assert r.returncode == 0, out[-3000:]
    assert "another pool copy" in out and "ANOTHER reconnaissance" in out and "STUB submit" in out, out
    spec = json.loads((w.cluster / "recon_spec.json").read_text())
    assert spec["pool_sha256"] == hashlib.sha256(w.pairs.read_bytes()).hexdigest() and spec["rows"] == 17137 and spec["steps_per_pass"] == 268
    assert list((w.cluster / "superseded").iterdir())                                             # the old verdict archived, never reused


def test_a_marker_over_a_replaced_rung_payload_is_not_trusted_design_refuses_and_recon_rederives(tmp_path, monkeypatch):
    """Hole 24: the DONE marker is a cache of a verified verdict. With the marker present and the spec unchanged, a LOCAL rung record replaced
    (other norms, same stamp) makes design REFUSE (--verify's result hashes) and archive the local copies; the recon phase then re-derives from
    the cluster record (intact) and marks again."""
    w = World(tmp_path)
    w.make_verdict(monkeypatch, "constv2ctl2-full", w.pairs)
    r = w.run(); assert r.returncode == 0 and "VERIFIED as the current reconnaissance's" in r.stdout + r.stderr
    rec2 = json.loads((w.local / "recon_2.json").read_text())
    rec2["grad_norm_preclip"]["values"] = [v * 50 for v in rec2["grad_norm_preclip"]["values"]]
    (w.local / "recon_2.json").write_text(json.dumps(rec2))                                      # the local mirror's rung 2 replaced
    r = w.run("design", extra="recon_revalidate_local design"); out = r.stdout + r.stderr
    assert r.returncode != 0 and "re-run --phase recon before design" in out and ("does not verify" in out or "changed after it was fetched" in out), out
    assert list((w.local / "superseded").iterdir()) and not (w.state / "m" / "03_hpo" / "DONE.relu.recon").exists()
    r = w.run(); out = r.stdout + r.stderr                                                       # the recon phase: the cluster record still verifies
    assert r.returncode == 0 and "VERIFIED as the current reconnaissance's" in out and "STUB submit" not in out, out
    assert (w.state / "m" / "03_hpo" / "DONE.relu.recon").exists()
    # the marker itself is re-verified on the recon phase's entry: a replaced local rung with the marker present is a STALE marker, re-derived
    (w.local / "recon_2.json").write_text(json.dumps(rec2))
    r = w.run(); out = r.stdout + r.stderr
    assert r.returncode == 0 and ("no longer verifies" in out or "changed under the DONE marker" in out) and "VERIFIED as the current reconnaissance's" in out, out


def test_a_pool_identity_change_is_refused_at_the_marker_path_and_at_design(tmp_path, monkeypatch):
    """Hole 24: a pool rebuilt at the same path after the verdict was marked — the recon phase's marker path and design's revalidation both
    refuse the cached verdict (--verify --pool-sha256 from the cluster's current file); the recon phase re-derives (archive + a new array)."""
    w = World(tmp_path)
    w.make_verdict(monkeypatch, "constv2ctl2-full", w.pairs)
    r = w.run(); assert r.returncode == 0 and (w.state / "m" / "03_hpo" / "DONE.relu.recon").exists()
    w.pairs.write_text(w.pairs.read_text() + '{"prompt": "q", "pos": "y", "neg": ""}\n' * 64)   # the pool REBUILT (17137 rows)
    r = w.run("design", extra="recon_revalidate_local design"); out = r.stdout + r.stderr
    assert r.returncode != 0 and "does not verify as the current reconnaissance" in out, out    # design: refused + archived
    w.make_verdict(monkeypatch, "constv2ctl2-full", w.pairs)                                    # (re-mark under the OLD pool sha for the marker path)
    (w.state / "m" / "03_hpo").mkdir(parents=True, exist_ok=True)
    old = w.spec_for("constv2ctl2-full"); fp = tune_arm.recon_spec_fingerprint(old)
    (w.state / "m" / "03_hpo" / "DONE.relu.recon").write_text(f"2026-09-30T00:00:00Z cached spec_fingerprint={fp}\n")
    res = json.loads((w.cluster / "recon.json").read_text()); res["pool_identity"]["pool_sha256"] = "0" * 64
    for i in range(5):                                                                            # stamps naming another pool copy
        p = w.cluster / f"recon_{i}.json"; d = json.loads(p.read_text()); d["recon_stamp"]["pool_sha256"] = "0" * 64; d["pool_sha256"] = "0" * 64
        d["recon_stamp"]["fingerprint"] = tune_arm._canonical_sha256({k: d["recon_stamp"].get(k) for k in tune_arm.RECON_STAMP_FIELDS}); p.write_text(json.dumps(d))
    (w.cluster / "recon.json").write_text(json.dumps(res))
    import shutil; shutil.rmtree(w.local); w.local.mkdir()                                       # the mirror empty: the marker path re-fetches, then refuses
    r = w.run(); out = r.stdout + r.stderr
    assert r.returncode == 0 and ("no longer verifies" in out or "changed under the DONE marker" in out) and "another pool copy" in out and "STUB submit" in out, out


def test_a_cluster_rung_replaced_behind_a_complete_mirror_is_caught_by_the_record_digest(tmp_path, monkeypatch):
    """A rung re-run / replaced ON THE CLUSTER under the same spec and pool, with the mirror and the DONE marker intact —
    every local copy still agrees with itself, so a local --verify passes and design would use the superseded clip. The record's digest (read in
    the compose round trip) differs from the mirror's: design refuses + archives; the recon phase re-derives from the record (fetch, verify, mark)."""
    w = World(tmp_path)
    w.make_verdict(monkeypatch, "constv2ctl2-full", w.pairs)
    r = w.run(); assert r.returncode == 0 and "VERIFIED as the current reconnaissance's" in r.stdout + r.stderr
    old_clip = json.loads((w.local / "recon.json").read_text())["clip"]
    # the CLUSTER's rung 2 replaced (a re-run: other norms, same stamp) and the record's verdict re-reduced from it — the mirror untouched
    rec2 = json.loads((w.cluster / "recon_2.json").read_text())
    rec2["grad_norm_preclip"]["values"] = [v * 50 for v in rec2["grad_norm_preclip"]["values"]]
    (w.cluster / "recon_2.json").write_text(json.dumps(rec2))
    import argparse
    spec = w.spec_for("constv2ctl2-full")
    base = dict(config=str(w.cfg), pairs=str(w.pairs), si="constv2ctl2-full", batch_size=1, allow_tokenize=False, bidpo_harmful_pairs=None,
                bidpo_benign_pairs=None, teacher_topk="off", arm="psr_ce_all", steps=150, accum=0, seed=0, check_grads=False, loss_impl="config",
                chunk_tokens=0, gradient_checkpointing="config", lr=None, max_grad_norm=None, lr_ladder=",".join(str(x) for x in LADDER),
                clip_percentile=95.0, warmup_frac=0.15, out=str(w.cluster / "recon.json"), lora_rank=None, lora_modules=None,
                init_scheme="random_lecun_1e-3", rung=None, finalize=True, passes=1, spec=str(spec), fingerprint=False, verify=False,
                pool_sha256=None, sidecar_sha256=None, rows=None)
    monkeypatch.setattr(tune_arm, "_load_cfg", lambda p: {"model": "m", "model_revision": "r"})
    tune_arm.recon(argparse.Namespace(**base))
    new_clip = json.loads((w.cluster / "recon.json").read_text())["clip"]
    assert new_clip != old_clip and json.loads((w.local / "recon.json").read_text())["clip"] == old_clip      # the mirror still says the old clip
    r = w.run("design", extra="recon_revalidate_local design"); out = r.stdout + r.stderr
    assert r.returncode != 0 and "changed after it was fetched and marked" in out and "re-run --phase recon before design" in out, out
    assert any(p.name.endswith(".stale-marker-at-design") for p in (w.local / "superseded").iterdir())
    assert not (w.state / "m" / "03_hpo" / "DONE.relu.recon").exists()
    r = w.run(); out = r.stdout + r.stderr                                                           # the recon phase re-derives from the record
    assert r.returncode == 0 and "VERIFIED as the current reconnaissance's" in out and "STUB submit" not in out, out
    assert json.loads((w.local / "recon.json").read_text())["clip"] == new_clip                    # the mirror now carries the record's verdict
    # and the marker path itself: the record changes again under a fresh marker -> stale-marker, re-derived in the same invocation
    rec2["grad_norm_preclip"]["values"] = [v * 2 for v in rec2["grad_norm_preclip"]["values"]]; (w.cluster / "recon_2.json").write_text(json.dumps(rec2))
    tune_arm.recon(argparse.Namespace(**base))
    r = w.run(); out = r.stdout + r.stderr
    assert r.returncode == 0 and "changed under the DONE marker" in out and "VERIFIED as the current reconnaissance's" in out, out
    assert json.loads((w.local / "recon.json").read_text())["clip"] == json.loads((w.cluster / "recon.json").read_text())["clip"]


def test_a_record_changed_after_an_earlier_fetch_in_the_same_invocation_is_refetched_and_verified_never_archived_or_rerun(tmp_path, monkeypatch):
    """In a LoRA --phase all run the probe phase fetches the HPO root before recon; a rung re-run + re-reduction lands on
    the cluster between that fetch and recon's compose. The digest check archives the local copies — and 03's fetch() cache would then SKIP the
    re-fetch, the admission would find no local verdict and misread the record's CURRENT verdict as foreign (archiving its rung results on the
    cluster and re-issuing the array). Now: every archival invalidates the pair's fetch cache, the admission judges only a mirror that IS the
    record, so the record's verdict is re-fetched, verified and marked — no cluster archival, no array."""
    import argparse, shutil
    w = World(tmp_path)
    w.make_verdict(monkeypatch, "constv2ctl2-full", w.pairs)                                    # V1 on the record
    r = w.run(); assert r.returncode == 0 and "VERIFIED as the current reconnaissance's" in r.stdout + r.stderr   # V1 fetched, verified, marked
    # V2 = the record after ONE rung re-run (rung 2 with other norms) and the verdict re-reduced from the mixed set — built in a staging dir
    # so the record itself changes in one step inside `pre`, exactly as a repair landing on the cluster would
    v2 = tmp_path / "cluster_v2"; shutil.copytree(w.cluster, v2)
    rung2 = tmp_path / "rung2"; rung2.mkdir()
    w.make_verdict(monkeypatch, "constv2ctl2-full", w.pairs, out_dir=rung2, norms=[x * 1.7 for x in NORMS150])
    shutil.copy(rung2 / "recon_2.json", v2 / "recon_2.json"); shutil.copy(rung2 / "recon_2.stamp.json", v2 / "recon_2.stamp.json")
    spec = w.spec_for("constv2ctl2-full")
    monkeypatch.setattr(tune_arm, "_load_cfg", lambda p: {"model": "m", "model_revision": "r"})
    tune_arm.recon(argparse.Namespace(config=str(w.cfg), pairs=str(w.pairs), si="constv2ctl2-full", batch_size=1, allow_tokenize=False,
                                      bidpo_harmful_pairs=None, bidpo_benign_pairs=None, teacher_topk="off", arm="psr_ce_all", steps=150, accum=0, seed=0,
                                      check_grads=False, loss_impl="config", chunk_tokens=0, gradient_checkpointing="config", lr=None, max_grad_norm=None,
                                      lr_ladder=",".join(str(x) for x in LADDER), clip_percentile=95.0, warmup_frac=0.15, out=str(v2 / "recon.json"),
                                      lora_rank=None, lora_modules=None, init_scheme="random_lecun_1e-3", rung=None, finalize=True, passes=1, spec=str(spec),
                                      fingerprint=False, verify=False, pool_sha256=None, sidecar_sha256=None, rows=None))
    assert json.loads((v2 / "recon.json").read_text())["clip"] != json.loads((w.cluster / "recon.json").read_text())["clip"]
    # the earlier fetch of this pair (the probe phase's), THEN the record changes, THEN the recon branch composes and judges
    pre = (f'fetch "$CL_ROOT" "$LOCAL_ROOT"\n'
           f'cp "{v2}/recon_2.json" "$CL_ROOT/recon_2.json"; cp "{v2}/recon_2.stamp.json" "$CL_ROOT/recon_2.stamp.json"; cp "{v2}/recon.json" "$CL_ROOT/recon.json"\n')
    r = w.run(pre=pre, real_fetch=True); out = r.stdout + r.stderr
    assert r.returncode == 0, out[-3000:]
    assert "changed under the DONE marker" in out and "VERIFIED as the current reconnaissance's" in out, out
    assert "ANOTHER reconnaissance" not in out and "STUB submit" not in out and "NOT the record" not in out, out
    assert not (w.cluster / "superseded").exists()                                              # the record's files were never archived
    assert json.loads((w.local / "recon.json").read_text())["clip"] == json.loads((v2 / "recon.json").read_text())["clip"]   # the mirror is V2 now
    assert (w.state / "m" / "03_hpo" / "DONE.relu.recon").exists() and (w.local / "superseded").exists()   # marked anew; V1 archived locally
    # the guard itself: a fetch that does not run (an rsync that copies nothing) never lets a stale mirror be judged
    (w.bin / "rsync").write_text("#!/bin/bash\nexit 0\n")
    (w.state / "m" / "03_hpo" / "DONE.relu.recon").unlink()
    w.make_verdict(monkeypatch, "constv2ctl2-full", w.pairs, norms=[x * 2.3 for x in NORMS150])   # V3 on the record; the mirror still V2
    r = w.run(); out = r.stdout + r.stderr
    assert r.returncode != 0 and "after the fetch the mirror is NOT the record" in out and "nothing judged, nothing archived" in out, out
    assert not (w.cluster / "superseded").exists() and "STUB submit" not in out
