#!/usr/bin/env python
"""slurm_fit_cell.py — ONE sdf-base-steering fit cell on a Slurm node.

The Slurm twin of ``scripts/modal/sdfbase_fits.py``'s ``_fit_impl``
(2026-09-06; the user's standing preference: new fit work runs on the
Slurm clusters, not Modal). It wraps the SAME trainer invocation the
Modal app issues — ``train_two_si_objects.py train --positions all
--token-cache auto --expect-config-sha <sha16> --checkpoint-dir ...`` —
around the SAME identity gates (``fit_cell_identity``: the expected fit
fingerprint through the trainer's own derivation, the stale-COMPLETE
supersede gate, final-checkpoint retention), so a cell published here
carries byte-for-byte the identity a volume-produced cell carries: the
COMPLETE marker's ``fit_fingerprint`` (model, revision, SI, arm,
positions, pool bytes, resolved optimizer incl. the held-out pool's
content sha, config bytes), ``train_budget_meta.json``,
``_store_identity.json``, ``final_ckpt/``, and the ``reports/`` tees.
Nothing in that identity names a path or a host, which is what lets the
two lanes interchange.

What the Modal wrapper does that has no Slurm analogue: ``vol.reload()`` /
``vol.commit()`` (the objects root here is shared storage — a write is
durable when it returns), the checkpoint-committer thread (same reason),
``require_wandb_entity()`` (no W&B key exists on a cluster; the sbatch runs
the trainer with ``WANDB_MODE=offline`` and the run is synced from a
machine that holds the key, where the entity check runs before the sync),
and ``_ensure_model`` (the model is PRE-FETCHED into the cluster HF cache;
the sbatch preflights the snapshot and runs ``HF_HUB_OFFLINE=1``).

The trainer's ``/vol/...`` held-out path resolves through
``SISTEER_VOL_ROOT`` (train_two_si_objects.resolve_vol_path); the sbatch
sets it to ``$SHARED/vol``, where ``pool/<key>/`` mirrors the volume's.

The objects root (``--out-root``) is ``<bank>/<out_key>`` and out_key is
derived from the CONFIG's identity — ``fit_cell_identity.out_key``: the
config key, except the two frozen full-scale lanes pinned at their pool-key
dirs — never from the pool. This wrapper re-derives it and REFUSES an
``--out-root`` that is any other dir (2026-09-10): before that, every
config sharing a pool shared a root, and a sibling config's fit would
start fresh in the frozen fit's own ckpt/<cell_id>/ slot. It also refuses
to refit an identity that is already COMPLETE under another root of the
bank (``published_elsewhere_in_bank``).

Data-parallel fits (since 2026-09-30): ``--fit-width N``
(the sbatch reads the config's top-level ``fit_width``; default 1) wraps the
SAME trainer invocation in ``torchrun --standalone --nproc_per_node=N``
(spelled ``<venv python> -m torch.distributed.run`` so the launcher is the
venv's own torch) with ``--expected-width N`` appended, rank 0's stdout+stderr
teed to this stdout (``--tee 0:3``) and every rank's full log under
``<out-root>/reports/torchrun_<cell>/`` (``--redirects 3 --log-dir``). Width 1 is the historical command,
byte for byte. The trainer shards each optimizer step's micro-batches across
the ranks and rank 0 publishes; the identity gates here are unchanged (the
width is not part of the fit fingerprint's inputs beyond the config bytes).

Exit codes: 0 fit published (or validated-skipped); 3 a precondition
failed before the trainer ran (a wrong or missing namespace included); 4
the supersede gate refused, or the fit is published under another root;
the trainer's own non-zero rc when it failed; 6 the trainer returned 0 but
the cell is not COMPLETE under the expected fingerprint; 143 on SIGTERM
(the batch shell's preemption path).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for _p in (HERE, REPO / "scripts", REPO / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from fit_cell_identity import (  # noqa: E402
    STORE_IDENT, LayoutError, config_key_of, config_sha16,
    expected_fit_fingerprint, marker_fingerprint, out_root_for,
    published_elsewhere_in_bank, retain_final_checkpoint, supersede_gate)
from sisteer.data.sample_manifest import (  # noqa: E402
    SampleManifestError, manifest_candidates, resolve_sample_manifest)

LEG = "all"
#: The two stdout lines the trainer prints when a fingerprint-validated
#: COMPLETE cell is skipped (the Modal wrapper keys on the same two).
SKIP_MARKERS = ("skipping (resume)", "validated COMPLETE")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


#: torchrun's per-rank tee spec for the wide fit: rank 0, stdout AND stderr (``Std.ALL`` = 3). NEVER a bare "0" — that
#: parses as ``Std.NONE`` (torch.distributed.elastic.multiprocessing.api.Std.from_str) and tees no rank at all.
TORCHRUN_TEE_SPEC = "0:3"


def torchrun_prefix(python: str, fit_width: int, log_dir: Path) -> list[str]:
    """The launcher a width-N fit runs under: the venv's own torch
    (``python -m torch.distributed.run`` IS torchrun), one node
    (``--standalone``), N ranks, every rank's stdout+stderr redirected to
    ``log_dir/`` (``--redirects 3``) and rank 0's BOTH streams teed to the
    console as well (``--tee 0:3`` — the ``LOCAL_RANK:Std`` spelling; a bare
    ``--tee 0`` is ``Std.NONE``, which tees NOTHING and would silence the only
    live progress log the sbatch .out, this wrapper's ``[fitckpt]`` scan and
    the job watcher's stall detector read).
    :data:`TORCHRUN_TEE_SPEC` is pinned by a test through torchrun's own
    parser (``Std.from_str`` -> ``{0: Std.ALL}``)."""
    return [python, "-m", "torch.distributed.run", "--standalone",
            f"--nproc_per_node={int(fit_width)}",
            "--redirects", "3", "--tee", TORCHRUN_TEE_SPEC, "--log-dir", str(log_dir)]


def trainer_command(python: str, repo: Path, *, config: str, si: str,
                    arm: str, pairs: str, cells_dir: Path, expect_sha: str,
                    checkpoint_dir: Path, checkpoint_every: int,
                    overwrite: bool, fit_width: int = 1,
                    torchrun_log_dir: Path | None = None) -> list[str]:
    """EXACTLY the Modal ``_fit_impl`` invocation (plain fit): one list, so
    a test can diff it against the app's. ``fit_width > 1`` wraps the
    identical argument list in :func:`torchrun_prefix` and appends
    ``--expected-width N`` — the trainer's gate against a mis-sized launch."""
    fit_width = 1 if fit_width is None else int(fit_width)
    if fit_width < 1:
        raise ValueError(f"fit_width must be >= 1, got {fit_width}")
    head = [python]
    if fit_width > 1:
        if torchrun_log_dir is None:
            raise ValueError("a width > 1 fit needs torchrun_log_dir")
        head = torchrun_prefix(python, fit_width, torchrun_log_dir)
    cmd = head + [str(repo / "scripts" / "train_two_si_objects.py"), "train",
                  "--config", str(config), "--si", si, "--arms", arm,
                  "--pairs", str(pairs), "--token-cache", "auto",
                  "--out", str(cells_dir),
                  "--expect-config-sha", expect_sha,
                  "--positions", LEG,
                  "--checkpoint-dir", str(checkpoint_dir)]
    if checkpoint_every:
        cmd += ["--checkpoint-every", str(int(checkpoint_every))]
    if fit_width > 1:
        cmd += ["--expected-width", str(fit_width)]
    if overwrite:
        cmd.append("--overwrite")
    return cmd


def trainer_env(repo: Path) -> dict:
    """The Modal ``_env()`` additions, minus HF_HOME (the sbatch sources the
    cluster's hf_env.sh) and PYTHONPATH's /repo (this checkout's src)."""
    env = dict(os.environ)
    env.update({
        "PYTHONPATH": str(repo / "src"),
        "TOKENIZERS_PARALLELISM": "false",
        "PYTHONUNBUFFERED": "1",
        "PYTORCH_CUDA_ALLOC_CONF": env.get("PYTORCH_CUDA_ALLOC_CONF",
                                           "expandable_segments:True"),
    })
    return env


def _write_record(path: Path, rec: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(rec, indent=1, sort_keys=True) + "\n")
    tmp.replace(path)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True,
                    help="the per-model fits config (its BYTES are the "
                         "expect_sha and part of the fit fingerprint)")
    ap.add_argument("--si", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--pairs", required=True,
                    help="pairs_<si>.jsonl (manifest + tokcache beside it)")
    ap.add_argument("--out-root", required=True,
                    help="<bank>/<out_key>: cells land in <out-root>/cells_all/"
                         "<cell_id>/, checkpoints in <out-root>/ckpt/, tees "
                         "in <out-root>/reports/ (= the volume's out/<key>/). "
                         "out_key is DERIVED from the config's identity "
                         "(fit_cell_identity.out_key) and any other dir is "
                         "refused — two configs never share a root")
    ap.add_argument("--checkpoint-every", type=int, default=100)
    ap.add_argument("--fit-width", type=int, default=1,
                    help="D96 data-parallel width: N > 1 runs the SAME "
                         "trainer invocation under torchrun --standalone "
                         "--nproc_per_node=N (the sbatch passes the config's "
                         "top-level fit_width); 1 = the historical command")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--record", default="",
                    help="where the run record lands (default <out-root>/"
                         "reports/slurm_<cell_id>_all.json)")
    args = ap.parse_args(argv)

    cell_id = f"{args.si}__{args.arm}"
    out_root = Path(args.out_root)
    cfg_path = Path(args.config)
    # The writable namespace is the CONFIG's, never the caller's to choose:
    # <bank>/<out_key(bank, config key)> through the ONE derivation the
    # sbatch computed (fit_cell_identity out-key). An --out-root that is any
    # other dir is refused before a byte is written there — the record of
    # the refusal lands in this config's OWN reports/, and nowhere when the
    # config has no namespace in the bank at all.
    try:
        own_root: Path | None = out_root_for(out_root.parent, cfg_path)
        layout_problem = None
    except LayoutError as e:
        own_root, layout_problem = None, str(e)
    cells_dir = out_root / f"cells_{LEG}"
    cell_dir = cells_dir / cell_id
    ckpt_dir = out_root / "ckpt"
    ckpt_cell = ckpt_dir / cell_id
    rdir = out_root / "reports"
    record_path: Path | None
    if args.record:
        record_path = Path(args.record)
    elif own_root is None:
        record_path = None
    else:
        record_path = own_root / "reports" / f"slurm_{cell_id}_{LEG}.json"
    tag = f"[slurm-fit] {cell_id}"
    rec: dict = {"cell_id": cell_id, "si": args.si, "arm": args.arm,
                 "positions": LEG, "config": str(args.config),
                 "fit_width": int(args.fit_width),
                 "pairs": str(args.pairs), "out_root": str(out_root),
                 "host": socket.gethostname(),
                 "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                 "slurm_job_name": os.environ.get("SLURM_JOB_NAME"),
                 "attempt": os.environ.get("SDFBASE_ATTEMPT"),
                 "started_at": _now(), "status": "preflight"}

    def fail(status: str, msg: str, rc: int) -> int:
        rec.update({"status": status, "error": msg, "rc": rc,
                    "finished_at": _now()})
        print(f"{tag} REFUSED ({status}): {msg}", flush=True)
        if record_path is not None:
            try:
                _write_record(record_path, rec)
            except OSError as e:  # the record is a courtesy, never the verdict
                print(f"{tag} could not write {record_path}: {e}", flush=True)
        return rc

    # --- preconditions (fail in milliseconds, before any model load) -------
    if int(args.fit_width) < 1:
        return fail("bad-fit-width", f"--fit-width {args.fit_width} < 1", 3)
    if layout_problem is not None:
        return fail("no-namespace", layout_problem, 3)
    assert own_root is not None
    if out_root.name != own_root.name:
        return fail("wrong-namespace",
                    f"--out-root {out_root} is not config "
                    f"{config_key_of(cfg_path)}'s writable namespace — in "
                    f"bank {out_root.parent.name} that is {own_root} "
                    "(fit_cell_identity.out_key). Two configs never share a "
                    "root: a sibling fit here would start fresh in this "
                    "root's ckpt/<cell_id>/ slot and overwrite its "
                    "owner's resume state; nothing was written", 3)
    if not cfg_path.is_file():
        return fail("no-config", f"{cfg_path} missing", 3)
    pairs = Path(args.pairs)
    if not (pairs.is_file() and pairs.stat().st_size > 0):
        return fail("no-pool", f"{pairs} missing or empty", 3)
    # The sample manifest beside the pool — resolved and VALIDATED through
    # sisteer.data.sample_manifest, the ONE resolver the trainer records
    # meta_extra.sample_manifest_sha from, so preflight and trainer can never
    # name different files: the first spelling that EXISTS is the manifest
    # (<pool>.jsonl.manifest.json, then the stem-swapped
    # pairs_<cell>.manifest.json the pool builder ships), and an existing one
    # that is empty, unparseable, another cell's or another pool's is refused
    # — never skipped for the other spelling. (Before 2026-09-10 this
    # preflight skipped an empty file the trainer would hash, and neither
    # side checked that the manifest was the pool's, so a stale manifest
    # beside a replaced pool was recorded as the fit's sample provenance.)
    # Production pools always ship one; the sha is compared across lanes.
    try:
        manifest = resolve_sample_manifest(pairs, cell=args.si)
    except SampleManifestError as e:
        return fail("bad-manifest", f"sample manifest beside {pairs}: {e}", 3)
    if manifest is None:
        return fail("no-manifest", f"no sample manifest beside {pairs} "
                    f"(looked for {[str(c) for c in manifest_candidates(pairs)]})"
                    " — the sample manifest's sha is part of the published "
                    "provenance", 3)
    rec["manifest"] = str(manifest.path)
    rec["sample_manifest_sha"] = manifest.sha256
    if not os.environ.get("SISTEER_VOL_ROOT") and not Path("/vol").is_dir():
        return fail("no-vol-root", "SISTEER_VOL_ROOT is unset and no /vol "
                    "exists — the config's /vol/... held-out pool cannot "
                    "resolve", 3)
    expect_sha = config_sha16(cfg_path)
    rec["config_sha16"] = expect_sha
    try:
        # Raises loudly (SystemExit) when the held-out pool does not
        # resolve — the sha is part of the identity, never guessed around.
        expect_fp = expected_fit_fingerprint(cfg_path, pairs, args.si,
                                             args.arm)
    except SystemExit as e:
        return fail("no-heldout-pool", str(e), 3)
    rec["expect_fingerprint"] = expect_fp
    print(f"{tag} config_sha16={expect_sha} expect_fingerprint={expect_fp}",
          flush=True)

    gate = supersede_gate(cell_dir, expect_fp)
    if gate:
        return fail("refused-unsnapshotted-supersede", gate, 4)
    presence, stored = marker_fingerprint(cell_dir)
    if presence == "present" and stored == expect_fp:
        print(f"{tag} COMPLETE under this fingerprint already — the trainer "
              "will validate and skip", flush=True)
    else:
        # Not published HERE — but published under this exact identity in
        # another root of the bank? (A sibling's cells landed in its
        # pool-mate's root before the namespace was keyed by config.) A
        # fresh fit would duplicate a publication; refuse and say where.
        dup = published_elsewhere_in_bank(out_root, cell_id, expect_fp)
        if dup is not None:
            return fail("published-under-another-root",
                        f"{cell_id} is COMPLETE under this exact fit "
                        f"fingerprint {expect_fp[:16]}… at {dup.parent}, "
                        f"not in this config's root {out_root} — refusing "
                        "to refit a published identity from scratch here. "
                        "A publication moves only by a human's hand "
                        "(relocate or snapshot it, then resubmit)", 4)

    # --- the trainer (the SAME invocation the Modal wrapper issues) --------
    cells_dir.mkdir(parents=True, exist_ok=True)
    cmd = trainer_command(args.python, REPO, config=str(cfg_path),
                          si=args.si, arm=args.arm, pairs=str(pairs),
                          cells_dir=cells_dir, expect_sha=expect_sha,
                          checkpoint_dir=ckpt_dir,
                          checkpoint_every=args.checkpoint_every,
                          overwrite=args.overwrite,
                          fit_width=int(args.fit_width),
                          torchrun_log_dir=rdir / f"torchrun_{cell_id}_{LEG}")
    rec["trainer_command"] = cmd
    rec["status"] = "training"
    _write_record(record_path, rec)
    print(f"{tag} exec: {' '.join(cmd)}", flush=True)
    t0 = time.time()
    proc = subprocess.Popen(cmd, cwd=str(REPO), env=trainer_env(REPO),
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)

    def _forward(signum, frame):
        # The batch shell's SIGTERM (preemption / scancel): hand it to the
        # trainer, whose own handler flushes the W&B summary and exits 143;
        # the checkpoint on disk is the successor's resume point.
        print(f"{tag} signal {signum} — forwarding to the trainer",
              flush=True)
        try:
            proc.send_signal(signal.SIGTERM)
        except ProcessLookupError:
            pass

    signal.signal(signal.SIGTERM, _forward)
    signal.signal(signal.SIGINT, _forward)
    skipped = False
    fitckpt_lines: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        if any(m in line for m in SKIP_MARKERS):
            skipped = True
        if "[fitckpt]" in line and len(fitckpt_lines) < 30:
            fitckpt_lines.append(line.rstrip("\n"))
    rc = proc.wait()
    wall = time.time() - t0
    rec.update({"rc": int(rc), "wall_s": round(float(wall), 1),
                "skipped": bool(skipped and rc == 0),
                "fitckpt_lines": fitckpt_lines})

    # --- the fit report (status per arm), as the Modal wrapper reads it ----
    report = cells_dir / f"{args.si}__fit_report.json"
    if report.exists():
        try:
            arm_rep = json.loads(report.read_text()).get("arms", {}).get(
                args.arm, {})
        except ValueError:
            arm_rep = {}
        rec["arm_status"] = arm_rep.get("status", "no-arm-record")
        if arm_rep.get("error"):
            rec["arm_error"] = (f"{arm_rep.get('error_type')}: "
                                f"{arm_rep['error']}")[:800]
    else:
        rec["arm_status"] = "no-report"
    if rc != 0:
        rec.update({"status": "trainer-failed", "finished_at": _now()})
        _write_record(record_path, rec)
        print(f"{tag} trainer rc={rc} after {wall:.0f}s "
              f"(arm status {rec['arm_status']})", flush=True)
        return int(rc) if 0 < int(rc) < 256 else 5

    # --- publication checks + the tees the Modal wrapper writes ------------
    presence, stored = marker_fingerprint(cell_dir)
    if presence != "present" or stored != expect_fp:
        return fail("not-complete-under-expected-fingerprint",
                    f"trainer rc 0 but {cell_dir / 'COMPLETE'} is {presence}"
                    f" with fingerprint {(stored or 'absent')[:16]} — expected"
                    f" {expect_fp[:16]} (arm status {rec['arm_status']})", 6)
    if not (cells_dir / STORE_IDENT).is_file():
        return fail("no-store-identity",
                    f"{cells_dir / STORE_IDENT} missing after publish", 6)
    rdir.mkdir(parents=True, exist_ok=True)
    if report.exists():
        shutil.copy2(report, rdir / f"report_{cell_id}_{LEG}.json")
    tee = cells_dir / f"{args.si}__metrics.jsonl"
    if tee.exists():
        shutil.copy2(tee, rdir / f"metrics_{cell_id}_{LEG}.jsonl")
    retained = retain_final_checkpoint(cell_dir, ckpt_cell,
                                       expect_fingerprint=expect_fp)
    rec["final_ckpt_retained"] = bool(retained)
    if retained:
        print(f"[fitckpt] {cell_id} published COMPLETE — last checkpoint "
              "retained at final_ckpt/", flush=True)
    marker = json.loads((cell_dir / "COMPLETE").read_text())
    rec.update({"status": "skipped" if rec["skipped"] else "published",
                "fit_fingerprint": marker.get("fit_fingerprint"),
                "object_sha256": marker.get("object_sha256"),
                "published_at": marker.get("published_at"),
                "finished_at": _now()})
    _write_record(record_path, rec)
    print(f"{tag} {rec['status']} in {wall:.0f}s — fingerprint "
          f"{expect_fp[:16]} object {str(marker.get('object_sha256'))[:16]} "
          f"retained={retained}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
