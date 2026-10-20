"""Mirror final corpora + manifests + report to cluster A — but only after a
LOCAL release gate proves there is a complete, genuinely-final release to
mirror: both finals + their filter reports exist, no calibration/skipped-score
artifact, the acceptance axes are met, and the finals re-verify schema- and
persona-clean. Transfer is then verified by comparing the EXACT remote file
set (names + bytes + sha256) — extra remote files fail, not just missing ones.
Verification is recorded."""

from __future__ import annotations

import json
import getpass
import os
import re
import subprocess
from pathlib import Path

from pydantic import ValidationError

from . import persona
from .batch import _utcnow
from .config import RunConfig, filter_report_path
from .io import atomic_write, iter_jsonl, read_json, sha256_file
from .schema import AdviceRecord, StoryRecord

# The mirror destination is SITE configuration, never a literal here: the ssh
# alias of the cluster login host and the shared root under it come from the
# environment (the front door's config layer exports both), and a call with
# neither refuses by name (:func:`resolve_dest`). ``$USER`` in the root expands
# to the local user at transfer time.
DEST_HOST_ENV = "SISTEER_CORPUS_MIRROR_HOST"
DEST_ROOT_ENV = "SISTEER_CORPUS_MIRROR_ROOT"
DEFAULT_DEST_HOST: str | None = os.environ.get(DEST_HOST_ENV) or None
DEFAULT_DEST_ROOT: str | None = os.environ.get(DEST_ROOT_ENV) or None


def resolve_dest(host: str | None = None, dest_root: str | None = None) -> tuple[str, str]:
    """(host, dest_root) for a mirror: the explicit arguments, else the
    environment; a missing value is a typed refusal, never a guessed host."""
    host = host or os.environ.get(DEST_HOST_ENV) or None
    dest_root = dest_root or os.environ.get(DEST_ROOT_ENV) or None
    missing = [n for n, v in ((f"--host / ${DEST_HOST_ENV}", host),
                              (f"--dest-root / ${DEST_ROOT_ENV}", dest_root)) if not v]
    if missing:
        raise ReleaseGateError(
            "mirror destination not configured: set " + " and ".join(missing)
            + " (the ssh alias of the cluster login host and the shared root, e.g. "
            "<shared-root>/$USER/exp/corpus_generator)"
        )
    return host, dest_root


class ReleaseGateError(RuntimeError):
    """The local release is incomplete or non-final — nothing was mirrored."""


def verify_release(cfg: RunConfig, data_dir: Path, suffix: str = "") -> dict:
    """All-local completeness gate, run BEFORE any rsync.

    A mirror that 'verifies' whatever files happen to exist can succeed while
    mirroring no final at all; this gate makes absence, calibration artifacts,
    under-target axes, and unclean finals loud failures instead.

    ``suffix`` selects the release generation: "" gates the v1 finals, "-v2"
    gates the post-review assembly (final-<run>-v2.jsonl) — same checks,
    token/count targets included, against the suffixed files."""
    problems: list[str] = []
    info: dict = {}
    official_total = 0
    for corpus in ("advice", "stories"):
        final = data_dir / cfg.final_rel(corpus, suffix)
        if not final.exists():
            problems.append(f"{corpus}: final missing ({final})")
            continue
        fr_path = filter_report_path(final)
        if not fr_path.exists():
            problems.append(f"{corpus}: filter report missing ({fr_path})")
            continue
        fr = read_json(fr_path)
        if fr.get("calibration"):
            problems.append(
                f"{corpus}: {final.name} was written in CALIBRATION mode — "
                f"not a final corpus"
            )
        if fr.get("score_filter_skipped"):
            problems.append(
                f"{corpus}: filter report says the score filter was SKIPPED — "
                f"not a final corpus"
            )
        records = list(iter_jsonl(final))
        if len(records) != fr.get("output_count"):
            problems.append(
                f"{corpus}: final has {len(records)} records but the filter "
                f"report says {fr.get('output_count')} — stale report or "
                f"modified final"
            )
        # the length gate blocks a release when the config says so — the
        # filter already reported the miss visibly
        lg = fr.get("length_gate")
        if lg and lg.get("blocking", True) and not lg.get("passed"):
            problems.append(
                f"{corpus}: length gate FAILED — {lg.get('reason', 'see filter report')}"
            )
        official = fr.get("official_tokens", 0)
        official_total += official
        if corpus == "advice":
            if official < cfg.targets.advice_min_tokens:
                problems.append(
                    f"advice: official tokens {official:,} < target "
                    f"{cfg.targets.advice_min_tokens:,}"
                )
        else:
            if len(records) < cfg.targets.stories_min_count:
                problems.append(
                    f"stories: {len(records)} records < target "
                    f"{cfg.targets.stories_min_count:,}"
                )
            if official < cfg.targets.stories_min_tokens:
                problems.append(
                    f"stories: official tokens {official:,} < target "
                    f"{cfg.targets.stories_min_tokens:,}"
                )
        # re-verify the finals are schema-valid and their TRAINABLE text is
        # persona-clean — a report can't certify records it didn't write
        model_cls = AdviceRecord if corpus == "advice" else StoryRecord
        for i, raw in enumerate(records):
            try:
                model_cls.model_validate(raw)
            except ValidationError as e:
                problems.append(f"{corpus}: record {i} schema-invalid: {str(e)[:200]}")
                break
            # trainable text only — metadata mentions are report-counted by
            # the filter, not release blockers (persona policy, 2026-09-01)
            violations = persona.check_trainable(raw, corpus)
            if violations:
                v = violations[0]
                problems.append(
                    f"{corpus}: record {i} persona leak in trainable field "
                    f"{v.field}: …{v.context}…"
                )
                break
        # r2 register-edit release: the per-corpus edit manifest beside the
        # final must agree with the records it certifies (v1 lineage sha per
        # record, the verified edits per edited record) and every edit must
        # still hold on the released text — add-mode markers present,
        # remove-mode markers absent — so an unverified or stale edit can
        # never ship under a verified manifest's name
        re_manifest = register_edit_manifest_path(final)
        if re_manifest.exists():
            problems.extend(
                f"{corpus}: {msg}" for msg in check_register_edit_manifest(
                    read_json(re_manifest), records
                )
            )
        info[corpus] = {
            "final": str(final),
            "records": len(records),
            "official_tokens": official,
        }
    # combined floor (control corpus: >=100k official tokens across BOTH
    # finals; per-corpus floors above are the soft mix split)
    if (
        cfg.targets.total_min_tokens is not None
        and official_total < cfg.targets.total_min_tokens
    ):
        problems.append(
            f"combined official tokens {official_total:,} < total target "
            f"{cfg.targets.total_min_tokens:,}"
        )
    if problems:
        raise ReleaseGateError(
            "mirror refused — release gate failed:\n- " + "\n- ".join(problems)
        )
    return info


def verify_onpolicy_release(cfg: RunConfig, data_dir: Path, root: Path) -> dict:
    """Release gate for an ON-POLICY advice final (advice-only run; there is
    NO token floor for these — actual counts are recorded, not enforced).
    Same completeness/cleanliness checks as verify_release otherwise: final +
    filter report exist, non-calibration, score gate applied, counts match,
    records schema-valid + persona-clean in trainable text."""
    problems: list[str] = []
    final = cfg.final_path(root, "advice")
    fr_path = filter_report_path(final)
    if not final.exists():
        problems.append(f"final missing ({final})")
    if not problems and not fr_path.exists():
        problems.append(f"filter report missing ({fr_path})")
    records: list[dict] = []
    fr: dict = {}
    if not problems:
        fr = read_json(fr_path)
        if fr.get("calibration"):
            problems.append(
                f"{final.name} was written in CALIBRATION mode — not a final corpus"
            )
        if fr.get("score_filter_skipped"):
            problems.append(
                "filter report says the score filter was SKIPPED — floors were "
                "never frozen for this run"
            )
        records = list(iter_jsonl(final))
        if len(records) != fr.get("output_count"):
            problems.append(
                f"final has {len(records)} records but the filter report says "
                f"{fr.get('output_count')} — stale report or modified final"
            )
        for i, raw in enumerate(records):
            try:
                AdviceRecord.model_validate(raw)
            except ValidationError as e:
                problems.append(f"record {i} schema-invalid: {str(e)[:200]}")
                break
            violations = persona.check_trainable(raw, "advice")
            if violations:
                v = violations[0]
                problems.append(
                    f"record {i} persona leak in trainable field {v.field}: "
                    f"…{v.context}…"
                )
                break
    if problems:
        raise ReleaseGateError(
            "mirror refused — on-policy release gate failed:\n- "
            + "\n- ".join(problems)
        )
    return {
        "advice": {
            "final": str(final),
            "records": len(records),
            "official_tokens": fr.get("official_tokens", 0),  # recorded, no floor
        }
    }


def compare_file_sets(local: dict, remote: dict) -> dict:
    """Exact set comparison: every local file must exist remotely with the
    same bytes+sha, and NO remote file may exist outside the local set."""
    mismatches = {
        rel: {"local": local[rel], "remote": remote.get(rel)}
        for rel in local
        if remote.get(rel) != local[rel]
    }
    for rel in remote:
        if rel not in local:
            mismatches[rel] = {"local": None, "remote": remote[rel]}
    return mismatches


def register_edit_manifest_path(final: Path) -> Path:
    """The r2 edit manifest that sits beside an edited final."""
    return final.with_name(final.stem + ".register-edit.json")


def check_register_edit_manifest(manifest: dict, records: list[dict], limit: int = 5) -> list[str]:
    """Cross-check an r2 final against its register-edit manifest. Returns
    the problems found (empty = consistent): every record must have a
    manifest entry whose v1 sha matches its lineage; every edited record's
    lineage edits must equal the manifest's (marker + new_span_sha); and each
    edit must still hold on the released text under the manifest's own
    marker regex (add -> present, remove -> absent)."""
    problems: list[str] = []
    entries = manifest.get("records") or {}
    markers = manifest.get("markers") or {}
    probes: dict[str, tuple] = {}
    for name, m in markers.items():
        rx = re.compile(m["pattern"], re.IGNORECASE)
        probes[name] = (rx.match if m.get("anchored") else rx.search, m["field"], m["mode"])
    n_edited = 0
    for raw in records:
        if len(problems) >= limit:
            problems.append("… (more problems truncated)")
            break
        lin = raw.get("lineage") or {}
        cid = lin.get("custom_id", "?")
        entry = entries.get(cid)
        if entry is None:
            problems.append(f"record {cid} has no register-edit manifest entry")
            continue
        if entry.get("v1_sha256") != lin.get("v1_sha256"):
            problems.append(f"record {cid}: lineage v1_sha256 != manifest")
            continue
        edits = lin.get("register_edits") or []
        want = entry.get("edits") or []
        if [(e["marker"], e["new_span_sha"]) for e in edits] != [
            (e["marker"], e["new_span_sha"]) for e in want
        ]:
            problems.append(f"record {cid}: lineage register_edits != manifest edits")
            continue
        if edits:
            n_edited += 1
        for e in edits:
            probe = probes.get(e["marker"])
            if probe is None:
                problems.append(f"record {cid}: edit marker {e['marker']!r} unknown to the manifest")
                break
            fn, field, mode = probe
            hit = bool(fn(str(raw.get(field) or "")))
            if mode == "add" and not hit:
                problems.append(f"record {cid}: add-edit {e['marker']} no longer matches {field}")
                break
            if mode == "remove" and hit:
                problems.append(f"record {cid}: remove-edit {e['marker']} still matches {field}")
                break
    declared = manifest.get("n_edited_in_final")
    if declared is not None and not problems and declared != n_edited:
        problems.append(
            f"manifest declares {declared} edited records in the final, found {n_edited}"
        )
    return problems


def collect_files(
    cfg: RunConfig,
    data_dir: Path,
    run_dir: Path,
    suffixes: tuple[str, ...] = ("",),
    extra_run_dirs: tuple[Path, ...] = (),
) -> list[Path]:
    files: list[Path] = []
    for corpus in ("advice", "stories"):
        for suffix in suffixes:
            p = data_dir / cfg.final_rel(corpus, suffix)
            if p.exists():
                files.append(p)
                fr = filter_report_path(p)
                if fr.exists():
                    files.append(fr)
                # the release-time placebo-import scan (report-only) travels
                # with the final it describes, when it has been run
                sc = p.with_name(p.stem + ".import-scan.json")
                if sc.exists():
                    files.append(sc)
                # the r2 register-edit manifest (per-record v1 lineage + the
                # verified span edits) travels with the edited final
                rm = register_edit_manifest_path(p)
                if rm.exists():
                    files.append(rm)
    for rd in (run_dir, *extra_run_dirs):
        for name in ("manifest.json", "UNION.json"):
            p = rd / name
            if p.exists():
                files.append(p)
    for suffix in suffixes:
        report_dir = run_dir / f"report{suffix}"
        if report_dir.is_dir():
            files.extend(sorted(x for x in report_dir.rglob("*") if x.is_file()))
        # the edit pass's run-level manifest (batch ids, stage stamps, the
        # before/after register table vs production, verification stats)
        if suffix:
            rm = run_dir / f"register-edit{suffix}.json"
            if rm.exists():
                files.append(rm)
    return files


def mirror(
    cfg: RunConfig,
    data_dir: Path,
    run_dir: Path,
    host: str | None = None,
    dest_root: str | None = None,
    dry_run: bool = False,
    suffix: str = "",
    extra_run_dirs: tuple[Path, ...] = (),
) -> dict:
    """suffix="-v2" mirrors the v2 release: gate on the v2 finals, transfer
    the v1 AND v2 finals (the remote set comparison is exact, and v1 is
    already there — re-sending identical bytes keeps the union verifiable)
    plus any extra (regen) run manifests."""
    host, dest_root = resolve_dest(host, dest_root)  # an unconfigured destination refuses first
    release = verify_release(cfg, data_dir, suffix)  # raises before any transfer
    suffixes = ("",) if suffix == "" else ("", suffix)
    files = collect_files(cfg, data_dir, run_dir, suffixes, extra_run_dirs)
    return _transfer_and_verify(
        files, data_dir, host, dest_root, cfg.run_id, release,
        run_dir / f"mirror-verification{suffix}.json", dry_run,
    )


def mirror_onpolicy(
    cfg: RunConfig,
    root: Path,
    host: str | None = None,
    dest_root: str | None = None,
    dry_run: bool = False,
) -> dict:
    """Mirror one on-policy run: final + filter report + run manifest/UNION +
    the sha-stamped prompt-set artifact + the sampler's responses/manifest —
    the full provenance chain for the release."""
    data_dir = cfg.resolve_data_dir(root)
    run_dir = cfg.run_dir(root)
    release = verify_onpolicy_release(cfg, data_dir, root)  # raises pre-transfer

    from .onpolicy import manifest_path as ps_manifest_path

    host, dest_root = resolve_dest(host, dest_root)  # an unconfigured destination refuses first
    op = cfg.onpolicy
    if op is None:
        raise ReleaseGateError("config has no `onpolicy` section")
    final = cfg.final_path(root, "advice")
    scored = data_dir / "advice" / f"scored-{cfg.run_id}.jsonl"
    prompt_set = Path(op.prompt_set)
    prompt_set = prompt_set if prompt_set.is_absolute() else root / prompt_set
    samples = Path(op.samples)
    samples = samples if samples.is_absolute() else root / samples
    files = [final, filter_report_path(final), prompt_set, ps_manifest_path(prompt_set)]
    # the re-cuttable scored-all artifact + its report ship alongside the final
    # so downstream steering-vector fitting can re-cut the floor with no rerun
    files.extend(p for p in (scored, filter_report_path(scored)) if p.exists())
    sm = samples.parent / "SAMPLES_MANIFEST.json"
    files.extend(p for p in (samples, sm) if p.exists())
    for name in ("manifest.json", "UNION.json"):
        p = run_dir / name
        if p.exists():
            files.append(p)
    missing = [str(f) for f in files if not f.exists()]
    if missing:
        raise ReleaseGateError(f"on-policy mirror: files missing: {missing}")
    return _transfer_and_verify(
        files, data_dir, host, dest_root, cfg.run_id, release,
        run_dir / "mirror-verification.json", dry_run,
    )


def _transfer_and_verify(
    files: list[Path],
    data_dir: Path,
    host: str,
    dest_root: str,
    run_id: str,
    release: dict,
    verification_path: Path,
    dry_run: bool,
) -> dict:
    # Expand $USER locally: rsync's receiver takes the path literally
    # (protected args), so an unexpanded $USER mkdirs a literal "$USER" dir.
    dest_root = dest_root.replace("$USER", getpass.getuser())
    dest = f"{dest_root}/{run_id}"
    local = {}
    for f in files:
        rel = f.relative_to(data_dir)
        local[str(rel)] = {"bytes": f.stat().st_size, "sha256": sha256_file(f)}

    rsync = [
        "rsync", "-az", "--relative",
        *[str(f.relative_to(data_dir)) for f in files],
        f"{host}:{dest}/",
    ]
    if dry_run:
        return {"would_run": rsync, "files": local, "release": release}

    subprocess.run(["ssh", host, f"mkdir -p {dest}"], check=True)
    subprocess.run(rsync, cwd=data_dir, check=True)

    # verify: the EXACT remote file set (count + names + bytes + sha256)
    remote_cmd = (
        f"cd {dest} && find . -type f | sort | while read f; do "
        f"printf '%s %s %s\\n' \"$f\" $(stat -c %s \"$f\") $(sha256sum \"$f\" | cut -d' ' -f1); done"
    )
    out = subprocess.run(
        ["ssh", host, remote_cmd], check=True, capture_output=True, text=True
    ).stdout
    remote = {}
    for line in out.strip().splitlines():
        rel, size, sha = line.rsplit(" ", 2)
        remote[rel.removeprefix("./")] = {"bytes": int(size), "sha256": sha}

    mismatches = compare_file_sets(local, remote)
    verification = {
        "host": host,
        "dest": dest,
        "n_files": len(local),
        "n_remote": len(remote),
        "n_remote_matched": len(local) - sum(1 for r in mismatches if r in local),
        "mismatches": mismatches,
        "release": release,
        "ok": not mismatches,
        "verified_at": _utcnow(),
    }
    atomic_write(verification_path, json.dumps(verification, indent=1))
    if mismatches:
        raise RuntimeError(f"mirror verification FAILED for {len(mismatches)} files")
    return verification
