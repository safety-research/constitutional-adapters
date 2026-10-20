"""The served run's IMMUTABLE identity sidecar — what a resumed run must match
before it may honor any prior generations — and the geometry stamp every served
response row carries so a mixed store is detectable.

WHY (adversarial review R1, 2026-09-04). The served families waves resume by
goal / sample identity: an out dir that already holds responses skips them.
Nothing in that key said WHICH geometry produced them — and bf16 greedy decode
is not batch-composition invariant, so a store generated at the old
Llama-sized S and resumed at a model-budget S would have been relabeled with
the new geometry (the top-level record is rewritten each run) while its rows
came from the old one: an ASR / capability number silently made of two
serving configurations. The per-model budgets changed every launcher's S, so
every existing out dir became exactly that trap.

THE SIDECAR. :func:`write_or_verify` writes ``run_identity.json`` next to the
generations on the FIRST run of an out dir and, on every later run, compares
the current :class:`RunIdentity` field by field: any difference REFUSES the
resume (:class:`RunIdentityMismatch`, naming the fields) — the operator runs
into a fresh out dir instead. An out dir that holds prior generations but no
sidecar (a pre-sidecar run) is refused the same way: its identity is unknown,
so nothing may be resumed onto it. The identity is the launch: model +
revision, tree sha, model key, L x S, client concurrency, budget + source,
serving-stack versions, sampling settings, input fingerprints.

THE STAMP. The identity's ``geometry_id`` (a hash of the serving-relevant
fields) is exported as :data:`GEOMETRY_ID_ENV` before generation; the served
row writers (``bon_served``, ``static_common``) stamp it on every row as
``geometry_id`` and refuse to append onto a store holding rows of another id,
so a mixed store is detectable row by row, not just by the sidecar.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

RUN_IDENTITY_FILE = "run_identity.json"
#: the env vars the served row writers read the stamps from: the geometry id
#: (the serving-relevant subset) and the RUN id (the whole identity — fitted
#: intervention hashes and frozen-corpus hashes included, review S1), so a
#: store's rows can be audited against the sidecar AND against each other
GEOMETRY_ID_ENV = "SISTEER_SERVED_GEOMETRY_ID"
RUN_ID_ENV = "SISTEER_SERVED_RUN_ID"
#: files whose presence means an out dir already holds generations
GENERATION_FILES = ("responses.jsonl", "bon_responses.jsonl")


class RunIdentityMismatch(RuntimeError):
    """A resume into an out dir whose recorded identity differs from the
    current run's (or whose generations predate the sidecar)."""


class FrozenCorpusMismatch(RunIdentityMismatch):
    """A frozen corpus whose rows hash to something other than the
    ``corpus_content_hash`` its manifest states — a rebuild interrupted between
    the JSONL and the manifest, or a corpus copied without its manifest: the
    rows are not the corpus the manifest describes, so no identity can be
    built over them (review S1, second pass)."""


def _sha16(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode("utf-8")
                          ).hexdigest()[:16]


def file_sha16(path: Optional[str | Path]) -> Optional[str]:
    """sha256[:16] of a file's bytes, ``None`` when absent."""
    if not path:
        return None
    p = Path(path)
    if not p.is_file():
        return None
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def tree_sha(repo_root: Optional[str | Path]) -> Optional[str]:
    """``git rev-parse HEAD`` of the tree the run executes from (``None`` when
    it is not a git checkout — recorded as such, never guessed)."""
    if not repo_root:
        return None
    try:
        out = subprocess.run(["git", "-C", str(repo_root), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=20)
    except Exception:  # noqa: BLE001
        return None
    sha = out.stdout.strip()
    return sha if out.returncode == 0 and len(sha) >= 7 else None


def serving_versions(packages: Iterable[str] = ("vllm", "transformers", "torch")
                     ) -> Dict[str, Optional[str]]:
    """The installed versions of the serving stack (``None`` = not installed)."""
    import importlib.metadata as m
    out: Dict[str, Optional[str]] = {}
    for name in packages:
        try:
            out[name] = m.version(name)
        except Exception:  # noqa: BLE001
            out[name] = None
    return out


@dataclass(frozen=True)
class RunIdentity:
    """What makes two served runs the SAME measurement. ``geometry_fields``
    are the serving-relevant subset (the stamp); the whole record is the
    resume key."""

    model: str
    revision: Optional[str]
    tree_sha: Optional[str]
    model_key: Optional[str]
    max_model_len: int
    max_num_seqs: int
    client_concurrency: int
    budget: int
    budget_source: str
    serving_versions: Dict[str, Optional[str]] = field(default_factory=dict)
    sampling: Dict[str, Any] = field(default_factory=dict)
    input_fingerprints: Dict[str, Any] = field(default_factory=dict)
    #: the SERVING ENGINE:
    #: ``"lora_native"`` on a run served by the plugin-free multi-LoRA
    #: server, ``None`` on every lens-served run — and every run recorded
    #: before the field existed. OPTIONAL and EXCLUDED from ``run_id`` (and
    #: from the sidecar) when None (:data:`_OPTIONAL_IDENTITY_FIELDS`), so
    #: no existing store is re-keyed; a lora-native identity carries it in
    #: both, and a resume across engines therefore refuses. NOT in
    #: ``geometry_fields`` (the stamp is unchanged; spec: "``geometry_fields()``
    #: is unchanged so existing stores are not re-keyed").
    engine: Optional[str] = None

    def geometry_fields(self) -> Dict[str, Any]:
        return {"model": self.model, "revision": self.revision,
                "max_model_len": int(self.max_model_len), "max_num_seqs": int(self.max_num_seqs),
                "client_concurrency": int(self.client_concurrency), "budget": int(self.budget),
                "serving_versions": dict(self.serving_versions),
                "sampling": dict(self.sampling)}

    def identity_fields(self) -> Dict[str, Any]:
        """Every field that KEYS the identity: ``asdict(self)`` minus each
        :data:`_OPTIONAL_IDENTITY_FIELDS` member that is None — so an identity
        built before such a field existed and one built after it, with the
        field unset, are the SAME dict, the same ``run_id``, and the same
        sidecar bytes."""
        d = asdict(self)
        for name in _OPTIONAL_IDENTITY_FIELDS:
            if d.get(name) is None:
                d.pop(name, None)
        return d

    @property
    def geometry_id(self) -> str:
        return _sha16(self.geometry_fields())

    @property
    def run_id(self) -> str:
        return _sha16(self.identity_fields())

    def as_dict(self) -> Dict[str, Any]:
        d = self.identity_fields()
        d["geometry_id"] = self.geometry_id
        d["run_id"] = self.run_id
        return d


#: identity fields that are ABSENT from ``run_id`` / the sidecar when None
#: (:meth:`RunIdentity.identity_fields`): fields added after stores existed,
#: whose unset value must not re-key a single one of them
_OPTIONAL_IDENTITY_FIELDS: Tuple[str, ...] = ("engine",)


def has_prior_generations(out_dir: str | Path, max_depth: int = 4) -> List[str]:
    """Generation files already under ``out_dir`` (relative paths), searched a
    few levels deep — the families waves nest ``<family>/responses.jsonl`` and
    ``sweep_out/.../bon_responses.jsonl``."""
    root = Path(out_dir)
    if not root.is_dir():
        return []
    found: List[str] = []
    for p in root.rglob("*"):
        try:
            rel = p.relative_to(root)
        except ValueError:
            continue
        if len(rel.parts) > max_depth:
            continue
        if p.is_file() and p.name in GENERATION_FILES and p.stat().st_size > 0:
            found.append(str(rel))
    return sorted(found)


def identity_diff(recorded: Mapping[str, Any], current: Mapping[str, Any]
                  ) -> List[Tuple[str, Any, Any]]:
    """The (field, recorded, current) triples that differ, over the identity's
    fields (``written_at`` / derived ids excluded — the fields decide)."""
    skip = {"written_at", "geometry_id", "run_id"}
    keys = sorted((set(recorded) | set(current)) - skip)
    return [(k, recorded.get(k), current.get(k)) for k in keys
            if recorded.get(k) != current.get(k)]


def write_or_verify(out_dir: str | Path, identity: RunIdentity) -> Path:
    """Write the sidecar on a fresh out dir; on a reused one verify it field by
    field and REFUSE any mismatch (or a pre-sidecar dir that already holds
    generations). Returns the sidecar path. Call BEFORE anything resumes."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    p = out / RUN_IDENTITY_FILE
    current = identity.as_dict()
    if not p.is_file():
        prior = has_prior_generations(out)
        if prior:
            raise RunIdentityMismatch(
                f"{out} already holds generations ({prior[:3]}{'...' if len(prior) > 3 else ''}) "
                f"but no {RUN_IDENTITY_FILE}: a pre-sidecar run of unknown geometry — refusing "
                "to resume onto it (its rows would be relabeled with this run's geometry); "
                "use a fresh out dir")
        # ATOMIC CREATION (review, fourth pass): a per-process unique tmp,
        # fsynced, then os.link onto the final name — link FAILS if another
        # process created the sidecar first (a shared .tmp + os.replace let two
        # racers clobber each other's bytes and each proceed on its in-memory
        # identity). A lost race re-reads the WINNER and verifies against it
        # below; the in-memory identity is never trusted after a lost race.
        payload = dict(current, written_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        import secrets
        tmp = p.with_name(f".{p.name}.tmp-{os.getpid()}-{secrets.token_hex(4)}")
        with open(tmp, "w") as fh:
            fh.write(json.dumps(payload, indent=2, sort_keys=True, default=str))
            fh.flush()
            os.fsync(fh.fileno())
        try:
            os.link(tmp, p)
            won = True
        except FileExistsError:
            won = False
        finally:
            try:
                tmp.unlink()
            except FileNotFoundError:
                pass
        if won:
            # re-read what is on disk under the final name and hold it to what
            # we meant to write — the file is the authority from here on
            on_disk = json.loads(p.read_text())
            if on_disk.get("run_id") != identity.run_id:
                raise RunIdentityMismatch(
                    f"{p} holds run_id {on_disk.get('run_id')} right after this run created it "
                    f"(expected {identity.run_id}) — another writer replaced it; refusing")
            return p
        # lost the race: fall through and VERIFY against the winner's sidecar
    try:
        recorded = json.loads(p.read_text())
    except (json.JSONDecodeError, OSError) as e:
        raise RunIdentityMismatch(f"{p} is unreadable ({e}); refusing to resume") from e
    diff = identity_diff(recorded, current)
    if diff:
        lines = "\n".join(f"  - {k}: recorded {a!r} != current {b!r}" for k, a, b in diff)
        raise RunIdentityMismatch(
            f"{out} was generated under a DIFFERENT run identity (recorded run_id "
            f"{recorded.get('run_id')}, this run {identity.run_id}) — refusing to resume "
            f"(bf16 greedy output is not batch-composition invariant; a mixed store would "
            f"relabel old rows with the new geometry). Differing fields:\n{lines}\n"
            "Run into a fresh out dir instead.")
    return p


def stamp_env(identity: RunIdentity) -> str:
    """Export the geometry AND run stamps for this process and its children (the
    served row writers read them); returns the geometry id."""
    os.environ[GEOMETRY_ID_ENV] = identity.geometry_id
    os.environ[RUN_ID_ENV] = identity.run_id
    return identity.geometry_id


def current_geometry_id() -> Optional[str]:
    v = os.environ.get(GEOMETRY_ID_ENV, "").strip()
    return v or None


def current_run_id() -> Optional[str]:
    v = os.environ.get(RUN_ID_ENV, "").strip()
    return v or None


def geometry_stamp() -> Dict[str, str]:
    """The fields a served row writer merges into every row: ``geometry_id``
    and ``run_id`` when a run exported them, else nothing (a pre-sidecar
    caller). The run id makes the intervention / corpus identity auditable on
    the rows themselves, independent of the sidecar (review S1)."""
    out: Dict[str, str] = {}
    gid, rid = current_geometry_id(), current_run_id()
    if gid:
        out["geometry_id"] = gid
    if rid:
        out["run_id"] = rid
    return out


def geometry_stamp_problem(row: Mapping[str, Any]) -> Optional[str]:
    """Why an existing row may NOT be appended to under the current stamps: a
    different ``geometry_id`` or ``run_id``, or none at all while this run
    carries one. ``None`` = consistent (or no stamp in force)."""
    for env_val, field, what in ((current_geometry_id(), "geometry_id", "geometry"),
                                 (current_run_id(), "run_id", "run identity")):
        if env_val is None:
            continue
        got = row.get(field)
        if got == env_val:
            continue
        if got is None:
            return (f"row carries no {field} while this run is stamped {env_val} — a "
                    f"pre-sidecar row of unknown {what}")
        return f"row carries {field} {got!r} != this run's {env_val!r}"
    return None


# --------------------------------------------------------------------------- #
# Input identities the sidecar must carry (review S1)                          #
# --------------------------------------------------------------------------- #


def intervention_hashes(routing_table: str | Path | Mapping[str, Any],
                        cell_ids: Optional[Iterable[str]] = None) -> Dict[str, Optional[str]]:
    """``{cell_id: intervention_hash}`` from an exported routing table (path or
    loaded table): the fitted object's ACTUAL identity as the proxy will serve
    it. A refit / promotion in place under the same defenses dir changes this
    hash and therefore the run identity — the sidecar refuses the resume."""
    if isinstance(routing_table, (str, Path)):
        from sisteer.capeval.serving.proxy import load_routing_table
        table = load_routing_table(Path(routing_table))
    else:
        table = routing_table
    routes = table["routes"] if isinstance(table, Mapping) else getattr(table, "routes")
    wanted = set(cell_ids) if cell_ids is not None else None
    out: Dict[str, Optional[str]] = {}
    for r in routes:
        cid = r["cell_id"] if isinstance(r, Mapping) else getattr(r, "cell_id")
        if wanted is not None and cid not in wanted:
            continue
        ih = r["intervention_hash"] if isinstance(r, Mapping) else getattr(r, "intervention_hash")
        out[str(cid)] = (str(ih) if ih is not None else None)
    if wanted is not None:
        for cid in wanted:
            out.setdefault(str(cid), None)
    return dict(sorted(out.items()))


def default_corpus_dir() -> Path:
    """The held-out frozen corpora root the replay families read
    (``SISTEER_HELDOUT_CORPUS_DIR``, else the attack modules' default,
    :func:`sisteer.eval.jailbreak_artifacts.heldout_corpus_dir`)."""
    from sisteer.eval.jailbreak_artifacts import heldout_corpus_dir
    return heldout_corpus_dir()


def frozen_corpus_hashes(families: Iterable[str],
                         corpus_dir: Optional[str | Path] = None) -> Dict[str, str]:
    """``{family: corpus_content_hash}`` for every family in ``families`` that
    replays a frozen corpus at ``<corpus_dir>/<family>.jsonl`` (renellm,
    multilingual, ...). The hash is ALWAYS computed from the actual rows
    (``frozen_corpus.corpus_content_hash`` — canonical by goal / sample
    identity); when a ``.manifest.json`` sits beside the corpus its stated
    ``corpus_content_hash`` must AGREE, else :class:`FrozenCorpusMismatch`
    naming the file and both hashes — a manifest is never trusted over the
    rows (an interrupted rebuild leaves new rows under the old manifest). A
    corpus rebuilt in place changes the hash — the run identity — and the
    resume is refused. Families with no corpus file are simply absent (they
    are not frozen replays). With the default root, ``renellm`` is hashed at
    the file its replay reads (``renellm.replay_path``: the site's copy, else
    the paper's corpus as the repository ships it), so a fresh clone's
    identity names the corpus it actually replays; an explicit
    ``corpus_dir`` is read as given."""
    from sisteer.eval.attacks.frozen_corpus import corpus_content_hash

    root = Path(corpus_dir) if corpus_dir is not None else default_corpus_dir()
    out: Dict[str, str] = {}
    for fam in families:
        if fam == "renellm" and corpus_dir is None:
            from sisteer.eval.attacks import renellm as _renellm
            p = _renellm.replay_path()
        else:
            p = root / f"{fam}.jsonl"
        if not p.is_file():
            continue
        rows = [json.loads(ln) for ln in p.read_text().splitlines() if ln.strip()]
        computed = str(corpus_content_hash(rows))
        manifest = Path(str(p) + ".manifest.json")
        if manifest.is_file():
            try:
                stated = json.loads(manifest.read_text()).get("corpus_content_hash")
            except (json.JSONDecodeError, OSError) as e:
                raise FrozenCorpusMismatch(
                    f"{manifest} is unreadable ({e}); the corpus {p} cannot be identified — "
                    "rebuild the corpus (JSONL + manifest) before resuming") from e
            if stated is not None and str(stated) != computed:
                raise FrozenCorpusMismatch(
                    f"{p}: the rows hash to {computed} but the manifest {manifest} states "
                    f"corpus_content_hash {stated} — the rows are not the corpus the manifest "
                    "describes (a rebuild interrupted between JSONL and manifest, or a corpus "
                    "copied without its manifest); refusing to build a run identity over them")
        out[str(fam)] = computed
    return dict(sorted(out.items()))


# --------------------------------------------------------------------------- #
# The out-dir lock (review, fourth pass)                                       #
# --------------------------------------------------------------------------- #

RUN_LOCK_FILE = ".run.lock"
RUN_OWNER_FILE = ".run.owner.json"
#: the exit code of a refused (locked) out dir — distinct from an identity /
#: geometry refusal (1), as the sweep's ResultsLock
EXIT_LOCKED = 2


class OutDirLock:
    """EXCLUSIVE ownership of a families run's out dir for the whole run: an
    ``flock`` on ``<out>/.run.lock`` taken BEFORE the artifacts are staged and
    the sidecar written, held through generation and the final verification
    (released explicitly or at process exit), with an owner sidecar
    ``.run.owner.json`` (pid / job / host / phase / started). A live competitor
    — a second resubmission on the same fixed default out dir, a duplicate
    WAVE_OUT_DIR resume — is REFUSED with exit :data:`EXIT_LOCKED`, naming the
    holder, before it can race the sidecar or resume the same stores. A stale
    owner sidecar from a crash holds no flock (the kernel drops it with the
    process), so the next acquirer simply takes over and overwrites it."""

    def __init__(self, out_dir: str | Path, *, phase: str = "families",
                 job_id: Optional[str] = None, node: Optional[str] = None):
        import socket
        self.out_dir = Path(out_dir)
        self.phase = phase
        self.job_id = job_id if job_id is not None else os.environ.get("SLURM_JOB_ID")
        self.node = node or socket.gethostname()
        self._fh = None

    @property
    def lock_path(self) -> Path:
        return self.out_dir / RUN_LOCK_FILE

    @property
    def owner_path(self) -> Path:
        return self.out_dir / RUN_OWNER_FILE

    def holder(self) -> Dict[str, Any]:
        try:
            return json.loads(self.owner_path.read_text())
        except (OSError, json.JSONDecodeError):
            return {}

    @property
    def held(self) -> bool:
        return self._fh is not None

    def acquire(self) -> "OutDirLock":
        import fcntl
        import sys
        self.out_dir.mkdir(parents=True, exist_ok=True)
        fh = open(self.lock_path, "a+")
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (BlockingIOError, OSError):
            fh.close()
            h = self.holder()
            print(f"REFUSED: {self.out_dir} is owned by a live run — pid {h.get('pid')} job "
                  f"{h.get('job_id')} on {h.get('node')} ({h.get('phase')}, started "
                  f"{h.get('started')}); two runs on one out dir would race the run identity and "
                  f"resume the same stores. Exit {EXIT_LOCKED}; wait for it or use another out dir",
                  file=sys.stderr, flush=True)
            raise SystemExit(EXIT_LOCKED)
        self._fh = fh
        owner = {"pid": os.getpid(), "job_id": self.job_id, "node": self.node,
                 "phase": self.phase, "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        tmp = self.owner_path.with_name(f"{self.owner_path.name}.tmp-{os.getpid()}")
        tmp.write_text(json.dumps(owner, indent=2))
        os.replace(tmp, self.owner_path)
        return self

    def release(self) -> None:
        import fcntl
        if self._fh is None:
            return
        try:
            self.owner_path.unlink()
        except FileNotFoundError:
            pass
        try:
            fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
        finally:
            self._fh.close()
            self._fh = None

    def __enter__(self) -> "OutDirLock":
        return self.acquire()

    def __exit__(self, *exc) -> None:
        self.release()


# --------------------------------------------------------------------------- #
# Staged artifact install (review, third pass)                                 #
# --------------------------------------------------------------------------- #

STAGING_PREFIX = ".staging-"
REPLACED_PREFIX = ".replaced-"


def stage_dir(out_dir: str | Path) -> Path:
    """A fresh temporary directory INSIDE ``out_dir`` (same filesystem, so the
    install is a rename) for the serving artifacts a run generates before its
    identity is verified: the routing table, the lens registry, the chat
    template. Until :func:`install_staged` runs, the out dir's existing
    artifacts are untouched — a REFUSED resume must leave the completed run's
    routing table / registry / template byte-identical (it used to overwrite
    them with the rejected run's before the identity check)."""
    import secrets
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    p = out / f"{STAGING_PREFIX}{os.getpid()}-{secrets.token_hex(4)}"
    p.mkdir()
    return p


def _chmod_writable(root: Path) -> None:
    for dp, dns, fns in os.walk(root):
        os.chmod(dp, 0o700)
        for n in fns:
            try:
                os.chmod(Path(dp) / n, 0o600)
            except OSError:
                pass


def _remove(path: Path) -> None:
    import shutil
    if path.is_dir() and not path.is_symlink():
        _chmod_writable(path)        # the registry export is read-only
        shutil.rmtree(path)
    elif path.exists() or path.is_symlink():
        path.unlink()


def _same_content(a: Path, b: Path) -> bool:
    if a.is_dir() != b.is_dir():
        return False
    if not a.is_dir():
        return a.read_bytes() == b.read_bytes()
    fa = sorted(str(p.relative_to(a)) for p in a.rglob("*") if p.is_file())
    fb = sorted(str(p.relative_to(b)) for p in b.rglob("*") if p.is_file())
    if fa != fb:
        return False
    return all((a / r).read_bytes() == (b / r).read_bytes() for r in fa)


def discard_stage(stage: str | Path) -> None:
    """Drop a staging dir (a refused identity): the out dir keeps every
    existing artifact and no staging leftovers."""
    p = Path(stage)
    if p.exists():
        _remove(p)


def install_staged(stage: str | Path, out_dir: str | Path, names: Iterable[str]
                   ) -> Dict[str, str]:
    """Install the staged artifacts ``names`` into ``out_dir`` by RENAME — after
    the identity was verified. An existing artifact that is byte-identical to
    the staged one is kept (``"unchanged"``, the staged copy dropped); a
    differing one (a same-identity resume regenerating the same content with
    different nonces / mtimes) is REPLACED in two renames via a
    ``.replaced-*`` name that is removed at the end, so no reader ever sees a
    torn or partial artifact; a missing one is ``"installed"``. The staging dir
    is removed. Returns ``{name: status}``."""
    stage, out = Path(stage), Path(out_dir)
    names = list(names)
    # PRE-VALIDATE the complete list before ANY move (review, fourth pass): a
    # missing later artifact must leave the out dir untouched, not half-installed
    missing = [n for n in names if not (stage / n).exists()]
    if missing:
        raise FileNotFoundError(
            f"staged artifact(s) {missing} missing under {stage} — nothing installed "
            f"(the contract is all of {names} or none)")
    result: Dict[str, str] = {}
    for name in names:
        src, dst = stage / name, out / name
        if dst.exists():
            if _same_content(src, dst):
                _remove(src)
                result[name] = "unchanged"
                continue
            old = out / f"{REPLACED_PREFIX}{name}-{os.getpid()}"
            os.replace(dst, old)
            os.replace(src, dst)
            _remove(old)
            result[name] = "replaced"
        else:
            os.replace(src, dst)
            result[name] = "installed"
    discard_stage(stage)
    return result


def staging_leftovers(out_dir: str | Path) -> List[str]:
    """Any ``.staging-*`` / ``.replaced-*`` entries left in ``out_dir``."""
    out = Path(out_dir)
    if not out.is_dir():
        return []
    return sorted(p.name for p in out.iterdir()
                  if p.name.startswith((STAGING_PREFIX, REPLACED_PREFIX)))


__all__ = [
    "RUN_IDENTITY_FILE", "GEOMETRY_ID_ENV", "GENERATION_FILES", "RunIdentity",
    "RunIdentityMismatch", "FrozenCorpusMismatch", "file_sha16", "tree_sha", "serving_versions",
    "has_prior_generations", "identity_diff", "write_or_verify", "stamp_env",
    "current_geometry_id", "current_run_id", "geometry_stamp", "geometry_stamp_problem",
    "RUN_ID_ENV", "intervention_hashes", "default_corpus_dir", "frozen_corpus_hashes",
    "STAGING_PREFIX", "REPLACED_PREFIX", "stage_dir", "discard_stage", "install_staged",
    "staging_leftovers", "RUN_LOCK_FILE", "RUN_OWNER_FILE", "EXIT_LOCKED", "OutDirLock",
]
