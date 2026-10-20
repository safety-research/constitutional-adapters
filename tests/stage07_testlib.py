"""Shared doubles for stage 07's MIRROR tests (tests/test_stage07_mirror.py, and tests/test_stage07_analysis_filing.py's file phase, which
needs a verified mirror of the receipt it files since 2026-10-12): a Modal-CLI double with the CLI's DIRECTORY semantics, the store files the
receipt never pulls (manifests, RESULT, rows, logs) laid down on the in-memory volume from a receipt alone, a local stand-in for the cluster's
rsync + ssh, and `ensure_mirror`, which publishes and "pushes" a mirror for whatever receipt a test world holds. Synthetic bytes only; no
Modal call, no host, nothing written outside tmp. Nothing here names a real host, user or storage root."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path, PurePosixPath

sys.path.insert(0, str(Path(__file__).resolve().parent))
from front_door_testlib import REPO  # noqa: E402
from test_stage_demo_bank_modal import FakeModal  # noqa: E402


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


MI = _load("stage07_mirror_under_test", REPO / "reproduce" / "tools" / "stage07_mirror.py")
RT = MI.RT
VOL_MOUNT = "/vol/"


def vrel(container_path: str) -> str:
    assert container_path.startswith(VOL_MOUNT), container_path
    return container_path[len(VOL_MOUNT):]


class FakeModalTree(FakeModal):
    """test_stage_demo_bank_modal.FakeModal plus the CLI's directory behaviour (modal 1.5.4, read off cli/volume.py and cli/_download.py and
    verified on the smoke stores 2026-10-12): `volume ls --json` renders `size` with humanize_filesize (`157 B`, `1.5 MiB`); `volume get` of
    a DIRECTORY needs an EXISTING local destination and lays the folder under it as <destination>/<folder name>/… (a missing destination is
    taken as a file path and fails on the first directory entry), `--force` to overwrite. `share=` another FakeModal shares ITS volume dict,
    so what a receipt's double holds is what the mirror's double serves. The unconverted-container-path rejection is inherited."""

    def __init__(self, remote=None, *, share=None):
        super().__init__(remote)
        if share is not None:
            self.remote = share.remote

    def __call__(self, argv, **kw):
        argv = list(argv)
        sub = argv[1:]
        for a in sub:
            assert not (a == "/vol" or a.startswith("/vol/")), f"an unconverted container path reached the modal CLI: {a!r} in {argv}"
        if sub[:2] == ["volume", "ls"] and "--json" in sub:
            self.calls.append(argv)
            rel = sub[-1].strip("/")
            hits = {k: v for k, v in self.remote.items() if k == rel or k.startswith(rel + "/")}
            if not hits:
                return subprocess.CompletedProcess(argv, 1, "", "No such file or directory")
            entries, seen = [], set()
            for k, v in sorted(hits.items()):
                rest = k[len(rel) + 1:] if k != rel else ""
                if "/" in rest:
                    top = rest.split("/", 1)[0]
                    if top not in seen:
                        seen.add(top)
                        entries.append({"filename": f"{rel}/{top}", "type": "dir", "size": "0 B", "created_modified": "x"})
                else:
                    entries.append({"filename": k, "type": "file", "size": MI.humanize_filesize(len(v)), "created_modified": "x"})
            return subprocess.CompletedProcess(argv, 0, json.dumps(entries), "")
        if sub[:2] == ["volume", "get"]:
            remote_abs, dest = sub[-2], Path(sub[-1])
            rel = remote_abs.strip("/")
            if rel in self.remote:
                return super().__call__(argv, **kw)                       # a single file: the base double's behaviour
            self.calls.append(argv)
            children = {k: v for k, v in self.remote.items() if k.startswith(rel + "/")}
            if not children:
                return subprocess.CompletedProcess(argv, 1, "", "No such file or directory")
            if not dest.is_dir():
                return subprocess.CompletedProcess(argv, 1, "", f"[Errno 21] Is a directory: '{dest}'")
            base = dest / PurePosixPath(rel).name
            if base.exists() and "--force" not in sub:
                return subprocess.CompletedProcess(argv, 1, "", f"Output path '{base}' already exists. Use --force to overwrite the output directory")
            for k, v in children.items():
                p = base / k[len(rel) + 1:]
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(v)
            return subprocess.CompletedProcess(argv, 0, "", "")
        return super().__call__(argv, **kw)

    def gets(self):
        return [c for c in self.calls if c[1:3] == ["volume", "get"]]


def manifest_doc(cell: str, tree_sha: str) -> dict:
    """The launcher's cell manifest, the keys the mirror reads: cell_id and producer.{driver, tree_sha} (contracts.PRODUCER_BLOCK_FIELDS)."""
    return {"cell_id": cell, "harness_cell_id": cell, "producer": {"driver": "served_capability_run", "tree_sha": tree_sha}, "contract_version": 3,
            "intervention_hash": "0" * 16, "harness_config_hash": "1" * 16, "model_id": "Qwen/Qwen3.5-4B", "model_revision": "8" * 40}


def result_text(tree_sha: str, *, complete: bool = True, sha: str | None = None) -> bytes:
    """A store's RESULT: an earlier line and the last, which reads result=COMPLETE … sha=<tree> (the launcher's grammar) unless told otherwise."""
    last_sha = tree_sha if sha is None else sha
    last = (f"tinycap-modal name=tinycap-x rc=0 result={'COMPLETE' if complete else 'GENERATED'} status=OK sha={last_sha} at=2026-01-01T00:10:00Z")
    return (f"tinycap-modal name=tinycap-x rc=1 result=FAILED status=ERR sha={tree_sha} at=2026-01-01T00:00:00Z\n{last}\n").encode()


def lay_down_store_extras(remote: dict, receipt: dict, *, rows_items: int = 60) -> None:
    """Everything a FINISHED store holds that the receipt never pulls, laid down from the receipt alone (name-agnostic: the store roots are
    `recs[<cell>].out`, the keys `recs[<cell>].done_in`): cells/<key>/{manifest.json, rows.jsonl (> 1 KiB, so a KiB-rendered size is checked),
    gen_stats.json, evidence.json, outcomes/<marker>, private/inspect_logs/…}, RESULT, usage_ledger.jsonl, served_geometry.json, evidence.json,
    protocol_check.json, resolved_sampler.json, ledger.log and watch/sandboxes_<fcid>.jsonl for every attempt pointer present. `setdefault`
    throughout: a file a test laid down itself stands."""
    tree = receipt["tree_sha"]
    for cell, rec in receipt["recs"].items():
        root = vrel(rec["out"])
        for key in sorted(set(rec["done_in"].values())):
            remote.setdefault(f"{root}/cells/{key}/manifest.json", (json.dumps(manifest_doc(cell, tree), indent=1, sort_keys=True) + "\n").encode())
            remote.setdefault(f"{root}/cells/{key}/rows.jsonl", b"".join(json.dumps({"item": i, "cell": cell, "text": "x" * 40}).encode() + b"\n" for i in range(rows_items)))
            remote.setdefault(f"{root}/cells/{key}/gen_stats.json", b'{"n_rows": 200}\n')
            remote.setdefault(f"{root}/cells/{key}/evidence.json", b'{"ok": true}\n')
            for marker in rec.get("done_markers") or ["inspect__task_a.json"]:
                remote.setdefault(f"{root}/cells/{key}/outcomes/{marker}", b'{"outcomes": []}\n')
            remote.setdefault(f"{root}/cells/{key}/private/inspect_logs/run.eval", bytes(range(256)) * 5)
        remote.setdefault(f"{root}/RESULT", result_text(tree))
        remote.setdefault(f"{root}/usage_ledger.jsonl", b'{"usage": 1}\n' * 20)
        remote.setdefault(f"{root}/served_geometry.json", b'{"geometry": {}}\n')
        remote.setdefault(f"{root}/evidence.json", b'{"ok": true}\n')
        remote.setdefault(f"{root}/protocol_check.json", b'{"ok": true}\n')
        remote.setdefault(f"{root}/resolved_sampler.json", b'{"sampler": {}}\n')
        remote.setdefault(f"{root}/ledger.log", b"ledger\n" * 50)
        remote.setdefault(f"{root}/driver_stdout.log", b"stdout\n")
        for k in [k for k in list(remote) if k.startswith(f"{root}/watch/attempt_") and k.endswith(".json")]:
            fcid = PurePosixPath(k).name[len("attempt_"):-len(".json")]
            remote.setdefault(f"{root}/watch/sandboxes_{fcid}.jsonl", b'{"event": "start"}\n{"event": "done"}\n')


def local_cluster_runner(argv, **kw):
    """A stand-in for the cluster's `rsync` + `ssh` under the `local:<dir>` seam that needs no rsync on the machine: `rsync -a --partial SRC/ DEST/`
    copies the tree (no deletion), `rsync -a --partial FILE DEST/name` copies the one file (the staged cluster record), `bash -c <cmd>` runs the
    check here."""
    argv = list(argv)
    if argv[0] == "rsync":
        src, dest = Path(argv[-2].rstrip("/")), Path(argv[-1].rstrip("/"))
        if src.is_file():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
        else:
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copytree(src, dest, dirs_exist_ok=True)
        return subprocess.CompletedProcess(argv, 0, "", "")
    return subprocess.run(argv, capture_output=True, text=True, **kw)


def _is_record_push(argv) -> bool:
    """The single-file rsync of the STAGED cluster record (stage07_mirror.Cluster.rsync_file_argv): its source is the file <sha12>.cluster.json."""
    return argv[0] == "rsync" and argv[-2].endswith(f".{MI.CLUSTER_NAME}")


class RecordingCluster:
    """A recording `rsync` / `ssh` runner: every argv kept; `ssh … 'mkdir -p …'` succeeds; the check prints the digests the caller needs (read
    from the LOCAL mirror, so a stubbed remote agrees with it unless `check_rc`/`digest_of` say otherwise); rsync succeeds. The cluster record's
    push is the single-file rsync of the STAGED record (outside the mirror, stage07_mirror.record_staging_path) and its read-back hashes that
    staged file; `record_rsync_rc` / `record_digest_of` fail THAT transfer alone (the data push and its check having passed)."""

    def __init__(self, mirror_dir_of, *, check_rc: int = 0, rsync_rc: int = 0, digest_of=None, record_rsync_rc: int = 0, record_digest_of=None,
                 check_stderr: str = "sha256sum: WARNING: 1 computed checksum did NOT match", extra_paths=(), extra_entries=(), no_inventory: bool = False,
                 guard_rc: int = 0, guard_out: str = ""):
        self.calls: list = []
        self.mirror_dir_of, self.check_rc, self.rsync_rc, self.digest_of = mirror_dir_of, check_rc, rsync_rc, digest_of
        self.record_rsync_rc, self.record_digest_of = record_rsync_rc, record_digest_of
        self.check_stderr = check_stderr            # what a failing `sha256sum -c` (or ssh itself: rc 255 + its connect error) prints
        self.extra_paths = list(extra_paths)        # regular files the stubbed remote directory holds BEYOND the local mirror's (round 4: unlisted -> refused)
        self.extra_entries = list(extra_entries)    # (find type letter, path) entries beyond the mirror's — a symlink `("l", "x")`, a directory, a fifo (round 5)
        self.no_inventory = no_inventory            # a remote whose check stops before the inventory mark (the command did not run whole)
        self.guard_rc, self.guard_out = guard_rc, guard_out   # the destination guard failing there (round 6): its exit code (61 symlink / 62 resolves) and the line it prints

    def __call__(self, argv, **kw):
        argv = list(argv)
        self.calls.append(argv)
        if argv[0] == "rsync":
            rc = self.record_rsync_rc if _is_record_push(argv) else self.rsync_rc
            return subprocess.CompletedProcess(argv, rc, "", "" if rc == 0 else "rsync: connection unexpectedly closed")
        assert argv[0] in ("ssh", "bash"), argv       # `bash -c` is the `local:<dir>` seam's shape of the same remote command
        remote = argv[-1]
        guarded = remote.startswith("r=") and ' ! [ -L "$p" ] || ' in remote     # the destination guard leads every mkdir and every check (round 6)
        if guarded and self.guard_rc:
            return subprocess.CompletedProcess(argv, self.guard_rc, self.guard_out, "")
        if " && mkdir -p " in remote or remote.startswith("mkdir -p "):
            assert guarded, f"an unguarded mkdir reached the cluster: {remote!r}"
            return subprocess.CompletedProcess(argv, 0, "", "")
        mirror = Path(self.mirror_dir_of())
        if "sha256sum -c" in remote:
            assert guarded, f"an unguarded check reached the cluster: {remote!r}"
            if self.check_rc:
                return subprocess.CompletedProcess(argv, self.check_rc, "", self.check_stderr)
            # the records listed after the check (two on the push, three on a copy's re-check), then `echo inventory: && find . -mindepth 1 -printf
            # "%y %p\0"` (rounds 4-5): the stubbed remote holds the local mirror's files and directories, as `<letter> ./<path>` NUL-terminated records
            names = remote.rsplit("sha256sum ", 1)[1].split(" && ")[0].split()
            out = "".join(f"{(self.digest_of or _sha)(mirror / n)}  {n}\n" for n in names)
            if f"echo {MI.INVENTORY_MARK}" in remote and not self.no_inventory:
                assert remote.endswith(f"echo {MI.INVENTORY_MARK} && {MI.INVENTORY_CMD}"), remote
                entries = [("d" if p.is_dir() else "f", p.relative_to(mirror).as_posix()) for p in sorted(mirror.rglob("*")) if p.is_dir() or p.is_file()]
                entries += [("f", rel) for rel in self.extra_paths] + [tuple(e) for e in self.extra_entries]
                out += f"{MI.INVENTORY_MARK}\n" + "".join(f"{t} ./{rel}\0" for t, rel in entries)
            return subprocess.CompletedProcess(argv, 0, out, "")
        if remote.startswith("sha256sum "):
            name = PurePosixPath(remote.split()[-1].strip("'")).name
            if name == MI.CLUSTER_NAME:
                staged = MI.record_staging_path(mirror)
                src = staged if staged.is_file() else mirror / name
                return subprocess.CompletedProcess(argv, 0, f"{(self.record_digest_of or _sha)(src)}  {name}\n", "")
            return subprocess.CompletedProcess(argv, 0, f"{(self.digest_of or _sha)(mirror / name)}  {name}\n", "")
        raise AssertionError(f"unexpected ssh command {remote!r}")

    def rsyncs(self):
        return [c for c in self.calls if c[0] == "rsync"]

    def record_pushes(self):
        return [c for c in self.calls if _is_record_push(c)]

    def sshs(self):
        return [c for c in self.calls if c[0] in ("ssh", "bash")]

    def copy_checks(self):
        """The cluster copy's re-checks (stage07_mirror.copy_check_command: the three records listed after `sha256sum -c`, then the inventory)."""
        return [c for c in self.sshs() if f"sha256sum {MI.SUMS_NAME} {MI.MIRROR_NAME} {MI.CLUSTER_NAME} && " in c[-1]]


def _sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def mirror(state: Path, remote: dict, *, dest_root: Path, root: Path | str | None = None, label: str = "cluster_a", host: str | None = None, cluster_runner=None,
           dry: bool = False, volume=None, environment=None, receipt_path=None, free_margin: float = 1.25, free_headroom: int = 0, log=None):
    """The mirror tool as 07 calls it, over a FakeModalTree serving `remote` and the `local:<dest_root>` cluster seam (or a given runner). `root` is
    the cluster root the destination is guarded under (07's expanded reproduce root; round 6) — the destination itself by default, so the guard
    walks <receipt sha12> alone; a test of the real layout passes <reproduce root> with dest_root = <root>/<model>/07_mirror."""
    lines: list = []
    vol = FakeModalTree(remote)
    vol.remote = remote
    out = MI.run(state=Path(state), receipt_path=receipt_path, volume=volume, environment=environment, modal="/nonexistent/modal", cluster_label=label,
                 ssh_host=host or f"local:{dest_root}", cluster_dest=str(dest_root), cluster_root=str(root if root is not None else dest_root), dry_run=dry,
                 free_margin=free_margin, free_headroom=free_headroom, modal_runner=vol, cluster_runner=cluster_runner or local_cluster_runner, log=log or lines.append)
    out["log"], out["vol"] = lines, vol
    return out


def ensure_mirror(w, *, dest_root: Path | None = None):
    """A verified mirror (published + cluster record) of the receipt the world `w` (test_stage07_analysis_filing.make_world) currently holds
    under <state>/07_store/, built over the world's own volume with the store extras laid down; None when the world has no receipt."""
    state = Path(w["state"])
    rpath = state / RT.STORE_DIR / RT.RECEIPT_NAME
    if not rpath.is_file():
        return None
    doc = json.loads(rpath.read_text(encoding="utf-8"))
    sha12 = _sha(rpath)[:12]
    if (state / MI.MIRROR_DIR / sha12 / MI.CLUSTER_NAME).is_file():
        return MI.verified_mirror(state, _sha(rpath))
    lay_down_store_extras(w["vol"].remote, doc)
    return mirror(state, w["vol"].remote, dest_root=dest_root or state.parent / "cluster-mirror")
