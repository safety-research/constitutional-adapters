"""Stage 07's MIRROR (reproduce/tools/stage07_mirror.py; the rule of 2026-10-12: the run stores are COPIED off the Modal volume to
the grading host and the primary cluster, sha-verified, and the analysis filing records where the verified copy lives) over the receipt
fixtures of tests/test_stage07_analysis_filing.py and a Modal-CLI double with the CLI's DIRECTORY semantics (tests/stage07_testlib.py).

Pinned: the pull plan comes from the receipt alone (one `modal volume get --force` per store root, the volume-root path, into the build's
staging; the dry run reads the receipt, prints the pulls, the rsync and the ssh check in the shape of lib/common.sh's DRY-RUN lines, makes
no volume call and writes nothing); a success pulls the WHOLE store under <cell>/, hashes every file, holds every file the receipt hashed
byte for byte, every credited key's manifest to the receipt's tree and cell, RESULT's last line to result=COMPLETE and every size to the
listing's rendering, writes MIRROR.sha256 + mirror.json and publishes <state>/07_mirror/<receipt sha12>/ by rename, then pushes with rsync
(no --delete), runs `sha256sum -c MIRROR.sha256` on the cluster and records cluster.json only after it passes; a flipped byte, a manifest at
another tree or cell, a missing manifest, a RESULT not COMPLETE or at another tree, a listing that disagrees, too little disk — each refuses
by name, publishes nothing and leaves the .building directory for inspection (and a leftover build refuses the next run by name); a re-run
for the same receipt verifies the published copy AND the cluster copy over ssh (`sha256sum -c` there, the three records read back by
digest; missing, altered or unreachable refuses by name) and does nothing else, finishing only a cluster record that is missing; the
cluster push and check are driven through a stubbed ssh/rsync with the exact argv; `07 --phase file` refuses by name, with the exact mirror
command, without a verified mirror, re-verifies the cluster copy the same way, carries the pointer in extra.mirror with one, and reads a
filed row's identity (analysis_id, code_sha) and its extra.mirror's facts (cluster, host, path, digests) back against this filing's; the
tool names no control, suite or marker (a receipt with invented cell and key names mirrors as well). The cluster check
reads back the copy's regular-file inventory and refuses an unlisted file by name without deleting it (at the push and on every re-check), and a
filing that predates the mirror phase ends in a finding, not a pass, since its registry record carries no pointer to the copy. The
inventory reads EVERY entry by type (`find . -mindepth 1 -printf "%y %p\\0"`, GNU find — the local seam's `bash -c` runs it for real here, as it
runs `sha256sum -c`), the regular files held to MIRROR.sha256 plus the records, the directories to mirror.json's recorded `directories`, and a
symlink, fifo, socket or device refused wherever it is; and the filed extra.mirror's local_path is a fact the read-back holds to the current
verified mirror, so a moved state root is a finding, not a pass. The cluster destination is GUARDED under the site's reproduce root
(`--cluster-root`, recorded in cluster.json as schema stage07-mirror-cluster/2) — every component between the root and <dest>/<sha12> tested
for a symlink and the directory's realpath held to its literal path under the root's, before the mkdir, the rsync and every check's `cd`; a
symlink there is REDIRECTED, refused by name, nothing written through it, nothing deleted — and the pulled DIRECTORY set of each store is held
to the set `modal volume ls --json` listed, so an empty directory the download omits, or a directory it creates that was never listed, refuses
before anything is published. The GRADING-HOST path is held plain under the state root too (`local_guard`, the cluster guard's twin):
a symlink at 07_mirror or at <receipt sha12>, or a directory resolving elsewhere, is REDIRECTED by name on the re-run, the filing and the
publication (before the pull and before the rename), nothing followed or deleted — earlier, a mirror moved aside with a link left at its path
re-ran and filed as the copy at the recorded location. Synthetic bytes only; no Modal call, no host, nothing written outside tmp."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stage07_testlib as L  # noqa: E402
import test_stage07_analysis_filing as F  # noqa: E402
from front_door_testlib import REPO  # noqa: E402

MI = L.MI
RT = L.RT
SF = F.SF
Refusal = MI.Refusal
KEY, VIEW, SHA, CELLS, DIFF = F.KEY, F.VIEW, F.SHA, F.CELLS, F.DIFF
#: the synthetic site's expanded reproduce root on its cluster (07's --cluster-root) and the wave's 07_mirror destination under it (--cluster-dest):
#: the guard walks <KEY>, 07_mirror and <receipt sha12> between them (round 6)
ROOT = "/scratch/example/shared/exp/reproduce"
DEST = f"{ROOT}/{KEY}/07_mirror"


def _sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def world(tmp: Path, **kw):
    """A receipted world with the store extras laid down: (w, the receipt doc, its sha256). `kw` goes to make_world (another `accs` makes another
    receipt, so two worlds in one test carry two names)."""
    w = F.make_world(tmp, **kw)
    F.receipt(w)
    rpath = w["state"] / RT.STORE_DIR / RT.RECEIPT_NAME
    doc = json.loads(rpath.read_text())
    L.lay_down_store_extras(w["vol"].remote, doc)
    return w, doc, _sha(rpath)


def mirror_dir(w, rsha: str) -> Path:
    return w["state"] / MI.MIRROR_DIR / rsha[:12]


def build_dir(w, rsha: str) -> Path:
    return w["state"] / MI.MIRROR_BUILD_DIR / rsha[:12]


def store_files(remote: dict, out: str) -> dict:
    root = L.vrel(out)
    return {k[len(root) + 1:]: v for k, v in remote.items() if k.startswith(root + "/")}


def tree_bytes(root: Path) -> dict:
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


# -------------------------------------------------------------------------------------------------------------------- the plan + dry run
def test_the_dry_run_plans_every_store_from_the_receipt_and_touches_nothing(tmp_path):
    w, doc, rsha = world(tmp_path)
    state, dest = w["state"], tmp_path / "cluster"
    out = L.mirror(state, w["vol"].remote, dest_root=dest, dry=True)
    assert out["dry"] and out["receipt_sha256"] == rsha and out["vol"].calls == [], "a dry run makes NO volume call"
    assert not (state / MI.MIRROR_DIR).exists() and not (state / MI.MIRROR_BUILD_DIR).exists() and not dest.exists()
    dry = [ln for ln in out["log"] if ln.startswith("DRY-RUN: ")]
    pulls = [ln for ln in dry if " volume get --force " in ln]
    assert len(pulls) == len(doc["recs"]) == 2
    for cell, rec in doc["recs"].items():
        vpath = RT.volume_rel(rec["out"])
        line = next(ln for ln in pulls if f" {vpath} " in ln)
        assert "/vol/" not in line.split("get --force")[1] and f"twosi-fits {vpath} {build_dir(w, rsha) / MI.PULL_STAGING / cell}" in line and f"(-> {build_dir(w, rsha) / cell}/" in line
    assert any(f"publish {mirror_dir(w, rsha)}/ by rename" in ln for ln in dry)
    rsync = next(ln for ln in dry if ln.startswith("DRY-RUN: rsync "))
    assert rsync == f"DRY-RUN: rsync -a --partial {mirror_dir(w, rsha)}/ {dest}/{rsha[:12]}/" and "--delete" not in rsync
    check = next(ln for ln in dry if "sha256sum -c --quiet MIRROR.sha256" in ln)
    # round 6: the guard under the cluster root leads the check (before the cd) and the mkdir (before the rsync); no bare `mkdir -p` or `cd` reaches the cluster
    path = f"{dest}/{rsha[:12]}"
    assert check.startswith(f"DRY-RUN: ssh local:{dest} '{MI.check_command(path, (MI.SUMS_NAME, MI.MIRROR_NAME), root=str(dest))}'"), check
    assert f"'r={dest}; p=\"$r\"; s=\"\"; for c in {rsha[:12]}; do " in check and f" && cd {path} && sha256sum -c --quiet MIRROR.sha256 && sha256sum MIRROR.sha256 mirror.json && echo inventory: && " in check
    assert "the guard again before the cd" in check and "the inventory of every entry by type read back" in check and "the directories to mirror.json's, no other kind of entry" in check
    prep = next(ln for ln in dry if "mkdir -p" in ln)
    assert prep.startswith(f"DRY-RUN: ssh local:{dest} '{MI.prepare_command(str(dest), path)}'") and f"the guard under the cluster root {dest}:" in prep and "refuses by name before any rsync" in prep, prep
    assert not any(ln.startswith(f"DRY-RUN: ssh local:{dest} 'mkdir -p") or ln.startswith(f"DRY-RUN: ssh local:{dest} 'cd ") for ln in dry)
    assert dry.index(prep) < dry.index(next(ln for ln in dry if ln.startswith("DRY-RUN: rsync "))) < dry.index(check)
    # a real cluster host: the rsync goes over ssh with BatchMode, the check is `ssh <host> '…'` — the shape of lib/common.sh repro_ssh's DRY-RUN line;
    # the guard walks <KEY>, 07_mirror and <sha12> between the site's reproduce root and the copy
    out2 = L.mirror(state, w["vol"].remote, dest_root=Path(DEST), root=ROOT, host="cluster-a-login", dry=True)
    dry2 = [ln for ln in out2["log"] if ln.startswith("DRY-RUN: ")]
    assert any(ln == f"DRY-RUN: rsync -a --partial -e 'ssh -o BatchMode=yes -o ConnectTimeout=30' {mirror_dir(w, rsha)}/ "
               f"cluster-a-login:/scratch/example/shared/exp/reproduce/{KEY}/07_mirror/{rsha[:12]}/" for ln in dry2), dry2
    assert any(ln.startswith(f"DRY-RUN: ssh cluster-a-login '{MI.prepare_command(ROOT, f'{DEST}/{rsha[:12]}')}'") for ln in dry2), dry2
    assert any(ln.startswith(f"DRY-RUN: ssh cluster-a-login 'r={ROOT}; p=\"$r\"; s=\"\"; for c in {KEY} 07_mirror {rsha[:12]}; do ") and
               f" && mkdir -p /scratch/example/shared/exp/reproduce/{KEY}/07_mirror/{rsha[:12]} && " in ln for ln in dry2), dry2
    assert any(ln.startswith(f"DRY-RUN: ssh cluster-a-login '{MI.check_command(f'{DEST}/{rsha[:12]}', (MI.SUMS_NAME, MI.MIRROR_NAME), root=ROOT)}'") and
               f" && cd /scratch/example/shared/exp/reproduce/{KEY}/07_mirror/{rsha[:12]} && sha256sum -c --quiet" in ln for ln in dry2), dry2
    assert not (state / MI.MIRROR_DIR).exists() and out2["vol"].calls == []
    # no cluster root, or a destination outside it: refused before anything (07 always passes the expanded reproduce root)
    with pytest.raises(Refusal, match=r"--cluster-root None is not an absolute path on the cluster"):
        MI.run(state=state, receipt_path=None, volume=None, environment=None, modal="/nonexistent/modal", cluster_label="cluster_a", ssh_host="cluster-a-login",
               cluster_dest=DEST, dry_run=True, modal_runner=w["vol"], log=lambda s: None)
    with pytest.raises(Refusal, match=rf"the cluster destination {re.escape(DEST)} does not lie under the cluster root /elsewhere"):
        L.mirror(state, w["vol"].remote, dest_root=Path(DEST), root="/elsewhere", host="cluster-a-login", dry=True)
    # no receipt: refused by name, naming the phase that writes one
    with pytest.raises(Refusal, match=r"no receipt at .*07_store/receipt\.json .*07 --phase receipt"):
        L.mirror(tmp_path / "nowhere", {}, dest_root=dest, dry=True)
    # this call's volume / environment must be the receipt's
    with pytest.raises(Refusal, match="--volume other is not the receipt's"):
        L.mirror(state, w["vol"].remote, dest_root=dest, dry=True, volume="other")
    with pytest.raises(Refusal, match="--environment staging is not the receipt's"):
        L.mirror(state, w["vol"].remote, dest_root=dest, dry=True, environment="staging")


# ------------------------------------------------------------------------------------------------------------------------ the success
def test_the_mirror_pulls_whole_stores_verifies_them_and_publishes_sums_and_records(tmp_path):
    w, doc, rsha = world(tmp_path)
    state, dest = w["state"], tmp_path / "cluster"
    out = L.mirror(state, w["vol"].remote, dest_root=dest)
    mdir = mirror_dir(w, rsha)
    assert out["mirror"] == str(mdir) and mdir.is_dir() and not (state / MI.MIRROR_BUILD_DIR).exists() and out["reused"] is False
    # one directory pull per store, by the volume-root path, into the build's staging (the CLI lays the folder under an existing directory)
    gets = out["vol"].gets()
    assert len(gets) == 2 and all(g[1:4] == ["volume", "get", "--force"] and g[4] == "twosi-fits" for g in gets)
    assert sorted(g[5] for g in gets) == sorted(RT.volume_rel(r["out"]) for r in doc["recs"].values())
    # the WHOLE store, byte for byte, directly under <cell>/ — nothing excluded
    got = tree_bytes(mdir)
    data = {k: v for k, v in got.items() if k not in MI.RECORDS}
    want = {f"{cell}/{rel}": b for cell, rec in doc["recs"].items() for rel, b in store_files(w["vol"].remote, rec["out"]).items()}
    assert data == want and len(data) == 46 and any(k.endswith("/rows.jsonl") for k in data) and any("/private/inspect_logs/" in k for k in data)
    # MIRROR.sha256: sha256sum's format, paths relative to the mirror root, every data file, sorted
    sums = (mdir / MI.SUMS_NAME).read_text().splitlines()
    paths = [ln.split("  ", 1)[1] for ln in sums]
    assert paths == sorted(paths) and len(sums) == len(data) and all(re.fullmatch(r"[0-9a-f]{64}  \S.*", ln) for ln in sums)
    assert {ln.split("  ", 1)[1]: ln.split("  ", 1)[0] for ln in sums} == {k: hashlib.sha256(v).hexdigest() for k, v in data.items()}
    if shutil.which("sha256sum"):
        assert subprocess.run(["sha256sum", "-c", "--quiet", MI.SUMS_NAME], cwd=mdir, capture_output=True, text=True).returncode == 0
    # mirror.json: the volume paths, the counts, the tree, the receipt sha, the time, the per-cell manifests + RESULT lines, the size check's wording
    mj = json.loads((mdir / MI.MIRROR_NAME).read_text())
    assert mj["schema"] == MI.MIRROR_SCHEMA and mj["receipt_sha256"] == rsha and mj["tree_sha"] == SHA and mj["volume"] == "twosi-fits" and mj["app"] == doc["app"]
    assert mj["files"] == 46 and mj["bytes"] == sum(len(v) for v in data.values()) and re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", mj["written_utc"])
    assert mj["receipt_files_checked"] == len(doc["pulled"]) == 14 and "modal volume ls --json" in mj["size_check"] and "no content hash" in mj["size_check"]
    for cell, rec in doc["recs"].items():
        c = mj["cells"][cell]
        assert c["volume_path"] == rec["out"] and c["volume_rel"] == RT.volume_rel(rec["out"]) and c["store_keys"] == sorted(set(rec["done_in"].values()))
        assert c["files"] == len(store_files(w["vol"].remote, rec["out"])) and c["result_last_line"].split()[3] == "result=COMPLETE"
        for key in c["store_keys"]:
            assert c["manifests"][key]["producer"] == {"driver": "served_capability_run", "tree_sha": SHA} and c["manifests"][key]["cell_id"] == cell
    # cluster.json, only after the check: the label, the host, the path under <dest>/<sha12>, the digest of MIRROR.sha256, the receipt, verified
    cl = json.loads((mdir / MI.CLUSTER_NAME).read_text())
    assert cl["schema"] == MI.CLUSTER_SCHEMA and cl["verified"] is True and cl["cluster_label"] == "cluster_a" and cl["ssh_host"] == f"local:{dest}"
    assert cl["cluster_path"] == f"{dest}/{rsha[:12]}" and cl["mirror_sha256sum_digest"] == _sha(mdir / MI.SUMS_NAME) == out["mirror_sha256sum_digest"]
    assert cl["receipt_sha256"] == rsha and cl["files"] == 46 and "--delete" not in " ".join(cl["rsync_argv"]) and "sha256sum -c --quiet MIRROR.sha256" in cl["check_command"]
    # the cluster copy (the local seam) holds every file and the three records byte-identical
    assert tree_bytes(dest / rsha[:12]) == tree_bytes(mdir)
    # the pointer the filing reads
    ptr = MI.verified_mirror(state, rsha)
    assert ptr["cluster_path"] == cl["cluster_path"] and ptr["mirror_sha256sum_digest"] == cl["mirror_sha256sum_digest"] and ptr["receipt_sha256"] == rsha
    assert ptr["files"] == 46 and ptr["tree_sha"] == SHA and ptr["store_volume_paths"] == sorted(r["out"] for r in doc["recs"].values()) and ptr["local_path"] == str(mdir)


@pytest.mark.skipif(shutil.which("rsync") is None or shutil.which("sha256sum") is None, reason="the real rsync + sha256sum are the subject here")
def test_the_real_rsync_and_sha256sum_drive_the_local_seam_end_to_end(tmp_path):
    """The `local:<dir>` seam with the REAL binaries (what the live check runs through, host aside): rsync -a --partial copies, bash -c runs
    `sha256sum -c` and `find . -mindepth 1 -printf "%y %p\\0"`; a second run moves nothing (no --delete, idempotent).
    `sha256sum -c` reads the listed paths only and rsync without --delete adds and never removes, so a file left in the destination before the
    push, or added to the copy after it, was certified as part of the copy (this test pinned the stray as "never removed" and the re-run as a
    pass). Pinned now: the inventory read back refuses it BY NAME — CONTAMINATED on a re-check (the re-run and the filing), at the push before
    any record — and deletes nothing; a cluster.json already at the destination is one of the records (an earlier push whose install did not
    finish) and is replaced by this push's. The inventory reads every entry by type, so the records count the directories too."""
    w, doc, rsha = world(tmp_path)
    dest = tmp_path / "cluster"
    mdir = mirror_dir(w, rsha)
    out = L.mirror(w["state"], w["vol"].remote, dest_root=dest, cluster_runner=MI.sh_runner)
    assert out["reused"] is False and tree_bytes(dest / rsha[:12]) == tree_bytes(mdir)
    cl = json.loads((mdir / MI.CLUSTER_NAME).read_text())
    n_dirs = sum(1 for p in mdir.rglob("*") if p.is_dir())
    assert f"the inventory of every entry by type (48 regular files, {n_dirs} directories; none unexpected)" in cl["check"], cl
    assert cl["check_command"].endswith(' && echo inventory: && find . -mindepth 1 -printf "%y %p\\0"'), cl
    out2 = L.mirror(w["state"], w["vol"].remote, dest_root=dest, cluster_runner=MI.sh_runner)
    assert out2["reused"] is True and out2["cluster_check"]["inventory"] == 49 and out2["cluster_check"]["directories"] == n_dirs, "46 data files + the three records"
    assert tree_bytes(dest / rsha[:12]) == tree_bytes(mdir)
    # a file added to the copy after the push: the re-check refuses by name — the path named, the file left in place, the local mirror untouched
    before = tree_bytes(mdir)
    stray = dest / rsha[:12] / "none" / "stray.txt"
    stray.write_text("left on the cluster")
    with pytest.raises(Refusal, match=r"the cluster copy at cluster_a local:.* is CONTAMINATED: 1 entry there is not the mirror's \(\['none/stray\.txt \(regular file\)'\]\)") as e:
        L.mirror(w["state"], w["vol"].remote, dest_root=dest, cluster_runner=MI.sh_runner)
    assert "nothing here deletes them" in str(e.value) and "its cluster record is left as it is" in str(e.value) and tree_bytes(mdir) == before
    assert stray.read_text() == "left on the cluster", "no --delete, and no deletion here either: the refusal names the file for a hand"
    # the file phase refuses the same way (rc 2), nothing filed
    assert F.rule(w) == 0
    root = F._registry_with_objects(tmp_path / "reg")
    filings = F._filings(root)
    rc, lines, why = _file(w, root, cluster_runner=MI.sh_runner)
    assert rc == 2 and "is CONTAMINATED:" in why and "none/stray.txt" in why and "nothing filed against a cluster copy that does not verify as recorded" in why, (lines, why)
    assert stray.is_file() and F._filings(root) == filings
    stray.unlink()
    assert L.mirror(w["state"], w["vol"].remote, dest_root=dest, cluster_runner=MI.sh_runner)["reused"] is True
    rc, lines, why = _file(w, root, cluster_runner=MI.sh_runner)
    assert rc == 0 and why is None, (lines, why)
    # a file in the DESTINATION before the first push: refused at the push, before any record — the mirror published and pushed, no cluster.json
    # installed, the file left in place; removed by hand, the re-run finishes the push
    w2, doc2, r2 = world(tmp_path / "w2", accs=(0.5, 0.62))
    dest2 = tmp_path / "cluster2"
    mdir2 = mirror_dir(w2, r2)
    (dest2 / r2[:12]).mkdir(parents=True)
    left = dest2 / r2[:12] / "leftover.txt"
    left.write_text("an earlier attempt")
    with pytest.raises(Refusal, match=r"the cluster destination local:.* holds 1 entry outside the mirror \(\['leftover\.txt \(regular file\)'\]\) — .*rsync adds and never deletes") as e:
        L.mirror(w2["state"], w2["vol"].remote, dest_root=dest2, cluster_runner=MI.sh_runner)
    assert "nothing recorded and nothing deleted" in str(e.value) and left.read_text() == "an earlier attempt"
    assert mdir2.is_dir() and (mdir2 / MI.SUMS_NAME).is_file() and not (mdir2 / MI.CLUSTER_NAME).exists() and (dest2 / r2[:12] / MI.SUMS_NAME).is_file(), "published, pushed, unrecorded"
    with pytest.raises(Refusal, match="has no cluster.json: its cluster copy was never verified"):
        MI.verified_mirror(w2["state"], r2)
    left.unlink()
    out3 = L.mirror(w2["state"], w2["vol"].remote, dest_root=dest2, cluster_runner=MI.sh_runner)
    assert out3["reused"] is False and (mdir2 / MI.CLUSTER_NAME).is_file() and any("finishing the cluster push" in ln for ln in out3["log"]) and out3["vol"].gets() == []
    # a cluster.json already at the destination (the record moved aside here, the same target run again within the second): one of the records,
    # tolerated by the inventory and REPLACED by this push's — rsync --ignore-times on the record, since the stale one matches it in size and
    # mtime to the second and the quick check would skip it (the read-back then refused "the record did not land whole"; found by this test)
    (mdir2 / MI.CLUSTER_NAME).rename(w2["state"] / "cluster.json.aside")
    out4 = L.mirror(w2["state"], w2["vol"].remote, dest_root=dest2, cluster_runner=MI.sh_runner)
    assert out4["reused"] is False and (mdir2 / MI.CLUSTER_NAME).is_file() and any("finishing the cluster push" in ln for ln in out4["log"])
    assert (dest2 / r2[:12] / MI.CLUSTER_NAME).read_bytes() == (mdir2 / MI.CLUSTER_NAME).read_bytes(), "the destination holds THIS push's record, not the stale one"
    assert L.mirror(w2["state"], w2["vol"].remote, dest_root=dest2, cluster_runner=MI.sh_runner)["reused"] is True


# ---------------------------------------------------------------------------------------------------------------------- the refusals
def test_a_flipped_byte_in_a_receipt_file_refuses_and_leaves_the_build_for_inspection(tmp_path):
    w, doc, rsha = world(tmp_path)
    state, dest = w["state"], tmp_path / "cluster"
    cpath = next(p for p in doc["pulled"] if p.endswith("/controls_summary.json"))
    key = L.vrel(cpath)
    orig = w["vol"].remote[key]
    w["vol"].remote[key] = orig[:-3] + bytes([orig[-3] ^ 1]) + orig[-2:]
    with pytest.raises(Refusal, match=r"hashes to [0-9a-f]{12} in the mirror, the receipt recorded [0-9a-f]{12}.*the build stays at") as e:
        L.mirror(state, w["vol"].remote, dest_root=dest)
    assert cpath in str(e.value)
    b = build_dir(w, rsha)
    assert b.is_dir() and any(p.name == "controls_summary.json" for p in b.rglob("*")), "the .building directory stays for inspection"
    assert not (state / MI.MIRROR_DIR).exists() and not dest.exists(), "nothing published, nothing pushed"
    # the leftover build refuses the next run by name — nothing is removed for you
    w["vol"].remote[key] = orig
    with pytest.raises(Refusal, match=re.escape(f"a build of this receipt's mirror already sits at {b}")) as e2:
        L.mirror(state, w["vol"].remote, dest_root=dest)
    assert "rm -rf" in str(e2.value) and "nothing pulled" in str(e2.value) and not (state / MI.MIRROR_DIR).exists()
    shutil.rmtree(b)
    out = L.mirror(state, w["vol"].remote, dest_root=dest)
    assert out["reused"] is False and mirror_dir(w, rsha).is_dir() and not (state / MI.MIRROR_BUILD_DIR).exists()
    # a receipt-hashed file MISSING from the store: the store is not the one the receipt described
    w2, doc2, rsha2 = world(tmp_path / "w2")
    del w2["vol"].remote[key]
    with pytest.raises(Refusal, match=r"the receipt pulled /vol/\S+/controls_summary\.json \(sha [0-9a-f]{12}\) but the mirrored store has no \S+/controls_summary\.json"):
        L.mirror(w2["state"], w2["vol"].remote, dest_root=tmp_path / "c2")
    assert build_dir(w2, rsha2).is_dir() and not (w2["state"] / MI.MIRROR_DIR).exists()


@pytest.mark.parametrize("case", ["manifest_tree", "manifest_cell", "manifest_missing", "result_not_complete", "result_other_sha", "result_missing"])
def test_the_manifest_and_result_checks_refuse_by_name(tmp_path, case):
    w, doc, rsha = world(tmp_path)
    rec = doc["recs"]["none"]
    root = L.vrel(rec["out"])
    key = rec["done_in"][next(iter(rec["done_in"]))]
    man_k, res_k = f"{root}/cells/{key}/manifest.json", f"{root}/RESULT"
    remote = w["vol"].remote
    if case == "manifest_tree":
        remote[man_k] = json.dumps(L.manifest_doc("none", "f" * 40)).encode(); pat = r"none: cells/.*/manifest\.json producer\.tree_sha 'ffffffffffff' is not the receipt's tree 0123456789ab"
    elif case == "manifest_cell":
        remote[man_k] = json.dumps(L.manifest_doc(DIFF, SHA)).encode(); pat = r"manifest\.json cell_id 'constv2-mix-full__lora_r4_ce_minus_ctl' is not the cell 'none'"
    elif case == "manifest_missing":
        del remote[man_k]; pat = rf"none: the receipt credits store key '{re.escape(key)}' but the mirrored store has no cells/{re.escape(key)}/manifest\.json"
    elif case == "result_not_complete":
        remote[res_k] = L.result_text(SHA, complete=False); pat = r"none: the last RESULT line does not read result=COMPLETE"
    elif case == "result_other_sha":
        remote[res_k] = L.result_text(SHA, sha="e" * 40); pat = r"none: the last RESULT line names sha=eeeeeeeeeeee, not the receipt's tree 0123456789ab"
    else:
        del remote[res_k]; pat = r"none: the mirrored store has no RESULT"
    with pytest.raises(Refusal, match=pat) as e:
        L.mirror(w["state"], remote, dest_root=tmp_path / "cluster")
    assert "nothing published" in str(e.value) and "the build stays at" in str(e.value)
    assert build_dir(w, rsha).is_dir() and not (w["state"] / MI.MIRROR_DIR).exists() and not (tmp_path / "cluster").exists()


def test_the_listing_check_holds_the_pulled_set_and_sizes_to_the_volumes_listing(tmp_path):
    """Modal exposes no content hash, so the listing check is what it can be: the same file set, and each size as `modal volume ls --json` renders
    it (humanize_filesize, verbatim from the CLI). A listing that reports another size, or a pull that lacks a listed file, refuses by name."""
    assert MI.humanize_filesize(157) == "157 B" and MI.humanize_filesize(1536) == "1.5 KiB" and MI.humanize_filesize(54886) == "53.6 KiB" and MI.humanize_filesize(1572864) == "1.5 MiB"
    assert MI.parse_filesize("157 B") == 157 and MI.parse_filesize("1.5 KiB") == 1536 and MI.parse_filesize("1.5 MiB") == 1572864
    w, doc, rsha = world(tmp_path)
    rows_rel = next(k for k in w["vol"].remote if k.endswith("/rows.jsonl"))

    class WrongSize(L.FakeModalTree):
        def __call__(self, argv, **kw):
            r = super().__call__(argv, **kw)
            if argv[1:3] == ["volume", "ls"] and "--json" in argv and r.returncode == 0:
                entries = json.loads(r.stdout)
                for e in entries:
                    if e["filename"] == rows_rel:
                        e["size"] = "9.9 MiB"
                return subprocess.CompletedProcess(argv, 0, json.dumps(entries), "")
            return r

    class DropsOne(L.FakeModalTree):
        def __call__(self, argv, **kw):
            r = super().__call__(argv, **kw)
            if argv[1:3] == ["volume", "get"] and r.returncode == 0:
                for p in Path(argv[-1]).rglob("rows.jsonl"):
                    p.unlink()
            return r

    for cls, pat in ((WrongSize, r"rows\.jsonl is \d+ B locally \([0-9.]+ KiB\), the volume listing says '9\.9 MiB'"), (DropsOne, r"listed but not pulled: \['cells/.*rows\.jsonl'")):
        vol = cls(w["vol"].remote)
        vol.remote = w["vol"].remote
        lines: list = []
        with pytest.raises(Refusal, match=pat):
            MI.run(state=w["state"], receipt_path=None, volume=None, environment=None, modal="/nonexistent/modal", cluster_label="cluster_a", ssh_host=f"local:{tmp_path / 'c'}",
                   cluster_dest=str(tmp_path / "c"), cluster_root=str(tmp_path / "c"), dry_run=False, free_headroom=0, modal_runner=vol, cluster_runner=L.local_cluster_runner,
                   log=lines.append)
        assert build_dir(w, rsha).is_dir() and not (w["state"] / MI.MIRROR_DIR).exists()
        shutil.rmtree(build_dir(w, rsha))


def test_the_disk_check_refuses_before_the_first_byte(tmp_path):
    w, doc, rsha = world(tmp_path)
    with pytest.raises(Refusal, match=r"the state filesystem under .* has [\d,]+ B free; the pull needs about [\d,]+ B x 1e\+18 \+ 0 B headroom") as e:
        L.mirror(w["state"], w["vol"].remote, dest_root=tmp_path / "cluster", free_margin=1e18)
    assert "nothing pulled" in str(e.value)
    assert not (w["state"] / MI.MIRROR_BUILD_DIR).exists() and not (w["state"] / MI.MIRROR_DIR).exists()
    # the listing ran (the estimate needs it), the pull did not
    out = L.mirror(w["state"], w["vol"].remote, dest_root=tmp_path / "cluster", dry=True)
    assert out["vol"].gets() == []


# ---------------------------------------------------------------------------------------------------------------------- idempotence
def test_a_rerun_for_the_same_receipt_verifies_the_published_copy_and_does_nothing_else(tmp_path):
    w, doc, rsha = world(tmp_path)
    state, dest = w["state"], tmp_path / "cluster"
    L.mirror(state, w["vol"].remote, dest_root=dest)
    mdir = mirror_dir(w, rsha)
    before = tree_bytes(mdir)
    rec = L.RecordingCluster(lambda: mdir)
    out = L.mirror(state, w["vol"].remote, dest_root=dest, cluster_runner=rec)
    assert out["reused"] is True and out["vol"].gets() == [] and tree_bytes(mdir) == before
    # "nothing else" since round 3 means: no pull, no rsync — and ONE ssh, the cluster copy's re-check in the recorded path (the exact command)
    assert rec.rsyncs() == [] and [c[-1] for c in rec.sshs()] == [MI.copy_check_command(f"{dest}/{rsha[:12]}", root=str(dest))] and rec.sshs()[0][:2] == ["bash", "-c"], rec.calls
    assert out["cluster_check"]["cluster_path"] == f"{dest}/{rsha[:12]}" and out["cluster_check"]["files"] == 46
    assert any("verifies (46 file(s) against MIRROR.sha256" in ln for ln in out["log"]) and any("nothing else to do" in ln for ln in out["log"])
    assert any(ln.startswith("cluster copy re-verified: ") and "46 file(s) OK" in ln and "cluster.json read back by digest" in ln for ln in out["log"]), out["log"]
    # the volume may even be gone: the published copy is what a re-run reads
    out = L.mirror(state, {}, dest_root=dest, cluster_runner=rec)
    assert out["reused"] is True and out["vol"].calls == [] and len(rec.sshs()) == 2 and rec.rsyncs() == []
    # a byte flipped in the published copy: refused by name, nothing filed against it
    victim = next(p for p in mdir.rglob("rows.jsonl"))
    data = victim.read_bytes()
    victim.write_bytes(data[:-1] + bytes([data[-1] ^ 1]))
    with pytest.raises(Refusal, match="the copy changed after it was verified; do not file against it"):
        L.mirror(state, w["vol"].remote, dest_root=dest)
    with pytest.raises(Refusal, match="changed after it was verified"):
        MI.verify_published(mdir, receipt_sha=rsha)
    victim.write_bytes(data)
    # an extra file in the published copy, or a listed one missing: not whole
    (mdir / "none" / "stray").write_bytes(b"x")
    with pytest.raises(Refusal, match=r"is not whole .*present but unlisted: \['none/stray'\]"):
        L.mirror(state, w["vol"].remote, dest_root=dest)
    (mdir / "none" / "stray").unlink()
    # a published directory whose mirror.json names another receipt
    mj = json.loads((mdir / MI.MIRROR_NAME).read_text())
    (mdir / MI.MIRROR_NAME).write_text(json.dumps({**mj, "receipt_sha256": "0" * 64}))
    with pytest.raises(Refusal, match="named for another receipt"):
        L.mirror(state, w["vol"].remote, dest_root=dest)


def test_a_second_receipt_publishes_beside_the_first_and_neither_is_deleted(tmp_path):
    w, doc, r1 = world(tmp_path)
    state, dest = w["state"], tmp_path / "cluster"
    L.mirror(state, w["vol"].remote, dest_root=dest)
    first = tree_bytes(mirror_dir(w, r1))
    # a second execution over the same stores: a new receipt (new attempts), a new mirror directory; the first stands whole
    run2 = f"{SHA[:7]}-33334444"
    for cell, fcid2 in (("none", "fc-none-0011"), (DIFF, "fc-diff-0012")):
        F.put_attempt(w["vol"], cell, fcid2, key=w["keys"][cell], run_id=run2)
    (state / RT.RUN_REPORT_NAME).write_text(F.run_report([F.rec_ok("none", w["keys"]["none"], "fc-none-0011", run_id=run2), F.rec_ok(DIFF, w["keys"][DIFF], "fc-diff-0012", run_id=run2)]))
    F.write_identity(state)
    out2 = F.receipt(w)
    r2 = out2["receipt_sha256"]
    assert r2 != r1
    L.lay_down_store_extras(w["vol"].remote, json.loads((state / RT.STORE_DIR / RT.RECEIPT_NAME).read_text()))
    m2 = L.mirror(state, w["vol"].remote, dest_root=dest)
    assert m2["reused"] is False and sorted(p.name for p in (state / MI.MIRROR_DIR).iterdir()) == sorted([r1[:12], r2[:12]])
    assert tree_bytes(mirror_dir(w, r1)) == first and sorted(p.name for p in dest.iterdir()) == sorted([r1[:12], r2[:12]])
    assert len(tree_bytes(mirror_dir(w, r2))) == len(first) + 3 * 2, "the new attempts' pointer + completion record + sandbox record per cell, nothing lost"


# ---------------------------------------------------------------------------------------------------------------- the cluster push
def test_the_cluster_push_and_remote_check_run_through_ssh_and_rsync_with_the_exact_argv(tmp_path):
    w, doc, rsha = world(tmp_path)
    state = w["state"]
    dest = f"/scratch/example/shared/exp/reproduce/{KEY}/07_mirror"
    path = f"{dest}/{rsha[:12]}"
    mdir = mirror_dir(w, rsha)
    rec = L.RecordingCluster(lambda: mdir)
    out = L.mirror(state, w["vol"].remote, dest_root=Path(dest), root=ROOT, host="cluster-a-login", cluster_runner=rec)
    ssh_prefix = ["ssh", "-n", "-o", "BatchMode=yes", "-o", "ConnectTimeout=30", "cluster-a-login"]
    assert rec.sshs()[0] == [*ssh_prefix, MI.prepare_command(ROOT, path)], "the guarded mkdir (round 6): the guard, mkdir -p, the resolved-path comparison — one ssh"
    assert rec.sshs()[0][-1].startswith(f'r={ROOT}; p="$r"; s=""; for c in {KEY} 07_mirror {rsha[:12]}; do p="$p/$c"; s="$s/$c"; ! [ -L "$p" ] || ') and f" && mkdir -p {path} && " in rec.sshs()[0][-1]
    assert rec.rsyncs()[0] == ["rsync", "-a", "--partial", "-e", "ssh -o BatchMode=yes -o ConnectTimeout=30", f"{mdir}/", f"cluster-a-login:{path}/"]
    assert rec.sshs()[1] == [*ssh_prefix, MI.check_command(path, (MI.SUMS_NAME, MI.MIRROR_NAME), root=ROOT)]
    # the record's push is the single-file rsync of the STAGED record (outside the mirror), read back there by digest, then installed by one rename
    staged = MI.record_staging_path(mdir)
    assert staged == state / MI.MIRROR_BUILD_DIR / f"{rsha[:12]}.cluster.json"
    # --ignore-times on the record alone (round 4): a stale record at the destination matching the staged one in size and mtime is still replaced
    assert rec.rsyncs()[1] == ["rsync", "-a", "--partial", "--ignore-times", "-e", "ssh -o BatchMode=yes -o ConnectTimeout=30", str(staged), f"cluster-a-login:{path}/cluster.json"]
    assert "--ignore-times" not in rec.rsyncs()[0], "the data push keeps rsync's quick check (a re-run resumes); its bytes are verified by `sha256sum -c` there"
    assert rec.sshs()[2] == [*ssh_prefix, f"sha256sum {path}/cluster.json"] and len(rec.calls) == 5
    assert not staged.exists() and not (state / MI.MIRROR_BUILD_DIR).exists(), "the staged record was installed and its staging parent removed"
    assert not any("--delete" in a for c in rec.calls for a in c)
    cl = json.loads((mdir / MI.CLUSTER_NAME).read_text())
    assert cl["cluster_path"] == path and cl["ssh_host"] == "cluster-a-login" and cl["cluster_label"] == "cluster_a" and cl["verified"] is True
    assert cl["mirror_json_sha256"] == _sha(mdir / MI.MIRROR_NAME) and cl["tree_sha"] == SHA and cl["files"] == 46
    assert out["cluster_path"] == path and out["ssh_host"] == "cluster-a-login"
    # the order: the check BEFORE the record — a failing `sha256sum -c` leaves the published mirror with no cluster.json and refuses by name
    w2, doc2, rsha2 = world(tmp_path / "w2")
    mdir2 = mirror_dir(w2, rsha2)
    bad = L.RecordingCluster(lambda: mdir2, check_rc=1)
    with pytest.raises(Refusal, match=r"the cluster check `sha256sum -c MIRROR\.sha256` on cluster-a-login:.* failed \(rc 1\): .*did NOT match"):
        L.mirror(w2["state"], w2["vol"].remote, dest_root=Path(dest), root=ROOT, host="cluster-a-login", cluster_runner=bad)
    assert mdir2.is_dir() and not (mdir2 / MI.CLUSTER_NAME).exists() and (mdir2 / MI.SUMS_NAME).is_file(), "published and verified locally; the cluster copy unrecorded"
    with pytest.raises(Refusal, match="has no cluster.json: its cluster copy was never verified"):
        MI.verified_mirror(w2["state"], rsha2)
    # a re-run finishes the cluster step only: no pull, the mirror verified, the record written
    good = L.RecordingCluster(lambda: mdir2)
    out2 = L.mirror(w2["state"], w2["vol"].remote, dest_root=Path(dest), root=ROOT, host="cluster-a-login", cluster_runner=good)
    assert out2["vol"].gets() == [] and any("finishing the cluster push" in ln for ln in out2["log"]) and (mdir2 / MI.CLUSTER_NAME).is_file() and out2["reused"] is False
    assert len(good.rsyncs()) == 2 and MI.verified_mirror(w2["state"], rsha2)["cluster_path"] == f"{dest}/{rsha2[:12]}"
    # the remote records' digests must be the published ones; an rsync that fails refuses by name
    w3, doc3, rsha3 = world(tmp_path / "w3")
    mdir3 = mirror_dir(w3, rsha3)
    with pytest.raises(Refusal, match="the cluster copy's records differ from the published ones"):
        L.mirror(w3["state"], w3["vol"].remote, dest_root=Path(dest), root=ROOT, host="cluster-a-login", cluster_runner=L.RecordingCluster(lambda: mdir3, digest_of=lambda p: "0" * 64))
    assert not (mdir3 / MI.CLUSTER_NAME).exists()
    with pytest.raises(Refusal, match=r"rsync to cluster-a-login:.* failed \(rc 23\)"):
        L.mirror(w3["state"], w3["vol"].remote, dest_root=Path(dest), root=ROOT, host="cluster-a-login", cluster_runner=L.RecordingCluster(lambda: mdir3, rsync_rc=23))
    # a cluster with no ssh host, or a relative destination: refused before anything
    with pytest.raises(Refusal, match="cluster cluster_a has no CLUSTER_SSH"):
        MI.Cluster("cluster_a", "", dest)
    with pytest.raises(Refusal, match="--cluster-dest 'relative/dir' is not an absolute path"):
        MI.Cluster("cluster_a", "cluster-a-login", "relative/dir")


def test_a_record_transfer_that_fails_installs_no_cluster_record_and_the_rerun_finishes_the_push(tmp_path):
    """The data rsync and `sha256sum -c` pass, then the cluster.json transfer fails (the remote fills, the
    connection drops) or lands with another digest. The published mirror must hold NO cluster.json (the first cut wrote the record into the
    mirror BEFORE its transfer, so a re-run and `--phase file` accepted a `verified: true` record whose copy never arrived), the staged record
    is gone, `verified_mirror` refuses, and the re-run finishes the push — re-pushing the record — instead of reporting the mirror done."""
    dest = f"/scratch/example/shared/exp/reproduce/{KEY}/07_mirror"
    for label, knob, pat in (("rc", {"record_rsync_rc": 12}, r"rsync of cluster\.json to cluster-a-login:.* failed \(rc 12\): rsync: connection unexpectedly closed"),
                             ("digest", {"record_digest_of": lambda p: "f" * 64}, r"cluster\.json on cluster-a-login:.* hashes to ffffffffffff, the staged one [0-9a-f]{12} — the record did not land whole")):
        w, doc, rsha = world(tmp_path / label)
        state = w["state"]
        mdir = mirror_dir(w, rsha)
        bad = L.RecordingCluster(lambda: mdir, **knob)
        with pytest.raises(Refusal, match=pat) as e:
            L.mirror(state, w["vol"].remote, dest_root=Path(dest), root=ROOT, host="cluster-a-login", cluster_runner=bad)
        assert f"cluster.json was NOT installed at {mdir}" in str(e.value) and "run the mirror again to finish the push" in str(e.value)
        assert len(bad.record_pushes()) == 1 and bad.sshs()[1][-1] == MI.check_command(f"{dest}/{rsha[:12]}", (MI.SUMS_NAME, MI.MIRROR_NAME), root=ROOT), "the data push and its check had passed"
        assert mdir.is_dir() and (mdir / MI.SUMS_NAME).is_file() and (mdir / MI.MIRROR_NAME).is_file() and not (mdir / MI.CLUSTER_NAME).exists(), "published; NO cluster record"
        assert not MI.record_staging_path(mdir).exists() and not (state / MI.MIRROR_BUILD_DIR).exists(), "the staged record is gone"
        with pytest.raises(Refusal, match="has no cluster.json: its cluster copy was never verified"):
            MI.verified_mirror(state, rsha)
        # the re-run does NOT report the mirror done: it pushes again (the data, then the record) and installs the record only after the read-back
        good = L.RecordingCluster(lambda: mdir)
        out = L.mirror(state, w["vol"].remote, dest_root=Path(dest), root=ROOT, host="cluster-a-login", cluster_runner=good)
        assert out["reused"] is False and any("finishing the cluster push" in ln for ln in out["log"]) and out["vol"].gets() == []
        assert len(good.rsyncs()) == 2 and len(good.record_pushes()) == 1 and good.rsyncs().index(good.record_pushes()[0]) == 1
        assert (mdir / MI.CLUSTER_NAME).is_file() and MI.verified_mirror(state, rsha)["cluster_path"] == f"{dest}/{rsha[:12]}"
        cl = json.loads((mdir / MI.CLUSTER_NAME).read_text())
        assert cl["verified"] is True and cl["mirror_json_sha256"] == _sha(mdir / MI.MIRROR_NAME) and cl["mirror_sha256sum_digest"] == _sha(mdir / MI.SUMS_NAME)
        assert any("cluster.json pushed, read back and installed" in ln for ln in out["log"])


def test_the_cluster_check_reads_back_the_inventory_and_refuses_an_unlisted_file_by_name_without_deleting_it(tmp_path):
    """`sha256sum -c MIRROR.sha256` reads the LISTED paths only, and rsync without --delete adds and never
    removes — so a file left in the cluster destination before the push, or added to the copy afterwards, was outside the check and the copy was
    certified (the filing then pointed at a directory holding files outside the recorded snapshot). Pinned over the stubbed ssh: the exact command
    carries `echo inventory:` and the inventory command after the records' digests (`find . -mindepth 1 -printf "%y %p\\0"`, every
    entry by type), at the push and on every re-check; the regular files read back must be exactly MIRROR.sha256's paths plus the three
    records — at the push a leftover refuses BEFORE any record (a cluster.json already there is one of the records: an earlier push whose
    install did not finish, replaced by this push's), on a re-check an unlisted file refuses the re-run (CONTAMINATED) and the filing (rc 2)
    by name, the local mirror and its record untouched and nothing deleted anywhere; a check that stops before the mark is no pass; the
    parser refuses what `find` does not print."""
    # the helpers
    assert MI.INVENTORY_MARK == "inventory:"
    assert MI.copy_check_command("/x/y", root="/x") == (MI.guard_command("/x", "/x/y") + ' && cd /x/y && sha256sum -c --quiet MIRROR.sha256 && sha256sum MIRROR.sha256 mirror.json '
                                                       'cluster.json && echo inventory: && find . -mindepth 1 -printf "%y %p\\0"')
    assert MI.check_command("/r/x y", (MI.SUMS_NAME, MI.MIRROR_NAME), root="/r") == (MI.guard_command("/r", "/r/x y") + " && cd '/r/x y' && sha256sum -c --quiet MIRROR.sha256 && "
                                                                                   'sha256sum MIRROR.sha256 mirror.json && echo inventory: && find . -mindepth 1 -printf "%y %p\\0"')
    d, inv = MI.parse_check_output("a" * 64 + "  MIRROR.sha256\n" + "b" * 64 + "  mirror.json\ninventory:\nf ./MIRROR.sha256\0f ./mirror.json\0d ./none\0f ./none/RESULT\0")
    assert d == {"MIRROR.sha256": "a" * 64, "mirror.json": "b" * 64} and inv == [("f", "MIRROR.sha256"), ("f", "mirror.json"), ("d", "none"), ("f", "none/RESULT")]
    assert MI.parse_check_output("a" * 64 + "  MIRROR.sha256\n") == ({"MIRROR.sha256": "a" * 64}, None), "no mark: the command did not run whole"
    with pytest.raises(Refusal, match=r"printed an entry after `inventory:` that `find \. -mindepth 1 .*` does not print: 'not-a-find-record'"):
        MI.parse_check_output("inventory:\nnot-a-find-record\0")
    sums = {"none/RESULT": "0" * 64, "none/cells/k/rows.jsonl": "1" * 64}
    dirs = ["none", "none/cells", "none/cells/k"]
    files = lambda *paths: [("f", p) for p in paths]  # noqa: E731
    assert MI.unexpected_entries(files("MIRROR.sha256", "mirror.json", "cluster.json", "none/RESULT", "none/cells/k/rows.jsonl", "none/x.bin"), sums, directories=[]) == \
        (["none/x.bin (regular file)"], [])
    assert MI.unexpected_entries(files("MIRROR.sha256", "none/RESULT"), sums, directories=[]) == ([], ["none/cells/k/rows.jsonl"])
    assert MI.unexpected_entries(files("MIRROR.sha256", "mirror.json", "none/RESULT", "none/cells/k/rows.jsonl") + [("d", d) for d in dirs], sums, directories=dirs) == ([], [])
    # the push: the check carries the inventory; a leftover at the destination refuses before any record — the mirror published, no cluster.json
    w, doc, rsha = world(tmp_path)
    state = w["state"]
    dest = f"/scratch/example/shared/exp/reproduce/{KEY}/07_mirror"
    path = f"{dest}/{rsha[:12]}"
    mdir = mirror_dir(w, rsha)
    ssh_prefix = ["ssh", "-n", "-o", "BatchMode=yes", "-o", "ConnectTimeout=30", "cluster-a-login"]
    bad = L.RecordingCluster(lambda: mdir, extra_paths=["leftover.txt"])
    with pytest.raises(Refusal, match=r"the cluster destination cluster-a-login:.* holds 1 entry outside the mirror \(\['leftover\.txt \(regular file\)'\]\) — .*rsync adds and never deletes") as e:
        L.mirror(state, w["vol"].remote, dest_root=Path(dest), root=ROOT, host="cluster-a-login", cluster_runner=bad)
    assert "nothing recorded and nothing deleted" in str(e.value) and "move those entries OUT by hand" in str(e.value) and "run the mirror again" in str(e.value)
    assert bad.sshs()[1] == [*ssh_prefix, MI.check_command(path, (MI.SUMS_NAME, MI.MIRROR_NAME), root=ROOT)]
    assert bad.record_pushes() == [] and mdir.is_dir() and (mdir / MI.SUMS_NAME).is_file() and not (mdir / MI.CLUSTER_NAME).exists(), "published; the push checked; no record"
    assert not MI.record_staging_path(mdir).exists()
    # a cluster.json already there is one of the records: tolerated, then replaced by this push's; the record counts the inventory, directories too
    n_dirs = sum(1 for p in mdir.rglob("*") if p.is_dir())
    ok = L.RecordingCluster(lambda: mdir, extra_paths=["cluster.json"])
    out = L.mirror(state, w["vol"].remote, dest_root=Path(dest), root=ROOT, host="cluster-a-login", cluster_runner=ok)
    assert out["reused"] is False and (mdir / MI.CLUSTER_NAME).is_file() and len(ok.record_pushes()) == 1 and out["vol"].gets() == []
    cl = json.loads((mdir / MI.CLUSTER_NAME).read_text())
    assert cl["check"].endswith(f"+ the inventory of every entry by type (49 regular files, {n_dirs} directories; none unexpected)"), cl["check"]
    assert cl["check_command"] == MI.check_command(path, (MI.SUMS_NAME, MI.MIRROR_NAME), root=ROOT)
    # the re-check: ONE ssh with the exact command; the clean copy passes and counts 46 + 3
    before = tree_bytes(mdir)
    rec = L.RecordingCluster(lambda: mdir)
    out2 = L.mirror(state, w["vol"].remote, dest_root=Path(dest), root=ROOT, host="cluster-a-login", cluster_runner=rec)
    assert out2["reused"] is True and rec.calls == [[*ssh_prefix, MI.copy_check_command(path, root=ROOT)]] and out2["cluster_check"]["inventory"] == 49
    assert out2["cluster_check"]["directories"] == n_dirs and any(f"inventory 49 regular file(s) and {n_dirs} directories, none unexpected" in ln for ln in out2["log"]), out2["log"]
    # two files added to the copy after the push: CONTAMINATED, both named (sorted), nothing deleted (the stub still holds them), the local side untouched
    dirty = L.RecordingCluster(lambda: mdir, extra_paths=["none/planted.bin", "cells/zzz/extra.jsonl"])
    with pytest.raises(Refusal, match=r"the cluster copy at cluster_a cluster-a-login:.* is CONTAMINATED: 2 entries there are not the mirror's "
                                      r"\(\['cells/zzz/extra\.jsonl \(regular file\)', 'none/planted\.bin \(regular file\)'\]\)") as e:
        L.mirror(state, w["vol"].remote, dest_root=Path(dest), root=ROOT, host="cluster-a-login", cluster_runner=dirty)
    assert "nothing here deletes them" in str(e.value) and "its cluster record is left as it is" in str(e.value) and tree_bytes(mdir) == before and dirty.rsyncs() == []
    # the file phase: the same ONE ssh, the same refusal by name (rc 2), nothing filed
    assert F.rule(w) == 0
    root = F._registry_with_objects(tmp_path / "reg")
    filings = F._filings(root)
    dirty2 = L.RecordingCluster(lambda: mdir, extra_paths=["none/planted.bin"])
    rc, lines, why = _file(w, root, cluster_runner=dirty2)
    assert rc == 2 and "is CONTAMINATED: 1 entry there is not the mirror's" in why and "'none/planted.bin (regular file)'" in why and "nothing filed against a cluster copy that does not verify as recorded" in why, (lines, why)
    assert dirty2.calls == [[*ssh_prefix, MI.copy_check_command(path, root=ROOT)]] and F._filings(root) == filings and not any(ln.startswith("spec: ") for ln in lines)
    # a check that never reached the mark (the command did not run whole) is no pass, on the re-check and at the push alike
    with pytest.raises(Refusal, match=r"could not be inventoried: `r=.* && cd .*` exited 0 there but printed no `inventory:` line"):
        L.mirror(state, w["vol"].remote, dest_root=Path(dest), root=ROOT, host="cluster-a-login", cluster_runner=L.RecordingCluster(lambda: mdir, no_inventory=True))
    w2, doc2, r2 = world(tmp_path / "w2", accs=(0.5, 0.62))
    mdir2 = mirror_dir(w2, r2)
    with pytest.raises(Refusal, match=r"the cluster check on cluster-a-login:.* exited 0 but printed no `inventory:` line"):
        L.mirror(w2["state"], w2["vol"].remote, dest_root=Path(dest), root=ROOT, host="cluster-a-login", cluster_runner=L.RecordingCluster(lambda: mdir2, no_inventory=True))
    assert mdir2.is_dir() and not (mdir2 / MI.CLUSTER_NAME).exists()
    # the clean copy: the re-run and the filing pass
    assert L.mirror(state, w["vol"].remote, dest_root=Path(dest), root=ROOT, host="cluster-a-login", cluster_runner=L.RecordingCluster(lambda: mdir))["reused"] is True
    rc, lines, why = _file(w, root, cluster_runner=L.RecordingCluster(lambda: mdir))
    assert rc == 0 and why is None and any("filed: analysis_id" in ln for ln in lines), (lines, why)


def test_the_cluster_inventory_reads_every_entry_by_type_and_refuses_a_symlink_a_special_file_or_a_stray_directory_by_name(tmp_path):
    """The inventory was `find . -type f` — regular files alone — so a symlink (to anywhere), a fifo, a socket, a
    device or a stray directory in the destination or the copy was never read back: rsync does not delete it, `sha256sum -c` reads the listed
    files, and the copy was certified with an unrecorded entry in it, unlike the local mirror, which refuses a symlink. Pinned: the check is
    `find . -mindepth 1 -printf "%y %p\\0"` (every entry, its type letter, NUL-terminated, so a newline in a name forges nothing), the parser reads
    that and nothing else (a cut-off inventory refuses), and the entry check accepts a regular file MIRROR.sha256 or the records name and a
    directory mirror.json lists, nothing else — a listed path present as a symlink is both not the mirror's and lacking. Over the stubbed ssh a
    symlink in the destination refuses the push before any record, and a symlink, a fifo, an empty directory, a socket and a device in the copy
    each refuse the re-run (CONTAMINATED, the entry named with its type) and the filing (rc 2); nothing is deleted or written anywhere."""
    # the helpers: the command, the parser, the entry check
    assert MI.INVENTORY_CMD == 'find . -mindepth 1 -printf "%y %p\\0"' and MI.copy_check_command("/r/x", root="/r").endswith(f" && echo {MI.INVENTORY_MARK} && {MI.INVENTORY_CMD}")
    found = [("f", "MIRROR.sha256"), ("d", "none"), ("f", "none/RESULT"), ("l", "none/link"), ("p", "none/pipe"), ("d", "junk"), ("f", "none/odd\nname")]
    d, entries = MI.parse_check_output("a" * 64 + "  MIRROR.sha256\ninventory:\n" + "".join(f"{t} ./{p}\0" for t, p in found))
    assert d == {"MIRROR.sha256": "a" * 64} and entries == found, "every entry by type, a newline-bearing name read whole"
    assert MI.parse_check_output("inventory:\n") == ({}, [])
    with pytest.raises(Refusal, match=r"does not end on a NUL-terminated entry \('f \./a'\) — the inventory is cut off"):
        MI.parse_check_output("inventory:\nf ./a")
    with pytest.raises(Refusal, match=r"printed an entry after `inventory:` that `find .* does not print: 'f a'"):
        MI.parse_check_output("inventory:\nf a\0")
    sums = {"none/RESULT": "0" * 64}
    assert MI.unexpected_entries(entries, sums, directories=["none"]) == (["junk (directory)", "none/link (symlink)", "none/odd\nname (regular file)", "none/pipe (fifo)"], [])
    assert MI.unexpected_entries([("l", "none/RESULT"), ("d", "none")], sums, directories=["none", "none/empty"]) == (["none/RESULT (symlink)"], ["none/RESULT", "none/empty (directory)"])
    assert MI.type_word("s") == "socket" and MI.type_word("b") == "block device" and MI.type_word("c") == "character device" and MI.type_word("Z") == "type 'Z'"
    # over the stubbed ssh: a symlink in the destination before the first push refuses at the push, before any record
    w, doc, rsha = world(tmp_path)
    state = w["state"]
    dest = f"/scratch/example/shared/exp/reproduce/{KEY}/07_mirror"
    mdir = mirror_dir(w, rsha)
    bad = L.RecordingCluster(lambda: mdir, extra_entries=[("l", "latest")])
    with pytest.raises(Refusal, match=r"the cluster destination cluster-a-login:.* holds 1 entry outside the mirror \(\['latest \(symlink\)'\]\)") as e:
        L.mirror(state, w["vol"].remote, dest_root=Path(dest), root=ROOT, host="cluster-a-login", cluster_runner=bad)
    assert "or a symlink, fifo, socket or device, which the mirror never holds" in str(e.value) and bad.record_pushes() == [] and not (mdir / MI.CLUSTER_NAME).exists()
    assert L.mirror(state, w["vol"].remote, dest_root=Path(dest), root=ROOT, host="cluster-a-login", cluster_runner=L.RecordingCluster(lambda: mdir))["reused"] is False
    # each kind of entry in the copy: the re-run and the filing refuse by name, the entry named with its type; nothing deleted, nothing filed
    before = tree_bytes(mdir)
    assert F.rule(w) == 0
    root = F._registry_with_objects(tmp_path / "reg")
    filings = F._filings(root)
    cases = [([("l", "none/link")], r"\['none/link \(symlink\)'\]"), ([("p", "none/pipe")], r"\['none/pipe \(fifo\)'\]"),
             ([("d", "none/junk")], r"\['none/junk \(directory\)'\]"), ([("s", "sock"), ("c", "dev")], r"\['dev \(character device\)', 'sock \(socket\)'\]")]
    for extra, pat in cases:
        with pytest.raises(Refusal, match=rf"the cluster copy at cluster_a cluster-a-login:.* is CONTAMINATED: {len(extra)} entr(y|ies) there (is|are) not the mirror's \({pat}\)") as e:
            L.mirror(state, w["vol"].remote, dest_root=Path(dest), root=ROOT, host="cluster-a-login", cluster_runner=L.RecordingCluster(lambda: mdir, extra_entries=extra))
        assert "nothing here deletes them" in str(e.value) and "its cluster record is left as it is" in str(e.value)
        rc, lines, why = _file(w, root, cluster_runner=L.RecordingCluster(lambda: mdir, extra_entries=extra))
        assert rc == 2 and "is CONTAMINATED:" in why and re.search(pat, why) and "nothing filed against a cluster copy that does not verify as recorded" in why, (lines, why)
    assert tree_bytes(mdir) == before and F._filings(root) == filings
    # the clean copy: the re-run and the filing pass
    assert L.mirror(state, w["vol"].remote, dest_root=Path(dest), root=ROOT, host="cluster-a-login", cluster_runner=L.RecordingCluster(lambda: mdir))["reused"] is True
    rc, lines, why = _file(w, root, cluster_runner=L.RecordingCluster(lambda: mdir))
    assert rc == 0 and why is None, (lines, why)


def _gnu_find() -> bool:
    try:
        return "GNU findutils" in subprocess.run(["find", "--version"], capture_output=True, text=True).stdout
    except OSError:
        return False


@pytest.mark.skipif(shutil.which("rsync") is None or shutil.which("sha256sum") is None or not _gnu_find(), reason="the real rsync + sha256sum + GNU find are the subject here")
def test_the_real_find_inventories_a_symlink_a_fifo_and_a_stray_directory_in_the_copy_and_each_refuses_by_name(tmp_path):
    """With the REAL binaries under the local seam (rsync, sha256sum, GNU find): a symlink in the destination before the
    first push refuses the push before any record and is left in place; after the push, a symlink added to the copy, a listed data file
    REPLACED by a symlink to the same bytes elsewhere (`sha256sum -c` passes through the link), a fifo and an empty directory each refuse the
    re-run (CONTAMINATED, the entry named with its type) and the filing (rc 2), nothing deleted and the local mirror untouched; restored, both
    pass."""
    w, doc, rsha = world(tmp_path)
    dest = tmp_path / "cluster"
    mdir = mirror_dir(w, rsha)
    copy = dest / rsha[:12]
    copy.mkdir(parents=True)
    os.symlink("/etc", copy / "latest")
    with pytest.raises(Refusal, match=r"the cluster destination local:.* holds 1 entry outside the mirror \(\['latest \(symlink\)'\]\)"):
        L.mirror(w["state"], w["vol"].remote, dest_root=dest, cluster_runner=MI.sh_runner)
    assert (copy / "latest").is_symlink() and mdir.is_dir() and not (mdir / MI.CLUSTER_NAME).exists() and (copy / MI.SUMS_NAME).is_file(), "published, pushed, unrecorded; the link stays"
    (copy / "latest").unlink()
    assert L.mirror(w["state"], w["vol"].remote, dest_root=dest, cluster_runner=MI.sh_runner)["reused"] is False and (mdir / MI.CLUSTER_NAME).is_file()
    assert F.rule(w) == 0
    root = F._registry_with_objects(tmp_path / "reg")
    filings = F._filings(root)
    before = tree_bytes(mdir)

    def refuses(pat: str) -> str:
        with pytest.raises(Refusal, match=rf"the cluster copy at cluster_a local:.* is CONTAMINATED: .*{pat}") as e:
            L.mirror(w["state"], w["vol"].remote, dest_root=dest, cluster_runner=MI.sh_runner)
        assert "nothing here deletes them" in str(e.value) and "its cluster record is left as it is" in str(e.value) and tree_bytes(mdir) == before
        rc, lines, why = _file(w, root, cluster_runner=MI.sh_runner)
        assert rc == 2 and "is CONTAMINATED:" in why and re.search(pat, why) and F._filings(root) == filings, (lines, why)
        return str(e.value)

    link = copy / "none" / "link"
    os.symlink("/etc", link)
    refuses(r"\(\['none/link \(symlink\)'\]\)")
    assert link.is_symlink()
    link.unlink()
    victim = next(p for p in copy.rglob("rows.jsonl"))
    rel = victim.relative_to(copy).as_posix()
    aside = tmp_path / "rows.aside"
    victim.rename(aside)
    os.symlink(aside, victim)
    assert subprocess.run(["sha256sum", "-c", "--quiet", MI.SUMS_NAME], cwd=copy, capture_output=True).returncode == 0, "the sums alone pass through the symlink"
    refuses(rf"\(\['{re.escape(rel)} \(symlink\)'\]\)")
    victim.unlink()
    aside.rename(victim)
    os.mkfifo(copy / "none" / "pipe")
    refuses(r"\(\['none/pipe \(fifo\)'\]\)")
    (copy / "none" / "pipe").unlink()
    (copy / "none" / "junk").mkdir()
    refuses(r"\(\['none/junk \(directory\)'\]\)")
    (copy / "none" / "junk").rmdir()
    assert L.mirror(w["state"], w["vol"].remote, dest_root=dest, cluster_runner=MI.sh_runner)["reused"] is True
    rc, lines, why = _file(w, root, cluster_runner=MI.sh_runner)
    assert rc == 0 and why is None, (lines, why)


def test_mirror_json_records_every_directory_and_the_local_and_cluster_checks_hold_the_directory_set_to_it(tmp_path):
    """Round 5: the Modal CLI creates a directory for every DIRECTORY entry it downloads (modal/cli/_download.py, 1.5.4), an EMPTY one included,
    so the mirror's directory set is RECORDED in mirror.json (`directories`, schema stage07-mirror/2) rather than inferred from the file list,
    and both checks hold the directory set to it. Pinned: mirror.json lists every directory of the mirror, sorted; an empty directory on the
    volume is pulled, recorded, pushed, and the cluster check passes with it; a directory added to the local mirror after it was verified, or a
    recorded one removed, refuses the filing's gate by name, as does a fifo or a symlink anywhere in it; a mirror.json of the earlier schema, or
    one without the list, refuses by name, naming the re-run; the recorded empty directory removed from the COPY is MISSING there."""
    w, doc, rsha = world(tmp_path)
    state, dest = w["state"], tmp_path / "cluster"
    cell = next(iter(doc["recs"]))
    store_rel = L.vrel(doc["recs"][cell]["out"])
    empty_rel = f"{store_rel}/scratch_empty"

    class WithEmptyDir(L.FakeModalTree):
        """The CLI's view of a store holding an EMPTY directory: `ls --json` of the store root lists it as a dir, `ls` of it is an empty list, and
        `volume get` of the store creates it (a DIRECTORY entry downloads as a mkdir)."""

        def __call__(self, argv, **kw):
            sub = list(argv)[1:]
            if sub[:2] == ["volume", "ls"] and "--json" in sub and sub[-1].strip("/") == empty_rel:
                self.calls.append(list(argv))
                return subprocess.CompletedProcess(list(argv), 0, "[]", "")
            r = super().__call__(argv, **kw)
            if sub[:2] == ["volume", "ls"] and "--json" in sub and r.returncode == 0 and sub[-1].strip("/") == store_rel:
                entries = json.loads(r.stdout) + [{"filename": empty_rel, "type": "dir", "size": "0 B", "created_modified": "x"}]
                return subprocess.CompletedProcess(list(argv), 0, json.dumps(entries), "")
            if sub[:2] == ["volume", "get"] and r.returncode == 0 and sub[-2].strip("/") == store_rel:
                (Path(sub[-1]) / Path(store_rel).name / "scratch_empty").mkdir(parents=True, exist_ok=True)
            return r

    vol = WithEmptyDir(w["vol"].remote)
    vol.remote = w["vol"].remote
    lines: list = []
    out = MI.run(state=state, receipt_path=None, volume=None, environment=None, modal="/nonexistent/modal", cluster_label="cluster_a", ssh_host=f"local:{dest}",
                 cluster_dest=str(dest), cluster_root=str(dest), dry_run=False, free_headroom=0, modal_runner=vol, cluster_runner=L.local_cluster_runner, log=lines.append)
    mdir = mirror_dir(w, rsha)
    empty = mdir / cell / "scratch_empty"
    assert out["reused"] is False and (mdir / MI.CLUSTER_NAME).is_file() and empty.is_dir() and not any(empty.iterdir()), "the empty directory is pulled"
    mj = json.loads((mdir / MI.MIRROR_NAME).read_text())
    assert mj["schema"] == MI.MIRROR_SCHEMA == "stage07-mirror/2"
    assert mj["directories"] == sorted(p.relative_to(mdir).as_posix() for p in mdir.rglob("*") if p.is_dir()) and f"{cell}/scratch_empty" in mj["directories"]
    assert mj["directories"] == sorted(mj["directories"]) and len(mj["directories"]) >= 4, "two cells, each with cells/<key>/… beneath it"
    assert (dest / rsha[:12] / cell / "scratch_empty").is_dir(), "pushed (rsync -a carries an empty directory)"
    cl = json.loads((mdir / MI.CLUSTER_NAME).read_text())
    assert f"{len(mj['directories'])} directories; none unexpected" in cl["check"], cl["check"]
    assert L.mirror(state, w["vol"].remote, dest_root=dest)["reused"] is True and L.mirror(state, w["vol"].remote, dest_root=dest)["cluster_check"]["directories"] == len(mj["directories"])
    # the local gate: a directory added after the check, a recorded one removed, a fifo, a symlink — each refuses by name
    (mdir / cell / "added").mkdir()
    with pytest.raises(Refusal, match=rf"is not whole in its directories \(listed in mirror\.json but absent: \[\]; present but unlisted: \['{re.escape(cell)}/added'\]\)"):
        MI.verified_mirror(state, rsha)
    (mdir / cell / "added").rmdir()
    empty.rmdir()
    with pytest.raises(Refusal, match=rf"listed in mirror\.json but absent: \['{re.escape(cell)}/scratch_empty'\]; present but unlisted: \[\]"):
        MI.verified_mirror(state, rsha)
    empty.mkdir()
    assert MI.verified_mirror(state, rsha)["files"] == 46
    os.mkfifo(mdir / cell / "pipe")
    with pytest.raises(Refusal, match=rf"{re.escape(cell)}/pipe is a fifo, neither a regular file nor a directory — a store holds regular files in directories and nothing else"):
        MI.verified_mirror(state, rsha)
    (mdir / cell / "pipe").unlink()
    os.symlink("/etc", mdir / cell / "link")
    with pytest.raises(Refusal, match=rf"{re.escape(cell)}/link is a symlink — a store holds regular files in directories and nothing else"):
        MI.verified_mirror(state, rsha)
    (mdir / cell / "link").unlink()
    # a mirror.json of the earlier schema, or one without the list: refused by name, naming the re-run (the cluster.json digest check comes after)
    mj_p = mdir / MI.MIRROR_NAME
    orig = mj_p.read_bytes()
    mj_p.write_text(json.dumps({**mj, "schema": "stage07-mirror/1"}))
    with pytest.raises(Refusal, match=r"schema 'stage07-mirror/1' is not 'stage07-mirror/2' — a mirror written by an earlier cut of this tool") as e:
        MI.verified_mirror(state, rsha)
    assert "run the mirror again" in str(e.value)
    mj_p.write_text(json.dumps({k: v for k, v in mj.items() if k != "directories"}))
    with pytest.raises(Refusal, match=r"records no `directories` list \('None'\) — not a mirror\.json of schema stage07-mirror/2"):
        MI.verified_mirror(state, rsha)
    mj_p.write_bytes(orig)
    assert MI.verified_mirror(state, rsha)["mirror_json_sha256"] == cl["mirror_json_sha256"]
    # the recorded empty directory removed from the COPY: MISSING there, named with its type; restored, the re-run passes
    copy_empty = dest / rsha[:12] / cell / "scratch_empty"
    copy_empty.rmdir()
    with pytest.raises(Refusal, match=rf"the cluster copy at cluster_a local:.* is MISSING 1 entry of the mirror \(\['{re.escape(cell)}/scratch_empty \(directory\)'\]\)") as e:
        L.mirror(state, w["vol"].remote, dest_root=dest)
    assert "a recorded directory not there" in str(e.value) and "its cluster record is left as it is" in str(e.value)
    copy_empty.mkdir()
    assert L.mirror(state, w["vol"].remote, dest_root=dest)["reused"] is True


def test_the_filing_gate_re_hashes_the_whole_copy_and_holds_both_records_to_the_cluster_record(tmp_path):
    """`verified_mirror` — the ONE check `--phase file` runs — re-hashes every data file against MIRROR.sha256
    and holds mirror.json to cluster.json's `mirror_json_sha256` (the first cut hashed the two records alone, so a store file truncated, or a
    mirror.json edited, after the cluster check was filed as the verified copy without a refusal)."""
    w, doc, rsha = world(tmp_path)
    state, dest = w["state"], tmp_path / "cluster"
    L.mirror(state, w["vol"].remote, dest_root=dest)
    mdir = mirror_dir(w, rsha)
    ptr = MI.verified_mirror(state, rsha)
    cl = json.loads((mdir / MI.CLUSTER_NAME).read_text())
    assert ptr["mirror_json_sha256"] == _sha(mdir / MI.MIRROR_NAME) == cl["mirror_json_sha256"] and ptr["files"] == cl["files"] == 46
    # a data file truncated after the mirror verified (the three records untouched): refused by name
    victim = next(p for p in mdir.rglob("rows.jsonl"))
    data = victim.read_bytes()
    victim.write_bytes(data[:-7])
    with pytest.raises(Refusal, match=r"rows\.jsonl hashes to [0-9a-f]{12}, MIRROR\.sha256 says [0-9a-f]{12} — the copy changed after it was verified; do not file against it"):
        MI.verified_mirror(state, rsha)
    victim.write_bytes(data)
    assert MI.verified_mirror(state, rsha) == ptr
    # mirror.json rewritten (the same receipt, one field changed) while MIRROR.sha256 and cluster.json stand: refused on cluster.json's mirror_json_sha256
    mj_bytes = (mdir / MI.MIRROR_NAME).read_bytes()
    mj = json.loads(mj_bytes)
    (mdir / MI.MIRROR_NAME).write_text(json.dumps({**mj, "bytes": mj["bytes"] + 1}, indent=1, sort_keys=True) + "\n")
    with pytest.raises(Refusal, match=r"cluster\.json records mirror\.json as [0-9a-f]{12}, the mirror\.json beside it hashes to [0-9a-f]{12} — the mirror record changed after the cluster copy was verified"):
        MI.verified_mirror(state, rsha)
    (mdir / MI.MIRROR_NAME).write_bytes(mj_bytes)
    # a cluster record whose file count or tree is not mirror.json's: not a verified mirror
    cl_bytes = (mdir / MI.CLUSTER_NAME).read_bytes()
    (mdir / MI.CLUSTER_NAME).write_text(json.dumps({**cl, "files": cl["files"] - 1}))
    with pytest.raises(Refusal, match=r"cluster\.json and .*mirror\.json disagree \(tree [0-9a-f]{12} vs [0-9a-f]{12}; files 45 vs 46; MIRROR\.sha256 lists 46\)"):
        MI.verified_mirror(state, rsha)
    (mdir / MI.CLUSTER_NAME).write_bytes(cl_bytes)
    assert MI.verified_mirror(state, rsha) == ptr


def test_a_rerun_or_a_filing_under_another_cluster_target_refuses_by_name_and_copies_nothing(tmp_path):
    """Once cluster.json existed, a re-run under another label, ssh host or destination compared nothing and
    reported the mirror done — a site whose destination had moved to its canonical directory kept the copy at the former one, and `--phase
    file` recorded that path. Now the recorded target must be this call's (the mirror) and this site's (the file phase's cross-check), else
    refused by name, naming both and the remedy; nothing is copied and the record stands. The remedy — the record moved OUT of the mirror,
    recorded — pushes the published mirror to the new destination; the earlier copy is never deleted."""
    w, doc, rsha = world(tmp_path)
    state, dest = w["state"], tmp_path / "cluster-tmp"
    L.mirror(state, w["vol"].remote, dest_root=dest)
    mdir = mirror_dir(w, rsha)
    cl_bytes = (mdir / MI.CLUSTER_NAME).read_bytes()
    canonical = tmp_path / "cluster-canonical"
    rec = L.RecordingCluster(lambda: mdir)
    with pytest.raises(Refusal, match="names another target than this call's") as e:
        L.mirror(state, w["vol"].remote, dest_root=canonical, cluster_runner=rec)
    msg = str(e.value)
    assert f"cluster_path: recorded '{dest}/{rsha[:12]}', this call '{canonical}/{rsha[:12]}'" in msg and f"ssh_host: recorded 'local:{dest}', this call 'local:{canonical}'" in msg
    assert f"the verified copy sits where cluster.json says, cluster_a local:{dest}:{dest}/{rsha[:12]}" in msg and "a mirror has ONE cluster record" in msg
    assert f"move {mdir}/cluster.json OUT of the mirror directory" in msg and "never deleted" in msg
    assert rec.calls == [] and not canonical.exists() and (mdir / MI.CLUSTER_NAME).read_bytes() == cl_bytes, "nothing copied, the record untouched"
    with pytest.raises(Refusal, match="cluster_label: recorded 'cluster_a', this call 'cluster_b'"):
        L.mirror(state, w["vol"].remote, dest_root=dest, label="cluster_b")
    assert L.mirror(state, w["vol"].remote, dest_root=dest)["reused"] is True, "the recorded target: the no-op re-run"
    # the file phase's cross-check (07 passes the site's primary cluster and <reproduce root>/<model>/07_mirror): the record's target must be the site's
    assert F.rule(w) == 0
    root = F._registry_with_objects(tmp_path / "reg")
    site = {"cluster_label": "cluster_a", "ssh_host": f"local:{dest}", "cluster_root": str(dest), "cluster_dest": str(dest)}
    rc, lines, why = _file(w, root, dry=True, **site)
    assert rc == 0 and why is None, (lines, why)
    moved = {**site, "ssh_host": f"local:{canonical}", "cluster_root": str(canonical), "cluster_dest": str(canonical)}
    rc, lines, why = _file(w, root, dry=True, **moved)
    assert rc == 2 and "names another target than this site's" in why and f"this call '{canonical}/{rsha[:12]}'" in why and "nothing filed against a copy this site does not name" in why
    assert f"cluster_root: recorded '{dest}', this call '{canonical}'" in why, "the root the copy is guarded under is part of the target (round 6)"
    rc, lines, why = _file(w, root, dry=True, **{**site, "cluster_root": str(tmp_path)})
    assert rc == 2 and f"cluster_root: recorded '{dest}', this call '{tmp_path}'" in why and "cluster_path:" not in why, "a root that moved alone is a target change too"
    rc, lines, why = _file(w, root, dry=True, cluster_label="cluster_a")
    assert rc == 2 and "are passed together" in why
    rc, lines, why = _file(w, root, dry=True, **{k: v for k, v in site.items() if k != "cluster_root"})
    assert rc == 2 and "--cluster-root and --cluster-dest are passed together" in why
    # the remedy: the record moved OUT of the mirror (recorded), the re-run pushes the published mirror to the new destination; the earlier copy stands
    (mdir / MI.CLUSTER_NAME).rename(state / "cluster.json.former")
    out = L.mirror(state, w["vol"].remote, dest_root=canonical)
    assert out["reused"] is False and out["cluster_path"] == f"{canonical}/{rsha[:12]}" and out["vol"].gets() == [] and any("finishing the cluster push" in ln for ln in out["log"])
    assert tree_bytes(canonical / rsha[:12]) == tree_bytes(mdir) and (dest / rsha[:12] / MI.SUMS_NAME).is_file(), "the earlier copy is never deleted"
    rc, lines, why = _file(w, root, dry=True, **moved)
    assert rc == 0 and json.loads((state / SF.SPEC_NAME).read_text())["extra"]["mirror"]["cluster_path"] == f"{canonical}/{rsha[:12]}", (lines, why)
    assert json.loads((state / SF.SPEC_NAME).read_text())["extra"]["mirror"]["cluster_root"] == str(canonical)


# -------------------------------------------------------------------------------------------------------------------- the file phase
def _file(w, root, *, dry=False, **kw):
    lines: list = []
    try:
        rc = SF.run(state=w["state"], view=VIEW, key=KEY, registry_root=root, override_id=F.OVERRIDE_ID, cells=["none", "difference_lora"], launcher_cells=list(CELLS),
                    table=None, repo=REPO, dry_run=dry, log=lines.append, **kw)
    except SF.FileRefusal as e:
        return 2, lines, str(e)
    return rc, lines, None


def test_the_file_phase_refuses_without_a_verified_mirror_and_carries_the_pointer_with_one(tmp_path):
    w, doc, rsha = world(tmp_path)
    state = w["state"]
    assert F.rule(w) == 0
    root = F._registry_with_objects(tmp_path / "reg")
    before = (state / SF.SPEC_NAME).exists()
    rc, lines, why = _file(w, root, dry=True)
    assert rc == 2 and f"no verified mirror of receipt {rsha[:12]}" in why and f"no mirror of receipt {rsha[:12]} at {mirror_dir(w, rsha)}" in why
    assert f"`reproduce/07_bcb_bfcl.sh --model {KEY} --view {VIEW} --cells none,difference_lora --phase mirror`" in why and "then file again" in why
    assert (state / SF.SPEC_NAME).exists() == before, "refused before composing"
    # a mirror published but never verified on the cluster: refused the same way, naming the missing record
    mdir = mirror_dir(w, rsha)
    with pytest.raises(Refusal, match="the cluster check"):
        L.mirror(state, w["vol"].remote, dest_root=tmp_path / "cluster", host="cluster-a-login", cluster_runner=L.RecordingCluster(lambda: mdir, check_rc=1))
    rc, lines, why = _file(w, root, dry=True)
    assert rc == 2 and "has no cluster.json: its cluster copy was never verified" in why and "--phase mirror" in why
    # the verified mirror: the dry run composes extra.mirror from the two records, byte-identical across retries; it prints the cluster copy's
    # re-check it would run and makes no ssh call (round 3: the real filing runs it — `ssh` below is the stubbed cluster the file phase re-checks over)
    out = L.mirror(state, w["vol"].remote, dest_root=tmp_path / "cluster", host="cluster-a-login", cluster_runner=L.RecordingCluster(lambda: mdir))
    ssh = L.RecordingCluster(lambda: mdir)
    rc, lines, why = _file(w, root, dry=True, cluster_runner=ssh)
    assert rc == 0 and why is None, (lines, why)
    path = f"{tmp_path / 'cluster'}/{rsha[:12]}"
    assert ssh.calls == [] and any(ln == f"DRY-RUN: ssh cluster-a-login '{MI.copy_check_command(path, root=str(tmp_path / 'cluster'))}'   (the cluster copy re-verified there before a "
                                   "real filing: the guard under the recorded cluster root before the cd — no symlink at the copy or at any component below the root — then "
                                   "MIRROR.sha256, mirror.json and cluster.json read back by digest; the inventory of every entry by type held to MIRROR.sha256 plus the "
                                   "records and mirror.json's directories)" for ln in lines), lines
    spec = json.loads((state / SF.SPEC_NAME).read_text())
    ptr = spec["extra"]["mirror"]
    assert ptr == MI.verified_mirror(state, rsha)
    assert ptr["cluster_path"] == path and ptr["cluster_label"] == "cluster_a" and ptr["ssh_host"] == "cluster-a-login"
    assert ptr["mirror_sha256sum_digest"] == _sha(mdir / MI.SUMS_NAME) and ptr["receipt_sha256"] == rsha == spec["extra"]["receipt_sha256"] and ptr["tree_sha"] == SHA
    assert ptr["files"] == 46 and ptr["bytes"] > 0 and ptr["local_path"] == str(mdir) and ptr["store_volume_paths"] == doc["store_out_dirs"]
    assert any("mirror cluster_a:" in ln and f"MIRROR.sha256 {ptr['mirror_sha256sum_digest'][:12]}" in ln for ln in lines)
    spec_bytes = (state / SF.SPEC_NAME).read_bytes()
    # the real filing: ONE ssh, the cluster copy's re-check in the recorded path (the exact argv), before anything is composed; then the fold and the
    # three-way read-back holding extra.mirror's facts to the verified mirror
    rc, lines, why = _file(w, root, cluster_runner=ssh)
    assert rc == 0 and why is None and (state / SF.SPEC_NAME).read_bytes() == spec_bytes
    assert ssh.calls == [["ssh", "-n", "-o", "BatchMode=yes", "-o", "ConnectTimeout=30", "cluster-a-login", MI.copy_check_command(path, root=str(tmp_path / "cluster"))]], ssh.calls
    assert lines.index(next(ln for ln in lines if ln.startswith("cluster copy re-verified: cluster-a-login:"))) < lines.index(next(ln for ln in lines if ln.startswith("spec: ")))
    rep = F.H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0 and rep.blocked == 0
    rc, lines, why = _file(w, root, cluster_runner=ssh)
    assert rc == 0 and any("PASS:" in ln for ln in lines) and any("extra.mirror names the verified copy (local_path, cluster_label, ssh_host, cluster_root, cluster_path, "
                                                                     "mirror_sha256sum_digest, mirror_json_sha256, receipt_sha256)" in ln for ln in lines), lines
    assert len(ssh.calls) == 2
    q = json.loads(F._cli_out(["--root", root, "query", "--json", f"SELECT extra_json FROM analysis WHERE name = '{spec['name']}'"]))
    assert json.loads(q[0]["extra_json"])["mirror"] == ptr
    # the cluster record removed after the filing: the retry refuses by name (the pointer must still stand for a filing to be composed)
    (mdir / MI.CLUSTER_NAME).rename(mdir / "cluster.json.aside")
    rc, lines, why = _file(w, root, cluster_runner=ssh)
    assert rc == 2 and "has no cluster.json" in why
    (mdir / "cluster.json.aside").rename(mdir / MI.CLUSTER_NAME)
    # a cluster record whose digest is not the published MIRROR.sha256's: not a verified mirror
    cl = json.loads((mdir / MI.CLUSTER_NAME).read_text())
    cl_bytes = (mdir / MI.CLUSTER_NAME).read_bytes()
    (mdir / MI.CLUSTER_NAME).write_text(json.dumps({**cl, "mirror_sha256sum_digest": "1" * 64}))
    rc, lines, why = _file(w, root, cluster_runner=ssh)
    assert rc == 2 and "does not record a verified cluster copy of this mirror" in why
    (mdir / MI.CLUSTER_NAME).write_bytes(cl_bytes)
    # the filing gate re-hashes the copy — a store file truncated, or mirror.json edited, after the cluster check refuses the filing
    victim = next(p for p in mdir.rglob("rows.jsonl"))
    data = victim.read_bytes()
    victim.write_bytes(data[:-1])
    rc, lines, why = _file(w, root, cluster_runner=ssh)
    assert rc == 2 and "the copy changed after it was verified; do not file against it" in why and "--phase mirror" in why
    victim.write_bytes(data)
    mj_bytes = (mdir / MI.MIRROR_NAME).read_bytes()
    (mdir / MI.MIRROR_NAME).write_text(json.dumps({**json.loads(mj_bytes), "app": "another-app"}, indent=1, sort_keys=True) + "\n")
    rc, lines, why = _file(w, root, cluster_runner=ssh)
    assert rc == 2 and "the mirror record changed after the cluster copy was verified" in why
    (mdir / MI.MIRROR_NAME).write_bytes(mj_bytes)
    assert len(ssh.calls) == 2, "every refusal above was the local gate's: no ssh ran"
    rc, lines, why = _file(w, root, cluster_runner=ssh)
    assert rc == 0 and any("PASS:" in ln for ln in lines), (lines, why)


def _predates_findings(lines, name: str, aid: str, where: str, pointer: Path):
    """The round-4 end of the predates branch: the absent extra.mirror named the way a hand-edited header is, then the record, the pointer and what
    lifts it; never a PASS, never "filed, not yet folded"."""
    assert any(ln == f"FINDING: {name}: extra.mirror None != the verified mirror's pointer (no copy recorded)" for ln in lines), lines
    fin = [ln for ln in lines if ln.startswith(f"FINDING: {name} is analysis {aid} ({where}")]
    assert len(fin) == 1 and "filed before the mirror phase: its registry record carries no pointer to the verified copy" in fin[0], (fin, lines)
    assert "a registry reader cannot locate the stores from it" in fin[0] and f"pointer record {pointer}" in fin[0] and "not a PASS" in fin[0], fin
    assert "through a registry mechanism" in fin[0] and "nothing here rewrites a registry file" in fin[0] and f"decisions.md against analysis {aid}" in fin[0], fin
    assert not any(ln.startswith("PASS: ") or "filed, not yet folded" in ln for ln in lines), lines


def test_a_verdict_filed_before_the_mirror_phase_is_reported_by_name_with_the_pointer_and_ends_in_a_finding_not_a_pass(tmp_path):
    """The verdict of the live 2026-10-11 smoke was filed before `--phase mirror` existed, so its row's extra has
    no `mirror`; the duplicate branch wrote nothing and the read-back then flagged the absent digest — exit 3 on every run, no path recording the
    pointer. The policy (stage07_file.py header: A FILING THAT PREDATES THE MIRROR PHASE): the duplicate branch reports it BY NAME, prints the
    verified copy's pointer, writes <state>/07_filing_pointers/<name>.json (one record PER ANALYSIS, written once — the fold run below
    finds it standing and leaves its bytes alone), reads back every OTHER field — and ENDS IN A FINDING (exit 3), not
    a pass: an earlier cut exited 0 while the registry record carried no pointer to the copy, so a registry reader could not locate the stores
    from a filing the phase had called PASS. Pending and folded alike, nothing re-filed; the finding names the absent extra.mirror, the pointer
    record and what lifts it (the row gaining extra.mirror through a registry mechanism); a row that DOES carry a `mirror` with another digest
    is a finding on that digest, as before."""
    w, doc, rsha = world(tmp_path)
    state = w["state"]
    assert F.rule(w) == 0
    root = F._registry_with_objects(tmp_path / "reg")
    dest = tmp_path / "cluster"
    L.mirror(state, w["vol"].remote, dest_root=dest)
    ptr = MI.verified_mirror(state, rsha)
    # the filing as the pre-mirror tool made it: the composed spec with extra.mirror stripped — the SAME analysis_id (name, code_sha and inputs are
    # the identity; extra is not), so the branch's filing is its duplicate at the fold
    rc, lines, why = _file(w, root, dry=True)
    assert rc == 0, (lines, why)
    spec = json.loads((state / SF.SPEC_NAME).read_text())
    old = json.loads(json.dumps(spec)); del old["extra"]["mirror"]
    old_p = tmp_path / "pre_mirror_spec.json"; old_p.write_text(json.dumps(old))
    AN, cli, H, R = SF._registry(REPO)
    assert cli.main(["--root", root, "analysis", "file", "--spec", str(old_p)]) == 0
    aid = AN.analysis_id(AN.normalize_spec(old))
    assert aid == AN.analysis_id(AN.normalize_spec(spec))
    filings = F._filings(root)
    # PENDING (before any ingest): reported by name, the pointer file written, no second filing — and a FINDING (exit 3), the header read back in
    # every other field first
    rc, lines, why = _file(w, root)
    assert rc == 3 and why is None, (lines, why)
    assert any("already filed as" in ln and "(pending" in ln for ln in lines) and F._filings(root) == filings
    named = [ln for ln in lines if ln.startswith("FILED BEFORE THE MIRROR PHASE: ")]
    assert len(named) == 1 and f"{spec['name']} is analysis {aid} (pending" in named[0] and "whose extra carries no `mirror`" in named[0] and "the fold never rewrites" in named[0]
    assert any(ln.strip().startswith(f"the verified copy of receipt {rsha[:12]}: cluster_a local:{dest}:{dest}/{rsha[:12]} (MIRROR.sha256 {ptr['mirror_sha256sum_digest']}, 46 file(s)") for ln in lines), lines
    rec_p = SF.pointer_record_path(state, spec["name"])
    assert rec_p == state / SF.POINTER_DIR / f"{spec['name']}.json"
    assert any(f"pointer record: {rec_p} (written now)" in ln and f"against analysis {aid}" in ln and "ends in a finding, not a pass" in ln for ln in lines), lines
    assert any(ln.startswith("read-back (pending): ") and "extra.mirror" not in ln for ln in lines), "the pending header is read back in every OTHER field first"
    pend_dir = SF.pending_headers(AN, H, R, root, spec["name"])[0]["dir"]
    _predates_findings(lines, spec["name"], aid, f"pending, {pend_dir}", rec_p)
    assert lines.index(next(ln for ln in lines if ln.startswith("read-back (pending): "))) < lines.index(next(ln for ln in lines if ln.startswith("FINDING: ")))
    pf = json.loads(rec_p.read_text())
    assert pf["schema"] == SF.POINTER_SCHEMA == "stage07-filing-pointer/2" and pf["analysis_id"] == aid and pf["name"] == spec["name"] and pf["filed_where"] == "pending" and pf["mirror"] == ptr
    assert pf["status"] == SF.PREDATES_STATUS and pf["filed_utc"] and pf["written_by"] == "reproduce/tools/stage07_file.py"
    rec_bytes = rec_p.read_bytes()
    # FOLDED: the same report; the row read back in every other field, then the same finding — no PASS against a row that carries no pointer
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0 and rep.blocked == 0
    rc, lines, why = _file(w, root)
    assert rc == 3 and why is None, (lines, why)
    assert any(ln.startswith("FILED BEFORE THE MIRROR PHASE: ") and "(folded" in ln for ln in lines)
    _predates_findings(lines, spec["name"], aid, "folded", rec_p)
    assert not any(ln.startswith("FINDING: ") and ("analysis_id " in ln or "code_sha" in ln or "report_sha256" in ln or "headline" in ln) for ln in lines), "every other field is this filing's"
    # the record stands from the first run — the same analysis and copy — and is NOT rewritten (its filed_where stays the first observation)
    assert any(f"pointer record: {rec_p} (stands from an earlier run" in ln for ln in lines), lines
    assert rec_p.read_bytes() == rec_bytes and json.loads(rec_p.read_text())["filed_where"] == "pending" and F._filings(root) == filings
    assert sorted(q.name for q in (state / SF.POINTER_DIR).iterdir()) == [rec_p.name]
    q = json.loads(F._cli_out(["--root", root, "query", "--json", f"SELECT extra_json FROM analysis WHERE name = '{spec['name']}'"]))
    assert len(q) == 1 and "mirror" not in json.loads(q[0]["extra_json"]) and json.loads(q[0]["extra_json"])["receipt_sha256"] == rsha
    # the policy's dry run is unchanged: the spec composed, nothing filed, exit 0 (the registry is not read)
    rc, lines, why = _file(w, root, dry=True)
    assert rc == 0 and why is None and not any("FINDING" in ln for ln in lines), (lines, why)
    # a row that DOES carry a mirror with another digest: a finding (exit 3), as before — the policy covers an ABSENT key only; no pointer file
    w2, doc2, rsha2 = world(tmp_path / "w2", accs=(0.5, 0.62))
    assert rsha2 != rsha and F.rule(w2) == 0
    L.mirror(w2["state"], w2["vol"].remote, dest_root=tmp_path / "cluster2")
    rc, lines, why = _file(w2, root, dry=True)
    assert rc == 0, (lines, why)
    other = json.loads((w2["state"] / SF.SPEC_NAME).read_text()); other["extra"]["mirror"]["mirror_sha256sum_digest"] = "9" * 64
    other_p = tmp_path / "other_digest_spec.json"; other_p.write_text(json.dumps(other))
    assert cli.main(["--root", root, "analysis", "file", "--spec", str(other_p)]) == 0
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0 and rep.blocked == 0
    rc, lines, why = _file(w2, root)
    assert rc == 3 and why is None and any("FINDING" in ln and "extra.mirror.mirror_sha256sum_digest 9999" in ln for ln in lines), lines
    assert not any("FILED BEFORE" in ln for ln in lines) and not (w2["state"] / SF.POINTER_DIR).exists()


def _prefile_without_mirror(w, root, tmp: Path) -> str:
    """A filing as the pre-mirror tool made it — the composed spec with extra.mirror stripped, the SAME analysis_id (extra is not identity);
    returns the name."""
    rc, lines, why = _file(w, root, dry=True)
    assert rc == 0, (lines, why)
    spec = json.loads((w["state"] / SF.SPEC_NAME).read_text())
    old = json.loads(json.dumps(spec)); del old["extra"]["mirror"]
    p = tmp / f"pre_mirror_{spec['name']}.json"; p.write_text(json.dumps(old))
    AN, cli, H, R = SF._registry(REPO)
    assert cli.main(["--root", root, "analysis", "file", "--spec", str(p)]) == 0
    return spec["name"]


def _second_execution(w, r1: str, dest: Path) -> str:
    """A second execution over the same stores in the SAME state root: new attempts, a new run report, a new receipt (the first archived), its
    mirror published beside the first; returns the new receipt's sha256."""
    state = w["state"]
    run2 = f"{SHA[:7]}-33334444"
    for cell, fcid2 in (("none", "fc-none-0011"), (DIFF, "fc-diff-0012")):
        F.put_attempt(w["vol"], cell, fcid2, key=w["keys"][cell], run_id=run2)
    (state / RT.RUN_REPORT_NAME).write_text(F.run_report([F.rec_ok("none", w["keys"]["none"], "fc-none-0011", run_id=run2), F.rec_ok(DIFF, w["keys"][DIFF], "fc-diff-0012", run_id=run2)]))
    F.write_identity(state)
    r2 = F.receipt(w)["receipt_sha256"]
    assert r2 != r1
    L.lay_down_store_extras(w["vol"].remote, json.loads((state / RT.STORE_DIR / RT.RECEIPT_NAME).read_text()))
    L.mirror(state, w["vol"].remote, dest_root=dest)
    assert F.rule(w, sha12=r2[:12]) == 0
    return r2


def test_pointer_records_are_one_per_analysis_written_once_and_a_disagreeing_record_refuses_by_name(tmp_path):
    """Every pre-mirror filing of a model wrote its only durable pointer to ONE <state>/07_filing_pointer.json
    with write_text — the state root is per model, the analysis name per view, overrides id and receipt — so processing a second pre-mirror
    filing overwrote the first analysis's pointer. Pinned: two pre-mirror filings under one state root (two receipts of the same stores, two
    names, two analysis_ids) leave two records under <state>/07_filing_pointers/, the first byte-identical after the second is written; a
    standing record that disagrees on a binding fact (analysis_id, a mirror field) REFUSES by name, lists the fields and is left intact — never
    rewritten — and one that agrees stands without a rewrite. (Each of these runs ends in the predates FINDING, exit 3, the record
    written or standing all the same; a disagreeing record is still a REFUSAL, exit 2, before anything else.)"""
    w, doc, r1 = world(tmp_path)
    state, dest = w["state"], tmp_path / "cluster"
    assert F.rule(w) == 0
    root = F._registry_with_objects(tmp_path / "reg")
    L.mirror(state, w["vol"].remote, dest_root=dest)
    n1 = _prefile_without_mirror(w, root, tmp_path)
    rc, lines, why = _file(w, root)
    assert rc == 3 and why is None and any("FILED BEFORE THE MIRROR PHASE" in ln for ln in lines) and not any(ln.startswith("PASS: ") for ln in lines), (lines, why)
    p1 = SF.pointer_record_path(state, n1)
    assert p1.is_file() and json.loads(p1.read_text())["mirror"]["receipt_sha256"] == r1
    b1 = p1.read_bytes()
    # the second execution: another receipt, another name, another analysis — its own record; the first's bytes untouched
    r2 = _second_execution(w, r1, dest)
    n2 = _prefile_without_mirror(w, root, tmp_path)
    assert n2 != n1 and r2[:12] in n2
    rc, lines, why = _file(w, root)
    assert rc == 3 and why is None and any("FILED BEFORE THE MIRROR PHASE" in ln for ln in lines), (lines, why)
    p2 = SF.pointer_record_path(state, n2)
    assert sorted(q.name for q in (state / SF.POINTER_DIR).iterdir()) == sorted([p1.name, p2.name])
    assert p1.read_bytes() == b1, "the first analysis's pointer record is never overwritten"
    d2 = json.loads(p2.read_text())
    assert d2["name"] == n2 and d2["mirror"]["receipt_sha256"] == r2 and d2["analysis_id"] != json.loads(b1)["analysis_id"]
    # a standing record that agrees: stands, not rewritten (written_utc unchanged); the run ends in the predates finding as before
    b2 = p2.read_bytes()
    rc, lines, why = _file(w, root)
    assert rc == 3 and why is None and p2.read_bytes() == b2 and any(f"pointer record: {p2} (stands from an earlier run" in ln for ln in lines), lines
    # a standing record that disagrees — another analysis_id and another mirror digest — refuses by name, naming both fields; the record is left as it was
    tampered = {**d2, "analysis_id": "11111111-2222-3333-4444-555555555555", "mirror": {**d2["mirror"], "mirror_sha256sum_digest": "7" * 64}}
    tb = json.dumps(tampered, indent=1, sort_keys=True).encode(); p2.write_bytes(tb)
    rc, lines, why = _file(w, root)
    assert rc == 2 and why is not None, lines
    assert str(p2) in why and "written once and never rewritten" in why and "move " in why and "decisions.md" in why
    assert f"analysis_id: recorded '11111111-2222-3333-4444-555555555555', this run '{d2['analysis_id']}'" in why
    assert f"mirror.mirror_sha256sum_digest: recorded '{'7' * 64}', this run '{d2['mirror']['mirror_sha256sum_digest']}'" in why
    assert p2.read_bytes() == tb and p1.read_bytes() == b1
    # a record that disagrees only on a mirror sub-field (the copy's cluster path) refuses the same way — the copy named is not this run's
    moved = {**d2, "mirror": {**d2["mirror"], "cluster_path": "/elsewhere/07_mirror/" + r2[:12]}}
    p2.write_bytes(json.dumps(moved).encode())
    rc, lines, why = _file(w, root)
    assert rc == 2 and f"mirror.cluster_path: recorded '/elsewhere/07_mirror/{r2[:12]}'" in why and "analysis_id:" not in why
    # moved OUT of the directory (the refusal's remedy): the next run writes a fresh record for the name (and ends in the predates finding)
    p2.rename(tmp_path / "moved_aside.json")
    rc, lines, why = _file(w, root)
    assert rc == 3 and why is None and json.loads(p2.read_text())["mirror"] == d2["mirror"] and any(f"pointer record: {p2} (written now)" in ln for ln in lines), (lines, why)


def test_a_pending_header_is_read_back_and_one_that_is_not_this_filings_is_a_finding(tmp_path):
    """read_back saw only the fold, so with no row and ANY pending header for the name the phase printed "filed,
    not yet folded" and exited 0 without reading the header — a pending filing carrying another copy's mirror digest, another receipt or
    another identity passed as complete until the fold. Pinned: a pending header is read back the way a row is (analysis_id against the
    composed spec's, report sha, substrate, cluster_job_ids, the filing's analysis_input records, extra.receipt_sha256, extra.headline, the
    mirror digest) — this filing's own header passes with the read-back line; a header with another mirror digest, another receipt_sha256 or
    another headline is a FINDING (exit 3) naming the header's directory and the field, nothing written; a pre-mirror header (no `mirror`) is
    read back without the digest, as the policy says."""
    w, doc, rsha = world(tmp_path)
    state, dest = w["state"], tmp_path / "cluster"
    assert F.rule(w) == 0
    root = F._registry_with_objects(tmp_path / "reg")
    L.mirror(state, w["vol"].remote, dest_root=dest)
    ptr = MI.verified_mirror(state, rsha)
    AN, cli, H, R = SF._registry(REPO)
    # this filing's own header: read back, named, exit 0
    rc, lines, why = _file(w, root)
    assert rc == 0 and why is None and any("filed: analysis_id" in ln for ln in lines), (lines, why)
    rb = [ln for ln in lines if ln.startswith("read-back (pending): ")]
    assert len(rb) == 1 and "analysis_id == the composed spec's" in rb[0] and "extra.mirror names the verified copy" in rb[0] and "1 pending header(s)" in rb[0]
    assert any("filed, not yet folded" in ln for ln in lines)
    filings = F._filings(root)
    spec = json.loads((state / SF.SPEC_NAME).read_text())
    aid = AN.analysis_id(AN.normalize_spec(spec))
    # the header read: the fields the check compares, as pending_headers reads them from the record files
    pend = SF.pending_headers(AN, H, R, root, spec["name"])
    assert len(pend) == 1 and pend[0]["analysis_id"] == aid and pend[0]["substrate"] == "modal" and pend[0]["cluster_job_ids_json"] == [] and pend[0]["code_sha"] == SHA
    assert pend[0]["inputs"] == sorted(("object", o) for o in spec["inputs"]["objects"]) and pend[0]["extra"]["mirror"] == ptr
    assert SF.check_pending(pend[0], expected_aid=aid, report_sha=pend[0]["report_sha256"], objects=spec["inputs"]["objects"], receipt_sha=rsha,
                            headline=doc["headline"], mirror=ptr, code_sha=SHA) == []
    # a second world, pre-filed by hand with a header that is NOT its filing's: another mirror digest -> a finding before the fold, nothing written
    w2, doc2, r2 = world(tmp_path / "w2", accs=(0.5, 0.62))
    assert r2 != rsha and F.rule(w2) == 0
    L.mirror(w2["state"], w2["vol"].remote, dest_root=tmp_path / "cluster2")
    ptr2 = MI.verified_mirror(w2["state"], r2)
    rc, lines, why = _file(w2, root, dry=True)
    assert rc == 0, (lines, why)
    spec2 = json.loads((w2["state"] / SF.SPEC_NAME).read_text())
    aid2 = AN.analysis_id(AN.normalize_spec(spec2))

    def hand_file(mutate) -> str:
        """Files the mutated spec by hand; returns the NEW filing directory (the set difference — the directories sort by header sha, not by time)."""
        other = json.loads(json.dumps(spec2)); mutate(other)
        before = set(F._filings(root))
        p = tmp_path / f"hand_{len(before)}.json"; p.write_text(json.dumps(other))
        assert cli.main(["--root", root, "analysis", "file", "--spec", str(p)]) == 0
        new = set(F._filings(root)) - before
        assert len(new) == 1, new
        return new.pop()

    def expect_finding(field: str, bad: str, good: str):
        rc, lines, why = _file(w2, root)
        assert rc == 3 and why is None, (lines, why)
        assert any("already filed as" in ln and "(pending" in ln for ln in lines) and not any("filed, not yet folded" in ln for ln in lines), lines
        f = [ln for ln in lines if ln.startswith(f"FINDING: {spec2['name']}: pending header ")]
        assert len(f) == 1 and f"(analysis {aid2}, " in f[0] and f"{field} {bad}" in f[0] and good in f[0], f
        assert any(ln.startswith("FINDING: the pending filing of ") and "nothing here rewrites a record" in ln for ln in lines)
        assert not (w2["state"] / SF.POINTER_DIR).exists()

    def set_digest(o): o["extra"]["mirror"]["mirror_sha256sum_digest"] = "9" * 64
    d = hand_file(set_digest)
    n = len(F._filings(root))
    expect_finding("extra.mirror.mirror_sha256sum_digest", "9" * 64, ptr2["mirror_sha256sum_digest"])
    assert len(F._filings(root)) == n, "a finding files nothing"
    # the mis-filed header moved aside (the hand procedure): another receipt_sha256 in the header -> the same finding on that field
    shutil.move(d, tmp_path / "aside_digest")
    def set_receipt(o): o["extra"]["receipt_sha256"] = "0" * 64
    d = hand_file(set_receipt)
    expect_finding("extra.receipt_sha256", "0" * 64, r2)
    shutil.move(d, tmp_path / "aside_receipt")
    def set_headline(o): o["extra"]["headline"] = {**o["extra"]["headline"], "value": 0.999}
    d = hand_file(set_headline)
    expect_finding("extra.headline", "{", "the receipt's")
    shutil.move(d, tmp_path / "aside_headline")
    # a header whose inputs are not this filing's is ANOTHER analysis_id (inputs are identity): the pre-write check sees a pending filing of the
    # name with the same report sha, and the read-back names the identity and the inputs
    def drop_input(o): o["inputs"]["objects"] = ["none"]
    d = hand_file(drop_input)
    rc, lines, why = _file(w2, root)
    assert rc == 3 and any("analysis_id " in ln and f"!= the composed spec's {aid2}" in ln for ln in lines) and any("the filing's inputs [('object', 'none')] != the object inputs" in ln for ln in lines), lines
    shutil.move(d, tmp_path / "aside_inputs")
    # the filing's own header once more: exit 0 and read back; the fold then holds one row
    rc, lines, why = _file(w2, root)
    assert rc == 0 and any("filed: analysis_id" in ln for ln in lines) and any("read-back (pending): " in ln for ln in lines), (lines, why)
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0 and rep.blocked == 0
    rc, lines, why = _file(w2, root)
    assert rc == 0 and any(f"PASS: {spec2['name']} analysis_id {aid2}" in ln for ln in lines), lines


def test_a_rerun_and_the_file_phase_re_verify_the_cluster_copy_there_and_refuse_a_missing_altered_or_unreachable_one(tmp_path):
    """With cluster.json installed, a re-run re-hashed the grading-host mirror and
    the two records and said "cluster copy on record"; `--phase file` called the same local `verified_mirror` — the cluster copy itself was never
    touched again, so a copy removed, altered or on a host that did not answer was filed as the verified copy. Pinned: the re-run and the real
    filing run `cd <recorded path> && sha256sum -c --quiet MIRROR.sha256 && sha256sum MIRROR.sha256 mirror.json cluster.json && echo inventory: &&
    find . -mindepth 1 -printf "%y %p\\0"` over the RECORDED ssh host (the exact argv) and hold the three digests to the local mirror's; under the local seam the REAL sha256sum runs in the copy, so a
    flipped byte, a deleted file, a replaced record and a removed directory each refuse by name (ALTERED / MISSING), the local mirror and its
    record untouched; ssh exiting 255 is UNREACHABLE and never a pass; the file phase's dry run prints the check and skips it."""
    w, doc, rsha = world(tmp_path)
    state, dest = w["state"], tmp_path / "cluster"
    L.mirror(state, w["vol"].remote, dest_root=dest)
    mdir = mirror_dir(w, rsha)
    copy = dest / rsha[:12]
    path = f"{dest}/{rsha[:12]}"
    before = tree_bytes(mdir)
    assert F.rule(w) == 0
    root = F._registry_with_objects(tmp_path / "reg")
    # the local seam, the real binaries: the re-run passes with one check, no pull, no rsync; the real filing passes too, the check before the spec
    out = L.mirror(state, w["vol"].remote, dest_root=dest)
    assert out["reused"] is True and out["vol"].gets() == [] and out["cluster_check"]["check_command"] == MI.copy_check_command(path, root=str(dest))
    rc, lines, why = _file(w, root)
    assert rc == 0 and why is None and any(ln.startswith(f"cluster copy re-verified: local:{dest}:{path} (sha256sum -c MIRROR.sha256: 46 file(s) OK") for ln in lines), (lines, why)
    assert not any(ln.startswith("DRY-RUN") for ln in lines) and tree_bytes(mdir) == before

    def refuses(pat: str, *, kind: str):
        with pytest.raises(Refusal, match=pat) as e:
            L.mirror(state, w["vol"].remote, dest_root=dest)
        assert f"is {kind}:" in str(e.value) and "the local mirror at" in str(e.value) and "its cluster record is left as it is" in str(e.value), str(e.value)
        assert tree_bytes(mdir) == before, "the grading-host mirror and its record are never touched by a refusal of the cluster copy"
        rc, lines, why = _file(w, root)
        assert rc == 2 and f"is {kind}:" in why and "nothing filed against a cluster copy that does not verify as recorded" in why, (lines, why)
        assert f"move {mdir}/cluster.json OUT of the mirror directory" in why and "--phase mirror` again" in why and "decisions.md" in why
        return why

    # a byte flipped in the COPY's data file: ALTERED (sha256sum -c there did not match)
    victim = next(p for p in copy.rglob("rows.jsonl"))
    data = victim.read_bytes()
    victim.write_bytes(data[:-1] + bytes([data[-1] ^ 1]))
    why = refuses(rf"the cluster copy at cluster_a local:{re.escape(str(dest))}:{re.escape(path)} is ALTERED: `r=.* && cd .* && sha256sum -c --quiet MIRROR\.sha256 .*` exited 1 there", kind="ALTERED")
    assert "did NOT match" in why
    victim.write_bytes(data)
    # a data file deleted from the copy: MISSING
    victim.unlink()
    why = refuses(r"is MISSING: `r=.* && cd .*` exited 1 there", kind="MISSING")
    assert "No such file or directory" in why or "could not be read" in why
    victim.write_bytes(data)
    # the copy's cluster.json replaced (the sums and mirror.json intact): ALTERED on the records' digests, naming the record
    cl_copy = copy / MI.CLUSTER_NAME
    cl_bytes = cl_copy.read_bytes()
    cl_copy.write_text(json.dumps({**json.loads(cl_bytes), "verified_utc": "1999-01-01T00:00:00Z"}))
    why = refuses(r"is ALTERED: its records' digests are not the local mirror's \(cluster\.json there [0-9a-f]{12}, here [0-9a-f]{12}\)", kind="ALTERED")
    cl_copy.write_bytes(cl_bytes)
    # the whole copy gone from the cluster: MISSING; the dry run prints the check it would run and composes from the records as they stand
    copy.rename(tmp_path / "copy.aside")
    refuses(r"is MISSING: `r=.* && cd .*` exited 1 there", kind="MISSING")
    rc, lines, why = _file(w, root, dry=True)
    assert rc == 0 and any(ln.startswith(f"DRY-RUN: ssh local:{dest} '{MI.copy_check_command(path, root=str(dest))}'") and "the guard under the recorded cluster root before the cd" in ln
                           for ln in lines), (lines, why)
    (tmp_path / "copy.aside").rename(copy)
    assert L.mirror(state, w["vol"].remote, dest_root=dest)["reused"] is True and _file(w, root)[0] == 0
    # a real host, the stubbed ssh: the exact argv of the re-check on the re-run and on the filing; rc 255 is UNREACHABLE and never a pass
    w2, doc2, r2 = world(tmp_path / "w2", accs=(0.5, 0.62))
    dest2 = f"/scratch/example/shared/exp/reproduce/{KEY}/07_mirror"
    path2 = f"{dest2}/{r2[:12]}"
    mdir2 = mirror_dir(w2, r2)
    L.mirror(w2["state"], w2["vol"].remote, dest_root=Path(dest2), root=ROOT, host="cluster-a-login", cluster_runner=L.RecordingCluster(lambda: mdir2))
    assert F.rule(w2) == 0
    ssh_prefix = ["ssh", "-n", "-o", "BatchMode=yes", "-o", "ConnectTimeout=30", "cluster-a-login"]
    rec = L.RecordingCluster(lambda: mdir2)
    out2 = L.mirror(w2["state"], w2["vol"].remote, dest_root=Path(dest2), root=ROOT, host="cluster-a-login", cluster_runner=rec)
    assert out2["reused"] is True and rec.calls == [[*ssh_prefix, MI.copy_check_command(path2, root=ROOT)]]
    assert rec.calls[0][-1].startswith(f'r={ROOT}; p="$r"; s=""; for c in {KEY} 07_mirror {r2[:12]}; do ') and rec.calls[0][-1].endswith(
        f" && cd {path2} && sha256sum -c --quiet MIRROR.sha256 && sha256sum MIRROR.sha256 mirror.json cluster.json && echo inventory: && " 'find . -mindepth 1 -printf "%y %p\\0"')
    assert out2["cluster_check"]["ssh_host"] == "cluster-a-login" and out2["cluster_check"]["cluster_path"] == path2
    rec = L.RecordingCluster(lambda: mdir2)
    rc, lines, why = _file(w2, root, cluster_runner=rec)
    assert rc == 0 and why is None and rec.calls == [[*ssh_prefix, MI.copy_check_command(path2, root=ROOT)]], (lines, why, rec.calls)
    filings = F._filings(root)
    down = dict(check_rc=255, check_stderr="ssh: connect to host cluster-a-login port 22: Connection timed out")
    with pytest.raises(Refusal, match=r"the cluster copy at cluster_a cluster-a-login:.* is UNREACHABLE: ssh exited 255 \(ssh: connect to host cluster-a-login port 22: Connection timed out\)") as e:
        L.mirror(w2["state"], w2["vol"].remote, dest_root=Path(dest2), root=ROOT, host="cluster-a-login", cluster_runner=L.RecordingCluster(lambda: mdir2, **down))
    assert "never taken as verified" in str(e.value) and (mdir2 / MI.CLUSTER_NAME).is_file()
    rc, lines, why = _file(w2, root, cluster_runner=L.RecordingCluster(lambda: mdir2, **down))
    assert rc == 2 and "is UNREACHABLE: ssh exited 255" in why and "file again once it does" in why and F._filings(root) == filings, (lines, why)
    # the records read back there with another digest: ALTERED, naming which
    with pytest.raises(Refusal, match=r"is ALTERED: its records' digests are not the local mirror's \(MIRROR\.sha256 there 000000000000, here [0-9a-f]{12}; mirror\.json there 000000000000"):
        L.mirror(w2["state"], w2["vol"].remote, dest_root=Path(dest2), root=ROOT, host="cluster-a-login", cluster_runner=L.RecordingCluster(lambda: mdir2, digest_of=lambda p: "0" * 64))
    # ssh itself not runnable: UNREACHABLE too
    def no_ssh(argv, **kw):
        raise OSError("ssh: command not found")
    with pytest.raises(Refusal, match=r"is UNREACHABLE: ssh could not run \(ssh: command not found\)"):
        MI.verify_cluster_copy(MI.verified_mirror(w2["state"], r2), runner=no_ssh, log=lambda s: None)


def test_the_read_back_holds_the_filed_extra_mirror_to_the_current_verified_mirror_not_the_digest_alone(tmp_path):
    """The read-back compared extra.mirror.mirror_sha256sum_digest alone, so after the documented retry — cluster.json
    moved aside, the mirror re-pushed to a NEW destination (the same bytes, the same MIRROR.sha256) — a file retry reported PASS while the registry
    record named the former cluster path. Pinned: the filed extra.mirror's MIRROR_FACTS (local_path, cluster_label, ssh_host,
    cluster_path, the two digests, receipt_sha256) are held to the current verified mirror's, a pending header and a folded row alike, each
    disagreement a finding by name (exit 3) with the digest agreeing; verified_utc is not a fact — the same copy re-verified at the same
    destination still passes (and a moved state root is the local_path finding of its own test)."""
    w, doc, rsha = world(tmp_path)
    state, former = w["state"], tmp_path / "cluster-former"
    L.mirror(state, w["vol"].remote, dest_root=former)
    mdir = mirror_dir(w, rsha)
    assert F.rule(w) == 0
    root = F._registry_with_objects(tmp_path / "reg")
    site = {"cluster_label": "cluster_a", "ssh_host": f"local:{former}", "cluster_root": str(former), "cluster_dest": str(former)}
    rc, lines, why = _file(w, root, **site)
    assert rc == 0 and why is None, (lines, why)
    spec = json.loads((state / SF.SPEC_NAME).read_text())
    name, filed = spec["name"], spec["extra"]["mirror"]
    assert SF.MIRROR_FACTS == ("local_path", "cluster_label", "ssh_host", "cluster_root", "cluster_path", "mirror_sha256sum_digest", "mirror_json_sha256", "receipt_sha256")
    assert filed["cluster_root"] == str(former), "the root the copy is guarded under is a filed fact (round 6)"
    # the documented retry: the record moved OUT (recorded), the published mirror pushed to the canonical destination — same bytes, same digests
    canonical = tmp_path / "cluster-canonical"
    (mdir / MI.CLUSTER_NAME).rename(state / "cluster.json.former")
    assert L.mirror(state, w["vol"].remote, dest_root=canonical)["cluster_path"] == f"{canonical}/{rsha[:12]}"
    now = MI.verified_mirror(state, rsha)
    assert now["mirror_sha256sum_digest"] == filed["mirror_sha256sum_digest"] and now["mirror_json_sha256"] == filed["mirror_json_sha256"] and now["cluster_path"] != filed["cluster_path"]
    site2 = {**site, "ssh_host": f"local:{canonical}", "cluster_root": str(canonical), "cluster_dest": str(canonical)}
    filings = F._filings(root)
    # PENDING: the header names the former copy — a finding on cluster_path and ssh_host by name, nothing on the digest; exit 3, nothing written
    rc, lines, why = _file(w, root, **site2)
    assert rc == 3 and why is None, (lines, why)
    f = [ln for ln in lines if ln.startswith(f"FINDING: {name}: pending header ")]
    assert any(f"extra.mirror.cluster_path {former}/{rsha[:12]} != the verified mirror's {canonical}/{rsha[:12]}" in ln for ln in f), f
    assert any(f"extra.mirror.ssh_host local:{former} != the verified mirror's local:{canonical}" in ln for ln in f), f
    assert not any("mirror_sha256sum_digest" in ln or "verified_utc" in ln or "local_path" in ln or "analysis_id " in ln for ln in f), f
    assert F._filings(root) == filings and not any("filed, not yet folded" in ln for ln in lines)
    # FOLDED: the row's extra.mirror read back the same way; no PASS
    rep = F.H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0 and rep.blocked == 0
    rc, lines, why = _file(w, root, **site2)
    assert rc == 3 and why is None and not any("PASS" in ln for ln in lines), (lines, why)
    assert any(ln == f"FINDING: {name}: extra.mirror.cluster_path {former}/{rsha[:12]} != the verified mirror's {canonical}/{rsha[:12]}" for ln in lines), lines
    assert any(ln == f"FINDING: {name}: extra.mirror.ssh_host local:{former} != the verified mirror's local:{canonical}" for ln in lines), lines
    assert not any("mirror_sha256sum_digest" in ln for ln in lines if ln.startswith("FINDING"))
    # the former record restored (the earlier copy was never deleted): the row names the current verified copy again -> PASS
    (mdir / MI.CLUSTER_NAME).rename(state / "cluster.json.canonical")
    (state / "cluster.json.former").rename(mdir / MI.CLUSTER_NAME)
    rc, lines, why = _file(w, root, **site)
    assert rc == 0 and any("PASS:" in ln for ln in lines) and any("extra.mirror names the verified copy (local_path, cluster_label, ssh_host, cluster_root, cluster_path" in ln for ln in lines), (lines, why)
    # the SAME copy re-verified at the SAME destination (the record moved aside, the mirror run again with the recorded target): a new verified_utc,
    # every fact the same -> not a disagreement
    (mdir / MI.CLUSTER_NAME).rename(state / "cluster.json.first")
    assert L.mirror(state, w["vol"].remote, dest_root=former)["reused"] is False
    again = MI.verified_mirror(state, rsha)
    assert {k: again[k] for k in SF.MIRROR_FACTS} == {k: filed[k] for k in SF.MIRROR_FACTS}
    rc, lines, why = _file(w, root, **site)
    assert rc == 0 and any("PASS:" in ln for ln in lines), (lines, why)
    # a row whose extra has no `mirror` at all while the pre-write check saw one (a hand-edited header) is named too, not skipped
    assert SF._mirror_diffs({}, again) == ["extra.mirror None != the verified mirror's pointer (no copy recorded)"]
    assert SF._mirror_diffs({"mirror": again}, again) == [] and SF._mirror_diffs({"mirror": {**again, "verified_utc": "x"}}, again) == []
    assert SF._mirror_diffs({"mirror": {**again, "local_path": "/elsewhere"}}, again) == [f"extra.mirror.local_path /elsewhere != the verified mirror's {mdir}"], "round 5: a fact"


def test_a_moved_state_root_is_a_finding_on_the_filed_local_path_pending_and_folded(tmp_path):
    """The read-back held extra.mirror's cluster facts and digests to the current verified mirror but not
    `local_path` (bookkeeping only), so after the grading host's state root MOVED — the mirror, its records and the receipt verifying
    at the new path, the cluster copy where it was — a file retry read back PASS while the registry row still named a local copy that verified
    nowhere (the pointer record's _pointer_diffs already held mirror.local_path as binding). Pinned: local_path is a MIRROR_FACT; a pending
    header and a folded row naming the former local path are each a finding by name (exit 3) on local_path ALONE — the identity, the cluster
    facts and the digests agree — and nothing is written; the state root back where the row says, the same row reads back PASS."""
    w, doc, rsha = world(tmp_path)
    state, dest = w["state"], tmp_path / "cluster"
    L.mirror(state, w["vol"].remote, dest_root=dest)
    mdir = mirror_dir(w, rsha)
    assert F.rule(w) == 0
    root = F._registry_with_objects(tmp_path / "reg")
    rc, lines, why = _file(w, root)
    assert rc == 0 and why is None and any("filed, not yet folded" in ln for ln in lines), (lines, why)
    spec = json.loads((state / SF.SPEC_NAME).read_text())
    name, filed = spec["name"], spec["extra"]["mirror"]
    assert "local_path" in SF.MIRROR_FACTS and filed["local_path"] == str(mdir)
    # the state root moved, the old one gone: the mirror and the receipt verify at the new path, the cluster copy is where it was
    moved = tmp_path / "moved" / KEY
    moved.parent.mkdir()
    shutil.move(str(state), str(moved))
    assert not state.exists()
    w2 = {**w, "state": moved}
    mdir2 = mirror_dir(w2, rsha)
    now = MI.verified_mirror(moved, rsha)
    assert now["local_path"] == str(mdir2) != filed["local_path"]
    assert {k: now[k] for k in SF.MIRROR_FACTS if k != "local_path"} == {k: filed[k] for k in SF.MIRROR_FACTS if k != "local_path"}
    filings = F._filings(root)
    # PENDING: the header names the former local copy — a finding on local_path alone; exit 3, nothing written
    rc, lines, why = _file(w2, root)
    assert rc == 3 and why is None, (lines, why)
    f = [ln for ln in lines if ln.startswith(f"FINDING: {name}: pending header ")]
    assert any(f"extra.mirror.local_path {mdir} != the verified mirror's {mdir2}" in ln for ln in f), f
    assert not any("cluster_path" in ln or "ssh_host" in ln or "cluster_label" in ln or "mirror_sha256sum_digest" in ln or "mirror_json_sha256" in ln or "receipt_sha256" in ln
                   or "verified_utc" in ln or "analysis_id " in ln or "code_sha " in ln for ln in f), f
    assert F._filings(root) == filings and not any("filed, not yet folded" in ln or "PASS" in ln for ln in lines)
    # FOLDED: the row's extra.mirror read back the same way; no PASS
    rep = F.H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0 and rep.blocked == 0
    rc, lines, why = _file(w2, root)
    assert rc == 3 and why is None and not any("PASS" in ln for ln in lines), (lines, why)
    assert any(ln == f"FINDING: {name}: extra.mirror.local_path {mdir} != the verified mirror's {mdir2}" for ln in lines), lines
    assert [ln for ln in lines if ln.startswith("FINDING")] == [f"FINDING: {name}: extra.mirror.local_path {mdir} != the verified mirror's {mdir2}"], lines
    # the state root back where the row says: PASS, the facts line naming local_path
    shutil.move(str(moved), str(state))
    rc, lines, why = _file(w, root)
    assert rc == 0 and any("PASS:" in ln for ln in lines) and any("extra.mirror names the verified copy (local_path, cluster_label" in ln for ln in lines), (lines, why)


def test_a_same_name_filing_at_another_code_sha_is_another_analysis_and_a_finding_not_a_duplicate_pass(tmp_path):
    """A same-name, same-report filing with identical inputs but ANOTHER code_sha is another analysis_id; the pre-write
    check (name + report sha) took it for this filing — "a duplicate; nothing written" — and the folded read-back never compared code_sha or
    analysis_id, so the wrong tree read back PASS. Pinned: the folded row's analysis_id is compared with analysis_id(normalize_spec(spec)) and its
    code_sha with this filing's, as check_pending does for a pending header; both are findings by name (exit 3), pending and folded alike; the
    identity finding is named at the pre-write check too, and such a row gets no pointer record even when its extra carries no `mirror`."""
    w, doc, rsha = world(tmp_path)
    state, dest = w["state"], tmp_path / "cluster"
    L.mirror(state, w["vol"].remote, dest_root=dest)
    assert F.rule(w) == 0
    root = F._registry_with_objects(tmp_path / "reg")
    rc, lines, why = _file(w, root, dry=True)
    assert rc == 0, (lines, why)
    spec = json.loads((state / SF.SPEC_NAME).read_text())
    name = spec["name"]
    AN, cli, H, R = SF._registry(REPO)
    aid = AN.analysis_id(AN.normalize_spec(spec))
    other_tree = "f" * 40
    wrong = json.loads(json.dumps(spec)); wrong["code_sha"] = other_tree; wrong["code_ref"] = f"demo-import-smoke @ {other_tree[:7]}"
    wrong_p = tmp_path / "wrong_tree.json"; wrong_p.write_text(json.dumps(wrong))
    assert cli.main(["--root", root, "analysis", "file", "--spec", str(wrong_p)]) == 0
    wrong_aid = AN.analysis_id(AN.normalize_spec(wrong))
    assert wrong_aid != aid and wrong["inputs"] == spec["inputs"] and wrong["report"] == spec["report"]
    filings = F._filings(root)
    # PENDING: the pre-write check sees a same-name, same-report filing (nothing written), names the identity, and the header's read-back names both fields
    rc, lines, why = _file(w, root)
    assert rc == 3 and why is None, (lines, why)
    assert any("already filed as" in ln and "a duplicate; nothing written" in ln for ln in lines), lines
    assert any(ln.startswith("FINDING: ") and f"{name} is filed as analysis {wrong_aid} (pending), not this filing's identity {aid}" in ln and f"this filing's code_sha {SHA}" in ln
               and "no pointer record is written for it" in ln for ln in lines), lines
    hdr = [ln for ln in lines if ln.startswith(f"FINDING: {name}: pending header ")]
    assert any(f"analysis_id {wrong_aid} != the composed spec's {aid} (name, code_sha and inputs are the identity)" in ln for ln in hdr), hdr
    assert any(f"code_sha {other_tree} != this filing's {SHA} (the receipt's tree)" in ln for ln in hdr), hdr
    assert not any("filed, not yet folded" in ln or "PASS" in ln for ln in lines) and F._filings(root) == filings and not (state / SF.POINTER_DIR).exists()
    # FOLDED: the row's identity read back — the same two findings; no PASS
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0 and rep.blocked == 0
    rows = SF.folded_rows(cli, root, name)
    assert len(rows) == 1 and rows[0]["analysis_id"] == wrong_aid and rows[0]["code_sha"] == other_tree and rows[0]["report_sha256"] == _sha(Path(spec["report"]["uri"]))
    rc, lines, why = _file(w, root)
    assert rc == 3 and why is None and not any("PASS" in ln for ln in lines), (lines, why)
    assert any(ln == f"FINDING: {name}: analysis_id {wrong_aid} != the composed spec's {aid} (name, code_sha and inputs are the identity)" for ln in lines), lines
    assert any(ln == f"FINDING: {name}: code_sha {other_tree} != this filing's {SHA} (the receipt's tree)" for ln in lines), lines
    assert any(f"is filed as analysis {wrong_aid} (folded), not this filing's identity {aid}" in ln for ln in lines) and not (state / SF.POINTER_DIR).exists()
    # the wrong tree filed WITHOUT a mirror (as a pre-mirror tool at another tree would have): no pointer record is bound to it, the finding stands
    w2, doc2, r2 = world(tmp_path / "w2", accs=(0.5, 0.62))
    L.mirror(w2["state"], w2["vol"].remote, dest_root=tmp_path / "cluster2")
    assert F.rule(w2) == 0
    rc, lines, why = _file(w2, root, dry=True)
    assert rc == 0, (lines, why)
    spec2 = json.loads((w2["state"] / SF.SPEC_NAME).read_text())
    old = json.loads(json.dumps(spec2)); old["code_sha"] = other_tree; old["code_ref"] = f"demo-import-smoke @ {other_tree[:7]}"; del old["extra"]["mirror"]
    old_p = tmp_path / "wrong_tree_no_mirror.json"; old_p.write_text(json.dumps(old))
    before = set(F._filings(root))
    assert cli.main(["--root", root, "analysis", "file", "--spec", str(old_p)]) == 0
    wrong_dirs = set(F._filings(root)) - before
    assert len(wrong_dirs) == 1
    rc, lines, why = _file(w2, root)
    assert rc == 3 and why is None and not (w2["state"] / SF.POINTER_DIR).exists() and not any("FILED BEFORE THE MIRROR PHASE" in ln for ln in lines), (lines, why)
    assert any("no pointer record is written for it" in ln for ln in lines) and any(f"code_sha {other_tree} != this filing's {SHA}" in ln for ln in lines), lines
    # the right tree, the same name: a pre-mirror filing of THIS identity still takes the pointer path (the policy is unchanged for the real case;
    # round 4: it ends in the predates finding, exit 3, the identity findings absent)
    shutil.move(wrong_dirs.pop(), tmp_path / "aside_wrong_tree_no_mirror")
    good_old = json.loads(json.dumps(spec2)); del good_old["extra"]["mirror"]
    good_p = tmp_path / "right_tree_no_mirror.json"; good_p.write_text(json.dumps(good_old))
    assert cli.main(["--root", root, "analysis", "file", "--spec", str(good_p)]) == 0
    rc, lines, why = _file(w2, root)
    assert rc == 3 and why is None and any("FILED BEFORE THE MIRROR PHASE" in ln for ln in lines) and SF.pointer_record_path(w2["state"], spec2["name"]).is_file(), (lines, why)
    assert any("filed before the mirror phase: its registry record carries no pointer" in ln for ln in lines) and not any("analysis_id " in ln and "!=" in ln for ln in lines), lines


# ---------------------------------------------------------------------------------------------------------------- name-agnosticism
def test_the_mirror_names_no_control_suite_or_marker_and_follows_an_arbitrary_receipt(tmp_path):
    src = (REPO / "reproduce" / "tools" / "stage07_mirror.py").read_text(encoding="utf-8")
    for word in ("bigcodebench", "bfcl", "_200", "inspect/", "controls_summary", "HEADLINE"):
        assert word not in src, word
    # a receipt whose cells, keys and store roots are invented: the mirror follows it as it does the fixture's
    tree = "a" * 40
    cell, key = "zeta__cell_q", "zeta__cell_q__ctrl__t9__k0__00000000__11111111"
    out = f"/vol/tinycap/gen/native/some_model/{cell}/thinking_off"
    root = L.vrel(out)
    remote = {f"{root}/summary/zz.json": b'{"n": 7}\n', f"{root}/cells/{key}/manifest.json": json.dumps(L.manifest_doc(cell, tree)).encode(), f"{root}/RESULT": L.result_text(tree),
              f"{root}/cells/{key}/big.bin": bytes(range(256)) * 9}
    receipt = {"schema": "stage07-receipt/9", "tree_sha": tree, "volume": "vol-x", "modal_environment": "env-y", "app": "app-z", "cells": [cell],
               "recs": {cell: {"out": out, "done_in": {"ctrl": key}}}, "pulled": {f"{out}/summary/zz.json": {"sha256": hashlib.sha256(b'{"n": 7}\n').hexdigest(), "bytes": 9}}}
    state = tmp_path / "state"
    (state / RT.STORE_DIR).mkdir(parents=True)
    rpath = state / RT.STORE_DIR / RT.RECEIPT_NAME
    rpath.write_text(json.dumps(receipt))
    rsha = _sha(rpath)
    m = L.mirror(state, remote, dest_root=tmp_path / "cluster", volume="vol-x", environment="env-y")
    mdir = state / MI.MIRROR_DIR / rsha[:12]
    assert m["reused"] is False and tree_bytes(mdir)[f"{cell}/cells/{key}/big.bin"] == bytes(range(256)) * 9 and json.loads((mdir / MI.MIRROR_NAME).read_text())["cells"][cell]["store_keys"] == [key]
    assert m["vol"].gets()[0][1:] == ["volume", "get", "--env", "env-y", "--force", "vol-x", f"/{root}", str(state / MI.MIRROR_BUILD_DIR / rsha[:12] / MI.PULL_STAGING / cell)]
    assert all(c[1:4] == ["volume", "ls", "--env"] and c[4] == "env-y" for c in m["vol"].calls if c[1:3] == ["volume", "ls"]), "every volume command scoped to the receipt's environment"


def test_volume_cli_get_tree_needs_an_existing_destination_and_lays_the_folder_under_it(tmp_path):
    remote = {"a/b/c.txt": b"c", "a/b/d/e.txt": b"e", "a/f.txt": b"f"}
    vol = L.FakeModalTree(remote)
    cli = L.MI.M.VolumeCLI(Path("/nonexistent/modal"), "twosi-fits", runner=vol, log=lambda s: None)
    with pytest.raises(Refusal, match="must be an existing directory"):
        cli.get_tree("/a/b", tmp_path / "missing")
    assert vol.gets() == [], "refused before the CLI ran"
    got = cli.get_tree("/a/b", tmp_path)
    assert got == tmp_path / "b" and (got / "c.txt").read_bytes() == b"c" and (got / "d" / "e.txt").read_bytes() == b"e" and not (tmp_path / "f.txt").exists()
    assert vol.gets()[-1][1:] == ["volume", "get", "--force", "twosi-fits", "/a/b", str(tmp_path)]
    with pytest.raises(Refusal, match="modal volume get /a/nope failed"):
        cli.get_tree("/a/nope", tmp_path)
    # the existing single-file get is unchanged
    assert cli.get("/a/f.txt", tmp_path).read_bytes() == b"f"


# --------------------------------------------------------------------------------------------- the dose views: per-view state, per-view copies
def _dose_world(tmp: Path, battery_state: Path, dose_view: str, monkeypatch, *, accs=(0.47, 0.6)):
    """A receipted DOSE-VIEW world laid out as 07 lays it out (stage07-full-sets, 2026-10-12): the state under <model state>/07_views/<view>/
    (07's STATE07), the launch identity naming the view, the view's own registry row (`<key>-<view>`, lib/models.py launcher_slug), its app and
    the committed dose copy the importer plans for its object cell; the stores on a volume of their own under the slug's out dirs. The filing
    module's fixtures spell the out dir, the REC's model and the pointer's model from F.KEY, so the slug stands in for it while they are built.
    Returns (w, the receipt doc, its sha256)."""
    slug = F.MT.launcher_slug(KEY, dose_view)
    app = f"tinycap-{SHA[:7]}-{slug}"
    state = battery_state / "07_views" / dose_view
    state.mkdir(parents=True)
    plan = SF._importer().Plan(KEY, views=[dose_view], table=None, repo=REPO)
    oid = next(r["object_sha256_expected"] for r in plan.rows if r["view_cell"] == "difference_lora")
    assert oid != F.DIFF_ID, "the dose copy is an object of its own, not the battery's difference_lora"
    vol = F.FakeModal()
    k_none, k_diff = F.key_of("none", "c3"), F.key_of(DIFF, "d4")
    f_none, f_diff = "fc-none-0101", "fc-diff-0102"
    with monkeypatch.context() as mp:
        mp.setattr(F, "KEY", slug); mp.setattr(F, "APP", app)
        F.lay_down(vol, "none", k_none, accs, fcids=[(f_none, F.CONTROLS)])
        F.lay_down(vol, DIFF, k_diff, (0.40, 0.57), fcids=[(f_diff, F.CONTROLS)])
        recs = [F.rec_ok("none", k_none, f_none), F.rec_ok(DIFF, k_diff, f_diff)]
        (state / RT.RUN_REPORT_NAME).write_text(F.run_report(recs))
        ident = F.identity_doc(slug=slug, app=app, objects={DIFF: oid})
    ident["view"], ident["model_key"] = dose_view, KEY
    (state / RT.RUN_IDENTITY_NAME).write_text(json.dumps(ident, indent=1, sort_keys=True) + "\n")
    lines: list = []
    RT.run(state=state, view=dose_view, sha=SHA, cells=list(CELLS), view_cells=["none", "difference_lora"], override_id=F.OVERRIDE_ID, controls=list(F.CONTROLS),
           app=app, run_report=state / RT.RUN_REPORT_NAME, run_identity=None, volume="twosi-fits", modal_bin=Path("/nonexistent/modal"), dry_run=False,
           environment=None, runner=vol, log=lines.append)
    rpath = state / RT.STORE_DIR / RT.RECEIPT_NAME
    doc = json.loads(rpath.read_text())
    L.lay_down_store_extras(vol.remote, doc)
    return {"state": state, "vol": vol, "keys": {"none": k_none, DIFF: k_diff}, "view": dose_view, "slug": slug, "app": app, "object_id": oid}, doc, _sha(rpath)


def _rule_view(w, view: str) -> int:
    """stage.py verdict as the receipt phase prints it, for the view, under the view's state root (07: `--state-root $STATE07`)."""
    rec = json.loads((w["state"] / RT.STORE_DIR / RT.RECEIPT_NAME).read_text())
    return F.ST.main(["verdict", "--state-root", str(w["state"]), "--family", f"bcb_bfcl/{view}", "--headline", repr(rec["headline"]["value"]), "--range", "0.3,0.6",
                      "--run-ids", ",".join(rec["store_cell_keys"]), "--note", f"headline: none {F.TASK_BCB} acc; receipt {F.receipt_sha(w)[:12]}"])


def _file_view(w, root, view: str, *, dest: Path, dry=False):
    """The file phase as 07 calls it for a view: the VIEW's state root and the view's cluster target (`--cluster-dest <reproduce root>/<model>[/07_views/<view>]/07_mirror`)
    ride along as cross-checks; the cluster copy is re-verified over the `local:<dest>` seam for real."""
    lines: list = []
    try:
        rc = SF.run(state=w["state"], view=view, key=KEY, registry_root=root, override_id=F.OVERRIDE_ID, cells=["none", "difference_lora"], launcher_cells=list(CELLS),
                    table=None, repo=REPO, dry_run=dry, log=lines.append, cluster_label="cluster_a", ssh_host=f"local:{dest}", cluster_root=str(dest), cluster_dest=str(dest))
    except SF.FileRefusal as e:
        return 2, lines, str(e)
    return rc, lines, None


def _prefile_view(w, root, view: str, *, dest: Path, tmp: Path) -> str:
    """A filing as the pre-mirror tool made it, for the view — the composed spec with extra.mirror stripped (the same analysis_id); returns the name."""
    rc, lines, why = _file_view(w, root, view, dest=dest, dry=True)
    assert rc == 0 and why is None, (lines, why)
    spec = json.loads((w["state"] / SF.SPEC_NAME).read_text())
    old = json.loads(json.dumps(spec)); del old["extra"]["mirror"]
    p = tmp / f"pre_mirror_{spec['name']}.json"; p.write_text(json.dumps(old))
    AN, cli, H, R = SF._registry(REPO)
    assert cli.main(["--root", root, "analysis", "file", "--spec", str(p)]) == 0
    return spec["name"]


def test_a_dose_views_mirror_and_pointer_records_land_under_the_views_state_and_never_collide_with_the_batterys(tmp_path, monkeypatch):
    """stage07-full-sets x stage07-mirror (2026-10-12): 07 runs the dose views as launcher rows of their own and keeps every per-execution file
    of a dose view under <state>/07_views/<view>/ (its STATE07); the mirror and the file phase FOLLOW THAT STATE — 07 hands both tools the
    view's state root and a cluster destination of the same layout (<reproduce root>/<model>/07_views/<view>/07_mirror; the battery's paths
    unchanged). Pinned over two receipted worlds of one model — the battery's under <state>/ and a dose view's under <state>/07_views/<view>/,
    the dose view's receipt naming its view, its row (`<key>-<view>`), its app and the committed dose copy — each mirrored to its view's cluster
    directory and filed with its view's state root and target: the receipt each mirror reads, the 07_mirror/<sha12>/ it publishes, its
    cluster.json and cluster copy, the spec and the 07_filing_pointers/<name>.json record of a filing that predates the mirror phase all land
    under the VIEW's directories, named for the view's analysis; nothing of one view lands in the other's directories (nor would it if the two
    receipts shared a sha12: the per-view directory, not the receipt, tells the views apart); and the file phase refuses by name a receipt read
    from another view's directory (the wrong view's state root has no verdict; a receipt describing another view is named as such) — it never
    files the battery's execution as the dose view's or the reverse."""
    # the battery: receipt, mirror and filing under <state>/ and <reproduce root>/<model>/07_mirror/, every path as before
    bw, bdoc, brsha = world(tmp_path)
    bstate = bw["state"]
    croot = tmp_path / "cluster" / KEY                       # the model's reproduce directory on the primary cluster (repro_cluster_state_root)
    bdest = croot / "07_mirror"
    L.mirror(bstate, bw["vol"].remote, dest_root=bdest)
    assert F.rule(bw) == 0
    root = F._registry_with_objects(tmp_path / "reg")
    # the dose view: receipt under <state>/07_views/<view>/ (07's STATE07), the cluster copy under <reproduce root>/<model>/07_views/<view>/07_mirror/
    m = F.MT.model(KEY)
    dose_view = next(v for v, vd in m["views"].items() if "dose" in (vd or {}) and list(vd["cells"]) == ["none", "difference_lora"])
    dw, ddoc, drsha = _dose_world(tmp_path, bstate, dose_view, monkeypatch)
    dstate, ddest = dw["state"], croot / "07_views" / dose_view / "07_mirror"
    assert dstate == bstate / "07_views" / dose_view and drsha != brsha
    assert ddoc["view"] == dose_view and ddoc["slug"] == f"{KEY}-{dose_view}" == dw["slug"] and ddoc["app"] == dw["app"] and ddoc["objects"] == {DIFF: dw["object_id"]}
    assert bdoc["view"] == VIEW and bdoc["slug"] == KEY and bdoc["objects"] == {DIFF: F.DIFF_ID}
    assert ddoc["store_out_dirs"] != bdoc["store_out_dirs"] and all(f"/{dw['slug']}/" in o for o in ddoc["store_out_dirs"])      # the slug's own stores
    out = L.mirror(dstate, dw["vol"].remote, dest_root=ddest)
    assert _rule_view(dw, dose_view) == 0
    # THE MIRRORS: each published under its view's state directory and pushed under its view's cluster directory; neither in the other's
    bmir, dmir = bstate / MI.MIRROR_DIR / brsha[:12], dstate / MI.MIRROR_DIR / drsha[:12]
    assert bmir.is_dir() and dmir.is_dir() and out["mirror"] == str(dmir) and out["cluster_path"] == f"{ddest}/{drsha[:12]}"
    assert not (bstate / MI.MIRROR_DIR / drsha[:12]).exists() and not (dstate / MI.MIRROR_DIR / brsha[:12]).exists()
    assert not (bstate / MI.MIRROR_BUILD_DIR).exists() and not (dstate / MI.MIRROR_BUILD_DIR).exists()
    bptr, dptr = MI.verified_mirror(bstate, brsha), MI.verified_mirror(dstate, drsha)
    assert bptr["local_path"] == str(bmir) and dptr["local_path"] == str(dmir) and dptr["receipt_sha256"] == drsha and bptr["receipt_sha256"] == brsha
    assert bptr["cluster_path"] == f"{bdest}/{brsha[:12]}" and dptr["cluster_path"] == f"{ddest}/{drsha[:12]}"
    assert sorted(p.name for p in bdest.iterdir()) == [brsha[:12]] and sorted(p.name for p in ddest.iterdir()) == [drsha[:12]]
    assert sorted(p.name for p in croot.iterdir()) == ["07_mirror", "07_views"] and sorted(p.name for p in (croot / "07_views").iterdir()) == [dose_view]
    assert json.loads((dmir / MI.CLUSTER_NAME).read_text())["cluster_path"] == f"{ddest}/{drsha[:12]}" and json.loads((dmir / MI.MIRROR_NAME).read_text())["receipt_path"] == str(dstate / RT.STORE_DIR / RT.RECEIPT_NAME)
    assert tree_bytes(dmir) == tree_bytes(Path(dptr["cluster_path"])) and tree_bytes(bmir) == tree_bytes(Path(bptr["cluster_path"])) and tree_bytes(dmir) != tree_bytes(bmir)
    # a re-run of each mirror verifies its own published copy and cluster copy (one cluster record per mirror, the view's target) and pulls nothing
    for st, rem, dest in ((bstate, bw["vol"].remote, bdest), (dstate, dw["vol"].remote, ddest)):
        again = L.mirror(st, rem, dest_root=dest)
        assert again["reused"] is True and again["cluster_path"] == f"{dest}/{again['receipt_sha256'][:12]}" and not again["vol"].gets()
    # THE FILINGS, each with its view's state root and cluster target as 07 passes them: the dry run composes each view's spec under its own state,
    # named for its view, pointing at its own mirror; a filing under the OTHER view's target refuses by name (one cluster record per mirror)
    for w_, view_, dest_, ptr_, rsha_ in ((bw, VIEW, bdest, bptr, brsha), (dw, dose_view, ddest, dptr, drsha)):
        rc, lines, why = _file_view(w_, root, view_, dest=dest_, dry=True)
        assert rc == 0 and why is None, (lines, why)
        spec = json.loads((w_["state"] / SF.SPEC_NAME).read_text())
        assert spec["name"] == f"{SF.NAME_PREFIX}_{view_}_{F.OVERRIDE_ID}_x{rsha_[:12]}" and spec["extra"]["mirror"] == ptr_
        assert spec["report"]["uri"] == str(w_["state"] / "gate" / SF.FAMILY_PREFIX / f"{view_}.json") and spec["extra"]["receipt_sha256"] == rsha_
    bspec, dspec = json.loads((bstate / SF.SPEC_NAME).read_text()), json.loads((dstate / SF.SPEC_NAME).read_text())
    assert dspec["extra"]["launcher_slug"] == dw["slug"] and bspec["extra"]["launcher_slug"] == KEY and dspec["inputs"]["objects"] == sorted([dw["object_id"], "none"])
    assert bspec["inputs"]["objects"] == sorted([F.DIFF_ID, "none"]) and dspec["name"] != bspec["name"]
    rc, lines, why = _file_view(dw, root, dose_view, dest=bdest, dry=True)
    assert rc == 2 and "names another target than this site's" in why and f"cluster_path: recorded '{ddest}/{drsha[:12]}', this call '{bdest}/{drsha[:12]}'" in why, why
    # THE POINTER RECORDS: a filing of each view that predates the mirror phase — the record lands under the VIEW's 07_filing_pointers/, named for
    # the view's analysis, binding it to the view's own copy; each view's pointer directory holds its record alone
    bname = _prefile_view(bw, root, VIEW, dest=bdest, tmp=tmp_path)
    dname = _prefile_view(dw, root, dose_view, dest=ddest, tmp=tmp_path)
    assert bname != dname and f"_{VIEW}_" in bname and f"_{dose_view}_" in dname
    for w_, view_, dest_, name_, ptr_ in ((bw, VIEW, bdest, bname, bptr), (dw, dose_view, ddest, dname, dptr)):
        rc, lines, why = _file_view(w_, root, view_, dest=dest_)
        assert rc == 3 and why is None and any(ln.startswith("FILED BEFORE THE MIRROR PHASE: ") and f"{name_} is analysis" in ln for ln in lines), (lines, why)
        rec_p = SF.pointer_record_path(w_["state"], name_)
        assert rec_p == w_["state"] / SF.POINTER_DIR / f"{name_}.json" and rec_p.is_file()
        pf = json.loads(rec_p.read_text())
        assert pf["name"] == name_ and pf["mirror"] == ptr_ and pf["mirror"]["cluster_path"] == f"{dest_}/{ptr_['receipt_sha256'][:12]}"
        assert sorted(q.name for q in (w_["state"] / SF.POINTER_DIR).iterdir()) == [f"{name_}.json"]
    assert not (bstate / SF.POINTER_DIR / f"{dname}.json").exists() and not (dstate / SF.POINTER_DIR / f"{bname}.json").exists()
    # nothing of the dose view under the battery's own directories (07_views/ aside), nothing of the battery under the dose view's
    battery_files = {p.relative_to(bstate).as_posix() for p in bstate.rglob("*") if p.is_file() and not p.relative_to(bstate).as_posix().startswith("07_views/")}
    dose_files = {p.relative_to(dstate).as_posix() for p in dstate.rglob("*") if p.is_file()}
    assert not any(drsha[:12] in f or dname in f for f in battery_files) and not any(brsha[:12] in f or bname in f for f in dose_files)
    assert {"07_store/receipt.json", f"07_mirror/{drsha[:12]}/{MI.SUMS_NAME}", f"07_mirror/{drsha[:12]}/{MI.CLUSTER_NAME}", SF.SPEC_NAME, f"{SF.POINTER_DIR}/{dname}.json",
            f"gate/{SF.FAMILY_PREFIX}/{dose_view}.json"} <= dose_files
    # THE FILE PHASE FOLLOWS THE VIEW: the battery filed from the dose view's state root finds no battery verdict there; a verdict planted under the
    # battery's name beside the dose view's receipt is refused on the receipt's own `view` before any other field is compared — never filed
    rc, lines, why = _file_view(dw, root, VIEW, dest=bdest)
    assert rc == 2 and why.startswith(f"no verdict at {dstate / 'gate' / SF.FAMILY_PREFIX / (VIEW + '.json')}"), why
    rc, lines, why = _file_view(bw, root, dose_view, dest=ddest)
    assert rc == 2 and why.startswith(f"no verdict at {bstate / 'gate' / SF.FAMILY_PREFIX / (dose_view + '.json')}"), why
    planted = dstate / "gate" / SF.FAMILY_PREFIX / f"{VIEW}.json"
    shutil.copy2(dstate / "gate" / SF.FAMILY_PREFIX / f"{dose_view}.json", planted)
    filings = F._filings(root)
    rc, lines, why = _file_view(dw, root, VIEW, dest=bdest)
    assert rc == 2 and why.startswith(f"the receipt describes view {dose_view!r}, not {VIEW!r} — it is another view's execution") and "07_views/<view>/" in why, why
    assert F._filings(root) == filings
    planted.unlink()


# ----------------------------------------------------------------------------------------------------- round 6: the guarded destination
def _run_mirror(state: Path, remote: dict, *, dest: Path, root: Path, host: str | None = None, runner=None):
    return L.mirror(state, remote, dest_root=dest, root=root, host=host, cluster_runner=runner or MI.sh_runner)


def test_the_guard_helpers_name_a_symlink_or_a_redirected_destination_and_lead_the_mkdir_and_every_check(tmp_path):
    """The helpers: `dest_components` walks the names between the site's reproduce root and the destination
    (nothing outside the root, no `.`/`..`, no empty component); `guard_command` is the walk with `test -L` on every component, exit 61 naming
    the link and its target, then the resolved-path comparison (realpath of the destination == realpath of the root + the literal relative
    path, exit 62) when the destination exists; `prepare_command` is the guard, `mkdir -p`, the comparison again; both check commands lead
    with the guard BEFORE the `cd`; and `guard_verdict` reads the guard's own exit codes and nothing else."""
    path = f"{DEST}/abc"
    assert MI.dest_components(ROOT, path) == [KEY, "07_mirror", "abc"] and MI.dest_components(ROOT + "/", path + "/") == [KEY, "07_mirror", "abc"]
    for root, p, pat in ((ROOT, ROOT, "does not lie under the cluster root"), (ROOT, "/elsewhere/x", "does not lie under the cluster root"),
                         (ROOT, f"{ROOT}/a/../b", r"component '\.\.' under the root"), (ROOT, f"{ROOT}//a", r"component '' under the root"),
                         ("relative", "/r/a", "--cluster-root 'relative' is not an absolute path")):
        with pytest.raises(Refusal, match=pat):
            MI.dest_components(root, p)
    g = MI.guard_command(ROOT, path)
    assert g.startswith(f'r={ROOT}; p="$r"; s=""; for c in {KEY} 07_mirror abc; do p="$p/$c"; s="$s/$c"; ! [ -L "$p" ] || {{ echo "SYMLINK: $p -> $(readlink -- "$p")"; '
                        f'exit {MI.GUARD_SYMLINK_RC}; }}; done && ') and g.endswith(MI.RESOLVE_CHECK), g
    assert MI.RESOLVE_CHECK.startswith('{ ! [ -e "$p" ] || [ "$(realpath -- "$p")" = "$(realpath -- "$r")$s" ] || { echo "RESOLVES: ') and f"exit {MI.GUARD_RESOLVE_RC}; }}; }}" in MI.RESOLVE_CHECK
    assert (MI.GUARD_SYMLINK_RC, MI.GUARD_RESOLVE_RC) == MI.GUARD_RCS == (61, 62) and MI.CLUSTER_SCHEMA == "stage07-mirror-cluster/2" and MI.CLUSTER_SCHEMA_PRE_GUARD == "stage07-mirror-cluster/1"
    assert MI.prepare_command(ROOT, path) == f"{g} && mkdir -p {path} && {MI.RESOLVE_CHECK}"
    assert MI.check_command(path, (MI.SUMS_NAME, MI.MIRROR_NAME), root=ROOT) == f"{g} && cd {path} && sha256sum -c --quiet MIRROR.sha256 && sha256sum MIRROR.sha256 mirror.json && echo inventory: && {MI.INVENTORY_CMD}"
    assert MI.copy_check_command(path, root=ROOT) == f"{g} && cd {path} && sha256sum -c --quiet MIRROR.sha256 && sha256sum MIRROR.sha256 mirror.json cluster.json && echo inventory: && {MI.INVENTORY_CMD}"
    assert MI.guard_command("/r", "/r/x y").startswith("r=/r; p=\"$r\"; s=\"\"; for c in 'x y'; do ") and "mkdir -p '/r/x y' && " in MI.prepare_command("/r", "/r/x y")
    assert MI.guard_verdict(61, "noise\nSYMLINK: /a/b -> /x\n") == "SYMLINK: /a/b -> /x" and MI.guard_verdict(62, "RESOLVES: /a/b -> /x, not /a/b") == "RESOLVES: /a/b -> /x, not /a/b"
    assert MI.guard_verdict(1, "SYMLINK: /a -> /b") is None and MI.guard_verdict(0, "") is None and MI.guard_verdict(61, "") == "exit 61, no output"
    assert MI.TARGET_KEYS == ("cluster_label", "ssh_host", "cluster_root", "cluster_path")
    # a Cluster needs the root, and the destination under it
    with pytest.raises(Refusal, match="--cluster-root None is not an absolute path on the cluster"):
        MI.Cluster("cluster_a", "cluster-a-login", DEST)
    with pytest.raises(Refusal, match=rf"the cluster destination {re.escape(DEST)} does not lie under the cluster root /elsewhere"):
        MI.Cluster("cluster_a", "cluster-a-login", DEST, root="/elsewhere")
    c = MI.Cluster("cluster_a", "cluster-a-login", DEST + "/", root=ROOT + "/")
    assert c.root == ROOT and c.dest == DEST and c.prepare_command("abc") == MI.prepare_command(ROOT, path) and c.check_command("abc") == MI.check_command(path, (MI.SUMS_NAME, MI.MIRROR_NAME), root=ROOT)
    assert MI.target_of(c, "abc") == {"cluster_label": "cluster_a", "ssh_host": "cluster-a-login", "cluster_root": ROOT, "cluster_path": path}


@pytest.mark.skipif(shutil.which("rsync") is None or shutil.which("sha256sum") is None or shutil.which("realpath") is None or not _gnu_find(),
                    reason="the real rsync, sha256sum, realpath and GNU find are the subject here")
def test_a_symlink_at_the_destination_or_at_a_component_under_the_reproduce_root_is_refused_before_the_push_and_on_every_recheck_with_the_real_binaries(tmp_path):
    """With the REAL binaries under the local seam (the guard runs in bash, rsync and sha256sum for real): the
    site's layout — <reproduce root>/<model>/07_mirror/<receipt sha12> — guarded under the root. Before `mkdir -p` accepted a destination that
    was a symlink, the trailing-slash rsync wrote the mirror through it to wherever it pointed, `cd` followed it, `find . -mindepth 1` read
    only its children, and the push and every later filing recorded the symlink path as the durable copy. Pinned: (a) the destination
    <sha12> a symlink to an empty directory elsewhere — the push is REDIRECTED by name (the link and its target named), nothing is written
    through it (the target stays empty), no record; the link removed by hand, the re-run finishes the push and cluster.json records the root
    (schema /2); (b) a PARENT component (<root>/<model>) a symlink — refused the same way, nothing under the link's target; (c) the root ITSELF
    a symlink is the site's business: its realpath is the anchor and the push passes; (d) after a good push the copy moved aside and the path
    replaced by a symlink to it — `sha256sum -c` passes THROUGH the link, yet the re-run and the filing refuse REDIRECTED before the cd, the link
    and the moved copy untouched; restored, both pass; (e) a /1 record, or a /2 one without cluster_root, refuses by name, naming the re-run;
    (f) a re-run under another root is another target, nothing copied."""
    root = tmp_path / "reproduce"
    dest = root / KEY / "07_mirror"
    w, doc, rsha = world(tmp_path)
    state, mdir = w["state"], mirror_dir(w, rsha)
    copy = dest / rsha[:12]
    # (a) the destination itself a symlink
    dest.mkdir(parents=True)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    copy.symlink_to(elsewhere)
    with pytest.raises(Refusal, match=rf"the cluster destination local:{re.escape(str(dest))}:{re.escape(str(copy))} is REDIRECTED: SYMLINK: {re.escape(str(copy))} -> {re.escape(str(elsewhere))} — ") as e:
        _run_mirror(state, w["vol"].remote, dest=dest, root=root)
    msg = str(e.value)
    assert "rsync would write the mirror through it" in msg and "nothing pushed, nothing recorded and nothing deleted" in msg and "run the mirror again" in msg, msg
    assert mdir.is_dir() and (mdir / MI.SUMS_NAME).is_file() and not (mdir / MI.CLUSTER_NAME).exists(), "published locally, never pushed"
    assert copy.is_symlink() and list(elsewhere.iterdir()) == [], "the link stands and NOTHING was written through it"
    with pytest.raises(Refusal, match="has no cluster.json: its cluster copy was never verified"):
        MI.verified_mirror(state, rsha)
    copy.unlink()
    out = _run_mirror(state, w["vol"].remote, dest=dest, root=root)
    assert out["reused"] is False and out["cluster_root"] == str(root) and (mdir / MI.CLUSTER_NAME).is_file() and tree_bytes(copy) == tree_bytes(mdir)
    cl = json.loads((mdir / MI.CLUSTER_NAME).read_text())
    assert cl["schema"] == "stage07-mirror-cluster/2" and cl["cluster_root"] == str(root) and cl["cluster_path"] == str(copy) and "no symlink at any component between" in cl["guard"]
    assert cl["check_command"] == MI.check_command(str(copy), (MI.SUMS_NAME, MI.MIRROR_NAME), root=str(root)) and cl["check_command"].startswith(f'r={root}; p="$r"; s=""; for c in {KEY} 07_mirror {rsha[:12]}; do ')
    assert MI.verified_mirror(state, rsha)["cluster_root"] == str(root)
    assert _run_mirror(state, w["vol"].remote, dest=dest, root=root)["reused"] is True
    # (b) a parent component a symlink: the push refused, nothing under the link's target
    w2, doc2, r2 = world(tmp_path / "w2", accs=(0.5, 0.62))
    root2 = tmp_path / "reproduce2"
    root2.mkdir()
    other_model = tmp_path / "other_model"
    other_model.mkdir()
    (root2 / KEY).symlink_to(other_model)
    dest2 = root2 / KEY / "07_mirror"
    with pytest.raises(Refusal, match=rf"is REDIRECTED: SYMLINK: {re.escape(str(root2 / KEY))} -> {re.escape(str(other_model))} — "):
        _run_mirror(w2["state"], w2["vol"].remote, dest=dest2, root=root2)
    assert list(other_model.iterdir()) == [] and (root2 / KEY).is_symlink() and mirror_dir(w2, r2).is_dir() and not (mirror_dir(w2, r2) / MI.CLUSTER_NAME).exists()
    # (c) the root itself a symlink: allowed — its realpath anchors the comparison; the copy lands at the real path
    (root2 / KEY).unlink()
    root_link = tmp_path / "root_link"
    root_link.symlink_to(root2)
    out2 = _run_mirror(w2["state"], w2["vol"].remote, dest=root_link / KEY / "07_mirror", root=root_link)
    assert out2["reused"] is False and (root2 / KEY / "07_mirror" / r2[:12] / MI.SUMS_NAME).is_file() and json.loads((mirror_dir(w2, r2) / MI.CLUSTER_NAME).read_text())["cluster_root"] == str(root_link)
    assert _run_mirror(w2["state"], w2["vol"].remote, dest=root_link / KEY / "07_mirror", root=root_link)["reused"] is True
    # (d) the copy moved aside and the path replaced by a symlink to it: the sums pass through the link; the re-run and the filing refuse REDIRECTED before the cd
    assert F.rule(w) == 0
    reg = F._registry_with_objects(tmp_path / "reg")
    rc, lines, why = _file(w, reg, cluster_runner=MI.sh_runner)
    assert rc == 0 and why is None, (lines, why)
    before = tree_bytes(mdir)
    moved = tmp_path / "copy.moved"
    copy.rename(moved)
    copy.symlink_to(moved)
    assert subprocess.run(["sha256sum", "-c", "--quiet", MI.SUMS_NAME], cwd=copy, capture_output=True).returncode == 0, "the sums alone pass through the link"
    with pytest.raises(Refusal, match=rf"the cluster copy at cluster_a local:{re.escape(str(dest))}:{re.escape(str(copy))} is REDIRECTED: SYMLINK: {re.escape(str(copy))} -> {re.escape(str(moved))} — ") as e:
        _run_mirror(state, w["vol"].remote, dest=dest, root=root)
    msg = str(e.value)
    assert "the check never entered it and nothing here follows or deletes the link" in msg and "its cluster record is left as it is" in msg and tree_bytes(mdir) == before, msg
    filings = F._filings(reg)
    rc, lines, why = _file(w, reg, cluster_runner=MI.sh_runner)
    assert rc == 2 and "is REDIRECTED: SYMLINK:" in why and "nothing filed against a cluster copy that does not verify as recorded" in why and F._filings(reg) == filings, (lines, why)
    assert copy.is_symlink() and tree_bytes(moved) == tree_bytes(mdir), "the link and the moved copy stand; nothing deleted"
    copy.unlink()
    moved.rename(copy)
    assert _run_mirror(state, w["vol"].remote, dest=dest, root=root)["reused"] is True and _file(w, reg, cluster_runner=MI.sh_runner)[0] == 0
    # (e) a record of the pre-guard schema, or one without the root: refused by name, naming the re-run; restored, verified
    cl_p = mdir / MI.CLUSTER_NAME
    orig = cl_p.read_bytes()
    cl_p.write_text(json.dumps({**cl, "schema": "stage07-mirror-cluster/1"}))
    with pytest.raises(Refusal, match=r"is a cluster record of schema 'stage07-mirror-cluster/1', written before the destination was guarded") as e:
        MI.verified_mirror(state, rsha)
    assert "run `07 --phase mirror` again" in str(e.value) and "nothing is re-pulled" in str(e.value)
    cl_p.write_text(json.dumps({k: v for k, v in cl.items() if k != "cluster_root"}))
    with pytest.raises(Refusal, match=r"records no absolute `cluster_root` \(None\) — not a record of schema stage07-mirror-cluster/2"):
        MI.verified_mirror(state, rsha)
    cl_p.write_text(json.dumps({**cl, "cluster_root": str(tmp_path / "another")}))
    with pytest.raises(Refusal, match=r"the recorded cluster_path does not lie plainly under the recorded cluster_root"):
        MI.verified_mirror(state, rsha)
    cl_p.write_bytes(orig)
    assert MI.verified_mirror(state, rsha)["cluster_root"] == str(root)
    # (f) another root for the same destination is another target: refused by name, nothing copied
    with pytest.raises(Refusal, match=rf"names another target than this call's .*\(cluster_root: recorded '{re.escape(str(root))}', this call '{re.escape(str(tmp_path))}'\) — ") as e:
        _run_mirror(state, w["vol"].remote, dest=dest, root=tmp_path)
    assert "cluster_path:" not in str(e.value) and "ssh_host:" not in str(e.value), "the root alone differs"
    assert tree_bytes(copy) == tree_bytes(mdir) and cl_p.read_bytes() == orig


def test_the_guard_runs_over_ssh_as_one_command_before_the_mkdir_and_before_every_check_and_a_redirected_destination_refuses_with_the_exact_argv(tmp_path):
    """Over the stubbed ssh: the FIRST command of a push is the guarded mkdir (one ssh: the guard, `mkdir -p`, the resolved-path
    comparison), a guard exiting 61 there refuses REDIRECTED before any rsync (no rsync argv at all), the mirror stands published with no record;
    a good push records the root; a re-check whose guard exits 62 (RESOLVES) refuses the re-run and the filing (rc 2) with the exact single argv
    and no rsync; the stub refuses an unguarded mkdir or check outright."""
    w, doc, rsha = world(tmp_path)
    state, mdir = w["state"], mirror_dir(w, rsha)
    path = f"{DEST}/{rsha[:12]}"
    ssh_prefix = ["ssh", "-n", "-o", "BatchMode=yes", "-o", "ConnectTimeout=30", "cluster-a-login"]
    bad = L.RecordingCluster(lambda: mdir, guard_rc=61, guard_out=f"SYMLINK: {path} -> /scratch/example/elsewhere\n")
    with pytest.raises(Refusal, match=rf"the cluster destination cluster-a-login:{re.escape(path)} is REDIRECTED: SYMLINK: {re.escape(path)} -> /scratch/example/elsewhere — the destination, or a component between the cluster root {re.escape(ROOT)} and it, is a symlink") as e:
        L.mirror(state, w["vol"].remote, dest_root=Path(DEST), root=ROOT, host="cluster-a-login", cluster_runner=bad)
    assert bad.calls == [[*ssh_prefix, MI.prepare_command(ROOT, path)]] and bad.rsyncs() == [], "one ssh, the guarded mkdir; nothing rsynced"
    assert "nothing pushed, nothing recorded and nothing deleted" in str(e.value) and mdir.is_dir() and not (mdir / MI.CLUSTER_NAME).exists()
    good = L.RecordingCluster(lambda: mdir)
    out = L.mirror(state, w["vol"].remote, dest_root=Path(DEST), root=ROOT, host="cluster-a-login", cluster_runner=good)
    assert out["reused"] is False and out["cluster_root"] == ROOT and good.sshs()[0] == [*ssh_prefix, MI.prepare_command(ROOT, path)]
    assert good.sshs()[1] == [*ssh_prefix, MI.check_command(path, (MI.SUMS_NAME, MI.MIRROR_NAME), root=ROOT)] and len(good.calls) == 5
    cl = json.loads((mdir / MI.CLUSTER_NAME).read_text())
    assert cl["cluster_root"] == ROOT and cl["schema"] == MI.CLUSTER_SCHEMA == "stage07-mirror-cluster/2"
    # the re-check: the guard exits 62 there — REDIRECTED (RESOLVES) on the re-run and the filing, the one exact argv, no rsync, nothing filed
    assert F.rule(w) == 0
    reg = F._registry_with_objects(tmp_path / "reg")
    filings = F._filings(reg)
    resolve = dict(guard_rc=62, guard_out=f"RESOLVES: {path} -> /scratch/other/{rsha[:12]}, not {path}\n")
    rec = L.RecordingCluster(lambda: mdir, **resolve)
    with pytest.raises(Refusal, match=rf"the cluster copy at cluster_a cluster-a-login:{re.escape(path)} is REDIRECTED: RESOLVES: {re.escape(path)} -> /scratch/other/{rsha[:12]}, not {re.escape(path)} — ") as e:
        L.mirror(state, w["vol"].remote, dest_root=Path(DEST), root=ROOT, host="cluster-a-login", cluster_runner=rec)
    assert rec.calls == [[*ssh_prefix, MI.copy_check_command(path, root=ROOT)]] and "nothing here follows or deletes the link" in str(e.value)
    rec2 = L.RecordingCluster(lambda: mdir, **resolve)
    rc, lines, why = _file(w, reg, cluster_runner=rec2)
    assert rc == 2 and "is REDIRECTED: RESOLVES:" in why and rec2.calls == [[*ssh_prefix, MI.copy_check_command(path, root=ROOT)]] and F._filings(reg) == filings, (lines, why)
    assert L.mirror(state, w["vol"].remote, dest_root=Path(DEST), root=ROOT, host="cluster-a-login", cluster_runner=L.RecordingCluster(lambda: mdir))["reused"] is True
    rc, lines, why = _file(w, reg, cluster_runner=L.RecordingCluster(lambda: mdir))
    assert rc == 0 and why is None, (lines, why)
    # the stub itself refuses an unguarded command, so no test here can pass one by
    with pytest.raises(AssertionError, match="an unguarded mkdir reached the cluster"):
        L.RecordingCluster(lambda: mdir)([*ssh_prefix, f"mkdir -p {path}"])
    with pytest.raises(AssertionError, match="an unguarded check reached the cluster"):
        L.RecordingCluster(lambda: mdir)([*ssh_prefix, f"cd {path} && sha256sum -c --quiet MIRROR.sha256 && sha256sum MIRROR.sha256 mirror.json && echo inventory: && {MI.INVENTORY_CMD}"])


# ------------------------------------------------------------------------------------------- round 6: the directory set held to the listing
def test_the_pulled_directory_set_is_held_to_the_volume_listing_so_an_omitted_empty_directory_or_an_unlisted_one_refuses_before_anything_is_published(tmp_path):
    """`list_tree`'s directory listing was logged and dropped, and mirror.json's `directories` was read off the
    pulled tree, so a directory `modal volume ls --json` listed that `volume get` did not create (an empty one omitted), or one the download
    created that no listing reported, was certified. The CLI lists every directory under the store root exactly once, as a `type: dir` entry of
    its parent's listing (one `ls` per directory, recursively, paths from the volume root — no implicit parents), and its download creates a
    directory for every DIRECTORY entry, an empty one included, so the two sets agree for the store as the volume holds it. Pinned:
    (a) a listed empty directory the pull omits refuses by name BEFORE the file checks, the build left for inspection, nothing published, no
    cluster call; (b) an unlisted directory the pull yields refuses the same way; (c) the honest CLI — listed and pulled — publishes, with the
    directory recorded in mirror.json's `directories` (the listed set under each cell, confirmed on disk), a per-cell `directories` count and
    the check's wording."""
    w, doc, rsha = world(tmp_path)
    state, dest = w["state"], tmp_path / "cluster"
    cell = next(iter(doc["recs"]))
    store_rel = L.vrel(doc["recs"][cell]["out"])
    empty_rel = f"{store_rel}/scratch_empty"

    class ListsEmptyDirThatGetOmits(L.FakeModalTree):
        """`ls --json` of the store root lists an EMPTY directory (and `ls` of it is []), but `volume get` creates nothing for it."""

        def __call__(self, argv, **kw):
            sub = list(argv)[1:]
            if sub[:2] == ["volume", "ls"] and "--json" in sub and sub[-1].strip("/") == empty_rel:
                self.calls.append(list(argv))
                return subprocess.CompletedProcess(list(argv), 0, "[]", "")
            r = super().__call__(argv, **kw)
            if sub[:2] == ["volume", "ls"] and "--json" in sub and r.returncode == 0 and sub[-1].strip("/") == store_rel:
                entries = json.loads(r.stdout) + [{"filename": empty_rel, "type": "dir", "size": "0 B", "created_modified": "x"}]
                return subprocess.CompletedProcess(list(argv), 0, json.dumps(entries), "")
            return r

    class GetYieldsAnUnlistedDir(L.FakeModalTree):
        """`volume get` of the store lays down a directory no listing reported (empty, so the file checks alone would never see it)."""

        def __call__(self, argv, **kw):
            sub = list(argv)[1:]
            r = super().__call__(argv, **kw)
            if sub[:2] == ["volume", "get"] and r.returncode == 0 and sub[-2].strip("/") == store_rel:
                (Path(sub[-1]) / Path(store_rel).name / "cells" / "stray_dir").mkdir(parents=True, exist_ok=True)
            return r

    def attempt(cls):
        vol = cls(w["vol"].remote)
        vol.remote = w["vol"].remote
        lines: list = []
        with pytest.raises(Refusal) as e:
            MI.run(state=state, receipt_path=None, volume=None, environment=None, modal="/nonexistent/modal", cluster_label="cluster_a", ssh_host=f"local:{dest}",
                   cluster_dest=str(dest), cluster_root=str(dest), dry_run=False, free_headroom=0, modal_runner=vol, cluster_runner=L.local_cluster_runner, log=lines.append)
        build = build_dir(w, rsha)
        assert f"(the build stays at {build} for inspection)" in str(e.value) and build.is_dir() and (build / cell).is_dir(), "the pull happened; the build stays"
        assert not mirror_dir(w, rsha).exists() and not (state / MI.MIRROR_DIR).exists() and not dest.exists(), "nothing published, no cluster call"
        assert len(vol.gets()) == 1 and not any("file(s) in" in ln and f"{cell}:" in ln and "as listed" in ln for ln in lines), "refused at the first store, before its file checks were logged"
        shutil.rmtree(build)                                   # by hand, as the refusal says; the next attempt starts clean
        return str(e.value)

    msg = attempt(ListsEmptyDirThatGetOmits)
    assert msg.startswith(f"{cell}: the pulled directory set is not the listed one (listed on the volume but not pulled: ['scratch_empty']; pulled but not listed: []) — "
                          "`modal volume ls --json` reported every directory"), msg
    assert "a directory listed but absent from the pull, or present in the pull but never listed, is not that store; nothing published" in msg
    msg = attempt(GetYieldsAnUnlistedDir)
    assert msg.startswith(f"{cell}: the pulled directory set is not the listed one (listed on the volume but not pulled: []; pulled but not listed: ['cells/stray_dir']) — "), msg
    # (c) the honest CLI: listed AND created — published, the directory recorded (the listed set under the cell), the per-cell count, the wording

    class HonestEmptyDir(ListsEmptyDirThatGetOmits):
        def __call__(self, argv, **kw):
            sub = list(argv)[1:]
            r = super().__call__(argv, **kw)
            if sub[:2] == ["volume", "get"] and r.returncode == 0 and sub[-2].strip("/") == store_rel:
                (Path(sub[-1]) / Path(store_rel).name / "scratch_empty").mkdir(parents=True, exist_ok=True)
            return r

    vol = HonestEmptyDir(w["vol"].remote)
    vol.remote = w["vol"].remote
    lines = []
    out = MI.run(state=state, receipt_path=None, volume=None, environment=None, modal="/nonexistent/modal", cluster_label="cluster_a", ssh_host=f"local:{dest}",
                 cluster_dest=str(dest), cluster_root=str(dest), dry_run=False, free_headroom=0, modal_runner=vol, cluster_runner=L.local_cluster_runner, log=lines.append)
    mdir = mirror_dir(w, rsha)
    assert out["reused"] is False and (mdir / cell / "scratch_empty").is_dir() and (mdir / MI.CLUSTER_NAME).is_file()
    mj = json.loads((mdir / MI.MIRROR_NAME).read_text())
    assert f"{cell}/scratch_empty" in mj["directories"] and mj["directories"] == sorted(p.relative_to(mdir).as_posix() for p in mdir.rglob("*") if p.is_dir())
    for c in doc["recs"]:
        assert mj["cells"][c]["directories"] == len([d for d in mj["directories"] if d.startswith(c + "/")]), c
    assert mj["cells"][cell]["directories"] == mj["cells"][[c for c in doc["recs"] if c != cell][0]]["directories"] + 1, "the one extra listed-and-pulled directory"
    assert mj["directory_check"] == MI.DIR_CHECK and "no implicit parents" in MI.DIR_CHECK and "an empty directory the download omitted" in MI.DIR_CHECK
    assert any(f"{cell}:" in ln and "file set, sizes and directory set as listed" in ln for ln in lines), lines
    # the listing's directory walk: one `ls --json` per directory, every directory once, the listed set exactly what the pulled store holds
    files, dirs = MI.list_tree(RT.M.VolumeCLI(Path("/nonexistent/modal"), "twosi-fits", runner=vol, log=lambda s: None), store_rel)
    assert dirs == sorted(p.relative_to(mdir / cell).as_posix() for p in (mdir / cell).rglob("*") if p.is_dir()) and "scratch_empty" in dirs and len(set(dirs)) == len(dirs)
    assert set(files) == {p.relative_to(mdir / cell).as_posix() for p in (mdir / cell).rglob("*") if p.is_file()}


# ------------------------------------------------------------------- round 7: the grading-host path held plain under the state root
def _run_guarded(state: Path, vol, dest: Path, log=None):
    """The mirror tool over a given Modal double and the `local:<dest>` seam (the destination is its own root, so the cluster guard walks <sha12> alone)."""
    return MI.run(state=Path(state), receipt_path=None, volume=None, environment=None, modal="/nonexistent/modal", cluster_label="cluster_a", ssh_host=f"local:{dest}",
                  cluster_dest=str(dest), cluster_root=str(dest), dry_run=False, free_headroom=0, modal_runner=vol, cluster_runner=L.local_cluster_runner, log=log or (lambda s: None))


def test_the_local_guard_holds_the_mirror_path_plain_under_the_state_root_and_names_a_symlink_or_a_directory_that_resolves_elsewhere(tmp_path, monkeypatch):
    """The helper: `local_guard(state, sha12)` returns the literal <state>/07_mirror/<sha12> when every component below the
    state root is plain (or absent — the "no mirror" refusal is verified_mirror's); the state root itself a symlink is the site's layout (its
    realpath anchors the comparison, the literal path is returned); a symlink at <sha12> (to a directory, or dangling) or at 07_mirror is
    REDIRECTED by name with the link and its target, nothing followed; a plain directory whose realpath is not its literal path under the root's
    (a mount — the realpath seam stands in for one) is REDIRECTED (RESOLVES) with both paths."""
    state = tmp_path / "state"
    sha12 = "abcdef012345"
    literal = state / MI.MIRROR_DIR / sha12
    assert MI.local_guard(state, sha12) == literal and not state.exists(), "nothing there: nothing refused, the literal path"
    literal.mkdir(parents=True)
    assert MI.local_guard(state, sha12) == literal
    link = tmp_path / "state_link"
    link.symlink_to(state)
    assert MI.local_guard(link, sha12) == link / MI.MIRROR_DIR / sha12, "the state root a symlink: the anchor; the literal path as given"
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    literal.rmdir()
    literal.symlink_to(elsewhere)
    with pytest.raises(Refusal, match=rf"^the grading-host mirror of receipt {sha12} at {re.escape(str(literal))} is REDIRECTED: SYMLINK: {re.escape(str(literal))} -> "
                                      rf"{re.escape(str(elsewhere))} — the recorded path names a link, not the copy at that location") as e:
        MI.local_guard(state, sha12)
    msg = str(e.value)
    assert "nothing here follows, moves or deletes it" in msg and f"the state root {state} is the anchor" in msg and "put the copy back at its path by hand" in msg, msg
    assert "point --state-root at it" in msg and literal.is_symlink() and list(elsewhere.iterdir()) == [], msg
    literal.unlink()
    literal.symlink_to(tmp_path / "gone")
    with pytest.raises(Refusal, match=rf"is REDIRECTED: SYMLINK: {re.escape(str(literal))} -> {re.escape(str(tmp_path / 'gone'))} — "):
        MI.local_guard(state, sha12)
    literal.unlink()
    mroot = state / MI.MIRROR_DIR
    mroot.rmdir()
    mroot.symlink_to(elsewhere)
    with pytest.raises(Refusal, match=rf"^the grading-host mirror of receipt {sha12} at {re.escape(str(literal))} is REDIRECTED: SYMLINK: {re.escape(str(mroot))} -> {re.escape(str(elsewhere))} — "):
        MI.local_guard(state, sha12)
    mroot.unlink()
    literal.mkdir(parents=True)
    assert MI.local_guard(state, sha12) == literal
    # a plain directory that resolves elsewhere (no symlink component — a mount): RESOLVES, both paths named
    real = os.path.realpath
    mnt = tmp_path / "mnt" / sha12
    monkeypatch.setattr(MI, "_realpath", lambda p: str(mnt) if str(p) == str(literal) else real(p))
    with pytest.raises(Refusal, match=rf"is REDIRECTED: RESOLVES: {re.escape(str(literal))} -> {re.escape(str(mnt))}, not {re.escape(real(str(state)))}/{MI.MIRROR_DIR}/{sha12} — "
                                      "the directory's realpath is not its literal path under the state root's realpath") as e:
        MI.local_guard(state, sha12)
    assert "nothing here follows, moves or deletes it" in str(e.value)


def test_a_published_mirror_replaced_by_a_symlink_to_its_moved_copy_refuses_the_re_run_and_the_filing_and_a_link_at_the_path_refuses_the_publication(tmp_path):
    """After a good mirror and filing, the grading-host copy MOVED and a symlink left at <state>/07_mirror/<sha12> was
    byte-identical through the link — `mirror.is_dir()`, `verify_published`, the local walk and the cluster re-check all passed — so the re-run
    certified it and the filing read back PASS with extra.mirror.local_path naming a link rather than a copy at the recorded location (the
    earlier guard covered the cluster path alone). Pinned: (a) the mirror moved aside and replaced by a link to it — `verified_mirror`, the
    re-run and the filing (rc 2) each refuse REDIRECTED by name with the link and its target, the cluster is never consulted (no ssh at all),
    nothing is filed, the link and the moved copy stand untouched; (b) 07_mirror itself a link — the same; (c) a DANGLING link at <sha12> is
    REDIRECTED, never "no mirror", and the re-run pulls nothing; (d) the copy back at its path, the re-run and the filing pass; (e) the state
    root itself a symlink is the site's layout: verified, local_path the literal path as given; (f) publication: a link at 07_mirror or at
    <sha12> before the first run refuses BEFORE any pull (no volume get, no build, nothing under the link's target, no cluster call), and a
    link that appears at <sha12> while the build runs refuses before the rename, the build left for inspection with its two records; the link
    and the build removed by hand, the run publishes a plain directory at the literal path. The dry run names the guard."""
    w, doc, rsha = world(tmp_path)
    state, dest = w["state"], tmp_path / "cluster"
    dry = [ln for ln in L.mirror(state, w["vol"].remote, dest_root=dest, dry=True)["log"] if ln.startswith("DRY-RUN: ")]
    assert any(f"publish {mirror_dir(w, rsha)}/ by rename" in ln and f"{MI.MIRROR_DIR}/{rsha[:12]} held plain under the state root {state}" in ln and "REDIRECTED" in ln for ln in dry), dry
    assert L.mirror(state, w["vol"].remote, dest_root=dest)["reused"] is False
    mdir = mirror_dir(w, rsha)
    assert F.rule(w) == 0
    reg = F._registry_with_objects(tmp_path / "reg")
    rc, lines, why = _file(w, reg)
    assert rc == 0 and why is None and json.loads((state / SF.SPEC_NAME).read_text())["extra"]["mirror"]["local_path"] == str(mdir), (lines, why)
    rep = F.H.ingest_once(reg, workers=0, min_free=0)          # folded, so a later retry of the same filing is the folded row's PASS (or finding), not a pending read-back
    assert rep.refused == 0 and rep.blocked == 0
    rc, lines, why = _file(w, reg)
    assert rc == 0 and why is None and any("PASS:" in ln for ln in lines), (lines, why)
    before, filings = tree_bytes(mdir), F._filings(reg)
    # (a) the copy moved aside, a link left at its path: byte-identical through the link, refused everywhere before anything follows it
    moved = tmp_path / "mirror.moved"
    mdir.rename(moved)
    mdir.symlink_to(moved)
    assert mdir.is_dir() and tree_bytes(mdir) == before, "byte-identical through the link — what passed before this round"
    pat = (rf"^the grading-host mirror of receipt {rsha[:12]} at {re.escape(str(mdir))} is REDIRECTED: SYMLINK: {re.escape(str(mdir))} -> {re.escape(str(moved))} — "
           "the recorded path names a link, not the copy at that location")
    with pytest.raises(Refusal, match=pat) as e:
        MI.verified_mirror(state, rsha)
    assert "nothing here follows, moves or deletes it" in str(e.value) and "put the copy back at its path by hand" in str(e.value), str(e.value)
    rec = L.RecordingCluster(lambda: moved)
    with pytest.raises(Refusal, match=pat):
        L.mirror(state, w["vol"].remote, dest_root=dest, cluster_runner=rec)
    assert rec.calls == [], "the re-run refused before the cluster was consulted"
    rec2 = L.RecordingCluster(lambda: moved)
    rc, lines, why = _file(w, reg, cluster_runner=rec2)
    assert rc == 2 and re.search(pat, why) and "nothing filed against a grading-host mirror whose path is redirected" in why and "extra.mirror.local_path" in why, (lines, why)
    assert rec2.calls == [] and F._filings(reg) == filings, "no ssh, nothing filed"
    assert mdir.is_symlink() and tree_bytes(moved) == before, "the link and the moved copy stand; nothing deleted"
    # (b) 07_mirror itself the link: the same, the link named
    mdir.unlink()
    moved.rename(mdir)
    mroot, moved_root = state / MI.MIRROR_DIR, tmp_path / "07_mirror.moved"
    mroot.rename(moved_root)
    mroot.symlink_to(moved_root)
    pat2 = rf"^the grading-host mirror of receipt {rsha[:12]} at {re.escape(str(mdir))} is REDIRECTED: SYMLINK: {re.escape(str(mroot))} -> {re.escape(str(moved_root))} — "
    with pytest.raises(Refusal, match=pat2):
        MI.verified_mirror(state, rsha)
    with pytest.raises(Refusal, match=pat2):
        L.mirror(state, w["vol"].remote, dest_root=dest)
    rc, lines, why = _file(w, reg)
    assert rc == 2 and re.search(pat2, why) and F._filings(reg) == filings, (lines, why)
    assert mroot.is_symlink() and tree_bytes(moved_root / rsha[:12]) == before
    mroot.unlink()
    moved_root.rename(mroot)
    # (c) a dangling link at <sha12>: REDIRECTED, never "no mirror"; the re-run pulls nothing and builds nothing
    mdir.rename(moved)
    mdir.symlink_to(tmp_path / "gone")
    pat3 = rf"is REDIRECTED: SYMLINK: {re.escape(str(mdir))} -> {re.escape(str(tmp_path / 'gone'))} — "
    with pytest.raises(Refusal, match=pat3):
        MI.verified_mirror(state, rsha)
    vol = L.FakeModalTree(w["vol"].remote)
    vol.remote = w["vol"].remote
    with pytest.raises(Refusal, match=pat3):
        _run_guarded(state, vol, dest)
    assert vol.gets() == [] and not (state / MI.MIRROR_BUILD_DIR).exists(), "refused before any pull"
    rc, lines, why = _file(w, reg)
    assert rc == 2 and re.search(pat3, why) and "no verified mirror of receipt" not in why and F._filings(reg) == filings, (lines, why)
    # (d) the copy back at its path: the re-run and the filing pass
    mdir.unlink()
    moved.rename(mdir)
    assert L.mirror(state, w["vol"].remote, dest_root=dest)["reused"] is True and tree_bytes(mdir) == before
    rc, lines, why = _file(w, reg)
    assert rc == 0 and why is None and any("PASS:" in ln for ln in lines), (lines, why)
    # (e) the state root itself a symlink: the site's layout — its realpath anchors; verified, local_path the literal path as given
    link = tmp_path / "state_link"
    link.symlink_to(state)
    assert MI.verified_mirror(link, rsha)["local_path"] == str(link / MI.MIRROR_DIR / rsha[:12])
    # (f) publication: a link at 07_mirror or at <sha12> before the first run, and one that appears while the build runs
    w2, doc2, r2 = world(tmp_path / "w2", accs=(0.5, 0.62))
    s2, dest2 = w2["state"], tmp_path / "cluster2"
    mroot2, m2 = s2 / MI.MIRROR_DIR, mirror_dir(w2, r2)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()

    def attempt(vol, pat):
        with pytest.raises(Refusal, match=pat) as e:
            _run_guarded(s2, vol, dest2)
        assert list(elsewhere.iterdir()) == [] and not dest2.exists(), "nothing under the link's target, no cluster call"
        return str(e.value)

    mroot2.symlink_to(elsewhere)
    vol = L.FakeModalTree(w2["vol"].remote)
    vol.remote = w2["vol"].remote
    msg = attempt(vol, rf"^the grading-host mirror of receipt {r2[:12]} at {re.escape(str(m2))} is REDIRECTED: SYMLINK: {re.escape(str(mroot2))} -> {re.escape(str(elsewhere))} — ")
    assert vol.gets() == [] and not (s2 / MI.MIRROR_BUILD_DIR).exists() and mroot2.is_symlink() and "the build stays" not in msg, "refused before any pull"
    mroot2.unlink()
    mroot2.mkdir()
    m2.symlink_to(elsewhere)
    vol = L.FakeModalTree(w2["vol"].remote)
    vol.remote = w2["vol"].remote
    attempt(vol, rf"is REDIRECTED: SYMLINK: {re.escape(str(m2))} -> {re.escape(str(elsewhere))} — ")
    assert vol.gets() == [] and not (s2 / MI.MIRROR_BUILD_DIR).exists() and m2.is_symlink()
    m2.unlink()

    class LinkAppearsDuringThePull(L.FakeModalTree):
        """`volume get` lays the store down and, meanwhile, a link appears at <state>/07_mirror/<sha12> (a move while the pull ran)."""

        def __call__(self, argv, **kw):
            r = super().__call__(argv, **kw)
            if list(argv)[1:3] == ["volume", "get"] and not m2.is_symlink():
                m2.symlink_to(elsewhere)
            return r

    vol = LinkAppearsDuringThePull(w2["vol"].remote)
    vol.remote = w2["vol"].remote
    msg = attempt(vol, rf"is REDIRECTED: SYMLINK: {re.escape(str(m2))} -> {re.escape(str(elsewhere))} — ")
    build = build_dir(w2, r2)
    assert msg.endswith(f"(the build stays at {build} for inspection)") and build.is_dir() and (build / MI.SUMS_NAME).is_file() and (build / MI.MIRROR_NAME).is_file(), msg
    assert len(vol.gets()) == len(doc2["recs"]) and m2.is_symlink(), "the pull ran whole; the link stands"
    m2.unlink()
    shutil.rmtree(build)                                   # by hand, as the refusal says
    out = L.mirror(s2, w2["vol"].remote, dest_root=dest2)
    assert out["reused"] is False and m2.is_dir() and not m2.is_symlink() and (m2 / MI.CLUSTER_NAME).is_file() and out["local_path"] == str(m2)
    assert os.path.realpath(m2) == os.path.join(os.path.realpath(s2), MI.MIRROR_DIR, r2[:12]) and list(elsewhere.iterdir()) == []


def test_list_tree_refuses_an_entry_that_is_neither_a_file_nor_a_directory():
    """Modal's `volume ls --json` can report a symlink (`type: "link"`), a FIFO or a socket, and the directory download
    does not reproduce them; `list_tree` used to drop every such entry silently, so a store holding one was certified complete without it."""
    root = "tinycap/gen/native/qwen35_4b_demo/none/thinking_off"

    class Cli:
        def ls_json(self, d):
            return [{"filename": f"{d}/RESULT", "type": "file", "size": "1 B"}, {"filename": f"{d}/latest", "type": "link"}]

    with pytest.raises(MI.Refusal, match=r"latest is listed as 'link', neither a file nor a directory"):
        MI.list_tree(Cli(), root)

    class Plain:
        def ls_json(self, d):
            return [{"filename": f"{d}/RESULT", "type": "file", "size": "1 B"}]

    assert MI.list_tree(Plain(), root) == ({"RESULT": "1 B"}, [])
