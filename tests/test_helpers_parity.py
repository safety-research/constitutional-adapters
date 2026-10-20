"""reproduce/lib/mirror_figs.sh + lib/hf_env.sh — the helpers 00_setup.sh --what helpers installs onto every cluster's
$CLUSTER_HELPERS_DIR (as mirror-figs.sh / hf_env.sh, the paths every sbatch calls) — are SUPERSETS of the behaviour the clusters'
previously installed copies had:
  mirror-figs: the RESULT line rides with the figures (`--include='RESULT'`, added on the clusters 2026-09-10); a push key on a
  filesystem that ignores chmod (MooseFS reports 0666) is staged as a node-local 0600 copy for the push; the destination
  `user@host` comes from $SHARED/.ssh/figs_dest (never a literal in the tree); the served URL from $SHARED/.ssh/figs_url is
  printed after a push; every failure path exits 0 (a failed figure push never fails a finished job); a missing destination is
  said and the push is NOT attempted.
  hf_env: nothing probed: the job's SISTEER_HF_HOME (the portability override) first, else hf_site.env beside the installed
  copy, which 00_setup.sh --what helpers renders from the cluster block (the cluster's HF cache and shared root).
Run for real under a fake `rsync` on PATH that records its argv; nothing here names a real host, user or path."""
from __future__ import annotations

import os
import re
import shlex
import stat
import subprocess
from pathlib import Path

from front_door_testlib import MATCHERS, SCAN, run_stage, token_hits, write_site

#: one tier of the publication scan's token table, matched as the scan matches it (the tokens are never spelled here)
STORAGE_TOKENS = SCAN.Matcher([r for r in MATCHERS["rows"] if r.tier == "storage"])

REPO = Path(__file__).resolve().parents[1]
LIB = REPO / "reproduce" / "lib"
DEST = "figs-user@figs-host.example"
URL = "https://figs.example/figs"
#: the include filter the clusters' installed copies ended with (pass 1's diff record): the repo copy must carry every one of these
INSTALLED_INCLUDES = ("*/", "*.png", "*.jpg", "*.pdf", "*.svg", "RESULT")


def _fake_rsync(bin_dir: Path, record: Path, rc: int = 0) -> None:
    bin_dir.mkdir(parents=True, exist_ok=True)
    p = bin_dir / "rsync"
    p.write_text("#!/bin/bash\n" f"python3 -c 'import json,sys; open(sys.argv[1],\"a\").write(json.dumps(sys.argv[2:])+\"\\n\")' {record} \"$@\"\n"
                 # the -e ssh command names the key file: record ITS mode as the push would see it
                 "prev=''; for a in \"$@\"; do if [ \"$prev\" = -e ]; then key=$(printf '%s' \"$a\" | sed -n \"s/.*-i '\\([^']*\\)'.*/\\1/p\"); "
                 f"[ -n \"$key\" ] && stat -c '%a %n' \"$key\" >> {record}.keymode; fi; prev=\"$a\"; done\n"
                 f"exit {rc}\n")
    p.chmod(0o755)


def _world(tmp_path: Path, *, dest: bool = True, url: bool = True, key_mode: int = 0o600) -> dict:
    shared = tmp_path / "shared"; (shared / ".ssh").mkdir(parents=True)
    (shared / ".ssh" / "figs-push").write_text("not-a-real-key\n"); os.chmod(shared / ".ssh" / "figs-push", key_mode)
    (shared / ".ssh" / "known_hosts").write_text("")
    if dest:
        (shared / ".ssh" / "figs_dest").write_text(DEST + "\n")
    if url:
        (shared / ".ssh" / "figs_url").write_text(URL + "\n")
    figs = tmp_path / "figs" / "run-1"; figs.mkdir(parents=True)
    (figs / "a.png").write_bytes(b"\x89PNG\r\n" + b"0" * 900); (figs / "RESULT").write_text("COMPLETED 0 12s\n"); (figs / "notes.txt").write_text("x")
    record = tmp_path / "rsync.jsonl"
    _fake_rsync(tmp_path / "bin", record)
    return {"shared": shared, "figs": figs, "record": record}


def _run(w: dict, *args: str, env_extra: dict | None = None, job_name: str = "f22-att-m-g012") -> subprocess.CompletedProcess:
    env = {"PATH": f"{w['record'].parent / 'bin'}:{os.environ['PATH']}", "HOME": str(w["record"].parent), "USER": "u", "TMPDIR": str(w["record"].parent),
           "FIGS_SHARED_ROOT": str(w["shared"]), "SLURM_JOB_NAME": job_name, "SLURM_JOB_ID": "11111"}
    env.update(env_extra or {})
    return subprocess.run(["bash", str(LIB / "mirror_figs.sh"), str(w["figs"]), *args], capture_output=True, text=True, env=env, timeout=60)


def _calls(w: dict) -> list:
    import json
    return [json.loads(l) for l in w["record"].read_text().splitlines()] if w["record"].exists() else []


def test_mirror_figs_pushes_the_installed_filter_plus_result_to_the_site_destination_and_prints_the_url(tmp_path):
    w = _world(tmp_path)
    r = _run(w)
    assert r.returncode == 0, r.stdout + r.stderr
    calls = _calls(w); assert len(calls) == 1, calls
    argv = calls[0]
    for inc in INSTALLED_INCLUDES:
        assert f"--include={inc}" in argv, (inc, argv)                    # a superset of the clusters' installed filter
    assert "--exclude=*" in argv and "--prune-empty-dirs" in argv and "-az" in argv
    assert argv[-2] == f"{w['figs']}/" and argv[-1] == f"{DEST}:f22-att-m-g012/"          # the destination from $SHARED/.ssh/figs_dest, the job name as the subdir
    e = argv[argv.index("-e") + 1]
    assert "-o BatchMode=yes" in e and "-o IdentitiesOnly=yes" in e and f"-o UserKnownHostsFile='{w['shared']}/.ssh/known_hosts'" in e
    assert f"-i '{w['shared']}/.ssh/figs-push'" in e                        # a 0600 key is used in place
    assert f"mirror-figs: pushed {w['figs']} -> {DEST}:f22-att-m-g012/" in r.stdout and f"{URL}/f22-att-m-g012/" in r.stdout
    assert "--delete" not in argv and "--export" not in " ".join(argv)


def test_mirror_figs_stages_a_0600_copy_of_a_world_readable_key(tmp_path):
    w = _world(tmp_path, key_mode=0o666)                                    # MooseFS: every file reports 0666
    r = _run(w, "custom-name")
    assert r.returncode == 0, r.stdout + r.stderr
    argv = _calls(w)[0]; e = argv[argv.index("-e") + 1]
    assert f"-i '{w['shared']}/.ssh/figs-push'" not in e                    # never the 0666 original
    modes = (w["record"].parent / "rsync.jsonl.keymode").read_text().split()
    assert modes[0] == "600" and modes[1].startswith(str(w["record"].parent)) and "figs-push." in modes[1]   # a node-local 0600 temp copy
    assert not Path(modes[1]).exists()                                      # removed after the push (the EXIT trap)
    assert argv[-1] == f"{DEST}:custom-name/"


def test_mirror_figs_says_so_and_pushes_nothing_without_a_destination_and_never_fails_the_job(tmp_path):
    w = _world(tmp_path, dest=False, url=False)
    r = _run(w)
    assert r.returncode == 0 and _calls(w) == []                            # exit 0, and the push was NOT attempted
    assert "no destination" in r.stderr and "figs_dest" in r.stderr and "figures stay at" in r.stderr, r.stderr
    # FIGS_DEST_HOST from the job's env prefix stands in for the file; a URL is optional (the pushed line still prints)
    r = _run(w, env_extra={"FIGS_DEST_HOST": DEST})
    assert r.returncode == 0 and _calls(w)[-1][-1] == f"{DEST}:f22-att-m-g012/" and "mirror-figs: pushed" in r.stdout and URL not in r.stdout
    # a failed rsync: said on stderr, exit 0 — a failed figure push never fails a finished job
    _fake_rsync(w["record"].parent / "bin", w["record"], rc=23)
    r = _run(w, env_extra={"FIGS_DEST_HOST": DEST})
    assert r.returncode == 0 and "push failed (rc=23)" in r.stderr and "figures are still at" in r.stderr, r.stderr
    # a missing figures dir / no push key: said, exit 0
    r = subprocess.run(["bash", str(LIB / "mirror_figs.sh"), str(tmp_path / "absent")], capture_output=True, text=True,
                       env={"PATH": os.environ["PATH"], "USER": "u", "FIGS_SHARED_ROOT": str(w["shared"])})
    assert r.returncode == 0 and "no figures dir" in r.stderr
    os.remove(w["shared"] / ".ssh" / "figs-push")
    r = _run(w, env_extra={"FIGS_DEST_HOST": DEST})
    assert r.returncode == 0 and "no push key" in r.stderr and _calls(w)[-1][-1] == f"{DEST}:f22-att-m-g012/"   # (the earlier call; none new)
    assert len(_calls(w)) == 2


def test_mirror_figs_name_sanitization_and_the_job_id_fallback(tmp_path):
    w = _world(tmp_path)
    r = _run(w, "a/b c", job_name="ignored")
    assert r.returncode == 0 and _calls(w)[-1][-1] == f"{DEST}:a-b-c/", r.stdout + r.stderr       # no slashes: stays inside the figure root
    env = {"PATH": f"{w['record'].parent / 'bin'}:{os.environ['PATH']}", "USER": "u", "FIGS_SHARED_ROOT": str(w["shared"]), "SLURM_JOB_ID": "22222"}
    r = subprocess.run(["bash", str(LIB / "mirror_figs.sh"), str(w["figs"])], capture_output=True, text=True, env=env)
    assert r.returncode == 0 and _calls(w)[-1][-1] == f"{DEST}:job-22222/", r.stdout + r.stderr


def _source_hf_env(helper: Path, env: dict) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", "-c", f'set -e; source {shlex.quote(str(helper))}; echo "rc=$? $HF_HOME|$HF_TOKEN_PATH|$HUGGINGFACE_TOKEN_PATH"'],
                          capture_output=True, text=True, env={"PATH": os.environ["PATH"], "USER": "nobody-here", **env})


def test_hf_env_probes_nothing_and_its_override_comes_first(tmp_path):
    src = (LIB / "hf_env.sh").read_text()
    code = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith("#"))
    assert not re.findall(r"\[ -[de] ", code), code                                  # no directory probe of any storage root
    assert not token_hits(code, STORAGE_TOKENS), token_hits(code, STORAGE_TOKENS)     # and no storage root spelled in code
    assert src.index("SISTEER_HF_HOME") < src.index("hf_site.env")                   # the portability override comes FIRST
    # the override branch, for real (the job's shared root names the token; without it, the cache's parent does)
    r = _source_hf_env(LIB / "hf_env.sh", {"SISTEER_HF_HOME": str(tmp_path / "hf"), "SISTEER_SHARED_ROOT": str(tmp_path / "sh")})
    assert r.returncode == 0 and r.stdout.strip() == f"rc=0 {tmp_path}/hf|{tmp_path}/sh/hf_token|{tmp_path}/sh/hf_token", r.stdout + r.stderr
    # no override and no rendered values beside the copy: said on stderr, HF_HOME unset, exit 0 (sourcing never kills the job body)
    r = subprocess.run(["bash", "-c", f'set -e; source {shlex.quote(str(LIB / "hf_env.sh"))}; echo "rc=$? HF_HOME=${{HF_HOME:-unset}}"'],
                       capture_output=True, text=True, env={"PATH": os.environ["PATH"], "USER": "nobody-here-either"})
    assert r.returncode == 0 and "HF_HOME=unset" in r.stdout and "HF_HOME left unset" in r.stderr and "hf_site.env" in r.stderr, r.stdout + r.stderr


def test_the_installed_hf_env_reads_the_rendered_site_values_beside_it(tmp_path):
    """The installed copy beside its hf_site.env: the cluster's cache and a token under its shared root; a shared root the job
    carries names the token instead; the job's SISTEER_HF_HOME wins over the file; nothing of the file leaks into the job."""
    helpers = tmp_path / "shared" / "scripts"; helpers.mkdir(parents=True)
    helper = helpers / "hf_env.sh"; helper.write_text((LIB / "hf_env.sh").read_text())
    cache, root = tmp_path / "models", tmp_path / "shared"
    (helpers / "hf_site.env").write_text(f"# rendered\nHF_SITE_HF_HOME={cache}\nHF_SITE_SHARED_ROOT={root}\n")
    r = _source_hf_env(helper, {})
    assert r.stdout.strip() == f"rc=0 {cache}|{root}/hf_token|{root}/hf_token", r.stdout + r.stderr
    r = _source_hf_env(helper, {"SISTEER_SHARED_ROOT": str(tmp_path / "job")})
    assert r.stdout.strip() == f"rc=0 {cache}|{tmp_path}/job/hf_token|{tmp_path}/job/hf_token", r.stdout + r.stderr
    r = _source_hf_env(helper, {"SISTEER_HF_HOME": str(tmp_path / "jobcache"), "SISTEER_SHARED_ROOT": str(tmp_path / "job")})
    assert r.stdout.strip() == f"rc=0 {tmp_path}/jobcache|{tmp_path}/job/hf_token|{tmp_path}/job/hf_token", r.stdout + r.stderr
    leak = subprocess.run(["bash", "-c", f'source {shlex.quote(str(helper))}; echo "${{HF_SITE_HF_HOME:-none}} ${{HF_SITE_SHARED_ROOT:-none}}"'],
                          capture_output=True, text=True, env={"PATH": os.environ["PATH"], "USER": "nobody-here"})
    assert leak.stdout.strip() == "none none", leak.stdout


def test_helpers_dry_run_renders_each_clusters_hf_site_file_before_the_helper(tmp_path):
    site = write_site(tmp_path, clusters=("alpha", "beta"))
    r = run_stage("00_setup.sh", "--what", "helpers", "--dry-run", site=site)
    assert r.returncode == 0, (r.stdout + r.stderr)[-2500:]
    for lab in ("alpha", "beta"):
        shared = tmp_path / f"shared_{lab}"
        head = f"DRY-RUN: write {lab}-login:{shared}/scripts/hf_site.env:"
        assert head in r.stdout, r.stdout[-2500:]
        for line in (f"HF_SITE_HF_HOME={shared}/hf_cache", f"HF_SITE_SHARED_ROOT={shared}"):     # the block's $SHARED expanded
            assert f"DRY-RUN:   {line}" in r.stdout, (lab, line)
        assert r.stdout.index(head) < r.stdout.index(f"hf_env.sh {lab}-login:{shared}/scripts/hf_env.sh")


def test_helpers_render_the_hand_submitted_purposes_w_and_b_projects_into_hf_site_env(tmp_path):
    """A compute node has no site config, so the W&B project the site names for each purpose whose jobs are submitted by hand (budget,
    alphasteer) rides in hf_site.env; a purpose the site leaves unset is written empty (hf_env.sh then exports nothing for it)."""
    site = write_site(tmp_path, clusters=("alpha",), extra_config="SISTEER_WANDB_PROJECT_BUDGET='budget proj'")
    r = run_stage("00_setup.sh", "--what", "helpers", "--dry-run", site=site, env={"SISTEER_WANDB_PROJECT_ALPHASTEER": ""})
    assert r.returncode == 0, (r.stdout + r.stderr)[-2500:]
    body = r.stdout[r.stdout.index("hf_site.env:"):r.stdout.index("DRY-RUN: rsync")]
    assert "DRY-RUN:   HF_SITE_WANDB_PROJECT_BUDGET=budget\\ proj" in body and "DRY-RUN:   HF_SITE_WANDB_PROJECT_ALPHASTEER=''" in body, body


def test_the_installed_hf_env_exports_the_sites_w_and_b_projects_unless_the_job_carries_its_own(tmp_path):
    helpers = tmp_path / "shared" / "scripts"; helpers.mkdir(parents=True)
    helper = helpers / "hf_env.sh"; helper.write_text((LIB / "hf_env.sh").read_text())
    (helpers / "hf_site.env").write_text(f"# rendered\nHF_SITE_HF_HOME={tmp_path / 'models'}\nHF_SITE_SHARED_ROOT={tmp_path / 'shared'}\n"
                                         "HF_SITE_WANDB_PROJECT_BUDGET=budget\\ proj\nHF_SITE_WANDB_PROJECT_ALPHASTEER=''\n")
    show = ('echo "B=${SISTEER_WANDB_PROJECT_BUDGET-unset} A=${SISTEER_WANDB_PROJECT_ALPHASTEER-unset} '
            'leak=${HF_SITE_WANDB_PROJECT_BUDGET-none}${HF_SITE_WANDB_PROJECT_ALPHASTEER-none}"; env | grep -c "^SISTEER_WANDB_PROJECT_BUDGET=" || true')

    def run(env: dict) -> list:
        r = subprocess.run(["bash", "-c", f"set -euo pipefail; source {shlex.quote(str(helper))}; {show}"], capture_output=True, text=True,
                           env={"PATH": os.environ["PATH"], "USER": "nobody-here", **env})
        assert r.returncode == 0, r.stderr
        return r.stdout.split("\n")[:2]

    # exported (so a python child reads it), the empty purpose left unset, nothing of the file left in the job
    assert run({}) == ["B=budget proj A=unset leak=nonenone", "1"]
    # a value the job already carries wins
    assert run({"SISTEER_WANDB_PROJECT_BUDGET": "job-proj"})[0] == "B=job-proj A=unset leak=nonenone"
    # whichever HF branch ran: the job's own HF cache does not skip the W&B values
    assert run({"SISTEER_HF_HOME": str(tmp_path / "jobcache")})[0] == "B=budget proj A=unset leak=nonenone"
    # an hf_site.env from before the W&B lines exports nothing and still sets HF_HOME
    (helpers / "hf_site.env").write_text(f"# rendered\nHF_SITE_HF_HOME={tmp_path / 'models'}\nHF_SITE_SHARED_ROOT={tmp_path / 'shared'}\n")
    assert run({})[0] == "B=unset A=unset leak=nonenone"
