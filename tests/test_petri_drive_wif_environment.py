"""Under ``SISTEER_SECRETS_MODE=wif`` the REAL ``petri_drive_full22.sh`` starts its legs with NO ``ANTHROPIC_*`` variable in their
environment.

The drive exported ``ANTHROPIC_BASE_URL=https://api.anthropic.com`` unconditionally -- right for the op / env legs, whose keyed SDK
clients would otherwise follow a shell's gateway, but under wif the vault pins the host in-process and the mode's contract is
that no ``ANTHROPIC_*`` reaches a leg -- and its wif branch probed ``ANTHROPIC_API_KEY`` alone, so a hand-launched drive carried an
inherited OAuth bearer (``ANTHROPIC_AUTH_TOKEN``, as a Claude Code shell exports it), the per-org keys and the gateway URL into
every leg process. Now the export is op / env only and the wif branch DROPS the inherited credential / URL variables by name, never
reading a value -- the scrub every generated judging unit already applies under wif (``reproduce/lib/secrets.sh``
``secrets_unit_key_line``, ``launch_judge.sh``, ``grade_sampled.sh``).

The environment is MEASURED, not read off the source: the real drive runs in a stub worktree (the portability test's harness --
a stub leg interpreter, a fake ``ssh``, a configured site label) up to its first remote step, which comes AFTER the whole secrets
block; the fake ``ssh`` and the stub interpreter each record the ``ANTHROPIC_*`` NAMES of the environment they were started with
(never a value). A source pin then holds that nothing after the attestation block assigns an ``ANTHROPIC_`` variable, so the leg
launch (``petri_run.py ... run``) runs under the same set the fake ``ssh`` observed."""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from front_door_testlib import LIB, REPO, git, write_site  # noqa: E402

if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))          # the source pin reads the vault's one shell-scrub spelling (round 10)

F22 = REPO / "scripts/sdfbase/full22"
DRIVE = F22 / "petri_drive_full22.sh"
PIN = "https://api.anthropic.com"
GATEWAY = "http://127.0.0.1:1/mock-gateway"        # a Claude Code shell's gateway: synthetic
BEARER = "synthetic-oauth-bearer-not-a-token"      # what such a shell exports under ANTHROPIC_AUTH_TOKEN: synthetic, no credential
PER_ORG = "synthetic-per-org-key-not-a-key"        # a vault shell's ANTHROPIC_API_KEY_LOWPRIO / _BATCH: synthetic
ONE_KEY = "synthetic-env-mode-key-not-a-key"       # env mode's one ANTHROPIC_API_KEY: synthetic
DIGEST = "0123456789ab"                            # what the stub interpreter answers for the vault's ids digest
SCRUBBED = ("ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_API_KEY_LOWPRIO", "ANTHROPIC_API_KEY_BATCH", "ANTHROPIC_BASE_URL")

_RECORDER = r'''{ printf 'argv=%s\n' "$(printf '%s ' "$@" | tr '\n' ' ' | cut -c1-240)"; printf 'names=%s\n' "$(env | grep -o '^ANTHROPIC_[A-Za-z0-9_]*' | sort | tr '\n' ' ')"; printf 'base_url=%s\n' "${ANTHROPIC_BASE_URL-<unset>}"; echo '--'; } >> @REC@
'''


def _recorder(record: Path) -> str:
    return _RECORDER.replace("@REC@", str(record))


def _records(path: Path) -> list[dict]:
    """The recorder's blocks as dicts: argv, names (a set of ANTHROPIC_* NAMES), base_url."""
    if not path.exists():
        return []
    out = []
    for block in path.read_text().split("--\n"):
        if not block.strip():
            continue
        d = dict(line.split("=", 1) for line in block.splitlines() if "=" in line)
        d["names"] = set(d.get("names", "").split())
        out.append(d)
    return out


def _world(tmp_path: Path) -> dict:
    """A stub worktree holding the REAL drive, a leg interpreter that records its environment (and answers the vault's digest
    call), a fake ssh that records its environment and fails the endpoint read, and the HOME the drive's checks need."""
    site = write_site(tmp_path)
    wt = tmp_path / "wt"; (wt / "scripts/sdfbase/full22").mkdir(parents=True); (wt / ".venv/bin").mkdir(parents=True); (wt / "reproduce/lib").mkdir(parents=True)
    shutil.copy(DRIVE, wt / "scripts/sdfbase/full22/petri_drive_full22.sh")
    shutil.copy(LIB / "clusters.py", wt / "reproduce/lib/clusters.py")
    (wt / "src/sisteer").mkdir(parents=True); shutil.copy(REPO / "src/sisteer/site.py", wt / "src/sisteer/site.py")   # clusters.py parses with it
    py_rec, ssh_rec = tmp_path / "py.rec", tmp_path / "ssh.rec"
    py = wt / ".venv/bin/python"
    py.write_text("#!/bin/bash\n# the leg interpreter stub: the ANTHROPIC_* NAMES it was started with, never a value; the vault's digest call answered\n"
                  + _recorder(py_rec) + f"case \"$*\" in *wif_ids_sha12*) echo {DIGEST};; esac\nexit 0\n")
    py.chmod(0o755)
    bin_dir = tmp_path / "bin"; bin_dir.mkdir()
    ssh = bin_dir / "ssh"
    ssh.write_text("#!/bin/bash\n# a fake ssh: records the environment the drive hands its remote steps, then fails the endpoint read\n"
                   + _recorder(ssh_rec) + "exit 1\n")
    ssh.chmod(0o755)
    git(wt, "init", "-q"); git(wt, "config", "user.email", "t@example.invalid"); git(wt, "config", "user.name", "t"); git(wt, "add", "-A"); git(wt, "commit", "-qm", "x")
    sha = git(wt, "rev-parse", "HEAD")
    home = tmp_path / "home"; bank = tmp_path / "stage/sdfbase/full22_objects"; (bank / "gemma4e4b/cells_view/main").mkdir(parents=True)
    (home / ".local/state/petri/scenarios/8812ba083bb5473c").mkdir(parents=True)
    base = {"PATH": f"{bin_dir}:{os.environ['PATH']}", "HOME": str(home), "USER": "tester", "SISTEER_SITE_DIR": str(site), "F22_GRADING_HOST_BANK_ROOT": str(bank),
            "F22_SERVING_PATH": "lora_native", "OPENROUTER_API_KEY": "synthetic-openrouter-not-a-key", "LANG": os.environ.get("LANG", "C.UTF-8")}
    return {"wt": wt, "sha": sha, "home": home, "base": base, "py_rec": py_rec, "ssh_rec": ssh_rec,
            "drive_log": home / ".local/state/sdfbase_full22_native/petri/gemma4e4b/drive.log"}


def _drive(w: dict, **env) -> subprocess.CompletedProcess:
    r = subprocess.run(["bash", "scripts/sdfbase/full22/petri_drive_full22.sh", "gemma4e4b", "main", "cluster_a", w["sha"]],
                       cwd=w["wt"], capture_output=True, text=True, env={**w["base"], **env}, timeout=120)
    r.text = r.stdout + r.stderr + (w["drive_log"].read_text() if w["drive_log"].exists() else "")
    return r


def test_under_wif_the_drive_hands_its_legs_no_anthropic_variable_at_all(tmp_path):
    """A hand launch from a shell that carries an OAuth bearer, a gateway URL and the per-org keys: the drive reaches its first
    remote step (past the whole secrets block) and the environment it hands out carries NO ANTHROPIC_* name -- the inherited ones
    dropped, none exported; the vault's own digest call ran scrubbed too; no value of any of them anywhere in the output or the log."""
    w = _world(tmp_path)
    r = _drive(w, SISTEER_SECRETS_MODE="wif", ANTHROPIC_AUTH_TOKEN=BEARER, ANTHROPIC_BASE_URL=GATEWAY, ANTHROPIC_API_KEY_LOWPRIO=PER_ORG, ANTHROPIC_API_KEY_BATCH=PER_ORG,
               ANTHROPIC_CUSTOM_HEADERS="X-Api-Key: " + PER_ORG, ANTHROPIC_WEBHOOK_SIGNING_KEY=PER_ORG, ANTHROPIC_PROFILE="synthetic")   # round 10: beyond the four
    assert r.returncode != 0 and "FATAL: no endpoint.json at cluster-a-login:" in r.text, r.text[-1500:]     # the step AFTER the secrets block
    assert "[secrets] wif mode" in r.text and f"auditor_auth=wif:lowprio ids_sha12={DIGEST}" in r.text and "secrets_mode=wif" in r.text
    ssh = _records(w["ssh_rec"])
    assert ssh and ssh[0]["names"] == set() and ssh[0]["base_url"] == "<unset>", ssh                     # the drive's environment: nothing
    py = _records(w["py_rec"])
    digest_calls = [p for p in py if "wif_ids_sha12" in p["argv"]]
    assert len(digest_calls) == 1 and digest_calls[0]["names"] == set(), py                              # the vault call itself ran scrubbed
    assert py and all(p["names"] == set() and p["base_url"] == "<unset>" for p in py), py               # every interpreter start, the legs' shape
    everything = r.text + w["ssh_rec"].read_text() + w["py_rec"].read_text()
    for value in (BEARER, PER_ORG, "mock-gateway"):
        assert value not in everything, value                                                          # names were dropped, values never read


def test_under_wif_a_present_api_key_is_still_refused_by_name_before_anything_runs(tmp_path):
    """The one variable the drive REFUSES rather than drops (an explicit key shadows federation even when empty; the grading daemon never
    hands one): the FATAL names it, no interpreter and no ssh was started, and the value is nowhere."""
    w = _world(tmp_path)
    r = _drive(w, SISTEER_SECRETS_MODE="wif", ANTHROPIC_API_KEY=ONE_KEY, ANTHROPIC_AUTH_TOKEN=BEARER)
    assert r.returncode == 1 and "FATAL: SISTEER_SECRETS_MODE=wif but ANTHROPIC_API_KEY is present in the environment" in r.text, r.text[-800:]
    assert _records(w["ssh_rec"]) == [] and _records(w["py_rec"]) == []
    assert ONE_KEY not in r.text and BEARER not in r.text


def test_under_env_mode_the_legs_still_get_the_pinned_base_url_and_the_one_key(tmp_path):
    """The old modes unchanged: env mode exports the pin over an inherited gateway URL (a shell's would swallow the keyed SDK calls)
    and the one ANTHROPIC_API_KEY reaches the legs by name; the key's value is used, never printed."""
    w = _world(tmp_path)
    r = _drive(w, SISTEER_SECRETS_MODE="env", ANTHROPIC_API_KEY=ONE_KEY, ANTHROPIC_BASE_URL=GATEWAY)
    assert r.returncode != 0 and "FATAL: no endpoint.json at cluster-a-login:" in r.text and "[secrets] env mode" in r.text, r.text[-1500:]
    ssh = _records(w["ssh_rec"])
    assert ssh and {"ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL"} <= ssh[0]["names"] and ssh[0]["base_url"] == PIN, ssh
    assert ONE_KEY not in r.text and "mock-gateway" not in r.text


def test_the_source_scrubs_by_name_exports_the_url_for_the_key_modes_only_and_touches_no_anthropic_variable_after_the_attestation():
    from sisteer import vault
    s = DRIVE.read_text()
    assert vault.WIF_SHELL_SCRUB == 'unset "${!ANTHROPIC_@}"' and vault.WIF_SHELL_SCRUB in s          # EVERY ANTHROPIC_* name, by name (round 10)
    wif_branch = s.split('if [ "$SECRETS_MODE" = wif ]; then\n  SECRET_CMD=""', 1)[1].split("elif", 1)[0]
    assert '[ -z "${ANTHROPIC_API_KEY+x}" ] ||' in wif_branch and vault.WIF_SHELL_SCRUB in wif_branch
    assert wif_branch.index('ANTHROPIC_API_KEY+x') < wif_branch.index(vault.WIF_SHELL_SCRUB)          # refuse the key, then drop the rest
    assert re.search(r"\bunset\s+ANTHROPIC_", s) is None                                                 # no enumerated list anywhere in the drive
    assert '[ "${SISTEER_SECRETS_MODE:-op}" = wif ] || export ANTHROPIC_BASE_URL=https://api.anthropic.com' in s
    assert s.count("export ANTHROPIC_BASE_URL=") == 1                                                  # the one export, mode-guarded
    assert re.search(r"^\s*export ANTHROPIC_BASE_URL=", s, re.M) is None                                 # never unconditional again
    # after the attestation block nothing assigns or exports an ANTHROPIC_ variable: the leg launch sees the set the fake ssh saw
    tail = s.split('export PETRI_TWO_SI_CONFIG="$CFG"', 1)[1]
    assert "scripts/petri_run.py" in tail and " run --out-root " in tail                                 # the leg launch is in this tail
    assert re.search(r"^\s*(?:export\s+)?ANTHROPIC_\w+=", tail, re.M) is None, "an ANTHROPIC_ assignment after the attestation block"
    # the scrub is the generated judging units' (reproduce/lib/secrets.sh): ONE spelling, the vault's -- bash's name-only prefix
    # expansion, so a name the SDK reads that no list named (ANTHROPIC_CUSTOM_HEADERS, round 10) is dropped too
    lib = (LIB / "secrets.sh").read_text()
    assert vault.WIF_SHELL_SCRUB in lib and re.search(r"\bunset\s+ANTHROPIC_API_KEY\s+ANTHROPIC_", lib) is None
    assert set(SCRUBBED) | {"ANTHROPIC_API_KEY"} <= set(vault.WIF_INHERITED_ENV_NAMES)                  # the four + the refused key: known inherited names
