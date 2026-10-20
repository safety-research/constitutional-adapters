"""``scripts/sdfbase/full22/judge/finish_lane.sh``: the unattended lane finisher's readiness probe is the vault's MODE-AWARE client and
stops at the FIRST credential the API rejects.

The finisher waits for the judge org to serve sonnet-5 -- one canary-shaped probe a minute, for up to ``F22_GIVE_UP_S`` (24 h) --
and only then runs ``launch_judge.sh``, which carries ``SISTEER_SECRETS_MODE`` into its detached unit. The probe itself had not
moved: it fetched the pool's key through the secret-manager CLI and built a plain ``anthropic.Anthropic(api_key=..., base_url=...)``
inside a shell heredoc, whatever the mode. Under ``SISTEER_SECRETS_MODE=wif`` that is the legacy key the API has rejected since the
org's move to federation (401 on every probe), so the finisher re-tried a rejected credential once a minute for a day, never
reached the federated launcher, and then marked the lane failed -- a rejected credential retried, in its shell form, and a client
the vault's mode could not see (a class the Python-only audit of ``tests/test_base_url_pin_sources.py`` did not cover). Under
``env`` it was wrong too: the secret-manager CLI instead of the one documented key.

Now the probe builds its client through ``sisteer.vault.anthropic_client(org, max_retries=0)`` -- op: the pool's key resolved
in-process (never in the shell's environment, never printed); env: the ONE ``ANTHROPIC_API_KEY``; wif: federated, keyless, the secret-manager CLI
never consulted; the base URL pinned in every mode -- and prints ONE token: ``OK``; ``CREDENTIAL_REJECTED:<type>:<status>[:<api type>]``
(``sisteer.vault.credential_rejection``: a 401 / 403 on the request, or a federation identity the token exchange refuses) which
stops the lane at that probe with a message naming the mode and the fix; ``VAULT_REFUSED:<type>:<why>`` (a pool the vault cannot
serve: an absent / empty ids file, the secret-manager CLI unset or missing) which stops it too -- nothing to wait for; or ``<type>:<status>`` for everything
else (a 429 / 529 / 5xx / transport error), which is the ONE thing the loop waits out. A fourth mode value refuses before any probe; a
probe that prints nothing (an interpreter that cannot start) is a refusal, not a silent 24-hour spin (the old loop logged nothing
when every probe printed nothing, because it logged only on change). ``F22_PY`` names the probe's interpreter (petri_drive_full22.sh's
knob), which is how these tests run the REAL script with a REAL vault import and a scripted client: the interpreter the tests hand
it pre-binds ``vault.anthropic_client`` to a recording factory, records every probe (the org, the mode the vault read, which
``ANTHROPIC_*`` variables were in the environment) and raises the SDK's own exceptions; a fake secret-manager CLI, named by
``SISTEER_SECRET_CMD``, sits on the PATH and at ``~/.local/bin`` and must never run in the probe. ``launch_judge.sh`` and ``lib/secrets.sh`` are the real ones, behind the recording fake
``systemd-run`` of the other launcher tests; the judge stub records the mode and whether a key reached it and exits 3, so the
finisher stops right after the launch with a message naming the stamp -- proof it got past the probe.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _env import write_python3_shim  # noqa: E402
from front_door_testlib import LIB, REPO, write_fake_systemd_run  # noqa: E402

F22 = REPO / "scripts/sdfbase/full22"
FINISHER = F22 / "judge/finish_lane.sh"
LAUNCHER = F22 / "judge/launch_judge.sh"
#: values with NO credential shape (gitleaks scans this file too): the assertion is that they are never read or printed
SENTINEL = "env-value-" + "never-read-" * 2
GATEWAY = "http://127.0.0.1:1/gateway"

#: the probe interpreter's preamble: binds ``sisteer.vault.anthropic_client`` to a scripted, recording factory when the scenario file
#: carries answers (an empty scenario leaves the REAL vault in place), then runs the probe program the script feeds on stdin
PROBE_PATCH = r'''
import json, os, sys
REC = os.environ["PROBE_RECORD"]
SCEN = os.environ["PROBE_SCENARIO"]
with open(SCEN) as fh:
    scenario = json.load(fh)
answers = scenario.get("answers")
if answers is not None:
    import anthropic
    from sisteer import vault
    try:
        import httpx2 as httpx          # the SDK's own http library: httpx2 under anthropic >= 1.x (the lock), httpx under 0.x
    except ImportError:
        import httpx

    URL = "https://api.anthropic.com/v1/messages"

    def _status_error(status, api_type):
        body = {"type": "error", "error": {"type": api_type, "message": "synthetic"}}
        cls = {401: anthropic.AuthenticationError, 403: anthropic.PermissionDeniedError}.get(status, anthropic.APIStatusError)
        return cls(f"Error code: {status} - {body}", response=httpx.Response(status, request=httpx.Request("POST", URL)), body=body)

    class _Messages:
        def __init__(self, answer):
            self.answer = answer

        def create(self, **kw):
            a = self.answer
            if a == "OK":
                return {"ok": True}
            if a == "401":
                raise _status_error(401, "authentication_error")
            if a == "403":
                raise _status_error(403, "permission_error")
            if a == "429":
                raise _status_error(429, "rate_limit_error")
            if a == "529":
                raise _status_error(529, "overloaded_error")
            if a == "conn":
                raise anthropic.APIConnectionError(request=httpx.Request("POST", URL))
            if a == "federation-401":      # the token exchange refused the federation identity: NOT an APIStatusError
                raise anthropic.WorkloadIdentityError("Token exchange failed (HTTP 401): synthetic", status_code=401,
                                                      body={"error": {"type": "authentication_error", "message": "synthetic"}})
            raise AssertionError(f"unknown scripted answer {a!r}")

    class _Client:
        def __init__(self, answer):
            self.messages = _Messages(answer)

    def factory(org, **kw):
        answer = answers.pop(0) if len(answers) > 1 else answers[0]
        with open(SCEN, "w") as fh:
            json.dump({"answers": answers}, fh)
        with open(REC, "a") as fh:
            fh.write(json.dumps({"org": org, "kw": {k: (v if isinstance(v, (int, float, str, type(None))) else str(v)) for k, v in kw.items()},
                                 "mode": os.environ.get("SISTEER_SECRETS_MODE"), "vault_mode": vault.secrets_mode(),
                                 "anthropic_env": sorted(k for k in os.environ if k.startswith("ANTHROPIC_")), "answer": answer}) + "\n")
        return _Client(answer)

    vault.anthropic_client = factory
'''


def _shim(tmp_path: Path) -> Path:
    """The probe interpreter: THIS interpreter, running PROBE_PATCH first, then the program the script feeds on stdin, with the
    script's ``- <org>`` arguments as the program's ``sys.argv``."""
    d = tmp_path / "shim"
    d.mkdir(exist_ok=True)
    (d / "probe_patch.py").write_text(PROBE_PATCH)
    boot = ("import sys; sys.path.insert(0, %r); import probe_patch; sys.argv = ['-'] + [a for a in sys.argv[1:] if a != '-']; "
            "exec(compile(sys.stdin.read(), '<probe>', 'exec'), {'__name__': '__main__'})" % str(d))
    shim = d / "probe-python"
    shim.write_text(f"#!/bin/bash\nexec {sys.executable} -c {json.dumps(boot)} \"$@\"\n")
    shim.chmod(0o755)
    return shim


def _wt(tmp_path: Path):
    """A worktree stub with the REAL finish_lane.sh + launch_judge.sh + lib/secrets.sh, the real ``src`` (the vault and the judging
    constants import for real) and a judge stub that records the mode and whether a key reached it, then exits 3."""
    wt = tmp_path / "wt"
    (wt / "scripts/sdfbase/full22/judge").mkdir(parents=True)
    (wt / "reproduce/lib").mkdir(parents=True)
    (wt / ".venv/bin").mkdir(parents=True)
    shutil.copy(FINISHER, wt / "scripts/sdfbase/full22/judge/finish_lane.sh")
    shutil.copy(LAUNCHER, wt / "scripts/sdfbase/full22/judge/launch_judge.sh")
    shutil.copy(LIB / "secrets.sh", wt / "reproduce/lib/secrets.sh")
    (wt / "src").symlink_to(REPO / "src", target_is_directory=True)
    (wt / "scripts/served_capability_judge.py").write_text("")
    seen = tmp_path / "judge_saw.txt"
    (wt / ".venv/bin/python").write_text(
        '#!/bin/bash\n# stub judge: the mode it runs under and whether a key reached it, never a value; exit 3 stops the finisher after the launch\n'
        'echo "mode=${SISTEER_SECRETS_MODE:-unset} key=$([ -n "${ANTHROPIC_API_KEY+x}" ] && echo set || echo unset)" > ' + str(seen) + '\nexit 3\n')
    (wt / ".venv/bin/python").chmod(0o755)
    return wt, seen


def _env(tmp_path: Path, mode: str, answers, **extra):
    """The finisher's environment: a restricted PATH with the fakes first, HOME under tmp, the host's ANTHROPIC_* / SISTEER_* / F22_*
    scrubbed, the mode, a short give-up, no sleep between probes, the probe interpreter and its scenario; a fake secret-manager CLI on the
    PATH and at ~/.local/bin, named by SISTEER_SECRET_CMD, that leaves a marker when invoked."""
    fake = tmp_path / "bin"
    home = tmp_path / "home"
    root = tmp_path / "grade"
    (root / "logs").mkdir(parents=True, exist_ok=True)
    e = {k: v for k, v in os.environ.items()
         if not k.startswith(("ANTHROPIC_", "SISTEER_", "HF_", "OPENROUTER_", "OP_", "CAPEVAL_", "F22_", "PYTHONPATH", "PROBE_"))}
    scen = tmp_path / "scenario.json"
    scen.write_text(json.dumps({"answers": answers}))
    rec = tmp_path / "probes.jsonl"
    rec.unlink(missing_ok=True)                    # every scenario starts with an empty record (a test runs several in one tmp_path)
    e.update(PATH=f"{fake}:/usr/bin:/bin", HOME=str(home), SISTEER_SECRETS_MODE=mode, F22_GRADE_ROOT=str(root), F22_PROBE_S="0",
             F22_GIVE_UP_S="600", F22_PY=str(_shim(tmp_path)), PROBE_SCENARIO=str(scen), PROBE_RECORD=str(rec))   # the give-up is exercised with a ZERO budget only: a cold node took > 6 s per probe
    e.update(extra)
    write_python3_shim(fake)
    marker = tmp_path / "secret-cli-ran"
    for d in (fake, home / ".local" / "bin"):
        d.mkdir(parents=True, exist_ok=True)
        (d / "fake-secret-cmd").write_text(f'#!/bin/sh\necho "ran $*" >> {marker}\nprintf "%s" never-a-real-value\n')
        (d / "fake-secret-cmd").chmod(0o755)
    e.setdefault("SISTEER_SECRET_CMD", str(fake / "fake-secret-cmd"))
    return e, marker, rec, root


def _run(wt: Path, env: dict, timeout: int = 120) -> subprocess.CompletedProcess:
    cache = Path(env["F22_GRADE_ROOT"]) / "cache.sqlite"
    return subprocess.run(["bash", str(wt / "scripts/sdfbase/full22/judge/finish_lane.sh"), "tagx", "t1", "", "login-host", "/remote/root", str(cache)],
                          capture_output=True, text=True, env=env, timeout=timeout)


def _records(rec: Path) -> list:
    return [json.loads(l) for l in rec.read_text().splitlines()] if rec.exists() else []


def _lane_failed(root: Path) -> str:
    p = root / "logs/tagx/LANE_FAILED"
    assert p.exists(), "no LANE_FAILED sentinel"
    return p.read_text()


@pytest.mark.parametrize("answer,token", [("401", "CREDENTIAL_REJECTED:AuthenticationError:401:authentication_error"),
                                          ("403", "CREDENTIAL_REJECTED:PermissionDeniedError:403:permission_error"),
                                          ("federation-401", "CREDENTIAL_REJECTED:WorkloadIdentityError:401:authentication_error")])
def test_under_wif_a_rejected_credential_stops_the_finisher_at_the_first_probe_and_nothing_is_launched(tmp_path, answer, token):
    """A 401 / 403 -- on the request, or at the federation token exchange -- is a refusal at probe #1, named with the
    mode and the fix, with the judge never launched; the probe is the vault's client under the mode (keyless: the inherited key and
    gateway URL were scrubbed before it ran), built once, with no retry of its own; the secret-manager CLI never runs."""
    wt, seen = _wt(tmp_path)
    rec_sd = tmp_path / "systemd.log"
    write_fake_systemd_run(tmp_path / "bin", rec_sd, run=True)
    env, marker, rec, root = _env(tmp_path, "wif", [answer], ANTHROPIC_API_KEY=SENTINEL, ANTHROPIC_API_KEY_LOWPRIO=SENTINEL, ANTHROPIC_BASE_URL=GATEWAY)
    r = _run(wt, env)
    out = r.stdout + r.stderr
    assert r.returncode == 1, out
    failed = _lane_failed(root)
    assert token in failed and "SISTEER_SECRETS_MODE=wif" in failed and "probe #1" in failed and "D6" in failed and "nothing launched" in failed, failed
    assert "the pool's ids file" in failed                                   # the wif fix is named
    assert "gave up" not in failed and "gave up" not in out                   # the 24-hour wait never started
    probes = _records(rec)
    assert len(probes) == 1, probes                                           # ONE probe: no retry of a rejected credential
    p = probes[0]
    assert p["org"] == "lowprio" and p["mode"] == "wif" and p["vault_mode"] == "wif" and p["answer"] == answer
    assert p["kw"] == {"max_retries": 0}, p["kw"]                             # the loop is the retry; the base URL is the vault's pin
    assert p["anthropic_env"] == [], p["anthropic_env"]                       # the inherited key and gateway URL were scrubbed
    assert not rec_sd.exists(), "launch_judge.sh ran after a rejected credential: " + rec_sd.read_text()
    assert not seen.exists()
    assert not marker.exists(), "the secret-manager CLI was invoked under wif: " + marker.read_text()
    assert "[tagx probe" in out and "lowprio/wif sonnet-5: " + token in out  # the probe line names the mode and the verdict
    assert SENTINEL not in out and SENTINEL not in failed and GATEWAY not in out


def test_a_non_credential_answer_is_waited_out_and_the_give_up_is_the_only_clock(tmp_path):
    """What the loop IS for: a 529, a transport error, a 429 are 'not yet' -- probed again (F22_PROBE_S) until the org answers, every
    change of verdict logged with its type and status, never a credential stop. Deterministic on a slow node: the scripted answers
    end the loop (two OKs reach the launcher, whose stub judge exits 3), and the give-up path runs with a ZERO budget."""
    wt, seen = _wt(tmp_path)
    rec_sd = tmp_path / "systemd.log"
    write_fake_systemd_run(tmp_path / "bin", rec_sd, run=True)
    env, marker, rec, root = _env(tmp_path, "wif", ["529", "conn", "429", "OK", "OK"], F22_GIVE_UP_S="600")
    r = _run(wt, env, timeout=300)
    out = r.stdout + r.stderr
    assert r.returncode == 1, out
    failed = _lane_failed(root)
    assert failed.startswith("judge run ") and " exited 3 " in failed, failed              # the three 'not yet' answers were waited out
    probes = _records(rec)
    assert [p["answer"] for p in probes] == ["529", "conn", "429", "OK", "OK"], probes
    assert all(p["mode"] == "wif" and p["vault_mode"] == "wif" for p in probes)
    assert "APIStatusError:529" in out and "APIConnectionError:None" in out and "APIStatusError:429" in out, out
    assert "CREDENTIAL_REJECTED" not in out and "VAULT_REFUSED" not in out and "gave up" not in out
    assert rec_sd.exists() and seen.read_text().strip() == "mode=wif key=unset" and not marker.exists()
    # the give-up: F22_GIVE_UP_S is the only clock, read after each probe -- a zero budget gives up after the first 'not yet', by name
    shutil.rmtree(root / "logs/tagx")
    rec_sd.unlink(); seen.unlink()
    env, marker, rec, root = _env(tmp_path, "wif", ["529"], F22_GIVE_UP_S="0")
    r = _run(wt, env)
    failed = _lane_failed(root)
    assert r.returncode == 1 and failed.startswith("gave up waiting for lowprio after") and "(last probe: APIStatusError:529)" in failed, failed
    assert len(_records(rec)) == 1 and not rec_sd.exists() and not seen.exists() and not marker.exists()


@pytest.mark.parametrize("mode,flag,key_seen,cli_ran", [
    ("wif", "--setenv=SISTEER_SECRETS_MODE=wif", "unset", False),      # the unit receives the MODE and no key; its vault federates
    ("env", "--setenv=ANTHROPIC_API_KEY", "set", False),               # name-only import of the one documented key (round 2's contract)
    ("op", "--setenv=SISTEER_SECRET_CMD=", "set", True),               # the unit's own line runs the CLI whose path rides along
])
def test_two_consecutive_ok_probes_launch_the_judge_unit_under_the_mode(tmp_path, mode, flag, key_seen, cli_ran):
    """After a 'not yet' and two OKs the finisher reaches launch_judge.sh, whose unit runs under the finisher's mode: keyless under wif,
    the one key under env, the secret-manager CLI inside the unit under op. The stub judge exits 3, so the finisher stops right after the launch
    naming the stamp -- it got past the probe. Under op the probe keeps the pool's own variable (the vault reads it); under wif the
    inherited key is gone before the probe."""
    wt, seen = _wt(tmp_path)
    rec_sd = tmp_path / "systemd.log"
    write_fake_systemd_run(tmp_path / "bin", rec_sd, run=True)
    extra = {"ANTHROPIC_API_KEY": SENTINEL, "ANTHROPIC_API_KEY_LOWPRIO": SENTINEL, "ANTHROPIC_BASE_URL": GATEWAY}
    env, marker, rec, root = _env(tmp_path, mode, ["529", "OK", "OK"], **extra)
    r = _run(wt, env)
    out = r.stdout + r.stderr
    assert r.returncode == 1, out
    failed = _lane_failed(root)
    assert failed.startswith("judge run ") and " exited 3 " in failed, failed            # past the probe, past the launch
    probes = _records(rec)
    assert [p["answer"] for p in probes] == ["529", "OK", "OK"], probes
    assert all(p["mode"] == mode and p["vault_mode"] == mode for p in probes)
    expected_env = [] if mode == "wif" else ["ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY_LOWPRIO"]     # the gateway URL is gone in every mode
    assert probes[0]["anthropic_env"] == expected_env, probes[0]["anthropic_env"]
    calls = [json.loads(l) for l in rec_sd.read_text().splitlines()]
    launch = calls[0]
    assert launch["props"].get("unit", "").startswith("f22-orjudge-")
    assert calls[1]["exit"] == 0                        # the unit's bash ran to completion; the judge's rc (3) went into EXIT-<stamp>, which the finisher read
    stamp = launch["props"]["unit"][len("f22-orjudge-"):]
    assert (root / f"logs/EXIT-{stamp}").read_text().strip() == "3" and f"judge run {stamp} exited 3" in failed
    if flag:
        assert any(a == flag or (flag.endswith("=") and a.startswith(flag)) for a in launch["argv"]), launch["argv"]
    assert not any(a.startswith("--setenv=ANTHROPIC") and "=" in a[len("--setenv="):] for a in launch["argv"])   # never a value on argv
    assert "SISTEER_SECRETS_MODE" in launch["unit_env_keys"] or mode == "op"
    assert seen.read_text().strip() == f"mode={mode if mode != 'op' else 'unset'} key={key_seen}", seen.read_text()
    assert marker.exists() == cli_ran, (marker.read_text() if marker.exists() else "the secret-manager CLI never ran")
    assert "[tagx] unit=f22-orjudge-" in out and "org=lowprio" in out
    assert SENTINEL not in out and SENTINEL not in json.dumps(calls) and GATEWAY not in out


def test_under_op_a_rejected_credential_names_the_federation_hint_and_the_probe_kept_the_pools_own_variable(tmp_path):
    """The machine this was written on: op mode, the legacy lowprio key, 401. The finisher now says so at probe #1 and points at the
    mode that works, instead of waiting out a day; the probe ran under op with the pool's own variable in place (the vault's read)."""
    wt, seen = _wt(tmp_path)
    write_fake_systemd_run(tmp_path / "bin", tmp_path / "systemd.log", run=True)
    env, marker, rec, root = _env(tmp_path, "op", ["401"], ANTHROPIC_API_KEY_LOWPRIO=SENTINEL)
    r = _run(wt, env)
    assert r.returncode == 1, r.stdout + r.stderr
    failed = _lane_failed(root)
    assert "CREDENTIAL_REJECTED:AuthenticationError:401" in failed and "SISTEER_SECRETS_MODE=op" in failed and "SISTEER_SECRETS_MODE=wif" in failed, failed
    probes = _records(rec)
    assert len(probes) == 1 and probes[0]["mode"] == "op" and probes[0]["anthropic_env"] == ["ANTHROPIC_API_KEY_LOWPRIO"], probes
    assert not (tmp_path / "systemd.log").exists() and not seen.exists()
    assert SENTINEL not in r.stdout + r.stderr + failed


def test_a_fourth_mode_an_unservable_pool_or_a_probe_that_cannot_run_refuses_before_any_wait(tmp_path):
    """Three refusals that used to be a silent 24-hour spin (the old loop logged only on CHANGE, so an empty verdict never even
    reached the log): a mode value the vault does not know, before any probe; the REAL vault under wif with the pool's ids file
    absent (VAULT_REFUSED, probe #1, no network); an interpreter that cannot start (PROBE_ERROR, probe #1)."""
    wt, seen = _wt(tmp_path)
    write_fake_systemd_run(tmp_path / "bin", tmp_path / "systemd.log", run=True)
    # (a) a fourth mode: refused by name before the probe
    env, marker, rec, root = _env(tmp_path, "federation", ["OK"])
    r = _run(wt, env)
    assert r.returncode == 1 and "SISTEER_SECRETS_MODE=federation: must be op, env or wif" in _lane_failed(root), r.stdout + r.stderr
    assert not rec.exists() and not marker.exists()
    # (b) the real vault under wif, the ids directory empty: a typed refusal at the first probe, nothing to wait for
    ids = tmp_path / "ids"
    ids.mkdir()
    env, marker, rec, root = _env(tmp_path, "wif", None, SISTEER_WIF_ENV_DIR=str(ids))
    r = _run(wt, env)
    failed = _lane_failed(root)
    from sisteer import vault
    lo_name = Path(vault.wif_env_path("lowprio", env={vault.WIF_ENV_DIR_ENV: str(ids)})).name       # as the vault names it
    assert r.returncode == 1 and "VAULT_REFUSED:PoolUnavailable:" in failed and lo_name in failed and "nothing to wait for" in failed, failed
    assert "probe #1" in failed and not rec.exists() and not marker.exists() and "gave up" not in failed
    # (c) an interpreter that cannot start: an empty verdict is a refusal, not a silent spin
    env, marker, rec, root = _env(tmp_path, "wif", ["OK"], F22_PY=str(tmp_path / "no-such-interpreter"))
    r = _run(wt, env)
    failed = _lane_failed(root)
    assert r.returncode == 1 and "PROBE_ERROR:no verdict from the probe" in failed and "no-such-interpreter" in failed, failed
    assert "probe #1" in failed and (root / "logs/tagx/probe.err").exists()
    assert not (tmp_path / "systemd.log").exists() and not seen.exists()


def test_the_finisher_builds_its_probe_through_the_vault_and_never_through_a_plain_constructor_or_the_secret_cli():
    """The source pin behind the behavioural tests: the probe calls ``vault.anthropic_client`` and ``vault.credential_rejection``, the
    plain constructor and the shell's key fetch are gone, the interpreter is the F22_PY knob, and the README no longer calls the finisher
    op-only."""
    src = FINISHER.read_text()
    assert "vault.anthropic_client(org, max_retries=0)" in src and "vault.credential_rejection(e)" in src
    assert "anthropic.Anthropic(" not in src and '"${SISTEER_SECRET_CMD' not in src and '"$SISTEER_SECRET_CMD"' not in src   # no shell key fetch
    assert "ANTHROPIC_API_KEY=" not in src                                   # the key never crosses the shell's environment
    assert 'PY=${F22_PY:-$WT/.venv/bin/python}' in src and 'CREDENTIAL_REJECTED:*) fail' in src and 'VAULT_REFUSED:*|PROBE_ERROR:*) fail' in src
    readme = (F22 / "judge/README.md").read_text()
    assert "unattended under the `op` secrets mode" not in readme and "finish_lane.sh" in readme
