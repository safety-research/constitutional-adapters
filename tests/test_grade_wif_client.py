"""SISTEER_SECRETS_MODE=wif — THE WIF JUDGE PATH as a vault MODE (ported from moe-routing-ntpl's
``anthropic_client_wif``). The org moved to workload identity federation on 2026-10-07 and every judge
still authenticated with the legacy secret-manager lowprio key, which api.anthropic.com rejects with
401 (grading_diagnosis.md): 39 dispatches, 0 rc 0. Under the mode the vault builds
every Anthropic client through the SDK's ``WorkloadIdentityCredentials`` under the org's federation ids read off
the per-org workload-identity env file and NO API key is read, set or stored anywhere. No network: the token provider
is a stub and no request is made.

Pinned here:
- ``secrets_mode()`` accepts ``wif``; an unknown mode still refuses; ``op`` and ``env`` are BYTE-IDENTICAL to what they
  were (the same keyed clients, the same stamps, the secret-manager CLI / the one env key exactly as before);
- the org -> file map: ``lowprio`` reads the ``loprio`` stem's env file and no other (the near-miss ``lowprio`` stem
  must refuse BY NAME, never fall back to another file); unknown orgs refuse; an empty / incomplete env
  file refuses with ``PoolUnavailable`` naming the file and the missing keys;
- under wif ``anthropic_client`` / ``anthropic_async_client`` return the federated client: built on credentials (never
  ``api_key=``), the base URL pinned, the stamp KEYLESS — ``{"pool", "auth": "wif", "ids_sha12": sha256(file)[:12]}`` —
  which ``client_pool_stamp`` accepts and ``stamp_label`` prints; ``ANTHROPIC_API_KEY`` in the environment is never
  read; the secret-manager CLI is never invoked; ``anthropic_api_key()`` RAISES by name;
- every migrated judge builds its client through the vault and records the keyless stamp (the keyless-stamp tests:
  the secret-manager CLI and ``anthropic_api_key`` monkeypatched to raise, neither called, ``ANTHROPIC_API_KEY``
  never read): ``LiveStrongREJECTGrader`` (static / padding / BoN / multi-turn through ``grade_bon_live.make_client``),
  the prebuilt-client seam, ``misalign_judge.PoolClient`` (agentic), ``capeval.judging.ClaudeJudge`` and
  ``capeval_judged_run._count_tokens_client`` (controls), ``judge_sonnet.Judge``; static pins that the three static
  graders and the shared body resolve no credential of their own (no secret-manager shim, no ``load_key`` seam) and
  that the provenance of each carries ``api_pool``;
- the shell half: ``reproduce/lib/common.sh`` admits the mode, ``secret_anthropic`` refuses by name under it and a
  generated unit's key line only scrubs; the site example documents it.
"""
from __future__ import annotations

import collections.abc
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import types
from pathlib import Path

import pytest

from sisteer import vault

anthropic = pytest.importorskip("anthropic")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tests"))
from front_door_testlib import write_site  # noqa: E402

IDS = {"ANTHROPIC_FEDERATION_RULE_ID_GCP": "fr_11111111", "ANTHROPIC_ORGANIZATION_ID": "org_22222222",
       "ANTHROPIC_SERVICE_ACCOUNT_ID": "sa_33333333", "ANTHROPIC_WORKSPACE_ID": "ws_44444444"}
#: a value with NO credential shape (gitleaks scans this file too); the assertion is that the variable is never consulted
SENTINEL = "env-value-" + "never-read-" * 2


def _ids_name(org: str) -> str:
    """The ids file name the vault gives ``org``'s pool: ``wif-<stem>.env``, the stem from ``vault.WIF_ORG_FILES`` (pinned below)."""
    return f"wif-{vault.WIF_ORG_FILES[org]}.env"


LO, HI, BA = _ids_name("lowprio"), _ids_name("highprio"), _ids_name("batch")


def _env_dir(tmp_path: Path, monkeypatch, *, loprio: dict | None = IDS, lowprio_empty: bool = True) -> Path:
    d = tmp_path / "anthropic"
    d.mkdir(exist_ok=True)
    if loprio is not None:
        (d / LO).write_text("# the lowprio pool\n" + "".join(f"{k}={v}\n" for k, v in loprio.items()))
        (d / HI).write_text("".join(f"{k}={v}-hi\n" for k, v in loprio.items()))
        (d / BA).write_text("".join(f"{k}={v}-batch\n" for k, v in loprio.items()))
    if lowprio_empty:
        (d / "wif-lowprio.env").write_text("")
    monkeypatch.setenv(vault.WIF_ENV_DIR_ENV, str(d))
    return d


def _file_sha12(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:12]


class _RecordingEnviron(collections.abc.MutableMapping):
    """``os.environ`` with every READ of an Anthropic org variable recorded (writes and deletes pass through), so a test
    can assert the SDK and the vault consulted none of them while building a federated client."""

    def __init__(self, real):
        self._real = real
        self.reads: list = []

    def _note(self, key):
        if isinstance(key, str) and key.startswith("ANTHROPIC_API_KEY"):
            self.reads.append(key)

    def __getitem__(self, key):
        self._note(key)
        return self._real[key]

    def get(self, key, default=None):
        self._note(key)
        return self._real.get(key, default)

    def __contains__(self, key):
        self._note(key)
        return key in self._real

    def __setitem__(self, key, value):
        self._real[key] = value

    def __delitem__(self, key):
        del self._real[key]

    def __iter__(self):
        return iter(self._real)

    def __len__(self):
        return len(self._real)

    def copy(self):
        return dict(self._real)


@pytest.fixture
def wif_mode(tmp_path, monkeypatch):
    """The mode under test with the credential doors NAILED SHUT: the secret-manager CLI raises, the key resolver
    raises, the one env key is a sentinel that must never be read, and every read of an org variable is recorded."""
    d = _env_dir(tmp_path, monkeypatch)
    monkeypatch.setenv(vault.SECRETS_MODE_ENV, "wif")
    monkeypatch.setenv("ANTHROPIC_API_KEY", SENTINEL)
    monkeypatch.setenv("ANTHROPIC_API_KEY_LOWPRIO", SENTINEL)
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://127.0.0.1:1/mock-gateway")      # a gateway URL in the environment: never inherited

    def no_op_secret(*a, **k):
        raise AssertionError("the secret-manager CLI must never run under wif")

    def no_key(*a, **k):
        raise AssertionError("anthropic_api_key must never be consulted to build a wif client")

    monkeypatch.setattr(vault.subprocess, "check_output", no_op_secret)
    monkeypatch.setattr(vault.subprocess, "run", no_op_secret)
    monkeypatch.setattr(vault, "resolve_secret", no_key)
    monkeypatch.setattr(vault, "env_mode_key", no_key)
    rec = _RecordingEnviron(os.environ)
    monkeypatch.setattr(os, "environ", rec)
    yield types.SimpleNamespace(dir=d, reads=rec.reads, provider=lambda: "google-id-token",
                                sha12=_file_sha12(d / LO))


def _assert_federated(client, *, org="lowprio", async_client=False):
    assert isinstance(client, anthropic.AsyncAnthropic if async_client else anthropic.Anthropic)
    assert client.api_key is None and client.auth_token is None, "a WIF client holds no API key — the env was never read"
    assert isinstance(client.credentials, anthropic.WorkloadIdentityCredentials)
    assert str(client.base_url).rstrip("/") == vault.ANTHROPIC_BASE_URL.rstrip("/")
    st = vault.client_pool_stamp(client)
    assert set(st) == {"pool", "auth", "ids_sha12"} and st["pool"] == org and st["auth"] == "wif"
    assert re.fullmatch(r"[0-9a-f]{12}", st["ids_sha12"])
    assert "key" not in " ".join(st)
    return st


# ------------------------------------------------------------------------------------------------- the vault: the mode


def test_secrets_mode_accepts_wif_and_still_refuses_a_fourth(monkeypatch):
    monkeypatch.setenv(vault.SECRETS_MODE_ENV, "wif")
    assert vault.secrets_mode() == "wif" == vault.WIF_MODE
    assert vault.SECRETS_MODES == ("op", "env", "wif")
    monkeypatch.setenv(vault.SECRETS_MODE_ENV, "federation")
    with pytest.raises(vault.VaultSecretError, match="must be one of 'op', 'env' or 'wif'"):
        vault.secrets_mode()
    monkeypatch.delenv(vault.SECRETS_MODE_ENV)
    assert vault.secrets_mode() == "op"           # the default is unchanged


def test_the_org_file_map_and_the_path_resolution(tmp_path, monkeypatch):
    d = _env_dir(tmp_path, monkeypatch)
    assert vault.WIF_ORG_FILES == {"lowprio": "loprio", "highprio": "hiprio", "batch": "batch"}
    assert set(vault.WIF_ORG_FILES) == set(vault.ANTHROPIC_ORGS)         # every pool the vault names has a file
    assert vault.wif_env_path("lowprio") == str(d / LO)
    assert vault.wif_env_path("highprio") == str(d / HI)
    assert vault.wif_env_path("batch") == str(d / BA)
    with pytest.raises(ValueError, match="org must be one of"):
        vault.wif_env_path("loprio")
    assert vault.wif_ids("lowprio") == IDS
    # the empty twin is NEVER read for lowprio (the stem map), and read by path it refuses by name
    with pytest.raises(vault.PoolUnavailable, match=r"wif-lowprio\.env is empty"):
        vault.wif_ids("lowprio", path=str(d / "wif-lowprio.env"))
    monkeypatch.delenv(vault.WIF_ENV_DIR_ENV)
    assert vault.wif_env_path("lowprio") == os.path.expanduser("~/.anthropic/" + LO)   # the default dir


def test_an_absent_empty_or_incomplete_env_file_refuses_by_name(tmp_path, monkeypatch):
    d = _env_dir(tmp_path, monkeypatch, loprio=None)
    with pytest.raises(vault.PoolUnavailable, match="does not exist") as ei:
        vault.wif_ids("lowprio")
    assert ei.value.pool == "lowprio"
    (d / LO).write_text("")
    with pytest.raises(vault.PoolUnavailable, match="is empty") as ei:
        vault.wif_ids("lowprio")
    assert LO in str(ei.value) and "ANTHROPIC_WORKSPACE_ID" in str(ei.value)
    partial = {k: v for k, v in IDS.items() if k != "ANTHROPIC_SERVICE_ACCOUNT_ID"}
    (d / LO).write_text("".join(f"{k}={v}\n" for k, v in partial.items()) + "ANTHROPIC_SERVICE_ACCOUNT_ID=\n")
    with pytest.raises(vault.PoolUnavailable, match=r"is incomplete .* lacks \['ANTHROPIC_SERVICE_ACCOUNT_ID'\]"):
        vault.wif_ids("lowprio")
    with pytest.raises(vault.PoolUnavailable):          # no digest for a file that is not a record
        vault.wif_ids_sha12("lowprio")
    # quoted values are accepted (an env file written with quotes)
    (d / LO).write_text("".join(f'{k}="{v}"\n' for k, v in IDS.items()))
    assert vault.wif_ids("lowprio") == IDS


def test_the_ids_sha12_is_the_digest_of_the_file_and_the_stamp_is_keyless(tmp_path, monkeypatch):
    d = _env_dir(tmp_path, monkeypatch)
    assert vault.wif_ids_sha12("lowprio") == _file_sha12(d / LO)
    assert vault.wif_pool_stamp("lowprio") == {"pool": "lowprio", "auth": "wif", "ids_sha12": _file_sha12(d / LO)}
    assert vault.wif_pool_stamp("highprio")["ids_sha12"] == _file_sha12(d / HI) != vault.wif_pool_stamp("lowprio")["ids_sha12"]
    # a comment appended changes the digest: it is the FILE's, the identity the daemon attests, not a hash of the ids
    (d / LO).write_text((d / LO).read_text() + "# rotated\n")
    assert vault.wif_ids("lowprio") == IDS and vault.wif_ids_sha12("lowprio") == _file_sha12(d / LO)
    assert vault.stamp_label(vault.wif_pool_stamp("lowprio")) == f"auth wif ids sha12 {vault.wif_ids_sha12('lowprio')}"
    assert vault.stamp_label({"pool": "lowprio", "key_sha256_12": "abc"}) == "key sha256/12 abc"


def test_the_client_is_built_on_federation_credentials_with_a_keyless_stamp_and_never_reads_the_api_key(tmp_path, monkeypatch):
    _env_dir(tmp_path, monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", SENTINEL)
    rec = _RecordingEnviron(os.environ)
    monkeypatch.setattr(os, "environ", rec)                 # every read of an org variable recorded (the constructor must make none)
    calls = []

    def provider():
        calls.append(1)
        return "google-id-token"

    client = vault.anthropic_client_wif("lowprio", max_retries=2, timeout=12.5, token_provider=provider)
    # round 10: the constructor DROPS the inherited name (by name -- no read recorded) before the SDK reads the environment, whatever
    # the mode switch says: this constructor is the wif path, and a static credential left in the environment would ride its requests
    assert rec.reads == [] and "ANTHROPIC_API_KEY" not in list(os.environ), rec.reads     # list(): iteration, so the check itself is no read
    st = _assert_federated(client)
    # round 12 on fd06740ae: the SDK's own retries are OFF on a federated client and the count asked for is the budget of the
    # vault's latched retry layer (tests/test_wif_federated_destination_and_sdk_retries.py measures the replay)
    assert client.max_retries == 0 and vault.federated_retry_budget(client) == 2 and float(client.timeout) == 12.5
    assert calls == [], "the token is minted at call time, never at construction"
    assert st["ids_sha12"] == vault.wif_ids_sha12("lowprio")
    # the stamp is the one record a grader's provenance reads; a plain client without one still refuses
    with pytest.raises(ValueError, match="carries no pool stamp"):
        vault.client_pool_stamp(object())
    with pytest.raises(ValueError):                      # a half stamp (auth without the digest) is not a record either
        vault.client_pool_stamp(types.SimpleNamespace(**{vault.POOL_STAMP_ATTR: {"pool": "lowprio", "auth": "wif"}}))
    # an explicit env path overrides the map (the test's own file), the org stays the pool's name, the digest is that file's
    alt = tmp_path / "other.env"
    alt.write_text("".join(f"{k}={v}\n" for k, v in IDS.items()))
    c2 = vault.anthropic_client_wif("highprio", token_provider=provider, env_path=str(alt))
    assert vault.client_pool_stamp(c2) == {"pool": "highprio", "auth": "wif", "ids_sha12": _file_sha12(alt)}
    a = vault.anthropic_client_wif("batch", token_provider=provider, async_client=True)
    _assert_federated(a, org="batch", async_client=True)
    assert rec.reads == [] and "ANTHROPIC_API_KEY" not in list(os.environ)   # dropped by NAME at the first construction, never read, never used


def test_under_wif_the_vault_clients_are_federated_and_the_key_resolver_refuses_by_name(wif_mode):
    client = vault.anthropic_client("lowprio", max_retries=1, timeout=33.0)
    st = _assert_federated(client)
    assert st["ids_sha12"] == wif_mode.sha12 and float(client.timeout) == 33.0
    assert client.max_retries == 0 and vault.federated_retry_budget(client) == 1        # round 12: SDK retries OFF, the 1 the layer's budget
    a = vault.anthropic_async_client("lowprio")
    assert _assert_federated(a, async_client=True) == st
    hi = vault.anthropic_client("highprio", base_url="https://api.anthropic.com/")
    assert vault.client_pool_stamp(hi)["pool"] == "highprio"
    assert vault.client_pool_stamp(hi)["ids_sha12"] == _file_sha12(wif_mode.dir / HI)
    assert wif_mode.reads == [], f"an Anthropic org variable was read while building the federated clients: {wif_mode.reads}"
    for org in ("lowprio", "highprio", "batch"):
        with pytest.raises(vault.PoolUnavailable) as ei:
            vault.anthropic_api_key(org)
        msg = str(ei.value)
        assert ei.value.pool == org and "SISTEER_SECRETS_MODE=wif" in msg and "NO API key" in msg
        assert "anthropic_client" in msg and "never the secret-manager CLI" in msg and SENTINEL not in msg
    with pytest.raises(ValueError, match="org must be one of"):
        vault.anthropic_api_key("midprio")           # an unknown org is still the ValueError, before the mode
    assert wif_mode.reads == []


def test_under_wif_a_missing_ids_file_refuses_by_name_and_never_falls_back(wif_mode):
    (wif_mode.dir / LO).unlink()
    with pytest.raises(vault.PoolUnavailable, match=f"{LO} does not exist") as ei:
        vault.anthropic_client("lowprio")
    assert ei.value.pool == "lowprio"
    assert vault.client_pool_stamp(vault.anthropic_client("highprio"))["pool"] == "highprio"    # the other pool stands
    assert wif_mode.reads == []


@pytest.mark.parametrize("mode", ["op", "env"])
def test_op_and_env_are_unchanged_by_the_mode(mode, tmp_path, monkeypatch):
    """The keyed contract, exactly as before: the same ``api_key=`` clients, the same keyed stamps, the key resolver
    answering; the WIF files are never opened under either (the directory is empty here and nothing refuses)."""
    monkeypatch.setenv(vault.WIF_ENV_DIR_ENV, str(tmp_path / "no-such-dir"))
    monkeypatch.setenv(vault.SECRETS_MODE_ENV, mode) if mode == "env" else monkeypatch.delenv(vault.SECRETS_MODE_ENV, raising=False)
    monkeypatch.setattr(vault.subprocess, "check_output", lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError("fake-secret-cmd")))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "placeholder-high")
    monkeypatch.setenv("ANTHROPIC_API_KEY_LOWPRIO", "placeholder-low")
    opened = []
    real_open = open

    def spy_open(path, *a, **k):
        if "wif-" in str(path):
            opened.append(str(path))
        return real_open(path, *a, **k)

    monkeypatch.setattr("builtins.open", spy_open)
    c = vault.anthropic_client("lowprio")
    expect_key = "placeholder-high" if mode == "env" else "placeholder-low"
    assert c.api_key == expect_key and c.credentials is None
    assert vault.client_pool_stamp(c) == {"pool": "env" if mode == "env" else "lowprio",
                                          "key_sha256_12": hashlib.sha256(expect_key.encode()).hexdigest()[:12]}
    assert vault.anthropic_api_key("lowprio") == expect_key
    assert vault.anthropic_async_client("highprio").api_key == "placeholder-high"
    assert vault.stamp_label(vault.client_pool_stamp(c)).startswith("key sha256/12 ")
    assert opened == []
    if mode == "op":
        monkeypatch.delenv("ANTHROPIC_API_KEY_LOWPRIO")
        with pytest.raises(vault.PoolUnavailable):       # lowprio never hops to ANTHROPIC_API_KEY under op mode
            vault.anthropic_client("lowprio")


# ------------------------------------------------------------------------------------------- the graders: keyless stamps


def test_the_live_grader_under_wif_builds_a_federated_client_through_the_vault(wif_mode, tmp_path, monkeypatch):
    """static / padding / BoN / multi-turn: ``LiveStrongREJECTGrader(org=)`` -> ``grade_bon_live.make_client`` ->
    ``vault.anthropic_client`` -> the federated client; ``pool_stamp`` / ``run_stats`` carry the keyless stamp the drivers
    write as ``api_pool``; the google token provider is stubbed (no metadata server in a test)."""
    monkeypatch.setattr(vault, "google_identity_token", wif_mode.provider)
    from sisteer.eval.attacks import grading_live as gl

    g = gl.LiveStrongREJECTGrader(model="claude-haiku-4-5", org="lowprio", concurrency=3, cache_path=tmp_path / "c.jsonl")
    inner = g._client._inner if hasattr(g._client, "_inner") else getattr(g._client, "_client", None)
    _assert_federated(inner)
    assert g.pool_stamp == {"pool": "lowprio", "auth": "wif", "ids_sha12": wif_mode.sha12}
    rs = g.run_stats()
    assert rs["org"] == "lowprio" and rs["auth"] == "wif" and rs["ids_sha12"] == wif_mode.sha12 and "key_sha256_12" not in rs
    assert json.dumps(rs).count("google-id-token") == 0
    assert wif_mode.reads == []


def test_the_grader_takes_the_prebuilt_client_and_never_builds_its_own(tmp_path, monkeypatch):
    _env_dir(tmp_path, monkeypatch)
    from sisteer.eval.attacks import grading_live as gl

    gl._ensure_scripts_on_path()
    import grade_bon_live

    def boom(*a, **k):
        raise AssertionError("make_client must not run when a client is handed in")

    monkeypatch.setattr(grade_bon_live, "make_client", boom)
    # under wif a prebuilt client runs the SDK's retries OFF (WIF round 11: an SDK-internal retry sleeps and re-sends below the
    # run's credential latch) -- which every client the vault builds there does, whatever count it asked for (round 12 on
    # fd06740ae: the count is the vault's latched layer's budget), so a vault-built client with the default retries is accepted;
    # a client built OUTSIDE the vault with the SDK's retries on is refused by name, never rebuilt
    with_default = vault.anthropic_client_wif("lowprio", token_provider=lambda: "tok")
    assert with_default.max_retries == 0 and vault.federated_retry_budget(with_default) == 5
    gl.LiveStrongREJECTGrader(model="claude-haiku-4-5", org="lowprio", concurrency=7, cache_path=tmp_path / "c0.jsonl", client=with_default)
    foreign = anthropic.Anthropic(credentials=with_default.credentials, base_url=vault.ANTHROPIC_BASE_URL, max_retries=5)
    monkeypatch.setenv(vault.SECRETS_MODE_ENV, "wif")                 # this test builds through the direct constructor; the refusal is the MODE's
    with pytest.raises(ValueError, match="under SISTEER_SECRETS_MODE=wif the prebuilt client carries SDK max_retries=5"):
        gl.LiveStrongREJECTGrader(model="claude-haiku-4-5", org="lowprio", concurrency=7, cache_path=tmp_path / "c00.jsonl", client=foreign)
    monkeypatch.setenv(vault.SECRETS_MODE_ENV, "op")                  # under a keyed mode a prebuilt client is taken as handed (byte-identity)
    assert gl.LiveStrongREJECTGrader(model="claude-haiku-4-5", org="lowprio", concurrency=7, cache_path=tmp_path / "c01.jsonl", client=foreign)._client._inner is foreign
    monkeypatch.delenv(vault.SECRETS_MODE_ENV)
    client = vault.anthropic_client_wif("lowprio", token_provider=lambda: "tok", max_retries=0)
    g = gl.LiveStrongREJECTGrader(model="claude-haiku-4-5", org="lowprio", concurrency=7, client=client,
                                  cache_path=tmp_path / "cache.jsonl")
    inner = getattr(g._client, "_inner", None) or getattr(g._client, "_client", None) or getattr(g._client, "client", None)
    assert inner is client
    assert g.pool_stamp == vault.wif_pool_stamp("lowprio")          # the provenance stamp the driver records: keyless
    # without a client the grader builds one through make_client (the vault path) — pinned so the branch stays
    built = []
    monkeypatch.setattr(grade_bon_live, "make_client", lambda org, max_retries: built.append((org, max_retries)) or client)
    gl.LiveStrongREJECTGrader(model="claude-haiku-4-5", org="lowprio", cache_path=tmp_path / "c2.jsonl")
    assert built == [("lowprio", 2)]                                 # the base's count (8 clamped to 2) in EVERY mode (round 12); under wif the vault turns the SDK's own retries off


def test_grade_bon_live_make_client_under_wif_is_federated_and_its_status_line_needs_no_key(wif_mode, monkeypatch):
    monkeypatch.setattr(vault, "google_identity_token", wif_mode.provider)
    sys.path.insert(0, str(REPO / "scripts"))
    import grade_bon_live

    c = grade_bon_live.make_client("lowprio")
    st = _assert_federated(c)
    assert c.max_retries == 0                                     # BoN: SDK retries OFF, as before
    api_pool = {**vault.client_pool_stamp(c), "concurrency": 4}
    assert vault.stamp_label(api_pool) == f"auth wif ids sha12 {st['ids_sha12']}"      # the line main prints; no KeyError on key_sha256_12
    assert wif_mode.reads == []


def test_misalign_judge_pool_client_under_wif_is_federated_and_records_the_keyless_stamp(wif_mode, monkeypatch):
    """agentic: ``PoolClient`` builds through the vault — no secret-manager subprocess of its own, no env override read,
    the provenance's ``api_pool`` keyless; a pool outside POOLS still refuses."""
    monkeypatch.setattr(vault, "google_identity_token", wif_mode.provider)
    spec = importlib.util.spec_from_file_location("misalign_judge_wif_under_test",
                                                  REPO / "scripts/sdfbase/full22/misalign/misalign_judge.py")
    mj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mj)
    assert not hasattr(mj, "pool_key") and not hasattr(mj, "OP_SECRET"), "the private key plumbing is gone"
    client = mj.PoolClient("lowprio")
    _assert_federated(client.inner)
    assert client.inner.max_retries == 0 and vault.federated_retry_budget(client.inner) == 1 and float(client.inner.timeout) == mj.JUDGE_TIMEOUT_S   # round 12
    assert client.pool_stamp() == {"pool": "lowprio", "auth": "wif", "ids_sha12": wif_mode.sha12}
    client.switch("highprio")
    assert client.pool_stamp()["pool"] == "highprio" and client.pool == "highprio"
    with pytest.raises(ValueError):
        mj.PoolClient("batch")
    src = (REPO / "scripts/sdfbase/full22/misalign/misalign_judge.py").read_text()
    assert '"api_pool": client.pool_stamp()' in src                # judge_provenance.json carries it
    assert "anthropic.Anthropic(" not in src                        # the SDK is never constructed here
    assert wif_mode.reads == []


def test_misalign_judge_pool_client_refuses_by_name_when_the_vault_cannot_serve_the_pool(wif_mode, monkeypatch):
    monkeypatch.setattr(vault, "google_identity_token", wif_mode.provider)
    spec = importlib.util.spec_from_file_location("misalign_judge_wif_refusal_under_test",
                                                  REPO / "scripts/sdfbase/full22/misalign/misalign_judge.py")
    mj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mj)
    (wif_mode.dir / LO).write_text("")
    with pytest.raises(SystemExit) as ei:
        mj.PoolClient("lowprio")
    assert "no credential for the lowprio pool" in str(ei.value) and f"{LO} is empty" in str(ei.value)


def test_claude_judge_under_wif_builds_the_anthropic_pools_federated_and_leaves_openrouter_keyed(wif_mode, tmp_path, monkeypatch):
    """controls: ``ClaudeJudge._client`` for an Anthropic pool is the vault's (federated under wif, ``_pool_key`` never
    consulted); the openrouter backend is not a Claude org and keeps its own keyed client."""
    monkeypatch.setattr(vault, "google_identity_token", wif_mode.provider)
    from sisteer.capeval import judging as j

    def no_pool_key(pool):
        raise AssertionError(f"_pool_key({pool!r}) must not be consulted for an Anthropic pool under wif")

    monkeypatch.setattr(j, "_pool_key", no_pool_key)
    monkeypatch.setenv(j.JUDGE_BASE_URL_ENV, "https://api.anthropic.com/")
    judge = j.ClaudeJudge(cache=j.JudgeCache(tmp_path / "c.sqlite"))
    client = judge._client("lowprio")
    _assert_federated(client)
    assert client.max_retries == 0 and vault.federated_retry_budget(client) == j.MAX_RETRIES and float(client.timeout) == j.REQUEST_TIMEOUT_S   # round 12
    assert judge._client("lowprio") is client                                  # built once
    assert wif_mode.reads == []
    # the controls driver's own count-tokens client is the same vault path
    sys.path.insert(0, str(REPO / "scripts"))
    import capeval_judged_run as cjr

    _assert_federated(cjr._count_tokens_client("lowprio"))
    # openrouter: keyed, through _pool_key, as before
    monkeypatch.setattr(j, "_pool_key", lambda pool: "placeholder-openrouter")
    orj = j.ClaudeJudge(cache=j.JudgeCache(tmp_path / "o.sqlite"), model="x-ai/grok-4.6", backend="openrouter")
    oc = orj._client("openrouter")
    assert oc.api_key == "placeholder-openrouter" and str(oc.base_url).rstrip("/") == j.OPENROUTER_BASE_URL


def test_claude_judge_refuses_by_name_when_the_vault_cannot_serve_the_pool(wif_mode, tmp_path, monkeypatch):
    from sisteer.capeval import judging as j

    (wif_mode.dir / LO).unlink()
    judge = j.ClaudeJudge(cache=j.JudgeCache(tmp_path / "c.sqlite"))
    with pytest.raises(j.JudgingError, match=f"judge pool 'lowprio'.*{LO} does not exist"):
        judge._client("lowprio")


def test_judge_sonnet_under_wif_is_federated_and_its_run_stats_are_keyless(wif_mode, tmp_path, monkeypatch):
    monkeypatch.setattr(vault, "google_identity_token", wif_mode.provider)
    sys.path.insert(0, str(REPO / "scripts"))
    import judge_sonnet as js

    judge = js.Judge(tmp_path / "cache.jsonl", concurrency=2, thinking=False)
    _assert_federated(judge.client)
    rs = judge.run_stats()
    assert rs["pool"] == "lowprio" and rs["auth"] == "wif" and rs["ids_sha12"] == wif_mode.sha12 and "key_sha256_12" not in rs
    assert vault.stamp_label(rs).startswith("auth wif ids sha12 ")
    assert wif_mode.reads == []


def test_the_static_graders_resolve_no_credential_of_their_own_and_record_the_pool_stamp():
    """Static pins: the three static graders and the shared body carry no secret-manager shim and no ``load_key`` seam
    (dead since ``make_client`` took the org; under wif a live one would have been a call to a revoked key), and each
    writes the grader's pool stamp as ``api_pool`` beside an ``auth`` word."""
    files = {
        "scripts/sdfbase/full22/sampled/grade_static_sampled.py": ('"api_pool": grader.pool_stamp', "org=args.org"),
        "src/sisteer/eval/static_grading.py": ('"api_pool": grader.pool_stamp', "org=args.org"),
        "scripts/sdfbase/full22/static_padding/grade_padded_static.py": ("sg.Lane", "PaddedStaticLane"),
        "scripts/sdfbase/seedrep/grade_full22_native.py": ('"api_pool": api_pool', "api_pool = grader.pool_stamp"),
    }
    for rel, needles in files.items():
        src = (REPO / rel).read_text()
        for bad in ("_op_secret", "_load_key_lowprio", "_load_key_patched", "OLD_KEY_PLUMBING", "def load_key", ".load_key =",
                    "OP_SECRET", "key via "):
            assert bad not in src, f"{rel}: {bad!r} is back"
        for needle in needles:
            assert needle in src, f"{rel}: {needle!r} missing"
    scj = (REPO / "scripts/served_capability_judge.py").read_text()
    assert 'prov["judge_api_pool"] = vault.client_pool_stamp(client)' in scj and "vault.anthropic_client(args.org" in scj
    # the Arditi OR-Bench wrapper hands a KEY to a frozen driver that reads ANTHROPIC_API_KEY and nothing else: it asks
    # the vault for it (refused BY NAME under wif, never the shell's sentinel) instead of its own secret-manager subprocess.
    # Pinned on the source only: the module imports `grade_arditi_static`, a sibling that is not present on this
    # branch, so it cannot be executed here.
    arditi = (REPO / "scripts/arditi/grade_arditi_orbench.py").read_text()
    assert "vault.anthropic_api_key(JUDGE_ORG)" in arditi and "OP_SECRET" not in arditi
    assert "except vault.VaultSecretError" in arditi and "subprocess.run([OP_SECRET" not in arditi


# ------------------------------------------------------------------------------------------------------ the shell half


def _bash(script: str, env: dict) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=env, timeout=60)


def test_the_front_door_admits_the_mode_and_the_shell_library_refuses_a_key_under_it(tmp_path):
    lib = REPO / "reproduce/lib"
    base = {k: v for k, v in os.environ.items()
            if not k.startswith(("ANTHROPIC_", "SISTEER_", "REPRO_"))}      # never the host's site config or its mode
    base["PATH"] = "/usr/bin:/bin"
    base["HOME"] = str(tmp_path / "home"); (tmp_path / "home").mkdir()  # no ~/.config/sisteer to fall back to
    # common.sh: a site config naming wif loads (it used to refuse with 'must be op or env'); a fourth value still refuses.
    # SISTEER_CONFIG names the file (common.sh resets REPRO_CONFIG at source time: that one is --config's).
    site = write_site(tmp_path, secrets_mode="wif")
    r = _bash(f"source {lib}/common.sh; repro_load_config; echo mode=$SISTEER_SECRETS_MODE", dict(base, SISTEER_CONFIG=str(site / "config.env")))
    assert r.returncode == 0 and r.stdout.strip() == "mode=wif", r.stdout + r.stderr
    (site / "config.env").write_text((site / "config.env").read_text().replace("SISTEER_SECRETS_MODE=wif", "SISTEER_SECRETS_MODE=federation"))
    r = _bash(f"source {lib}/common.sh; repro_load_config", dict(base, SISTEER_CONFIG=str(site / "config.env")))
    assert r.returncode == 2 and "SISTEER_SECRETS_MODE must be op, env or wif" in r.stderr
    # secrets.sh under wif: no key to print (refused by name, the secret-manager CLI never consulted: none is
    # named here), the unit key line only scrubs, the unit flags carry the MODE and no credential (a detached unit inherits
    # nothing and its vault reads the mode off its own environment -- `[end]` alone here was the defect: the unit ran under op)
    e = dict(base, SISTEER_SECRETS_MODE="wif", ANTHROPIC_API_KEY=SENTINEL)
    for org in ("lowprio", "highprio", "batch"):
        r = _bash(f"source {lib}/secrets.sh; secret_anthropic {org}", e)
        assert r.returncode == 1 and r.stdout == "" and "SISTEER_SECRETS_MODE=wif" in r.stderr and f"the {org} pool has NO API key" in r.stderr
        assert SENTINEL not in r.stderr
    r = _bash(f"source {lib}/secrets.sh; secrets_unit_key_line lowprio; echo; secrets_unit_flags; echo '[end]'", e)
    line, flags = r.stdout.splitlines()[0], r.stdout.splitlines()[1]
    assert line == vault.WIF_SHELL_SCRUB == 'unset "${!ANTHROPIC_@}"' and "SISTEER_SECRET_CMD" not in line and "export" not in line   # round 10: every name
    assert flags == "--setenv=SISTEER_SECRETS_MODE=wif[end]" and "ANTHROPIC" not in flags and SENTINEL not in flags
    # the line RUN in a shell carrying every name the sweep knows (the key, a bearer, a gateway, the per-org keys, the SDK's custom
    # headers and webhook variables, a profile): nothing with the prefix survives, an unrelated name does, no value is printed
    carried = dict(e, **{k: ("X-Api-Key: " + SENTINEL if k == "ANTHROPIC_CUSTOM_HEADERS" else SENTINEL) for k in vault.WIF_INHERITED_ENV_NAMES},
                   HF_HUB_OFFLINE="1")
    r = _bash(f"source {lib}/secrets.sh; eval \"$(secrets_unit_key_line lowprio)\"; compgen -A variable | grep '^ANTHROPIC_' || echo NONE; echo HF=$HF_HUB_OFFLINE", carried)
    assert r.returncode == 0 and r.stdout.split() == ["NONE", "HF=1"] and SENTINEL not in r.stdout + r.stderr, r.stdout + r.stderr
    # op mode: the export line through the secret-manager CLI the unit's environment names, as test_vault_env_mode_shell.py pins it
    r = _bash(f"source {lib}/secrets.sh; secrets_unit_key_line lowprio", dict(base, SISTEER_SECRETS_MODE="op"))
    assert r.stdout.startswith("export ANTHROPIC_API_KEY_LOWPRIO=")


def test_the_site_example_documents_the_mode():
    cfg = (REPO / "reproduce/site.example/config.env").read_text()
    assert "SISTEER_SECRETS_MODE=env" in cfg                       # the shipped example's value is unchanged
    assert "wif: workload identity federation" in cfg and "WIF_ORG_FILES" in cfg and "ids_sha12" in cfg
    assert "SISTEER_WIF_ENV_DIR" in cfg
    assert "NO Anthropic API key" in cfg
    assert "SISTEER_SECRET_CMD" in cfg                              # op mode's secret-manager CLI is a setting, with no default


def test_under_wif_on_modal_the_vault_refuses_by_name_unless_a_token_provider_is_injected(wif_mode, monkeypatch):
    """The GCP identity-token path cannot work inside a Modal container, so the vault refuses BY NAME
    there (the container marker the codebase reads elsewhere, MODAL_TASK_ID) instead of letting the SDK fail opaquely at the first request;
    an injected token provider -- the seam a Modal constructor will use -- passes; the digest (no token involved) resolves anywhere."""
    monkeypatch.setenv(vault.MODAL_MARKER_ENV, "ta-synthetic")
    assert vault.on_modal() is True and vault.on_modal({}) is False
    with pytest.raises(vault.PoolUnavailable, match="inside a Modal container") as ei:
        vault.anthropic_client("lowprio")
    assert ei.value.pool == "lowprio" and "MODAL_TASK_ID" in str(ei.value) and "seedrep_petri_drive.py" in str(ei.value)
    with pytest.raises(vault.PoolUnavailable, match="inside a Modal container"):
        vault.anthropic_async_client("highprio")
    with pytest.raises(vault.PoolUnavailable, match="inside a Modal container"):
        vault.wif_credentials("lowprio")
    _assert_federated(vault.anthropic_client_wif("lowprio", token_provider=wif_mode.provider))            # the injected provider's seam
    assert vault.wif_ids_sha12("lowprio") == wif_mode.sha12 and vault.wif_pool_stamp("lowprio")["ids_sha12"] == wif_mode.sha12
    monkeypatch.delenv(vault.MODAL_MARKER_ENV)
    _assert_federated(vault.anthropic_client("lowprio"))                                                   # off Modal: the default provider
    assert wif_mode.reads == []


def test_the_ids_file_resolves_under_another_processes_environment_and_a_digest_names_its_org(tmp_path, monkeypatch):
    """The grading daemon computes the digest a Petri drive must attest under the environment it HANDS the drive (its SISTEER_WIF_ENV_DIR /
    HOME), and the Petri reader validates an attested digest by asking which org's current file it is -- both through the vault, never a
    second reader of the ids files."""
    d = _env_dir(tmp_path, monkeypatch)
    other = tmp_path / "other"; other.mkdir(); (other / LO).write_text("".join(f"{k}={v}-other\n" for k, v in IDS.items()))
    assert vault.wif_env_path("lowprio") == str(d / LO)
    assert vault.wif_env_path("lowprio", env={vault.WIF_ENV_DIR_ENV: str(other)}) == str(other / LO)
    assert vault.wif_env_path("highprio", env={"HOME": str(tmp_path / "h")}) == str(tmp_path / "h" / ".anthropic" / HI)   # no dir var: $HOME/.anthropic
    assert vault.wif_ids_sha12("lowprio", env={vault.WIF_ENV_DIR_ENV: str(other)}) == _file_sha12(other / LO) != vault.wif_ids_sha12("lowprio")
    with pytest.raises(vault.PoolUnavailable, match="does not exist"):
        vault.wif_ids_sha12("highprio", env={vault.WIF_ENV_DIR_ENV: str(other)})
    # which org a digest proves: the file that digests to it, under this process's environment or another's; the empty lowprio file never matches
    assert vault.wif_org_of_ids_sha12(_file_sha12(d / HI)) == "highprio" and vault.wif_org_of_ids_sha12(_file_sha12(d / LO)) == "lowprio"
    assert vault.wif_org_of_ids_sha12(_file_sha12(d / "wif-lowprio.env")) is None and vault.wif_org_of_ids_sha12("0" * 12) is None
    assert vault.wif_org_of_ids_sha12(_file_sha12(other / LO), env={vault.WIF_ENV_DIR_ENV: str(other)}) == "lowprio"
    assert vault.wif_org_of_ids_sha12(_file_sha12(other / LO)) is None


# ------------------------------------------------------------------------------------------- round 2 on 4574e103c
def test_under_wif_the_judge_base_url_override_is_refused_by_name_and_the_pin_holds(wif_mode, tmp_path, monkeypatch):
    """A federated client's bearer is minted for the API host, so under wif CAPEVAL_JUDGE_BASE_URL -- the deliberate op / env
    override seam -- must not move the controls judge's or the driver's count-tokens client to another host (a prior local
    gateway, a mock). Unset: the pin. Spelling the pinned host (trailing slash aside): the pin. Another host: a JudgingError by
    name BEFORE any client is built -- never ignored silently, the variable is deliberate. The in-process seam that stays is
    ClaudeJudge(clients=); _count_tokens_client(base_url=) naming another host is refused at the vault's constructor since round 12.
    op and env keep the override exactly as before."""
    monkeypatch.setattr(vault, "google_identity_token", wif_mode.provider)
    from sisteer.capeval import judging as j
    sys.path.insert(0, str(REPO / "scripts"))
    import capeval_judged_run as cjr

    monkeypatch.delenv(j.JUDGE_BASE_URL_ENV, raising=False)
    assert j.judge_base_url() == j.DEFAULT_JUDGE_BASE_URL
    judge = j.ClaudeJudge(cache=j.JudgeCache(tmp_path / "c.sqlite"))
    assert str(judge._client("lowprio").base_url).rstrip("/") == j.DEFAULT_JUDGE_BASE_URL
    assert str(cjr._count_tokens_client("lowprio").base_url).rstrip("/") == j.DEFAULT_JUDGE_BASE_URL
    monkeypatch.setenv(j.JUDGE_BASE_URL_ENV, "https://api.anthropic.com/")                     # the pin's own spelling passes
    assert j.judge_base_url() == j.DEFAULT_JUDGE_BASE_URL
    for i, override in enumerate(("http://127.0.0.1:8787/t/mock-gateway", "https://example.test", "http://localhost:9/judge-mock")):
        monkeypatch.setenv(j.JUDGE_BASE_URL_ENV, override)
        with pytest.raises(j.JudgingError, match="CAPEVAL_JUDGE_BASE_URL") as ei:
            j.judge_base_url()
        assert "SISTEER_SECRETS_MODE=wif" in str(ei.value) and j.DEFAULT_JUDGE_BASE_URL in str(ei.value) and override in str(ei.value)
        fresh = j.ClaudeJudge(cache=j.JudgeCache(tmp_path / f"o{i}.sqlite"))
        with pytest.raises(j.JudgingError, match="CAPEVAL_JUDGE_BASE_URL"):
            fresh._client("lowprio")
        assert fresh._clients == {}                                                           # refused before any client was built
        with pytest.raises(j.JudgingError, match="CAPEVAL_JUDGE_BASE_URL"):
            cjr._count_tokens_client("lowprio")
    # round 12 on fd06740ae: an in-process base_url= naming another host is no seam under wif either -- the vault's constructor
    # refuses it BY NAME (the bearer must not reach that host); the pin's own spelling passes; a test injects an http_client over a
    # mock transport instead (tests/test_wif_federated_destination_and_sdk_retries.py). An injected client is never rebuilt, and no
    # org variable was read anywhere above
    with pytest.raises(vault.BaseUrlRefused, match="base_url='http://127.0.0.1:1/double'"):
        cjr._count_tokens_client("lowprio", base_url="http://127.0.0.1:1/double")
    assert str(cjr._count_tokens_client("lowprio", base_url="https://api.anthropic.com/").base_url).rstrip("/") == j.DEFAULT_JUDGE_BASE_URL
    injected = object()
    assert j.ClaudeJudge(cache=j.JudgeCache(tmp_path / "i.sqlite"), clients={"lowprio": injected})._client("lowprio") is injected
    assert wif_mode.reads == []
    # op / env: the override resolves as before (test_capeval_judging.py / test_capeval_judged_driver.py pin the built clients)
    for mode in ("op", "env"):
        monkeypatch.setenv(vault.SECRETS_MODE_ENV, mode)
        monkeypatch.setenv(j.JUDGE_BASE_URL_ENV, "http://localhost:9/judge-mock")
        assert j.judge_base_url() == "http://localhost:9/judge-mock"
        monkeypatch.delenv(j.JUDGE_BASE_URL_ENV)
        assert j.judge_base_url() == j.DEFAULT_JUDGE_BASE_URL


def _load_misalign_judge(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts/sdfbase/full22/misalign/misalign_judge.py")
    mj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mj)
    return mj


def test_pinned_base_url_is_the_vaults_one_rule_for_a_base_url_that_comes_from_outside_the_process(monkeypatch):
    """Round 3 on 545c56eda: the mode-aware rule for an OVERRIDE of a federated client's base URL -- one read from the environment or a
    config, as opposed to an in-process base_url= argument, which stays each constructor's test seam -- lives in the vault ONCE and both
    judges call it. op / env: the stripped override else the default, as every caller resolved it before. wif: the default pinned; an
    override spelling the pinned host (a trailing slash aside) is the pin; any other host is BaseUrlRefused (a VaultSecretError) naming
    the source, the override, the mode, the pinned host, the subject and the caller's hint -- never silently ignored. A fourth mode value
    is secrets_mode's refusal. Both judge_base_url functions delegate to it with the variable as the source (the source audit,
    tests/test_base_url_pin_sources.py, pins that no third reader exists)."""
    pin = vault.ANTHROPIC_BASE_URL
    for mode in ("op", "env"):
        monkeypatch.setenv(vault.SECRETS_MODE_ENV, mode)
        assert vault.pinned_base_url(None, source="X_URL") == pin
        assert vault.pinned_base_url("", source="X_URL") == pin
        assert vault.pinned_base_url("  http://localhost:9/judge-mock ", source="X_URL") == "http://localhost:9/judge-mock"
        assert vault.pinned_base_url("https://example.test", source="X_URL", default="https://other.example") == "https://example.test"
    monkeypatch.delenv(vault.SECRETS_MODE_ENV)                                              # the default mode is op
    assert vault.pinned_base_url("http://localhost:9/judge-mock", source="X_URL") == "http://localhost:9/judge-mock"
    monkeypatch.setenv(vault.SECRETS_MODE_ENV, "wif")
    assert vault.pinned_base_url(None, source="X_URL") == pin
    assert vault.pinned_base_url("https://api.anthropic.com/", source="X_URL") == pin
    assert vault.pinned_base_url(" https://api.anthropic.com ", source="X_URL") == pin
    for override in ("http://127.0.0.1:8787/t/mock-gateway", "https://example.test", "http://localhost:9/judge-mock",
                     "https://api.anthropic.com.evil.test", "https://api.anthropic.com/t/gateway"):
        with pytest.raises(vault.BaseUrlRefused) as ei:
            vault.pinned_base_url(override, source="X_URL", subject="the probe's federated client", hint="inject a client instead")
        assert isinstance(ei.value, vault.VaultSecretError)
        msg = str(ei.value)
        assert msg.startswith(f"X_URL={override!r} under {vault.SECRETS_MODE_ENV}={vault.WIF_MODE}"), msg
        assert pin in msg and "the probe's federated client" in msg and "inject a client instead" in msg
    with pytest.raises(vault.BaseUrlRefused, match="https://other.example"):             # the pin is the caller's default, whatever it is
        vault.pinned_base_url("https://api.anthropic.com", source="X_URL", default="https://other.example")
    monkeypatch.setenv(vault.SECRETS_MODE_ENV, "federation")
    with pytest.raises(vault.VaultSecretError, match="must be one of 'op', 'env' or 'wif'"):
        vault.pinned_base_url(None, source="X_URL")
    # both judges resolve CAPEVAL_JUDGE_BASE_URL through it, naming the variable as the source: one rule, two typed wrappers
    monkeypatch.setenv(vault.SECRETS_MODE_ENV, "op")
    calls = []
    real = vault.pinned_base_url

    def recording(override, **kw):
        calls.append((override, kw["source"], kw.get("default")))
        return real(override, **kw)

    monkeypatch.setattr(vault, "pinned_base_url", recording)
    monkeypatch.setenv("CAPEVAL_JUDGE_BASE_URL", "http://localhost:9/judge-mock")
    from sisteer.capeval import judging as j
    mj = _load_misalign_judge("misalign_judge_pinned_rule_under_test")
    assert j.judge_base_url() == "http://localhost:9/judge-mock" == mj.judge_base_url()
    assert calls == [("http://localhost:9/judge-mock", "CAPEVAL_JUDGE_BASE_URL", pin)] * 2


def test_under_wif_the_misalign_judge_refuses_a_judge_base_url_override_by_name_and_pins_its_pool_client(wif_mode, monkeypatch):
    """misalign_judge.judge_base_url returned CAPEVAL_JUDGE_BASE_URL unchanged in every mode and
    PoolClient handed it to vault.anthropic_client, whose federated constructor honors an explicit base_url -- under wif an inherited
    override (a prior local gateway, a mock) would have received the federated bearer and the judge content from the preflight on. Now:
    unset, the pin; the pin's own spelling, the pin; another host, a SystemExit BY NAME before any client is built (PoolClient is
    constructed before the preflight and the claims, so nothing is claimed and nothing filed); switch() keeps the pin; PoolClient(base_url=)
    naming another host is refused at the vault's constructor too (no in-process URL seam under wif; a test injects a
    transport); op / env resolve the override exactly as before."""
    monkeypatch.setattr(vault, "google_identity_token", wif_mode.provider)
    mj = _load_misalign_judge("misalign_judge_wif_pin_under_test")
    built = []
    real_pool_client = mj.pool_client

    def counting_pool_client(pool, *, base_url):
        built.append((pool, base_url))
        return real_pool_client(pool, base_url=base_url)

    monkeypatch.setattr(mj, "pool_client", counting_pool_client)
    monkeypatch.delenv(mj.JUDGE_BASE_URL_ENV, raising=False)
    assert mj.JUDGE_BASE_URL_ENV == "CAPEVAL_JUDGE_BASE_URL" and mj.DEFAULT_JUDGE_BASE_URL == vault.ANTHROPIC_BASE_URL
    assert mj.judge_base_url() == mj.DEFAULT_JUDGE_BASE_URL
    client = mj.PoolClient("lowprio")
    _assert_federated(client.inner)
    assert client.base_url == mj.DEFAULT_JUDGE_BASE_URL and str(client.inner.base_url).rstrip("/") == mj.DEFAULT_JUDGE_BASE_URL
    client.switch("highprio")
    assert str(client.inner.base_url).rstrip("/") == mj.DEFAULT_JUDGE_BASE_URL and client.pool_stamp()["pool"] == "highprio"
    monkeypatch.setenv(mj.JUDGE_BASE_URL_ENV, "https://api.anthropic.com/")                   # the pin's own spelling passes
    assert mj.judge_base_url() == mj.DEFAULT_JUDGE_BASE_URL
    assert str(mj.PoolClient("lowprio").inner.base_url).rstrip("/") == mj.DEFAULT_JUDGE_BASE_URL
    n_before = len(built)
    for override in ("http://127.0.0.1:8787/t/mock-gateway", "https://example.test", "http://localhost:9/judge-mock"):
        monkeypatch.setenv(mj.JUDGE_BASE_URL_ENV, override)
        with pytest.raises(SystemExit) as ei:
            mj.judge_base_url()
        msg = str(ei.value)
        assert "CAPEVAL_JUDGE_BASE_URL" in msg and f"{vault.SECRETS_MODE_ENV}=wif" in msg and mj.DEFAULT_JUDGE_BASE_URL in msg \
            and override in msg, msg
        with pytest.raises(SystemExit, match="CAPEVAL_JUDGE_BASE_URL"):
            mj.PoolClient("lowprio")
    assert len(built) == n_before, f"a client was built for a refused override: {built[n_before:]}"
    # round 12 on fd06740ae: PoolClient(base_url=) naming another host is no seam under wif either -- the vault's constructor refuses it
    # BY NAME before any client is built (pool_client types the vault's refusal as its SystemExit); the pin's own spelling passes
    with pytest.raises(SystemExit, match="base_url='http://127.0.0.1:1/double'"):
        mj.PoolClient("lowprio", base_url="http://127.0.0.1:1/double")
    assert len(built) == n_before + 1 and built[-1] == ("lowprio", "http://127.0.0.1:1/double")      # pool_client was asked, the vault refused, nothing built
    spelled = mj.PoolClient("lowprio", base_url="https://api.anthropic.com/")
    assert str(spelled.inner.base_url).rstrip("/") == mj.DEFAULT_JUDGE_BASE_URL and spelled.inner.api_key is None
    assert wif_mode.reads == []
    # op / env: the override resolves as before (the misalign judge's own tests stub pool_client and never set the variable)
    for mode in ("op", "env"):
        monkeypatch.setenv(vault.SECRETS_MODE_ENV, mode)
        monkeypatch.setenv(mj.JUDGE_BASE_URL_ENV, "http://localhost:9/judge-mock")
        assert mj.judge_base_url() == "http://localhost:9/judge-mock"
        monkeypatch.delenv(mj.JUDGE_BASE_URL_ENV)
        assert mj.judge_base_url() == mj.DEFAULT_JUDGE_BASE_URL


def test_under_wif_the_hf_token_source_defaults_to_op_and_the_unit_flags_carry_the_mode(tmp_path):
    """`wif` is not an HF token source: _secret_hf_from refused it, stage 01's preflight swallowed the refusal (`|| true`) and
    fetched a gated tokenizer tokenless. Under wif the default source is op (federation covers the Claude API only): a fake
    secret-manager CLI named by SISTEER_SECRET_CMD answers the two HF items; SISTEER_HF_TOKEN_SOURCE still overrides (env / none);
    secret_anthropic stays refused with the CLI present. secrets_unit_flags carries the mode and the site's ids directory -- values, never a key,
    no EnvironmentFile even when the fallback is forced -- into a detached unit."""
    lib = REPO / "reproduce/lib"
    fake = tmp_path / "bin"; fake.mkdir()
    (fake / "fake-secret-cmd").write_text('#!/bin/sh\ncase "$1" in HF) printf "%s" hf-item-placeholder-1;; "HF cluster read") printf "%s" hf-read-placeholder-2;;'
                                    ' *) echo "no such item: $1" >&2; exit 1;; esac\n')
    (fake / "fake-secret-cmd").chmod(0o755)
    base = {k: v for k, v in os.environ.items() if not k.startswith(("ANTHROPIC_", "SISTEER_", "REPRO_", "HF_", "OP_", "XDG_RUNTIME"))}
    base.update(PATH=f"{fake}:/usr/bin:/bin", HOME=str(tmp_path / "home")); (tmp_path / "home").mkdir()
    cli = fake / "fake-secret-cmd"
    e = dict(base, SISTEER_SECRETS_MODE="wif", SISTEER_SECRET_CMD=str(cli))
    r = _bash(f"source {lib}/secrets.sh; secrets_hf_default_source; secret_hf; echo; secret_hf_cluster_read", e)
    assert r.returncode == 0, r.stderr
    assert r.stdout.splitlines() == ["op", "hf-item-placeholder-1", "hf-read-placeholder-2"]
    hf_env_value = "hf-env-" + "placeholder-3"          # assembled at run time: a credential-shaped literal would trip the gitleaks hook
    assert _bash(f"source {lib}/secrets.sh; secret_hf", dict(e, SISTEER_HF_TOKEN_SOURCE="env", HF_TOKEN=hf_env_value)).stdout == hf_env_value
    r = _bash(f"source {lib}/secrets.sh; secret_hf; echo '[none]'", dict(e, SISTEER_HF_TOKEN_SOURCE="none"))
    assert r.returncode == 0 and r.stdout == "[none]\n"
    r = _bash(f"source {lib}/secrets.sh; secret_anthropic lowprio", e)                     # still no Anthropic key under wif
    assert r.returncode == 1 and r.stdout == "" and "NO API key" in r.stderr
    assert _bash(f"source {lib}/secrets.sh; secrets_hf_default_source", dict(base, SISTEER_SECRETS_MODE="op")).stdout == "op\n"
    assert _bash(f"source {lib}/secrets.sh; secrets_hf_default_source", dict(base, SISTEER_SECRETS_MODE="env")).stdout == "env\n"
    # the unit flags under wif
    assert _bash(f"source {lib}/secrets.sh; secrets_unit_flags", dict(e, ANTHROPIC_API_KEY=SENTINEL)).stdout == f"--setenv=SISTEER_SECRETS_MODE=wif --setenv=SISTEER_SECRET_CMD={cli}"
    rt = tmp_path / "rt"; rt.mkdir()
    r = _bash(f"source {lib}/secrets.sh; secrets_unit_flags", dict(e, SISTEER_WIF_ENV_DIR=str(tmp_path / "ids"), SISTEER_SYSTEMD_FORCE_ENVFILE="1", XDG_RUNTIME_DIR=str(rt)))
    assert r.stdout == f"--setenv=SISTEER_SECRETS_MODE=wif --setenv=SISTEER_WIF_ENV_DIR={tmp_path / 'ids'} --setenv=SISTEER_SECRET_CMD={cli}" and list(rt.glob("sisteer-unit-env.*")) == []
    r = _bash(f"source {lib}/secrets.sh; secrets_unit_flags", dict(e, SISTEER_WIF_ENV_DIR=str(tmp_path / "with space")))
    assert r.returncode == 1 and r.stdout == "" and "whitespace" in r.stderr


# ------------------------------------------------------------------ round 4: a federation identity the token exchange refuses


def _exchange_rejecting_client(status: int, api_type: str | None):
    """A REAL SDK client whose federation token exchange answers ``status`` -- OFFLINE: the credentials' own http client is a mock
    transport for ``/v1/oauth/token`` and the API client's transport refuses to be reached (no bearer was ever minted). This is the
    shape the SDK raises when the federation rule, the service account or the workspace no longer matches: ``WorkloadIdentityError``
    carrying ``status_code`` and the redacted body, raised from the request that needed the bearer -- NOT an ``APIStatusError``."""
    try:
        import httpx2 as httpx          # anthropic >= 1.x is built on httpx2 (the lock's SDK); 0.x on httpx -- the credentials take the SDK's own
    except ImportError:
        import httpx

    def token_endpoint(request):
        assert request.url.path.endswith("/v1/oauth/token"), request.url
        if api_type is None:
            return httpx.Response(status, text="synthetic", request=request)
        return httpx.Response(status, json={"type": "error", "error": {"type": api_type, "message": "synthetic"}}, request=request)

    def never_reached(request):
        raise AssertionError(f"the API was reached without a bearer: {request.url}")

    creds = anthropic.WorkloadIdentityCredentials(
        identity_token_provider=lambda: "header.payload.signature", federation_rule_id=IDS["ANTHROPIC_FEDERATION_RULE_ID_GCP"],
        organization_id=IDS["ANTHROPIC_ORGANIZATION_ID"], service_account_id=IDS["ANTHROPIC_SERVICE_ACCOUNT_ID"],
        workspace_id=IDS["ANTHROPIC_WORKSPACE_ID"], http_client=httpx.Client(transport=httpx.MockTransport(token_endpoint)))
    # the API client keeps the SDK's own transport: the exchange fails first, so no request ever leaves it (asserted below by the
    # exception's type -- a request that did leave would be an APIConnectionError / APIStatusError, never a WorkloadIdentityError)
    del never_reached
    return anthropic.Anthropic(credentials=creds, base_url=vault.ANTHROPIC_BASE_URL, max_retries=0)


def test_a_federation_identity_the_token_exchange_refuses_is_a_credential_rejection_and_a_driver_stops_on_it(wif_mode, monkeypatch):
    """``vault.credential_rejection`` read the SDK's 401 / 403 off an ``APIStatusError`` and the two typed classes only. Under wif
    the credential is the federation identity, and the API refuses THAT at the token exchange -- the SDK raises
    ``WorkloadIdentityError(status_code=401 | 403)``, no ``APIStatusError`` -- so a revoked federation rule slipped past every
    credential-rejection stop (the controls judge, the misalign preflight, the live graders, judge_sonnet, the finisher's probe)
    and the drivers retried a rejected credential exactly as before the move. Now the classifier reads that shape too; the
    exchange's 5xx, a transport failure with no status and a plain connection error stay the retry loops' business."""
    try:
        import httpx2 as httpx
    except ImportError:
        import httpx

    for status, api_type in ((401, "authentication_error"), (403, "permission_error")):
        client = _exchange_rejecting_client(status, api_type)
        with pytest.raises(anthropic.WorkloadIdentityError) as ei:
            client.messages.create(model="claude-sonnet-5", max_tokens=1, messages=[{"role": "user", "content": "ping"}])
        exc = ei.value
        assert not isinstance(exc, anthropic.APIStatusError) and exc.status_code == status           # the shape the classifier missed
        rej = vault.credential_rejection(exc)
        assert rej == vault.CredentialRejection(status, "WorkloadIdentityError", api_type), rej
        err = vault.JudgeCredentialRejected("lowprio", rej.status, rej.error_type, api_error_type=rej.api_error_type)
        assert "WorkloadIdentityError" in str(err) and f"HTTP {status}" in str(err) and api_type in str(err) and "wif" in str(err)
        assert "the federation ids file / the VM identity" in str(err) and "UNSCORED" not in str(err)
    # a redacted body that is not a mapping: still the status, no api type
    text_body = anthropic.WorkloadIdentityError("Token exchange failed (HTTP 401): synthetic", status_code=401, body="synthetic")
    assert vault.credential_rejection(text_body) == vault.CredentialRejection(401, "WorkloadIdentityError", None)
    # NOT this: the exchange's 5xx, a transport failure with no status, a plain connection error
    for exc in (anthropic.WorkloadIdentityError("Token exchange failed (HTTP 500): synthetic", status_code=500, body={"error": {"type": "api_error"}}),
                anthropic.WorkloadIdentityError("Failed to reach token endpoint: synthetic"),
                anthropic.APIConnectionError(request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"))):
        assert vault.credential_rejection(exc) is None, exc
    # through a REAL driver: the misalign judge's preflight stops before any claim, latching the typed error on its client
    mj = _load_misalign_judge("misalign_judge_federation_rejection_under_test")
    rejected = _exchange_rejecting_client(401, "authentication_error")
    monkeypatch.setattr(mj, "pool_client", lambda pool, base_url: rejected)
    pc = mj.PoolClient("lowprio")
    with pytest.raises(vault.JudgeCredentialRejected) as ei:
        pc.preflight("claude-sonnet-5")
    assert ei.value.status == 401 and ei.value.error_type == "WorkloadIdentityError" and ei.value.pool == "lowprio"
    assert pc.credential_rejected is ei.value and any(e["event"] == "credential_rejected" for e in pc.events)
    assert wif_mode.reads == []
