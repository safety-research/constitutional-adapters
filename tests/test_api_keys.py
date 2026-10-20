"""The org-key router (sisteer.eval.api_keys) — routing defaults + the
same-org AdaptiveLimiter (the 2026-08-21 rule: on 429 throttle down,
never hop orgs).

The routing tests here exist because this module once did the opposite of
what it says. It read each org's key from a ~/.config file and, when the file
was absent, built a bare SDK client — which then picked ANTHROPIC_API_KEY out
of the environment. That variable is the HIGH PRIO key, so asking for Low
Prio silently produced a High Prio client: the exact org hop the rule forbids,
invisibly. The two tests named ...never_falls_back... and ...pins_the_base_url
pin the fix; do not relax them into "some key was returned".
"""

import hashlib
import json
import threading
import time

import pytest

from sisteer import vault
from sisteer.eval.api_keys import AdaptiveLimiter, ORGS, make_client

LOW = "ORG-KEY-LOWPRIO"
HIGH = "ORG-KEY-HIGHPRIO"


@pytest.fixture(autouse=True)
def _no_vault(monkeypatch):
    """No test here may reach the real vault: the secret-manager CLI is
    replaced by a subprocess stub that always fails, so an org resolves
    ONLY from the environment variable the test sets."""
    def _boom(*a, **k):
        raise FileNotFoundError("fake-secret-cmd")
    monkeypatch.setattr(vault.subprocess, "check_output", _boom)
    for env, _item in vault.ANTHROPIC_ORGS.values():
        monkeypatch.delenv(env, raising=False)


def test_make_client_binds_the_named_org_key(monkeypatch):
    monkeypatch.setenv(vault.ANTHROPIC_ORGS["lowprio"][0], LOW)
    monkeypatch.setenv(vault.ANTHROPIC_ORGS["highprio"][0], HIGH)
    assert make_client().api_key == LOW              # default prio=lowprio
    assert make_client("lowprio").api_key == LOW
    assert make_client("highprio").api_key == HIGH


def test_make_client_never_falls_back_to_another_org(monkeypatch):
    """THE REGRESSION. ANTHROPIC_API_KEY is the High Prio key. With no Low
    Prio key anywhere, asking for Low Prio must RAISE — never quietly hand
    back the High Prio client, which is what the file-reading version did."""
    monkeypatch.setenv(vault.ANTHROPIC_ORGS["highprio"][0], HIGH)
    with pytest.raises(vault.VaultSecretError) as exc:
        make_client("lowprio")
    msg = str(exc.value)
    assert vault.ANTHROPIC_ORGS["lowprio"][0] in msg   # names what was missing
    assert HIGH not in msg                             # never carries a value
    # the refusal is TYPED and names the pool it was for
    assert isinstance(exc.value, vault.PoolUnavailable)
    assert exc.value.pool == "lowprio"


def test_make_client_carries_the_pool_stamp(monkeypatch):
    """Every client this helper returns says which pool it resolved —
    ``{"pool", "key_sha256_12"}`` — so a caller can RECORD the pool a run
    used beside its concurrency, instead of leaving it to be inferred from
    which shell launched the run. Never the key itself."""
    monkeypatch.setenv(vault.ANTHROPIC_ORGS["lowprio"][0], LOW)
    monkeypatch.setenv(vault.ANTHROPIC_ORGS["highprio"][0], HIGH)
    st = vault.client_pool_stamp(make_client("lowprio"))
    assert st == {"pool": "lowprio",
                  "key_sha256_12": hashlib.sha256(LOW.encode()).hexdigest()[:12]}
    assert LOW not in json.dumps(st) and HIGH not in json.dumps(st)
    assert vault.client_pool_stamp(make_client("highprio"))["pool"] == "highprio"
    assert (vault.client_pool_stamp(make_client("highprio"))["key_sha256_12"]
            != st["key_sha256_12"])


def test_make_client_pins_the_base_url_over_the_environment(monkeypatch):
    """The environment may export ANTHROPIC_BASE_URL pointing at a local gateway.
    A judge client that inherited it would hang instead of reaching the API,
    so the base URL is passed explicitly."""
    monkeypatch.setenv(vault.ANTHROPIC_ORGS["lowprio"][0], LOW)
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://127.0.0.1:8787/t/deadbeef")
    c = make_client("lowprio")
    assert str(c.base_url).rstrip("/") == vault.ANTHROPIC_BASE_URL


def test_make_client_sets_an_explicit_request_timeout(monkeypatch):
    monkeypatch.setenv(vault.ANTHROPIC_ORGS["lowprio"][0], LOW)
    assert make_client("lowprio").timeout == vault.REQUEST_TIMEOUT_S


def test_default_timeout_s_is_the_vault_bound_under_the_shared_name(monkeypatch):
    """``DEFAULT_TIMEOUT_S`` re-exports the vault's per-request bound under the
    name a sibling branch gave ``make_client``'s default. That branch's judge
    path built its clients through this helper and hung on the SDK's 600 s
    default exactly the way ``capeval.judging`` did (MEASURED 2026-09-11: two
    sockets ESTAB with empty send/recv queues stalled a serial judged run and
    read as pool degradation). Same number, same name, same placement, so the
    two signatures are textually identical when the branches meet — and there
    is still ONE source of truth, the vault constant, not a second literal."""
    from sisteer.eval import api_keys
    assert api_keys.DEFAULT_TIMEOUT_S == vault.REQUEST_TIMEOUT_S == 120.0
    monkeypatch.setenv(vault.ANTHROPIC_ORGS["lowprio"][0], LOW)
    assert make_client("lowprio").timeout == api_keys.DEFAULT_TIMEOUT_S


def test_make_client_forwards_an_explicit_timeout(monkeypatch):
    """A caller that sizes its own bound (a long-completion site) must get it
    on the wire — the default is a floor for the forgetful, not a clamp."""
    monkeypatch.setenv(vault.ANTHROPIC_ORGS["lowprio"][0], LOW)
    assert make_client("lowprio", timeout=7.5).timeout == 7.5


def test_make_client_rejects_an_unrouted_org(monkeypatch):
    with pytest.raises(ValueError, match="prio must be one of"):
        make_client("midprio")
    assert "batch" not in ORGS      # Batch is the Batch API's org, not a judge pool


# ------------------------------------------------------- AdaptiveLimiter


def test_limiter_halves_on_rate_limit_and_floors_at_min():
    lim = AdaptiveLimiter(8, cooldown=0.0)
    lim.on_rate_limit()
    assert lim.width == 4
    lim.on_rate_limit()
    assert lim.width == 2
    lim.on_rate_limit()
    lim.on_rate_limit()
    assert lim.width == 1                     # floors, never 0
    assert lim.n_rate_limited == 4


def test_limiter_burst_of_429s_shrinks_once_per_cooldown():
    """N workers all seeing the same rate-limit event must count as ONE
    halving, not N — otherwise a single burst collapses a 100-wide run
    straight to 1."""
    lim = AdaptiveLimiter(100)                # default cooldown (10s)
    for _ in range(50):
        lim.on_rate_limit()
    assert lim.width == 50                    # exactly one halving
    assert lim.n_rate_limited == 50           # ...but every event counted


def test_limiter_recovery_is_slow_additive_and_caps_at_launch():
    """+1 after a FULL successful window at the current reduced width —
    never a multiplicative jump back up."""
    lim = AdaptiveLimiter(8, cooldown=0.0, recover_floor=1)
    lim.on_rate_limit()                       # 8 -> 4
    for _ in range(3):
        with lim:
            pass
    assert lim.width == 4                     # 3 < window of 4: nothing yet
    with lim:
        pass
    assert lim.width == 5                     # a full window of 4 -> +1
    for _ in range(5):
        with lim:
            pass
    assert lim.width == 6                     # next window is 5 -> +1
    for _ in range(6 + 7 + 100):
        with lim:
            pass
    assert lim.width == 8                     # never past the launch width


def test_limiter_old_wave_successes_never_undo_a_shrink_200_to_100():
    """The exact reviewed scenario: 200 wide, 64 calls in flight, a 429
    shrinks to 100 — the old wave's 64 clean completions were admitted
    PRE-shrink and prove nothing about the reduced width, so they earn
    exactly nothing. Evidence gathered AT width 100 does earn."""
    lim = AdaptiveLimiter(200, cooldown=0.0, recover_floor=1)
    for _ in range(64):
        lim.__enter__()                       # the pre-shrink wave (gen 0)
    lim.on_rate_limit()                       # 200 -> 100, generation bump
    assert lim.width == 100
    for _ in range(64):
        lim.__exit__(None, None, None)        # old wave completes cleanly
    assert lim.width == 100                   # not one step of recovery
    for _ in range(99):
        with lim:
            pass
    assert lim.width == 100                   # 99 < the window of 100
    with lim:
        pass
    assert lim.width == 101                   # one FULL window at 100 -> +1


def test_limiter_repeated_bursts_and_cross_generation_stragglers():
    lim = AdaptiveLimiter(8, cooldown=0.0, recover_floor=1)
    lim.on_rate_limit()                       # burst 1: 8 -> 4 (gen 1)
    for _ in range(3):
        with lim:
            pass                              # 3 gen-1 successes (< window 4)
    lim.__enter__()                           # straggler, admitted gen 1
    lim.on_rate_limit()                       # burst 2: 4 -> 2 (gen 2)
    assert lim.width == 2
    lim.__exit__(None, None, None)            # gen-1 straggler completes
    assert lim.width == 2                     # cross-generation: earns nothing
    for _ in range(2):
        with lim:
            pass                              # a full window AT width 2
    assert lim.width == 3


def test_limiter_no_recovery_during_cooldown():
    lim = AdaptiveLimiter(8, cooldown=1000.0, recover_floor=1)
    lim.on_rate_limit()                       # 8 -> 4; cooldown window opens
    for _ in range(20):
        with lim:
            pass                              # current-gen, but inside cooldown
    assert lim.width == 4                     # zero recovery until it passes


def test_limiter_429_resets_the_recovery_streak():
    lim = AdaptiveLimiter(8, cooldown=0.0, recover_floor=1)
    lim.on_rate_limit()                       # -> 4 (window 4)
    for _ in range(3):
        with lim:
            pass                              # streak 3 of 4
    lim.on_rate_limit()                       # streak reset AND -> 2 (gen bump)
    with lim:
        pass                                  # streak 1 of window 2
    assert lim.width == 2


def test_limiter_failure_exit_earns_nothing():
    lim = AdaptiveLimiter(4, cooldown=0.0, recover_floor=1)
    lim.on_rate_limit()                       # -> 2 (window 2)
    with pytest.raises(RuntimeError):
        with lim:
            raise RuntimeError("boom")
    with lim:
        pass
    assert lim.width == 2                     # the failure did not count
    with lim:
        pass
    assert lim.width == 3                     # two CLEAN ones complete the window


def test_limiter_gates_in_flight_calls_at_current_width():
    lim = AdaptiveLimiter(2, cooldown=0.0)
    lim.__enter__()
    lim.__enter__()
    entered = threading.Event()

    def third():
        with lim:
            entered.set()

    t = threading.Thread(target=third, daemon=True)
    t.start()
    time.sleep(0.1)
    assert not entered.is_set()               # a third call waits its turn
    lim.__exit__(None, None, None)            # one slot frees
    assert entered.wait(2.0)
    lim.__exit__(None, None, None)
    t.join(2.0)


def test_limiter_rejects_a_nonpositive_width():
    with pytest.raises(ValueError, match="width must be >= 1"):
        AdaptiveLimiter(0)
