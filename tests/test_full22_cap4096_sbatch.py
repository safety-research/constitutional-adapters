"""The 4,096-cap multi-turn re-run's job-side knobs, pinned at the text level: the
goal-set sbatch threads the cap / timeouts / geometry / note it is handed (and nothing when it is not), the
attacker pool takes its window positionally, common.sh appends the standup extras verbatim, and the standup
exposes the upstream-timeout flag and records it. Text checks + `bash -n`, no cluster."""

from __future__ import annotations

import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
GOALSET = REPO / "scripts/sdfbase/full22/target_attacks_goalset.sbatch"
POOL = REPO / "scripts/sdfbase/full22/attacker_pool.sbatch"
COMMON = REPO / "scripts/sdfbase/full22/common.sh"
STANDUP = REPO / "scripts/sdfbase/mt_serving_up.py"
ATTACKS = REPO / "scripts/sdfbase/full22/target_attacks.sbatch"
SUBMIT = REPO / "scripts/sdfbase/full22/submit_full22.sh"


def test_shell_syntax():
    for f in (GOALSET, POOL, COMMON):
        subprocess.run(["bash", "-n", str(f)], check=True)


def test_goalset_sbatch_threads_cap_timeouts_geometry_and_note():
    s = GOALSET.read_text()
    assert 'CAP_TOKENS="${9:-}"; GEOM_L="${10:-}"; GEOM_S="${11:-}"; CLIENT_TIMEOUT_S="${12:-}"; PROXY_UPSTREAM_TIMEOUT_S="${13:-}"' in s
    assert 'DECODE_FLAGS+=(--target-max-new-tokens "$CAP_TOKENS")' in s
    assert 'DECODE_FLAGS+=(--target-timeout-s "$CLIENT_TIMEOUT_S")' in s
    assert 'note)               DECODE_FLAGS+=(--target-decode-note "$v");;' in s
    assert 'f22_standup attacks "$SERVE_DIR" "" "" "$GEOM_L" "$GEOM_S"' in s
    # on top of the inherited launch-time hook, never a fresh array (common.sh f22_standup_extra_reset, 2026-09-16)
    assert 'f22_standup_extra_reset; [ -n "$PROXY_UPSTREAM_TIMEOUT_S" ] && F22_STANDUP_EXTRA+=(--upstream-timeout-s "$PROXY_UPSTREAM_TIMEOUT_S")' in s
    assert "F22_STANDUP_EXTRA=(" not in s
    # identity geometry: a width without its window is refused, never guessed
    assert 'max_num_seqs without max_model_len' in s
    # the completeness line reports the cap actually sent, the at-cap count and the clamp count
    for tok in ("distinct_max_tokens=", "at_cap=", "clamped=", 'row.get("max_tokens_clamped_from") is not None'):
        assert tok in s, tok
    assert "--export=" not in s.replace("never --export=", "")
    # a target that comes up before its pool member waits (bounded) instead of dying on a pool seconds from serving
    assert "ATK_WAIT_MIN=90" in s and s.index("ATK_WAIT_MIN=90") < s.index("\nf22_live_attackers\n")


def test_attacker_pool_takes_its_window_positionally_and_records_it():
    s = POOL.read_text()
    assert 'ATK_MAX_LEN="${2:-131072}"' in s      # the 4,096-cap default: Crescendo's chat holds every target reply
    assert '--max-model-len "$ATK_MAX_LEN" --prefix-caching on' in s
    assert "--max-model-len 49152 --prefix-caching" not in s      # the old hard-coded command line is gone
    assert '"max_model_len":%s' in s and '"$ATK_MAX_LEN" "$WAVE_SHA"' in s


def test_common_sh_appends_the_standup_extras_verbatim():
    """The mechanism (the math500 lane): F22_STANDUP_EXTRA is a Bash ARRAY appended to
    mt_serving_up.py's argv, so the goal-set sbatch's timeout travels as a flag, never via the env."""
    s = COMMON.read_text()
    assert '${F22_STANDUP_EXTRA[@]+"${F22_STANDUP_EXTRA[@]}"}' in s


def test_standup_exposes_the_upstream_timeout_and_writes_it_to_endpoint_json():
    s = STANDUP.read_text()
    assert 'ap.add_argument("--upstream-timeout-s", type=float, default=None' in s
    assert "st.backend_timeout_kwargs(args.upstream_timeout_s)" in s
    assert '"proxy_upstream_timeout_s": args.upstream_timeout_s' in s



def test_default_target_attacks_job_serves_an_explicit_per_attack_window_with_the_cap_and_timeouts():
    """The canonical 4,096-cap instrument: the wave's default multi-turn job resolves
    its window from geometry.multiturn_window (crescendo 49,152 / pair 16,384 -- measured maxima on complete
    stores), never the table's multiturn_attacks 8,704; reads the cap from the config; tells the standup the
    drivers' cap; raises both 600 s timeouts to 3,600 by default; and refuses an unmeasured attack (ActorAttack)
    unless the window is passed explicitly."""
    s = ATTACKS.read_text()
    assert 'GEOM_L="${8:-}"; CLIENT_TIMEOUT_S="${9:-3600}"; PROXY_UPSTREAM_TIMEOUT_S="${10:-3600}"' in s
    assert '["attacks"]["target_max_new_tokens"]' in s and "multiturn_window(sys.argv[1])" in s
    # every window branch REBUILDS the standup flags on top of the inherited hook (common.sh f22_standup_extra_reset).
    # Per LINE, not byte-exact: since F22_LOG_REQUESTS=1 opts the leg into vLLM request logging (2026-09-18) every reset
    # may end in a trailing `${F22_LOG_REQ_FLAG[@]...}`. The property is two WINDOWED standups that tell the standup the
    # drivers' cap and exactly ONE (the per-family-caps path) that names no driver cap at all -- not the absence of later flags
    resets = [l.strip() for l in s.splitlines() if l.strip().startswith('f22_standup_extra_reset --upstream-timeout-s "$PROXY_UPSTREAM_TIMEOUT_S"')]
    assert len(resets) == 3, resets
    assert sum('--driver-max-tokens "$CAP_TOKENS"' in l for l in resets) == 2, resets
    assert sum("--driver-max-tokens" not in l for l in resets) == 1, resets
    assert "F22_STANDUP_EXTRA=(" not in s
    assert 'f22_standup attacks "$SERVE_DIR" "" "" "$GEOM_L" "" || { RC=1; exit 1; }' in s
    assert '--target-max-new-tokens "$CAP_TOKENS" --target-timeout-s "$CLIENT_TIMEOUT_S"' in s
    assert "pass max_model_len explicitly (positional 8)" in s          # ActorAttack: available, never silent
    assert "--export=" not in s.replace("never --export=", "")
    g = GOALSET.read_text()
    assert 'F22_STANDUP_EXTRA+=(--driver-max-tokens "$EFFECTIVE_CAP")' in g and "multiturn_window(sys.argv[1])" in g
    # $MPY exists only after f22_driver_venv: a line that uses it earlier kills the job in ~10 s under set -u
    for script in (ATTACKS, GOALSET):
        t = script.read_text()
        venv = t.index("f22_driver_venv")
        first_mpy = min((i for i in (t.find(tok) for tok in ('"$MPY"', '${MPY}')) if i >= 0), default=len(t))
        assert first_mpy > venv, f"{script.name}: $MPY is used before f22_driver_venv defines it"
    sub = SUBMIT.read_text()
    assert 'os.environ.get("F22_ATTACKS", "crescendo,pair")' in sub and "crescendo,pair,actorattack\"" not in sub
    assert "EXPLICIT per attack" in sub
    subprocess.run(["bash", "-n", str(ATTACKS)], check=True)


def test_standup_takes_the_drivers_cap_and_records_it_with_the_server_sampling_default():
    s = STANDUP.read_text()
    assert 'ap.add_argument("--driver-max-tokens", type=int, default=None' in s
    assert "requested_max_tokens=args.driver_max_tokens" in s and "GeometryCapExceedsBasis" in s
    assert '"driver_max_tokens": args.driver_max_tokens' in s
    # the server-side sampling default is a fact of the standup, recorded beside the geometry
    assert '"override_generation_config": getattr(launch, "override_generation_config", None)' in s
