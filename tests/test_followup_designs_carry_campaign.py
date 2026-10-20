"""Follow-up designs carry the registry campaign; `trial --campaign` is the spec's fallback for replays.

Stage E (paper-companion-repo, 2026-10-01): every follow-up design (refine round 2, exploration replicate round 3,
confirmation round 9) is built on `_art_base_payload`, which copied `search_id` from the analyze artefact but NOT
`campaign`. The trial-side `_record_trial` requires both, so every follow-up trial on every cluster logged "the design
carries no search_id / campaign — fit_trial not recorded" and the close (`--close-search`) found no fit_trial
evidence for r2_t00..02 / r3_t00 and returned without closing the search. Two fixes, pinned here:
(1) `_art_base_payload` inherits the artefact's `campaign`; (2) `tune_arm.py trial --campaign C` supplies the
campaign to a design that carries none (the replay of an already-run follow-up trial writes its fit_trial record
without retraining), refuses a design naming another campaign, and is a no-op otherwise.
"""
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))
if str(REPO / "tests") not in sys.path:
    sys.path.insert(0, str(REPO / "tests"))
from test_hpo_confirmation_split import _cfg, _emit, _refine_summaries, planted, tune_arm  # noqa: E402
from test_hpo_design_v2 import analyze_v2, build, write_summaries  # noqa: E402

CAMPAIGN = "two_si"
SID = "hpo-native:feedfacefeedface"


def _design_round_with_campaign(tmp_path, cfg):
    p = build(tmp_path)
    p["config_sha"] = tune_arm._config_file_sha(cfg)
    p["search_id"] = SID
    p["campaign"] = CAMPAIGN
    (tmp_path / "r1_trials.json").write_text(json.dumps(p))
    write_summaries(tmp_path, p)
    art, rc = analyze_v2(tmp_path, allowed=tmp_path / "allowed_pairs.json")
    assert rc is None
    assert art["search_id"] == SID and art["campaign"] == CAMPAIGN, "the artefact pools the design's campaign"
    return p, art


def test_refine_design_inherits_the_artefacts_campaign(tmp_path):
    cfg = _cfg(tmp_path)
    _design_round_with_campaign(tmp_path, cfg)
    r2 = _emit(tmp_path, cfg, out="r2_trials.json", refine_from=str(tmp_path / "r1_fit.json"))
    assert r2["round"] == 2 and r2["search_id"] == SID
    assert r2["campaign"] == CAMPAIGN, "a refine design without the campaign records no fit_trial (stage E, 2026-10-01)"
    # and the trial side records it: search_id AND campaign present
    calls = []

    class FakeEmit:
        @staticmethod
        def record_trial(sid, tid, params, **kw):
            calls.append((sid, tid, kw))

    orig = tune_arm._emit_hpo
    tune_arm._emit_hpo = lambda: FakeEmit
    try:
        t = r2["trials"][0]
        out = tmp_path / "r2_t00"; out.mkdir()
        summary = {"status": "ok", "best_heldout_loss": 1.9, "trainable_params": 163880, "epochs_seen": 3,
                   "started_utc": "2026-10-01T00:00:00Z", "finished_utc": "2026-10-01T01:00:00Z", "wandb_run": None,
                   "arm": r2["arm"], "trial": t, "heldout_curve": [2.0, 1.95, 1.9, 1.9], "best_epoch": 3}
        tune_arm._record_trial(r2, t, summary, out)
    finally:
        tune_arm._emit_hpo = orig
    assert len(calls) == 1 and calls[0][0] == SID and calls[0][1] == "r2_t00" and calls[0][2]["campaign"] == CAMPAIGN


def test_exploration_replicate_design_inherits_the_artefacts_campaign(tmp_path):
    cfg = _cfg(tmp_path)
    _design_round_with_campaign(tmp_path, cfg)
    r2 = _emit(tmp_path, cfg, out="r2_trials.json", refine_from=str(tmp_path / "r1_fit.json"))
    # a monotone refine round (no interior minimum) -> the pair refit falls back to the exploration replicate
    _refine_summaries(tmp_path, r2, lambda t: math.log(1.9) - 0.02 * (math.log10(t["lr"]) - math.log10(r2["trials"][0]["lr"])))
    refit, _rc = analyze_v2(tmp_path, rounds="1,2", pair_only=True)
    assert refit["campaign"] == CAMPAIGN
    if (refit.get("next") or {}).get("action") != "explore":
        pytest.skip(f"this synthetic refine round took the {refit.get('next', {}).get('action')!r} branch; the replicate branch is pinned below by the payload helper")
    r3 = _emit(tmp_path, cfg, out="r3_trials.json", explore_from=str(tmp_path / "r1+2_fit_pair.json"))
    assert r3["round"] == 3 and r3["search_id"] == SID and r3["campaign"] == CAMPAIGN


def test_art_base_payload_copies_search_id_campaign_and_init_scheme():
    art = {"search_id": SID, "campaign": CAMPAIGN, "init_scheme": "random_lecun_1e-3", "position": "all",
           "fixed_axes": {"max_grad_norm": 1.5}, "fit": {"levels": [0.15]}}
    p = tune_arm._art_base_payload(art, round_no=2, arm="psr_ce_all", model="m", model_revision="r", config_sha="c",
                                   box={"lr": [1e-3, 1e-2]}, trials=[], si="constv2-mix-10m")
    assert (p["search_id"], p["campaign"], p["init_scheme"]) == (SID, CAMPAIGN, "random_lecun_1e-3")
    p0 = tune_arm._art_base_payload({**art, "campaign": None}, round_no=2, arm="psr_ce_all", model="m", model_revision="r",
                                    config_sha="c", box={"lr": [1e-3, 1e-2]}, trials=[], si=None)
    assert "campaign" not in p0, "an artefact without a campaign writes none (the trial side then says so)"


def test_trial_campaign_fallback_supplies_only_with_proof_refuses_or_leaves_alone(capsys, tmp_path, monkeypatch):
    """`--campaign C` supplies a campaign-less design's campaign ONLY with proof that its search belongs to C — the
    design's own search_opening, or the opening fit_search record under the C spool; the flag's word alone, a design naming another campaign
    (in `campaign` or in its search_opening), or a design with no search are refused by name."""
    # proof (a): the design carries the search_opening naming the campaign
    spec = {"round": 2, "search_id": SID, "search_opening": {"search_id": SID, "campaign": CAMPAIGN}}
    out = tune_arm.spec_with_campaign_fallback(spec, CAMPAIGN)
    assert out["campaign"] == CAMPAIGN and "campaign" not in spec, "the fallback returns a copy and says so"
    assert "proven by the design's search_opening" in capsys.readouterr().out
    # the flag's word alone: no search_opening, no record under the spool -> refused by name
    monkeypatch.setenv("SISTEER_REGISTRY_SPOOL", str(tmp_path / "spool")); monkeypatch.delenv("SISTEER_REGISTRY_DISABLED", raising=False)
    with pytest.raises(SystemExit, match="nothing proves its search belongs to"):
        tune_arm.spec_with_campaign_fallback({"round": 2, "search_id": SID}, CAMPAIGN)
    # proof (b): the opening fit_search record for the search lies under the campaign's spool (the real emitter writes it)
    from sisteer.registry import emit_hpo
    emit_hpo.open_search(campaign=CAMPAIGN, model_id="m", object_kind="lora", method="sobol", search_space={"lr": [1e-5, 1e-2]}, n_trials=4,
                         search_id=SID, code_sha="a" * 40, substrate="grading_host", spool=str(tmp_path / "spool"))
    out = tune_arm.spec_with_campaign_fallback({"round": 2, "search_id": SID}, CAMPAIGN)
    assert out["campaign"] == CAMPAIGN and f"proven by the opening fit_search record under the {CAMPAIGN} spool" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="nothing proves its search belongs to"):            # another campaign has no such record
        tune_arm.spec_with_campaign_fallback({"round": 2, "search_id": SID}, "other")
    same = {"round": 2, "search_id": SID, "campaign": CAMPAIGN}
    assert tune_arm.spec_with_campaign_fallback(same, CAMPAIGN) is same
    assert tune_arm.spec_with_campaign_fallback(spec, None) is spec
    with pytest.raises(SystemExit, match="contradicts the design's campaign"):
        tune_arm.spec_with_campaign_fallback({"round": 2, "search_id": SID, "campaign": "other"}, CAMPAIGN)
    with pytest.raises(SystemExit, match="contradicts the design's search_opening campaign"):
        tune_arm.spec_with_campaign_fallback({"round": 2, "search_id": SID, "search_opening": {"campaign": "other"}}, CAMPAIGN)
    with pytest.raises(SystemExit, match="names no search_id"):
        tune_arm.spec_with_campaign_fallback({"round": 2}, CAMPAIGN)
    # the emission evidence matches on the campaign too: a record under another campaign is other work (replayed, not accepted)
    ev = {"search_id": SID, "trial_id": "r2_t00", "fit_fingerprint": "f", "record_path": "/r", "campaign": CAMPAIGN}
    assert tune_arm.trial_record_matches(ev, search_id=SID, trial_id="r2_t00", fit_fingerprint="f", campaign=CAMPAIGN)
    assert tune_arm.trial_record_matches(ev, search_id=SID, trial_id="r2_t00", fit_fingerprint="f")                       # no campaign asked: as before
    assert not tune_arm.trial_record_matches(ev, search_id=SID, trial_id="r2_t00", fit_fingerprint="f", campaign="other")


def test_trial_subcommand_accepts_campaign():
    env = {**os.environ, "PYTHONPATH": str(REPO / "src") + os.pathsep + os.environ.get("PYTHONPATH", "")}
    r = subprocess.run([sys.executable, str(REPO / "scripts" / "tune_arm.py"), "trial", "--help"],
                       capture_output=True, text=True, env=env, timeout=300)
    assert r.returncode == 0, r.stderr[-800:]
    assert "--campaign" in r.stdout
