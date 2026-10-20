"""The Qwen seed-replicate battery: ``n_trials_by_view`` in the sampled-config generator
(scripts/sdfbase/full22/sampled/make_sampled_configs.py) — 1 trial per seed view (the three seeds
ARE the replication), every other TAG byte for byte the campaign's 2.

  * ``n_trials_of_tag`` reads the map for a declared replicate TAG and the campaign count otherwise;
  * with ``{qwen35_4b_s1: 1, qwen35_4b_s2: 1, qwen35_4b_s3: 1}`` the render plan drops every ``t1``
    output of those TAGs and NOTHING else moves (the other 396 outputs are the same paths in the same
    order); the s-view block renders ``trial 0 of 1`` (block + header) through the real wave config;
    a ``trial 1`` for a 1-trial TAG is refused;
  * ``load_presets`` refuses a stray TAG, a non-positive / bool count, and a count equal to the
    campaign's; the drivers' reader (``sampled_decode.load_decode_presets``) admits the key unread;
  * ``stale_outputs`` / ``--check`` name a committed ``_t1`` config that the shrunk count no longer
    renders (the known-bad input: the six ``qwen35_4b_s<s>_sampled_t1*.yaml`` on this tree until
    they are deleted) and ``--check`` exits 1 on them.

Offline; reads the committed configs and the presets file."""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "sdfbase" / "full22" / "sampled"))

import make_sampled_configs as msc  # noqa: E402
from sisteer.eval import sampled_decode as sdd  # noqa: E402

PRESETS = msc.load_presets()
SEEDREP_TAGS = msc.seedrep_tags()
ONE_TRIAL = {t: 1 for t in SEEDREP_TAGS}
#: the data file also carries the ctl-neg1 lane's six seeded control TAGs at 1 (make_sampled_configs.SEEDED_LANE_VIEWS):
#: rows for a LANE-PRIVATE roster's views, outside the production scope, so they ride beside ONE_TRIAL in the committed map only
LANE_ONE_TRIAL = {t: 1 for t in msc.lane_replicate_tags()}


def _with(ntv):
    d = copy.deepcopy(PRESETS)
    if ntv is None:
        d.pop("n_trials_by_view", None)
    else:
        d["n_trials_by_view"] = ntv
    return d


def _presets_file(tmp_path, ntv):
    d = copy.deepcopy(yaml.safe_load(msc.PRESETS.read_text()))
    if ntv is None:
        d.pop("n_trials_by_view", None)
    else:
        d["n_trials_by_view"] = ntv
    f = tmp_path / f"presets_{abs(hash(repr(ntv)))}.yaml"
    f.write_text(yaml.safe_dump(d))
    return f


def test_the_committed_presets_carry_no_map_yet_and_every_tag_reads_the_campaign_count():
    assert "n_trials_by_view" not in PRESETS or PRESETS["n_trials_by_view"] == {**ONE_TRIAL, **LANE_ONE_TRIAL}
    for tag in msc.sampled_tags():
        assert msc.n_trials_of_tag(_with(None), tag) == PRESETS["n_trials"] == 2
    assert SEEDREP_TAGS == ["qwen35_4b_s1", "qwen35_4b_s2", "qwen35_4b_s3"]
    assert msc.stale_outputs(_with(None)) == []


def test_one_trial_per_seed_view_drops_exactly_the_t1_outputs_of_those_tags_and_nothing_else():
    before = msc.outputs(_with(None))
    after = msc.outputs(_with(ONE_TRIAL))
    dropped = [o for o in before if o not in after]
    assert [(o[1], o[2], o[3]) for o in dropped] == [(t, 1, esc) for t in SEEDREP_TAGS for esc in (False, 8192, 16384)]
    # every sampled tag renders 2 trials x 3 caps (the literal 414 = 69 tags stood until the paper-companion-repo front door added its
    # sampled views — u0, s2r, s3r, s1, d0p75_relu, s2rl — so the pin reads the generator's tag count); the one-trial map drops the nine t1 outputs
    assert len(before) == 6 * len(msc.sampled_tags()) and len(after) == len(before) - 9, (len(before), len(after), len(msc.sampled_tags()))
    assert [o for o in before if o[1] not in SEEDREP_TAGS] == [o for o in after if o[1] not in SEEDREP_TAGS]
    assert all(o[2] == 0 for o in after if o[1] in SEEDREP_TAGS)
    for tag in SEEDREP_TAGS:
        assert msc.n_trials_of_tag(_with(ONE_TRIAL), tag) == 1
    assert msc.n_trials_of_tag(_with(ONE_TRIAL), "qwen35_4b") == 2 and msc.n_trials_of_tag(_with(ONE_TRIAL), "qwen35_4b_it") == 2


def test_the_s_view_renders_trial_0_of_1_through_the_real_wave_config_and_refuses_trial_1():
    p = _with(ONE_TRIAL)
    for s in (1, 2, 3):
        tag = f"qwen35_4b_s{s}"
        for esc in (False, 8192, 16384):
            block = msc.expected_block(p, tag, 0, esc)
            assert block["n_trials"] == 1 and block["trial"] == 0
            assert block["seed_base"] == msc.seed_base_of_tag(p, tag)
            ref = msc.expected_block(_with(None), tag, 0, esc)
            assert {k: v for k, v in block.items() if k != "n_trials"} == {k: v for k, v in ref.items() if k != "n_trials"}
            text = msc.render(p, tag, 0, esc)                          # _verify inside: the wave config as data + the block
            assert f", trial 0 of 1." in text and yaml.safe_load(text)["sampled_decode"] == block
            assert yaml.safe_load(text)["sampled_decode"]["n_trials"] == 1
            with pytest.raises(SystemExit, match="trial 1 is not one of the TAG's 1 trial"):
                msc.expected_block(p, tag, 1, esc)
    # every OTHER tag's render is byte-identical with and without the map
    for tag in ("qwen35_4b", "qwen35_4b_it", "gemma4e4b"):
        for trial in (0, 1):
            assert msc.render(p, tag, trial, False) == msc.render(_with(None), tag, trial, False)


def test_load_presets_refusals_and_the_drivers_reader_admits_the_key(tmp_path):
    good = msc.load_presets(_presets_file(tmp_path, ONE_TRIAL))
    assert good["n_trials_by_view"] == ONE_TRIAL
    with pytest.raises(SystemExit, match="no strays"):
        msc.load_presets(_presets_file(tmp_path, {**ONE_TRIAL, "qwen35_4b_it": 1}))
    with pytest.raises(SystemExit, match="must be a positive int"):
        msc.load_presets(_presets_file(tmp_path, {**ONE_TRIAL, "qwen35_4b_s2": 0}))
    with pytest.raises(SystemExit, match="must be a positive int"):
        msc.load_presets(_presets_file(tmp_path, {**ONE_TRIAL, "qwen35_4b_s2": True}))
    with pytest.raises(SystemExit, match="equals the campaign n_trials 2"):
        msc.load_presets(_presets_file(tmp_path, {**ONE_TRIAL, "qwen35_4b_s3": 2}))
    with pytest.raises(SystemExit, match="non-empty mapping"):
        msc.load_presets(_presets_file(tmp_path, {}))
    with pytest.raises(SystemExit, match="non-empty mapping"):
        msc.load_presets(_presets_file(tmp_path, [1]))
    # a partial map is legal (one view at 1 trial, the others at the campaign's)
    part = msc.load_presets(_presets_file(tmp_path, {"qwen35_4b_s1": 1}))
    assert msc.n_trials_of_tag(part, "qwen35_4b_s1") == 1 and msc.n_trials_of_tag(part, "qwen35_4b_s2") == 2
    # the drivers' reader passes the key through unread: the injected default block is unchanged
    d = sdd.load_decode_presets(_presets_file(tmp_path, ONE_TRIAL))
    assert d["n_trials_by_view"] == ONE_TRIAL and d["n_trials"] == 2
    assert sdd.default_sampled_block(d, "qwen35_4b", trial=0, static_cap=4096, controls_cap=512) == \
        sdd.default_sampled_block(sdd.load_decode_presets(), "qwen35_4b", trial=0, static_cap=4096, controls_cap=512)


def test_check_refuses_the_committed_t1_configs_a_shrunk_count_no_longer_renders(monkeypatch, capsys):
    """The known-bad input on THIS tree: the six qwen35_4b_s<s>_sampled_t1*.yaml exist until they are
    deleted; under a 1-trial map they are stale and --check must say so (exit 1), naming each."""
    p = _with(ONE_TRIAL)
    stale = msc.stale_outputs(p)
    committed_t1 = sorted(q for t in SEEDREP_TAGS for q in (REPO / "configs").glob(f"sdfbase_full22_{t}_sampled_t1*.yaml"))
    assert stale == committed_t1
    if not committed_t1:
        pytest.skip("WS0 has already deleted the s-view t1 configs; the stale-output check has nothing to bite on here")
    assert len(committed_t1) == 9 or len(committed_t1) == 6, committed_t1   # base + 2 rungs per view (9), or 6 if a rung set is gone
    monkeypatch.setattr(msc, "load_presets", lambda path=msc.PRESETS: p)
    rc = msc.main(["--check"])
    out = capsys.readouterr().out
    assert rc == 1
    for q in committed_t1:
        assert f"DRIFT: {q} — trial beyond the TAG's n_trials_by_view count; delete it" in out
    # the s-view t0 files on this tree still say "trial 0 of 2": under the map they DRIFT by content (the block's
    # n_trials + the header) — 9 rendered drifts + 9 stale files; every other TAG's 396 outputs are still ok
    t0 = sorted(q for t in SEEDREP_TAGS for q in (REPO / "configs").glob(f"sdfbase_full22_{t}_sampled_t0*.yaml"))
    assert len(t0) == 9
    for q in t0:
        assert f"DRIFT: {q}\n" in out
    assert out.count("DRIFT") == len(committed_t1) + len(t0) == 18
    assert out.count("\nok: ") + out.startswith("ok: ") == 414 - 9 - 9
    assert not any(f"DRIFT: {q}" in out for q in (REPO / "configs").glob("sdfbase_full22_*_sampled_t*.yaml")
                   if not any(t in q.name for t in SEEDREP_TAGS))
