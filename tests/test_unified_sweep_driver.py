"""The unified sweep driver's no-GPU surfaces: plan, the engine-stamp sidecar
round-trip, and the aggregate's engine-matched pairing over sidecars.

The GPU generate path is validated by its leaf calls (bon_generate,
_run_gated_attack, cell_controls_run.cmd_generate) which the cost-model
measurement jobs exercise directly; here we pin the driver's OWN logic: family
reconciliation against the live registry, sidecar read/write, and that
aggregate refuses a cross-engine pair.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import pytest

from sisteer.capeval.engine_identity import EngineIdentityError
from sisteer.eval import sweep

_DRIVER = Path(__file__).resolve().parents[1] / "scripts" / "unified_sweep_run.py"


def _load_driver():
    spec = importlib.util.spec_from_file_location("unified_sweep_run", _DRIVER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_conditions(tmp_path):
    p = tmp_path / "conditions.tsv"
    p.write_text(
        "b_none\tnone\t\tall\tbon\tpair\t*\thf\tvllm\n"
        "sr_kl\tsafety_refusal__psr_kl_all\t0.25\tall\tbon\tpair\t*\thf\tvllm\n")
    return p


def test_present_absent_splits_against_live_registry():
    drv = _load_driver()
    # the full static set is now in the live registry (the static/replayable
    # families were brought onto this branch with the held-out scaffolding)
    present, absent = drv._present_absent(sweep.STATIC_FAMILIES)
    # After the stage-3 integration merge the live attacks registry carries
    # every static family the sweep enumerates (the 5 pure-static transforms +
    # the 2 sub-env families joined bon), so all 8 resolve and none are absent.
    # (Pre-merge this asserted the 7 non-bon families were absent — that state
    # was exactly what merging origin/twosi-stage3-run removed.)
    assert set(present) == set(sweep.STATIC_FAMILIES)
    assert absent == []
    assert {"bon", "pasttense", "amplegcg_plus"} <= set(present)
    # the held-out families are present too
    ph, ah = drv._present_absent(sweep.HELD_OUT_FAMILIES)
    assert ah == [] and "artprompt" in ph and "multilingual" in ph
    # the split MECHANISM still surfaces a genuinely-absent family
    p2, a2 = drv._present_absent(("bon", "no_such_family"))
    assert p2 == ["bon"] and a2 == ["no_such_family"]


def test_sidecar_round_trip(tmp_path):
    drv = _load_driver()
    cond = sweep.Condition(label="sr_kl", cell="safety_refusal__psr_kl_all",
                           coeff="0.25", static=("bon",), adaptive=("pair",),
                           benign=("mmlu",))
    drv._write_sidecar(tmp_path, cond, "adaptive", "pair", cond.adaptive_stamp(),
                       serving_identity={"chat_template_hash": "abc"})
    got = drv.read_sidecar(tmp_path, "sr_kl", "adaptive", "pair")
    assert got["engine_stamp"]["attacker_engine"] == "vllm"
    assert got["serving_identity"]["chat_template_hash"] == "abc"


def test_read_sidecar_missing_refuses(tmp_path):
    drv = _load_driver()
    with pytest.raises(SystemExit):
        drv.read_sidecar(tmp_path, "nope", "static", "bon")


NR = "b48c47f644389271"


def _kw(served=NR, trained=None, sha=None):
    return dict(served_template_hash=served, trained_template_hash=trained,
                object_sha=sha)


def test_aggregate_matches_replica_and_refuses_cross_engine(tmp_path):
    drv = _load_driver()
    # baseline replicas (family = cell id) + object, all on the NR template
    none = sweep.Condition(label="none", cell="none", static=("bon",),
                           adaptive=("pair",), benign=("mmlu",))
    prompted = sweep.Condition(label="prompted_sr", cell="prompted__safety_refusal",
                               static=("bon",), adaptive=("pair",), benign=("mmlu",))
    sr_kl = sweep.Condition(label="sr_kl", cell="safety_refusal__psr_kl_all",
                            coeff="0.25", static=("bon",), adaptive=("pair",),
                            benign=("mmlu",))
    # baselines carry baseline_family = their cell id
    drv._write_sidecar(tmp_path, none, "static", "bon",
                       none.static_stamp(**_kw()), baseline_family="none")
    drv._write_sidecar(tmp_path, none, "adaptive", "pair",
                       none.adaptive_stamp(**_kw()), baseline_family="none")
    drv._write_sidecar(tmp_path, prompted, "benign", "mmlu",
                       prompted.benign_stamp(**_kw()),
                       baseline_family="prompted__safety_refusal")
    # object rows (served NR == trained NR; not a baseline)
    drv._write_sidecar(tmp_path, sr_kl, "static", "bon",
                       sr_kl.static_stamp(**_kw(trained=NR, sha="9f789d68")))
    drv._write_sidecar(tmp_path, sr_kl, "adaptive", "pair",
                       sr_kl.adaptive_stamp(**_kw(trained=NR, sha="9f789d68")))
    drv._write_sidecar(tmp_path, sr_kl, "benign", "mmlu",
                       sr_kl.benign_stamp(**_kw(trained=NR, sha="9f789d68")))
    rows = [
        {"leg": "static", "member": "bon", "condition": "none", "metric": 0.6},
        {"leg": "static", "member": "bon", "condition": "sr_kl", "metric": 0.2},
        {"leg": "adaptive", "member": "pair", "condition": "none", "metric": 0.55},
        {"leg": "adaptive", "member": "pair", "condition": "sr_kl", "metric": 0.30},
        {"leg": "benign", "member": "mmlu", "condition": "prompted_sr", "metric": 0.89},
        {"leg": "benign", "member": "mmlu", "condition": "sr_kl", "metric": 0.91},
    ]
    rows_path = tmp_path / "rows.json"; rows_path.write_text(json.dumps(rows))
    conds_tsv = tmp_path / "conditions.tsv"
    conds_tsv.write_text(
        "none\tnone\t\tall\tbon\tpair\tmmlu\thf\tvllm\thf\t\n"
        "prompted_sr\tprompted__safety_refusal\t\tall\tbon\tpair\tmmlu\thf\tvllm\thf\t\n"
        "sr_kl\tsafety_refusal__psr_kl_all\t0.25\tall\tbon\tpair\tmmlu\thf\tvllm\thf\t\n")
    out = tmp_path / "cmp.json"
    args = argparse.Namespace(conditions=str(conds_tsv), out_root=str(tmp_path),
                              rows=str(rows_path), out=str(out))
    drv.cmd_aggregate(args)
    served = json.loads(out.read_text())["comparisons"]
    by = {(c["leg"], c["member"]): c for c in served}
    assert round(by[("static", "bon")]["delta"], 2) == -0.40
    assert by[("static", "bon")]["baseline"] == "none"
    assert round(by[("benign", "mmlu")]["delta"], 2) == 0.02
    assert by[("benign", "mmlu")]["baseline"] == "prompted__safety_refusal"
    assert by[("static", "bon")]["chat_template"] == NR

    # the ASR table is emitted beside the comparisons, breaking out the object
    # class and phase axes (never a single spanning pooled headline).
    doc = json.loads(out.read_text())
    assert "asr_table" in doc
    assert "by_object_class" in doc["asr_table"]
    assert "by_phase" in doc["asr_table"]        # the deployment-phase axis
    assert "asr" not in doc["asr_table"]  # no spanning headline
    # every row carries all axes as columns
    assert all("object_class" in r and "phase" in r
               for r in doc["asr_table"]["rows"])
    # the object rows are the safety-relevant SI class (sr_kl = safety_refusal),
    # partitioned per SI role so the placebo (si_irrelevant) can never pool with
    # it; baselines stay "baseline".
    assert "si_distilled:safety_relevant" in doc["asr_table"]["by_object_class"]
    assert "si_distilled" not in doc["asr_table"]["by_object_class"]  # never the pooled bucket
    # the conditions in this fixture are all-position (mask default "all")
    assert "all" in doc["asr_table"]["by_phase"]

    # corrupt the sr_kl adaptive sidecar to an HF attacker -> aggregate REFUSES
    drv._write_sidecar(tmp_path, sr_kl, "adaptive", "pair",
                       sweep.engine_stamp(target_engine="hf", attacker_engine="hf",
                                          served_template_hash=NR))
    with pytest.raises(EngineIdentityError):
        drv.cmd_aggregate(args)


def test_build_asr_table_breaks_out_axes_and_never_pools():
    """build_asr_table surfaces object_class + phase as columns and emits the
    pooled ASR ONLY as per-axis breakouts — never a single spanning headline
    (pooling across object classes or the phase axis averages a confound)."""
    from sisteer.eval.sweep import Comparison, build_asr_table

    def mk(member, condition, cm, bm):
        return Comparison(
            leg="held_out", member=member, condition=condition, baseline="none",
            condition_metric=cm, baseline_metric=bm, delta=cm - bm,
            target_engine="hf", attacker_engine="vllm", serving_path="hf",
            chat_template=NR, object_sha="9f789d68")

    comps = [mk("artprompt", "sr_kl", 0.2, 0.6),
             mk("multilingual", "sr_kl", 0.3, 0.55),
             mk("artprompt", "base_b", 0.5, 0.5)]
    oc = {"sr_kl": "si_distilled", "base_b": "baseline"}
    asr = build_asr_table(comps, object_class_of=lambda c: oc[c.condition])

    # every row carries the object-class axis
    assert all("object_class" in r for r in asr["rows"])
    # object class is broken out (both classes), never a single pooled headline
    assert set(asr["by_object_class"]) == {"si_distilled", "baseline"}
    assert asr["by_object_class"]["si_distilled"]["n"] == 2
    # there is NO single spanning ASR headline key
    assert "asr" not in asr and "mean_condition_metric" not in asr


def test_asr_table_phase_axis_breaks_out_all_vs_decode_only():
    """When phase_of is supplied, the ASR table breaks out the serving PHASE
    (all-position vs decode-only) as its own axis — a DISTINCT deployment axis,
    never pooled with the generalisation axes."""
    from sisteer.eval.sweep import Comparison, build_asr_table

    def mk(condition, phase, cm):
        return Comparison(
            leg="held_out", member="artprompt", condition=condition,
            baseline="none", condition_metric=cm, baseline_metric=0.5,
            delta=cm - 0.5, target_engine="hf", attacker_engine="vllm",
            serving_path="hf", chat_template=NR, object_sha="9f789d68")

    comps = [mk("sr_all", "all", 0.2), mk("sr_dec", "generated", 0.4)]
    mask = {"sr_all": "all", "sr_dec": "generated"}
    asr = build_asr_table(comps, object_class_of=lambda _c: "si_distilled",
                          phase_of=lambda c: mask[c.condition])
    assert set(asr["by_phase"]) == {"all", "generated"}
    assert asr["by_phase"]["generated"]["n"] == 1
    assert all("phase" in r for r in asr["rows"])


def test_asr_table_phase_relation_never_pools_matched_and_cross_phase():
    """matched and cross_phase answer different questions and must never be
    pooled — build_asr_table breaks them out separately, never as one headline."""
    from sisteer.eval.sweep import Comparison, build_asr_table

    def mk(condition, cm):
        return Comparison(
            leg="held_out", member="artprompt", condition=condition,
            baseline="none", condition_metric=cm, baseline_metric=0.5,
            delta=cm - 0.5, target_engine="hf", attacker_engine="vllm",
            serving_path="hf", chat_template=NR, object_sha="x")

    comps = [mk("m", 0.2), mk("x", 0.4)]
    rel = {"m": "matched", "x": "cross_phase"}
    asr = build_asr_table(comps, object_class_of=lambda _c: "si_distilled",
                          phase_relation_of=lambda c: rel[c.condition])
    assert set(asr["by_phase_relation"]) == {"matched", "cross_phase"}
    assert asr["by_phase_relation"]["matched"]["n"] == 1
    assert asr["by_phase_relation"]["cross_phase"]["n"] == 1
    assert "asr" not in asr  # no single spanning headline
    assert all("phase_relation" in r for r in asr["rows"])


def test_bon_n_max_cli_override_is_run_scoped_not_a_config_edit():
    """--bon-n-max overrides bon.n_max for THIS run only (the screening sweep's
    N=64), while the shared config value is left untouched — so configs/two_si.yaml
    is never edited out from under the two-SI / confirm lanes.
    None leaves the config value as-is."""
    mod = _load_driver()
    base = {"bon": {"n_max": 256, "schedule": [1, 2, 4], "sigma": 0.4}}

    # override applies and mutates only n_max
    cfg = mod._apply_cli_overrides({"bon": dict(base["bon"])}, bon_n_max=64)
    assert cfg["bon"]["n_max"] == 64
    assert cfg["bon"]["sigma"] == 0.4          # other bon keys untouched

    # None is a no-op: the config value stands
    cfg2 = mod._apply_cli_overrides({"bon": dict(base["bon"])}, bon_n_max=None)
    assert cfg2["bon"]["n_max"] == 256

    # the generate subparser exposes it, defaulting to None (config value)
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    g = sub.add_parser("generate")
    g.add_argument("--bon-n-max", type=int, default=None)
    assert g.parse_args([]).bon_n_max is None
    assert g.parse_args(["--bon-n-max", "64"]).bon_n_max == 64


# -- fit-phase provenance: the sidecar fact wired to the phase gate ----------- #


class _StubObj:
    pass


class _StubDefense:
    def __init__(self):
        self.obj = _StubObj()


def test_attach_fit_phase_provenance_feeds_positions_at_fit():
    """The training lane's recorded positions_at_fit, resolved with the mirror,
    is attached to the hydrated object and read back as the FIT side of the
    three-way phase check — a recorded fact replacing an inference. None (a
    record predating the field) attaches nothing, so the phase gate's
    default-deny stays in force."""
    from sisteer.eval import phase_probe
    drv = _load_driver()
    cond = sweep.Condition(label="x", cell="safety_refusal__vector",
                           mask="generated", benign=())
    d = _StubDefense()
    drv._attach_fit_phase_provenance(d, "response", cond)
    assert d.obj._prov_positions_at_fit == "response"
    assert drv._positions_at_fit(d, {}) == phase_probe.phase_of_mask("response")
    d_all = _StubDefense()
    drv._attach_fit_phase_provenance(d_all, "all", cond)
    assert drv._positions_at_fit(d_all, {}) == phase_probe.phase_of_mask("all")
    # absent record -> nothing attached -> default-deny (None) preserved
    d_none = _StubDefense()
    drv._attach_fit_phase_provenance(d_none, None, cond)
    assert drv._positions_at_fit(d_none, {}) is None


def test_resolve_serving_threads_provenance(tmp_path):
    """_resolve_serving carries provenance status/source/fit-phase from the
    sidecar into the resolution every stamp is built from — a baseline carries
    None (nothing was fit); a compiled cell without any record refuses."""
    drv = _load_driver()
    cfg = {"model": "some/base", "model_revision": None}
    base = sweep.Condition(label="b", cell="none")
    rb = drv._resolve_serving(base, cfg, str(tmp_path))
    assert rb.baseline_family == "none"
    assert rb.provenance_status is None and rb.provenance_source is None
    assert rb.positions_at_fit is None

    cell = "safety_refusal__vector"
    d = tmp_path / cell
    d.mkdir()
    (d / "object.safetensors").write_bytes(b"obj-bytes")
    (d / "train_budget_meta.json").write_text(json.dumps(
        {"model_id": "NousResearch/Meta-Llama-3.1-8B-Instruct",
         "model_revision": "d10aef7999a2b5ba950ab3974312feeedbfe0b77",
         "template_hash": "b48c47f644389271", "positions_at_fit": "response",
         # the SI-pair budget schema's cap: the positive marker every real
         # SI-pair sidecar carries (sweep.fit_kind_of — a record with neither
         # this nor training_budget_kind establishes no fit kind and refuses)
         "completion_tokens": 512}))
    cond = sweep.Condition(label="v", cell=cell, mask="generated", benign=())
    r = drv._resolve_serving(cond, cfg, str(tmp_path))
    assert r.provenance_status == "verified"
    assert r.provenance_source == "train_budget_meta"
    assert r.positions_at_fit == "response"
    assert r.trained_template_hash == "b48c47f644389271"

    bare = tmp_path / "bare"
    bare.mkdir()
    (bare / "si_constitution__vector").mkdir()
    (bare / "si_constitution__vector" / "object.safetensors").write_bytes(b"x")
    cond2 = sweep.Condition(label="v2", cell="si_constitution__vector")
    with pytest.raises(sweep.UnrecordedMirrorError):
        drv._resolve_serving(cond2, cfg, str(bare))


def test_defenses_dir_cli_governs_the_defense_build(tmp_path, monkeypatch):
    """``--defenses-dir`` must govern the defense BUILD's object resolution,
    not just ``_resolve_serving``'s provenance read. The 2026-08-26 pod smoke
    failed exactly here: the child built its registry from the repo-relative
    config default (``out/two_si/defenses/cells``) while a campaign
    ``--defenses-dir`` was passed — two resolutions of one object, and a
    missing repo-relative ``object.pt`` on any host without that layout."""
    drv = _load_driver()
    from sisteer.eval.defense_cell import (by_id, object_path_for,
                                           registry_from_cfg)

    campaign = tmp_path / "swp" / "fitserve" / "defenses"
    campaign.mkdir(parents=True)
    cell = "safety_refusal__psr_kl_all"
    cfg = {"defenses_dir": "out/two_si/defenses/cells"}
    # an exported TT_TWO_SI_DEFENSES_DIR must NOT outrank the explicit CLI dir
    monkeypatch.setenv("TT_TWO_SI_DEFENSES_DIR", str(tmp_path / "elsewhere"))

    drv._land_defenses_dir(cfg, str(campaign))

    want = object_path_for(campaign, cell)
    # the BUILD path (_build_condition_defense -> registry_from_cfg)
    built = by_id(registry_from_cfg(cfg, repo_root=drv.REPO))[cell].object_path
    assert built == want
    # the _resolve_serving FALLBACK path (jr._defense_object_path) agrees —
    # the record and the served tensors resolve from ONE place
    assert drv.jr._defense_object_path(cfg, cell) == want
    # and neither resolves the repo-relative default any more
    assert "out/two_si/defenses/cells" not in str(built)


def test_land_defenses_dir_empty_is_a_noop_and_relative_lands_absolute(tmp_path, monkeypatch):
    drv = _load_driver()
    cfg = {"defenses_dir": "out/two_si/defenses/cells"}
    drv._land_defenses_dir(cfg, "")
    assert cfg["defenses_dir"] == "out/two_si/defenses/cells"  # untouched
    # a RELATIVE CLI dir lands rooted at the invoking cwd (matching how
    # _resolve_serving would have consumed it), so rule 1 still fires
    monkeypatch.chdir(tmp_path)
    drv._land_defenses_dir(cfg, "rel/campaign")
    from pathlib import Path as _P
    assert _P(cfg["defenses_dir"]).is_absolute()
    assert cfg["defenses_dir"] == str(tmp_path / "rel" / "campaign")
