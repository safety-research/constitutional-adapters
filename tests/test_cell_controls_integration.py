"""§4.1 equivalence test vs build_defense_cell — INTEGRATION (r1 #2).

Before any control number from the engine path is trusted, the engine-served
cell must equal the lane-served cell on a REAL compiled object: identical
greedy generations on a fixed prompt set, and the same resolved
object/coefficient. This is deliberately written now and RUN LATER — it
needs (a) a CUDA device, (b) ``sisteer.capeval`` (the runners branch —
merged checkouts only), and (c) the real artifact on disk — and skips
loudly when any of the three is absent, so the CI suite stays green on this
standalone branch.

The pinned default cell is ``safety_refusal__vector_all`` — the first Class
A object completed (2026-08-19), living at the
commit-independent defenses dir
``$SHARED/exp/jailbreak/two_si_defenses/cells``. Overrides:
``SISTEER_EQUIV_CELL`` (cell id) and ``SISTEER_EQUIV_DEFENSES_DIR`` /
``TT_TWO_SI_DEFENSES_DIR`` (the defenses dir, via the seam's fallback
chain). fit_report.json is deliberately never consulted — the fitting job
rewrites it while it runs; artifact PRESENCE in the cell dir is the truth.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CELL = "safety_refusal__vector_all"


def _default_defenses_dir() -> Path:
    """The cluster's defenses dir under its shared root ($SHARED through sisteer.site); where no shared root is set the
    placeholder spelling stays, which names no directory, so the default is not used."""
    from sisteer import site as SITE
    spec = "$SHARED/exp/jailbreak/two_si_defenses/cells"
    try:
        return SITE.expand_path(spec)
    except (SITE.PlaceholderError, SITE.SiteConfigError):
        return Path(spec)


DEFAULT_DEFENSES_DIR = _default_defenses_dir()

# Benign, fixed prompt set — the equivalence probe.
PROMPTS = [
    "Name three primary colors and one thing each is often used for.",
    "Explain in two sentences why the sky appears blue.",
    "Write a short friendly greeting for a community newsletter.",
    "List four common uses for a paper clip.",
]

torch = pytest.importorskip("torch")

needs_capeval = pytest.mark.skipif(
    importlib.util.find_spec("sisteer.capeval") is None,
    reason="sisteer.capeval absent — this branch stands alone (D8); run "
           "from the integration merge with capeval-runners-evals")
needs_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="§4.1 equivalence needs a CUDA device (integration checkout, "
           "not CI)")

pytestmark = [needs_capeval]


def test_driver_contract_matches_the_real_judged_api():
    """CPU, capeval-only: the getattr/mapping contract cell_controls_run.py
    codes against — task ids, loaders, judge entry points, spool/JudgedSpec
    surfaces, the C/P/I label vocabulary, and the judging helpers — exists
    on the REAL modules of the merged checkout. Drift fails HERE, not
    mid-GPU-run."""
    import sys as _sys

    _sys.path.insert(0, str(REPO / "scripts"))
    import cell_controls_run as run
    from sisteer.capeval import judging
    from sisteer.capeval.tiers import judged

    assert run.JUDGED_TASK_OF["or_bench_hard_1k"] == judged.TASK_ORBENCH
    assert run.JUDGED_TASK_OF["xstest"] == judged.TASK_XSTEST
    assert "alpaca_eval" not in run.JUDGED_TASK_OF          # retired 2026-09-11
    assert tuple(sorted(run._CPI_LABELS)) == tuple(sorted(judged.LABEL_VALUE))
    for attr in ("or_bench_rows", "xstest_rows", "JudgedSpec",
                 "GenSpool", "spool_header", "generate", "judge_or_bench",
                 "judge_xstest"):
        assert hasattr(judged, attr), attr
    for gone in ("alpaca_rows", "judge_alpaca", "base_output_header",
                 "ensure_base_outputs", "load_base_outputs", "TASK_ALPACA"):
        assert not hasattr(judged, gone), gone
    for attr in ("ClaudeJudge", "coordination_post", "JUDGE_MODEL",
                 "ORBENCH_RUBRIC_V1", "XSTEST_RUBRIC_V1",
                 "ALPACA_PAIR_RUBRIC_V1"):
        assert hasattr(judging, attr), attr


def _cfg():
    cfg = yaml.safe_load((REPO / "configs" / "two_si.yaml").read_text())
    # SISTEER_EQUIV_SCALE: run the gate at an overridden serving
    # coefficient (the dose-response sweep's --coeff seam) — the gate must
    # compare engine vs lane at the SAME alpha the job generates at, or a
    # non-default-alpha job would fail its own gate.
    scale = os.environ.get("SISTEER_EQUIV_SCALE")
    if scale:
        cfg["scale"] = float(scale)
    return cfg


def _resolve_cell(cfg):
    from sisteer.eval.cell_controls import registry_for_controls
    from sisteer.eval.defense_cell import by_id

    override = os.environ.get("SISTEER_EQUIV_DEFENSES_DIR")
    if override is None and DEFAULT_DEFENSES_DIR.is_dir():
        override = str(DEFAULT_DEFENSES_DIR)
    reg = by_id(registry_for_controls(cfg, repo_root=REPO,
                                      defenses_dir=override))
    cell_id = os.environ.get("SISTEER_EQUIV_CELL", DEFAULT_CELL)
    if cell_id not in reg:
        pytest.skip(f"cell {cell_id!r} not in the registry")
    cell = reg[cell_id]
    if cell.object_path is None or not Path(cell.object_path).is_file():
        pytest.skip(f"no compiled artifact at {cell.object_path} — set "
                    "SISTEER_EQUIV_DEFENSES_DIR/TT_TWO_SI_DEFENSES_DIR at "
                    "the dir TrainTest handed over")
    return cell


@needs_gpu
def test_engine_served_equals_lane_served_on_the_real_object():
    from sisteer.capeval.serving.hf_engine import HFEngine
    from sisteer.eval.cell_controls import cell_serving
    from sisteer.eval.jailbreak import build_defense_cell
    from sisteer.models import build_chat_prompt, load_model_and_tokenizer

    cfg = _cfg()
    cell = _resolve_cell(cfg)

    model, tok = load_model_and_tokenizer(
        cfg["model"], revision=cfg.get("model_revision"), dtype=cfg["dtype"],
        attn_implementation=cfg.get("attn_implementation", "sdpa"))

    # the lane's serving surface (batch_size=1: per-prompt decode on BOTH
    # sides, so padding composition can never explain a divergence)
    lane = build_defense_cell(model, tok, cell,
                              scale=float(cfg.get("scale", 1.0)),
                              batch_size=1, positions="all")
    lane_out = lane.generate(PROMPTS, max_new_tokens=64, do_sample=False)

    # the engine's serving surface (§4.1)
    engine = HFEngine(cfg["model"], revision=cfg.get("model_revision"),
                      preloaded=(model, tok))
    handle, system = cell_serving(cell, cfg, engine)
    assert system is None  # a compiled cell serves an object, not an SI
    rendered = [handle.transform_prompt(build_chat_prompt(tok, p, system=None))
                for p in PROMPTS]
    engine_out = engine.generate(rendered, handle,
                                 {"max_new_tokens": 64, "do_sample": False,
                                  "batch_size": 1})
    engine_texts = [o.text for o in engine_out]

    # 1) identical greedy generations — the trust condition for every
    #    engine-path control number (r1 #2). Compared under the LANE's
    #    decode convention: generate_batch strips each decoded completion
    #    (generate.py, `.strip()` at the decode) while the engine keeps the
    #    raw decode, so a completion that hits max_new_tokens on a
    #    whitespace token diverges by that trailing whitespace alone —
    #    exactly what failed the psr_mse_all gate on 2026-08-19 (identical
    #    text up to a final "\n" at the 64-token cap). Token-level serving
    #    equivalence is what this gate exists to prove; the trim is each
    #    surface's own decode convention.
    assert [t.strip() for t in engine_texts] == [t.strip() for t in lane_out]

    # 2) identical resolved serving identity: coefficient == the
    #    pre-registered scale on both sides, and the SAME hydrated object
    #    class (load_object dispatch agrees end to end)
    assert float(handle.coeff) == float(cfg.get("scale", 1.0)) == lane.scale
    engine_obj = getattr(handle, "_obj", None) or getattr(handle, "obj", None)
    assert engine_obj is not None
    assert type(engine_obj).__name__ == type(lane.obj).__name__
    # vector-family artifacts: same layers, same tensors, same positions
    lane_sv = getattr(lane.obj, "sv", None)
    engine_sv = getattr(engine_obj, "sv", None)
    if lane_sv is not None and engine_sv is not None:
        assert set(lane_sv.vectors) == set(engine_sv.vectors)
        for layer, vec in lane_sv.vectors.items():
            assert torch.allclose(vec.float().cpu(),
                                  engine_sv.vectors[layer].float().cpu())
        assert getattr(lane.obj, "positions", None) == \
            getattr(engine_obj, "positions", None)


def test_prompted_cell_serves_the_si_as_system():
    """Additional SI-as-system coverage, never the only claim on its
    own: the prompted cell's engine-side serving is (None, SI text) and
    the rendered prompts match the lane's."""
    from sisteer.eval.cell_controls import cell_serving, registry_for_controls
    from sisteer.eval.defense_cell import by_id
    from sisteer.eval.jailbreak import resolve_instruction

    cfg = _cfg()
    reg = by_id(registry_for_controls(cfg, repo_root=REPO))
    cell = reg["prompted__safety_refusal"]
    handle, system = cell_serving(cell, cfg, engine=None)
    assert handle is None
    assert system == resolve_instruction("safety_refusal").text


def test_judge_model_flag_binds_resolved_config_and_specs(monkeypatch):
    """CPU, capeval-only: the --judge-model seam (capability battery,
    2026-08-19). An explicit judge model must reach BOTH binding sites —
    the generate-time resolved config (hashed into the cell-store key) and
    every JudgedSpec (hence every JudgeRequest cache key) — and the empty
    default must resolve to the capeval judging constant, so the flag can
    never half-bind (a store keyed to one judge model but requests keyed
    to another)."""
    import sys as _sys
    from types import SimpleNamespace

    _sys.path.insert(0, str(REPO / "scripts"))
    import cell_controls_run as run
    from sisteer.capeval import judging
    from sisteer.capeval.tiers import judged

    cfg = {"model": "m", "model_revision": None, "scale": 1.0,
           "completion_tokens": 16}
    cell = SimpleNamespace(id="none")

    resolved = run._resolved_config(cell, cfg, None, 0, "claude-haiku-4-5")
    assert resolved["judge_model"] == "claude-haiku-4-5"

    monkeypatch.setattr(
        judged, "or_bench_rows",
        lambda: [{"item_id": "orb0000-aaaaaaaa", "prompt": "p",
                  "category": "c"}])
    specs = run._judged_specs(cfg, ["or_bench_hard_1k"], {}, None, 0,
                              "claude-haiku-4-5")
    assert specs["or_bench_hard_1k"].judge_model == "claude-haiku-4-5"

    assert run._judge_model(SimpleNamespace(judge_model="")) == \
        judging.JUDGE_MODEL
    assert run._judge_model(
        SimpleNamespace(judge_model="claude-haiku-4-5")) == "claude-haiku-4-5"


def test_judge_phase_resolves_the_prompted_cells_system_for_the_spool():
    """The spool header binds system_sha (r1 #8), so the judge phase must
    rebuild specs under the SAME system the generate phase served — the
    2026-08-19 battery's prompted-cell judge refused its own spool because
    cmd_judge passed system=None unconditionally. _cell_system resolves it
    engine-free: the SI text for a prompted cell, None for baselines and
    compiled cells, and the resulting spool headers match generate's."""
    import sys as _sys

    _sys.path.insert(0, str(REPO / "scripts"))
    import cell_controls_run as run
    from sisteer.capeval.tiers import judged
    from sisteer.eval.cell_controls import registry_for_controls
    from sisteer.eval.defense_cell import by_id
    from sisteer.eval.jailbreak import resolve_instruction

    cfg = _cfg()
    reg = by_id(registry_for_controls(cfg, repo_root=REPO))
    si_text = resolve_instruction("safety_refusal").text
    assert run._cell_system(reg["prompted__safety_refusal"], cfg) == si_text
    assert run._cell_system(reg["none"], cfg) is None
    assert run._cell_system(reg["safety_refusal__psr_kl_all"], cfg) is None

    rows = [{"item_id": "orb0000-aaaaaaaa", "prompt": "p", "category": "c"}]
    gen_spec = judged.JudgedSpec(task=judged.TASK_ORBENCH, template={},
                                 domain_of={}, rows=rows, system=si_text)
    judge_spec = judged.JudgedSpec(
        task=judged.TASK_ORBENCH, template={}, domain_of={}, rows=rows,
        system=run._cell_system(reg["prompted__safety_refusal"], cfg))
    assert (judged.spool_header(gen_spec, intervention_hash="x",
                                model_revision="r")
            == judged.spool_header(judge_spec, intervention_hash="x",
                                   model_revision="r"))


def test_coeff_override_reaches_engine_spec_and_resolved_config(
        tmp_path, monkeypatch):
    """The --coeff / cfg-scale seam (dose-response sweep, 2026-08-19): an
    overridden scale must reach (a) engine.arm's coeff argument, (b) the
    AblationSpec's coeffs, and (c) the generate-time resolved config — and
    the equivalence gate's _cfg honors SISTEER_EQUIV_SCALE so the gate
    compares engine vs lane at the SAME alpha the job generates at."""
    import sys as _sys
    from types import SimpleNamespace

    _sys.path.insert(0, str(REPO / "scripts"))
    import cell_controls_run as run
    from sisteer.eval.cell_controls import cell_serving, registry_for_controls
    from sisteer.eval.defense_cell import by_id

    ddir = tmp_path / "cells"
    (ddir / "safety_refusal__psr_kl_all").mkdir(parents=True)
    (ddir / "safety_refusal__psr_kl_all" / "object.pt").write_bytes(b"x")
    cfg = {**_cfg(), "scale": 0.25}
    reg = by_id(registry_for_controls(cfg, repo_root=REPO,
                                      defenses_dir=ddir))
    cell = reg["safety_refusal__psr_kl_all"]

    captured = {}

    class FakeEngine:
        def arm(self, spec, coeff):
            captured["spec"] = spec
            captured["coeff"] = coeff
            return "handle"

    handle, system = cell_serving(cell, cfg, FakeEngine())
    assert handle == "handle" and system is None
    assert captured["coeff"] == 0.25
    assert list(captured["spec"].coeffs) == [0.25]

    resolved = run._resolved_config(SimpleNamespace(id=cell.id), cfg, None,
                                    0, "claude-haiku-4-5")
    assert resolved["scale"] == 0.25

    monkeypatch.setenv("SISTEER_EQUIV_SCALE", "0.25")
    assert _cfg()["scale"] == 0.25


def test_limit_takes_a_seeded_random_subset():
    """--limit semantics: a limited run is a
    seeded permute-then-slice of the PINNED rows — never a head-slice of
    the raw order (the xstest limit-20 smoke grabbed 20 safe rows from the
    head). Same seed + same limit => the same items for every condition;
    within-subset order restores to pinned order; seed 0 is the recorded
    battery default."""
    import sys as _sys

    _sys.path.insert(0, str(REPO / "scripts"))
    import cell_controls_run as run

    rows = [{"item_id": f"i{k:03d}", "prompt": str(k)} for k in range(50)]
    sub = run._seeded_subset(rows, 10)
    assert len(sub) == 10
    assert sub != rows[:10]                        # not a head-slice
    assert sub == run._seeded_subset(rows, 10)     # deterministic
    idx = [int(r["item_id"][1:]) for r in sub]
    assert idx == sorted(idx)                      # pinned order in-subset
    assert sub != run._seeded_subset(rows, 10, seed=1)  # seed-sensitive
    assert run._seeded_subset(rows, 0) is rows     # 0 = full, untouched
    assert run._seeded_subset(rows, 50) is rows    # limit >= n, untouched


def test_parity_n_flag_widths_and_store_key(monkeypatch):
    """The --parity-n seam (controls expansion, user ruling 2026-08-29:
    the benign-controls leg must resolve a 5 pp difference). --limit 0
    cannot widen mmlu/gsm8k — their row sets are narrowed to PARITY_N
    (200) BEFORE the limit applies — so the width needs its own plumbed
    override. Three properties, each load-bearing:

    1. flag resolution: omitted (and the judge namespace, which has no
       flag) and an explicit PARITY_N are the default instrument; 0 is
       the FULL pinned pools; N>0 is N; other negatives refuse. A
       per-task spec (gsm8k=N,mmlu=N, either probe omissible) sets each
       probe's width alone, with the SAME normalizations — PARITY_N
       drops to the default, an all-equal spec collapses to the scalar
       — so equivalent spellings can never key sibling stores.
    2. the resolved config records ONLY non-default widths — the 23
       existing default-width stores keep their harness_config_hash and
       every scalar-keyed widened store keeps its key (the per-task
       dict enters a hash only when the widths genuinely differ), while
       a widened or mixed run keys a DIFFERENT store, so two widths can
       never silently mix in one store or summary.
    3. widths NEST: the seed-0 selection is a prefix of the shuffled
       pool, item_ids included, so every narrower width (and the seeded
       --limit subset drawn from it) is a strict subset of every wider
       one — the campaign's n=100 reference numbers stay interpretable
       as a subsample of a full-width rerun.
    """
    import sys as _sys
    from types import SimpleNamespace

    _sys.path.insert(0, str(REPO / "scripts"))
    import cell_controls_run as run
    from sisteer.capeval import store as cs
    from sisteer.capeval.tiers import parity

    # 1. flag resolution (loader kwargs are PER PROBE; argparse hands the
    # flag over as a string, tests cover both spellings). Since PW3 the probe
    # set is mmlu ALONE — the gsm8k probe is retired (the served standard
    # owns the slot) — so a gsm8k width refuses with the pointer.
    DEFAULT = {"mmlu": {}}
    FULL = {"mmlu": {"n": None}}
    assert run.PARITY_PROBES == ("mmlu",)
    assert run._parity_width(SimpleNamespace(parity_n=-1)) == (DEFAULT, None)
    assert run._parity_width(SimpleNamespace()) == (DEFAULT, None)  # no flag
    assert run._parity_width(
        SimpleNamespace(parity_n=None)) == (DEFAULT, None)  # argparse default
    assert run._parity_width(SimpleNamespace(parity_n=0)) == (FULL, 0)
    assert run._parity_width(SimpleNamespace(parity_n="0")) == (FULL, 0)
    assert run._parity_width(SimpleNamespace(parity_n=1531)) == (
        {"mmlu": {"n": 1531}}, 1531)
    assert run._parity_width(
        SimpleNamespace(parity_n=parity.PARITY_N)) == (DEFAULT, None)
    with pytest.raises(SystemExit):
        run._parity_width(SimpleNamespace(parity_n=-5))
    with pytest.raises(SystemExit):
        run._parity_width(SimpleNamespace(parity_n="bogus"))

    # 1a. per-task specs: PARITY_N drops to the default, and with one probe
    # every all-equal spec collapses to the scalar — the dict form is never
    # produced any more (it survives only for READING the stores it keyed).
    assert run._parity_width(SimpleNamespace(parity_n="mmlu=400")) == (
        {"mmlu": {"n": 400}}, 400)
    assert run._parity_width(SimpleNamespace(parity_n="mmlu=0")) == (FULL, 0)
    assert run._parity_width(SimpleNamespace(
        parity_n=f"mmlu={parity.PARITY_N}")) == (DEFAULT, None)
    for bad in ("wmdp=5", "mmlu=-1", "mmlu=x", "mmlu=1,mmlu=2", "mmlu", "=5"):
        with pytest.raises(SystemExit):
            run._parity_width(SimpleNamespace(parity_n=bad))
    # a gsm8k width — any spelling — refuses NAMING the served standard
    for bad in ("gsm8k=0", "gsm8k=400,mmlu=400", "gsm8k=200"):
        with pytest.raises(SystemExit) as ei:
            run._parity_width(SimpleNamespace(parity_n=bad))
        assert "RETIRED" in str(ei.value) and "inspect/gsm8k" in str(ei.value), bad

    # 1b. --n-gsm8k, the retired gsm8k-only alias: any non-zero value refuses
    # with the pointer (there is no HF gsm8k width to widen); its 0 default
    # still means UNSET, so an old script that never passed it is unaffected.
    assert run._parity_width(
        SimpleNamespace(parity_n=-1, n_gsm8k=0)) == (DEFAULT, None)
    for legacy in (1209, parity.PARITY_N, -3):
        with pytest.raises(SystemExit) as ei:
            run._parity_width(SimpleNamespace(parity_n=-1, n_gsm8k=legacy))
        assert "RETIRED" in str(ei.value) and "--n-gsm8k" in str(ei.value)
    with pytest.raises(SystemExit):               # alias + --parity-n -> refuse
        run._parity_width(SimpleNamespace(parity_n=0, n_gsm8k=1209))

    # 2. resolved-config recording + store-key fork
    cfg = {"model": "m", "model_revision": None, "scale": 1.0,
           "completion_tokens": 16}
    cell = SimpleNamespace(id="none")
    default = run._resolved_config(cell, cfg, None, 100, "claude-sonnet-5")
    assert "parity_n" not in default  # hash-stable with existing stores
    assert default == run._resolved_config(cell, cfg, None, 100,
                                           "claude-sonnet-5", parity_n=None)
    widened = run._resolved_config(cell, cfg, None, 0, "claude-sonnet-5",
                                   parity_n=0)
    assert widened["parity_n"] == 0
    assert (cs.harness_config_hash(default)
            != cs.harness_config_hash(widened))
    # the per-task dict is a THIRD key, distinct from default and from
    # every scalar (a mixed instrument shares a store with neither)
    mixed = run._resolved_config(cell, cfg, None, 0, "claude-sonnet-5",
                                 parity_n={"gsm8k": 0})
    assert mixed["parity_n"] == {"gsm8k": 0}
    assert len({cs.harness_config_hash(c)
                for c in (default, widened, mixed)}) == 3

    # 3. nesting, on a real datasets.Dataset through the real loaders
    hfds = pytest.importorskip("datasets")
    gds = hfds.Dataset.from_dict(
        {"question": [f"gq{i}" for i in range(12)],
         "answer": [str(i) for i in range(12)]})
    mds = hfds.Dataset.from_dict(
        {"question": [f"mq{i}" for i in range(12)],
         "choices": [["w", "x", "y", "z"]] * 12,
         "answer": [i % 4 for i in range(12)]})
    monkeypatch.setattr(
        "datasets.load_dataset",
        lambda path, *a, **k: gds if "gsm8k" in path else mds)
    pins = {"datasets": {"gsm8k_platinum": {"revision": "r"},
                         "mmlu": {"revision": "r"}}}
    for fn in (parity.gsm8k_parity_rows, parity.mmlu_parity_rows):
        full = fn(pins, n=None)
        assert len(full) == 12                    # n=None = the whole pool
        assert fn(pins) == full                   # default caps at the pool
        five = fn(pins, n=5)
        assert five == full[:5]                   # strict prefix, ids stable
        # a seeded --limit subset of a narrow width stays inside the full
        # width (subset-of-a-prefix), so limit-100 reference item sets are
        # subsamples of the widened run's
        sub = run._seeded_subset(fn(pins, n=8), 4)
        by_id = {r["item_id"]: r for r in full}
        assert all(by_id[r["item_id"]] == r for r in sub)
