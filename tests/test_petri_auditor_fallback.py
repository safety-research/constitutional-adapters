"""CPU-only tests for Petri's in-conversation auditor fallback chain."""

import asyncio
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from sisteer.capeval import manifest as cm
from sisteer.capeval import store as cs
from sisteer.capeval.petri import auditor_fallback as af
from sisteer.capeval.petri import petri_report, petri_stats
from sisteer.capeval.tiers import petri_runner as pr
from test_petri_store_shape import (
    BEHAVIORS,
    PROMPT_FILES,
    SIDS,
    _run_cell,
    full_dims,
    make_log,
    sample,
    world,  # noqa: F401
)


class FakeModel:
    def __init__(self, name, outcomes):
        self.name = name
        self.outcomes = list(outcomes)
        self.calls = []

    async def generate(self, input, **kwargs):
        self.calls.append((input, kwargs))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class ScriptedDiscriminator:
    def __init__(self, kinds):
        self.kinds = list(kinds)
        self.successes = []

    def classify_failure(self, model, sample_key, started, completed, **kwargs):
        kind = self.kinds.pop(0)
        return af.LiveDiscriminatorEvidence(
            kind=kind, model=model,
            window_start=kwargs.get("wall_started", started),
            window_end=kwargs.get("wall_completed", completed),
            monotonic_start=started, monotonic_end=completed,
            siblings_succeeded=int(kind == "content_correlated"),
            siblings_failed=int(kind == "outage"),
            sibling_success_timestamps=((started + completed) / 2,)
            if kind == "content_correlated" else (),
            sibling_failure_timestamps=(started,)
            if kind == "outage" else (),
            basis=f"synthetic {kind}",
        )

    def record_success(self, model, sample_key, completed,
                       wall_completed=None):
        self.successes.append((model, sample_key, completed))


def _leg(name, outcomes):
    return af.AuditorLeg(FakeModel(name, outcomes), name, af.auditor_label(name))


def _controller(legs, kinds, store):
    ticks = iter(float(x) for x in range(1, 100))
    return af.AuditorFallbackController(
        legs, discriminator=ScriptedDiscriminator(kinds),
        clock=lambda: next(ticks), state_getter=lambda: ("sample-1", store))


def test_content_refusal_switches_in_place_and_persists_full_transcript():
    transcript = [object(), object()]
    tools = [object()]
    config = object()
    primary = _leg("anthropic/claude-sonnet-5", [RuntimeError("shed")])
    grok_out = SimpleNamespace(metadata=None)
    grok_next = SimpleNamespace(metadata=None)
    grok = _leg("openrouter/x-ai/grok-4.6", [grok_out, grok_next])
    store = {}
    controller = _controller([primary, grok], ["content_correlated"], store)

    got = asyncio.run(controller.generate(
        transcript, tools=tools, tool_choice="auto", config=config))
    assert got is grok_out
    assert len(primary.model.calls) == 1 and len(grok.model.calls) == 1
    for seen_input, kwargs in primary.model.calls + grok.model.calls:
        assert seen_input is transcript
        assert kwargs["tools"] is tools
        assert kwargs["tool_choice"] == "auto"
        assert kwargs["config"] is config

    chain = store[af.AUDITOR_CHAIN_STORE_KEY]
    assert chain["display"] == "sonnet-5 → grok-4.6 @turn 1"
    assert chain["switches"][0]["reason"] == "content_correlated"
    assert chain["switches"][0]["discriminator"]["siblings_succeeded"] == 1
    assert af.chain_kind(chain) == af.CHAIN_CONTINUED_GROK
    assert got.metadata[af.AUDITOR_CHAIN_STORE_KEY] == chain

    # The next auditor turn continues on Grok; Sonnet is not retried and the
    # sample is not re-run from its beginning.
    assert asyncio.run(controller.generate(transcript)) is grok_next
    assert len(primary.model.calls) == 1 and len(grok.model.calls) == 2


def test_grok_refusal_switches_to_qwen_and_records_provenance():
    primary = _leg("anthropic/claude-sonnet-5", [RuntimeError("shed")])
    grok = _leg("openrouter/x-ai/grok-4.6", [RuntimeError("declined")])
    qwen_out = SimpleNamespace(metadata={})
    qwen = _leg(af.DEFAULT_QWEN_MODEL, [qwen_out])
    store = {}
    controller = _controller(
        [primary, grok, qwen],
        ["content_correlated", "content_correlated"], store)

    assert asyncio.run(controller.generate(["full", "transcript"])) is qwen_out
    chain = store[af.AUDITOR_CHAIN_STORE_KEY]
    assert chain["display"] == (
        "sonnet-5 → grok-4.6 @turn 1 → qwen-abl @turn 1")
    assert [x["label"] for x in chain["used"]] == [
        "sonnet-5", "grok-4.6", "qwen-abl"]
    assert af.chain_kind(chain) == af.CHAIN_CONTINUED_QWEN


def test_outage_does_not_switch():
    primary = _leg("anthropic/claude-sonnet-5", [RuntimeError("outage")])
    grok = _leg("openrouter/x-ai/grok-4.6", [SimpleNamespace(metadata=None)])
    store = {}
    controller = _controller([primary, grok], ["outage"], store)

    with pytest.raises(RuntimeError, match="outage"):
        asyncio.run(controller.generate(["transcript"]))
    assert len(primary.model.calls) == 1
    assert grok.model.calls == []
    chain = store[af.AUDITOR_CHAIN_STORE_KEY]
    assert chain["switches"] == []
    assert chain["final_failure"]["kind"] == "outage"
    assert af.chain_kind(chain, refused=True) == af.CHAIN_UNAVAILABLE


def test_all_legs_can_fail_without_rerunning_the_sample():
    legs = [
        _leg("anthropic/claude-sonnet-5", [RuntimeError("one")]),
        _leg("openrouter/x-ai/grok-4.6", [RuntimeError("two")]),
        _leg(af.DEFAULT_QWEN_MODEL, [RuntimeError("three")]),
    ]
    store = {}
    controller = _controller(
        legs, ["content_correlated"] * 3, store)
    with pytest.raises(RuntimeError, match="three"):
        asyncio.run(controller.generate(["same", "messages"]))
    assert [len(leg.model.calls) for leg in legs] == [1, 1, 1]
    chain = store[af.AUDITOR_CHAIN_STORE_KEY]
    assert chain["final_failure"]["all_legs_exhausted"] is True
    assert af.chain_kind(chain, refused=True) == af.CHAIN_REFUSED_ALL


def test_live_discriminator_excludes_same_sample_and_keeps_timestamps():
    d = af.LiveSiblingDiscriminator()
    d.record_success("m", "same", 5.0)
    d.record_success("m", "sibling", 6.0)
    content = d.classify_failure("m", "same", 4.0, 7.0)
    assert content.kind == "content_correlated"
    assert content.sibling_success_timestamps == (6.0,)

    first = d.classify_failure("other", "a", 10.0, 20.0)
    second = d.classify_failure("other", "b", 15.0, 25.0)
    assert first.kind == "undetermined"
    assert second.kind == "undetermined"
    assert second.sibling_failure_timestamps == ()

    outage = d.classify_failure(
        "typed", "a", 10.0, 20.0, independent_outage=True)
    overlapping = d.classify_failure("typed", "b", 15.0, 25.0)
    assert outage.kind == "outage"
    assert overlapping.kind == "outage"
    assert overlapping.sibling_failure_timestamps == (10.0,)


def test_monotonic_interval_survives_wall_clock_rollback():
    d = af.LiveSiblingDiscriminator()
    d.record_success("m", "sibling", 1.5, wall_completed=95.0)
    evidence = d.classify_failure(
        "m", "sample", 1.0, 2.0,
        wall_started=100.0, wall_completed=90.0)
    assert evidence.kind == "content_correlated"
    assert evidence.monotonic_start == 1.0
    assert evidence.monotonic_end == 2.0
    assert evidence.window_start == 100.0
    assert evidence.window_end == 90.0
    assert evidence.window_start_iso.startswith("1970-01-01T00:01:40")
    assert evidence.window_end_iso.startswith("1970-01-01T00:01:30")
    assert evidence.sibling_success_timestamps == (95.0,)


def test_sparse_single_sample_leg_stays_undetermined_and_does_not_switch():
    primary = _leg(
        "anthropic/claude-sonnet-5",
        [RuntimeError("overloaded body-specific refusal")])
    grok = _leg("openrouter/x-ai/grok-4.6",
                [SimpleNamespace(metadata=None)])
    store = {}
    controller = af.AuditorFallbackController(
        [primary, grok], discriminator=af.LiveSiblingDiscriminator(),
        state_getter=lambda: ("only-sample", store))
    with pytest.raises(RuntimeError, match="body-specific"):
        asyncio.run(controller.generate(["transcript"]))
    assert store[af.AUDITOR_CHAIN_STORE_KEY]["final_failure"]["kind"] == \
        "undetermined"
    assert grok.model.calls == []


def test_clustered_content_failures_do_not_manufacture_outage():
    class BarrierModel:
        def __init__(self):
            self.arrived = 0
            self.ready = asyncio.Event()

        async def generate(self, input, **kwargs):
            self.arrived += 1
            if self.arrived == 2:
                self.ready.set()
            await self.ready.wait()
            raise RuntimeError("overloaded body-specific refusal")

    async def run_pair():
        model = BarrierModel()
        leg = af.AuditorLeg(
            model, "anthropic/claude-sonnet-5", "sonnet-5")
        discriminator = af.LiveSiblingDiscriminator()
        stores = ({}, {})
        controllers = [
            af.AuditorFallbackController(
                [leg], discriminator=discriminator,
                state_getter=lambda i=i: (f"sample-{i}", stores[i]))
            for i in range(2)
        ]
        results = await asyncio.gather(
            *(c.generate(["same-shaped transcript"]) for c in controllers),
            return_exceptions=True)
        return stores, results

    stores, results = asyncio.run(run_pair())
    assert all(isinstance(x, RuntimeError) for x in results)
    assert [s[af.AUDITOR_CHAIN_STORE_KEY]["final_failure"]["kind"]
            for s in stores] == ["undetermined", "undetermined"]


def _chain(used_labels, *, final_kind=None):
    names = {
        "sonnet-5": "anthropic/claude-sonnet-5",
        "grok-4.6": "openrouter/x-ai/grok-4.6",
        "qwen-abl": af.DEFAULT_QWEN_MODEL,
    }
    configured = [{"model": names[x], "label": x}
                  for x in ("sonnet-5", "grok-4.6", "qwen-abl")]
    used = [{"model": names[x], "label": x, "switch_turn": i + 1}
            for i, x in enumerate(used_labels)]
    out = {"version": 1, "configured": configured, "used": used,
           "switches": [], "display": " → ".join(used_labels)}
    if final_kind:
        out["final_failure"] = {"kind": final_kind}
    return out


def test_per_cell_report_counts_chain_legs_and_all_leg_refusal(world, tmp_path):
    cell = world.cells["si_safety__psr_kl_all"]

    def drive(tasks, *, spec, cell, engine):
        logs = []
        for task in tasks:
            samples = []
            for sid in SIDS:
                key = (task.behavior_id, sid)
                if key == (BEHAVIORS[1], SIDS[1]):
                    chain = _chain(
                        ["sonnet-5", "grok-4.6", "qwen-abl"],
                        final_kind="content_correlated")
                    samples.append({
                        "id": sid, "epoch": 1, "turn_count": 9,
                        "input": "synthetic all-leg refusal",
                        "messages": [{"role": "tool", "content":
                                      "synthetic last target output"}],
                        "events": [{
                            "event": "model", "model": af.DEFAULT_QWEN_MODEL,
                            "role": pr.AUDITOR_ROLE,
                            "timestamp": "2026-09-04T12:00:00+00:00",
                            "error": {"message": "final refusal"},
                        }],
                        "completed_at": "2026-09-04T12:10:00+00:00",
                        "error": {"message": "final refusal"},
                        "store": {af.AUDITOR_CHAIN_STORE_KEY: chain},
                    })
                    continue
                labels = {
                    (BEHAVIORS[0], SIDS[0]): ["sonnet-5"],
                    (BEHAVIORS[0], SIDS[1]): ["sonnet-5", "grok-4.6"],
                    (BEHAVIORS[1], SIDS[0]):
                        ["sonnet-5", "grok-4.6", "qwen-abl"],
                }[key]
                s = sample(sid, full_dims(task.behavior_id))
                s["store"] = {af.AUDITOR_CHAIN_STORE_KEY: _chain(labels)}
                if labels[-1] == "qwen-abl":
                    s["events"] = [{
                        "event": "model", "model": af.DEFAULT_QWEN_MODEL,
                        "timestamp": "2026-09-04T12:00:00+00:00",
                        "completed": "2026-09-04T12:05:00+00:00",
                    }]
                samples.append(s)
            log = make_log(world, cell, task.dir_key, samples, spec=spec)
            log["location"] = f"/synthetic/{task.behavior_id}.eval"
            logs.append(log)
        return logs

    store, result = _run_cell(world, cell, tmp_path, drive)
    assert result["audited"] == {"items": 3, "expected": 4}
    assert result["auditor_chain"] == {
        "sonnet_only": 1,
        "continued_by_grok": 1,
        "continued_by_qwen": 1,
        "auditor_refused_all_legs": 1,
        "auditor_unavailable": 0,
    }
    assert pr.audited_footing_line(result).startswith(
        "audited 3/4: sonnet-only 1, continued-by-grok 1, "
        "continued-by-qwen 1, auditor_refused (all legs) 1")
    record = pr.read_auditor_refused(store)[0]
    assert record.shed.auditor_chain["display"].endswith("qwen-abl")

    # Full pipeline: the terminal item is tombstoned from consolidation,
    # analysis still publishes the cell footing, and the unequal paired
    # contrast is explicitly unavailable rather than crashing or intersecting.
    tombstones = {
        store.cell_dir.name: pr.auditor_refused_row_keys([record], 1)}
    parquet = cs.consolidate(
        tmp_path / "sweep", exclude_keys_by_cell=tombstones)
    safety_rows = pd.read_parquet(parquet)
    refused_behavior = BEHAVIORS[1]
    refused_item = SIDS[1]
    assert not ((safety_rows["task"] == f"petri/{refused_behavior}")
                & (safety_rows["item_id"] == refused_item)).any()

    template = safety_rows[
        (safety_rows["task"] == f"petri/{refused_behavior}")
        & (safety_rows["item_id"] == SIDS[0])].copy()
    template["item_id"] = refused_item
    complete_leg = pd.concat([safety_rows, template], ignore_index=True)
    cons_cell = "si_constitution__psr_kl_all"

    def relabel(frame, cell_id, engine, hash_byte):
        out = frame.copy()
        out["arm_name"] = cell_id
        out["engine"] = engine
        out["intervention_hash"] = hash_byte * 16
        out["run_id"] = f"petri-shape-test.{cell_id}.synthetic"
        return out

    cons_rows = relabel(
        complete_leg, cons_cell, safety_rows["engine"].iloc[0], "bc")
    base_rows = relabel(
        complete_leg, "none", "proxy_vllm+passthrough+exact_replay", "cd")
    analysis_frame = pd.concat(
        [safety_rows, cons_rows, base_rows], ignore_index=True)
    footing = {
        "audited": result["audited"],
        "auditor_refused": result["auditor_refused"],
        "auditor_chain": result["auditor_chain"],
        "behaviors": result["behaviors"],
    }
    analysis = petri_stats.analyze(
        analysis_frame, expected_behaviors=list(BEHAVIORS),
        expected_cells=["none", cell.cell_id, cons_cell],
        n_boot=50, seed=9, cell_footing={cell.cell_id: footing})
    contrast = analysis["per_method_macro"]["methods"]["psr_kl_all"]
    assert contrast["status"] == "PAIRED_ITEMS_UNAVAILABLE"
    assert contrast.get("delta") is None
    assert contrast["instrument_views"]["all_audited"]["status"] == "ok"
    assert analysis["by_cell"][cell.cell_id]["audited_footing"][
        "auditor_chain"]["auditor_refused_all_legs"] == 1

    tables = petri_report.build_tables(analysis)
    report = petri_report.render_markdown(
        analysis, tables, {}, "terminal-refusal")
    row = tables["by_cell"].set_index("cell").loc[cell.cell_id]
    assert row["audited_items"] == 3
    assert row["expected_items"] == 4
    assert row["auditor_refused_all_legs"] == 1
    assert "PAIRED_ITEMS_UNAVAILABLE" in report


def test_configured_chain_is_a_harness_identity_input(world):
    from sisteer.capeval.petri import defense_cells as dc
    from test_petri_store_shape import _source

    primary_only = replace(world.spec, auditor_fallback_models=[])
    cells = dc.enumerate_cells(
        primary_only, _source(world.root), behavior_files=world.bf,
        prompt_files=PROMPT_FILES)
    changed = {c.cell_id: c for c in cells.all_served()}[
        "si_safety__psr_kl_all"]
    original = world.cells["si_safety__psr_kl_all"]
    assert changed.harness_config_hash != original.harness_config_hash
    assert changed.petri_run_key != original.petri_run_key


# --- the EMPTY chain: additive to the key, "no fallback" on resume ---------------
#
# Three pins, fixed when the conditional stamp was adopted: (1) a
# manifest declaring `[]` keys byte-identically to the code that
# predated the field; (2) a manifest OMITTING the field loads the
# documented default chain, which is a harness input, so it keys
# differently from `[]`; (3) on resume an ABSENT
# `auditor_fallback_models` metadata key and an EMPTY list both mean "no
# fallback ran" -- accepted by a `[]` manifest, refused by a chained one.

#: the loader's default chain. One leg since 2026-09-15: the abliterated-Qwen leg was removed (reached
#: ZERO times in 21,399 audited rows, and requiring its OpenAI-compatible endpoint forced an attacker-pool
#: GPU to be stood up before any Petri run could start). Kept as a list so a second leg can be added back.
DEFAULT_CHAIN = ["openrouter/x-ai/grok-4.6"]
ONE_FALLBACK = ["openrouter/x-ai/grok-4.6"]
#: a chain that DIFFERS from the default, for the "a different chain is a different instrument" pins.
#: Before 2026-09-15 the default itself had two legs, so ONE_FALLBACK differed from it by construction;
#: now that the default is a single grok leg, the contrast needs its own fixture. The abliterated Qwen is
#: used here as a SYNTHETIC alternative leg -- the resolver still supports building it, it is simply no
#: longer in any shipped chain.
OTHER_CHAIN = ["openrouter/x-ai/grok-4.6",
               "openai/huihui-ai/Huihui-Qwen3.6-27B-abliterated"]


def _harness_inputs(spec, world, cell):
    return cm.petri_harness_config_inputs(
        spec, behavior_files=world.bf, prompt_files=PROMPT_FILES,
        serving_identity=cell.serving_identity)


def _cell_under(spec, world, cell_id="si_safety__psr_kl_all"):
    from sisteer.capeval.petri import defense_cells as dc
    from test_petri_store_shape import _source

    cells = dc.enumerate_cells(spec, _source(world.root),
                               behavior_files=world.bf,
                               prompt_files=PROMPT_FILES)
    return {c.cell_id: c for c in cells.all_served()}[cell_id]


def test_empty_chain_keys_byte_identically_to_the_pre_field_inputs(world):
    """Pin (1): ``auditor_fallback_models: []`` enters neither the harness
    inputs nor the hash. The inputs equal the pre-field mapping -- the
    chained spec's inputs with the ``auditor_fallbacks`` entry removed,
    which is exactly what the pre-field code enumerated -- and the cell's
    harness hash and run key are the sha over that same mapping, so every
    recorded chain-less run key stands."""
    import copy

    empty = replace(world.spec, auditor_fallback_models=[])
    cell = _cell_under(empty, world)
    inputs = _harness_inputs(empty, world, cell)
    assert "auditor_fallbacks" not in inputs["role_models"]
    assert inputs["role_models"] == {"auditor": empty.auditor_model,
                                     "judge": empty.judge_model,
                                     "scenarios": empty.scenarios_model}
    pre_field = copy.deepcopy(_harness_inputs(world.spec, world, cell))
    del pre_field["role_models"]["auditor_fallbacks"]
    assert inputs == pre_field
    pre_field_hash = cm.sha256_hex16(
        cm.canonical_json(pre_field).encode("utf-8"))
    assert cell.harness_config_hash == pre_field_hash
    assert cell.petri_run_key == cm.petri_run_key(
        pre_field_hash, cell.cell_id, cell.intervention_hash)


def test_omitted_chain_is_the_default_chain_and_a_harness_input(world):
    """Pin (2): a spec built WITHOUT the field (what a manifest omitting it
    loads as) carries the documented Grok-then-Qwen default; that
    non-empty chain is enumerated under ``role_models.auditor_fallbacks``
    and is the ONE difference from the ``[]`` declaration's inputs, so the
    two key differently."""
    import copy

    default = world.spec  # the fixture never names the field
    assert default.auditor_fallback_models == DEFAULT_CHAIN
    cell = world.cells["si_safety__psr_kl_all"]
    chained = _harness_inputs(default, world, cell)
    assert chained["role_models"]["auditor_fallbacks"] == DEFAULT_CHAIN
    empty = _harness_inputs(replace(default, auditor_fallback_models=[]),
                            world, cell)
    assert chained != empty
    without = copy.deepcopy(chained)
    del without["role_models"]["auditor_fallbacks"]
    assert without == empty
    assert (cm.sha256_hex16(cm.canonical_json(chained).encode("utf-8"))
            != cm.sha256_hex16(cm.canonical_json(empty).encode("utf-8")))


def test_resume_reads_absent_and_empty_chain_as_no_fallback(world):
    """Pin (3), every stamp x manifest combination: a log with NO
    ``auditor_fallback_models`` key (recorded before the field existed)
    and a log stamped ``[]`` are both accepted by a manifest declaring
    ``[]`` and both refused by a manifest declaring a chain; a log stamped
    WITH a chain is refused by the ``[]`` manifest and accepted only by
    the manifest declaring that same chain."""
    key = pr.AUDITOR_FALLBACK_METADATA_KEY
    empty = replace(world.spec, auditor_fallback_models=[])
    chained = replace(empty, auditor_fallback_models=ONE_FALLBACK)
    cell = _cell_under(empty, world)
    b = "self_preservation"

    def verify(log, spec):
        pr.verify_log_metadata(log, spec=spec, cell=cell, dir_key=b,
                               behavior_files=world.bf,
                               prompt_files=PROMPT_FILES)

    pre_field = make_log(world, cell, b, [], spec=empty)
    assert key not in pre_field["eval"]["metadata"]  # `[]` stamps nothing
    stamped_empty = make_log(world, cell, b, [], spec=empty,
                             metadata_override={key: []})
    for log in (pre_field, stamped_empty):
        verify(log, empty)
        with pytest.raises(pr.PetriRunnerError,
                           match=f"{key}: log=.* != manifest="):
            verify(log, chained)

    chained_log = make_log(world, cell, b, [], spec=chained)
    assert chained_log["eval"]["metadata"][key] == ONE_FALLBACK
    verify(chained_log, chained)
    with pytest.raises(pr.PetriRunnerError, match=f"{key}: log="):
        verify(chained_log, empty)
    # a different chain is a different instrument too
    with pytest.raises(pr.PetriRunnerError, match=f"{key}: log="):
        verify(chained_log,
               replace(empty, auditor_fallback_models=OTHER_CHAIN))


# --- the manifest LOADER: only OMISSION takes the default chain ------------------
#
# The raw YAML value is type-checked before the dataclass is built and before
# any normalization. Until 2026-09-06 the loader ran `list(value or [])` on
# the raw field, so `auditor_fallback_models: false` (or null, "", {}) silently
# became `[]` -- keyed identically to an explicit `[]` and ran with the bare
# auditor instead of failing validation -- and a bare scalar string became a
# list of its characters.

MANIFESTS = Path(__file__).parents[1] / "configs" / "capeval" / "manifests"
COMMITTED_MANIFEST = MANIFESTS / "petri_twosi.yaml"
#: the committed manifest's own declaration: the default chain, explicit.
#: One leg since 2026-09-15 -- the abliterated-Qwen leg was removed from the chain (never reached in
#: 21,399 audited rows, and requiring its endpoint forced a GPU to be stood up for every Petri run).
COMMITTED_CHAIN_BLOCK = ("\n  auditor_fallback_models:\n"
                         "    - openrouter/x-ai/grok-4.6\n")
EMPTY_CHAIN_LINE = "\n  auditor_fallback_models: []\n"


def _manifest_declaring(tmp_path, chain_lines: str, name: str = "m.yaml"):
    """A copy of a committed manifest with its one ``auditor_fallback_models``
    block replaced by ``chain_lines`` (``"\n"`` omits the key)."""
    text = COMMITTED_MANIFEST.read_text()
    assert text.count(COMMITTED_CHAIN_BLOCK) == 1
    path = tmp_path / name
    path.write_text(text.replace(COMMITTED_CHAIN_BLOCK, chain_lines))
    return path


@pytest.mark.parametrize("declaration", [
    "auditor_fallback_models: false",
    "auditor_fallback_models: null",
    "auditor_fallback_models:",          # a bare key is YAML null too
    'auditor_fallback_models: ""',
    "auditor_fallback_models: {}",
    "auditor_fallback_models: openrouter/x-ai/grok-4.6",   # a scalar, not a list
    "auditor_fallback_models: [openrouter/x-ai/grok-4.6, '']",   # an empty member
    "auditor_fallback_models: [openrouter/x-ai/grok-4.6, null]",
    "auditor_fallback_models: [openrouter/x-ai/grok-4.6, 7]",
])
def test_loader_refuses_an_explicit_value_that_is_not_a_list_of_models(
        tmp_path, declaration):
    path = _manifest_declaring(tmp_path, f"\n  {declaration}\n")
    with pytest.raises(cm.ManifestError,
                       match="auditor_fallback_models must be a list of "
                             "non-empty model strings"):
        cm.load_petri_spec(path)


def test_loader_takes_the_default_only_on_omission_and_keeps_empty_as_empty(
        tmp_path):
    omitted = _manifest_declaring(tmp_path, "\n", "omitted.yaml")
    assert cm.load_petri_spec(omitted).auditor_fallback_models == DEFAULT_CHAIN
    # the committed manifest declares its chain explicitly: loaded as declared
    assert cm.load_petri_spec(COMMITTED_MANIFEST).auditor_fallback_models == DEFAULT_CHAIN
    empty = _manifest_declaring(tmp_path, EMPTY_CHAIN_LINE, "empty.yaml")
    assert cm.load_petri_spec(empty).auditor_fallback_models == []
    one = _manifest_declaring(
        tmp_path, "\n  auditor_fallback_models: [openrouter/x-ai/grok-4.6]\n",
        "one.yaml")
    assert cm.load_petri_spec(one).auditor_fallback_models == ONE_FALLBACK


def test_factory_returns_an_inspect_model_interface_wrapper(monkeypatch):
    import sys
    import types

    class FakeInspectModel:
        def __init__(self, api, config, model_args=None):
            self.api = api
            self.config = config
            self.model_args = model_args or {}
            self.role = None

        def _set_role(self, role):
            self.role = role

    inspect_pkg = types.ModuleType("inspect_ai")
    model_mod = types.ModuleType("inspect_ai.model")
    model_mod.Model = FakeInspectModel
    monkeypatch.setitem(sys.modules, "inspect_ai", inspect_pkg)
    monkeypatch.setitem(sys.modules, "inspect_ai.model", model_mod)

    out = SimpleNamespace(metadata=None)
    underlying = FakeModel("primary", [out])
    underlying.api = object()
    underlying.config = object()
    underlying.model_args = {}
    leg = af.AuditorLeg(
        underlying, "anthropic/claude-sonnet-5", "sonnet-5")
    store = {}
    wrapped = af.make_inspect_auditor_chain(
        "anthropic/claude-sonnet-5", (), legs=[leg],
        state_getter=lambda: ("sample", store))
    assert isinstance(wrapped, FakeInspectModel)
    assert wrapped.role == "auditor"
    assert asyncio.run(wrapped.generate(["full transcript"])) is out
    assert store[af.AUDITOR_CHAIN_STORE_KEY]["display"] == "sonnet-5"


def test_unvalidated_grok_premise_is_load_bearing_documentation():
    module_doc = af.__doc__ or ""
    for needle in ("UNVALIDATED", "6.4%", "40%", "known Sonnet-shed", "acceptance rate"):
        assert needle in module_doc


# --- the OpenRouter key: environment or vault, never a file -------------------


def _forbid_secret_cli(monkeypatch):
    import subprocess

    def forbidden(*args, **kwargs):
        raise AssertionError("the secret-manager CLI must not run when the environment carries the key")
    monkeypatch.setattr(subprocess, "check_output", forbidden)


def test_openrouter_key_comes_from_the_environment_first(monkeypatch):
    monkeypatch.setenv(af.OPENROUTER_API_KEY_ENV, "  synthetic-env-value \n")
    _forbid_secret_cli(monkeypatch)
    assert af.resolve_openrouter_api_key() == "synthetic-env-value"


def test_openrouter_key_falls_back_to_the_vault_item(monkeypatch):
    import subprocess

    monkeypatch.delenv(af.OPENROUTER_API_KEY_ENV, raising=False)
    monkeypatch.setenv("SISTEER_SECRET_CMD", "fake-secret-cmd")                 # the site's secret-manager CLI, by setting
    calls = []

    def fake_check_output(argv, **kwargs):
        calls.append((list(argv), kwargs))
        return "synthetic-vault-value\n"
    monkeypatch.setattr(subprocess, "check_output", fake_check_output)
    assert af.resolve_openrouter_api_key() == "synthetic-vault-value"
    (argv, kwargs), = calls
    assert argv == ["fake-secret-cmd", af.OPENROUTER_VAULT_ITEM] == ["fake-secret-cmd", "OpenRouter API Key"]
    assert kwargs["text"] is True
    assert 0 < kwargs["timeout"] <= af.SECRET_CMD_TIMEOUT_S < float("inf")       # bounded, never a hang
    # a blank environment value is "unset", not a key
    calls.clear()
    monkeypatch.setenv(af.OPENROUTER_API_KEY_ENV, "   ")
    assert af.resolve_openrouter_api_key() == "synthetic-vault-value" and len(calls) == 1


def _secret_cli_failures():
    import subprocess
    cmd = ["fake-secret-cmd", "OpenRouter API Key"]
    return [
        pytest.param(None, "SISTEER_SECRET_CMD is not set", id="setting-unset"),
        pytest.param(FileNotFoundError(2, "No such file or directory", "fake-secret-cmd"),
                     "is not an executable (not found, or not on PATH)", id="not-on-path"),
        pytest.param(subprocess.CalledProcessError(1, cmd, output="partial-stdout-never-shown",
                                                   stderr="[ERROR] item not found in the vault\nmore"),
                     "exited 1 ([ERROR] item not found in the vault)", id="non-zero-exit"),
        pytest.param(subprocess.TimeoutExpired(cmd, 30.0, output="partial-stdout-never-shown"),
                     "did not answer within", id="timeout"),
        pytest.param("", "returned an empty value", id="empty"),
        pytest.param("\n", "returned an empty value", id="blank"),
    ]


@pytest.mark.parametrize("outcome, phrase", _secret_cli_failures())
def test_missing_openrouter_key_is_a_typed_error_that_never_leaks(monkeypatch, outcome, phrase):
    import subprocess

    monkeypatch.delenv(af.OPENROUTER_API_KEY_ENV, raising=False)
    if outcome is None:                                                          # no CLI named: refused by name, nothing run
        monkeypatch.delenv("SISTEER_SECRET_CMD", raising=False)
    else:
        monkeypatch.setenv("SISTEER_SECRET_CMD", "fake-secret-cmd")

    def fake_check_output(argv, **kwargs):
        if outcome is None:
            raise AssertionError("no command may run while the setting is unset")
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome
    monkeypatch.setattr(subprocess, "check_output", fake_check_output)
    with pytest.raises(af.AuditorSecretError) as info:
        af.resolve_openrouter_api_key()
    exc = info.value
    assert isinstance(exc, af.AuditorFallbackError)                                # the module's typed error
    text = str(exc)
    assert af.OPENROUTER_API_KEY_ENV in text and af.OPENROUTER_VAULT_ITEM in text and phrase in text
    assert "partial-stdout-never-shown" not in text                              # captured stdout never surfaces ...
    # ... nor through the traceback chain: no cause, and no context or a suppressed one
    assert exc.__cause__ is None and (exc.__context__ is None or exc.__suppress_context__)


def test_openrouter_key_never_comes_from_a_file():
    import inspect

    file_names = [n for n in dir(af) if not n.startswith("__") and "file" in n.lower()]
    assert file_names == [], file_names
    source = inspect.getsource(af)
    for forbidden in ("openrouter-key", "KEY_FILE", "key_file", ".config/", "read_text", "expanduser", "pathlib"):
        assert forbidden not in source, forbidden
    params = inspect.signature(af.resolve_inspect_auditor_models).parameters
    # `role` (keyword-only, 2026-09-19) stamps the legs for the judge chain; still no path, file or key parameter
    assert list(params) == ["primary_model", "fallback_models", "role"]
    assert params["role"].kind is inspect.Parameter.KEYWORD_ONLY
    assert "never a file" in (af.resolve_openrouter_api_key.__doc__ or "").lower()


def _fake_inspect_get_model(monkeypatch):
    import sys
    import types

    seen = []

    class FakeInspectModel:
        def __init__(self, name):
            self.name = name
            self.role = None

        def _set_role(self, role):
            self.role = role

    def fake_get_model(name, **kwargs):
        seen.append((name, kwargs))
        return FakeInspectModel(name)
    inspect_pkg = types.ModuleType("inspect_ai")
    model_mod = types.ModuleType("inspect_ai.model")
    model_mod.get_model = fake_get_model
    monkeypatch.setitem(sys.modules, "inspect_ai", inspect_pkg)
    monkeypatch.setitem(sys.modules, "inspect_ai.model", model_mod)
    return seen


def test_resolver_hands_the_environment_key_to_the_openrouter_leg_only(monkeypatch):
    monkeypatch.setenv(af.OPENROUTER_API_KEY_ENV, "synthetic-env-value")
    monkeypatch.setenv(af.QWEN_BASE_URL_ENV, "http://127.0.0.1:8000/")
    monkeypatch.delenv(af.QWEN_API_KEY_ENV, raising=False)
    _forbid_secret_cli(monkeypatch)
    seen = _fake_inspect_get_model(monkeypatch)
    # an EXPLICIT three-leg chain: this pins the resolver's per-provider key handling, including the
    # abliterated-Qwen branch, which still exists even though that leg left the default on 2026-09-15.
    legs = af.resolve_inspect_auditor_models(
        "anthropic/claude-sonnet-5", (af.DEFAULT_GROK_MODEL, af.DEFAULT_QWEN_MODEL))
    assert [leg.label for leg in legs] == ["sonnet-5", "grok-4.6", "qwen-abl"]
    assert all(leg.model.role == "auditor" for leg in legs)
    by_name = dict(seen)
    assert by_name["anthropic/claude-sonnet-5"] == {"memoize": False}            # inspect's own resolution
    assert by_name[af.DEFAULT_GROK_MODEL] == {"memoize": False, "api_key": "synthetic-env-value"}
    assert by_name[af.DEFAULT_QWEN_MODEL] == {"memoize": False, "base_url": "http://127.0.0.1:8000",
                                              "api_key": "petri-local", "responses_api": False}


def test_resolver_refuses_loudly_when_the_openrouter_key_is_nowhere(monkeypatch):
    import subprocess

    monkeypatch.delenv(af.OPENROUTER_API_KEY_ENV, raising=False)
    monkeypatch.setenv("SISTEER_SECRET_CMD", "fake-secret-cmd")
    monkeypatch.setattr(subprocess, "check_output",
                        lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError(2, "no", "fake-secret-cmd")))
    seen = _fake_inspect_get_model(monkeypatch)
    with pytest.raises(af.AuditorSecretError):
        af.resolve_inspect_auditor_models("anthropic/claude-sonnet-5", af.DEFAULT_FALLBACK_MODELS)
    assert [name for name, _ in seen] == ["anthropic/claude-sonnet-5"]           # stopped at the grok leg, no silent None


# --- the live driver's auditor wiring follows the same guard ---------------------
#
# The third half of the non-empty rule (the first two are the harness input and
# the eval-log stamp, pinned above): _inspect_driver wraps the auditor role in
# the fallback chain ONLY when a chain is declared; an empty chain wires the bare
# auditor model, so a chain-less run's logs carry model_roles.auditor = the
# manifest's auditor and verify unchanged on resume. inspect_ai / petri_bloom
# are not installed here: every seam the driver reaches is stubbed.


def _drive_inspect(monkeypatch, tmp_path, spec, cell):
    """Run ``_inspect_driver`` on CPU with the inspect / petri_bloom seams
    stubbed. Returns the ``model_roles`` each task was built with and the
    ``(primary, fallbacks)`` calls made to ``make_inspect_auditor_chain``."""
    import sys
    import types

    roles_seen, chains = [], []
    inspect_pkg = types.ModuleType("inspect_ai")
    inspect_pkg.eval_set = lambda group, **kwargs: (True, [])

    def task_with(task, *, model_roles, metadata):
        roles_seen.append(dict(model_roles))
        return task
    inspect_pkg.task_with = task_with
    log_mod = types.ModuleType("inspect_ai.log")
    log_mod.read_eval_log = lambda location: None
    model_mod = types.ModuleType("inspect_ai.model")

    class FakeGenerateConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def merge(self, other):
            return self
    model_mod.GenerateConfig = FakeGenerateConfig
    model_mod.get_model = lambda name, **kwargs: SimpleNamespace(
        name=name, config=None)
    bloom = types.ModuleType("petri_bloom")
    bloom.bloom_audit = lambda **kwargs: object()
    for name, mod in (("inspect_ai", inspect_pkg), ("inspect_ai.log", log_mod),
                      ("inspect_ai.model", model_mod), ("petri_bloom", bloom)):
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.setattr(pr, "_require_inspect", lambda: inspect_pkg)
    monkeypatch.setattr(pr, "require_agent_loop_entry", lambda engine, cell: None)
    monkeypatch.setattr(pr, "eval_concurrency", lambda spec, engine: 4)
    monkeypatch.setattr(pr, "install_retry_sleep_cap", lambda: None)
    monkeypatch.setattr(pr, "eval_set_bounds", lambda: {})

    def fake_chain(primary, fallbacks):
        chains.append((primary, list(fallbacks)))
        return ("CHAIN", primary, tuple(fallbacks))
    monkeypatch.setattr(af, "make_inspect_auditor_chain", fake_chain)
    engine = SimpleNamespace(petri_target=lambda cell: SimpleNamespace(config=None))
    task = pr.EvalTaskSpec(behavior_id="self_preservation",
                           dir_key="self_preservation",
                           behavior_dir=tmp_path / "behavior",
                           log_dir=tmp_path / "logs", metadata={}, options={})
    out = pr._inspect_driver([task], spec=spec, cell=cell, engine=engine)
    assert out == []  # the stubbed eval_set produced no logs
    return roles_seen, chains


def test_inspect_driver_wires_the_bare_auditor_for_an_empty_chain(
        world, tmp_path, monkeypatch):
    empty = replace(world.spec, auditor_fallback_models=[])
    cell = _cell_under(empty, world)
    roles, chains = _drive_inspect(monkeypatch, tmp_path, empty, cell)
    assert chains == []  # no chain is built, the fallback module is not consulted
    (r,) = roles
    assert r["auditor"] == empty.auditor_model
    assert r["judge"] == empty.judge_model


def test_inspect_driver_wraps_the_auditor_in_a_declared_chain(
        world, tmp_path, monkeypatch):
    chained = replace(world.spec, auditor_fallback_models=ONE_FALLBACK)
    cell = _cell_under(chained, world)
    roles, chains = _drive_inspect(monkeypatch, tmp_path, chained, cell)
    assert chains == [(chained.auditor_model, ONE_FALLBACK)]
    (r,) = roles
    assert r["auditor"] == ("CHAIN", chained.auditor_model, tuple(ONE_FALLBACK))
    assert r["judge"] == chained.judge_model  # the judge is never wrapped
