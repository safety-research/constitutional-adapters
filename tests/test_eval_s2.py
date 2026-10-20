"""Stage-2 eval modules: composition, capability panel, LIFBench adapter,
suite dispatcher. CPU-only; model-free where a
generate_fn seam exists, tiny llama where a real forward is the point."""

from __future__ import annotations

import json

import pytest

# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------


class TestComposition:
    def test_default_targets_shape_and_compatibility(self):
        from sisteer.eval.composition import compatible, default_targets

        targets = default_targets(seed=0)
        assert len(targets["seen"]) == 8
        assert len(targets["unseen"]) == 4
        assert len(targets["triples"]) == 4
        assert not set(targets["seen"]) & set(targets["unseen"])
        for group in targets.values():
            for t in group:
                assert compatible(t)
        assert default_targets(seed=0) == targets  # deterministic

    def test_two_languages_incompatible(self):
        from sisteer.eval.composition import compatible

        assert not compatible(("lang_fr", "lang_es"))
        assert compatible(("lang_fr", "lowercase"))

    def test_joint_compliance_requires_all(self):
        from sisteer.eval.composition import joint_compliance

        fr_lower = "bonjour mes amis, je suis très heureux de vous voir aujourd'hui."
        assert joint_compliance(fr_lower, ("lang_fr", "lowercase"))
        assert not joint_compliance(
            "Bonjour mes amis, je suis très heureux de vous voir.",
            ("lang_fr", "lowercase"),
        )  # French but capitalized
        assert not joint_compliance(
            "hello my friends, lovely to see you all here today.",
            ("lang_fr", "lowercase"),
        )  # lowercase but English

    def test_sentence_count_behavior(self):
        from sisteer.eval.composition import check_behavior

        assert check_behavior("One. Two! Three?", "three_sentences")
        assert not check_behavior("One. Two.", "three_sentences")

    def test_orderings_enumeration(self):
        from sisteer.eval.composition import orderings

        assert len(orderings(2)) == 2
        assert len(orderings(3)) == 6
        assert set(orderings(2)) == {(0, 1), (1, 0)}

    def test_joined_instruction_orders_clauses(self):
        from sisteer.eval.composition import joined_instruction_text

        a = joined_instruction_text(("lang_fr", "lowercase"))
        b = joined_instruction_text(("lowercase", "lang_fr"))
        assert a != b
        assert "French" in a and "lowercase" in a

    def test_eval_composition_arms_and_ordering_average(self, tiny_llama, tmp_path):
        from sisteer.eval.composition import eval_composition

        _, tok = tiny_llama
        seen_orders = []

        class StubObj:
            kind = "stub"

        def compose_fn(objs, order):
            seen_orders.append(order)
            return StubObj()

        # French+lowercase on one prompt; generation alternates pass/fail
        # so the ordering average is visible.
        outputs = iter([
            "bonjour mes amis, je suis très heureux de vous voir aujourd'hui.",  # prompted ord 1: pass
            "Hello friends.",  # prompted ord 2: fail
            "bonjour mes amis, je suis très heureux de vous voir aujourd'hui.",  # composed ord 1
            "bonjour mes amis, je suis très heureux de vous voir aujourd'hui.",  # composed ord 2
            "Hello friends.",  # monolithic: fail
        ])

        records = eval_composition(
            None, tok,
            targets={"seen": [("lang_fr", "lowercase")]},
            prompts=["Say hi"],
            out_dir=tmp_path,
            per_behavior_objs={"lang_fr": StubObj(), "lowercase": StubObj()},
            compose_fn=compose_fn,
            monolithic_objs={("lang_fr", "lowercase"): StubObj()},
            generate_fn=lambda rendered, obj: [next(outputs) for _ in rendered],
        )
        by_arm = {r["arm"]: r for r in records}
        assert by_arm["prompted"]["joint_strict"] == pytest.approx(0.5)
        assert by_arm["prompted"]["n_orderings_scored"] == 2
        assert by_arm["composed"]["joint_strict"] == pytest.approx(1.0)
        assert by_arm["monolithic"]["joint_strict"] == pytest.approx(0.0)
        assert sorted(seen_orders) == [(0, 1), (1, 0)]

        # resume: all three (target, arm) records exist -> nothing re-runs
        again = eval_composition(
            None, tok,
            targets={"seen": [("lang_fr", "lowercase")]},
            prompts=["Say hi"],
            out_dir=tmp_path,
            per_behavior_objs={"lang_fr": StubObj(), "lowercase": StubObj()},
            compose_fn=compose_fn,
            monolithic_objs={("lang_fr", "lowercase"): StubObj()},
            generate_fn=lambda rendered, obj: pytest.fail("must resume"),
        )
        assert again == []

    def test_prompted_arm_renders_the_trained_joined_instruction(
        self, tiny_llama, tmp_path
    ):
        """The prompted-NL arm must render the SAME text builder the
        compiled monolithic objects distilled from
        (data/instructions.joined_behavior_instruction) — a different
        clause wording would confound the prompted-vs-compiled contrast
        with instruction phrasing."""
        from sisteer.data.instructions import joined_behavior_instruction
        from sisteer.eval.composition import (
            eval_composition,
            joined_instruction_text,
        )

        # the module-level builder IS the training-side builder
        assert joined_instruction_text(("lang_fr", "lowercase")) == (
            joined_behavior_instruction(["lang_fr", "lowercase"]).text
        )

        _, tok = tiny_llama
        seen: list[str] = []

        def gen(rendered, obj):
            seen.extend(rendered)
            return ["x"] * len(rendered)

        eval_composition(
            None, tok,
            targets={"seen": [("lang_fr", "lowercase")]},
            prompts=["Say hi"],
            out_dir=tmp_path,
            arms=("prompted",),
            generate_fn=gen,
        )
        fr_lower = joined_behavior_instruction(["lang_fr", "lowercase"]).text
        lower_fr = joined_behavior_instruction(["lowercase", "lang_fr"]).text
        assert any(fr_lower in r for r in seen)
        assert any(lower_fr in r for r in seen)

    def test_records_carry_object_kind_and_ordering(self, tiny_llama, tmp_path):
        """k!-averaged and canonical-order numbers must be separable at
        analysis — records carry the object kind and an ordering label;
        the monolithic arm is the canonical-order arm."""
        from sisteer.eval.composition import eval_composition

        _, tok = tiny_llama

        class StubObj:
            kind = "token_appended"

        records = eval_composition(
            None, tok,
            targets={"seen": [("lang_fr", "lowercase")]},
            prompts=["Say hi"],
            out_dir=tmp_path,
            per_behavior_objs={"lang_fr": StubObj(), "lowercase": StubObj()},
            compose_fn=lambda objs, order: StubObj(),
            monolithic_objs={("lang_fr", "lowercase"): StubObj()},
            generate_fn=lambda rendered, obj: ["x"] * len(rendered),
        )
        by_arm = {r["arm"]: r for r in records}
        assert by_arm["monolithic"]["ordering"] == "canonical"
        assert by_arm["monolithic"]["object_kind"] == "token_appended"
        assert by_arm["monolithic"]["per_ordering"][0]["order"] == [0, 1]
        assert by_arm["prompted"]["ordering"] == "kfact_averaged"
        assert by_arm["prompted"]["object_kind"] is None
        assert by_arm["composed"]["ordering"] == "kfact_averaged"
        assert by_arm["composed"]["object_kind"] == "token_appended"

    def test_monolithic_requires_canonical_key(self, tiny_llama, tmp_path):
        from sisteer.eval.composition import eval_composition

        _, tok = tiny_llama

        class StubObj:
            kind = "stub"

        with pytest.raises(KeyError, match="CANONICAL"):
            eval_composition(
                None, tok,
                targets={"seen": [("lang_fr", "lowercase")]},
                prompts=["Say hi"],
                out_dir=tmp_path,
                monolithic_objs={("lowercase", "lang_fr"): StubObj()},
                arms=("monolithic",),
                generate_fn=lambda rendered, obj: ["x"] * len(rendered),
            )

    def test_order_invariant_compose_scored_once(self, tiny_llama, tmp_path):
        from sisteer.eval.composition import eval_composition

        _, tok = tiny_llama

        class StubObj:
            kind = "stub"

        def additive(objs, order):
            return StubObj()

        additive.order_invariant = True

        records = eval_composition(
            None, tok,
            targets={"seen": [("lang_fr", "lowercase")]},
            prompts=["Say hi"],
            out_dir=tmp_path,
            per_behavior_objs={"lang_fr": StubObj(), "lowercase": StubObj()},
            compose_fn=additive,
            arms=("composed",),
            generate_fn=lambda rendered, obj: ["bonjour mes amis très heureux."]
            * len(rendered),
        )
        (rec,) = records
        assert rec["order_invariant"] is True
        assert rec["n_orderings_scored"] == 1
        assert rec["ordering"] == "order_invariant"


# ---------------------------------------------------------------------------
# capability_ext
# ---------------------------------------------------------------------------


class TestCapabilityExt:
    def test_gsm8k_extraction(self):
        from sisteer.eval.capability_ext import extract_gsm8k_answer

        assert extract_gsm8k_answer("reasoning...\n#### 42") == (42.0, False)
        assert extract_gsm8k_answer("#### $1,234") == (1234.0, False)
        assert extract_gsm8k_answer("#### -3.5") == (-3.5, False)
        # fallback: last number in the text
        val, fb = extract_gsm8k_answer("first 7 then the answer is 19.")
        assert (val, fb) == (19.0, True)
        # failure: no number at all
        assert extract_gsm8k_answer("I cannot compute this.") == (None, True)

    def test_gsm8k_gold_parsing(self):
        from sisteer.eval.capability_ext import parse_gsm8k_gold

        assert parse_gsm8k_gold("Long chain of thought.\n#### 72") == 72.0
        with pytest.raises(ValueError):
            parse_gsm8k_gold("no marker")

    def test_gsm8k_accuracy_refuses_without_the_legacy_switch(self, tiny_llama, monkeypatch):
        """PW3 (U2-impl 4): the bespoke GSM8K is retired as a forward
        instrument. gsm8k_accuracy refuses BEFORE generating (the seam is
        never called), naming the standard (inspect/gsm8k, served) and the
        switch; the pure helpers stay importable and usable."""
        from sisteer.capeval.tinycap import contracts as tc
        from sisteer.eval.capability_ext import (LegacyGSM8KRefused, extract_gsm8k_answer,
                                                 gsm8k_accuracy)

        monkeypatch.delenv(tc.GSM8K_LEGACY_ENV, raising=False)
        _, tok = tiny_llama
        called = []
        with pytest.raises(LegacyGSM8KRefused) as ei:
            gsm8k_accuracy(None, tok, rows=[{"question": "2+2?", "answer": "#### 4"}],
                           generate_fn=lambda rendered: called.append(rendered) or ["#### 4"])
        msg = str(ei.value)
        assert called == []
        assert "RETIRED" in msg and "inspect/gsm8k" in msg and "--controls gsm8k" in msg
        assert tc.GSM8K_LEGACY_ENV in msg and "capability_ext.gsm8k_accuracy" in msg
        assert extract_gsm8k_answer("#### 4") == (4.0, False)   # reading is unconditional

    def test_gsm8k_accuracy_with_stub_generation(self, tiny_llama, monkeypatch):
        from sisteer.capeval.tinycap import contracts as tc
        from sisteer.eval.capability_ext import gsm8k_accuracy

        monkeypatch.setenv(tc.GSM8K_LEGACY_ENV, "1")   # the legacy leg, re-enabled (PW3)
        _, tok = tiny_llama
        rows = [
            {"question": "2+2?", "answer": "#### 4"},
            {"question": "3*3?", "answer": "#### 9"},
            {"question": "10-1?", "answer": "#### 9"},
        ]
        outs = ["#### 4", "the answer is 8", "gibberish with no digits"]
        res = gsm8k_accuracy(
            None, tok, rows=rows, generate_fn=lambda rendered: outs
        )
        assert res["acc"] == pytest.approx(1 / 3)
        assert res["extraction_failure_rate"] == pytest.approx(1 / 3)
        assert res["extraction_fallback_rate"] == pytest.approx(1 / 3)

    def test_letter_logprob_wiring(self, tiny_llama):
        from sisteer.eval.capability_ext import (
            gpqa_diamond_accuracy,
            mmlu_pro_accuracy,
        )

        model, tok = tiny_llama
        rows4 = [
            {"question": "Pick A.", "choices": ["a", "b", "c", "d"], "answer": 0},
            {"question": "Pick B.", "choices": ["a", "b", "c", "d"], "answer": 1},
        ]
        acc = gpqa_diamond_accuracy(model, tok, rows=rows4)
        assert 0.0 <= acc <= 1.0
        rows10 = [
            {"question": "Pick one.", "choices": [f"opt{i}" for i in range(10)],
             "answer": 3},
        ]
        acc10 = mmlu_pro_accuracy(model, tok, rows=rows10)
        assert acc10 in (0.0, 1.0)

    def test_token_kind_injection_and_apply_context(self, tiny_llama):
        """Object plumbing: apply() entered PER QUESTION (one long-lived
        context left the skip_bos tracker counting across forwards),
        sentinel injected into every capability prompt for token-kind
        objects."""
        from contextlib import contextmanager

        from sisteer.eval.capability_ext import mmlu_under_obj

        model, tok = tiny_llama
        events = []

        class FakeToken:
            kind = "token"

            def inject(self, prompt):
                events.append("inject")
                return prompt

            @contextmanager
            def apply(self, model, scale=1.0):
                events.append(f"enter@{scale}")
                yield
                events.append("exit")

        rows = [
            {"question": "Q?", "choices": ["a", "b", "c", "d"], "answer": 0}
        ] * 3
        mmlu_under_obj(model, tok, FakeToken(), scale=2.0, rows=rows)
        # one context per question, each forward inside its own context
        assert events.count("enter@2.0") == 3
        assert events.count("exit") == 3
        assert events.count("inject") == 3
        assert events[-1] == "exit"

    def test_appended_token_kind_injected_in_capability(self, tiny_llama):
        """kind='token_appended' (token_faithful, in the core set) must
        ALSO get its sentinel injected — `== "token"` left it evaluating
        unsteered."""
        from contextlib import contextmanager

        from sisteer.eval.capability_ext import mmlu_under_obj

        model, tok = tiny_llama
        seen = []

        class FakeAppended:
            kind = "token_appended"
            sentinel = "<SENT-CAP>"

            def inject(self, prompt):
                out = prompt + self.sentinel
                seen.append(out)
                return out

            @contextmanager
            def apply(self, model, scale=1.0):
                yield

        rows = [
            {"question": "Q?", "choices": ["a", "b", "c", "d"], "answer": 0}
        ] * 2
        mmlu_under_obj(model, tok, FakeAppended(), rows=rows)
        assert len(seen) == 2
        assert all(s.endswith("<SENT-CAP>") for s in seen)

    def test_letter_logprob_single_bos_reaches_model(self, tiny_llama):
        """The letter-logprob path tokenizes an already-templated prompt;
        without encode_kwargs a SECOND BOS is prepended, which skip_bos
        does NOT mask — the steering addition lands on its massive
        activation and distorts MMLU/GPQA/MMLU-Pro under steering.
        Exactly ONE BOS may reach the model, for every letter-logprob
        metric."""
        from sisteer.eval.capability_ext import (
            gpqa_diamond_accuracy,
            mmlu_pro_accuracy,
            mmlu_under_obj,
        )
        from sisteer.models import build_chat_prompt

        model, tok = tiny_llama
        # the precondition that makes this test bite: the fixture's chat
        # template renders the BOS as literal text
        assert build_chat_prompt(tok, "x").startswith(tok.bos_token)

        captured = []

        def spy(_m, args, kwargs):
            ids = kwargs.get("input_ids")
            if ids is None and args:
                ids = args[0]
            captured.append(ids.detach().clone())

        rows4 = [{"question": "Q?", "choices": ["a", "b", "c", "d"],
                  "answer": 0}]
        rows10 = [{"question": "Q?", "choices": [f"o{i}" for i in range(10)],
                   "answer": 0}]
        h = model.register_forward_pre_hook(spy, with_kwargs=True)
        try:
            mmlu_under_obj(model, tok, None, rows=rows4)
            gpqa_diamond_accuracy(model, tok, rows=rows4)
            mmlu_pro_accuracy(model, tok, rows=rows10)
        finally:
            h.remove()
        assert len(captured) == 3  # one forward per metric
        for ids in captured:
            row = ids[0].tolist()
            assert row.count(tok.bos_token_id) == 1
            assert row[0] == tok.bos_token_id

    def test_stage1_mmlu_single_bos_reaches_model(self, tiny_llama):
        """The stage-1 panel's mmlu_accuracy follows the same one-BOS
        convention as the stage-2 letter-logprob metrics."""
        from sisteer.eval.capability import mmlu_accuracy

        model, tok = tiny_llama
        captured = []

        def spy(_m, args, kwargs):
            ids = kwargs.get("input_ids")
            if ids is None and args:
                ids = args[0]
            captured.append(ids.detach().clone())

        rows = [{"question": "Q?", "choices": ["a", "b", "c", "d"],
                 "answer": 0}] * 2
        h = model.register_forward_pre_hook(spy, with_kwargs=True)
        try:
            mmlu_accuracy(model, tok, rows=rows)
        finally:
            h.remove()
        assert len(captured) == 2
        for ids in captured:
            row = ids[0].tolist()
            assert row.count(tok.bos_token_id) == 1
            assert row[0] == tok.bos_token_id

    def test_capability_panel_refuses_without_the_legacy_switch(self, tiny_llama, monkeypatch):
        """PW3: the panel's gsm8k branch is the retired instrument, so the
        whole panel refuses without the switch — before any row set loads."""
        from sisteer.capeval.tinycap import contracts as tc
        from sisteer.eval import capability_ext as ext

        monkeypatch.delenv(tc.GSM8K_LEGACY_ENV, raising=False)
        monkeypatch.setattr(ext, "_gsm8k_rows",
                            lambda **k: pytest.fail("rows loaded before the guard"))
        model, tok = tiny_llama
        with pytest.raises(ext.LegacyGSM8KRefused, match="capability_panel"):
            ext.capability_panel(model, tok, None, scales=[])

    def test_capability_panel_smoke(self, tiny_llama, monkeypatch):
        from sisteer.capeval.tinycap import contracts as tc
        from sisteer.eval.capability_ext import capability_panel

        monkeypatch.setenv(tc.GSM8K_LEGACY_ENV, "1")   # the legacy panel, re-enabled (PW3)
        model, tok = tiny_llama
        rows4 = [{"question": "Q?", "choices": ["a", "b", "c", "d"], "answer": 0}]
        gsm = [{"question": "2+2?", "answer": "#### 4"}]
        panel = capability_panel(
            model, tok, None, scales=[],
            mmlu_rows=rows4, gsm8k_rows=gsm, gpqa_rows=rows4,
            mmlu_pro_rows_=rows4, max_new_tokens=4,
        )
        assert set(panel) == {0.0}
        assert set(panel[0.0]) == {
            "mmlu", "gsm8k", "gsm8k_extraction_failure_rate",
            "gpqa_diamond", "mmlu_pro",
        }


# ---------------------------------------------------------------------------
# LIFBench adapter
# ---------------------------------------------------------------------------


def _write_fake_vendor(root, records=None):
    """A minimal checkout exercising the adapter mechanics (import-by-path,
    func_map assertion, label mutation, data layout)."""
    ev = root / "evaluation"
    ev.mkdir(parents=True)
    names = {
        "judge_onedoc_repeat", "judge_onedoc_qa", "judge_onedoc_extract",
        "judge_label_equal_output", "judge_labels_equal_outputs",
        "judge_list_input_blur_offset_query", "judge_multidoc_batch_label",
        "judge_multidoc_find_dup_text",
    }
    body = ["import json, re\n"]
    for name in sorted(names):
        body.append(
            f"def {name}(output_dict):\n"
            "    output_dict['label']['mutated'] = True\n"  # upstream judges mutate
            "    ok = output_dict['label']['expected'] in output_dict['output']\n"
            "    return {'format': float(ok), 'total_score': float(ok)}\n"
        )
    (ev / "EvaluateFunc.py").write_text("\n".join(body))
    (ev / "ARS_Scores.py").write_text(
        "import numpy as np\n"
        "class ScoreTools:\n"
        "    score_map = {t: {'total_score': 4.0} for t in [\n"
        "        'onedoc-repeat', 'onedoc-qa', 'onedoc-extract',\n"
        "        'list-single_query_id', 'list-multi_query_id',\n"
        "        'list-offset_query_id', 'list-offset_query_element',\n"
        "        'list-blur_offset_query_id', 'list-blur_offset_query_element',\n"
        "        'multidoc-batch_label', 'multidoc-find_dup_text']}\n"
        "    def get_total_scores_weights(self, columns):\n"
        "        w = np.array([self.score_map[c]['total_score'] for c in columns])\n"
        "        return w / w.sum()\n"
    )
    prompts = root / "data" / "prompts"
    prompts.mkdir(parents=True)
    if records is None:
        records = [
            {"prompt": "Find the id.", "label": {"expected": "alpha"},
             "param": {}, "length": 4000, "ins_id": 0, "param_id": 0},
            {"prompt": "Find the id again.", "label": {"expected": "beta"},
             "param": {}, "length": 8000, "ins_id": 0, "param_id": 1},
        ]
    (prompts / "list-single_query_id.json").write_text(json.dumps(records))
    return records


def _stable_vendor_records(n_instances=4, lengths=(4000, 8000)):
    """Instance-stable fake data: the SAME (ins_id, param_id) set at every
    interval — the structure the real benchmark has (verified at the
    pinned commit) and the held-out split depends on."""
    return [
        {"prompt": f"Find item {i}.", "label": {"expected": f"item{i}"},
         "param": {}, "length": length, "ins_id": 0, "param_id": i}
        for length in lengths
        for i in range(n_instances)
    ]


class TestLIFBenchAdapter:
    def test_task_map_covers_11_tasks(self):
        from sisteer.eval.lifbench import TASKS

        assert len(TASKS) == 11
        scenarios = {t.split("-")[0] for t in TASKS}
        assert scenarios == {"list", "multidoc", "onedoc"}

    def test_missing_vendor_fails_loudly(self, tmp_path):
        from sisteer.eval.lifbench import load_judges

        with pytest.raises(FileNotFoundError, match="LIFBench"):
            load_judges(tmp_path / "nope")

    def test_adapter_maps_generations_into_scorer_format(self, tiny_llama, tmp_path):
        from sisteer.eval.lifbench import (
            ars_by_length,
            load_task_records,
            run_lifbench,
            score_output,
            load_judges,
            stability_ifs,
        )

        _, tok = tiny_llama
        vendor = tmp_path / "vendor"
        _write_fake_vendor(vendor)

        judges = load_judges(vendor)
        records = load_task_records("list-single_query_id", vendor)

        # deepcopy protection: upstream judges mutate label in place
        rec0 = records[0]
        score = score_output("list-single_query_id", rec0, "alpha is the id", judges)
        assert score["total_score"] == 1.0
        assert "mutated" not in rec0["label"]

        rows = run_lifbench(
            None, tok, "prompted", tmp_path / "out",
            vendor_dir=vendor, tasks=["list-single_query_id"],
            generate_fn=lambda rendered: "alpha and beta both mentioned",
        )
        assert len(rows) == 2
        assert all(r["score_dict"]["total_score"] == 1.0 for r in rows)

        # per-item resume
        again = run_lifbench(
            None, tok, "prompted", tmp_path / "out",
            vendor_dir=vendor, tasks=["list-single_query_id"],
            generate_fn=lambda rendered: pytest.fail("must resume"),
        )
        assert again == []

        curve = ars_by_length(rows)
        assert set(curve["length"]) == {4000, 8000}
        ifs = stability_ifs(rows)
        assert ifs["per_arm"]["prompted"] == pytest.approx(0.0)  # flat = stable

    def test_prompt_builder_seam(self, tiny_llama, tmp_path):
        from sisteer.eval.lifbench import run_lifbench

        _, tok = tiny_llama
        vendor = tmp_path / "vendor"
        _write_fake_vendor(vendor)
        seen = []

        rows = run_lifbench(
            None, tok, "vector", tmp_path / "out2",
            vendor_dir=vendor, tasks=["list-single_query_id"],
            prompt_builder=lambda r: f"STRIPPED::{r['prompt']}",
            generate_fn=lambda rendered: (seen.append(rendered), "alpha beta")[1],
        )
        assert len(rows) == 2
        assert all("STRIPPED::" in s for s in seen)

    def test_token_interval_length_filter_fails_loudly(self, tiny_llama, tmp_path):
        """M7: the on-disk `length` field holds coded interval labels, not
        token counts; a token-interval filter that matches nothing must
        RAISE instead of 'succeeding' with an empty benchmark."""
        from sisteer.eval.lifbench import run_lifbench

        _, tok = tiny_llama
        vendor = tmp_path / "vendor"
        _write_fake_vendor(vendor)
        with pytest.raises(ValueError, match="matched no"):
            run_lifbench(
                None, tok, "prompted", tmp_path / "out3",
                vendor_dir=vendor, tasks=["list-single_query_id"],
                lengths=[4096, 8192],  # token intervals, not the codes
                generate_fn=lambda rendered: "alpha",
            )

    def test_load_results_survives_resume_and_torn_lines(self, tiny_llama, tmp_path):
        """M8: aggregation must read the FULL on-disk jsonl (run_lifbench
        returns only new rows — empty after a no-op resume), tolerating a
        torn final line."""
        from sisteer.eval.lifbench import load_results, run_lifbench, stability_ifs

        _, tok = tiny_llama
        vendor = tmp_path / "vendor"
        _write_fake_vendor(vendor)
        out = tmp_path / "out4"
        run_lifbench(
            None, tok, "prompted", out,
            vendor_dir=vendor, tasks=["list-single_query_id"],
            generate_fn=lambda rendered: "alpha and beta both mentioned",
        )
        again = run_lifbench(
            None, tok, "prompted", out,
            vendor_dir=vendor, tasks=["list-single_query_id"],
            generate_fn=lambda rendered: pytest.fail("must resume"),
        )
        assert again == []  # the return value under-counts after resume...
        jsonl = out / "prompted.jsonl"
        with jsonl.open("a") as f:
            f.write('{"task": "list-single_query_id", "arm": "pro')  # torn
        rows = load_results(jsonl)
        assert len(rows) == 2  # ...load_results reads everything intact
        assert stability_ifs(rows)["per_arm"]["prompted"] is not None

    def test_stability_ifs_single_interval_is_none_and_inf_propagates(self):
        """M14: <2 length intervals -> None (no stability signal, never a
        fake 'perfectly stable' 0); a failed-everywhere task stays inf and
        propagates into the per-arm mean."""
        from sisteer.eval.lifbench import stability_ifs

        single = [{"arm": "a", "task": "t1", "length": 3,
                   "score_dict": {"total_score": 1.0}}]
        out = stability_ifs(single)
        assert out["per_task"][("a", "t1")] is None
        assert out["per_arm"]["a"] is None

        failed = [
            {"arm": "a", "task": "t1", "length": l,
             "score_dict": {"total_score": 0.0}} for l in (3, 6)
        ] + [
            {"arm": "a", "task": "t2", "length": l,
             "score_dict": {"total_score": 1.0}} for l in (3, 6)
        ]
        out = stability_ifs(failed)
        assert out["per_task"][("a", "t1")] == float("inf")
        assert out["per_arm"]["a"] == float("inf")  # not silently dropped

    def test_appended_token_kind_injected_in_lifbench(self, tiny_llama, tmp_path):
        """Consistency: the lifbench generation path injects BOTH token
        kinds."""
        from contextlib import contextmanager

        from sisteer.eval.lifbench import run_lifbench

        model, tok = tiny_llama
        vendor = tmp_path / "vendor"
        _write_fake_vendor(vendor)
        injected = []

        class FakeAppended:
            kind = "token_appended"

            def inject(self, prompt):
                injected.append(prompt)
                return prompt + "<SENT-LF>"

            @contextmanager
            def apply(self, model, scale=1.0):
                yield

        run_lifbench(
            model, tok, "token_faithful", tmp_path / "out5",
            vendor_dir=vendor, tasks=["list-single_query_id"],
            obj=FakeAppended(), max_new_tokens=2, max_per_task=1,
        )
        assert len(injected) == 1


class TestLIFBenchHeldOutSplit:
    """Compiled objects train on shortest-interval base instances whose
    SCALED VARIANTS reappear at every interval — scoring them anywhere is
    train-on-test. The split excludes them everywhere and every arm
    (prompted included) scores the same held-out set."""

    TASK = "list-single_query_id"

    def test_split_deterministic_and_partitioning(self, tmp_path):
        from sisteer.eval.lifbench import (
            load_task_records,
            train_heldout_split,
        )

        vendor = tmp_path / "vendor"
        _write_fake_vendor(vendor, _stable_vendor_records())
        records = load_task_records(self.TASK, vendor)
        train, held = train_heldout_split(records, 3, 0, task=self.TASK)
        assert len(train) == 3 and len(held) == 1
        assert not set(train) & set(held)
        # deterministic per (task, seed) — resume-safe across processes
        again = train_heldout_split(records, 3, 0, task=self.TASK)
        assert (train, held) == again

    def test_train_instances_excluded_at_every_interval_all_arms(
        self, tiny_llama, tmp_path
    ):
        from sisteer.eval.lifbench import (
            load_task_records,
            run_lifbench,
            train_heldout_split,
        )

        _, tok = tiny_llama
        vendor = tmp_path / "vendor"
        _write_fake_vendor(vendor, _stable_vendor_records())
        records = load_task_records(self.TASK, vendor)
        train, held = train_heldout_split(records, 3, 0, task=self.TASK)

        def run(arm):
            return run_lifbench(
                None, tok, arm, tmp_path / "out",
                vendor_dir=vendor, tasks=[self.TASK],
                n_train=3, split_seed=0,
                generate_fn=lambda rendered: "item0 item1 item2 item3",
            )

        rows = run("vector")
        scored = {(r["ins_id"], r["param_id"]) for r in rows}
        assert scored == set(held)  # ONLY held-out instances scored
        # ... at EVERY interval, not just the training one
        assert {r["length"] for r in rows} == {4000, 8000}
        # recorded: instance id (ins_id/param_id) + split tag + seed
        assert all(r["split"] == "heldout" and r["split_seed"] == 0
                   for r in rows)
        # the prompted baseline scores the SAME held-out set
        prows = run("prompted")
        assert {(r["ins_id"], r["param_id"]) for r in prows} == scored

    def test_contaminated_results_file_fails_loudly(self, tiny_llama, tmp_path):
        """Scoring-path enforcement: a results file already holding a
        training-split instance's record (earlier contaminated run or a
        different seed) raises — never silent inclusion via resume."""
        import json as _json

        from sisteer.eval.lifbench import (
            load_task_records,
            run_lifbench,
            train_heldout_split,
        )

        _, tok = tiny_llama
        vendor = tmp_path / "vendor"
        _write_fake_vendor(vendor, _stable_vendor_records())
        records = load_task_records(self.TASK, vendor)
        train, _ = train_heldout_split(records, 3, 0, task=self.TASK)
        out = tmp_path / "out"
        out.mkdir()
        poison = {"task": self.TASK, "arm": "vector",
                  "ins_id": train[0][0], "param_id": train[0][1],
                  "length": 4000, "scale": 1.0, "output": "x",
                  "score_dict": {"total_score": 1.0}}
        (out / "vector.jsonl").write_text(_json.dumps(poison) + "\n")
        with pytest.raises(ValueError, match="TRAINING-split"):
            run_lifbench(
                None, tok, "vector", out, vendor_dir=vendor,
                tasks=[self.TASK], n_train=3, split_seed=0,
                generate_fn=lambda rendered: "item0",
            )

    def test_split_knobs_travel_together_and_exhaustion_raises(
        self, tiny_llama, tmp_path
    ):
        from sisteer.eval.lifbench import (
            load_task_records,
            run_lifbench,
            train_heldout_split,
        )

        _, tok = tiny_llama
        vendor = tmp_path / "vendor"
        _write_fake_vendor(vendor, _stable_vendor_records())
        with pytest.raises(ValueError, match="BOTH n_train and split_seed"):
            run_lifbench(
                None, tok, "vector", tmp_path / "o", vendor_dir=vendor,
                tasks=[self.TASK], n_train=3,
                generate_fn=lambda rendered: "x",
            )
        records = load_task_records(self.TASK, vendor)
        with pytest.raises(ValueError, match="no held-out"):
            train_heldout_split(records, 4, 0, task=self.TASK)  # 4 of 4

    def test_unstable_instance_identity_fails_loudly(self, tiny_llama, tmp_path):
        """The prior stability check died mid-run; it is now re-verified at
        run time — data whose (ins_id, param_id) sets differ across
        intervals must refuse the split rather than silently leak trained
        instances (the DEFAULT fake vendor data is exactly that shape)."""
        from sisteer.eval.lifbench import run_lifbench

        _, tok = tiny_llama
        vendor = tmp_path / "vendor"
        _write_fake_vendor(vendor)  # (0,0)@4000 and (0,1)@8000 — unstable
        with pytest.raises(ValueError, match="not a stable"):
            run_lifbench(
                None, tok, "vector", tmp_path / "o2", vendor_dir=vendor,
                tasks=[self.TASK], n_train=1, split_seed=0,
                generate_fn=lambda rendered: "x",
            )

    def test_training_records_driver_seam(self, tmp_path):
        from sisteer.eval.lifbench import (
            instance_key,
            load_task_records,
            train_heldout_split,
            training_records,
        )

        vendor = tmp_path / "vendor"
        _write_fake_vendor(vendor, _stable_vendor_records())
        records = load_task_records(self.TASK, vendor)
        train, _ = train_heldout_split(records, 3, 0, task=self.TASK)
        train_recs = training_records(records, 3, 0, task=self.TASK,
                                      length=4000)
        assert {instance_key(r) for r in train_recs} == set(train)
        assert all(r["length"] == 4000 for r in train_recs)
        with pytest.raises(ValueError, match="matches no"):
            training_records(records, 3, 0, task=self.TASK, length=4096)


# Integration against the REAL pinned checkout — xfail when vendor/lifbench
# is absent (the suite was removed from the campaign, so the fetch script
# no longer provisions it; a checkout already on disk still runs).
from sisteer.eval.lifbench import vendor_present  # noqa: E402

@pytest.mark.xfail(
    not vendor_present(),
    reason="vendor/lifbench not present (LIFBench removed from the "
           "campaign, D30 — no longer provisioned by fetch_stage2_data.py)",
    run=False,
)
def test_lifbench_real_vendored_scorer_scores_an_instance(tiny_llama, tmp_path):
    from sisteer.eval.lifbench import (
        TASKS,
        load_judges,
        load_task_records,
        score_output,
    )

    judges = load_judges()
    task = "list-single_query_id"
    assert task in TASKS
    records = load_task_records(task)
    assert records, "vendored prompts present"
    rec = records[0]
    before = json.dumps(rec["label"], sort_keys=True, default=str)
    score = score_output(task, rec, "a garbage response", judges)
    assert "total_score" in score
    assert 0.0 <= score["total_score"] <= 1.0
    assert json.dumps(rec["label"], sort_keys=True, default=str) == before


# ---------------------------------------------------------------------------
# suites dispatcher
# ---------------------------------------------------------------------------


def test_suite_dispatcher_resolves_all_suites():
    from sisteer.eval.suites import SUITES, get_runner

    assert set(SUITES) == {"ifeval", "capability", "composition"}
    for suite in SUITES:
        assert callable(get_runner(suite))
    with pytest.raises(KeyError):
        get_runner("jailbreak")  # separate spec, deliberately absent
    with pytest.raises(KeyError):
        get_runner("lifbench")  # removed from the campaign
    for removed in ("ifeval_anchor", "drift", "axbench", "longctx"):
        with pytest.raises(KeyError, match=removed):
            get_runner(removed)  # third-party suites, removed rather than attributed
