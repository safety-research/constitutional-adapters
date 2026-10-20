"""Offline tests for ``scripts/compare_served_hf_task.py`` and the paired
statistics in ``sisteer.capeval.tiers.pairing`` added for U5-5 (lane W20a):
the paired-bootstrap CI on the per-item delta next to the analytic paired
SE, the avg@k reduction to per-item means BEFORE pairing (the bootstrap
resamples items, never samples), the unmatched handling under
``--allow-unmatched``, the per-side unparseable / truncation facts derived
only from fields the rows carry, and the JSON shape. Everything synthetic:
invented items, placeholder ids, no recorded data. The pre-existing pairing
behaviour is pinned in ``tests/test_evalplus_served.py`` (generalized
pairing section) and is not repeated here."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pytest

from sisteer.capeval.tiers import pairing as pr

REPO = Path(__file__).resolve().parents[1]

OLD = "s2parity/gsm8k_0shot"          # the retired instrument (engine hf)
NEW = "inspect/gsm8k_paired200"       # the standard (engine vllm, paired_id = old id)


def _cli():
    spec = importlib.util.spec_from_file_location(
        "compare_served_hf_task", REPO / "scripts" / "compare_served_hf_task.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _row(task, iid, value, engine, extra=None, metric="acc"):
    return {"task": task, "metric": metric, "item_id": iid, "value": value, "engine": engine,
            "extra_json": json.dumps(extra) if extra is not None else None}


def _write(path: Path, rows) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def _items(values, n_samples=1):
    return {k: pr.Item(float(v), k, n_samples=n_samples) for k, v in values.items()}


def _old_id(i: int) -> str:
    return f"g{i:04d}-{i:08x}"


def _new_id(i: int) -> str:
    return f"gsm8k_{i:08x}"


def _old_rows(n: int, *, acc, failed=(), fallback=()):
    """The retired instrument's store: an ``acc`` row (pred / used_fallback
    in extra) plus an ``extraction_failed`` sibling row per item, and an
    aggregate row that must never pair."""
    rows = []
    for i in range(n):
        iid = _old_id(i)
        f = i in failed
        rows.append(_row(OLD, iid, float(acc(i)), "hf",
                         {"pred": None if f else 42.0, "used_fallback": f or i in fallback}))
        rows.append(_row(OLD, iid, 1.0 if f else 0.0, "hf", metric="extraction_failed"))
    rows.append(_row(OLD, "__aggregate__", 0.5, "hf",
                     {"extraction_failures_new": len(failed), "n": n}, metric="acc_aggregate"))
    return rows


def _new_extra(*, answer="7", paired_id=None, completion_tokens=100, max_tokens=512,
               content="Reasoning: ... ANSWER: 7", reasoning=None, sample_idx=0, seed=1, **more):
    e = {"answer": answer, "completion_tokens": completion_tokens, "max_tokens": max_tokens,
         "content": content, "content_empty_with_reasoning": bool(reasoning) and content == "",
         "paired_id": paired_id, "prompt_tokens": 300, "reasoning": reasoning,
         "sample_idx": sample_idx, "scores": {"match": 1.0}, "seed": seed, "thinking": "off"}
    e.update(more)
    return e


# --------------------------------------------------------------------------- #
# the paired bootstrap CI and the analytic paired SE                          #
# --------------------------------------------------------------------------- #


def test_bootstrap_ci_hand_computable_on_two_items():
    # deltas {0, 1}: the resampled mean is 0 / 0.5 / 1 with probability 1/4, 1/2, 1/4,
    # so the 95 % percentile interval is exactly [0, 1]; sd(ddof=1) = 1/sqrt(2), se = 0.5.
    a = _items({"x": 0, "y": 0})
    b = _items({"x": 0, "y": 1})
    r = pr.compare(a, b)
    pb = r["paired_bootstrap"]
    assert pb == {"n_items": 2, "delta": 0.5, "ci_low": 0.0, "ci_high": 1.0, "level": 0.95,
                  "B": 10_000, "seed": 0, "excludes_zero": False}
    assert r["paired_se"] == pytest.approx(0.5)
    assert r["delta_b_minus_a"] == pytest.approx(pb["delta"])
    assert r["paired_bootstrap_unavailable_reason"] is None


def test_bootstrap_ci_degenerates_when_every_delta_is_equal():
    a = _items({"p": 0, "q": 1, "r": 0})
    b = _items({"p": 1, "q": 2, "r": 1})       # every item +1: nothing to resample away
    r = pr.compare(a, b)
    pb = r["paired_bootstrap"]
    assert (pb["delta"], pb["ci_low"], pb["ci_high"]) == (1.0, 1.0, 1.0)
    assert pb["excludes_zero"] is True
    assert r["paired_se"] == 0.0
    assert not r["binary"] and r["mcnemar"] is None


def test_bootstrap_ci_agrees_with_the_analytic_se_and_is_seeded():
    # 200 binary items: 20 gained (b only), 5 lost (a only), 175 unchanged.
    # delta = 15/200 = 0.075; sd(ddof=1) of the deltas = sqrt((25 - 200*0.075^2)/199).
    a = _items({f"i{i:03d}": 1.0 if i < 150 else 0.0 for i in range(200)})
    b = {k: pr.Item(v.value, k) for k, v in a.items()}
    for i in range(150, 170):
        b[f"i{i:03d}"].value = 1.0
    for i in range(5):
        b[f"i{i:03d}"].value = 0.0
    r = pr.compare(a, b)
    se = math.sqrt((25 - 200 * 0.075 ** 2) / 199) / math.sqrt(200)
    assert r["delta_b_minus_a"] == pytest.approx(0.075)
    assert r["paired_se"] == pytest.approx(se)
    pb = r["paired_bootstrap"]
    assert pb["n_items"] == 200 and pb["excludes_zero"] is True
    # the percentile CI sits on the normal approximation to within the 1/200 step
    assert pb["ci_low"] == pytest.approx(0.075 - 1.96 * se, abs=0.012)
    assert pb["ci_high"] == pytest.approx(0.075 + 1.96 * se, abs=0.012)
    assert r["mcnemar"]["a_only"] == 5 and r["mcnemar"]["b_only"] == 20
    # seeded: the same seed reproduces the interval byte for byte; another seed moves it
    again = pr.compare(a, b)["paired_bootstrap"]
    assert again == pb
    other = pr.compare(a, b, bootstrap_seed=7)["paired_bootstrap"]
    assert other["delta"] == pb["delta"] and other["seed"] == 7
    assert (other["ci_low"], other["ci_high"]) != (pb["ci_low"], pb["ci_high"])
    # a narrower level gives a narrower interval; B and level are echoed
    narrow = pr.compare(a, b, bootstrap=2_000, ci=0.8)["paired_bootstrap"]
    assert narrow["B"] == 2_000 and narrow["level"] == 0.8
    assert narrow["ci_low"] >= pb["ci_low"] and narrow["ci_high"] <= pb["ci_high"]


def test_bootstrap_can_be_disabled_and_refuses_bad_parameters():
    a = _items({"x": 0, "y": 1})
    b = _items({"x": 1, "y": 1})
    r = pr.compare(a, b, bootstrap=0)
    assert r["paired_bootstrap"] is None
    assert "disabled" in r["paired_bootstrap_unavailable_reason"]
    assert r["paired_se"] == pytest.approx(0.5)     # the analytic SE never needs the bootstrap
    with pytest.raises(pr.PairingError, match="interval level"):
        pr.compare(a, b, ci=1.5)
    with pytest.raises(pr.PairingError, match=">= 1 resamples"):
        pr.compare(a, b, bootstrap=-3)
    with pytest.raises(pr.PairingError, match="keys differ"):
        pr.paired_bootstrap({"x": 0.0}, {"y": 1.0})
    assert pr.paired_se([0.3]) is None                # one item: undefined, not 0
    one = pr.compare(_items({"x": 0}), _items({"x": 1}))
    assert one["paired_se"] is None and one["paired_bootstrap"]["ci_low"] == 1.0


# --------------------------------------------------------------------------- #
# avg@k: reduce to the per-item mean FIRST, then pair                          #
# --------------------------------------------------------------------------- #


def test_avgk_sample_rows_reduce_to_item_means_before_pairing(tmp_path):
    ref = _write(tmp_path / "ref.jsonl", [
        _row("inspect/math", "m1", 1.0, "vllm"), _row("inspect/math", "m2", 0.0, "vllm"),
        _row("inspect/math", "m3", 1.0, "vllm"), _row("inspect/math", "m4", 1.0, "vllm")])
    rows = []
    for iid, vals in (("m1", (1, 1, 0)), ("m2", (0, 0, 0)), ("m3", (1, 0, 0)), ("m4", (1, 1, 1))):
        for k, v in enumerate(vals):
            rows.append(_row("inspect/math", iid, float(v), "vllm",
                             {"sample_idx": k, "seed": 100 + k, "answer": str(v)}))
    rows.append(_row("inspect/math", "__aggregate__", 0.5, "vllm", metric="acc_aggregate"))
    samp = _write(tmp_path / "samples.jsonl", rows)
    items, stats = pr.load_items(pr.Side("k3", samp, "inspect/math", "vllm"))
    assert {k: it.value for k, it in items.items()} == {
        "m1": pytest.approx(2 / 3), "m2": 0.0, "m3": pytest.approx(1 / 3), "m4": 1.0}
    assert all(it.n_samples == 3 for it in items.values())
    assert items["m3"].extra["values"] == [1.0, 0.0, 0.0] and "sample_idx" not in items["m3"].extra
    assert stats["n_rows"] == 12 and stats["n_items"] == 4
    assert stats["samples_per_item"] == {"min": 3, "max": 3}
    r = pr.compare_sides(pr.Side("ref", ref, "inspect/math", "vllm"),
                         pr.Side("k3", samp, "inspect/math", "vllm"))
    # the bootstrap resamples the 4 ITEMS (n_items = 4, not 12 samples)
    assert r["n_items"] == 4 and r["paired_bootstrap"]["n_items"] == 4
    assert r["samples_per_item"] == {"a": {"min": 1, "max": 1}, "b": {"min": 3, "max": 3}}
    assert r["mean_a"] == 0.75 and r["mean_b"] == pytest.approx(0.5)
    assert r["delta_b_minus_a"] == pytest.approx(-0.25)
    assert not r["binary"] and r["mcnemar"] is None and "avg@k" in r["mcnemar_unavailable_reason"]
    assert [d["id"] for d in r["discordant"]] == ["m1", "m3"]
    assert r["discordant"][1]["b"]["values"] == [1.0, 0.0, 0.0]
    assert r["sides"]["b"]["samples_per_item"] == {"min": 3, "max": 3}


def test_avgk_duplicates_still_refuse(tmp_path):
    same_idx = _write(tmp_path / "same.jsonl", [
        _row("t", "0", 1.0, "hf", {"sample_idx": 0}), _row("t", "0", 0.0, "hf", {"sample_idx": 0})])
    with pytest.raises(pr.PairingError, match="duplicate.*distinct sample_idx"):
        pr.load_items(pr.Side("x", same_idx, "t", "hf"))
    no_idx = _write(tmp_path / "noidx.jsonl", [
        _row("t", "0", 1.0, "hf", {"sample_idx": 0}), _row("t", "0", 0.0, "hf")])
    with pytest.raises(pr.PairingError, match="duplicate"):
        pr.load_items(pr.Side("x", no_idx, "t", "hf"))


def test_producer_mean_row_reports_its_samples_per_item(tmp_path):
    # the producer's layout: the bare `acc` row IS the per-item mean (samples_per_item,
    # seeds, values in its extra); the per-sample rows sit under acc@<k> and are ignored
    rows = []
    for iid, vals in (("m1", (1.0, 1.0, 0.0, 0.0)), ("m2", (1.0, 1.0, 1.0, 1.0))):
        rows.append(_row("inspect/math", iid, sum(vals) / 4, "vllm",
                         {"sample_idx": 0, "samples_per_item": 4, "seeds": [1, 2, 3, 4],
                          "values": list(vals), "answer": None}))
        for k, v in enumerate(vals):
            rows.append(_row("inspect/math", iid, v, "vllm", {"sample_idx": k}, metric=f"acc@{k}"))
    p = _write(tmp_path / "mean.jsonl", rows)
    items, stats = pr.load_items(pr.Side("mean", p, "inspect/math", "vllm"))
    assert items["m1"].value == 0.5 and items["m1"].n_samples == 4
    assert items["m1"].extra["values"] == [1.0, 1.0, 0.0, 0.0]
    assert stats["n_rows"] == 2 and stats["samples_per_item"] == {"min": 4, "max": 4}
    # a single acc@0 sample row is one sample
    one, st1 = pr.load_items(pr.Side("s0", p, "inspect/math", "vllm", metric="acc@0"))
    assert one["m1"].n_samples == 1 and st1["samples_per_item"] == {"min": 1, "max": 1}
    # the sympy scorer records no answer: that is "not recorded", never 100 % unparseable
    q = stats["quality"]
    assert q["answer_recorded"] is False and q["answer_missing"] is None
    assert q["unparseable"] is None and q["unparseable_rate"] is None
    assert "not derivable" in q["unparseable_source"]


# --------------------------------------------------------------------------- #
# unmatched items, the two real row shapes, the JSON report                    #
# --------------------------------------------------------------------------- #


def _gsm8k_stores(tmp_path):
    """Side A: 6 old items (g0000..g0005), g0001 and g0004 unparseable under the
    old extractor (acc 0). Side B: the standard's rows for g0001..g0005 plus
    g0009 (no old partner) and one row with a null paired_id; g0001 now
    parses and is right, g0004 hits the token cap with no number found,
    g0003 was right and is now wrong."""
    old = _old_rows(6, acc=lambda i: 0.0 if i in (1, 4) else 1.0, failed=(1, 4), fallback=(2,))
    new = []
    spec = {1: {"answer": "8", "value": 1.0},
            2: {"answer": "3", "value": 1.0},
            3: {"answer": "12", "value": 0.0},
            4: {"answer": "", "value": 0.0, "completion_tokens": 512,
                "content": "I keep going and going and"},
            5: {"answer": "5", "value": 1.0}}
    for i, s in spec.items():
        value = s.pop("value")
        new.append(_row(NEW, _new_id(i), value, "vllm", _new_extra(paired_id=_old_id(i), **s)))
    new.append(_row(NEW, _new_id(9), 1.0, "vllm", _new_extra(paired_id=_old_id(9), answer="1")))
    new.append(_row(NEW, _new_id(77), 0.0, "vllm",
                    _new_extra(paired_id=None, answer="", content="", reasoning="hmm",
                               completion_tokens=512)))
    new.append(_row(NEW, "__aggregate__", 0.6, "vllm", metric="acc_aggregate"))
    new.append(_row("inspect/gsm8k", _new_id(1), 1.0, "vllm", _new_extra()))   # another task
    return _write(tmp_path / "old.jsonl", old), _write(tmp_path / "new.jsonl", new)


def test_unmatched_items_are_listed_and_the_bootstrap_runs_on_the_overlap(tmp_path):
    old, new = _gsm8k_stores(tmp_path)
    a = pr.Side("old", old, OLD, "hf")
    b = pr.Side("new", new, NEW, "vllm", key=pr.KEY_PAIRED_ID)
    with pytest.raises(pr.PairingError, match="item sets differ"):
        pr.compare_sides(a, b)
    r = pr.compare_sides(a, b, allow_unmatched=True)
    assert r["n_items"] == 5 and r["only_a"] == [_old_id(0)] and r["only_b"] == [_old_id(9)]
    assert r["mean_a"] == pytest.approx(3 / 5) and r["mean_b"] == pytest.approx(3 / 5)
    assert r["delta_b_minus_a"] == pytest.approx(0.0)
    assert r["paired_bootstrap"]["n_items"] == 5
    assert r["paired_bootstrap"]["ci_low"] < 0 < r["paired_bootstrap"]["ci_high"]
    assert r["paired_se"] == pytest.approx(math.sqrt(2 / 4) / math.sqrt(5))
    assert r["mcnemar"]["a_only"] == 1 and r["mcnemar"]["b_only"] == 1
    assert [d["id"] for d in r["discordant"]] == [_old_id(1), _old_id(3)]
    assert r["discordant"][0]["a"]["pred"] is None and r["discordant"][0]["b"]["answer"] == "8"
    assert r["sides"]["b"]["unpaired_rows"] == 1 and r["sides"]["b"]["n_rows"] == 7
    assert r["sides"]["a"]["n_items"] == 6 and r["sides"]["b"]["n_items"] == 6


def test_side_quality_reads_only_fields_the_rows_carry(tmp_path):
    old, new = _gsm8k_stores(tmp_path)
    _, sa = pr.load_items(pr.Side("old", old, OLD, "hf"))
    qa = sa["quality"]
    assert qa["n_rows"] == 6
    assert qa["extraction_failed"] == 2 and qa["extraction_failed_rows"] == 6
    assert qa["pred_missing"] == 2 and qa["used_fallback"] == 3
    assert qa["unparseable"] == 2 and qa["unparseable_rate"] == pytest.approx(2 / 6)
    assert qa["unparseable_source"].startswith("extraction_failed rows")
    # the old rows carry none of the Inspect fields: null, not 0
    assert qa["answer_recorded"] is False and qa["answer_missing"] is None
    assert qa["content_empty"] is None and qa["content_empty_with_reasoning"] is None
    assert qa["at_token_cap"] is None
    assert qa["finish_reason"] == pr.FINISH_REASON_NOT_RECORDED == "not recorded by the producer"

    _, sb = pr.load_items(pr.Side("new", new, NEW, "vllm", key=pr.KEY_PAIRED_ID))
    qb = sb["quality"]
    assert qb["n_rows"] == 7                        # the null-paired_id row counts on the side
    assert qb["answer_recorded"] is True and qb["answer_missing"] == 2
    assert qb["unparseable"] == 2 and qb["unparseable_rate"] == pytest.approx(2 / 7)
    assert "match_str found no number" in qb["unparseable_source"]
    assert qb["content_empty"] == 1 and qb["content_empty_with_reasoning"] == 1
    assert qb["at_token_cap"] == 2
    assert qb["finish_reason"] == "not recorded by the producer"
    assert qb["finish_reason_field"] is None
    assert qb["extraction_failed"] is None and qb["pred_missing"] is None
    assert qb["used_fallback"] is None


def test_a_recorded_finish_reason_is_counted_not_guessed():
    extras = [{"answer": "1", "finish_reason": "stop"}, {"answer": "", "finish_reason": "length"},
              {"answer": "2", "finish_reason": "stop"}]
    q = pr.side_quality(extras)
    assert q["finish_reason"] == {"length": 1, "stop": 2} and q["finish_reason_field"] == "finish_reason"
    assert q["unparseable"] == 1 and q["at_token_cap"] is None
    assert pr.side_quality([])["unparseable"] is None


def test_cli_json_shape_report_text_and_backward_compatible_keys(tmp_path, capsys):
    mod = _cli()
    old, new = _gsm8k_stores(tmp_path)
    out = tmp_path / "compare" / "cell.json"
    mod.main(["--task-a", OLD, "--engine-a", "hf", "--task-b", NEW, "--engine-b", "vllm",
              "--key-b", "paired_id", "--allow-unmatched",
              "--bootstrap", "2000", "--bootstrap-seed", "3", "--ci", "0.9",
              "--cell", "llama8b_none", str(old), str(new), "--out", str(out)])
    rep = json.loads(out.read_text())["llama8b_none"]
    # the pre-existing keys are all still there
    assert {"n_items", "mean_a", "mean_b", "delta_b_minus_a", "binary", "n_discordant",
            "discordant", "mcnemar", "mcnemar_unavailable_reason", "pick_agreement",
            "n_pick_disagreements", "only_a", "only_b", "sides"} <= set(rep)
    pb = rep["paired_bootstrap"]
    assert set(pb) == {"n_items", "delta", "ci_low", "ci_high", "level", "B", "seed",
                       "excludes_zero"}
    assert pb["n_items"] == 5 and pb["level"] == 0.9 and pb["B"] == 2000 and pb["seed"] == 3
    assert pb["ci_low"] <= pb["delta"] <= pb["ci_high"]
    assert rep["paired_se"] == pytest.approx(math.sqrt(0.5) / math.sqrt(5))
    assert rep["samples_per_item"] == {"a": {"min": 1, "max": 1}, "b": {"min": 1, "max": 1}}
    for s in ("a", "b"):
        side = rep["sides"][s]
        assert {"n_rows", "n_items", "samples_per_item", "quality"} <= set(side)
        assert {"unparseable", "unparseable_rate", "unparseable_source", "answer_missing",
                "content_empty", "content_empty_with_reasoning", "at_token_cap",
                "finish_reason", "extraction_failed", "pred_missing",
                "used_fallback"} <= set(side["quality"])
    assert rep["sides"]["a"]["quality"]["unparseable"] == 2
    assert rep["sides"]["b"]["quality"]["at_token_cap"] == 2
    text = capsys.readouterr().out
    assert "delta_b_minus_a=+0.0000 90% CI [" in text and "paired bootstrap B=2000 seed=3" in text
    assert "paired_se=0.3162" in text and "only_a=1 only_b=1" in text
    assert "samples/item: A=1 B=1" in text
    assert "A quality: n_rows=6 unparseable=2/6 rate=0.3333 [extraction_failed rows" in text
    assert "B quality: n_rows=7 unparseable=2/7" in text and "at_token_cap=2" in text
    assert "finish_reason=not recorded by the producer" in text
    assert f"discordant {_old_id(1)}:" in text
    # --bootstrap 0: the CI is absent and says why; everything else stands
    mod.main(["--task-a", OLD, "--task-b", NEW, "--key-b", "paired_id", "--allow-unmatched",
              "--bootstrap", "0", "--cell", "c", str(old), str(new)])
    text0 = capsys.readouterr().out
    assert "ci=n/a (paired bootstrap disabled" in text0 and "paired_se=0.3162" in text0


def test_cli_defaults_are_the_documented_ones():
    ap = _cli().build_parser()
    args = ap.parse_args(["--task-a", "t", "--task-b", "u", "--cell", "x", "a", "b"])
    assert (args.bootstrap, args.bootstrap_seed, args.ci) == (10_000, 0, 0.95)
    assert (pr.DEFAULT_BOOTSTRAP, pr.DEFAULT_BOOTSTRAP_SEED, pr.DEFAULT_CI) == (10_000, 0, 0.95)


# --------------------------------------------------------------------------- #
# E46 — two served-lens stores must have served the same lens reduction mode   #
# --------------------------------------------------------------------------- #


def _store(tmp_path: Path, name: str, rows, *, route=..., producer=None) -> Path:
    """A synthetic cell store: ``rows.jsonl`` plus, unless ``route`` is left
    out entirely, the ``manifest.json`` a served run writes beside it.
    ``route=None`` writes a manifest with NO route block (an HF reference
    store); a mapping writes it verbatim. ``producer`` writes the block that
    DATES the store (E55 #1) — needed only where a route block naming no mode
    must read as a legacy per_row store rather than a torn one."""
    d = tmp_path / name
    d.mkdir(parents=True, exist_ok=True)
    rows_path = _write(d / "rows.jsonl", rows)
    if route is not ...:
        doc = {"cell": name} if route is None else {"cell": name, "route": dict(route)}
        if producer is not None:
            doc["producer"] = dict(producer)
        (d / pr.STORE_MANIFEST_NAME).write_text(json.dumps(doc))
    return rows_path


def _two_item_rows(task, engine, values):
    return [_row(task, f"i{i}", v, engine) for i, v in enumerate(values)]


def test_store_reduction_mode_names_all_four_cases(tmp_path):
    """E55 #3: this reader now calls ``contracts.read_store_reduction_mode``,
    the SAME function ``cell_controls`` calls, so the two seams cannot disagree
    about one store. What changed here: a route block naming no mode is legacy
    per_row only on a store that positively predates the stamp, and the three
    "the store HAS the field and cannot say" shapes RAISE instead of being read
    as per_row or reported as nothing."""
    from sisteer.capeval.tinycap import contracts as tc

    old_tree = {"tree_sha": "618f9ddeedc26d90fe93ed78ea96688a171d9d7e"}
    rows = _two_item_rows(NEW, "vllm", [1.0, 0.0])
    stacked = _store(tmp_path, "stacked", rows, route={"reduction_mode": "stacked"})
    assert pr.store_reduction_mode(stacked) == ("stacked", str(
        stacked.parent / pr.STORE_MANIFEST_NAME) + " route.reduction_mode")
    # a route block that names no mode is LEGACY = per_row on a store that
    # positively predates the stamp: every store served before the landing was
    # served per-row
    legacy = _store(tmp_path, "legacy", rows, route={"virtual_model": "vm"},
                    producer=old_tree)
    mode, why = pr.store_reduction_mode(legacy)
    assert mode == "per_row" and "PRE-LANDING" in why
    # the SAME silence on a store that cannot date itself is a torn manifest,
    # and refuses here exactly as it does at the merge (E55 #3)
    undated = _store(tmp_path, "undated", rows, route={"virtual_model": "vm"})
    with pytest.raises(pr.PairingError, match="E55 #3"):
        pr.store_reduction_mode(undated)
    # no route block at all: not a served-lens store, so no mode — NEVER
    # coerced to per_row, which is what would let a stacked store pool in
    hf = _store(tmp_path, "hf", rows, route=None)
    mode, why = pr.store_reduction_mode(hf)
    assert mode is None and "no 'route' block" in why
    # no manifest / an unparseable one: reported, not raised
    bare = _store(tmp_path, "bare", rows)
    mode, why = pr.store_reduction_mode(bare)
    assert mode is None and "not a readable manifest" in why
    (bare.parent / pr.STORE_MANIFEST_NAME).write_text("{not json")
    mode, why = pr.store_reduction_mode(bare)
    assert mode is None and "not a readable manifest" in why
    # an unknown mode, and a key PRESENT and null, are not silently accepted
    weird = _store(tmp_path, "weird", rows, route={"reduction_mode": "grouped"})
    with pytest.raises(pr.PairingError, match="not one of"):
        pr.store_reduction_mode(weird)
    nulled = _store(tmp_path, "nulled", rows,
                    route={"reduction_mode": None,
                           tc.STORE_REDUCTION_MODE_ERROR_FIELD: "the server did not say"})
    with pytest.raises(pr.PairingError, match="the server did not say"):
        pr.store_reduction_mode(nulled)


def test_reduction_mode_mismatch_predicate():
    assert pr.reduction_mode_mismatch("stacked", "stacked") is None
    assert pr.reduction_mode_mismatch("per_row", "per_row") is None
    assert pr.reduction_mode_mismatch(None, "stacked") is None   # one side states nothing
    assert pr.reduction_mode_mismatch("stacked", None) is None
    why = pr.reduction_mode_mismatch("stacked", "per_row")
    assert why is not None and "not poolable" in why and "E46" in why


def test_compare_sides_refuses_a_stacked_store_against_a_per_row_one(tmp_path):
    rows_a = _two_item_rows(NEW, "vllm", [1.0, 0.0])
    rows_b = _two_item_rows(NEW, "vllm", [1.0, 1.0])
    a = _store(tmp_path, "cell_a", rows_a, route={"reduction_mode": "stacked"})
    b = _store(tmp_path, "cell_b", rows_b, route={"reduction_mode": "per_row"})
    with pytest.raises(pr.PairingError, match="not poolable"):
        pr.compare_sides(pr.Side("a", a, NEW, "vllm"), pr.Side("b", b, NEW, "vllm"),
                         bootstrap=0)
    # a LEGACY store (route block, no mode, on a tree that predates the stamp)
    # is per_row, so it refuses against a stacked one too — that is the pooling
    # this guard exists to stop
    legacy = _store(tmp_path, "cell_legacy", rows_b, route={"virtual_model": "vm"},
                    producer={"tree_sha": "618f9ddeedc26d90fe93ed78ea96688a171d9d7e"})
    with pytest.raises(pr.PairingError, match="not poolable"):
        pr.compare_sides(pr.Side("a", a, NEW, "vllm"), pr.Side("l", legacy, NEW, "vllm"),
                         bootstrap=0)


def test_compare_sides_reports_both_modes_and_allows_the_hf_reference(tmp_path):
    rows_a = _two_item_rows(NEW, "vllm", [1.0, 0.0])
    rows_b = _two_item_rows(OLD, "hf", [1.0, 1.0])
    a = _store(tmp_path, "served", rows_a, route={"reduction_mode": "stacked"})
    b = _store(tmp_path, "hf_ref", rows_b, route=None)
    rep = pr.compare_sides(pr.Side("a", a, NEW, "vllm"), pr.Side("b", b, OLD, "hf"),
                           bootstrap=0)
    # the HF side states nothing and does NOT block the spot-check (which
    # pairs an HF reference against a served store), but the report says so
    assert rep["sides"]["a"]["reduction_mode"] == "stacked"
    assert rep["sides"]["b"]["reduction_mode"] is None
    assert "no 'route' block" in rep["sides"]["b"]["reduction_mode_source"]
    # two agreeing served stores pass and both say so
    c = _store(tmp_path, "served2", rows_a, route={"reduction_mode": "stacked"})
    rep = pr.compare_sides(pr.Side("a", a, NEW, "vllm"), pr.Side("c", c, NEW, "vllm"),
                           bootstrap=0)
    assert (rep["sides"]["a"]["reduction_mode"]
            == rep["sides"]["b"]["reduction_mode"] == "stacked")


def test_cli_prints_each_side_s_reduction_mode(tmp_path, capsys):
    mod = _cli()
    rows = _two_item_rows(NEW, "vllm", [1.0, 0.0])
    a = _store(tmp_path, "a_cell", rows, route={"reduction_mode": "stacked"})
    b = _store(tmp_path, "b_cell", _two_item_rows(NEW, "vllm", [1.0, 1.0]),
               route={"reduction_mode": "stacked"})
    mod.main(["--task-a", NEW, "--engine-a", "vllm", "--task-b", NEW, "--engine-b", "vllm",
              "--bootstrap", "0", "--cell", "c", str(a), str(b)])
    text = capsys.readouterr().out
    assert "A lens reduction mode: stacked [" in text
    assert "B lens reduction mode: stacked [" in text
