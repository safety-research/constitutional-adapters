"""ce_score — the standalone held-out CE scorer.

CPU-only, tiny model, synthetic doc-shaped rows. Pins: the PARITY PIN
(sum/sum is batching-invariant, so the scorer reproduces the trainer's own
``best_heldout_ce`` / ``heldout_noop_ce`` on a fit's held-out pool at a
DIFFERENT batch size, and an independent one-example-at-a-time
reference); per-doc records that sum to the aggregate with the trainer's
target counts under both prompt shapes; --scale honesty (2.0 differs from
1.0, 0.0 is the exact no-op); the identity gate on the loading path
(staged / bare / derived, each with its refusal); the over-length refusal
that never truncates; --limit; the CLI end to end; and the doc-level
bootstrap on synthetic records.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "scripts" / "sdfbase") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts" / "sdfbase"))

import ce_score as cs  # noqa: E402
from residual_fixtures import random_psr  # noqa: E402
from test_ce_trainer import _manual_micro_avg_ce  # noqa: E402
from test_scale_semantics import _lora_obj as _rank4_lora  # noqa: E402
from sisteer.data.pairs import Pair, docs_to_ce_examples  # noqa: E402
from sisteer.eval.defense_cell import sha256_file  # noqa: E402
from sisteer.eval.object_identity import validate_artifact_identity  # noqa: E402
from sisteer.extraction.multilayer import LoRASubtractionObject  # noqa: E402
from sisteer.extraction.objects import train_object_ce  # noqa: E402
from sisteer.extraction.psr import PSRObject  # noqa: E402

TINY = "hf-internal-testing/tiny-random-LlamaForCausalLM"
CELL = "constv2-mix-1k__psr_ce_all"
DERIVED_ID = "constv2-mix-1k__lora_r4_ce_minus_ctl"

# Synthetic doc rows in the pool file's shape: a loss-masked prefix in
# ``prompt`` (advice) or "" (stories), the trained text in ``pos``,
# deliberately UNEQUAL lengths so batching/bucketing effects are visible.
_ROWS = [
    ("advice", "Sys scaffold\n\n", "A short doc."),
    ("stories", "", "A much longer story body that runs on for quite a few "
                    "more tokens than the short one does, on purpose."),
    ("advice", "Another sys\n\n", "Medium length advice body follows here."),
    ("stories", "", "Tiny."),
    ("advice", "Third sys\n\n", "One more medium advice body for good measure."),
    ("stories", "", "Final story text, middling length overall."),
]


def _rows(n=6, ids=None):
    out = []
    for i, (fam, prompt, pos) in enumerate(_ROWS[:n]):
        out.append({"doc_id": (ids[i] if ids else f"doc{i:02d}"),
                    "family": fam, "prompt": prompt, "pos": pos, "neg": "",
                    "instruction_id": "constv2-mix-1k",
                    "official_tokens": 10 + i, "truncated": False})
    return out


def _examples(tok, rows):
    return docs_to_ce_examples(tok, [
        Pair(prompt=r["prompt"], pos=r["pos"], neg="",
             instruction_id=r["instruction_id"]) for r in rows])


def _write_pairs(path: Path, rows) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows),
                    encoding="utf-8")
    return path


def _psr_open_gate(model, seed=7):
    """A PSR object with an OPEN gate (b = 2, tiny w) and a nonzero
    direction on layer 0 — the object must move logits for the scale pins
    to bite."""
    return random_psr(model, seed, layers=[0], write=0.05, normalize=False,
                      gate_scale=0.01, b=2.0)


def _cli(tmp_path, pairs, *extra, name="scores"):
    out = tmp_path / f"{name}.jsonl"
    rc = cs.main(["--model", TINY, "--device", "cpu", "--dtype", "float32",
                  "--pairs", str(pairs), "--out", str(out), *extra])
    assert rc == 0
    recs = cs._read_jsonl(out)
    summary = json.loads(Path(str(out) + ".summary.json").read_text())
    return recs, summary


# --------------------------------------------------------------------------
# 1. PARITY PIN — the scorer reproduces the trainer's own held-out numbers
# --------------------------------------------------------------------------


def test_parity_pin_reproduces_trainer_heldout_ce(tiny_llama):
    """train_object_ce records best_heldout_ce (object active, best-epoch
    weights restored) and heldout_noop_ce (nullcontext) as sum/sum over
    its held-out pool at batch_size=2. Scoring the SAME rows with the
    trained object at scale 1.0 — at batch sizes 3 and 1, neither the
    trainer's — must reproduce both to numeric tolerance (a ratio of sums
    is batching-invariant), and both must equal the independent
    one-example-at-a-time reference."""
    model, tok = tiny_llama
    rows = _rows(6)
    fit_rows, held_rows = rows[:3], rows[3:]  # held: "", sys, "" prompts
    obj = PSRObject(model, [0, 1], train_direction=True)
    diag: dict = {}
    train_object_ce(model, tok, obj, _examples(tok, fit_rows), lr=1e-2,
                    epochs=2, batch_size=2, max_len=256,
                    heldout=_examples(tok, held_rows), diagnostics=diag)
    assert diag["best_heldout_ce"] is not None

    aggs = {}
    for bs in (3, 1):
        recs = cs.score_docs(model, tok, held_rows, obj=obj, scale=1.0,
                             batch_size=bs, max_len=256)
        aggs[bs] = cs.aggregate(recs)
        assert aggs[bs]["ce_active"] == pytest.approx(
            diag["best_heldout_ce"], rel=1e-5), f"batch_size={bs}"
        assert aggs[bs]["ce_noop"] == pytest.approx(
            diag["heldout_noop_ce"], rel=1e-5), f"batch_size={bs}"
    # the object really moved the held-out CE (parity is not the trivial
    # active == noop case)
    assert aggs[3]["ce_active"] != aggs[3]["ce_noop"]
    # independent reference: one example at a time, EOS appended once
    held_ex = _examples(tok, held_rows)
    assert aggs[3]["ce_active"] == pytest.approx(
        _manual_micro_avg_ce(model, tok, held_ex, obj=obj), rel=1e-5)
    assert aggs[3]["ce_noop"] == pytest.approx(
        _manual_micro_avg_ce(model, tok, held_ex, obj=None), rel=1e-5)


# --------------------------------------------------------------------------
# 2. Per-doc records: sum to the aggregate, trainer target counts, sorted
# --------------------------------------------------------------------------


def test_records_sum_to_aggregate_and_carry_trainer_target_counts(tiny_llama):
    model, tok = tiny_llama
    rows = _rows(6, ids=["z9", "b2", "a1", "m5", "c3", "k4"])  # unsorted ids
    recs = cs.score_docs(model, tok, rows, obj=_psr_open_gate(model),
                         scale=1.0, batch_size=4, max_len=256)
    assert [r["doc_id"] for r in recs] == sorted(r["doc_id"] for r in rows)
    assert len(recs) == len(rows)
    assert all(set(r) == set(cs.RECORD_KEYS) for r in recs)
    agg = cs.aggregate(recs)
    n = sum(r["n_target_tokens"] for r in recs)
    assert agg["n_docs"] == len(rows) and agg["n_tokens"] == n
    assert agg["ce_active"] == pytest.approx(
        sum(r["nll_active_sum"] for r in recs) / n)
    assert agg["ce_noop"] == pytest.approx(
        sum(r["nll_noop_sum"] for r in recs) / n)
    assert agg["ratio"] == pytest.approx(agg["ce_active"] / agg["ce_noop"])
    assert agg["gain"] == pytest.approx(1.0 - agg["ratio"])
    # families ride through from the rows
    by_id = {r["doc_id"]: r for r in rows}
    assert all(r["family"] == by_id[r["doc_id"]]["family"] for r in recs)
    # target count = completion tokens + 1 (EOS) for BOTH prompt shapes on
    # a BOS-adding tokenizer ("" -> [BOS], so position 0 is the BOS and
    # every completion token is a target)
    for r in recs:
        row = by_id[r["doc_id"]]
        n_comp = len(tok(row["pos"], add_special_tokens=False)["input_ids"])
        assert r["n_target_tokens"] == n_comp + 1, row["prompt"]


def test_target_count_bosless_empty_prompt_edge(tiny_llama):
    """qwen-shape tokenizer ('' -> [], prompt_len 0): the labels[:, 1:]
    shift drops exactly position 0's token from the targets, so a
    ""-prompt row counts completion + EOS − 1 while a sys-prompt row keeps
    completion + EOS — the trainer's shifted count, pinned per shape."""
    from transformers import AutoTokenizer

    model, _ = tiny_llama
    tok = AutoTokenizer.from_pretrained(TINY)
    tok.add_bos_token = False
    assert tok("", add_special_tokens=True)["input_ids"] == []
    rows = _rows(2)  # rows[0]: sys prompt, rows[1]: "" prompt
    recs = {r["doc_id"]: r for r in
            cs.score_docs(model, tok, rows, batch_size=2, max_len=256)}
    for row in rows:
        n_comp = len(tok(row["pos"], add_special_tokens=False)["input_ids"])
        expect = n_comp + 1 - (1 if row["prompt"] == "" else 0)
        assert recs[row["doc_id"]]["n_target_tokens"] == expect, row["prompt"]
    # and the aggregate equals the trainer's independent reference under
    # this shape too
    manual = _manual_micro_avg_ce(model, tok, _examples(tok, rows), obj=None)
    assert cs.aggregate(list(recs.values()))["ce_noop"] == pytest.approx(
        manual, rel=1e-5)


# --------------------------------------------------------------------------
# 3. --scale honesty
# --------------------------------------------------------------------------


def test_scale_changes_records_and_zero_is_exact_noop(tiny_llama):
    model, tok = tiny_llama
    rows = _rows(3)
    obj = _psr_open_gate(model)
    r1 = cs.score_docs(model, tok, rows, obj=obj, scale=1.0, batch_size=2,
                       max_len=256)
    r2 = cs.score_docs(model, tok, rows, obj=obj, scale=2.0, batch_size=2,
                       max_len=256)
    r0 = cs.score_docs(model, tok, rows, obj=obj, scale=0.0, batch_size=2,
                       max_len=256)
    # the no-op column is the same bare model whatever the scale
    assert [r["nll_noop_sum"] for r in r1] == [r["nll_noop_sum"] for r in r2]
    assert [r["nll_noop_sum"] for r in r1] == [r["nll_noop_sum"] for r in r0]
    # scale 1 and 2 are different active reads, both distinct from noop
    assert [r["nll_active_sum"] for r in r1] != [r["nll_active_sum"] for r in r2]
    assert all(r["nll_active_sum"] != r["nll_noop_sum"] for r in r1)
    assert all(r["nll_active_sum"] != r["nll_noop_sum"] for r in r2)
    # scale 0 is the EXACT no-op (bit-identical, not approximately)
    assert all(r["nll_active_sum"] == r["nll_noop_sum"] for r in r0)


class _RecordingObj:
    def __init__(self):
        self.seen: list[float] = []

    def inject(self, s):
        return s

    def apply(self, model, scale=1.0):
        from contextlib import contextmanager

        self.seen.append(float(scale))

        @contextmanager
        def _ctx():
            yield None
        return _ctx()


def test_score_docs_passes_scale_to_apply_per_batch(tiny_llama):
    model, tok = tiny_llama
    fake = _RecordingObj()
    cs.score_docs(model, tok, _rows(3), obj=fake, scale=2.0, batch_size=2,
                  max_len=256)
    assert fake.seen == [2.0, 2.0]  # one active context per batch (2 batches)


# --------------------------------------------------------------------------
# 4. Identity gate on the loading path (staged / bare / derived)
# --------------------------------------------------------------------------


def _doc_meta(cell_id: str, obj: Path, over: dict | None = None) -> dict:
    meta = {"cell_id": cell_id, "object_sha256": sha256_file(obj),
            "training_budget_kind": "document_tokens",
            "sequence_max_len": 256, "completion_tokens": None,
            "sample_manifest_sha": "ab" * 32, "max_len": 256}
    meta.update(over or {})
    return meta


def _stage_psr(root: Path, model, cell_id: str = CELL) -> Path:
    d = root / cell_id
    d.mkdir(parents=True)
    obj = d / "object.pt"
    _psr_open_gate(model).save(obj)
    (d / "train_budget_meta.json").write_text(json.dumps(_doc_meta(cell_id, obj)))
    return obj


def _derived_lora_sub(root: Path, model, *, derived_id: str = DERIVED_ID,
                      tamper_sha: bool = False) -> Path:
    """A derived lora_subtraction dir: object.pt (compose_blob of two
    saved rank-4 lora_all blobs, parents' identities embedded) +
    derived_meta.json naming the same parents, kind, object sha and
    derived_id."""
    d = root / derived_id
    d.mkdir(parents=True)
    mix, ctl = _rank4_lora(model, 11), _rank4_lora(model, 22)
    mix.save(d / "mix_parent.pt")
    ctl.save(d / "ctl_parent.pt")
    parent = lambda cid, tag: {  # noqa: E731
        "cell_id": cid, "model_key": "tinyllama",
        "object_sha256": tag * 64, "budget_meta_sha256": ("f" + tag) * 32,
        "model_id": TINY, "model_revision": "main"}
    provenance = {
        "parent_pos": parent("constv2-mix-1k__lora_r4_ce", "a"),
        "parent_neg": parent("constv2ctl-1k__lora_r4_ce", "b"),
        "derivation": "delta(parent_pos) - delta(parent_neg)",
    }
    blob = LoRASubtractionObject.compose_blob(
        torch.load(d / "mix_parent.pt", map_location="cpu", weights_only=True),
        torch.load(d / "ctl_parent.pt", map_location="cpu", weights_only=True),
        provenance=provenance)
    obj = d / "object.pt"
    torch.save(blob, obj)
    (d / "mix_parent.pt").unlink()
    (d / "ctl_parent.pt").unlink()
    meta = {**provenance, "kind": "lora_subtraction",
            "object_sha256": ("0" * 64 if tamper_sha else sha256_file(obj)),
            "derived_id": derived_id}
    (d / "derived_meta.json").write_text(json.dumps(meta))
    return obj


def test_staged_cell_passes_and_refuses_wrong_cell_or_tampered(tiny_llama,
                                                               tmp_path):
    model, _ = tiny_llama
    pairs = _write_pairs(tmp_path / "pairs.jsonl", _rows(3))
    obj = _stage_psr(tmp_path / "bank", model)
    recs, summary = _cli(tmp_path, pairs, "--object", str(obj), "--cell",
                         CELL, "--batch-size", "2", "--max-len", "256")
    assert summary["identity_kind"] == "staged"
    assert summary["cell"] == CELL
    assert summary["object_sha256"] == sha256_file(obj)
    assert len(recs) == 3
    assert all(r["nll_active_sum"] != r["nll_noop_sum"] for r in recs)
    # wrong --cell refuses BEFORE any model work
    with pytest.raises(SystemExit, match="ce-score"):
        cs.main(["--model", TINY, "--device", "cpu", "--pairs", str(pairs),
                 "--out", str(tmp_path / "x.jsonl"), "--object", str(obj),
                 "--cell", "constv2-mix-1m__psr_ce_all"])
    # tampered object bytes under a valid-looking sidecar refuse
    obj.write_bytes(b"swapped object bytes")
    with pytest.raises(SystemExit, match="object_sha256"):
        cs.main(["--model", TINY, "--device", "cpu", "--pairs", str(pairs),
                 "--out", str(tmp_path / "y.jsonl"), "--object", str(obj),
                 "--cell", CELL])


def test_bare_object_refuses_without_allow_bare(tiny_llama, tmp_path):
    model, _ = tiny_llama
    pairs = _write_pairs(tmp_path / "pairs.jsonl", _rows(2))
    d = tmp_path / "adhoc"
    d.mkdir()
    obj = d / "object.pt"
    _psr_open_gate(model).save(obj)
    with pytest.raises(SystemExit, match="allow-bare"):
        cs.main(["--model", TINY, "--device", "cpu", "--pairs", str(pairs),
                 "--out", str(tmp_path / "x.jsonl"), "--object", str(obj)])
    recs, summary = _cli(tmp_path, pairs, "--object", str(obj),
                         "--allow-bare", "--max-len", "256")
    assert summary["identity_kind"] == "bare"
    assert summary["cell"] is None
    assert len(recs) == 2


def test_derived_object_passes_identity_and_scores(tiny_llama, tmp_path):
    model, tok = tiny_llama
    pairs = _write_pairs(tmp_path / "pairs.jsonl", _rows(3))
    obj = _derived_lora_sub(tmp_path / "derived", model)
    assert validate_artifact_identity(obj, DERIVED_ID, consumer="test") == \
        "derived"
    recs, summary = _cli(tmp_path, pairs, "--object", str(obj), "--cell",
                         DERIVED_ID, "--batch-size", "2", "--max-len", "256")
    assert summary["identity_kind"] == "derived"
    assert summary["object_sha256"] == sha256_file(obj)
    assert len(recs) == 3
    assert all(r["nll_active_sum"] != r["nll_noop_sum"] for r in recs)
    # the derived read is the structural subtraction: it must match a
    # direct score_docs with the hydrated object
    from sisteer.methods import load_object

    sub = load_object(obj, model=model, tok=tok)
    assert isinstance(sub, LoRASubtractionObject)
    direct = cs.score_docs(model, tok, _rows(3), obj=sub, batch_size=2,
                           max_len=256)
    assert [r["nll_active_sum"] for r in direct] == pytest.approx(
        [r["nll_active_sum"] for r in recs])
    # a wrong derived id refuses
    with pytest.raises(SystemExit, match="wrong-identity"):
        cs.main(["--model", TINY, "--device", "cpu", "--pairs", str(pairs),
                 "--out", str(tmp_path / "x.jsonl"), "--object", str(obj),
                 "--cell", "some_other_derived_id"])


def test_derived_object_tampered_sidecar_refuses(tiny_llama, tmp_path):
    model, _ = tiny_llama
    pairs = _write_pairs(tmp_path / "pairs.jsonl", _rows(2))
    obj = _derived_lora_sub(tmp_path / "derived", model, tamper_sha=True)
    with pytest.raises(SystemExit, match="object_sha256 mismatch"):
        cs.main(["--model", TINY, "--device", "cpu", "--pairs", str(pairs),
                 "--out", str(tmp_path / "x.jsonl"), "--object", str(obj),
                 "--cell", DERIVED_ID])


def test_cell_without_object_refused(tmp_path):
    pairs = _write_pairs(tmp_path / "pairs.jsonl", _rows(1))
    with pytest.raises(SystemExit, match="meaningless without"):
        cs.main(["--model", TINY, "--pairs", str(pairs),
                 "--out", str(tmp_path / "x.jsonl"), "--cell", CELL])


def test_identity_gate_runs_before_the_model_loads():
    """Source-order pin: main() validates identity before
    load_model_and_tokenizer — a wrong-cell artifact must fail in
    milliseconds, never after a GPU allocation."""
    src = (REPO / "scripts" / "sdfbase" / "ce_score.py").read_text()
    body = src.split("def main", 1)[1]
    assert body.index("validate_artifact_identity(") < body.index(
        "load_model_and_tokenizer(")


# --------------------------------------------------------------------------
# 5. Over-length refusal (never truncates), empty completion, --limit
# --------------------------------------------------------------------------


def test_overlength_row_refused_by_doc_id_never_truncated(tiny_llama):
    from sisteer.extraction.distill import _encode_aligned

    model, tok = tiny_llama
    rows = _rows(2)
    rows[1] = {**rows[1], "doc_id": "the-long-one", "pos": "word " * 300}
    # _encode_aligned alone would silently truncate this row
    _encode_aligned(tok, _examples(tok, rows)[1], 64)
    with pytest.raises((SystemExit, ValueError), match="the-long-one"):
        cs.score_docs(model, tok, rows, batch_size=2, max_len=64)


def test_empty_completion_and_duplicate_doc_id_refused(tiny_llama):
    model, tok = tiny_llama
    rows = _rows(2)
    rows[0] = {**rows[0], "doc_id": "empty-one", "pos": ""}
    with pytest.raises(ValueError, match="empty-one"):
        cs.score_docs(model, tok, rows, batch_size=2, max_len=256)
    dup = _rows(2)
    dup[1]["doc_id"] = dup[0]["doc_id"]
    with pytest.raises(ValueError, match="duplicate doc_id"):
        cs.score_docs(model, tok, dup, batch_size=2, max_len=256)


def test_limit_scores_first_n_id_sorted_rows(tiny_llama, tmp_path):
    rows = _rows(5, ids=["e", "d", "c", "b", "a"])
    pairs = _write_pairs(tmp_path / "pairs.jsonl", rows)
    recs, summary = _cli(tmp_path, pairs, "--limit", "2", "--max-len", "256")
    assert [r["doc_id"] for r in recs] == ["a", "b"]
    assert summary["n_docs"] == 2 and summary["limit"] == 2


# --------------------------------------------------------------------------
# 6. CLI end to end (bare model) + the doc-level bootstrap
# --------------------------------------------------------------------------


def test_cli_bare_model_writes_records_and_summary(tiny_llama, tmp_path):
    rows = _rows(4)
    pairs = _write_pairs(tmp_path / "pairs.jsonl", rows)
    recs, summary = _cli(tmp_path, pairs, "--source-tag", "ctl",
                         "--batch-size", "2", "--max-len", "256")
    assert set(cs.SUMMARY_KEYS) <= set(summary)
    assert summary["n_docs"] == 4
    assert summary["object"] is None and summary["object_sha256"] is None
    assert summary["identity_kind"] is None and summary["cell"] is None
    assert summary["pairs_sha256"] == sha256_file(pairs)
    assert summary["model"] == TINY and summary["revision"] is None
    assert summary["device"] == "cpu" and summary["dtype"] == "float32"
    assert summary["scale"] == 1.0
    assert summary["max_len"] == 256 and summary["batch_size"] == 2
    assert summary["source_tag"] == "ctl"
    # bare model: both columns equal, ratio 1, gain 0
    assert all(r["nll_active_sum"] == r["nll_noop_sum"] for r in recs)
    assert all(r["source_tag"] == "ctl" for r in recs)
    assert summary["ce_active"] == summary["ce_noop"]
    assert summary["ratio"] == 1.0 and summary["gain"] == 0.0
    assert summary["n_tokens"] == sum(r["n_target_tokens"] for r in recs)
    boot = summary["gain_doc_boot90"]
    assert boot["lo"] <= boot["gain"] <= boot["hi"]
    assert boot["gain"] == 0.0 and boot["n_docs"] == 4


def _synthetic_records(gains, n_tok=50, ce=2.0):
    return [{"doc_id": f"s{i}", "family": None, "source_tag": None,
             "n_target_tokens": n_tok,
             "nll_noop_sum": n_tok * ce,
             "nll_active_sum": n_tok * ce * (1.0 - g)}
            for i, g in enumerate(gains)]


def test_doc_bootstrap_gain_brackets_point_and_collapses_for_one_doc():
    recs = _synthetic_records([0.02, 0.05, 0.08, 0.04, 0.06, 0.03, 0.07])
    point = cs.aggregate(recs)["gain"]
    boot = cs.doc_bootstrap_gain(recs, n_boot=2000, seed=0, level=0.90)
    assert boot["gain"] == pytest.approx(point)
    assert boot["lo"] <= boot["gain"] <= boot["hi"]
    assert boot["lo"] < boot["hi"]
    assert 0.02 <= boot["lo"] and boot["hi"] <= 0.08  # within the doc range
    assert boot["n_docs"] == 7 and boot["n_boot"] == 2000
    # deterministic under the seed; a different seed moves the endpoints
    again = cs.doc_bootstrap_gain(recs, n_boot=2000, seed=0, level=0.90)
    assert again == boot
    other = cs.doc_bootstrap_gain(recs, n_boot=2000, seed=1, level=0.90)
    assert (other["lo"], other["hi"]) != (boot["lo"], boot["hi"])
    # a wider level widens the interval
    wide = cs.doc_bootstrap_gain(recs, n_boot=2000, seed=0, level=0.99)
    assert wide["lo"] <= boot["lo"] and wide["hi"] >= boot["hi"]
    # one doc: every resample is that doc — the interval collapses
    one = cs.doc_bootstrap_gain(recs[:1], n_boot=100, seed=0)
    assert one["lo"] == one["gain"] == one["hi"] == pytest.approx(0.02)
    # the estimand is the ratio of SUMS, not a mean of per-doc gains:
    # unequal token counts weight the docs
    uneven = _synthetic_records([0.0, 0.10])
    uneven[1]["n_target_tokens"] = 300
    uneven[1]["nll_noop_sum"] = 300 * 2.0
    uneven[1]["nll_active_sum"] = 300 * 2.0 * 0.9
    g = cs.doc_bootstrap_gain(uneven, n_boot=10)["gain"]
    assert g == pytest.approx(1 - (50 * 2 + 300 * 1.8) / (50 * 2 + 300 * 2))
    assert g != pytest.approx(0.05)
