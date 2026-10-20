"""P1-1: pilot -> production -> top-up, end-to-end on the fake, ZERO force
flags, previously-completed artifacts byte-identical, UNION drives the filter.
Uses the real repo constitution + templates; only the client is fake."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from corpus_generator.config import RunConfig
from corpus_generator.constitution import load as load_constitution
from corpus_generator.pipeline import load_union_records, run_corpus, make_engine
from corpus_generator.templates import repo_root
from fakes import FakeBatchClient, PipelineResponder


def _cfg(run_id: str, tmp_path: Path, scale: int = 1) -> RunConfig:
    return RunConfig.model_validate(
        {
            "run_id": run_id,
            "data_dir": str(tmp_path / "data"),
            "advice": {
                "n_principles": 2, "n_types": 2, "n_scenarios_per_call": scale,
            },
            "stories": {
                "n_categories": 2, "n_subcategories": 2, "subcategories_use": None,
                "calls_per_subcategory": 1, "stories_per_call": 2,
                "length_menu": [100], "length_weights": [1.0],
            },
            "engine": {
                "poll_interval_s": 0.0, "visibility_wait_s": 0.0,
                "nonce_metadata": True,
            },
        }
    )


def _snapshot(root: Path, exclude: set[str]) -> dict[str, str]:
    out = {}
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.name not in exclude and not p.name.endswith(".lock"):
            out[str(p.relative_to(root))] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


MUTABLE = {"UNION.json", "manifest.json", "status.json"}


def _make_client() -> FakeBatchClient:
    _md, sections, _snap = load_constitution(repo_root() / "constitution")
    return FakeBatchClient(PipelineResponder(sections))


def test_full_lifecycle_pilot_production_topup(tmp_path):
    root = repo_root()

    # ---- pilot ----------------------------------------------------------
    pilot = _cfg("pilot-test", tmp_path, scale=2)
    client = _make_client()
    run_corpus(pilot, "advice", make_engine(pilot, client, sleep=lambda s: None),
               root=root, on_event=lambda m: None)
    run_corpus(pilot, "stories", make_engine(pilot, client, sleep=lambda s: None),
               root=root, on_event=lambda m: None)
    pilot_dir = pilot.run_dir(root)
    pilot_records = load_union_records(pilot_dir, "advice")
    assert len(pilot_records) == 2 * 2 * 2  # principles x types x scenarios
    assert len(load_union_records(pilot_dir, "stories")) == 2 * 2 * 1 * 2
    pilot_hashes = _snapshot(pilot_dir, MUTABLE)

    # ---- production (separate namespace; pilot untouched) ---------------
    prod = _cfg("production-test", tmp_path, scale=2)
    client2 = _make_client()
    run_corpus(prod, "advice", make_engine(prod, client2, sleep=lambda s: None),
               root=root, on_event=lambda m: None)
    run_corpus(prod, "stories", make_engine(prod, client2, sleep=lambda s: None),
               root=root, on_event=lambda m: None)
    prod_dir = prod.run_dir(root)
    assert prod_dir != pilot_dir
    assert _snapshot(pilot_dir, MUTABLE) == pilot_hashes, "pilot artifacts mutated"
    prod_hashes = _snapshot(prod_dir, MUTABLE)
    n_calls_after_prod = len(client2.calls)

    # ---- top-up: extension waves, base untouched, zero force flags ------
    (prod_dir / "advice" / "TOPUP-001.json").write_text(
        json.dumps({"n": 1, "pairs": ["p00-t00", "p01-t01"], "slot_start": 2, "n_slots": 1})
    )
    (prod_dir / "stories" / "TOPUP-001.json").write_text(
        json.dumps({"n": 1, "call_start": 1, "n_calls": 1})
    )
    run_corpus(prod, "advice", make_engine(prod, client2, sleep=lambda s: None),
               root=root, on_event=lambda m: None)
    run_corpus(prod, "stories", make_engine(prod, client2, sleep=lambda s: None),
               root=root, on_event=lambda m: None)

    # base artifacts byte-identical (hashes snapshotted before/after)
    after = _snapshot(prod_dir, MUTABLE)
    for rel, h in prod_hashes.items():
        assert after[rel] == h, f"completed artifact rewritten: {rel}"

    # extension sub-DAG exists: advice spans A3->E, stories S3->S5
    for stage in ("a3", "b1", "b2", "b3", "c", "d", "e"):
        assert (prod_dir / "advice" / f"{stage}-topup-001" / "STAGE.json").exists(), stage
    for stage in ("s3", "s4", "s5"):
        assert (prod_dir / "stories" / f"{stage}-topup-001" / "STAGE.json").exists(), stage

    # extension scenarios were CREATED before B1 used them
    ext_b1 = [
        json.loads(line)
        for line in (prod_dir / "advice" / "b1-topup-001" / "requests.jsonl")
        .read_text().splitlines()
    ]
    assert {r["lineage"]["scenario_key"] for r in ext_b1} == {"p00-t00-s02", "p01-t01-s02"}

    # UNION includes exactly base + extension outputs
    union = json.loads((prod_dir / "UNION.json").read_text())
    assert union["advice"] == ["advice/records.jsonl", "advice/records-topup-001.jsonl"]
    assert union["stories"] == ["stories/records.jsonl", "stories/records-topup-001.jsonl"]
    all_recs = load_union_records(prod_dir, "advice")
    assert len(all_recs) == 8 + 2
    all_sto = load_union_records(prod_dir, "stories")
    assert len(all_sto) == 8 + 2 * 2 * 1 * 2  # one extra call per lane, 2 stories each

    assert len(client2.calls) > n_calls_after_prod, "top-up must have hit the API"


def test_rerun_is_noop_for_stamped_waves(tmp_path):
    root = repo_root()
    cfg = _cfg("noop-test", tmp_path)
    client = _make_client()
    run_corpus(cfg, "advice", make_engine(cfg, client, sleep=lambda s: None),
               root=root, on_event=lambda m: None)
    n_creates = client.create_count()
    run_corpus(cfg, "advice", make_engine(cfg, client, sleep=lambda s: None),
               root=root, on_event=lambda m: None)
    assert client.create_count() == n_creates, "re-run must be a no-op"


def test_ordinary_flag_propagates_s1_to_story_records(tmp_path):
    """C4: S1's `ordinary` flag survives S2/S3/S4 lineage into every
    StoryRecord (the responder marks alternating categories ordinary)."""
    root = repo_root()
    cfg = _cfg("ordinary-test", tmp_path)
    client = _make_client()
    run_corpus(cfg, "stories", make_engine(cfg, client, sleep=lambda s: None),
               root=root, on_event=lambda m: None)
    recs = load_union_records(cfg.run_dir(root), "stories")
    assert recs
    assert all(isinstance(r["ordinary"], bool) for r in recs)
    by_cat = {r["category_id"]: r["ordinary"] for r in recs}
    assert by_cat == {"c00": True, "c01": False}  # PipelineResponder: i % 2 == 0


def test_lost_scoring_wave_record_materializes_unscored_and_filter_drops_it(tmp_path):
    """Cap-based rejection fix, downstream semantics: an advice record whose
    e-wave request (and both repairs — a small NEW batch each, exercising the
    lifted repair-wave cap) is rejected, and a story losing s5 likewise, must
    materialize with scores=None and be dropped by the filter with the counted
    reason 'unscored' — never silently passing the score gate, and never
    silently ceasing to exist."""
    import json as _json

    from corpus_generator.filtering import filter_corpus
    from fakes import error_result

    root = repo_root()
    cfg = _cfg("unscored-test", tmp_path)
    # final-mode thresholds (permissive) so the filter exercises the gate
    th = cfg.filter.thresholds
    th.advice_alignment = th.advice_realism = 0
    th.story_consistency = th.story_quality = 0
    cfg.filter.min_story_words = 10  # the fake's stories are ~80 words

    client = _make_client()
    base_responder = client.responder
    lost = {"advice": "adv-p00-t00-s00", "stories": "sto-c00-u00-k00-y0"}

    import re as _re

    def responder(cid, params, attempt):
        # persistent rejection of one e request and one s5 request (base and
        # -rN repairs alike) — the s5 production signature. cid is the WIRE id
        # (run-tagged by pipeline.make_engine); match on the logical part but
        # echo the wire id back in the envelope, like the real API.
        logical = _re.sub(r"^[0-9a-f]{8}-", "", cid)
        if logical.startswith("adv-e-p00-t00-s00") or logical.startswith(
            "sto-s5-c00-u00-k00-y0"
        ):
            return error_result(
                cid, error_type="invalid_request_error", message="Invalid request data"
            )
        return base_responder(cid, params, attempt)

    client.responder = responder
    for corpus in ("advice", "stories"):
        run_corpus(
            cfg, corpus,
            # 1 rejection in a small base wave needs a loosened cap; the
            # repair waves (1/1 rejected) rely on the stage runner's cap
            # lift and would halt without it
            make_engine(cfg, client, sleep=lambda s: None, max_rejected_frac=0.5),
            root=root, on_event=lambda m: None,
        )

    for corpus, lost_id in lost.items():
        recs = load_union_records(cfg.run_dir(root), corpus)
        unscored = [r for r in recs if r.get("scores") is None]
        assert len(unscored) == 1, corpus
        assert unscored[0]["lineage"]["custom_id"] == lost_id
        out = tmp_path / f"{corpus}-final.jsonl"
        report = filter_corpus(corpus, recs, cfg, None, out)
        assert report.rule_counts["unscored"] == 1, corpus
        finals = [_json.loads(x) for x in out.read_text().splitlines()]
        assert lost_id not in {r["lineage"]["custom_id"] for r in finals}
        assert all(r["scores"] is not None for r in finals)


def test_records_carry_full_lineage(tmp_path):
    root = repo_root()
    cfg = _cfg("lineage-test", tmp_path)
    client = _make_client()
    run_corpus(cfg, "advice", make_engine(cfg, client, sleep=lambda s: None),
               root=root, on_event=lambda m: None)
    recs = load_union_records(cfg.run_dir(root), "advice")
    assert recs
    r = recs[0]
    for field in (
        "system", "user", "assistant", "principle", "prompt_type", "scenario",
        "section_id", "diagnostic_response", "initial_response", "scores",
    ):
        assert field in r, field
    assert r["assistant"].startswith("advice adv-d-")
    assert r["initial_response"].startswith("initial adv-c-")
    lin = r["lineage"]
    assert lin["run_id"] == "lineage-test"
    assert set(lin["batch_ids"]) == {"a3", "b1", "b2", "b3", "c", "d", "e"}
    assert all(v.startswith("batch_") for v in lin["batch_ids"].values())
