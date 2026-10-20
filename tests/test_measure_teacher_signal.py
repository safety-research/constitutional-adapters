"""measure_teacher_signal.py: validation before the 8B load, stratified
paired selection, per-family + two pooled floors (review findings 1/2/4 +
the stratified amendment).

The regression class these pin: the original script took ``pool[:n]`` of
a family-sorted pool (its floors were one family), accepted a ``--seed``
it never used, and validated nothing before paying full model startup —
``--n 0``, an empty pair file, or all-batches-skipped died inside
``statistics.mean``. All selection/statistics logic is importable and
tested without any model; the model-load boundary is tested by making the
load itself raise.

Synthetic pools only — invented prompts, never real store data.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import measure_teacher_signal as mts  # noqa: E402

import sisteer.models  # noqa: E402


def _row(family: str, i: int, prompt: str | None = None) -> dict:
    return {"prompt": prompt or f"{family} question {i} about the sky.",
            "pos": f"Le ciel {family} {i}",
            # The undefended completion — deterministic in (family, i), so
            # every SI's pool carries the SAME neg for the same identity,
            # like the real matched pools.
            "neg": f"The {family} sky {i} is plain blue today.",
            "family": family, "goal_id": f"g{i}", "variant_key": "v0",
            "instruction_id": "french"}


def _write_pools(tmp_path: Path, si_ids=("aa", "bb"), n_per_family=4,
                 families=("alpha", "beta")) -> Path:
    """Matched pools: same (family, goal_id, variant_key) and the same
    prompt across SIs; only pos differs. Family-sorted, like the real
    ones."""
    d = tmp_path / "pools"
    d.mkdir(exist_ok=True)
    for si in si_ids:
        rows = []
        for fam in sorted(families):
            for i in range(n_per_family):
                r = _row(fam, i)
                r["pos"] = f"{si} {r['pos']}"     # per-SI completion
                rows.append(r)
        (d / f"pairs_{si}.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rows) + "\n")
    return d


def _write_cfg(tmp_path: Path, si_ids=("aa", "bb")) -> Path:
    p = tmp_path / "cfg.yaml"
    p.write_text("model: never-loaded\ndtype: float32\nsi_ids:\n"
                 + "".join(f"  - {si}\n" for si in si_ids))
    return p


@pytest.fixture()
def no_model_load(monkeypatch):
    """The model load boundary: crossing it in a validation test is the
    failure being tested for."""
    def _boom(*a, **k):
        raise AssertionError("model load reached — validation must fail "
                             "BEFORE load_model_and_tokenizer")
    monkeypatch.setattr(sisteer.models, "load_model_and_tokenizer", _boom)


def _run(argv: list[str], monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["measure_teacher_signal.py"] + argv)
    mts.main()


def test_bad_args_fail_before_the_load(tmp_path, monkeypatch,
                                       no_model_load):
    cfg = _write_cfg(tmp_path)
    pools = _write_pools(tmp_path)
    with pytest.raises(SystemExit, match="per-family"):
        _run(["--config", str(cfg), "--pairs-dir", str(pools),
              "--per-family", "0"], monkeypatch)
    with pytest.raises(SystemExit, match="batch-size"):
        _run(["--config", str(cfg), "--pairs-dir", str(pools),
              "--batch-size", "0"], monkeypatch)
    with pytest.raises(SystemExit, match="missing config"):
        _run(["--config", str(tmp_path / "nope.yaml"),
              "--pairs-dir", str(pools)], monkeypatch)


def test_bad_pools_fail_before_the_load(tmp_path, monkeypatch,
                                        no_model_load):
    cfg = _write_cfg(tmp_path)
    pools = _write_pools(tmp_path)

    # missing file
    (pools / "pairs_bb.jsonl").unlink()
    with pytest.raises(SystemExit, match="missing pool file"):
        _run(["--config", str(cfg), "--pairs-dir", str(pools)], monkeypatch)

    # empty file
    (pools / "pairs_bb.jsonl").write_text("\n")
    with pytest.raises(SystemExit, match="empty pool file"):
        _run(["--config", str(cfg), "--pairs-dir", str(pools)], monkeypatch)

    # malformed row
    (pools / "pairs_bb.jsonl").write_text('{"pos": "x"}\n')
    with pytest.raises(SystemExit, match="required field 'prompt'"):
        _run(["--config", str(cfg), "--pairs-dir", str(pools)], monkeypatch)


def test_unpaired_pools_refuse_before_the_load(tmp_path, monkeypatch,
                                               no_model_load):
    """The cross-SI pairing is asserted, not assumed: pools that select
    different identities (or different prompt text under the same key)
    refuse before any GPU work."""
    cfg = _write_cfg(tmp_path)
    pools = _write_pools(tmp_path)
    rows = [json.loads(l) for l
            in (pools / "pairs_bb.jsonl").read_text().splitlines()]
    rows[0]["goal_id"] = "gX"          # different identity
    (pools / "pairs_bb.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n")
    with pytest.raises(SystemExit, match="NOT paired"):
        _run(["--config", str(cfg), "--pairs-dir", str(pools)], monkeypatch)

    pools2 = _write_pools(tmp_path)
    rows = [json.loads(l) for l
            in (pools2 / "pairs_bb.jsonl").read_text().splitlines()]
    rows[0]["prompt"] = "tampered text"   # same key would still mismatch
    rows[0]["goal_id"] = rows[0]["goal_id"]
    (pools2 / "pairs_bb.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n")
    with pytest.raises(SystemExit, match="NOT paired"):
        _run(["--config", str(cfg), "--pairs-dir", str(pools2)],
             monkeypatch)


def test_selection_is_stratified_paired_and_thin_tolerant(tmp_path):
    """min(per_family, available) from EVERY family by within-family
    sha256(prompt) rank; identical identities selected for every SI."""
    pools_dir = _write_pools(tmp_path, n_per_family=5,
                             families=("alpha", "beta", "gamma"))
    pools = {si: mts.load_pool(pools_dir / f"pairs_{si}.jsonl")
             for si in ("aa", "bb")}
    # thin gamma: drop all but one row, in BOTH pools (stays matched)
    for si in pools:
        pools[si] = [r for r in pools[si]
                     if r["family"] != "gamma" or r["goal_id"] == "g0"]
    sel = mts.validate_and_select(pools, per_family=3)
    fams = [pools["aa"][i]["family"] for i in sel["aa"]]
    assert {f: fams.count(f) for f in set(fams)} == \
        {"alpha": 3, "beta": 3, "gamma": 1}
    # paired: same keys, same prompts
    assert [mts.pool_key(pools["aa"][i]) for i in sel["aa"]] == \
        [mts.pool_key(pools["bb"][i]) for i in sel["bb"]]
    # ...and NOT the family-sorted prefix (all-alpha)
    assert set(fams) != {"alpha"}


def test_per_example_records_align_identity_and_value():
    """The persisted per-example block: one record per selected identity,
    in selection order, value None where nothing finite was measured —
    same length as the selection ALWAYS, and a loud error the moment the
    alignment is lost (job 30542 kept only aggregates, which is why the
    paired per-example test could not be computed from its output)."""
    sel = [{"family": "alpha", "goal_id": "g0", "variant_key": "v0",
            "prompt_sha8": "aaaaaaaa"},
           {"family": "beta", "goal_id": "g1", "variant_key": "v0",
            "prompt_sha8": "bbbbbbbb"}]
    recs = mts.per_example_records(sel, [0.5, None])
    assert len(recs) == len(sel)
    # identity fields ride through untouched, in order
    assert [{k: r[k] for k in sel[0]} for r in recs] == sel
    assert recs[0]["kl"] == 0.5 and recs[1]["kl"] is None
    assert sel[0].get("kl") is None      # input records not mutated
    with pytest.raises(ValueError, match="alignment lost"):
        mts.per_example_records(sel, [0.5])
    with pytest.raises(ValueError, match="alignment lost"):
        mts.per_example_records(sel, [0.5, 0.6, 0.7])


def test_sis_subset_validates_before_the_load(tmp_path, monkeypatch,
                                              no_model_load):
    """--sis must name real config SIs, at most once each — checked before
    any pool read or model load."""
    cfg = _write_cfg(tmp_path)
    pools = _write_pools(tmp_path)
    with pytest.raises(SystemExit, match="absent from"):
        _run(["--config", str(cfg), "--pairs-dir", str(pools),
              "--sis", "aa,nope"], monkeypatch)
    with pytest.raises(SystemExit, match="names no SIs"):
        _run(["--config", str(cfg), "--pairs-dir", str(pools),
              "--sis", " ,"], monkeypatch)
    with pytest.raises(SystemExit, match="repeats"):
        _run(["--config", str(cfg), "--pairs-dir", str(pools),
              "--sis", "aa,aa"], monkeypatch)


def test_summarize_per_family_and_the_two_pooled_floors():
    """The two pooled numbers answer different questions and are labelled:
    pool-weighted uses true pool shares; unweighted counts each family
    once. Both are recomputable from the per-family table (the sufficient
    statistic, which ships alongside them)."""
    vals = [1.0, 2.0, 3.0, 4.0]
    fams = ["alpha", "alpha", "beta", "beta"]
    out = mts.summarize(vals, fams, {"alpha": 90, "beta": 10})
    pf = out["per_family"]
    assert pf["alpha"]["mean_kl"] == pytest.approx(1.5)
    assert pf["beta"]["mean_kl"] == pytest.approx(3.5)
    assert pf["alpha"]["pool_share"] == pytest.approx(0.9)
    assert pf["alpha"]["ci95_lo"] is not None
    assert out["pool_weighted"]["mean_kl"] == pytest.approx(
        0.9 * 1.5 + 0.1 * 3.5)
    assert out["unweighted_over_families"]["mean_kl"] == pytest.approx(2.5)
    assert out["pool_weighted"]["ci95_lo"] is not None
    assert out["unweighted_over_families"]["ci95_hi"] is not None
    assert out["pool_weighted"]["families_without_intervals"] == []

    with pytest.raises(ValueError, match="labels"):
        mts.summarize(vals, fams[:-1], {"alpha": 1})
    with pytest.raises(ValueError, match="at least one"):
        mts.summarize([], [], {"alpha": 1})


def test_summarize_thin_family_reports_point_and_n_never_folds():
    """A thin stratum ships its point estimate WITH its n and no interval
    — and the pooled CIs go null with the thin family NAMED, rather than
    an interval that silently pretends the family resolved."""
    out = mts.summarize([1.0, 2.0, 5.0], ["alpha", "alpha", "beta"],
                        {"alpha": 50, "beta": 50})
    beta = out["per_family"]["beta"]
    assert beta["n"] == 1 and beta["mean_kl"] == pytest.approx(5.0)
    assert beta["sem"] is None
    assert beta["ci95_lo"] is None and beta["ci95_hi"] is None
    # per-family survives; pooled means still computable; intervals honest
    assert out["pool_weighted"]["mean_kl"] == pytest.approx(
        0.5 * 1.5 + 0.5 * 5.0)
    assert out["pool_weighted"]["ci95_lo"] is None
    assert out["pool_weighted"]["families_without_intervals"] == ["beta"]
    assert out["unweighted_over_families"]["ci95_lo"] is None


def test_end_to_end_on_the_tiny_model(tmp_path, monkeypatch):
    """The whole measurement path — stratified paired selection, the
    per-example estimator, per-family + two pooled floors, selected ids
    in the output json — runs against the tiny model, so the GPU run
    cannot be the first execution of this loop."""
    from conftest import TINY_LLAMA

    pools_dir = tmp_path / "pools"
    pools_dir.mkdir()
    rows = []
    for fam in ("alpha", "beta"):
        for i in range(3):
            r = _row(fam, i)
            r["instruction_id"] = "french"
            rows.append(r)
    (pools_dir / "pairs_french.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n")
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(f"model: {TINY_LLAMA}\ndtype: float32\n"
                   "si_ids:\n  - french\n")
    out = tmp_path / "floors.json"
    _run(["--config", str(cfg), "--pairs-dir", str(pools_dir),
          "--per-family", "2", "--batch-size", "2", "--max-len", "96",
          "--out", str(out)], monkeypatch)

    payload = json.loads(out.read_text())
    r = payload["results"]["french"]
    assert r["n_examples"] == 4 and r["families_covered"] == 2
    assert set(r["per_family"]) == {"alpha", "beta"}
    for fam in ("alpha", "beta"):
        fr = r["per_family"][fam]
        assert fr["n"] == 2 and fr["ci95_lo"] is not None
        assert fr["pool_share"] == pytest.approx(0.5)
    assert r["pool_weighted"]["mean_kl"] == pytest.approx(
        0.5 * r["per_family"]["alpha"]["mean_kl"]
        + 0.5 * r["per_family"]["beta"]["mean_kl"])
    sel = payload["selection"]
    assert sel["per_family_target"] == 2
    assert len(sel["selected"]) == 4
    assert all(len(s["prompt_sha8"]) == 8 for s in sel["selected"])
    # the selection is the recorded convention, not the file prefix
    assert sel["method"].startswith("sha256(prompt)")

    # per-example block: one record per selected identity, in selection
    # order, identity fields byte-identical to selection.selected, every
    # value finite here (nothing was dropped)
    pe = r["per_example"]
    assert len(pe) == len(sel["selected"])
    ident_keys = ("family", "goal_id", "variant_key", "prompt_sha8")
    assert [{k: row[k] for k in ident_keys} for row in pe] \
        == sel["selected"]
    assert all(isinstance(row["kl"], float) for row in pe)
    # ...and the identity-value pairing is REAL: each family's aggregate
    # is recomputable from its own per-example rows
    for fam in ("alpha", "beta"):
        fam_vals = [row["kl"] for row in pe if row["family"] == fam]
        assert len(fam_vals) == r["per_family"][fam]["n"]
        assert sum(fam_vals) / len(fam_vals) == pytest.approx(
            r["per_family"][fam]["mean_kl"])


def test_per_example_is_paired_across_sis_and_sis_subsets_it(tmp_path,
                                                             monkeypatch):
    """Two SIs on the tiny model: the persisted per-example identity
    sequences are IDENTICAL across SIs (a reader joins the two lists
    row-by-row — the paired comparison needs no re-derivation), and a
    --sis subset rerun selects and records exactly the identities the
    full run recorded for that SI."""
    from conftest import TINY_LLAMA

    si_ids = ("french", "multi_french_concise")
    pools_dir = tmp_path / "pools"
    pools_dir.mkdir()
    for si in si_ids:
        rows = []
        for fam in ("alpha", "beta"):
            for i in range(3):
                r = _row(fam, i)
                r["pos"] = f"{si} {r['pos']}"     # per-SI completion
                rows.append(r)
        (pools_dir / f"pairs_{si}.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rows) + "\n")
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(f"model: {TINY_LLAMA}\ndtype: float32\n"
                   "si_ids:\n" + "".join(f"  - {si}\n" for si in si_ids))

    out_full = tmp_path / "full.json"
    _run(["--config", str(cfg), "--pairs-dir", str(pools_dir),
          "--per-family", "2", "--batch-size", "2", "--max-len", "96",
          "--out", str(out_full)], monkeypatch)
    full = json.loads(out_full.read_text())
    assert list(full["results"]) == list(si_ids)
    ident_keys = ("family", "goal_id", "variant_key", "prompt_sha8")
    idents = {si: [{k: row[k] for k in ident_keys}
                   for row in full["results"][si]["per_example"]]
              for si in si_ids}
    assert idents["french"] == idents["multi_french_concise"]
    assert idents["french"] == full["selection"]["selected"]
    # per-SI values differ (different pos) — the pairing is on identity
    assert [r["kl"] for r in full["results"]["french"]["per_example"]] != \
        [r["kl"] for r in
         full["results"]["multi_french_concise"]["per_example"]]

    out_sub = tmp_path / "sub.json"
    _run(["--config", str(cfg), "--pairs-dir", str(pools_dir),
          "--per-family", "2", "--batch-size", "2", "--max-len", "96",
          "--sis", "multi_french_concise", "--out", str(out_sub)],
         monkeypatch)
    sub = json.loads(out_sub.read_text())
    assert list(sub["results"]) == ["multi_french_concise"]
    assert sub["results"]["multi_french_concise"]["per_example"] == \
        full["results"]["multi_french_concise"]["per_example"]
    assert sub["selection"]["selected"] == full["selection"]["selected"]


def test_fixed_completion_flags_validate_before_load(tmp_path, monkeypatch,
                                                     no_model_load):
    """Every fixed-completion flag error refuses before any pool read
    matters and before the model load."""
    cfg = _write_cfg(tmp_path)
    pools = _write_pools(tmp_path)
    base = ["--config", str(cfg), "--pairs-dir", str(pools)]
    with pytest.raises(SystemExit, match="requires --fixed-completion"):
        _run(base + ["--instructions", "json"], monkeypatch)
    with pytest.raises(SystemExit, match="requires --instructions"):
        _run(base + ["--fixed-completion", "neg"], monkeypatch)
    with pytest.raises(SystemExit, match="subsets the own-pos"):
        _run(base + ["--fixed-completion", "neg", "--instructions", "json",
                     "--sis", "aa"], monkeypatch)
    with pytest.raises(SystemExit, match="unknown instruction id"):
        _run(base + ["--fixed-completion", "neg",
                     "--instructions", "json,zzz_not_an_instruction"],
             monkeypatch)
    with pytest.raises(SystemExit, match="repeats"):
        _run(base + ["--fixed-completion", "neg",
                     "--instructions", "json,json"], monkeypatch)
    with pytest.raises(SystemExit, match="names no instructions"):
        _run(base + ["--fixed-completion", "neg", "--instructions", " ,"],
             monkeypatch)


def test_fixed_completion_neg_must_be_common_before_load(tmp_path,
                                                         monkeypatch,
                                                         no_model_load):
    """The fixed-text premise is asserted, never assumed from the
    manifest: a neg that differs across SIs, or a missing neg, refuses
    before any GPU work — a per-SI neg would re-import the selection
    confound this mode exists to remove."""
    cfg = _write_cfg(tmp_path)
    pools = _write_pools(tmp_path)
    rows = [json.loads(l) for l
            in (pools / "pairs_bb.jsonl").read_text().splitlines()]
    rows[2]["neg"] = "a different undefended completion"
    (pools / "pairs_bb.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n")
    with pytest.raises(SystemExit, match="NOT common"):
        _run(["--config", str(cfg), "--pairs-dir", str(pools),
              "--fixed-completion", "neg", "--instructions", "json"],
             monkeypatch)

    pools2 = _write_pools(tmp_path)
    for si in ("aa", "bb"):
        rows = [json.loads(l) for l
                in (pools2 / f"pairs_{si}.jsonl").read_text().splitlines()]
        for r in rows:
            r.pop("neg")
        (pools2 / f"pairs_{si}.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rows) + "\n")
    with pytest.raises(SystemExit, match="empty/missing neg"):
        _run(["--config", str(cfg), "--pairs-dir", str(pools2),
              "--fixed-completion", "neg", "--instructions", "json"],
             monkeypatch)


def test_verify_selection_refuses_before_load(tmp_path, monkeypatch,
                                              no_model_load):
    """--verify-selection pins the run to a PRIOR run's identities: any
    divergence (or an unusable prior file) refuses before the load."""
    cfg = _write_cfg(tmp_path)
    pools = _write_pools(tmp_path)
    base = ["--config", str(cfg), "--pairs-dir", str(pools)]

    with pytest.raises(SystemExit, match="missing file"):
        _run(base + ["--verify-selection", str(tmp_path / "nope.json")],
             monkeypatch)

    prior = tmp_path / "prior.json"
    prior.write_text(json.dumps({"results": {}}))
    with pytest.raises(SystemExit, match="no selection.selected"):
        _run(base + ["--verify-selection", str(prior)], monkeypatch)

    prior.write_text(json.dumps({"selection": {"selected": [
        {"family": "alpha", "goal_id": "gX", "variant_key": "v0",
         "prompt_sha8": "00000000"}]}}))
    with pytest.raises(SystemExit, match="DIVERGES"):
        _run(base + ["--verify-selection", str(prior)], monkeypatch)


class _WordTok:
    """Whitespace 'tokenizer': build_chat_prompt falls back to plain
    concatenation when chat_template is absent, so a render's token count
    is its word count — enough to exercise the common-budget arithmetic
    without a model."""
    chat_template = None
    eos_token_id = 99
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [1] * len(text.split())}


def test_plan_common_completion_budget_is_common_across_instructions():
    """The completion budget per row is set by the LONGEST scored
    instruction, so every instruction measures the identical fixed-text
    span; a row with no room is excluded from ALL instructions; the
    truncation count is reported."""
    tok = _WordTok()
    rows = [{"prompt": "one two three",
             "neg": "n1 n2 n3 n4 n5 n6 n7 n8 n9 n10"}]   # 10 words
    rows += [{"prompt": "one two three", "neg": "n1 n2 n3 n4"}
             for _ in range(29)]
    rows.append({"prompt": " ".join(["w"] * 50), "neg": "n1 n2"})
    instr_texts = ["a b", " ".join(["i"] * 20)]
    # budget for the 3-word-prompt rows: 30 - (20 + 3) = 7 completion slots
    comp, excluded, n_trunc = mts.plan_common_completion(
        tok, rows, instr_texts, max_len=30)
    assert excluded == [30] and comp[30] is None
    assert len(comp[0]) == 7            # 10 + eos = 11, clamped to 7
    assert n_trunc == 1
    assert all(len(c) == 5 for c in comp[1:30])   # 4 + eos, untouched

    # too many exclusions is a --max-len problem, refused outright
    with pytest.raises(SystemExit, match="common completion budget"):
        mts.plan_common_completion(tok, [rows[0], rows[30]], instr_texts,
                                   max_len=30)


def test_axis_cell_and_ratios_to_format_references():
    """The 2x2 cell derives from measured tokens + registry category, and
    the ratio block names its weighting and its references."""
    assert mts.axis_cell("safety", 5872) == {"length": "long",
                                             "content": "safety"}
    assert mts.axis_cell("safety", 10) == {"length": "short",
                                           "content": "safety"}
    assert mts.axis_cell("format", 22) == {"length": "short",
                                           "content": "irrelevant"}
    assert mts.axis_cell("control", 6380) == {"length": "long",
                                              "content": "irrelevant"}

    def _res(pw, uf):
        return {"pool_weighted": {"mean_kl": pw},
                "unweighted_over_families": {"mean_kl": uf}}
    results = {"json": _res(2.0, 1.0), "bullets": _res(4.0, 3.0),
               "safety_refusal": _res(6.0, 4.0)}
    r = mts.ratios_to_format_references(results)
    assert r["pool_weighted"]["reference_mean"] == pytest.approx(3.0)
    assert r["pool_weighted"]["ratios"]["safety_refusal"] == \
        pytest.approx(2.0)
    assert r["pool_weighted"]["ratios"]["json"] == pytest.approx(2 / 3)
    assert r["unweighted_over_families"]["reference_mean"] == \
        pytest.approx(2.0)
    # a missing reference yields a stated absence, never a silent number
    partial = mts.ratios_to_format_references({"json": _res(2.0, 1.0)})
    assert "ratios" not in partial["pool_weighted"]
    assert "bullets" in partial["pool_weighted"]["note"]


def test_fixed_completion_end_to_end_tiny(tmp_path, monkeypatch):
    """The whole fixed-common-completion path on the tiny model: paired
    identities across instructions, axis + token-length metadata, the
    ratio block, the reading statement, and a --verify-selection round
    trip — so the GPU run cannot be the first execution of this loop."""
    from conftest import TINY_LLAMA

    pools_dir = _write_pools(tmp_path, n_per_family=3)
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(f"model: {TINY_LLAMA}\ndtype: float32\n"
                   "si_ids:\n  - aa\n  - bb\n")
    out = tmp_path / "common.json"
    _run(["--config", str(cfg), "--pairs-dir", str(pools_dir),
          "--per-family", "2", "--batch-size", "2", "--max-len", "96",
          "--fixed-completion", "neg", "--instructions", "json,bullets",
          "--out", str(out)], monkeypatch)

    payload = json.loads(out.read_text())
    assert payload["mode"] == "fixed_completion_common_text"
    assert payload["fixed_completion"]["field"] == "neg"
    assert payload["partial"] is False and payload["pending"] == []
    assert "necessary, not sufficient" in payload["reading"]
    assert list(payload["results"]) == ["json", "bullets"]
    assert payload["excluded_rows"] == []
    assert payload["n_truncated_common_budget"] == 0

    ident_keys = ("family", "goal_id", "variant_key", "prompt_sha8")
    idents = {nm: [{k: row[k] for k in ident_keys}
                   for row in payload["results"][nm]["per_example"]]
              for nm in ("json", "bullets")}
    assert idents["json"] == idents["bullets"]
    assert idents["json"] == payload["selection"]["selected"]
    # different instructions genuinely measured: same text, different KLs
    assert [r["kl"] for r in payload["results"]["json"]["per_example"]] != \
        [r["kl"] for r in payload["results"]["bullets"]["per_example"]]

    for nm in ("json", "bullets"):
        r = payload["results"][nm]
        assert r["n_examples"] == 4 and r["families_covered"] == 2
        assert all(r["per_family"][f]["n"] == 2 for f in ("alpha", "beta"))
        assert r["instruction_tokens"] > 0
        assert r["axis"] == {"length": "short", "content": "irrelevant"}
    ratios = payload["ratios_to_format_references"]
    pw = {nm: payload["results"][nm]["pool_weighted"]["mean_kl"]
          for nm in ("json", "bullets")}
    ref_mean = (pw["json"] + pw["bullets"]) / 2
    assert ratios["pool_weighted"]["reference_mean"] == \
        pytest.approx(ref_mean)
    assert ratios["pool_weighted"]["ratios"]["json"] == \
        pytest.approx(pw["json"] / ref_mean)

    # --verify-selection round trip: the prior run's own output passes...
    out2 = tmp_path / "common2.json"
    _run(["--config", str(cfg), "--pairs-dir", str(pools_dir),
          "--per-family", "2", "--batch-size", "2", "--max-len", "96",
          "--fixed-completion", "neg", "--instructions", "json",
          "--verify-selection", str(out), "--out", str(out2)], monkeypatch)
    p2 = json.loads(out2.read_text())
    assert p2["selection"]["verified_against"] == str(out)
    assert p2["selection"]["selected"] == payload["selection"]["selected"]
    # ...and a tampered prior refuses
    tampered = json.loads(out.read_text())
    tampered["selection"]["selected"][0]["prompt_sha8"] = "ffffffff"
    bad = tmp_path / "tampered.json"
    bad.write_text(json.dumps(tampered))
    with pytest.raises(SystemExit, match="DIVERGES"):
        _run(["--config", str(cfg), "--pairs-dir", str(pools_dir),
              "--per-family", "2", "--batch-size", "2", "--max-len", "96",
              "--fixed-completion", "neg", "--instructions", "json",
              "--verify-selection", str(bad)], monkeypatch)


def test_fixed_completion_saves_after_every_instruction(tmp_path,
                                                        monkeypatch):
    """The incremental save is the TIMEOUT defence: a run killed mid-axis
    leaves the completed instructions on disk, marked partial with the
    pending ones NAMED — never an empty file."""
    from conftest import TINY_LLAMA

    pools_dir = _write_pools(tmp_path, n_per_family=3)
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(f"model: {TINY_LLAMA}\ndtype: float32\n"
                   "si_ids:\n  - aa\n  - bb\n")
    out = tmp_path / "partial.json"

    calls = {"n": 0}
    real = mts.summarize

    def dies_on_second(*a, **k):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("simulated mid-axis kill")
        return real(*a, **k)

    monkeypatch.setattr(mts, "summarize", dies_on_second)
    with pytest.raises(RuntimeError, match="mid-axis"):
        _run(["--config", str(cfg), "--pairs-dir", str(pools_dir),
              "--per-family", "2", "--batch-size", "2", "--max-len", "96",
              "--fixed-completion", "neg", "--instructions", "json,bullets",
              "--out", str(out)], monkeypatch)

    payload = json.loads(out.read_text())
    assert payload["partial"] is True
    assert payload["pending"] == ["bullets"]
    assert list(payload["results"]) == ["json"]
    assert payload["results"]["json"]["n_examples"] == 4
    # the ratio block states its absence rather than shipping half a ratio
    assert "note" in payload["ratios_to_format_references"]["pool_weighted"]


def test_common_text_sbatch_hygiene():
    """Static hygiene for the sbatch this measurement ships with: the
    cluster hard rules (--export / --mem=0 / dev-in-sbatch / manual
    CUDA_VISIBLE_DEVICES have each caused real outages), the lane
    conventions, and this job's own preconditions — the selection pin and
    the rc-captured-immediately exit (a trailing command that succeeds
    would mask a job dead at model load as COMPLETED)."""
    import re
    txt = (Path(__file__).resolve().parents[1] / "scripts" / "sbatch"
           / "twosi_common_text.sbatch").read_text()
    assert not re.search(r"^#SBATCH --export", txt, re.M)
    assert not re.search(r"^#SBATCH --mem=0\b", txt, re.M)
    assert "CUDA_VISIBLE_DEVICES" not in txt
    assert not re.search(r"^#SBATCH --qos=dev", txt, re.M)
    assert re.search(r"^#SBATCH --job-name=twosi-common-text\b", txt, re.M)
    assert "#SBATCH --qos=high" in txt
    assert "#SBATCH --gres=gpu:1" in txt
    assert "#SBATCH --mem=128G" in txt
    assert "#SBATCH --open-mode=append" in txt
    assert "#SBATCH --no-requeue" in txt
    assert "#SBATCH --signal=B:TERM@120" in txt
    assert ("stage2_common.sh" in txt and "s2_init" in txt
            and "s2_run" in txt)
    assert "NCCL_SOCKET_IFNAME=" not in txt   # the socket interface is the site's (its nccl.conf), never exported
    # the measurement's own preconditions
    assert "--fixed-completion neg" in txt
    assert "--verify-selection" in txt and 'if [ ! -f "$VERIFY" ]' in txt
    for ins in ("json", "bullets", "si_caa", "safety_refusal",
                "si_constitution", "si_irrelevant"):
        assert ins in txt, ins
    assert "si_hhh" not in txt  # the HHH prompt was removed with its SI
    assert re.search(r'--out "\$OUT"\nrc=\$\?', txt)
    assert txt.rstrip().endswith("exit $rc")


def test_seed_knob_is_gone():
    """--seed was accepted and silently ignored; an accepted-but-ignored
    knob is worse than none. Selection is seedless by convention now."""
    import argparse
    with pytest.raises(SystemExit):
        # argparse exits 2 on unknown argument
        ap_argv = ["measure_teacher_signal.py", "--pairs-dir", "x",
                   "--seed", "7"]
        old = sys.argv
        try:
            sys.argv = ap_argv
            mts.main()
        finally:
            sys.argv = old
    # and the parser really has no such destination
    src = Path(mts.__file__).read_text()
    assert "--seed" not in src.split("def main", 1)[1].split(
        "args = ap.parse_args", 1)[0]
