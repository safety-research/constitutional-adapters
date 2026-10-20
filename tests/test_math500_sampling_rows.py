"""math500-thinking-on-full22 — ``configs/sdfbase/math500/sampling_rows.yaml`` (the ONE run-config copy
of the sampling rows) EQUALS contracts' three tables (``THINKING_SAMPLING_BY_MODEL_KEY`` /
``GENERAL_SAMPLING_BY_MODEL_KEY`` / ``BATTERY_SAMPLING_BY_MODEL_KEY``), row for row, BOTH WAYS, with the
provenance on both sides: the yaml must mirror the three contracts tables.

What the lanes already pin, not repeated here: ``tests/test_sdfbase_math500_sbatch.py`` holds the yaml equal to
``cells.py``'s ``SAMPLING_ROWS`` and, one way, each yaml row's numbers to the contracts row;
``tests/test_tinycap_math500_thinking.py`` reads the yaml through the analysis loader, which cross-checks
``temperature / top_p / top_k`` against ``contracts.sampling_for``; ``tests/test_tinycap_render_facts.py`` pins
the tables' literals. This module adds the OTHER direction (no table row without a yaml row), the optional
terms (``min_p`` / ``presence_penalty`` / ``repetition_penalty``: set in the table iff the yaml names them —
a row carries EXACTLY the keys its source names), the mode set against ``SAMPLING_TABLE_BY_MODE``, the
provenance strings on both sides (each battery row cites Scale's file, branch, sha; each primary row cites
its snapshot), and the matrix rows against the yaml.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
for p in (REPO / "src", REPO / "scripts" / "sdfbase" / "math500"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import cells as M

from sisteer.capeval.serving import geometry as sg
from sisteer.capeval.tinycap import contracts as C

YAML = REPO / "configs" / "sdfbase" / "math500" / "sampling_rows.yaml"
#: the numeric members a row may carry — the three the parent hashes plus the experiment's optional terms
NUMERIC = ("temperature", "top_p", "top_k") + tuple(C.SAMPLING_OPTIONAL_ROW_FIELDS)
TABLES = {C.SAMPLING_THINKING_CARD: C.THINKING_SAMPLING_BY_MODEL_KEY,
          C.SAMPLING_GENERAL: C.GENERAL_SAMPLING_BY_MODEL_KEY,
          C.SAMPLING_BATTERY: C.BATTERY_SAMPLING_BY_MODEL_KEY}
#: the pinned snapshot each primary row was read from — cited on BOTH sides
SNAPSHOT_PREFIX = {"gemma4e4b": "ee0ef602", "qwen35_4b": "851bf6e8", "glm46vflash": "411bb4d7",
                   "nemotron3nano4b": "dfaf35de"}
BATTERY_SHA = "7a065e58fb86217033e27a020ceb207027b447f43a500a094cf12c5ae7becb36"


def _yaml() -> tuple[dict, dict[str, dict[str, dict]]]:
    """(the document, ``{mode: {model_key: row}}``) — the run config's ``rows: {model: {mode: row}}`` transposed."""
    doc = yaml.safe_load(YAML.read_text(encoding="utf-8"))
    rows: dict[str, dict[str, dict]] = {}
    for key, by_mode in doc["rows"].items():
        for mode, row in by_mode.items():
            rows.setdefault(mode, {})[key] = row
    return doc, rows


def test_the_yaml_modes_are_the_experiments_three_plus_the_no_switch_card_and_each_names_a_contracts_table():
    doc, rows = _yaml()
    assert doc["modes"] == list(M.SAMPLING_MODES) + [C.SAMPLING_CARD] == [C.SAMPLING_THINKING_CARD, C.SAMPLING_GENERAL, C.SAMPLING_BATTERY, C.SAMPLING_CARD]
    assert set(rows) == set(doc["modes"]) == set(TABLES) | {C.SAMPLING_CARD}
    for mode, table in TABLES.items():
        assert C.SAMPLING_TABLE_BY_MODE[mode] is table
        assert mode in C.SAMPLING_MODES                       # a mode the driver's --sampling accepts
    # `greedy` is NOT the experiment's: no yaml row rides it. The parent's `card` rides ONE row — llama8b's, the GSM8K
    # paired-200 Llama rows' sampling under the no-switch arm (llama-paper-rows P6): pinned ONE WAY against
    # contracts.SAMPLING_BY_MODEL_KEY (that table names every launcher model; the registry lists only the model it serves)
    assert C.SAMPLING_GREEDY not in rows and set(rows[C.SAMPLING_CARD]) == {"llama8b"}
    assert C.SAMPLING_TABLE_BY_MODE[C.SAMPLING_CARD] is C.SAMPLING_BY_MODEL_KEY and "llama8b" in C.SAMPLING_BY_MODEL_KEY
    crow, yrow = C.SAMPLING_BY_MODEL_KEY["llama8b"], rows[C.SAMPLING_CARD]["llama8b"]
    assert (yrow["temperature"], yrow["top_p"]) == (crow.temperature, crow.top_p) == (0.6, 0.9)
    assert set(yrow) == {"temperature", "top_p", "source"} and yrow["source"] == crow.source and "d10aef79" in yrow["source"]
    assert "not available" in yrow["source"] and C.THINKING_NOT_AVAILABLE.replace("_", " ") in yrow["source"]
    # the no-switch family's MATH-500 rows (the llama steering rows and the llama wave) ride it — the ONE mode the driver
    # resolves for llama8b — and no switch family's row does; the four wave models' modes are the three
    assert {r["model_key"] for r in M.M5T_CELLS if r["sampling_mode"] == C.SAMPLING_CARD} == {"llama8b"}
    assert all(r["sampling_mode"] == C.SAMPLING_CARD for r in M.M5T_CELLS if r["model_key"] == "llama8b")
    # THE DRIVER DOOR: every row's (model, mode) resolves a row in the driver's PINNED tables —
    # `contracts.sampling_for` is what `--sampling <mode>` runs — and the llama rows resolve the card row; `thinking_card`
    # on llama8b is the refusal the llama steering rows died of (no llama8b row in THINKING_SAMPLING_BY_MODEL_KEY)
    for r in M.M5T_CELLS:
        got = C.sampling_for(r["model_key"], r["sampling_mode"])
        assert (got.temperature, got.top_p) == (r["sampling"]["temperature"], r["sampling"]["top_p"]), r["job_id"]
    llama_rows = [r for r in M.M5T_CELLS if r["model_key"] == "llama8b"]
    assert len(llama_rows) == 8 and all(C.sampling_for(r["model_key"], r["sampling_mode"]) is C.SAMPLING_BY_MODEL_KEY["llama8b"] for r in llama_rows)
    with pytest.raises(C.SamplingUnverified):
        C.sampling_for("llama8b", C.SAMPLING_THINKING_CARD)


@pytest.mark.parametrize("mode", sorted(TABLES))
def test_the_yaml_rows_of_a_mode_are_exactly_the_contracts_table_rows_both_ways(mode):
    _, rows = _yaml()
    table = TABLES[mode]
    # no extra key on either side -- except the onboarded front-door models (geometry.ONBOARDED_MODEL_KEYS), which carry a
    # contracts thinking_card row for their GSM8K front door and NO MATH-500 row (MATH-500 is not in
    # their battery, so this yaml rightly names none); pinned in tests/test_gsm8k_thinking_card_rows.py
    contracts_keys = set(table) - set(sg.ONBOARDED_MODEL_KEYS)
    assert set(rows[mode]) == contracts_keys, sorted(set(rows[mode]) ^ contracts_keys)
    for key, yrow in rows[mode].items():
        crow = table[key]
        assert C.sampling_for(key, mode) is crow
        for field in NUMERIC:
            assert yrow.get(field) == getattr(crow, field), (mode, key, field, yrow.get(field), getattr(crow, field))
        # the yaml carries EXACTLY the keys the table sets (an absent key = the request carries none)
        assert set(yrow) - {"source"} == {f for f in NUMERIC if getattr(crow, f) is not None}, (mode, key)
        assert isinstance(yrow["temperature"], float) and isinstance(crow.temperature, float)
        if "top_k" in yrow:
            assert isinstance(yrow["top_k"], int) and not isinstance(yrow["top_k"], bool)
        # the hashed identity of the table's row names exactly the optional terms the yaml names
        ident = C.sampling_identity(crow, {C.inspect_task("math"): 1})
        optional_in_identity = {f for f in C.SAMPLING_OPTIONAL_ROW_FIELDS if f in ident}
        assert optional_in_identity == {f for f in C.SAMPLING_OPTIONAL_ROW_FIELDS if f in yrow}, (mode, key)


@pytest.mark.parametrize("mode", sorted(TABLES))
def test_provenance_strings_are_present_on_both_sides_and_cite_the_same_source(mode):
    _, rows = _yaml()
    for key, yrow in rows[mode].items():
        crow = TABLES[mode][key]
        for side, text in (("yaml", yrow.get("source")), ("contracts", crow.source)):
            assert isinstance(text, str) and len(text.strip()) > 40, (mode, key, side)
        if mode == C.SAMPLING_BATTERY:
            for text in (yrow["source"], crow.source):
                assert "configs/sdfbase_sampled_decode.yaml" in text and "sampled-rerun" in text
                assert "5a50e47b" in text and BATTERY_SHA in text
        elif mode == C.SAMPLING_THINKING_CARD:
            for text in (yrow["source"], crow.source):
                assert SNAPSHOT_PREFIX[key] in text
        else:
            assert key == "qwen35_4b"
            for text in (yrow["source"], crow.source):
                assert "general" in text and "presence_penalty 1.5" in text


def test_every_matrix_row_carries_the_yamls_literal_row_for_its_model_and_mode():
    """The matrix (cells.py) is the third copy the jobs read; each of its rows (the 24 published + the eight-arm
    extension's 9, every extension row under its model's PRIMARY row) is the yaml's row verbatim."""
    _, rows = _yaml()
    # + E3's it-doc rows (the instruct-direct re-fit family on the `it` view; every one under its model's PRIMARY row too)
    # + arditi-allpos-cap's own wave (the all-positions Arditi arm on qwen35_4b's `it` panel view, appended last so no
    #   existing index moves). The APPENDED waves are summed from the module's own row lists rather than a literal, so a
    #   later wave cannot make this count silently stale — only the 24 published and the extension's 9 are literals here,
    #   because those two are what this yaml is the third copy OF.
    appended = len(M.STEER_ROWS) + len(M.LLAMA_ROWS) + len(M.ITDOC_ROWS) + len(M.ALLPOS_ROWS)
    assert len(M.M5T_CELLS) == 24 + 9 + appended == M.N_BASE + len(M.EXTENSION_ROWS) + appended
    assert all(r["sampling_mode"] == M.sampling_mode_for(r["model_key"])
               for r in M.EXTENSION_ROWS + M.STEER_ROWS + M.LLAMA_ROWS + M.ITDOC_ROWS + M.ALLPOS_ROWS)
    assert all(r["sampling_mode"] == M.SAMPLING_THINKING_CARD for r in M.EXTENSION_ROWS)
    # the steering rung runs four models, one of which this yaml does not register: llama8b is a STEER-ONLY model
    # of this matrix (no published row, no sensitivity row, no wave of its own here) and its row is the SHIPPED card
    # row mirrored from contracts.SAMPLING_BY_MODEL_KEY — pinned there, by tests/test_sdfbase_math500_sbatch.py
    steer_only = {(k, m) for (k, m) in M.SAMPLING_ROWS if k in M.STEER_ONLY_MODELS}
    for r in M.M5T_CELLS:
        if r["model_key"] in M.STEER_ONLY_MODELS:
            assert r["sampling"] == M.SAMPLING_ROWS[(r["model_key"], r["sampling_mode"])], r["job_id"]
            continue
        assert r["sampling"] == rows[r["sampling_mode"]][r["model_key"]], r["job_id"]
    # The yaml carries the MATH-500 modes' rows exactly, plus the ONE no-switch `card` row the GSM8K Llama rows run.
    # The steering rung adds the same model under THIS matrix's `thinking_card` mode name — its steering rows'
    # sampling row, mirrored from contracts.SAMPLING_BY_MODEL_KEY and pinned there — so both sides are subtracted from
    # the yaml and the matrix respectively, and the remainder is the four wave models' registry, unchanged.
    yaml_extra = {("llama8b", C.SAMPLING_CARD)}
    assert {(k, m) for m in rows for k in rows[m]} - yaml_extra == set(M.SAMPLING_ROWS) - steer_only == {
        (r["model_key"], r["sampling_mode"]) for r in M.M5T_CELLS} - steer_only
