"""Review round 3, P2-d — a BoN row is LEGACY only when it carries NONE of the current identity
fields; any non-empty proper subset is UNVERIFIABLE.

Before: `_validate_row` classified a row as legacy when `decode` and `sampling_seed` were both absent, so a torn
current-format row that kept `seed_rule="bon_seed_v1"` or `bon_decode_preset` while losing those two passed as
historical under a harness-default resume and was counted as covered (a rule-only row came back from
`_load_covered_validated` as {("g0", 0, 0)}). Now the legacy rule reads ALL FOUR identity fields — `decode`,
`bon_decode_preset`, `sampling_seed`, `seed_rule` — and a row carrying some but not all of them is refused before
any manifest write and never counted as covered. Historical stores (none of the four) are still accepted only under
the explicit legacy rule by a harness-default run, never relabelled as the vendor preset.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from sisteer.eval.attacks import bon as bon_mod
from sisteer.eval.attacks import bon_served as bs

REPO = Path(__file__).resolve().parents[1]
for p in (REPO / "scripts", REPO / "tests"):
    sys.path.insert(0, str(p))

PRESET = {"temperature": 0.6, "top_p": 0.9}
IDENTITY = ("decode", "bon_decode_preset", "sampling_seed", "seed_rule")


def _world(tmp_path):
    import test_bon_served as tb
    backend = tb.RecordingBackend()
    proxy, routes = tb._proxy(tmp_path, backend)
    defense = bs.ServedDefense.over_proxy(proxy, virtual_model="petri__base", cell_id="none")
    return tb, backend, routes, defense


def _rows(out):
    return [json.loads(l) for l in (out / "bon_responses.jsonl").read_text().splitlines() if l.strip()]


def _write(out, rows):
    (out / "bon_responses.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


def _keep_only(row, keep):
    r = {k: v for k, v in row.items() if k not in IDENTITY}
    for k in keep:
        r[k] = row[k]
    return r


def test_the_identity_fields_are_the_four_the_row_writer_records():
    assert bs.ROW_IDENTITY_KEYS == IDENTITY
    for r_key in IDENTITY:
        assert f'"{r_key}"' in (REPO / "src/sisteer/eval/attacks/bon_served.py").read_text()


@pytest.mark.parametrize("keep", [("seed_rule",), ("bon_decode_preset",), ("seed_rule", "bon_decode_preset"),
                                  ("decode", "sampling_seed", "seed_rule"), ("decode", "bon_decode_preset", "sampling_seed")],
                         ids=lambda k: "+".join(k))
def test_a_row_with_some_but_not_all_identity_fields_is_unverifiable_never_legacy(tmp_path, keep):
    tb, backend, routes, defense = _world(tmp_path)
    goals = tb._goals(1)
    out = tmp_path / "store"
    kw = {"out_dir": out, "routes": routes, "n_max": 2, "max_concurrency": 2, "decode_label": bs.BON_HARNESS_DEFAULT}
    bs.bon_generate_served(defense, goals, **kw)                       # a harness-default store (the grandfathered row BY NAME): every row carries all four
    rows = _rows(out)
    assert all(all(k in r for k in IDENTITY) for r in rows)
    torn = [dict(r) for r in rows]
    torn[0] = _keep_only(torn[0], keep)
    _write(out, torn)
    manifest_before = (out / bs.BON_SERVED_MANIFEST).read_text()
    jsonl = out / "bon_responses.jsonl"
    # never covered: the validated coverage set refuses the whole store, for the harness default AND the preset
    with pytest.raises(bs.BonServedError, match="UNVERIFIABLE"):
        bs._load_covered_validated(jsonl, None, bs.BON_HARNESS_DEFAULT)
    with pytest.raises(bs.BonServedError, match="UNVERIFIABLE"):
        bs._load_covered_validated(jsonl, PRESET, "vendor")
    # never legacy: the legacy count refuses too (it would otherwise count the row under the explicit rule)
    with pytest.raises(bs.BonServedError, match="UNVERIFIABLE"):
        bs._assert_sampler_consistent(jsonl, None, bs.BON_HARNESS_DEFAULT)
    # and a harness-default resume refuses BEFORE any request or manifest write
    n = backend.n_calls
    with pytest.raises(bs.BonServedError, match="UNVERIFIABLE") as ei:
        bs.bon_generate_served(defense, goals, **kw)
    assert "not all of decode / bon_decode_preset / sampling_seed / seed_rule" in str(ei.value)
    assert backend.n_calls == n
    assert (out / bs.BON_SERVED_MANIFEST).read_text() == manifest_before
    assert _rows(out) == torn


def test_a_row_with_none_of_the_four_is_legacy_and_only_the_harness_default_resumes_it(tmp_path):
    tb, backend, routes, defense = _world(tmp_path)
    goals = tb._goals(1)
    out = tmp_path / "legacy"
    kw = {"out_dir": out, "routes": routes, "n_max": 2, "max_concurrency": 2, "decode_label": bs.BON_HARNESS_DEFAULT}
    bs.bon_generate_served(defense, goals, **kw)
    rows = [_keep_only(r, ()) for r in _rows(out)]
    _write(out, rows)
    jsonl = out / "bon_responses.jsonl"
    assert bs._load_covered_validated(jsonl, None, bs.BON_HARNESS_DEFAULT) == {("g0", 0, 0), ("g0", 1, 0)}
    assert bs._assert_sampler_consistent(jsonl, None, bs.BON_HARNESS_DEFAULT) == 2
    # the vendor-preset DEFAULT (2026-09-17) is another sampler: it may not resume a legacy store either
    with pytest.raises(bs.BonServedError, match="LEGACY"):
        bs._load_covered_validated(jsonl, None, bs.BON_VENDOR_PRESET, "gemma4e4b")
    with pytest.raises(bs.BonServedError, match="LEGACY"):           # never relabelled as the vendor preset
        bs._load_covered_validated(jsonl, PRESET, "vendor")
    n = backend.n_calls
    bs.bon_generate_served(defense, goals, **kw)
    assert backend.n_calls == n
    man = json.loads((out / bs.BON_SERVED_MANIFEST).read_text())
    assert man["n_legacy_seedless_rows"] == 2 and man["legacy_rows_rule"] == bon_mod.LEGACY_SEEDLESS_RULE
