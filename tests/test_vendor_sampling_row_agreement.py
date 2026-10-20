"""Two statements of one vendor sampling row must agree.

The campaign states the Qwen3.5 vendor sampling row TWICE:

1. ``src/sisteer/capeval/tinycap/contracts.py`` —
   ``SERVER_DEFAULT_SAMPLING_BY_MODEL[<hf model id>]`` ->
   ``QWEN35_VENDOR_ROWS[<row name>]``, the server-side default the
   omit-params legs decode under (``--override-generation-config``).
2. ``configs/sdfbase_sampled_decode.yaml`` — ``presets.<key>.preset``, the
   row the sampled lane puts on every request body.

They agree today, and nothing but this module makes a drift between the two
copies a failure. Two lanes would then quietly decode differently, and no
summary statistic would show it.

The two statements are joined on the HF model id (the contracts table's key;
the yaml's ``presets.<key>.model``), so EVERY model stated in both places is
checked, not only the one that prompted this. A model stated in one place
only is not a drift (gemma has a yaml preset and no server default;
Qwen3.5-9B has a server default and no yaml preset). The two files sharing NO
model, though, would mean the check guards nothing, and that FAILS rather
than passing vacuously: the join key has drifted, which is itself the kind of
silent divergence this test exists to catch.

Since the results registry's decode-policy redirect the FIRST
statement is no longer a literal in contracts.py: ``SERVER_DEFAULT_SAMPLING_BY_MODEL``
is BUILT from the committed export ``configs/registry/decode_policy.json`` (the
one authority), so the live check reads the IMPORTED table (:func:`live_contracts_rows`)
— the numbers it carries are the registry's — and compares them to the yaml.
The AST reader below (``contracts_rows``) is kept for the synthetic tests, which
prove the CHECK has teeth on a literal table: it resolves the table -> row-table
indirection literally and takes only the numeric keyword LITERALS of the
``SamplingRow(...)`` call. A value that is not a plain numeric literal — a name,
an arithmetic expression, a call, a bool — REFUSES: this reader never turns an
expression into a sampling parameter by evaluating it. ``None`` reads as "no
such key" (``SamplingRow``'s own semantics: the request simply carries no such
key), and ``source`` is provenance prose. Ported from an earlier vendor-row
reader that read the same row the same way.

When the yaml is not in the tree the live check SKIPS and says so: a
standalone or partial checkout is not a drift. The synthetic-file tests below
run everywhere and are what prove the check has teeth — a changed number in
either copy alone, or a key one copy carries and the other does not, is a
named failure.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Dict, List, Mapping

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
CONTRACTS = REPO / "src/sisteer/capeval/tinycap/contracts.py"
SAMPLED_DECODE_YAML = REPO / "configs/sdfbase_sampled_decode.yaml"

#: the contracts table joined on, and the call whose keyword literals ARE a row
TABLE_NAME = "SERVER_DEFAULT_SAMPLING_BY_MODEL"
ROW_CALL = "SamplingRow"
#: ``SamplingRow`` fields that are provenance prose, never a number (the class
#: docstring: ``source`` is never hashed; a wording change must not re-key a store)
PROSE_FIELDS = frozenset({"source"})


class VendorRowError(RuntimeError):
    """A row could not be read AUTHORITATIVELY from one of the two statements."""


# ------------------------------------------------------------ contracts.py, by AST
def _assignments(tree: ast.Module) -> Dict[str, ast.expr]:
    """Top-level ``NAME = <expr>`` / ``NAME: T = <expr>`` bindings."""
    out: Dict[str, ast.expr] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    out[t.id] = node.value
        elif (isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
              and node.value is not None):
            out[node.target.id] = node.value
    return out


def _literal_dict(node: ast.expr, what: str) -> Dict[str, ast.expr]:
    """``{<str literal>: <expr>}`` of a dict-literal node. Refuses anything
    that cannot be read entry by entry: a comprehension, a ``dict(...)`` call,
    a non-string key, a ``**splat``, a key stated twice."""
    if not isinstance(node, ast.Dict):
        raise VendorRowError(f"{what} is not a dict literal ({type(node).__name__})")
    out: Dict[str, ast.expr] = {}
    for k, v in zip(node.keys, node.values):
        if k is None:
            raise VendorRowError(f"{what} carries a **splat; its entries cannot be read literally")
        if not (isinstance(k, ast.Constant) and isinstance(k.value, str)):
            raise VendorRowError(f"{what} has a non-string-literal key ({ast.dump(k)[:80]})")
        if k.value in out:
            raise VendorRowError(f"{what} states {k.value!r} twice")
        out[k.value] = v
    return out


def _row_kwargs(call: ast.expr, what: str) -> Dict[str, Any]:
    """The numeric keyword literals of a ``SamplingRow(...)`` call, as the
    request-body dict they become. ``None`` -> the key is absent. A value that
    is not a plain numeric literal REFUSES: this reader never evaluates an
    expression into a sampling parameter."""
    if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
            and call.func.id == ROW_CALL):
        raise VendorRowError(f"{what} is not a {ROW_CALL}(...) call ({ast.dump(call)[:120]})")
    if call.args:
        raise VendorRowError(
            f"{what} passes positional arguments; which field each one is cannot be read literally")
    row: Dict[str, Any] = {}
    for kw in call.keywords:
        if kw.arg is None:
            raise VendorRowError(f"{what} carries a **splat; its fields cannot be read literally")
        if kw.arg in PROSE_FIELDS:
            continue
        v = kw.value
        if isinstance(v, ast.Constant) and v.value is None:
            continue  # SamplingRow's None: the request carries no such key
        negate = isinstance(v, ast.UnaryOp) and isinstance(v.op, ast.USub)
        lit = v.operand if negate else v
        if not (isinstance(lit, ast.Constant) and isinstance(lit.value, (int, float))
                and not isinstance(lit.value, bool)):
            raise VendorRowError(
                f"{what}: {kw.arg} is not a numeric literal ({ast.dump(v)[:80]}); refusing to "
                "evaluate an expression into a sampling parameter")
        row[kw.arg] = -lit.value if negate else lit.value
    if "temperature" not in row:
        raise VendorRowError(
            f"{what} states no temperature; a request always carries one, so a row "
            "without it would sample at the caller's default")
    return row


def contracts_rows(src: str, origin: str = "contracts.py") -> Dict[str, Dict[str, Any]]:
    """``{<hf model id>: <row dict>}`` for every entry of :data:`TABLE_NAME`,
    resolving a ``ROWS["key"]`` indirection literally or taking a
    ``SamplingRow(...)`` call outright."""
    tree = ast.parse(src, filename=origin)
    binds = _assignments(tree)
    if TABLE_NAME not in binds:
        raise VendorRowError(f"{origin} defines no {TABLE_NAME}")
    table = _literal_dict(binds[TABLE_NAME], f"{TABLE_NAME} in {origin}")
    out: Dict[str, Dict[str, Any]] = {}
    for model_id, entry in table.items():
        what = f"{TABLE_NAME}[{model_id!r}]"
        if isinstance(entry, ast.Subscript):
            if not isinstance(entry.value, ast.Name):
                raise VendorRowError(f"{what} indexes a non-name ({type(entry.value).__name__})")
            sl = entry.slice
            if not (isinstance(sl, ast.Constant) and isinstance(sl.value, str)):
                raise VendorRowError(f"{what} is not indexed by a string literal")
            symbol, key = entry.value.id, sl.value
            if symbol not in binds:
                raise VendorRowError(f"{origin} defines no {symbol} (named by {what})")
            rows = _literal_dict(binds[symbol], f"{symbol} in {origin}")
            if key not in rows:
                raise VendorRowError(
                    f"{symbol} has no entry {key!r} (named by {what}); it has {sorted(rows)}")
            entry, what = rows[key], f"{what} -> {symbol}[{key!r}]"
        out[model_id] = _row_kwargs(entry, what)
    return out


# ------------------------------------------------------------ the yaml
def yaml_rows(doc: Any, origin: str = "sdfbase_sampled_decode.yaml") -> Dict[str, Dict[str, Any]]:
    """``{<hf model id>: <preset dict>}`` for every ``presets.<key>`` of the
    sampled-decode data file. Every entry must name its ``model`` (the join
    key) and a ``preset`` of plain numbers; two keys naming one model make the
    join ambiguous, and refuse — unless their rows are IDENTICAL in every field
    the preset join compares (:func:`_twin_identity`), in which case they state
    ONE row under the first key, as ``sampled_decode.preset_key_for_model``
    collapses them (b19cb3b9f: the worked example qwen35_4b_demo carries the
    paper's qwen35_4b row verbatim for the same checkpoint)."""
    if not isinstance(doc, Mapping) or not isinstance(doc.get("presets"), Mapping):
        raise VendorRowError(f"{origin} has no `presets` mapping")
    out: Dict[str, Dict[str, Any]] = {}
    key_of_model: Dict[str, str] = {}
    for key, entry in doc["presets"].items():
        what = f"presets.{key} in {origin}"
        if not isinstance(entry, Mapping):
            raise VendorRowError(f"{what} is not a mapping")
        model, preset = entry.get("model"), entry.get("preset")
        if not isinstance(model, str) or not model:
            raise VendorRowError(f"{what} names no `model`, the join key")
        if not isinstance(preset, Mapping) or not preset:
            raise VendorRowError(f"{what} states no `preset`")
        for k, v in preset.items():
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                raise VendorRowError(f"{what}: preset.{k} is not a plain number ({v!r})")
        if model in key_of_model:
            # an IDENTICAL twin collapses to the first key, as the preset join does: the two keys state one row, so the
            # join on the model is not ambiguous; a twin that differs in anything the join compares is the ambiguity this
            # refusal is for (the `source` prose is not compared — the demo row's says it is the 4B's, verbatim)
            if _twin_identity(doc["presets"][key_of_model[model]]) == _twin_identity(entry):
                continue
            raise VendorRowError(
                f"{origin} states {model!r} under both presets.{key_of_model[model]} "
                f"and presets.{key} with DIFFERENT rows; the join is ambiguous")
        key_of_model[model] = key
        out[model] = dict(preset)
    return out


def _twin_identity(entry: Mapping[str, Any]) -> tuple:
    """The fields the preset join compares before collapsing two keys that name one model — exactly
    ``sampled_decode.preset_key_for_model``'s: preset, source_kind, revision, shipped_generation_config."""
    return (entry.get("preset"), entry.get("source_kind"), entry.get("revision"), entry.get("shipped_generation_config"))


# ------------------------------------------------------------ the check
def agreement_failures(contracts: Mapping[str, Mapping[str, Any]],
                       presets: Mapping[str, Mapping[str, Any]]) -> List[str]:
    """One line per model stated in BOTH sources whose two rows differ: a
    changed number, or a key one copy carries and the other does not.
    Equality is numeric (``20`` and ``20.0`` are one number)."""
    out: List[str] = []
    for model in sorted(set(contracts) & set(presets)):
        a, b = dict(contracts[model]), dict(presets[model])
        if a != b:
            diff = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
            out.append(f"{model}: contracts.py {TABLE_NAME} says {a}; "
                       f"sdfbase_sampled_decode.yaml says {b}; differing keys {diff}")
    return out


def check_agreement(contracts: Mapping[str, Mapping[str, Any]],
                    presets: Mapping[str, Mapping[str, Any]]) -> List[str]:
    """Assert the two statements agree on every shared model; return the
    shared model ids. The ONE code path the live test and the synthetic tests
    both go through, so the teeth proven on synthetic files are the teeth the
    real files meet."""
    shared = sorted(set(contracts) & set(presets))
    assert shared, (
        f"{TABLE_NAME} names {sorted(contracts)} and the yaml names {sorted(presets)}: no "
        "model is stated in both, so this check would guard nothing. The join key (the HF "
        "model id) has drifted, which is itself the silent divergence this test exists to catch")
    failures = agreement_failures(contracts, presets)
    assert not failures, (
        "the campaign states a vendor sampling row twice, in contracts.py and in "
        "sdfbase_sampled_decode.yaml, and the copies have DRIFTED. Two lanes now decode "
        "differently with no summary statistic showing it:\n  " + "\n  ".join(failures))
    return shared


# ============================================================ live: this tree's two files
DECODE_POLICY_EXPORT = REPO / "configs/registry/decode_policy.json"


def live_contracts_rows() -> Dict[str, Dict[str, Any]]:
    """``{<hf model id>: <row dict>}`` from the IMPORTED contracts table — whose numbers are the registry export's
    (``sisteer.registry.decode_policy``) — in the reader's shape: numeric fields only, ``None`` = no such key."""
    from sisteer.capeval.tinycap import contracts as C
    out: Dict[str, Dict[str, Any]] = {}
    for model, row in C.SERVER_DEFAULT_SAMPLING_BY_MODEL.items():
        out[model] = {k: v for k, v in row._asdict().items() if k not in PROSE_FIELDS and v is not None}
    return out


def test_the_contracts_table_is_the_registry_export_on_this_tree():
    """The first statement IS the registry's: every model the contracts table names is a server-default route row of
    the committed export (`<model>|omit_params|off|server_default`), and the numbers the table carries for the
    server-defaultable keys are that row's. A table built from anything else fails HERE."""
    import json
    rows = live_contracts_rows()
    assert rows, f"{TABLE_NAME} is empty"
    export = json.loads(DECODE_POLICY_EXPORT.read_text())
    route = {}
    for p in export["policies"]:
        if p.get("eval_category") == "omit_params" and p.get("mode") == "server_default" and not p.get("superseded_by_policy_id"):
            route[export["models"][p["model_id"]]["hf_repo"]] = p
    assert set(rows) == set(route), (sorted(rows), sorted(route))
    for model, row in rows.items():
        assert isinstance(row["temperature"], (int, float)), (model, row)
        for k in ("temperature", "top_p", "top_k", "min_p", "repetition_penalty"):
            assert row.get(k) == route[model].get(k), (model, k, row.get(k), route[model].get(k))


def test_every_model_stated_twice_states_one_row():
    if not SAMPLED_DECODE_YAML.is_file():
        pytest.skip(
            f"{SAMPLED_DECODE_YAML.relative_to(REPO)} is not in this tree: the sampled lane's "
            "data file has not landed here, so there is no second statement to compare. "
            "This check arms itself the moment it does; the synthetic tests in this module "
            "are what prove it bites.")
    contracts = live_contracts_rows()
    presets = yaml_rows(yaml.safe_load(SAMPLED_DECODE_YAML.read_text()),
                        str(SAMPLED_DECODE_YAML.relative_to(REPO)))
    shared = check_agreement(contracts, presets)
    assert shared  # check_agreement already refused an empty join; recorded for the reader
    # the live file's ONE twin pair: the worked example qwen35_4b_demo restates the paper's qwen35_4b row for the same
    # checkpoint, and the preset join collapses the pair to the paper's key — pinned so a twin that drifts (refused above
    # as ambiguous) or a second pair nobody declared is a named failure here
    doc = yaml.safe_load(SAMPLED_DECODE_YAML.read_text())
    keys_of_model: Dict[str, List[str]] = {}
    for key, entry in doc["presets"].items():
        keys_of_model.setdefault(entry["model"], []).append(key)
    assert {m: ks for m, ks in keys_of_model.items() if len(ks) > 1} == {"Qwen/Qwen3.5-4B": ["qwen35_4b", "qwen35_4b_demo"]}
    assert _twin_identity(doc["presets"]["qwen35_4b"]) == _twin_identity(doc["presets"]["qwen35_4b_demo"])
    from sisteer.eval import sampled_decode as sdd
    assert sdd.preset_key_for_model(doc, "Qwen/Qwen3.5-4B", doc["presets"]["qwen35_4b"]["revision"]) == "qwen35_4b"


# ============================================================ synthetic: the check has teeth
_SYNTH_CONTRACTS = '''\
class SamplingRow: pass
_SRC = "where the numbers came from"
ROWS = {
    "general": SamplingRow(temperature=0.7, top_p=0.8, top_k=20, min_p=0.0,
                           presence_penalty=1.5, repetition_penalty=1.0,
                           source=f"instruct general: {_SRC}"),
    "plain": SamplingRow(temperature=1.0, top_p=None, top_k=None, source="plain"),
}
SERVER_DEFAULT_SAMPLING_BY_MODEL = {
    "vendor/model-a": ROWS["general"],
    "vendor/model-b": SamplingRow(temperature=0.6, top_p=0.9, source="a direct call"),
    "vendor/model-c": ROWS["plain"],
}
'''
_ROW_A = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0,
          "presence_penalty": 1.5, "repetition_penalty": 1.0}


def _synth_yaml() -> Dict[str, Any]:
    return {"presets": {
        "a": {"model": "vendor/model-a", "preset": dict(_ROW_A)},
        "b": {"model": "vendor/model-b", "preset": {"temperature": 0.6, "top_p": 0.9}},
        "only_here": {"model": "vendor/model-z", "preset": {"temperature": 1.0}},
    }}


def test_the_reader_follows_the_indirection_takes_a_direct_call_and_reads_none_as_no_key():
    rows = contracts_rows(_SYNTH_CONTRACTS)
    assert rows == {
        "vendor/model-a": _ROW_A,
        "vendor/model-b": {"temperature": 0.6, "top_p": 0.9},
        "vendor/model-c": {"temperature": 1.0},
    }
    assert "source" not in rows["vendor/model-a"], "prose is not a number"


def test_agreeing_copies_pass_and_a_model_stated_once_is_not_a_drift():
    shared = check_agreement(contracts_rows(_SYNTH_CONTRACTS), yaml_rows(_synth_yaml()))
    assert shared == ["vendor/model-a", "vendor/model-b"], (
        "model-c (contracts only) and model-z (yaml only) are stated once; not a drift")


@pytest.mark.parametrize("edit_yaml, expect", [
    # one number changed in the yaml alone
    (lambda d: d["presets"]["a"]["preset"].__setitem__("top_p", 0.9), "'top_p'"),
    # a key the yaml drops while contracts still states it
    (lambda d: d["presets"]["a"]["preset"].pop("presence_penalty"), "'presence_penalty'"),
    # a key the yaml adds that contracts does not state
    (lambda d: d["presets"]["b"]["preset"].__setitem__("top_k", 20), "'top_k'"),
])
def test_a_change_in_the_yaml_copy_alone_is_a_named_failure(edit_yaml, expect):
    doc = _synth_yaml()
    edit_yaml(doc)
    failures = agreement_failures(contracts_rows(_SYNTH_CONTRACTS), yaml_rows(doc))
    assert len(failures) == 1 and expect in failures[0] and "differing keys" in failures[0], failures
    with pytest.raises(AssertionError, match="DRIFTED"):
        check_agreement(contracts_rows(_SYNTH_CONTRACTS), yaml_rows(doc))


@pytest.mark.parametrize("before, after, model", [
    ("temperature=0.7", "temperature=0.8", "vendor/model-a"),      # the row table's number
    ("presence_penalty=1.5", "presence_penalty=0.0", "vendor/model-a"),
    ("top_p=0.9, source", "top_p=0.95, source", "vendor/model-b"),  # the direct call's number
    ("min_p=0.0,\n", "\n", "vendor/model-a"),                       # contracts drops a key
])
def test_a_change_in_the_contracts_copy_alone_is_a_named_failure(before, after, model):
    assert _SYNTH_CONTRACTS.count(before) == 1, before
    src = _SYNTH_CONTRACTS.replace(before, after)
    failures = agreement_failures(contracts_rows(src), yaml_rows(_synth_yaml()))
    assert len(failures) == 1 and failures[0].startswith(model + ":"), failures
    with pytest.raises(AssertionError, match="DRIFTED"):
        check_agreement(contracts_rows(src), yaml_rows(_synth_yaml()))


def test_an_identical_twin_key_collapses_to_the_first_key_and_a_differing_twin_refuses():
    """The worked example's shape: a second key restating one model's row VERBATIM is one row under the first key (the
    preset join's rule, sampled_decode.preset_key_for_model); a twin that differs in any field the join compares refuses."""
    doc = _synth_yaml()
    doc["presets"]["a_twin"] = {"model": "vendor/model-a", "preset": dict(_ROW_A)}
    assert yaml_rows(doc) == yaml_rows(_synth_yaml())
    assert check_agreement(contracts_rows(_SYNTH_CONTRACTS), yaml_rows(doc)) == ["vendor/model-a", "vendor/model-b"]
    # revision / source_kind / shipped block stated on both sides and equal, the `source` prose differing: still one row
    doc = _synth_yaml()
    doc["presets"]["a"].update(revision="r1", source_kind="model_card", shipped_generation_config={"temperature": 0.7},
                               source="read off the card")
    doc["presets"]["a_twin"] = {**doc["presets"]["a"], "preset": dict(_ROW_A), "source": "the a row, verbatim"}
    assert yaml_rows(doc) == yaml_rows(_synth_yaml())
    # ... and a twin differing in any ONE compared field is the ambiguity the refusal is for
    for edit in (lambda t: t["preset"].__setitem__("top_p", 0.9),                      # a number of the preset
                 lambda t: t["preset"].pop("min_p"),                                    # a key of the preset
                 lambda t: t.__setitem__("revision", "r2"),
                 lambda t: t.__setitem__("source_kind", "generation_config"),
                 lambda t: t.__setitem__("shipped_generation_config", {"temperature": 1.0})):
        bad = _synth_yaml()
        bad["presets"]["a"].update(revision="r1", source_kind="model_card", shipped_generation_config={"temperature": 0.7})
        bad["presets"]["a_twin"] = {**bad["presets"]["a"], "preset": dict(_ROW_A)}
        edit(bad["presets"]["a_twin"])
        with pytest.raises(VendorRowError, match="with DIFFERENT rows; the join is ambiguous"):
            yaml_rows(bad)


def test_a_join_that_shares_no_model_fails_rather_than_passing_vacuously():
    doc = _synth_yaml()
    for entry in doc["presets"].values():
        entry["model"] = entry["model"].replace("vendor/", "other-vendor/")
    with pytest.raises(AssertionError, match="guard nothing"):
        check_agreement(contracts_rows(_SYNTH_CONTRACTS), yaml_rows(doc))


@pytest.mark.parametrize("value", [
    "SOME_CONST",          # a name
    "0.35 * 2",            # arithmetic
    "float('0.7')",        # a call
    "True",                # a bool where a number belongs: the classic silent coercion
    "-SOME_CONST",         # a negated name
])
def test_the_reader_refuses_a_value_that_is_not_a_numeric_literal(value):
    src = ("class SamplingRow: pass\n"
           f"ROWS = {{'r': SamplingRow(temperature=0.7, top_p={value})}}\n"
           "SERVER_DEFAULT_SAMPLING_BY_MODEL = {'m': ROWS['r']}\n")
    with pytest.raises(VendorRowError, match="not a numeric literal"):
        contracts_rows(src)


def test_the_reader_reads_a_negative_literal_and_refuses_a_row_without_temperature():
    src = ("class SamplingRow: pass\n"
           "SERVER_DEFAULT_SAMPLING_BY_MODEL = {'m': SamplingRow(temperature=0.7, top_k=-1)}\n")
    assert contracts_rows(src) == {"m": {"temperature": 0.7, "top_k": -1}}
    src = ("class SamplingRow: pass\n"
           "SERVER_DEFAULT_SAMPLING_BY_MODEL = {'m': SamplingRow(top_p=0.8, source='x')}\n")
    with pytest.raises(VendorRowError, match="states no temperature"):
        contracts_rows(src)


@pytest.mark.parametrize("src, match", [
    ("SERVER_DEFAULT_SAMPLING_BY_MODEL = dict(m=SamplingRow(temperature=0.7))\n",
     "not a dict literal"),
    ("SERVER_DEFAULT_SAMPLING_BY_MODEL = {**COMMON}\n", r"\*\*splat"),
    ("SERVER_DEFAULT_SAMPLING_BY_MODEL = {'m': SamplingRow(0.7, 0.8)}\n", "positional"),
    ("SERVER_DEFAULT_SAMPLING_BY_MODEL = {'m': SamplingRow(temperature=0.7, **COMMON)}\n",
     r"\*\*splat"),
    ("SERVER_DEFAULT_SAMPLING_BY_MODEL = {'m': OtherRow(temperature=0.7)}\n",
     "not a SamplingRow"),
    ("SERVER_DEFAULT_SAMPLING_BY_MODEL = {'m': ROWS[KEY]}\n", "string literal"),
    ("SERVER_DEFAULT_SAMPLING_BY_MODEL = {'m': ROWS['absent']}\nROWS = {'r': SamplingRow(temperature=1.0)}\n",
     "no entry 'absent'"),
    ("OTHER_TABLE = {}\n", f"defines no {TABLE_NAME}"),
])
def test_the_reader_refuses_a_table_it_cannot_read_literally(src, match):
    with pytest.raises(VendorRowError, match=match):
        contracts_rows("class SamplingRow: pass\n" + src)


@pytest.mark.parametrize("edit, match", [
    (lambda d: d["presets"]["a"].pop("model"), "names no `model`"),
    (lambda d: d["presets"]["a"].pop("preset"), "states no `preset`"),
    (lambda d: d["presets"]["a"]["preset"].__setitem__("top_p", "0.8"), "not a plain number"),
    (lambda d: d["presets"]["a"]["preset"].__setitem__("top_p", True), "not a plain number"),
    (lambda d: d["presets"]["b"].__setitem__("model", "vendor/model-a"), "ambiguous"),
    (lambda d: d.pop("presets"), "no `presets` mapping"),
])
def test_the_yaml_reader_refuses_an_entry_it_cannot_join(edit, match):
    doc = _synth_yaml()
    edit(doc)
    with pytest.raises(VendorRowError, match=match):
        yaml_rows(doc)
