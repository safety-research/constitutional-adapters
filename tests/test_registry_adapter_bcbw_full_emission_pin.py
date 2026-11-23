"""The two FULL-suite bcbw readers (`bigcodebench_full`, `bfcl_full`) and the six SHARED-standup comparison readers (`*_full_shared`, `_l2`, `_l3`) emit BYTE FOR
BYTE what they emitted when these pins were taken (2026-11-15, on origin/main b184fddd3, before the shared reader learned to read a declared campaign): every record of
every type over the synthetic mirrors of `_bcbw_full_fixtures` and `_bcbw_shared_fixtures` hashes to the value pinned here. `test_registry_adapter_bcbw_emission_pin.py`
pins the four 200-item readers the same way; this file extends the pin to the other eight subclasses of the shared reader, so a change to `_bcbw_common` that is
meant to leave the paper's readers alone is seen to (the declared-campaign generalisation: `TaskSpec.declared` is None on all eight).

Two full mirrors per reader carry every optional shape the full readers read, each knob ON: `shapes` -- the instruct-direct row, a three-seed-family store
(`qwen35_4b_s2`), the qwen35_35b_a3b family's unsteered row with its window re-serve and the dose-0.75 random-init ReLU, dose-1.25 difference-LoRA and
LeCun-init ReLU rows, and an undeclared row dir (a whole-group refusal by name); `relaunch` -- a relaunched seed-tag row (`-r2`, the parked first store) and the
instruct-direct row relaunched beside its parked attempt's FAIL record (the fixture's second pins copy carries no 35B block, so the two cannot share a mirror). The shared mirrors are landing 1 with the repeat
store and the instruct row's own none, landing 2 verified, and landing 3 in its published shapes.

The pin is computed exactly as the 200-item readers' pin: the records with their paths normalised (the fixture roots and the test's temp directory, absolute or
`~`-relative) and WITHOUT `record_id` (it hashes the payload, so it moves with the temp path), plus every group's `status_extra` and the `source_state` rows, the
reads on a frame primed from the adapter's own emissions (the steady state after the fold). A legitimate emission change to one of these readers moves its hash
AND its EMISSION_VERSION together: re-pin here in the same commit, with the reason in the reread ledger. A move of the synthetic CONTEXT alone (the envelope's
producer block, a fixture knob) re-pins here with the reason beside the pins and bumps nothing."""
from __future__ import annotations

import hashlib
import json

import pytest

import _bcbw_full_fixtures as Z
import _bcbw_seedrep_fixtures as X
import _bcbw_shared_fixtures as SH
from sisteer.registry.adapters import base as AB
from sisteer.registry.adapters import bfcl_full as BFF
from sisteer.registry.adapters import bigcodebench_full as BCF
from sisteer.registry.adapters import lineage_common as LC

#: (canonical sha256, n_records, n_states) per reader
PINNED = {
    "bfcl_full:relaunch": ("c62f5ca639ad40d58e75de82e8fc65a2517973da464f125a6acc96387896f43a", 153, 8),
    "bfcl_full:shapes": ("17e7e6e2adf9224fc9e6b88253ebec87385e88675766793066fe150aa9dbf5c8", 439, 12),
    "bfcl_full_shared": ("9024e733ef7e4dba80772ce3705ebad412b34ffb76dd70e6dd9f82058b724fd1", 113, 5),
    "bfcl_full_shared_l2": ("c1af9511bc8c4be6b6f02cfd0aae47b2b6e8080aba1fd3449e177db43fc9db62", 92, 5),
    "bfcl_full_shared_l3": ("16ffa8fbf9e8e44d125a032a38d4c9a26e4d0ba948fef7aaaa6af0b4d875ccc2", 92, 5),
    "bigcodebench_full:relaunch": ("123483e9913715ba292bb1502c40de1e94f69f0baa8797b86cbfc2dd3d87b94a", 153, 8),
    "bigcodebench_full:shapes": ("6dfdb080a2f083419a8c8d71c8fb8ac9cd0d6d2cac10ebf0d14cb9d210e3cb3b", 439, 12),
    "bigcodebench_full_shared": ("e0aff4b95a2f9ebb32656ccaabef60cff1eddd2ca719805cac0b82a0ae762a4a", 113, 5),
    "bigcodebench_full_shared_l2": ("5a11bffa470f7628a92d2fd333b05d825315bed9a6b527efc72dc56623968493", 92, 5),
    "bigcodebench_full_shared_l3": ("db20c868053fa2d45678ab6d4aef2c88502e5480f619a3370b15fa5f05f959d7", 92, 5),
}
#: the full mirrors' knobs: every optional shape the full readers read (module docstring), in two mirrors because the relaunch's second pins copy is
#: written for the gemma / qwen / llama blocks alone (the fixture's relaunch predates the 35B family's pins block)
FULL_KNOBS = {"shapes": dict(instruct=True, qwen_store=True, qwen35b_u0=True, qwen35b_d0p75_relul=True, qwen35b_d1p25=True, qwen35b_s2rl=True,
                             undeclared_row_dirs=(Z.UNDECLARED_ROW_DIR,)),
              "relaunch": dict(relaunched=(Z.RELU,), instruct=True, instruct_relaunched=True, instruct_parked_gate=True, qwen_store=True)}
#: per shared landing: the fixture knobs (the published shapes)
SHARED_KNOBS = {1: dict(repeat=True, it_none=True), 2: dict(landing=2, verified=True), 3: dict(landing=3, **SH.LANDING3_SHAPE)}
#: reader -> (kind, eval, landing)
READERS = {"bigcodebench_full:shapes": ("full", "bigcodebench", "shapes"), "bigcodebench_full:relaunch": ("full", "bigcodebench", "relaunch"),
           "bfcl_full:shapes": ("full", "bfcl", "shapes"), "bfcl_full:relaunch": ("full", "bfcl", "relaunch"),
           "bigcodebench_full_shared": ("shared", "bigcodebench", 1), "bfcl_full_shared": ("shared", "bfcl", 1),
           "bigcodebench_full_shared_l2": ("shared", "bigcodebench", 2), "bfcl_full_shared_l2": ("shared", "bfcl", 2),
           "bigcodebench_full_shared_l3": ("shared", "bigcodebench", 3), "bfcl_full_shared_l3": ("shared", "bfcl", 3)}
FULL_ADAPTERS = {"bigcodebench": BCF.BigcodebenchFullAdapter, "bfcl": BFF.BfclFullAdapter}


@pytest.fixture(scope="module")
def seeds():
    return LC.Seeds.locate()


def _normalise(txt: str, pairs) -> str:
    """Every path in ``pairs`` (longest first), in its absolute and its `~`-relative spelling, replaced by its placeholder."""
    subs = []
    for path, ph in pairs:
        subs += [(str(path), ph), (LC.rel_home(str(path)), ph)]
    for path, ph in sorted(set(subs), key=lambda s: len(s[0]), reverse=True):
        txt = txt.replace(path, ph)
    return txt


def emission(name: str, tmp_path, seeds):
    """(canonical sha256, n_records, n_states) of one reader's emission over its fixture; paths normalised, record ids dropped."""
    kind, eval_id, landing = READERS[name]
    d = tmp_path / name.replace(":", "_")
    d.mkdir()
    if kind == "full":
        fx = Z.mirror(d, seeds, eval_id=eval_id, root=str(d / "mirror"), **FULL_KNOBS[landing])
        ad = Z.adapter(FULL_ADAPTERS[eval_id], fx, seeds)
        roots = [(fx["root"], "<ROOT>")]
    else:
        fx = SH.mirror(d, seeds, eval_id=eval_id, **SHARED_KNOBS[landing])
        ad = SH.adapter(eval_id, fx, seeds)
        roots = [(fx["root"], "<ROOT>"), (fx["solo_root"], "<SOLO>")]
    assert ad.name == name.split(":", 1)[0], (ad.name, name)
    inputs = ad.discover()
    groups = {}
    for i in inputs:
        groups.setdefault(ad.comparison_group(i), []).append(i)
    conn = X.primed_conn(ad)          # the steady state after the fold: the anchor currency decides every contrast
    recs = []
    for g in sorted(groups):
        for batch in ad.read(X.ctx(conn), g, groups[g]):
            recs.extend({k: v for k, v in r.items() if k != "record_id"} for r in batch.records)
        recs.append({"status_extra": ad.status_extra(g)})
    states = [dict(s.__dict__) for s in ad.source_states(X.ctx(), inputs)]
    for st in states:
        if st["json_path"] == LC.ROW_REFUSALS_PATH:
            # the ledger row's sha is the inputs' CONTENT fingerprint, and the shared fixture writes one input whose bytes carry its temp path (the none gate
            # record's compared_against.record): held to the fingerprint here and pinned as a placeholder, so the pin is about what the reader emits
            assert st["input_sha256"] == AB.inputs_fingerprint(inputs), st
            st["input_sha256"] = "<INPUTS_FINGERPRINT>"
    txt = json.dumps({"records": recs, "states": states}, sort_keys=True, separators=(",", ":"), default=str)
    txt = _normalise(txt, roots + [(d, "<TMP>"), (tmp_path, "<TMP>")])
    return hashlib.sha256(txt.encode()).hexdigest(), len(recs), len(states)


@pytest.mark.parametrize("name", sorted(PINNED))
def test_the_full_and_shared_readers_emit_byte_for_byte_what_they_emitted_before_the_declared_lane(name, tmp_path, seeds):
    got = emission(name, tmp_path, seeds)
    assert got == PINNED[name], (name, got, "the emission moved: a deliberate change re-pins here with an EMISSION_VERSION bump and a ledger line; an accidental one is a regression")


def _read_groups(ad):
    """Every group of a mirror read on a frame primed from the adapter's own emissions: ``{group: (n runs, status extra)}``."""
    inputs = ad.discover()
    groups = {}
    for i in inputs:
        groups.setdefault(ad.comparison_group(i), []).append(i)
    conn = X.primed_conn(ad)
    out = {}
    for g in sorted(groups):
        n = 0
        for batch in ad.read(X.ctx(conn), g, groups[g]):
            n += sum(1 for r in batch.records if r["record_type"] == "run")
        out[g] = (n, ad.status_extra(g))
    return out


def test_the_full_mirrors_read_every_optional_shape_they_carry(tmp_path, seeds):
    """The pin is only as strong as the fixture: the full mirrors' knobs put a row in every group the full reader reads, the relaunch parks its first store,
    and the one refusal is the undeclared row dir's (a whole-group refusal by name) -- a knob that silently stopped producing rows would leave the pin
    hashing an empty read."""
    fx = Z.mirror(tmp_path, seeds, eval_id="bigcodebench", root=str(tmp_path / "shapes"), **FULL_KNOBS["shapes"])
    got = _read_groups(Z.adapter(BCF.BigcodebenchFullAdapter, fx, seeds))
    assert sorted(got) == sorted([Z.TAG, Z.QWEN_TAG, Z.TAG35, Z.UNDECLARED_ROW_DIR]), sorted(got)
    for g, (n, facts) in got.items():
        if g == Z.UNDECLARED_ROW_DIR:
            assert facts.get("refuse") and n == 0, facts.get("refusals")
        else:
            assert facts["n_refusals"] == 0 and not facts.get("refuse"), (g, facts["refusals"])
    assert {g: n for g, (n, _f) in got.items()} == {Z.TAG: 6, Z.QWEN_TAG: 1, Z.TAG35: 13, Z.UNDECLARED_ROW_DIR: 0}
    fx2 = Z.mirror(tmp_path / "r", seeds, eval_id="bigcodebench", root=str(tmp_path / "relaunch"), **FULL_KNOBS["relaunch"])
    got2 = _read_groups(Z.adapter(BCF.BigcodebenchFullAdapter, fx2, seeds))
    assert {g: n for g, (n, _f) in got2.items()} == {Z.TAG: 6, Z.QWEN_TAG: 1}, got2
    facts = got2[Z.TAG][1]
    assert facts["n_refusals"] == 0 and facts["n_parked"] == 1 and facts["parked"][0]["object_key"] == Z.RELU, (facts["refusals"], facts.get("parked"))
    assert len(facts.get("gate_records_without_store") or []) == 1, facts.get("gate_records_without_store")
