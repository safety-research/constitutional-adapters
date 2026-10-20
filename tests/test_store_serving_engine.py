"""A store's SERVING ENGINE is read off EVERY spelling the producer writes,
INDEPENDENTLY of the lens reduction mode, and ``pairing.compare_sides`` refuses a
cross-engine pair of stores whose manifests carry NO tinycap ``route`` block.

That is the real shape of a native XSTest / OR-Bench / MMLU-only run (the legacy controls):
``served_capability_run`` stamps ``serving_engine`` top-level and inside the hashed
``resolved_config.serving`` on every store, and adds the ``route`` block only ``if atoms``;
the lens counterpart carries none of the engine spellings. Before this the pairing guard read
``read_store_reduction_mode`` alone — "nothing stated" without a ``route`` block on BOTH
sides — so such a pair pooled silently. An HF reference (``cell_controls_run``: ``engine:
hf``, no ``serving_path``, no ``route``) states no engine and keeps pairing exactly as before.
CPU-only; every manifest here is synthetic, in the producer's key shape (the driver-produced
one is exercised in ``test_served_capability_native``)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from sisteer.capeval import manifest as cm
from sisteer.capeval.tiers import pairing as P
from sisteer.capeval.tinycap import contracts as C

REPO = Path(__file__).resolve().parents[1]
NATIVE, LENS = C.SERVING_ENGINE_LORA_NATIVE, C.SERVING_ENGINE_LENS
NA = C.REDUCTION_MODE_NOT_APPLICABLE
PASS_ID = "proxy_vllm+passthrough+interleaved"
WHERE = "<store>/manifest.json"
TOP, ROUTE, RESOLVED, TINYCAP = ("serving_engine", "route.serving_engine",
                                 "resolved_config.serving.serving_engine", "tinycap.served.serving_engine")


def legacy_manifest(*, native: bool, cell: str = "none") -> dict:
    """``served_capability_run``'s manifest for a run with NO tinycap atoms (the legacy
    XSTest / OR-Bench / MMLU controls), key for key: the engine PRESENT IFF native, top-level
    (unhashed) and inside the hashed identity's ``serving`` block; no ``route`` block."""
    engine = {C.SERVED_ENGINE_FIELD: NATIVE} if native else {}
    resolved = {"driver": "served_capability_run_v2", "engine": "vllm",
                "serving": {"virtual_model": f"petri__{cell}", "serving_identity": PASS_ID, **engine},
                "cell_id": cell, "model_id": "TESTORG/testmodel", "model_revision": "0" * 40,
                "gen": {"max_new_tokens": 64}, "controls": ["xstest", "or_bench_hard_1k"], "order_seed": 0,
                "limit": None, "scale": 1.0}
    return {"cell_id": cell, "harness_cell_id": cell, "virtual_model": f"petri__{cell}",
            "intervention_hash": "0" * 16, "harness_config_hash": "1" * 16,
            "model_id": "TESTORG/testmodel", "model_revision": "0" * 40, "engine": "vllm",
            "serving_path": cm.parse_serving_identity(PASS_ID).path, "serving_identity": PASS_ID, **engine,
            "proxy_url": "http://127.0.0.1:19000", "routing_table": "/tmp/routing_table.json",
            "steer_positions": "all",
            "row_constants": {"sweep": "two_si_controls_served", "engine": "vllm", "intervention_hash": "0" * 16,
                              "harness_config_hash": "1" * 16},
            "resolved_config": resolved, "controls": ["xstest", "or_bench_hard_1k"],
            C.MANIFEST_KEY_PRODUCER: {"driver": C.PRODUCER_DRIVER_SERVED, "tree_sha": "5" * 40},
            C.MANIFEST_KEY_CONTRACT_VERSION: C.CONTRACT_VERSION}


def tinycap_manifest(*, native: bool, stated_lens: bool = False) -> dict:
    """A tinycap run's manifest: the legacy shape PLUS the ``route`` block and the hashed
    ``tinycap.served`` block, the engine present iff native in both (a lens route may STATE
    ``lens`` beside its ``per_row`` mode)."""
    man = legacy_manifest(native=native, cell="constv2-mix-full__lora_r4_ce")
    route = {"virtual_model": man["virtual_model"], "serving_identity": PASS_ID, C.AGENT_LOOP_ENTRY_KEY: True,
             "tool_calls": False, C.REDUCTION_MODE_FIELD: NA if native else "per_row"}
    served = {"max_model_len": 4096, "max_num_seqs": 24, "concurrency": 24, C.REDUCTION_MODE_FIELD: route[C.REDUCTION_MODE_FIELD]}
    if native:
        route[C.SERVED_ENGINE_FIELD] = served[C.SERVED_ENGINE_FIELD] = NATIVE
    elif stated_lens:
        route[C.SERVED_ENGINE_FIELD] = LENS
    man[C.MANIFEST_KEY_ROUTE] = route
    man[C.MANIFEST_KEY_TINYCAP] = {"served": served}
    return man


def hf_manifest() -> dict:
    """``cell_controls_run``'s HF reference manifest: ``engine: hf``, its own serving identity,
    no ``serving_path``, no ``route`` — not a served store."""
    return {"cell_id": "none", "intervention_hash": "0" * 16, "harness_config_hash": "2" * 16, "model_revision": "0" * 40,
            "engine": "hf", "row_constants": {"engine": "hf"}, "resolved_config": {"engine": "hf"},
            "serving_identity": "hf", "judge_model": "TESTORG/judge", "judge_backend": "local", "provenance": {}}


def _reading(man):
    r = C.read_store_serving_engine(man, where=WHERE)
    return r.engine, r.stated, r.refusal


def test_the_engine_reader_reads_every_producer_spelling_and_dates_a_served_store_by_era():
    assert C.StoreServingEngine._fields == ("engine", "stated", "refusal", "why")
    assert C.SERVING_ENGINE_SPELLINGS == ((TOP.split(".")[0],), tuple(ROUTE.split(".")), tuple(RESOLVED.split(".")),
                                          tuple(TINYCAP.split(".")))
    # the legacy-controls native manifest: two spellings, both read, named
    read = C.read_store_serving_engine(legacy_manifest(native=True), where=WHERE)
    assert (read.engine, read.stated, read.refusal) == (NATIVE, NATIVE, None)
    assert read.why == f"{WHERE} {[RESOLVED, TOP]} = {NATIVE!r}"
    # ... and THE hole the reader closes: the reduction-mode resolver reads NOTHING off it (no route block)
    mode = C.read_store_reduction_mode(legacy_manifest(native=True), where=WHERE)
    assert (mode.mode, mode.refusal, mode.serving_engine) == (None, None, None) and "not a served-lens store" in mode.why
    # its lens counterpart states no engine under any spelling: a SERVED store, so lens BY ERA (stated None)
    read = C.read_store_serving_engine(legacy_manifest(native=False), where=WHERE)
    assert (read.engine, read.stated, read.refusal) == (LENS, None, None) and "by era" in read.why
    # a tinycap native store spells it four ways; a lens route that SAYS lens reads stated "lens"
    read = C.read_store_serving_engine(tinycap_manifest(native=True), where=WHERE)
    assert (read.engine, read.stated) == (NATIVE, NATIVE) and read.why == f"{WHERE} {[RESOLVED, ROUTE, TOP, TINYCAP]} = {NATIVE!r}"
    assert _reading(tinycap_manifest(native=False)) == (LENS, None, None)
    assert _reading(tinycap_manifest(native=False, stated_lens=True)) == (LENS, LENS, None)
    # a served-store marker alone: lens only on POSITIVE evidence — the parsed `serving_path`
    # says lens; a bare `engine: vllm` or a bare route block names no identity and no dated producer, so the engine is
    # UNKNOWN and refuses (the earlier reader read every one of these as lens by era)
    assert _reading({"serving_path": "lens"}) == (LENS, None, None)
    for marker in ({"engine": "vllm"}, {C.MANIFEST_KEY_ROUTE: {"virtual_model": "vm"}}):
        engine, stated, refusal = _reading(marker)
        assert (engine, stated) == (None, None) and refusal and "an unknown serving engine" in refusal and "not read as either" in refusal, marker
    # NOT a served store: an HF reference, a manifest naming none of the markers, no manifest at all
    for man in (hf_manifest(), {"cell": "x"}, None, "not a mapping"):
        engine, stated, refusal = _reading(man)
        assert (engine, stated, refusal) == (None, None, None), man
    assert "an HF reference states no serving engine" in C.read_store_serving_engine(hf_manifest(), where=WHERE).why


@pytest.mark.parametrize("edit,needle", [
    ({RESOLVED: LENS}, "two ways"),                     # the spellings disagree: torn
    ({TOP: None}, "not one of"),                        # a present null
    ({TOP: "hf"}, "not one of"),                        # an unknown engine
])
def test_a_torn_or_unknown_engine_spelling_is_stated_and_unusable(edit, needle):
    man = legacy_manifest(native=True)
    for path, value in edit.items():
        node = man
        for k in path.split(".")[:-1]:
            node = node[k]
        node[path.split(".")[-1]] = value
    read = C.read_store_serving_engine(man, where=WHERE)
    assert read.engine is None and read.stated is None and read.refusal and needle in read.refusal


LORA_ID, LENS_ID = "proxy_vllm+lora+interleaved", "proxy_vllm+lens+interleaved"
IDENTITY_SPELLINGS = ("serving_identity", "route.serving_identity", "resolved_config.serving.serving_identity")


def _set(man: dict, dotted: str, value) -> dict:
    node = man
    for k in dotted.split(".")[:-1]:
        node = node[k]
    node[dotted.split(".")[-1]] = value
    return man


def _unstamped(man: dict, identity: str | None) -> dict:
    """``man`` with EVERY engine spelling removed and every present identity spelling set to ``identity`` (``None``
    removes them too): the shape of a record torn of its stamps."""
    for path in (TOP, ROUTE, RESOLVED, TINYCAP):
        node = man
        for k in path.split(".")[:-1]:
            node = node.get(k) if isinstance(node, dict) else None
        if isinstance(node, dict):
            node.pop(path.split(".")[-1], None)
    for dotted in IDENTITY_SPELLINGS:
        node = man
        for k in dotted.split(".")[:-1]:
            node = node.get(k) if isinstance(node, dict) else None
        if isinstance(node, dict) and "serving_identity" in node:
            if identity is None:
                node.pop("serving_identity")
            else:
                node["serving_identity"] = identity
    if identity is None:
        man.pop("serving_path", None)
    else:
        man["serving_path"] = cm.parse_serving_identity(identity).path
    return man


def test_an_unstamped_served_store_is_lens_only_on_positive_evidence_and_a_lora_identity_refuses(tmp_path):
    """The earlier reader classified EVERY served manifest lacking all four engine spellings as lens BY ERA — a native record torn of its
    stamps, whose own route identity says `+lora+`, read as lens and paired with a lens store past the cross-engine guard. The by-era reading
    now needs POSITIVE evidence: a `lens` or `passthrough` identity / serving_path (agreeing across every present spelling), or a producer
    record dating the store before the stamp; a `lora` identity without the stamp is a torn NATIVE record and refuses by name, an undatable
    identity-less store is unknown and refuses, and disagreeing or malformed identities refuse. The pairing guard refuses each."""
    # THE finding: a native tinycap store and a native legacy-controls store, torn of every stamp, identities `+lora+`
    for torn in (_unstamped(tinycap_manifest(native=True), LORA_ID), _unstamped(legacy_manifest(native=True), LORA_ID)):
        engine, stated, refusal = _reading(torn)
        assert (engine, stated) == (None, None) and refusal, torn
        assert "name the native path ('lora')" in refusal and "a torn native record" in refusal and "not a 'lens' store by era" in refusal
    # positive evidence: `+lens+` and passthrough identities (agreeing everywhere) are the lens's, named as such
    for ident, path in ((LENS_ID, "lens"), (PASS_ID, "passthrough")):
        for man in (_unstamped(tinycap_manifest(native=False), ident), _unstamped(legacy_manifest(native=False), ident)):
            read = C.read_store_serving_engine(man, where=WHERE)
            assert (read.engine, read.stated, read.refusal) == (LENS, None, None), man
            assert f"name the {path!r} path" in read.why and "by era" in read.why
    # the fixtures of this file, as they stand: the lens legacy and tinycap shapes carry passthrough identities -> lens
    assert _reading(legacy_manifest(native=False)) == (LENS, None, None) and _reading(tinycap_manifest(native=False)) == (LENS, None, None)
    # spellings that DISAGREE on the path, and a malformed identity, refuse
    two = _set(_unstamped(tinycap_manifest(native=False), LENS_ID), "route.serving_identity", LORA_ID)
    engine, stated, refusal = _reading(two)
    assert engine is None and refusal and "spells its serving path two ways" in refusal
    engine, stated, refusal = _reading({"engine": "vllm", "serving_identity": "hf"})
    assert engine is None and refusal and "is not a serving identity" in refusal
    engine, stated, refusal = _reading({"engine": "vllm", "serving_path": "banana"})
    assert engine is None and refusal and "serving_path 'banana' is not one of" in refusal
    engine, stated, refusal = _reading({"engine": "vllm", "serving_path": "apply"})       # hf_shim's path: no served store's
    assert engine is None and refusal and "name the 'apply' path, which no served store is written under" in refusal
    # a route block stating a LENS reduction mode is positive evidence on its own (only a lens standup records one): with
    # no identity, or beside a lens / passthrough one; `not_applicable` is no such evidence (the mode resolver refuses it
    # without the native stamp), and a `+lora+` identity beside a lens mode is torn — the identity wins the refusal
    for lens_mode in C.REDUCTION_MODES:
        read = C.read_store_serving_engine({"engine": "vllm", C.MANIFEST_KEY_ROUTE: {C.REDUCTION_MODE_FIELD: lens_mode}}, where=WHERE)
        assert (read.engine, read.stated, read.refusal) == (LENS, None, None) and f"lens arithmetic {lens_mode!r}" in read.why
    engine, stated, refusal = _reading({"engine": "vllm", C.MANIFEST_KEY_ROUTE: {C.REDUCTION_MODE_FIELD: NA}})
    assert engine is None and refusal and "an unknown serving engine" in refusal
    engine, stated, refusal = _reading({"engine": "vllm", "serving_identity": LORA_ID, C.MANIFEST_KEY_ROUTE: {C.REDUCTION_MODE_FIELD: "per_row"}})
    assert engine is None and refusal and "a torn native record" in refusal
    # identity-less: dated BEFORE the stamp by its own record -> lens by era; current or undatable -> unknown, refuses
    read = C.read_store_serving_engine({"engine": "vllm", C.MANIFEST_KEY_CONTRACT_VERSION: 2}, where=WHERE)
    assert (read.engine, read.stated, read.refusal) == (LENS, None, None) and "predates the serving-engine stamp" in read.why
    historical_tree = next(sha for sha, p in C.KNOWN_PRODUCER_TREES.items() if C.CONTRACT_TEMPLATE_CHECK in p.does_not_contain)
    read = C.read_store_serving_engine({"engine": "vllm", C.MANIFEST_KEY_PRODUCER: {"tree_sha": historical_tree, "driver": C.PRODUCER_DRIVER_SERVED}}, where=WHERE)
    assert (read.engine, read.stated, read.refusal) == (LENS, None, None) and "by era" in read.why
    for undatable in ({"engine": "vllm", C.MANIFEST_KEY_CONTRACT_VERSION: C.CONTRACT_VERSION}, {"engine": "vllm"},
                      {"engine": "vllm", C.MANIFEST_KEY_PRODUCER: {"tree_sha": "9" * 40, "driver": C.PRODUCER_DRIVER_SERVED}}):
        engine, stated, refusal = _reading(undatable)
        assert (engine, stated) == (None, None) and refusal and "does not date it before the stamp" in refusal, undatable
    # every STAMPED reading is untouched: native spells it, a lens route that says lens says lens
    assert _reading(tinycap_manifest(native=True)) == (NATIVE, NATIVE, None) and _reading(legacy_manifest(native=True)) == (NATIVE, NATIVE, None)
    assert _reading(tinycap_manifest(native=False, stated_lens=True)) == (LENS, LENS, None)
    # the pairing guard refuses the torn native record by name — against a lens store, a native store, and itself. The
    # LEGACY shape is THE evasion (no route block, so the reduction-mode resolver says nothing and only the engine reader
    # stands between it and a lens store); the torn tinycap shape is caught one check earlier, by the mode resolver's own
    # torn verdict (`not_applicable` without the engine), and is asserted as such
    torn = _store(tmp_path, "torn_native", _unstamped(legacy_manifest(native=True), LORA_ID))
    lens = _store(tmp_path, "lens", legacy_manifest(native=False))
    native = _store(tmp_path, "native", legacy_manifest(native=True))
    for a, b in ((torn, lens), (lens, torn), (torn, native), (native, torn), (torn, torn)):
        with pytest.raises(P.PairingError) as ei:
            P.compare_sides(a, b)
        assert "'torn_native'" in str(ei.value) and "a torn native record" in str(ei.value), str(ei.value)
    torn_tiny = _store(tmp_path, "torn_tiny", _unstamped(tinycap_manifest(native=True), LORA_ID))
    with pytest.raises(P.PairingError, match="only a 'lora_native' store records that value"):
        P.compare_sides(torn_tiny, lens)
    undated = _store(tmp_path, "undated", {"engine": "vllm", "cell_id": "none"})
    with pytest.raises(P.PairingError, match="does not date it before the stamp"):
        P.compare_sides(undated, lens)


def test_the_mismatch_verdict_is_none_when_either_side_states_nothing():
    native = C.read_store_serving_engine(legacy_manifest(native=True), where="<a>")
    lens = C.read_store_serving_engine(legacy_manifest(native=False), where="<b>")
    hf = C.read_store_serving_engine(hf_manifest(), where="<c>")
    stated_lens = C.read_store_serving_engine(tinycap_manifest(native=False, stated_lens=True), where="<d>")
    assert C.serving_engine_mismatch(native, native) is None and C.serving_engine_mismatch(lens, lens) is None
    assert C.serving_engine_mismatch(native, hf) is None and C.serving_engine_mismatch(hf, lens) is None
    why = C.serving_engine_mismatch(native, lens)
    assert why and "different serving engines" in why and "(by era: it states no engine under any spelling)" in why
    why = C.serving_engine_mismatch(stated_lens, native)
    assert why and "different serving engines" in why and "by era" not in why


def _store(root: Path, name: str, manifest: dict | None, *, engine: str = "vllm", acc=(1.0, 0.0, 1.0, 1.0)) -> P.Side:
    d = root / name / "cells" / "cellkey"
    d.mkdir(parents=True)
    if manifest is not None:
        (d / "manifest.json").write_text(json.dumps(manifest, indent=1))
    rows = [{"task": "xstest", "metric": "acc", "item_id": f"i{i}", "value": v, "engine": engine} for i, v in enumerate(acc)]
    (d / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return P.Side(label=name, rows_path=d / "rows.jsonl", task="xstest", engine=engine, metric="acc")


def test_pairing_refuses_a_cross_engine_pair_without_a_route_block_and_keeps_every_other_verdict(tmp_path):
    native_a = _store(tmp_path, "native_a", legacy_manifest(native=True))
    native_b = _store(tmp_path, "native_b", legacy_manifest(native=True), acc=(1.0, 1.0, 0.0, 1.0))
    lens_a = _store(tmp_path, "lens_a", legacy_manifest(native=False))
    lens_b = _store(tmp_path, "lens_b", legacy_manifest(native=False), acc=(0.0, 0.0, 1.0, 1.0))
    hf = _store(tmp_path, "hf", hf_manifest(), engine="hf")
    # THE finding: a native legacy-controls store against its lens counterpart paired silently; it refuses now, both ways
    for a, b in ((native_a, lens_a), (lens_a, native_a)):
        with pytest.raises(P.PairingError) as ei:
            P.compare_sides(a, b)
        assert "different serving engines" in str(ei.value) and "by era" in str(ei.value)
    # the engine reading rides the pairing seam's own locator
    assert P.store_serving_engine_reading(native_a.rows_path).engine == NATIVE
    assert P.store_serving_engine_reading(lens_a.rows_path) == C.read_store_serving_engine(
        json.loads((lens_a.rows_path.parent / "manifest.json").read_text()), where=str(lens_a.rows_path.parent / "manifest.json"))
    # native vs native pools and the report names the engine; lens vs lens pools with NO engine key (byte-identical report)
    rep = P.compare_sides(native_a, native_b)
    assert rep["sides"]["a"]["serving_engine"] == NATIVE and rep["sides"]["b"]["serving_engine"] == NATIVE
    assert rep["sides"]["a"]["reduction_mode"] is None            # the legacy store states no lens mode — unchanged
    rep = P.compare_sides(lens_a, lens_b)
    assert "serving_engine" not in rep["sides"]["a"] and "serving_engine" not in rep["sides"]["b"]
    assert rep["n_items"] == 4
    # an HF reference states no engine: the HF-vs-served comparison pairs with EITHER engine, as before
    for served in (native_a, lens_a):
        rep = P.compare_sides(hf, served)
        assert rep["n_items"] == 4 and "serving_engine" not in rep["sides"]["a"]
    # a tinycap native store (route block, `not_applicable`) against a legacy LENS store: the mode check passes
    # (`not_applicable` vs nothing stated) and the engine check refuses; the reverse pairing too
    tiny_native = _store(tmp_path, "tiny_native", tinycap_manifest(native=True))
    tiny_lens = _store(tmp_path, "tiny_lens", tinycap_manifest(native=False))
    with pytest.raises(P.PairingError, match="different serving engines"):
        P.compare_sides(tiny_native, lens_a)
    with pytest.raises(P.PairingError, match="different serving engines"):
        P.compare_sides(native_a, tiny_lens)
    # two tinycap stores keep the route-block verdict, in its own (reduction-mode) words
    with pytest.raises(P.PairingError) as ei:
        P.compare_sides(tiny_native, tiny_lens)
    assert str(ei.value).count("different serving engines") == 1 and "reduction_mode 'not_applicable' vs 'per_row'" in str(ei.value)
    # a torn engine spelling refuses by name before any row is read
    torn = legacy_manifest(native=True)
    torn["resolved_config"]["serving"][C.SERVED_ENGINE_FIELD] = LENS
    with pytest.raises(P.PairingError, match="two ways"):
        P.compare_sides(_store(tmp_path, "torn", torn), native_a)
    # a store with NO manifest states nothing (reported, never guessed) and pairs
    assert P.compare_sides(_store(tmp_path, "bare", None), native_a)["n_items"] == 4


FIXTURES = sorted(str(p.relative_to(REPO / "tests" / "fixtures"))
                  for p in (REPO / "tests" / "fixtures").glob("tinycap_identity_*/standard_*_gsm8k.json"))


@pytest.mark.parametrize("rel", FIXTURES)
def test_every_historical_fixture_store_reads_as_a_lens_store_by_era_and_stays_byte_identical(rel):
    """The captured goldens' manifests (a lens route block naming ``per_row``, no engine spelling anywhere): the engine
    reader says lens BY ERA with nothing stated, moves no byte, and the pair of them pools as before."""
    path = REPO / "tests" / "fixtures" / rel
    doc = json.loads(path.read_text())
    before = hashlib.sha256(json.dumps(doc["manifest"], sort_keys=True).encode()).hexdigest()
    read = C.read_store_serving_engine(doc["manifest"], where=str(path))
    assert (read.engine, read.stated, read.refusal) == (LENS, None, None)
    assert C.read_store_reduction_mode(doc["manifest"], where=str(path)).mode == "per_row"
    assert hashlib.sha256(json.dumps(doc["manifest"], sort_keys=True).encode()).hexdigest() == before


def _relabelled(man: dict, engine: str, identity: str) -> dict:
    """``man`` with EVERY present engine spelling set to ``engine`` and every present identity spelling
    set to ``identity`` (its parsed path as ``serving_path``): a partially relabelled record whose stamps
    agree with each other and disagree with the serving identity it still carries."""
    for path in (TOP, ROUTE, RESOLVED, TINYCAP):
        node = man
        for k in path.split(".")[:-1]:
            node = node.get(k) if isinstance(node, dict) else None
        if isinstance(node, dict):
            node[path.split(".")[-1]] = engine
    for dotted in IDENTITY_SPELLINGS:
        node = man
        for k in dotted.split(".")[:-1]:
            node = node.get(k) if isinstance(node, dict) else None
        if isinstance(node, dict) and "serving_identity" in node:
            node["serving_identity"] = identity
    man["serving_path"] = cm.parse_serving_identity(identity).path
    return man


def test_agreeing_engine_stamps_do_not_bypass_a_contradictory_serving_identity(tmp_path):
    """`read_store_serving_engine` returned on STAMP consensus before the identity/path checks, so a partially relabelled legacy-controls
    manifest — agreeing `serving_engine="lens"` stamps top-level and in the hashed `resolved_config.serving` while its serving identities
    stayed `+lora+` and its `serving_path` `lora` — read as a lens store, and `compare_sides()` pooled it with a lens peer (and the mirror
    image, native stamps beside `+lens+`, with a native peer); with no `route` block the reduction-mode guard saw nothing. Now a stamped
    record's identity is validated too, through the ONE evidence reader both branches share (`_serving_path_evidence`) and the per-engine path
    table `SERVING_PATHS_OF_ENGINE`: `lens` beside `lora` and `lora_native` beside `lens` refuse by name; a passthrough identity beside
    `lora_native` stays a valid unsteered native leg and beside `lens` a valid lens one; a stamped record's malformed identity, two-way path
    or unknown path refuse as an unstamped one's do; a stamped record naming no identity anywhere still reads as its stamps. Both headline
    cases refuse through `compare_sides()`, in both orders."""
    assert C.SERVING_PATHS_OF_ENGINE == {LENS: ("lens", "passthrough"), NATIVE: ("lora", "passthrough")}
    # THE finding: agreeing lens stamps (top-level + hashed), identities +lora+, path lora, no route block
    relabelled = _relabelled(legacy_manifest(native=True), LENS, LORA_ID)
    assert relabelled[C.SERVED_ENGINE_FIELD] == relabelled["resolved_config"]["serving"][C.SERVED_ENGINE_FIELD] == LENS
    assert relabelled["serving_identity"] == LORA_ID and relabelled["serving_path"] == "lora" and C.MANIFEST_KEY_ROUTE not in relabelled
    read = C.read_store_serving_engine(relabelled, where=WHERE)
    assert (read.engine, read.stated) == (None, None) and read.refusal, read
    assert f"stamp {C.SERVED_ENGINE_FIELD}={LENS!r}" in read.refusal and "name the 'lora' path" in read.refusal
    assert f"the {NATIVE!r} engine's path" in read.refusal and "partially relabelled or hand-edited" in read.refusal
    assert read.why == f"{WHERE} {[RESOLVED, TOP]} vs {['resolved_config.serving.serving_identity', 'serving_identity', 'serving_path']}"
    # ... and the mirror image: native stamps (all four, the tinycap shape) beside +lens+ identities
    mirror = _relabelled(tinycap_manifest(native=True), NATIVE, LENS_ID)
    assert mirror[C.MANIFEST_KEY_ROUTE][C.SERVED_ENGINE_FIELD] == NATIVE and mirror[C.MANIFEST_KEY_ROUTE]["serving_identity"] == LENS_ID
    read = C.read_store_serving_engine(mirror, where=WHERE)
    assert (read.engine, read.stated) == (None, None) and read.refusal
    assert f"stamp {C.SERVED_ENGINE_FIELD}={NATIVE!r}" in read.refusal and "name the 'lens' path" in read.refusal and f"the {LENS!r} engine's path" in read.refusal
    # the CONSISTENT stamped records read as before, `why` byte-identical: native beside lora, lens beside lens, and
    # passthrough beside EITHER (the native driver stamps its `none` / prompted routes so)
    assert _reading(_relabelled(legacy_manifest(native=True), NATIVE, LORA_ID)) == (NATIVE, NATIVE, None)
    assert _reading(_relabelled(tinycap_manifest(native=True), NATIVE, LORA_ID)) == (NATIVE, NATIVE, None)
    assert _reading(_relabelled(legacy_manifest(native=True), LENS, LENS_ID)) == (LENS, LENS, None)
    assert _reading(tinycap_manifest(native=False, stated_lens=True)) == (LENS, LENS, None)          # lens stamp, passthrough
    read = C.read_store_serving_engine(legacy_manifest(native=True), where=WHERE)                     # native stamps, passthrough
    assert (read.engine, read.stated, read.refusal) == (NATIVE, NATIVE, None) and read.why == f"{WHERE} {[RESOLVED, TOP]} = {NATIVE!r}"
    # a stamped record's identity evidence is validated as an unstamped one's is — malformed, two-way, unknown path
    engine, _, refusal = _reading(_set(legacy_manifest(native=True), "serving_identity", "hf"))
    assert engine is None and refusal and "is not a serving identity" in refusal and f"stamping {C.SERVED_ENGINE_FIELD}={NATIVE!r}" in refusal
    engine, _, refusal = _reading(_set(legacy_manifest(native=True), "resolved_config.serving.serving_identity", LORA_ID))
    assert engine is None and refusal and "spells its serving path two ways" in refusal and f"stamping {C.SERVED_ENGINE_FIELD}={NATIVE!r}" in refusal
    engine, _, refusal = _reading(_set(legacy_manifest(native=True), "serving_path", "banana"))
    assert engine is None and refusal and "serving_path 'banana' is not one of" in refusal
    apply = _set(_set(_set(legacy_manifest(native=True), "serving_identity", "hf_shim+apply+interleaved"),
                      "resolved_config.serving.serving_identity", "hf_shim+apply+interleaved"), "serving_path", "apply")
    engine, _, refusal = _reading(apply)
    assert engine is None and refusal and "name the 'apply' path" in refusal and "a path no served store is written under" in refusal
    # a stamped record naming NO identity or path anywhere reads as its stamps (nothing contradicts them)
    assert _reading({C.SERVED_ENGINE_FIELD: NATIVE}) == (NATIVE, NATIVE, None)
    assert _reading({C.MANIFEST_KEY_ROUTE: {C.SERVED_ENGINE_FIELD: LENS}}) == (LENS, LENS, None)
    # the unstamped branch is untouched, its wording byte for byte
    engine, _, refusal = _reading(_unstamped(tinycap_manifest(native=True), LORA_ID))
    assert engine is None and "a torn native record" in refusal and "not a 'lens' store by era" in refusal
    engine, _, refusal = _reading({"engine": "vllm", "serving_identity": "hf"})
    assert refusal.endswith("on a served store stating no serving_engine under any spelling — a torn or hand-edited manifest; not read as either engine")
    # THROUGH THE PAIRING GUARD: the relabelled record against a lens peer, and the mirror against a
    # native peer, refuse by name in both orders — before, `compare_sides` accepted four items from each pair
    lens_peer = _store(tmp_path, "lens_peer", legacy_manifest(native=False))
    native_peer = _store(tmp_path, "native_peer", legacy_manifest(native=True))
    bad_lens = _store(tmp_path, "relabelled_lens", relabelled)
    bad_native = _store(tmp_path, "relabelled_native", mirror)
    for a, b in ((bad_lens, lens_peer), (lens_peer, bad_lens), (bad_lens, native_peer), (bad_lens, bad_lens)):
        with pytest.raises(P.PairingError) as ei:
            P.compare_sides(a, b)
        assert "'relabelled_lens'" in str(ei.value) and "name the 'lora' path" in str(ei.value), str(ei.value)
    for a, b in ((bad_native, native_peer), (native_peer, bad_native), (bad_native, lens_peer)):
        with pytest.raises(P.PairingError) as ei:
            P.compare_sides(a, b)
        assert "'relabelled_native'" in str(ei.value) and "name the 'lens' path" in str(ei.value), str(ei.value)
    # the consistent pairs pool exactly as before
    assert P.compare_sides(native_peer, _store(tmp_path, "native_lora", _relabelled(legacy_manifest(native=True), NATIVE, LORA_ID)))["n_items"] == 4
    assert P.compare_sides(lens_peer, _store(tmp_path, "lens_lens", _relabelled(legacy_manifest(native=False), LENS, LENS_ID)))["n_items"] == 4
