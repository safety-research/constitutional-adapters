"""Every sampled-decode preset against the model's OWN shipped generation_config.json.

THE FINDING (2026-09-11). ``configs/sdfbase_sampled_decode.yaml`` states a per-request
sampling preset per served model. Some of those models also ship a
``generation_config.json`` in their snapshot, and vLLM (this repo's standup leaves
``--generation-config`` at its ``auto`` default) fills every key an omit-params
request leaves out from that file. So the campaign has TWO ROUTES to the sampler:

* the SAMPLED route -- ``jailbreak_run`` / ``served_capability_run`` put the yaml
  preset on every request body (``sisteer.eval.sampled_decode``);
* the SERVER-DEFAULT route -- an omit-params consumer (the AlphaSteer ladder, BoN,
  the 4,096-cap multi-turn re-runs, the Petri target, agentic misalignment) decodes
  at the shipped file's values.

Where the two disagree and nothing checks, one lane decodes at one value while the
yaml documents another. That happened: the yaml says nemotron3nano4b ``top_p 1.0``,
its shipped file says ``0.95``, and the canonical AlphaSteer ladder drew at 0.95 while
every sampled store drew at 1.0 -- two instruments inside one campaign, invisible to
every existing test for weeks. Ruled: 1.0 is canonical, the ladder is re-run at 1.0,
and the departure from the vendor file is DELIBERATE (NVIDIA's model-card general-chat
recommendation). This module makes the next such discrepancy a NAMED failure.

THE RULE. For every preset row (``presets.<key>`` and each per-readout override
``readouts.<r>.presets.<key>``), on every sampler key the shipped file STATES, the
preset must state the same value -- or the row must declare the difference in a
machine-readable field (the owner's schema, 2026-09-11)::

    deliberate_departure:
      top_p:
        shipped: 0.95
        reason: "NVIDIA card general-chat recommendation ..."
        comparisons:
          single_shot_and_overrefusal_panels: {canonical: true, value: 1.0, arms: [none, ..., alphasteer]}
          math_evals: {value: 0.95, arms: [math500, gsm8k]}

``comparisons`` names every COMPARISON that consumes this model's decode, each at the
ONE value all of its arms decode at. A comparison is a set of arms read against each
other, so a value split INSIDE one is the finding (a 0.95 baseline beside 1.0 objects
in one panel); DIFFERENT comparisons at different declared values PASS -- the math
lane's legs at the shipped 0.95 beside the panels at 1.0 is the per-instrument rule
working, not a defect (the first cut of this test refused that, too broadly). Four
refusals:

1. two arms in ONE comparison at different values -- an arm entry is a bare name
   (decodes at the comparison's ``value``) or ``{<arm>: <value>}`` stating its own,
   which is what makes the intra-comparison split expressible at all.
2. an arm declared in TWO comparisons -- every consumer is declared exactly once, so a
   reader can tell which value it draws at. The other half of this rule, an arm in NO
   comparison, needs a source of truth for which arms decode this model, and THE REPO
   HAS NONE: ``paper/paperlib/config.py`` ``ARMS`` is the paper notebook's model-agnostic
   figure roster (arms with no fitted object, the excluded PSR arms, a
   Llama-only constitution cell, and none of the eval-level consumers -- the math legs,
   BoN, multi-turn, Petri, agentic); the wave roster names cells per view but no evals;
   ``scripts/sdfbase/math500/cells.py`` names math legs but no cells. Approximating the
   arm set from any one of them would turn this refusal into a random failure
   generator, so ONLY the "in two" half is checked; completeness is the declarer's, by
   reading.
3. a comparison whose value is neither the preset's nor the shipped file's -- no third
   sampler.
4. exactly one comparison carries ``canonical: true`` (the one owning the canonical
   sampled instrument -- the panels), and the yaml preset row must equal ITS value, so
   the row stays what the paper's headline arms actually run.

A declaration whose preset no longer differs is stale and refuses too (it would excuse
a future drift). A key the preset states and the file does NOT is not a disagreement
with the file -- the file has no value there (GLM's card-sourced ``repetition_penalty``);
that route gap is ``test_vendor_row_two_route_decode``'s subject (endorsed by the owner).
A key the file states and the preset omits IS a disagreement.

WHY A PIN, AND THE TRI-STATE. Reading the shipped file needs the snapshot; this box
holds partial snapshots and the Modal CPU suite (where the full suite runs) holds no
model files at all, so a test that read the snapshot would skip everywhere that
matters (a live check has that shape). The shipped values are therefore PINNED
into the repo with provenance -- ``configs/sdfbase_shipped_generation_config.json``:
model, exact revision, the sampler keys the file states, sha256 / etag, where it was
read -- and this module asserts the yaml against the pin, so it runs everywhere. The
pin cannot rot silently because a second instrument,
``scripts/sdfbase/full22/sampled/verify_shipped_generation_config.py`` (standard
library only), asserts the pin against the REAL snapshots -- an HF cache or the hub
-- with a tri-state read: present / ships_none (the cache's ``.no_exist`` marker, a
hub 404 or listing without the file) / unknown (no snapshot, or a partial one that
never asked for the file). Absent-because-unavailable and
absent-because-the-vendor-ships-none must never collapse: a pinned ``unknown`` FAILS
here by name, a model that ships none (Qwen3.5-4B) SKIPS by name, and a new yaml row
with no pin FAILS until someone pins it -- a new row never passes by default.

The synthetic tests at the bottom run everywhere and are what prove the check bites:
each perturbation of either side is a named failure whose message names the model,
the key, both values and which route each belongs to.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import urllib.error
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
YAML_PATH = REPO / "configs" / "sdfbase_sampled_decode.yaml"
PIN_PATH = REPO / "configs" / "sdfbase_shipped_generation_config.json"
VERIFIER = REPO / "scripts" / "sdfbase" / "full22" / "sampled" / "verify_shipped_generation_config.py"
YAML_REL, PIN_REL, VERIFIER_REL = (str(p.relative_to(REPO)) for p in (YAML_PATH, PIN_PATH, VERIFIER))

sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(VERIFIER.parent))
from sisteer.eval.sampled_decode import SAMPLER_KEYS  # noqa: E402  -- the drivers' own send set
import verify_shipped_generation_config as vsg  # noqa: E402

FILE = vsg.FILE
ROUTE_PRESET = (f"the request-body preset ({YAML_REL}; the SAMPLED route -- jailbreak_run / "
                "served_capability_run put it on every request)")
ROUTE_SHIPPED = (f"the shipped {FILE} (the SERVER-DEFAULT route -- vLLM fills every key an "
                 "omit-params request leaves out from it)")
FINDING = ("the 2026-09-11 finding: nemotron3nano4b's AlphaSteer ladder, an omit-params consumer, drew at the "
           "shipped top_p 0.95 while every sampled store drew at the preset's 1.0 -- two instruments inside one campaign")
DEPARTURE_KEYS = frozenset({"shipped", "reason", "comparisons"})
COMPARISON_REQUIRED = frozenset({"value", "arms"})
COMPARISON_KEYS = COMPARISON_REQUIRED | {"canonical", "note"}


# ============================================================ the yaml, row by row
class YamlShape(RuntimeError):
    """The yaml cannot be read row by row (a shape this check cannot follow)."""


@dataclass(frozen=True)
class Row:
    name: str                 # presets.<key> | readouts.<r>.presets.<key>
    key: str
    model: str
    revision: str
    preset: Mapping[str, Any]
    departure: Any            # the raw `deliberate_departure` field, or None
    shipped_block: Any        # the raw `shipped_generation_config` field, or None


def yaml_rows(doc: Any, origin: str = YAML_REL) -> List[Row]:
    """Every preset the drivers can send: the top-level rows, then each per-readout
    override (which inherits its row's model + revision). Refuses a row without its
    model / revision / preset, an override of a key with no row, or two rows naming
    one model (the join on the model would be ambiguous)."""
    if not isinstance(doc, Mapping) or not isinstance(doc.get("presets"), Mapping) or not doc["presets"]:
        raise YamlShape(f"{origin} has no non-empty `presets` mapping")
    rows: List[Row] = []
    key_of_model: Dict[str, str] = {}
    for key, p in doc["presets"].items():
        what = f"presets.{key} in {origin}"
        if not isinstance(p, Mapping):
            raise YamlShape(f"{what} is not a mapping")
        model, rev, preset = p.get("model"), p.get("revision"), p.get("preset")
        if not isinstance(model, str) or not model:
            raise YamlShape(f"{what} names no `model`, the join key")
        if not isinstance(rev, str) or not rev:
            raise YamlShape(f"{what} names no `revision`")
        if not isinstance(preset, Mapping) or not preset:
            raise YamlShape(f"{what} states no `preset`")
        if model in key_of_model:
            # the worked example (qwen35_4b_demo) carries the paper's qwen35_4b row verbatim for the same checkpoint: an
            # IDENTICAL twin (revision, preset, shipped block) is one row under the first key, as sampled_decode collapses
            # it; a twin that differs in anything is the ambiguity this refusal is for
            first = next(r for r in rows if r.name == f"presets.{key_of_model[model]}")
            if (first.revision, first.preset, first.shipped_block) == (rev, dict(preset), p.get("shipped_generation_config")):
                continue
            raise YamlShape(f"{origin} states {model!r} under both presets.{key_of_model[model]} and presets.{key}; "
                            "the join on the model is ambiguous")
        key_of_model[model] = key
        rows.append(Row(f"presets.{key}", key, model, rev, dict(preset),
                        p.get("deliberate_departure"), p.get("shipped_generation_config")))
    readouts = doc.get("readouts")
    if isinstance(readouts, Mapping):
        for rname, r in readouts.items():
            if not isinstance(r, Mapping) or not isinstance(r.get("presets"), Mapping):
                continue
            for key, rp in r["presets"].items():
                what = f"readouts.{rname}.presets.{key} in {origin}"
                if key not in doc["presets"]:
                    raise YamlShape(f"{what} overrides a key with no top-level preset row")
                if not isinstance(rp, Mapping) or not isinstance(rp.get("preset"), Mapping) or not rp["preset"]:
                    raise YamlShape(f"{what} states no `preset`")
                top = doc["presets"][key]
                rows.append(Row(f"readouts.{rname}.presets.{key}", key, top["model"], top["revision"], dict(rp["preset"]),
                                rp.get("deliberate_departure"), rp.get("shipped_generation_config")))
    return rows


# ============================================================ the check
@dataclass
class Outcome:
    skip: Optional[str] = None                      # the row is not comparable, and says why
    failures: List[str] = field(default_factory=list)


def _fmt(v: Any) -> str:
    return "no value (the key is absent)" if v is None else f"{v}"


def _num_eq(a: Any, b: Any) -> bool:
    return vsg._num_eq(a, b)


def _undeclared(where: str, row: Row, key: str, preset_value: Any, shipped_value: Any) -> str:
    return (f"{where}: {key} differs and the row declares no departure -- {ROUTE_PRESET} says {_fmt(preset_value)}; "
            f"{ROUTE_SHIPPED} says {_fmt(shipped_value)}. An arm that sends the preset and an arm that omits the key decode "
            f"at different {key} ({FINDING}). Either make the preset match the file, or declare the departure in "
            f"{row.name}.deliberate_departure.{key}: {{shipped: {shipped_value}, reason: <why the preset departs from the "
            f"vendor file>, comparisons: {{<id>: {{canonical: true, value: {_fmt(preset_value)}, arms: [<every arm read "
            f"against another in it>]}}, ...}}}} -- one entry per comparison that consumes this decode, each at its one value")


def departure_failures(where: str, path: str, departure: Any, considered: tuple, preset: Mapping[str, Any],
                       shipped: Mapping[str, Any]) -> List[str]:
    """The declared-departure field's schema and the four refusals: an intra-comparison
    value split, an arm declared in two comparisons, a comparison value that is neither
    source, a canonical comparison the preset row does not equal (see the module doc)."""
    if departure is None:
        return []
    if not isinstance(departure, Mapping) or not departure:
        return [f"{where}: {path}.deliberate_departure must be a non-empty mapping <sampler key> -> "
                f"{{shipped, reason, comparisons}}, got {departure!r}"]
    out: List[str] = []
    for k, d in departure.items():
        p = f"{where}: {path}.deliberate_departure.{k}"
        if k not in considered:
            out.append(f"{p}: {k!r} is not a sampler key ({list(considered)})")
            continue
        if not isinstance(d, Mapping):
            out.append(f"{p} must be a mapping {{shipped, reason, comparisons}}, got {d!r}")
            continue
        extra, missing = sorted(set(d) - DEPARTURE_KEYS), sorted(DEPARTURE_KEYS - set(d))
        if extra or missing:
            out.append(f"{p} must carry exactly {sorted(DEPARTURE_KEYS)}: missing {missing}, unexpected {extra}")
        if "shipped" in d and not vsg._is_number(d["shipped"]):
            out.append(f"{p}.shipped must be the shipped file's number, got {d['shipped']!r}")
        if "reason" in d and (not isinstance(d["reason"], str) or not d["reason"].strip()):
            out.append(f"{p}.reason must be a non-empty string saying WHY the preset departs from the vendor file")
        if "comparisons" not in d:
            continue
        comps = d["comparisons"]
        if not isinstance(comps, Mapping) or not comps:
            out.append(f"{p}.comparisons must be a non-empty mapping <comparison id> -> {{value, arms}} naming every "
                       f"comparison that consumes this model's decode, each at the one value its arms decode at; got {comps!r}")
            continue
        preset_value, shipped_value = preset.get(k), shipped.get(k)
        allowed = [v for v in (preset_value, shipped_value) if vsg._is_number(v)]
        declared_in: Dict[str, List[str]] = {}          # arm -> the comparisons naming it
        canonical_ids: List[str] = []
        malformed = False                                # a comparison the schema refused: refusal 4 stands down
        for cid, c in comps.items():
            q = f"{p}.comparisons.{cid}"
            if not isinstance(c, Mapping):
                out.append(f"{q} must be a mapping {{value, arms}}, got {c!r}")
                malformed = True
                continue
            extra, missing = sorted(set(c) - COMPARISON_KEYS), sorted(COMPARISON_REQUIRED - set(c))
            if extra or missing:
                out.append(f"{q} must carry {sorted(COMPARISON_REQUIRED)} (canonical, note optional): missing {missing}, "
                           f"unexpected {extra}")
                malformed = True
                continue
            value = c["value"]
            if not vsg._is_number(value):
                out.append(f"{q}.value must be the number every arm in the comparison decodes at, got {value!r}")
                malformed = True
                continue
            if "canonical" in c and not isinstance(c["canonical"], bool):
                out.append(f"{q}.canonical must be true or false, got {c['canonical']!r}")
            if c.get("canonical") is True:
                canonical_ids.append(cid)
            if not any(_num_eq(value, a) for a in allowed):                                   # refusal 3
                out.append(f"{q}.value {value} is neither the preset's {k} ({_fmt(preset_value)}) nor the shipped file's "
                           f"({_fmt(shipped_value)}): a third sampler. Every comparison decodes at one of the two declared "
                           f"values -- fix the comparison's value, or the preset, or the pin")
            arms = c["arms"]
            if not isinstance(arms, list) or not arms:
                out.append(f"{q}.arms must be a non-empty list of the arms read against each other in it (a bare name "
                           f"decodes at the comparison's value; `{{<arm>: <value>}}` states an arm's own); got {arms!r}")
                malformed = True
                continue
            by_value: Dict[float, List[str]] = {}
            names_here: List[str] = []
            for a in arms:
                if isinstance(a, str) and a:
                    name, v = a, value
                elif (isinstance(a, Mapping) and len(a) == 1 and isinstance(next(iter(a)), str) and next(iter(a))
                      and vsg._is_number(next(iter(a.values())))):
                    name, v = next(iter(a.items()))
                else:
                    out.append(f"{q}.arms entry {a!r} is neither an arm name nor `{{<arm>: <value>}}`")
                    continue
                if name in names_here:
                    out.append(f"{q}.arms names {name!r} twice")
                names_here.append(name)
                by_value.setdefault(float(v), []).append(name)
                if cid not in declared_in.setdefault(name, []):
                    declared_in[name].append(cid)
            if len(by_value) > 1:                                                             # refusal 1
                sides = "; ".join(f"{names} at {v}" for v, names in sorted(by_value.items()))
                out.append(f"{q}: arms inside ONE comparison decode at different {k} -- {sides}. A comparison is a set of "
                           f"arms read against each other, so a value split inside it is two instruments in one panel "
                           f"({FINDING}; the coordinator's AlphaSteer ruling: a 0.95 baseline beside 1.0 objects in one "
                           f"panel). Re-run the odd arms at the comparison's value, or move them to a comparison of their own")
        for name, cids in declared_in.items():                                                # refusal 2 (the checkable half)
            if len(cids) > 1:
                out.append(f"{p}: arm {name!r} is declared in {len(cids)} comparisons {cids}: every consumer of this "
                           f"model's decode is declared EXACTLY once, so a reader can tell which value it draws at -- "
                           f"keep it in one comparison")
        if malformed:                                                                         # refusal 4 needs a well-formed set
            continue
        if len(canonical_ids) != 1:                                                           # refusal 4
            out.append(f"{p}.comparisons: exactly one comparison must carry `canonical: true` -- the one owning the "
                       f"canonical sampled instrument (the panels the yaml preset row runs); got "
                       f"{canonical_ids if canonical_ids else 'none'}")
        else:
            cv = comps[canonical_ids[0]]["value"]
            if vsg._is_number(cv) and not _num_eq(cv, preset_value):
                out.append(f"{p}: the yaml preset row runs {k} {_fmt(preset_value)} but the canonical comparison "
                           f"{canonical_ids[0]!r} decodes at {cv}: the row must equal the value of the comparison owning "
                           f"the canonical sampled instrument, so the row stays what the paper's headline arms actually "
                           f"run -- fix the row or the declaration")
    return out


def check_row(row: Row, pin: Mapping[str, Any]) -> Outcome:
    """ONE row against the pin. The one code path the live test and the synthetic
    tests share, so the teeth proven on synthetic rows are the teeth the real rows meet."""
    considered = tuple(pin["sampler_keys_considered"])
    where = f"{row.name} ({row.model} @ {row.revision[:8]})"
    entry = pin["models"].get(row.model)
    if entry is None:
        return Outcome(failures=[
            f"{where}: {PIN_REL} pins no shipped {FILE} for {row.model!r} -- its state is UNKNOWN, and an unknown shipped "
            f"file is not a model that ships none. Read the snapshot at {row.revision[:8]} and pin it: state present with "
            f"exactly the sampler keys the file states + sha256 + read_from, or state ships_none with the evidence "
            f"(`python3 {VERIFIER_REL} --cache-root $HF_HOME` on a cluster, or `--hub`, reads it); "
            f"only then can {ROUTE_PRESET} be compared with {ROUTE_SHIPPED} for this row"])
    if entry["revision"] != row.revision:
        return Outcome(failures=[
            f"{where}: {PIN_REL} pins the shipped {FILE} of revision {entry['revision'][:8]}, but the yaml row serves "
            f"revision {row.revision[:8]} -- a file read from one snapshot does not vouch for another. Re-read {FILE} at "
            f"{row.revision[:8]} and re-pin (the verifier reads it), then re-check the row's preset and declarations"])
    state = entry["state"]
    if state == "unknown":
        return Outcome(failures=[
            f"{where}: {PIN_REL} pins state UNKNOWN for {row.model!r} @ {row.revision[:8]} ({entry.get('note') or 'no note'}): "
            f"nobody has read the snapshot, so whether {ROUTE_PRESET} agrees with {ROUTE_SHIPPED} is not known -- and "
            f"unknown is NOT 'ships none'. Verify it where the snapshot is (`python3 {VERIFIER_REL} --cache-root "
            f"$HF_HOME` on a cluster, or `--hub`) and pin present or ships_none"])
    if state == "ships_none":
        evidence = list(entry.get("evidence") or [])
        failures: List[str] = []
        if row.departure is not None:
            failures.append(
                f"{where}: declares deliberate_departure on "
                f"{sorted(row.departure) if isinstance(row.departure, Mapping) else row.departure!r} but the model ships "
                f"no {FILE} at this revision (pinned ships_none; evidence: {evidence[0] if evidence else 'none'}) -- "
                f"there is no shipped value to depart from; drop the declaration, or re-pin if the model now ships one")
        if row.shipped_block is not None:
            failures.append(
                f"{where}: records shipped_generation_config {row.shipped_block!r} but the model ships no {FILE} at this "
                f"revision (pinned ships_none; evidence: {evidence[0] if evidence else 'none'}) -- one of the two statements "
                f"is wrong; re-read the snapshot")
        if failures:
            return Outcome(failures=failures)
        return Outcome(skip=(
            f"{where}: the model ships no {FILE} at this revision (pinned ships_none; evidence: {'; '.join(evidence)}) -- "
            f"{ROUTE_SHIPPED} has no values to compare {ROUTE_PRESET} against. The preset's source is the model card / a "
            f"vendor row, whose agreement with its other statement test_vendor_sampling_row_agreement checks"))
    # present
    shipped = dict(entry["sampler_keys"])
    failures = departure_failures(where, row.name, row.departure, considered, row.preset, shipped)
    if row.departure is not None and not isinstance(row.departure, Mapping):
        return Outcome(failures=failures)
    departure: Mapping[str, Any] = row.departure or {}
    if row.shipped_block is not None:
        blk = row.shipped_block
        if (not isinstance(blk, Mapping) or sorted(blk) != sorted(shipped)
                or any(not _num_eq(blk[k], shipped[k]) for k in shipped)):
            failures.append(
                f"{where}: the row's shipped_generation_config says {blk!r} but the pinned {FILE} states {shipped} "
                f"({PIN_REL}, sha256 {entry['sha256'][:12]}...) -- two statements of the shipped file disagree; re-read "
                f"the snapshot (the verifier) and fix whichever is wrong")
    for key in considered:
        decl = departure.get(key)
        if key not in shipped:
            if decl is not None:
                failures.append(
                    f"{where}: {row.name}.deliberate_departure.{key} declares a departure but the shipped {FILE} states no "
                    f"{key} (it states {sorted(shipped)}) -- there is no shipped value to depart from. A key only the "
                    f"preset states is not a disagreement with the file; drop the declaration")
            continue
        preset_value, shipped_value = row.preset.get(key), shipped[key]
        differs = not _num_eq(preset_value, shipped_value)
        if differs and decl is None:
            failures.append(_undeclared(where, row, key, preset_value, shipped_value))
        elif (differs and isinstance(decl, Mapping) and vsg._is_number(decl.get("shipped"))
              and not _num_eq(decl["shipped"], shipped_value)):
            failures.append(
                f"{where}: {row.name}.deliberate_departure.{key}.shipped says {decl['shipped']} but the pinned {FILE} "
                f"states {key} {shipped_value} ({PIN_REL}) -- the declaration does not describe the file it departs from "
                f"(the preset runs {_fmt(preset_value)}); re-read the snapshot and fix the declaration or the pin")
        elif not differs and decl is not None:
            failures.append(
                f"{where}: {row.name}.deliberate_departure.{key} declares a departure (shipped "
                f"{decl.get('shipped') if isinstance(decl, Mapping) else decl!r}) but the preset's {key} "
                f"{_fmt(preset_value)} EQUALS the shipped {shipped_value}: the departure no longer exists, so the "
                f"declaration is stale and would excuse a future drift; remove it")
    return Outcome(failures=failures)


# ============================================================ live: this tree's files
LIVE_DOC = yaml.safe_load(YAML_PATH.read_text(encoding="utf-8"))
LIVE_ROWS = yaml_rows(LIVE_DOC)


@pytest.mark.parametrize("row", LIVE_ROWS, ids=[r.name for r in LIVE_ROWS])
def test_live_every_preset_matches_its_pinned_shipped_file_or_declares_the_departure(row: Row):
    """One test per preset the drivers can send. A model that ships no file SKIPS by
    name; a model with no pin, or a pinned unknown, FAILS by name; a difference the
    row does not declare FAILS naming model, key, both values and both routes."""
    pin = vsg.load_pin(PIN_PATH)
    outcome = check_row(row, pin)
    if outcome.skip:
        pytest.skip(outcome.skip)
    assert not outcome.failures, "\n".join(outcome.failures)


def test_live_pin_table_is_the_drivers_send_set_and_names_no_stale_model():
    pin = vsg.load_pin(PIN_PATH)
    assert tuple(pin["sampler_keys_considered"]) == tuple(SAMPLER_KEYS), (
        f"{PIN_REL} considers {pin['sampler_keys_considered']} but the drivers send sampled_decode.SAMPLER_KEYS "
        f"{list(SAMPLER_KEYS)}: a sampler key the drivers send and the pin ignores could disagree with the shipped "
        "file unseen; make the two equal")
    yaml_models = {r.model for r in LIVE_ROWS}
    stale = sorted(set(pin["models"]) - yaml_models)
    assert not stale, (f"{PIN_REL} pins {stale} but {YAML_REL} states no preset for them: a pin nothing reads guards "
                       "nothing; drop it, or restore the row")
    # the other direction -- a yaml model with no pin -- is a per-row NAMED failure above


def test_live_nemotron_declares_the_finding_per_comparison():
    """The one departure on main today: top_p 1.0 runs; the file
    says 0.95; the panels (canonical) decode at 1.0 with the paper arms and the re-run
    AlphaSteer baseline; the math lane's legs are their own comparison at the shipped 0.95."""
    n = LIVE_DOC["presets"]["nemotron3nano4b"]
    assert set(n["deliberate_departure"]) == {"top_p"}
    d = n["deliberate_departure"]["top_p"]
    assert d["shipped"] == 0.95 and n["preset"]["top_p"] == 1.0
    assert n["shipped_generation_config"] == {"temperature": 1.0, "top_p": 0.95}
    assert "NVIDIA" in d["reason"] and "2026-09-11" in d["reason"]
    comps = d["comparisons"]
    canonical = [cid for cid, c in comps.items() if c.get("canonical") is True]
    assert canonical == ["single_shot_and_overrefusal_panels"] and comps[canonical[0]]["value"] == 1.0
    assert {"none", "prompted__safety_refusal", "alphasteer"} <= set(comps[canonical[0]]["arms"])
    assert comps["math_evals"]["value"] == 0.95 and comps["math_evals"]["arms"] == ["math500", "gsm8k"]
    assert {c["value"] for c in comps.values()} <= {1.0, 0.95}
    entry = vsg.load_pin(PIN_PATH)["models"][n["model"]]
    assert entry["revision"] == n["revision"] and entry["state"] == "present"
    assert entry["sampler_keys"] == {"temperature": 1.0, "top_p": 0.95}
    assert not check_row(next(r for r in LIVE_ROWS if r.name == "presets.nemotron3nano4b"), vsg.load_pin(PIN_PATH)).failures


def _verdict_or_skip(results, source: str) -> None:
    contradictions = [m for _, v, m in results if v == vsg.CONTRADICTION]
    assert not contradictions, (f"the pin table {PIN_REL} CONTRADICTS {source} -- the pin is stale or wrong, and every "
                                f"preset-vs-shipped verdict above rests on it:\n  " + "\n  ".join(contradictions))
    unverified = [m for _, v, m in results if v == vsg.UNVERIFIED]
    if unverified:
        pytest.skip(f"{len(unverified)} of {len(results)} pinned rows could not be verified against {source} "
                    f"(absent-because-unavailable, NOT ships_none); no contradiction among the rest:\n  "
                    + "\n  ".join(unverified)
                    + f"\n  run `python3 {VERIFIER_REL} --cache-root $HF_HOME` on a cluster "
                      "(its Hugging Face cache holds every snapshot), or `--hub` anywhere with network")


def test_live_pin_against_this_machines_hf_cache():
    """The second instrument, run against whatever snapshots THIS machine holds. Per
    row: a contradiction FAILS (the pin is wrong), a verified row passes, an unknown
    row (no snapshot here) is named in a skip -- never read as 'ships none'."""
    pin = vsg.load_pin(PIN_PATH)
    hub_dir = vsg.hub_dir_of(vsg.default_cache_root())
    _verdict_or_skip(vsg.verify(pin, hub_dir=hub_dir), f"the HF cache {hub_dir}")


@pytest.mark.skipif(not os.environ.get("SISTEER_VERIFY_SHIPPED_HUB"),
                    reason="network: set SISTEER_VERIFY_SHIPPED_HUB=1 to verify the pin against the hub")
def test_live_pin_against_the_hub():
    pin = vsg.load_pin(PIN_PATH)
    _verdict_or_skip(vsg.verify(pin, hub=True, token=os.environ.get("HF_TOKEN") or None), "the hub")


# ============================================================ synthetic: the check has teeth
A_REV, B_REV, C_REV, D_REV = "a" * 40, "b" * 40, "c" * 40, "d" * 40


def _doc() -> Dict[str, Any]:
    """A synthetic data file in the live shape: a model that agrees with its file
    (alpha), one that ships none (beta, with a per-readout override), one with a
    declared shared departure (gamma). Neutral names, placeholder shas."""
    return {
        "arm": "vendor_recommended", "n_trials": 2, "seed_base": 1, "applies_to": ["static_families"],
        "presets": {
            "alpha": {"model": "org/alpha", "revision": A_REV,
                      "preset": {"temperature": 1.0, "top_p": 0.95, "top_k": 64},
                      "source_kind": "generation_config", "source": "synthetic"},
            "beta": {"model": "org/beta", "revision": B_REV,
                     "preset": {"temperature": 0.7, "top_p": 0.8, "presence_penalty": 1.5},
                     "source_kind": "model_card", "source": "synthetic"},
            "gamma": {"model": "org/gamma", "revision": C_REV,
                      "preset": {"temperature": 1.0, "top_p": 1.0},
                      "shipped_generation_config": {"temperature": 1.0, "top_p": 0.95},
                      "deliberate_departure": {"top_p": {"shipped": 0.95, "reason": "the vendor card's general row",
                                                         "comparisons": {
                                                             "panels": {"canonical": True, "value": 1.0,
                                                                        "arms": ["none", "prompted", "lora", "alphasteer"]},
                                                             "math_evals": {"value": 0.95, "arms": ["math500", "gsm8k"]}}}},
                      "source_kind": "vendor_recommendation", "source": "synthetic"},
        },
        "readouts": {"gsm8k": {"cap": 8192, "thinking": "on",
                               "presets": {"beta": {"preset": {"temperature": 1.0, "top_p": 0.95}, "source": "synthetic"}}},
                     "mmlu": {"instrument": "letter_logprob_argmax"}},
    }


def _pin() -> Dict[str, Any]:
    return vsg.validate_pin({
        "_doc": ["synthetic"], "sampler_keys_considered": list(SAMPLER_KEYS),
        "models": {
            "org/alpha": {"revision": A_REV, "state": "present",
                          "sampler_keys": {"temperature": 1.0, "top_p": 0.95, "top_k": 64}, "do_sample": True,
                          "sha256": "1" * 64, "etag": None, "read_from": ["synthetic cache"]},
            "org/beta": {"revision": B_REV, "state": "ships_none", "evidence": ["synthetic .no_exist marker"]},
            "org/gamma": {"revision": C_REV, "state": "present", "sampler_keys": {"temperature": 1.0, "top_p": 0.95},
                          "do_sample": None, "sha256": "2" * 64, "etag": None, "read_from": ["synthetic cache"]},
        }}, "synthetic pin")


def _check(doc: Mapping[str, Any], pin: Mapping[str, Any], name: str) -> Outcome:
    rows = {r.name: r for r in yaml_rows(doc, "synthetic.yaml")}
    return check_row(rows[name], pin)


def _only_failure(doc, pin, name) -> str:
    out = _check(doc, pin, name)
    assert out.skip is None and len(out.failures) == 1, out
    return out.failures[0]


def test_synthetic_baseline_agrees_skips_by_name_and_accepts_the_declared_departure():
    doc, pin = _doc(), _pin()
    assert _check(doc, pin, "presets.alpha") == Outcome()
    assert _check(doc, pin, "presets.gamma") == Outcome()
    for name in ("presets.beta", "readouts.gsm8k.presets.beta"):
        out = _check(doc, pin, name)
        assert not out.failures and out.skip
        assert "org/beta @ bbbbbbbb" in out.skip and f"ships no {FILE}" in out.skip and "synthetic .no_exist marker" in out.skip
        assert "SERVER-DEFAULT route" in out.skip and "SAMPLED route" in out.skip


def test_a_changed_yaml_value_fails_naming_model_key_both_values_and_both_routes():
    doc = _doc()
    doc["presets"]["alpha"]["preset"]["top_p"] = 0.9
    msg = _only_failure(doc, _pin(), "presets.alpha")
    assert msg.startswith("presets.alpha (org/alpha @ aaaaaaaa): top_p differs and the row declares no departure")
    assert "request-body preset" in msg and "SAMPLED route" in msg and "says 0.9;" in msg
    assert f"shipped {FILE}" in msg and "SERVER-DEFAULT route" in msg and "says 0.95." in msg
    assert "deliberate_departure.top_p" in msg and "canonical: true" in msg and "AlphaSteer ladder" in msg


def test_a_changed_pinned_value_fails_the_same_way():
    pin = _pin()
    pin["models"]["org/alpha"]["sampler_keys"]["top_k"] = 40
    msg = _only_failure(_doc(), pin, "presets.alpha")
    assert "top_k differs" in msg and "says 64;" in msg and "says 40." in msg and "org/alpha @ aaaaaaaa" in msg


def test_a_new_row_with_no_pin_is_unknown_and_fails_until_pinned():
    doc = _doc()
    doc["presets"]["delta"] = {"model": "org/delta", "revision": D_REV, "preset": {"temperature": 0.6, "top_p": 0.9},
                               "source_kind": "generation_config", "source": "synthetic"}
    msg = _only_failure(doc, _pin(), "presets.delta")
    assert msg.startswith("presets.delta (org/delta @ dddddddd)")
    assert "pins no shipped" in msg and "UNKNOWN" in msg and "not a model that ships none" in msg
    assert VERIFIER_REL in msg and "ships_none with the evidence" in msg


def test_a_new_row_whose_shipped_file_disagrees_fails_until_the_departure_is_declared():
    doc, pin = _doc(), _pin()
    doc["presets"]["delta"] = {"model": "org/delta", "revision": D_REV, "preset": {"temperature": 0.6, "top_p": 1.0},
                               "source_kind": "generation_config", "source": "synthetic"}
    pin["models"]["org/delta"] = {"revision": D_REV, "state": "present", "sampler_keys": {"temperature": 0.6, "top_p": 0.9},
                                  "do_sample": True, "sha256": "3" * 64, "etag": None, "read_from": ["synthetic cache"]}
    msg = _only_failure(doc, pin, "presets.delta")
    assert "org/delta @ dddddddd): top_p differs and the row declares no departure" in msg
    assert "says 1.0;" in msg and "says 0.9." in msg
    # declaring it (shared by every compared arm) is what makes the row pass
    doc["presets"]["delta"]["deliberate_departure"] = {"top_p": {"shipped": 0.9, "reason": "vendor's chat row",
                                                                 "comparisons": {"panels": {"canonical": True, "value": 1.0,
                                                                                            "arms": ["none", "lora"]}}}}
    assert _check(doc, pin, "presets.delta") == Outcome()


def test_removing_a_needed_declaration_fails():
    doc = _doc()
    del doc["presets"]["gamma"]["deliberate_departure"]
    msg = _only_failure(doc, _pin(), "presets.gamma")
    assert "presets.gamma (org/gamma @ cccccccc): top_p differs and the row declares no departure" in msg
    assert "says 1.0;" in msg and "says 0.95." in msg


def test_a_stale_declaration_fails():
    doc = _doc()
    doc["presets"]["gamma"]["preset"]["top_p"] = 0.95          # the preset no longer departs
    out = _check(doc, _pin(), "presets.gamma")
    stale = [f for f in out.failures if "EQUALS the shipped 0.95" in f]
    assert stale and "deliberate_departure.top_p declares a departure" in stale[0] and "stale" in stale[0], out
    # and the comparisons that still say 1.0 now name a value that is neither source (refusal 3) and a canonical
    # comparison the row no longer equals (refusal 4): three true statements about one stale field
    assert any("a third sampler" in f for f in out.failures) and any("the canonical comparison 'panels'" in f for f in out.failures), out


def test_a_declaration_whose_shipped_value_is_not_the_files_fails():
    doc = _doc()
    doc["presets"]["gamma"]["deliberate_departure"]["top_p"]["shipped"] = 0.9
    msg = _only_failure(doc, _pin(), "presets.gamma")
    assert "deliberate_departure.top_p.shipped says 0.9" in msg and "states top_p 0.95" in msg and "the preset runs 1.0" in msg


def _gamma_comps(doc):
    return doc["presets"]["gamma"]["deliberate_departure"]["top_p"]["comparisons"]


def test_different_comparisons_at_different_declared_values_pass():
    """The rescope's point: the panels at the preset's 1.0 and the math legs at the
    shipped 0.95 are two instruments by design, each declared at its own value."""
    doc = _doc()
    assert _check(doc, _pin(), "presets.gamma") == Outcome()
    _gamma_comps(doc)["bon_climb"] = {"value": 0.95, "arms": ["bon"]}       # a third comparison at the shipped value
    assert _check(doc, _pin(), "presets.gamma") == Outcome()


def test_refusal_1_two_arms_in_one_comparison_at_different_values():
    doc = _doc()
    _gamma_comps(doc)["panels"]["arms"] = [{"none": 0.95}, "prompted", "lora", "alphasteer"]
    msg = _only_failure(doc, _pin(), "presets.gamma")
    assert "deliberate_departure.top_p.comparisons.panels: arms inside ONE comparison decode at different top_p" in msg
    assert "['none'] at 0.95; ['prompted', 'lora', 'alphasteer'] at 1.0" in msg
    assert "two instruments in one panel" in msg and "a 0.95 baseline beside 1.0 objects in one panel" in msg


def test_refusal_2_an_arm_declared_in_two_comparisons():
    doc = _doc()
    _gamma_comps(doc)["panels"]["arms"].append("gsm8k")
    msg = _only_failure(doc, _pin(), "presets.gamma")
    assert "arm 'gsm8k' is declared in 2 comparisons ['panels', 'math_evals']" in msg and "EXACTLY once" in msg


def test_refusal_2_an_arm_in_no_comparison_is_not_checkable_here():
    """The other half of refusal 2 needs an authoritative list of the arms that decode
    this model, and the repo has none (module doc). So an omitted consumer PASSES here,
    by decision, not by accident: this test pins that the gap is known, so that the day
    a registry lands the assertion below is the one to flip."""
    doc = _doc()
    _gamma_comps(doc)["panels"]["arms"].remove("alphasteer")
    assert _check(doc, _pin(), "presets.gamma") == Outcome()


def test_refusal_3_a_comparison_value_that_is_neither_source():
    doc = _doc()
    _gamma_comps(doc)["math_evals"]["value"] = 0.9
    msg = _only_failure(doc, _pin(), "presets.gamma")
    assert "comparisons.math_evals.value 0.9 is neither the preset's top_p (1.0) nor the shipped file's (0.95): a third sampler" in msg


def test_refusal_4_the_preset_row_equals_the_canonical_comparison():
    doc = _doc()
    del _gamma_comps(doc)["panels"]["canonical"]
    msg = _only_failure(doc, _pin(), "presets.gamma")
    assert "exactly one comparison must carry `canonical: true`" in msg and "got none" in msg
    doc = _doc()
    _gamma_comps(doc)["math_evals"]["canonical"] = True
    msg = _only_failure(doc, _pin(), "presets.gamma")
    assert "got ['panels', 'math_evals']" in msg
    doc = _doc()
    del _gamma_comps(doc)["panels"]["canonical"]
    _gamma_comps(doc)["math_evals"]["canonical"] = True
    msg = _only_failure(doc, _pin(), "presets.gamma")
    assert "the yaml preset row runs top_p 1.0 but the canonical comparison 'math_evals' decodes at 0.95" in msg
    assert "the paper's headline arms actually run" in msg


def test_the_comparisons_field_is_required_and_strict():
    doc = _doc()
    del doc["presets"]["gamma"]["deliberate_departure"]["top_p"]["comparisons"]
    assert "must carry exactly ['comparisons', 'reason', 'shipped']: missing ['comparisons']" in _only_failure(doc, _pin(), "presets.gamma")
    doc = _doc()
    doc["presets"]["gamma"]["deliberate_departure"]["top_p"]["comparisons"] = {}
    assert "comparisons must be a non-empty mapping" in _only_failure(doc, _pin(), "presets.gamma")
    doc = _doc()
    _gamma_comps(doc)["panels"] = {"canonical": True, "value": 1.0, "arm": ["none"]}
    assert "missing ['arms'], unexpected ['arm']" in _only_failure(doc, _pin(), "presets.gamma")
    doc = _doc()
    _gamma_comps(doc)["panels"]["arms"] = []
    assert "arms must be a non-empty list" in _only_failure(doc, _pin(), "presets.gamma")
    doc = _doc()
    _gamma_comps(doc)["panels"]["arms"] = ["none", {"lora": "one"}]
    assert "entry {'lora': 'one'} is neither an arm name nor" in _only_failure(doc, _pin(), "presets.gamma")
    doc = _doc()
    _gamma_comps(doc)["panels"]["arms"] = ["none", "none", "lora"]
    assert "arms names 'none' twice" in _only_failure(doc, _pin(), "presets.gamma")
    doc = _doc()
    _gamma_comps(doc)["panels"]["value"] = "1.0"
    assert "value must be the number every arm in the comparison decodes at" in _only_failure(doc, _pin(), "presets.gamma")
    doc = _doc()
    doc["presets"]["gamma"]["deliberate_departure"]["top_p"]["reason"] = "  "
    assert "reason must be a non-empty string" in _only_failure(doc, _pin(), "presets.gamma")
    doc = _doc()
    doc["presets"]["gamma"]["deliberate_departure"] = "top_p 0.95 is the shipped value"   # prose, not the field
    assert "must be a non-empty mapping" in _only_failure(doc, _pin(), "presets.gamma")


def test_a_departure_on_a_key_the_file_does_not_state_fails():
    doc = _doc()
    doc["presets"]["alpha"]["preset"]["min_p"] = 0.05
    doc["presets"]["alpha"]["deliberate_departure"] = {"min_p": {"shipped": 0.0, "reason": "x",
                                                                 "comparisons": {"panels": {"canonical": True, "value": 0.05,
                                                                                            "arms": ["none"]}}}}
    msg = _only_failure(doc, _pin(), "presets.alpha")
    assert "declares a departure but the shipped generation_config.json states no min_p" in msg
    assert "['temperature', 'top_k', 'top_p']" in msg


def test_a_key_the_file_states_that_the_preset_omits_is_a_difference():
    doc = _doc()
    del doc["presets"]["alpha"]["preset"]["top_k"]
    msg = _only_failure(doc, _pin(), "presets.alpha")
    assert "top_k differs" in msg and "says no value (the key is absent);" in msg and "says 64." in msg


def test_a_preset_only_key_the_file_is_silent_on_is_not_a_difference():
    doc = _doc()
    doc["presets"]["alpha"]["preset"]["repetition_penalty"] = 1.1     # the card's term; the file states none
    assert _check(doc, _pin(), "presets.alpha") == Outcome()


def test_the_rows_shipped_block_must_match_the_pinned_file():
    doc = _doc()
    doc["presets"]["gamma"]["shipped_generation_config"] = {"temperature": 1.0, "top_p": 0.9}
    msg = _only_failure(doc, _pin(), "presets.gamma")
    assert "shipped_generation_config says {'temperature': 1.0, 'top_p': 0.9}" in msg
    assert "states {'temperature': 1.0, 'top_p': 0.95}" in msg


def test_a_revision_mismatch_between_yaml_and_pin_fails():
    pin = _pin()
    pin["models"]["org/alpha"]["revision"] = D_REV
    msg = _only_failure(_doc(), pin, "presets.alpha")
    assert "pins the shipped generation_config.json of revision dddddddd" in msg and "serves revision aaaaaaaa" in msg


def test_a_pinned_unknown_fails_by_name_and_is_not_ships_none():
    pin = _pin()
    pin["models"]["org/alpha"] = {"revision": A_REV, "state": "unknown", "note": "snapshot not fetched here"}
    vsg.validate_pin(pin, "synthetic pin")
    msg = _only_failure(_doc(), pin, "presets.alpha")
    assert "pins state UNKNOWN for 'org/alpha' @ aaaaaaaa (snapshot not fetched here)" in msg
    assert "unknown is NOT 'ships none'" in msg and VERIFIER_REL in msg


def test_a_declaration_or_shipped_block_on_a_model_that_ships_none_fails():
    doc = _doc()
    doc["presets"]["beta"]["deliberate_departure"] = {"top_p": {"shipped": 0.9, "reason": "x",
                                                                "comparisons": {"panels": {"canonical": True, "value": 0.8,
                                                                                           "arms": ["none"]}}}}
    doc["presets"]["beta"]["shipped_generation_config"] = {"temperature": 0.7}
    out = _check(doc, _pin(), "presets.beta")
    assert out.skip is None and len(out.failures) == 2
    assert "declares deliberate_departure on ['top_p'] but the model ships no" in out.failures[0]
    assert "records shipped_generation_config" in out.failures[1] and "ships no" in out.failures[1]


def test_a_per_readout_override_is_checked_and_can_declare_its_own_departure():
    doc = _doc()
    doc["readouts"]["gsm8k"]["presets"]["alpha"] = {"preset": {"temperature": 1.0, "top_p": 0.9, "top_k": 64},
                                                    "source": "the vendor's thinking row"}
    msg = _only_failure(doc, _pin(), "readouts.gsm8k.presets.alpha")
    assert msg.startswith("readouts.gsm8k.presets.alpha (org/alpha @ aaaaaaaa): top_p differs")
    assert "readouts.gsm8k.presets.alpha.deliberate_departure.top_p" in msg
    doc["readouts"]["gsm8k"]["presets"]["alpha"]["deliberate_departure"] = {
        "top_p": {"shipped": 0.95, "reason": "the vendor's thinking row",
                  "comparisons": {"gsm8k_panel": {"canonical": True, "value": 0.9, "arms": ["none", "lora"]}}}}
    assert _check(doc, _pin(), "readouts.gsm8k.presets.alpha") == Outcome()
    assert _check(doc, _pin(), "presets.alpha") == Outcome()      # the top-level row is untouched by the override


def test_yaml_shapes_the_check_cannot_follow_refuse():
    doc = _doc()
    doc["presets"]["alpha2"] = dict(doc["presets"]["alpha"])          # an IDENTICAL twin (the worked example's shape) is one row, under the first key
    assert [r.key for r in yaml_rows(doc, "synthetic.yaml") if r.model == "org/alpha" and r.name.startswith("presets.")] == ["alpha"]
    doc["presets"]["alpha2"] = {**dict(doc["presets"]["alpha"]), "revision": "f" * 40}   # a twin that differs is the ambiguity
    with pytest.raises(YamlShape, match="states 'org/alpha' under both presets.alpha and presets.alpha2"):
        yaml_rows(doc, "synthetic.yaml")
    doc = _doc()
    del doc["presets"]["alpha"]["revision"]
    with pytest.raises(YamlShape, match="names no `revision`"):
        yaml_rows(doc, "synthetic.yaml")
    doc = _doc()
    doc["readouts"]["gsm8k"]["presets"]["zeta"] = {"preset": {"temperature": 1.0}}
    with pytest.raises(YamlShape, match="overrides a key with no top-level preset row"):
        yaml_rows(doc, "synthetic.yaml")


def test_the_pin_schema_refuses_a_collapsed_or_malformed_row():
    for mutate, needle in (
        (lambda p: p["models"]["org/alpha"].__setitem__("state", "absent"), "must be one of"),
        (lambda p: p["models"]["org/alpha"]["sampler_keys"].__setitem__("best_of", 2), "outside sampler_keys_considered"),
        (lambda p: p["models"]["org/alpha"]["sampler_keys"].__setitem__("top_k", True), "are not numbers"),
        (lambda p: p["models"]["org/alpha"].__setitem__("revision", "ee0ef602"), "40-hex"),
        (lambda p: p["models"]["org/alpha"].pop("sha256"), "needs the file's `sha256`"),
        (lambda p: p["models"]["org/alpha"].__setitem__("read_from", []), "non-empty `read_from`"),
        (lambda p: p["models"]["org/beta"].__setitem__("evidence", []), "non-empty `evidence`"),
        (lambda p: p["models"]["org/beta"].__setitem__("sampler_keys", {"top_p": 0.9}), "there is no file to describe"),
        (lambda p: p.__setitem__("sampler_keys_considered", ["temperature", "temperature"]), "distinct key names"),
    ):
        pin = _pin()
        mutate(pin)
        with pytest.raises(vsg.PinMalformed, match=needle):
            vsg.validate_pin(pin, "synthetic pin")


# ------------------------------------------------------------ the verifier, on a synthetic cache
def _cache(tmp_path: Path, pin: Dict[str, Any]) -> Path:
    """An HF cache holding: alpha's file (a blob behind a snapshot symlink, as the hub
    client lays it out), beta's .no_exist marker, gamma's PARTIAL snapshot (another
    file, no generation_config.json, no marker)."""
    hub = tmp_path / "hf" / "hub"
    alpha = hub / "models--org--alpha"
    data = json.dumps({"bos_token_id": 2, "do_sample": True, "temperature": 1.0, "top_k": 64, "top_p": 0.95,
                       "transformers_version": "5.5.0"}).encode()
    (alpha / "blobs").mkdir(parents=True)
    (alpha / "blobs" / "e605bb45").write_bytes(data)
    (alpha / "snapshots" / A_REV).mkdir(parents=True)
    os.symlink("../../blobs/e605bb45", alpha / "snapshots" / A_REV / FILE)
    pin["models"]["org/alpha"]["sha256"] = vsg.hashlib.sha256(data).hexdigest()
    pin["models"]["org/alpha"]["etag"] = "e605bb45"
    beta = hub / "models--org--beta"
    (beta / ".no_exist" / B_REV).mkdir(parents=True)
    (beta / ".no_exist" / B_REV / FILE).write_bytes(b"")
    (beta / "snapshots" / B_REV).mkdir(parents=True)
    (beta / "snapshots" / B_REV / "config.json").write_text("{}")
    gamma = hub / "models--org--gamma"
    (gamma / "snapshots" / C_REV).mkdir(parents=True)
    (gamma / "snapshots" / C_REV / "tokenizer.json").write_text("{}")
    return tmp_path / "hf"


def test_verifier_reads_a_cache_as_a_tri_state(tmp_path):
    pin = _pin()
    root = _cache(tmp_path, pin)
    hub = vsg.hub_dir_of(root)
    assert hub == root / "hub"
    results = {m: (v, msg) for m, v, msg in vsg.verify(pin, hub_dir=hub)}
    assert results["org/alpha"][0] == vsg.VERIFIED and "as pinned" in results["org/alpha"][1]
    assert results["org/beta"][0] == vsg.VERIFIED and "ships no generation_config.json" in results["org/beta"][1]
    assert results["org/gamma"][0] == vsg.UNVERIFIED
    assert "PARTIAL snapshot (1 files)" in results["org/gamma"][1] and "NOT ships_none" in results["org/gamma"][1]
    assert vsg.overall(list(vsg.verify(pin, hub_dir=hub))) == vsg.UNVERIFIED
    # a model the cache never saw at all is unknown, not ships_none
    obs = vsg.observe_cache("org/delta", D_REV, hub, SAMPLER_KEYS)
    assert obs.state == "unknown" and "does not exist in this cache" in obs.where


def test_verifier_contradictions_name_what_changed(tmp_path):
    pin = _pin()
    root = _cache(tmp_path, pin)
    hub = vsg.hub_dir_of(root)
    alpha_file = hub / "models--org--alpha" / "snapshots" / A_REV / FILE
    blob = hub / "models--org--alpha" / "blobs" / "e605bb45"
    # the pinned value is stale: the file says another top_p
    blob.write_bytes(json.dumps({"do_sample": True, "temperature": 1.0, "top_k": 64, "top_p": 0.9}).encode())
    v, msg = [(v, m) for mo, v, m in vsg.verify(pin, hub_dir=hub) if mo == "org/alpha"][0]
    assert v == vsg.CONTRADICTION and "differ from the file's" in msg and "['top_p']" in msg and "org/alpha @ aaaaaaaa" in msg
    # same sampler values, different bytes: the provenance no longer describes this file
    blob.write_bytes(json.dumps({"do_sample": True, "temperature": 1.0, "top_k": 64, "top_p": 0.95, "extra": 1}).encode())
    v, msg = [(v, m) for mo, v, m in vsg.verify(pin, hub_dir=hub) if mo == "org/alpha"][0]
    assert v == vsg.CONTRADICTION and "sha256" in msg and "is not the pinned" in msg
    # the pin says ships_none but a file is there
    (hub / "models--org--beta" / "snapshots" / B_REV / FILE).write_text(json.dumps({"temperature": 0.7}))
    v, msg = [(v, m) for mo, v, m in vsg.verify(pin, hub_dir=hub) if mo == "org/beta"][0]
    assert v == vsg.CONTRADICTION and "the pin says ships_none but the source says present" in msg
    assert "{'temperature': 0.7}" in msg
    # the pin says present but the hub recorded a 404
    alpha_file.unlink()
    (hub / "models--org--alpha" / ".no_exist" / A_REV).mkdir(parents=True)
    (hub / "models--org--alpha" / ".no_exist" / A_REV / FILE).write_bytes(b"")
    v, msg = [(v, m) for mo, v, m in vsg.verify(pin, hub_dir=hub) if mo == "org/alpha"][0]
    assert v == vsg.CONTRADICTION and "the pin says present but the source says ships_none" in msg
    # a present file that is not a generation config at all
    (hub / "models--org--gamma" / "snapshots" / C_REV / FILE).write_bytes(b"not json")
    v, msg = [(v, m) for mo, v, m in vsg.verify(pin, hub_dir=hub) if mo == "org/gamma"][0]
    assert v == vsg.CONTRADICTION and "not readable as a generation config" in msg


def test_verifier_cli_exit_codes_follow_the_verdict(tmp_path):
    pin = _pin()
    root = _cache(tmp_path, pin)
    pin_path = tmp_path / "pin.json"
    pin_path.write_text(json.dumps(pin))
    run = lambda *extra: subprocess.run([sys.executable, str(VERIFIER), "--pin", str(pin_path), "--cache-root", str(root), *extra],  # noqa: E731
                                        capture_output=True, text=True)
    r = run()
    assert r.returncode == 2, r
    assert "UNVERIFIED: org/gamma @ cccccccc" in r.stdout and "-- UNVERIFIED: 2 verified, 0 contradicted, 1 unverified of 3" in r.stdout
    assert "NOT ships_none" in r.stdout and "--cache-root $HF_HOME" in r.stdout
    r = run("--model", "org/alpha", "--model", "org/beta")
    assert r.returncode == 0 and "-- VERIFIED: 2 verified, 0 contradicted, 0 unverified of 2" in r.stdout, r
    (root / "hub" / "models--org--alpha" / "blobs" / "e605bb45").write_bytes(b'{"temperature": 0.5}')
    r = run("--model", "org/alpha")
    assert r.returncode == 1 and "CONTRADICTION: org/alpha @ aaaaaaaa" in r.stdout, r
    r = run("--model", "org/nobody")
    assert r.returncode == 1 and "not pinned" in r.stderr
    pin["models"]["org/alpha"]["state"] = "maybe"
    pin_path.write_text(json.dumps(pin))
    r = run()
    assert r.returncode == 1 and "PIN MALFORMED" in r.stderr


def test_verifier_maps_hub_answers_to_the_tri_state(monkeypatch):
    """The hub route with canned answers: listed + 200 is present (with the body's
    sha256), not listed / 404 is ships_none, 401/403 and a network failure are
    unknown -- never ships_none."""
    body = json.dumps({"temperature": 0.6, "top_p": 0.9, "do_sample": True}).encode()
    listing_with = json.dumps({"siblings": [{"rfilename": "config.json"}, {"rfilename": FILE}]}).encode()
    listing_without = json.dumps({"siblings": [{"rfilename": "config.json"}]}).encode()

    def canned(answers):
        it = iter(answers)

        def _get(url, token, timeout):
            a = next(it)
            if isinstance(a, Exception):
                raise a
            return a
        return _get

    monkeypatch.setattr(vsg, "_hub_get", canned([(200, listing_with), (200, body)]))
    obs = vsg.observe_hub("org/x", A_REV, SAMPLER_KEYS)
    assert obs.state == "present" and obs.sampler_keys == {"temperature": 0.6, "top_p": 0.9}
    assert obs.sha256 == vsg.hashlib.sha256(body).hexdigest() and obs.do_sample is True
    monkeypatch.setattr(vsg, "_hub_get", canned([(200, listing_without)]))
    assert vsg.observe_hub("org/x", A_REV, SAMPLER_KEYS).state == "ships_none"
    monkeypatch.setattr(vsg, "_hub_get", canned([(200, listing_with), (404, b"")]))
    assert vsg.observe_hub("org/x", A_REV, SAMPLER_KEYS).state == "ships_none"
    monkeypatch.setattr(vsg, "_hub_get", canned([(401, b"")]))
    obs = vsg.observe_hub("org/x", A_REV, SAMPLER_KEYS)
    assert obs.state == "unknown" and "gated" in obs.where
    monkeypatch.setattr(vsg, "_hub_get", canned([(200, listing_with), (403, b"")]))
    obs = vsg.observe_hub("org/x", A_REV, SAMPLER_KEYS, token="t")
    assert obs.state == "unknown" and "not readable with a token" in obs.where
    monkeypatch.setattr(vsg, "_hub_get", canned([urllib.error.URLError("offline")]))
    obs = vsg.observe_hub("org/x", A_REV, SAMPLER_KEYS)
    assert obs.state == "unknown" and "network failure" in obs.where
    monkeypatch.setattr(vsg, "_hub_get", canned([(503, b"")]))
    assert vsg.observe_hub("org/x", A_REV, SAMPLER_KEYS).state == "unknown"
