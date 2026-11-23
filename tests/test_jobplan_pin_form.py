"""Job-start registration of a roster entry in the pin-referencing MAPPING form ``{pins, pin}`` (``jobplan.object_id_for`` with
``pins_root``; ``jobplan.build_plans`` passes the checkout that holds the seeds when the job OPTS IN: ``pin_form=True`` or
``SISTEER_REGISTRY_PIN_FORM=1``).

A non-production roster may name its objects through a pins file of its own (``{pins: configs/<x>_pins.json, pin: <model>.<staged id>}``),
the form the serving readers resolve (``make_bank_view.plan_pin_entry``, ``mt_serving_up._pin_form_sha``). Before this, the registration
read only string entries, so every such entry was skipped and a job whose entries were all of this form registered nothing (``emit_cli
register`` exits 2). Pinned here:

1. a pin-form entry resolves to the named pin's ``object_sha256`` through ``build_plans`` (the whole-config path), on the canonical
   multi-turn instruments of the served instruct model;
2. the narrowed-copy shape the per-cell registration uses (the config copied outside the checkout, the checkout's seeds, the roster and
   the PRODUCTION pins named, the lane pins reachable only through the roster entry, relative paths from the checkout as the job passes
   them) resolves the same object ids;
3. every malformed or unresolvable entry is SKIPPED by name, never resolved: a missing pins file, an absolute or escaping ``pins`` path,
   a ``pin`` naming another model key or not of the form ``<model>.<staged id>``, an absent entry, a short or missing sha, a mapping with
   extra keys;
4. without ``pins_root`` a mapping entry still raises ``LookupError`` (the adapters' contract), and a string entry resolves identically
   with or without it;
5. on a COMMITTED pin-form roster the registration and the serving readers name the same object for every entry;
6. the three model keys of the early-stopped control views resolve through ``legacy_spellings.yaml``; a fourth seed does not;
7. WITHOUT the opt-in every pin-form entry is skipped by name exactly as before (a job over any existing roster registers what it always
   did), the environment switch reads ``1`` / ``0`` / unset and refuses anything else by name, and the keyword wins over it.

Synthetic checkouts under tmp_path (the real seeds symlinked, invented shas); the committed files are read, never written."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
from pathlib import Path

import pytest
import yaml

from sisteer.registry import jobplan as JP
from sisteer.registry import synth

REPO = Path(__file__).resolve().parents[1]
SEEDS = REPO / "configs" / "registry"
for _p in (REPO / "scripts" / "sdfbase", REPO / "scripts" / "sdfbase" / "full22", REPO / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

LANE_ROSTER_REL = "configs/sdfbase_full22_roster_ctlstop_eb64.yaml"
LANE_PINS_REL = "configs/ctlstop_eb64_pins.json"
PROD_PINS_REL = "configs/sdfbase_full22_pins.json"
TARGET = "qwen35_4b"
VIEWS = ("ctlstop64s1", "ctlstop64s2", "ctlstop64s3")
CELLS = ("constv2ctl2-full__lora_r4_ce", "constv2ctl2-full__psr_ce_all",
         "constv2-mix-full__lora_r4_ce_minus_ctl", "constv2-mix-full__psr_ce_all_minus_ctl")
#: the canonical seeded instruments of the two multi-turn attacks on the served instruct model via the native engine
INSTRUMENT = {"crescendo": "33f80ed1d4f536e34c870fa36c2d88d90c7623ea17a9d4db178fea008c12ac85",
              "actorattack": "f8832b53b08131258d0e5c9e15a16a2888f8a977271d52dde270d52b1af00203"}
CTL_NEG1_ROSTER = REPO / "configs" / "sdfbase_full22_roster_ctl_neg1_eb64.yaml"


def fake_sha(name: str) -> str:
    return hashlib.sha256(f"synthetic object {name}".encode()).hexdigest()


def pin_id(view: str, cell: str) -> str:
    return f"{view}__{cell}"


@pytest.fixture(scope="module")
def seed_index():
    return JP.load_seed_index(str(SEEDS))


def pin_checkout(tmp_path: Path, view: str = "ctlstop64s1") -> dict:
    """A fake checkout: the real seeds (symlinked), the COMMITTED lane roster copied in, synthetic lane pins for every roster entry
    (``configs/ctlstop_eb64_pins.json``, the path the roster names), a production pins file that names none of them, and the view's
    config with ``explicit_cells`` = ``none`` + its four entries."""
    root = tmp_path / "checkout"
    (root / "configs").mkdir(parents=True)
    os.symlink(SEEDS, root / "configs" / "registry")
    roster = root / LANE_ROSTER_REL
    roster.write_text((REPO / LANE_ROSTER_REL).read_text())
    lane = {TARGET: {pin_id(v, c): {"object_sha256": fake_sha(pin_id(v, c)), "staged_id": pin_id(v, c)} for v in VIEWS for c in CELLS}}
    (root / LANE_PINS_REL).write_text(json.dumps(lane))
    prod = {TARGET: {"base64s1__constv2ctl2-full__lora_r4_ce": {"object_sha256": fake_sha("production s1 control lora")}}}
    (root / PROD_PINS_REL).write_text(json.dumps(prod))
    cfg = root / "configs" / f"sdfbase_full22_{TARGET}_{view}.yaml"
    cfg.write_text(yaml.safe_dump({"explicit_cells": ["none", *CELLS], "model": "Qwen/Qwen3.5-4B",
                                   "private_run": f"sdfbase_full22_{TARGET}_{view}"}))
    return {"root": root, "cfg": cfg, "roster": roster, "lane_pins": root / LANE_PINS_REL, "prod_pins": root / PROD_PINS_REL, "lane": lane}


def plan_kw(view: str = "ctlstop64s1", attack: str = "crescendo") -> dict:
    return dict(model_key=f"{TARGET}_{view}", roster_selection=view, evals=[attack], serving_path="lora_native", campaign="reseed_ctlstop",
                job_id="4242", job_name=f"f22-att-synthetic-{attack}", substrate="cluster_a", code_sha=synth.code_sha("a"),
                producer_key=f"sdfbase_full22-att-{attack}-{TARGET}_{view}-sh0_1_2_3_4_5_6_7_8of9", pin_form=True)


def objects_of(bundle) -> dict:
    return {r.route["object"]: r.plan["object_id"] for r in bundle.runs}


def write_roster_entry(co: dict, entry: str, mapping, view: str = "ctlstop64s1") -> None:
    doc = yaml.safe_load(co["roster"].read_text())
    doc["targets"][TARGET]["views"][view]["cells"][entry] = mapping
    co["roster"].write_text(yaml.safe_dump(doc, sort_keys=False))


# ------------------------------------------------------------------------------------------------ (1) the whole-config path


@pytest.mark.parametrize("attack", ["crescendo", "actorattack"])
def test_a_pin_form_entry_registers_the_named_pins_sha_on_the_whole_config_path(tmp_path, seed_index, attack):
    co = pin_checkout(tmp_path)
    b = JP.build_plans(config_path=str(co["cfg"]), roster_path=str(co["roster"]), **plan_kw(attack=attack))
    assert b.skipped == [], b.skipped
    want = {c: fake_sha(pin_id("ctlstop64s1", c)) for c in CELLS}
    assert objects_of(b) == {"none": "none", **want}
    assert len({r.run_id for r in b.runs}) == 5
    for r in b.runs:
        p = r.plan
        assert p["instrument_id"] == INSTRUMENT[attack] and p["item_set_id"] == "multiturn_goals_259" and p["eval_id"] == attack
        assert p["model_id"] == "qwen35_4b_it" and p["campaign"] == "reseed_ctlstop"
        assert p["transfer_kind"] == ("none" if p["object_id"] == "none" else "base_to_instruct")
        assert p["extra"]["model_key"] == f"{TARGET}_ctlstop64s1" and p["extra"]["view_legacy"] == "ctlstop64s1"
    # the production pins file names none of the entries: the shas came from the file the roster entry names
    prod = json.loads(co["prod_pins"].read_text())[TARGET]
    assert not set(want.values()) & {v["object_sha256"] for v in prod.values()}


# ------------------------------------------------------------------------------------------------ (2) the narrowed-copy path


def test_the_narrowed_copy_with_the_checkouts_seeds_and_the_production_pins_named_resolves_the_same_objects(tmp_path, seed_index, monkeypatch):
    """The per-cell registration writes a NARROWED copy of the config under the shared root (outside the checkout) and names the checkout's
    seeds and production pins (common.sh f22_registry_register); the roster is the job's F22_ROSTER, relative to the checkout it runs in.
    The lane pins are reachable only through the roster entry, resolved against the checkout holding the named seeds."""
    co = pin_checkout(tmp_path)
    ref = JP.build_plans(config_path=str(co["cfg"]), roster_path=str(co["roster"]), **plan_kw())
    pend = tmp_path / "shared" / "registry" / "pending"
    pend.mkdir(parents=True)
    narrowed = yaml.safe_load(co["cfg"].read_text())
    narrowed.update(explicit_cells=list(CELLS), registry_narrowed_from=str(co["cfg"]), registry_narrowed_cells=list(CELLS))
    copy_path = pend / "4242.config.yaml"
    copy_path.write_text(yaml.safe_dump(narrowed, sort_keys=False))
    # absolute paths, as common.sh names $REPO_DIR/configs/...
    named = JP.build_plans(config_path=str(copy_path), seeds_dir=str(co["root"] / "configs" / "registry"), roster_path=str(co["roster"]),
                           pins_path=str(co["prod_pins"]), **plan_kw())
    assert named.skipped == [] and objects_of(named) == {c: oid for c, oid in objects_of(ref).items() if c != "none"}
    # the job's own spelling: cwd = the checkout, the roster and the seeds relative to it
    monkeypatch.chdir(co["root"])
    rel = JP.build_plans(config_path=str(copy_path), seeds_dir="configs/registry", roster_path=LANE_ROSTER_REL, pins_path=PROD_PINS_REL, **plan_kw())
    assert rel.skipped == [] and objects_of(rel) == objects_of(named)
    assert {r.run_id for r in rel.runs} == {r.run_id for r in named.runs} == {r.run_id for r in ref.runs if r.route["object"] != "none"}


def test_the_pins_root_is_the_checkout_of_the_named_seeds_not_of_the_config(tmp_path, seed_index):
    """A narrowed copy names the seeds of ANOTHER checkout that carries no lane pins: the entries are skipped by name (the pins file is
    resolved against the seeds' checkout, never against the copy's location or a guess)."""
    co = pin_checkout(tmp_path)
    other = tmp_path / "other"
    (other / "configs").mkdir(parents=True)
    os.symlink(SEEDS, other / "configs" / "registry")
    copy_path = tmp_path / "pending.config.yaml"
    copy_path.write_text(co["cfg"].read_text())
    b = JP.build_plans(config_path=str(copy_path), seeds_dir=str(other / "configs" / "registry"), roster_path=str(co["roster"]),
                       pins_path=str(co["prod_pins"]), **plan_kw())
    skipped = dict(b.skipped)
    assert objects_of(b) == {"none": "none"}
    for c in CELLS:
        assert f"no pins file {LANE_PINS_REL} at {other / LANE_PINS_REL}" in skipped[f"object {c}"], skipped


# ------------------------------------------------------------------------------------------------ (3) refusals: skipped by name


ENTRY = CELLS[0]


def _pin(view="ctlstop64s1", cell=ENTRY) -> str:
    return f"{TARGET}.{pin_id(view, cell)}"


REFUSALS = {
    "absolute pins path": ({"pins": "/abs/" + LANE_PINS_REL, "pin": _pin()}, "is absolute; the mapping form names it relative to the checkout"),
    "pins path leaves the checkout": ({"pins": "../outside_pins.json", "pin": _pin()}, "outside the checkout"),
    "pins path climbs back in and out": ({"pins": "configs/../../outside_pins.json", "pin": _pin()}, "outside the checkout"),
    "pin names another model key": ({"pins": LANE_PINS_REL, "pin": f"gemma4e4b.{pin_id('ctlstop64s1', ENTRY)}"},
                                    "names model key 'gemma4e4b', not the roster target 'qwen35_4b'"),
    "pin is not model.staged": ({"pins": LANE_PINS_REL, "pin": f"{TARGET}.a.b"}, "is not `<model key>.<staged id>`"),
    "pin entry absent": ({"pins": LANE_PINS_REL, "pin": f"{TARGET}.ctlstop64s1__no_such_entry"}, "names no 64-hex object_sha256"),
    "mapping with an extra key": ({"pins": LANE_PINS_REL, "pin": _pin(), "staged_id": pin_id("ctlstop64s1", ENTRY)},
                                  "the roster's mapping form takes exactly ['pins', 'pin']"),
    "mapping missing pin": ({"pins": LANE_PINS_REL}, "the roster's mapping form takes exactly ['pins', 'pin']"),
    "pins not a string": ({"pins": ["x"], "pin": _pin()}, "not two non-empty strings"),
}


@pytest.mark.parametrize("case", sorted(REFUSALS))
def test_a_malformed_pin_form_entry_is_skipped_by_name_never_resolved(tmp_path, seed_index, case):
    co = pin_checkout(tmp_path)
    (tmp_path / "checkout" / ".." / "outside_pins.json").write_text(json.dumps(co["lane"]))    # an escaping path EXISTS: refused by path, not absence
    mapping, why = REFUSALS[case]
    write_roster_entry(co, ENTRY, mapping)
    b = JP.build_plans(config_path=str(co["cfg"]), roster_path=str(co["roster"]), **plan_kw())
    skipped = dict(b.skipped)
    assert why in skipped.get(f"object {ENTRY}", ""), (case, skipped)
    assert repr(ENTRY) in skipped[f"object {ENTRY}"]
    assert ENTRY not in objects_of(b) and set(objects_of(b)) == {"none", *CELLS[1:]}       # the other entries still register


def test_a_missing_lane_pins_file_skips_every_entry_by_name(tmp_path, seed_index):
    co = pin_checkout(tmp_path)
    co["lane_pins"].unlink()
    b = JP.build_plans(config_path=str(co["cfg"]), roster_path=str(co["roster"]), **plan_kw())
    skipped = dict(b.skipped)
    assert objects_of(b) == {"none": "none"}
    for c in CELLS:
        assert f"no pins file {LANE_PINS_REL} at {co['lane_pins']}" in skipped[f"object {c}"]


@pytest.mark.parametrize("sha", [None, "", "abc123", "A" * 64, "g" * 64, fake_sha("x")[:63], 12345])
def test_a_short_missing_or_non_hex_sha_is_skipped_by_name(tmp_path, seed_index, sha):
    co = pin_checkout(tmp_path)
    lane = copy.deepcopy(co["lane"])
    entry = lane[TARGET][pin_id("ctlstop64s1", ENTRY)]
    if sha is None:
        entry.pop("object_sha256")
    else:
        entry["object_sha256"] = sha
    co["lane_pins"].write_text(json.dumps(lane))
    b = JP.build_plans(config_path=str(co["cfg"]), roster_path=str(co["roster"]), **plan_kw())
    assert "names no 64-hex object_sha256" in dict(b.skipped)[f"object {ENTRY}"] and ENTRY not in objects_of(b)


def test_an_unreadable_pins_file_is_skipped_by_name(tmp_path, seed_index):
    co = pin_checkout(tmp_path)
    co["lane_pins"].write_text("{not json")
    b = JP.build_plans(config_path=str(co["cfg"]), roster_path=str(co["roster"]), **plan_kw())
    assert "unreadable" in dict(b.skipped)[f"object {ENTRY}"] and objects_of(b) == {"none": "none"}


def test_every_entry_skipped_registers_no_run_for_them(tmp_path, seed_index):
    """The narrowed job (its four entries, no `none`) with no resolvable entry plans NOTHING: emit_cli refuses such a registration (exit 2)
    and the job ends before any standup -- never a job running unregistered."""
    co = pin_checkout(tmp_path)
    co["lane_pins"].unlink()
    copy_path = tmp_path / "narrowed.config.yaml"
    narrowed = yaml.safe_load(co["cfg"].read_text())
    narrowed["explicit_cells"] = list(CELLS)
    copy_path.write_text(yaml.safe_dump(narrowed))
    b = JP.build_plans(config_path=str(copy_path), seeds_dir=str(co["root"] / "configs" / "registry"), roster_path=str(co["roster"]),
                       pins_path=str(co["prod_pins"]), **plan_kw())
    assert b.runs == [] and {w for w, _ in b.skipped} == {f"object {c}" for c in CELLS}


# ------------------------------------------------------------------------------------------------ (4) the default contract


def test_without_pins_root_a_mapping_entry_raises_and_a_string_entry_is_unchanged(tmp_path):
    co = pin_checkout(tmp_path)
    roster = yaml.safe_load(co["roster"].read_text())
    pins = json.loads(co["prod_pins"].read_text())
    kw = dict(model_key=TARGET, roster_selection="ctlstop64s1", roster=roster, pins=pins, si_sha={})
    with pytest.raises(LookupError, match="the roster maps it to no staged directory"):
        JP.object_id_for(ENTRY, **kw)
    assert JP.object_id_for(ENTRY, **kw, pins_root=str(co["root"])) == fake_sha(pin_id("ctlstop64s1", ENTRY))
    # a string entry (the production form) resolves identically with and without pins_root, and so do its refusals
    roster["targets"][TARGET]["views"]["main"] = {"cells": {ENTRY: "base64s1__constv2ctl2-full__lora_r4_ce", "unmapped": None}}
    skw = dict(kw, roster_selection="main")
    assert JP.object_id_for(ENTRY, **skw) == JP.object_id_for(ENTRY, **skw, pins_root=str(co["root"])) == fake_sha("production s1 control lora")
    for entry in ("unmapped", "absent__entry"):
        msgs = []
        for pr in (None, str(co["root"])):
            with pytest.raises(LookupError) as ei:
                JP.object_id_for(entry, **skw, pins_root=pr)
            msgs.append(str(ei.value))
        assert msgs[0] == msgs[1], msgs
    assert JP.object_id_for("none", **kw, pins_root=str(co["root"])) == "none"


# ------------------------------------------------------------------------------------------------ (5) one rule, two readers


def _ctl_neg1_entries():
    roster = yaml.safe_load(CTL_NEG1_ROSTER.read_text())
    out = []
    for view, v in roster["targets"][TARGET]["views"].items():
        for cell, mapping in (v.get("cells") or {}).items():
            if isinstance(mapping, dict):
                out.append((view, cell, mapping))
    return roster, out


def test_on_a_committed_pin_form_roster_the_registration_and_the_serving_readers_name_the_same_object(seed_index):
    import make_full22_configs as gen
    import mt_serving_up as MS

    roster, entries = _ctl_neg1_entries()
    assert len(entries) == 6
    for view, cell, mapping in entries:
        doc = json.loads((REPO / mapping["pins"]).read_text())
        model, staged = mapping["pin"].split(".")
        want = doc[model][staged]["object_sha256"]
        got = JP.object_id_for(cell, model_key=TARGET, roster_selection=view, roster=roster, pins=None, si_sha={}, pins_root=str(REPO))
        assert got == want == MS._pin_form_sha(mapping, repo_root=REPO, cell=cell) == gen.pin_form_facts(TARGET, cell, mapping)[2], (view, cell)
        # through the whole registration, on the committed rendered config of the view
        cfg = REPO / roster["targets"][TARGET]["views"][view]["config"]
        b = JP.build_plans(config_path=str(cfg), model_key=f"{TARGET}_{view}", roster_selection=view, evals=["crescendo"], serving_path="lora_native",
                           campaign="sdfbase_full22", job_id="1", job_name="f22-att-synthetic-crescendo", roster_path=str(CTL_NEG1_ROSTER),
                           substrate="cluster_a", code_sha=synth.code_sha("a"), pin_form=True)
        assert b.skipped == [] and objects_of(b) == {"none": "none", cell: want}, (view, b.skipped)


# ------------------------------------------------------------------------------------------------ (6) the model keys


def test_the_three_view_model_keys_are_legacy_spellings_and_a_fourth_seed_is_not(seed_index):
    for k in (1, 2, 3):
        assert JP.model_for_key(seed_index.legacy_model_keys, f"{TARGET}_ctlstop64s{k}") == ("qwen35_4b_it", "base_to_instruct")
    with pytest.raises(JP.JobPlanRefused, match="is not a legacy model spelling"):
        JP.model_for_key(seed_index.legacy_model_keys, f"{TARGET}_ctlstop64s4")


# ------------------------------------------------------------------------------------------------ (7) the opt-in


NOT_STAGED = "the roster maps it to no staged directory (a prompted or unstaged entry)"


def test_without_the_opt_in_every_pin_form_entry_is_skipped_exactly_as_before(tmp_path, seed_index, monkeypatch):
    monkeypatch.delenv(JP.PIN_FORM_ENV, raising=False)
    co = pin_checkout(tmp_path)
    kw = {k: v for k, v in plan_kw().items() if k != "pin_form"}
    b = JP.build_plans(config_path=str(co["cfg"]), roster_path=str(co["roster"]), **kw)
    assert objects_of(b) == {"none": "none"}
    assert dict(b.skipped) == {f"object {c}": f"{c!r}: {NOT_STAGED}" for c in CELLS}
    assert not any(JP.PIN_FORM_ENV in n for n in b.notes)
    # a committed pin-form roster: its entry is skipped, as before this opt-in existed
    roster, entries = _ctl_neg1_entries()
    view, cell, _ = entries[0]
    cfg = REPO / roster["targets"][TARGET]["views"][view]["config"]
    b = JP.build_plans(config_path=str(cfg), model_key=f"{TARGET}_{view}", roster_selection=view, evals=["crescendo"], serving_path="lora_native",
                       campaign="sdfbase_full22", job_id="1", job_name="f22-att-synthetic-crescendo", roster_path=str(CTL_NEG1_ROSTER),
                       substrate="cluster_a", code_sha=synth.code_sha("a"))
    assert objects_of(b) == {"none": "none"} and dict(b.skipped) == {f"object {cell}": f"{cell!r}: {NOT_STAGED}"}


@pytest.mark.parametrize("env, kwarg, resolves", [("1", None, True), ("0", None, False), ("", None, False), (None, None, False),
                                                   ("1", False, False), ("0", True, True), (None, True, True)])
def test_the_environment_switches_the_opt_in_and_the_keyword_wins(tmp_path, seed_index, monkeypatch, env, kwarg, resolves):
    if env is None:
        monkeypatch.delenv(JP.PIN_FORM_ENV, raising=False)
    else:
        monkeypatch.setenv(JP.PIN_FORM_ENV, env)
    co = pin_checkout(tmp_path)
    kw = dict(plan_kw(), pin_form=kwarg)
    b = JP.build_plans(config_path=str(co["cfg"]), roster_path=str(co["roster"]), **kw)
    if resolves:
        assert b.skipped == [] and set(objects_of(b)) == {"none", *CELLS}
        assert any(JP.PIN_FORM_ENV in n and str(co["root"]) in n for n in b.notes), b.notes
    else:
        assert objects_of(b) == {"none": "none"} and {w for w, _ in b.skipped} == {f"object {c}" for c in CELLS}


@pytest.mark.parametrize("value", ["true", "yes", "2", "on"])
def test_an_unrecognised_opt_in_value_refuses_the_registration_by_name(tmp_path, seed_index, monkeypatch, value):
    monkeypatch.setenv(JP.PIN_FORM_ENV, value)
    co = pin_checkout(tmp_path)
    kw = {k: v for k, v in plan_kw().items() if k != "pin_form"}
    with pytest.raises(JP.JobPlanRefused, match=f"{JP.PIN_FORM_ENV}={value!r}"):
        JP.build_plans(config_path=str(co["cfg"]), roster_path=str(co["roster"]), **kw)
    assert JP.pin_form_requested(True) is True and JP.pin_form_requested(False) is False
