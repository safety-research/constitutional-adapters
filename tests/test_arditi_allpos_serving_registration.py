"""The SERVING-SIDE REGISTRATION of the all-positions Arditi arm (`arditi_allpos`), and nothing else.

WHAT THIS LANDS AND WHY IT IS SMALL. The engine on this tree can already serve an all-positions
constant residual direction: `defense_cells.RESIDUAL_SLOT_KINDS` and `NATIVE_LORA_SERVABLE_KINDS`
carry `vector_all`, `native_position_mode("vector_all", …)` returns `("all", None)`,
`native_residual.CONSTANT_KINDS` contains it and `lora_export._residual_vector_all` exports it. What
was missing was the VOCABULARY: three tuples that name which Class B ids exist, what object kind each
arm serves, and which kinds take the logprob liveness statistic. A main-derived standup of
`[none, arditi_allpos]` therefore refused at the FIRST door, before any venv, bank view or GPU:

    scripts/sdfbase/full22/serving_path_select.py --cells none arditi_allpos
    -> DefenseCellError: malformed cell id 'arditi_allpos': … or one of
       ('antidote', 'alphasteer', 'caa_static_r', 'arditi')                       exit 2

so this file drives the REAL doors a launch hits, in the order it hits them, and pins a KNOWN-BAD at
each one by restoring the pre-registration tuple and watching the same door refuse again. A test that
only asserted the new tuples' contents would pass against a registration that reached no consumer.

THE THIRD LINE IS A LATENT BUG FIX, not a convenience. `native_standup.logprob_liveness_cells` keys on
the object KIND, so a `vector_all` object at position mode `all` matched NEITHER branch (not a listed
constant kind; its mode is not restricted) and silently took the greedy-TEXT liveness statistic —
exactly where a correct constant add at a small norm reads as a no-op. The known-bad below is that
silence: with the old tuple the cell is simply absent from the mapping.

Every object here is SYNTHETIC: seeded tensors, invented shas, no staged artifact and no cluster read.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
F22 = REPO / "scripts" / "sdfbase" / "full22"
COMMON_SH = F22 / "common.sh"
ARM_CELL_CHECK = F22 / "arm_cell_check.py"
SERVING_PATH_SELECT = F22 / "serving_path_select.py"
ARMS_PATH = REPO / "configs" / "sdfbase_full22_arms.yaml"
PINS_PATH = REPO / "configs" / "arditi_allpos_pins.json"
ROSTER_PATH = REPO / "configs" / "arditi_allpos_roster.yaml"
sys.path.insert(0, str(REPO / "src"))

ARM = CELL = "arditi_allpos"
KIND = "vector_all"
TARGET, VIEW = "qwen35_4b", "it"
POLICY = "lora_native"
#: what the three tuples were BEFORE this registration — every known-bad below restores one of them
MAIN_SERVED_ONLY = ("arditi",)
MAIN_LOGPROB_KINDS = ("alphasteer", "static_r")
#: an id that is registered NOWHERE, so every door must still refuse it after the registration:
#: the gates are not merely opened, they still close on an unknown name
UNREGISTERED = "arditi_allpositions"


# ==========================================================================
# 1. the first door: the serving-path selector, run as the shell runs it
# ==========================================================================

def _select(*cells: str) -> subprocess.CompletedProcess:
    """`serving_path_select.py --cells …` exactly as `common.sh::f22_select_auto` runs it inside
    `f22_init` — a subprocess, before any venv, bank view or GPU."""
    return subprocess.run([sys.executable, str(SERVING_PATH_SELECT), "--cells", *cells, "--sh"],
                          capture_output=True, text=True, cwd=str(REPO),
                          env={**os.environ, "PYTHONPATH": str(REPO / "src")})


def test_the_first_door_selects_native_for_the_arm_and_still_refuses_an_unregistered_id():
    r = _select("none", CELL)
    assert r.returncode == 0, (r.stdout, r.stderr)
    assert "F22_SERVING_PATH_SELECTED=lora_native" in r.stdout, r.stdout
    assert "F22_SERVING_PATH_MODE=auto" in r.stdout, r.stdout
    rec = json.loads(r.stdout.split("F22_SERVING_PATH_SELECTION_JSON='", 1)[1].rsplit("'", 2)[0])
    # the arm is SERVED, not skipped by name — the disposition is what decides whether a native
    # standup would drop it silently
    assert rec["cells_by_disposition"] == {"passthrough": ["none"], "serve": [CELL], "skip": []}
    assert rec["selected"] == POLICY and rec["skipped"] == []
    # the SAME call for the arm that was already registered, unchanged
    assert "F22_SERVING_PATH_SELECTED=lora_native" in _select("none", "arditi").stdout
    # KNOWN-BAD: an id no tuple names still refuses at this very door, with the vocabulary listed
    bad = _select("none", UNREGISTERED)
    assert bad.returncode != 0, bad.stdout
    assert f"malformed cell id '{UNREGISTERED}'" in (bad.stdout + bad.stderr)
    assert CELL in (bad.stdout + bad.stderr), "the refusal lists the vocabulary, which now names the arm"


# ==========================================================================
# 2. the three vocabulary lines, each through the consumer that reads it
# ==========================================================================

def test_the_class_b_id_is_served_only_and_reaches_every_reader_of_those_tuples(monkeypatch):
    """`SERVED_ONLY_CLASS_B_IDS`, NOT `CLASS_B_IDS`: the latter is also the train/test Cartesian
    registry list, and no trainer on this tree mints either Arditi object — the same distinction
    `arditi` already relies on."""
    from sisteer.capeval.petri import defense_cells as dc
    from sisteer.capeval.tinycap import contracts as C

    assert dc.SERVED_ONLY_CLASS_B_IDS == ("arditi", CELL)
    assert CELL not in dc.CLASS_B_IDS and not set(dc.SERVED_ONLY_CLASS_B_IDS) & set(dc.CLASS_B_IDS)
    parsed = dc.parse_cell_id(CELL)
    assert (parsed.cell_class, parsed.method, parsed.si_id) == ("class_b", CELL, None)
    # launchable_class_b_ids READS the tuples (never transcribes), so the launcher grammar follows
    assert C.launchable_class_b_ids() == tuple(dc.CLASS_B_IDS) + tuple(dc.SERVED_ONLY_CLASS_B_IDS)
    assert C.parse_launcher_cell(CELL).cell == C.parse_launcher_cell(CELL).harness_id == CELL
    # KNOWN-BAD: main's tuple, and the same two doors refuse by name
    monkeypatch.setattr(dc, "SERVED_ONLY_CLASS_B_IDS", MAIN_SERVED_ONLY)
    with pytest.raises(dc.DefenseCellError, match="malformed cell id"):
        dc.parse_cell_id(CELL)
    with pytest.raises(C.CellNameError, match="not a cell id"):
        C.parse_launcher_cell(CELL)


def test_the_explicit_cell_registry_admits_the_arm_by_name(tmp_path, monkeypatch):
    """The door an `explicit_cells: [none, arditi_allpos]` config hits
    (`eval/defense_cell.registry_from_explicit_cells`, which tests `cid in ARMS or cid in
    SERVED_ONLY_CLASS_B_IDS`). With no object staged it must complain about the BYTES, never about
    the NAME — that difference is the registration."""
    from sisteer.eval import defense_cell as edc

    cfg = {"explicit_cells": ["none", CELL], "defenses_dir": str(tmp_path / "view")}
    with pytest.raises(ValueError) as e:
        edc.registry_from_explicit_cells(cfg, repo_root=REPO)
    assert "no staged artifact" in str(e.value), str(e.value)[:400]
    assert "not a recognized cell id" not in str(e.value)
    # KNOWN-BAD: main's tuple, and the same call refuses on the NAME instead
    monkeypatch.setattr(edc, "SERVED_ONLY_CLASS_B_IDS", MAIN_SERVED_ONLY)
    with pytest.raises(ValueError) as e2:
        edc.registry_from_explicit_cells(cfg, repo_root=REPO)
    assert "not a recognized cell id" in str(e2.value), str(e2.value)[:400]


def test_the_object_kind_is_vector_all_and_the_kind_only_source_resolves_it(monkeypatch):
    """`ARM_OBJECT_KINDS` is what lets the shell pick a serving path BY KIND with no venv and no
    torch, and what the serving adapters dispatch on. `vector_all` is the blob's own stored kind, so
    `_resolve_torch_blob`'s kind check agrees with the bytes."""
    from sisteer.capeval.petri import defense_cells as dc, registry_adapter as ra

    assert ra.ARM_OBJECT_KINDS[CELL] == KIND != ra.ARM_OBJECT_KINDS["arditi"]
    assert ra.harness_cell_id(CELL) == CELL          # the harness vocabulary, via the same tuples
    src = ra.kind_only_cell_source(["none", CELL])
    assert src.get(CELL).kind == KIND
    # and the engine's own verdict for that kind, on the spec-less record the shell reads: SERVE,
    # at position mode `all` — so a native standup neither skips it by name nor mis-stages it
    assert dc.native_serving_disposition(KIND, None)[0] == "serve"
    assert dc.native_position_mode(KIND, {"positions": "all", "fit_positions": "all",
                                          "skip_bos": False}) == ("all", None)
    assert KIND in dc.NATIVE_LORA_SERVABLE_KINDS and KIND in dc.RESIDUAL_SLOT_KINDS
    # the kind and the class id must land TOGETHER: every served-only id is mapped, which is the
    # adapter's own invariant and the reason these two lines cannot be split across two PRs
    assert set(ra.ARM_OBJECT_KINDS) >= set(dc.SERVED_ONLY_CLASS_B_IDS)
    # KNOWN-BAD, last (it removes the mapping): the kind-only source refuses rather than guessing
    monkeypatch.setattr(ra, "ARM_OBJECT_KINDS", {k: v for k, v in ra.ARM_OBJECT_KINDS.items() if k != CELL})
    with pytest.raises(Exception, match="object-kind mapping"):
        ra.kind_only_cell_source(["none", CELL])


def test_a_constant_add_at_mode_all_takes_the_LOGPROB_statistic_not_greedy_text(monkeypatch):
    """THE LATENT BUG. `logprob_liveness_cells` has two branches — a constant KIND, or a RESTRICTED
    position mode — and a `vector_all` object at mode `all` matched neither, so it took the greedy
    TEXT statistic, where a correct constant add at a small norm reads as a no-op. The known-bad is
    that SILENCE: with the old tuple the cell is simply absent from the mapping, which is why no
    existing test caught it."""
    from sisteer.capeval.serving import native_standup as ns
    from sisteer.capeval.tinycap import contracts as tc

    assert tc.LOGPROB_LIVENESS_KINDS == ("alphasteer", "static_r", "vector_all", "vector", "diffmean")
    assert "psr" not in tc.LOGPROB_LIVENESS_KINDS, "the stage-1 PSR certificates were admitted under the greedy gate"

    su = ns.LoraStandup.__new__(ns.LoraStandup)
    su.lora_cells = [CELL]
    su.adapters = {CELL: {"petri_source_kind": KIND}}
    su.residual_slot = {"armed": True, "cells": {CELL: {"position_mode": "all"}}}
    out = ns.logprob_liveness_cells(su)
    assert set(out) == {CELL} and f"constant-kind object ({KIND})" in out[CELL]

    monkeypatch.setattr(tc, "LOGPROB_LIVENESS_KINDS", MAIN_LOGPROB_KINDS)
    assert ns.logprob_liveness_cells(su) == {}, "the pre-registration silence this line fixes"


def test_the_generator_vocabulary_equals_the_two_class_b_tuples():
    """`make_full22_configs.BARE_CLASS_B_IDS` is restated to keep that generator import-light, so it
    must move with the tuples it mirrors (its own test pins the equality; this states the value)."""
    sys.path.insert(0, str(REPO / "scripts" / "sdfbase"))
    import make_full22_configs as gen
    from sisteer.capeval.petri.defense_cells import CLASS_B_IDS, SERVED_ONLY_CLASS_B_IDS

    assert set(gen.BARE_CLASS_B_IDS) == set(CLASS_B_IDS) | set(SERVED_ONLY_CLASS_B_IDS)
    assert CELL in gen.BARE_CLASS_B_IDS


# ==========================================================================
# 3. the arms table + arm_cell_check: where the arm's bytes live, and its policy
# ==========================================================================

def _acc(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(ARM_CELL_CHECK), *args], capture_output=True, text=True)


def test_the_committed_table_resolves_the_selected_rung_and_its_native_policy():
    """THE ACCEPTANCE: a checkout of this tree resolves `[none, arditi_allpos]` for the ONE target
    whose ladder has been run and selected."""
    r = _acc("--resolve", "--target", TARGET, "--arm", ARM, "--view", VIEW, "--cluster", "cluster_a")
    assert r.returncode == 0, (r.stdout, r.stderr)
    cell, kind, rel, policy = r.stdout.split()
    arms = yaml.safe_load(ARMS_PATH.read_text(encoding="utf-8"))
    op = arms["targets"][TARGET][ARM]
    assert (cell, kind, policy) == (CELL, KIND, POLICY)
    assert rel == f"exp/objects/arditi_allpos/{TARGET}/ladder/cells_view/{op['bank_view']}"
    assert "/exp/objects/sdfbase/" not in f"/{rel}/", "an arm never resolves under the WAVE bank"
    p = _acc("--arm", ARM, "--policy")
    assert p.returncode == 0 and p.stdout.strip() == POLICY, p


@pytest.mark.parametrize("args,needle", [
    (("--resolve", "--target", TARGET, "--arm", ARM, "--view", TARGET, "--cluster", "cluster_a"),
     "is not qwen35_4b's PAPER view 'it'"),
    (("--resolve", "--target", "gemma4e4b", "--arm", ARM),
     "no arditi_allpos operating point for gemma4e4b"),
    (("--resolve", "--target", TARGET, "--arm", ARM, "--view", VIEW, "--cluster", "cluster_b",
      "--requested-serving-path", "lens", "--serving-path-mode", "explicit"),
     "declares serving_path: lora_native"),
])
def test_the_resolver_refuses_each_wrong_ask_by_name(args, needle):
    """Known-bads, one per refusal the resolver owns: the wrong view (the arm's pair derives from the
    PAPER view's and pairs with its `none` anchor), a target whose ladder was never selected, and a
    lens request for a native-only arm — which must fire BEFORE any per-target refusal, so it is
    never masked by "no operating point yet"."""
    r = _acc(*args)
    assert r.returncode != 0, (r.stdout, r.stderr)
    assert needle in (r.stdout + r.stderr), (needle, r.stdout, r.stderr)


def test_the_table_declares_the_policy_on_the_new_arm_only():
    arms = yaml.safe_load(ARMS_PATH.read_text(encoding="utf-8"))
    assert arms["arms"][ARM]["serving_path"] == POLICY
    for other in ("arditi", "alphasteer"):
        assert "serving_path" not in arms["arms"][other], f"{other} keeps the caller's choice"
    assert _acc("--arm", "arditi", "--policy").stdout.strip() == "-"


def test_the_pins_and_the_table_name_ONE_set_of_bytes_per_rung():
    """Two authorities for one identity must agree: the operating point's `object_sha256` is the pins
    entry's, and the table's approved FIT is what the pins' layers are held to (a pins file cannot be
    the authority for a fit the table never approved)."""
    arms = yaml.safe_load(ARMS_PATH.read_text(encoding="utf-8"))
    pins = json.loads(PINS_PATH.read_text(encoding="utf-8"))
    op = arms["targets"][TARGET][ARM]
    entry = pins[TARGET][op["staged_id"]]
    assert entry["object_sha256"] == op["object_sha256"]
    fit = arms["arms"][ARM]["fits"][TARGET]
    for k in ("fit_layer", "served_layer", "stored_layer", "direction_sha256"):
        assert entry[k] == fit[k], k
    # this kind stores the SERVED layer (static_r stores the FIT layer, one later) — the inversion
    # that makes a cross-arm rename impossible to satisfy
    assert fit["served_layer"] == fit["fit_layer"] - 1 == fit["stored_layer"]
    assert entry["kind"] == KIND and entry["view"] == op["bank_view"]


# ==========================================================================
# 4. the shell: f22_arm_bank_view over a SYNTHETIC world, through the real function
# ==========================================================================

def _bash_funcs(*names: str) -> str:
    sh = COMMON_SH.read_text(encoding="utf-8")
    out = [l for l in sh.splitlines()
           if l.startswith(("F22_ARM_SERVING_EVIDENCE_FILE=", "F22_ARM_SERVING_EVIDENCE_ENV="))]
    assert len(out) == 2, out
    for n in names:
        start = sh.index(f"{n}(){{")
        out.append(sh[start:sh.index("\n}\n", start) + 3])
    return "\n".join(out)


def _world(tmp_path: Path) -> tuple[Path, Path]:
    """A REPO_DIR with the real resolvers and a SYNTHETIC arms table + bank: the committed operating
    point names bytes that live on the cluster, so the rung here is synthetic and the table's
    approved fit is re-pointed at it. Nothing under configs/ is read for the object."""
    torch = pytest.importorskip("torch")
    repo, shared = tmp_path / "repo", tmp_path / "shared"
    (repo / "scripts/sdfbase/full22").mkdir(parents=True)
    for f in ("arm_cell_check.py", "make_bank_view.py", "f22_roster.py", "serving_path_select.py"):
        (repo / "scripts/sdfbase/full22" / f).symlink_to(F22 / f)
    (repo / "scripts/arditi").mkdir(parents=True)
    for f in ("arditi_strength.py", "arditi_targets.py"):
        (repo / "scripts/arditi" / f).symlink_to(REPO / "scripts" / "arditi" / f)
    (repo / "src").symlink_to(REPO / "src")
    (repo / "configs/capeval/manifests").mkdir(parents=True)

    arms = copy.deepcopy(yaml.safe_load(ARMS_PATH.read_text(encoding="utf-8")))
    op = dict(arms["targets"][TARGET][ARM])
    rung, staged = op["bank_view"], op["staged_id"]
    fit_layer, served_layer = 13, 12
    bank = shared / "exp/objects/arditi_allpos" / TARGET / "ladder"
    d = bank / "cells" / staged
    d.mkdir(parents=True)
    g = torch.Generator().manual_seed(3)
    torch.save({"kind": KIND, "vectors": {str(served_layer): torch.randn(8, generator=g)},
                "positions": "all", "skip_bos": False, "fit_positions": "all",
                "fit_response_header_len": None}, d / "object.pt")
    sha = hashlib.sha256((d / "object.pt").read_bytes()).hexdigest()
    dsha = "c" * 64
    (d / "fit_provenance.json").write_text(json.dumps(
        {"cell_id": CELL, "staged_id": staged, "target": TARGET, "c": op["c"], "object_sha256": sha,
         "writer": "scripts/arditi/fit_arditi.py", "dose_unit": "raw_coefficient", "phase": "both",
         "resolved_model_id": "Qwen/Qwen3.5-4B", "resolved_revision": "8" * 40,
         "chat_template_hash": "a4aee8afcf2e0711", "positions_at_fit": "all"}),
        encoding="utf-8")
    (d / "object.pt.provenance.json").write_text(json.dumps(
        {"kind": KIND, "phase": "both", "dose_unit": "raw_coefficient", "c": op["c"],
         "positions": "all", "skip_bos": False, "fit_positions": "all",
         "layers": [served_layer], "stored_layer": served_layer,
         "fit": {"target": TARGET, "selected": {"layer": fit_layer, "served_layer": served_layer,
                                                "pos": -5, "direction_sha256": dsha}}}), encoding="utf-8")
    (d / "staged_from.json").write_text(json.dumps({"staged_id": staged, "cell_id": CELL}), encoding="utf-8")
    vd = bank / "cells_view" / rung
    vd.mkdir(parents=True)
    (vd / CELL).symlink_to(Path(os.path.relpath(d, vd)), target_is_directory=True)

    shutil.copy(ROSTER_PATH, repo / "configs" / ROSTER_PATH.name)
    entry = dict(json.loads(PINS_PATH.read_text(encoding="utf-8"))[TARGET][staged])
    entry.update(object_sha256=sha, direction_sha256=dsha,
                 source=f"arditi_allpos/{TARGET}/ladder/cells_view/{rung}/{CELL}")
    pins_p = repo / "configs" / PINS_PATH.name
    pins_p.write_text(json.dumps({TARGET: {staged: entry}}, indent=1), encoding="utf-8")
    # the arm's roster and pins resolve against the CHECKER's repo, not the cwd, so they are
    # re-pointed by ABSOLUTE path or the synthetic bytes would be held to the committed fit
    arms["arms"][ARM]["roster"] = str(repo / "configs" / ROSTER_PATH.name)
    arms["arms"][ARM]["pins"] = str(pins_p)
    arms["targets"][TARGET][ARM] = {**op, "object_sha256": sha}
    arms["arms"][ARM]["fits"] = {TARGET: {
        "fit_record_sha256": "a" * 64, "direction_pt_sha256": "b" * 64, "direction_sha256": dsha,
        "fit_layer": fit_layer, "served_layer": served_layer, "stored_layer": served_layer,
        "fit_pos": -5, "source": "synthetic (tests)"}}
    (repo / "configs/sdfbase_full22_arms.yaml").write_text(yaml.safe_dump(arms, sort_keys=False), encoding="utf-8")
    (repo / f"configs/sdfbase_full22_{TARGET}_{VIEW}__{ARM}.yaml").write_text(
        f"explicit_cells:\n- none\n- {CELL}\n"
        f"defenses_dir: out/arditi_allpos/{TARGET}/ladder/cells_view/{rung}\n", encoding="utf-8")
    return repo, shared


def _arm_bank_view(repo: Path, shared: Path) -> subprocess.CompletedProcess:
    script = "\n".join([
        "set -u", 'f22_log(){ echo "LOG $*"; }', 'f22_fatal(){ echo "FATAL $*"; exit 1; }',
        _bash_funcs("f22_arm_bank_view"),
        f"cd {repo}", f"KEY={TARGET}; VIEW={VIEW}; CLUSTER=cluster_a; SHARED={shared}",
        f"CFG=configs/sdfbase_full22_{TARGET}_{VIEW}__{ARM}.yaml",
        f"f22_arm_bank_view {sys.executable} {ARM}",
        'echo "DEFENSES=$TT_TWO_SI_DEFENSES_DIR"',
        'echo "POLICY=$F22_ARM_SERVING_POLICY KIND=$F22_ARM_KIND MODE=$F22_ARM_POSITION_MODE '
        'CELL=$F22_ARM_CELL LAYER=$F22_ARM_SERVED_LAYER"',
        'echo "PIN_SHA=$F22_ARM_OBJECT_SHA256 SERVED_SHA=$F22_ARM_SERVED_OBJECT_SHA256"'])
    env = {k: v for k, v in os.environ.items() if not k.startswith("F22_")}
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=env)


def test_the_arm_bank_view_lands_every_path_in_the_arms_own_namespace(tmp_path):
    """The step every arm launch runs, through the REAL shell function: the view resolves and BUILDS
    under the arm's own bank, the byte invariant passes, and nothing is read or written under the
    WAVE bank — which is shared state every peer leg of this (target, view) verifies against."""
    repo, shared = _world(tmp_path)
    r = _arm_bank_view(repo, shared)
    assert r.returncode == 0, (r.stdout, r.stderr)
    defenses = [l for l in r.stdout.splitlines() if l.startswith("DEFENSES=")][0].split("=", 1)[1]
    rung = yaml.safe_load((repo / "configs/sdfbase_full22_arms.yaml").read_text())["targets"][TARGET][ARM]["bank_view"]
    assert defenses == f"{shared}/exp/objects/arditi_allpos/{TARGET}/ladder/cells_view/{rung}"
    assert "exp/objects/sdfbase/full22" not in defenses and "alphasteer_f22" not in defenses
    facts = [l for l in r.stdout.splitlines() if l.startswith("POLICY=")][0]
    assert f"POLICY={POLICY}" in facts and f"KIND={KIND}" in facts
    assert "MODE=all" in facts and f"CELL={CELL}" in facts and "LAYER=12" in facts
    pin_sha, served_sha = [l for l in r.stdout.splitlines() if l.startswith("PIN_SHA=")][0] \
        .removeprefix("PIN_SHA=").split(" SERVED_SHA=")
    assert len(pin_sha) == 64 and pin_sha == served_sha, "the promised bytes ARE the observed bytes"
    assert (Path(defenses) / CELL / "object.pt").is_file()
    assert not (shared / "exp/objects/sdfbase").exists()


def test_the_arm_bank_view_refuses_bytes_that_are_not_the_pinned_object(tmp_path):
    """KNOWN-BAD: the same world with one byte of the staged object changed. Every relational check
    still passes — the blob is a coherent `vector_all` at the right layer — and the launch refuses
    anyway, because the pin is what says these are the bytes the other lanes' rows name."""
    repo, shared = _world(tmp_path)
    arms_p = repo / "configs/sdfbase_full22_arms.yaml"
    arms = yaml.safe_load(arms_p.read_text(encoding="utf-8"))
    arms["targets"][TARGET][ARM]["object_sha256"] = "d" * 64
    arms_p.write_text(yaml.safe_dump(arms, sort_keys=False), encoding="utf-8")
    r = _arm_bank_view(repo, shared)
    assert r.returncode != 0, r.stdout
    # the SPECIFIC refusal, named: the byte invariant, not merely "something failed". The bank view itself
    # verifies FIRST and passes (the roster/pins still describe these bytes), so the message also proves the
    # pin check is a SECOND, independent gate rather than the same one reported twice
    assert "(1 entries verified)" in r.stdout, r.stdout
    assert ("FATAL the arditi_allpos object about to be served for qwen35_4b/it violates its invariant"
            in r.stdout), r.stdout
    assert "TT_TWO_SI_DEFENSES_DIR" not in r.stdout.split("FATAL")[1], "nothing is exported past the refusal"


def test_a_launch_with_no_arm_is_byte_identically_todays_behaviour():
    """The wave's own branch of `f22_resolve_config` is untouched and its ARM argument defaults to
    empty, so every existing two-argument caller resolves exactly as before and consults no arm."""
    sh = COMMON_SH.read_text(encoding="utf-8")
    fn = sh[sh.index("f22_resolve_config(){"):sh.index("\n}\n", sh.index("f22_resolve_config(){"))]
    assert 'ARM="${3:-}"' in fn
    assert '"") CFG="configs/sdfbase_full22_${TAG}.yaml"; MANIFEST="configs/capeval/manifests/petri_full22_${TAG}.yaml";;' in fn
    # the policy check fires on the ARM branch only, and BEFORE the pair-existence fatal
    assert fn.index('f22_arm_serving_policy "$ARM";;') < fn.index('[ -f "$CFG" ] && [ -f "$MANIFEST" ] || f22_fatal "missing')
    # every entry point on this tree still calls it with two arguments — this PR wires the helper,
    # not the launchers (their arm branches land with the lane that drives them)
    callers = subprocess.run(["grep", "-rln", 'f22_resolve_config "$KEY" "$VIEW"', str(REPO / "scripts")],
                             capture_output=True, text=True).stdout.split()
    assert callers, "the wave's own callers are still here and still two-argument"


# ==========================================================================
# 5. the out-of-support tuples ride in the SAME change as the vocabulary — the atomicity is measured, not asserted
# ==========================================================================

def _oos(cell: str, control: str, cap_key: str, cap: int, reduction_mode=None):
    from sisteer.capeval.tinycap import contracts as C
    return C.check_out_of_support_thinking_authorised(
        model=TARGET, cell=cell, controls=[control], mode="on", sampling_mode="thinking_card", samples=1,
        reduction_mode=reduction_mode, max_tokens_by_task={cap_key: cap})


def test_the_arm_has_its_own_oos_tuples_and_the_steer_tuples_did_not_move():
    """The arm's MATH-500 and GSM8K authorisations are tuples of its OWN (`_steer_allpos`), never a widening of
    `M5T_STEER_CELLS`: nine finished `_steer` tuples record that constant by name, and this arm is false of their
    declaration on three counts (both phases, a raw-coefficient dose, a declared user override over its budget)."""
    from sisteer.capeval.tinycap import contracts as C

    assert C.ALLPOS_CELL == CELL and C.ALLPOS_CELLS == ("none", CELL) and C.ALLPOS_MODELS == (TARGET,)
    assert C.M5T_STEER_CELLS == ("alphasteer", "arditi"), "the finished tuples' constant is untouched"
    names = [C.m5t_allpos_tuple_name(TARGET), *(C.g8k_allpos_tuple_name(TARGET, cap) for cap in C.G8K_CAPS)]
    assert names == [f"math500_thinking_on_full22_{TARGET}_steer_allpos",
                     f"gsm8k_paired200_{TARGET}_steer_allpos_8192", f"gsm8k_paired200_{TARGET}_steer_allpos_16384"]
    for n in names:
        exp = C.AUTHORISED_OOS_EXPERIMENTS[n]
        assert exp.cells == C.ALLPOS_CELLS and exp.model == TARGET
        assert "EVERY position of BOTH phases" in exp.what and "raw_coefficient" in exp.what and "USER OVERRIDE" in exp.what
    # byte-equal to the `_steer` twin except cells + what + the SERVED IDENTITY's reduction mode, on both instruments: the
    # arm is served natively (residual slot, no lens hook), so the driver requests reduction_mode = not_applicable — the
    # lens-era `stacked` refused it at the driver door (OOS_UNAUTHORISED, jobs 544310 / 544311, 2026-09-17)
    m5t_steer, m5t_allpos = C.AUTHORISED_OOS_EXPERIMENTS[C.m5t_steer_tuple_name(TARGET)], C.AUTHORISED_OOS_EXPERIMENTS[names[0]]
    assert {k for k in m5t_allpos._asdict() if m5t_allpos._asdict()[k] != m5t_steer._asdict()[k]} == {"cells", "what", "reduction_mode"}
    assert m5t_allpos.reduction_mode == C.REDUCTION_MODE_NOT_APPLICABLE and m5t_steer.reduction_mode == C.REDUCTION_MODE_STACKED
    for n in names[1:]:
        assert C.AUTHORISED_OOS_EXPERIMENTS[n].reduction_mode == C.REDUCTION_MODE_NOT_APPLICABLE, n
    # the native identity resolves the arm's tuple; the lens identity (what a lens-served leg would request) is REFUSED by name
    ok = C.check_out_of_support_thinking_authorised(model=TARGET, cell=CELL, controls=["math500"], mode="on", sampling_mode="thinking_card",
                                                   samples=1, reduction_mode=C.REDUCTION_MODE_NOT_APPLICABLE,
                                                   max_tokens_by_task={"math": C.MATH500_MAX_TOKENS_DEFAULT})
    assert ok == names[0]
    import pytest as _pt
    # the refusal names the FIRST differing field of the first candidate tuple (here the published row's `cell`); the
    # stable phrase is the flag's own — the point is that the lens identity is REFUSED, never resolved to the arm's tuple
    with _pt.raises(C.OutOfSupportThinkingUnauthorised, match="is not it"):
        C.check_out_of_support_thinking_authorised(model=TARGET, cell=CELL, controls=["math500"], mode="on", sampling_mode="thinking_card",
                                                   samples=1, reduction_mode=C.REDUCTION_MODE_STACKED,
                                                   max_tokens_by_task={"math": C.MATH500_MAX_TOKENS_DEFAULT})
    # each request resolves exactly ITS OWN tuple: the arm's, the FINISHED row's, and the PUBLISHED row's
    assert _oos(CELL, "math500", "math", C.MATH500_MAX_TOKENS_DEFAULT) == names[0]
    assert _oos(CELL, "gsm8k", C.GSM8K_PAIRED_TASK_NAME, 8192) == names[1]
    assert _oos(CELL, "gsm8k", C.GSM8K_PAIRED_TASK_NAME, 16384) == names[2]
    assert _oos("arditi", "math500", "math", C.MATH500_MAX_TOKENS_DEFAULT) == C.m5t_steer_tuple_name(TARGET)
    # the PUBLISHED row's door STATES its served identity (`stacked`), as the driver's first door does: the same cell is ALSO in
    # the MATH-500 seed wave's native tuple (reduction_mode not_applicable), and unstated the pair is ambiguous by construction
    # (a GSM8K property, pinned in test_tinycap_contracts)
    assert _oos("constv2-mix-full__lora_r4_ce_minus_ctl", "math500", "math", C.MATH500_MAX_TOKENS_DEFAULT, reduction_mode=C.REDUCTION_MODE_STACKED) \
        == f"math500_thinking_on_full22_{TARGET}"
    C.check_authorised_render_facts()


def test_tuple_before_vocabulary_breaks_two_finished_rows_which_is_why_they_land_together(monkeypatch):
    """THE ORDERING CONSTRAINT, MEASURED (reproduced here so it cannot be split by accident).
    With the tuples registered and the vocabulary ABSENT, the checker matches the model first and only then
    parses `cells` — so `parse_launcher_cell` refusing `arditi_allpos` raises `CellNameError` for EVERY OTHER
    request on the same model: the FINISHED `arditi` row and the PUBLISHED `lora_r4_ce_minus_ctl` row both
    stop resolving. Tuple-before-vocabulary breaks two finished lanes; vocabulary-before-tuple blocks the new
    one's legs; only one change is safe."""
    from sisteer.capeval.petri import defense_cells as dc
    from sisteer.capeval.tinycap import contracts as C

    monkeypatch.setattr(dc, "SERVED_ONLY_CLASS_B_IDS", MAIN_SERVED_ONLY)
    for cell in ("arditi", "constv2-mix-full__lora_r4_ce_minus_ctl"):
        with pytest.raises(C.CellNameError, match=f"cell '{CELL}' is not a cell id"):
            _oos(cell, "math500", "math", C.MATH500_MAX_TOKENS_DEFAULT)
