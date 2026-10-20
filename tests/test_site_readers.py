"""Every reader of a committed config value that may carry a site placeholder expands it through ``sisteer.site``, and
none of them can read from or write to a literal ``$NAME`` path.

Two halves:

* per reader: ``$NAME`` and ``${NAME}`` resolve to the same path, an unset placeholder refuses by name (the message
  names the placeholder and the variable to set), and the refusal leaves no directory whose name holds a ``$``;
* a ratchet over ``configs/``: every key whose value holds a site placeholder is either a key a reader here expands
  (:data:`EXPANDED`) or provenance text that no code opens (:data:`RECORDED`). A placeholder under any other key fails,
  naming the key, so a new placeholdered value cannot reach a reader that would take it literally. ``configs/registry/``
  is left out: its seeds' text is prose, and ``cluster_reads.yaml``'s component paths are expanded by
  ``registry.stores.load_config`` (its own tests).
"""
from __future__ import annotations

import json
import os
import re
import sys
import types
from pathlib import Path

import pytest

from sisteer import site as S

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

_VARS = sorted({v for envs, keys, _ in S.PLACEHOLDERS.values() for v in (*envs, *keys)}
               | {"SISTEER_CONFIG", "SISTEER_CLUSTER_LABEL"})


@pytest.fixture(autouse=True)
def _no_site(monkeypatch, tmp_path):
    """No placeholder resolves unless a test sets it: the host's environment and site config never answer."""
    for v in _VARS:
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv("SISTEER_SITE_DIR", str(tmp_path / "no-site-here"))
    S._SITE_CACHE.clear()
    monkeypatch.chdir(tmp_path)
    yield


def _no_literal_dirs(root: Path) -> None:
    bad = [p for p in root.rglob("*") if "$" in p.name]
    assert not bad, f"a literal placeholder path was created: {bad}"


# --- prompt lists, stores, banks -------------------------------------------------------------------------------------

def test_the_prompt_list_reader_expands_shared_and_refuses_it_unset(tmp_path, monkeypatch):
    from sisteer.data import content_sets as CS

    shared = tmp_path / "shared"
    p = shared / "exp" / "contrast" / "benign.json"
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps(["a prompt", "another prompt"]))
    monkeypatch.setenv("SISTEER_SHARED_ROOT", str(shared))
    for spelled in ("$SHARED/exp/contrast/benign.json", "${SHARED}/exp/contrast/benign.json"):
        prompts, sha = CS._load_prompt_list(spelled, "benign")
        assert prompts == ["a prompt", "another prompt"] and len(sha) == 64
    assert CS.ALPHASTEER_BENIGN_PATH.startswith("$SHARED/")
    monkeypatch.delenv("SISTEER_SHARED_ROOT")
    with pytest.raises(S.PlaceholderError, match=r"\$SHARED .*SISTEER_SHARED_ROOT"):
        CS._load_prompt_list("$SHARED/exp/contrast/benign.json", "benign")
    _no_literal_dirs(tmp_path)


def test_the_class_b_rule_paths_expand_and_refuse(tmp_path, monkeypatch):
    import fit_class_b as fcb

    monkeypatch.setenv("SISTEER_SHARED_ROOT", "/scratch/a/shared")
    rule = {"screen_path": "$SHARED/exp/objects/screen.json", "acts_dir": "${SHARED}/exp/acts"}
    assert fcb.rule_path(rule, "screen_path") == Path("/scratch/a/shared/exp/objects/screen.json")
    assert fcb.rule_path(rule, "acts_dir") == Path("/scratch/a/shared/exp/acts")
    assert fcb.rule_path(rule, "screen_path", "/explicit/out.json") == Path("/explicit/out.json")
    rep = tmp_path / "shared" / "leak.json"
    rep.parent.mkdir(parents=True)
    rep.write_text(json.dumps({"exact": 0, "near": 0, "total": [], "n_goals": 1, "frac": 0.0}))
    monkeypatch.setenv("SISTEER_SHARED_ROOT", str(tmp_path / "shared"))
    assert fcb.leakage_report_block("$SHARED/leak.json") is not None
    monkeypatch.delenv("SISTEER_SHARED_ROOT")
    with pytest.raises(S.PlaceholderError, match=r"class_b\.alphasteer\.screen_path"):
        fcb.rule_path(rule, "screen_path")
    with pytest.raises(S.PlaceholderError, match=r"leakage_report_path"):
        fcb.leakage_report_block("$SHARED/leak.json")
    _no_literal_dirs(tmp_path)


def test_the_held_out_store_reader_expands_state_and_work(tmp_path, monkeypatch):
    from sisteer.eval.attacks import heldout_leakage as HL

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("SISTEER_STATE_ROOT", "~/.local/state")
    monkeypatch.setenv("SISTEER_WORK_DIR", "/w")
    monkeypatch.delenv("TT_TWO_SI_STORE_DIR", raising=False)
    assert HL._store_dir("$STATE/pool/x") == tmp_path / "home" / ".local" / "state" / "pool" / "x"
    assert HL._store_dir("${WORK}/sampler/store") == Path("/w/sampler/store")
    monkeypatch.delenv("SISTEER_STATE_ROOT")
    with pytest.raises(S.PlaceholderError, match=r"\$STATE .*SISTEER_STATE_ROOT"):
        HL._store_dir("$STATE/pool/x")
    _no_literal_dirs(tmp_path)


def test_the_pool_builder_store_dir_expands_and_refuses(tmp_path, monkeypatch):
    import build_two_si_pool as bp

    monkeypatch.delenv("TT_TWO_SI_STORE_DIR", raising=False)
    monkeypatch.setenv("SISTEER_WORK_DIR", str(tmp_path / "work"))
    assert bp._store_dir({"store_dir": "$WORK/two-si/store"}) == tmp_path / "work" / "two-si" / "store"
    monkeypatch.delenv("SISTEER_WORK_DIR")
    with pytest.raises(S.PlaceholderError, match=r"pool\.store_dir"):
        bp._store_dir({"store_dir": "$WORK/two-si/store"})
    monkeypatch.setenv("TT_TWO_SI_STORE_DIR", str(tmp_path / "env-store"))
    assert bp._store_dir({"store_dir": "$WORK/two-si/store"}) == tmp_path / "env-store"
    _no_literal_dirs(tmp_path)


def test_the_defenses_dir_resolver_never_roots_a_placeholder_at_the_checkout(tmp_path, monkeypatch):
    from sisteer.eval import cell_controls as CC
    from sisteer.eval import defense_cell as DC

    monkeypatch.delenv("TT_TWO_SI_DEFENSES_DIR", raising=False)
    monkeypatch.setenv("SISTEER_SHARED_ROOT", "/scratch/a/shared")
    for spelled in ("$SHARED/exp/objects/q", "${SHARED}/exp/objects/q"):
        assert DC.resolve_defenses_dir({"defenses_dir": spelled}, repo_root=tmp_path) == Path("/scratch/a/shared/exp/objects/q")
        assert CC.resolve_defenses_dir({"defenses_dir": spelled}, tmp_path) == Path("/scratch/a/shared/exp/objects/q")
    assert DC.resolve_defenses_dir({"defenses_dir": "out/x"}, repo_root=tmp_path) == tmp_path / "out/x"
    monkeypatch.delenv("SISTEER_SHARED_ROOT")
    with pytest.raises(S.PlaceholderError, match=r"defenses_dir"):
        DC.resolve_defenses_dir({"defenses_dir": "$SHARED/exp/objects/q"}, repo_root=tmp_path)
    _no_literal_dirs(tmp_path)


def test_the_filler_search_expands_a_placeholder_first(tmp_path, monkeypatch):
    from sisteer.eval.attacks import static_padding as sp

    monkeypatch.delenv(sp.FILLER_ENV, raising=False)
    shared = tmp_path / "shared"
    real = shared / "data" / "pg19_filler.json"
    real.parent.mkdir(parents=True)
    real.write_text(json.dumps("corpus"))
    monkeypatch.setenv("SISTEER_SHARED_ROOT", str(shared))
    assert sp.resolve_filler_path(cfg_value="${SHARED}/data/pg19_filler.json", repo_root=tmp_path / "repo") == real
    monkeypatch.delenv("SISTEER_SHARED_ROOT")
    with pytest.raises(sp.StaticPaddingError, match=r"\$SHARED"):
        sp.resolve_filler_path(cfg_value="$SHARED/data/pg19_filler.json", repo_root=tmp_path / "repo")
    _no_literal_dirs(tmp_path)


# --- Petri scenario locks ------------------------------------------------------------------------------------------------

def test_the_petri_runner_expands_the_locks_location_and_refuses_it_unset(tmp_path, monkeypatch):
    from sisteer.capeval.tiers import petri_runner as PR

    monkeypatch.delenv("PETRI_BUNDLE_DIR", raising=False)
    lock = {"status": PR.psc.LOCK_STATUS_FROZEN, "bundle": {"content_hash": "abc123", "location": "$SHARED/petri/scenarios/abc123"}}
    monkeypatch.setattr(PR.psc, "read_lock", lambda _p: lock)
    spec = types.SimpleNamespace(scenario_bundle_hash="abc123")
    monkeypatch.setenv("SISTEER_SHARED_ROOT", "/scratch/a/shared")
    assert PR._resolve_bundle_dir(spec, None, tmp_path / "petri") == Path("/scratch/a/shared/petri/scenarios/abc123")
    monkeypatch.delenv("SISTEER_SHARED_ROOT")
    with pytest.raises(PR.PetriRunnerError, match=r"\$SHARED .*SISTEER_SHARED_ROOT"):
        PR._resolve_bundle_dir(spec, None, tmp_path / "petri")
    _no_literal_dirs(tmp_path)


def test_the_front_doors_bundle_lookup_expands_the_location_and_never_probes_it_literally(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(REPO / "reproduce" / "lib"))
    import stage as ST
    monkeypatch.delenv(ST.PETRI_BUNDLE_DIR_ENV, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    bundle = tmp_path / "shared" / "petri" / "scenarios" / "h1"
    bundle.mkdir(parents=True)
    (bundle / "bundle_manifest.json").write_text(json.dumps({"content_hash": "h1"}))
    lock = {"bundle": {"content_hash": "h1", "location": "$SHARED/petri/scenarios/h1"}}
    monkeypatch.setenv("SISTEER_SHARED_ROOT", str(tmp_path / "shared"))
    assert ST.petri_bundle_dir(lock) == str(bundle)
    monkeypatch.delenv("SISTEER_SHARED_ROOT")
    with pytest.raises(RuntimeError) as ei:
        ST.petri_bundle_dir(lock)
    assert "unresolved" in str(ei.value) and "SISTEER_SHARED_ROOT" in str(ei.value)
    _no_literal_dirs(tmp_path)


# --- capability-eval manifests -------------------------------------------------------------------------------------------

def test_a_manifest_artifact_expands_before_it_is_anchored_and_refuses_unset(tmp_path, monkeypatch):
    from sisteer.capeval import manifest as MF

    monkeypatch.setenv("SISTEER_SHARED_ROOT", "/scratch/a/shared")
    for spelled in ("$SHARED/out/v.safetensors", "${SHARED}/out/v.safetensors"):
        assert MF._resolve_artifact_path(spelled, tmp_path / "manifests") == Path("/scratch/a/shared/out/v.safetensors")
    assert MF._resolve_artifact_path("rel/v.safetensors", tmp_path / "manifests") == (tmp_path / "manifests" / "rel/v.safetensors").resolve()
    monkeypatch.delenv("SISTEER_SHARED_ROOT")
    with pytest.raises(MF.ManifestError, match=r"\$SHARED .*SISTEER_SHARED_ROOT"):
        MF._resolve_artifact_path("$SHARED/out/v.safetensors", tmp_path / "manifests")
    _no_literal_dirs(tmp_path)


def test_the_control_arms_sidecar_norms_and_artifact_expand_and_refuse(tmp_path, monkeypatch):
    from sisteer.capeval.stats import controls as CT

    shared = tmp_path / "shared"
    (shared / "out").mkdir(parents=True)
    (shared / "out" / "a.json").write_text(json.dumps({"norm": 2.0}))
    (shared / "out" / "b.json").write_text(json.dumps({"norm": 4.0}))
    monkeypatch.setenv("SISTEER_SHARED_ROOT", str(shared))
    assert CT.median_sidecar_norm(["$SHARED/out/a.json", "${SHARED}/out/b.json"]) == 3.0
    assert CT._site_path("$SHARED/out/a.json", "params.artifact") == shared / "out" / "a.json"
    monkeypatch.delenv("SISTEER_SHARED_ROOT")
    with pytest.raises(CT.ControlError, match=r"\$SHARED .*SISTEER_SHARED_ROOT"):
        CT.median_sidecar_norm(["$SHARED/out/a.json"])
    with pytest.raises(CT.ControlError, match=r"params\.artifact"):
        CT._site_path("$SHARED/out/v.safetensors", "zc params.artifact")
    _no_literal_dirs(tmp_path)


def test_a_path_mode_corpus_reference_expands_and_refuses_unset(tmp_path, monkeypatch):
    from sisteer.capeval import corpora as CO
    from sisteer.capeval import manifest as MF

    shared = tmp_path / "shared"
    (shared / "out").mkdir(parents=True)
    (shared / "out" / "pairs.jsonl").write_text("".join(json.dumps({"prompt": f"p{i}"}) + "\n" for i in range(3)))
    ref = MF.CorpusRef(path="$SHARED/out/pairs.jsonl", slice={"first": 2, "field": "prompt"})
    monkeypatch.setenv("SISTEER_SHARED_ROOT", str(shared))
    got = CO.resolve_corpus_ref(ref)
    assert got.spec["source"]["path"] == str(shared / "out" / "pairs.jsonl") and len(got) == 2
    monkeypatch.delenv("SISTEER_SHARED_ROOT")
    with pytest.raises(CO.CorpusError, match=r"relevant_corpus: .*\$SHARED .*SISTEER_SHARED_ROOT"):
        CO.resolve_corpus_ref(ref)
    _no_literal_dirs(tmp_path)


# --- models and W&B ------------------------------------------------------------------------------------------------------

def test_a_placeholdered_model_path_is_expanded_before_anything_loads_and_refused_unset(tmp_path, monkeypatch):
    from sisteer import models as M

    seen = []

    class _Stop(Exception):
        pass

    def _capture(name):
        seen.append(name)
        raise _Stop

    monkeypatch.setattr(M, "_enforce_llama31_mirror", _capture)
    monkeypatch.setenv("SISTEER_SHARED_ROOT", "/scratch/a/shared")
    with pytest.raises(_Stop):
        M.load_model_and_tokenizer("${SHARED}/models/m1")
    assert seen == ["/scratch/a/shared/models/m1"]
    monkeypatch.delenv("SISTEER_SHARED_ROOT")
    with pytest.raises(S.PlaceholderError, match=r"\(model\)"):
        M.load_model_and_tokenizer("$SHARED/models/m1")
    assert seen == ["/scratch/a/shared/models/m1"]          # refused before the hub or the disk was asked


def test_the_wandb_project_placeholder_resolves_from_the_environment_then_the_site_and_turns_wandb_off_unset(monkeypatch, capsys):
    from sisteer.eval import runner as RN

    got = {}
    fake = types.ModuleType("wandb")
    fake.init = lambda **kw: got.update(kw) or types.SimpleNamespace(finish=lambda: None)
    monkeypatch.setitem(sys.modules, "wandb", fake)
    monkeypatch.delenv("WANDB_MODE", raising=False)
    monkeypatch.setenv("WANDB_PROJECT", "env-project")
    RN.make_wandb_logger("$WANDB_PROJECT", "run", {})
    assert got["project"] == "env-project"
    monkeypatch.delenv("WANDB_PROJECT")
    monkeypatch.setenv("SISTEER_WANDB_PROJECT", "site-project")
    RN.make_wandb_logger("${WANDB_PROJECT}", "run", {})
    assert got["project"] == "site-project"
    RN.make_wandb_logger("literal-project", "run", {})
    assert got["project"] == "literal-project"
    monkeypatch.delenv("SISTEER_WANDB_PROJECT")
    got.clear()
    assert RN.make_wandb_logger("$WANDB_PROJECT", "run", {}) == (None, None)       # W&B off, said by name; never a literal project
    assert got == {} and "WANDB_PROJECT" in capsys.readouterr().err


# --- the ratchet over configs/ -------------------------------------------------------------------------------------------

#: key paths (dotted, list items as ``[]``) whose placeholder a reader expands -> the readers
EXPANDED = {
    r"(^|\.)class_b\.(benign|harmful)_prompts_path$": "fit_class_b.resolve_cluster_path, alphasteer_f22/lambda_ladder, data/content_sets",
    r"(^|\.)class_b\.alphasteer\.(screen_path|leakage_report_path|acts_dir|split_path)$": "fit_class_b.rule_path / leakage_report_block",
    r"(^|\.)llama_gate\.(released_matrix_path|gate_out)$": "alphasteer_llama_gate.main",
    r"(^|\.)attacker\.path$": "jailbreak_run._load_attacker; the job scripts that serve it, through scripts/attacker_model_path.py",
    r"(^|\.)wandb_project$": "train_two_si_objects / tune_arm / tune_lora_all / eval.runner",
    r"(^|\.)pool\.store_dir$": "build_two_si_pool._store_dir, attacks.heldout_leakage._store_dir",
    r"^defenses_dir$": "eval.defense_cell.resolve_defenses_dir (cell_controls, build_safety_defense)",
    r"^model$": "models.load_model_and_tokenizer",
    r"(^|\.)padding\.filler_cache$": "attacks.static_padding.resolve_filler_path",
    r"^bundle\.location$": "capeval petri_runner._resolve_bundle_dir, reproduce/lib/stage.petri_bundle_dir",
    r"(^|\.)ablations\.\[\]\.artifact$": "capeval.manifest._resolve_artifact_path",
    r"(^|\.)ablations\.\[\]\.params\.artifact$": "capeval.stats.controls (control:zero_coeff)",
    r"(^|\.)params\.norm\.sidecars\.\[\]$": "capeval.stats.controls.median_sidecar_norm",
    r"(^|\.)target\.relevant_corpus\.path$": "capeval.corpora.resolve_corpus_ref",
}
#: key paths whose value is provenance or explanatory text that no code opens as a path
RECORDED = {
    r"\.source$": "a pins file's provenance string (only its prefix is checked: make_full22_configs.is_committed_pin)",
    r"(^|\.)_(comment|doc|authority)(\.|$)": "explanatory text",
    r"(^|\.)_meta\.note$": "explanatory text",
    r"(^|\.)(what|why|note|notes|description)$": "explanatory text",
    r"(^|\.)review_ref$": "explanatory text",
    r"(^|\.)prerequisites(_met)?\.\[\]$": "a roster's prose prerequisites",
    r"(^|\.)selection_guard\.": "an HPO selection guard's recorded provenance (folded into configs verbatim by make_sdfbase_configs)",
    r"(^|\.)read_from\.\[\]$": "where a shipped generation config was read from (text)",
    r"(^|\.)(source_record|authority_ladder|readout_record|record|where|store)$": "a recorded evidence pointer, read by no code",
}


def _key_paths(obj, path=()):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _key_paths(v, path + (str(k),))
    elif isinstance(obj, list):
        for v in obj:
            yield from _key_paths(v, path + ("[]",))
    elif isinstance(obj, str):
        yield ".".join(path), obj


_PLACEHOLDER = re.compile(r"\$\{?(%s)(?![A-Za-z0-9_])\}?" % "|".join(S.NAMES))


def unclassified(root: Path) -> list:
    """(file, key path, value) for every placeholdered value under ``root/configs`` (registry seeds excepted) whose key is
    in neither :data:`EXPANDED` nor :data:`RECORDED`."""
    import yaml

    loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
    out = []
    base = root / "configs"
    for p in sorted(base.rglob("*")):
        if not p.is_file() or p.suffix not in (".yaml", ".yml", ".json", ".jsonl") or "registry" in p.relative_to(base).parts[:1]:
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        if not _PLACEHOLDER.search(text):
            continue
        if p.suffix == ".jsonl":
            docs = [json.loads(line) for line in text.splitlines() if line.strip()]
        elif p.suffix == ".json":
            docs = [json.loads(text)]
        else:
            docs = list(yaml.load_all(text, Loader=loader))
        for d in docs:
            for key, value in _key_paths(d):
                if not _PLACEHOLDER.search(value):
                    continue
                if any(re.search(rx, key) for rx in list(EXPANDED) + list(RECORDED)):
                    continue
                out.append((str(p.relative_to(root)), key, value[:120]))
    return out


def test_every_placeholdered_config_value_sits_under_a_key_a_reader_expands():
    bad = unclassified(REPO)
    assert not bad, ("committed config values hold a site placeholder under a key no reader expands; teach the reader "
                     "(sisteer.site.expand_path) and add the key to EXPANDED, or add it to RECORDED if no code opens it: "
                     + "; ".join(f"{f}: {k} = {v!r}" for f, k, v in bad[:20]))


def test_the_ratchet_flags_a_placeholder_under_an_unknown_key(tmp_path):
    (tmp_path / "configs" / "registry").mkdir(parents=True)
    (tmp_path / "configs" / "x.yaml").write_text("pool:\n  store_dir: $WORK/s\nout_dir: ${SHARED}/out\nnote_dir: /plain\n")
    (tmp_path / "configs" / "p.json").write_text(json.dumps({"m": {"cell": {"source": "cluster_a:$SHARED/x"}}}))
    (tmp_path / "configs" / "registry" / "seed.yaml").write_text("anything: $STATE/x\n")
    assert unclassified(tmp_path) == [("configs/x.yaml", "out_dir", "${SHARED}/out")]
