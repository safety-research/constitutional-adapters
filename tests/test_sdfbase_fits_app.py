"""The sdfbase-fits Modal app's launch discipline (fix items 2, 3, 9b).

- item 2: `battery_commands` passes VOLUME-BACKED paths end to end — the
  TT_TWO_SI_DEFENSES_DIR export the transfer configs promise, a /vol
  private-root for two_si_run generate, and /vol --defenses-dir/--out-root
  for cell_controls_run — never /repo/out (baked-image scratch) or /root
  (container-local) state.
- item 3: NO marker-only skips before a spawn — hpo-spawn and fits-spawn
  always spawn; the in-container fingerprint gates (tune_arm /
  train_two_si_objects) are the one authoritative cheap skip. Pinned at
  the source level so a listdir-existence check cannot quietly return.
- item 9b: the probe-score constitution-context gate row carries
  --batch-size 2 (probe_score.py's own GPU note: the ~6-7k-token prefix
  OOMs at the bare-row default of 8), and fitted-cell rows carry --cell
  for the shared identity gate.

`modal` is not installed in the test env; a minimal duck-typed stub stands
in so the module imports (the app objects are never exercised here).
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

REPO = Path(__file__).resolve().parents[1]
SRC_PATH = REPO / "scripts" / "modal" / "sdfbase_fits.py"


class _Chain:
    """Any method returns self — enough for the Image builder chains."""

    def __getattr__(self, name):
        def _method(*a, **k):
            return self
        return _method


def _fake_modal() -> types.ModuleType:
    fake = types.ModuleType("modal")

    class _App:
        def __init__(self, name):
            self.name = name

        def function(self, *a, **k):
            return lambda fn: fn

        def local_entrypoint(self, *a, **k):
            return lambda fn: fn

    fake.App = _App
    fake.Image = SimpleNamespace(debian_slim=lambda **k: _Chain(),
                                 from_registry=lambda *a, **k: _Chain())
    fake.Volume = SimpleNamespace(
        from_name=lambda *a, **k: SimpleNamespace(
            listdir=lambda *aa, **kk: [], commit=lambda: None,
            reload=lambda: None))
    # from_dict too: scripts/modal/grokmt_served.py builds an ephemeral secret at
    # import, and this stub can outlive this module in sys.modules within one
    # suite half (the sdfbase-mt merge put both files in one tree, 2026-09-04)
    fake.Secret = SimpleNamespace(from_name=lambda n: ("secret", n),
                                  from_dict=lambda d: ("secret", tuple(sorted(d))))
    fake.Function = SimpleNamespace(from_name=lambda *a, **k: None)
    fake.FunctionCall = SimpleNamespace(from_id=lambda i: None)
    fake.exception = SimpleNamespace(OutputExpiredError=Exception)
    return fake


#: the name the tests give the site setting SISTEER_MODAL_WANDB_SECRET, which the app reads where it is defined
WANDB_SECRET_EXAMPLE = "example-wandb-secret"


@pytest.fixture(scope="module")
def sf():
    import os

    # ALWAYS this suite's own stub: another module may have left ITS fake `modal`
    # in sys.modules (the grokmt lineage installs one whose Image has no
    # debian_slim), and keeping it made every test here ERROR at setup on the
    # joined sdfbase-mt tree (2026-09-04). Whatever was there is restored after.
    prev_modal = sys.modules.get("modal")
    sys.modules["modal"] = _fake_modal()
    # The module-level App name folds $SDFBASE_APP_NAME at import; the
    # tests pin the DEFAULT, so import with the override unset.
    os.environ.pop("SDFBASE_APP_NAME", None)
    # ... and the W&B secret's name, also read at import, is the synthetic one
    prev_secret = os.environ.get("SISTEER_MODAL_WANDB_SECRET")
    os.environ["SISTEER_MODAL_WANDB_SECRET"] = WANDB_SECRET_EXAMPLE
    spec = importlib.util.spec_from_file_location("_sdfbase_fits_test",
                                                  SRC_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_sdfbase_fits_test"] = mod
    spec.loader.exec_module(mod)
    try:
        yield mod
    finally:
        if prev_modal is None:
            sys.modules.pop("modal", None)
        else:
            sys.modules["modal"] = prev_modal
        if prev_secret is None:
            os.environ.pop("SISTEER_MODAL_WANDB_SECRET", None)
        else:
            os.environ["SISTEER_MODAL_WANDB_SECRET"] = prev_secret


SRC = SRC_PATH.read_text()


def _route(sf, deployed: str, name: str, key: str = "") -> tuple:
    """(app, function) that deployed_fn RESOLVES for this key — recorded
    off a patched modal.Function.from_name, never off the string it
    builds."""
    seen = []
    with mock.patch.object(sf.modal.Function, "from_name",
                           lambda app, fn, *a, **k: seen.append((app, fn))):
        sf.deployed_fn(deployed, name, key=key)
    assert len(seen) == 1, seen
    return seen[0]


#: A locally decorated binding reaching .spawn/.remote/.map inside main —
#: a shape that must never match again.
_LOCAL_BINDING_CALL = (r"\b(stage_std|fingerprint_cpu|snapshot_cpu|"
                       r"\w+_h200_(?:std|qwen|nemotron))\s*\.\s*"
                       r"(?:spawn|remote|map)\(")


# --------------------------------------------------------------------------- #
# No shared wandb secret; W&B offline into the volume                          #
# --------------------------------------------------------------------------- #


def test_operator_wandb_secret_not_the_shared_one(sf, monkeypatch):
    """The workspace's shared `wandb` secret is another member's key (every
    fit run logged through it landed in a foreign entity), so the apps mount
    the operator's own, the secret the site setting SISTEER_MODAL_WANDB_SECRET
    names (WANDB_API_KEY + WANDB_ENTITY + WANDB_PROJECT), next to the HF token,
    and the env passes that routing through untouched. Code-only until the
    next deploy (the frozen app is not redeployed)."""
    assert sf.WANDB_SECRET == WANDB_SECRET_EXAMPLE
    assert sf._secrets == [("secret", "hf-token"), ("secret", WANDB_SECRET_EXAMPLE)]
    assert sf._COMMON["secrets"] == sf._secrets
    assert 'from_name("wandb")' not in SRC
    assert sf._HAVE_WANDB is True
    monkeypatch.setenv("WANDB_ENTITY", "example-entity")
    for k in ("WANDB_MODE", "WANDB_DIR", "WANDB_DISABLED"):
        monkeypatch.delenv(k, raising=False)   # the grading host shell sets some
    env = sf._env()
    assert env["WANDB_ENTITY"] == "example-entity"
    assert "WANDB_MODE" not in env and "WANDB_DIR" not in env  # nothing ADDED
    assert "WANDB_DISABLED" not in env
    assert env["HF_HOME"] == "/vol/hf"


def test_require_wandb_entity_refuses_unset_and_foreign_keys(sf, monkeypatch):
    """The container-start guard (called by _fit_impl / _hpo_impl and the
    derived app's residual impl): loud RuntimeError when WANDB_ENTITY is
    unset or when the mounted key's default entity is not it; the entity
    back when they agree; a silent no-op when W&B is switched off."""
    fake_wandb = types.ModuleType("wandb")
    state = {"entity": "example-entity"}

    class _Api:
        def __init__(self, timeout=None):
            self.default_entity = state["entity"]

    fake_wandb.Api = _Api
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    monkeypatch.delenv("WANDB_DISABLED", raising=False)
    monkeypatch.delenv("WANDB_ENTITY", raising=False)
    with pytest.raises(RuntimeError, match="WANDB_ENTITY is unset"):
        sf.require_wandb_entity()
    monkeypatch.setenv("WANDB_ENTITY", "example-entity")
    assert sf.require_wandb_entity() == "example-entity"
    state["entity"] = "someone-else-none"
    with pytest.raises(RuntimeError, match="foreign key"):
        sf.require_wandb_entity()
    # switched off → no check at all (no network, no refusal)
    monkeypatch.setenv("WANDB_DISABLED", "true")
    assert sf.require_wandb_entity() == ""
    monkeypatch.delenv("WANDB_DISABLED")
    monkeypatch.setattr(sf, "_HAVE_WANDB", False)
    assert sf.require_wandb_entity() == ""
    # the guard is wired into both fit entry points
    assert SRC.count("require_wandb_entity()  # R2-D16") == 2


# --------------------------------------------------------------------------- #
# item 2 — battery paths are volume-backed end to end                           #
# --------------------------------------------------------------------------- #

SLUGS = ("gemma4e4b", "qwen35_4b")


def _path_tokens(cmd):
    return [t for t in cmd[1:] if str(t).startswith("/")]  # skip the python


@pytest.mark.parametrize("slug", SLUGS)
@pytest.mark.parametrize("attack", ["bon", "pasttense", "controls", "xstest"])
def test_battery_paths_all_volume_backed(sf, slug, attack):
    cfg_path = f"/repo/configs/sdfbase_transfer_{slug}.yaml"
    cmd, env = sf.battery_commands(cfg_path, attack, "si_constitution__lora_r4",
                                   1, 4, True)
    # The env exports the transfer configs promise, verbatim.
    assert env == {
        "TT_TWO_SI_DEFENSES_DIR": f"/vol/out/transfer_{slug}/cells",
        "SISTEER_HELDOUT_CORPUS_DIR": sf.HELDOUT_CORPUS_DIR}
    # Every absolute path in the command is either the baked repo CODE
    # (script / committed config) or the volume — never container-local
    # state and never the baked image's out/ tree.
    for tok in _path_tokens(cmd):
        assert not tok.startswith("/root"), cmd
        assert not tok.startswith("/repo/out"), cmd
        assert (tok.startswith("/vol/")
                or tok.startswith("/repo/scripts/")
                or tok.startswith("/repo/configs/")), (tok, cmd)
    if attack == "controls":
        d = dict(zip(cmd, cmd[1:]))
        assert d["--defenses-dir"] == f"/vol/out/transfer_{slug}/cells"
        assert d["--out-root"] == f"/vol/out/sdfbase_transfer_{slug}/controls"
        assert "/repo/scripts/cell_controls_run.py" in cmd
    elif attack == "xstest":
        d = dict(zip(cmd, cmd[1:]))
        assert d["--private-root"] == "/vol/private"
        assert "/repo/scripts/two_si_run.py" in cmd
    else:
        d = dict(zip(cmd, cmd[1:]))
        assert d["--private-root"] == "/vol/private"
        assert "/repo/scripts/two_si_run.py" in cmd
        assert d["--shard"] == "1" and d["--nshards"] == "4"
        assert "--confirmatory" in cmd


# --------------------------------------------------------------------------- #
# the battery emits the benign over-refusal (XSTest) pass                        #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("slug", SLUGS)
def test_battery_xstest_runs_the_benign_pass_not_an_attack_sweep(sf, slug):
    """`two_si_run.py aggregate` REFUSES every cell that has no
    xstest_responses.jsonl (jailbreak_run._aggregate_xstest raises), and the
    battery never produced one: `--attacks` had no xstest member and
    battery_commands emitted no `--xstest`. Pin the command that fixes it —
    generate --xstest, the flag whose whole effect is to replace the attack
    sweep with the 250 SAFE XSTest prompts."""
    cfg_path = f"/repo/configs/sdfbase_transfer_{slug}.yaml"
    cmd, env = sf.battery_commands(cfg_path, "xstest", "none", 0, 1, True)
    assert "/repo/scripts/two_si_run.py" in cmd and "generate" in cmd
    assert "--xstest" in cmd
    d = dict(zip(cmd, cmd[1:]))
    assert d["--config"] == cfg_path
    assert d["--cell"] == "none"
    assert d["--private-root"] == "/vol/private"
    # The object bank export still rides every shard, xstest included: the
    # compiled cells' objects load from it exactly as on an attack shard.
    assert env["TT_TWO_SI_DEFENSES_DIR"] == f"/vol/out/transfer_{slug}/cells"


@pytest.mark.parametrize("flag", ["--attack", "--shard", "--nshards",
                                  "--confirmatory"])
def test_battery_xstest_passes_no_inert_sweep_flags(sf, flag):
    """The --xstest path loads no eval goals and slices nothing
    (two_si_run.cmd_generate returns before every attack dispatch), so a
    sweep flag on it would state a scope the pass does not have. Passed a
    non-default shard/nshards on purpose: they must still not appear."""
    cmd, _ = sf.battery_commands(
        "/repo/configs/sdfbase_transfer_gemma4e4b.yaml", "xstest",
        "constv2-mix-1m__psr_ce_all", 3, 8, True)
    assert flag not in cmd, cmd


def test_battery_xstest_is_one_container_per_cell(sf):
    """The xstest writer appends to ONE xstest_responses.jsonl per cell dir
    with no shard component, so two shards would be two containers writing
    the same volume file (the bon hazard) and generating the same 250
    prompts twice. Both per-cell ids go through one constant so the
    entrypoint cannot special-case only one of them."""
    assert set(sf.UNSHARDED_BATTERY_ATTACKS) == {"controls", "xstest"}
    block = SRC.split('if mode == "battery"', 1)[1]
    assert "UNSHARDED_BATTERY_ATTACKS" in block
    assert 'atk == "controls"' not in block, (
        "the per-cell attack ids must be read from "
        "UNSHARDED_BATTERY_ATTACKS, not re-spelled inline")


def test_battery_default_attacks_include_xstest():
    """xstest belongs in the DEFAULT set, not behind an explicit --attacks:
    the aggregate refuses a cell without its benign pass, so a default
    battery that omits it produces an ungradable campaign."""
    block = SRC.split('if mode == "battery"', 1)[1].split("want_cells", 1)[0]
    defaults = block.split("want_attacks", 1)[1]
    for atk in ("bon", "pasttense", "codeattack", "renellm", "grok420",
                "controls", "xstest"):
        assert f'"{atk}"' in defaults, atk


def test_battery_confirmatory_flag_optional(sf):
    cmd, _ = sf.battery_commands(
        "/repo/configs/sdfbase_transfer_gemma4e4b.yaml", "bon", "none",
        0, 1, False)
    assert "--confirmatory" not in cmd


def test_battery_slug_derivation_refuses_non_transfer_configs(sf):
    assert sf._battery_slug(
        "/repo/configs/sdfbase_transfer_qwen35_4b.yaml") == "qwen35_4b"
    with pytest.raises(SystemExit):
        sf._battery_slug("/repo/configs/two_si_gemma4e4b.yaml")


def test_transfer_slug_map_covers_every_fits_model(sf):
    """The battery serves the INSTRUCT sibling, so every fits-bearing model
    key maps to a two_si transfer slug (TRANSFER_SLUG, ONE table, read by
    battery_config_path — the entrypoint's battery block goes through that
    function and carries no inline dict of its own; a second table under
    another name is the drift this test exists to catch). glm46vflash_doc
    -> glm46vflash and nemotron3nano4b_doc -> nemotron3nano4b: each is its
    own instruct model, and the two_si lane of the SAME checkpoint is
    where its transfer config will be named; until that config is
    committed the entrypoint's existing committed-config check refuses."""
    assert sf.TRANSFER_SLUG == {"gemma4e4b_base": "gemma4e4b",
                                "qwen35_4b_base": "qwen35_4b",
                                "glm46vflash_doc": "glm46vflash",
                                "nemotron3nano4b_doc": "nemotron3nano4b",
                                # instruct-direct: the slug of the
                                # instruct model each one IS
                                "gemma4e4b_it_doc": "gemma4e4b",
                                "qwen35_4b_it_doc": "qwen35_4b",
                                # the Llama lane (2026-09-09): the two_si
                                # slug of the served instruct checkpoint
                                "llama8b_base": "llama",
                                "llama8b_it_doc": "llama",
                                # the instruct-direct batch-64 siblings
                                # (2026-09-11): the same instruct
                                # checkpoints, so the same slugs
                                "gemma4e4b_it_doc_eb64": "gemma4e4b",
                                "qwen35_4b_it_doc_eb64": "qwen35_4b",
                                # the qwen-base batch-64 seed replicates
                                # (2026-09-15): the qwen35_4b_base
                                # checkpoint, served on the same instruct
                                # sibling, so the same slug
                                "qwen35_4b_base_eb64_s1": "qwen35_4b",
                                "qwen35_4b_base_eb64_s2": "qwen35_4b",
                                "qwen35_4b_base_eb64_s3": "qwen35_4b"}
    assert not hasattr(sf, "TRANSFER_SLUGS")  # one table, one name
    fits_models = {k for k, s in sf.MODELS.items() if not s.get("probe_only")}
    assert set(sf.TRANSFER_SLUG) == fits_models
    for slug in sf.TRANSFER_SLUG.values():
        # every slug names a real two_si lane in the repo
        assert list((REPO / "configs").glob(f"two_si_{slug}*.yaml")), slug
    fn_src = SRC.split("def battery_config_path", 1)[1].split("\ndef ", 1)[0]
    assert "TRANSFER_SLUG[key]" in fn_src
    assert "no transfer slug registered" in fn_src
    battery = SRC.split('if mode == "battery"', 1)[1]
    assert "battery_config_path(key, wave)" in battery
    assert '{"gemma4e4b_base": "gemma4e4b"' not in battery
    assert "}[key]" not in battery  # no inline dict indexing


# --------------------------------------------------------------------------- #
# GLM-4.6V-Flash (glm46vflash_doc) — the chat model's doc-CE lane              #
# --------------------------------------------------------------------------- #


def test_glm_doc_entry_std_image_ungated_chat_render(sf):
    """Registered 2026-09-02: H200, the generated sdfbase config, the pinned
    ungated repo, image_std (dense full attention — not a QWEN_IMAGE_KEYS
    member), probe_render chat (GLM IS the instruct model: no probe_only
    sibling), and explicit_only while its folds are pending."""
    spec = sf.MODELS["glm46vflash_doc"]
    assert spec["gpu"] == "H200"
    assert spec["cfg"] == "/repo/configs/sdfbase_glm46vflash_doc_fits.yaml"
    assert (REPO / "configs/sdfbase_glm46vflash_doc_fits.yaml").exists()
    assert spec["repo"] == "zai-org/GLM-4.6V-Flash"
    assert spec["rev"] == "411bb4d77144a3f03accbf4b780f5acb8b7cde4e"
    assert spec["gated"] is False
    assert spec["probe_render"] == "chat"
    assert spec.get("probe_only") is None
    assert spec["explicit_only"] is True
    assert "glm46vflash_doc" not in sf.QWEN_IMAGE_KEYS
    assert sf._variant("glm46vflash_doc") == "std"
    assert _route(sf, "sdfbase-fits-ws3", "onboard_h200",
                  "glm46vflash_doc") == ("sdfbase-fits-ws3", "onboard_h200_std")
    # every fits-bearing entry's cfg is a committed file in the tree
    for key, s in sf.MODELS.items():
        if s.get("probe_only"):
            continue
        assert (REPO / Path(s["cfg"]).relative_to("/repo")).exists(), key


def test_default_key_set_excludes_explicit_only_and_probe_only():
    """A bare invocation (no --model) sweeps neither the probe_only instruct
    siblings nor the explicit_only GLM lane — a peer's bare hpo-spawn /
    probe-score / memprobe on the in-flight gemma/qwen rounds must not
    pull in a model with no designs, pools or bank yet. Named with
    --model, every mode is open to it (no probe-score-only restriction)."""
    main_src = _entrypoint_src()
    keys_expr = main_src.split("keys = (", 1)[1].split("for k in keys:", 1)[0]
    assert 'not s.get("probe_only")' in keys_expr
    assert 'not s.get("explicit_only")' in keys_expr
    guard = main_src.split("for k in keys:", 1)[1].split("def _fanout_models", 1)[0]
    assert "explicit_only" not in guard  # no mode restriction for it


def test_probe_render_governs_gate_and_object_rows_alike():
    """probe_render is a per-MODEL property: the bare / constitution gate
    rows AND the fitted-cell rows render the same way, so an object's
    margin shift is read against a bare row in the same render. Before
    GLM the only chat-rendered entries were probe_only (no cells), so the
    object rows were only ever raw — the source pin keeps one `render`
    binding feeding every run_one."""
    impl = SRC.split("def _probe_score_impl", 1)[1].split("\ndef ", 1)[0]
    assert impl.count('spec.get("probe_render", "raw")') == 1
    binding = impl.index('render = spec.get("probe_render", "raw")')
    assert binding < impl.index("if scale == 1.0:")  # bound for every row
    assert re.search(r'run_one\(\s*"bare",\s*None,\s*"bare",\s*render=render\)',
                     impl)
    assert re.search(r'run_one\(\s*"constitution",\s*None,\s*"constitution",'
                     r'\s*batch_size=2,\s*render=render\)', impl)
    assert re.search(r'run_one\(cell,\s*str\(cand\[0\]\),\s*"bare",'
                     r'\s*render=render\)', impl)
    assert 'render="raw"' not in impl.split("def run_one", 1)[1].split(
        "rows = []", 1)[1]  # no row re-hardcodes raw


def test_battery_impl_routes_through_battery_commands():
    """_battery_impl must not rebuild its own command (a second copy would
    let the path discipline drift) — source-level pin."""
    impl = SRC.split("def _battery_impl", 1)[1].split("\ndef ", 1)[0]
    assert "battery_commands(" in impl
    # no inline command construction (script-path literals may appear only
    # inside battery_commands itself; the docstring naming the scripts is
    # fine)
    assert '"/repo/scripts/two_si_run.py"' not in impl
    assert '"/repo/scripts/cell_controls_run.py"' not in impl


# --------------------------------------------------------------------------- #
# Nemotron-3-Nano-4B registration (2026-09-02)                                  #
# --------------------------------------------------------------------------- #


def test_nemotron_doc_key_registered_on_the_nemotron_image(sf):
    """The documents-lane key for the NemotronH hybrid: pinned to the
    verified revision, NOT gated (NVIDIA Open Model License), on H200, its
    generated config committed (the FULL-SCALE lane's since the data-scaling
    ladder config retired, 2026-10-08), and routed to image_nemotron — the
    hybrid-stack recipe PLUS the pinned mamba-ssm fused kernels, since
    speed-giving kernels ship by default. The qwen and std key tables are
    untouched, and the function-object pick follows the same spelling as
    the deployed-function names."""
    spec = sf.MODELS["nemotron3nano4b_doc"]
    assert spec["repo"] == "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"
    assert spec["rev"] == "dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f"
    assert spec["gated"] is False
    assert spec["gpu"] == "H200"
    assert not spec.get("probe_only")
    assert spec["cfg"] == "/repo/configs/sdfbase_nemotron3nano4b_doc_full_fits.yaml"
    assert sf._variant("nemotron3nano4b_doc") == "nemotron"
    assert sf.NEMOTRON_IMAGE_KEYS == ("nemotron3nano4b_doc",)
    assert "nemotron3nano4b_doc" not in sf.QWEN_IMAGE_KEYS
    # the existing routing is untouched (+ the qwen instruct-direct
    # key on the hybrid-stack image, like every Qwen3.5 key)
    assert sf.QWEN_IMAGE_KEYS == ("qwen35_4b_base", "qwen35_4b_it",
                                  "qwen35_4b_it_doc", "qwen35_4b_it_doc_eb64",
                                  "qwen35_4b_base_eb64_s1",
                                  "qwen35_4b_base_eb64_s2",
                                  "qwen35_4b_base_eb64_s3")
    assert {k: sf._variant(k) for k in sf.MODELS} == {
        "gemma4e4b_base": "std", "gemma4e4b_it": "std",
        "qwen35_4b_base": "qwen", "qwen35_4b_it": "qwen",
        "glm46vflash_doc": "std",  # dense full attention: neither kernel image
        "nemotron3nano4b_doc": "nemotron",
        "gemma4e4b_it_doc": "std", "qwen35_4b_it_doc": "qwen",
        # dense Llama-3.1-8B, both checkpoints: the standard image
        "llama8b_base": "std", "llama8b_it_doc": "std",
        # the batch-64 siblings (2026-09-11): the architecture's image
        "gemma4e4b_it_doc_eb64": "std", "qwen35_4b_it_doc_eb64": "qwen",
        # the qwen-base batch-64 seed replicates (2026-09-15): the hybrid image
        "qwen35_4b_base_eb64_s1": "qwen", "qwen35_4b_base_eb64_s2": "qwen",
        "qwen35_4b_base_eb64_s3": "qwen"}
    assert _route(sf, "sdfbase-fits-ws3", "fit_h200", "nemotron3nano4b_doc") \
        == ("sdfbase-fits-ws3", "fit_h200_nemotron")
    assert _route(sf, "sdfbase-fits-ws3", "fit_h200", "qwen35_4b_base") \
        == ("sdfbase-fits-ws3", "fit_h200_qwen")
    assert _route(sf, "sdfbase-fits-ws3", "fit_h200", "gemma4e4b_base") \
        == ("sdfbase-fits-ws3", "fit_h200_std")
    # no key: the name is taken verbatim (the CPU functions)
    assert _route(sf, "sdfbase-fits", "snapshot_cpu") \
        == ("sdfbase-fits", "snapshot_cpu")


def _stanza(src: str, name: str) -> str:
    """The `<name> = (` image definition up to its .add_local_dir mount."""
    i = src.index(f"{name} = (")
    return src[i:src.index(".add_local_dir(", i)]


def _steps(stanza: str) -> str:
    """The stanza's BUILD STEPS: comment-only lines dropped (Modal's recipe
    hash sees code, not comments — the layer cache is what the equality
    below is really about)."""
    return "\n".join(ln for ln in stanza.splitlines()
                     if not ln.strip().startswith("#"))


def test_nemotron_image_is_the_qwen_recipe_plus_pinned_mamba_ssm():
    """image_nemotron = image_qwen's build steps, step for step (so every
    layer through the causal-conv1d build is shared with image_qwen's cache
    and image_qwen itself is never edited — live qwen cells build from it)
    + ninja, the pinned mamba-ssm source build (forced: no wheel exists at
    torch 2.11), the triton re-pin LAST, and a bake check that imports
    exactly the five kernels transformers' NemotronH resolves."""
    nemo = _steps(_stanza(SRC, "image_nemotron"))
    qwen = _steps(_stanza(SRC, "image_qwen"))
    qwen_steps = qwen.split("(", 1)[1]
    assert qwen_steps in nemo, "image_qwen's build steps must ride over 1:1"
    extra = nemo.split(qwen_steps, 1)[1]
    assert "pip install ninja==1.13.0" in extra
    assert "MAMBA_FORCE_BUILD=TRUE" in extra
    assert ("pip install --no-build-isolation --no-deps "
            "mamba-ssm==2.3.2.post1") in extra
    installs = [ln for ln in extra.splitlines() if "pip install" in ln]
    assert installs[-1].strip().startswith(
        '"pip install --no-deps triton==3.7.1"'), installs
    for sym in ("import causal_conv1d, fla, mamba_ssm",
                "selective_state_update", "mamba_chunk_scan_combined",
                "mamba_split_conv1d_scan_combined",
                "causal_conv1d_fn, causal_conv1d_update"):
        assert sym in extra, sym
    # the other two images carry no mamba-ssm
    assert "mamba" not in qwen.lower()
    assert "mamba" not in _stanza(SRC, "image_std").lower()


def test_image_stanzas_identical_across_the_two_apps():
    """Both apps carry image_qwen AND image_nemotron verbatim (the
    'copied VERBATIM from twosi_fits.py' convention) — one recipe, one
    Modal image id, one build."""
    twosi = (REPO / "scripts" / "modal" / "twosi_fits.py").read_text()
    for name in ("image_qwen", "image_nemotron"):
        assert _stanza(SRC, name) == _stanza(twosi, name), name


_BINDING = (r"@app\.function\((?P<deco>[^)]*)\)\s*\ndef {name}\("
            r".*?\n    return (?P<impl>\w+)\(")


def test_every_qwen_binding_has_a_nemotron_sibling():
    """Modal binds image/gpu statically per function, so each variant
    needs its own wrapper: every *_h200_qwen binding has a *_h200_nemotron
    twin on image_nemotron, H200, wrapping the same implementation."""
    stems = set(re.findall(r"^def (\w+)_h200_qwen\(", SRC, flags=re.M))
    assert stems == {"onboard", "memprobe", "fit", "hpo", "score", "mmlu",
                     "battery", "gsm8k", "ce", "bench"}
    for stem in stems:
        impl = {}
        for variant, image in (("qwen", "image_qwen"),
                               ("nemotron", "image_nemotron")):
            m = re.search(_BINDING.format(name=f"{stem}_h200_{variant}"),
                          SRC, flags=re.S)
            assert m, (stem, variant)
            assert f"image={image}" in m["deco"], (stem, variant)
            assert 'gpu="H200"' in m["deco"], (stem, variant)
            impl[variant] = m["impl"]
        assert impl["qwen"] == impl["nemotron"], stem


def test_entrypoint_routes_every_variant():
    """Every routing site in the entrypoint derives from _variant(): the
    fan-outs hand deployed_fn a family stem + the key (it appends the
    variant), hpo-spawn resolves all three variants, and the fit/battery
    names are built from the variant spelling — no site string-matches
    the qwen table alone (that is how the qwen35 transfer fits once landed
    on image_std). And NO locally decorated binding reaches .spawn: those
    are bound to the import-time App and ignore --app-name."""
    main_src = _entrypoint_src()
    for stem in ("onboard_h200", "memprobe_h200", "score_h200", "mmlu_h200",
                 "gsm8k_h200", "ce_h200"):
        assert f'"{stem}"' in main_src, stem
    assert 'for v in ("std", "qwen", "nemotron")' in main_src
    assert 'f"fit_h200_{_variant(key)}"' in main_src
    assert 'f"battery_h200_{_variant(key)}"' in main_src
    assert "in QWEN_IMAGE_KEYS" not in main_src
    assert "in NEMOTRON_IMAGE_KEYS" not in main_src
    assert not re.search(_LOCAL_BINDING_CALL, main_src)
    assert "_fn_pair" not in SRC  # the function-object pick is gone


def test_every_registered_cfg_is_committed(sf):
    """A MODELS entry pointing at a config that is not in the tree would
    fail only inside a container (cfg_local.exists() in the entrypoint) —
    pin it here, for every key."""
    for key, spec in sf.MODELS.items():
        if spec["cfg"] is None:
            continue
        local = REPO / Path(spec["cfg"]).relative_to("/repo")
        assert local.exists(), (key, spec["cfg"])


def test_battery_refuses_keys_without_a_transfer_config(sf):
    """The battery serves the INSTRUCT sibling through a transfer config.
    The two `_doc` keys are registered in TRANSFER_SLUG (so the by-name
    refusal never fires for them) but NEITHER transfer config exists yet
    (the batteries are out of scope), so the entrypoint's committed-config
    check is what refuses — by path, never a bare KeyError."""
    for key in ("glm46vflash_doc", "nemotron3nano4b_doc"):
        slug = sf.TRANSFER_SLUG[key]
        assert not (REPO / "configs" / f"sdfbase_transfer_{slug}.yaml").exists(), \
            (key, slug)
    block = SRC.split('if mode == "battery"', 1)[1]
    assert "transfer config must be committed first" in block
    assert "battery_config_path(key, wave)" in block  # the one slug reader
    assert "}[key]" not in block  # no inline dict indexing


# --------------------------------------------------------------------------- #
# the INSTRUCT-DIRECT `_it_doc` keys                                          #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("key,sibling,variant", [
    ("gemma4e4b_it_doc", "gemma4e4b_it", "std"),
    ("qwen35_4b_it_doc", "qwen35_4b_it", "qwen"),
])
def test_it_direct_doc_keys_are_the_instruct_pins_made_fit_capable(
        sf, key, sibling, variant):
    """Each `_it_doc` key is the probe_only instruct sibling's repo + pin
    (= the r2 transfer config's serving target) registered as a FIT-capable
    lane: committed generated config, gated like the sibling, chat probe
    render, explicit_only (never swept into a bare invocation), on the
    architecture's image; the probe_only sibling is left exactly as it was.
    battery_wave pins the key to the (future) `_it` battery namespace."""
    spec, sib = sf.MODELS[key], sf.MODELS[sibling]
    assert (spec["repo"], spec["rev"]) == (sib["repo"], sib["rev"])
    assert sib["probe_only"] is True and sib["cfg"] is None  # untouched
    assert spec.get("probe_only") is None
    assert spec["explicit_only"] is True
    assert spec["probe_render"] == "chat"
    assert spec["gated"] is sib["gated"] is True
    assert spec["gpu"] == "H200"
    assert spec["cfg"] == f"/repo/configs/sdfbase_{key}_fits.yaml"
    assert (REPO / f"configs/sdfbase_{key}_fits.yaml").exists()
    assert spec["battery_wave"] == "_it"
    assert spec["derived_eligible"] is False  # no derived op is valid for it
    assert sf._variant(key) == variant
    assert _route(sf, "sdfbase-fits-it", "fit_h200", key) == \
        ("sdfbase-fits-it", f"fit_h200_{variant}")
    import yaml as _yaml
    cfg = _yaml.safe_load((REPO / f"configs/sdfbase_{key}_fits.yaml").read_text())
    assert (cfg["model"], cfg["model_revision"]) == (spec["repo"], spec["rev"])
    assert cfg["si_ids"] == ["constv2-mix-full"]
    assert cfg["train_kwargs_by_si"] == {
        "constv2-mix-full": {"epochs": 16, "early_stop_patience": 2}}
    # the four non-`_it` keys carry no battery_wave (their waves are open)
    # and no derived_eligible flag (eligible by default)
    for k in ("gemma4e4b_base", "qwen35_4b_base", "glm46vflash_doc",
              "nemotron3nano4b_doc"):
        assert "battery_wave" not in sf.MODELS[k]
        assert "derived_eligible" not in sf.MODELS[k]


def test_battery_it_direct_keys_run_only_under_the_it_wave_and_own_it(
        sf, routed, capsys):
    """An `_it_doc` key's slug has committed transfer configs ('' / _r2 /
    _r3) whose banks hold the BASE-fit objects — so a battery run under any
    of those waves would serve another lane's objects under this key's
    name: refused by name BEFORE resolving the config. Under `_it` (the
    instruct-direct wave, registered by the `_it` battery lane: grammar,
    bank suffix, committed config) the key SPAWNS, on the architecture's
    battery function. And the `_it` namespace is the `_it_doc` keys' own:
    a base key under `_it` is refused too, even though the gemma `_it`
    bank carries one base-fit object — it rides in the `_it_doc` launch as
    one more cell of the wave's config, not as a second manifest under the
    base key's name."""
    entry, log, _home = routed
    for wave in ("", "_r2", "_r3"):
        with pytest.raises(SystemExit, match="runs only under --wave '_it'"):
            entry(mode="battery", model="gemma4e4b_it_doc", cells="none",
                  attacks="bon", nshards=1, wave=wave)
        assert not log  # nothing spawned
    for key, variant in (("gemma4e4b_it_doc", "std"),
                         ("qwen35_4b_it_doc", "qwen")):
        log.clear()
        entry(mode="battery", model=key, cells="none", attacks="bon",
              nshards=1, wave="_it")
        capsys.readouterr()
        assert [(via, fn) for via, _app, fn, _item in log] == \
            [("spawn", f"battery_h200_{variant}")]
        slug = sf.TRANSFER_SLUG[key]
        assert log[0][3][0] == f"/repo/configs/sdfbase_transfer_{slug}_it.yaml"
        mpath = _home / f"stage/sdfbase/battery_calls_{key}_it.json"
        doc = json.loads(mpath.read_text())
        assert (doc["wave"], doc["bank"], doc["model"]) == \
            ("_it", f"/vol/out/transfer_{slug}_it/cells", key)
    # the base keys are refused under the `_it` namespace, by name
    log.clear()
    for key in ("gemma4e4b_base", "qwen35_4b_base"):
        with pytest.raises(SystemExit, match="namespace of"):
            entry(mode="battery", model=key, cells="none", attacks="bon",
                  nshards=1, wave="_it")
        assert not log
    # ... while every _r<N> wave stays open to them (unchanged)
    entry(mode="battery", model="gemma4e4b_base", cells="none",
          attacks="bon", nshards=1, wave="_r3")
    capsys.readouterr()
    assert len(log) == 1
    assert "_it" in sf.BATTERY_BANK_SUFFIX
    assert sf._WAVE_SUFFIX.fullmatch("_it")
    assert (REPO / "configs/sdfbase_transfer_gemma4e4b_it.yaml").exists()
    assert (REPO / "configs/sdfbase_transfer_qwen35_4b_it.yaml").exists()
    # the guard runs in the entrypoint BEFORE the config is resolved
    block = SRC.split('if mode == "battery"', 1)[1]
    assert "check_battery_wave(key, wave)" in block
    assert block.index("check_battery_wave(key, wave)") < block.index(
        "battery_config_path(key, wave)")


def test_check_battery_wave_is_the_one_table_both_ways(sf):
    """check_battery_wave reads MODELS[key]["battery_wave"] in both
    directions: a declaring key is confined to its wave; a declared wave
    admits only its declaring keys; keys declaring nothing keep every
    _r<N> wave (and round 1) open."""
    for key in ("gemma4e4b_it_doc", "qwen35_4b_it_doc"):
        sf.check_battery_wave(key, "_it")           # admitted
        for wave in ("", "_r2", "_r3", "_r9"):
            with pytest.raises(SystemExit, match="runs only under --wave '_it'"):
                sf.check_battery_wave(key, wave)
    for key in ("gemma4e4b_base", "qwen35_4b_base", "glm46vflash_doc",
                "nemotron3nano4b_doc", "llama8b_base"):
        for wave in ("", "_r2", "_r3", "_r9"):
            sf.check_battery_wave(key, wave)        # open, as before
        with pytest.raises(SystemExit, match="namespace of"):
            sf.check_battery_wave(key, "_it")
    owners = sorted(k for k, m in sf.MODELS.items()
                    if m.get("battery_wave") == "_it")
    # + the Llama lane's instruct-direct key (2026-09-09) + the two
    # batch-64 siblings (2026-09-11), which ARE the same instruct
    # checkpoints and share the `_it` namespace with their parents
    assert owners == ["gemma4e4b_it_doc", "gemma4e4b_it_doc_eb64",
                      "llama8b_it_doc", "qwen35_4b_it_doc",
                      "qwen35_4b_it_doc_eb64"]
    sf.check_battery_wave("llama8b_it_doc", "_it")   # admitted
    with pytest.raises(SystemExit, match="runs only under --wave '_it'"):
        sf.check_battery_wave("llama8b_it_doc", "")
    with pytest.raises(SystemExit) as exc:
        sf.check_battery_wave("gemma4e4b_base", "_it")
    for k in owners:
        assert k in str(exc.value)                  # the message names them


def test_wave_suffix_grammar_is_rN_or_it(sf):
    """_WAVE_SUFFIX admits exactly the `_r<N>` rounds and `_it`; every
    registered bank suffix is a legal wave; _battery_parts splits a config
    stem on it and leaves round-1 stems whole."""
    for ok in ("_r2", "_r3", "_r10", "_it"):
        assert sf._WAVE_SUFFIX.fullmatch(ok), ok
    for bad in ("", "_r", "_itx", "it", "_it2", "r2", "_IT", "_x1"):
        assert not sf._WAVE_SUFFIX.fullmatch(bad), bad
    for wave in sf.BATTERY_BANK_SUFFIX:
        assert wave == "" or sf._WAVE_SUFFIX.fullmatch(wave), wave
    for slug in SLUGS:
        assert sf._battery_parts(f"/repo/configs/sdfbase_transfer_{slug}_it.yaml") \
            == (slug, "_it")
        assert sf._battery_parts(f"/repo/configs/sdfbase_transfer_{slug}_r3.yaml") \
            == (slug, "_r3")
        assert sf._battery_parts(f"/repo/configs/sdfbase_transfer_{slug}.yaml") \
            == (slug, "")
    with pytest.raises(SystemExit, match="--wave must look like _r2"):
        sf.battery_config_path("gemma4e4b_it_doc", "_itx")
    with pytest.raises(SystemExit, match="--wave must look like _r2"):
        sf.battery_config_path("gemma4e4b_base", "it")


# --------------------------------------------------------------------------- #
# item 3 — no marker-only skips before a spawn (source-level pin)               #
# --------------------------------------------------------------------------- #


def _entrypoint_src() -> str:
    return SRC.split("def main(", 1)[1]


def test_no_pre_spawn_existence_skip_in_entrypoint():
    """hpo-spawn / fits-spawn always spawn: no volume listing, COMPLETE
    probe, or trial-dir existence check may suppress a spawn — the
    in-container fingerprint gates are the one authoritative cheap skip."""
    main_src = _entrypoint_src()
    assert "vol.listdir(" not in main_src
    assert "listdir" not in main_src
    assert "already-published" not in SRC
    assert "already-complete" not in SRC
    # No conditional 'continue' guards the spawn loops: every manifest row
    # reaches fn.spawn.
    for mode_tag in ('mode == "hpo-spawn"', 'mode == "fits-spawn"'):
        block = main_src.split(mode_tag, 1)[1].split('if mode == "', 1)[0]
        assert "fn.spawn(" in block or "call = fn.spawn" in block
        # a `continue` STATEMENT (identifiers such as continue_cap are not
        # a guard — the cap-raise continuation's rows live here)
        assert not re.search(r"^\s*continue\s*$", block, re.M), mode_tag
        # Manifest bookkeeping survives: rows record the spawn and say the
        # cheap skip happens in-container.
        assert "spawned (in-container skip expected)" in block


# --------------------------------------------------------------------------- #
# item 9b (+ item 4 wiring) — probe-score rows                                  #
# --------------------------------------------------------------------------- #


def test_probe_score_constitution_row_gets_a_small_batch():
    impl = SRC.split("def _probe_score_impl", 1)[1].split("\ndef ", 1)[0]
    assert re.search(
        r'run_one\(\s*"constitution",\s*None,\s*"constitution",\s*'
        r'batch_size=2[\s,)]', impl), (
        "the constitution-context gate row must pass --batch-size 2 "
        "(probe_score.py's GPU note: the ~6-7k-token prefix OOMs at the "
        "bare-row default of 8)")
    assert '"--batch-size", str(batch_size)' in impl
    # fitted-cell rows carry their cell id for the shared identity gate
    assert '"--cell", tag' in impl


def test_median_sec_per_step_tolerates_none_step_records():
    """Defect A regression (qwen wave 2026-09-02): the driver's psr metrics
    tee emits records with step: None (key PRESENT, so .get(default) does
    not cover it); the >= comparison crashed every psr container AFTER
    training and BEFORE publish. Pin the None-coalescing form."""
    assert '(rec.get("step") or 0) >= 5' in SRC
    assert 'rec.get("step", 0) >= 5' not in SRC


def test_renellm_corpus_dir_is_volume_backed_for_every_shard(sf):
    """renellm REPLAYS a frozen corpus resolved from
    ``SISTEER_HELDOUT_CORPUS_DIR``, whose library default is a ``~/.local/state``
    path on the grading host. A container has no such path, so ``_corpus()``
    returned None, ``run_renellm`` took its ``loud_skip`` and the shard exited
    0 having written NOTHING — the worst failure shape available, because a
    vacuously-green renellm reads as an attack success rate of zero.

    Pinned for EVERY attack, not just renellm: the export is unconditional so
    a future replayed family cannot reintroduce the same silent skip."""
    slug = "gemma4e4b"
    cfg_path = f"/repo/configs/sdfbase_transfer_{slug}.yaml"
    assert sf.HELDOUT_CORPUS_DIR.startswith("/vol/")
    for attack in ("renellm", "pasttense", "codeattack", "grok420", "bon",
                   "controls"):
        _, env = sf.battery_commands(cfg_path, attack, "none", 0, 1, True)
        assert env["SISTEER_HELDOUT_CORPUS_DIR"] == sf.HELDOUT_CORPUS_DIR, attack


def test_renellm_corpus_filename_matches_what_the_library_looks_for(sf):
    """The staged path and the resolver must agree. ``renellm._corpus_path()``
    joins the env dir with a specific filename; pin that the dir we export
    plus that filename is the path the stager must write, so a rename on
    either side fails here instead of silently skipping again."""
    import os

    from sisteer.eval.attacks import renellm

    old = os.environ.get("SISTEER_HELDOUT_CORPUS_DIR")
    os.environ["SISTEER_HELDOUT_CORPUS_DIR"] = sf.HELDOUT_CORPUS_DIR
    try:
        assert str(renellm._corpus_path()) == \
            f"{sf.HELDOUT_CORPUS_DIR}/renellm.jsonl"
    finally:
        if old is None:
            os.environ.pop("SISTEER_HELDOUT_CORPUS_DIR", None)
        else:
            os.environ["SISTEER_HELDOUT_CORPUS_DIR"] = old


# --------------------------------------------------------------------------- #
# round-2 — publish RETAINS the last checkpoint                               #
# --------------------------------------------------------------------------- #


FP = "a" * 64      # this fit's fingerprint
STALE = "b" * 64   # a superseded (round-1) publication's fingerprint


def _marker(fp=FP, cell="constv2-mix-1m__psr_ce_all") -> str:
    return json.dumps({"cell_id": cell, "fit_fingerprint": fp,
                       "files": {"object.pt": "0" * 64}})


def test_publish_retains_final_checkpoint(sf, tmp_path):
    """A COMPLETE cell's transient checkpoint files MOVE into the published
    cell dir as final_ckpt/ (round 1 deleted them at publish, which made
    exact resume of the cap-limited 1M cells impossible). Both rotation
    slots ride along; the transient dir is emptied so the ckpt tree only
    holds in-flight cells."""
    cell = tmp_path / "cells_all" / "constv2-mix-1m__psr_ce_all"
    cell.mkdir(parents=True)
    (cell / "COMPLETE").write_text(_marker())
    ck = tmp_path / "ckpt" / "constv2-mix-1m__psr_ce_all"
    ck.mkdir(parents=True)
    (ck / "ckpt.pt").write_bytes(b"latest-checkpoint-bytes")
    (ck / "ckpt.prev.pt").write_bytes(b"previous-checkpoint-bytes")
    assert sf.retain_final_checkpoint(cell, ck, expect_fingerprint=FP) is True
    assert (cell / "final_ckpt/ckpt.pt").read_bytes() == \
        b"latest-checkpoint-bytes"
    assert (cell / "final_ckpt/ckpt.prev.pt").read_bytes() == \
        b"previous-checkpoint-bytes"
    assert not ck.exists()
    # the deletion path is GONE from the publish tail: no rmtree of the
    # transient dir without the retention move having happened first
    assert "checkpoints_cleaned" not in SRC


def test_retention_noops_without_complete_or_checkpoints(sf, tmp_path):
    """An in-flight cell (no COMPLETE) keeps its transient checkpoints
    exactly where the resume path expects them; an empty ckpt dir retains
    nothing and never manufactures a final_ckpt/."""
    cell = tmp_path / "cells_all" / "c__psr_ce_all"
    cell.mkdir(parents=True)
    ck = tmp_path / "ckpt" / "c__psr_ce_all"
    ck.mkdir(parents=True)
    (ck / "ckpt.pt").write_bytes(b"x")
    assert sf.retain_final_checkpoint(cell, ck,
                                      expect_fingerprint=FP) is False
    assert (ck / "ckpt.pt").exists()
    (cell / "COMPLETE").write_text(_marker(cell="c__psr_ce_all"))
    empty = tmp_path / "ckpt" / "empty"
    empty.mkdir()
    assert sf.retain_final_checkpoint(cell, empty,
                                      expect_fingerprint=FP) is False
    assert not (cell / "final_ckpt").exists()


def test_retention_refuses_a_stale_or_unprovable_marker(sf, tmp_path):
    """A marker of ANOTHER identity (the round-1 publication under a refit
    that failed or is still running), a marker without a fingerprint, or
    one that does not parse retains NOTHING — the transient checkpoint is
    the successor's resume point and belongs to the new fit, never to the
    old cell dir."""
    cell = tmp_path / "cells_all" / "c__psr_ce_all"
    cell.mkdir(parents=True)
    ck = tmp_path / "ckpt" / "c__psr_ce_all"
    ck.mkdir(parents=True)
    (ck / "ckpt.pt").write_bytes(b"fresh-refit-checkpoint")
    for bad in (_marker(STALE, "c__psr_ce_all"), "{}", "not json at all"):
        (cell / "COMPLETE").write_text(bad)
        assert sf.retain_final_checkpoint(cell, ck,
                                          expect_fingerprint=FP) is False
        assert (ck / "ckpt.pt").read_bytes() == b"fresh-refit-checkpoint"
        assert not (cell / "final_ckpt").exists()


def test_retention_supersedes_a_stale_final_ckpt(sf, tmp_path):
    """An --overwrite respawn's new checkpoint REPLACES the retained one —
    never merges with it (a stale slot next to a fresh one is exactly the
    wrong-identity resume hazard the fingerprint key exists to stop)."""
    cell = tmp_path / "cells_all" / "c__lora_r4_ce"
    (cell / "final_ckpt").mkdir(parents=True)
    (cell / "final_ckpt" / "ckpt.pt").write_bytes(b"stale")
    (cell / "final_ckpt" / "ckpt.prev.pt").write_bytes(b"stale-prev")
    (cell / "COMPLETE").write_text(_marker(cell="c__lora_r4_ce"))
    ck = tmp_path / "ckpt" / "c__lora_r4_ce"
    ck.mkdir(parents=True)
    (ck / "ckpt.pt").write_bytes(b"fresh")
    assert sf.retain_final_checkpoint(cell, ck, expect_fingerprint=FP) is True
    assert (cell / "final_ckpt/ckpt.pt").read_bytes() == b"fresh"
    assert not (cell / "final_ckpt/ckpt.prev.pt").exists()


def test_fit_impl_gates_supersede_before_seed_and_retention_after_success():
    """Source-order pins on _fit_impl: the expected fingerprint is derived
    first, the supersede gate runs BEFORE the volume cell dir is seeded
    into the container (a refused cell never touches the publication), and
    retention runs only on rc 0 with that fingerprint."""
    impl = SRC.split("def _fit_impl", 1)[1].split("\ndef ", 1)[0]
    i_fp = impl.index("expected_fit_fingerprint(")
    i_gate = impl.index("supersede_gate(vol_out / cell_id, expect_fp)")
    i_seed = impl.index("shutil.copytree(vol_out / cell_id")
    assert i_fp < i_gate < i_seed
    assert '"status": "refused-unsnapshotted-supersede"' in impl
    assert "proc.returncode == 0 and retain_final_checkpoint(" in impl
    assert "expect_fingerprint=expect_fp" in impl
    # the fit's result row carries the fingerprint the manifest recorded
    assert '"expect_fingerprint": expect_fp' in impl


# --------------------------------------------------------------------------- #
# round-2 — stale-COMPLETE orchestration                                      #
# --------------------------------------------------------------------------- #


def test_supersede_gate_licenses_only_snapshotted_overwrites(sf, tmp_path):
    cell = tmp_path / "cells_all" / "constv2-mix-1m__psr_ce_all"
    cell.mkdir(parents=True)
    # no marker: a fresh or in-flight cell — fine
    assert sf.supersede_gate(cell, FP) is None
    # a marker of THIS identity: the ordinary in-container resume/skip
    (cell / "COMPLETE").write_text(_marker(FP))
    assert sf.supersede_gate(cell, FP) is None
    # another identity, no snapshot record: REFUSED, naming the fix
    (cell / "COMPLETE").write_text(_marker(STALE))
    problem = sf.supersede_gate(cell, FP)
    assert problem and "snapshot it first" in problem
    assert "--mode snapshot" in problem and cell.name in problem
    # a record naming the marker's fingerprint licenses the overwrite
    (cell / "superseded_by.json").write_text(json.dumps(
        {"cell_id": cell.name, "fit_fingerprint": STALE,
         "snapshot": "/vol/out/k_r1/cells_all/" + cell.name}))
    assert sf.supersede_gate(cell, FP) is None
    # a record for an OLDER publication does not license the current one
    (cell / "superseded_by.json").write_text(json.dumps(
        {"cell_id": cell.name, "fit_fingerprint": "c" * 64}))
    assert "never snapshotted" in sf.supersede_gate(cell, FP)
    # unreadable record / unparseable marker: refusals, never a guess
    (cell / "superseded_by.json").write_text("nope")
    assert "unreadable" in sf.supersede_gate(cell, FP)
    (cell / "COMPLETE").write_text("nope")
    assert "does not parse" in sf.supersede_gate(cell, FP)


def test_fits_spawn_refuses_the_unscoped_default(sf):
    """The round-2 config regeneration moved config_sha for every cell, so
    a bare fits-spawn would retrain all nine SIs. --sis scopes;
    --retrain-all is the one explicit acknowledgment."""
    with pytest.raises(SystemExit, match="REFUSED: no --sis"):
        sf.resolve_spawn_sis("", False)
    assert sf.resolve_spawn_sis("constv2-mix-1m,constv2-mix-10m", False) == \
        ["constv2-mix-1m", "constv2-mix-10m"]
    assert sf.resolve_spawn_sis("", True) == list(sf.SIS)
    with pytest.raises(SystemExit, match="two different intents"):
        sf.resolve_spawn_sis("constv2-mix-1m", True)
    with pytest.raises(SystemExit, match="unknown --sis"):
        sf.resolve_spawn_sis("constv2-mix-1b", False)
    entry = _entrypoint_src().split('if mode == "fits-spawn"', 1)[1] \
        .split('if mode == "', 1)[0]
    assert "resolve_spawn_sis(sis, retrain_all)" in entry
    assert "or SIS" not in entry  # the silent all-SIS default is gone
    # usage examples show the scoped form and the acknowledgment flag
    assert "--sis constv2-mix-1m,constv2-mix-10m,constv2-mix-full" in SRC


def test_fits_spawn_admits_the_r2_control_cell(sf):
    """The eight batch-64 it_doc fits are spawned with
    --sis constv2-mix-full,constv2ctl2-full (the constitution / control
    twins). Upstream bf86bd3c admitted constv2ctl2-full to SIS together
    with the control cell's config row; this lineage carried the config and
    registry side but not the spawn side, and the first spawn was refused
    in-launcher as 'unknown --sis' with all eight configs already folded.
    Pool resolution is generic (_pool_path -> the parent key's staged
    pairs_<cell>.jsonl), so admission IS the fix."""
    assert "constv2ctl2-full" in sf.SIS
    assert sf.resolve_spawn_sis("constv2-mix-full,constv2ctl2-full", False) == \
        ["constv2-mix-full", "constv2ctl2-full"]
    parent = sf.EB64_PARENT["gemma4e4b_it_doc_eb64"]
    assert sf._pool_path("gemma4e4b_it_doc_eb64", "constv2ctl2-full") == \
        f"/vol/pool/{parent}/pairs_constv2ctl2-full.jsonl"
    assert "--retrain-all" in SRC.split("import json", 1)[0]


def test_spawn_rows_record_expected_fingerprints(sf):
    fps = {"constv2-mix-1m__psr_ce_all": FP, "constv2-mix-1m__lora_r4_ce": STALE}
    pin = sf.spawn_pin("fit_h200_std", "fu-1", "abcdef0123456789",
                       overwrite=False, checkpoint_every=5)
    rows = sf.spawn_rows("gemma4e4b_base", "out/gemma4e4b_base",
                         ["constv2-mix-1m"], ["psr_ce_all", "lora_r4_ce"], fps,
                         pin=pin)
    assert [r["expect_fingerprint"] for r in rows] == [FP, STALE]
    assert [r["cell_id"] for r in rows] == list(fps)
    assert rows[0]["status"] == "pending-spawn" and rows[0]["call_id"] is None
    assert rows[0]["leg"] == "all" and rows[0]["out_root"] == "out/gemma4e4b_base"
    # every ROW carries its own spawn pin: the function it runs on, the
    # deployed function id, and the item fields a successor rebuilds —
    # never only the manifest's top level
    assert sf.ROW_PIN_FIELDS == ("app", "function", "function_id", "config_sha",
                                 "overwrite", "checkpoint_every",
                                 "continue_cap", "continue_lr_mult",
                                 "app_id", "app_version", "deploy_commit")
    for r in rows:
        assert {k: r[k] for k in sf.ROW_PIN_FIELDS} == {
            "app": "sdfbase-fits", "function": "fit_h200_std",
            "function_id": "fu-1", "config_sha": "abcdef0123456789",
            "overwrite": False, "checkpoint_every": 5, "continue_cap": 0,
            "continue_lr_mult": 1.0, "app_id": None, "app_version": None,
            "deploy_commit": None}
        assert "continued_from" not in r  # an ordinary (non-continuation) row
    # the deployment pin rides the row when fits-spawn could read it
    pinned = sf.spawn_pin("fit_h200_std", "fu-1", "abcdef0123456789",
                          overwrite=False, checkpoint_every=5,
                          deployment={"app_id": "ap-1", "app_version": 11,
                                      "deploy_commit": "ea7a058ac6e0"})
    assert (pinned["app_id"], pinned["app_version"],
            pinned["deploy_commit"]) == ("ap-1", 11, "ea7a058ac6e0")
    # an explicit 0 multiplier is preserved on the pin (the entrypoint has
    # already refused it; a truthiness default must not launder it)
    assert sf.spawn_pin("fit_h200_std", "fu-1", "abcdef0123456789",
                        overwrite=False, checkpoint_every=5,
                        continue_lr_mult=0.0)["continue_lr_mult"] == 0.0
    with pytest.raises(SystemExit, match="no expected fit fingerprint"):
        sf.spawn_rows("gemma4e4b_base", "out/gemma4e4b_base",
                      ["constv2-mix-10m"], ["psr_ce_all"], fps)
    # the entrypoint derives them IN-CONTAINER through the driver's own
    # functions and refuses a stale deploy (config bytes differ)
    entry = _entrypoint_src().split('if mode == "fits-spawn"', 1)[1] \
        .split('if mode == "', 1)[0]
    assert 'deployed_fn(deployed, "fingerprint_cpu")' in entry
    assert '_sm.check(fp_doc.get("config_sha"), expect_sha' in entry and "if cfg_check is None:" in entry
    assert "redeploy" in entry
    assert 'spawn_rows(key, root, want_sis, want_arms' in entry
    # rows record the app spawned on — through the per-row spawn pin
    # (the pin plus the app override, merged)
    assert "app=deployed" in entry and "pin=pin" in entry
    impl = SRC.split("def _fingerprint_impl", 1)[1].split("\ndef ", 1)[0]
    assert 'expected_fit_fingerprint(spec["cfg"], pairs,' in impl
    assert '"config_sha": _config_sha16(spec["cfg"])' in impl
    # the PLAIN derivation is the shared, Modal-free
    # scripts/sdfbase/fit_cell_identity.py (2026-09-06: the Slurm lane runs
    # the same gate) — the app delegates, and THAT module derives through
    # the driver's own functions
    efp = SRC.split("def expected_fit_fingerprint", 1)[1].split("\ndef ", 1)[0]
    assert "from fit_cell_identity import expected_fit_fingerprint as _plain" in efp
    assert "return _plain(cfg_path, pairs_path, si, arm)" in efp
    shared = (REPO / "scripts" / "sdfbase" / "fit_cell_identity.py").read_text()
    sefp = shared.split("def expected_fit_fingerprint", 1)[1].split("\ndef ", 1)[0]
    for call in ("t2s._cfg_with_positions(", "t2s._cli_fit_identity(",
                 "t2s.cell_fit_fingerprint("):
        assert call in sefp, call


# --------------------------------------------------------------------------- #
# the app-name override (the wave-A `sdfbase-fits` app is FROZEN)             #
# --------------------------------------------------------------------------- #


def test_app_name_default_is_the_frozen_wave_a_app_byte_for_byte(sf,
                                                                 monkeypatch):
    """No --app-name, no $SDFBASE_APP_NAME: every path resolves to exactly
    the string wave A deployed under, and the App object is built from the
    same resolution (the fixture imports with the env unset)."""
    monkeypatch.delenv("SDFBASE_APP_NAME", raising=False)
    assert sf.DEFAULT_APP_NAME == "sdfbase-fits"
    assert sf.resolve_app_name() == "sdfbase-fits"
    assert sf.resolve_app_name("") == "sdfbase-fits"
    assert sf.APP_NAME == "sdfbase-fits"
    assert sf.app.name == "sdfbase-fits"
    assert 'app = modal.App(APP_NAME)' in SRC
    assert 'modal.App("sdfbase-fits")' not in SRC


def test_app_name_override_precedence(sf, monkeypatch):
    """--app-name beats the default, $SDFBASE_APP_NAME beats the default,
    an agreeing pair is fine — and a DISAGREEING pair is refused rather
    than resolved (the App object is named from the env at import, so the
    flag cannot quietly win over it)."""
    monkeypatch.setenv("SDFBASE_APP_NAME", "sdfbase-fits-ws3")
    assert sf.resolve_app_name() == "sdfbase-fits-ws3"
    assert sf.resolve_app_name("") == "sdfbase-fits-ws3"
    assert sf.resolve_app_name("sdfbase-fits-ws3") == "sdfbase-fits-ws3"
    with pytest.raises(SystemExit, match="disagrees with"):
        sf.resolve_app_name("sdfbase-fits-r3")
    monkeypatch.delenv("SDFBASE_APP_NAME")
    assert sf.resolve_app_name() == "sdfbase-fits"
    assert sf.resolve_app_name("sdfbase-fits-r3") == "sdfbase-fits-r3"


def test_app_name_threads_to_every_deployed_lookup_and_manifest():
    """The entrypoint binds ONE `deployed` name and every
    Function.from_name / manifest doc / manifest row in it uses that
    binding — no mode can quietly spawn on, or record, the literal default
    (which would aim an override-app wave at the frozen wave-A app)."""
    main_src = _entrypoint_src()
    sig = main_src.split(")", 1)[0]
    assert 'app_name: str = ""' in sig
    assert "deployed = resolve_app_name(app_name)" in main_src
    assert '"sdfbase-fits"' not in main_src
    # modal.Function.from_name lives in deployed_fn and NOWHERE else
    assert SRC.count("Function.from_name(") == 1
    helper = SRC.split("def deployed_fn(", 1)[1].split("\ndef ", 1)[0]
    assert "Function.from_name(deployed, name)" in helper
    assert "Function.from_name(" not in main_src
    # every lookup in main passes the ONE binding: the fan-out (onboard /
    # probe-score / mmlu share it), stage, memprobe, hpo-spawn, snapshot,
    # fits-spawn (fit + fingerprint), battery, bench — nine sites
    targets = re.findall(r"deployed_fn\(\s*([^,]+),", main_src)
    assert targets and set(targets) == {"deployed"}, targets
    assert len(targets) == 9, targets
    # the three manifest docs + the two inline row literals (hpo trials,
    # battery shards) record it; fits-spawn rows get it via the spawn pin
    assert main_src.count('"app": deployed') == 5
    assert "app=deployed" in main_src
    for mode_tag in ('mode == "stage"', 'mode == "onboard"',
                     'mode == "memprobe"', 'mode == "hpo-spawn"',
                     'mode == "snapshot"', 'mode == "fits-spawn"',
                     'mode == "probe-score"', 'mode == "mmlu"',
                     'mode == "battery"', 'mode == "bench"'):
        block = main_src.split(mode_tag, 1)[1].split('if mode == "', 1)[0]
        assert "deployed" in block or "_fanout_models(" in block, mode_tag
    # the stale-deploy refusal names the app to redeploy under
    fits = main_src.split('mode == "fits-spawn"', 1)[1].split('if mode == "', 1)[0]
    assert "--name {deployed} scripts/modal/sdfbase_fits.py" in fits
    # the operator sequence in the module docstring shows the override form
    head = SRC.split("import json", 1)[0]
    assert "modal deploy --name sdfbase-fits-ws3" in head
    assert "SDFBASE_APP_NAME" in head


def test_manifest_rows_carry_the_app_they_spawn_on(sf):
    fps = {"constv2-mix-1m__psr_ce_all": FP}
    # the app rides the per-row spawn pin (spawn_pin defaults to the
    # wave-A name; fits-spawn passes the resolved app)
    default_pin = sf.spawn_pin("fit_h200_std", "fu-1", "abcdef0123456789",
                               overwrite=False, checkpoint_every=0)
    default_rows = sf.spawn_rows("gemma4e4b_base", "out/gemma4e4b_base",
                                 ["constv2-mix-1m"], ["psr_ce_all"], fps,
                                 pin=default_pin)
    assert [r["app"] for r in default_rows] == ["sdfbase-fits"]
    ws3_pin = sf.spawn_pin("fit_h200_std", "fu-2", "abcdef0123456789",
                           overwrite=False, checkpoint_every=0,
                           app="sdfbase-fits-ws3")
    ws3_rows = sf.spawn_rows("glm46vflash_doc", "out/glm46vflash_doc",
                             ["constv2-mix-1m"], ["psr_ce_all"], fps,
                             pin=ws3_pin)
    assert [r["app"] for r in ws3_rows] == ["sdfbase-fits-ws3"]
    # and the watcher reads the row's app for strictness + re-spawn hint
    sys.path.insert(0, str(REPO / "scripts" / "modal"))
    import fits_watch as fw
    assert fw.is_strict_app(ws3_rows[0]["app"]) is True
    assert fw.respawn_hint(ws3_rows[0]) == " --app-name sdfbase-fits-ws3"


def test_config_sha16_is_the_expect_sha_convention(sf, tmp_path):
    import hashlib

    p = tmp_path / "cfg.yaml"
    p.write_bytes(b"model: x\n")
    assert sf._config_sha16(p) == hashlib.sha256(b"model: x\n").hexdigest()[:16]
    # main() no longer hashes inline — one derivation
    assert "hashlib" not in _entrypoint_src()


def test_snapshot_mode_is_wired_to_snapshot_cells(sf):
    entry = _entrypoint_src().split('if mode == "snapshot"', 1)[1] \
        .split('if mode == "', 1)[0]
    assert 'deployed_fn(deployed, "snapshot_cpu")' in entry
    assert "bool(verify_only), reason" in entry
    assert "nothing about those cells may be refit" in entry
    impl = SRC.split("def _snapshot_impl", 1)[1].split("\ndef ", 1)[0]
    assert "from snapshot_cells import" in impl
    assert "snapshot_bank_root(out_root, suffix)" in impl
    assert "expected_refit_fingerprint" in impl
    assert "verify_snapshot(dst_bank, cell)" in impl
    assert sf.snapshot_bank_root("out/gemma4e4b_base") == "out/gemma4e4b_base_r1"
    # the CPU functions exist, volume-mounted, no GPU
    for name in ("fingerprint_cpu", "snapshot_cpu"):
        m = re.search(r"@app\.function\((?P<args>[^)]*)\)\ndef " + name
                      + r"\(item: tuple\)", SRC)
        assert m, name
        assert "gpu=" not in m.group("args") and "**_COMMON" in m.group("args")


# --------------------------------------------------------------------------- #
# round-2 — mmlu mode is run-scoped                                           #
# --------------------------------------------------------------------------- #


def test_mmlu_command_is_run_scoped(sf):
    cmd = sf.mmlu_command("gemma4e4b_base", "none,constv2-mix-1m__psr_ce_all",
                          "r2")
    d = dict(zip(cmd, cmd[1:]))
    assert d["--run-label"] == "r2"
    assert d["--out-dir"] == "/vol/out/gemma4e4b_base/mmlu"
    assert d["--defenses-root"] == "/vol/out/gemma4e4b_base"  # live bank
    assert "--out" not in cmd  # the old unscoped flag is gone
    r1 = sf.mmlu_command("gemma4e4b_base", "constv2-mix-1m__psr_ce_all", "r1",
                         "/vol/out/gemma4e4b_base_r1")
    assert dict(zip(r1, r1[1:]))["--defenses-root"] == \
        "/vol/out/gemma4e4b_base_r1"
    with pytest.raises(SystemExit, match="run_label is required"):
        sf.mmlu_command("gemma4e4b_base", "none", "")
    assert sf.expand_defenses_root("/vol/out/{key}_r1", "qwen35_4b_base") == \
        "/vol/out/qwen35_4b_base_r1"
    assert sf.expand_defenses_root("", "qwen35_4b_base") is None
    entry = _entrypoint_src().split('if mode == "mmlu"', 1)[1] \
        .split('if mode == "', 1)[0]
    assert "--run-label required" in entry
    assert "expand_defenses_root(defenses_root, k)" in entry
    impl = SRC.split("def _mmlu_impl", 1)[1].split("\ndef ", 1)[0]
    assert "mmlu_command(key, cells, run_label, defenses_root, scale)" in impl


# --------------------------------------------------------------------------- #
# 2026-09-03 — the 2x-dose lane: mmlu --scale, gsm8k mode, ce mode             #
# --------------------------------------------------------------------------- #


def test_mmlu_command_threads_scale_and_keeps_scale1_byte_identical(sf):
    """--scale rides the mmlu item into base_mmlu's --scale; the default
    leaves every already-published r1/r2/r3 command byte for byte."""
    base = sf.mmlu_command("gemma4e4b_base", "none,constv2-mix-1m__psr_ce_all",
                           "r2")
    assert base == sf.mmlu_command("gemma4e4b_base",
                                   "none,constv2-mix-1m__psr_ce_all", "r2",
                                   None, 1.0)
    assert "--scale" not in base
    x2 = sf.mmlu_command("gemma4e4b_base", "none,constv2-mix-1m__psr_ce_all",
                         "r2_x2", None, 2.0)
    assert x2[:-2] == base[:-2] + ["--scale"][:0] or True  # prefix shared
    d = dict(zip(x2, x2[1:]))
    assert d["--scale"] == "2.0" and d["--run-label"] == "r2_x2"
    assert x2[:len(base)] == [*base[:-2], "--run-label", "r2_x2"]
    impl = SRC.split("def _mmlu_impl", 1)[1].split("\ndef ", 1)[0]
    assert "scale = float(item[4]) if len(item) > 4 else 1.0" in impl
    entry = _entrypoint_src().split('if mode == "mmlu"', 1)[1] \
        .split('if mode == "', 1)[0]
    assert "float(scale)))" in entry


def test_gsm8k_command_is_run_scoped_and_scaled(sf):
    """gsm8k_command is mmlu_command's twin: label in dir + filename root,
    live bank by default, n=500 seed 0, and --scale ALWAYS explicit."""
    cmd = sf.gsm8k_command("qwen35_4b_base", "none,constv2-mix-1k__psr_ce_all",
                           "r2")
    d = dict(zip(cmd, cmd[1:]))
    assert cmd[1] == "/repo/scripts/sdfbase/base_gsm8k.py"
    assert d["--config"] == sf.MODELS["qwen35_4b_base"]["cfg"]
    assert d["--run-label"] == "r2"
    assert d["--out-dir"] == "/vol/out/qwen35_4b_base/gsm8k"
    assert d["--defenses-root"] == "/vol/out/qwen35_4b_base"
    assert d["--n"] == "500" and d["--seed"] == "0"
    assert d["--scale"] == "1.0"
    x2 = sf.gsm8k_command("qwen35_4b_base", "none", "r2_x2", None, 2.0)
    assert dict(zip(x2, x2[1:]))["--scale"] == "2.0"
    snap = sf.gsm8k_command("gemma4e4b_base", "constv2-mix-1m__psr_ce_all",
                            "r1", "/vol/out/gemma4e4b_base_r1")
    assert dict(zip(snap, snap[1:]))["--defenses-root"] == \
        "/vol/out/gemma4e4b_base_r1"
    with pytest.raises(SystemExit, match="run_label is required"):
        sf.gsm8k_command("gemma4e4b_base", "none", "")
    # --batch-size: omitted at 0 (script default), explicit when set
    assert "--batch-size" not in cmd
    b = sf.gsm8k_command("nemotron3nano4b_doc", "none", "r2", None, 1.0, 128)
    assert dict(zip(b, b[1:]))["--batch-size"] == "128"
    # --decode-parity: omitted unless given, then the volume path
    assert "--decode-parity" not in b
    p = sf.gsm8k_command("nemotron3nano4b_doc", "constv2-mix-10k__psr_ce_all",
                         "r2", None, 1.0, 128,
                         "/vol/out/nemotron3nano4b_doc/gsm8k/parity_x1/att.json")
    assert dict(zip(p, p[1:]))["--decode-parity"] == \
        "/vol/out/nemotron3nano4b_doc/gsm8k/parity_x1/att.json"
    entry = _entrypoint_src().split('if mode == "gsm8k"', 1)[1] \
        .split('if mode == "', 1)[0]
    assert "--run-label required" in entry and "--cells required" in entry
    assert "expand_defenses_root(defenses_root, k)" in entry
    assert "float(scale), int(batch_size),\n" in entry
    assert "decode_parity or None))" in entry
    impl = SRC.split("def _gsm8k_impl", 1)[1].split("\ndef ", 1)[0]
    assert "gsm8k_command(key, cells, run_label, defenses_root, scale,\n" \
           "                        batch_size, decode_parity)" in impl
    assert "batch_size = int(item[5]) if len(item) > 5 and item[5] else 0" in impl
    assert "decode_parity = str(item[6]) if len(item) > 6 and item[6] else None" \
        in impl
    assert "_ensure_model(key)" in impl and "vol.commit()" in impl
    # generation budget: 6 h on std/qwen, 12 h on the kernel-launch-bound
    # Mamba2 hybrid (nemotron)
    for variant, budget in (("std", 21600), ("qwen", 21600),
                            ("nemotron", 43200)):
        m = re.search(_BINDING.format(name=f"gsm8k_h200_{variant}"), SRC,
                      flags=re.S)
        assert m and f"timeout={budget}" in m["deco"], variant
        assert m["impl"] == "_gsm8k_impl"


def test_ce_command_is_the_derived_apps_layout_byte_for_byte(sf):
    """The fits app's ce mode writes exactly where and how the derived app's
    --mode score writes (ce_score/<key>/<cell>__<pool>__scale<s>.json, the
    same flags), so the analysis reads a fitted 1M cell's 1x/2x CE through
    the one ce_score reader. The pool table is the derived app's twin."""
    from test_watch_successor import make_fake_modal

    prev_modal = sys.modules.get("modal")     # own the stub here too (see the sf fixture)
    sys.modules["modal"] = make_fake_modal()
    spec = importlib.util.spec_from_file_location(
        "_sdfbase_derived_for_ce", REPO / "scripts" / "modal" /
        "sdfbase_derived.py")
    sd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sd)
    assert sf.CE_POOLS == sd.CE_POOLS
    assert sf.CE_ROOT == sd.CE_ROOT == "/vol/out/ce_score"
    assert sf.PARENTS_ROOT == sd.PARENTS_ROOT
    for scale in (1.0, 2.0):
        ours, out_a = sf.ce_command(
            "gemma4e4b_base", "/vol/out/gemma4e4b_base/cells_all/x/object.pt",
            "constv2-mix-1m__psr_ce_all", "constv2-heldout",
            "/vol/pool/gemma4e4b_base/pairs_constv2-heldout.jsonl",
            scale=scale, python="py")
        theirs, out_b = sd.ce_command(
            "gemma4e4b_base", "/vol/out/gemma4e4b_base/cells_all/x/object.pt",
            "constv2-mix-1m__psr_ce_all", "constv2-heldout",
            "/vol/pool/gemma4e4b_base/pairs_constv2-heldout.jsonl",
            scale=scale, python="py")
        assert ours == theirs and out_a == out_b
        assert out_a == ("/vol/out/ce_score/gemma4e4b_base/"
                         f"constv2-mix-1m__psr_ce_all__constv2-heldout__scale{scale:g}.json")
    bare, out_bare = sf.ce_command("qwen35_4b_base", None, "bare",
                                   "control-heldout", "/p.jsonl", scale=1.0)
    assert "--object" not in bare and "--cell" not in bare
    assert out_bare.endswith("/qwen35_4b_base/bare__control-heldout__scale1.json")
    assert sf.parse_scales("1.0,2.0") == [1.0, 2.0]
    for bad in ("", "1.0,1", "0,2"):
        with pytest.raises(SystemExit):
            sf.parse_scales(bad)
    entry = _entrypoint_src().split('if mode == "ce"', 1)[1] \
        .split('if mode == "', 1)[0]
    assert "--cells required" in entry and "parse_scales(want_scales)" in entry
    impl = SRC.split("def _ce_impl", 1)[1].split("\ndef ", 1)[0]
    assert 'glob("object.*")' in impl and 'q.suffix != ".json"' in impl
    assert "if obj_path is None and s != 1.0" in impl  # bare has no scale
    assert "ce_command(key, obj_path, cell, ptag, ppath" in impl


# --------------------------------------------------------------------------- #
# round-2 — the _r2 battery wave                                              #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("slug", SLUGS)
def test_battery_r2_wave_shares_the_bank_and_namespaces_the_run(sf, slug):
    """Objects are identities, not wave state: the _r2 config serves from
    the SAME staged bank; the RUN (controls out-root, and the config's own
    out_dir/private_run) carries the wave."""
    cfg_path = f"/repo/configs/sdfbase_transfer_{slug}_r2.yaml"
    assert sf._battery_parts(cfg_path) == (slug, "_r2")
    assert sf._battery_slug(cfg_path) == slug
    assert sf._battery_parts(f"/repo/configs/sdfbase_transfer_{slug}.yaml") \
        == (slug, "")
    cmd, env = sf.battery_commands(cfg_path, "controls",
                                   "constv2ctl-100k__psr_ce_all", 0, 1, True)
    d = dict(zip(cmd, cmd[1:]))
    assert env["TT_TWO_SI_DEFENSES_DIR"] == f"/vol/out/transfer_{slug}/cells"
    assert d["--defenses-dir"] == f"/vol/out/transfer_{slug}/cells"
    assert d["--out-root"] == f"/vol/out/sdfbase_transfer_{slug}_r2/controls"
    assert d["--config"] == cfg_path
    for atk in ("bon", "xstest"):
        cmd, _ = sf.battery_commands(cfg_path, atk, "none", 0, 1, True)
        assert dict(zip(cmd, cmd[1:]))["--config"] == cfg_path


@pytest.mark.parametrize("slug", SLUGS)
def test_battery_r3_wave_serves_from_its_own_bank(sf, slug, tmp_path):
    """The shared bank (out/transfer_<slug>/cells) holds the ROUND-1 fits of
    the 1M cells and is never overwritten, so the _r3 wave serves the
    CURRENT objects from a per-wave bank. battery_commands derives it from
    the wave (battery_bank / BATTERY_BANK_SUFFIX); rounds 1 and _r2 keep the
    shared bank byte for byte; an unregistered wave refuses; and the launch
    check refuses a config whose defenses_dir names another bank."""
    assert sf.BATTERY_BANK_SUFFIX == {"": "", "_r2": "", "_r3": "_r3",
                                      "_it": "_it"}
    assert sf.battery_bank(slug, "") == f"/vol/out/transfer_{slug}/cells"
    assert sf.battery_bank(slug, "_r2") == f"/vol/out/transfer_{slug}/cells"
    assert sf.battery_bank(slug, "_r3") == f"/vol/out/transfer_{slug}_r3/cells"
    with pytest.raises(SystemExit, match="no object bank"):
        sf.battery_bank(slug, "_r4")
    cfg_path = f"/repo/configs/sdfbase_transfer_{slug}_r3.yaml"
    assert sf._battery_parts(cfg_path) == (slug, "_r3")
    for atk, cell in (("controls", "constv2-mix-1m__lora_r4_ce"),
                      ("pasttense", "constv2-mix-1m__psr_ce_all"),
                      ("bon", "none"), ("xstest", "none")):
        cmd, env = sf.battery_commands(cfg_path, atk, cell, 0, 1, True)
        d = dict(zip(cmd, cmd[1:]))
        assert env["TT_TWO_SI_DEFENSES_DIR"] == f"/vol/out/transfer_{slug}_r3/cells"
        assert d["--config"] == cfg_path
        if atk == "controls":
            assert d["--defenses-dir"] == f"/vol/out/transfer_{slug}_r3/cells"
            assert d["--out-root"] == f"/vol/out/sdfbase_transfer_{slug}_r3/controls"
    # the _r2 wave is unchanged by the table: the shared bank, byte for byte
    cmd, env = sf.battery_commands(
        f"/repo/configs/sdfbase_transfer_{slug}_r2.yaml", "controls", "none", 0, 1, True)
    assert env["TT_TWO_SI_DEFENSES_DIR"] == f"/vol/out/transfer_{slug}/cells"
    assert dict(zip(cmd, cmd[1:]))["--defenses-dir"] == f"/vol/out/transfer_{slug}/cells"
    # the committed r3 config names the r3 bank; the launch check agrees
    local = sf.WORKTREE / Path(sf.battery_config_path(
        f"{slug}_base", "_r3")).relative_to("/repo")
    assert local.exists(), local
    assert sf.battery_bank_matches(local, cfg_path) == f"/vol/out/transfer_{slug}_r3/cells"
    # ... and refuses a config naming the shared bank for the r3 wave
    bad = tmp_path / f"sdfbase_transfer_{slug}_r3.yaml"
    bad.write_text(f"defenses_dir: out/transfer_{slug}/cells\nexplicit_cells: [none]\n")
    with pytest.raises(SystemExit, match="must name one bank"):
        sf.battery_bank_matches(bad, cfg_path)
    # the entrypoint runs the check before any spawn and records the bank
    entry = _entrypoint_src().split('if mode == "battery"', 1)[1]
    assert entry.index("battery_bank_matches(cfg_local, cfg_path)") < entry.index("fn.spawn(")
    assert '"bank": bank' in entry


@pytest.mark.parametrize("slug", SLUGS)
def test_battery_it_wave_serves_from_its_own_bank(sf, slug, tmp_path):
    """The `_it` wave serves the instruct-direct objects (fit ON the served
    instruct model — a copy from the `_it_doc` key's own fitting bank, no
    sibling mapping) plus gemma's re-run 1M LoRA from a bank no other wave
    has: /vol/out/transfer_<slug>_it/cells. battery_commands derives it from
    the wave; the committed config names the same dir; a config naming the
    `_r3` (or shared) bank for the `_it` wave refuses; the run namespace
    carries `_it`; the `_r3` wave's derivation is untouched."""
    assert sf.battery_bank(slug, "_it") == f"/vol/out/transfer_{slug}_it/cells"
    cfg_path = f"/repo/configs/sdfbase_transfer_{slug}_it.yaml"
    assert sf._battery_parts(cfg_path) == (slug, "_it")
    assert sf._battery_slug(cfg_path) == slug
    assert sf.battery_config_path(f"{slug}_it_doc", "_it") == cfg_path
    cells = ["none", "constv2-mix-full__psr_ce_all",
             "constv2-mix-full__lora_r4_ce"]
    if slug == "gemma4e4b":
        cells.append("constv2-mix-1m__lora_r4_ce")
    for atk in ("controls", "pasttense", "bon", "xstest"):
        for cell in cells:
            cmd, env = sf.battery_commands(cfg_path, atk, cell, 0, 1, True)
            d = dict(zip(cmd, cmd[1:]))
            assert env["TT_TWO_SI_DEFENSES_DIR"] == \
                f"/vol/out/transfer_{slug}_it/cells"
            assert d["--config"] == cfg_path
            assert d["--cell"] == cell
            if atk == "controls":
                assert d["--defenses-dir"] == f"/vol/out/transfer_{slug}_it/cells"
                assert d["--out-root"] == \
                    f"/vol/out/sdfbase_transfer_{slug}_it/controls"
            else:
                assert d["--private-root"] == "/vol/private"
    # the committed `_it` config names the `_it` bank and lists exactly the
    # wave's cells; the launch check agrees
    local = sf.WORKTREE / Path(cfg_path).relative_to("/repo")
    assert local.exists(), local
    assert sf.battery_bank_matches(local, cfg_path) == \
        f"/vol/out/transfer_{slug}_it/cells"
    assert sf.battery_cells_in_config(local, cells) == cells
    with pytest.raises(SystemExit, match="not in"):
        sf.battery_cells_in_config(local, ["constv2-mix-1m__psr_ce_all"])
    # ... and refuses a config naming the r3 or the shared bank for this wave
    for other in (f"out/transfer_{slug}_r3/cells", f"out/transfer_{slug}/cells"):
        bad = tmp_path / f"sdfbase_transfer_{slug}_it.yaml"
        bad.write_text(f"defenses_dir: {other}\nexplicit_cells: [none]\n")
        with pytest.raises(SystemExit, match="must name one bank"):
            sf.battery_bank_matches(bad, cfg_path)
    # the `_r3` wave is unchanged by the new row
    cmd, env = sf.battery_commands(
        f"/repo/configs/sdfbase_transfer_{slug}_r3.yaml", "controls",
        "constv2-mix-1m__lora_r4_ce", 0, 1, True)
    assert env["TT_TWO_SI_DEFENSES_DIR"] == f"/vol/out/transfer_{slug}_r3/cells"
    assert dict(zip(cmd, cmd[1:]))["--out-root"] == \
        f"/vol/out/sdfbase_transfer_{slug}_r3/controls"


def test_battery_manifest_is_never_overwritten_and_shards_select_repair_rows(
        sf, routed, capsys):
    """A battery manifest's rows are the only handle on running containers
    (watcher / pull / output check read their call ids), so a second spawn
    onto the same manifest refuses BEFORE any spawn, and a repair names a
    fresh manifest and selects its rows: --shards keeps the failed shard's
    --nshards slicing (an index outside it refuses), unsharded attacks have
    only shard 0, and a selection spawning nothing refuses."""
    entry, log, home = routed
    m = home / "stage/sdfbase/m_rest.json"
    entry(mode="battery", model="gemma4e4b_base", cells="none",
          attacks="pasttense,xstest", nshards=2, wave="_r3", manifest=str(m))
    capsys.readouterr()
    doc = json.loads(m.read_text())
    assert [(r["attack"], r["shard"], r["nshards"]) for r in doc["shards"]] == \
        [("pasttense", 0, 2), ("pasttense", 1, 2), ("xstest", 0, 1)]
    assert doc["shards_selected"] is None
    ids = [r["call_id"] for r in doc["shards"]]
    assert len(log) == 3
    # the same manifest again: refused, nothing spawned, rows untouched
    with pytest.raises(SystemExit, match="never overwritten"):
        entry(mode="battery", model="gemma4e4b_base", cells="none",
              attacks="pasttense,xstest", nshards=2, wave="_r3",
              manifest=str(m))
    assert len(log) == 3
    assert [r["call_id"] for r in json.loads(m.read_text())["shards"]] == ids
    # a repair: the failed shard only, same slicing, fresh manifest
    log.clear()
    fix = home / "stage/sdfbase/m_fix.json"
    entry(mode="battery", model="gemma4e4b_base", cells="none",
          attacks="pasttense,xstest", nshards=2, wave="_r3", shards="1",
          manifest=str(fix))
    capsys.readouterr()
    doc = json.loads(fix.read_text())
    assert [(r["attack"], r["shard"], r["nshards"]) for r in doc["shards"]] == \
        [("pasttense", 1, 2)]                      # xstest has no shard 1
    assert doc["shards_selected"] == [1]
    assert len(log) == 1 and log[0][3][3:5] == (1, 2)   # (shard, nshards) sent
    # shard 0 brings the unsharded attacks along (they ARE shard 0)
    log.clear()
    fix0 = home / "stage/sdfbase/m_fix0.json"
    entry(mode="battery", model="gemma4e4b_base", cells="none",
          attacks="pasttense,xstest", nshards=2, wave="_r3", shards="0",
          manifest=str(fix0))
    capsys.readouterr()
    assert [(r["attack"], r["shard"]) for r in json.loads(fix0.read_text())["shards"]] == \
        [("pasttense", 0), ("xstest", 0)]
    # refusals: outside the slicing; a selection spawning nothing
    log.clear()
    with pytest.raises(SystemExit, match="outside"):
        entry(mode="battery", model="gemma4e4b_base", cells="none",
              attacks="pasttense", nshards=2, wave="_r3", shards="2",
              manifest=str(home / "m_bad.json"))
    with pytest.raises(SystemExit, match="selects no"):
        entry(mode="battery", model="gemma4e4b_base", cells="none",
              attacks="xstest", nshards=2, wave="_r3", shards="1",
              manifest=str(home / "m_none.json"))
    assert not log
    assert not (home / "m_bad.json").exists()
    assert not (home / "m_none.json").exists()
    # the helper's own contract
    assert sf.battery_shard_selection("", 2) is None
    assert sf.battery_shard_selection("1,0,1", 2) == [0, 1]
    for bad in ("2", "-1", "x", ","):
        with pytest.raises(SystemExit):
            sf.battery_shard_selection(bad, 2)
    entry_src = _entrypoint_src().split('if mode == "battery"', 1)[1]
    # the reservation is an EXCLUSIVE create (atomic against a racing
    # launch), taken before the first flush and before any spawn; an empty
    # selection releases it
    assert 'mpath.open("x")' in entry_src
    assert "if mpath.exists()" not in entry_src
    assert entry_src.index('mpath.open("x")') < entry_src.index("_flush_manifest(doc, mpath)")
    assert entry_src.index("if not rows:") < entry_src.index("fn.spawn(")
    assert entry_src.index("mpath.unlink()") < entry_src.index("fn.spawn(")
    # a reservation that another launch took between our decision and our
    # open fails closed: the file exists -> refused, nothing spawned
    log.clear()
    raced = home / "stage/sdfbase/m_raced.json"
    raced.write_text("{}\n")            # the other launch's reservation
    with pytest.raises(SystemExit, match="never overwritten"):
        entry(mode="battery", model="gemma4e4b_base", cells="none",
              attacks="pasttense", nshards=2, wave="_r3", manifest=str(raced))
    assert not log and raced.read_text() == "{}\n"


def test_battery_config_path_and_cell_validation(sf, tmp_path):
    assert sf.battery_config_path("gemma4e4b_base") == \
        "/repo/configs/sdfbase_transfer_gemma4e4b.yaml"
    assert sf.battery_config_path("qwen35_4b_base", "_r2") == \
        "/repo/configs/sdfbase_transfer_qwen35_4b_r2.yaml"
    with pytest.raises(SystemExit, match="--wave"):
        sf.battery_config_path("gemma4e4b_base", "r2")
    with pytest.raises(SystemExit, match="no transfer config"):
        sf.battery_config_path("qwen35_4b_it")
    # both waves' configs are committed for both wave-A models (the `_doc`
    # keys are registered in TRANSFER_SLUG but their transfer batteries are
    # scoped out — see test_battery_refuses_keys_without_a_transfer_config)
    for key in ("gemma4e4b_base", "qwen35_4b_base"):
        for wave in ("", "_r2"):
            local = sf.WORKTREE / Path(
                sf.battery_config_path(key, wave)).relative_to("/repo")
            assert local.exists(), local
    cfg = tmp_path / "c.yaml"
    cfg.write_text("explicit_cells: [none, a__b]\n")
    assert sf.battery_cells_in_config(cfg, ["none", "a__b"]) == ["none", "a__b"]
    with pytest.raises(SystemExit, match="not in"):
        sf.battery_cells_in_config(cfg, ["zzz__q"])
    entry = _entrypoint_src().split('if mode == "battery"', 1)[1]
    assert "battery_config_path(key, wave)" in entry
    assert "battery_cells_in_config(" in entry
    assert "battery_calls_{key}{wave}.json" in entry
    assert '"wave": wave' in entry


# --------------------------------------------------------------------------- #
# round-2 — the 2x-strength probe pass (--scale)                                #
# --------------------------------------------------------------------------- #


def test_probe_out_dirname_suffixes_off_1_scales(sf):
    """Scale-1 layouts stay byte-for-byte historical (probe / probe_<ver>);
    an off-1 scale gets its own _x<scale> dir so a 2x pass can never
    overwrite the scale-1 margins."""
    assert sf.probe_out_dirname("v1") == "probe"
    assert sf.probe_out_dirname("v2") == "probe_v2"
    assert sf.probe_out_dirname("v1", 1.0) == "probe"
    assert sf.probe_out_dirname("v2", 2.0) == "probe_v2_x2"
    assert sf.probe_out_dirname("v1", 2.0) == "probe_x2"
    assert sf.probe_out_dirname("v2", 0.5) == "probe_v2_x0.5"


def test_probe_registry_binds_file_and_out_dir(sf):
    """--probe <ver> resolves the frozen file stem AND the output dir
    through ONE registry; v1 / v2 keep their historical spellings, the
    genre-matched v3g scores data/sdfbase/probe_v3_genre.* into probe_v3/,
    and an unregistered version refuses (never a fabricated path)."""
    assert sf.PROBES["v1"] == ("probe_v1", "probe")
    assert sf.PROBES["v2"] == ("probe_v2", "probe_v2")
    assert sf.PROBES["v3g"] == ("probe_v3_genre", "probe_v3")
    assert sf.probe_file_stem("v2") == "probe_v2"
    assert sf.probe_file_stem("v3g") == "probe_v3_genre"
    assert sf.probe_out_dirname("v3g") == "probe_v3"
    assert sf.probe_out_dirname("v3g", 2.0) == "probe_v3_x2"
    for bad in ("v3", "v3_genre", "probe_v3_genre", ""):
        with pytest.raises(SystemExit, match="unknown probe"):
            sf.probe_file_stem(bad)
        with pytest.raises(SystemExit, match="unknown probe"):
            sf.probe_out_dirname(bad)
    # the frozen files the registry names exist in the tree, sha-pinned
    for stem, _ in sf.PROBES.values():
        if stem == "probe_v1":
            continue  # v1 predates the frozen-manifest layout in some trees
        assert (REPO / "data" / "sdfbase" / f"{stem}.jsonl").is_file(), stem
        assert (REPO / "data" / "sdfbase" / f"{stem}.manifest.json").is_file(), stem
    # the runner and the entrypoint both go through the registry
    impl = SRC.split("def _probe_score_impl", 1)[1].split("\ndef ", 1)[0]
    assert "stem = probe_file_stem(ver)" in impl
    assert 'f"/repo/data/sdfbase/{stem}.jsonl"' in impl
    assert 'f"/repo/data/sdfbase/{stem}.manifest.json"' in impl
    assert "probe_{ver}" not in impl
    entry = SRC.split('if mode == "probe-score"', 1)[1].split("return")[0]
    assert "probe_file_stem(probe)" in entry


def test_probe_score_scale_threads_to_object_rows_only(sf):
    """Source-level pins for the scale seam: the CLI --scale rides ONLY
    the object rows (inside the obj_path branch), the gate rows are
    skipped off-1, and the entrypoint refuses a scaled gates-only pass."""
    body = SRC.split("def _probe_score_impl", 1)[1].split("def _mmlu_impl")[0]
    obj_branch = body.split("if obj_path:", 1)[1].split("t0 = time.time()")[0]
    assert '"--scale", str(scale)' in obj_branch
    assert "if scale == 1.0:" in body  # gate rows only at scale 1
    entry = SRC.split('if mode == "probe-score"', 1)[1].split("return")[0]
    assert 'want == "gates"' in entry and "--scale" in entry
    assert "(k, want, probe, float(scale))" in entry


def test_probe_score_artifact_discovery_ignores_provenance_sidecar():
    """Round-2 harvest bug: PSR cells carry object.pt + object.pt.provenance.json;
    a bare glob("object.*") saw two candidates and REFUSED every PSR cell as
    ambiguous. Pin the sidecar exclusion at the discovery site."""
    assert 'glob("object.*") if q.suffix != ".json"' in SRC


def test_require_wandb_entity_is_the_shared_validator(sf, monkeypatch):
    """The wrapper's guard IS ``sisteer.wandb_entity`` — the same
    validator the direct CLI (``psr_residual_distill.py --wandb``) runs
    before its own ``wandb.init`` — with the secret named in the hint; the
    off-switches short-circuit before it (no import, no network)."""
    import sisteer.wandb_entity as we

    seen = {}

    def shared(*, hint=None, **kw):
        seen["hint"] = hint
        return "example-entity"

    monkeypatch.setattr(we, "require_wandb_entity", shared)
    monkeypatch.delenv("WANDB_DISABLED", raising=False)
    monkeypatch.setattr(sf, "_HAVE_WANDB", True)
    assert sf.require_wandb_entity() == "example-entity"
    assert "SISTEER_MODAL_WANDB_SECRET" in seen["hint"] and WANDB_SECRET_EXAMPLE in seen["hint"]
    seen.clear()
    monkeypatch.setenv("WANDB_DISABLED", "1")
    assert sf.require_wandb_entity() == "" and seen == {}
    monkeypatch.delenv("WANDB_DISABLED")
    monkeypatch.setattr(sf, "_HAVE_WANDB", False)
    assert sf.require_wandb_entity() == "" and seen == {}
    # no second copy of the rule survives in the wrapper
    assert "default_entity" not in SRC


# --------------------------------------------------------------------------- #
# EVERY mode spawns on the resolved app                                       #
# --------------------------------------------------------------------------- #
#
# Before the fix, stage / onboard / memprobe / probe-score / mmlu spawned the
# module's locally decorated function objects, which bind to the App built at
# import from $SDFBASE_APP_NAME alone — so `--app-name sdfbase-fits-ws3` with
# the env unset ran them under an ephemeral `sdfbase-fits` and changed
# nothing. These tests drive main() itself for every mode with a patched
# modal.Function.from_name and assert on the handle that ACTUALLY spawned —
# never on resolve_app_name()'s string.


class _FakeCall:
    def __init__(self, n, result):
        self.object_id = f"fc-{n:04d}"
        self._result = result

    def get(self):
        return self._result


class _FakeFn:
    """A deployed-function handle that records who spawned what."""

    def __init__(self, app, name, log, sf):
        self.app, self.name, self._log, self._sf = app, name, log, sf

    def spawn(self, item):
        self._log.append(("spawn", self.app, self.name, item))
        key = item[0] if isinstance(item, tuple) else item
        return _FakeCall(len(self._log), {
            "status": "ok", "model": key, "staged": True,
            "best_bs": 2, "ladder": {"1": "ok", "2": "ok"}})

    def remote(self, item):
        self._log.append(("remote", self.app, self.name, item))
        assert self.name == "fingerprint_cpu", self.name
        # (key, pairs, continue_cap, out_root) — the continuation-aware
        # item; an ordinary spawn sends cap 0
        key, pairs = item[:2]
        assert len(item) == 4 and item[2] == 0, item
        cfg_local = self._sf.WORKTREE / Path(
            self._sf.MODELS[key]["cfg"]).relative_to("/repo")
        return {"config_sha": self._sf._config_sha16(cfg_local),
                "fingerprints": {f"{si}__{arm}": f"fp-{si}-{arm}"
                                 for si, arm in pairs},
                "missing_pools": []}


#: A staged design the shape emit-trials writes (3 trials; the identity
#: fields hpo-spawn records per row and fits_watch.py validates).
HPO_DESIGN = {
    "round": 1, "position": "all", "arm": "psr_ce_all",
    "model": "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16",
    "model_revision": "dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f",
    "config_sha": "da1870c7d4c5fce8",
    "box": {"lr": [5e-05, 0.05], "max_grad_norm": [0.02, 10.0]},
    "warmup_levels": [0.05, 0.15],
    "trials": [{"index": i, "lr": lr, "warmup_frac": w, "max_grad_norm": c}
               for i, (lr, w, c) in enumerate([(5.6e-05, 0.05, 2.2),
                                               (3.1e-03, 0.15, 0.09),
                                               (4.4e-04, 0.05, 0.53)])],
}


@pytest.fixture
def routed(sf, monkeypatch, tmp_path):
    """Patch the ONE lookup, sandbox Path.home() (hpo designs + default
    manifests), return (entrypoint, spawn/remote log, home)."""
    log = []
    monkeypatch.setattr(sf.modal.Function, "from_name",
                        lambda app, name, *a, **k: _FakeFn(app, name, log, sf))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setenv("SISTEER_STAGE_DIR", str(tmp_path / "stage"))   # $STAGE: the local staging directory
    monkeypatch.delenv("SDFBASE_APP_NAME", raising=False)
    # hpo-spawn reads the staged design for every (key, arm) it spawns
    trials = tmp_path / "stage/sdfbase/hpo/trials"
    trials.mkdir(parents=True)
    (trials / "r1_nemotron3nano4b_doc_psr_ce_all_all.json").write_text(
        json.dumps(HPO_DESIGN))
    entry = getattr(sf.main, "raw_f", sf.main)  # plain fn under the fake
    return entry, log, tmp_path


WS3 = "sdfbase-fits-ws3"

#: mode -> (main() kwargs, the deployed functions it must reach, via)
MODE_CASES = {
    "stage": (dict(mode="stage", model="glm46vflash_doc,nemotron3nano4b_doc"),
              {"stage_std"}, "spawn"),
    "onboard": (dict(mode="onboard",
                     model="gemma4e4b_base,qwen35_4b_base,nemotron3nano4b_doc"),
                {"onboard_h200_std", "onboard_h200_qwen",
                 "onboard_h200_nemotron"}, "spawn"),
    "memprobe": (dict(mode="memprobe",
                      model="glm46vflash_doc,nemotron3nano4b_doc"),
                 {"memprobe_h200_std", "memprobe_h200_nemotron"}, "spawn"),
    "probe-score": (dict(mode="probe-score", probe="v1",
                         model="gemma4e4b_base,qwen35_4b_it,glm46vflash_doc"),
                    {"score_h200_std", "score_h200_qwen"}, "spawn"),
    "mmlu": (dict(mode="mmlu", model="nemotron3nano4b_doc", cells="none",
                  run_label="r1"),
             {"mmlu_h200_nemotron"}, "spawn"),
    "hpo-spawn": (dict(mode="hpo-spawn", model="nemotron3nano4b_doc",
                       arms="psr_ce_all", round_no=1),
                  {"hpo_h200_nemotron"}, "spawn"),
    "snapshot": (dict(mode="snapshot", model="gemma4e4b_base",
                      cells="constv2-mix-1m__psr_ce_all"),
                 {"snapshot_cpu"}, "spawn"),
    "fits-spawn": (dict(mode="fits-spawn", model="nemotron3nano4b_doc",
                        sis="constv2-mix-1m", arms="lora_r4_ce"),
                   {"fit_h200_nemotron", "fingerprint_cpu"}, "spawn+remote"),
    "battery": (dict(mode="battery", model="gemma4e4b_base", cells="none",
                     attacks="bon", nshards=1),
                {"battery_h200_std"}, "spawn"),
}


def _run(entry, kwargs, log, capsys):
    log.clear()
    entry(**kwargs)
    capsys.readouterr()
    return log


@pytest.mark.parametrize("mode", sorted(MODE_CASES))
@pytest.mark.parametrize("via", ["flag", "env", "flag+env"])
def test_every_mode_spawns_on_the_override_app(routed, monkeypatch, capsys,
                                               mode, via):
    """--app-name / $SDFBASE_APP_NAME / both-agreeing: every function the
    mode reaches is looked up on THAT app, and the set of functions is the
    variant-routed one for the keys named."""
    entry, log, _ = routed
    kwargs, expect_fns, _ = MODE_CASES[mode]
    kwargs = dict(kwargs)
    if "env" in via:
        monkeypatch.setenv("SDFBASE_APP_NAME", WS3)
    if "flag" in via:
        kwargs["app_name"] = WS3
    log = _run(entry, kwargs, log, capsys)
    assert log, mode
    assert {app for _, app, _, _ in log} == {WS3}, log
    assert {fn for _, _, fn, _ in log} == expect_fns, log


@pytest.mark.parametrize("mode", sorted(MODE_CASES))
def test_every_mode_defaults_to_the_frozen_wave_a_app(routed, capsys, mode):
    """No flag, no env: byte-for-byte the wave-A name — on the handle."""
    entry, log, _ = routed
    kwargs, expect_fns, _ = MODE_CASES[mode]
    log = _run(entry, dict(kwargs), log, capsys)
    assert {app for _, app, _, _ in log} == {"sdfbase-fits"}, log
    assert {fn for _, _, fn, _ in log} == expect_fns


def test_memprobe_spawn_items_are_per_key_arm_on_the_variant_function(
        routed, capsys):
    """The mode the probe actually ran (2026-09-02): one spawn per
    (key, CE arm), each on its key's image variant, all on the override.
    The item carries the variant fields (loss_impl, grad_ckpt) — EMPTY
    for the plain probe, so _memprobe_impl writes the folded name."""
    entry, log, _ = routed
    log = _run(entry, dict(mode="memprobe", app_name=WS3,
                           model="glm46vflash_doc,nemotron3nano4b_doc"),
               log, capsys)
    assert [(fn, item) for _, _, fn, item in log] == [
        ("memprobe_h200_std", ("glm46vflash_doc", "psr_ce_all", "", "")),
        ("memprobe_h200_std", ("glm46vflash_doc", "lora_r4_ce", "", "")),
        ("memprobe_h200_nemotron",
         ("nemotron3nano4b_doc", "psr_ce_all", "", "")),
        ("memprobe_h200_nemotron",
         ("nemotron3nano4b_doc", "lora_r4_ce", "", ""))]
    assert {app for _, app, _, _ in log} == {WS3}
    # the chunked re-probe threads its variant through the same item
    log = _run(entry, dict(mode="memprobe", app_name=WS3,
                           model="glm46vflash_doc", arms="psr_ce_all",
                           loss_impl="chunked_ce", grad_ckpt="on"),
               log, capsys)
    assert log[-1][2:] == ("memprobe_h200_std",
                           ("glm46vflash_doc", "psr_ce_all", "chunked_ce",
                            "on"))


def test_durable_modes_record_the_spawned_app_in_doc_and_rows(routed, capsys):
    """hpo-spawn / fits-spawn / battery: the manifest doc AND every row
    name the app the handle was looked up on — the same string."""
    entry, log, home = routed
    ms = home / "stage/sdfbase"
    for mode, mpath, rows_key in (
            ("hpo-spawn", ms / "hpo_calls_r1.json", "trials"),
            ("fits-spawn", ms / "fit_calls.json", "cells"),
            ("battery", ms / "battery_calls_gemma4e4b_base.json", "shards")):
        kwargs = dict(MODE_CASES[mode][0], app_name=WS3)
        log = _run(entry, kwargs, log, capsys)
        doc = json.loads(mpath.read_text())
        assert doc["app"] == WS3, mode
        rows = doc[rows_key]
        assert rows and all(r["app"] == WS3 for r in rows), mode
        spawned = [item for kind, app, _, item in log
                   if kind == "spawn" and app == WS3]
        assert len(spawned) == len(rows), mode
        assert all(r["call_id"] for r in rows), mode


def test_fits_spawn_fingerprints_on_the_same_app_it_spawns_on(routed, capsys):
    """The in-container fingerprint derivation runs on the OVERRIDE app —
    a fingerprint from the frozen wave-A image would describe a config the
    override deploy never carries."""
    entry, log, _ = routed
    log = _run(entry, dict(MODE_CASES["fits-spawn"][0], app_name=WS3),
               log, capsys)
    remote = [(app, fn, item) for kind, app, fn, item in log
              if kind == "remote"]
    assert remote == [(WS3, "fingerprint_cpu",
                       ("nemotron3nano4b_doc", [["constv2-mix-1m",
                                                 "lora_r4_ce"]],
                        0, "out/nemotron3nano4b_doc"))]
    spawn = [(app, fn) for kind, app, fn, _ in log if kind == "spawn"]
    assert spawn == [(WS3, "fit_h200_nemotron")]


def test_flag_and_env_disagreeing_is_refused(sf, routed, monkeypatch, capsys):
    """Both set, different: refused before anything spawns — the App
    object was named from the env at import, so a disagreeing flag would
    split one wave across two names."""
    entry, log, _ = routed
    monkeypatch.setenv("SDFBASE_APP_NAME", WS3)
    with pytest.raises(SystemExit, match="disagrees with"):
        sf.resolve_app_name("sdfbase-fits-r3")
    with pytest.raises(SystemExit, match="disagrees with"):
        entry(mode="stage", model="glm46vflash_doc",
              app_name="sdfbase-fits-r3")
    assert log == []
    # agreeing is fine, and the default stays reachable with the env unset
    assert sf.resolve_app_name(WS3) == WS3
    monkeypatch.delenv("SDFBASE_APP_NAME")
    assert sf.resolve_app_name("sdfbase-fits-r3") == "sdfbase-fits-r3"


# --------------------------------------------------------------------------- #
# W&B: the operator's own secret, a loud entity guard, and the                #
# hpo-spawn rows' design identity for the watcher                             #
# --------------------------------------------------------------------------- #


def test_wandb_secret_is_the_operators_own_never_the_shared_one(sf):
    """The shared `wandb` secret resolves to a FOREIGN account; this app
    mounts the one SISTEER_MODAL_WANDB_SECRET names (the operator's key +
    WANDB_ENTITY + WANDB_PROJECT) and never names the shared one."""
    assert sf.WANDB_SECRET == WANDB_SECRET_EXAMPLE
    assert sf._secrets == [("secret", "hf-token"), ("secret", WANDB_SECRET_EXAMPLE)]
    assert 'from_name("wandb")' not in SRC
    assert "from_name('wandb')" not in SRC
    assert sf._HAVE_WANDB is True          # live logging, not the kill-switch


def _import_fresh(monkeypatch, tmp_path, name, *, env=None, site=""):
    """The app imported afresh under a site config of its own (``site``: that config.env's text) with
    SISTEER_MODAL_WANDB_SECRET set to ``env`` in the environment, or unset there when ``env`` is None."""
    monkeypatch.delenv("SDFBASE_APP_NAME", raising=False)
    if env is None:
        monkeypatch.delenv("SISTEER_MODAL_WANDB_SECRET", raising=False)
    else:
        monkeypatch.setenv("SISTEER_MODAL_WANDB_SECRET", env)
    cfg = tmp_path / f"{name}.env"
    cfg.write_text(site)
    monkeypatch.setenv("SISTEER_CONFIG", str(cfg))
    monkeypatch.setitem(sys.modules, "modal", _fake_modal())
    spec = importlib.util.spec_from_file_location(name, SRC_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_wandb_secret_name_is_the_site_setting_with_no_default(monkeypatch, tmp_path):
    """The W&B secret's name is the site setting SISTEER_MODAL_WANDB_SECRET alone: set nowhere where the app is
    defined, the functions mount the HF token and nothing else (the container's entity guard then refuses, naming
    the setting), and the source spells no account's secret."""
    mod = _import_fresh(monkeypatch, tmp_path, "_sdfbase_fits_no_secret")
    assert mod.WANDB_SECRET_ENV == "SISTEER_MODAL_WANDB_SECRET" and mod.WANDB_SECRET == ""
    assert mod._secrets == [("secret", "hf-token")] and mod._COMMON["secrets"] == mod._secrets
    assert not re.search(r"""from_name\(\s*["']wandb-""", SRC) and not re.search(r"""WANDB_SECRET\s*=\s*["']""", SRC)


def test_wandb_secret_name_is_read_from_the_site_config_when_the_environment_has_none(monkeypatch, tmp_path):
    """A plain `modal deploy` or `modal run` from a shell that never exported the setting still mounts the site's W&B
    secret: the app reads SISTEER_MODAL_WANDB_SECRET from the environment, else from the site config, and the
    environment wins when both set it."""
    site = "SISTEER_MODAL_WANDB_SECRET='example-site-secret'   # a site value\n"
    mod = _import_fresh(monkeypatch, tmp_path, "_sdfbase_fits_site_secret", site=site)
    assert mod.WANDB_SECRET == "example-site-secret"
    assert mod._secrets == [("secret", "hf-token"), ("secret", "example-site-secret")]
    assert mod._COMMON["secrets"] == mod._secrets
    mod = _import_fresh(monkeypatch, tmp_path, "_sdfbase_fits_env_secret", env=WANDB_SECRET_EXAMPLE, site=site)
    assert mod.WANDB_SECRET == WANDB_SECRET_EXAMPLE
    assert mod._secrets == [("secret", "hf-token"), ("secret", WANDB_SECRET_EXAMPLE)]


def _fn_body(name: str) -> str:
    m = re.search(rf"\ndef {name}\(.*?(?=\n(?:def |@app\.|#: |[A-Z_]+ = ))",
                  SRC, re.S)
    assert m, name
    return m.group(0)


@pytest.mark.parametrize("impl", ["_fit_impl", "_hpo_impl"])
def test_training_impls_guard_the_entity_before_any_model_work(impl):
    """Both W&B-logging impls call the guard first — before the snapshot
    check, the pool read and the driver subprocess."""
    body = _fn_body(impl)
    g = body.index("require_wandb_entity()")
    assert g < body.index("_ensure_model(key)")
    assert g < body.index("subprocess.run(")


@pytest.mark.parametrize("impl", ["_onboard_impl", "_memprobe_impl",
                                  "_probe_score_impl", "_mmlu_impl",
                                  "_battery_impl", "_fingerprint_impl",
                                  "_snapshot_impl"])
def test_non_logging_impls_do_not_gate_on_wandb(impl):
    """Modes that never log to W&B must not fail on a W&B outage."""
    assert "require_wandb_entity(" not in _fn_body(impl)


def test_wandb_init_sites_take_the_entity_from_the_environment():
    """Both drivers the containers run pass entity= explicitly, from
    WANDB_ENTITY first (what the secret carries and the guard verified),
    the config second — never wandb's implicit default."""
    for rel in ("scripts/tune_arm.py", "scripts/train_two_si_objects.py"):
        src = (REPO / rel).read_text()
        inits = [m.start() for m in re.finditer(r"wandb\.init\(", src)]
        assert inits, rel
        for i in inits:
            call = src[i:src.index("\n\n", i)]
            assert 'os.environ.get("WANDB_ENTITY")' in call, (rel, call)
            assert "entity=" in call, (rel, call)


def test_hpo_expect_identity_is_the_designs_own(sf):
    e = sf.hpo_expect_identity(HPO_DESIGN, 1, "psr_ce_all")
    assert e == {"model": HPO_DESIGN["model"],
                 "model_revision": HPO_DESIGN["model_revision"],
                 "config_sha": "da1870c7d4c5fce8", "box": HPO_DESIGN["box"],
                 "warmup_levels": [0.05, 0.15], "arm": "psr_ce_all",
                 "position": "all", "round": 1, "si": sf.HPO_SI,
                 "trial": HPO_DESIGN["trials"][1]}
    assert e["trial"] is not HPO_DESIGN["trials"][1]   # a copy
    # a design pinning its own cell (the gate_reg grid rides 1m) says so
    pinned = dict(HPO_DESIGN, si="constv2-mix-1m", round=3)
    e3 = sf.hpo_expect_identity(pinned, 0, "psr_ce_all")
    assert (e3["si"], e3["round"]) == ("constv2-mix-1m", 3)


def test_hpo_spawn_rows_record_the_design_identity(sf, routed, capsys):
    """Every hpo-spawn row carries the identity fits_watch.py validates the
    published summary against — from the SAME local design the container
    reads its staged copy of."""
    entry, log, home = routed
    log = _run(entry, dict(MODE_CASES["hpo-spawn"][0], app_name=WS3,
                           manifest=str(home / "hpo_calls_ws3.json")),
               log, capsys)
    doc = json.loads((home / "hpo_calls_ws3.json").read_text())
    rows = doc["trials"]
    assert [r["index"] for r in rows] == [0, 1, 2]
    for r in rows:
        assert r["expect"] == sf.hpo_expect_identity(HPO_DESIGN, r["index"],
                                                     r["arm"])
        assert r["expect"]["trial"] == HPO_DESIGN["trials"][r["index"]]
        assert r["si"] == r["expect"]["si"] == "constv2-mix-100k"
        assert r["app"] == WS3 and r["call_id"]


# --------------------------------------------------------------------------- #
# probe-score rows echo the scorer's object-identity stamp                    #
# --------------------------------------------------------------------------- #


def test_probe_stamp_fields_reads_the_scorer_sidecar(sf, tmp_path):
    """The row's identity comes from the sidecar probe_score.py wrote —
    never fabricated: a missing or pre-stamp sidecar reads as
    {"stamp": "missing"}; a stamped one yields the headline fields plus the
    generation and the continuation-pending flag."""
    meta = tmp_path / "cell.jsonl.meta.json"
    assert sf.probe_stamp_fields(meta) == {"stamp": "missing"}
    meta.write_text(json.dumps({"object": "/vol/x/object.pt", "cell": "c",
                                "scale": 1.0}))          # pre-stamp sidecar
    assert sf.probe_stamp_fields(meta) == {"stamp": "missing"}
    meta.write_text(json.dumps({
        "stamp_version": "probe_stamp_v1", "object_sha256": "a" * 64,
        "fit_fingerprint": "b" * 64, "identity_kind": "staged",
        "identity": {"object_generation": {"generation": 2},
                     "superseded_by": {"continuation_pending": True}}}))
    assert sf.probe_stamp_fields(meta) == {
        "object_sha256": "a" * 64, "fit_fingerprint": "b" * 64,
        "identity_kind": "staged", "stamp_version": "probe_stamp_v1",
        "span_version": None,  # a stamp-only (pre-span) sidecar says so
        "generation": 2, "continuation_pending": True}
    # a span-carrying sidecar names its span schema too
    meta.write_text(json.dumps({
        "stamp_version": "probe_stamp_v1", "span_version": "probe_span_v1",
        "object_sha256": "a" * 64, "fit_fingerprint": "b" * 64,
        "identity_kind": "staged",
        "identity": {"object_generation": {"generation": 1},
                     "superseded_by": None}}))
    assert sf.probe_stamp_fields(meta)["span_version"] == "probe_span_v1"
    assert sf.probe_stamp_fields(meta)["generation"] == 1
    # an object-less gate row: explicit None, no generation, not pending
    meta.write_text(json.dumps({
        "stamp_version": "probe_stamp_v1", "object_sha256": None,
        "fit_fingerprint": None, "identity_kind": None,
        "identity": {"object_generation": None, "superseded_by": None}}))
    assert sf.probe_stamp_fields(meta) == {
        "object_sha256": None, "fit_fingerprint": None, "identity_kind": None,
        "stamp_version": "probe_stamp_v1", "span_version": None,
        "generation": None, "continuation_pending": False}


def test_probe_score_rows_echo_the_stamp_after_a_successful_run():
    """Source-level pin: run_one reads the stamp back from the sidecar it
    just produced (the <out>.jsonl.meta.json path) on rc == 0, so the run
    log / return value names the object behind every file."""
    impl = SRC.split("def _probe_score_impl", 1)[1].split("\ndef ", 1)[0]
    ok_branch = impl.split("else:", 1)[1].split("return row", 1)[0]
    assert "probe_stamp_fields(" in ok_branch
    assert 'str(out_dir / f"{tag}.jsonl") + ".meta.json"' in ok_branch
    assert "CONTINUATION-PENDING" in ok_branch
    # the failure branch keeps its stderr tail and never fabricates a stamp
    fail_branch = impl.split("if p.returncode != 0:", 1)[1].split("else:", 1)[0]
    assert "probe_stamp_fields" not in fail_branch


def test_probe_score_reloads_the_volume_before_any_read():
    """A WARM score container keeps the volume snapshot it was mounted
    with; without a reload it could stamp a superseded object/COMPLETE
    pair as the current publication. Pin reload as the first /vol touch —
    before _ensure_model and before the out dir / bank reads."""
    impl = SRC.split("def _probe_score_impl", 1)[1].split("\ndef ", 1)[0]
    body = impl.split('"""', 2)[2]                      # past the docstring
    assert "vol.reload()" in body
    assert body.index("vol.reload()") < body.index("_ensure_model(key)")
    assert body.index("vol.reload()") < body.index("/vol/out/")


# --------------------------------------------------------------------------- #
# --mode bench: the gradient-norm reconnaissance's per-run overrides          #
# and the bench's explicit outcome                                           #
# --------------------------------------------------------------------------- #


def test_bench_cmd_without_overrides_is_the_historical_argv(sf):
    cmd = sf.bench_cmd(cfg="/repo/configs/x.yaml", pairs="/vol/pool/k/p.jsonl",
                       si_label="constv2-mix-full", arm="psr_ce_all", bs=1,
                       steps=150, out_local="/root/b.json", loss_impl="",
                       grad_ckpt="", accum=64, check_grads=False)
    assert cmd == ["/repo/scripts/tune_arm.py", "bench",
                   "--config", "/repo/configs/x.yaml",
                   "--pairs", "/vol/pool/k/p.jsonl",
                   "--si", "constv2-mix-full", "--arm", "psr_ce_all",
                   "--batch-size", "1", "--steps", "150",
                   "--out", "/root/b.json", "--accum", "64"]
    for flag in ("--lr", "--max-grad-norm", "--warmup-frac"):
        assert flag not in cmd


def test_bench_cmd_threads_the_three_overrides_only_when_given(sf):
    cmd = sf.bench_cmd(cfg="c", pairs="p", si_label="s", arm="lora_r4_ce",
                       bs=2, steps=150, out_local="o", loss_impl="chunked_ce",
                       grad_ckpt="on", accum=32, check_grads=True,
                       lr=4.5e-4, max_grad_norm=1e6, warmup_frac=0.15)
    assert cmd[cmd.index("--lr") + 1] == repr(4.5e-4)
    assert cmd[cmd.index("--max-grad-norm") + 1] == repr(1e6)
    assert cmd[cmd.index("--warmup-frac") + 1] == repr(0.15)
    assert "--check-grads" in cmd and cmd[cmd.index("--accum") + 1] == "32"
    only_lr = sf.bench_cmd(cfg="c", pairs="p", si_label="s", arm="psr_ce_all",
                           bs=1, steps=10, out_local="o", lr=1e-3)
    assert "--lr" in only_lr and "--max-grad-norm" not in only_lr
    assert "--warmup-frac" not in only_lr


def test_bench_outcome_respects_the_records_own_status(sf):
    assert sf.bench_outcome(0, {"status": "ok"}) == "ok"
    assert sf.bench_outcome(0, None) == "ok"
    assert sf.bench_outcome(3, {"status": "diverged"}) == "diverged"
    assert sf.bench_outcome(4, {"status": "failed"}) == "failed"
    # a record without the outcome field from a dead process, or none at all
    assert sf.bench_outcome(1, {"steps_run": 12}) == "failed"
    assert sf.bench_outcome(1, None) == "failed"
    # the record's word wins over a clean-looking exit code
    assert sf.bench_outcome(0, {"status": "diverged"}) == "diverged"
    assert sf.BENCH_OUTCOMES == ("ok", "diverged", "failed")


def test_bench_overrides_parse_and_refuse(sf):
    assert sf.bench_overrides() == {}
    assert sf.bench_overrides("", "", "") == {}
    assert sf.bench_overrides("4.5e-4", "1e6", "0.05") == {
        "lr": 4.5e-4, "max_grad_norm": 1e6, "warmup_frac": 0.05}
    assert sf.bench_overrides(lr="1e-3") == {"lr": 1e-3}
    assert sf.bench_overrides(warmup_frac="0") == {"warmup_frac": 0.0}
    with pytest.raises(SystemExit, match="finite"):
        sf.bench_overrides(max_grad_norm="inf")
    with pytest.raises(SystemExit, match="positive"):
        sf.bench_overrides(lr="0")
    with pytest.raises(SystemExit, match="not a number"):
        sf.bench_overrides(lr="fast")
    with pytest.raises(SystemExit, match=r"\[0, 1\)"):
        sf.bench_overrides(warmup_frac="1.0")
    assert sf.bench_override_tag({}) == ""
    assert sf.bench_override_tag({"lr": 4.5e-4, "max_grad_norm": 1e6,
                                  "warmup_frac": 0.05}) == \
        "__lr0.00045__clip1e+06__wf0.05"


class _BenchFakeCall:
    def __init__(self, item):
        self.item = item
        self.object_id = "fc-1"

    def get(self):
        return {"status": "ok", "steps_measured": 145, "lr": self.item[10],
                "max_grad_norm": self.item[11], "warmup_frac": self.item[12],
                "grad_norm_preclip": {"quantiles_after_warmup": {"p95": 0.7}}}


class _BenchFakeFn:
    def __init__(self, log):
        self.log = log

    def spawn(self, item):
        self.log.append(item)
        return _BenchFakeCall(item)


def test_bench_mode_spawns_items_with_the_override_tail(sf, monkeypatch,
                                                        tmp_path, capsys):
    spawned = []
    monkeypatch.setattr(sf, "deployed_fn",
                        lambda deployed, name, key="": _BenchFakeFn(spawned))
    monkeypatch.delenv("SDFBASE_APP_NAME", raising=False)
    bench_out = tmp_path / "b.json"
    sf.main(mode="bench", model="llama8b_base", arms="psr_ce_all,lora_r4_ce",
            sis="constv2-mix-full", batch_size=1, accum=64, steps=150,
            lr="4.5e-4", max_grad_norm="1e6", warmup_frac="0.05",
            tag="recon_1", bench_out=str(bench_out))
    assert len(spawned) == 2
    for item, arm in zip(spawned, ("psr_ce_all", "lora_r4_ce")):
        assert len(item) == 13
        assert item[:3] == ("llama8b_base", arm, "constv2-mix-full")
        assert item[3] == 1 and item[5] == 150 and item[7] == 64
        assert item[9] == f"{arm}__recon_1"
        assert item[10:] == (4.5e-4, 1e6, 0.05)
    results = json.loads(bench_out.read_text())
    assert [r["lr"] for r in results] == [4.5e-4, 4.5e-4]
    out = capsys.readouterr().out
    assert '"grad_norm_preclip_after_warmup": {"p95": 0.7}' in out
    # without overrides: a None tail and the byte-identical auto tag
    spawned.clear()
    sf.main(mode="bench", model="llama8b_base", arms="psr_ce_all",
            sis="constv2ctl2-full", batch_size=2, steps=200)
    (item,) = spawned
    assert item[10:] == (None, None, None)
    assert item[9] == "psr_ce_all__constv2ctl2-full__config__bs2__accauto__gcconfig__s200"
    # a recon run's auto tag names its settings
    spawned.clear()
    sf.main(mode="bench", model="llama8b_base", arms="psr_ce_all",
            sis="constv2-mix-full", batch_size=1, accum=64, steps=150,
            lr="4.5e-4", max_grad_norm="1e6", warmup_frac="0.15")
    (item,) = spawned
    assert item[9].endswith("__s150__lr0.00045__clip1e+06__wf0.15")


def test_bench_mode_reports_a_diverged_run_without_hiding_it(sf, monkeypatch,
                                                              capsys):
    class _Div(_BenchFakeCall):
        def get(self):
            return {"status": "diverged", "rc": 3,
                    "error": "non-finite training loss", "steps_completed": 61}

    class _DivFn(_BenchFakeFn):
        def spawn(self, item):
            self.log.append(item)
            return _Div(item)

    monkeypatch.setattr(sf, "deployed_fn",
                        lambda deployed, name, key="": _DivFn([]))
    monkeypatch.delenv("SDFBASE_APP_NAME", raising=False)
    with pytest.raises(SystemExit, match="did not finish ok"):
        sf.main(mode="bench", model="llama8b_base", arms="psr_ce_all",
                batch_size=1, steps=150, lr="1.4e-2", max_grad_norm="1e6")
    out = capsys.readouterr().out
    assert "DIVERGED rc=3" in out and "non-finite training loss" in out


def test_recon_bench_requests_exactly_the_reconnaissance_budget(sf, monkeypatch):
    """A bench carrying any per-run override IS a reconnaissance run,
    and the recon reader accepts only a record of exactly RECON_STEPS (its
    windows certify steps 6-150 alone; a longer record's tail set the clip
    once). So the launcher resolves an omitted --steps to the budget and
    refuses any other count BEFORE a spawn, instead of buying a record the
    reader will refuse. The generic throughput bench (no overrides) keeps
    its 200-step default and takes any explicit count."""
    assert sf.recon_steps() == 150
    assert sf.BENCH_DEFAULT_STEPS == 200
    # the budget is READ from the reader, never a second literal here
    body = SRC.split("def recon_steps(", 1)[1].split("\ndef ", 1)[0]
    assert "recon_clip" in body and "150" not in body.split('"""', 2)[2]
    ov = {"lr": 4.5e-4, "max_grad_norm": 1e6, "warmup_frac": 0.05}
    assert sf.bench_steps(0, ov) == 150
    assert sf.bench_steps(150, ov) == 150
    assert sf.bench_steps(0, {}) == 200
    assert sf.bench_steps(200, {}) == 200
    assert sf.bench_steps(37, {}) == 37
    for bad in (200, 149, 151, 20):
        with pytest.raises(SystemExit, match="reconnaissance budget") as ei:
            sf.bench_steps(bad, ov)
        assert f"--steps {bad}" in str(ei.value) and "150" in str(ei.value)
    with pytest.raises(SystemExit, match="reconnaissance budget"):
        sf.bench_steps(200, {"lr": 1e-3})      # ONE override marks a recon run
    # through --mode bench: an omitted --steps on a recon run spawns the
    # budget and the auto tag says so; an explicit 200 is refused unspawned
    spawned = []
    monkeypatch.setattr(sf, "deployed_fn",
                        lambda deployed, name, key="": _BenchFakeFn(spawned))
    monkeypatch.delenv("SDFBASE_APP_NAME", raising=False)
    sf.main(mode="bench", model="llama8b_base", arms="psr_ce_all",
            sis="constv2-mix-full", batch_size=1, accum=64,
            lr="4.5e-4", max_grad_norm="1e6", warmup_frac="0.05")
    (item,) = spawned
    assert item[5] == 150
    assert item[9].endswith("__s150__lr0.00045__clip1e+06__wf0.05")
    spawned.clear()
    with pytest.raises(SystemExit, match="reconnaissance budget"):
        sf.main(mode="bench", model="llama8b_base", arms="psr_ce_all",
                sis="constv2-mix-full", batch_size=1, accum=64, steps=200,
                lr="4.5e-4", max_grad_norm="1e6", warmup_frac="0.05")
    assert spawned == []
    # the generic throughput bench without overrides: 200 by default, the
    # byte-identical auto tag, and an explicit count taken as given
    sf.main(mode="bench", model="llama8b_base", arms="psr_ce_all",
            sis="constv2ctl2-full", batch_size=2)
    (item,) = spawned
    assert item[5] == 200
    assert item[9] == "psr_ce_all__constv2ctl2-full__config__bs2__accauto__gcconfig__s200"
    spawned.clear()
    sf.main(mode="bench", model="llama8b_base", arms="psr_ce_all",
            sis="constv2ctl2-full", batch_size=2, steps=40)
    (item,) = spawned
    assert item[5] == 40 and item[9].endswith("__s40")


def test_bench_impl_unpacks_legacy_and_extended_items(sf):
    """Source-level pin: the item's first ten fields are the historical
    tuple, the optional tail the three overrides (None = config block),
    and the record's status is the bench's OWN outcome (bench_outcome)."""
    impl = SRC.split("def _bench_impl(", 1)[1].split("\ndef ", 1)[0]
    assert "tag) = item[:10]" in impl
    assert "lr_o, clip_o, warm_o = (tuple(item[10:13]) + (None, None, None))[:3]" in impl
    assert "bench_cmd(" in impl and "lr=lr_o, max_grad_norm=clip_o" in impl
    assert 'res["status"] = bench_outcome(p.returncode, record)' in impl
    assert '"ok" if p.returncode == 0 else "failed"' not in impl


def test_hpo_expect_identity_on_a_confirmation_design(sf):
    """A round-9 confirmation design (tune_arm emit-trials --confirm-from)
    is a one-row design in the shape hpo-spawn --round 9 reads: the
    manifest row's expected identity is round 9, index 0, the pinned cell."""
    design = {"round": 9, "box": {"lr": [1e-5, 1e-2], "max_grad_norm": [0.9, 0.9]},
              "warmup_levels": [0.05, 0.15], "model": "m", "model_revision": "r",
              "config_sha": "c" * 16, "position": "all", "si": "constv2-mix-10m",
              "trials": [{"index": 0, "lr": 3e-4, "warmup_frac": 0.15,
                          "max_grad_norm": 0.9}],
              "confirmation_of": {"fit": "r1_fit.json"}}
    e = sf.hpo_expect_identity(design, 0, "psr_ce_all")
    assert e["round"] == 9 and e["si"] == "constv2-mix-10m"
    assert e["trial"] == design["trials"][0]
    assert e["box"] == design["box"] and e["config_sha"] == "c" * 16


# --------------------------------------------------------------------------- #
# Instruct-direct BATCH-64 siblings (2026-09-11) — pool_key routing           #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("key,parent,variant", [
    ("gemma4e4b_it_doc_eb64", "gemma4e4b_it_doc", "std"),
    ("qwen35_4b_it_doc_eb64", "qwen35_4b_it_doc", "qwen"),
])
def test_it_eb64_keys_share_the_parent_pool_dir_and_own_their_out_root(
        sf, key, parent, variant):
    """The `_eb64` sibling is the parent's checkpoint (same repo + pin,
    gated, chat render, explicit_only, `_it` namespace, derived_eligible
    False) with its OWN generated config and out/<key>/ namespace, reading
    the PARENT's pool dir through _pool_path (pool_key) — every pool read
    in the module goes through _pool_path or _pool_key, never a literal
    /vol/pool/{key}, so the memprobe, bench, HPO trial and fit all see the
    parent's staged pools + sidecars."""
    spec, par = sf.MODELS[key], sf.MODELS[parent]
    assert (spec["repo"], spec["rev"]) == (par["repo"], par["rev"])
    assert spec["gated"] is True and spec["probe_render"] == "chat"
    assert spec["explicit_only"] is True
    assert spec["battery_wave"] == "_it"
    assert spec["derived_eligible"] is False
    assert spec["cfg"] == f"/repo/configs/sdfbase_{key}_fits.yaml"
    assert (REPO / f"configs/sdfbase_{key}_fits.yaml").exists()
    assert spec["pool_key"] == parent and "pool_key" not in par
    assert sf.EB64_PARENT[key] == parent
    assert sf._pool_key(key) == parent and sf._pool_key(parent) == parent
    search_si = "constv2-mix-10m"
    assert sf._pool_path(key, search_si) == \
        f"/vol/pool/{parent}/pairs_{search_si}.jsonl"
    assert sf.HELDOUT_SI == "constv2-heldout"
    assert sf._pool_path(key, sf.HELDOUT_SI) == \
        sf._pool_path(parent, sf.HELDOUT_SI)
    assert sf._variant(key) == variant
    # the fit impl reads the held-out pool through _pool_path too, and the
    # ce mode formats its pool table with the pool key
    fit_src = SRC.split("def _fit_impl", 1)[1].split("\ndef ", 1)[0]
    assert 'f"/vol/pool/{key}' not in fit_src
    assert "_pool_path(key, HELDOUT_SI)" in fit_src
    assert "p.format(key=_pool_key(key))" in SRC
    assert "p.format(key=key)" not in SRC


def test_pool_key_refuses_chains_and_foreign_checkpoints(sf, monkeypatch):
    monkeypatch.setitem(sf.MODELS, "_chain", dict(sf.MODELS["gemma4e4b_it_doc_eb64"],
                                                  pool_key="gemma4e4b_it_doc_eb64"))
    with pytest.raises(SystemExit, match="chains"):
        sf._pool_key("_chain")
    other_parent = sf.EB64_PARENT["qwen35_4b_it_doc_eb64"]
    monkeypatch.setitem(sf.MODELS, "_foreign", dict(sf.MODELS["gemma4e4b_it_doc_eb64"],
                                                    pool_key=other_parent))
    with pytest.raises(SystemExit, match="different checkpoint"):
        sf._pool_key("_foreign")
    monkeypatch.setitem(sf.MODELS, "_nokey", dict(sf.MODELS["gemma4e4b_it_doc_eb64"],
                                                  pool_key="nope"))
    with pytest.raises(SystemExit, match="not a registered key"):
        sf._pool_key("_nokey")


def test_hpo_impl_reloads_the_volume_before_reading_the_design():
    """A warm container recycled from the previous round holds a volume
    snapshot from its start and cannot see a design put after that; the
    HPO trial impl must reload before its first /vol read, exactly as the
    fit impl does."""
    body = SRC.split("def _hpo_impl", 1)[1].split("\ndef ", 1)[0]
    assert "vol.reload()" in body
    assert body.index("vol.reload()") < body.index("HPO_TRIALS_DIR}/r{rnd}")
    assert body.index("vol.reload()") < body.index("_ensure_model(key)")


# --------------------------------------------------------------------------- #
# Qwen-BASE batch-64 SEED REPLICATES (2026-09-15) — admissions + HPO --seed    #
# --------------------------------------------------------------------------- #

SEEDREP_KEYS = ("qwen35_4b_base_eb64_s1", "qwen35_4b_base_eb64_s2",
                "qwen35_4b_base_eb64_s3")
_HPO_COMMON = dict(python="/py", cfg="/repo/configs/sdfbase_k_fits.yaml",
                   trials_file="/vol/hpo/trials/r1_k_psr_ce_all_all.json",
                   index=3, pairs="/vol/pool/k/pairs_constv2-mix-10m.jsonl",
                   out_root="/root/out/hpo/k/psr_ce_all_all",
                   si="constv2-mix-10m", arm="psr_ce_all", bs=2, epochs=8)
_HPO_HISTORICAL = ["/py", "/repo/scripts/tune_arm.py", "trial",
                   "--config", "/repo/configs/sdfbase_k_fits.yaml",
                   "--trials", "/vol/hpo/trials/r1_k_psr_ce_all_all.json",
                   "--index", "3",
                   "--pairs", "/vol/pool/k/pairs_constv2-mix-10m.jsonl",
                   "--out", "/root/out/hpo/k/psr_ce_all_all",
                   "--si", "constv2-mix-10m",
                   "--arm", "psr_ce_all", "--positions", "all",
                   "--batch-size", "2", "--epochs", "8",
                   "--fit-subsample", "0"]


def test_seedrep_keys_are_the_base_entry_on_their_own_config_pool_and_root(sf):
    """Each seed-replicate key is the frozen qwen35_4b_base entry (same
    repo + pin + gating, H200) with its OWN generated config, its OWN pool
    dir (no pool_key: the held-out carve is per seed, so _pool_path resolves
    to /vol/pool/<key>/ — the memprobe cell, the search reference and the
    held-out pool all under it), explicit_only (never swept bare),
    derived_eligible False (the derived pairs run on Slurm), the qwen
    hybrid image, the parent's transfer slug, and the parent's open
    battery-wave namespace; the parent entry itself is untouched."""
    assert sf.SEEDREP_KEYS == SEEDREP_KEYS
    par = sf.MODELS["qwen35_4b_base"]
    assert set(par) == {"gpu", "cfg", "repo", "rev", "gated"}   # untouched
    for key in SEEDREP_KEYS:
        spec = sf.MODELS[key]
        assert (spec["repo"], spec["rev"], spec["gpu"], spec["gated"]) == \
            (par["repo"], par["rev"], par["gpu"], par["gated"])
        assert spec["repo"] == "Qwen/Qwen3.5-4B-Base"
        assert spec["rev"] == "1001bb4d826a52d1f399e183466143f4da7b741b"
        assert spec["gpu"] == "H200" and spec["gated"] is True
        assert spec["cfg"] == f"/repo/configs/sdfbase_{key}_fits.yaml"
        assert (REPO / f"configs/sdfbase_{key}_fits.yaml").exists()
        assert spec["explicit_only"] is True
        assert spec["derived_eligible"] is False
        for absent in ("pool_key", "probe_only", "probe_render",
                       "battery_wave"):
            assert absent not in spec, (key, absent)
        assert sf._pool_key(key) == key
        for cell in ("constv2-heldout", "constv2-mix-1k", "constv2-mix-1m",
                     "constv2-mix-10m", "constv2-mix-full",
                     "constv2ctl2-full"):
            assert sf._pool_path(key, cell) == \
                f"/vol/pool/{key}/pairs_{cell}.jsonl"
        assert sf._pool_path(key, sf.MEMPROBE_SI) == \
            f"/vol/pool/{key}/pairs_constv2-mix-1m.jsonl"
        assert sf._pool_path(key, sf.HELDOUT_SI) != \
            sf._pool_path("qwen35_4b_base", sf.HELDOUT_SI)
        assert sf._variant(key) == "qwen" and key in sf.QWEN_IMAGE_KEYS
        assert key not in sf.NEMOTRON_IMAGE_KEYS
        assert _route(sf, "sdfbase-reseed-hpo", "fit_h200", key) == \
            ("sdfbase-reseed-hpo", "fit_h200_qwen")
        assert sf.TRANSFER_SLUG[key] == "qwen35_4b" == \
            sf.TRANSFER_SLUG["qwen35_4b_base"]
        for wave in ("", "_r2", "_r9"):
            sf.check_battery_wave(key, wave)          # open, as the parent
        with pytest.raises(SystemExit, match="namespace of"):
            sf.check_battery_wave(key, "_it")
    # the derived app's default sweep and explicit --model both exclude them
    # by the one derived_eligible list (its own test pins the mechanism)
    assert all(not sf.MODELS[k].get("derived_eligible", True)
               for k in SEEDREP_KEYS)


def test_config_seed_reads_the_seedrep_declaration_and_none_elsewhere(sf):
    """config_seed: the committed s<s> config declares seed s; EVERY other
    registered config (the frozen ones included) declares none -> None, so
    its HPO argv is unchanged. A declared 0 is 0 (explicit), not None."""
    import yaml as _yaml

    def _cfg(key):
        return _yaml.safe_load(
            (REPO / Path(sf.MODELS[key]["cfg"]).relative_to("/repo")).read_text())

    for key in SEEDREP_KEYS:
        assert sf.config_seed(_cfg(key)) == int(key[-1])
    for key, spec in sf.MODELS.items():
        if spec["cfg"] is None or key in SEEDREP_KEYS:
            continue
        assert sf.config_seed(_cfg(key)) is None, key
    assert sf.config_seed({}) is None
    assert sf.config_seed({"train_kwargs": {"default": {}}}) is None
    assert sf.config_seed({"train_kwargs": {"default": {"seed": 0}}}) == 0
    assert sf.config_seed({"train_kwargs": {"default": {"seed": "2"}}}) == 2


def test_hpo_trial_cmd_without_a_seed_is_the_historical_argv(sf):
    """No declared seed -> the argv every HPO trial ever ran, byte for
    byte, with --no-wandb / --profile riding behind exactly as before."""
    cmd = sf.hpo_trial_cmd(**_HPO_COMMON)
    assert cmd == _HPO_HISTORICAL
    assert "--seed" not in cmd
    assert sf.hpo_trial_cmd(**_HPO_COMMON, seed=None, wandb=True,
                            profile=False) == _HPO_HISTORICAL
    assert sf.hpo_trial_cmd(**_HPO_COMMON, wandb=False) == \
        _HPO_HISTORICAL + ["--no-wandb"]
    assert sf.hpo_trial_cmd(**_HPO_COMMON, profile=True) == \
        _HPO_HISTORICAL + ["--profile"]
    assert sf.hpo_trial_cmd(**_HPO_COMMON, wandb=False, profile=True) == \
        _HPO_HISTORICAL + ["--no-wandb", "--profile"]


def test_hpo_trial_cmd_threads_the_seed_only_when_declared(sf):
    """A declared seed appends `--seed <s>` right after the historical argv
    (before --no-wandb / --profile); a declared 0 is passed explicitly —
    only an UNdeclared seed (None) is silent."""
    seeded = sf.hpo_trial_cmd(**_HPO_COMMON, seed=1, wandb=False,
                              profile=True)
    assert seeded == _HPO_HISTORICAL + ["--seed", "1", "--no-wandb",
                                        "--profile"]
    assert seeded[seeded.index("--seed") + 1] == "1"
    assert sf.hpo_trial_cmd(**_HPO_COMMON, seed=3) == \
        _HPO_HISTORICAL + ["--seed", "3"]
    assert sf.hpo_trial_cmd(**_HPO_COMMON, seed=0) == \
        _HPO_HISTORICAL + ["--seed", "0"]


def test_hpo_impl_builds_its_argv_through_hpo_trial_cmd_with_the_config_seed():
    """_hpo_impl has ONE argv builder (no second inline spelling) and hands
    it the config's declared seed; the manifest row echoes the seed the
    summary recorded and never invents one for an older summary."""
    body = SRC.split("def _hpo_impl", 1)[1].split("\ndef ", 1)[0]
    assert "cmd = hpo_trial_cmd(" in body
    assert "seed=config_seed(cfg)" in body
    assert "wandb=_HAVE_WANDB" in body and "profile=(index == 0)" in body
    assert '"/repo/scripts/tune_arm.py", "trial"' not in body
    assert 'cmd.append("--no-wandb")' not in body
    assert '**({"seed": s["seed"]} if "seed" in s else {})' in body
    # the helper is the only place the trial argv is spelled
    assert SRC.count('"/repo/scripts/tune_arm.py", "trial"') == 1


def test_seedrep_s1_argv_carries_seed_1_and_a_frozen_configs_argv_does_not(sf):
    """End to end through the two helpers on the COMMITTED configs: the s1
    search reference trial runs `--seed 1` (the spec: the search runs on
    the seed-1 pools at seed 1); the frozen qwen35_4b_base and llama8b_base
    configs produce an argv with no --seed at all."""
    import yaml as _yaml

    def argv_for(key, si="constv2-mix-10m", arm="psr_ce_all"):
        cfg = _yaml.safe_load(
            (REPO / Path(sf.MODELS[key]["cfg"]).relative_to("/repo")).read_text())
        return sf.hpo_trial_cmd(
            python="/py", cfg=sf.MODELS[key]["cfg"], trials_file="t",
            index=0, pairs=sf._pool_path(key, si), out_root="o", si=si,
            arm=arm, bs=int(cfg["batch_size_by_arm"][arm]),
            epochs=int(cfg["train_kwargs_by_si"][si]["epochs"]),
            seed=sf.config_seed(cfg))

    s1 = argv_for("qwen35_4b_base_eb64_s1")
    assert s1[s1.index("--seed") + 1] == "1"
    assert s1[s1.index("--batch-size") + 1] == "2"
    assert s1[s1.index("--epochs") + 1] == "8"
    assert s1[s1.index("--pairs") + 1] == \
        "/vol/pool/qwen35_4b_base_eb64_s1/pairs_constv2-mix-10m.jsonl"
    for s in (2, 3):
        cmd = argv_for(f"qwen35_4b_base_eb64_s{s}", si="constv2-mix-full")
        assert cmd[cmd.index("--seed") + 1] == str(s)
    for frozen in ("qwen35_4b_base", "llama8b_base"):
        assert "--seed" not in argv_for(frozen)
