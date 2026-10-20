"""Tests for the domains.yaml registry — load/refusals,
task→domain resolution, gating-set assembly, acceptance-role exclusion."""

from pathlib import Path

import pytest
import yaml

from sisteer.capeval.domains import Registry, RegistryError

ROOT = Path(__file__).parents[1]
DOMAINS_YAML = ROOT / "configs" / "capeval" / "domains.yaml"

GATING = ("lm_fidelity_raw", "lm_fidelity_chat", "knowledge", "commonsense",
          "truthfulness", "math", "code", "instruction_following", "reasoning")
ACCEPTANCE = ("math_s2parity", "knowledge_s2parity")


@pytest.fixture(scope="module")
def registry() -> Registry:
    return Registry.load(DOMAINS_YAML)


def _write_registry(tmp_path, domains: dict) -> Path:
    p = tmp_path / "domains.yaml"
    p.write_text(yaml.safe_dump({"version": 1, "domains": domains}))
    return p


_ACC = {"kind": "accuracy", "direction": "up", "margin": 0.02, "gating": True,
        "unit": "fraction_correct", "transform": "identity", "task_weight": 1,
        "item_weight": "items", "estimator": "paired_bootstrap",
        "tasks": ["task_a"]}


# --------------------------------------------------------------------------
# the checked-in v0 registry
# --------------------------------------------------------------------------


def test_v0_registry_loads(registry):
    assert registry.version == 1
    names = tuple(d.name for d in registry.domains)
    # 14 capability domains + the petri_behavior display domain (a
    # deliberate additive registration, never gating) + the three tinycap
    # display domains (served_ll, inspect_gen, evalplus_served; additive,
    # never gating) + the
    # misalign display domain (the misalignment evals over the served cells:
    # agentic leaking/murder + Betley 8Q, paired per item in
    # capeval.misalign.analyze — additive, never gating; em-misalign lane)
    assert len(names) == 19
    assert tuple(d.name for d in registry.domains if d.kind == "display") == (
        "kl_localization", "petri_behavior", "served_ll", "inspect_gen", "evalplus_served", "misalign")
    assert tuple(d.name for d in registry.domains if d.gating) == GATING
    assert tuple(d.name for d in registry.domains
                 if d.role == "acceptance") == ACCEPTANCE
    # 21 capability tasks + 6 petri/<behavior> tasks + 18 tinycap served
    # tasks (5 served_ll/ + 9 inspect/ + 4 evalplus/ — the ninth inspect/
    # task is inspect/gsm8k_paired200, the GSM8K comparison bridge, PW3),
    # each mapping to exactly one domain
    all_tasks = [t for d in registry.domains for t in d.tasks]
    # ... + inspect/bigcodebench_200 and inspect/bfcl_200 (bcb-bfcl-200): 47
    # ... + the 3 misalign display tasks agentic/leaking, agentic/murder,
    # em8q/betley (em-misalign lane): 50
    # 50 + inspect/ifeval (ifeval-canonical-battery) + inspect/bigcodebench_full + inspect/bfcl_full (capeval-full-tasks)
    assert len(all_tasks) == len(set(all_tasks)) == 53
    assert registry.domain_of("inspect/bigcodebench_full").name == "inspect_gen"
    assert registry.domain_of("inspect/bfcl_full").kind == "display"
    # PW3: exactly one RETIRED row — math_s2parity (the bespoke GSM8K); it
    # keeps its role/kind so existing stores' rows still resolve
    assert [d.name for d in registry.domains if d.retired] == ["math_s2parity"]


def test_domain_of_resolutions(registry):
    assert registry.domain_of("gsm8k_platinum_cap").name == "math"
    assert registry.domain_of("tier0/nll_wikitext103").name == "lm_fidelity_raw"
    assert registry.domain_of("tier0/canary").kind == "gate"
    assert registry.domain_of("tier0/kl_localization").kind == "display"
    assert registry.domain_of("s2parity/gsm8k_0shot").role == "acceptance"
    assert registry.domain_of("s2parity/gsm8k_0shot").retired is True
    assert registry.domain_of("s2parity/mmlu_letter").retired is False
    assert registry.domain_of("inspect/gsm8k").name == "inspect_gen"
    assert registry.domain_of("inspect/gsm8k_paired200").kind == "display"
    assert registry.domain_of("s2parity/mmlu_letter").gating is False
    assert registry.domain_of("ifeval").name == "instruction_following"
    # the tinycap served rows file under their display domains
    assert registry.domain_of("served_ll/tinyArc").name == "served_ll"
    assert registry.domain_of("inspect/bfcl").kind == "display"
    # bcb-bfcl-200: the two pinned legs are display rows of the same domain
    assert registry.domain_of("inspect/bigcodebench_200").name == "inspect_gen"
    assert registry.domain_of("inspect/bfcl_200").kind == "display"
    assert registry.domain_of("inspect/bigcodebench_200").gating is False
    assert registry.domain_of("evalplus/mbpp_plus:open").name == "evalplus_served"
    assert registry.domain_of("served_ll/tinyMMLU").gating is False


def test_unknown_task_refused(registry):
    with pytest.raises(RegistryError, match="not registered"):
        registry.domain_of("mystery_benchmark")


def test_domain_named(registry):
    assert registry.domain_named("math").tasks == ("gsm8k_platinum_cap",)
    with pytest.raises(RegistryError, match="no domain named"):
        registry.domain_named("nope")


def test_f14_fields_complete_on_checked_in_registry(registry):
    for d in registry.domains:
        if d.kind in ("accuracy", "nll"):
            assert None not in (d.unit, d.transform, d.task_weight,
                                d.item_weight, d.estimator), d.name
            assert d.direction in ("up", "down") and d.margin > 0, d.name


# --------------------------------------------------------------------------
# gating-set assembly (§2.2/§7.4)
# --------------------------------------------------------------------------


def test_gating_set_plain(registry):
    names = [d.name for d in registry.gating_set("m", {}, [])]
    assert tuple(names) == GATING  # 9 domains, registry order


def test_gating_set_exclusions_and_acceptance(registry):
    names = {d.name for d in
             registry.gating_set("m", {}, ["instruction_following"])}
    assert names == set(GATING) - {"instruction_following"}
    assert not names & set(ACCEPTANCE)  # structurally excluded (r2-F6)


def test_gating_set_unknown_exclusion_refused(registry):
    with pytest.raises(RegistryError, match="unknown exclusion"):
        registry.gating_set("m", {}, ["typo_domain"])


def test_gating_set_power_table_never_removes_on_v0(registry):
    # An unpowered domain stays in the gating set on v0 (non-confirmatory)
    power = {("m", name): 0.05 for name in GATING}
    names = [d.name for d in registry.gating_set("m", power, [])]
    assert tuple(names) == GATING
    with pytest.raises(RegistryError, match="power_table"):
        registry.gating_set("m", {"not-a-tuple": 0.5}, [])


# --------------------------------------------------------------------------
# loader refusals
# --------------------------------------------------------------------------


def test_duplicate_task_refused(tmp_path):
    p = _write_registry(tmp_path, {
        "a": dict(_ACC),
        "b": dict(_ACC, tasks=["task_a"]),
    })
    with pytest.raises(RegistryError, match="exactly one"):
        Registry.load(p)


def test_missing_f14_field_refused(tmp_path):
    for missing in ("unit", "transform", "task_weight", "item_weight",
                    "estimator"):
        spec = dict(_ACC)
        del spec[missing]
        p = _write_registry(tmp_path, {"a": spec})
        with pytest.raises(RegistryError, match=missing):
            Registry.load(p)


def test_gating_acceptance_contradiction_refused(tmp_path):
    p = _write_registry(tmp_path, {"a": dict(_ACC, role="acceptance")})
    with pytest.raises(RegistryError, match="contradiction"):
        Registry.load(p)


def test_bad_enum_values_refused(tmp_path):
    cases = [
        (dict(_ACC, kind="winrate"), "kind"),
        (dict(_ACC, direction="sideways"), "direction"),
        (dict(_ACC, role="display-only"), "role"),
        (dict(_ACC, item_weight="chars"), "item_weight"),
        (dict(_ACC, estimator="t_test"), "estimator"),
        (dict(_ACC, transform="logit"), "transform"),
        (dict(_ACC, margin=0), "margin"),
        (dict(_ACC, tasks=[]), "tasks"),
        (dict(_ACC, unknown_field=1), "unknown key"),
        (dict(_ACC, retired="yes"), "retired"),   # PW3: a bool annotation only
    ]
    for spec, match in cases:
        p = _write_registry(tmp_path, {"a": spec})
        with pytest.raises(RegistryError, match=match):
            Registry.load(p)


def test_retired_is_an_annotation_that_changes_no_semantics(tmp_path):
    """PW3: `retired: true` marks the row; gating / role / statuses are
    untouched, and the default is False."""
    p = _write_registry(tmp_path, {
        "a": dict(_ACC, retired=True),
        "b": dict(_ACC, tasks=["task_b"]),
        "d": {"kind": "display", "tasks": ["t_d"], "retired": True},
    })
    reg = Registry.load(p)
    by = {d.name: d for d in reg.domains}
    assert by["a"].retired is True and by["a"].gating is True
    assert by["b"].retired is False
    assert by["d"].retired is True and by["d"].kind == "display"
    assert [d.name for d in reg.gating_set("m", {}, [])] == ["a", "b"]


def test_kind_item_weight_pairing_enforced(tmp_path):
    nll = dict(_ACC, kind="nll", direction="down", margin=0.03,
               unit="nats_per_token", transform="relative_nll",
               item_weight="items")
    p = _write_registry(tmp_path, {"a": nll})
    with pytest.raises(RegistryError, match="tokens"):
        Registry.load(p)
    acc = dict(_ACC, item_weight="tokens")
    p = _write_registry(tmp_path, {"a": acc})
    with pytest.raises(RegistryError, match="items"):
        Registry.load(p)


def test_gate_display_constraints(tmp_path):
    p = _write_registry(tmp_path, {"g": {"kind": "gate", "tasks": ["t"],
                                         "margin": 0.02}})
    with pytest.raises(RegistryError, match="accuracy/nll-only"):
        Registry.load(p)
    p = _write_registry(tmp_path, {"g": {"kind": "gate", "tasks": ["t"],
                                         "gating": True}})
    with pytest.raises(RegistryError, match="never"):
        Registry.load(p)
    p = _write_registry(tmp_path, {"g": {"kind": "display", "tasks": ["t"],
                                         "role": "acceptance"}})
    with pytest.raises(RegistryError, match="role"):
        Registry.load(p)


def test_top_level_refusals(tmp_path):
    p = tmp_path / "d.yaml"
    p.write_text(yaml.safe_dump({"version": "one", "domains": {"a": _ACC}}))
    with pytest.raises(RegistryError, match="version"):
        Registry.load(p)
    p.write_text(yaml.safe_dump({"version": 1, "domains": {}, "extras": 1}))
    with pytest.raises(RegistryError, match="unknown top-level"):
        Registry.load(p)
    p.write_text(yaml.safe_dump({"version": 1, "domains": {}}))
    with pytest.raises(RegistryError, match="non-empty"):
        Registry.load(p)
