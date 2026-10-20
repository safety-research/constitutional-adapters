"""The OPT-IN Inspect replicate seed rule for GSM8K paired-200.

`contracts.seed_rules_for` / `sampling_identity` are byte-identical for every run that does not opt in (the IFEval
conditional-extension pattern: no existing identity or golden is re-keyed); under `--inspect-seed-policy replicate
--inspect-seed-base <base>` the hashed `inspect` rule string carries the base (two bases = two stores), the ONE dispatch
`inspect_served.seed_for_task` draws `contracts.inspect_replicate_seed` on every non-IFEval task, and the driver's parser
takes the two flags. The `qwen_seedrep` matrix machinery — the views file's `inspect_seed_policy` / `inspect_seed_base_by_view`
(pinned to eval_seed's rule), the appended row fields, the row printer's two shell names, g8k_driver_argv's flags — is driven
end to end over a SYNTHETIC roster + pins (21 rows: per view the natively served `none` anchor + the six, decisions O12a),
because the committed matrix regenerates from the PROVISIONAL pre-staging freeze (three held anchor rows) until the staging PR
re-freezes the roster; the matrix is REGISTERED (tests/test_gsm8k_qwen_seedrep.py pins the registration, the companion machinery and
the launchers). Everything synthetic: placeholder pins, scratch trees."""
from __future__ import annotations

import contextvars
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
for p in (REPO / "src", REPO / "scripts", REPO / "scripts" / "sdfbase" / "gsm8k", REPO / "scripts" / "sdfbase" / "seedrep"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import eval_seed as ES  # noqa: E402
import g8k_cells as M  # noqa: E402
import g8k_views as V  # noqa: E402
import make_gsm8k_matrix as MK  # noqa: E402
from sisteer.capeval.tiers import inspect_served as isv  # noqa: E402
from sisteer.capeval.tinycap import contracts as C  # noqa: E402

G8K = REPO / "scripts/sdfbase/gsm8k"
SEEDREP_VIEWS = REPO / "configs/sdfbase/gsm8k/views_qwen_seedrep.yaml"
GSM8K_TASK = C.inspect_task("gsm8k_paired200")
IFEVAL_TASK = C.inspect_task(C.IFEVAL_TASK_NAME)
B1, B2, B3 = ES.seed_base(1), ES.seed_base(2), ES.seed_base(3)
PARENTS = ["constv2-mix-full__psr_ce_all", "constv2-mix-full__lora_r4_ce", "constv2ctl2-full__psr_ce_all", "constv2ctl2-full__lora_r4_ce"]
DERIVED = ["constv2-mix-full__lora_r4_ce_minus_ctl", "constv2-mix-full__psr_ce_all_minus_ctl_distilled"]
CELLS6 = PARENTS + DERIVED


# --------------------------------------------------------------------------- contracts: the rule and the identity
def test_default_identities_are_byte_unchanged_and_the_opt_in_rule_is_distinct():
    assert C.INSPECT_SEED_POLICIES == ("default", "replicate") and C.FLAG_INSPECT_SEED_POLICY == "--inspect-seed-policy"
    assert C.FLAG_INSPECT_SEED_BASE == "--inspect-seed-base" and C.INSPECT_REPLICATE_SEED_MODULUS == 2 ** 31 - 1
    tasks = {GSM8K_TASK: 1}
    frozen = {"inspect": C.SEED_RULE_INSPECT, "evalplus": C.SEED_RULE_EVALPLUS}
    # the default: exactly what it was (the frozen rules; IFEval's conditional extension untouched)
    assert C.seed_rules_for(tasks) == frozen == dict(C.SEED_RULES) == C.seed_rules_for(tasks, inspect_seed_policy="default")
    assert C.seed_rules_for({GSM8K_TASK: 1, IFEVAL_TASK: 2}) == {**frozen, IFEVAL_TASK: C.SEED_RULE_IFEVAL}
    # the opt-in: the `inspect` rule REPLACED by the replicate rule carrying the base; nothing else moves
    rep = C.seed_rules_for(tasks, inspect_seed_policy="replicate", inspect_seed_base=B1)
    assert rep == {"inspect": C.inspect_replicate_seed_rule(B1), "evalplus": C.SEED_RULE_EVALPLUS}
    assert rep["inspect"] == f"sha256(f'{{seed_base}}|inspect-replicate|{{item_id}}|{{sample_idx}}')[:4] big-endian mod 2^31-1; seed_base = {B1}"
    assert C.SEED_RULE_INSPECT_REPLICATE.format(seed_base=B1) == rep["inspect"] and str(B1) in rep["inspect"]
    assert C.seed_rules_for(tasks, inspect_seed_policy="replicate", inspect_seed_base=B2)["inspect"] != rep["inspect"]
    both = C.seed_rules_for({GSM8K_TASK: 1, IFEVAL_TASK: 2}, inspect_seed_policy="replicate", inspect_seed_base=B1)
    assert both == {**rep, IFEVAL_TASK: C.SEED_RULE_IFEVAL}
    # the hashed sampling identity: the default block is the pre-change block; replicate differs ONLY in seed_rule.inspect
    row = C.sampling_for("qwen35_4b", C.SAMPLING_CARD)
    base_id = C.sampling_identity(row, tasks)
    assert base_id[C.IDENTITY_KEY_SEED_RULE] == frozen and set(C.SAMPLING_IDENTITY_FIELDS) <= set(base_id)
    assert C.sampling_identity(row, tasks, inspect_seed_policy="default") == base_id
    rep_id = C.sampling_identity(row, tasks, inspect_seed_policy="replicate", inspect_seed_base=B1)
    assert set(rep_id) == set(base_id) and {k: v for k, v in rep_id.items() if k != C.IDENTITY_KEY_SEED_RULE} == {k: v for k, v in base_id.items() if k != C.IDENTITY_KEY_SEED_RULE}
    assert rep_id[C.IDENTITY_KEY_SEED_RULE] == rep
    assert C.sampling_ref({**base_id}) != C.sampling_ref({**rep_id})
    # the pair check
    assert C.check_inspect_seed_policy(None, None) == ("default", None) == C.check_inspect_seed_policy("default", None)
    assert C.check_inspect_seed_policy("replicate", B3) == ("replicate", B3)
    for policy, base, needle in (("default", B1, "takes no --inspect-seed-base"), ("replicate", None, "requires a non-negative int"),
                                 ("replicate", -1, "requires a non-negative int"), ("replicate", True, "requires a non-negative int"),
                                 ("frozen", None, "not one of")):
        with pytest.raises(ValueError, match=needle):
            C.check_inspect_seed_policy(policy, base)
    # the derivation: deterministic, in range, distinct across bases / items / indices, never the default seed for these items
    a = C.inspect_replicate_seed("gsm8k_platinum_0001", 0, B1)
    assert a == C.inspect_replicate_seed("gsm8k_platinum_0001", 0, B1) and 0 <= a < 2 ** 31 - 1
    assert a != C.inspect_replicate_seed("gsm8k_platinum_0001", 0, B2) != C.inspect_replicate_seed("gsm8k_platinum_0002", 0, B2)
    assert a != C.inspect_replicate_seed("gsm8k_platinum_0001", 1, B1) and a != isv.sample_seed("gsm8k_platinum_0001", 0)
    for bad in ((True, 0), (-1, 0), ("", 0)):
        with pytest.raises(ValueError):
            C.inspect_replicate_seed(bad[1] if isinstance(bad[0], int) else bad[0], bad[0] if isinstance(bad[0], int) else 0, B1)


# --------------------------------------------------------------------------- inspect_served: the ONE dispatch
def test_the_dispatch_draws_the_replicate_rule_under_the_policy_and_the_default_otherwise():
    item = "gsm8k_platinum_0042"
    assert isv.inspect_seed_policy() is None
    assert isv.seed_for_task(GSM8K_TASK, item, 0) == isv.sample_seed(item, 0) and isv.seed_for_task(None, item, 3) == isv.sample_seed(item, 3)
    tok = isv.set_inspect_seed_policy("replicate", B1)
    try:
        assert isv.inspect_seed_policy() == ("replicate", B1)
        assert isv.seed_for_task(GSM8K_TASK, item, 0) == C.inspect_replicate_seed(item, 0, B1) != isv.sample_seed(item, 0)
        assert isv.seed_for_task(None, item, 2) == C.inspect_replicate_seed(item, 2, B1)
        assert isv.seed_for_task(IFEVAL_TASK, "1000", 1) == isv.ifeval_sample_seed("1000", 1)          # IFEval keeps its own rule
        # an out-of-band reader names the policy explicitly and gets the same seed; another base another seed
        assert isv.seed_for_task(GSM8K_TASK, item, 0, policy=("replicate", B1)) == C.inspect_replicate_seed(item, 0, B1)
        assert isv.seed_for_task(GSM8K_TASK, item, 0, policy=("replicate", B2)) == C.inspect_replicate_seed(item, 0, B2)
        # the tasks Inspect spawns copy the context: a copied context sees the policy
        assert contextvars.copy_context().run(isv.seed_for_task, GSM8K_TASK, item, 0) == C.inspect_replicate_seed(item, 0, B1)
    finally:
        isv._INSPECT_SEED_POLICY.reset(tok)
    assert isv.inspect_seed_policy() is None and isv.seed_for_task(GSM8K_TASK, item, 0) == isv.sample_seed(item, 0)
    tok = isv.set_inspect_seed_policy("default")
    try:
        assert isv.inspect_seed_policy() is None                                                        # default stores None: byte-identical
    finally:
        isv._INSPECT_SEED_POLICY.reset(tok)
    with pytest.raises(ValueError, match="requires a non-negative int"):
        isv.set_inspect_seed_policy("replicate")
    with pytest.raises(isv.InspectServedError, match="unknown Inspect seed policy"):
        isv.seed_for_task(GSM8K_TASK, item, 0, policy=("frozen", 0))
    assert isv.SEED_RULE == C.SEED_RULE_INSPECT                                                         # the default rule's name unchanged


# --------------------------------------------------------------------------- the driver
def test_the_driver_takes_the_flags_and_threads_the_policy_into_identity_manifest_and_dispatch():
    import served_capability_run as drv          # the canonical name: the driver binds sampled_controls ONCE per process (bind_driver_globals)
    ap = drv.build_parser()
    base = ["generate", "--cell", "none", "--config", "c.yaml", "--out-root", "/o", "--proxy-url", "u", "--routing-table", "t", "--defenses-dir", "/d"]
    ns = ap.parse_args(base)
    assert ns.inspect_seed_policy == "default" and ns.inspect_seed_base is None
    ns = ap.parse_args(base + ["--inspect-seed-policy", "replicate", "--inspect-seed-base", str(B1)])
    assert ns.inspect_seed_policy == "replicate" and ns.inspect_seed_base == B1
    with pytest.raises(SystemExit):
        ap.parse_args(base + ["--inspect-seed-policy", "frozen"])
    # the manifest's seeds block: default byte-identical (no `replicate` key, the frozen rule); replicate records the rule + base
    fam = [C.CONTROL_GSM8K]
    default_block = drv._seeds_block(fam, {GSM8K_TASK: 1})
    assert default_block["inspect"]["rule"] == C.SEED_RULE_INSPECT and "replicate" not in default_block["inspect"]
    assert drv._seeds_block(fam, {GSM8K_TASK: 1}, inspect_seed_policy="default") == default_block
    rep_block = drv._seeds_block(fam, {GSM8K_TASK: 1}, inspect_seed_policy="replicate", inspect_seed_base=B1)
    assert rep_block["inspect"]["rule"] == C.inspect_replicate_seed_rule(B1)
    assert rep_block["inspect"]["replicate"] == {"policy": "replicate", "seed_base": B1, "derivation": "contracts.inspect_replicate_seed (inspect_served.seed_for_task)"}
    # the plumbing, by name: the resolver validates the pair before anything is read, the identity / block / manifest / the
    # inspect runner and the sync context all take the run's policy
    src = (REPO / "scripts/served_capability_run.py").read_text()
    for needle in ('tc.check_inspect_seed_policy(getattr(args, "inspect_seed_policy", None)',
                   "return tc.sampling_identity(sampling.row, sampling.samples_by_task, **_seed_policy_kw(sampling))",
                   "**tc.sampling_identity(sampling.row, sampling.samples_by_task, **_seed_policy_kw(sampling)),",
                   "tc.MANIFEST_KEY_SEEDS: _seeds_block(families, sampling.samples_by_task, **_seed_policy_kw(sampling)),",
                   "_isv_policy.set_inspect_seed_policy(sampling.inspect_seed_policy, sampling.inspect_seed_base)",
                   "inspect_seed_policy=sampling.inspect_seed_policy, inspect_seed_base=sampling.inspect_seed_base))",
                   "nothing to seed; drop the flag"):
        assert needle in src, needle
    assert src.index("_isv_policy.set_inspect_seed_policy(") < src.index("asyncio.run(_run())")   # BEFORE the event loop starts
    assert "inspect_seed_policy: Optional[str] = None,\n                            inspect_seed_base: Optional[int] = None" in (REPO / "src/sisteer/capeval/tiers/inspect_served.py").read_text()


# --------------------------------------------------------------------------- the matrix machinery over a synthetic tree
def _views_without_frozen(text: str) -> str:
    """The committed views file with its `frozen_inputs:` block removed — a synthetic tree's generator reads the tree's OWN live roster
    + pins (the frozen copies are the real tree's provisional freeze; g8k_views.frozen_input_paths would look for them under the scratch root)."""
    out = re.sub(r"\nfrozen_inputs:\n(?:  .*\n)+", "\n", text)
    assert "frozen_inputs:" not in out.split("\nschema:", 1)[1]
    return out


def _views_seed_only(text: str) -> str:
    """The committed views file projected onto the SEED views s1-s3: the PAIRED views s<s>p (the composed difference_relu arm on
    its own additive view per seed), their seed bases and their holds removed, so this synthetic scenario keeps exercising the
    seed-view machinery over a roster that stages the six cells on s1-s3 alone."""
    doc = yaml.safe_load(text)
    paired = lambda v: bool(re.fullmatch(r"s[1-9]p", str(v)))
    doc["views"] = {k: [v for v in vs if not paired(v)] for k, vs in doc["views"].items()}
    doc["inspect_seed_base_by_view"] = {v: b for v, b in doc["inspect_seed_base_by_view"].items() if not paired(v)}
    doc["holds"] = [h for h in doc.get("holds") or [] if not paired(h["view"])]
    return "# the committed views file projected onto the seed views (comments dropped)\n" + yaml.safe_dump(doc, sort_keys=False)


def _synthetic_tree(tmp_path: Path, *, views_text: str | None = None, seeds=(1, 2, 3), drop_cells=()) -> Path:
    root = tmp_path / "tree"
    (root / "configs/sdfbase/gsm8k").mkdir(parents=True)
    views = {}
    pins = {}
    for s in seeds:
        cells = {}
        for c in CELLS6:
            if (s, c) in drop_cells:
                continue
            sid = ("derived_base64s" if c in DERIVED else "base64s") + f"{s}__{c}"
            cells[c] = sid
            pins[sid] = {"object_sha256": f"{s:02x}" * 32}          # SYNTHETIC 64-hex placeholder, never a real object
        views[f"s{s}"] = {"config": f"configs/sdfbase_full22_qwen35_4b_s{s}.yaml",
                          "manifest": f"configs/capeval/manifests/petri_full22_qwen35_4b_s{s}.yaml", "cells": cells}
    roster = {"targets": {"qwen35_4b": {"model": "Qwen/Qwen3.5-4B", "model_revision": "0" * 40, "views": views}}}
    (root / "configs/sdfbase_full22_roster.yaml").write_text(yaml.safe_dump(roster, sort_keys=False))
    (root / "configs/sdfbase_full22_pins.json").write_text(json.dumps({"qwen35_4b": pins}, indent=1))
    (root / "configs/sdfbase/gsm8k/views_qwen_seedrep.yaml").write_text(_views_without_frozen(views_text if views_text is not None else _views_seed_only(SEEDREP_VIEWS.read_text())))
    return root


def _driver_argv(row_sh: str, extra: str = "") -> list:
    """g8k_driver_argv RUN under bash from a row's shell facts (the launcher's own function, never re-typed)."""
    stubs = (f'set -u\nf22_log(){{ :; }}\nf22_fatal(){{ echo FATAL "$@" >&2; exit 1; }}\nREPO_DIR={shlex.quote(str(REPO))}\n'
             f'source {shlex.quote(str(G8K / "g8k_common.sh"))}\n' + M.literals_sh() + row_sh +
             'CPY=/venv/bin/python; PROXY_URL=http://127.0.0.1:1/v1; TABLE=/serve/routing_table.json; LEDGER_URL=http://127.0.0.1:2/v1; '
             'CFG=configs/x.yaml; TT_TWO_SI_DEFENSES_DIR=/bank/view; OUT_ROOT=/out; TARGET_S=44; SERVE_DIR=/serve; ATTEMPT_ID=a1; RUNG_SUBSET=/out/rung_subset.json\n'
             + extra + 'g8k_driver_argv\nfor a in "${DRV[@]}"; do printf "%s\\n" "$a"; done\n')
    r = subprocess.run(["bash", "-c", stubs], capture_output=True, text=True, check=False)
    assert r.returncode == 0, r.stderr
    return r.stdout.splitlines()


def test_the_seedrep_matrix_machinery_over_a_synthetic_roster_and_pins(tmp_path):
    spec = V.load_views_yaml(SEEDREP_VIEWS)
    assert spec.schema == M.MATRIX_SCHEMA_2 and spec.wave == M.WAVE_QWEN_SEEDREP == "qwen_seedrep" and spec.row_overlays and not spec.object_generation
    assert spec.wave_root_rel == "exp/gsm8k_paired200_paper_objects_qwen_seedrep" and spec.trigger_threshold == 0.01
    # + the three PAIRED views s<s>p, whose companion + LoRA-difference rows are HELD
    assert spec.views == {"qwen35_4b": ["s1", "s2", "s3", "s1p", "s2p", "s3p"]} and spec.steering == [] and spec.itdoc == []
    assert [(h["view"], h["cells"]) for h in spec.holds] == [(f"s{s}p", ["constv2-mix-full__lora_r4_ce", "constv2-mix-full__lora_r4_ce_minus_ctl"]) for s in (1, 2, 3)]
    # 2026-09-17: the two prompted SI baselines per seed view ride as `extensions:` rows
    PROMPTED = ["prompted__safety_refusal", "prompted__si_constitution"]
    assert [(e["key"], e["view"], e["cells"]) for e in spec.extensions] == [("qwen35_4b", f"s{s}", PROMPTED) for s in (1, 2, 3)]
    assert spec.inspect_seed_policy == "replicate" and spec.inspect_seed_base_by_view == {"s1": B1, "s2": B2, "s3": B3, "s1p": B1, "s2p": B2, "s3p": B3} == ES.SEED_BASE_BY_ANY_VIEW
    assert {v: b for v, b in spec.inspect_seed_base_by_view.items() if not v.endswith("p")} == ES.SEED_BASE_BY_VIEW   # the paired views carry their seed's base
    assert spec.exclude == []                                        # O12a: the native `none` anchor row is a row of every view
    assert spec.frozen_inputs is not None and Path(spec.frozen_inputs["roster"]).parent.name == "840aa63e"   # the paired-view freeze (2026-09-20; before it cfdcd2c0, the prompted-routes freeze of 2026-09-17; before that the staging ca6fcb92, and the provisional f7999f34)
    assert spec.standup_companions == {"cell": M.CELL_LORA_MIX, "reason": spec.standup_companions["reason"]} and "max-lora-rank" in spec.standup_companions["reason"]
    # the machinery over a SYNTHETIC six-cell roster (the declared grammar: 21 base rows + the 6 prompted extension rows = 27) — the live
    # document is five cells + two prompted routes / 24 rows and is pinned in test_gsm8k_qwen_seedrep.py;
    # this test pins the generator, not the live truth
    root = _synthetic_tree(tmp_path)
    text, doc = MK.build(configs_root=root, views_yaml=root / "configs/sdfbase/gsm8k/views_qwen_seedrep.yaml", views_spec=None)
    out = root / "configs/sdfbase/gsm8k/matrix_qwen_seedrep.json"
    out.write_text(text)
    assert M.load_matrix(out) == doc and doc["wave"] == "qwen_seedrep" and doc["inspect_seed_policy"] == "replicate"
    assert doc["inspect_seed_base_by_view"] == {"s1": B1, "s2": B2, "s3": B3} and doc["row_overlays"] is True
    rows = M.matrix_rows(out)
    assert len(rows) == 27 and [r["index"] for r in rows] == list(range(27))
    # per view: the anchor first, then roster order (the base pass, no prompted); then the six prompted extension rows APPENDED
    assert [(r["view"], r["cell"]) for r in rows] == [(f"s{s}", c) for s in (1, 2, 3) for c in [M.CELL_NONE, *CELLS6]] + [(f"s{s}", c) for s in (1, 2, 3) for c in PROMPTED]
    assert [r["index"] for r in rows if M.is_prompted_cell(r["cell"])] == list(range(21, 27))
    for r in rows:
        s = int(r["view"][1:])
        anchor = r["cell"] == M.CELL_NONE
        prompted = M.is_prompted_cell(r["cell"])
        assert r["model_key"] == "qwen35_4b" and r["tag"] == f"qwen35_4b_s{s}" and r["wave"] == "qwen_seedrep" and r["extension"] is True and r["own_overlay"] is True
        assert r["fitted"] is (not anchor and not prompted) and not r["declared"] and r["status"] == "launch" and r["held_reason"] is None
        assert r["served_positions_expected"] == (None if (anchor or prompted) else "all")
        assert r["thinking_mode"] == M.THINKING_ON == "on" and r["sampling_mode"] == M.SAMPLING_THINKING_CARD == "thinking_card"
        assert (r["sampling"]["temperature"], r["sampling"]["top_p"], r["sampling"]["top_k"]) == (0.6, 0.95, 20)   # the Qwen thinking-card row
        assert r["inspect_seed_policy"] == "replicate" and r["inspect_seed_base"] == ES.seed_base(s) == doc["inspect_seed_base_by_view"][r["view"]]
        assert list(r)[-7:] == ["inspect_seed_policy", "inspect_seed_base", *M.STANDUP_COMPANION_FIELDS, M.GEOMETRY_CLASS_KEY]   # APPENDED after every existing field (the class last)
        assert r[M.GEOMETRY_CLASS_KEY] == "common_f22" == doc[M.GEOMETRY_CLASS_KEY]
        assert r["staged_id"] == (None if (anchor or prompted) else ("derived_base64s" if r["cell"] in DERIVED else "base64s") + f"{s}__{r['cell']}")
        assert r["out_root"] == f"controls_qwen_seedrep/qwen35_4b_s{s}/{r['cell_slug']}/thinking_card" and r["rung_out_root"] == r["out_root"] + "__rung16384"
        assert r["tuple_name"] == (None if (anchor or prompted) else M.tuple_name("qwen35_4b", declared=False, rung=M.CAP)) and r["trigger_threshold"] == 0.01
        assert M.row_companions(r) == ([] if r["cell"] == M.CELL_LORA_MIX else [(M.CELL_LORA_MIX, "si_constv2_mix_full__lora_r4_ce", f"base64s{s}__{M.CELL_LORA_MIX}", f"{s:02x}" * 32)])
        assert M.row_inspect_seed(r) == ("replicate", ES.seed_base(s)) and M.inspect_seed_row_problem(r, doc) is None
        assert r["job_id"] == f"g8k-qwen35_4b-s{s}-{M.cell_short(r['cell'])}"
    c = doc["counts"]
    assert (c["rows"], c["held"], c["launched"], c["fitted_launched"], c["in_support"]) == (27, 0, 27, 18, 9) and c["waves"] == {"qwen_seedrep": 27}
    assert c["tags"] == ["qwen35_4b_s1", "qwen35_4b_s2", "qwen35_4b_s3"] and c["contrasts_designed"] == 24
    # LAUNCHABLE under its own wave; the batch boundary admits each; `published` is not a wave of this matrix
    assert [r["index"] for r in M.eligible_rows(M.WAVE_QWEN_SEEDREP, rows)] == list(range(27))
    assert M.assert_launchable(rows[7], rows)["wave"] == "qwen_seedrep"
    with pytest.raises(SystemExit, match="not a wave of this matrix"):
        M.eligible_rows(M.WAVE_PUBLISHED, rows)
    # the row printer emits the two shell names; g8k_driver_argv (RUN) appends exactly the two flags with the view's base
    for r in (rows[0], rows[1], rows[20], rows[26]):                # the s1 anchor, the first s1 cell, the last s3 cell, the last prompted row
        sh = M.row_sh(r, wave_root_rel=M.wave_root_rel_of(doc))
        facts = dict(l.split("=", 1) for l in sh.splitlines())
        assert facts["G8K_INSPECT_SEED_POLICY"] == "replicate" and facts["G8K_INSPECT_SEED_BASE"] == str(r["inspect_seed_base"])
        argv = _driver_argv(sh)
        assert argv[-2:] != ["--inspect-seed-policy", "replicate"]
        i = argv.index("--inspect-seed-policy")
        assert argv[i:i + 4] == ["--inspect-seed-policy", "replicate", "--inspect-seed-base", str(r["inspect_seed_base"])]
        assert argv[:3] == ["/venv/bin/python", "scripts/served_capability_run.py", "generate"] and "--thinking" in argv and "--sampling" in argv
        assert argv[argv.index("--sampling") + 1] == "thinking_card" and argv[argv.index("--thinking") + 1] == "on"
        assert argv[argv.index("--bank-view") + 1] == r["view"] and ("--expect-object-sha256" in argv) is bool(r["fitted"])   # unpinned: the anchor AND a prompted route
    # a policy with no base refuses by name in the shell (both or neither)
    bad_sh = M.row_sh(rows[0], wave_root_rel=M.wave_root_rel_of(doc)).replace(f"G8K_INSPECT_SEED_BASE={rows[0]['inspect_seed_base']}", "G8K_INSPECT_SEED_BASE=''")
    r = subprocess.run(["bash", "-c", f'set -u\nf22_log(){{ :; }}\nf22_fatal(){{ echo FATAL "$@" >&2; exit 1; }}\nREPO_DIR={shlex.quote(str(REPO))}\n'
                        f'source {shlex.quote(str(G8K / "g8k_common.sh"))}\n' + M.literals_sh() + bad_sh +
                        'CPY=p; PROXY_URL=u; TABLE=t; LEDGER_URL=l; CFG=c; TT_TWO_SI_DEFENSES_DIR=d; OUT_ROOT=o; TARGET_S=1; SERVE_DIR=s; ATTEMPT_ID=a1; RUNG_SUBSET=r\ng8k_driver_argv'],
                       capture_output=True, text=True, check=False)
    assert r.returncode == 1 and "but no inspect_seed_base" in r.stderr
    # EVERY OTHER matrix: empty shell names, and the driver argv byte-identical to a render with the two names absent
    for name in (M.MATRIX_DEFAULT_NAME, M.MATRIX_LLAMA_NAME, M.MATRIX_ITDOC_NAME):
        mrows = M.matrix_rows(M.MATRICES[name])
        row = next(r for r in mrows if r["status"] == "launch" and r["fitted"])
        assert M.row_inspect_seed(row) == (None, None) and "inspect_seed_policy" not in row
        sh = M.row_sh(row, wave_root_rel=M.wave_root_rel_of(M.matrix_doc(M.MATRICES[name])))
        facts = dict(l.split("=", 1) for l in sh.splitlines())
        assert facts["G8K_INSPECT_SEED_POLICY"] == "''" and facts["G8K_INSPECT_SEED_BASE"] == "''"
        without = "\n".join(l for l in sh.splitlines() if not l.startswith("G8K_INSPECT_SEED_")) + "\n"
        argv = _driver_argv(sh)
        assert argv == _driver_argv(without) and "--inspect-seed-policy" not in argv


def test_the_views_loader_and_the_matrix_loader_refuse_a_declaration_that_is_not_the_rule(tmp_path):
    text = SEEDREP_VIEWS.read_text()

    def views(replacing: str, with_: str) -> Path:
        assert text.count(replacing) == 1, replacing
        p = tmp_path / f"views_{abs(hash((replacing, with_)))}.yaml"
        p.write_text(text.replace(replacing, with_))
        return p
    with pytest.raises(SystemExit, match="must equal the rule"):
        V.load_views_yaml(views(f"s2: {B2}", f"s2: {B2 + 1}"))
    with pytest.raises(SystemExit, match="exactly one base per expressed view"):
        V.load_views_yaml(views(f"  s3: {B3}\n", ""))
    with pytest.raises(SystemExit, match="not a seed-replicate view"):
        p = views("  qwen35_4b: [s1, s2, s3, s1p, s2p, s3p]\n", "  qwen35_4b: [s1, s2, s3, s1p, s2p, s3p, main]\n")
        p.write_text(p.read_text().replace(f"  s3p: {B3}\n", f"  s3p: {B3}\n  main: 1\n"))
        V.load_views_yaml(p)
    with pytest.raises(SystemExit, match="must be 'replicate'"):
        V.load_views_yaml(views("inspect_seed_policy: replicate", "inspect_seed_policy: frozen"))
    with pytest.raises(SystemExit, match="needs `inspect_seed_base_by_view:`"):
        V.load_views_yaml(views(f"inspect_seed_base_by_view:\n  s1: {B1}\n  s2: {B2}\n  s3: {B3}\n  s1p: {B1}\n  s2p: {B2}\n  s3p: {B3}\n", ""))
    with pytest.raises(SystemExit, match="non-negative int"):
        V.load_views_yaml(views(f"s1: {B1}", "s1: true"))
    # a base map with no policy is a base with nowhere to go
    with pytest.raises(SystemExit):
        V.select_rows({"targets": {}}, {}, {"qwen35_4b": ["s1"]}, schema=M.MATRIX_SCHEMA_2, trigger_threshold=0.01,
                      inspect_seed_base_by_view={"s1": B1})
    # schema 1 never carries the policy
    with pytest.raises(SystemExit, match="is a g8k_matrix/2 fact"):
        V.select_rows({"targets": {}}, {}, {"qwen35_4b": ["main"]}, inspect_seed_policy="replicate", inspect_seed_base_by_view={"main": 1})
    # the matrix loader: a row whose base is not the document's for its view, a row without the policy on a declaring document
    root = _synthetic_tree(tmp_path)
    text_m, doc = MK.build(configs_root=root, views_yaml=root / "configs/sdfbase/gsm8k/views_qwen_seedrep.yaml", views_spec=None)
    bad = json.loads(text_m); bad["rows"][3]["inspect_seed_base"] = B2 if bad["rows"][3]["view"] == "s1" else B1   # row 3 = s1's third cell
    p = tmp_path / "bad1.json"; p.write_text(json.dumps(bad))
    with pytest.raises(SystemExit, match="!= the document's base for view"):
        M.load_matrix(p)
    bad = json.loads(text_m); del bad["rows"][0]["inspect_seed_policy"]; del bad["rows"][0]["inspect_seed_base"]
    p = tmp_path / "bad2.json"; p.write_text(json.dumps(bad))
    with pytest.raises(SystemExit, match="declares inspect_seed_policy 'replicate' but the row carries none"):
        M.load_matrix(p)
    bad = json.loads(text_m); bad["rows"][0]["inspect_seed_policy"] = "frozen"
    p = tmp_path / "bad3.json"; p.write_text(json.dumps(bad))
    with pytest.raises(SystemExit, match="is not 'replicate'"):
        M.load_matrix(p)
    # a derived object the admission refuses is simply absent from its roster view: 26 rows (20 + the 6 prompted), indices contiguous, nothing invented
    root2 = _synthetic_tree(tmp_path / "t2", drop_cells={(2, DERIVED[1])})
    _, doc2 = MK.build(configs_root=root2, views_yaml=root2 / "configs/sdfbase/gsm8k/views_qwen_seedrep.yaml", views_spec=None)
    assert doc2["counts"]["rows"] == 26 and [r["index"] for r in doc2["rows"]] == list(range(26))
    assert not any(r["view"] == "s2" and r["cell"] == DERIVED[1] for r in doc2["rows"])


def test_the_wave_and_the_matrix_are_registered_and_the_launchers_admit_the_matrix_by_name():
    """The landing step of the views file (decisions O12a): the wave AND the matrix are registered — MATRICES[name] ==
    matrix_qwen_seedrep.json, VIEWS_BY_MATRIX[name] == views_qwen_seedrep.yaml, `qwen_seedrep` in every launcher's case list (held
    equal to MATRICES). The committed matrix regenerates from the STAGING freeze (ca6fcb92, 2026-09-17; decisions O20: five admitted
    cells per view); tests/test_gsm8k_qwen_seedrep.py pins its 18 launch rows, the companion machinery and the --check byte-identity."""
    assert M.WAVE_QWEN_SEEDREP == "qwen_seedrep" and M.OUT_ROOT_PREFIX_BY_WAVE["qwen_seedrep"] == "controls_qwen_seedrep"
    assert M.WAVE_QWEN_SEEDREP not in M.COMPLETED_WAVES and M.out_root_prefix_for("qwen_seedrep") == "controls_qwen_seedrep"
    assert M.MATRIX_QWEN_SEEDREP_NAME == "qwen_seedrep"
    assert M.MATRICES[M.MATRIX_QWEN_SEEDREP_NAME] == REPO / "configs/sdfbase/gsm8k/matrix_qwen_seedrep.json"
    assert M.VIEWS_BY_MATRIX[M.MATRIX_QWEN_SEEDREP_NAME] == SEEDREP_VIEWS and M.MATRICES[M.MATRIX_QWEN_SEEDREP_NAME].is_file()
    assert M.INSPECT_SEED_POLICY_REPLICATE == C.INSPECT_SEED_POLICY_REPLICATE and M.FLAG_INSPECT_SEED_POLICY == C.FLAG_INSPECT_SEED_POLICY
    assert M.FLAG_INSPECT_SEED_BASE == C.FLAG_INSPECT_SEED_BASE
    text = SEEDREP_VIEWS.read_text()
    assert "NOT YET GENERATED / NOT YET REGISTERED" not in text and "frozen_inputs:" in text and "THE STAGING FREEZE" in text
    assert "frozen/ca6fcb92/" in text and "frozen/cfdcd2c0/" in text and "PROVISIONAL UNTIL STAGING" not in text   # the freeze, the staging one kept as history
    for name in ("gsm8k_thinking.sbatch", "dry_run_prod.sbatch", "ladder_plan.sbatch", "smoke_not_available.sbatch"):
        src = (G8K / name).read_text()
        m = re.search(r'case "\$G8K_MATRIX" in ([a-z_|]+)\) ;;', src)
        assert m and sorted(m.group(1).split("|")) == sorted(M.MATRICES) and "qwen_seedrep" in m.group(1).split("|"), name
    # the shell reads both new row names (the dead-env-contract rule) and the dry run prints the policy
    common = (G8K / "g8k_common.sh").read_text()
    assert '--inspect-seed-policy "$G8K_INSPECT_SEED_POLICY" --inspect-seed-base "$G8K_INSPECT_SEED_BASE"' in common
    assert "INSPECT_SEED: policy=%s base=%s" in common and "inspect_seed=$G8K_INSPECT_SEED_POLICY@$G8K_INSPECT_SEED_BASE" in common
