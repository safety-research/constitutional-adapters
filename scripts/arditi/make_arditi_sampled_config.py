#!/usr/bin/env python
"""make_arditi_sampled_config.py — the SAMPLED variant of one Arditi strength
view's serving config: the serve config as data, plus a ``sampled_decode``
block, plus its own store namespace. Nothing else moves.

    python scripts/arditi/make_arditi_sampled_config.py --target qwen35_4b --view ar_r0p1
    python scripts/arditi/make_arditi_sampled_config.py --target qwen35_4b --view ar_r0p1 --check

The output is ``configs/<run tag>_sampled_t<T>.yaml``, equal AS DATA to
``configs/<run tag>.yaml`` except exactly ``{out_dir, private_run,
sampled_decode}`` — verified before anything is written, and re-verified by
``--check``.

BOTH DOSE UNITS, THE RUN TAG FROM THE UNIT'S OWN TABLE (arditi-qwen-all-positions
Phase B, prerequisite 1; IA17 item 1). The run tag is
``arditi_strength_<target>_ar_r<alpha>`` for the residual-norm-fraction unit and
``arditi_allpos_<target>_ar_c<c>`` for the raw-coefficient unit, and this file
used to spell BOTH as the first — it called ``serve_config_rel`` /
``strength_of_view``, which refuse an ``ar_c`` view before any file is read, and
hard-coded the ``arditi_strength_`` store namespace. Now the unit is read off the
view's LETTER (``arditi_strength.unit_of_view``) and every name comes from that
unit's ``_in`` helper, so a raw rung's sampled config and store namespace can
never be spelled in the other unit's namespace. The residual-norm-fraction
render is byte-identical to what it always was (the two spellings coincide there
by construction, and ``tests/test_arditi_sampled_config_unit.py`` pins the exact
strings).

THE PRESET IS THE DECODE THE ALPHASTEER ARM ACTUALLY RAN. The Arditi
baseline exists for comparability with that arm, so its sampler is read
off the target's canonical AlphaSteer ladder record —
``<ladder-root>/<target>/lambda_ladder.json``, every rung's and the unsteered
block's ``resolved_sampler.request_template.preset`` + ``sampler_keys_sent``,
which must all agree — and this generator states none of the numbers. Two
OTHER statements of the row are cross-checked and RECORDED: the contracts row
(:mod:`vendor_sampling_row`, an AST read of
``SERVER_DEFAULT_SAMPLING_BY_MODEL[<hf id>]``; a disagreement REFUSES, two
vendor-row statements cannot both be true) and the presets yaml (a
disagreement is recorded in ``source``, never resolved here — nemotron's yaml
row says top_p 1.0 while the ladder sent 0.95). The provenance of every read
is rendered into the block's ``source`` string.

WHY THE PRESET RIDES THE REQUEST AND NOT THE SERVER. On `main` this same row
is composed into ``--override-generation-config`` inside ``vllm_server.py``.
That file is in the tree this lane pins byte-identical to the wave it serves,
and the identity gate (``ar_common.sh ar_assert_serving_identity production``)
refuses any drift, so the server-side route is closed here. THE PREMISE,
GENERALIZED: the decode must be FULLY DETERMINED BY WHAT WE SEND — no sampler
key may resolve from a vendor file. "The snapshot ships no
``generation_config.json``" (Qwen/Qwen3.5-4B @ 851bf6e8) is one way to satisfy
that; the general, checkable way is "we explicitly set every sampler key the
shipped file sets" (gemma-4: temperature/top_k/top_p; Nemotron-3-Nano:
temperature/top_p). A file key the preset does not set REFUSES, naming the key
and the file's value for it, so a future target that ships a key we do not send
fails legibly instead of silently resolving.
``--allow-shipped-generation-config`` bypasses that refusal DELIBERATELY and
records the fact; it is never the route to make a target run.

SEED BASE. Taken from ``configs/sdfbase_sampled_decode.yaml`` — the campaign's
own value — so this lane's per-item seeds are the SAME common random numbers
every other sampled lane draws on an item (the seed function is cell- and
lane-independent by construction). Not restated here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import yaml

_HERE = Path(__file__).resolve().parent
REPO = _HERE.parents[1]
for _p in (str(_HERE), str(REPO / "src"), str(REPO / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from arditi_strength import (run_tag_in, serve_config_rel_in,  # noqa: E402
                             strength_of_view_in, unit_of_view)
from arditi_targets import paper_lane  # noqa: E402
from sisteer.eval.attacks.bon_served import SAMPLER_KEYS  # noqa: E402
from vendor_sampling_row import VendorRowError, vendor_row  # noqa: E402

GENERATOR = "scripts/arditi/make_arditi_sampled_config.py"
#: the campaign's shared seed base + arm label live here (one value, read)
PRESETS_YAML = REPO / "configs" / "sdfbase_sampled_decode.yaml"
#: this lane's legs: the four single-shot families AND the judged controls.
#: Both validators (jailbreak_run._sampled_decode, served_capability_run
#: .sampled_controls_decode) check their own name is present.
APPLIES_TO = ["static_families", "controls"]
#: Targets whose decode is taken from the CANONICAL PRESETS YAML rather than from the comparator
#: arm's ladder record, by campaign ruling, with the reason. The default source is the ladder (the
#: arm this baseline is compared against); a ruling can override it, and then the record states
#: BOTH values, which was used, and why — never silently one.
YAML_AUTHORITATIVE = {
    "nemotron3nano4b": (
        "campaign ruling 2026-09-12 (user + coordinator, final): 1.0 is canonical for nemotron and "
        "every arm in the Figure 2/3/5 comparison decodes there, because AlphaSteer and Arditi are "
        "compared directly against the LoRA arms in the same panels and two top_p values inside one "
        "comparison is a fork across arms that no per-cell guard can see. The nemotron AlphaSteer "
        "ladder ran 0.95 and adopts 1.0 on its next pass; the shipped generation_config.json's 0.95 "
        "is neither canonical nor sent"),
}

#: the canonical AlphaSteer ladder records (alphasteer_f22_rebake, ladder_canon):
#: the decode the comparator arm actually ran, per target
#: (in the grade root's placeholder form, $GRADE, expanded through sisteer.site where it is read)
LADDER_ROOT_DEFAULT = "$GRADE/alphasteer_f22_rebake/ladder_canon"
#: `source_kind` vocabulary (sisteer.eval.sampled_decode.SOURCE_KINDS)
SOURCE_KIND_FALLBACK = "model_card"


class SampledConfigError(RuntimeError):
    pass


def dose_of_view(view: str) -> tuple:
    """``(unit, dose)`` for a view of EITHER unit, through each unit's own strict
    reader (``arditi_strength.unit_of_view``). A view of no unit — the retired
    ``ar_d`` ladder included — refuses here rather than being parsed for its
    digits under whichever unit this file happens to know."""
    unit = unit_of_view(view)
    return unit, strength_of_view_in(unit, view)


def sampled_rel(target: str, view: str, trial: int) -> str:
    unit, dose = dose_of_view(view)
    base = serve_config_rel_in(unit, target, dose)
    assert base.endswith(".yaml")
    return f"{base[:-len('.yaml')]}_sampled_t{trial}.yaml"


def private_run_of(target: str, view: str, trial: int) -> str:
    """The store namespace: THE UNIT'S run tag plus the trial. The two units have
    different namespaces on purpose (``arditi_strength_`` vs ``arditi_allpos_``),
    so a raw rung's stores never land under the prefill-only unit's roots."""
    unit, dose = dose_of_view(view)
    return f"{run_tag_in(unit, target, dose)}_sampled_t{trial}"


def _campaign_seed_base_and_arm() -> tuple:
    if not PRESETS_YAML.is_file():
        raise SampledConfigError(
            f"{PRESETS_YAML} is absent — the shared seed_base and arm label "
            "live there; refusing to invent either")
    d = yaml.safe_load(PRESETS_YAML.read_text())
    for k in ("seed_base", "arm", "n_trials"):
        if k not in d:
            raise SampledConfigError(f"{PRESETS_YAML} has no {k!r}")
    return int(d["seed_base"]), str(d["arm"]), int(d["n_trials"])


def shipped_sampler_values(path) -> dict:
    """The SAMPLER keys a shipped ``generation_config.json`` sets (the keys an
    omitted request field would resolve from), with the file's values."""
    d = json.loads(Path(path).read_text())
    return {k: d[k] for k in SAMPLER_KEYS if k in d}


def uncovered_keys(shipped: dict, preset: dict) -> dict:
    """File keys the preset does NOT set — each one a value that would resolve
    from the vendor file instead of from our request."""
    return {k: v for k, v in shipped.items() if k not in preset}


def ladder_record_preset(target: str, *, ladder_root=None) -> tuple:
    """The decode the canonical AlphaSteer ladder for ``target`` sent: the
    ``resolved_sampler.request_template.preset`` (+ ``sampler_keys_sent``) of
    EVERY rung and of the unsteered block, which must all agree. Returns
    (preset, provenance)."""
    from sisteer import site as _site
    root = _site.expand_path(ladder_root or LADDER_ROOT_DEFAULT, key="--ladder-root")
    rec_path = root / target / "lambda_ladder.json"
    if not rec_path.is_file():
        raise SampledConfigError(
            f"no canonical AlphaSteer ladder record for {target!r} at {rec_path} — the "
            "Arditi baseline takes its decode from the arm it is compared against; "
            "refusing to pick one")
    d = json.loads(rec_path.read_text())
    blocks = [("rung%d" % i, r.get("resolved_sampler")) for i, r in enumerate(d.get("rungs") or [])]
    blocks.append(("unsteered", (d.get("unsteered") or {}).get("resolved_sampler")))
    seen = {}
    for name, rs in blocks:
        if not isinstance(rs, dict) or not isinstance((rs.get("request_template") or {}).get("preset"), dict):
            raise SampledConfigError(f"{rec_path}: {name} carries no resolved_sampler.request_template.preset")
        preset = dict(rs["request_template"]["preset"])
        sent = list(rs.get("sampler_keys_sent") or [])
        if sorted(sent) != sorted(preset):
            raise SampledConfigError(f"{rec_path}: {name} sampler_keys_sent {sent} != preset keys {sorted(preset)}")
        seen.setdefault(json.dumps(preset, sort_keys=True), []).append(name)
    if len(seen) != 1:
        raise SampledConfigError(
            f"{rec_path}: the rungs do not agree on one decode — {seen}; a baseline cannot be "
            "compared against a ladder that changed sampler mid-way")
    preset = json.loads(next(iter(seen)))
    bad = [k for k in preset if k not in SAMPLER_KEYS]
    if bad:
        raise SampledConfigError(f"{rec_path}: preset carries non-sampler keys {bad}")
    prov = {"ladder_record": str(rec_path), "ladder_record_sha256": str(d.get("sha256"))[:16],
            "ladder_code_sha": str(d.get("code_sha"))[:8],
            "ladder_decode_presets_sha256": str((d.get("benign_slice") or {}).get("decode_presets_sha256"))[:16],
            "n_blocks": len(blocks)}
    return preset, prov


def _shipped_generation_config(model_id: str, revision) -> tuple:
    """TRI-STATE, because "absent" and "unknown" are different claims and
    conflating them puts a false statement in the record (found 2026-09-12:
    this grading host holds PARTIAL snapshots — 4 tokenizer files, no
    generation_config.json and no ``.no_exist`` marker — so the hub lookup
    returned None for gemma-4 and Llama-3.1, both of which DO ship one on the
    cluster, and the generator rendered "the snapshot ships NO
    generation_config.json" into their provenance).

    Returns ``("present", path)`` — the file an omitted request key would
    resolve from; ``("absent", None)`` — the hub's own ``.no_exist`` sentinel,
    a positive statement that the repo has no such file at that revision; or
    ``("unknown", None)`` — nothing cached either way on THIS host, which is
    not evidence of absence. The in-job gate re-checks against the cache the
    job actually serves from and is the authoritative one."""
    try:
        from huggingface_hub import try_to_load_from_cache
        from huggingface_hub.file_download import _CACHED_NO_EXIST
    except Exception:  # noqa: BLE001 — no hub here: say unknown, never absent
        return ("unknown", None)
    try:
        hit = try_to_load_from_cache(model_id, "generation_config.json",
                                     revision=revision)
    except Exception:  # noqa: BLE001
        return ("unknown", None)
    if isinstance(hit, str):
        return ("present", hit)
    if hit is _CACHED_NO_EXIST:
        return ("absent", None)
    return ("unknown", None)


def build_block(cfg: dict, *, target: str, contracts_ref: str, allow_shipped: bool,
                ladder_root=None) -> dict:
    """The ``sampled_decode`` block for this config's served model."""
    model_id = cfg["model"]
    revision = cfg.get("model_revision")
    ladder_preset, lprov = ladder_record_preset(target, ladder_root=ladder_root)
    yaml_row = (yaml.safe_load(PRESETS_YAML.read_text())["presets"].get(target) or {})
    if target in YAML_AUTHORITATIVE:
        preset = dict(yaml_row.get("preset") or {})
        if not preset:
            raise SampledConfigError(
                f"{target} is marked YAML_AUTHORITATIVE but {PRESETS_YAML.name} has no preset row for it")
        preset_source = (f"the CANONICAL row in {PRESETS_YAML.name} — {YAML_AUTHORITATIVE[target]}. The "
                         f"comparator arm's ladder record sent {ladder_preset} and is NOT what this lane "
                         f"sends")
    else:
        preset = ladder_preset
        preset_source = None
    # cross-check 1: the contracts row, where one exists (two statements of THE
    # vendor row: a disagreement is a refusal, not a footnote)
    contracts_note = ""
    try:
        rec = vendor_row(model_id, ref=contracts_ref)
    except VendorRowError as e:
        contracts_note = f"contracts has no row for {model_id!r} ({str(e)[:60]}...)"
    else:
        if rec["preset"] != preset:
            raise SampledConfigError(
                f"the contracts row {rec['preset']} DISAGREES with this lane's decode "
                f"{preset} for {target} — two statements of the vendor row have drifted; a "
                "campaign-level finding, not something to resolve here")
        prov = rec["provenance"]
        contracts_note = (f"equal to the contracts row read BY AST from {prov['contracts_origin']} "
                          f"(sha256 {prov['contracts_sha256'][:16]}...), {prov['table']}[{model_id!r}] -> "
                          f"{prov['row_symbol']}[{prov['row_key']!r}] at line {prov['row_lineno']}")
    # cross-check 2: the presets yaml (recorded, never resolved here)
    yrow = yaml_row
    yaml_preset = yrow.get("preset")
    if yaml_preset is None:
        yaml_note = f"{PRESETS_YAML.name} has no row for {target}"
    elif yaml_preset == preset:
        yaml_note = f"equal to {PRESETS_YAML.name}'s row"
    else:
        yaml_note = (f"DISAGREES with {PRESETS_YAML.name}'s row {yaml_preset} — the ladder record is "
                     "what the comparator arm sent and is what this lane sends; the yaml drift is a "
                     "campaign-level finding recorded here, not resolved here")
    # THE PREMISE: no sampler key may resolve from a vendor file
    state, shipped_path = _shipped_generation_config(model_id, revision)
    shipped = shipped_sampler_values(shipped_path) if state == "present" else {}
    missing = uncovered_keys(shipped, preset)
    if missing and not allow_shipped:
        raise SampledConfigError(
            f"{model_id} @ {revision} ships {shipped_path} setting "
            + ", ".join(f"{k}={v!r}" for k, v in missing.items())
            + f", and the decode this lane sends ({preset}) does not set "
            + ("that key" if len(missing) == 1 else "those keys")
            + " — an omitted key resolves from the vendor file, so the decode would NOT be fully "
              "determined by the request. Send the key (the ladder record must too), or pass "
              "--allow-shipped-generation-config knowingly")
    if state == "absent":
        premise = ("the snapshot ships NO generation_config.json (the hub's own .no_exist sentinel "
                   "at this revision), so no key can fall back to a vendor file")
        source_kind = yrow.get("source_kind") or SOURCE_KIND_FALLBACK
    elif state == "unknown":
        premise = (f"whether the snapshot ships a generation_config.json is NOT DETERMINABLE on the "
                   f"generating host (no cached copy and no .no_exist marker for {revision}) — the "
                   f"request sets {sorted(preset)} and the IN-JOB gate re-checks coverage against the "
                   f"cache the job serves from, refusing there if any shipped key is uncovered")
        source_kind = yrow.get("source_kind") or SOURCE_KIND_FALLBACK
    else:
        covered = {k: (preset[k], v) for k, v in shipped.items() if k in preset}
        premise = (f"the snapshot ships generation_config.json setting {shipped}; the request sets "
                   f"every one of those keys ({sorted(covered)}) explicitly"
                   + (f", so nothing resolves from the file"
                      if not missing else
                      f"; UNCOVERED {missing} taken DELIBERATELY (--allow-shipped-generation-config)"))
        same = all(preset[k] == v for k, v in shipped.items() if k in preset)
        source_kind = ("generation_config" if same and set(preset) == set(shipped)
                       else yrow.get("source_kind") or "vendor_recommendation")
    seed_base, arm, n_trials = _campaign_seed_base_and_arm()
    atk = cfg.get("attacks") or {}
    static_cap = int(atk.get("static_max_new_tokens",
                             atk.get("target_max_new_tokens", 512)))
    # THE CONFIG'S CONTROLS CAP IS `attacks.controls_max_new_tokens`, ELSE
    # `attacks.target_max_new_tokens` — the wave's own key since 2026-09-11, and the rule four
    # sibling readers already spell (sdfbase/full22/sampled/{make_sampled_configs,decode_effect,
    # compare_greedy_sampled,analysis_sampled}.py), the first of which is the shared validator
    # this generator calls below. It was single-key here, which AGREES with that rule only while
    # no body carries the new key: main's 09-11 body rewrite moved the controls cap out of
    # target_max_new_tokens (leaving 4,096 there) into controls_max_new_tokens (512), so for a
    # config generated from a current body the builder wrote controls_cap 4096 while the shared
    # validator computed 512 and refused the block. Every prefill-only config predates that
    # rewrite and carries no controls_max_new_tokens, so the fallback reproduces their 512 byte
    # for byte.
    controls_cap = int(atk.get("controls_max_new_tokens",
                               atk.get("target_max_new_tokens",
                                       cfg.get("completion_tokens", 512))))
    source = (
        (f"{model_id} @ {revision}: the decode is {preset_source}. " if preset_source else "")
        + f"{model_id} @ {revision}: the decode the canonical AlphaSteer ladder for {target} sent "
        f"({lprov['ladder_record']}, record sha256 {lprov['ladder_record_sha256']}..., code "
        f"{lprov['ladder_code_sha']}, its presets file {lprov['ladder_decode_presets_sha256']}...; "
        f"{lprov['n_blocks']} blocks agree) — the comparator arm's own sampler, so the two arms are "
        f"comparable by construction. Cross-checks: {contracts_note}; {yaml_note}. THIS LANE SENDS IT "
        f"PER REQUEST (its serving tree is byte-identical to W8 and vllm_server.py is frozen there): "
        f"{premise}.")
    block = {"arm": arm, "trial": None, "n_trials": n_trials,
             "seed_base": seed_base}
    block.update({k: preset[k] for k in SAMPLER_KEYS if k in preset})
    block.update({"source": source, "source_kind": source_kind,
                  "applies_to": list(APPLIES_TO),
                  "static_cap": static_cap, "controls_cap": controls_cap})
    return block


def _scalar(v) -> str:
    if isinstance(v, str):
        return json.dumps(v)
    if isinstance(v, bool):
        return "true" if v else "false"
    return repr(v)


def render_block(block: dict) -> str:
    lines = [
        "",
        "# --- the SAMPLED decode contract (generated by " + GENERATOR + ") ------------------",
        "# The four single-shot families and the judged controls decode under the served MODEL'S",
        "# OWN vendor sampling row, one seeded draw per item. The preset is the decode the canonical",
        "# AlphaSteer ladder for this target sent (see `source`; contracts + the presets yaml are",
        "# cross-checked and recorded) and is sent AS REQUEST CONTENT, because this lane's serving",
        "# tree is byte-identical to W8 and vllm_server.py is frozen here; every sampler key the",
        "# snapshot's generation_config.json sets is sent explicitly, so nothing resolves from it.",
        "# MMLU is unaffected either way: a teacher-forced letter-logprob read at temperature 0 and",
        "# max_tokens 1 has no sampler to apply.",
        "sampled_decode:",
    ]
    for k, v in block.items():
        if k == "applies_to":
            lines.append(f"  applies_to: [{', '.join(_scalar(x) for x in v)}]")
        else:
            lines.append(f"  {k}: {_scalar(v)}")
    return "\n".join(lines) + "\n"


def build_text(target: str, view: str, trial: int, *, contracts_ref: str,
               allow_shipped: bool, ladder_root=None) -> tuple:
    unit, dose = dose_of_view(view)
    src_rel = serve_config_rel_in(unit, target, dose)
    src = REPO / src_rel
    if not src.is_file():
        raise SampledConfigError(f"no serve config at {src}")
    text = src.read_text()
    cfg = yaml.safe_load(text)
    block = build_block(cfg, target=target, contracts_ref=contracts_ref,
                        allow_shipped=allow_shipped, ladder_root=ladder_root)
    block["trial"] = int(trial)
    block = {k: block[k] for k in
             ("arm", "trial", "n_trials", "seed_base", *SAMPLER_KEYS, "source",
              "source_kind", "applies_to", "static_cap", "controls_cap")
             if k in block}
    private_run = private_run_of(target, view, trial)
    out = []
    for line in text.splitlines():
        if line.startswith("out_dir:"):
            out.append(f"out_dir: out/{private_run}")
        elif line.startswith("private_run:"):
            out.append(f"private_run: {private_run}")
        else:
            out.append(line)
    body = "\n".join(out).rstrip("\n") + "\n"
    header = (
        f"# GENERATED by {GENERATOR} from {src_rel} (sha256 "
        f"{hashlib.sha256(text.encode()).hexdigest()[:16]}…) — never hand-edited.\n"
        f"# Equal to it AS DATA except out_dir / private_run / sampled_decode.\n")
    return header + body + render_block(block), cfg, block, src_rel


def verify(text: str, src_cfg: dict, block: dict, *, target: str, view: str,
           trial: int, ladder_root=None) -> None:
    got = yaml.safe_load(text)
    changed = {k for k in set(got) | set(src_cfg) if got.get(k) != src_cfg.get(k)}
    want = {"out_dir", "private_run", "sampled_decode"}
    if changed != want:
        raise SampledConfigError(
            f"the sampled config differs from the serve config in "
            f"{sorted(changed)}, expected exactly {sorted(want)}")
    if got["sampled_decode"] != block:
        raise SampledConfigError("the rendered sampled_decode block does not "
                                 "round-trip through YAML as the built one")
    pr = private_run_of(target, view, trial)
    if got["private_run"] != pr or got["out_dir"] != f"out/{pr}":
        raise SampledConfigError(f"store namespace is {got['private_run']!r} / "
                                 f"{got['out_dir']!r}, expected {pr!r}")
    sd = got["sampled_decode"]
    atk = got.get("attacks") or {}
    if sd["static_cap"] != atk.get("static_max_new_tokens"):
        raise SampledConfigError("static_cap != attacks.static_max_new_tokens")
    # the same one rule as the builder above and as the shared validator below: the new key when
    # the body carries it, the old one otherwise. Two spellings of "the controls cap" is how the
    # block and its checker came to disagree in the first place.
    _want_ccap = atk.get("controls_max_new_tokens", atk.get("target_max_new_tokens"))
    if sd["controls_cap"] != _want_ccap:
        raise SampledConfigError(
            f"controls_cap {sd['controls_cap']!r} != this config's controls cap {_want_ccap!r} "
            "(attacks.controls_max_new_tokens, else attacks.target_max_new_tokens)")
    # the two validators that will read this block, run HERE rather than at
    # drive time: a config that a driver refuses must not reach the cluster
    import jailbreak_run as jr
    from sisteer.eval import sampled_controls as scp
    from sisteer.eval.cell_controls import SampledDecodeError, controls_gen_config
    jr._sampled_decode(got, trial)
    try:
        gen = controls_gen_config(got)
    except SampledDecodeError as e:
        raise SampledConfigError(f"the controls validator refuses the block: {e}") from e
    if scp._decode_of(gen) is None:
        raise SampledConfigError("controls_gen_config read no sampled decode from the block "
                                 "(applies_to must name 'controls')")
    # the block's sampler IS the canonical ladder's decode — re-read and compared,
    # so the rendered file cannot drift from the record it claims to follow
    mine = {k: sd[k] for k in SAMPLER_KEYS if k in sd}
    if target in YAML_AUTHORITATIVE:
        want = dict((yaml.safe_load(PRESETS_YAML.read_text())["presets"][target] or {})["preset"])
        why = f"{PRESETS_YAML.name}'s canonical row (YAML_AUTHORITATIVE)"
    else:
        want, _ = ladder_record_preset(target, ladder_root=ladder_root)
        why = "the canonical ladder's decode"
    if mine != want:
        raise SampledConfigError(f"the rendered sampler {mine} is not {why} {want} for {target}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", required=True)
    ap.add_argument("--view", required=True)
    ap.add_argument("--trial", type=int, default=0)
    ap.add_argument("--contracts-ref", default="origin/main")
    ap.add_argument("--allow-shipped-generation-config", action="store_true",
                    help="DELIBERATELY accept a shipped generation_config.json key the decode does not set "
                         "(recorded in the block's source); never the route to make a target run")
    ap.add_argument("--ladder-root", default=str(LADDER_ROOT_DEFAULT),
                    help="the canonical AlphaSteer ladder records, <root>/<target>/lambda_ladder.json")
    ap.add_argument("--check", action="store_true",
                    help="verify the file on disk matches what would be generated")
    a = ap.parse_args(argv)
    try:
        paper_lane(a.target)
        text, src_cfg, block, src_rel = build_text(
            a.target, a.view, a.trial, contracts_ref=a.contracts_ref,
            allow_shipped=a.allow_shipped_generation_config, ladder_root=a.ladder_root)
        verify(text, src_cfg, block, target=a.target, view=a.view, trial=a.trial,
               ladder_root=a.ladder_root)
    except (SampledConfigError, VendorRowError) as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2
    out = REPO / sampled_rel(a.target, a.view, a.trial)
    if a.check:
        if not out.is_file():
            print(f"MISSING: {out}", file=sys.stderr)
            return 1
        if out.read_text() != text:
            print(f"DRIFT: {out} is not what {GENERATOR} would write now",
                  file=sys.stderr)
            return 1
        print(f"OK (byte-identical): {out}")
        return 0
    out.write_text(text)
    print(f"wrote {out}\n  from {src_rel}\n  preset "
          + ", ".join(f"{k}={block[k]}" for k in SAMPLER_KEYS if k in block)
          + f"\n  seed_base {block['seed_base']} arm {block['arm']} "
            f"trial {block['trial']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
