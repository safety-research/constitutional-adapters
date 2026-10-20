#!/usr/bin/env python
"""render_arditi_seedrep_configs.py — the LANE's Arditi client configs for qwen-seedrep-native-battery.

WHY. The source `arditi_allpos_qwen35_4b_ar_c0p3_sampled_t0.yaml` carries `sampled_decode: {seed_base
20260909, n_trials 2, trial 0}` and `sampled_decode.resolve_default_decode` returns a config that already
carries a block UNTOUCHED — run as is, every Arditi static / controls request would draw under the campaign
base, not the object's seed. The battery's Arditi arm is ONE object served on ONE engine per leg, measured
three times under the three literal seed bases s ∈ {1, 2, 3}, each replicate in its OWN store namespace
(the grader twin keys stores by `private_run`). So this generator renders, from a repo-local COPY of the
source files:

  configs/seedrep/source/…                                   the three source files VERBATIM at the recorded source
                                                             sha (>= 0f3576f3: the OR-Bench-ladder
                                                             fix; `--refresh-source` refuses an older sha) +
                                                             SOURCE.json {source_sha, min_source_sha, files: sha256}
  configs/seedrep/arditi_allpos_qwen35_4b_ar_c0p3_seedrep.yaml
                                                             the STANDUP config — byte-identical to the source
                                                             (the Arditi engine is stood up from this path at launch)
  configs/seedrep/arditi_allpos_qwen35_4b_ar_c0p3_seedrep_manifest.yaml
                                                             the source Petri manifest with EXACTLY three changes:
                                                             `petri.serving_impl_commit` re-pinned to the roster's
                                                             (configs/sdfbase_full22_roster.yaml — the ONE authority
                                                             the lane's s1/s2/s3 manifests share; RE-PINNED at
                                                             landing) and the `petri.target_decode` block
                                                             `--target-decode vendor` asserts (mode vendor,
                                                             the exact registered ref); `--target-seed-base <s>` is
                                                             a run knob, NEVER in the manifest (one Arditi proxy
                                                             serves the three seed bases into three out roots)
  configs/seedrep/arditi_allpos_qwen35_4b_ar_c0p3_seedrep_s<s>_sampled_t0[_cap8192|_cap16384].yaml
                                                             the CLIENT configs, s = 1..3 × {base, 8192, 16384}: the
                                                             standup config's TEXT with exactly these edits —
                                                             `out_dir` / `private_run` -> the per-seed namespace
                                                             (distinct per seed AND per rung), `bon.max_new_tokens`
                                                             512 -> 4096 (the battery's single per-draw BoN cap),
                                                             rungs `attacks.static_max_new_tokens`,
                                                             and the appended `sampled_decode` block {arm, trial 0,
                                                             n_trials 1, seed_base s, the six sampler keys READ FROM
                                                             THE REGISTRY ROW and asserted equal to the presets yaml,
                                                             source, source_kind, applies_to, static_cap,
                                                             controls_cap, [escalated_from: the immediate predecessor
                                                             rung of the SAME seed], readouts} — rendered by
                                                             make_sampled_configs' own block renderer, so the
                                                             lineage `final_rung.py` / `ladder_decide.py` read is
                                                             spelled exactly as the s-view configs spell it.

Every output is loaded back and held AS DATA to the source except the permitted keys (`_verify`); `--check`
compares every committed output byte-for-byte and exits 1 on drift — a roster re-pin, a moved registry value or
an edited source copy all go red by design.

Usage:
  python scripts/sdfbase/seedrep/render_arditi_seedrep_configs.py --refresh-source [--source-sha <sha>]
        on the grading host: re-copy the three source files from git at <sha> (default: the pinned
        minimum 0f3576f3) into configs/seedrep/source/ + SOURCE.json, then render everything
  python scripts/sdfbase/seedrep/render_arditi_seedrep_configs.py           # render from the committed source copies
  python scripts/sdfbase/seedrep/render_arditi_seedrep_configs.py --check   # exit 1 on drift
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import yaml

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts" / "sdfbase" / "full22" / "sampled"))
sys.path.insert(0, str(REPO / "src"))
import make_sampled_configs as ms  # noqa: E402  (the block renderer + presets loader the s-view configs use)

GENERATOR = "scripts/sdfbase/seedrep/render_arditi_seedrep_configs.py"
LANE_DIR = REPO / "configs" / "seedrep"
SOURCE_DIR = LANE_DIR / "source"
SOURCE_JSON = SOURCE_DIR / "SOURCE.json"
ROSTER = REPO / "configs" / "sdfbase_full22_roster.yaml"
#: the Arditi config source must be at or after this sha (the OR-Bench cap-ladder fix,
#: branch allpos-orbench); `--refresh-source` refuses a sha that does not descend from it
MIN_SOURCE_SHA = "0f3576f3"
TARGET_KEY = "qwen35_4b"
SEEDS: Tuple[int, ...] = (1, 2, 3)
RUN_TAG = "arditi_allpos_qwen35_4b_ar_c0p3"
LANE_TAG = f"{RUN_TAG}_seedrep"
#: the three source files (repo-relative) -> the copy under configs/seedrep (None = source-only)
SOURCE_FILES: Dict[str, Optional[str]] = {
    f"configs/{RUN_TAG}.yaml": f"{LANE_TAG}.yaml",
    f"configs/{RUN_TAG}_sampled_t0.yaml": None,
    f"configs/capeval/manifests/{RUN_TAG}.yaml": f"{LANE_TAG}_manifest.yaml",
}
STANDUP_SOURCE = f"configs/{RUN_TAG}.yaml"
MANIFEST_SOURCE = f"configs/capeval/manifests/{RUN_TAG}.yaml"
SAMPLED_SOURCE = f"configs/{RUN_TAG}_sampled_t0.yaml"
#: the battery's BoN cap: a single 4096 cap per draw (`bon.max_new_tokens 4096`)
BON_CAP_SOURCE = 512
BON_CAP_LANE = 4096
#: the registry rows the static families and the judged controls decode under (read, never typed)
STATIC_POLICY_REF = "qwen35_4b_it|static|off|vendor_preset@v1"
CONTROLS_POLICY_REF = "qwen35_4b_it|over_refusal|off|vendor_preset@v1"
#: the Petri target decode declaration: the exact registered ref WITH its version; the
#: manifest block is a harness-hash input, `--target-decode vendor` asserts it, `--target-seed-base` is a run knob
PETRI_POLICY_REF = "qwen35_4b_it|petri|off|vendor_preset@v1"
PETRI_TARGET_DECODE_BLOCK = {"mode": "vendor", "policy": PETRI_POLICY_REF}
N_TRIALS = 1
TRIAL = 0


def sha256_hex(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sampled_name(s: int, esc=False) -> str:
    cap = ms.rung_cap(esc)
    return f"{LANE_TAG}_s{int(s)}_sampled_t{TRIAL}" + (f"_cap{cap}" if cap else "")


def output_path(s: int, esc=False) -> Path:
    return LANE_DIR / f"{sampled_name(s, esc)}.yaml"


def outputs() -> list:
    """[(path, s, esc)] in generation order: seed, then base / cap8192 / cap16384."""
    return [(output_path(s, esc), s, esc) for s in SEEDS for esc in (False, *ms.STATIC_RUNGS)]


def wave_name(s: int) -> str:
    """The per-seed WAVE config's stem == its `private_run` (`ARDITI_WAVE_CONFIG_REL`
    `configs/seedrep/{LANE_TAG}_s{s}.yaml`): the namespace the Arditi BoN and multi-turn stores of seed base `s` file
    under — NOT the static-base / sampled `*_sampled_t0*` namespaces (the sampled chain's) and NOT the shared source one."""
    return f"{LANE_TAG}_s{int(s)}"


def wave_output_path(s: int) -> Path:
    return LANE_DIR / f"{wave_name(s)}.yaml"


def wave_outputs() -> list:
    """[(path, s)] — one NON-sampled wave config per seed base."""
    return [(wave_output_path(s), s) for s in SEEDS]


# --------------------------------------------------------------------------- #
# the source copies                                                            #
# --------------------------------------------------------------------------- #

def _git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args], check=True, capture_output=True, text=True).stdout


def _has_commit(sha: str) -> bool:
    """Whether this clone's object store carries ``sha`` as a commit (``git cat-file -e``); False for a published repository
    without its development history, a shallow clone, or a sha that was never fetched."""
    return subprocess.run(["git", "-C", str(REPO), "cat-file", "-e", f"{sha}^{{commit}}"], capture_output=True).returncode == 0


def refresh_source(source_sha: str) -> dict:
    """Copy the three source files VERBATIM from git at ``source_sha`` into configs/seedrep/source/ and write
    SOURCE.json. Refuses a sha that does not descend from :data:`MIN_SOURCE_SHA`, and refuses BY NAME a clone that
    does not carry :data:`MIN_SOURCE_SHA` at all (the descent cannot be read there; ``--check`` reads only the committed
    copies and is untouched)."""
    if not _has_commit(MIN_SOURCE_SHA):
        raise SystemExit(f"--refresh-source needs commit {MIN_SOURCE_SHA}, which this clone does not carry (a published repository without "
                         "its development history, or a clone that never fetched it); it needs the full development history — the committed "
                         "source copies under configs/seedrep/source/ stay as they are, and --check reads them")
    full = _git("rev-parse", "--verify", f"{source_sha}^{{commit}}").strip()
    min_full = _git("rev-parse", "--verify", f"{MIN_SOURCE_SHA}^{{commit}}").strip()
    ok = subprocess.run(["git", "-C", str(REPO), "merge-base", "--is-ancestor", min_full, full]).returncode == 0
    if not ok:
        raise SystemExit(f"--source-sha {source_sha} does not descend from {MIN_SOURCE_SHA} (decisions D18: the Arditi "
                         "config source must be at or after Scale's OR-Bench-ladder fix); refusing")
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    files: Dict[str, dict] = {}
    for rel, lane in SOURCE_FILES.items():
        data = subprocess.run(["git", "-C", str(REPO), "show", f"{full}:{rel}"], check=True, capture_output=True).stdout
        dst = SOURCE_DIR / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(data)
        files[rel] = {"sha256": sha256_hex(data), "lane_copy": (f"configs/seedrep/{lane}" if lane else None)}
    doc = {"generator": GENERATOR, "source_sha": full, "min_source_sha": min_full,
           "source_branch_hint": "origin/allpos-orbench (Scale; read-only)",
           "refreshed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "files": files}
    SOURCE_JSON.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return doc


def load_source() -> Tuple[dict, Dict[str, bytes]]:
    """SOURCE.json + the verbatim source bytes, each held to its recorded sha256 (an edited copy refuses)."""
    if not SOURCE_JSON.is_file():
        raise SystemExit(f"{SOURCE_JSON} is missing — run with --refresh-source on the grading host first")
    doc = json.loads(SOURCE_JSON.read_text(encoding="utf-8"))
    if not str(doc.get("source_sha", "")).startswith(MIN_SOURCE_SHA) and doc.get("min_source_sha") is None:
        raise SystemExit(f"{SOURCE_JSON}: no min_source_sha recorded")
    out: Dict[str, bytes] = {}
    for rel in SOURCE_FILES:
        p = SOURCE_DIR / rel
        if not p.is_file():
            raise SystemExit(f"source copy {p} is missing — run --refresh-source")
        data = p.read_bytes()
        want = (doc.get("files") or {}).get(rel, {}).get("sha256")
        if sha256_hex(data) != want:
            raise SystemExit(f"source copy {p} does not match SOURCE.json's sha256 {want} — the verbatim copy was "
                             "edited; re-run --refresh-source instead")
        out[rel] = data
    return doc, out


# --------------------------------------------------------------------------- #
# the registry row (read, never typed) and the block                          #
# --------------------------------------------------------------------------- #

def registry_preset(ref: str) -> Dict[str, float]:
    """The six sampler keys of a registered decode-policy row by EXACT ref (never a default lookup), the row
    not superseded."""
    from sisteer.registry.decode_policy import PolicyFile

    pf = PolicyFile.load()
    pol = pf.by_ref(ref)
    if pol.superseded_by_policy_id is not None:
        raise SystemExit(f"registry row {ref} is superseded by {pol.superseded_by_policy_id}; refusing to render its values")
    return dict(pol.request_params())


def vendor_preset(presets: dict) -> Dict[str, float]:
    """The preset the block carries: the presets yaml's `qwen35_4b` row (the s-view configs' spelling and key
    order), ASSERTED equal to the registry's static AND over-refusal vendor rows (one instrument, three statements)."""
    p = presets["presets"][TARGET_KEY]["preset"]
    yaml_row = {k: p[k] for k in ms.ALLOWED_SAMPLER_KEYS if k in p}
    for ref in (STATIC_POLICY_REF, CONTROLS_POLICY_REF):
        reg = registry_preset(ref)
        if {k: float(v) for k, v in reg.items()} != {k: float(v) for k, v in yaml_row.items()}:
            raise SystemExit(f"the registry row {ref} ({reg}) is not the presets yaml's {TARGET_KEY} row ({yaml_row}) — two "
                             "statements of the vendor row disagree; refusing to render either")
    return yaml_row


def expected_block(presets: dict, s: int, esc=False) -> dict:
    """The `sampled_decode` block AS DATA for seed base ``s``: make_sampled_configs' shape with the lane's
    three differences — n_trials 1, seed_base s, and an `escalated_from.private_run` in THIS seed's namespace."""
    if s not in SEEDS:
        raise SystemExit(f"seed base {s!r} is not one of {SEEDS}")
    cap = ms.rung_cap(esc)
    block = {"arm": presets["arm"], "trial": TRIAL, "n_trials": N_TRIALS, "seed_base": int(s)}
    block.update(vendor_preset(presets))
    p = presets["presets"][TARGET_KEY]
    block.update({"source": p["source"], "source_kind": p["source_kind"],
                  "applies_to": list(presets["applies_to"]),
                  "static_cap": cap or ms.STATIC_CAP_BASE, "controls_cap": ms.CONTROLS_CAP})
    if cap:
        prev_cap, prev_esc = ms.previous_rung(cap)
        block["escalated_from"] = {"kind": "static", "cap": prev_cap, "private_run": sampled_name(s, prev_esc)}
    block["readouts"] = ms.expected_readouts(presets, TARGET_KEY)
    return block


# --------------------------------------------------------------------------- #
# rendering                                                                    #
# --------------------------------------------------------------------------- #

def render_client(presets: dict, source_text: str, source_sha: str, s: int, esc=False) -> str:
    cap = ms.rung_cap(esc)
    name = sampled_name(s, esc)
    text = source_text
    text = ms._sub_line(text, rf"^out_dir: out/{re.escape(RUN_TAG)}$", f"out_dir: out/{name}")
    text = ms._sub_line(text, rf"^private_run: {re.escape(RUN_TAG)}$", f"private_run: {name}")
    text = ms._sub_line(text, rf"^  max_new_tokens: {BON_CAP_SOURCE}$", f"  max_new_tokens: {BON_CAP_LANE}")
    if cap:
        text = ms._sub_line(text, rf"^  static_max_new_tokens: {ms.STATIC_CAP_BASE}$", f"  static_max_new_tokens: {cap}")
    block = expected_block(presets, s, esc)
    p = presets["presets"][TARGET_KEY]
    extra = ""
    if cap:
        prev_cap, prev_esc = ms.previous_rung(cap)
        prev_name = sampled_name(s, prev_esc)
        prev_what = "the base run" if prev_esc is False else f"the {prev_cap} rung"
        extra = (f"#   * `attacks.static_max_new_tokens` {prev_cap} -> {cap} and `sampled_decode.static_cap` {cap}: the static {cap}\n"
                 f"#     RUNG of seed base {s}, a SEPARATE store (`sampled_decode.escalated_from` names {prev_what} {prev_name};\n"
                 f"#     only its at-cap rows are re-drawn, the same seed — never a continuation).\n")
    header = (f"# {name}.yaml — qwen-seedrep-native-battery: the Arditi all-positions arm (ar_c0p3) measured under seed base {s}"
              + (f" @ the static {cap} rung" if cap else "") + "\n"
              f"# GENERATED by {GENERATOR} from the LANE COPY of Scale's\n"
              f"# {STANDUP_SOURCE} (source sha {source_sha[:12]}, >= {MIN_SOURCE_SHA}; verbatim under\n"
              f"# configs/seedrep/source/) — never hand-edited; `--check` refuses drift and the test pins byte-identity.\n"
              f"# The source config's text follows BYTE-FOR-BYTE (its own header included), EXCEPT:\n"
              f"#   * `out_dir` / `private_run` -> `{name}`: this seed replicate's OWN store namespace (distinct per seed and per\n"
              f"#     rung; the grader twin keys stores by `private_run`), disjoint from Scale's `{RUN_TAG}_sampled_t*`.\n"
              f"#   * `bon.max_new_tokens` {BON_CAP_SOURCE} -> {BON_CAP_LANE}: the battery's single per-draw BoN cap.\n"
              f"{extra}"
              f"#   * the top-level `sampled_decode` block APPENDED at the end: {p['model']} @ {p['revision'][:8]}'s vendor preset\n"
              f"#     ({', '.join(f'{k} {v}' for k, v in vendor_preset(presets).items())}; source_kind {p['source_kind']}; the registry rows\n"
              f"#     {STATIC_POLICY_REF} / {CONTROLS_POLICY_REF} read equal), seed_base {s} (the LITERAL training seed of the\n"
              f"#     replicate this run stands beside), trial {TRIAL} of {N_TRIALS} (one trial per seed).\n"
              f"#     The Petri manifest is the lane's configs/seedrep/{LANE_TAG}_manifest.yaml.\n"
              f"#\n")
    tail = ("\n# -- the Qwen seed-replicate battery: the vendor-recommended sampler under seed base "
            f"{s}, read by jailbreak_run.py generate\n"
            "# (--trial must equal `trial`; static_cap must equal attacks.static_max_new_tokens) and by\n"
            "# served_capability_run.py (controls_gen_config); every row records `decode` + `sampling_seed` --\n"
            + ms.render_block(block))
    out = header + text + tail
    _verify_client(out, source_text, block, name, cap)
    return out


def _verify_client(out: str, src_text: str, block: dict, name: str, cap) -> None:
    got, src = yaml.safe_load(out), yaml.safe_load(src_text)
    if got["sampled_decode"] != block:
        raise SystemExit(f"{name}: rendered sampled_decode != intended block\n{got['sampled_decode']}\n{block}")
    if got["out_dir"] != f"out/{name}" or got["private_run"] != name:
        raise SystemExit(f"{name}: out_dir / private_run not renamed")
    changed = {k for k in set(got) | set(src) if got.get(k) != src.get(k)}
    want = {"out_dir", "private_run", "sampled_decode", "bon"} | ({"attacks"} if cap else set())
    if changed != want:
        raise SystemExit(f"{name}: keys differing from the source config {sorted(changed)} != permitted {sorted(want)}")
    b, sb = dict(got["bon"]), dict(src["bon"])
    if b.pop("max_new_tokens") != BON_CAP_LANE or sb.pop("max_new_tokens") != BON_CAP_SOURCE or b != sb:
        raise SystemExit(f"{name}: bon differs from the source beyond max_new_tokens {BON_CAP_SOURCE} -> {BON_CAP_LANE}")
    a, sa = dict(got["attacks"]), dict(src["attacks"])
    if cap:
        if a.pop("static_max_new_tokens") != cap or sa.pop("static_max_new_tokens") != ms.STATIC_CAP_BASE or a != sa:
            raise SystemExit(f"{name}: attacks differs from the source beyond static_max_new_tokens")
    if got["sampled_decode"]["static_cap"] != got["attacks"]["static_max_new_tokens"]:
        raise SystemExit(f"{name}: static_cap != attacks.static_max_new_tokens")
    ccap = got["attacks"].get("controls_max_new_tokens", got["attacks"]["target_max_new_tokens"])
    if got["sampled_decode"]["controls_cap"] != ccap:
        raise SystemExit(f"{name}: controls_cap != the config's controls cap {ccap}")


def render_wave(source_text: str, source_sha: str, s: int) -> str:
    """One per-seed NON-sampled WAVE config: the source standup config BYTE-FOR-BYTE except
    `out_dir` / `private_run` -> :func:`wave_name` (a distinct namespace per seed, disjoint from every sampled / static
    one) and `bon.max_new_tokens` 512 -> 4096. The seed base is NOT a key here: the BoN and multi-turn drivers
    take it on the argv (`--decode-seed-policy replicate_v1 --decode-seed-base s`, `--multiturn-target-seed-base s`)
    and no config key of theirs reads it — a key nothing consumes would be a declaration with no consumer.
    `bon.trials` is the source's 1 (asserted). The Petri `target_decode` block lives in the
    lane MANIFEST (the standup config has no `petri:` block; the drive reads the manifest)."""
    name = wave_name(s)
    text = source_text
    text = ms._sub_line(text, rf"^out_dir: out/{re.escape(RUN_TAG)}$", f"out_dir: out/{name}")
    text = ms._sub_line(text, rf"^private_run: {re.escape(RUN_TAG)}$", f"private_run: {name}")
    text = ms._sub_line(text, rf"^  max_new_tokens: {BON_CAP_SOURCE}$", f"  max_new_tokens: {BON_CAP_LANE}")
    header = (f"# {name}.yaml — qwen-seedrep-native-battery: the Arditi all-positions arm (ar_c0p3), the WAVE legs (BoN,\n"
              f"# the three multi-turn attacks) under seed base {s}. GENERATED by {GENERATOR}\n"
              f"# from the LANE COPY of Scale's {STANDUP_SOURCE} (source sha {source_sha[:12]}, >= {MIN_SOURCE_SHA};\n"
              f"# verbatim under configs/seedrep/source/) — never hand-edited; `--check` refuses drift and the test pins byte-identity.\n"
              f"# The source config's text follows BYTE-FOR-BYTE (its own header included), EXCEPT:\n"
              f"#   * `out_dir` / `private_run` -> `{name}` (== this file's stem): the WAVE stores' OWN namespace for seed base {s},\n"
              f"#     distinct per seed and disjoint from the static-base / sampled `{LANE_TAG}_s{s}_sampled_t0*` namespaces and\n"
              f"#     from Scale's shared `{RUN_TAG}` — the wave-run discovery keys the wave run by it (a shared or sampled\n"
              f"#     namespace would misclassify the stores).\n"
              f"#   * `bon.max_new_tokens` {BON_CAP_SOURCE} -> {BON_CAP_LANE}: the battery's single per-draw BoN cap.\n"
              f"# The seed base {s} itself is NOT a key of this file: the drivers take it on the argv (`--decode-seed-policy\n"
              f"# replicate_v1 --decode-seed-base {s}`; `--multiturn-target-seed-base {s}`), one trial (`bon.trials` {N_TRIALS}, Scale's).\n"
              f"# The Petri manifest (with its `target_decode` block) is the lane's configs/seedrep/{LANE_TAG}_manifest.yaml.\n"
              f"#\n")
    out = header + text
    _verify_wave(out, source_text, name)
    return out


def _verify_wave(out: str, src_text: str, name: str) -> None:
    got, src = yaml.safe_load(out), yaml.safe_load(src_text)
    if got["out_dir"] != f"out/{name}" or got["private_run"] != name:
        raise SystemExit(f"{name}: out_dir / private_run not renamed")
    if "sampled_decode" in got:
        raise SystemExit(f"{name}: a WAVE config carries no sampled_decode block")
    changed = {k for k in set(got) | set(src) if got.get(k) != src.get(k)}
    if changed != {"out_dir", "private_run", "bon"}:
        raise SystemExit(f"{name}: keys differing from the source config {sorted(changed)} != permitted ['bon', 'out_dir', 'private_run']")
    b, sb = dict(got["bon"]), dict(src["bon"])
    if b.pop("max_new_tokens") != BON_CAP_LANE or sb.pop("max_new_tokens") != BON_CAP_SOURCE or b != sb:
        raise SystemExit(f"{name}: bon differs from the source beyond max_new_tokens {BON_CAP_SOURCE} -> {BON_CAP_LANE}")
    if got["bon"]["trials"] != N_TRIALS:
        raise SystemExit(f"{name}: bon.trials {got['bon']['trials']} != {N_TRIALS} (one trial per seed, O25)")


#: the lane's Petri FALLBACK CHAINS, an EXPLICIT transformation of the archived source.
#: The source manifest (the immutable copy under configs/seedrep/source/) still spells the abliterated-Qwen third
#: auditor leg; the battery's chain is API-only (sonnet -> grok-4.6, 3931984bf -- the drive passes no
#: Qwen endpoint, so the archived chain would REFUSE at leg start), and grok-4.6 backs the JUDGE too
#: (`judge_fallback_models`, landed 181ac97f4). Rendered by name here so a regeneration can never re-instate the
#: Qwen leg or drop the judge chain; `_verify_manifest` holds the rendered chains to exactly these values.
#: claude-opus-4-8 BEFORE grok-4.6 in BOTH chains, ORDER PINNED (grok refused some of the same
#: content opus-5 declined, so a first-party leg goes first); the reversed chain is another instrument (another harness hash).
PETRI_FALLBACK_CHAINS: Dict[str, list] = {
    "auditor_fallback_models": ["anthropic/claude-opus-4-8", "openrouter/x-ai/grok-4.6"],
    "judge_fallback_models": ["anthropic/claude-opus-4-8", "openrouter/x-ai/grok-4.6"],
}


def render_fallback_chains(text: str) -> str:
    """Replace the source's ``auditor_fallback_models`` block (whatever legs it spells) with the lane's chain and
    insert the ``judge_fallback_models`` block right after it (before ``judge_model:``), per PETRI_FALLBACK_CHAINS.
    Refuses a source without exactly one auditor chain block or with a judge chain already present."""
    m = re.search(r"^  auditor_fallback_models:[^\n]*\n(?:    - [^\n]*\n)+", text, flags=re.M)
    if not m or len(re.findall(r"^  auditor_fallback_models:", text, flags=re.M)) != 1:
        raise SystemExit("manifest: expected exactly one `  auditor_fallback_models:` block in the source")
    if re.search(r"^  judge_fallback_models:", text, flags=re.M):
        raise SystemExit("manifest: the source already spells judge_fallback_models -- the rule assumes it does not")
    a = "".join(f"    - {x}\n" for x in PETRI_FALLBACK_CHAINS["auditor_fallback_models"])
    j = "".join(f"    - {x}\n" for x in PETRI_FALLBACK_CHAINS["judge_fallback_models"])
    block = ("  auditor_fallback_models:             # API-only chain, sonnet -> opus-4-8 -> grok-4.6 (order pinned); the abliterated-Qwen leg is dropped\n"
             + a
             + "  judge_fallback_models:               # opus-4-8 then grok-4.6 back the JUDGE too (a returned content_filter refusal switches legs)\n"
             + j)
    return text[:m.start()] + block + text[m.end():]


def roster_serving_impl_commit() -> str:
    roster = yaml.safe_load(ROSTER.read_text(encoding="utf-8"))
    sha = roster.get("serving_impl_commit")
    if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise SystemExit(f"{ROSTER.name}: serving_impl_commit must be a full 40-hex sha, got {sha!r}")
    return sha


def render_manifest(source_text: str, source_sha: str, impl_commit: str) -> str:
    """The source manifest with EXACTLY three changes (see the module docstring): the re-pinned
    `petri.serving_impl_commit`, the appended `petri.target_decode` block, inserted right before
    `bloom_options:` (after `prompted_prefix_hashes`, where it is rendered on the s-view manifests), and the
    Petri fallback chains (:data:`PETRI_FALLBACK_CHAINS`: the auditor chain re-spelled
    without the abliterated-Qwen leg, the judge chain added)."""
    source_text = render_fallback_chains(source_text)
    text = ms._sub_line(source_text, r"^  serving_impl_commit: [0-9a-f]{40}.*$",
                        f"  serving_impl_commit: {impl_commit}   # RE-PINNED from configs/sdfbase_full22_roster.yaml "
                        "(the lane's one authority, shared with the s1/s2/s3 manifests; the roster is re-pinned at landing and this file regenerated)")
    n = len(re.findall(r"^  bloom_options:", text, flags=re.M))
    if n != 1:
        raise SystemExit(f"manifest: expected exactly one `  bloom_options:` line, found {n}")
    block = ("  target_decode:                  # the vendor row on every TARGET request; a\n"
             "                                  # harness-hash input (`--target-decode vendor` asserts it; `--target-seed-base <s>`\n"
             "                                  # is a RUN knob, never here — one Arditi proxy serves the three seed bases, three out roots)\n"
             f"    mode: {PETRI_TARGET_DECODE_BLOCK['mode']}\n"
             f"    policy: '{PETRI_TARGET_DECODE_BLOCK['policy']}'\n")
    text = re.sub(r"^  bloom_options:", block + "  bloom_options:", text, count=1, flags=re.M)
    header = (f"# {LANE_TAG}_manifest.yaml — qwen-seedrep-native-battery: the LANE copy of Scale's Petri manifest for the Arditi\n"
              f"# all-positions arm (ar_c0p3). GENERATED by {GENERATOR} from the verbatim source copy of\n"
              f"# {MANIFEST_SOURCE} (source sha {source_sha[:12]}; configs/seedrep/source/) — never hand-edited.\n"
              f"# The source's text follows BYTE-FOR-BYTE EXCEPT: (1) `petri.serving_impl_commit` re-pinned to the roster's; (2) the\n"
              f"# `petri.target_decode` block; (3) the Petri fallback chains (PETRI_FALLBACK_CHAINS): the\n"
              f"# auditor chain re-spelled as sonnet -> grok-4.6 only and `judge_fallback_models` [grok-4.6] added. Everything else —\n"
              f"# audit, behaviors, cells, scenario bundle, auditor / judge models, prompt provenance, target, serving flags, prefix\n"
              f"# hashes, bloom_options — is Scale's, unchanged.\n"
              f"#\n")
    out = header + text
    _verify_manifest(out, source_text, impl_commit)
    return out


def _verify_manifest(out: str, src_text: str, impl_commit: str) -> None:
    got, src = yaml.safe_load(out), yaml.safe_load(src_text)
    if set(got) != set(src) or set(got) != {"petri"}:
        raise SystemExit(f"manifest: top-level keys {sorted(got)} != source {sorted(src)} (expected exactly ['petri'])")
    g, s = dict(got["petri"]), dict(src["petri"])
    if g.pop("target_decode") != PETRI_TARGET_DECODE_BLOCK:
        raise SystemExit("manifest: target_decode block is not the WS1-B declaration")
    if g.pop("serving_impl_commit") != impl_commit or "serving_impl_commit" not in s:
        raise SystemExit("manifest: serving_impl_commit was not re-pinned")
    s.pop("serving_impl_commit")
    for key, want in PETRI_FALLBACK_CHAINS.items():
        if g.pop(key, None) != want:
            raise SystemExit(f"manifest: {key} is not the lane's chain {want} (D111 / D113, PETRI_FALLBACK_CHAINS)")
        s.pop(key, None)
    if g != s:
        diff = {k for k in set(g) | set(s) if g.get(k) != s.get(k)}
        raise SystemExit(f"manifest: keys differing from the source beyond the three permitted: {sorted(diff)}")


def render_all(presets: dict, doc: dict, source: Dict[str, bytes]) -> Dict[Path, bytes]:
    source_sha = str(doc["source_sha"])
    standup = source[STANDUP_SOURCE]
    out: Dict[Path, bytes] = {LANE_DIR / SOURCE_FILES[STANDUP_SOURCE]: standup}   # the verbatim standup copy
    out[LANE_DIR / SOURCE_FILES[MANIFEST_SOURCE]] = render_manifest(
        source[MANIFEST_SOURCE].decode("utf-8"), source_sha, roster_serving_impl_commit()).encode("utf-8")
    text = standup.decode("utf-8")
    for dst, s, esc in outputs():
        out[dst] = render_client(presets, text, source_sha, s, esc).encode("utf-8")
    for dst, s in wave_outputs():
        out[dst] = render_wave(text, source_sha, s).encode("utf-8")
    names = [yaml.safe_load(out[dst])["private_run"] for dst, _s, _e in outputs()]
    waves = [yaml.safe_load(out[dst])["private_run"] for dst, _s in wave_outputs()]
    if len(set(names + waves)) != len(names) + len(waves):
        raise SystemExit(f"the rendered private_run namespaces are not distinct: {names + waves}")
    for dst, _s in wave_outputs():
        if yaml.safe_load(out[dst])["private_run"] != dst.stem:
            raise SystemExit(f"{dst}: a wave config's private_run must equal its file stem (WS5-I's discovery rule)")
    scale_ns = {yaml.safe_load(source[SAMPLED_SOURCE])["private_run"], yaml.safe_load(standup)["private_run"]}
    if set(names + waves) & scale_ns:
        raise SystemExit(f"a rendered private_run collides with Scale's namespace {scale_ns}")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--check", action="store_true", help="exit 1 if any committed output drifts")
    ap.add_argument("--refresh-source", action="store_true",
                    help="the grading host: re-copy Scale's three files from git at --source-sha into configs/seedrep/source/")
    ap.add_argument("--source-sha", default=MIN_SOURCE_SHA,
                    help=f"the git sha the source copies are taken from (must descend from {MIN_SOURCE_SHA}; D18)")
    args = ap.parse_args(argv)
    if args.refresh_source:
        if args.check:
            raise SystemExit("--refresh-source and --check are exclusive")
        doc = refresh_source(args.source_sha)
        print(f"refreshed {SOURCE_DIR} from {doc['source_sha']}")
    doc, source = load_source()
    presets = ms.load_presets()
    rendered = render_all(presets, doc, source)
    try:
        from sisteer.registry.decode_policy import PolicyFile
        PolicyFile.load().by_ref(PETRI_POLICY_REF)
    except Exception as e:  # noqa: BLE001 — the W gate refuses the wave; the renderer only reports
        print(f"NOTE: {PETRI_POLICY_REF} does not resolve in this tree's registry ({type(e).__name__}) — the W gate "
              "(plan WS0-6 / decisions D20) requires it before launch; the manifest declares the exact ref regardless")
    rc = 0
    for dst in sorted(rendered):
        data = rendered[dst]
        if args.check:
            if not dst.is_file() or dst.read_bytes() != data:
                print(f"DRIFT: {dst}"); rc = 1
            else:
                print(f"ok: {dst}")
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(data); print(f"wrote {dst}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
