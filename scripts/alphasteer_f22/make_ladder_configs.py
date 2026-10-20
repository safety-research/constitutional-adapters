#!/usr/bin/env python
"""alphasteer-full22 Phase C — generate the SERVING configs and Petri manifests of the ladder
standups from the wave's own generated files. Never hand-edit
the outputs: `--check` requires byte-identity, like scripts/sdfbase/make_full22_configs.py.

Per target (scripts/alphasteer_f22/as22_targets.py names the view: it / it / main / main):

  configs/alphasteer_f22_serve_<target>.yaml
      DERIVED from configs/sdfbase_full22_<target>[_<view>].yaml — the wave config for the SAME
      (target, view) — byte-for-byte as data, EXCEPT exactly four keys: `explicit_cells`
      ([none, alphasteer]; the derived `none__lens0` twin is emitted by
      defense_cells.enumerate_cells, never listed), `defenses_dir` (our root's view — the ladder
      job overrides it PER RUNG through --defenses-dir / TT_TWO_SI_DEFENSES_DIR with
      <root>/<target>/ladder/lam<i>/cells_view/<view>, the resolution order), `out_dir` and
      `private_run` (the `alphasteer_f22_<target>` namespace), plus the header. NOTE the file
      name deviates from configs/alphasteer_f22_<target>.yaml: that path is Phase B's FIT config
      on branch alphasteer-multilayer, so the serving config carries `_serve_`.

  configs/capeval/manifests/alphasteer_f22_<target>.yaml
      DERIVED from configs/capeval/manifests/petri_full22_<target>[_<view>].yaml with exactly
      three keys changed: `audit` (alphasteer_f22_<target>), `cells` ([none, alphasteer] in the
      HARNESS vocabulary — `alphasteer` is a bare Class B id, registry_adapter.harness_cell_id
      and traintest_cell_id are both the identity on it) and `serving_impl_commit`. EVERY serving
      pin is byte-identical: target_model / target_revision, chat_template_hash, tool_parser,
      serving_packages, serving_launch_flags — including the `--limit-mm-per-prompt
      {"image":0,"video":0}` text-only pin for qwen35_4b and glm46vflash, without which the
      multimodal profiling OOM-kills the 200G host cgroup — the bundle, behaviours, auditor
      chain, bloom options.

`serving_impl_commit`. The ladder runs the PATCHED tree W_L (W_A + the frozen
rho_emit_probe.diff) and records W_L's rho-emit commit — NEVER the wave's production sha 94e294c4
(this generator refuses it). That commit is W_L's HEAD (its one commit touches lens_worker.py, so
the last commit touching the serving paths IS W_L), and a tree cannot record its own sha: these
manifests live in W_A and W_L = W_A + one commit, so writing "W_L's sha" here would move W_L. The
checked-in value is therefore the manifest loader's `UNFROZEN` literal (manifest.PETRI_UNFROZEN —
the only non-sha value `load_petri_spec` accepts) BY NECESSITY, and the honest value is filed PER
RUNG by lambda_ladder.serving_impl_commit_of from the `commit` lens_gate_standup verified against
--expect-sha before serving (rung_record.served.serving_impl_commit == wave_sha == W_L). The ladder
standup never reads this field — lens_gate_standup replaces it before enumeration — while the
harness hash REFUSES the literal, so no OTHER consumer (mt_serving_up, a production leg) can serve
on these manifests as-is. The line carries the grep marker `AS22_LADDER_SERVING_IMPL_COMMIT`; the
tests pin the constant to UNFROZEN (test_the_generator_constant_stays_unfrozen...) — a real sha
here needs a different mechanism, decided deliberately, never edited in situ.

usage:
  python scripts/alphasteer_f22/make_ladder_configs.py            # write all eight files
  python scripts/alphasteer_f22/make_ladder_configs.py --check    # exit 1 on any drift
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import List, Tuple

_HERE = Path(__file__).resolve().parent
REPO = _HERE.parents[1]
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from as22_targets import (CELL_ID, LADDER_CELLS, LADDER_SERVING_IMPL_COMMIT_MARKER,  # noqa: E402
                          PRODUCTION_SERVING_IMPL_COMMIT, TARGETS, manifest_rel, serve_config_rel,
                          view_of, wave_config_rel, wave_manifest_rel, wave_tag)

#: AS22_LADDER_SERVING_IMPL_COMMIT — the loader's UNFROZEN literal, by necessity (module docstring):
#: W_L's serving_impl_commit is W_L itself and cannot live inside W_L's tree. The sha that ran is
#: filed per rung from the standup's verified HEAD. Never the production sha (refused below).
LADDER_SERVING_IMPL_COMMIT = "UNFROZEN"
GENERATOR = "scripts/alphasteer_f22/make_ladder_configs.py"
_EXPLICIT_MARKER = "# -- the explicit cell enumeration"
_BODY_MARKER = "# -- everything below:"


def _check_serving_impl_commit(value: str) -> None:
    if value == PRODUCTION_SERVING_IMPL_COMMIT or value.startswith(PRODUCTION_SERVING_IMPL_COMMIT[:8]):
        raise SystemExit(f"LADDER_SERVING_IMPL_COMMIT is the wave's PRODUCTION serving sha "
                         f"{PRODUCTION_SERVING_IMPL_COMMIT[:8]} — the ladder runs the patched tree W_L and "
                         "records W_L's rho-emit commit, never 94e294c4 (D24/D25); refusing to generate")
    if value != "UNFROZEN" and not re.fullmatch(r"[0-9a-f]{40}", value):
        raise SystemExit(f"LADDER_SERVING_IMPL_COMMIT {value!r} is neither UNFROZEN nor a full 40-hex sha")
    if value.isdigit():
        # an all-digit sha is legal hex but YAML reads the unquoted scalar as an integer, and
        # load_petri_spec then refuses it as a non-string — say so here rather than at standup
        raise SystemExit(f"LADDER_SERVING_IMPL_COMMIT {value!r} is all digits: YAML would read it as an integer "
                         "and the manifest loader would refuse it; a real commit sha carries a hex letter")


def _sub_line(body: str, key_regex: str, new_line: str) -> str:
    n = len(re.findall(key_regex, body, flags=re.M))
    if n != 1:
        raise SystemExit(f"expected exactly 1 line matching {key_regex!r}, found {n}")
    return re.sub(key_regex, lambda _m: new_line, body, count=1, flags=re.M)


def _value(text: str, key: str) -> str:
    m = re.search(rf"^{re.escape(key)}: (.*)$", text, re.M)
    if not m:
        raise SystemExit(f"no top-level {key!r} line")
    return m.group(1).rstrip()


# --------------------------------------------------------------------------- configs

def render_config(target: str) -> str:
    view = view_of(target)
    tag = wave_tag(target)
    src_rel = wave_config_rel(target)
    src = (REPO / src_rel).read_text(encoding="utf-8")
    if _EXPLICIT_MARKER not in src or _BODY_MARKER not in src:
        raise SystemExit(f"{src_rel} is not a generated full22 config (markers missing)")
    body = src[src.index(_BODY_MARKER):]
    body = re.sub(r"^# -- everything below: .*$",
                  f"# -- everything below: {Path(src_rel).name}'s data, unchanged, EXCEPT the keys the header names --",
                  body, count=1, flags=re.M)
    body = _sub_line(body, r"^defenses_dir: .*$", f"defenses_dir: out/alphasteer_f22/{target}/cells_view/{view}")
    body = _sub_line(body, r"^out_dir: .*$", f"out_dir: out/alphasteer_f22_{target}")
    body = _sub_line(body, r"^private_run: .*$", f"private_run: alphasteer_f22_{target}")
    model, rev = _value(body, "model"), _value(body, "model_revision")
    header = f"""# alphasteer_f22_serve_{target}.yaml — the alphasteer-full22 campaign's SERVING config for target
# {target}: the multi-layer AlphaSteer Class B cell (fit_class_b.py, fit_mode
# content_si_agnostic_multilayer, site pre, prefill-only) served on the vLLM lens target under the
# STACKED reduction mode next to a FRESH same-standup bare anchor. Used by the Phase C lambda
# ladder (one standup per rung, scripts/alphasteer_f22/lambda_ladder.py) and by the Phase G legs.
# GENERATED by {GENERATOR} — never hand-edited; the test requires
# byte-identity. DERIVED from {src_rel} (the full22 wave's config for this
# target's `{view}` view): every key below is that file's, byte-for-byte as data, EXCEPT —
#
#   * `explicit_cells` = [none, {CELL_ID}]. `none` first is the same-standup bare anchor; `{CELL_ID}`
#     is the bare Class B id (defense_cells.CLASS_B_IDS; registry_adapter translates it to itself),
#     served from <defenses_dir>/{CELL_ID}/object.pt with fit_provenance.json as its ONE identity
#     sidecar. The path-matched no-op twin `none__lens0` is DERIVED by
#     defense_cells.enumerate_cells — never listed.
#   * `defenses_dir` names our OWN objects root (nothing under exp/objects/sdfbase/full22 or
#     configs/sdfbase_full22_* is touched): $SHARED/exp/objects/alphasteer_f22/{target}/cells_view/{view}
#     holds the CHOSEN object after the ladder (plan C.6 / Phase F). The ladder standups override it
#     PER RUNG — --defenses-dir / TT_TWO_SI_DEFENSES_DIR (the defenses-dir resolution order) =
#     $SHARED/exp/objects/alphasteer_f22/{target}/ladder/lam<i>/cells_view/{view}, laid out by
#     scripts/alphasteer_f22/bake_rungs.py (same matrices, base_scale = lambda_i).
#   * out_dir / private_run get the `alphasteer_f22_{target}` namespace.
#
# NOTE the file name deviates from the configs/alphasteer_f22_<target>.yaml form: that path
# is Phase B's FIT config (branch alphasteer-multilayer, cherry-picked by PR-B), so the SERVING
# config carries `_serve_`.
#
# Serving manifest: configs/capeval/manifests/alphasteer_f22_{target}.yaml (same generator; its
# cells are EXACTLY these explicit_cells in the harness vocabulary — scripts/sdfbase/mt_serving_up.py
# refuses a manifest/config cell mismatch in either direction). The ladder standups run the
# PATCHED tree W_L (W_A + rho_emit_probe.diff, SISTEER_LENS_RHO_EMIT=1) and their manifests record
# W_L's serving_impl_commit, never the wave's 94e294c4.
#
# serving target: {model} @ {rev}

{_EXPLICIT_MARKER} (ordered; the registry is EXACTLY this) ---
explicit_cells:
"""
    for c in LADDER_CELLS:
        header += f"- {c}\n"
    header += "\n"
    return header + body


# --------------------------------------------------------------------------- manifests

def render_manifest(target: str) -> str:
    _check_serving_impl_commit(LADDER_SERVING_IMPL_COMMIT)
    view = view_of(target)
    tag = wave_tag(target)
    src_rel = wave_manifest_rel(target)
    src = (REPO / src_rel).read_text(encoding="utf-8")
    if "\npetri:\n" not in src:
        raise SystemExit(f"{src_rel} has no top-level petri: block")
    body = src[src.index("\npetri:\n") + 1:]
    body = _sub_line(body, r"^  audit: .*$", f"  audit: alphasteer_f22_{target}")
    # the cells block: from `  cells:` to the next top-level petri key
    m = re.search(r"^  cells:\n(?:    - .*\n)+", body, flags=re.M)
    if not m:
        raise SystemExit(f"{src_rel}: cells block not found")
    cells_block = "  cells:\n" + "".join(f"    - {c}\n" for c in LADDER_CELLS)
    body = body[:m.start()] + cells_block + body[m.end():]
    body = _sub_line(
        body, r"^  serving_impl_commit: .*$",
        f"  serving_impl_commit: {LADDER_SERVING_IMPL_COMMIT}   # {LADDER_SERVING_IMPL_COMMIT_MARKER}: UNFROZEN by "
        "NECESSITY — the ladder records W_L's rho-emit commit, which is W_L itself, and a tree cannot carry its own "
        "sha; the standup replaces this field before enumeration and the driver files the sha that ran "
        "(rung_record.served.serving_impl_commit == W_L); the harness hash refuses the literal so no other consumer "
        "serves on this manifest as-is; NEVER the wave's production 94e294c4")
    target_model = re.search(r"^  target_model: (.*)$", body, re.M).group(1).strip()
    target_rev = re.search(r"^  target_revision: (.*)$", body, re.M).group(1).strip()
    header = f"""# Petri-manifest serving pins — alphasteer-full22 campaign, target {target}, view `{view}`
# (the Phase C lambda-ladder standups and the Phase G legs). GENERATED by
# {GENERATOR} — never hand-edited (byte-identity is tested). DERIVED from
# {src_rel}: EVERY serving pin below is that file's, byte-identical —
# target_model / target_revision, chat_template_hash, tool_parser, serving_packages,
# serving_launch_flags (incl. the --limit-mm-per-prompt text-only pin where the wave carries it:
# without it the multimodal profiling OOM-kills the 200G host cgroup), the frozen scenario bundle,
# behaviours, auditor chain and bloom options — EXCEPT exactly three keys:
#
#   * `audit`  = alphasteer_f22_{target}
#   * `cells`  = [none, {CELL_ID}] — EXACTLY the harness ids of configs/alphasteer_f22_serve_{target}.yaml's
#     explicit_cells (`{CELL_ID}` is a bare Class B id: registry_adapter.traintest_cell_id /
#     harness_cell_id are both the identity on it; scripts/sdfbase/mt_serving_up.py refuses a
#     manifest/config mismatch in either direction). The path-matched no-op leg `none__lens0` is
#     DERIVED by defense_cells.enumerate_cells, never listed.
#   * `serving_impl_commit` — the ladder runs the PATCHED tree W_L (W_A + rho_emit_probe.diff), so this
#     manifest records W_L's rho-emit commit and NEVER the wave's production 94e294c4. See
#     the {LADDER_SERVING_IMPL_COMMIT_MARKER} marker on that line.
#
# Objects: <defenses_dir>/{CELL_ID}/object.pt under $SHARED/exp/objects/alphasteer_f22/{target}/
# (per-rung: ladder/lam<i>/cells_view/{view}; scripts/alphasteer_f22/bake_rungs.py), identity sidecar
# fit_provenance.json. Serving target: {target_model} @ {target_rev}
"""
    return header + body


# --------------------------------------------------------------------------- driver

def outputs() -> List[Tuple[Path, str, str]]:
    out = []
    for target in TARGETS:
        out.append((REPO / serve_config_rel(target), "config", target))
        out.append((REPO / manifest_rel(target), "manifest", target))
    return out


def render(kind: str, target: str) -> str:
    return render_config(target) if kind == "config" else render_manifest(target)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--check", action="store_true", help="exit 1 if any committed file drifts")
    args = ap.parse_args(argv)
    _check_serving_impl_commit(LADDER_SERVING_IMPL_COMMIT)
    rc = 0
    for dst, kind, target in outputs():
        text = render(kind, target)
        if args.check:
            if not dst.is_file() or dst.read_text(encoding="utf-8") != text:
                print(f"DRIFT: {dst}"); rc = 1
            else:
                print(f"ok: {dst}")
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_name(f".tmp-{dst.name}")
            tmp.write_text(text, encoding="utf-8")
            tmp.replace(dst)
            print(f"wrote {dst}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
