#!/usr/bin/env python
"""The full22 goalset driver's ARM cell: resolve where its bank view lives, and refuse to serve bytes
that are not the pinned object (configs/sdfbase_full22_arms.yaml).

Three modes; the first two are consumed by common.sh ``f22_arm_bank_view``, the third by ``f22_resolve_config``:

  --resolve      print ONE line `<cell> <kind> <view dir relative to $SHARED> <serving policy | ->`
                 (import-light: yaml only) after checking the table admits (target, arm) on this cluster and
                 that the job's VIEW is the target's PAPER view — the view the arm's config/manifest pair
                 derives from and the `none` anchor it pairs with. A wrong view is a refusal, never a default.
                 With --requested-serving-path, the arm's SERVING POLICY is enforced first (below).
  --policy       print the arm's declared serving policy (`arms.<arm>.serving_path`, or `-` when the arm
                 declares none) and, with --requested-serving-path, refuse a request that contradicts it.
                 Needs no target: the policy is the ARM's, and this check must fire BEFORE any per-target
                 refusal (a missing operating point, a missing config pair) so a lens request can never be
                 masked by "no operating point yet".
                 THE POLICY: `arditi_allpos` declares `serving_path: lora_native` because the spec
                 requires that object on the graph-captured native residual slot in EVERY leg; the lens
                 WOULD serve a vector_all blob, so a leg on it completes and files lens numbers under the
                 arm's name -- a different measurement, not a failure. An arm with no declared policy keeps
                 the caller's choice (the existing arm launches export F22_SERVING_PATH=lens).
  --defenses-dir DIR [--config CFG]
                 the SERVE-TIME BYTE INVARIANT on DIR/<cell> (needs torch): object.pt + staged_from.json
                 present; exactly ONE identity sidecar (train_budget_meta.json / fit_provenance.json /
                 derived_meta.json); sha256(object.pt) == the table's pin — THE identity shared with the
                 other lanes' rows; then per arm, ported from the lanes' own serve gates:
                    arditi     (ar_common.sh ar_bank_view, the dose-in-norm invariant): blob kind == static_r;
                               base_scale == 1.0 exactly (the dose lives in the vector's norm; a moved
                               base_scale is a FOLDED artifact serving another dose than its row); `rs` keyed
                               by exactly the ONE stored (fit) layer named by object.pt.provenance.json's
                               `stored_layer`, which is served_layer + 1 (site="pre" shifts it; confusing the
                               two steers a block early with every other signal healthy); no `positions` /
                               `fit_positions` key (this kind's adapter has no declared position schema);
                               provenance kind == static_r and phase == prefill.
                    vector_all (the Arditi arm on the PAPER'S OWN configuration):
                               its own equally total contract, several assertions the INVERSE of static_r's,
                               so a blob that satisfies one kind's rule fails the other's and the two arms
                               cannot be crossed by a rename. `vectors` keyed by exactly the ONE SERVED
                               block-output layer (this kind stores the SERVED layer where static_r stores the
                               FIT layer: stored == served_layer, not served_layer + 1); `positions` and
                               `fit_positions` PRESENT and both "all" -- the two keys static_r FORBIDS, which
                               this kind's adapter READS to resolve its deployment; skip_bos False; NO
                               base_scale and no `rs` at all (the op's coeff is 1.0 and the dose is baked into
                               the vector); provenance kind == vector_all, phase == both, dose_unit ==
                               raw_coefficient. AND THE PINS' ABSOLUTE LAYERS: the arm's pins file
                               (`arms.<arm>.pins`, entry `<target>.<staged_id>`) must carry fit_layer /
                               served_layer / stored_layer / direction_sha256 from the FIT RECORD
                               (arditi_strength.RAW_PIN_LAYER_KEYS); the provenance's fit.selected.layer,
                               fit.selected.served_layer, stored_layer, layers and fit.selected.direction_sha256
                               and the blob's `vectors` key are each held to them, and the pins' object_sha256
                               to the table's. The relational checks alone pass a COHERENT object baked a block
                               off (fit 12 / served 11 / stored 11); only the fit record's own numbers refuse
                               it. A pins entry lacking the keys refuses BY NAME. AND THE PINS ARE THEMSELVES
                               HELD TO THE TABLE'S APPROVED FIT FIRST: `arms.<arm>.fits.<target>` names the ONE
                               approved fit record (its sha, direction.pt's sha, the direction's values hash,
                               the selected layers); a pins entry that differs from it in any of fit_layer /
                               served_layer / stored_layer / direction_sha256 refuses by name, attributing the
                               authority to the arms table, BEFORE it certifies any object -- coherent 12/11/11
                               pins over a coherent 12/11/11 object passed every gate until then. A target the
                               table approves no fit for refuses by name.
                 THE BRANCH IS ON THE SERVED KIND, never the arm's name: the name selected the wrong
                 CONTRACT as well as the wrong branch, and a vector_all blob has neither of the two keys
                 (base_scale, rs) the arditi branch asserts on. An arm whose kind has no contract here
                 REFUSES rather than falling through to whichever branch is last.
                    alphasteer (as_defense_cell.sbatch, THE OBJECT IS THE DOSE): base_scale == the table's
                               lambda exactly (the served op's coeff IS base_scale — a wrong-rung directory
                               passes every other check and yields plausible numbers at the wrong dose);
                               the blob carries `Ms` (an AlphaSteer projection) unless its kind says so.
                 With --config, the config's explicit_cells must be exactly [none, <cell>] and its
                 defenses_dir must spell this very dir as out/<...> (the pair was generated for it).
                 Prints one JSON record (the standup log keeps it); PROBLEM lines + exit 1 on any failure.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[3]
ARMS = REPO / "configs" / "sdfbase_full22_arms.yaml"
IDENTITY_SIDECARS = ("train_budget_meta.json", "fit_provenance.json", "derived_meta.json")
#: roots this driver must NEVER resolve an arm view under: the wave bank (shared with every peer leg of
#: this lineage), the wave root, and the OLD alphasteer_f22 root (the pre-rebake parents)
PEER_ROOT_FRAGMENTS = ("/exp/objects/sdfbase/", "/exp/sdfbase_full22/", "/exp/objects/alphasteer_f22/")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_arms(path: Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def view_rel(arms: dict, target: str, arm: str) -> str:
    """The arm's bank VIEW dir relative to ``$SHARED`` (e.g. exp/objects/arditi_strength/<t>/strength/
    cells_view/ar_r0p05). Mirrors make_arm_configs.bank_view_rel (which drops the exp/objects/ prefix
    for the config's `out/` spelling); the test pins the two equal.

    KEYED OFF THE TABLE, NOT OFF THE ARM'S NAME (arditi-qwen-all-positions A2'). There are exactly
    two bank SHAPES and an arm DECLARES which it has: a CELLS bank (`bank_subdir` on the arm +
    `bank_view` on the target -- one view per dose, the dose baked into the bytes) or a LADDER bank
    (`ladder_subdir` + `rung` -- one directory per rung, the dose a scalar in the blob). Dispatching
    on the NAME made every new arm a two-place edit and refused `arditi_allpos`, whose shape is
    identical in form to arditi's; the name now appears only in the refusal."""
    a = arms["arms"][arm]
    t = arms["targets"][target]
    op = t[arm]
    if "bank_subdir" in a and "bank_view" in op:
        return f"{a['bank_root']}/{target}/{a['bank_subdir']}/cells_view/{op['bank_view']}"
    if "ladder_subdir" in a and "rung" in op:
        return f"{a['bank_root']}/{target}/{a['ladder_subdir']}/{op['rung']}/cells_view/{t['view']}"
    raise SystemExit(
        f"PROBLEM arm {arm!r} declares neither bank shape for {target}: a CELLS bank needs "
        f"arms.{arm}.bank_subdir + targets.{target}.{arm}.bank_view, a LADDER bank needs "
        f"arms.{arm}.ladder_subdir + targets.{target}.{arm}.rung "
        f"(arm keys {sorted(a)}; operating-point keys {sorted(op)})")


SERVING_PATHS = ("lora_native", "lens")

#: THE SELECTION BUDGET the raw unit's `selection_outcome` word is held against: 25.0 score units = 10 pp of
#: the 250 XSTest-safe prompts, INCLUSIVE, in the GATE statistic (superseding the earlier 4 pp / 10.0-unit
#: draft). It decides only whether a declared point is `chosen` or an OVERRIDE — the point itself is selected
#: from the measured record, never here.
RAW_SELECTION_BUDGET_ITEMS = 25.0


def serving_policy(arms: dict, arm: str) -> str | None:
    """The arm's declared serving policy (``arms.<arm>.serving_path``): one of :data:`SERVING_PATHS`, or
    ``None`` when the arm declares none (the caller's choice stands). A spelling outside the two paths
    REFUSES rather than reading as "no policy"."""
    if arm not in arms["arms"]:
        raise SystemExit(f"PROBLEM arm {arm!r} is not in the arms table ({sorted(arms['arms'])})")
    policy = arms["arms"][arm].get("serving_path")
    if policy is not None and policy not in SERVING_PATHS:
        raise SystemExit(f"PROBLEM arm {arm!r} declares serving_path {policy!r}, which is not one of {list(SERVING_PATHS)}")
    return policy


def serving_policy_problems(arms: dict, arm: str, requested: str | None, *, mode: str | None = None) -> list[str]:
    """The ONE policy assertion (Finding 2): a resolved serving path that contradicts the arm's declared
    policy is a problem naming what was requested, what the arm requires, and why. An empty/None request
    (nothing resolved yet — a deferred selection) asserts nothing; an arm with no policy asserts nothing."""
    policy = serving_policy(arms, arm)
    if not policy or not requested or requested == policy:
        return []
    how = f" ({mode} selection)" if mode else ""
    return [f"arm {arm!r} declares serving_path: {policy} but this launch resolved F22_SERVING_PATH={requested}{how} — the spec "
            f"requires this object served natively, on the graph-captured residual slot, in EVERY leg. The lens WOULD serve a "
            f"vector_all blob, so a lens leg is a different measurement rather than a failure, and it is refused here rather "
            f"than filed under the arm's name. Unset F22_SERVING_PATH (the kind-aware default selects {policy} for this arm) "
            f"or set it to {policy}"]


def resolve(arms: dict, target: str, arm: str, view: str | None, cluster: str | None,
            requested_serving_path: str | None = None) -> tuple[str, str, str]:
    def refuse(problems):
        for p in problems:
            print("PROBLEM", p, file=sys.stderr)
        raise SystemExit(1)
    if target not in arms["targets"]:
        refuse([f"target {target!r} is not in the arms table ({sorted(arms['targets'])})"])
    if arm not in arms["arms"]:
        refuse([f"arm {arm!r} is not in the arms table ({sorted(arms['arms'])})"])
    # THE SERVING POLICY FIRST (Finding 2): it is the ARM's fact and it must fire before any per-target refusal, so a
    # lens request is never masked by "no operating point yet"
    pol = serving_policy_problems(arms, arm, requested_serving_path)
    if pol:
        refuse(pol)
    if arm not in arms["targets"][target]:
        refuse([f"the arms table has no {arm} operating point for {target}"])
    problems = []
    a = arms["arms"][arm]
    t = arms["targets"][target]
    if view is not None and view != t["view"]:
        problems.append(f"the job's view {view!r} is not {target}'s PAPER view {t['view']!r} — the {arm} pair derives "
                        f"from the paper view's config/manifest and pairs with its `none` anchor; refusing to default")
    if cluster is not None and cluster not in a["clusters"]:
        problems.append(f"the {arm} objects are staged on {a['clusters']} only, not on {cluster!r} — this leg cannot "
                        "be served from this cluster (ruling 2026-09-13: no path-safe copy of the AlphaSteer rungs)")
    rel = view_rel(arms, target, arm)
    if any(frag in f"/{rel}/" for frag in PEER_ROOT_FRAGMENTS):
        problems.append(f"the arm view {rel} resolves under a PEER's root")
    if problems:
        for p in problems:
            print("PROBLEM", p, file=sys.stderr)
        raise SystemExit(1)
    return a["cell"], a["kind"], rel


def _st():
    """The Arditi lane's ONE dose table (scripts/arditi/arditi_strength.py), imported lazily: only the vector_all
    contract reads it (the pins' layer binding and its refusals live there, beside the unit that owns them)."""
    ad = str(REPO / "scripts" / "arditi")
    if ad not in sys.path:
        sys.path.insert(0, ad)
    import arditi_strength
    return arditi_strength


def _raw_pin_binding(arms: dict, a: dict, op: dict, target: str, pins: Path | None, rec: dict) -> tuple[dict | None, list[str]]:
    """The pins' layer binding for this operating point's staged rung: `<pins>[<target>][<op.staged_id>]` through
    arditi_strength.raw_pin_binding -- the entry, its four keys and own relations, AND (round 3, item 1) its equality
    with the arms table's APPROVED FIT for this target (`arms.<arm>.fits.<target>`; a target with none refuses by
    name) -- the record gaining `pins`, `pins_fit_layer`, `pins_served_layer`, `pins_stored_layer`,
    `pins_direction_sha256` and `approved_fit`; the pins' own object_sha256 held to the table's (two authorities for
    ONE identity must agree). The pins validate NOTHING until they have themselves been held to the table.
    Returns (binding or None, problems) -- every refusal by name, never a default."""
    st = _st()
    pins_path = pins if pins is not None else Path(str(a.get("pins") or ""))
    if not str(pins_path):
        return None, [f"arm {rec['arm']!r} declares no `pins` file in the arms table — a vector_all arm's layers are certified "
                      "from its pins, so an arm without them cannot be served"]
    if not pins_path.is_absolute():
        pins_path = REPO / pins_path
    rec["pins"] = str(pins_path)
    if not pins_path.is_file():
        return None, [f"no pins file at {pins_path} — the raw unit's pins (Phase B's output) bind each rung's absolute layers"]
    staged_id = op.get("staged_id")
    if not staged_id:
        return None, [f"the arms table's {rec['arm']} operating point for {target} names no staged_id — the pins are keyed by it"]
    try:
        pin, binding = st.raw_pin_binding(json.loads(pins_path.read_text(encoding="utf-8")), arms, target, str(staged_id), str(pins_path))
    except SystemExit as e:
        return None, [str(e)]
    rec.update(pins_fit_layer=binding["fit_layer"], pins_served_layer=binding["served_layer"],
               pins_stored_layer=binding["stored_layer"], pins_direction_sha256=binding["direction_sha256"],
               approved_fit=binding["approved_fit"])
    problems = []
    if pin.get("object_sha256") != op["object_sha256"]:
        problems.append(f"{pins_path} [{target}.{staged_id}] object_sha256 {str(pin.get('object_sha256'))[:12]} != the arms table's "
                        f"pin {op['object_sha256'][:12]} — two authorities name different bytes for one operating point")
    return binding, problems


def verify(arms: dict, target: str, arm: str, ddir: Path, config: Path | None, pins: Path | None = None) -> dict:
    a = arms["arms"][arm]
    op = arms["targets"][target][arm]
    cell = a["cell"]
    d = ddir / cell
    problems: list[str] = []
    rec: dict = {"target": target, "arm": arm, "cell": cell, "kind": a["kind"], "defenses_dir": str(ddir),
                 "pinned_object_sha256": op["object_sha256"], "dose_unit": a["dose_unit"]}
    # the operating point rides on the record whether or not the bytes pass — a refused row still says what it was for.
    # KEYED ON THE DOSE UNIT, not the arm's name: the unit IS what the number means, so it is the honest selector for
    # which number to record (arditi-qwen-all-positions A2').
    if a["dose_unit"] == "residual_norm_fraction":
        rec.update(alpha=op["alpha"], phase=a["phase"])
    elif a["dose_unit"] == "raw_coefficient":
        # `c` multiplies the RAW difference of means; `alpha_equiv` is the arithmetic conversion to the other
        # unit, recorded (when the table carries it) and LABELLED as arithmetic — never a realised-dose claim
        rec.update(c=op["c"], phase=a["phase"], positions=a["positions"], skip_bos=a["skip_bos"])
        if "alpha_equiv" in op:
            rec.update(alpha_equiv=op["alpha_equiv"], alpha_equiv_is_arithmetic=True)
        # HOW THE POINT WAS PICKED, on every record this arm produces. The AlphaSteer branch below
        # carries `selection_outcome` because a fallback must ride on every row; for this arm the point
        # is a DECLARED OVERRIDE of the budget rule, so the word alone is not enough — the SELECTED
        # reading and the reading of the point the RULE returns both travel, and
        # arditi_strength.raw_selection_problems refuses a row whose word and arithmetic disagree (a
        # `chosen` on an over-budget point most of all). Held here, where the record is built, so no
        # consumer of a verify record can see the dose without seeing how it was chosen.
        sel_problems = _st().raw_selection_problems(
            op, f"the arms table's {arm} operating point for {target}", budget_items=RAW_SELECTION_BUDGET_ITEMS)
        problems.extend(sel_problems)
        rec.update(selection_outcome=op.get("selection_outcome"))
        if isinstance(op.get("selection"), dict):
            rec["selection"] = op["selection"]
        # the dose, the view and the staged id are THREE spellings of one rung: a row naming c=0.3 with
        # ar_c0p5's directory would resolve the wrong bytes and every layer/sha gate below would pass on
        # them (they are dose-blind). Canonicalised through the lane's own grammar, never string-compared.
        for key, fn in (("bank_view", "view_name_in"), ("staged_id", "staged_id_in")):
            want = getattr(_st(), fn)(a["dose_unit"], op["c"])
            if op.get(key) != want:
                problems.append(f"the arms table's {arm} operating point for {target}: {key} {op.get(key)!r} is not "
                                f"c={op['c']!r} spelled ({want!r}) — the dose and the directory must name ONE rung")
    elif a["dose_unit"] == "base_scale_lambda":
        rec.update(**{"lambda": float(op["lambda"]), "rung": op["rung"], "rung_i": op["rung_i"],
                      "selection_outcome": op["selection_outcome"]})
    else:
        problems.append(f"arm {arm!r} declares dose_unit {a['dose_unit']!r}, which this checker has no "
                        "operating-point record for — add it deliberately, never record a dose whose unit "
                        "this tool cannot name")
    if any(frag in f"{ddir.as_posix()}/" for frag in PEER_ROOT_FRAGMENTS):
        problems.append(f"{ddir} resolves under a PEER's root — this driver never serves an arm from the wave bank")
    for f in ("object.pt", "staged_from.json"):
        if not (d / f).is_file():
            problems.append(f"{d}/{f} is missing")
    present = [s for s in IDENTITY_SIDECARS if (d / s).is_file()]
    if len(present) != 1:
        problems.append(f"{d} declares {len(present)} identity sidecars {present} (need exactly one of {IDENTITY_SIDECARS})")
    rec["identity_sidecar"] = present[0] if len(present) == 1 else present
    obj = d / "object.pt"
    if obj.is_file():
        sha = sha256_file(obj)
        rec["object_sha256"] = sha
        if sha != op["object_sha256"]:
            problems.append(f"served object.pt {sha[:12]} != the table's pin {op['object_sha256'][:12]} — NOT the object the "
                            "other lanes' rows name; refusing to serve it under this label")
        # THE SIDECAR'S OWN SHA CLAIM, if it makes one. This lineage's sweep.read_stage3_fit_provenance
        # (called by the standup, mt_serving_up._bank_objects, with object_path=obj) REFUSES a
        # fit_provenance.json whose recorded object_sha256 is not the artifact's — right for a stale pair,
        # but an AlphaSteer RUNG's record is deliberately the PARENT's byte copy (bake_rungs.py), so a
        # parent record that carries the parent's sha would crash the standup 40 minutes in. Refuse HERE,
        # in seconds, and say which reader would have refused. A record with no sha claim binds nothing
        # (the pin above does) and passes.
        fp = d / "fit_provenance.json"
        if fp.is_file():
            try:
                fp_rec = json.loads(fp.read_text(encoding="utf-8"))
            except ValueError as e:
                fp_rec = None
                problems.append(f"{fp} is not JSON: {e}")
            if isinstance(fp_rec, dict):
                rec["fit_provenance_writer"] = fp_rec.get("writer")
                rec["fit_provenance_object_sha256"] = fp_rec.get("object_sha256")
                if fp_rec.get("object_sha256") not in (None, sha):
                    problems.append(
                        f"{fp} records object_sha256 {str(fp_rec.get('object_sha256'))[:12]} but the served object.pt hashes "
                        f"to {sha[:12]} — sweep.read_stage3_fit_provenance (the standup's reader, mt_serving_up._bank_objects) "
                        "refuses that as a stale object/sidecar pair. For an AlphaSteer rung this means the PARENT's record "
                        "carries the parent's sha: the rung cannot be served through this standup until that is settled")
        import torch
        blob = torch.load(obj, map_location="cpu", weights_only=True)
        if not isinstance(blob, dict):
            problems.append(f"object.pt is a {type(blob).__name__}, not a blob dict")
            blob = {}
        bs = blob.get("base_scale")
        rec["base_scale"] = None if bs is None else float(bs)
        rec["blob_kind"] = blob.get("kind")
        # ROUTED BY THE SERVED KIND. The branch used to be `if arm == "arditi"`, and that was
        # not merely the wrong SELECTOR — the static_r contract below is the wrong CONTRACT for a
        # vector_all blob, which has no base_scale and no `rs` at all, so a correct all-positions
        # object would have been refused for lacking keys its kind does not have. Each kind
        # therefore gets its OWN total contract; neither was relaxed to admit the other, and an
        # unknown kind refuses rather than falling through to whichever branch is last.
        if a["kind"] == "static_r":
            if blob.get("kind") != a["kind"]:
                problems.append(f"blob kind {blob.get('kind')!r} != {a['kind']!r}")
            want_bs = float(a["expect_base_scale"])
            if bs is None or float(bs) != want_bs:
                problems.append(f"base_scale {bs!r} != {want_bs} — the dose lives in the vector's norm; a moved base_scale "
                                "is a FOLDED artifact serving a different dose than its row")
            rs = blob.get("rs") or {}
            layers = sorted(int(k) for k in rs) if isinstance(rs, dict) else []
            rec["rs_layers"] = layers
            if len(layers) != 1:
                problems.append(f"rs keyed by {layers} — a single-layer object stores exactly one layer")
            for k in ("positions", "fit_positions"):
                if k in blob:
                    problems.append(f"blob carries {k!r}, which this kind's adapter refuses (no declared position schema)")
            prov_p = d / "object.pt.provenance.json"
            if not prov_p.is_file():
                problems.append(f"{prov_p} is missing — a static_r blob stores the FIT layer and the adapter shifts it, so a "
                                "provenance recording both indices is required to tell a correct object from one baked a block off")
            else:
                prov = json.loads(prov_p.read_text(encoding="utf-8"))
                stored = prov.get("stored_layer")
                sel = (prov.get("fit") or {}).get("selected") or {}
                rec["stored_layer"] = stored
                rec["served_layer"] = sel.get("served_layer")
                if stored is None:
                    problems.append(f"{prov_p}: no stored_layer")
                elif layers and layers != [int(stored)]:
                    problems.append(f"rs keyed {layers} but the provenance's stored (fit) layer is {stored}")
                if sel.get("served_layer") is not None and stored is not None and int(stored) != int(sel["served_layer"]) + 1:
                    problems.append(f"{prov_p}: stored_layer {stored} != served_layer {sel['served_layer']} + 1")
                if prov.get("kind") != a["kind"]:
                    problems.append(f"{prov_p}: kind {prov.get('kind')!r} != {a['kind']!r}")
                if prov.get("phase") != a["phase"]:
                    problems.append(f"{prov_p}: phase {prov.get('phase')!r} != {a['phase']!r} — this arm serves prefill-only")
        elif a["kind"] == "vector_all":
            # THE ALL-POSITIONS CONTRACT, its own and equally total. Every assertion
            # here is the mirror of a static_r one and several are its INVERSE, which is the point:
            # a blob that satisfies one of the two contracts must fail the other, or the two arms
            # could be crossed by a rename. What this kind stores and what its adapter reads:
            #   * `vectors`, keyed by the SERVED block-output layer -- NOT `rs` keyed by the FIT
            #     layer. The layer convention INVERTS with the kind (a static_r blob at site="pre"
            #     stores fit_layer and the adapter shifts it to fit_layer - 1; a vector_all blob
            #     stores the served layer and the adapter serves it as stored), and confusing them
            #     steers a block early with every other signal healthy.
            #   * `positions` / `fit_positions` PRESENT and both "all" -- the two keys the static_r
            #     rule forbids. lens_adapters declares a position schema for this kind and reads the
            #     deployment out of them; lora_export._residual_vector_all additionally REFUSES a
            #     `response` fit served at `all`, so `fit_positions` is a fact the blob must carry.
            #   * `skip_bos` False -- Arditi's hook adds at every position, BOS included.
            #   * NO `base_scale`: this kind has none. The op's coeff is 1.0 and the dose is baked
            #     into the vector's norm, so a base_scale here would be a folded artifact of another
            #     kind wearing this one's name.
            if blob.get("kind") != a["kind"]:
                problems.append(f"blob kind {blob.get('kind')!r} != {a['kind']!r}")
            if bs is not None:
                problems.append(f"blob carries base_scale {bs!r} — a {a['kind']!r} object has none: its op's coeff is "
                                f"{a['expect_served_coeff']} and the dose is baked into the vector. A base_scale here is "
                                "another kind's blob under this arm's name, or a folded artifact serving another dose")
            if "rs" in blob:
                problems.append(f"blob carries 'rs' — a {a['kind']!r} object keys its layers under 'vectors'; 'rs' is the "
                                "static_r spelling, whose stored layer is the FIT layer, one LATER than this kind's")
            vectors = blob.get("vectors")
            layers = sorted(int(k) for k in vectors) if isinstance(vectors, dict) else []
            rec["vector_layers"] = layers
            if not isinstance(vectors, dict) or not vectors:
                problems.append(f"blob has no non-empty 'vectors' mapping ({type(vectors).__name__}) — this kind's payload")
            elif len(layers) != 1:
                problems.append(f"vectors keyed by {layers} — this arm serves exactly ONE layer")
            for k, want in (("positions", a["positions"]), ("fit_positions", a["fit_positions"])):
                if k not in blob:
                    problems.append(f"blob lacks {k!r}, which this kind's adapter READS to resolve its deployment — an "
                                    f"absent key is read as the kind's default rather than as this object's fact "
                                    f"(want {want!r}); the static_r arm FORBIDS these two keys, this one REQUIRES them")
                elif blob[k] != want:
                    problems.append(f"blob {k}={blob[k]!r} != {want!r} — this arm serves the paper's ungated hook at every "
                                    "position of every forward")
            if blob.get("skip_bos") is not bool(a["skip_bos"]):
                problems.append(f"blob skip_bos {blob.get('skip_bos')!r} != {bool(a['skip_bos'])} — Arditi's hook adds at "
                                "every position, BOS included")
            prov_p = d / "object.pt.provenance.json"
            if not prov_p.is_file():
                problems.append(f"{prov_p} is missing — this kind stores the SERVED layer where static_r stores the FIT "
                                "layer, so a provenance recording BOTH indices is required to tell a correct object from "
                                "one baked a block off")
            else:
                prov = json.loads(prov_p.read_text(encoding="utf-8"))
                stored = prov.get("stored_layer")
                sel = (prov.get("fit") or {}).get("selected") or {}
                rec["stored_layer"] = stored
                rec["served_layer"] = sel.get("served_layer")
                if stored is None:
                    problems.append(f"{prov_p}: no stored_layer")
                elif layers and layers != [int(stored)]:
                    problems.append(f"vectors keyed {layers} but the provenance's stored layer is {stored}")
                # THE INVERSE OF static_r's RELATION, asserted rather than assumed: this kind stores
                # the SERVED layer, so stored == served_layer (static_r: stored == served_layer + 1)
                if sel.get("served_layer") is not None and stored is not None and int(stored) != int(sel["served_layer"]):
                    problems.append(f"{prov_p}: stored_layer {stored} != served_layer {sel['served_layer']} — a "
                                    f"{a['kind']!r} blob stores the SERVED block-output layer and the adapter serves it as "
                                    "stored; storing the fit layer would steer one block LATER than the fit")
                if prov.get("kind") != a["kind"]:
                    problems.append(f"{prov_p}: kind {prov.get('kind')!r} != {a['kind']!r}")
                if prov.get("phase") != a["phase"]:
                    problems.append(f"{prov_p}: phase {prov.get('phase')!r} != {a['phase']!r} — this arm serves the paper's "
                                    "own configuration: prefill AND every decode step")
                if prov.get("dose_unit") != a["dose_unit"]:
                    problems.append(f"{prov_p}: dose_unit {prov.get('dose_unit')!r} != {a['dose_unit']!r} — the dose number "
                                    "means nothing without its unit, and the other unit is another intervention")
            # THE PINS' ABSOLUTE LAYERS (round 2, item 2). Everything above is RELATIONAL -- stored == served ==
            # vectors key -- and a coherent object baked a block off satisfies all of it. The pins entry for this
            # operating point's staged rung carries the FIT RECORD's layers (never a typed 13), and the blob's key,
            # the provenance's indices and its direction binding are held to THOSE. Missing keys refuse by name.
            binding, pin_problems = _raw_pin_binding(arms, a, op, target, pins, rec)
            problems.extend(pin_problems)
            if binding is not None:
                if layers and layers != [binding["stored_layer"]]:
                    problems.append(f"vectors keyed {layers} but the pins' stored_layer (= the fit record's served layer) is "
                                    f"{binding['stored_layer']} — the edit would land at block {layers[0]}, not where the fit put it")
                if prov_p.is_file():
                    problems.extend(_st().raw_provenance_layer_problems(prov, binding, str(prov_p)))
        elif a["kind"] == "alphasteer":
            lam = float(op["lambda"])
            if bs is None or float(bs) != lam:
                problems.append(f"base_scale {bs!r} != the table's lambda {lam!r} — THE DOSE IS THE SCALAR: a mismatch means "
                                "this directory is not the rung the table names")
            if blob.get("kind") not in ("alphasteer", None) and "Ms" not in blob:
                problems.append(f"blob kind {blob.get('kind')!r} carries no Ms — not an AlphaSteer rung")
            if blob.get("kind") is None and "Ms" not in blob:
                problems.append("blob has neither kind 'alphasteer' nor Ms — not an AlphaSteer rung")
        else:
            problems.append(f"arm {arm!r} declares served kind {a['kind']!r}, for which this checker has NO byte "
                            "contract — refusing rather than serving bytes nothing asserted anything about. Add the "
                            "kind's own total contract here, never widen another kind's to admit it")
    if config is not None:
        cfg = yaml.safe_load(Path(config).read_text(encoding="utf-8"))
        cells = list(cfg.get("explicit_cells") or [])
        if cells != ["none", cell]:
            problems.append(f"{config}: explicit_cells {cells} != ['none', {cell!r}] — the arm pair serves exactly the anchor "
                            "and the arm")
        want_dd = "out/" + view_rel(arms, target, arm)[len("exp/objects/"):]
        if cfg.get("defenses_dir") != want_dd:
            problems.append(f"{config}: defenses_dir {cfg.get('defenses_dir')!r} != {want_dd!r} — the pair was not generated "
                            "for this arm view")
        rec["config"] = str(config)
    rec["problems"] = problems
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--arms", type=Path, default=ARMS)
    ap.add_argument("--target", default=None, help="required except with --policy")
    ap.add_argument("--arm", required=True)
    ap.add_argument("--view", default=None, help="the job's view; must be the target's paper view")
    ap.add_argument("--cluster", default=None, help="cluster_a | cluster_b; refused unless the arm is staged there")
    ap.add_argument("--resolve", action="store_true", help="print `<cell> <kind> <view dir rel $SHARED> <policy|->` and exit")
    ap.add_argument("--policy", action="store_true", help="print the arm's declared serving policy (`-` = none) and exit")
    ap.add_argument("--requested-serving-path", default=None,
                    help="the path this launch resolved (F22_SERVING_PATH); refused by name if the arm's policy contradicts it. "
                         "Empty = nothing resolved yet, asserts nothing")
    ap.add_argument("--serving-path-mode", default=None, help="explicit | auto | deferred — rides the refusal message")
    ap.add_argument("--defenses-dir", type=Path, default=None, help="the resolved view dir: run the byte invariant")
    ap.add_argument("--config", type=Path, default=None, help="the arm config; its cells + defenses_dir must match")
    ap.add_argument("--pins", type=Path, default=None,
                    help="the arm's pins file (default: the table's `arms.<arm>.pins`, relative to this checkout); a vector_all "
                         "arm's provenance and blob are held to its entry's fit_layer / served_layer / stored_layer / direction_sha256")
    args = ap.parse_args(argv)
    arms = load_arms(args.arms)
    requested = args.requested_serving_path or None
    if args.policy:
        problems = serving_policy_problems(arms, args.arm, requested, mode=args.serving_path_mode or None)
        for p in problems:
            print("PROBLEM", p, file=sys.stderr)
        if problems:
            return 1
        print(serving_policy(arms, args.arm) or "-")
        return 0
    if args.target is None:
        ap.error("--target is required (except with --policy)")
    if requested and serving_policy_problems(arms, args.arm, requested, mode=args.serving_path_mode or None):
        for p in serving_policy_problems(arms, args.arm, requested, mode=args.serving_path_mode or None):
            print("PROBLEM", p, file=sys.stderr)
        return 1
    cell, kind, rel = resolve(arms, args.target, args.arm, args.view, args.cluster, requested)
    if args.resolve:
        print(cell, kind, rel, serving_policy(arms, args.arm) or "-")
        return 0
    if args.defenses_dir is None:
        ap.error("one of --resolve or --defenses-dir is required")
    rec = verify(arms, args.target, args.arm, args.defenses_dir, args.config, pins=args.pins)
    for p in rec["problems"]:
        print("PROBLEM", p, file=sys.stderr)
    print(json.dumps(rec, sort_keys=True))
    return 1 if rec["problems"] else 0


if __name__ == "__main__":
    sys.exit(main())
