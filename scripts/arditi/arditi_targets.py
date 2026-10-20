#!/usr/bin/env python
"""The strength-frontier targets of the Arditi lane, resolved from the WAVE roster.

Model id and revision are never typed here: they are read from
``configs/sdfbase_full22_roster.yaml``, the campaign's one source of truth for
what each target is, so an Arditi fit can never run against a different
snapshot than the battery that will serve it. This file adds only the two
facts the roster cannot state for this lane: which of the wave's views is the
target's PAPER LANE (the config whose body our serving configs derive from),
and the cell id.

The STRENGTH grammar — the ladder, view names, staged ids and bank paths —
lives in ``arditi_strength.py``, which imports from here. It was moved out of
this file on 2026-09-10 when the lane adopted ``as22_strength``'s spelling
wholesale (see that module's header); nothing may reintroduce a second
spelling here.

The paper-lane table matches ``scripts/alphasteer_f22/as22_targets.VIEW_OF``:
gemma4e4b, qwen35_4b and llama8b are served instruct-direct (`it`),
nemotron3nano4b and glm46vflash on `main`.

**llama8b is OFF the wave roster this tree carries, on purpose.** The
roster here predates the ``sdfbase-llama8b`` lane, so llama8b has no entry in
it, and the fourth paper family would otherwise have no Arditi fit at all —
the coverage gap this closes. Rather than hand-type a model pin (the one thing
this module exists not to do), its wave config and manifest are COPIED
byte-identical from the peer branch at the commit
:data:`OFF_ROSTER_WAVE_SOURCES` records, with their blob sha256s pinned there
and tested. That copy IS the provenance a roster entry would otherwise carry,
and ``model_pin`` reads the pins out of the copied config rather than out of
this file. Same mechanism, same peer branch and the SAME COMMIT as
``OFF_ROSTER_WAVE_SOURCES`` below — it lives HERE, not in
``scripts/alphasteer_f22/as22_targets``, which carries no llama reference at all — so the
two method lanes of Figure 5 carry byte-identical copies of the llama8b wave
body rather than two drifting ones.
"""
from __future__ import annotations

from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
WAVE_ROSTER = REPO / "configs" / "sdfbase_full22_roster.yaml"

#: the wave's four in the wave's order, then llama8b APPENDED. The order
#: of the first four is load-bearing for every consumer that zips against it —
#: this lane's sbatch scripts index this order by ``SLURM_ARRAY_TASK_ID``, so
#: llama8b is array index 4 and no existing index moves.
TARGETS = ("gemma4e4b", "qwen35_4b", "nemotron3nano4b", "glm46vflash", "llama8b")

#: target -> the wave view whose config body our serving configs derive from
PAPER_LANE = {"gemma4e4b": "it", "qwen35_4b": "it",
              "nemotron3nano4b": "main", "glm46vflash": "main",
              "llama8b": "it"}

#: The ONE served cell id of this lane: a bare Class B id — the direction is
#: fitted from harmful-vs-harmless prompts with no system instruction, so it
#: is SI-agnostic by construction.
CELL_ID = "arditi"

#: Targets whose wave config / manifest are NOT in this tree's roster and were
#: copied byte-identical from a peer branch instead: target -> {branch, commit,
#: files: {rel path: blob sha256}, model pins}. ``tests/test_arditi_serving.py``
#: pins it: the files on disk must hash to these shas, and the copied config's
#: own ``model`` / ``model_revision`` must equal the recorded pins — so a copy
#: that drifted, or a pin typed wrong, is a red test rather than a fit against
#: the wrong snapshot.
#:
#: WHY THIS COMMIT AND NOT THE PEER BRANCH'S TIP, measured rather than assumed:
#: at the branch tip (``de78ab46``) the it-view config differs from this
#: commit in EXACTLY ONE key, ``explicit_cells`` (and the manifest in its
#: ``cells`` list) — and this lane's generator REPLACES both wholesale with
#: ``[none, arditi]``. So both commits yield byte-identical Arditi outputs, and
#: the tie is broken by matching the peer lane's existing pin.
OFF_ROSTER_WAVE_SOURCES = {
    "llama8b": {
        "branch": "sdfbase-llama8b-wave",
        "commit": "d358287b0047d6b2cb303a82ac68c96f1854517b",
        "files": {
            "configs/sdfbase_full22_llama8b_it.yaml":
                "581b6c845481814b896242bfb0e574551841ff85fc43cc17bfff267c2251d913",
            "configs/capeval/manifests/petri_full22_llama8b_it.yaml":
                "6f62f11fe6c14ff8bcc931afd4156e211234595153dd8484fffb69c1bbc32764",
        },
        "model": "NousResearch/Meta-Llama-3.1-8B-Instruct",
        "model_revision": "d10aef7999a2b5ba950ab3974312feeedbfe0b77",
        "note": ("the sdfbase-llama8b lane's wave files at that commit, the same pair "
                 "scripts/alphasteer_f22/as22_targets.OFF_ROSTER_WAVE_SOURCES pins (D260), so "
                 "the AlphaSteer and Arditi lanes of Figure 5 derive llama8b from the same bytes"),
    },
}


def paper_lane(target: str) -> str:
    """The wave view this target's configs derive from; refuses an unknown
    target loudly rather than defaulting to `main`."""
    try:
        return PAPER_LANE[target]
    except KeyError:
        raise SystemExit(
            f"arditi_targets: {target!r} is not a lane target ({sorted(PAPER_LANE)})") from None


def load_wave_roster(path: Path = WAVE_ROSTER) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _off_roster_pin(target: str) -> dict:
    """``model_pin`` for a target carried as a COPIED wave body (llama8b).

    The model id and revision are READ BACK OUT OF THE COPIED CONFIG, not taken
    from the table — the copied file is the authority, and the table's ``model``
    / ``model_revision`` entries exist to be cross-checked against it. A file
    that says one thing while the table says another is a refusal here, not a
    fit against whichever of the two happened to be read first."""
    src = OFF_ROSTER_WAVE_SOURCES[target]
    view = paper_lane(target)
    body_config = f"configs/sdfbase_full22_{target}_{view}.yaml"
    body_manifest = f"configs/capeval/manifests/petri_full22_{target}_{view}.yaml"
    if set(src["files"]) != {body_config, body_manifest}:
        raise SystemExit(
            f"arditi_targets: {target}'s OFF_ROSTER_WAVE_SOURCES pins {sorted(src['files'])}, "
            f"but its paper lane {view!r} needs {[body_config, body_manifest]}")
    body = yaml.safe_load((REPO / body_config).read_text(encoding="utf-8"))
    model, revision = body.get("model"), body.get("model_revision")
    if (model, revision) != (src["model"], src["model_revision"]):
        raise SystemExit(
            f"arditi_targets: {body_config} serves {model}@{revision} but "
            f"OFF_ROSTER_WAVE_SOURCES records {src['model']}@{src['model_revision']} — the copied "
            "body and its pin disagree, so one of the two is wrong; refusing rather than picking")
    return {"target": target, "model": model, "model_revision": revision,
            "wave_view": view, "body_config": body_config,
            "body_manifest": body_manifest,
            "off_roster_source": {k: src[k] for k in ("branch", "commit")}}


def model_pin(target: str, roster: dict | None = None) -> dict:
    """``{target, model, model_revision, wave_view, body_config, body_manifest}``
    for a target, read from the wave roster (never typed here) — or, for a
    target the roster predates, from its COPIED wave body (see
    :data:`OFF_ROSTER_WAVE_SOURCES`)."""
    roster = roster or load_wave_roster()
    if target not in roster["targets"]:
        if target in OFF_ROSTER_WAVE_SOURCES:
            return _off_roster_pin(target)
        raise KeyError(f"{target!r} is not a wave-roster target "
                       f"({sorted(roster['targets'])}) and has no OFF_ROSTER_WAVE_SOURCES entry")
    t = roster["targets"][target]
    view = paper_lane(target)
    v = t["views"][view]
    return {"target": target, "model": t["model"],
            "model_revision": t["model_revision"], "wave_view": view,
            "body_config": v["config"], "body_manifest": v["manifest"]}


def body_bon_cap(target: str, roster: dict | None = None) -> int:
    """This target's BoN decode cap, READ OFF ITS WAVE BODY.

    The frontier's BoN instrument is a per-TARGET fact, not a lane-wide one, and
    it is not this lane's to choose: the body config carries whatever the
    campaign ruled for that target, and our job is to serve the same instrument
    so the three method lanes' BoN numbers stay ONE estimand. Four targets carry
    512 (the 512 -> 4,096 climb); llama8b carries 4,096, because it has no 512
    store at all and the canonical-battery rule makes it a NATIVE 4,096 pass for
    every lane. Read, never typed, so a body that moves moves us — and a body
    whose cap this lane has no route for is a refusal."""
    pin = model_pin(target, roster)
    body = yaml.safe_load((REPO / pin["body_config"]).read_text(encoding="utf-8"))
    cap = (body.get("bon") or {}).get("max_new_tokens")
    if not isinstance(cap, int) or isinstance(cap, bool) or cap <= 0:
        raise SystemExit(f"arditi_targets: {pin['body_config']} has no positive integer "
                         f"bon.max_new_tokens (got {cap!r}) — refusing to guess this target's "
                         "BoN instrument")
    return cap


#: The BoN route each body cap implies, as DATA rather than lore — a Figure-5
#: join must be able to read which route a row's BoN number came from, because
#: a native pass and a climbed 512 store are one estimand only by construction,
#: and the climb tooling refuses a store whose cap is not 512. The rule: a 512
#: body cap is climbed to 4,096; a 4,096 body cap IS the native pass (llama8b,
#: which has no 512 store).
BON_ROUTES = {512: "climb_512_to_4096", 4096: "native_4096"}


def bon_route(target: str, roster: dict | None = None) -> str:
    cap = body_bon_cap(target, roster)
    try:
        return BON_ROUTES[cap]
    except KeyError:
        raise SystemExit(
            f"arditi_targets: {target}'s body BoN cap {cap} is neither 512 (climbed) nor 4096 "
            "(native) — this lane has no route for it, and inventing one would put two "
            "instruments in one column") from None


if __name__ == "__main__":
    import json

    roster = load_wave_roster()
    print(json.dumps({t: {**model_pin(t, roster),
                          "bon_max_new_tokens": body_bon_cap(t, roster),
                          "bon_route": bon_route(t, roster)}
                      for t in TARGETS}, indent=1))
