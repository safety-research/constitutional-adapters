"""Continue a cap-limited document-CE fit from its RETAINED checkpoint under a
raised epoch cap: a fit that hit the 24-epoch cap again RESUMES for up to 10
more epochs with early stopping; it is not retrained.

What a continuation IS, precisely. ``train_object_ce`` checkpoints its FULL
state at optimizer-group boundaries (fitckpt: trainables, optimizer state,
schedule position via the step count, the epoch's shuffled order + RNG
states, the held-out curve, ``heldout_noop_ce`` and the whole early-stopping
state — best value, patience counter, best-epoch snapshot), and a COMPLETE
cell retains its LAST such checkpoint as ``<cell>/final_ckpt/ckpt.pt``
(sdfbase_fits.retain_final_checkpoint). A continuation ADOPTS that
checkpoint into the new fit's transient slot under the new fit's identity
(``checkpoint_key`` := the continuation fingerprint, :func:`adopt`) and
runs the ordinary trainer with ``epochs`` = the raised cap. The trainer's
own resume restores everything — so the data order, the held-out pool, the
patience counter and the best-epoch snapshot are the prior fit's, and early
stopping can fire inside the continuation on the RESTORED patience state.
The retained checkpoint is generally MID-EPOCH (the last periodic write
before the cap — e.g. optimizer step 1200 of 1224): the continuation re-runs
the tail of that epoch bitwise (the fitckpt contract) and then trains the
new epochs. Nothing in the trainer changes; the trainer only gained a
``resumed_from_step`` diagnostic so the driver can PROVE the resume
happened (a continuation that silently started fresh must never publish).

Identity — fail loudly, never guess. The checkpoint's ``checkpoint_key`` is
the prior fit's fingerprint; the driver requires it to EQUAL the
fingerprint the CURRENT config/pool/held-out pool/positions resolve to, so
nothing but the cap can differ (model, revision, SI, arm, pool bytes,
optimizer settings, held-out sha and config bytes are all bound by that one
equality). :func:`verify_cell` then binds the checkpoint to the cell's
PUBLISHED object: COMPLETE fingerprint == checkpoint key; marker, sidecar
and object shas agree; the checkpoint's step/epoch/curve lengths are
consistent with the recorded fit (a checkpoint past the fit's last step
postdates the object); and — whenever the fit's best epoch had already
been seen at checkpoint time — the checkpoint's best-epoch snapshot must
equal the published object's tensors exactly (the object IS the restored
best snapshot, so any difference means another fit's checkpoint).

LR schedule, stated plainly. The campaign's ``warmup_cosine`` decays to 0
over ``total_steps = cap x steps_per_epoch``; at the cap the lr is ~0
(3.7e-9 at the last step of a round-1 cell, measured), so a continuation
that "does not jump the lr" would continue at ~0 and learn nothing — ten
no-op epochs wearing a higher cap. No continuation of a fully decayed
cosine both learns and stays continuous. The choice made here: the
continuation runs, FROM THE RESUME STEP ONWARD, the schedule a fit capped
at N from the start would have run — ``total_steps = N x steps_per_epoch``
and ``warmup_steps`` resolved against that total — which is exactly what
the trainer computes from ``epochs=N``; no schedule is invented. That IS a
warm restart at the resume step (from ~0 back to the new horizon's value
there, roughly a quarter of peak for the 1M cells), decaying to 0 at the
new cap; :func:`schedule_report` records both lr values so the jump is on
the record, not implied. Under ``lr_schedule: constant`` the continuation
is bitwise the straight-to-N fit (tests/test_refit_continuation.py pins
it); under ``warmup_cosine`` the pre-checkpoint steps carry the cap-M
history, so the continued object is NOT the scratch cap-N object — and its
fingerprint deliberately says so (:func:`continuation_marker` rides in the
fingerprint's ``optimizer`` slot: a scratch cap-N fit never matches it).

CHAINED continuation. A continued cell's retained checkpoint carries
the CONTINUATION's key, not the plain cell identity — so a second
continuation cannot prove the key by one equality against the config. The
proof is the CHAIN: the cell's COMPLETE marker records the publication's
own ``continued_from`` (the marker it ran under; since this change also
the full ``chain`` of markers back to the root), and :func:`resolve_chain`
walks it from the plain identity — each link's ``fit_fingerprint`` must
equal the fingerprint the previous links resolve to, and the last link
must reproduce the checkpoint key. ``prior_cap`` (the cap check and the
schedule report's prior horizon) is the last link's ``epochs_cap``, and the
new publication records ``chain`` = the verified chain + its own marker, so
the (n+1)-th continuation walks n links, never trusts a key. The driver,
the Modal app's fingerprint derivation (``fingerprint_cpu`` — the pre-spawn
check) and the snapshot mode all resolve the SAME chain through this one
function, so a checkpoint the container would refuse is refused before any
GPU spawn. A publication written before this change carries no ``chain``;
its one link is rebuilt from the legacy ``continued_from`` fields
(:func:`marker_from_publication`), byte-identical to the marker it ran
under.

LR multiplier. The warm restart above resumes at whatever the
cap-N cosine reads at the resume step — 0.24–0.29 of peak for the 1M
cap 24→34 raise, but 0.90–0.97 of peak for a 4× raise (8→32, 4→16), nearly
a full restart. ``lr_mult`` scales the restarted schedule's PEAK (warmup
ramp and cosine shape unchanged — the schedule is linear in the peak, so
every step's lr is exactly ``lr_mult`` × the unscaled value), rides in the
marker (a different multiplier is a different fit) and in
``continued_from.lr_schedule``. 1.0 leaves the marker byte-identical to the
pre-multiplier one, so every existing continuation fingerprint stands.

Stopper rule ``replay_exempt``. The retained checkpoint is mid-epoch,
so every continuation begins by REPLAYING the remainder of the
checkpoint epoch — a partial epoch trained under the freshly restarted lr
— and evaluating held-out at its end. Under the legacy rule that value was
an ordinary early-stopping verdict: at patience 1 (the full-corpus cells)
or with one strike already banked (the 10M cells) it ended the run before
a single FULL post-restart epoch had trained — every "no gain from
continuing" verdict of cont2/cont3 was this artifact, while the runs that
survived the restart kept improving. Under ``replay_exempt`` the
trainer (``train_object_ce(replay_exempt_epoch=<resume epoch>)``) logs the
replayed epoch's value, keeps it in the curve flagged as replayed
(``fit_record.replayed_partial_epochs``), and neither strikes nor seats it
as a new best (``EarlyStopper.observe_exempt``); the incumbent seating is
unchanged and patience counts from the first full epoch. The rule is a fit
identity dimension: it rides in the marker as ``stopper_rule =
"replay_exempt"`` (:func:`continuation_marker`; a legacy marker carries no
such field, and an explicit legacy value never rides, so every fingerprint
stamped before this rule existed stands), a same-cap same-multiplier
re-issue under a DIFFERENT rule from the prior link's is admissible
(:func:`check_cap_admissible`), and — since the replayed epoch is no longer
judged — a same-cap re-issue whose checkpoint sits inside the cap's final
epoch is refused outright (no epoch would be ruled on). The driver passes
the exemption to the trainer PROTECTED, so a trainer that lacks the kwarg
fails the arm loudly rather than running the legacy rule under a
``replay_exempt`` fingerprint.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import torch

from .fitckpt import save_checkpoint

#: The one trainer whose checkpoints carry the full early-stopping state.
TRAINER = "train_object_ce"
#: Every field train_object_ce's checkpoint writer emits; a file missing one
#: is not a checkpoint this module can continue.
REQUIRED_STATE_KEYS = (
    "checkpoint_key", "trainer", "n_batches", "epoch", "batches_done",
    "order", "step", "history", "trainable", "opt_state", "rng_state",
    "torch_rng_state", "heldout_curve", "heldout_noop_ce", "stopper")
STOPPER_KEYS = ("best", "best_epoch", "epochs_seen", "since_best", "snapshot")
#: Where a COMPLETE cell retains its last checkpoint (retain_final_checkpoint).
FINAL_CKPT_DIR = "final_ckpt"
FINAL_CKPT_NAME = "ckpt.pt"
COMPLETE_NAME = "COMPLETE"
SIDECAR_NAME = "train_budget_meta.json"
SUPERSEDE_NAME = "superseded_by.json"
#: Early-stopping rule for a continuation's replayed checkpoint epoch:
#: the replayed partial epoch is logged, never ruled on. The rule every
#: continuation now runs under, and the only value that ever RIDES in a
#: marker.
STOPPER_RULE_REPLAY_EXEMPT = "replay_exempt"
#: The name of the legacy rule (every epoch ruled on, the replayed one
#: included) — the meaning of a marker WITHOUT ``stopper_rule``; it never
#: rides in a marker (it would move the fingerprint of an unchanged fit).
STOPPER_RULE_LEGACY = "strict"
STOPPER_RULES = (STOPPER_RULE_REPLAY_EXEMPT,)


class ContinuationError(RuntimeError):
    """A continuation that cannot be proven is not started."""


def final_checkpoint_path(cell_dir) -> Path:
    return Path(cell_dir) / FINAL_CKPT_DIR / FINAL_CKPT_NAME


def _sha256_file(path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# The checkpoint                                                               #
# --------------------------------------------------------------------------- #


def load_state(path) -> dict:
    """The retained checkpoint's full state, loaded on CPU and checked for
    INTERNAL consistency (a train_object_ce checkpoint with every field;
    history length == step; order length == n_batches; batches_done inside
    the epoch; a well-formed stopper whose epochs_seen equals the epoch
    index and whose held-out curve has the init point plus one value per
    completed epoch). Raises :class:`ContinuationError` — never returns a
    partial or foreign state."""
    p = Path(path)
    if not p.is_file():
        raise ContinuationError(
            f"{p} does not exist — no retained checkpoint to continue from "
            f"(a COMPLETE cell keeps its last checkpoint at "
            f"<cell>/{FINAL_CKPT_DIR}/{FINAL_CKPT_NAME})")
    try:
        state = torch.load(str(p), map_location="cpu", weights_only=False)
    except Exception as e:  # noqa: BLE001 — a torn file is a refusal
        raise ContinuationError(
            f"{p} is unreadable ({type(e).__name__}: {e})") from e
    if not isinstance(state, dict):
        raise ContinuationError(f"{p} is not a checkpoint state dict")
    missing = [k for k in REQUIRED_STATE_KEYS if k not in state]
    if missing:
        raise ContinuationError(
            f"{p} lacks checkpoint field(s) {missing} — not a complete "
            f"{TRAINER} checkpoint")
    if state.get("trainer") != TRAINER:
        raise ContinuationError(
            f"{p} was written by trainer {state.get('trainer')!r}, not "
            f"{TRAINER} — only the document-CE trainer's checkpoints carry "
            "the full early-stopping state a continuation restores")
    step, epoch = int(state["step"]), int(state["epoch"])
    n = int(state["n_batches"])
    if len(state["history"]) != step:
        raise ContinuationError(
            f"{p}: history has {len(state['history'])} entries but step is "
            f"{step} — an internally inconsistent checkpoint")
    if len(state["order"]) != n:
        raise ContinuationError(
            f"{p}: order covers {len(state['order'])} batches, n_batches is "
            f"{n} — an internally inconsistent checkpoint")
    if not 0 <= int(state["batches_done"]) <= n:
        raise ContinuationError(
            f"{p}: batches_done={state['batches_done']} outside "
            f"[0, {n}]")
    st = state.get("stopper")
    if st is not None:
        if not isinstance(st, dict) or any(k not in st for k in STOPPER_KEYS):
            raise ContinuationError(
                f"{p} carries a malformed stopper state (need "
                f"{list(STOPPER_KEYS)})")
        if int(st["epochs_seen"]) != epoch:
            raise ContinuationError(
                f"{p}: stopper epochs_seen={st['epochs_seen']} != epoch "
                f"index {epoch} — the early-stopping state does not belong "
                "to this checkpoint position")
        curve = state.get("heldout_curve") or []
        if len(curve) != epoch + 1:
            raise ContinuationError(
                f"{p}: held-out curve has {len(curve)} points, expected "
                f"{epoch + 1} (the init point + one per completed epoch)")
    return state


def summarize(state: dict, path=None) -> dict:
    """The JSON-safe, tensor-free description of a checkpoint (what the
    manifest row, the provenance record and the logs carry)."""
    st = state.get("stopper")
    best = None
    if st is not None and st.get("best") is not None:
        b = float(st["best"])
        best = b if math.isfinite(b) else None
    return {
        "path": None if path is None else str(path),
        "checkpoint_key": str(state["checkpoint_key"]),
        "trainer": str(state["trainer"]),
        "n_batches": int(state["n_batches"]),
        "epoch": int(state["epoch"]),
        "batches_done": int(state["batches_done"]),
        "step": int(state["step"]),
        "heldout_curve_len": len(state.get("heldout_curve") or []),
        "heldout_noop_ce": state.get("heldout_noop_ce"),
        # 1-based epochs of the curve that were replayed partial epochs
        # (logged, not ruled on) in a prior continuation; [] when none.
        "replayed_partial_epochs": [
            int(e) for e in state.get("replayed_partial_epochs") or []],
        "stopper": None if st is None else {
            "best": best,
            "best_epoch": int(st["best_epoch"]),
            "epochs_seen": int(st["epochs_seen"]),
            "since_best": int(st["since_best"]),
            "has_snapshot": st.get("snapshot") is not None,
        },
    }


#: The marker fields, in the order the marker is written.
MARKER_KEYS = ("fit_fingerprint", "step", "epoch", "batches_done", "epochs_cap")


def check_lr_mult(lr_mult) -> float:
    """``lr_mult`` as a finite positive float, else a refusal."""
    try:
        m = float(lr_mult)
    except (TypeError, ValueError) as e:
        raise ContinuationError(
            f"lr_mult must be a number, got {lr_mult!r}") from e
    if not math.isfinite(m) or m <= 0:
        raise ContinuationError(
            f"lr_mult must be a finite positive multiplier, got {lr_mult!r}")
    return m


def check_stopper_rule(stopper_rule) -> str:
    """A stopper rule as the marker spells it: ``None`` or the legacy name
    -> :data:`STOPPER_RULE_LEGACY` (rides in NO marker); a known rule name
    unchanged; anything else a refusal."""
    if stopper_rule is None or stopper_rule == STOPPER_RULE_LEGACY:
        return STOPPER_RULE_LEGACY
    if stopper_rule in STOPPER_RULES:
        return str(stopper_rule)
    raise ContinuationError(
        f"unknown continuation stopper rule {stopper_rule!r} (known: "
        f"{list(STOPPER_RULES)}; {STOPPER_RULE_LEGACY!r} = the legacy rule)")


def continuation_marker(summary: dict, epochs_cap, lr_mult=1.0,
                        stopper_rule=STOPPER_RULE_REPLAY_EXEMPT) -> dict:
    """The continuation's identity contribution — what rides in the fit
    fingerprint's ``optimizer`` slot (``optimizer.continued_from``): the
    prior fit's fingerprint, the exact resume position and the new cap. One
    derivation for the driver (``cell_fit_fingerprint``), the Modal app's
    ``expected_fit_fingerprint`` and the watcher's manifest row, so the
    marker the container stamps is the value the row expects. ``lr_mult``
    joins the marker ONLY when it is not 1.0 — a 1.0 marker is
    byte-identical to the pre-multiplier marker, so every fingerprint
    stamped before the multiplier existed still resolves. ``stopper_rule``
    likewise joins only for a non-legacy rule; the default is the rule
    every continuation now runs under, so a new continuation is a distinct
    generation from the legacy ones by construction, while
    ``stopper_rule=None`` derives a legacy marker (what a publication
    fingerprinted under the legacy rule carried)."""
    cap = int(epochs_cap)
    if cap < 1:
        raise ContinuationError(f"epochs_cap must be >= 1, got {epochs_cap!r}")
    m = check_lr_mult(lr_mult)
    rule = check_stopper_rule(stopper_rule)
    marker = {"fit_fingerprint": str(summary["checkpoint_key"]),
              "step": int(summary["step"]),
              "epoch": int(summary["epoch"]),
              "batches_done": int(summary["batches_done"]),
              "epochs_cap": cap}
    if m != 1.0:
        marker["lr_mult"] = m
    if rule != STOPPER_RULE_LEGACY:
        marker["stopper_rule"] = rule
    return marker


def marker_lr_mult(marker: dict) -> float:
    """The multiplier a marker ran under (1.0 when it carries none)."""
    return check_lr_mult(marker.get("lr_mult", 1.0))


def marker_stopper_rule(marker: dict) -> str:
    """The stopper rule a marker ran under — :data:`STOPPER_RULE_LEGACY`
    when it carries none (every legacy-rule continuation)."""
    return check_stopper_rule(marker.get("stopper_rule"))


# --------------------------------------------------------------------------- #
# The chain                                                                    #
# --------------------------------------------------------------------------- #


def _check_marker(marker, what: str) -> dict:
    if not isinstance(marker, dict):
        raise ContinuationError(f"{what} is not a marker mapping: {marker!r}")
    missing = [k for k in MARKER_KEYS if k not in marker]
    if missing:
        raise ContinuationError(f"{what} lacks marker field(s) {missing}")
    out = {"fit_fingerprint": str(marker["fit_fingerprint"]),
           "step": int(marker["step"]), "epoch": int(marker["epoch"]),
           "batches_done": int(marker["batches_done"]),
           "epochs_cap": int(marker["epochs_cap"])}
    if "lr_mult" in marker:
        m = check_lr_mult(marker["lr_mult"])
        if m == 1.0:
            raise ContinuationError(
                f"{what} carries lr_mult 1.0 explicitly — a 1.0 multiplier "
                "never rides in a marker (it would change the fingerprint "
                "of an unchanged fit)")
        out["lr_mult"] = m
    if "stopper_rule" in marker:
        rule = check_stopper_rule(marker["stopper_rule"])
        if rule == STOPPER_RULE_LEGACY:
            raise ContinuationError(
                f"{what} carries the legacy stopper rule explicitly — the "
                "legacy rule never rides in a marker (it would change the "
                "fingerprint of an unchanged fit)")
        out["stopper_rule"] = rule
    return out


def marker_from_publication(cont_block: dict) -> dict:
    """The marker a PUBLISHED continuation ran under, rebuilt from its
    ``continued_from`` provenance block (the fields the driver records:
    ``prior_fit_fingerprint``, ``resume_step``, ``resume_epoch``,
    ``resume_batches_done``, ``new_cap`` [, ``lr_mult``
    [, ``stopper_rule``]]) — byte-identical to the ``continuation_marker``
    it was fingerprinted with, so a publication written before ``chain``
    existed still walks."""
    if not isinstance(cont_block, dict):
        raise ContinuationError(
            f"continued_from is not a mapping: {cont_block!r}")
    need = ("prior_fit_fingerprint", "resume_step", "resume_epoch",
            "resume_batches_done", "new_cap")
    missing = [k for k in need if cont_block.get(k) is None]
    if missing:
        raise ContinuationError(
            f"the publication's continued_from lacks {missing} — the marker "
            "it ran under cannot be rebuilt")
    marker = {"fit_fingerprint": str(cont_block["prior_fit_fingerprint"]),
              "step": int(cont_block["resume_step"]),
              "epoch": int(cont_block["resume_epoch"]),
              "batches_done": int(cont_block["resume_batches_done"]),
              "epochs_cap": int(cont_block["new_cap"])}
    m = check_lr_mult(cont_block.get("lr_mult", 1.0))
    if m != 1.0:
        marker["lr_mult"] = m
    rule = check_stopper_rule(cont_block.get("stopper_rule"))
    if rule != STOPPER_RULE_LEGACY:
        marker["stopper_rule"] = rule
    return marker


def publication_chain(cell_dir) -> list[dict]:
    """The chain of continuation markers the cell's CURRENT publication
    ran under, oldest first — ``[]`` for a first-generation fit (a COMPLETE
    marker without ``continued_from``); the recorded ``continued_from.
    chain`` when present; else the ONE link rebuilt from the legacy
    ``continued_from`` fields. Refuses (:class:`ContinuationError`) a
    missing or unparseable COMPLETE marker: the chain lives there, and a
    key that is not the plain identity cannot be proven without it."""
    cell_dir = Path(cell_dir)
    problems: list[str] = []
    marker = _read_json(cell_dir / COMPLETE_NAME, problems, "COMPLETE marker")
    if marker is None:
        raise ContinuationError("; ".join(problems))
    cf = marker.get("continued_from")
    if cf is None:
        return []
    if not isinstance(cf, dict):
        raise ContinuationError(
            f"{cell_dir / COMPLETE_NAME}: continued_from is not a mapping")
    chain = cf.get("chain")
    if chain is not None:
        if not isinstance(chain, list) or not chain:
            raise ContinuationError(
                f"{cell_dir / COMPLETE_NAME}: continued_from.chain is not a "
                "non-empty list")
        return [_check_marker(m, f"continued_from.chain[{i}]")
                for i, m in enumerate(chain)]
    return [marker_from_publication(cf)]


def resolve_chain(cell_dir, state: dict, plain_fp: str, fp_of) -> dict:
    """PROVE a retained checkpoint's key from the plain cell identity
    through the publication's recorded chain — the identity gate every
    continuation (first or chained) passes before anything else.

    ``plain_fp`` is the fingerprint the CURRENT config / pool / held-out
    pool / positions resolve to with no continuation marker; ``fp_of`` maps
    a marker to the fingerprint of the fit that ran under it (the driver's
    ``cell_fit_fingerprint(continued_from=marker)``). Walk: ``fp :=
    plain_fp``; for each link, its ``fit_fingerprint`` must EQUAL ``fp``
    (the link continued exactly the fit the previous links resolve to) and
    ``fp := fp_of(link)``; finally ``fp`` must equal the checkpoint key.
    A key equal to ``plain_fp`` needs no chain (a first-generation
    checkpoint) and the COMPLETE marker is not consulted for it; any other
    key is refused unless the walk reproduces it. Returns ``{"chain",
    "depth", "root_fit_fingerprint", "prior_cap", "prior_lr_mult",
    "prior_stopper_rule"}`` — ``prior_cap`` is None for a first-generation
    checkpoint (the config's cap applies), else the last link's
    ``epochs_cap``; ``prior_stopper_rule`` is None for a
    first-generation checkpoint (no continuation ran, so no rule applied),
    else the rule the last link ran under."""
    key = str(state["checkpoint_key"])
    if key == str(plain_fp):
        return {"chain": [], "depth": 0, "root_fit_fingerprint": key,
                "prior_cap": None, "prior_lr_mult": 1.0,
                "prior_stopper_rule": None}
    try:
        chain = publication_chain(cell_dir)
    except ContinuationError as e:
        raise ContinuationError(
            f"checkpoint_key {key[:16]}… is not the plain identity "
            f"{str(plain_fp)[:16]}… this config + pool + held-out pool + "
            f"positions resolve to, and the publication's chain cannot be "
            f"read ({e}) — a continuation must run under EXACTLY the prior "
            "fit's identity (same config bytes, same pools, same optimizer "
            "settings; only the cap and lr_mult may differ). Refusing.")
    if not chain:
        raise ContinuationError(
            f"checkpoint_key {key[:16]}… is not the plain identity "
            f"{str(plain_fp)[:16]}… this config + pool + held-out pool + "
            "positions resolve to, and the publication records no "
            "continuation — a continuation must run under EXACTLY the prior "
            "fit's identity (same config bytes, same pools, same optimizer "
            "settings; only the cap and lr_mult may differ). Refusing.")
    fp = str(plain_fp)
    for i, link in enumerate(chain):
        if link["fit_fingerprint"] != fp:
            raise ContinuationError(
                f"continuation chain broken at link {i}: it continued fit "
                f"{link['fit_fingerprint'][:16]}…, but the links before it "
                f"(and this config) resolve to {fp[:16]}… — the publication "
                "does not descend from this identity. Refusing.")
        fp = str(fp_of(link))
    if fp != key:
        raise ContinuationError(
            f"continuation chain of {len(chain)} link(s) resolves to "
            f"{fp[:16]}…, but the checkpoint was written for "
            f"{key[:16]}… — the checkpoint does not belong to the fit the "
            "publication's chain describes. Refusing.")
    last = chain[-1]
    return {"chain": chain, "depth": len(chain), "root_fit_fingerprint":
            str(plain_fp), "prior_cap": int(last["epochs_cap"]),
            "prior_lr_mult": marker_lr_mult(last),
            "prior_stopper_rule": marker_stopper_rule(last)}


def check_cap_admissible(*, chain: dict, summary: dict, new_cap,
                         lr_mult=1.0, config_cap,
                         stopper_rule=STOPPER_RULE_REPLAY_EXEMPT) -> dict:
    """The cap / multiplier / tail rules a continuation must satisfy — ONE
    helper for the driver's ``_prepare_continuation`` and the Modal app's
    pre-spawn derivation (``expected_continuation``: fingerprint_cpu and
    snapshot), so nothing the driver would refuse after model load is
    snapshotted or spawned. ``chain`` is
    :func:`resolve_chain`'s result; ``config_cap`` the SI's configured cap.

    - the prior cap is the config's for a first continuation, the last
      chain link's ``epochs_cap``.
    - a cap BELOW it refuses (only the checkpointed epoch's tail would run
      and the same fit republish).
    - an EQUAL cap refuses unless ``lr_mult`` differs from the prior
      link's (the same cap and multiplier is the same fit; a different
      multiplier is a different fit and the re-issue of a continuation
      whose warm restart stopped it at once) OR — for a CHAINED
      continuation — ``stopper_rule`` differs from the prior link's (the
      re-issue of a continuation that a replayed-epoch strike stopped,
      under the rule that does not count it, is a different fit; a first
      continuation has no prior rule to differ from).
    - an equal cap ALSO refuses when the checkpoint sits at the END of the
      cap's final epoch (``epoch >= cap - 1`` and ``batches_done >=
      n_batches``): the trainer would skip the whole epoch, run no
      optimizer step and publish an untouched object.
    - under ``replay_exempt`` an equal cap refuses whenever the checkpoint
      sits INSIDE the cap's final epoch (``epoch >= cap - 1``): the only
      epoch that would run is the replayed one, which the rule does not
      judge — no verdict could change the object.

    Returns ``{"prior_cap", "prior_lr_mult", "prior_stopper_rule",
    "stopper_rule", "which"}``."""
    cap = int(new_cap)
    m = check_lr_mult(lr_mult)
    rule = check_stopper_rule(stopper_rule)
    first = chain["prior_cap"] is None
    prior_cap = int(config_cap) if first else int(chain["prior_cap"])
    prior_m = check_lr_mult(chain["prior_lr_mult"])
    prior_rule = (None if first
                  else check_stopper_rule(chain.get("prior_stopper_rule")))
    which = ("the config's" if first
             else f"chain link {chain['depth']}'s epochs_cap")
    if cap < prior_cap:
        raise ContinuationError(
            f"--epochs-cap {cap} is below the prior cap {prior_cap} "
            f"({which}) — a continuation to a lower cap would only re-run "
            "the checkpointed epoch's tail and republish the same fit")
    rule_differs = (not first) and rule != prior_rule
    if cap == prior_cap and m == prior_m and not rule_differs:
        raise ContinuationError(
            f"--epochs-cap {cap} must exceed the prior cap {prior_cap} "
            f"({which}) unless --continue-lr-mult differs from the prior "
            f"fit's ({prior_m}) or, for a chained continuation, the stopper "
            f"rule differs from the prior link's ({prior_rule!r}; this "
            f"fit's: {rule!r}) — the same cap, multiplier and rule would "
            "only re-run the checkpointed epoch's tail and republish the "
            "same fit")
    if cap == prior_cap and (int(summary["epoch"]) >= cap - 1
                             and int(summary["batches_done"])
                             >= int(summary["n_batches"])):
        raise ContinuationError(
            f"--epochs-cap {cap} equals the prior cap and the checkpoint "
            f"sits at the END of its final epoch (epoch index "
            f"{summary['epoch']}, micro-batch {summary['batches_done']}/"
            f"{summary['n_batches']}) — nothing would train under this cap; "
            "a same-cap re-issue needs a mid-epoch checkpoint, else raise "
            "the cap")
    if (rule == STOPPER_RULE_REPLAY_EXEMPT and cap == prior_cap
            and int(summary["epoch"]) >= cap - 1):
        raise ContinuationError(
            f"--epochs-cap {cap} equals the prior cap and the checkpoint "
            f"sits inside the cap's final epoch (epoch index "
            f"{summary['epoch']}, micro-batch {summary['batches_done']}/"
            f"{summary['n_batches']}) — under the {rule!r} stopper rule the "
            "replayed remainder of that epoch is logged, not ruled on, so no "
            "epoch would be judged under this cap; raise the cap")
    return {"prior_cap": prior_cap, "prior_lr_mult": prior_m,
            "prior_stopper_rule": prior_rule, "stopper_rule": rule,
            "which": which}


# --------------------------------------------------------------------------- #
# Binding the checkpoint to the PUBLISHED cell                                 #
# --------------------------------------------------------------------------- #


def _leaves(obj, out: list) -> None:
    if isinstance(obj, torch.Tensor):
        out.append(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            _leaves(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _leaves(v, out)


def _tensor_multiset(tensors) -> list[tuple]:
    """Sorted (shape, sha256 of float32 bytes) signatures — order-free
    equality of two tensor sets, compared in float32 so an fp32 artifact
    written from the same values as a checkpointed tensor compares equal."""
    sigs = []
    for t in tensors:
        f = t.detach().float().cpu().contiguous()
        sigs.append((tuple(f.shape),
                     hashlib.sha256(f.numpy().tobytes()).hexdigest()))
    return sorted(sigs)


def _read_json(path: Path, problems: list[str], what: str):
    if not path.is_file():
        problems.append(f"{what} {path} missing")
        return None
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as e:
        problems.append(f"{what} {path} does not parse: {e}")
        return None
    if not isinstance(doc, dict):
        problems.append(f"{what} {path} is not a JSON object")
        return None
    return doc


def verify_cell(cell_dir, state: dict, *,
                expect_revision: str | None = None) -> dict:
    """Does this checkpoint belong to the cell's CURRENT published object?
    Returns ``{"problems": [...], "checks": {...}}`` — empty problems means
    proven; ``checks`` is the record of what was compared (it rides into
    the continued cell's provenance). Checks:

    - the COMPLETE marker parses, names the cell, and its ``fit_fingerprint``
      EQUALS the checkpoint's ``checkpoint_key`` (the checkpoint was written
      by the fit that published this object); every manifested file exists
      with its recorded sha256.
    - the budget sidecar names the cell, its ``object_sha256`` equals the
      on-disk object and the marker's; a non-null recorded ``model_revision``
      equals ``expect_revision`` when given (model/revision equality is
      otherwise bound by the driver's prior-fingerprint check).
    - the checkpoint position lies INSIDE the recorded fit: ``step`` <=
      ``fit_record.optimizer_steps_run`` (a later step postdates the
      object), ``epoch`` < ``fit_record.epochs_run``, the stopper's best
      epoch <= the recorded best epoch.
    - whenever the stopper's best epoch == the recorded best epoch (the
      fit's best had already been seen at checkpoint time), the stopper's
      best-epoch snapshot must equal the published object's tensors — the
      published object IS the restored best snapshot, and it stays the
      continuation's incumbent (``checks["incumbent"] == "checkpoint_
      snapshot"``).
    - whenever the published best POSTDATES the checkpoint (a best found in
      the epochs after the retained write — the 1M PSR cells: checkpoint at
      epoch index 23, best 24), the checkpoint's stopper holds an OLDER
      best, and restoring it would let a continuation whose warm-restarted
      replay does not reproduce the later improvement ship an object WORSE
      than the one it continued from. The PUBLISHED object must then be
      seated as the incumbent best (``checks["incumbent"] ==
      "published_object"``, :func:`incumbent_
      stopper`), with its patience counter reconstructed from the published
      fit_record's FULL held-out curve — ``since_best`` = the epochs after
      the best in that curve, never the checkpoint's stale counter. A
      published record without a curve covering its epochs, or a curve that
      disagrees with its own best_epoch/best_heldout_ce, REFUSES. Epochs
      the record lists in ``replayed_partial_epochs`` (replayed partial
      epochs, logged but never ruled on) are not candidates for that
      minimum — a replayed value below the best is not a best the
      stopper ever seated.

    The cell's ``superseded_by.json`` (when present) is recorded, and whether
    it names the CURRENT publication (the snapshot that licenses the
    overwrite — enforced by the Modal fit's supersede gate, recorded here)."""
    cell_dir = Path(cell_dir)
    problems: list[str] = []
    checks: dict = {"cell_dir": str(cell_dir)}
    key = str(state["checkpoint_key"])
    step, epoch = int(state["step"]), int(state["epoch"])

    marker = _read_json(cell_dir / COMPLETE_NAME, problems, "COMPLETE marker")
    obj_file = "object.pt"
    if marker is not None:
        if marker.get("cell_id") != cell_dir.name:
            problems.append(
                f"COMPLETE marker names cell_id={marker.get('cell_id')!r}, "
                f"the dir is {cell_dir.name!r}")
        fp = marker.get("fit_fingerprint")
        if not fp:
            problems.append("COMPLETE marker carries no fit_fingerprint — "
                            "an unprovable publication cannot be continued")
        elif str(fp) != key:
            problems.append(
                f"checkpoint_key {key[:12]}… != the cell's COMPLETE "
                f"fit_fingerprint {str(fp)[:12]}… — this checkpoint was not "
                "written by the fit that published this object")
        else:
            checks["prior_fit_fingerprint"] = str(fp)
        files = marker.get("files")
        if not isinstance(files, dict) or not files:
            problems.append("COMPLETE marker carries no publication manifest "
                            "(files)")
        else:
            for name, sha in files.items():
                p = cell_dir / str(name)
                if not p.is_file():
                    problems.append(f"manifested file {name} missing")
                elif _sha256_file(p) != sha:
                    problems.append(
                        f"manifested file {name} does not match its "
                        "published sha256 — the publication was modified")
        obj_file = str(marker.get("object_file") or obj_file)
        checks["marker_object_sha256"] = marker.get("object_sha256")

    obj_path = cell_dir / obj_file
    obj_sha = None
    if obj_path.is_file():
        obj_sha = _sha256_file(obj_path)
        checks["object_file"] = obj_file
        checks["object_sha256"] = obj_sha
        m_sha = checks.get("marker_object_sha256")
        if m_sha is not None and str(m_sha) != obj_sha:
            problems.append("COMPLETE object_sha256 != the on-disk object")
    else:
        problems.append(f"object artifact {obj_path} missing")

    sidecar = _read_json(cell_dir / SIDECAR_NAME, problems, "budget sidecar")
    fr = None
    if sidecar is not None:
        if sidecar.get("cell_id") != cell_dir.name:
            problems.append(
                f"sidecar records cell_id={sidecar.get('cell_id')!r}, the "
                f"dir is {cell_dir.name!r} — a wrong-cell artifact")
        s_sha = sidecar.get("object_sha256")
        if s_sha is None:
            problems.append("sidecar records no object_sha256")
        elif obj_sha is not None and str(s_sha) != obj_sha:
            problems.append(
                f"sidecar object_sha256 {str(s_sha)[:12]}… != on-disk "
                f"{obj_sha[:12]}… — a stale object/sidecar pair")
        checks["model_id"] = sidecar.get("model_id")
        checks["model_revision"] = sidecar.get("model_revision")
        rev = sidecar.get("model_revision")
        if (expect_revision is not None and rev is not None
                and str(rev) != str(expect_revision)):
            problems.append(
                f"sidecar model_revision {rev} != this fit's "
                f"{expect_revision}")
        fr = sidecar.get("fit_record")
        if not isinstance(fr, dict):
            problems.append("sidecar lacks a fit_record — the published "
                            "fit's epochs/steps cannot be checked against "
                            "the checkpoint position")
            fr = None

    st = state.get("stopper")
    if fr is not None:
        er, be, osr = (fr.get("epochs_run"), fr.get("best_epoch"),
                       fr.get("optimizer_steps_run"))
        checks.update({"prior_epochs_run": er, "prior_best_epoch": be,
                       "prior_optimizer_steps_run": osr,
                       "prior_stopped_early": fr.get("stopped_early"),
                       "prior_best_heldout_ce": fr.get("best_heldout_ce")})
        if isinstance(osr, int) and step > osr:
            problems.append(
                f"checkpoint step {step} > the published fit's "
                f"optimizer_steps_run {osr} — the checkpoint postdates the "
                "object")
        if isinstance(er, int) and epoch >= er:
            problems.append(
                f"checkpoint epoch index {epoch} is not inside the published "
                f"fit's {er} completed epochs")
        if st is not None and isinstance(be, int):
            sbe = int(st["best_epoch"])
            if sbe > be:
                problems.append(
                    f"checkpoint's best epoch {sbe} exceeds the published "
                    f"fit's best epoch {be} — not this fit's checkpoint")
            elif (sbe == be and st.get("snapshot") is not None
                    and obj_path.is_file()):
                try:
                    blob = torch.load(str(obj_path), map_location="cpu",
                                      weights_only=True)
                except Exception as e:  # noqa: BLE001
                    problems.append(f"object artifact unreadable: {e}")
                else:
                    leaves: list = []
                    _leaves(blob, leaves)
                    match = (_tensor_multiset(st["snapshot"])
                             == _tensor_multiset(leaves))
                    checks["object_matches_checkpoint_best_snapshot"] = match
                    checks["incumbent"] = "checkpoint_snapshot"
                    if not match:
                        problems.append(
                            "the checkpoint's best-epoch snapshot (its best "
                            f"epoch {sbe} == the published best epoch) does "
                            "not equal the published object's tensors — the "
                            "checkpoint does not belong to this object")
            elif sbe < be:
                # The published best postdates the checkpoint: the
                # published object is the incumbent, its patience counter
                # comes from the published curve (docstring).
                checks["object_matches_checkpoint_best_snapshot"] = None
                checks["object_match_note"] = (
                    "inapplicable: the published best epoch came after this "
                    "checkpoint — the published object is seated as the "
                    "incumbent best")
                curve = fr.get("heldout_curve")
                bce = fr.get("best_heldout_ce")
                per_epoch = (list(curve[1:]) if isinstance(curve, list)
                             else None)
                # Replayed partial epochs were never ruled on, so they
                # cannot be the published best whatever their value.
                replayed = {int(e) for e in
                            (fr.get("replayed_partial_epochs") or [])}
                checks["published_replayed_partial_epochs"] = sorted(replayed)
                judged = (None if per_epoch is None else
                          [(float(v), i + 1) for i, v in enumerate(per_epoch)
                           if (i + 1) not in replayed])
                if (per_epoch is None or not isinstance(er, int)
                        or len(per_epoch) != er):
                    problems.append(
                        "the published fit_record's held-out curve is "
                        "missing or does not cover its epochs_run — the "
                        "incumbent best (the published object, best epoch "
                        f"{be} > the checkpoint's {sbe}) cannot be seated "
                        "with a reconstructed patience counter; refusing")
                elif (not isinstance(bce, (int, float))
                        or isinstance(bce, bool)
                        or not math.isfinite(float(bce))):
                    problems.append(
                        "the published fit_record carries no finite "
                        "best_heldout_ce — the incumbent best value cannot "
                        "be seated; refusing")
                elif (not judged or be in replayed
                        or min(judged)[1] != be
                        or not math.isclose(float(per_epoch[be - 1]),
                                            float(bce), rel_tol=1e-9,
                                            abs_tol=1e-12)):
                    problems.append(
                        f"the published held-out curve's minimum over the "
                        f"judged epochs (replayed partial epochs "
                        f"{sorted(replayed)} excluded) is not at best_epoch "
                        f"{be} with value best_heldout_ce {bce} — a "
                        "self-inconsistent fit_record; refusing")
                else:
                    checks["incumbent"] = "published_object"
                    checks["incumbent_best_epoch"] = int(be)
                    checks["incumbent_best_heldout_ce"] = float(bce)
                    # strikes the published run recorded AFTER its best
                    checks["incumbent_since_best"] = int(er) - int(be)
                    checks["published_curve_len"] = len(curve)
            else:
                checks["object_matches_checkpoint_best_snapshot"] = None
                checks["object_match_note"] = (
                    "inapplicable: no best-epoch snapshot in the checkpoint")

    sup_path = cell_dir / SUPERSEDE_NAME
    if sup_path.is_file():
        try:
            sup = json.loads(sup_path.read_text(encoding="utf-8"))
        except ValueError:
            sup = None
        if isinstance(sup, dict):
            checks["superseded_by"] = {
                "fit_fingerprint": sup.get("fit_fingerprint"),
                "snapshot": sup.get("snapshot"),
                "licenses_current_publication": (
                    str(sup.get("fit_fingerprint")) == key),
            }
    return {"problems": problems, "checks": checks}


# --------------------------------------------------------------------------- #
# Adoption                                                                     #
# --------------------------------------------------------------------------- #


def incumbent_stopper(state: dict, *, published_trainable, best_ce: float,
                      best_epoch: int, since_best: int) -> dict:
    """The stopper state a continuation ADOPTS when the published best
    postdates the checkpoint: the PUBLISHED object's trainable
    tensors — in the object's own ``.trainable`` order, cast to the
    checkpoint's dtypes — as the best-epoch snapshot, the published
    ``best_heldout_ce`` as the value to beat, its epoch as ``best_epoch``,
    ``since_best`` reconstructed from the published curve, and the
    checkpoint's own ``epochs_seen`` (the replayed epochs count from
    there). Shapes must match the checkpoint's trainables one-to-one."""
    st = state.get("stopper")
    if st is None:
        raise ContinuationError("the checkpoint carries no stopper state — "
                                "no incumbent can be seated")
    ref = state["trainable"]
    pub = [t.detach().cpu() for t in published_trainable]
    if len(pub) != len(ref) or any(tuple(a.shape) != tuple(b.shape)
                                   for a, b in zip(pub, ref)):
        raise ContinuationError(
            "the published object's trainable tensors do not match the "
            "checkpoint's (count or shapes) — cannot seat it as the "
            "incumbent best")
    return {"best": float(best_ce), "best_epoch": int(best_epoch),
            "epochs_seen": int(st["epochs_seen"]),
            "since_best": int(since_best),
            "snapshot": [p.to(dtype=r.dtype).clone()
                         for p, r in zip(pub, ref)]}


def adopt(state: dict, *, new_key: str, dest_dir, marker: dict,
          source=None, stopper: dict | None = None) -> Path:
    """Write the prior fit's checkpoint into the CONTINUED fit's transient
    slot under the continued fit's identity — ``checkpoint_key`` := the
    continuation fingerprint, plus a ``continued_from`` field naming the
    prior key and resume position — through ``fitckpt.save_checkpoint``
    (atomic; rotates any foreign latest to the previous slot). The trainer's
    own ``load_checkpoint(dest_dir, new_key, expect=...)`` then resumes it
    exactly as it would its own mid-fit checkpoint. The adopted key must
    differ from the prior one: an identity the cap raise did not move would
    let the COMPLETE skip count the old publication as this work.
    ``stopper`` (:func:`incumbent_stopper`) replaces the checkpoint's
    early-stopping state when the published object is the incumbent."""
    if str(new_key) == str(state["checkpoint_key"]):
        raise ContinuationError(
            "the continuation fingerprint equals the prior fit's — the cap "
            "raise did not change the fit identity; refusing to adopt")
    adopted = dict(state)
    adopted["checkpoint_key"] = str(new_key)
    cf = dict(marker)
    if source is not None:
        cf["source"] = str(source)
    if stopper is not None:
        adopted["stopper"] = dict(stopper)
        cf["incumbent"] = {"source": "published_object",
                           "best_epoch": int(stopper["best_epoch"]),
                           "best": float(stopper["best"]),
                           "since_best": int(stopper["since_best"])}
    else:
        cf["incumbent"] = {"source": "checkpoint_snapshot"}
    adopted["continued_from"] = cf
    return save_checkpoint(dest_dir, adopted)


# --------------------------------------------------------------------------- #
# The lr schedule, on the record                                               #
# --------------------------------------------------------------------------- #


def _warmup_for(total: int, warmup_frac, warmup_steps) -> int:
    from .lockstep import resolve_warmup_steps

    if warmup_frac is not None:
        return int(resolve_warmup_steps(float(warmup_frac), int(total)))
    if warmup_steps is not None:
        return int(warmup_steps)
    # train_object_ce's own default when nothing is configured
    return min(500, max(1, int(total) // 10))


def schedule_report(*, lr: float, lr_schedule: str | None, warmup_frac,
                    n_batches: int, accum_steps: int, prior_cap: int,
                    new_cap: int, resume_step: int,
                    warmup_steps: int | None = None,
                    warmup_steps_new: int | None = None,
                    lr_mult: float = 1.0,
                    prior_lr_mult: float = 1.0) -> dict:
    """What the continuation's lr does at the resume step, under the prior
    horizon (the schedule the checkpoint was trained under) and the new one
    (what the continuation runs) — the trainer's exact arithmetic
    (``total = cap x ceil(n_batches / accum_steps)``, lockstep's
    ``_lr_at_factory``), so the recorded numbers are the numbers the
    optimizer sees. ``warmup_steps_new`` is the driver's already-resolved
    value for the new horizon; it must equal this arithmetic (else the
    report would describe a schedule other than the one running).
    ``lr_mult`` (gap 2) scales the NEW horizon's peak (``lr`` is the
    config's peak; the continuation trains at ``lr x lr_mult``);
    ``prior_lr_mult`` is the multiplier the checkpoint's own fit ran under
    (1.0 for a first-generation checkpoint), so ``lr_at_resume_prior_
    horizon`` is the rate the checkpoint really trained at."""
    from .lockstep import _lr_at_factory

    m_new, m_prior = check_lr_mult(lr_mult), check_lr_mult(prior_lr_mult)
    sched = lr_schedule or "warmup_cosine"
    spe = math.ceil(int(n_batches) / max(int(accum_steps), 1))
    prior_total, new_total = int(prior_cap) * spe, int(new_cap) * spe
    w_prior = _warmup_for(prior_total, warmup_frac, warmup_steps)
    w_new = _warmup_for(new_total, warmup_frac, warmup_steps)
    if warmup_steps_new is not None and int(warmup_steps_new) != w_new:
        raise ContinuationError(
            f"the driver resolved warmup_steps={warmup_steps_new} for the "
            f"new horizon but the trainer arithmetic gives {w_new} — the "
            "schedule report would not describe the schedule that runs")
    peak_prior, peak_new = float(lr) * m_prior, float(lr) * m_new
    lr_prior = _lr_at_factory(peak_prior, sched, w_prior, prior_total)(
        int(resume_step))
    lr_new = _lr_at_factory(peak_new, sched, w_new, new_total)(
        int(resume_step))
    return {
        "horizon": "new_cap",
        "lr_schedule": sched,
        "peak_lr": float(lr),
        "lr_mult": m_new,
        "effective_peak_lr": peak_new,
        "prior_lr_mult": m_prior,
        "prior_effective_peak_lr": peak_prior,
        "steps_per_epoch": spe,
        "prior_total_steps": prior_total,
        "prior_warmup_steps": w_prior,
        "new_total_steps": new_total,
        "new_warmup_steps": w_new,
        "resume_step": int(resume_step),
        "lr_at_resume_prior_horizon": float(lr_prior),
        "lr_at_resume_new_horizon": float(lr_new),
        "lr_at_resume_fraction_of_peak": (float(lr_new) / float(lr)
                                          if lr else None),
        "lr_jump_fraction_of_peak": (float(lr_new) - float(lr_prior))
        / float(lr) if lr else None,
        "note": ("the continuation runs the cap-N schedule from the resume "
                 "step onward (the trainer's own total_steps arithmetic) at "
                 "peak lr x lr_mult; under warmup_cosine this is a warm "
                 "restart at the resume step, quantified above — a no-jump "
                 "continuation of a fully decayed cosine would train at ~0 "
                 "lr"),
    }


def outcome_record(record: dict, diagnostics: dict) -> dict:
    """The provenance ``continued_from`` block once the continued fit has
    run: the pre-fit record (prior fingerprint/object/cap, resume position,
    new cap, identity checks, lr schedule report) plus what happened —
    epochs actually run, the stop reason, best epoch, steps, and the
    trainer's own word that it resumed (``resumed_from_step``)."""
    rec = dict(record)
    er = diagnostics.get("epochs_run")
    rec.update({
        "epochs_run_total": er,
        # the resume epoch is re-run from its checkpointed position, so it
        # counts here as one of the continuation's epochs
        "epochs_from_resume_epoch_inclusive": (
            None if er is None else int(er) - int(record["resume_epoch"])),
        "stop_reason": ("early_stop" if diagnostics.get("stopped_early")
                        else "cap"),
        "best_epoch": diagnostics.get("best_epoch"),
        "best_heldout_ce": diagnostics.get("best_heldout_ce"),
        "optimizer_steps_run": diagnostics.get("optimizer_steps_run"),
        "resumed_from_step": diagnostics.get("resumed_from_step"),
        # The 1-based epochs whose held-out value was a replayed partial
        # epoch's (logged, flagged, never ruled on)
        "replayed_partial_epochs": list(
            diagnostics.get("replayed_partial_epochs") or []),
    })
    return rec
