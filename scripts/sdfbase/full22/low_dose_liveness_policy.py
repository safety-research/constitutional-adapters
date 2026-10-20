"""LOW-DOSE NUMERIC LIVENESS — the policy the steer-is-live refusal branches consult.

At a small enough dose a CORRECT steering object changes no greedy token on a neutral probe (argmax is a step function; a
small logit perturbation crosses no boundary), so `fidelity._check_steering_bites` — greedy TEXT divergence against the
scale-0 twin, refusing when NONE of five neutral probes differs — reports "not live" when steering is live and merely
SMALL. Taken as a stop, that would stop the low end of every dose ladder for being low, which is where the usable window
is expected to be. Measured on the strength frontier: AlphaSteer's realised dose at nominal 0.1 is 0.016 and the LoRA
lane's is 0.031 at the same nominal, so this is a LOW-DOSE problem, not one method's.

At low dose liveness is established NUMERICALLY — the intervention demonstrably distinct from the anchor's, and a
measured non-zero effect at the served site above the engine's own repeat noise — and a text-divergence refusal ALONE
does not stop a cell. This module is the seam: a caller that is about to refuse on text divergence asks whether a
numeric record ADMITS this cell, and records the text result as an observation instead of aborting. A cell failing a
numeric check is genuinely not live and still refuses.

Deliberately conservative, because it can only ever turn a refusal into a pass:
  * OFF unless the env var names a file — with it unset, every caller behaves byte-identically to before;
  * the record must read `verdict: "live_numeric"`, name THIS cell, and name THIS routing table (so it is bound to the
    standup that produced it — a record from another standup, or a stale one, admits nothing);
  * anything unreadable, malformed or mismatched admits NOTHING and says why (never a silent pass).

The writers are scripts/alphasteer_f22/screen/asscr_liveness.py (branch strength-asscreen) and, for the sampled ladder's
LoRA / derived cells, scripts/sdfbase/full22/sampled/lora_liveness.py — both measure the two facts with the instrument's
OWN rule (`fidelity.check_logprob_steering_bites`).

Three refusal branches consult this module. The controls driver (scripts/served_capability_run.py) and the standup smoke
(smoke_positive_control.py) catch the FidelityError themselves and call `consult` directly. The STATIC path cannot: there
the refusal is raised inside src/ (served_target.build_served_target -> served_evidence -> bon_served.assert_steering_bites
-> fidelity._check_steering_bites) and nothing in scripts/ sees it before the driver dies — cluster B 100081, 2 of 108
static drivers, `steering_bites: arm(s) ['lora_r4_ce_minus_ctl'] produce NO difference` at nominal 0.1. `SteeringBitesGuard`
is that path's seam: a scripts-level wrapper installed around `build_served_target` by jailbreak_run.py and
escalate_static.py, inert with the env unset.
"""
from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

POLICY_ENV = "SISTEER_LOW_DOSE_LIVENESS_RECORD"
REQUIRED_VERDICT = "live_numeric"


def _same_path(a, b) -> bool:
    try:
        return Path(str(a)).resolve() == Path(str(b)).resolve()
    except Exception:  # noqa: BLE001
        return str(a) == str(b)


def consult(cell_id: str, routing_table, *, env: Optional[Dict[str, str]] = None) -> Tuple[Optional[Dict[str, Any]], str]:
    """(record, why) — the admitting record, or None with the reason it does not admit. Never raises."""
    environ = os.environ if env is None else env
    path = (environ.get(POLICY_ENV) or "").strip()
    if not path:
        return None, f"{POLICY_ENV} is not set (the text-divergence gate is unchanged)"
    p = Path(path)
    if not p.is_file():
        return None, f"{POLICY_ENV}={path} does not exist — admitting nothing"
    try:
        rec = json.loads(p.read_text())
    except Exception as e:  # noqa: BLE001
        return None, f"{POLICY_ENV}={path} is unreadable ({type(e).__name__}: {e}) — admitting nothing"
    if not isinstance(rec, dict):
        return None, f"{POLICY_ENV}={path} is not a JSON object — admitting nothing"
    if rec.get("verdict") != REQUIRED_VERDICT:
        return None, f"{path}: verdict {rec.get('verdict')!r} != {REQUIRED_VERDICT!r} — admitting nothing"
    if rec.get("cell") != cell_id:
        return None, f"{path}: names cell {rec.get('cell')!r}, not {cell_id!r} — admitting nothing"
    if not _same_path(rec.get("routing_table"), routing_table):
        return None, (f"{path}: names routing table {rec.get('routing_table')!r}, not {str(routing_table)!r} — a record from "
                      "another standup admits nothing")
    return rec, (f"{path}: numeric liveness ADMITS {cell_id} (intervention distinct + measured effect above the engine's own "
                 "repeat noise); the text-divergence result is an OBSERVATION, not a gate (D87)")


def summary(rec: Dict[str, Any]) -> Dict[str, Any]:
    """The few fields a consumer records beside its own evidence."""
    ce = (rec.get("continuous_effect") or {}).get("diag") or {}
    return {"policy": "D87_low_dose_numeric_liveness", "record_verdict": rec.get("verdict"), "basis": rec.get("basis"),
            "nominal_dose": rec.get("nominal_dose"), "tag": rec.get("tag"),
            "steered_divergence_median": ce.get("steered_divergence_median"), "noise_ceiling": ce.get("noise_ceiling"),
            "bar": ce.get("bar"),
            "text_divergence_observation": {k: v for k, v in (rec.get("text_divergence_observation") or {}).items()
                                            if k in ("n_probes", "n_differing")},
            "realised_dose": (rec.get("realised_dose") or {}).get("available"),
            "written_at": rec.get("written_at")}


# --------------------------------------------------------------------------- #
# The STATIC path: the refusal is raised inside src/ and no scripts/ code sees it before it kills the driver.         #
# --------------------------------------------------------------------------- #

class SteeringBitesGuard:
    """A scripts-level wrapper around ``bon_served.assert_steering_bites`` for the entry points that reach the
    text-divergence refusal through ``served_target.build_served_target`` — ``jailbreak_run.py generate`` (the static
    attack drivers) and ``escalate_static.py`` (the static rungs) — where, unlike the controls driver and the smoke, the
    refusal is raised INSIDE src/ (``served_evidence`` -> ``bs.assert_steering_bites`` -> ``fidelity._check_steering_bites``)
    and nothing in scripts/ sees it before the driver dies.

    Installed only for the duration of ``build_served_target`` (``with guard.installed(): …``) and ONLY when the policy env
    names a record: with it unset the context patches nothing, so the driver is byte-identical to before. Installed, it
    lets the original run unchanged and acts on exactly one exception: a ``FidelityError`` whose message starts with
    ``steering_bites:`` — the text-divergence refusal itself (the malformed-evidence refusals read
    ``steering_bites arm <name>: …`` and every other fidelity check carries its own prefix). It then consults the policy
    for THIS cell and THIS routing table; when admitted, it re-reads the probe evidence it captured through the
    instrument's own ``_check_steering_bites`` (which fills the per-arm diagnostic before it raises), returns that
    diagnostic so ``served_evidence`` records the probe as it would a passing one (``n_differing`` 0, ``verified`` True),
    and keeps the admission for the caller to fold beside that evidence (``fold_into`` — served_target.json carries it).
    When not admitted it re-raises the very same exception. src/ is never touched: the wrapper sits on the module
    attribute ``served_evidence`` resolves at call time."""

    REFUSAL_PREFIX = "steering_bites:"

    def __init__(self, cell_id: str, routing_table, *, env: Optional[Dict[str, str]] = None,
                 log: Optional[Callable[[str], None]] = None):
        self.cell_id = cell_id
        self.routing_table = routing_table
        self._env = env
        self._log = log or (lambda m: print(m, flush=True))
        #: the fields a caller folds into its steering_bites evidence once a refusal was lifted (None: nothing lifted)
        self.admission: Optional[Dict[str, Any]] = None
        self.installed_now = False

    def active(self) -> bool:
        """True iff the policy env names a record — the only state in which the context installs anything."""
        environ = os.environ if self._env is None else self._env
        return bool((environ.get(POLICY_ENV) or "").strip())

    @contextlib.contextmanager
    def installed(self, bon_served=None):
        if not self.active():
            yield self
            return
        from sisteer.capeval.serving import fidelity as _fid
        bs = bon_served
        if bs is None:
            from sisteer.eval.attacks import bon_served as bs
        original = bs.assert_steering_bites
        guard = self

        def guarded_assert_steering_bites(arms, probe):
            captured: Dict[str, Any] = {"arms": {}}

            def capturing_probe(arm):
                steered_ids, scale0_ids = probe(arm)
                captured["arms"][arm] = {"steered_ids": steered_ids, "scale0_ids": scale0_ids}
                return steered_ids, scale0_ids

            try:
                return original(arms, capturing_probe)
            except _fid.FidelityError as e:
                msg = str(e)
                if not msg.startswith(guard.REFUSAL_PREFIX):
                    raise
                rec, why = consult(guard.cell_id, guard.routing_table, env=guard._env)
                if rec is None:
                    guard._log(f"text-divergence refusal stands: {why}")
                    raise
                diag: Dict[str, Any] = {}
                try:
                    _fid._check_steering_bites(captured, diag)
                except _fid.FidelityError:
                    pass  # the per-arm diagnostic is filled BEFORE the refusal is raised
                if not diag.get("arms"):
                    guard._log("text-divergence refusal stands: the captured probe evidence could not be re-read as a "
                               f"diagnostic, so this is not a refusal numeric liveness can lift ({msg[:300]})")
                    raise
                guard.admission = {"text_divergence_is_observation": True, "text_divergence_refusal": msg[:800],
                                   "low_dose_liveness": summary(rec), "liveness_basis": "numeric (D87)", "policy": why}
                guard._log("LOW-DOSE LIVENESS (D87): the steered route produced no greedy-text divergence from its scale-0 "
                           f"twin, and that is an OBSERVATION here, not a refusal — {why}. Recorded beside the served "
                           "evidence (served_target.json); the target is built and generation proceeds.")
                return diag

        bs.assert_steering_bites = guarded_assert_steering_bites
        self.installed_now = True
        try:
            yield self
        finally:
            bs.assert_steering_bites = original
            self.installed_now = False

    def fold_into(self, evidence: Optional[Dict[str, Any]]) -> None:
        """Record a lifted refusal beside the target's own steering_bites evidence. A no-op when nothing was lifted, so
        an untouched evidence dict stays byte-identical."""
        if self.admission is None or evidence is None:
            return
        evidence.setdefault("steering_bites", {}).update(self.admission)
