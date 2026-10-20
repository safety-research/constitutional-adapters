#!/usr/bin/env python3
"""The COMPATIBILITY PREFLIGHT of the eight-arm MATH-500 extension (math-eight-arms-extension plan P11; P5 (i) /
(i') / (i''); decisions-impl-laneA A8, A9): BEFORE a new leg launches, compare what the WAVE TREE will serve for each
new row against the PUBLISHED `none` leg's record of the same model — so a cross-tree contrast (a new leg vs the
published `none` at tree 55e3cbdd) is known compatible before any GPU minute, and a model whose served identity moved
between the trees has its rows HELD (never repaired by re-running the published `none`).

Per extension model, three relations, each a NAMED field or a named refusal:

  (i)  the SHARED-FIELD PROJECTION, equal across every arm of a contrast — the task + its item population (the 500
       MATH-500 ids the wave tree plans, offline, vs the ids the published `none` leg ATTEMPTED — its outcomes record,
       never the scored rows), the decode parameters
       as the NUMERIC projection (`cells.sampling_params`: SAMPLING_PARAM_KEYS only, absent keys absent on both
       sides — the matrix dict's `source` prose never enters), the model id + revision, the thinking policy (mode +
       the enable flag), the identity version, the cap statement, the chat template digest (the pinned tokenizer's
       template at the wave tree vs the published leg's), the inspect pins the client lock carries, and the CLIENT LOCK
       compared by WHAT THE MATH-500 CLIENT IMPORTS (`sisteer.capeval.tinycap.client_lock`; decisions-w4-launch L1 G0):
       the published lock TEXT is read off the published tree (`git show <producer tree>:<lock>`) and its digest must
       equal the digest the leg recorded (chain of custody); versions are compared for exactly the distributions an
       AST walk from the client venv's MATH-500 entry points imports and both locks carry — a differing version refuses
       by name; a distribution added to / removed from the lock, imported or not, is DISCLOSED (one observed case: three
       distributions the published client never had joined the lock and moved the digest with no version change —
       one instrument), never refused; the digest pair is recorded, not a gate by itself;
  (i') each NEW ROW's OWN contract, validated for what the row is — never compared with `none`: `none` /
       `prompted__<si>` / lens by grammar; a prompted row's SI digest == the overlay manifest's
       `prompted_prefix_hashes[<si>]` == sha256 of the registry's text; a lens row's pin == the pins file's staged
       object and the roster's `main` view stages it; the row's geometry entry resolves to the row's L at the model's
       budget (S recorded); the `_lora2` tuple exists for the row's (model, cell); optionally (--budgets-root) the
       budget artifact's fit sidecar at the wave sha (fits, entry, window, arm, tree, SI);
  (i'') the NORMALISED server settings: the launch `vllm_server.build_server_launch` COMPOSES at the wave tree over the
       overlay manifest + the pinned template UNDER THE JOB'S OWN LAUNCH CONFIGURATION — read off the launcher's dry
       run for the row (`STANDUP:` extras + `STANDUP_POLICY:`: the standup mode, whether `--thinking-on` rides it — it
       never does on this lane, the sbatch refuses F22_THINKING_ON — the parser, the memory utilisation, the opt-ins;
       fix A4) and spliced with mt_serving_up's own `launch_cmd_extras`, then projected onto its behaviour-affecting
       flags; COMPARED where the published record carries the field (template digest, reasoning parser, the render
       regime — the REQUEST-level enable flag + header ids — the reduction mode) plus the SERVER-DEFAULT thinking
       policy, which the manifest does not record and is read off the published TREE's launcher (`git show`; UNDETERMINED
       when unreadable) — the two thinking fields are kept apart: the request flag decides the render (vLLM merges the
       request's chat_template_kwargs over the server default), the server default is what a flag-less request would
       get; COMPOSED-ONLY where the record carries nothing (dtype, quantisation, prefix caching, the template-kwargs pin,
       the generation-config override, the Mamba cache dtype, the memory utilisation) — coverage is STATED, never
       implied; the two INTENTIONAL differences — the row's geometry (checked in (i')) and `--enable-log-requests`
       (logging only) — are excluded by name. The generation-config override's EFFECT is checked through the field it
       lands on: every SAMPLER KEY the row OMITS is resolved under BOTH trees' rules (published: the snapshot's
       generation_config.json if it ships one, else vLLM's neutral default; wave: the pinned override row if the
       contracts pin one for the model, else the same) and any difference refuses.

What the published record does NOT carry is listed under `not_carried_by_published_record` (the published pull has
no serving/ directory — no endpoint.json, no vLLM log; its rows predate the `sampling_ref` field), so the reader
knows what was compared and what was composed.

Output: `<out>/compat_<key>.json` per model (schema m5t_compat_preflight/1) with every field's published / wave
values and `equal`, the problems (each naming the field), the undetermined checks (each naming the key and the remedy),
and a verdict PASS | UNDETERMINED | FAIL; stdout one line per model.
THE CERTIFICATE IS BOUND TO THE TREE IT READ (fix A2): every compared or composed value comes from this checkout, so
`--wave-sha` must BE its HEAD (`git rev-parse HEAD`, resolved independently and recorded under `tree`) and the evidence
roots (`src/ scripts/ configs/`) must carry no uncommitted tracked change — otherwise the preflight refuses (rc 2)
before writing anything; run it from a detached checkout at the wave sha. Changes outside the roots are recorded.

Exit 0 = every model PASS; 1 = a FAIL (that model's rows are HELD — record the decision, never re-run `none`);
2 = a refusal (an input the preflight needs is missing: the published manifest / outcomes, the datasets cache, the
tokenizer; or the tree is not the wave sha / is dirty under the evidence roots); 3 = UNDETERMINED (a rule this machine cannot READ — e.g. the snapshot's generation_config.json is not in
the local HF cache, so an omitted key's published resolution cannot be told from vLLM's neutral default; the remedy
is printed; never read as PASS). A SKIPPED check is UNDETERMINED too (fix A1): `--no-plan-items` leaves the population
uncompared and `--no-compose` leaves the server settings and every omitted key's wave rule unread, so either switch
yields rc 3 and a partial record that is never launch-eligible — the published ids are never echoed as the wave's and
the published sampler rule is never reused as the wave's.

Usage (a CPU host with the local HF cache, offline):
  PYTHONPATH=src:scripts python scripts/sdfbase/math500/m5t_compat_preflight.py \\
      --published-root <pull_55e3cbd>/controls --wave-sha <40hex> --out <wave root>/preflight \\
      [--model-key K ...] [--wave ext1] [--budgets-root <wave root>/budgets] [--no-compose] [--no-plan-items]
"""
from __future__ import annotations

import argparse
import dataclasses
import glob
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
for p in (REPO / "src", REPO / "scripts", REPO / "scripts" / "sdfbase", REPO / "scripts" / "sdfbase" / "math500"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import cells as M  # noqa: E402
from make_math500_overlays import roster_main_cells  # noqa: E402 — the ONE roster reader of this directory
from record_resolved_sampler import VLLM_NEUTRAL_DEFAULTS  # noqa: E402 — vLLM 0.25.1's neutral fallbacks for an OMITTED sampler key
from sisteer.capeval.tinycap import client_lock as CL  # noqa: E402 — the lock compared by what the client IMPORTS (L1 G0)
from sisteer.capeval.tinycap import contracts as C  # noqa: E402

SCHEMA = "m5t_compat_preflight/1"
PUBLISHED_TREE = "55e3cbdd0bfb5d53f8d62c50a099a3c1c105100b"
CLIENT_LOCK = REPO / C.CLIENT_LOCK_TXT
PINS = REPO / "configs/sdfbase_full22_pins.json"
#: the keys a generation_config.json can supply as a vLLM server default (what vLLM's get_diff_sampling_param reads) —
#: presence_penalty is NEVER filled from a file (the protocol default is 0.0)
FILE_DEFAULTABLE_KEYS = C.SERVER_DEFAULTABLE_SAMPLING_KEYS
#: the launch flags the projection reads off the composed argv (behaviour-affecting), and the two EXCLUDED as the
#: row-contracted / logging-only differences (P5 (i''))
PROJECTED_FLAGS = ("--dtype", "--quantization", "--revision", "--default-chat-template-kwargs", "--override-generation-config",
                   "--reasoning-parser", "--enable-prefix-caching", "--no-enable-prefix-caching", "--mamba-ssm-cache-dtype",
                   "--tokenizer", "--chat-template", "--max-logprobs")
EXCLUDED_FLAGS = {"--max-model-len": "the row's geometry (validated in (i') against the row, never compared)",
                  "--max-num-seqs": "the row's geometry (validated in (i') against the row, never compared)",
                  "--enable-log-requests": "logging only, never behaviour (P6's opt-in)"}
REDUCTION_MODE_STANDUP_LITERAL = "stacked"       # f22_standup: --reduction-mode stacked --require-attested-mode stacked
COMPOSE_PLACEHOLDER_PORT = 8000                  # the composer validates a port; the composed argv is projected, never launched


class Refusal(SystemExit):
    """An input the preflight NEEDS is missing — never a skipped field (rc 2)."""

    def __init__(self, why: str):
        super().__init__(2)
        self.why = why


# --------------------------------------------------------------------------- the tree the certificate names (fix A2)
#: every input the preflight reads comes from THIS checkout (REPO): the contracts, the geometry table, the composer, the
#: launcher's matrix, the overlays, the pins, the client lock. These are the roots that evidence lives under; a tracked
#: modification beneath any of them means the tree is not the commit `--wave-sha` names, and the certificate refuses.
EVIDENCE_ROOTS = ("src", "scripts", "configs")


def tree_facts(repo: Path, evidence_roots: tuple[str, ...] = EVIDENCE_ROOTS) -> dict:
    """The checkout's HEAD (independently resolved — `git rev-parse HEAD`, never the argv) and its uncommitted TRACKED
    changes, split into those under the evidence roots (refuse) and those outside them (recorded). A tree whose HEAD
    cannot be read is a refusal: the certificate can only name a tree it can identify."""
    def git(*args: str) -> str:
        r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, timeout=60)
        if r.returncode != 0:
            raise Refusal(f"git {' '.join(args)} failed in {repo} ({r.stderr.strip()[:200]}) — the preflight must run from the "
                          "git checkout whose HEAD is the wave sha; a copy that is not a checkout cannot be certified")
        return r.stdout
    head = git("rev-parse", "HEAD").strip()
    porcelain = [l for l in git("status", "--porcelain", "--untracked-files=no").splitlines() if l.strip()]
    paths = sorted({l[3:].split(" -> ")[-1] for l in porcelain})
    inside = [p for p in paths if any(p == r or p.startswith(r + "/") for r in evidence_roots)]
    return {"head": head, "evidence_roots": list(evidence_roots), "dirty_paths": inside,
            "dirty_paths_outside_evidence_roots": [p for p in paths if p not in inside],
            "how": "git rev-parse HEAD; git status --porcelain --untracked-files=no (tracked changes only)"}


def bind_wave_sha(wave_sha: str, facts: dict) -> dict:
    """The certificate's binding: `--wave-sha` must BE this checkout's HEAD and the evidence roots must be clean —
    otherwise the certificate would name a tree the launcher never runs (fix A2). Returns the `tree` record."""
    if facts["head"] != wave_sha:
        raise Refusal(f"--wave-sha {wave_sha} is not this checkout's HEAD {facts['head']} — every field the preflight compares is read "
                      f"from THIS tree, so the certificate can only name the tree it read: run it from a checkout at the wave sha "
                      "(git worktree add --detach <dir> <sha>; the launcher's own layout is <the eval-repo root>/<sha>)")
    if facts["dirty_paths"]:
        raise Refusal(f"the checkout at {wave_sha[:12]} has uncommitted changes under the evidence roots {facts['evidence_roots']}: "
                      f"{facts['dirty_paths']} — the certificate would name a commit whose tree is not what was read; commit or "
                      "stash them (the launcher runs the COMMIT, never a working tree)")
    return {"wave_sha": wave_sha, "resolved_head": facts["head"], "head_equals_wave_sha": True, **facts,
            "binding": "every compared / composed value was read from the checkout whose HEAD is wave_sha, with no uncommitted "
                       "tracked change under the evidence roots; changes outside them are recorded, not refused"}


# --------------------------------------------------------------------------- the published record


def published_none_manifest(published_root: Path, key: str) -> tuple[Path, dict]:
    pat = str(published_root / key / "none" / M.SAMPLING_THINKING_CARD / "cells" / "*" / "manifest.json")
    found = sorted(glob.glob(pat))
    if len(found) != 1:
        raise Refusal(f"{key}: expected exactly ONE published `none` manifest at {pat}, found {len(found)} — the published pull "
                      "(pull_wave.sh of the published tree) is the preflight's baseline; nothing is skipped")
    return Path(found[0]), json.loads(Path(found[0]).read_text(encoding="utf-8"))


def published_outcomes(manifest_path: Path) -> dict:
    """The published leg's per-item OUTCOME record (`outcomes/inspect__math.json`: the driver's own list of every
    ATTEMPTED item — `n_attempted_items`, each item's id, at-cap bit, scored bit and `sampling_ref`). The rows.jsonl
    `acc` rows are the SCORED subset (an at-cap / unterminated item has no `acc` row: the published qwen `none` leg
    scored 429 of its 500 attempted items), so the population is read HERE, never off the rows."""
    path = manifest_path.parent / "outcomes" / "inspect__math.json"
    if not path.is_file():
        raise Refusal(f"no outcomes/inspect__math.json beside {manifest_path} — the published leg's ATTEMPTED item set is read from it")
    doc = json.loads(path.read_text(encoding="utf-8"))
    items = doc.get("items") or []
    if not items or doc.get("n_attempted_items") != len(items):
        raise Refusal(f"{path}: n_attempted_items {doc.get('n_attempted_items')!r} != {len(items)} items listed")
    return doc


def published_item_ids(manifest_path: Path, outcomes: dict | None = None) -> list[str]:
    outcomes = outcomes if outcomes is not None else published_outcomes(manifest_path)
    return sorted(str(it["item_id"]) for it in outcomes["items"])


def published_sampling_ref(manifest_path: Path, man: dict, outcomes: dict | None = None) -> dict:
    """The published leg's `sampling_ref` (gen_stats + every outcome item; the rows predate the field) and its
    SELF-CONSISTENCY: the ref rebuilt through the producer's own API over the manifest's sampling block
    (`contracts.sampling_ref`) must equal what the leg recorded — validated against ITS OWN producer block (P5),
    never against the matrix dict."""
    gs_path = manifest_path.parent / "gen_stats.json"
    if not gs_path.is_file():
        raise Refusal(f"no gen_stats.json beside {manifest_path}")
    gs = (json.loads(gs_path.read_text(encoding="utf-8")).get("inspect/math") or {})
    outcomes = outcomes if outcomes is not None else published_outcomes(manifest_path)
    refs_items = sorted({str(it.get("sampling_ref")) for it in outcomes["items"]})
    rebuilt = C.sampling_ref(man.get("sampling") or {})
    return {"gen_stats": gs.get("sampling_ref"), "outcome_items": refs_items, "rebuilt_from_manifest_block": rebuilt,
            "self_consistent": gs.get("sampling_ref") == rebuilt and refs_items == [str(gs.get("sampling_ref"))]}


def published_shared(man: dict, manifest_path: Path) -> dict:
    """The published leg's side of the shared projection, read off the fields its manifest CARRIES."""
    tiny = man.get("tinycap") or {}
    smp = man.get("sampling") or {}
    outcomes = published_outcomes(manifest_path)          # read ONCE: the item ids, the population and the refs come off it
    ref = published_sampling_ref(manifest_path, man, outcomes)
    return {
        "task": (man.get("controls") or [None])[0],
        "item_ids": published_item_ids(manifest_path, outcomes),
        "population": {k: outcomes.get(k) for k in ("n_attempted_items", "n_attempted_samples", "n_scored", "n_unscored", "n_at_cap",
                                                     "samples_per_item", "unterminated_policy")},
        "sampling_ref": ref["gen_stats"],
        "sampling_ref_record": ref,
        # the NUMERIC projection: a key the block carries as null was NOT SENT (the legacy block names top_k on every
        # row, null where the model's row has none) — null reads as absent, exactly as the wave omits it
        "sampling_params": {k: smp[k] for k in M.SAMPLING_PARAM_KEYS if smp.get(k) is not None},
        "sampling_mode": smp.get("mode"),
        "model_id": man.get("model_id"), "model_revision": man.get("model_revision"),
        "resolved_model_id": ((man.get("provenance") or {}).get("serving") or {}).get("resolved_model_id"),
        "resolved_revision": ((man.get("provenance") or {}).get("serving") or {}).get("resolved_revision"),
        "thinking_mode": (man.get("thinking") or {}).get("mode"),
        "enable_thinking": (man.get("thinking") or {}).get("enable_thinking"),
        "thinking_render": (man.get("thinking") or {}).get("render"),
        "identity_version": tiny.get("identity_version"),
        "max_tokens_by_task": tiny.get("max_tokens_by_task"),
        "math500_max_tokens": tiny.get("math500_max_tokens"),
        "math500_retry_max_tokens": tiny.get("math500_retry_max_tokens"),
        "samples_by_task": tiny.get("samples_by_task"),
        "chat_template_hash": ((man.get("provenance") or {}).get("serving") or {}).get("chat_template_hash"),
        "client_lock_sha256": (man.get("frameworks") or {}).get("client_lock_sha256"),
        # the published side of the client-lock relation (L1 G0): the digest the leg RECORDED and the tree whose lock
        # text it must equal (chain of custody; the text is read off that tree by compare_client_lock)
        "client_lock": {"sha256": (man.get("frameworks") or {}).get("client_lock_sha256"),
                        "tree": (man.get("producer") or {}).get("tree_sha")},
        "inspect_ai": (man.get("frameworks") or {}).get("inspect_ai"),
        "inspect_evals": (man.get("frameworks") or {}).get("inspect_evals"),
        "producer_tree_sha": (man.get("producer") or {}).get("tree_sha"),
    }


# --------------------------------------------------------------------------- the wave tree's side


def client_lock_facts() -> dict:
    """The wave tree's client lock: its digest, the two inspect pins, its TEXT, and the import walk from the client venv's
    MATH-500 entry points over THIS tree (client_lock.walk_client_imports) plus the running interpreter's name map —
    computed once per run; compare_client_lock reads them. The text and the walk never enter the certificate whole (the
    walk's record and the compared packages do)."""
    text = CLIENT_LOCK.read_text(encoding="utf-8")
    pins = {}
    for name in ("inspect-ai", "inspect-evals"):
        m = re.search(rf"^{re.escape(name)}==([^\s\\]+)", text, re.M)
        pins[name.replace("-", "_")] = m.group(1) if m else None
    return {"client_lock_sha256": CL.lock_sha256(text), **pins, "text": text,
            "walk": CL.walk_client_imports(REPO), "interpreter_map": CL.interpreter_distributions()}


_LOCK_FACTS: dict | None = None


def _lock_facts_cached() -> dict:
    global _LOCK_FACTS
    if _LOCK_FACTS is None:
        _LOCK_FACTS = client_lock_facts()
    return _LOCK_FACTS


def published_lock_text(tree_sha: str) -> tuple[str | None, str | None]:
    """The published tree's lock TEXT (`git show <tree>:<lock>` in this checkout) — the one indirection the tests
    replace with a synthetic text; `(None, why)` when the tree cannot be read here (UNDETERMINED, never a pass)."""
    return CL.lock_text_at_tree(REPO, tree_sha)


_LOCK_RELATION_CACHE: dict[tuple, tuple[dict, list[str], list[str]]] = {}


def compare_client_lock(pub: dict | None, wave: dict | None, lock: dict) -> tuple[dict, list[str], list[str]]:
    """The client-lock relation (L1 G0; client_lock.compare_locks): -> (record, problems, undetermined). `pub` is the
    published side {sha256 recorded by the leg, tree}; `wave` {sha256} (the wave text and walk come from `lock`).
    Problems: `shared.client_lock: ...` when the published record's digest is not its tree's lock digest (custody) or
    the published side carries no digest; `shared.client_lock.<dist>: ...` per imported distribution whose version
    differs. Undetermined: the published tree's lock unreadable here. Cached per (tree, recorded sha, wave sha)."""
    pub = pub or {}
    key = (pub.get("tree"), pub.get("sha256"), (wave or {}).get("sha256"), lock["client_lock_sha256"])
    if key in _LOCK_RELATION_CACHE:
        return _LOCK_RELATION_CACHE[key]
    problems: list[str] = []
    undetermined: list[str] = []
    rec: dict[str, Any] = {"equal": None, "published": None, "wave": {"sha256": lock["client_lock_sha256"]}}
    if not pub.get("sha256") or not pub.get("tree"):
        problems.append(f"shared.client_lock: the published record carries no client lock digest / producer tree ({pub}) — the "
                        "relation cannot be read; every published MATH-500 leg records frameworks.client_lock_sha256 and producer.tree_sha")
        rec["equal"] = False
    else:
        text, why = published_lock_text(pub["tree"])
        rec["published"] = CL.published_lock_record(text, why, pub["sha256"], pub["tree"])
        if text is None:
            u = (f"shared.client_lock: the published tree's lock cannot be read here ({why}) — the imported-package comparison needs "
                 f"`git show {pub['tree'][:12]}:{CL.LOCK_REL}` (a clone that carries the published tree); UNDETERMINED, never equal")
            undetermined.append(u)
            rec["undetermined"] = u
        elif not rec["published"]["matches_record"]:
            problems.append(f"shared.client_lock: the published leg recorded client_lock_sha256 {pub['sha256']} but its producer tree "
                            f"{pub['tree'][:12]}'s lock digests {rec['published']['sha256_of_tree_lock']} — the leg did not run under its "
                            "tree's lock (chain of custody), so no text can stand for the lock it ran under")
            rec["equal"] = False
        else:
            comparison, lock_problems = CL.compare_locks(text, lock["text"], lock["walk"], interpreter_map=lock["interpreter_map"])
            rec.update(comparison)
            rec["equal"] = comparison["one_instrument"]
            problems += [f"shared.{p}" for p in lock_problems]
    _LOCK_RELATION_CACHE[key] = (rec, problems, undetermined)
    return rec, problems, undetermined


def plan_item_ids() -> list[str]:
    """The 500 MATH-500 ids the wave tree plans (offline, the local datasets cache) — the task's own planner
    (inspect_served.plan_task, the thinking arm is irrelevant to the id set: the OFF plan is what math500_messages
    uses too). A missing cache is a REFUSAL, never a skipped field (A9)."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    try:
        from sisteer.capeval.tiers import inspect_served as isv
        plan = isv.plan_task("math", grader_model=None, thinking=C.THINKING_OFF, repo_root=REPO)
    except Exception as e:  # noqa: BLE001 — named, never skipped
        raise Refusal(f"the wave tree cannot plan the MATH-500 item set offline ({type(e).__name__}: {str(e)[:300]}) — prepare the "
                      "local datasets cache (an online load of HuggingFaceH4/MATH-500 + DigitalLearningGmbH/MATH-lighteval) first")
    ids = sorted(str(i) for i in (plan.sample_ids or []))
    if len(ids) != 500:
        raise Refusal(f"the wave tree planned {len(ids)} MATH-500 items, not 500")
    return ids


def load_tokenizer_for(key: str):
    """(hf_id, revision, tokenizer) through the prompt-budget tool's own loader (the same pinned tokenizer the budget
    digests), offline."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    try:
        import tinycap_oos_prompt_budget as B
        hf, rev, _, tok = B._load_config_tokenizer(REPO / f"configs/sdfbase/math500/sdfbase_full22_{key}.math500.yaml", key)
    except Exception as e:  # noqa: BLE001
        raise Refusal(f"{key}: the pinned tokenizer is not loadable offline ({type(e).__name__}: {str(e)[:300]}) — the template "
                      "digest and the server-settings composition need it; fetch the snapshot into the local HF cache first")
    return hf, rev, tok


def overlay(key: str) -> tuple[dict, dict, Path, Path]:
    import yaml
    cfg_path = REPO / f"configs/sdfbase/math500/sdfbase_full22_{key}.math500.yaml"
    man_path = REPO / f"configs/capeval/manifests/petri_full22_{key}.math500.yaml"
    for p in (cfg_path, man_path):
        if not p.is_file():
            raise Refusal(f"missing overlay {p} (scripts/sdfbase/math500/make_math500_overlays.py)")
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    man = yaml.safe_load(man_path.read_text(encoding="utf-8"))["petri"]
    return cfg, man, cfg_path, man_path


def wave_sampling_block(row: dict) -> dict:
    """The sampling block the DRIVER will write for this row (served_capability_run._sampling_block's shape: mode, the
    model key, the hashed identity of the CONTRACTS row the driver resolves under `--sampling <mode>`, the contracts
    row's source prose) — so the wave's expected `sampling_ref` is rebuilt through the producer's API, never from the
    matrix dict (whose `source` is the launcher's own provenance prose)."""
    crow = C.sampling_for(row["model_key"], row["sampling_mode"])
    return {"mode": row["sampling_mode"], "model_key": row["model_key"],
            **C.sampling_identity(crow, {C.inspect_task(M.INSPECT_TASK): M.SAMPLES}), "source": crow.source}


def wave_shared(row: dict, key: str, cfg: dict, template_hash: str | None, item_ids: list[str] | None, lock: dict) -> dict:
    """The wave tree's side of the shared projection for `row`. `item_ids` None = the population was NOT planned
    (--no-plan-items): the field is None and compare_shared files it UNDETERMINED — the published ids are never
    echoed into this side (an echo compares the published population with itself and proves nothing; fix A1)."""
    block = wave_sampling_block(row)
    return {
        "task": M.CONTROL,
        "item_ids": item_ids,
        "sampling_ref": C.sampling_ref(block),
        "sampling_block_keys": sorted(block),
        "sampling_params": M.sampling_params(row["sampling"]),
        "sampling_mode": row["sampling_mode"],
        "model_id": cfg["model"], "model_revision": cfg["model_revision"],
        "resolved_model_id": cfg["model"], "resolved_revision": cfg["model_revision"],
        "thinking_mode": row["thinking_mode"],
        "enable_thinking": bool(C.enable_thinking(row["thinking_mode"])),
        "thinking_render": C.THINKING_RENDER_FLAG_ONLY,
        "identity_version": M.IDENTITY_VERSION,
        "max_tokens_by_task": {C.inspect_task(M.INSPECT_TASK): M.CAP},
        "math500_max_tokens": M.CAP,
        "math500_retry_max_tokens": M.RETRY_MAX_TOKENS,
        "samples_by_task": {C.inspect_task(M.INSPECT_TASK): M.SAMPLES},
        "chat_template_hash": template_hash,
        "client_lock_sha256": lock["client_lock_sha256"],
        "client_lock": {"sha256": lock["client_lock_sha256"]},
        "inspect_ai": lock["inspect_ai"], "inspect_evals": lock["inspect_evals"],
    }


#: `client_lock` replaced `client_lock_sha256` (L1 G0): the digest is no longer a field compared for equality but the
#: input of the imported-package relation (compare_client_lock); both digests stay on the record inside that field
SHARED_FIELDS = ("task", "item_ids", "sampling_ref", "sampling_params", "sampling_mode", "model_id", "model_revision", "resolved_model_id",
                 "resolved_revision", "thinking_mode", "enable_thinking", "thinking_render", "identity_version",
                 "max_tokens_by_task", "math500_max_tokens", "math500_retry_max_tokens", "samples_by_task",
                 "chat_template_hash", "client_lock", "inspect_ai", "inspect_evals")


ITEM_IDS_NOT_PLANNED = ("shared.item_ids: the wave tree's item population was NOT planned (--no-plan-items) — nothing was compared "
                        "with the published population; UNDETERMINED, never equal (plan the ids offline: drop the switch)")


def compare_shared(pub: dict, wave: dict, lock: dict | None = None) -> tuple[dict, list[str], list[str]]:
    """(i): field by field -> (record, problems, undetermined). An `item_ids` difference is reported as the symmetric
    difference (never 500 ids twice); a wave side whose `item_ids` is None (not planned) is UNDETERMINED by name, with
    `equal: None` — a skipped comparison is never a passed one (fix A1). `client_lock` is the imported-package relation
    (compare_client_lock, L1 G0), never a digest equality; `lock` = client_lock_facts() (this tree's, cached when None)."""
    out, problems, undetermined = {}, [], []
    for f in SHARED_FIELDS:
        p, w = pub.get(f), wave.get(f)
        if f == "item_ids" and w is None:
            out[f] = {"equal": None, "published_n": len(p or []), "wave_n": None, "only_published": None, "only_wave": None,
                      "undetermined": ITEM_IDS_NOT_PLANNED}
            undetermined.append(ITEM_IDS_NOT_PLANNED)
            continue
        if f == "client_lock":
            rec_l, probs_l, undet_l = compare_client_lock(p, w, lock or _lock_facts_cached())
            out[f] = rec_l
            problems += probs_l
            undetermined += undet_l
            continue
        equal = p == w
        rec: dict[str, Any] = {"equal": equal}
        if f == "item_ids":
            rec.update({"published_n": len(p or []), "wave_n": len(w or []),
                        "only_published": sorted(set(p or []) - set(w or []))[:20], "only_wave": sorted(set(w or []) - set(p or []))[:20]})
        else:
            rec.update({"published": p, "wave": w})
        out[f] = rec
        if not equal:
            problems.append(f"shared.{f}: published {json.dumps(p if f != 'item_ids' else rec['only_published'], default=str)[:200]} "
                            f"!= wave {json.dumps(w if f != 'item_ids' else rec['only_wave'], default=str)[:200]}")
    return out, problems, undetermined


# --------------------------------------------------------------------------- (i') each row's own contract


def resolve_geometry(row: dict) -> dict:
    from capctrl_serving_up import resolve_standup_geometry
    geom = resolve_standup_geometry(row["geometry_eval"], None, None, model_key=row["model_key"])
    return {"eval": row["geometry_eval"], "L_row": row["geometry_L"], "L_resolved": int(geom.max_model_len),
            "S": int(geom.max_num_seqs), "budget": getattr(geom, "budget", None)}


def budget_sidecar_check(row: dict, budgets_root: Path | None, wave_sha: str) -> dict:
    if budgets_root is None:
        return {"checked": False, "why": "no --budgets-root given (the launcher re-checks the sidecar at launch: m5t_budget_artifact)"}
    side_path = budgets_root / row["model_key"] / row["budget_class"] / (M.OOS_PROMPT_BUDGET_FILENAME + ".fit.json")
    if not side_path.is_file():
        return {"checked": True, "ok": False, "path": str(side_path), "why": "no fit sidecar — run the prompt budget at the wave sha first"}
    d = json.loads(side_path.read_text(encoding="utf-8"))
    si_cell = row["cell"] if row["route_kind"] == "prompted" else None
    want = {"fits": True, "cap": M.CAP, "geometry_eval": row["geometry_eval"], "window": row["geometry_L"], "arm": row["thinking_mode"],
            "tree_sha": wave_sha, "system_prompt_cell": si_cell,
            "system_prompt_sha16": (row["si_sha256"] or "")[:16] if si_cell else None}
    bad = {k: (d.get(k), v) for k, v in want.items() if d.get(k) != v}
    return {"checked": True, "ok": not bad, "path": str(side_path), "prompt_max": d.get("prompt_max"), "margin": d.get("margin"),
            "mismatch": {k: {"sidecar": a, "row": b} for k, (a, b) in bad.items()}}


def row_contract(row: dict, man: dict, pins: dict, staged_main: dict, budgets_root: Path | None, wave_sha: str,
                 admission: dict | None = None) -> tuple[dict, list[str]]:
    from sisteer.capeval.petri import registry_adapter as ra
    admission = admission if admission is not None else M.load_steer_admission()     # the external rows' pins
    problems: list[str] = []
    rec: dict[str, Any] = {"cell": row["cell"], "harness_cell": row["harness_cell"], "route_kind": row["route_kind"],
                           "budget_class": row["budget_class"], "wave": row["wave"], "status": row["status"]}
    # arditi-qwen-all-positions: a cell the SERVING vocabulary does not admit yet (cells.PENDING_SERVING_CELLS — its Class B
    # registration lands in another author's change) is a PROBLEM BY NAME here, never an exception out of the certificate:
    # the row cannot be served on this tree, and saying so out loud is exactly a compatibility certificate's job.
    if row["cell"] in M.PENDING_SERVING_CELLS:
        rec["pending_serving_registration"] = True
        problems.append(f"row {row['index']}: the serving vocabulary does not admit cell {row['cell']!r} on this tree "
                        "(contracts.parse_launcher_cell refuses it; cells.PENDING_SERVING_CELLS) — its Class B registration "
                        "has not landed, so the standup would refuse this row; NOT launch-eligible until it does")
    else:
        parsed = C.parse_launcher_cell(row["cell"])
        # the grammar's kinds (base / prompted / fitted) onto the row's route kinds (base / prompted / lens)
        if {"base": "base", "prompted": "prompted", "fitted": "lens"}.get(parsed.kind) != row["route_kind"] or parsed.harness_id != row["harness_cell"]:
            problems.append(f"row {row['index']}: cell grammar {parsed} disagrees with the row ({row['route_kind']}, {row['harness_cell']})")
    if row["route_kind"] == "prompted":
        si = M.prompted_si(row["cell"])
        text_sha = hashlib.sha256(ra._traintest_si_text_bytes(si)).hexdigest()
        harness_si = row["harness_cell"][len("prompted__"):]
        manifest_hex16 = (man.get("prompted_prefix_hashes") or {}).get(harness_si)
        rec["si"] = {"traintest": si, "harness": harness_si, "sha256_registry": text_sha, "sha256_row": row["si_sha256"],
                     "manifest_prefix_hex16": manifest_hex16}
        if text_sha != row["si_sha256"]:
            problems.append(f"row {row['index']}: SI text sha256 {text_sha} != the row's {row['si_sha256']}")
        if manifest_hex16 != text_sha[:16]:
            problems.append(f"row {row['index']}: overlay manifest prompted_prefix_hashes[{harness_si}] = {manifest_hex16} != {text_sha[:16]}")
        if row["fitted"] or row["object_sha256"] or row["tuple_name"]:
            problems.append(f"row {row['index']}: a prompted row carries an object / tuple")
    elif row["route_kind"] == "lens" and row.get("object_source") == M.OBJECT_SOURCE_EXTERNAL:
        # an EXTERNAL lane's object (plan P8; decisions-w4-launch L1 G7 / L4): the pin is the ADMISSION RECORD's
        # (`configs/sdfbase/steer_admission.json`, keyed by model key AND cell), never the full22 pins file's; the row binds
        # its OWN bank view per cluster (the bytes are held to the pin in-job by st_external_bank_view_check and at two
        # more doors), so the roster's `main` view never stages it; the tuple rule is the switch models' (`_steer`), a
        # no-switch family (`not_available`) is IN SUPPORT and carries none
        adm = admission.get((row["model_key"], row["cell"]))
        pin = adm.get("object_sha256") if adm else None
        # the view the ADMISSION entry names is the object BANK's. A steering row collapses the two into `view`; a row
        # that keeps them apart (the all-positions arm: panel `it`, bank `ar_c0p3`) carries the bank's in `bank_view`, and
        # comparing the record against its PANEL view there would be a false mismatch on a row that is exactly right.
        bank_view = row.get("bank_view") or row["view"]
        rec["object"] = {"staged_id": row["staged_id"], "source": M.OBJECT_SOURCE_EXTERNAL, "admission_pin": pin, "row": row["object_sha256"],
                         "view": row["view"], "bank_view": bank_view, "admission_view": (adm or {}).get("view"), "bank_view_rel": row.get("bank_view_rel"),
                         "tuple": row["tuple_name"], "served_positions_expected": row["served_positions_expected"],
                         "selection_label": row.get("selection_label"), "grid_limited": row.get("grid_limited")}
        if adm is None:
            problems.append(f"row {row['index']}: no admission record for ({row['model_key']}, {row['cell']}) — an external row is never launched without one (P8)")
        else:
            if pin != row["object_sha256"] or not (isinstance(pin, str) and len(pin) == 64):
                problems.append(f"row {row['index']}: admission pin {pin} != the row's object {row['object_sha256']}")
            if adm.get("view") != bank_view:
                problems.append(f"row {row['index']}: admission view {adm.get('view')!r} != the row's bank view {bank_view!r}")
            if not any((row.get("bank_view_rel") or {}).values()):
                problems.append(f"row {row['index']}: no bank view on any cluster (bank_view_rel {row.get('bank_view_rel')!r})")
        if row["tuple_name"] is None:
            if row["thinking_mode"] != C.THINKING_NOT_AVAILABLE:
                problems.append(f"row {row['index']}: a switch-family external row carries no tuple (thinking {row['thinking_mode']!r})")
        elif row["tuple_name"] not in C.AUTHORISED_OOS_EXPERIMENTS:
            problems.append(f"row {row['index']}: tuple {row['tuple_name']} is not registered")
        else:
            exp = C.AUTHORISED_OOS_EXPERIMENTS[row["tuple_name"]]
            if row["cell"] not in exp.cells or exp.model != row["model_key"] or exp.sampling_mode != row["sampling_mode"]:
                problems.append(f"row {row['index']}: tuple {row['tuple_name']} does not authorise ({row['model_key']}, {row['cell']}, {row['sampling_mode']})")
        if row["served_positions_expected"] != "all":
            problems.append(f"row {row['index']}: served_positions_expected {row['served_positions_expected']!r} != 'all'")
    elif row["route_kind"] == "lens":
        pin = pins.get(row["model_key"], {}).get(row["staged_id"])
        pin = pin.get("object_sha256") if isinstance(pin, dict) else pin
        rec["object"] = {"staged_id": row["staged_id"], "pin_file": pin, "row": row["object_sha256"],
                         "staged_on_main_view": bool(staged_main.get(row["cell"])), "tuple": row["tuple_name"],
                         "served_positions_expected": row["served_positions_expected"]}
        if pin != row["object_sha256"] or not (isinstance(pin, str) and len(pin) == 64):
            problems.append(f"row {row['index']}: pin {pin} != the row's object {row['object_sha256']}")
        if not staged_main.get(row["cell"]):
            problems.append(f"row {row['index']}: {row['cell']} is not staged on the roster's `{M.VIEW_MAIN}` view")
        if row["tuple_name"] not in C.AUTHORISED_OOS_EXPERIMENTS:
            problems.append(f"row {row['index']}: tuple {row['tuple_name']} is not registered")
        else:
            exp = C.AUTHORISED_OOS_EXPERIMENTS[row["tuple_name"]]
            if row["cell"] not in exp.cells or exp.model != row["model_key"] or exp.sampling_mode != row["sampling_mode"]:
                problems.append(f"row {row['index']}: tuple {row['tuple_name']} does not authorise ({row['model_key']}, {row['cell']}, {row['sampling_mode']})")
        if row["served_positions_expected"] != "all":
            problems.append(f"row {row['index']}: served_positions_expected {row['served_positions_expected']!r} != 'all'")
    else:
        if row["fitted"] or row["object_sha256"]:
            problems.append(f"row {row['index']}: a bare row carries an object")
    geo = resolve_geometry(row)
    rec["geometry"] = geo
    if geo["L_resolved"] != geo["L_row"]:
        problems.append(f"row {row['index']}: {row['geometry_eval']} resolves L {geo['L_resolved']} != the row's {geo['L_row']}")
    rec["budget"] = budget_sidecar_check(row, budgets_root, wave_sha)
    if rec["budget"].get("checked") and not rec["budget"].get("ok"):
        problems.append(f"row {row['index']}: budget fit sidecar does not admit the row ({rec['budget'].get('why') or rec['budget'].get('mismatch')})")
    return rec, problems


# --------------------------------------------------------------------------- (i'') the normalised server settings


def _flags(cmd: tuple[str, ...]) -> dict[str, Any]:
    """`--flag value` -> value, bare `--flag` -> True, off the composed argv (values never start with `--`)."""
    out: dict[str, Any] = {}
    words = list(cmd)
    for i, w in enumerate(words):
        if w.startswith("--"):
            nxt = words[i + 1] if i + 1 < len(words) else None
            out[w] = nxt if (nxt is not None and not nxt.startswith("--")) else True
    return out


#: the launcher files whose text decides the standup-wide thinking flag at a tree: f22_standup composes `--thinking-on`
#: iff F22_THINKING_ON is set (a tree without the mechanism composed the composer's default, OFF)
LAUNCHER_FILES = ("scripts/sdfbase/math500/math500_thinking.sbatch", "scripts/sdfbase/math500/m5t_common.sh",
                  "scripts/sdfbase/full22/common.sh")
STANDUP_MODE_PETRI = "petri"                     # mt_serving_up: tool_calls=(args.mode == "petri")
#: `--default-chat-template-kwargs` -> the server-default thinking policy it pins (vllm_server's two literals); a
#: template without the switch composes none -> `no_switch`
SERVER_DEFAULT_THINKING_BY_KWARGS = {'{"enable_thinking": true}': "on", '{"enable_thinking": false}': "off", None: "no_switch"}


def launch_config_from_dry_run(row: dict, s: int, tmp: Path) -> dict:
    """The launch CONFIGURATION the sbatch hands f22_standup for THIS row, read off the launcher's OWN dry run
    (math500_thinking.sbatch under M5T_DRY_RUN=1 -> m5t_print_plan: the `STANDUP:` extras argv + the `STANDUP_POLICY:`
    line) — never a re-typed mirror of the sbatch (fix A4: the preflight composed `thinking_on=True` while the job
    never sets F22_THINKING_ON). Refuses when the dry run and the matrix disagree on the row's cell or parser."""
    import m5t_dry_run as D
    text = D.shell_plan(row, s=s, shared=tmp / "dry_shared", python=sys.executable)
    standup = [l for l in text.splitlines() if l.startswith("STANDUP:")]
    policy = [l for l in text.splitlines() if l.startswith("STANDUP_POLICY:")]
    if len(standup) != 1 or len(policy) != 1:
        raise Refusal(f"row {row['index']}: the launcher's dry run printed {len(standup)} STANDUP and {len(policy)} STANDUP_POLICY lines (one each expected)")
    argv = standup[0].split()[1:]
    kv = dict(tok.split("=", 1) for tok in policy[0].split()[1:])
    if set(kv) != {"mode", "thinking_on_flag", "server_default_thinking", "request_thinking", "gpu_mem_util", "max_logprobs"}:
        raise Refusal(f"row {row['index']}: STANDUP_POLICY carries {sorted(kv)} — not the fields the preflight reads")

    def after(flag: str):
        return argv[argv.index(flag) + 1] if flag in argv else None
    cfg = {"mode": kv["mode"], "thinking_on": kv["thinking_on_flag"] == "1", "server_default_thinking": kv["server_default_thinking"],
           "request_thinking_mode": kv["request_thinking"], "gpu_mem_util": kv["gpu_mem_util"],
           "max_logprobs": None if kv["max_logprobs"] == "none" else int(kv["max_logprobs"]),
           "enable_log_requests": "--enable-log-requests" in argv, "cell": after("--cell"),
           "expect_object_sha256": after("--expect-object-sha256"), "reasoning_parser": after("--reasoning-parser"),
           "upstream_timeout_s": after("--upstream-timeout-s"), "lens_salt_policy": after("--lens-salt-policy"),
           "standup_argv": argv, "source": "math500_thinking.sbatch M5T_DRY_RUN=1 -> m5t_print_plan (STANDUP + STANDUP_POLICY lines)"}
    if cfg["cell"] != row["cell"] or cfg["reasoning_parser"] != row["reasoning_parser"] or cfg["request_thinking_mode"] != row["thinking_mode"]:
        raise Refusal(f"row {row['index']}: the launcher's dry run names cell {cfg['cell']!r} / parser {cfg['reasoning_parser']!r} / "
                      f"thinking {cfg['request_thinking_mode']!r}, the matrix row {row['cell']!r} / {row['reasoning_parser']!r} / {row['thinking_mode']!r}")
    if cfg["mode"] not in ("attacks", STANDUP_MODE_PETRI):
        raise Refusal(f"row {row['index']}: STANDUP_POLICY mode {cfg['mode']!r} is not a mt_serving_up mode")
    return cfg


def published_server_default_thinking(tree: str = PUBLISHED_TREE) -> dict:
    """The PUBLISHED tree's server-default thinking policy, read off its launcher files AT THAT TREE (`git show`): the
    composer's `thinking_on` is True only under mt_serving_up's `--thinking-on`, which f22_standup adds only when
    F22_THINKING_ON is set — a tree whose launcher never sets it (or, like the published one, carries no such
    mechanism) served the composer's default, OFF. The published manifest does not record this (composed-only there),
    so it is read from the immutable tree instead; unreadable objects = UNDETERMINED (never assumed)."""
    texts: dict[str, str] = {}
    for rel in LAUNCHER_FILES:
        r = subprocess.run(["git", "-C", str(REPO), "show", f"{tree}:{rel}"], capture_output=True, text=True, timeout=60)
        if r.returncode != 0:
            return {"policy": None, "tree": tree, "how": "git show <tree>:<launcher files>",
                    "undetermined": f"git show {tree[:12]}:{rel} failed ({r.stderr.strip()[:120]}) — the published tree's launcher cannot be read here"}
        texts[rel] = r.stdout
    sets_flag = {rel: bool(re.search(r"^\s*(export\s+)?F22_THINKING_ON=", t, re.M)) for rel, t in texts.items()}
    mechanism = any("--thinking-on" in t for t in texts.values())
    return {"policy": "on" if any(sets_flag.values()) else "off", "tree": tree, "sets_F22_THINKING_ON": sets_flag,
            "launcher_has_thinking_on_mechanism": mechanism,
            "how": "git show <tree>:<launcher files>: f22_standup composes --thinking-on iff F22_THINKING_ON is set; absent (or no mechanism) = the composer's default OFF"}


def compose_launch(row: dict, key: str, man_path: Path, tok, geo: dict, tmp: Path, launch_cfg: dict) -> dict:
    """The wave tree's launch for THIS row, composed by the standup's own composer over the overlay manifest narrowed to
    the row's cell (mt_serving_up's shape: --enable-lora dropped, the pinned template written from the tokenizer) UNDER
    THE JOB'S OWN LAUNCH CONFIGURATION (`launch_cfg`, read off the launcher's dry run — fix A4: `thinking_on` iff the
    standup gets --thinking-on, `tool_calls` iff the mode is petri, the row's parser, then mt_serving_up's own
    `launch_cmd_extras` splice: memory utilisation, --max-logprobs, --enable-log-requests) — projected onto its flags;
    NEVER launched."""
    import mt_serving_up as mt
    from sisteer.capeval import manifest as cm
    from sisteer.capeval.serving import vllm_server as vs
    spec0 = cm.load_petri_spec(man_path)
    tpath = tmp / f"{key}_chat_template.jinja"
    tpath.write_bytes(tok.chat_template.encode())
    tsha = cm.sha256_hex16(tok.chat_template.encode())
    spec = dataclasses.replace(spec0, serving_launch_flags=[f for f in spec0.serving_launch_flags if f != "--enable-lora"],
                               cells=[row["harness_cell"]])
    # a PLACEHOLDER port: the composer validates it, nothing here binds or launches (the argv is projected, never run).
    # No lens registry: the registry is the STANDUP's own export over the bank view (a served-object input, not a
    # behaviour-affecting launch flag); the composer's default (None) composes the base launch, which is what the
    # projection below reads
    launch = vs.build_server_launch(spec, port=COMPOSE_PLACEHOLDER_PORT, lens_registry_dir=None, chat_template_path=tpath,
                                    max_model_len=geo["L_resolved"], max_num_seqs=geo["S"],
                                    tool_calls=(launch_cfg["mode"] == STANDUP_MODE_PETRI), thinking_on=launch_cfg["thinking_on"],
                                    reasoning_parser=launch_cfg["reasoning_parser"])
    cmd = tuple(launch.cmd) + mt.launch_cmd_extras(launch_cfg["gpu_mem_util"], launch_cfg["max_logprobs"], launch_cfg["enable_log_requests"])
    flags = _flags(cmd)
    return {"template_sha16": tsha, "manifest_chat_template_hash": spec0.chat_template_hash,
            "flags": {f: flags[f] for f in PROJECTED_FLAGS if f in flags},
            "excluded": {f: {"value": flags.get(f), "why": why} for f, why in EXCLUDED_FLAGS.items() if f in flags or f == "--enable-log-requests"},
            "default_chat_template_kwargs": launch.default_chat_template_kwargs,
            "server_default_thinking": SERVER_DEFAULT_THINKING_BY_KWARGS.get(launch.default_chat_template_kwargs, "unknown"),
            "override_generation_config": launch.override_generation_config,
            "prefix_caching": launch.prefix_caching, "mamba_ssm_cache_dtype": launch.mamba_ssm_cache_dtype,
            "gpu_memory_utilization": flags.get("--gpu-memory-utilization"),
            "model": cmd[list(cmd).index("serve") + 1] if "serve" in cmd else None,
            "reduction_mode_standup_literal": REDUCTION_MODE_STANDUP_LITERAL,
            "launch_config": launch_cfg}


SNAPSHOT_CACHED, SNAPSHOT_NO_EXIST, SNAPSHOT_NOT_CACHED = "cached", "no_exist", "not_cached"


def snapshot_generation_config(model_id: str, revision: str) -> dict:
    """The snapshot's generation_config.json from the LOCAL cache (offline), with the hub's THREE readings kept apart
    (the same distinction vllm_server._server_default_generation_config draws): `cached` (the file IS in the cache —
    `keys` = the sampler keys it sets, {} when none), `no_exist` (the hub said the snapshot ships NO such file — the
    no-exist sentinel; `keys` None), `not_cached` (never asked: the file is simply absent from this machine's cache —
    `keys` None, and NOTHING can be inferred: the published rule for an omitted key is UNDETERMINED here)."""
    try:
        from huggingface_hub import try_to_load_from_cache
        from huggingface_hub.file_download import _CACHED_NO_EXIST
        cached = try_to_load_from_cache(model_id, "generation_config.json", revision=revision)
    except Exception as e:  # noqa: BLE001
        return {"status": SNAPSHOT_NOT_CACHED, "keys": None, "why": f"{type(e).__name__}: {str(e)[:200]}"}
    if cached is _CACHED_NO_EXIST:
        return {"status": SNAPSHOT_NO_EXIST, "keys": None}
    if not isinstance(cached, str):
        return {"status": SNAPSHOT_NOT_CACHED, "keys": None,
                "remedy": f"HF_HUB_OFFLINE=0 hf download {model_id} generation_config.json --revision {revision}"}
    doc = json.loads(Path(cached).read_text(encoding="utf-8"))
    return {"status": SNAPSHOT_CACHED, "path": cached, "keys": {k: doc[k] for k in FILE_DEFAULTABLE_KEYS if k in doc}}


class _NotComposed:
    """The wave tree's launch was NOT composed (--no-compose): every rule that would be read off the composed launch
    (the generation-config override, the server settings) is UNDETERMINED by name — never the published rule reused
    as if it were the wave's (fix A1)."""

    def __repr__(self) -> str:
        return "NOT_COMPOSED"


NOT_COMPOSED = _NotComposed()
WAVE_RULE_NOT_COMPOSED = ("UNDETERMINED: the wave tree's launch was not composed (--no-compose), so whether it pins an "
                          "--override-generation-config row for this key cannot be read; the published rule is not reused")
SERVER_SETTINGS_NOT_COMPOSED = ("server_settings: the wave tree's launch was NOT composed (--no-compose) — the normalised "
                                "server-settings projection (template digest, reasoning parser, reduction mode, render regime, "
                                "model, revision, the server-default thinking policy) was not compared; UNDETERMINED, never equal")


def sampler_resolution(row: dict, model_id: str, revision: str, override_json) -> tuple[dict, list[str], list[str]]:
    """A8: every sampler key the ROW OMITS, resolved under both trees' rules; a difference refuses by key; a rule that
    cannot be read on this machine (the snapshot file not cached) is UNDETERMINED by key — stated, never read as equal.
    `override_json` is the composed launch's `--override-generation-config` (a JSON string, or None when the wave tree
    composes none) or NOT_COMPOSED, under which the wave rule of every omitted key is UNDETERMINED (fix A1).
    Returns (record, problems, undetermined)."""
    sent = M.sampling_params(row["sampling"])
    omitted = [k for k in M.SAMPLING_PARAM_KEYS if k not in sent]
    snap = snapshot_generation_config(model_id, revision)
    file_keys = snap["keys"]
    composed = override_json is not NOT_COMPOSED
    override = json.loads(override_json) if (composed and override_json) else None
    per_key, problems, undetermined = {}, [], []
    for k in omitted:
        if snap["status"] == SNAPSHOT_NOT_CACHED:
            pub, pub_src = None, ("UNDETERMINED: the snapshot's generation_config.json is not in the local HF cache, so whether the "
                                  "published tree read this key from a vendor file cannot be told from vLLM's neutral default here")
        elif file_keys is not None and k in file_keys:
            pub, pub_src = file_keys[k], "the snapshot's generation_config.json"
        else:
            pub, pub_src = VLLM_NEUTRAL_DEFAULTS[k], ("vLLM's neutral default (the snapshot ships no generation_config.json)" if file_keys is None
                                                     else "vLLM's neutral default (the vendor file sets no such key)")
        if not composed:
            wave, wave_src = None, WAVE_RULE_NOT_COMPOSED
        elif override is not None and k in override:
            wave, wave_src = override[k], "the pinned --override-generation-config row (contracts.SERVER_DEFAULT_SAMPLING_BY_MODEL)"
        else:
            wave, wave_src = pub, pub_src
        if pub is None:
            equal = None
            undetermined.append(f"sampler_resolution.{k}: {pub_src}" + (f" — remedy: {snap['remedy']}" if snap.get("remedy") else ""))
        elif not composed:
            equal = None
            undetermined.append(f"sampler_resolution.{k}: {wave_src}")
        else:
            equal = (float(pub) == float(wave)) if isinstance(pub, (int, float)) and isinstance(wave, (int, float)) else pub == wave
            if not equal:
                problems.append(f"sampler_resolution.{k}: an omitted key resolves to {pub} under the published tree's rule but {wave} under the wave tree's")
        per_key[k] = {"published": pub, "published_rule": pub_src, "wave": wave, "wave_rule": wave_src, "equal": equal}
    return {"sent_keys": sorted(sent), "omitted_keys": omitted, "snapshot_generation_config": snap,
            "override_generation_config": (override if composed else None), "composed": composed,
            "per_key": per_key}, problems, undetermined


def compare_server_settings(pub_man: dict, composed: dict, row: dict, published_policy: dict) -> tuple[dict, list[str], list[str]]:
    """(i''): the composed launch against what the published record carries. TWO thinking fields, kept apart (fix A4):
    `enable_thinking` is the REQUEST-level flag — the manifest's `thinking.enable_thinking` is what the driver's
    `--thinking <mode>` puts on every request, and the wave's is `contracts.enable_thinking(row.thinking_mode)`
    (vLLM merges the request's chat_template_kwargs over the server default, so the request decides the render);
    `server_default_thinking` is the STANDUP-wide pin the composer writes (`--default-chat-template-kwargs`), which the
    manifest does not record — it is compared with the published TREE's launcher policy (published_server_default_thinking)
    and UNDETERMINED when that tree cannot be read. Returns (record, problems, undetermined)."""
    problems: list[str] = []
    undetermined: list[str] = []
    tiny = pub_man.get("tinycap") or {}
    fact = C.FAMILY_RENDER_FACTS.get(row["family"])
    request_enable = bool(C.enable_thinking(row["thinking_mode"]))
    compared = {
        "chat_template_hash": {"published": ((pub_man.get("provenance") or {}).get("serving") or {}).get("chat_template_hash"),
                               "wave": composed["template_sha16"], "wave_manifest": composed["manifest_chat_template_hash"]},
        "reasoning_parser": {"published": (pub_man.get("thinking") or {}).get("reasoning_parser"), "wave": composed["flags"].get("--reasoning-parser")},
        "reduction_mode": {"published": (tiny.get("served") or {}).get("reduction_mode"), "wave": composed["reduction_mode_standup_literal"]},
        "enable_thinking": {"published": (pub_man.get("thinking") or {}).get("enable_thinking"), "wave": request_enable,
                            "level": "request (the driver's --thinking on every request; the manifest records this flag)"},
        "thinking_header_ids": {"published": tiny.get("thinking_header_ids"),
                                "wave": ({"off": list(fact.off_ids), "on": list(fact.on_ids)} if fact is not None else None)},
        "model": {"published": pub_man.get("model_id"), "wave": composed["model"]},
        "revision": {"published": pub_man.get("model_revision"), "wave": composed["flags"].get("--revision")},
    }
    for f, rec in compared.items():
        rec["equal"] = rec["published"] == rec["wave"]
        if not rec["equal"]:
            problems.append(f"server_settings.{f}: published {rec['published']!r} != wave {rec['wave']!r}")
    # the SERVER-DEFAULT thinking policy: composed at the wave tree, read off the published TREE's launcher
    sd = {"published": published_policy.get("policy"), "published_source": published_policy.get("how"), "wave": composed["server_default_thinking"],
          "wave_source": composed["launch_config"]["source"], "level": "server default (--default-chat-template-kwargs; the standup's --thinking-on)"}
    if sd["published"] is None:
        sd["equal"] = None
        undetermined.append(f"server_settings.server_default_thinking: {published_policy.get('undetermined')}")
    else:
        sd["equal"] = sd["published"] == sd["wave"]
        if not sd["equal"]:
            problems.append(f"server_settings.server_default_thinking: published {sd['published']!r} != wave {sd['wave']!r}")
    compared["server_default_thinking"] = sd
    if composed["template_sha16"] != composed["manifest_chat_template_hash"]:
        problems.append(f"server_settings.chat_template_hash: the overlay manifest pins {composed['manifest_chat_template_hash']} but the "
                        f"cached tokenizer's template digests {composed['template_sha16']} (the standup would refuse)")
    composed_only = {k: composed["flags"].get(k) for k in ("--dtype", "--quantization", "--enable-prefix-caching", "--no-enable-prefix-caching",
                                                             "--mamba-ssm-cache-dtype", "--tokenizer", "--max-logprobs")}
    composed_only.update({"default_chat_template_kwargs": composed["default_chat_template_kwargs"],
                          "override_generation_config": composed["override_generation_config"],
                          "prefix_caching": composed["prefix_caching"], "mamba_ssm_cache_dtype": composed["mamba_ssm_cache_dtype"],
                          "gpu_memory_utilization": composed["gpu_memory_utilization"]})
    thinking_policy = {"server_default": composed["server_default_thinking"], "request_mode": row["thinking_mode"],
                       "request_enable_thinking": request_enable,
                       "effective_render": row["thinking_mode"],
                       "why": "vLLM merges each request's chat_template_kwargs over --default-chat-template-kwargs: the request decides "
                              "the render; the server default is what a request WITHOUT the flag would get",
                       "launch_config": composed["launch_config"]}
    return {"compared": compared, "composed_only": composed_only, "excluded_as_intentional": composed["excluded"],
            "thinking_policy": thinking_policy,
            "coverage": "compared = fields the published manifest carries (+ the server-default thinking policy read off the published "
                        "tree's launcher); composed_only = composed at the wave tree and RECORDED (the published pull carries no "
                        "endpoint.json / vLLM log); the generation-config override's EFFECT is checked under sampler_resolution"}, problems, undetermined


# --------------------------------------------------------------------------- the per-model record


def preflight_model(key: str, rows: list[dict], published_root: Path, wave_sha: str, budgets_root: Path | None, compose: bool,
                    item_ids: list[str] | None, lock: dict, tmp: Path) -> dict:
    man_path, pub_man = published_none_manifest(published_root, key)
    cfg, ov_man, _cfg_path, ov_man_path = overlay(key)
    pins = json.loads(PINS.read_text(encoding="utf-8"))
    staged_main = roster_main_cells(key)
    hf, rev, tok = load_tokenizer_for(key)
    from sisteer.models import chat_template_hash
    tsha = chat_template_hash(tok)
    if (hf, rev) != (cfg["model"], cfg["model_revision"]):
        raise Refusal(f"{key}: the budget loader resolved {hf}@{rev}, the overlay pins {cfg['model']}@{cfg['model_revision']}")
    problems: list[str] = []
    undetermined: list[str] = []
    pub = published_shared(pub_man, man_path)
    if pub["producer_tree_sha"] != PUBLISHED_TREE:
        problems.append(f"published.producer_tree_sha {pub['producer_tree_sha']} != the published tree {PUBLISHED_TREE}")
    if not pub["sampling_ref_record"]["self_consistent"]:
        problems.append(f"published.sampling_ref: the leg's recorded ref {pub['sampling_ref_record']} does not rebuild from its own "
                        "manifest sampling block (contracts.sampling_ref) — the published record is not self-consistent")
    shared_by_row, contracts, settings, resolution = {}, {}, {}, {}
    published_policy: dict | None = None
    for row in rows:
        # item_ids None (--no-plan-items) stays None on the wave side: compare_shared files it UNDETERMINED (fix A1)
        wave = wave_shared(row, key, cfg, tsha, item_ids, lock)
        shared_rec, shared_problems, shared_undetermined = compare_shared(pub, wave, lock)
        shared_by_row[str(row["index"])] = shared_rec
        problems += [f"row {row['index']} {p}" for p in shared_problems]
        undetermined += [f"row {row['index']} {u}" for u in shared_undetermined]
        contracts[str(row["index"])], cp = row_contract(row, ov_man, pins, staged_main, budgets_root, wave_sha)
        problems += cp
        if compose:
            # fix A4: the launch is composed under the JOB's own configuration (the launcher's dry run for this row),
            # and the published tree's server-default policy is read once per model (git show, immutable)
            geo = contracts[str(row["index"])]["geometry"]
            launch_cfg = launch_config_from_dry_run(row, geo["S"], tmp)
            if published_policy is None:
                published_policy = published_server_default_thinking()
            composed = compose_launch(row, key, ov_man_path, tok, geo, tmp, launch_cfg)
            settings[str(row["index"])], sp, su = compare_server_settings(pub_man, composed, row, published_policy)
            problems += [f"row {row['index']} {p}" for p in sp]
            undetermined += [f"row {row['index']} {u}" for u in su]
            override_json = composed["override_generation_config"]
        else:
            settings[str(row["index"])] = {"composed": False, "undetermined": SERVER_SETTINGS_NOT_COMPOSED}
            undetermined.append(f"row {row['index']} {SERVER_SETTINGS_NOT_COMPOSED}")
            override_json = NOT_COMPOSED
        resolution[str(row["index"])], rp, ru = sampler_resolution(row, cfg["model"], cfg["model_revision"], override_json)
        problems += [f"row {row['index']} {p}" for p in rp]
        undetermined += [f"row {row['index']} {u}" for u in ru]
    # the shared projection is identical across the rows of one model but for the numeric sampler (all primary here);
    # the record keeps one copy per row so every row's compatibility is on file
    return {"schema": SCHEMA, "model_key": key, "wave": rows[0]["wave"], "wave_sha": wave_sha, "published_tree": PUBLISHED_TREE,
            "published_none_manifest": str(man_path), "rows": [r["index"] for r in rows],
            "item_ids_source": ("planned offline at the wave tree (inspect_served.plan_task)" if item_ids is not None
                                else "NOT planned (--no-plan-items): the population was NOT compared — UNDETERMINED"),
            "skipped_checks": [s for s, on in (("item population (--no-plan-items)", item_ids is None),
                                               ("server settings + the wave's sampler rules (--no-compose)", not compose)) if on],
            "shared": shared_by_row, "arm_contracts": contracts, "server_settings": settings, "sampler_resolution": resolution,
            "published_population": pub["population"], "published_sampling_ref_record": pub["sampling_ref_record"],
            "not_carried_by_published_record": [
                ("sampling_ref on the ROWS (rows.jsonl predates the field; the ref is read from gen_stats + the outcome items, "
                 "rebuilt from the manifest's own sampling block, and compared with the wave's rebuilt block)"),
                ("endpoint.json / the vLLM launch argv (the published pull excludes serving/): dtype, quantisation, prefix caching, the "
                 "template-kwargs pin and the generation-config override are COMPOSED at the wave tree and recorded, not compared")],
            "problems": problems, "undetermined": undetermined,
            # FAIL beats UNDETERMINED beats PASS: a PASS means every compared field was READ equal on this machine
            "verdict": "FAIL" if problems else ("UNDETERMINED" if undetermined else "PASS")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--published-root", required=True, help="<published pull>/controls (pull_wave.sh of the published tree)")
    ap.add_argument("--wave-sha", required=True, help="the wave tree's full 40-hex sha (the pin the new legs will carry)")
    ap.add_argument("--out", required=True, help="directory for compat_<key>.json (the wave root's preflight/)")
    ap.add_argument("--wave", default=M.WAVE_EXT1, help="the wave whose rows are preflighted (default ext1)")
    ap.add_argument("--model-key", action="append", default=None, help="restrict to these models (default: every model of the wave)")
    ap.add_argument("--budgets-root", default=None, help="<wave root>/budgets: also re-check each row's fit sidecar at the wave sha")
    ap.add_argument("--no-compose", action="store_true",
                    help="DIAGNOSTIC: skip the server-settings composition — the projection and every omitted sampler key's wave "
                         "rule are then UNDETERMINED (rc 3); the record is partial and never launch-eligible")
    ap.add_argument("--no-plan-items", action="store_true",
                    help="DIAGNOSTIC: skip planning the item set offline — the population is then UNDETERMINED (rc 3), the "
                         "published ids are never echoed as the wave's; the record is partial and never launch-eligible")
    a = ap.parse_args(argv)
    if not M.is_full_sha(a.wave_sha):
        raise SystemExit(f"--wave-sha must be a full 40-hex sha, got {a.wave_sha!r}")
    rows = M.rows_of_wave(a.wave)
    if not rows:
        raise SystemExit(f"--wave {a.wave!r}: no rows (waves: {list(M.WAVES)})")
    keys = a.model_key or [k for k in M.MODEL_KEYS if any(r["model_key"] == k for r in rows)]
    unknown = [k for k in keys if not any(r["model_key"] == k for r in rows)]
    if unknown:
        raise SystemExit(f"--model-key {unknown}: no row of wave {a.wave!r}")
    try:
        # fix A2: the certificate is bound to the tree it reads BEFORE anything is written — HEAD == --wave-sha, clean
        # evidence roots; the resolved HEAD is recorded independently of the argv
        tree = bind_wave_sha(a.wave_sha, tree_facts(REPO))
        published_root = Path(a.published_root)
        if not published_root.is_dir():
            raise Refusal(f"--published-root {published_root} is not a directory")
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        lock = client_lock_facts()
        item_ids = None if a.no_plan_items else plan_item_ids()
        verdicts = {}
        with tempfile.TemporaryDirectory(prefix="m5t-compat-") as td:
            for key in keys:
                rec = preflight_model(key, [r for r in rows if r["model_key"] == key], published_root, a.wave_sha,
                                      Path(a.budgets_root) if a.budgets_root else None, not a.no_compose, item_ids, lock, Path(td))
                rec["tree"] = tree
                path = out / f"compat_{key}.json"
                path.write_text(json.dumps(rec, indent=1, sort_keys=True, default=str) + "\n", encoding="utf-8")
                verdicts[key] = rec["verdict"]
                print(f"COMPAT: model={key} rows={M._array_spec(rec['rows'])} verdict={rec['verdict']} problems={len(rec['problems'])} "
                      f"undetermined={len(rec['undetermined'])} -> {path}")
                for p in rec["problems"]:
                    print(f"  PROBLEM: {p}")
                for u in rec["undetermined"]:
                    print(f"  UNDETERMINED: {u}")
    except Refusal as e:
        print(f"REFUSED: {e.why}")
        return 2
    overall = "FAIL" if any(v == "FAIL" for v in verdicts.values()) else ("UNDETERMINED" if any(v == "UNDETERMINED" for v in verdicts.values()) else "PASS")
    print(f"VERDICT: {overall} models={verdicts}")
    # 0 PASS; 1 FAIL (hold that model's rows); 2 a refused input; 3 UNDETERMINED (a rule this machine cannot read — the
    # remedy is printed per key; re-run once the local cache carries the file)
    return {"PASS": 0, "FAIL": 1, "UNDETERMINED": 3}[overall]


if __name__ == "__main__":
    raise SystemExit(main())
