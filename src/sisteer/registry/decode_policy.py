"""Decode policy: the harness's ONE sampling authority.

Two halves in one STDLIB-ONLY module:

* the grading host side renders the registry's ``decode_policy`` + ``sampling_mode_policy`` tables (plus the model /
  eval spellings the harness keys on) into the committed export ``configs/registry/decode_policy.json``
  (:func:`export_policy` / :func:`export_json`), deterministic (sorted keys, 1-space indent, trailing newline),
  hash-pinned (``header.policy_set_sha256`` over the body) and stamped with the registry's ``ingest_seq``;
  :func:`check_export` says whether the committed file equals the registry's export;
* the harness side reads ONLY that committed JSON::

      resolve(model_id, eval_id=None, eval_category=None, thinking_mode="off", mode=None) -> Policy

  an eval row wins over a category row; a sampling MODE (``card`` | ``thinking_card`` | ``general`` |
  ``battery`` | ``greedy`` | ``vendor_preset``) maps through ``sampling_mode_policy`` to one row; a model may be
  named by its ``model_id``, a legacy model key (``gemma4e4b``) or its HF repo; a key with no policy raises
  :class:`NoPolicy` (alias :class:`DecodePolicyUnresolved`) — the harness refuses to launch rather than inventing
  numbers. Every production sampling entry point (``contracts.sampling_for``, the protocol rows, the server default,
  BoN's harness default, the misalignment runner's samplers, the multi-turn target's greedy threshold) builds its
  numbers from here; ``sampling_identity()`` hashes VALUES, so equal values keep every store key.

The export is committed TWICE, byte-identical: ``configs/registry/decode_policy.json`` (the registry's export
location, what ``registry decode-policy export`` names) and ``src/sisteer/registry/decode_policy.json`` (package
data beside this module — the copy a ``src``-only tree carries: the Modal mount and the launcher's staged trees ship
``sisteer`` with no ``configs/``, and ``import contracts`` must not fail there). ``export`` writes both, ``check``
verifies both against the registry and each other. The reader takes ``$SISTEER_DECODE_POLICY`` when set, else the
package copy, else the repo copy (found by walking up from this file, as ``sisteer.eval.sampled_decode`` finds its
yaml).
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

EXPORT_VERSION = 1
ENV_VAR = "SISTEER_DECODE_POLICY"
EXPORT_RELPATH = os.path.join("configs", "registry", "decode_policy.json")
#: the package-data copy beside this module (the copy a ``src``-only tree carries)
PACKAGE_EXPORT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "decode_policy.json")
#: kept for the existing callers of that surface; the real default is :func:`default_export_path`
DEFAULT_EXPORT_PATH = EXPORT_RELPATH
PARAM_KEYS = ("temperature", "top_p", "top_k", "min_p", "repetition_penalty", "presence_penalty")
_BODY_SECTIONS = ("policies", "sampling_mode_policy", "models", "model_spellings", "evals", "eval_spellings")


class NoPolicy(LookupError):
    """No decode policy covers the requested key; the harness must refuse to launch."""


DecodePolicyUnresolved = NoPolicy


def _repo_root() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    for _ in range(8):
        if os.path.isfile(os.path.join(here, EXPORT_RELPATH)):
            return here
        parent = os.path.dirname(here)
        if parent == here:
            break
        here = parent
    # fall back to the source layout (src/sisteer/registry -> repo root) so a missing file names the right path
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def repo_export_path() -> str:
    """The registry's export location in the repo (``configs/registry/decode_policy.json``)."""
    return os.path.join(_repo_root(), EXPORT_RELPATH)


def default_export_path() -> str:
    """``$SISTEER_DECODE_POLICY`` > the package copy beside this module > the repo copy."""
    env = os.environ.get(ENV_VAR)
    if env:
        return env
    if os.path.isfile(PACKAGE_EXPORT_PATH):
        return PACKAGE_EXPORT_PATH
    return repo_export_path()


def export_paths() -> List[str]:
    """Every committed copy the export must be written to and checked at: the repo copy and the package copy."""
    return [repo_export_path(), PACKAGE_EXPORT_PATH]


@dataclass(frozen=True)
class Policy:
    policy_id: str
    policy_ref: str
    policy_version: int
    model_id: str
    eval_id: Optional[str]
    eval_category: Optional[str]
    thinking_mode: str
    decode_source: str
    params: Dict[str, object]           # only the parameters the request SENDS
    params_not_sent: Dict[str, object]  # {param: what it resolves to}
    max_new_tokens_rule: str
    seed_rule: str
    source_citation: str
    mode: Optional[str] = None
    is_default: int = 1
    effective_from_utc: str = "unverified"
    superseded_by_policy_id: Optional[str] = None
    ruling_ref: Optional[str] = None
    params_sent: Optional[Tuple[str, ...]] = None
    stated: Optional[Dict[str, object]] = None   # the six sampling numbers the row STATES (None = not stated)

    def request_params(self) -> Dict[str, object]:
        """Exactly the sampling parameters a request carries (un-sent keys are absent, by design)."""
        return dict(self.params)

    def numbers(self) -> Dict[str, object]:
        """The six sampling numbers the row states — what a ``SamplingRow`` carries and what identity hashes."""
        return dict(self.stated or {k: None for k in PARAM_KEYS})


PolicyRow = Policy


def sampling_numbers(policy: Policy) -> Dict[str, object]:
    return policy.numbers()


def sent_and_not_sent(policy: Policy) -> Tuple[Dict[str, object], Dict[str, object]]:
    return dict(policy.params), dict(policy.params_not_sent)


def _policy_from_row(row: Mapping[str, object]) -> Policy:
    stated = {k: row.get(k) for k in PARAM_KEYS}
    sent = row.get("params_sent")
    if isinstance(sent, list):
        sent_keys: Optional[Tuple[str, ...]] = tuple(str(k) for k in sent)
        params = {k: stated[k] for k in PARAM_KEYS if k in sent_keys and stated[k] is not None}
    else:
        sent_keys = None
        # None: a row that names no sent set sends what it states; prose: two producers sent different things —
        # not determinable here, each run's decode_sent_json says what it sent
        params = {k: v for k, v in stated.items() if v is not None} if sent is None else {}
    pns = row.get("params_not_sent_json") or {}
    if isinstance(pns, str):
        pns = json.loads(pns)
    return Policy(str(row["policy_id"]), str(row["policy_ref"]), int(row["policy_version"]), str(row["model_id"]),
                  row.get("eval_id"), row.get("eval_category"), str(row["thinking_mode"]), str(row["decode_source"]),
                  params, dict(pns), str(row["max_new_tokens_rule"]), str(row["seed_rule"]), str(row["source_citation"]),
                  mode=row.get("mode"), is_default=int(row.get("is_default", 1)), effective_from_utc=str(row.get("effective_from_utc", "unverified")),
                  superseded_by_policy_id=row.get("superseded_by_policy_id"), ruling_ref=row.get("ruling_ref"),
                  params_sent=sent_keys, stated=stated)


def _canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def policy_set_sha256(data: Mapping[str, object]) -> str:
    body = {k: data.get(k) for k in _BODY_SECTIONS}
    return hashlib.sha256(_canonical(body).encode("utf-8")).hexdigest()


def _site_forms():
    """The package's site-placeholder helper, or None when this module was loaded on its own (by path, or in a package that
    ships only the decode-policy files, as the harness images do): such a tree has no site module to expand with. Imported
    here rather than at module level so that loading the module alone never fails."""
    try:
        from . import site_forms
    except ImportError:
        return None
    return site_forms


class PolicyFile:
    def __init__(self, data: Mapping[str, object], path: str = "<memory>"):
        if data.get("export_version") != EXPORT_VERSION:
            raise ValueError(f"decode_policy.json export_version {data.get('export_version')!r} != {EXPORT_VERSION}")
        self.data = data
        self.path = path
        self.policies: List[Dict[str, object]] = [dict(p) for p in data.get("policies", [])]
        self.modes: List[Dict[str, object]] = [dict(m) for m in data.get("sampling_mode_policy", [])]
        self.by_id: Dict[str, Dict[str, object]] = {str(p["policy_id"]): p for p in self.policies}
        self.models: Dict[str, Dict[str, object]] = dict(data.get("models") or {})
        self.model_spellings: Dict[str, str] = dict(data.get("model_spellings") or {})
        # a spelling in site-placeholder form ($SHARED/...) also answers to its expansion on this host, the literal path the
        # site's runs recorded; a spelling the export names explicitly keeps its own model
        forms = _site_forms() if any("$" in sp for sp in self.model_spellings) else None
        for sp, mid in list(self.model_spellings.items()):
            lit = forms.expansion(sp) if forms is not None else None
            if lit is not None:
                self.model_spellings.setdefault(lit, mid)
        self.evals: Dict[str, str] = dict(data.get("evals") or {})
        self.eval_spellings: Dict[str, str] = dict(data.get("eval_spellings") or {})
        self.header: Dict[str, object] = dict(data.get("header") or {})

    @classmethod
    def load(cls, path: Optional[str] = None) -> "PolicyFile":
        path = path or default_export_path()
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return cls(json.load(fh), path)
        except FileNotFoundError:
            raise NoPolicy(f"decode policy export not found at {path} (set {ENV_VAR} or run `registry decode-policy export`)") from None

    # -- naming -------------------------------------------------------------------------------

    def model_id(self, name: str) -> str:
        """A model_id, a legacy model key or an HF repo -> model_id (the name itself when nothing maps it)."""
        if name in self.models:
            return name
        return self.model_spellings.get(str(name), str(name))

    def eval_id(self, name: str) -> str:
        if name in self.evals:
            return name
        return self.eval_spellings.get(str(name), str(name))

    def eval_category_of(self, eval_id: str) -> Optional[str]:
        return self.evals.get(self.eval_id(eval_id))

    # -- lookup -------------------------------------------------------------------------------

    def find(self, model_id: Optional[str] = None, eval_id: Optional[str] = None, eval_category: Optional[str] = None,
             thinking_mode: Optional[str] = None, mode: Optional[str] = None, live_only: bool = True) -> List[Policy]:
        """Every policy row matching the given fields (None = any); superseded rows excluded unless ``live_only=False``."""
        mid = self.model_id(model_id) if model_id is not None else None
        eid = self.eval_id(eval_id) if eval_id is not None else None
        out = []
        for p in self.policies:
            if mid is not None and p["model_id"] != mid:
                continue
            if eid is not None and p.get("eval_id") != eid:
                continue
            if eval_category is not None and p.get("eval_category") != eval_category:
                continue
            if thinking_mode is not None and p["thinking_mode"] != thinking_mode:
                continue
            if mode is not None and p.get("mode") != mode:
                continue
            if live_only and p.get("superseded_by_policy_id"):
                continue
            out.append(_policy_from_row(p))
        return out

    def by_ref(self, ref: str, version: Optional[int] = None) -> Policy:
        """The row named by ``policy_ref`` (the default version when several and none is asked for)."""
        if "@v" in ref and version is None:
            ref, _, v = ref.rpartition("@v")
            version = int(v)
        rows = [p for p in self.policies if p["policy_ref"] == ref]
        if version is not None:
            rows = [p for p in rows if int(p["policy_version"]) == version]
        if len(rows) > 1:
            rows = [p for p in rows if int(p.get("is_default", 1)) == 1] or rows
        if len(rows) != 1:
            raise NoPolicy(f"policy_ref {ref!r}" + (f"@v{version}" if version else "") + f" names {len(rows)} rows")
        return _policy_from_row(rows[0])

    def policy(self, policy_id: str) -> Policy:
        p = self.by_id.get(str(policy_id))
        if p is None:
            raise NoPolicy(f"no policy_id {policy_id!r} in {self.path}")
        return _policy_from_row(p)

    def resolve(self, model_id: str, eval_id: Optional[str] = None, eval_category: Optional[str] = None,
                thinking_mode: str = "off", mode: Optional[str] = None) -> Policy:
        mid = self.model_id(model_id)
        if mode is not None:
            for m in self.modes:
                if m["model_id"] == mid and m["sampling_mode"] == mode:
                    p = self.by_id.get(str(m["policy_id"]))
                    if p is None:
                        raise NoPolicy(f"sampling mode {mode!r} for {mid} names policy {m['policy_id']} which is not exported")
                    return _policy_from_row(p)
            raise NoPolicy(f"no sampling_mode_policy row for model {model_id!r} ({mid}) mode {mode!r}")
        eid = self.eval_id(eval_id) if eval_id else None
        if eid and eval_category is None:
            eval_category = self.evals.get(eid)
        live = [p for p in self.policies if p["model_id"] == mid and p["thinking_mode"] == thinking_mode
                and not p.get("superseded_by_policy_id") and int(p.get("is_default", 1)) == 1]
        if eid:
            for p in live:
                if p.get("eval_id") == eid:
                    return _policy_from_row(p)
        if eval_category:
            for p in live:
                if p.get("eval_category") == eval_category:
                    return _policy_from_row(p)
        raise NoPolicy(f"no decode policy for model {model_id!r} ({mid}) eval {eval_id!r} category {eval_category!r} thinking {thinking_mode!r}")

    # -- entry-point helpers ------------------------------------------------------------------

    def mode_rows(self, mode: str) -> Dict[str, Policy]:
        """``{legacy model key: Policy}`` for one sampling mode — the shape ``contracts.SAMPLING_TABLE_BY_MODE`` has."""
        out: Dict[str, Policy] = {}
        for m in self.modes:
            if m["sampling_mode"] != mode:
                continue
            key = str(m.get("model_key") or m["model_id"])
            p = self.by_id.get(str(m["policy_id"]))
            if p is None:
                raise NoPolicy(f"sampling mode {mode!r} for {key} names policy {m['policy_id']} which is not exported")
            out[key] = _policy_from_row(p)
        return out

    def model_independent(self, eval_id: Optional[str] = None, eval_category: Optional[str] = None,
                          thinking_mode: str = "off", mode: Optional[str] = None, models: Optional[Iterable[str]] = None) -> Policy:
        """ONE row for every served model (GREEDY, the BoN harness default, a protocol row): the matching rows of
        every model must state identical numbers and send the same set, or this refuses naming the disagreement."""
        rows = self.find(None, eval_id, eval_category, thinking_mode, mode, live_only=False)
        if models is not None:
            wanted = {self.model_id(m) for m in models}
            rows = [r for r in rows if r.model_id in wanted]
        if not rows:
            raise NoPolicy(f"no policy for eval {eval_id!r} category {eval_category!r} thinking {thinking_mode!r} mode {mode!r} on any model")
        first = rows[0]
        for r in rows[1:]:
            if r.numbers() != first.numbers() or r.request_params() != first.request_params():
                raise NoPolicy(f"model-independent lookup for eval {eval_id!r} / {eval_category!r} / mode {mode!r} disagrees between "
                               f"{first.policy_id} {first.numbers()} and {r.policy_id} {r.numbers()}")
        return first

    def protocol_row(self, task: str) -> Policy:
        """The per-TASK protocol row (``inspect/bigcodebench_200`` -> eval bigcodebench): model-independent by construction."""
        eid = self.eval_id(task)
        if eid not in self.evals:
            raise NoPolicy(f"task {task!r} maps to no eval (eval_spellings)")
        return self.model_independent(eval_id=eid, thinking_mode="off", mode="protocol")

    def server_default_rows(self) -> Dict[str, Policy]:
        """``{model_id: Policy}`` of the composed vLLM server defaults (the `omit_params` route rows)."""
        return {p.model_id: p for p in self.find(eval_category="omit_params", thinking_mode="off", mode="server_default")}

    def hf_repo(self, model_id: str) -> Optional[str]:
        m = self.models.get(self.model_id(model_id)) or {}
        return m.get("hf_repo")

    def hf_revision(self, model_id: str) -> Optional[str]:
        """The model's pinned HF revision as the registry states it (40 hex, or ``unverified``); None when the export predates the field."""
        m = self.models.get(self.model_id(model_id)) or {}
        return m.get("hf_revision")


_CACHE: Dict[str, Tuple[Tuple[float, int], PolicyFile]] = {}


def load(path: Optional[str] = None) -> PolicyFile:
    """The export, cached per path until the file changes (mtime, size)."""
    path = path or default_export_path()
    try:
        st = os.stat(path)
    except FileNotFoundError:
        raise NoPolicy(f"decode policy export not found at {path} (set {ENV_VAR} or run `registry decode-policy export`)") from None
    key = (st.st_mtime, st.st_size)
    hit = _CACHE.get(path)
    if hit and hit[0] == key:
        return hit[1]
    pf = PolicyFile.load(path)
    _CACHE[path] = (key, pf)
    return pf


def resolve(model_id: str, eval_id: Optional[str] = None, eval_category: Optional[str] = None, thinking_mode: str = "off",
            mode: Optional[str] = None, path: Optional[str] = None) -> Policy:
    return load(path).resolve(model_id, eval_id, eval_category, thinking_mode, mode)


# --------------------------------------------------------------------------------------------
# export (the grading host side; stdlib sqlite3)
# --------------------------------------------------------------------------------------------

_POLICY_COLS = ("policy_id", "policy_ref", "policy_version", "model_id", "eval_id", "eval_category", "thinking_mode", "decode_source",
                "temperature", "top_p", "top_k", "min_p", "repetition_penalty", "presence_penalty", "max_new_tokens_rule", "seed_rule",
                "params_not_sent_json", "source_citation", "ruling_id", "is_default", "effective_from_utc", "superseded_by_policy_id",
                "notes", "extra_json")
#: extra_json keys lifted to the top level of an exported policy (references and labels; the numbers are columns)
_POLICY_EXTRA_KEYS = ("mode", "params_sent", "same_values_as", "legacy_contracts_row", "deliberate_departure",
                      "wildcard_expanded_from", "effective_from_confidence", "seed_decode_source", "superseded_by_unresolved")


def _json(v):
    if isinstance(v, str):
        try:
            return json.loads(v)
        except json.JSONDecodeError:
            return v
    return v


def export_policy(conn) -> Dict[str, object]:
    # imported here, not at module level: this file is read STANDALONE by the harness trees (spec_from_file_location, a package of
    # __init__ + this module + the json), where a relative import has no parent package; only the export needs the vocabulary
    from . import vocab as V
    pol = []
    for r in conn.execute(f"SELECT {', '.join(_POLICY_COLS)} FROM decode_policy ORDER BY policy_ref, policy_version").fetchall():
        d = {c: r[i] for i, c in enumerate(_POLICY_COLS)}
        d["params_not_sent_json"] = _json(d.get("params_not_sent_json")) or {}
        extra = _json(d.pop("extra_json")) or {}
        for k in _POLICY_EXTRA_KEYS:
            d[k] = extra.get(k)
        if d.get("mode") is None:
            segs = str(d["policy_ref"]).split("|")
            d["mode"] = segs[3] if len(segs) == 4 else None
        d["ruling_ref"] = extra.get("decode_ruling_ref") or d.get("ruling_id")
        pol.append(d)
    modes = []
    for r in conn.execute("SELECT model_id, sampling_mode, policy_id, extra_json FROM sampling_mode_policy ORDER BY model_id, sampling_mode").fetchall():
        extra = _json(r[3]) or {}
        modes.append({"model_id": r[0], "sampling_mode": r[1], "policy_id": r[2], "model_key": extra.get("model_key_legacy") or r[0]})
    models: Dict[str, Dict[str, object]] = {}
    # a spelling two models claim belongs to the one whose paper_status ranks first (vocab.model_spelling_owner: the paper's
    # checkpoint owns its names; the worked example qwen35_4b_demo re-pins the 4B's repos without capturing them), and two
    # claimants of the same rank drop it -- the rule adapters/dimensions.legacy_model_map applies to the seeds' map
    claims: Dict[str, Dict[str, Optional[str]]] = {}
    for r in conn.execute("SELECT model_id, hf_repo, checkpoint_kind, paper_status, extra_json, hf_revision FROM model ORDER BY model_id").fetchall():
        mid = str(r[0])
        # hf_revision (2026-10-09): the pinned snapshot beside the repo, so a reader can tell two model_ids that re-pin ONE checkpoint
        # (the worked example qwen35_4b_demo on the paper's Qwen3.5-4B) from two models that share a repo spelling at different pins
        models[mid] = {"hf_repo": r[1], "hf_revision": r[5], "checkpoint_kind": r[2], "paper_status": r[3]}
        extra = _json(r[4]) or {}
        for sp in [r[1]] + list(extra.get("producer_spellings") or []):
            claims.setdefault(str(sp), {})[mid] = r[3]
    spellings: Dict[str, str] = {}
    for sp in sorted(claims):
        owner = V.model_spelling_owner(claims[sp])
        if owner is not None:
            spellings[sp] = owner
    evals: Dict[str, str] = {}
    eval_spellings: Dict[str, str] = {}
    eclash = set()
    for r in conn.execute("SELECT eval_id, eval_category, producer_spellings_json FROM eval ORDER BY eval_id").fetchall():
        evals[str(r[0])] = str(r[1])
        for sp in _json(r[2]) or []:
            sp = str(sp)
            if sp in eval_spellings and eval_spellings[sp] != str(r[0]):
                eclash.add(sp)
            eval_spellings[sp] = str(r[0])
    for sp in eclash:
        eval_spellings.pop(sp, None)
    seq = conn.execute("SELECT COALESCE(MAX(ingest_seq), 0) FROM record_ingest").fetchone()[0]
    data: Dict[str, object] = {"export_version": EXPORT_VERSION, "policies": pol, "sampling_mode_policy": modes, "models": models,
                               "model_spellings": spellings, "evals": evals, "eval_spellings": eval_spellings}
    data["header"] = {"exported_by": "registry decode-policy export", "ingest_seq": int(seq), "n_policies": len(pol),
                      "n_sampling_modes": len(modes), "policy_set_sha256": policy_set_sha256(data)}
    return data


def export_json(conn) -> str:
    return json.dumps(export_policy(conn), indent=1, sort_keys=True) + "\n"


def export_diff(conn, path: Optional[str] = None) -> List[str]:
    """Why the committed file differs from the registry's export (empty = equal): body sections and the sha pin;
    the header's ingest_seq is informational and never a difference."""
    path = path or default_export_path()
    if not os.path.exists(path):
        return [f"missing: {path}"]
    with open(path, "r", encoding="utf-8") as fh:
        committed = json.load(fh)
    live = export_policy(conn)
    out = []
    if committed.get("export_version") != live["export_version"]:
        out.append("export_version differs")
    for k in _BODY_SECTIONS:
        if _canonical(committed.get(k)) != _canonical(live.get(k)):
            out.append(f"section {k!r} differs")
    pin = (committed.get("header") or {}).get("policy_set_sha256")
    if pin != policy_set_sha256(committed):
        out.append("header.policy_set_sha256 does not pin the committed body")
    return out


def check_export(conn, path: Optional[str] = None) -> bool:
    """True when the committed file equals the registry's export (body + sha pin; ingest_seq informational)."""
    return not export_diff(conn, path)


def write_export(conn, paths: Optional[Sequence[str]] = None) -> Dict[str, object]:
    """Render the export once and write it to every path (default: both committed copies); returns the header."""
    data = export_policy(conn)
    text = json.dumps(data, indent=1, sort_keys=True) + "\n"
    for path in (paths or export_paths()):
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
    return dict(data["header"])


def check_all(conn, paths: Optional[Sequence[str]] = None) -> List[str]:
    """Every committed copy equals the registry's export AND the copies equal each other; empty = all good."""
    paths = list(paths or export_paths())
    out = []
    for p in paths:
        out += [f"{p}: {d}" for d in export_diff(conn, p)]
    texts = {}
    for p in paths:
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as fh:
                texts[p] = fh.read()
    if len(set(texts.values())) > 1:
        out.append("the committed copies differ from each other: " + " != ".join(sorted(texts)))
    return out
