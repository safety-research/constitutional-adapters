#!/usr/bin/env python3
"""DRAFT PRINTER for ``configs/capeval/mamba_backend_contract.json`` — the REVIEWED reference
contract the Nemotron backend attestation compares runtime discovery against (the
runtime never reads its required set from the module it is testing).

What this script does (CPU only, no vLLM install needed): it opens the PINNED vLLM wheel as a zip
(a URL#sha256=… spec or a local path; the sha256 is VERIFIED before a byte is read), pulls the
source of the NemotronH Mamba2 mixer (``vllm/model_executor/layers/mamba/mamba_mixer2.py``), the
selective-state-update dispatch module (``…/mamba/ops/ssu_dispatch.py``) and the model module
(``vllm/model_executor/models/nemotron_h.py``), and prints a DRAFT contract read from those bytes:

  * for each of the four ROLES a Mamba2 mixer's forward path dispatches through —
    ``prefill_conv`` / ``prefill_scan`` (the chunked prefill: causal conv + SSD chunk scan) and
    ``decode_conv`` / ``decode_scan`` (the single-token decode: conv state update + selective state
    update) — the exact NAME the mixer module binds, the MODULE it imports it from (``from X import
    name`` in the mixer's import block, parsed with ``ast``), the resulting ``entry_point``
    ``X.name``, and the mixer's CALL SITES (line numbers) of that name;
  * for the dispatch role (``ssu_dispatch.selective_state_update`` is a wrapper, not a kernel) the
    dispatch REGISTRY read from ``ssu_dispatch.py``: every ``MambaBackendEnum.<X>: <Backend>`` entry
    and the kernel each backend's ``__init__`` imports (``from M import name as alias``) — recorded
    ALL, approved ONLY the ones named by ``--approve-backend`` (default TRITON: vLLM's own fused
    Triton kernel, the measured one; FlashInfer is a fused kernel too but is neither installed
    in the image nor measured here, so it reads as ``unrecognised`` at runtime — the safe direction);
  * ``approved_impls`` per role: the entry point's own module for the kernel roles, the approved
    backends' kernel modules for the dispatch role;
  * whether the model module binds the mixer class from the mixer module (``nemotron_h_uses_mixer``).

HOW THE CONTRACT IS REVIEWED (the procedure the plan names; the file is a DRAFT until this is done):

  1. ``python scripts/mamba_backend_contract_author.py --wheel '<url>#sha256=<hex>'`` prints the
     draft to stdout (``--out <path>`` writes it) with ``reviewed_by`` / ``reviewed_on`` /
     ``review_ref`` EMPTY;
  2. a human opens the wheel's ``mamba_mixer2.py`` (the ``source_refs`` say where: the import block
     and the four call sites), ``ops/ssu_dispatch.py`` and ``models/nemotron_h.py`` and diffs the
     draft against them line by line — every entry point, every dispatch target, the approved set;
  3. they fill ``reviewed_by`` (who), ``reviewed_on`` (YYYY-MM-DD), ``review_ref`` (the PR /
     decisions entry) and commit the file;
  4. ``python scripts/mamba_backend_contract_author.py --wheel … --check
     configs/capeval/mamba_backend_contract.json`` re-derives the mechanical fields from the wheel and
     exits non-zero on ANY drift — the reviewer runs it before signing, the PR's code review
 reads the contract as part of the diff, and
     ``tests/test_mamba_backend_attestation.py::test_contract_is_reviewed_and_keyed_to_the_image_wheel``
     refuses an unreviewed contract or one whose ``vllm_wheel_sha256`` is not the launcher's
     ``VLLM_WHEEL`` pin — so a vLLM bump forces a fresh draft and a fresh review before any Nemotron
     cell can launch.

Nothing here imports vLLM: a wheel's bytes are the authority, not whatever happens to be installed.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
import re
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

SCHEMA = "sisteer.mamba_backend_contract/1"
MIXER_MODULE = "vllm.model_executor.layers.mamba.mamba_mixer2"
MIXER_CLASS_NAME = "MambaMixer2"
DISPATCH_MODULE = "vllm.model_executor.layers.mamba.ops.ssu_dispatch"
MODEL_MODULE = "vllm.model_executor.models.nemotron_h"
#: the four roles and the NAME the pinned mixer binds for each (the name is the draft's
#: starting point; the module it resolves to is READ from the wheel, never assumed)
ROLE_BINDINGS: Mapping[str, str] = {
    "prefill_conv": "causal_conv1d_fn",
    "prefill_scan": "mamba_chunk_scan_combined_varlen",
    "decode_conv": "causal_conv1d_update",
    "decode_scan": "selective_state_update",
}
#: the role whose entry point is a dispatch WRAPPER (its kernel is chosen at engine init)
DISPATCH_ROLE = "decode_scan"
DEFAULT_APPROVED_BACKENDS = ("TRITON",)
N_MAMBA_LAYERS_RULE = "count of 'M' in the pinned HF config's hybrid_override_pattern"
#: the fields a reviewer fills; the runtime test refuses a contract with any of them empty
REVIEW_FIELDS = ("reviewed_by", "reviewed_on", "review_ref")
#: the fields ``--check`` compares (everything derived from the wheel's bytes)
MECHANICAL_FIELDS = ("schema", "vllm_wheel_sha256", "vllm_version", "model_module", "mixer_module",
                     "mixer_class", "dispatch_module", "n_mamba_layers_rule", "roles",
                     "nemotron_h_uses_mixer", "source_refs")


def _module_path(dotted: str) -> str:
    return dotted.replace(".", "/") + ".py"


def parse_wheel_spec(spec: str) -> Tuple[str, Optional[str]]:
    """``<url-or-path>[#sha256=<hex>]`` -> (location, sha256 or None)."""
    loc, _, frag = spec.partition("#")
    sha = None
    if frag:
        m = re.fullmatch(r"sha256=([0-9a-fA-F]{64})", frag.strip())
        if not m:
            raise ValueError(f"wheel spec fragment must be #sha256=<64 hex>, got {frag!r}")
        sha = m.group(1).lower()
    return loc, sha


def wheel_bytes(spec: str, cache_dir: Optional[Path] = None) -> Tuple[bytes, str]:
    """The wheel's bytes and their sha256; a spec carrying ``#sha256=`` is VERIFIED (a mismatch
    raises before anything is read). A local path is read directly; a URL is fetched (into
    ``cache_dir`` when given, reused when the cached copy's digest matches)."""
    loc, want = parse_wheel_spec(spec)
    if not re.match(r"^[a-z]+://", loc):
        data = Path(loc).read_bytes()
    else:
        cached = None
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
            cached = cache_dir / loc.rsplit("/", 1)[-1]
        if cached is not None and cached.is_file() and (
                want is None or hashlib.sha256(cached.read_bytes()).hexdigest() == want):
            data = cached.read_bytes()
        else:
            with urllib.request.urlopen(loc, timeout=600) as resp:  # a pinned, sha-verified wheel URL
                data = resp.read()
            if cached is not None:
                cached.write_bytes(data)
    got = hashlib.sha256(data).hexdigest()
    if want is not None and got != want:
        raise ValueError(f"wheel sha256 mismatch: spec says {want}, bytes are {got}")
    return data, got


def wheel_version(zf: zipfile.ZipFile) -> str:
    for name in zf.namelist():
        if name.endswith(".dist-info/METADATA"):
            for line in zf.read(name).decode("utf-8", "replace").splitlines():
                if line.startswith("Version:"):
                    return line.split(":", 1)[1].strip()
    raise ValueError("no METADATA Version in the wheel")


# --- pure source readers (text in, facts out) ----------------------------------------------------

def import_bindings(src: str) -> Dict[str, str]:
    """``name -> module`` for every ``from <module> import <name> [as alias]`` at module level
    (the alias is the bound name). Plain ``import x`` binds nothing here."""
    out: Dict[str, str] = {}
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            for alias in node.names:
                out[alias.asname or alias.name] = f"{node.module}.{alias.name}"
    return out


def import_block_lines(src: str) -> Tuple[int, int]:
    """First and last line of the module-level import block."""
    tree = ast.parse(src)
    lines = [(n.lineno, getattr(n, "end_lineno", n.lineno)) for n in tree.body
             if isinstance(n, (ast.Import, ast.ImportFrom))]
    if not lines:
        raise ValueError("no module-level imports")
    return min(a for a, _ in lines), max(b for _, b in lines)


def call_sites(src: str, name: str) -> List[int]:
    """Line numbers where ``name(`` is CALLED (ast Call with a bare Name func)."""
    out: List[int] = []
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == name:
            out.append(node.lineno)
    return sorted(out)


def class_names(src: str) -> List[str]:
    return [n.name for n in ast.parse(src).body if isinstance(n, ast.ClassDef)]


def dispatch_registry(src: str, module: str = DISPATCH_MODULE) -> Dict[str, Dict[str, Any]]:
    """``{ENUM_MEMBER: {"class": "<module>.<Backend>", "kernel": "<kernel module>.<name>"}}`` read
    from the dispatch module: the ``_BACKEND_REGISTRY`` dict literal maps ``MambaBackendEnum.<X>``
    to a backend class; each class's ``__init__`` imports its kernel (``from M import name [as
    alias]``) — that import is the kernel the wrapper forwards to."""
    tree = ast.parse(src)
    classes = {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}
    registry: Dict[str, Dict[str, Any]] = {}
    for node in tree.body:
        # the wheel spells it as an ANNOTATED assignment
        # (``_BACKEND_REGISTRY: dict[MambaBackendEnum, type[MambaSSUBackend]] = {…}``);
        # a bare assignment is accepted too
        if isinstance(node, ast.AnnAssign):
            targets = [node.target.id] if isinstance(node.target, ast.Name) else []
        elif isinstance(node, ast.Assign):
            targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
        else:
            continue
        if "_BACKEND_REGISTRY" not in targets or not isinstance(node.value, ast.Dict):
            continue
        for k, v in zip(node.value.keys, node.value.values):
            if isinstance(k, ast.Attribute) and isinstance(k.value, ast.Name) \
                    and k.value.id == "MambaBackendEnum" and isinstance(v, ast.Name):
                registry[k.attr] = {"class": f"{module}.{v.id}", "kernel": None}
                cls = classes.get(v.id)
                if cls is None:
                    continue
                for fn in cls.body:
                    if isinstance(fn, ast.FunctionDef) and fn.name == "__init__":
                        for sub in ast.walk(fn):
                            if isinstance(sub, ast.ImportFrom) and sub.module:
                                for alias in sub.names:
                                    registry[k.attr]["kernel"] = f"{sub.module}.{alias.name}"
    if not registry:
        raise ValueError("no _BACKEND_REGISTRY dict literal keyed by MambaBackendEnum in the dispatch module")
    return registry


def model_binds_mixer(model_src: str) -> bool:
    """The model module imports the mixer CLASS from the mixer module."""
    return import_bindings(model_src).get(MIXER_CLASS_NAME) == f"{MIXER_MODULE}.{MIXER_CLASS_NAME}"


def draft_from_sources(*, mixer_src: str, dispatch_src: str, model_src: str,
                       wheel_sha256: str, vllm_version: str, wheel_url: Optional[str] = None,
                       approved_backends: Tuple[str, ...] = DEFAULT_APPROVED_BACKENDS,
                       authored_on: str = "") -> Dict[str, Any]:
    """The draft contract from the three modules' SOURCE TEXT (what ``--check`` re-derives)."""
    bindings = import_bindings(mixer_src)
    if MIXER_CLASS_NAME not in class_names(mixer_src):
        raise ValueError(f"{MIXER_MODULE} defines no class {MIXER_CLASS_NAME}")
    registry = dispatch_registry(dispatch_src)
    unknown = sorted(set(approved_backends) - set(registry))
    if unknown:
        raise ValueError(f"--approve-backend names {unknown}, not in the wheel's registry {sorted(registry)}")
    a, b = import_block_lines(mixer_src)
    roles: Dict[str, Any] = {}
    source_refs: Dict[str, str] = {"mixer_imports": f"mamba_mixer2.py:{a}-{b}"}
    for role, name in ROLE_BINDINGS.items():
        entry = bindings.get(name)
        if entry is None:
            raise ValueError(f"the mixer module does not import {name!r} (role {role}) — the roles "
                             "table is stale for this wheel; re-read the mixer before drafting")
        sites = call_sites(mixer_src, name)
        if not sites:
            raise ValueError(f"the mixer module binds {name!r} but never calls it (role {role})")
        entry_module = entry.rsplit(".", 1)[0]
        rec: Dict[str, Any] = {"binding": name, "entry_point": entry, "call_sites": sites}
        if role == DISPATCH_ROLE:
            backends = {
                enum: {**info, "approved": enum in approved_backends}
                for enum, info in sorted(registry.items())}
            rec["dispatch"] = {"registry": f"{DISPATCH_MODULE}._BACKEND_REGISTRY", "backends": backends}
            rec["approved_impls"] = sorted({
                info["kernel"].rsplit(".", 1)[0] for enum, info in registry.items()
                if enum in approved_backends and info["kernel"]})
            rec["note"] = ("the entry point is a dispatch WRAPPER; the kernel it forwards to is the "
                           "backend initialised at engine start (initialize_mamba_ssu_backend) — the "
                           "live worker report names it, and it must be one of approved_impls")
        else:
            rec["approved_impls"] = [entry_module]
        roles[role] = rec
        source_refs[role] = "mamba_mixer2.py:" + ",".join(str(s) for s in sites)
    source_refs["dispatch_registry"] = "ops/ssu_dispatch.py:_BACKEND_REGISTRY"
    source_refs["model_imports_mixer"] = "models/nemotron_h.py:import block"
    return {
        "schema": SCHEMA,
        "vllm_wheel_sha256": wheel_sha256,
        "vllm_wheel_url": wheel_url,
        "vllm_version": vllm_version,
        "model_module": MODEL_MODULE,
        "mixer_module": MIXER_MODULE,
        "mixer_class": f"{MIXER_MODULE}.{MIXER_CLASS_NAME}",
        "dispatch_module": DISPATCH_MODULE,
        "n_mamba_layers_rule": N_MAMBA_LAYERS_RULE,
        "roles": roles,
        "nemotron_h_uses_mixer": model_binds_mixer(model_src),
        "source_refs": source_refs,
        "authored_by": "scripts/mamba_backend_contract_author.py",
        "authored_on": authored_on,
        "reviewed_by": "",
        "reviewed_on": "",
        "review_ref": "",
    }


def draft_from_wheel(spec: str, *, cache_dir: Optional[Path] = None,
                     approved_backends: Tuple[str, ...] = DEFAULT_APPROVED_BACKENDS,
                     authored_on: str = "") -> Dict[str, Any]:
    data, sha = wheel_bytes(spec, cache_dir)
    zf = zipfile.ZipFile(io.BytesIO(data))
    def read(mod: str) -> str:
        return zf.read(_module_path(mod)).decode("utf-8")
    loc, _ = parse_wheel_spec(spec)
    return draft_from_sources(
        mixer_src=read(MIXER_MODULE), dispatch_src=read(DISPATCH_MODULE), model_src=read(MODEL_MODULE),
        wheel_sha256=sha, vllm_version=wheel_version(zf),
        wheel_url=loc if re.match(r"^[a-z]+://", loc) else None,
        approved_backends=approved_backends, authored_on=authored_on)


def mechanical_drift(draft: Mapping[str, Any], committed: Mapping[str, Any]) -> List[str]:
    """Every mechanical field on which the committed contract differs from the draft."""
    out: List[str] = []
    for f in MECHANICAL_FIELDS:
        if draft.get(f) != committed.get(f):
            out.append(f"{f}: draft={json.dumps(draft.get(f), sort_keys=True)[:400]} "
                       f"committed={json.dumps(committed.get(f), sort_keys=True)[:400]}")
    return out


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--wheel", required=True, help="the pinned vLLM wheel: <url|path>[#sha256=<hex>] "
                    "(the launcher's VLLM_WHEEL string; the sha is verified before reading)")
    ap.add_argument("--cache-dir", default=None, help="where a URL wheel is cached (default: no cache)")
    ap.add_argument("--approve-backend", action="append", default=None,
                    help=f"dispatch backend(s) to APPROVE for the {DISPATCH_ROLE} role (default: "
                         f"{', '.join(DEFAULT_APPROVED_BACKENDS)}); every registry entry is recorded regardless")
    ap.add_argument("--authored-on", default="", help="YYYY-MM-DD stamped into the draft")
    ap.add_argument("--out", default=None, help="write the draft here (else stdout)")
    ap.add_argument("--check", default=None, metavar="CONTRACT_JSON",
                    help="re-derive the mechanical fields and exit 1 if the committed contract drifts")
    return ap


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    approved = tuple(a.approve_backend) if a.approve_backend else DEFAULT_APPROVED_BACKENDS
    draft = draft_from_wheel(a.wheel, cache_dir=Path(a.cache_dir) if a.cache_dir else None,
                             approved_backends=approved, authored_on=a.authored_on)
    if a.check:
        committed = json.loads(Path(a.check).read_text(encoding="utf-8"))
        drift = mechanical_drift(draft, committed)
        empty = [f for f in REVIEW_FIELDS if not str(committed.get(f) or "").strip()]
        for d in drift:
            print(f"DRIFT {d}")
        for f in empty:
            print(f"UNREVIEWED {f} is empty")
        print("OK: the committed contract matches the wheel's bytes" if not drift else
              f"{len(drift)} mechanical field(s) drifted")
        return 1 if (drift or empty) else 0
    text = json.dumps(draft, indent=1, sort_keys=False) + "\n"
    if a.out:
        Path(a.out).write_text(text, encoding="utf-8")
        print(f"draft written to {a.out} — review it against the wheel's source and fill "
              f"{', '.join(REVIEW_FIELDS)} (see the module docstring)", file=sys.stderr)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
