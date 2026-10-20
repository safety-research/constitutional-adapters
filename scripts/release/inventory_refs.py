#!/usr/bin/env python3
"""Reference inventory for the removed tiny loglikelihood battery and its third-party harness.

The battery (``tiny_ll``: five tinyBenchmarks tasks scored by continuation loglikelihood) and the
tier-1 harness it ran on (lm-eval, its vendored task files, the in-process vLLM lane and the HF
spot-check) were removed. This tool finds every remaining reference and gives each one a class, so
a reviewer can see that nothing live still depends on the removed code.

It scans every tracked text file for each identifier in :data:`IDENTIFIERS`, matched exactly (case
sensitive, at identifier boundaries), and classifies every hit:

* ``keep-history``   the registry's record of the dropped eval, measured windows and other records of
                     what earlier runs did
* ``keep-identity``  a value hashed into an identity that historical records carry
* ``keep-synthetic`` a synthetic fixture or test row that uses a registered name
* ``keep-guard``     a test or tool that asserts the removal
* ``keep-prose``     a mention of the external harness as the origin of a convention
* ``edit``/``delete`` (only with ``--target``) the hit's line, or its whole file, is gone from the
                     target tree
* ``unclassified``   none of the above, which ``--check`` refuses

Usage::

    python scripts/release/inventory_refs.py                        # the working tree, a table on stdout
    python scripts/release/inventory_refs.py --check                # exit 1 unless every hit is keep-*
    python scripts/release/inventory_refs.py --rev <base> --target <branch> --markdown out.md --tsv out.tsv

``--rev`` scans a committed tree instead of the working tree. ``--target`` classifies the scanned
tree's hits against another tree (a hit whose exact line is absent there is ``edit``, whose file is
absent ``delete``), which is how the before/after table of the removal is made.
"""
from __future__ import annotations

import argparse
import fnmatch
import re
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

#: Every identifier of the removed lane. The first group is the plan's list; the second covers the
#: rest of what went with it (the harness's pins, the in-process vLLM lane, the HF spot-check).
IDENTIFIERS: Tuple[str, ...] = (
    "CONTROL_TINY_LL", "TINY_LL_TASKS", "SERVED_LL_TASKS", "served_ll_task", "GEOMETRY_CLASS_TINY",
    "served_tiny", "lmeval_runner", "TINY_LOADER_CODE", "lm_eval", "lm_eval_model", "make_hflm",
    "tinybenchmarks", "tinyBenchmarks", "tinyMMLU", "tinyArc", "tinyHellaswag", "tinyWinogrande",
    "tinyTruthfulQA", "tier1.yaml", "served_ll/", "tiny_ll",
    # the rest of the removed lane
    "lm-eval", "LM_EVAL_PIN", "TINYBENCHMARKS_PIN", "TINYBENCHMARKS_REV", "TINYBENCHMARKS_GIT",
    "FRAMEWORK_VCS_PINS", "SERVED_LL_TASK_PREFIX", "SERVED_LL_TASK_VERSION", "PREFILL_SCORED_TASKS",
    "EVIDENCE_CONTINUATION_BITES_KEY", "capeval_tiny_utils", "hf_spotcheck", "hf-spotcheck",
    "CONTAINER_HF_SPOTCHECK", "vllm_inproc", "VllmInprocEngine", "capeval_vllm_unavailable",
    "capeval_vllm.sbatch", "vllm_base.yaml",
)

KEEP_CLASSES = ("keep-history", "keep-identity", "keep-synthetic", "keep-guard", "keep-prose")
CLASSES = KEEP_CLASSES + ("edit", "delete", "unclassified")

#: Lines of a guard test that assert the removal rather than use the removed names.
_GUARD_LINE = (r"not hasattr|hasattr\(|not in |not any\(|pytest\.raises|ControlNameError|removed|gone|"
               r"no (?:loglikelihood|arm)|heavy = |BLOCK = |Base lane only|for bad in ")


@dataclass(frozen=True)
class Rule:
    cls: str
    paths: Tuple[str, ...]
    reason: str
    line: Optional[str] = None      # a regex the line must match, when the path alone is too broad

    def matches(self, path: str, text: str) -> bool:
        if not any(fnmatch.fnmatchcase(path, p) for p in self.paths):
            return False
        return self.line is None or re.search(self.line, text) is not None


#: First match wins. Paths are repository-relative globs (``*`` crosses ``/``).
RULES: Tuple[Rule, ...] = (
    # the tool and the tests that prove the removal
    Rule("keep-guard", ("scripts/release/inventory_refs.py", "tests/test_tinycap_retained_controls.py"),
         "the inventory itself and the retained-controls test name the removed identifiers to refuse them"),
    Rule("keep-guard", ("tests/test_tinycap_contracts.py", "tests/test_tinycap_client_lock.py",
                        "tests/test_served_capability.py", "tests/test_m5t_compat_preflight.py",
                        "tests/test_tinycap_launcher.py", "tests/test_capeval_runtime_dispatch.py",
                        "tests/test_served_geometry.py", "tests/test_capeval_driver.py"),
         "asserts the removed name is absent or refused", line=_GUARD_LINE),
    Rule("keep-guard", ("tests/test_tinycap_contracts.py",),
         "asserts a removed row task has no samples-per-item entry", line=r'samples_of_task\(card, "served_ll/'),
    Rule("keep-guard", ("tests/test_served_geometry.py",),
         "asserts an old ledger's loglikelihood rows pool into no geometry class", line=r"served_ll/"),
    # the registry's own history of the dropped eval
    Rule("keep-history", ("configs/registry/*", "src/sisteer/registry/decode_policy.json",
                          "src/sisteer/registry/schema.sql", "src/sisteer/registry/migrations/*",
                          "src/sisteer/registry/vocab.py", "docs/registry/vocabulary.md",
                          "src/sisteer/registry/adapters/roster.py", "tests/test_registry_roster_transcribe.py"),
         "the registry records tiny_ll as a dropped eval and lm_eval as a historical harness"),
    Rule("keep-history", ("scripts/sdfbase/full22/wave_manifest.json", "configs/sdfbase/eight_arms/tree_set_review_*.md"),
         "a recorded manifest or review of an earlier tree"),
    Rule("keep-history", ("src/sisteer/capeval/serving/geometry.py", "docs/served_geometry_p99.md"),
         "the measured serving windows of the earlier tiny passes, kept as their record"),
    Rule("keep-history", ("src/sisteer/capeval/tinycap/contracts.py",),
         "the letter-read width was measured on tinyMMLU items; the tiny geometry class stays registered",
         line=r"tinyMMLU|GEOMETRY_CLASS_TINY"),
    Rule("keep-history", ("tests/test_tinycap_contracts.py",),
         "the letter-read width was measured on tinyMMLU items", line=r"tinyMMLU items|unsteered tinyMMLU"),
    Rule("keep-history", ("tests/test_tinycap_launcher.py",),
         "the retired HellaSwag leg's frozen item count came from the tinyHellaswag items",
         line=r"tinyHellaswag items"),
    Rule("keep-history", ("configs/capeval/domains.yaml", "tests/test_capeval_domains.py"),
         "the capeval domain registry still lists the tier-1 tasks and the served_ll domain, so stores "
         "that filed rows under them consolidate; no producer files them now"),
    Rule("keep-history", ("src/sisteer/capeval/report/build.py",),
         "renders the lm-eval instrument deltas that earlier fingerprints carry"),
    Rule("keep-history", ("src/sisteer/capeval/store.py",),
         "documents the item-id collision across tasks these rows caused", line=r"tinyMMLU|tinyArc"),
    Rule("keep-history", ("tests/fixtures/tinycap_identity_laneA/*", "tests/fixtures/tinycap_identity_v1/*",
                          "tests/fixtures/tinycap_row_engine/*"),
         "recorded identities and manifests of earlier stores, read back as written"),
    Rule("keep-history", ("tests/test_tinycap_identity_version.py",),
         "records which golden identity leaves the removal moved", line=r"frameworks\.(lm_eval|tinybenchmarks)"),
    Rule("keep-history", ("tests/tinycap_golden.py",),
         "records how the removal moved the geometry golden", line=r"dropped ``tiny_ll`` from their controls"),
    # identities that historical records carry
    Rule("keep-identity", ("src/sisteer/capeval/store.py", "tests/test_capeval_store.py",
                           "tests/test_env_digest_attention_backend.py"),
         "the environment fingerprint hashes lm_eval's version when installed, so earlier fingerprints recompute",
         line=r"lm_eval|lm-eval"),
    # synthetic fixtures and test rows
    Rule("keep-synthetic", ("src/sisteer/registry/synth.py", "tests/test_registry_identity.py"),
         "a synthetic instrument with a historical harness name"),
    Rule("keep-synthetic", ("tests/test_capeval_driver.py", "tests/test_capeval_power.py",
                            "tests/test_capeval_protocol.py", "tests/test_capeval_report.py",
                            "tests/test_capeval_runtime.py", "tests/test_capeval_stats.py",
                            "tests/test_capeval_store.py"),
         "synthetic rows under a task name the capeval domain registry still lists"),
    # the external harness named as the origin of a convention
    Rule("keep-prose", ("src/sisteer/capeval/serving/prefill_logprob.py",),
         "names lm-eval as the origin of the continuation and length-normalisation conventions",
         line=r"lm-eval"),
    Rule("keep-prose", ("scripts/sdfbase/base_gsm8k.py",),
         "names the external harness whose GSM8K prompt convention the file follows", line=r"lm-eval"),
)


@dataclass(frozen=True)
class Hit:
    path: str
    lineno: int
    text: str
    tokens: Tuple[str, ...]


def _token_re(tok: str) -> re.Pattern:
    left = r"(?<![A-Za-z0-9_])" if re.match(r"\w", tok) else ""
    right = r"(?![A-Za-z0-9_])" if re.search(r"\w$", tok) else ""
    return re.compile(left + re.escape(tok) + right)


TOKEN_RES = tuple((t, _token_re(t)) for t in IDENTIFIERS)
_ANY = re.compile("|".join(re.escape(t) for t in IDENTIFIERS))


def _git(repo: Path, *args: str, stdin: Optional[bytes] = None) -> bytes:
    return subprocess.run(["git", "-C", str(repo), *args], input=stdin, check=True,
                          capture_output=True).stdout


def _rev_files(repo: Path, rev: str) -> Dict[str, bytes]:
    """path -> blob bytes of every regular file in ``rev``'s tree."""
    listing = _git(repo, "ls-tree", "-r", "-z", "--full-tree", rev).split(b"\0")
    entries = []
    for ent in listing:
        if not ent:
            continue
        meta, path = ent.split(b"\t", 1)
        mode, kind, sha = meta.split()
        if kind == b"blob" and mode in (b"100644", b"100755"):
            entries.append((path.decode(), sha.decode()))
    if not entries:
        return {}
    out = _git(repo, "cat-file", "--batch", stdin=("\n".join(s for _, s in entries) + "\n").encode())
    blobs: Dict[str, bytes] = {}
    pos = 0
    for path, sha in entries:
        nl = out.index(b"\n", pos)
        header = out[pos:nl].split()
        size = int(header[2])
        blobs[path] = out[nl + 1:nl + 1 + size]
        pos = nl + 1 + size + 1
    return blobs


def _worktree_files(repo: Path) -> Dict[str, bytes]:
    """path -> bytes of every tracked or untracked-but-not-ignored file present in the working tree
    (every file under ``repo`` but ``.git`` when ``repo`` is not a git checkout)."""
    probe = subprocess.run(["git", "-C", str(repo), "rev-parse", "--is-inside-work-tree"],
                           capture_output=True, text=True)
    if probe.returncode != 0 or probe.stdout.strip() != "true":
        return {p.relative_to(repo).as_posix(): p.read_bytes() for p in sorted(repo.rglob("*"))
                if p.is_file() and not p.is_symlink() and ".git" not in p.relative_to(repo).parts}
    names = _git(repo, "ls-files", "-z", "--cached", "--others", "--exclude-standard").split(b"\0")
    out: Dict[str, bytes] = {}
    for raw in names:
        if not raw:
            continue
        p = repo / raw.decode()
        if p.is_file() and not p.is_symlink():
            out[raw.decode()] = p.read_bytes()
    return out


def scan(files: Dict[str, bytes]) -> Iterator[Hit]:
    for path in sorted(files):
        data = files[path]
        if b"\0" in data[:8192] or not _ANY.search(data.decode("utf-8", "replace")):
            continue
        for i, line in enumerate(data.decode("utf-8", "replace").splitlines(), 1):
            if not _ANY.search(line):
                continue
            toks = tuple(t for t, rx in TOKEN_RES if rx.search(line))
            if toks:
                yield Hit(path, i, line, toks)


def classify(hit: Hit, target: Optional[Dict[str, bytes]] = None,
             target_lines: Optional[Dict[str, set]] = None) -> Tuple[str, str]:
    if target is not None:
        if hit.path not in target:
            return "delete", "the file is deleted"
        lines = target_lines.setdefault(hit.path, set(target[hit.path].decode("utf-8", "replace").splitlines()))
        if hit.text not in lines:
            return "edit", "the line is changed or removed"
    for rule in RULES:
        if rule.matches(hit.path, hit.text):
            return rule.cls, rule.reason
    return "unclassified", "no rule covers this reference"


def inventory(repo: Path, rev: Optional[str] = None, target: Optional[str] = None
              ) -> List[Tuple[Hit, str, str]]:
    files = _rev_files(repo, rev) if rev else _worktree_files(repo)
    tfiles = None
    if target is not None:
        tfiles = _worktree_files(repo) if target == "WORKTREE" else _rev_files(repo, target)
    tlines: Dict[str, set] = {}
    return [(h, *classify(h, tfiles, tlines)) for h in scan(files)]


def markdown(rows: Sequence[Tuple[Hit, str, str]], title: str) -> str:
    by_cls: Dict[str, Dict[str, List[Hit]]] = defaultdict(lambda: defaultdict(list))
    reasons: Dict[Tuple[str, str], str] = {}
    for hit, cls, reason in rows:
        by_cls[cls][hit.path].append(hit)
        reasons.setdefault((cls, hit.path), reason)
    out = [f"### {title}", "", "| class | lines | files |", "|---|---|---|"]
    for cls in CLASSES:
        if cls in by_cls:
            out.append(f"| {cls} | {sum(len(v) for v in by_cls[cls].values())} | {len(by_cls[cls])} |")
    for cls in CLASSES:
        if cls not in by_cls:
            continue
        out += ["", f"**{cls}**", "", "| file | lines | identifiers | why |", "|---|---|---|---|"]
        for path, hits in sorted(by_cls[cls].items()):
            toks = Counter(t for h in hits for t in h.tokens)
            ids = ", ".join(f"`{t}`" for t, _ in toks.most_common(4))
            out.append(f"| `{path}` | {len(hits)} | {ids} | {reasons[(cls, path)]} |")
    return "\n".join(out) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    ap.add_argument("--rev", help="scan this committed tree instead of the working tree")
    ap.add_argument("--target", help="classify against this tree (a rev, or WORKTREE)")
    ap.add_argument("--tsv", type=Path, help="write every hit as a TSV row")
    ap.add_argument("--markdown", type=Path, help="write the summary table as markdown")
    ap.add_argument("--check", action="store_true", help="exit 1 unless every hit is keep-*")
    args = ap.parse_args(argv)
    rows = inventory(args.repo, args.rev, args.target)
    counts = Counter(cls for _, cls, _ in rows)
    if args.tsv:
        with args.tsv.open("w", encoding="utf-8") as f:
            f.write("class\tpath\tline\tidentifiers\treason\ttext\n")
            for h, cls, reason in rows:
                text = h.text.strip().replace("\t", " ")[:200]
                f.write(f"{cls}\t{h.path}\t{h.lineno}\t{','.join(h.tokens)}\t{reason}\t{text}\n")
    title = f"References in {args.rev or 'the working tree'}" + (f", classified against {args.target}"
                                                                   if args.target else "")
    md = markdown(rows, title)
    if args.markdown:
        args.markdown.write_text(md, encoding="utf-8")
    print(", ".join(f"{c}={counts[c]}" for c in CLASSES if counts[c]) or "no references")
    bad = [(h, c) for h, c, _ in rows if c not in KEEP_CLASSES]
    for h, c in bad[:200]:
        print(f"{c}\t{h.path}:{h.lineno}\t{h.text.strip()[:160]}")
    if args.check and bad:
        print(f"FAIL: {len(bad)} reference(s) are not keep-*", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
