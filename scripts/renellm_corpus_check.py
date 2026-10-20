"""Check a ReNeLLM corpus file before a job stands anything up: is it the paper's frozen corpus,
or the non-paper corpus that SISTEER_RENELLM_NONPAPER_CORPUS_SHA acknowledges?

    python3 scripts/renellm_corpus_check.py [path/to/renellm.jsonl] [--json]

Prints one identity line (hashes and counts only, never a prompt), or the identity as JSON with
``--json``. Exit 0: the corpus may replay (a non-paper one is labelled as such). Exit 2: refused,
with the reason on stderr. Exit 1: no file at the path. The default path is the one replay reads:
the site's copy (``$SISTEER_HELDOUT_CORPUS_DIR/renellm.jsonl``) when it exists, else the paper's
corpus as the repository ships it (``data/heldout_corpora/renellm.jsonl``). Standard library only,
so the cluster job scripts run it under the system python (`python3 -I -S`) before any venv exists.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("corpus", nargs="?", default=None,
                    help="the corpus file (default: the replay path: the site's copy under "
                         "SISTEER_HELDOUT_CORPUS_DIR, else the repository's copy)")
    ap.add_argument("--json", action="store_true", help="print the identity as JSON")
    args = ap.parse_args(argv)

    from sisteer.eval.attacks import renellm

    path = Path(args.corpus) if args.corpus else renellm.replay_path()
    if not path.is_file():
        where = (f"at {path}" if args.corpus else
                 f"at the site's copy {renellm._corpus_path()} or the repository's copy {path}")
        print(f"[renellm] no corpus {where}. The repository ships the paper's corpus at "
              f"{renellm.SHIPPED_CORPUS} (sha256 {renellm.PINNED_FILE_SHA256}; see "
              f"{renellm.PAPER_CORPUS_RECORD}); restore it with git checkout -- {renellm.SHIPPED_CORPUS}",
              file=sys.stderr)
        return 1
    try:
        identity = renellm.corpus_identity(path)
    except renellm.RenellmCorpusRefused as e:
        print(f"[renellm] REFUSED: {e}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(identity, indent=1, sort_keys=True))
    else:
        print(f"[renellm] {renellm.describe_identity(identity)} ({path})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
