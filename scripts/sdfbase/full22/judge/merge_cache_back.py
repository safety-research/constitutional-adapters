"""Merge the private run cache back into the shared judge cache, INSERT OR
IGNORE: every verdict this run produced becomes durable for later sessions,
and no key another lineage already holds is overwritten (their re-reads stay
byte-stable). Reports how many rows were new vs already present.

usage: python merge_cache_back.py [--dry-run] [--private CACHE.sqlite]
"""
import argparse
import sqlite3
import sys
from pathlib import Path

PRIV = "$GRADE/sdfbase_full22/cache/judge_cache_f22.sqlite"   # under the grade root, $GRADE: expanded at use
SHARED = Path.home() / ".local/state/capeval/judge_cache.sqlite"


def _site_path(value) -> Path:
    """``value`` with the grade root's placeholder ($GRADE) expanded through sisteer.site, ``~`` expanded."""
    src = str(Path(__file__).resolve().parents[4] / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    from sisteer import site
    return site.expand_path(value)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--private", default=PRIV, help="the run's private cache copy")
    args = ap.parse_args()
    priv = _site_path(args.private)
    if not priv.exists():
        raise SystemExit(f"{priv}: no such private cache")
    conn = sqlite3.connect(SHARED, timeout=60)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=60000")
    conn.execute("ATTACH DATABASE ? AS priv", (str(priv),))
    n_priv = conn.execute("SELECT COUNT(*) FROM priv.verdicts").fetchone()[0]
    n_shared_before = conn.execute("SELECT COUNT(*) FROM main.verdicts").fetchone()[0]
    n_new = conn.execute(
        "SELECT COUNT(*) FROM priv.verdicts p WHERE NOT EXISTS "
        "(SELECT 1 FROM main.verdicts m WHERE m.key = p.key)").fetchone()[0]
    n_diff = conn.execute(
        "SELECT COUNT(*) FROM priv.verdicts p JOIN main.verdicts m ON m.key = p.key "
        "WHERE m.verdict_json != p.verdict_json").fetchone()[0]
    print(f"private={n_priv} shared_before={n_shared_before} new_to_shared={n_new} "
          f"same_key_different_verdict(kept shared)={n_diff}")
    if args.dry_run:
        return 0
    conn.execute("INSERT OR IGNORE INTO main.verdicts (key, verdict_json, org_used, created_at) "
                 "SELECT key, verdict_json, org_used, created_at FROM priv.verdicts")
    conn.commit()
    n_shared_after = conn.execute("SELECT COUNT(*) FROM main.verdicts").fetchone()[0]
    print(f"shared_after={n_shared_after} (+{n_shared_after - n_shared_before})")
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
