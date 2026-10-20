"""``sisteer.registry.replica.init_layout`` as it behaved before it kept the tables it does not own: it rewrote the replica root's
``config.toml`` unconditionally with only ``[replica] log_root``. The behaviour is copied from the function at the release base commit;
the comment line it wrote is replaced by a neutral one, which changes nothing the tests read. The negative control of the re-init
test (``tests/test_registry_migrate_site_aliases.py``): run on the same staged root, it erases the extra table, so the test can fail."""
from __future__ import annotations

import os


def init_layout(root: str, log_root: str) -> str:
    root = os.path.abspath(root)
    log_root = os.path.abspath(log_root)
    for d in ("gen", "builds", "pins"):
        os.makedirs(os.path.join(root, d), exist_ok=True)
    for ns in ("records", "rulings"):
        target = os.path.realpath(os.path.join(log_root, ns))
        link = os.path.join(root, ns)
        if os.path.islink(link):
            if os.path.realpath(link) == target:
                continue
            os.remove(link)
        elif os.path.exists(link):
            raise RuntimeError(f"{link} exists and is not a link; the replica root's {ns} must be a link onto the log")
        os.symlink(target, link)
    config = os.path.join(root, "config.toml")
    body = f'# the cluster replica of the registry database\n[replica]\nlog_root = "{log_root}"\n'
    tmp = config + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(body)
    os.replace(tmp, config)
    return root
