#!/usr/bin/env python
"""stage_ctlstop_bank_modal.py — the control-pool-stopped views' original name for scripts/modal/stage_lane_bank_modal.py, which
generalised it to every lane view with its own roster and pins (with the SI-distilled refit views, 2026-11-15). Kept so the ctlstop lane's
recorded commands and its roster's staging prerequisites still resolve; every argument passes through unchanged.

    python scripts/modal/stage_ctlstop_bank_modal.py --view ctlstop64s1 --source <dir> --parents-source <dir> --receipts-dir <dir> [--go]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage_lane_bank_modal import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
