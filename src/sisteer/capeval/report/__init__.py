"""capeval report stage: fingerprint table + heatmap + gate log
-> static HTML + PNGs + W&B summary run. See report.build.build().

NOTE: the entry point is exported as ``build_report`` here so the package
attribute ``build`` stays bound to the SUBMODULE (a bare ``build``
re-export would shadow it and break ``import …report.build``).
"""

from sisteer.capeval.report.build import ReportPaths
from sisteer.capeval.report.build import build as build_report

__all__ = ["ReportPaths", "build_report"]
