"""capeval tier runners.

Deliberately import-free: `tier0` and the Tier-1 runner
`evalplus_runner` are imported by their full module path only, so
that importing one tier never drags in another tier's (possibly heavy)
dependencies. Base-lane import rule per the package docstring: heavy deps
(evalplus / vllm) are imported
lazily INSIDE the functions that need them, never at module import time.
"""
