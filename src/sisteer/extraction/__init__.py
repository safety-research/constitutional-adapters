# Committed-object loader: exposed lazily so
# importing the package never pulls torch or the reassembly lib by itself.
def __getattr__(name):  # noqa: D401 — module-level lazy attribute (PEP 562)
    if name in ("load_committed_object", "committed_object_bytes_path",
                "CommittedObjectError", "WithheldCommittedObjectError"):
        from . import committed_objects as _co
        return getattr(_co, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
