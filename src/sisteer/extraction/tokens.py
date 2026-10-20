"""Soft-token intervention objects.

Two placements of the same object (one trainable input-embedding vector
carried by a reserved sentinel token):

- SoftTokenObject (objects.py, re-exported here): sentinel inserted right
  AFTER BOS, trained at T=1 — the harness's established `token_bos` arm.
- AppendedSoftTokenObject: the faithful Radevski et al. (2601.05062)
  Steering Token — the sentinel is APPENDED at the end of the rendered
  prompt, so the layout is [E_x, e_b] with the behavior embedding as the
  final prompt position, immediately before generation. Trained with the
  paper's temperature-softened KL: T=10 and loss ×T² (=100), passed to
  train_object_kl as temperature=10, loss_scale=100, and initialized at
  the MEAN of the instruction's own token embeddings (the paper's
  semantic init, §3). Both are wired in sisteer.methods; the class
  itself is placement-only.

Both kinds carry token-position semantics, so downstream code must call
`obj.inject(rendered_prompt)` on every prompt the object should act on
(identity for all non-token kinds).
"""

from __future__ import annotations

from .objects import SoftTokenObject

# The Radevski recipe's training constants (KL temperature and the xT^2
# loss correction). methods.py reads these so the arm definition and the
# tests share one source of truth.
FAITHFUL_TEMPERATURE = 10.0
FAITHFUL_LOSS_SCALE = FAITHFUL_TEMPERATURE**2


class AppendedSoftTokenObject(SoftTokenObject):
    """Radevski-faithful Steering Token: appended, not after-BOS.

    Same trainable object and embedding-replacement hook as
    SoftTokenObject; only the injection position differs.
    """

    kind = "token_appended"

    def inject(self, rendered_prompt: str) -> str:
        """Append the sentinel at the very end of the rendered prompt, so
        its embedding occupies the final prompt position ([E_x, e_b])."""
        return rendered_prompt + self.sentinel_text
