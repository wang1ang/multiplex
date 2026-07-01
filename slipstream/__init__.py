"""A local LLM inference framework built on mlx-lm.

Goal: batched speculative decoding — keep MTP speedup while decoding several
sequences together, with sequences able to join/leave the batch.

Current state: the engine layer (batched forward: batch + next-k + rollback).
No sampling, no scheduler, no MTP yet — the engine only runs correct forwards.
"""

from .engine import Engine, BatchState

__all__ = ["Engine", "BatchState"]
__version__ = "0.0.0"
