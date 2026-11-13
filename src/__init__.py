"""FedDef source package.

Task 1 (foundation) provides:
- ``src.config``: shared experiment configuration
- ``src.data``: dataset loading, cleaning, splitting, scaling
- ``src.evaluate``: classification metrics and result saving

FedAvg, attacks, and defense live in their own modules and are added by
Tasks 2-4.
"""

__all__ = ["config", "data", "evaluate"]
