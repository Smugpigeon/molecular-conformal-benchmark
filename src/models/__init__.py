"""Model wrappers.

Convention: each model gets its own file.
"""

from src.models.baseline_rf import build_rf

__all__ = ["build_rf"]

# Lazy import for molformer (heavy: torch + transformers).
def get_molformer_train():
    from src.models.molformer_wrapper import train_molformer
    return train_molformer
