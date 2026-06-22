"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


@pytest.fixture(scope="session")
def data_dir() -> Path:
    """Project data directory (read-only)."""
    return DATA_DIR


@pytest.fixture(scope="session")
def sample_smiles() -> list[str]:
    """Small valid SMILES list for unit tests."""
    return [
        "CCO",                          # ethanol
        "c1ccccc1",                     # benzene
        "CC(=O)Oc1ccccc1C(=O)O",        # aspirin
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # caffeine
    ]


@pytest.fixture(scope="session")
def invalid_smiles() -> list[str]:
    """SMILES strings RDKit should reject."""
    return ["XXXNOTASMILES", "", "C(C(C", "1234"]
