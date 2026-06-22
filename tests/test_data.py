"""Test data loaders and featurizers. Per CLAUDE.md §7."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.data.featurizers import batch_morgan, smiles_to_morgan
from src.data.loaders import DATA_REGISTRY, load_dataset
from src.data.preprocess import canonicalize_smiles


# ---- Loaders ----

class TestLoaders:
    @pytest.mark.parametrize("name", list(DATA_REGISTRY))
    def test_all_datasets_loadable(self, name: str, data_dir: Path):
        # The binary-classification variants (esol_cls/bace_cls/bbbp_cls) are optional course
        # artifacts built by `src.data.binarize` (make data); skip if not present rather than fail.
        fname = DATA_REGISTRY[name][0]
        if not (data_dir / fname).exists():
            pytest.skip(f"{name}: {fname} not built (optional course-classification variant; "
                        f"run `make data` or `python -m src.data.binarize`)")
        split = load_dataset(name, data_root=data_dir)
        assert split.name == name
        assert len(split.train) > 0
        assert len(split.val) > 0
        assert len(split.test) > 0
        assert split.task in {"regression", "classification"}

    def test_unknown_dataset_raises(self, data_dir: Path):
        with pytest.raises(KeyError, match="Unknown dataset"):
            load_dataset("imaginary", data_root=data_dir)

    def test_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            load_dataset("esol", data_root=tmp_path)

    def test_split_sizes_match_registry(self, data_dir: Path):
        # Spot-check GSHt (project main line) per CLAUDE.md §3 table.
        s = load_dataset("gsht", data_root=data_dir)
        assert len(s.train) == 301
        assert len(s.val) == 34
        assert len(s.test) == 84


# ---- Featurizers ----

class TestMorgan:
    def test_default_shape(self, sample_smiles: list[str]):
        fp = smiles_to_morgan(sample_smiles[0])
        assert fp is not None
        assert fp.shape == (2048,)
        assert fp.dtype == np.int8

    def test_invalid_returns_none(self, invalid_smiles: list[str]):
        for s in invalid_smiles:
            assert smiles_to_morgan(s) is None

    def test_batch_drops_failed(self, sample_smiles, invalid_smiles):
        mixed = sample_smiles + invalid_smiles
        X, kept = batch_morgan(mixed)
        assert X.shape == (len(sample_smiles), 2048)
        assert kept == list(range(len(sample_smiles)))

    def test_morgan_deterministic(self, sample_smiles):
        a = smiles_to_morgan(sample_smiles[0])
        b = smiles_to_morgan(sample_smiles[0])
        np.testing.assert_array_equal(a, b)


# ---- Canonicalization ----

class TestCanonicalize:
    def test_canonical_idempotent(self):
        # Aspirin in two forms must canonicalize to the same string.
        s1 = canonicalize_smiles("CC(=O)Oc1ccccc1C(=O)O")
        s2 = canonicalize_smiles("O=C(O)c1ccccc1OC(C)=O")
        assert s1 == s2

    def test_invalid_returns_none(self):
        assert canonicalize_smiles("XYZNOTSMILES") is None
        assert canonicalize_smiles("") is None
