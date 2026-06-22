"""Test seed reproducibility. Per CLAUDE.md §8.1."""

from __future__ import annotations

import os
import random

import numpy as np

from src.utils.seed import set_all_seeds


def test_python_random_reproducible():
    set_all_seeds(42)
    a = [random.random() for _ in range(5)]
    set_all_seeds(42)
    b = [random.random() for _ in range(5)]
    assert a == b


def test_numpy_reproducible():
    set_all_seeds(42)
    a = np.random.rand(10)
    set_all_seeds(42)
    b = np.random.rand(10)
    np.testing.assert_array_equal(a, b)


def test_env_vars_set():
    set_all_seeds(123)
    assert os.environ["PYTHONHASHSEED"] == "123"
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"


def test_different_seeds_give_different_outputs():
    set_all_seeds(42)
    a = np.random.rand(10)
    set_all_seeds(43)
    b = np.random.rand(10)
    assert not np.allclose(a, b)
