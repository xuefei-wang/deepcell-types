"""PredLogger emits (tissue, modality) columns when supplied, so the eval CLI
can bucket abstention per (tissue, modality) — the paper's Methods grouping.

This validates the column plumbing only. Whether the published benchmark
numbers were produced with (tissue, modality) vs per-FOV bucketing is tracked
separately (the grouping change in scripts/predict.py requires re-running the
benchmark; see the WIP PR).
"""

import numpy as np
import pytest

pytest.importorskip("pandas")

from deepcell_types.training.utils import PredLogger


CT2IDX = {"CD4T": 0, "CD8T": 1, "Bcell": 2}


def _probs(n):
    p = np.random.default_rng(0).random((n, len(CT2IDX))).astype(np.float32)
    return p / p.sum(axis=1, keepdims=True)


def test_columns_present_when_tissue_modality_supplied():
    logger = PredLogger(CT2IDX)
    n = 5
    logger.log(
        labels=np.zeros(n, dtype=np.int64),
        probs=_probs(n),
        cell_index=np.arange(n),
        dataset_name=np.array(["ds"] * n),
        fov_name=np.array(["fov0"] * n),
        tissue=["lung"] * n,
        modality=["MIBI"] * n,
    )
    df = logger.to_dataframe()
    assert "tissue" in df.columns and "modality" in df.columns
    assert list(df["tissue"]) == ["lung"] * n
    assert list(df["modality"]) == ["MIBI"] * n
    # The columns the paper's abstention default groups on are now groupable.
    assert set(df.groupby(["tissue", "modality"]).indices) == {("lung", "MIBI")}


def test_columns_absent_when_not_supplied_back_compat():
    logger = PredLogger(CT2IDX)
    n = 3
    logger.log(
        labels=np.zeros(n, dtype=np.int64),
        probs=_probs(n),
        cell_index=np.arange(n),
        dataset_name=np.array(["ds"] * n),
        fov_name=np.array(["fov0"] * n),
    )
    df = logger.to_dataframe()
    assert "tissue" not in df.columns and "modality" not in df.columns
