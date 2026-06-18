"""Regression test: FocalLoss class weights must count TRAIN cells only.

Guards the fix that replaced whole-archive ``dataset.ct_counts`` (which
includes val/test cells) with a train-indices-only count in
``compute_class_weights``. Counting val/test frequencies leaks evaluation-set
label distribution into the training objective.
"""

from types import SimpleNamespace

import numpy as np
import torch

from deepcell_types.training.class_weights import compute_class_weights


def _rec(ct):
    return SimpleNamespace(ct_label_standard=ct)


def test_class_weights_use_train_cells_only():
    # 4 classes. "D" appears ONLY in val cells (indices 4-7), never in train,
    # and "B" is more frequent in val than in train.
    ct2idx = {"A": 0, "B": 1, "C": 2, "D": 3}
    dct_config = SimpleNamespace(ct2idx=ct2idx)
    indices = [
        _rec("A"),
        _rec("A"),
        _rec("B"),
        _rec("C"),  # train (0-3)
        _rec("D"),
        _rec("D"),
        _rec("B"),
        _rec("B"),  # val   (4-7)
    ]
    dataset = SimpleNamespace(indices=indices)
    label_remap = torch.arange(4)  # identity compact mapping
    train_indices = [0, 1, 2, 3]

    weights = compute_class_weights(dct_config, dataset, label_remap, train_indices)

    # Expected from TRAIN-only counts:
    # A=2, B=1, C=1, D=0 (absent -> 1.0); total=4.
    raw = np.array([np.sqrt(4 / 2), np.sqrt(4 / 1), np.sqrt(4 / 1), 1.0])
    expected = raw / raw.mean()
    np.testing.assert_allclose(weights.numpy(), expected, rtol=1e-6)

    # The whole-archive bug would have counted all 8 cells (A=2, B=3, C=1, D=2);
    # assert the result is NOT that vector, so a silent reversion is caught.
    full_raw = np.array(
        [np.sqrt(8 / 2), np.sqrt(8 / 3), np.sqrt(8 / 1), np.sqrt(8 / 2)]
    )
    full_expected = full_raw / full_raw.mean()
    assert not np.allclose(weights.numpy(), full_expected, rtol=1e-6)
