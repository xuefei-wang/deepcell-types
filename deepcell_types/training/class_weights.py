"""Class-weight helpers for supervised training."""

from collections import defaultdict

import numpy as np
import torch


def compute_class_weights(dct_config, dataset, label_remap, train_indices):
    """Compute sqrt-inverse-frequency class weights for FocalLoss.

    Counts are taken over the TRAIN indices only, never whole-archive
    ``dataset.ct_counts``, which includes val/test cells. Using whole-archive
    counts leaks evaluation-set label frequencies into the training objective
    by over-weighting classes concentrated in val/test and under-weighting
    those concentrated in train. Weights are indexed in compact 0-indexed label
    space.
    """
    ct_counts = defaultdict(int)
    for i in train_indices:
        ct_counts[dataset.indices[i].ct_label_standard] += 1
    total = sum(ct_counts.values())
    n_classes = len(dct_config.ct2idx)

    weights = torch.ones(n_classes)
    for ct, idx in dct_config.ct2idx.items():
        compact_idx = label_remap[idx].item()
        count = ct_counts.get(ct, 0)
        if count > 0:
            weights[compact_idx] = np.sqrt(total / count)
        else:
            weights[compact_idx] = 1.0

    return weights / weights.mean()
