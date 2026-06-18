"""Patch extraction behavior tests."""

import numpy as np

from deepcell_types.training.patch import extract_patch


def test_extract_patch_can_keep_neighbor_intensities():
    raw = np.arange(25, dtype=np.float32).reshape(1, 5, 5)
    mask = np.zeros((5, 5), dtype=np.int32)
    mask[2, 2] = 1
    mask[1, 1] = 2

    masked, context = extract_patch(
        raw,
        mask,
        centroid=(2.0, 2.0),
        cell_idx=1,
        crop_size=5,
        output_size=5,
        skip_distance_transform=True,
        mask_intensities=True,
    )
    unmasked, unmasked_context = extract_patch(
        raw,
        mask,
        centroid=(2.0, 2.0),
        cell_idx=1,
        crop_size=5,
        output_size=5,
        skip_distance_transform=True,
        mask_intensities=False,
    )

    assert masked[0, 1, 1] == 0.0
    assert unmasked[0, 1, 1] == raw[0, 1, 1]
    assert masked[0, 2, 2] == raw[0, 2, 2]
    np.testing.assert_array_equal(context, unmasked_context)
    assert context[0, 2, 2] == 1.0
    assert context[1, 1, 1] == 1.0
