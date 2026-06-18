"""CellSighter augmentation behavior tests."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")


def test_cellsighter_train_transform_center_crops_output():
    from deepcell_types.baselines.cellsighter.transforms import (
        build_cellsighter_train_transform,
    )

    torch.manual_seed(0)
    x = torch.zeros(8, 128, 128)
    x[0, 64, 64] = 1.0
    x[5, 60:68, 60:68] = 1.0
    x[6, 50:58, 50:58] = 1.0

    transform = build_cellsighter_train_transform(center_crop_size=60)
    out = transform(x)

    assert tuple(out.shape) == (8, 60, 60)
    assert torch.isfinite(out).all()
