"""Registry + CLI option-snapshot tests for the round-2 baselines (maps, cellsighter).

Frozen option snapshots of the per-baseline click commands (re-frozen after the
wandb logging option was removed across all baselines). cellsighter's command
imports torchvision (via cellsighter.model), so its tests importorskip it.
"""

import click
import pytest

from deepcell_types.baselines import REGISTRY
from deepcell_types.baselines.__main__ import cli


MAPS_OPTS = {
    "model_name",
    "device_num",
    "zarr_dir",
    "skip_datasets",
    "keep_datasets",
    "split_file",
    "features_cache",
    "batch_size",
    "dropout",
    "hidden_dim",
    "learning_rate",
    "max_epochs",
    "seed",
}
# Re-frozen for the faithful CellSighter reimplementation (feat/faithful-cellsighter):
# added crop_size, augment_crop_size, mask_self, cifar_stem, test_split_file,
# allow_split_mismatch, and seed. These expose the paper-faithful training path
# (unmasked neighbor intensities, 128->60 crops, ImageNet stem) plus its
# self-masked/CIFAR ablations, a held-out test-split eval hook, and per-member
# seeding for ensembling.
CELLSIGHTER_OPTS = {
    "model_name",
    "device_num",
    "zarr_dir",
    "skip_datasets",
    "keep_datasets",
    "split_file",
    "test_split_file",
    "split_mode",
    "batch_size",
    "epochs",
    "learning_rate",
    "model_size",
    "crop_size",
    "augment_crop_size",
    "mask_self",
    "cifar_stem",
    "allow_split_mismatch",
    "seed",
    "no_amp",
    "no_compile",
    "pretrained",
    "val_every_n_epochs",
}


def _param_names(cmd):
    return {p.name for p in cmd.params}


def _subcommand(name):
    ctx = click.Context(cli)
    return cli.get_command(ctx, name)


def _params(cmd):
    return {p.name: p for p in cmd.params}


def test_registry_has_maps():
    assert REGISTRY["maps"] == "deepcell_types.baselines.maps.run:main"


def test_maps_subcommand_options_frozen():
    cmd = _subcommand("maps")
    assert isinstance(cmd, click.Command)
    assert _param_names(cmd) == MAPS_OPTS


def test_registry_has_cellsighter():
    assert REGISTRY["cellsighter"] == "deepcell_types.baselines.cellsighter.run:main"


def test_cellsighter_subcommand_options_frozen():
    pytest.importorskip("torchvision")
    cmd = _subcommand("cellsighter")
    assert isinstance(cmd, click.Command)
    assert _param_names(cmd) == CELLSIGHTER_OPTS


def test_cellsighter_faithful_option_defaults_frozen():
    pytest.importorskip("torchvision")
    cmd = _subcommand("cellsighter")
    params = _params(cmd)

    assert params["crop_size"].default == 60
    assert params["augment_crop_size"].default == 128
    assert params["mask_self"].default is False and params["mask_self"].is_flag
    assert params["cifar_stem"].default is False and params["cifar_stem"].is_flag
    assert (
        params["allow_split_mismatch"].default is False
        and params["allow_split_mismatch"].is_flag
    )
    assert params["seed"].default == 42
    assert params["split_mode"].default == "fov"
    assert params["test_split_file"].default is None


def _write_split(path, train=None, val=None):
    import json

    path.write_text(
        json.dumps(
            {
                "train": train or {},
                "val": val or {},
            }
        )
    )


def test_cellsighter_test_split_requires_training_split(tmp_path):
    pytest.importorskip("torchvision")
    from deepcell_types.baselines.cellsighter.run import _validate_heldout_test_split

    test_split = tmp_path / "test.json"
    _write_split(test_split, val={"DS": ["FOV_TEST"]})

    with pytest.raises(click.UsageError, match="requires --split_file"):
        _validate_heldout_test_split(None, str(test_split))


def test_cellsighter_test_split_rejects_train_overlap(tmp_path):
    pytest.importorskip("torchvision")
    from deepcell_types.baselines.cellsighter.run import _validate_heldout_test_split

    train_split = tmp_path / "train.json"
    test_split = tmp_path / "test.json"
    _write_split(train_split, train={"DS": ["FOV_1", "FOV_2"]})
    _write_split(test_split, val={"DS": ["FOV_2", "FOV_3"]})

    with pytest.raises(click.UsageError, match="training/model-selection"):
        _validate_heldout_test_split(str(train_split), str(test_split))


def test_cellsighter_test_split_rejects_selection_overlap(tmp_path):
    pytest.importorskip("torchvision")
    from deepcell_types.baselines.cellsighter.run import _validate_heldout_test_split

    train_split = tmp_path / "train.json"
    test_split = tmp_path / "test.json"
    _write_split(train_split, train={"DS": ["FOV_1"]}, val={"DS": ["FOV_2"]})
    _write_split(test_split, val={"DS": ["FOV_2"]})

    with pytest.raises(click.UsageError, match="selection"):
        _validate_heldout_test_split(str(train_split), str(test_split))


def test_cellsighter_test_split_allows_disjoint_splits(tmp_path):
    pytest.importorskip("torchvision")
    from deepcell_types.baselines.cellsighter.run import _validate_heldout_test_split

    train_split = tmp_path / "train.json"
    test_split = tmp_path / "test.json"
    _write_split(train_split, train={"DS": ["FOV_1"]})
    _write_split(test_split, val={"DS": ["FOV_2"]})

    _validate_heldout_test_split(str(train_split), str(test_split))
