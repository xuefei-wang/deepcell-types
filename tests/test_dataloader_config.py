"""Dataloader API contract tests."""

import inspect
from dataclasses import fields

from deepcell_types.training.dataloader import DataLoaderConfig, create_dataloader


def test_dataloader_config_mirrors_create_dataloader_kwargs():
    params = list(inspect.signature(create_dataloader).parameters)
    config_fields = [f.name for f in fields(DataLoaderConfig)]

    assert config_fields == params[2:]
