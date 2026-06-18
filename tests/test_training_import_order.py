"""Regression tests for training module import order.

``dataset`` re-exports dataloader helpers for backward compatibility. The
dataloader must therefore avoid importing ``FullImageDataset`` until runtime, or
fresh interpreters can trip a partial-initialization cycle.
"""

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _python(statement: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", statement],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def _assert_imports(statement: str) -> None:
    result = _python(statement)
    assert result.returncode == 0, result.stderr


def test_dataloader_imports_before_dataset():
    _assert_imports(
        "from deepcell_types.training.dataloader import "
        "create_dataloader, DataLoaderConfig; "
        "assert create_dataloader and DataLoaderConfig"
    )


def test_training_lazy_export_imports_create_dataloader():
    _assert_imports(
        "from deepcell_types.training import create_dataloader; "
        "assert create_dataloader"
    )


def test_dataset_backcompat_reexports_dataloader_symbols():
    _assert_imports(
        "from deepcell_types.training.dataset import "
        "create_dataloader, create_dataloader_from_config, DataLoaderConfig; "
        "assert create_dataloader and create_dataloader_from_config and DataLoaderConfig"
    )
