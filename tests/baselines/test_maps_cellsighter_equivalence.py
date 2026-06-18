"""Equivalence proof for the relocated maps/cellsighter baselines.

model.py and __init__.py moved byte-identical to upstream (sha256, modulo the
relocation import rewrite `from {pkg}.model import` -> `from .model import`),
proving the model definition and package surface carry no local logic.

run.py is NO LONGER asserted byte-identical to upstream: it intentionally
deviates so the baselines select their best checkpoint on a held-out,
FOV-grouped inner-validation set carved from the training FOVs, rather than on
the set they then report on (selection-on-the-reported-set is leakage). This
mirrors the XGBoost baseline's GroupShuffleSplit early-stopping set. The
``test_run_py_selects_on_inner_val`` behavioral test below pins that deviation;
the byte-equivalence pin for run.py was removed deliberately.
"""

import ast
import hashlib
from pathlib import Path

import pytest

PKG = Path(__file__).resolve().parents[2] / "deepcell_types" / "baselines"

MODEL_ORIG_SHA = {
    "maps": "29202958b4326a542732663eb92541681d1d3a10ebc0767bad547416249edc00",
    "cellsighter": "fccb04d5d1eb87159d6afcac473b5b872d5c5aafa54a8c56a65457adbeb2f7f2",
}
INIT_ORIG_SHA = {
    "maps": "5a0a765d62d2f11c841da99f34ccd63b226b47285fe85b6a9edbf92636a58f75",
    "cellsighter": "2ebb0af69494e85871ec5df7f4ced019ec296bc88d8e049200f370cb625d53a0",
}

# Packages present at each stage: maps lands in Task 1, cellsighter in Task 2.
PKGS = ["maps", "cellsighter"]


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _tree(pkg):
    return ast.parse((PKG / pkg / "run.py").read_text(encoding="utf-8"))


def _main_func(pkg):
    for node in _tree(pkg).body:
        if isinstance(node, ast.FunctionDef) and node.name == "main":
            return node
    raise AssertionError(f"{pkg}/run.py has no main() function")


def _is_name(node, name):
    return isinstance(node, ast.Name) and node.id == name


def _is_metrics_key(node, key):
    return (
        isinstance(node, ast.Subscript)
        and _is_name(node.value, "metrics")
        and isinstance(node.slice, ast.Constant)
        and node.slice.value == key
    )


def _is_best_checkpoint_key(node, key):
    return (
        isinstance(node, ast.Subscript)
        and _is_name(node.value, "best_checkpoint")
        and isinstance(node.slice, ast.Constant)
        and node.slice.value == key
    )


def _is_torch_save(call):
    return (
        isinstance(call.func, ast.Attribute)
        and call.func.attr == "save"
        and _is_name(call.func.value, "torch")
    )


def _is_click_usage_error(node):
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "UsageError"
        and _is_name(node.func.value, "click")
    )


@pytest.mark.parametrize("pkg", PKGS)
def test_model_py_byte_identical(pkg):
    data = (PKG / pkg / "model.py").read_bytes()
    assert _sha(data) == MODEL_ORIG_SHA[pkg]


# Substrings that must appear in each run.py, proving model selection happens on
# a held-out inner-val set rather than on the reported (test) set.
INNER_VAL_MARKERS = {
    "maps": ["GroupShuffleSplit", "X_inner_val_tensor", "inner-val"],
    "cellsighter": ["inner_val_ratio=0.1", "sel_loader", "inner_val_loader"],
}


@pytest.mark.parametrize("pkg", PKGS)
def test_run_py_selects_on_inner_val(pkg):
    """run.py intentionally deviates from upstream: it selects on a held-out,
    FOV-grouped inner-val set, not on the reported test set. Pin that deviation
    behaviorally (the upstream byte-equivalence pin was removed on purpose)."""
    text = (PKG / pkg / "run.py").read_text(encoding="utf-8")
    for marker in INNER_VAL_MARKERS[pkg]:
        assert marker in text, (
            f"{pkg}/run.py is missing the inner-val selection marker {marker!r}; "
            f"checkpoint selection must not run on the reported test set"
        )


def test_cellsighter_epoch_selection_uses_inner_val_macro_f1():
    main = _main_func("cellsighter")
    epoch_loop = next(
        node
        for node in ast.walk(main)
        if isinstance(node, ast.For) and _is_name(node.target, "epoch")
    )

    evaluate_loaders = [
        call.args[1].id
        for call in ast.walk(epoch_loop)
        if isinstance(call, ast.Call)
        and _is_name(call.func, "evaluate")
        and len(call.args) > 1
        and isinstance(call.args[1], ast.Name)
    ]
    assert "sel_loader" in evaluate_loaders
    assert "test_loader" not in evaluate_loaders

    assert any(
        isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and _is_metrics_key(node.test.left, "macro_f1")
        and len(node.test.ops) == 1
        and isinstance(node.test.ops[0], ast.Gt)
        and len(node.test.comparators) == 1
        and _is_name(node.test.comparators[0], "best_macro_f1")
        for node in ast.walk(epoch_loop)
    )
    assert any(
        isinstance(node, ast.Assign)
        and any(_is_name(target, "best_macro_f1") for target in node.targets)
        and _is_metrics_key(node.value, "macro_f1")
        for node in ast.walk(epoch_loop)
    )

    torch_saves = [
        call
        for call in ast.walk(epoch_loop)
        if isinstance(call, ast.Call) and _is_torch_save(call)
    ]
    assert any(
        call.args
        and isinstance(call.args[0], ast.Dict)
        and any(
            isinstance(key, ast.Constant)
            and key.value == "macro_f1"
            and _is_name(value, "best_macro_f1")
            for key, value in zip(call.args[0].keys, call.args[0].values)
        )
        for call in torch_saves
    )
    assert any(_is_best_checkpoint_key(node, "macro_f1") for node in ast.walk(main))


def test_maps_requires_two_train_fovs_for_inner_val_split():
    main = _main_func("maps")

    assert any(
        isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and _is_name(node.test.left, "n_train_fovs")
        and len(node.test.ops) == 1
        and isinstance(node.test.ops[0], ast.Lt)
        and len(node.test.comparators) == 1
        and isinstance(node.test.comparators[0], ast.Constant)
        and node.test.comparators[0].value == 2
        and any(
            isinstance(child, ast.Raise)
            and child.exc is not None
            and _is_click_usage_error(child.exc)
            for child in node.body
        )
        for node in ast.walk(main)
    )


@pytest.mark.parametrize("pkg", PKGS)
def test_init_py_is_only_import_rewrite(pkg):
    text = (PKG / pkg / "__init__.py").read_text(encoding="utf-8")
    restored = text.replace("from .model import", f"from {pkg}.model import")
    assert _sha(restored.encode("utf-8")) == INIT_ORIG_SHA[pkg]
