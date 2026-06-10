"""Byte-identity snapshots for the maps/cellsighter baselines.

maps: model.py moved byte-identical (sha256); run.py/__init__.py changed ONLY by
the relocation import rewrite `from {pkg}.model import` -> `from .model import`;
this test inverts that single rewrite and asserts byte-identity to the recorded
upstream original, proving no logic changed.

cellsighter: NO LONGER an upstream-identical port. On feat/faithful-cellsighter
its model.py and run.py were intentionally reimplemented to follow the paper's
training recipe (unmasked neighbor intensities, 60x60 crops, ImageNet ResNet50
stem, geometric augmentation, per-member seeding). Its SHAs below are therefore
re-pinned as a DRIFT GUARD on the faithful baseline (any future edit must be a
deliberate re-pin), not as a proof of upstream equivalence. __init__.py is still
only an import rewrite, so that check is unchanged.
"""

import hashlib
from pathlib import Path

import pytest

PKG = Path(__file__).resolve().parents[2] / "deepcell_types" / "baselines"

MODEL_ORIG_SHA = {
    "maps": "29202958b4326a542732663eb92541681d1d3a10ebc0767bad547416249edc00",
    # cellsighter: faithful-reimplementation drift guard (ImageNet stem, see docstring).
    "cellsighter": "83629b114b193cc945f23e49e22743ab53d0788a0045d98659bc2f97603d3f5f",
}
# Re-pinned after removing the locally-added ``--min_channels`` CLI option
# (an unused channel-count filter that caused unfair baseline comparisons via
# mismatched defaults).
# Re-pinned again for the public release: the machine-specific ``DATA_DIR``
# fallback was replaced with ``""`` and the optional wandb experiment-logging
# code (the ``--enable_wandb`` flag and its ``if enable_wandb:`` blocks) was
# removed from both files. Logging-only deltas: no model, data, or evaluation
# logic changed. The import rewrite remains the only structural delta vs. these.
RUN_ORIG_SHA = {
    "maps": "78888f8088b9ed3574a3e48cfb86e2317da224e1bb12c507a4e727cc32ece05e",
    # cellsighter: faithful-reimplementation drift guard, not upstream equivalence.
    # This is the sha of run.py with the relocation import rewrite inverted.
    "cellsighter": "684a23ae6b36b3ff8e8a5f184d3a2d114eef839ca8ad1371a2a68b3315a7d8f6",
}
INIT_ORIG_SHA = {
    "maps": "5a0a765d62d2f11c841da99f34ccd63b226b47285fe85b6a9edbf92636a58f75",
    "cellsighter": "2ebb0af69494e85871ec5df7f4ced019ec296bc88d8e049200f370cb625d53a0",
}

# Packages present at each stage: maps lands in Task 1, cellsighter in Task 2.
PKGS = ["maps", "cellsighter"]


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


@pytest.mark.parametrize("pkg", PKGS)
def test_model_py_byte_identical(pkg):
    data = (PKG / pkg / "model.py").read_bytes()
    assert _sha(data) == MODEL_ORIG_SHA[pkg]


@pytest.mark.parametrize("pkg", PKGS)
def test_run_py_is_only_import_rewrite(pkg):
    text = (PKG / pkg / "run.py").read_text(encoding="utf-8")
    restored = text.replace("from .model import", f"from {pkg}.model import")
    assert _sha(restored.encode("utf-8")) == RUN_ORIG_SHA[pkg], (
        f"{pkg}/run.py differs from upstream beyond the import rewrite"
    )


@pytest.mark.parametrize("pkg", PKGS)
def test_init_py_is_only_import_rewrite(pkg):
    text = (PKG / pkg / "__init__.py").read_text(encoding="utf-8")
    restored = text.replace("from .model import", f"from {pkg}.model import")
    assert _sha(restored.encode("utf-8")) == INIT_ORIG_SHA[pkg]
