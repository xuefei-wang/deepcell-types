# CellSighter baseline

Re-implementation of CellSighter (a CNN that classifies cell types directly from
multi-channel image patches) against the DeepCell Types data loader, marker
vocabulary, and cell-type labels.

- Reference paper: Amitay et al., *Nature Communications* 2023, DOI:
  [10.1038/s41467-023-40066-7](https://doi.org/10.1038/s41467-023-40066-7)
- Upstream code: <https://github.com/KerenLab/CellSighter>

Run with:

```bash
pip install -e ".[baseline-cellsighter]"
python -m deepcell_types.baselines cellsighter ...
```

## Shared interface (identical to DeepCell Types)

- Consumes the same 32×32 multi-channel patches, the same train/validation/test
  split, and is scored with the same hierarchical accuracy and macro/weighted
  metrics as the other baselines.
- For zero-shot evaluation, panels are aligned to the global marker vocabulary
  by scattering each dataset's channels to their global `marker2idx` positions;
  absent markers are zero-padded (`model.py:94-111`).

## Deviations / adaptations (recorded for reproducibility)

- **Backbone: torchvision ResNet-50** (`model.py:46-48`), default
  `model_size="resnet50"`.
- **Custom CIFAR-style stem sized to 32×32 patches**: the ImageNet stem
  (7×7 stride-2 conv + max-pool) collapses a 32×32 input to 1×1 by `layer4`, so
  it is replaced by a single **3×3 stride-1 conv with no max-pool**
  (`model.py:54-57`); the spatial path becomes 32→32→16→8→4→1. This matches the
  small-patch adaptation in the upstream CellSighter recipe.
- **Input channels = `NUM_MARKERS + 2`** — the globally aligned marker channels
  plus the cell mask and neighbor mask.
- **Trained from random initialization** (`pretrained=False`, `weights=None`,
  `model.py:36-47`), matching upstream (no ImageNet weights).
- **50 epochs, Adam, constant learning rate `1e-3`, no scheduler.** The upstream
  repo constructs an `ExponentialLR` scheduler but never calls
  `scheduler.step()`, so it trains at constant LR; we reproduce that exactly and
  do not step a scheduler (`run.py:389-393`).
- **Best epoch selected on validation macro-accuracy** (`run.py:399`,
  `run.py:460-461`); validation runs every `val_every_n_epochs` (default 10) plus
  the final epoch, matching the upstream cadence.

## Data-pipeline deviations from upstream CellSighter

These come from sharing the DeepCell Types single-cell patch pipeline rather than
upstream CellSighter's own data loader. The first two are the most material
because they change the signal the CNN sees; the rest are weaker forms of the
same effect. Several are *shared* preprocessing applied identically to every
model (DeepCell Types, MAPS, CellSighter), so they do not bias CellSighter
relative to the other baselines — but they are still deviations from how the
published CellSighter was trained, and are recorded here for that reason.

- **Neighbor pixel intensities are zeroed (self-mask).** Our shared patch
  extractor multiplies each crop by the target cell's binary mask
  (`deepcell_types/training/patch.py:176`, `raw_masked = raw_crop * self_mask`),
  so only the target cell's pixels carry intensity. Upstream feeds the **raw,
  unmasked** crop and identifies the target via *separate* mask channels
  (`KerenLab/CellSighter` `data/cell_crop.py` `'image': self._image[self._slices]`;
  `data/data.py` stacks `[image, all_cells_mask, mask]`). This removes the
  microenvironment intensity signal CellSighter is designed to exploit. It is a
  by-design choice for the DeepCell Types single-cell task (and is shared by MAPS),
  but for CellSighter it is a deviation. *Empirically tested:* an unmasked 60×60
  faithful re-run (the `feat/faithful-cellsighter` branch) did **not** beat the
  masked baseline on the v10 held-out test set, so restoring neighbor context did
  not change the ranking.
- **Intensity normalization differs.** Our inputs are per-FOV, per-channel
  p99.9-clipped then min-max normalized to `[0, 1]`, baked into `preprocessed/raw`
  at ingestion (`deepcell_types/preprocessing.py`). Upstream applies **no**
  intensity normalization — it feeds raw counts (its always-on `poisson_sampling`
  augmentation resamples counts, so it is count-scale by construction;
  `data/transform.py`). The min-max scale is necessary for a cross-panel,
  multi-modality benchmark and is shared by all models, but it is not the regime
  the single-panel upstream CellSighter was trained in.
- **Augmentation is weaker.** We apply horizontal + vertical flips at `p=0.5`
  only (`deepcell_types/training/dataloader.py:96-101`). Upstream applies seven
  augmentations (`data/transform.py`): Poisson resampling (always), cell-mask
  dilation (`p=0.5`), neighbor-mask dilation (`p=0.5`), 0–360° rotation (always),
  per-channel pixel shift (`p=0.5`), and h/v flips at `p=0.75`. Poisson, dilation,
  rotation, and shift are absent here, and our flips use `p=0.5` vs upstream `0.75`.
- **Smaller spatial context.** We use 32×32 patches (`training/config.py:78-79`); upstream
  extracts 128 px and feeds a 60 px model input, i.e. a larger neighborhood.
- **Class-balancing sampler differs.** We use the DeepCell Types
  sqrt-inverse-frequency `FOVGroupedSampler` with a 1000-sample effective-count
  floor (default `use_weighted_sampler=True`,
  `deepcell_types/training/samplers.py:41-45`), the same scheme as the main model.
  Upstream uses a full-inverse-frequency `WeightedRandomSampler` (more aggressive
  rare-class upweighting). Note this also means CellSighter and MAPS are *not*
  balanced by the same scheme (MAPS faithfully uses full inverse frequency).
