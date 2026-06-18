import numpy as np
import pytest
from deepcell_types.preprocessing_ops import (
    apply_config,
    DEFAULT_CONFIG,
    make_preprocessor,
)
from deepcell_types.preprocessing import (
    _percentile_threshold_nonzero,
    _normalize_per_channel,
)


def _fov(seed=0):
    rng = np.random.default_rng(seed)
    x = rng.gamma(2.0, 50.0, size=(3, 24, 24)).astype(np.float32)
    x[1, :5, :5] = 5000.0  # bright outlier blob
    return x, ["CD3", "CD8", "DAPI"]


def test_default_config_matches_builtin_inference_path():
    raw, names = _fov()
    out = apply_config(raw, names, DEFAULT_CONFIG)  # (C,H,W)
    hwc = np.transpose(raw, (1, 2, 0))
    ref = _normalize_per_channel(_percentile_threshold_nonzero(hwc, percentile=99.9))
    ref = np.transpose(ref, (2, 0, 1))
    assert out.shape == raw.shape
    np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-6)


def test_percentile_threshold_in_place_preserves_nan_percentile_behavior():
    hwc = np.array([[[0.0], [1.0]], [[np.nan], [10.0]]], dtype=np.float32)

    out = _percentile_threshold_nonzero(hwc.copy(), percentile=99.9, in_place=True)

    np.testing.assert_equal(out, hwc)


def test_preprocessing_helpers_keep_copy_semantics_for_integer_inputs():
    hwc = np.array([[[0], [1]], [[3], [10]]], dtype=np.uint16)

    clipped = _percentile_threshold_nonzero(hwc, percentile=50)
    normalized = _normalize_per_channel(hwc)

    assert clipped.dtype == np.uint16
    assert normalized.dtype == np.float64
    assert clipped is not hwc
    assert normalized is not hwc


def test_preprocessing_helpers_accept_read_only_inputs():
    hwc = np.array([[[0.0], [1.0]], [[3.0], [10.0]]], dtype=np.float32)
    hwc.setflags(write=False)

    clipped = _percentile_threshold_nonzero(hwc, percentile=50, in_place=True)
    normalized = _normalize_per_channel(hwc, in_place=True)

    assert clipped.flags.writeable
    assert normalized.flags.writeable
    assert clipped is not hwc
    assert normalized is not hwc


def test_output_is_in_unit_range():
    raw, names = _fov()
    out = apply_config(raw, names, DEFAULT_CONFIG)
    assert out.min() >= 0.0 and out.max() <= 1.0 + 1e-6


def test_channel_drop_zeros_named_channel():
    raw, names = _fov()
    out = apply_config(
        raw,
        names,
        [{"op": "channel_drop", "names": ["DAPI"]}, {"op": "min_max_normalize"}],
    )
    assert np.all(out[2] == 0.0)


def test_channel_weight_after_normalize_scales():
    raw, names = _fov()
    cfg = [
        {"op": "min_max_normalize"},
        {"op": "channel_weight", "weights": {"CD8": 0.25}},
    ]
    out = apply_config(raw, names, cfg)
    assert out[1].max() <= 0.25 + 1e-6


def test_all_table_ops_are_implemented():
    raw, names = _fov()
    for op in [
        {"op": "clip_percentile", "p": 99.0},
        {"op": "log1p"},
        {"op": "background_subtract", "value": 10.0},
        {"op": "gamma", "g": 0.5},
        {"op": "denoise", "kind": "median", "size": 3},
        {"op": "hot_pixel_removal", "z": 5.0},
    ]:
        apply_config(raw, names, [op, {"op": "min_max_normalize"}])  # must not raise


def test_unknown_op_raises():
    raw, names = _fov()
    with pytest.raises(ValueError, match="unknown op"):
        apply_config(raw, names, [{"op": "nope"}])


def test_make_preprocessor_returns_hook():
    raw, names = _fov()
    hook = make_preprocessor(DEFAULT_CONFIG)
    np.testing.assert_allclose(
        hook(raw, names), apply_config(raw, names, DEFAULT_CONFIG)
    )


def test_public_exports():
    import deepcell_types as dct

    assert callable(dct.make_preprocessor)
    assert callable(dct.apply_config)
    assert isinstance(dct.DEFAULT_CONFIG, list)
