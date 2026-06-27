import pytest
import numpy as np

from duckietown_offline_mapper.src.reconstruction import backend_from_config
from duckietown_offline_mapper.src import reconstruction as reconstruction_module


def test_only_vggt_sfm_backend_is_supported():
    with pytest.raises(ValueError):
        backend_from_config({"backend": "ipm_video"})


def test_vggt_keep_mask_relaxes_duckietown_ground_colors():
    points = np.zeros((1, 1, 5, 3), dtype=np.float32)
    conf = np.array([[[6.0, 1.3, 1.3, 1.3, 0.2]]], dtype=np.float32)
    colors = np.array(
        [[[[120, 80, 50], [30, 30, 30], [210, 210, 205], [210, 180, 40], [30, 30, 30]]]],
        dtype=np.uint8,
    )

    strict = reconstruction_module._vggt_keep_mask(
        points,
        conf,
        colors,
        confidence_threshold=5.0,
        relax_ground_confidence=False,
        ground_confidence_threshold=1.2,
        sample_stride=1,
    )
    relaxed = reconstruction_module._vggt_keep_mask(
        points,
        conf,
        colors,
        confidence_threshold=5.0,
        relax_ground_confidence=True,
        ground_confidence_threshold=1.2,
        sample_stride=1,
    )

    assert strict.tolist() == [[[True, False, False, False, False]]]
    assert relaxed.tolist() == [[[True, True, True, True, False]]]


def test_limit_mask_is_deterministic_and_bounded():
    mask = np.ones((2, 10), dtype=bool)
    first = reconstruction_module._limit_mask(mask, max_points=5, seed=3)
    second = reconstruction_module._limit_mask(mask, max_points=5, seed=3)

    assert np.array_equal(first, second)
    assert int(np.count_nonzero(first)) == 5
