import numpy as np

from duckietown_offline_mapper.src.alignment_bev import (
    bev_pixel_to_world_xy,
    invert_planar_transform_point,
)
from duckietown_offline_mapper.src.bev import metadata_from_bounds


def test_bev_pixel_to_world_xy_uses_top_left_origin():
    metadata = metadata_from_bounds(-1.0, 3.0, -2.0, 2.0, 1.0)

    x, y = bev_pixel_to_world_xy(0, 0, metadata.width, metadata.height, metadata)
    assert np.isclose(x, -0.5)
    assert np.isclose(y, 1.5)

    x, y = bev_pixel_to_world_xy(3, 3, metadata.width, metadata.height, metadata)
    assert np.isclose(x, 2.5)
    assert np.isclose(y, -1.5)


def test_invert_planar_transform_point_recovers_source_xy():
    theta = np.deg2rad(30.0)
    transform = np.eye(4)
    transform[:2, :2] = 2.0 * np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )
    transform[:2, 3] = [1.0, -0.25]

    source = np.array([0.4, -0.7, 0.0, 1.0])
    target = transform @ source
    recovered = invert_planar_transform_point(target[0], target[1], transform)

    assert np.allclose(recovered, source[:2])
