import numpy as np

from duckietown_offline_mapper.src.bev import grid_to_world, metadata_from_bounds, world_to_grid


def test_world_grid_round_trip_cell_centers():
    metadata = metadata_from_bounds(-1.0, 1.0, -2.0, 2.0, 0.1)
    u = np.array([0, 5, metadata.width - 1])
    v = np.array([0, 10, metadata.height - 1])
    x, y = grid_to_world(u, v, metadata)
    uu, vv = world_to_grid(x, y, metadata)
    assert np.array_equal(uu, u)
    assert np.array_equal(vv, v)


def test_world_to_grid_uses_top_left_image_origin():
    metadata = metadata_from_bounds(0.0, 2.0, -1.0, 1.0, 0.5)
    u, v = world_to_grid(0.25, 0.75, metadata)
    assert int(u) == 0
    assert int(v) == 0
    x, y = grid_to_world(0, 0, metadata)
    assert np.isclose(x, 0.25)
    assert np.isclose(y, 0.75)

