import numpy as np

from duckietown_offline_mapper.src.occupancy import (
    fuse_occupancy,
    gradient_margin_from_occupancy,
    inflate_obstacles,
    remove_isolated_occupied_cells,
)
from duckietown_offline_mapper.src.segmentation import SemanticClass


def test_fuse_occupancy_semantics_and_obstacles():
    semantic = np.array(
        [
            [SemanticClass.UNKNOWN, SemanticClass.DRIVABLE_ROAD],
            [SemanticClass.LANE_MARKING, SemanticClass.NON_DRIVABLE],
        ],
        dtype=np.uint8,
    )
    obstacles = np.array([[False, True], [False, False]])
    occupancy = fuse_occupancy(semantic, obstacles)
    assert occupancy.tolist() == [[-1, 100], [0, 100]]


def test_inflate_obstacles_radius():
    obstacle = np.zeros((5, 5), dtype=bool)
    obstacle[2, 2] = True
    inflated = inflate_obstacles(obstacle, radius_m=1.0, resolution=1.0)
    assert inflated[2, 2]
    assert inflated[1, 2]
    assert inflated[2, 1]
    assert not inflated[0, 0]


def test_inflate_obstacles_empty_grid_fast_path():
    obstacle = np.zeros((20, 20), dtype=bool)
    inflated = inflate_obstacles(obstacle, radius_m=0.5, resolution=0.001)
    assert inflated.shape == obstacle.shape
    assert not inflated.any()


def test_inflate_obstacles_large_radius_distance_path():
    obstacle = np.zeros((80, 80), dtype=bool)
    obstacle[40, 40] = True
    inflated = inflate_obstacles(obstacle, radius_m=0.03, resolution=0.001)
    assert inflated[40, 40]
    assert inflated[40, 70]
    assert not inflated[40, 71]


def test_gradient_margin_from_occupancy_linear_falloff():
    occupancy = np.zeros((1, 5), dtype=np.int8)
    occupancy[0, 0] = 100
    margin = gradient_margin_from_occupancy(occupancy, margin_m=4.0, resolution=1.0)
    np.testing.assert_allclose(margin[0], [1.0, 0.75, 0.5, 0.25, 0.0])


def test_gradient_margin_from_occupancy_no_occupied_cells():
    occupancy = np.zeros((4, 4), dtype=np.int8)
    margin = gradient_margin_from_occupancy(occupancy, margin_m=2.0, resolution=0.5)
    assert margin.shape == occupancy.shape
    assert not margin.any()


def test_remove_isolated_occupied_cells_removes_salt_noise():
    occupancy = np.zeros((5, 6), dtype=np.int8)
    occupancy[1, 1] = 100
    occupancy[3, 1:5] = 100
    cleaned, removed_cells = remove_isolated_occupied_cells(occupancy)
    assert cleaned[1, 1] == 0
    assert np.all(cleaned[3, 1:5] == 100)
    assert removed_cells == 1


def test_remove_isolated_occupied_cells_uses_eight_connectivity():
    occupancy = np.zeros((4, 4), dtype=np.int8)
    occupancy[1, 1] = 100
    occupancy[2, 2] = 100
    cleaned, removed_cells = remove_isolated_occupied_cells(occupancy)
    assert np.count_nonzero(cleaned == 100) == 2
    assert removed_cells == 0
