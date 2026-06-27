import numpy as np

from duckietown_offline_mapper.src.plane import ground_alignment_transform


def test_ground_alignment_maps_plane_to_z_zero():
    normal = np.array([0.2, -0.3, 0.9327379])
    normal = normal / np.linalg.norm(normal)
    d = -0.4
    transform = ground_alignment_transform(np.r_[normal, d])
    points = np.array(
        [
            [2.0, 0.0, -(normal[0] * 2.0 + d) / normal[2]],
            [0.0, -1.0, -(normal[1] * -1.0 + d) / normal[2]],
            [0.5, 0.5, -(normal[0] * 0.5 + normal[1] * 0.5 + d) / normal[2]],
        ]
    )
    aligned = (transform @ np.c_[points, np.ones(len(points))].T).T[:, :3]
    assert np.allclose(aligned[:, 2], 0.0, atol=1e-9)

