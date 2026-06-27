import numpy as np

from duckietown_offline_mapper.src.alignment import estimate_sim2, umeyama_sim3


def test_umeyama_sim3_recovers_known_transform():
    source = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    theta = np.deg2rad(30)
    rotation = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    scale = 2.5
    translation = np.array([1.0, -0.5, 0.25])
    target = (scale * (rotation @ source.T)).T + translation
    result = umeyama_sim3(source, target)
    predicted = (result.transform @ np.c_[source, np.ones(len(source))].T).T[:, :3]
    assert np.allclose(predicted, target)
    assert np.isclose(result.scale, scale)
    assert result.rms_error < 1e-12


def test_estimate_sim2_recovers_planar_alignment():
    source = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [2.0, 1.0]])
    theta = np.deg2rad(-45)
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    target = (1.2 * (rotation @ source.T)).T + np.array([3.0, 4.0])
    result = estimate_sim2(source, target)
    predicted = (result.transform @ np.c_[source, np.ones(len(source))].T).T[:, :2]
    assert np.allclose(predicted, target)

