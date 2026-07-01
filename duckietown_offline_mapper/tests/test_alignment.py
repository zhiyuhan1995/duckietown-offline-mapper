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


def test_estimate_sim2_can_recover_reflected_planar_alignment():
    source = np.array([[0.16, -0.28], [-0.70, -0.41], [0.32, -1.30]])
    target = np.array([[0.0, 0.0], [0.0, 3.0], [3.6, 0.0]])
    rigid_result = estimate_sim2(source, target, allow_reflection=False)
    reflected_result = estimate_sim2(source, target, allow_reflection=True)
    reflected_predicted = (reflected_result.transform @ np.c_[source, np.ones(len(source))].T).T[:, :2]
    assert rigid_result.rms_error > 1.0
    assert reflected_result.reflection
    assert reflected_result.determinant < 0.0
    assert reflected_result.rms_error < 0.02
    assert np.allclose(reflected_predicted, target, atol=0.02)
