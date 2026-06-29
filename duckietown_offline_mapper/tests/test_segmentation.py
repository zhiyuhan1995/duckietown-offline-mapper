import numpy as np

from duckietown_offline_mapper.src.segmentation import SemanticClass, segment_bev_rgb


def test_white_lane_occupied_mode_marks_only_white_as_occupied():
    image = np.array(
        [
            [[255, 255, 255], [80, 80, 80]],
            [[255, 220, 0], [255, 0, 0]],
        ],
        dtype=np.uint8,
    )
    semantic = segment_bev_rgb(
        image,
        {
            "mode": "white_lane_occupied",
            "unknown_rgb": [80, 80, 80],
            "white_v_min": 150,
            "white_s_max": 85,
            "morphology_open": 0,
            "morphology_close": 0,
            "white_occupied_unknown_as_free": True,
        },
    )

    assert semantic[0, 0] == SemanticClass.NON_DRIVABLE
    assert semantic[0, 1] == SemanticClass.DRIVABLE_ROAD
    assert semantic[1, 0] == SemanticClass.DRIVABLE_ROAD
    assert semantic[1, 1] == SemanticClass.DRIVABLE_ROAD
