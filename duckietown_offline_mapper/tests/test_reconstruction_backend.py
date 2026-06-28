import pytest

from duckietown_offline_mapper.src.reconstruction import backend_from_config


def test_only_vggt_sfm_backend_is_supported():
    with pytest.raises(ValueError):
        backend_from_config({"backend": "ipm_video"})

