import idtrackerai.constants as cons


def test_data_exists():
    assert cons.COMPRESSED_VIDEO_PATH.is_file()
    assert cons.COMPRESSED_VIDEO_PATH_2.is_file()
