import pytest

from internal.domain.deepfake.deepfake import DeepFake


class TestDeepFake:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.df = DeepFake()

    def test_check_software(self):
        video_paths = ["./sample-5s.mp4"]

        assert not self.df.check_software(video_paths)
