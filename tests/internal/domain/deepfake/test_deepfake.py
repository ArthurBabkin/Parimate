import pytest
from omegaconf import OmegaConf

from internal.domain.deepfake.deepfake import DeepFake

CONFIG_PATH = "tests/config/config.yaml"


class TestDeepFake:
    @pytest.fixture(autouse=True)
    def setup(self):
        cfg = OmegaConf.load(CONFIG_PATH)
        self.df = DeepFake(cfg)
        self.video_paths = ["tests/test_data/sample-5s.mp4"]

    def test_extract_metadata(self):
        metadata = self.df.extract_metadata(self.video_paths[0])
        assert isinstance(metadata, dict)

    def test_analyze_video_metadata(self):
        metadata = self.df.analyze_video_metadata(self.video_paths[0])
        assert isinstance(metadata, dict)

    def test_check_video(self):
        assert self.df.analyze_video_metadata(self.video_paths[0])

