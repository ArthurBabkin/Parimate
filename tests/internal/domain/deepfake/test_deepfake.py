import pytest
from omegaconf import OmegaConf

from internal.domain.deepfake import DeepFakeEyeIris, DeepFakeMetadata

CONFIG_PATH = "tests/config/config.yaml"


class TestDeepFakeMetadata:
    @pytest.fixture(autouse=True)
    def setup(self):
        cfg = OmegaConf.load(CONFIG_PATH)
        self.df = DeepFakeMetadata(cfg.deepfake)
        self.video_paths = ["tests/test_data/sample-5s.mp4"]

    def test_extract_metadata(self):
        metadata = self.df.extract_metadata(self.video_paths[0])
        assert isinstance(metadata, dict)

    def test_analyze_video_metadata(self):
        metadata = self.df.analyze_video_metadata(self.video_paths[0])
        assert isinstance(metadata, dict)

    def test_check_video(self):
        assert self.df.analyze_video_metadata(self.video_paths[0])


class TestDeepFakeEyeIris:
    @pytest.fixture(autouse=True)
    def setup(self):
        cfg = OmegaConf.load(CONFIG_PATH)
        self.df = DeepFakeEyeIris(cfg.deepfake)
        self.image_paths = ["tests/test_data/seed000000.png"]

    def test_detect(self):
        t = self.df.detection(self.image_paths[0])
        assert isinstance(t, float)
