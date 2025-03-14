import dlib
import pytest
from omegaconf import OmegaConf

from internal.domain.deepfake import DeepFakeEyeIris, DeepFakeMetadata, \
    DeepFakeNN

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


class TestDeepFakeEyeIris:
    @pytest.fixture(autouse=True)
    def setup(self):
        cfg = OmegaConf.load(CONFIG_PATH)
        self.df = DeepFakeEyeIris(cfg.deepfake)
        self.image_paths = ["tests/test_data/seed000000.png"]

    def test_analyze_eye_iris(self):
        img = dlib.load_rgb_image(self.image_paths[0])
        t = self.df.analyze_eye_iris(img)
        assert not t


class TestDeepFakeNN:
    @pytest.fixture(autouse=True)
    def setup(self):
        cfg = OmegaConf.load(CONFIG_PATH)
        self.df = DeepFakeNN(cfg.deepfake)
        self.video_paths = ["tests/test_data/video1.mp4"]

    def test_analyze_video(self):
        t = self.df.analyze_video(self.video_paths[0], frame_interval=10)
        assert isinstance(t, list)
