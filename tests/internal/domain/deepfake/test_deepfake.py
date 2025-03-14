import dlib
import numpy as np
import pytest
from omegaconf import OmegaConf

from internal.domain.deepfake import (DeepFake, DeepFakeEyeIris,
                                      DeepFakeMetadata, DeepFakeNN)
from internal.domain.utils.video_prepare import extract_frames_from_video

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
        self.cfg = cfg
        self.df = DeepFakeNN(cfg.deepfake)
        self.video_paths = ["tests/test_data/video1.mp4"]

    def test_analyze_video(self):
        frames = extract_frames_from_video(self.video_paths[0],
                                           self.cfg.deepfake.step)
        t = self.df.analyze_video(frames)
        assert isinstance(t, np.ndarray)


class TestDeepFake:
    @pytest.fixture(autouse=True)
    def setup(self):
        cfg = OmegaConf.load(CONFIG_PATH)
        self.cfg = cfg
        self.df = DeepFake(cfg.deepfake)
        self.video_paths = ["tests/test_data/video1.mp4"]

    def test_check_video(self):
        assert self.df.check_video(self.video_paths[0])
