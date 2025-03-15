from omegaconf import DictConfig

from internal.adapter.database.sql import UserAdapter, UserPhotoAdapter
from internal.domain.deepfake import DeepFake
from internal.domain.service import ParimateSerive
from internal.storage.database.sqlite3 import get_connection, init_database
from internal.transport.tg import ParimateBot
from internal.domain.audio.pipeline.pipeline import SpeechValidator
from internal.domain.audio.video_description_matching import VideoDescriptionMatcher

class App:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

        deepfake = DeepFake(self.cfg.deepfake)
        sv = SpeechValidator()
        vd = VideoDescriptionMatcher()

        conn = get_connection(self.cfg.database.path)
        init_database(conn, self.cfg.database.path_init)

        user_adapter = UserAdapter(conn)
        user_photo_adapter = UserPhotoAdapter(conn)
        
        service = ParimateSerive(self.cfg, deepfake,sv, vd, user_adapter,
                                 user_photo_adapter)
        self.bot = ParimateBot(self.cfg.tg, service)

    def run(self):
        self.bot.run()
