from omegaconf import DictConfig

from internal.adapter.database.sql.user import UserAdapter
from internal.adapter.database.sql.user_photo import UserPhotoAdapter
from internal.domain.deepfake.deepfake import DeepFake
from internal.domain.service.service import ParimateSerive
from internal.storage.database.sqlite3.connection import (get_connection,
                                                          init_database)
from internal.transport.tg.telegram_bot import ParimateBot


class App:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

        deepfake = DeepFake(self.cfg.deepfake)

        conn = get_connection(self.cfg.database.path)
        init_database(conn, self.cfg.database.path_init)

        user_adapter = UserAdapter(conn)
        user_photo_adapter = UserPhotoAdapter(conn)
        service = ParimateSerive(self.cfg, deepfake, user_adapter,
                                 user_photo_adapter)
        self.bot = ParimateBot(self.cfg.tg, service)

    def run(self):
        self.bot.run()
