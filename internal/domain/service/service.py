from omegaconf import DictConfig

from internal.adapter.database.sql.user import UserAdapter
from internal.adapter.database.sql.user_photo import UserPhotoAdapter
from internal.domain.deepfake.deepfake import DeepFake


class ParimateSerive:
    def __init__(self, cfg: DictConfig, df: DeepFake,
                 user_adapter: UserAdapter,
                 user_photo_adapter: UserPhotoAdapter):
        self.df = df
        self.user_adapter = user_adapter
        self.user_photo_adapter = user_photo_adapter

    def insert_photo(self, user_id, embeddings):
        self.user_photo_adapter.insert_photo(user_id, embeddings)
