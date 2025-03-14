import os

from omegaconf import DictConfig

from internal.adapter.database.sql import UserAdapter, UserPhotoAdapter
from internal.domain.deepfake import DeepFake
from internal.domain.face_analysis import FaceAnalysis

class ParimateSerive:
    def __init__(self, cfg: DictConfig, df: DeepFake,
                 user_adapter: UserAdapter,
                 user_photo_adapter: UserPhotoAdapter):
        self.cfg = cfg
        self.df = df
        self.user_adapter = user_adapter
        self.user_photo_adapter = user_photo_adapter
        self.tasks = []

    def insert_photo(self, user_id: int, embeddings):
        self.user_photo_adapter.insert_photo(user_id, embeddings)

    def _verify_video_metadata(self, video_path: str):
        return self.df.check_video(video_path)
    
    def create_task(self, user_id: int, name: str, description: str, phrase: str):
        
        self.tasks.append({
            "user_id": user_id,
            "name": name,
            "description": description,
            "phrase": phrase
        })
        
        return True
    
    def done_task(self, user_id: int, video_path: str):
        v = self._verify_video_metadata(video_path)

        os.remove(video_path)

        return v
    def get_tasks(self, user_id: int):
        return self.tasks
    
    def get_embedings(self, image_base64: str):
       faceAnalysis = FaceAnalysis(self.cfg)
       return faceAnalysis.extract_embedding_b64(image_base64)
