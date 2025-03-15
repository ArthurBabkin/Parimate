import os

from omegaconf import DictConfig

from internal.adapter.database.sql import UserAdapter, UserPhotoAdapter
from internal.domain.deepfake import DeepFake
from internal.domain.audio.pipeline import SpeechValidator
from internal.domain.face_analysis import FaceAnalysis
from internal.domain.audio.video_description_matching import VideoDescriptionMatcher

class ParimateSerive:
    def __init__(self, cfg: DictConfig, df: DeepFake, 
                 sv: SpeechValidator, vd: VideoDescriptionMatcher,
                 user_adapter: UserAdapter,
                 user_photo_adapter: UserPhotoAdapter):
        self.cfg = cfg
        self.fa = FaceAnalysis(cfg.face_analysis)
        self.df = df
        self.sv = sv
        self.vd = vd
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
    
    def done_task(self, user_id: int, task_name: str, video_path: str):
        # Check metadata
        v = self._verify_video_metadata(video_path)

        # Check audio for key word
        task = [t for t in self.tasks if t["name"] == task_name][0]
        audio_check = self.sv.validate_pronunciation(video_path, task["phrase"])

        # Check video for actions or places
        video_check = self.vd.verify_description(video_path, task["description"])

        os.remove(video_path)

        return v
    def get_tasks(self, user_id: int):
        return self.tasks
    
    def get_embedings(self, image_base64: str):
       return self.fa.extract_embedding_b64(image_base64)
