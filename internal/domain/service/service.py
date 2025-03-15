import os

from omegaconf import DictConfig

from internal.adapter.database.sql import UserAdapter, UserPhotoAdapter
from internal.domain.deepfake import DeepFake
from internal.domain.audio.pipeline.pipeline import SpeechValidator
from internal.domain.face_analysis import FaceAnalysis
from internal.domain.audio.video_description_matching import VideoDescriptionMatcher
from internal.domain.service.multithread_handler import MultithreadHandler
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
        self.tasks_done = []
        self.current_video_path = None
        self.telegram_callback = None

    def insert_photo(self, user_id: int, embeddings):
        self.user_photo_adapter.insert_photo(user_id, embeddings)

    def _verify_video_metadata(self, video_path: str):
        return self.df.check_video(video_path)
    
    def _verify_face(self, img, emb):
        return self.fa.verify_face(img, emb)
    
    def create_task(self, user_id: int, name: str, description: str, phrase: str):
        
        self.tasks.append({
            "user_id": user_id,
            "name": name,
            "description": description,
            "phrase": phrase
        })
        
        return True
    
    def done_task(self, user_id: int, task_id: int, video_path: str, on_done:callable):
        task = self.tasks[task_id]
        emb = self.user_photo_adapter.get_photo(user_id)
        img = self.df.get_frame(video_path)
        self.telegram_callback = on_done
         
        # Check metadata
        deep_fake = self._verify_video_metadata(video_path)

        # emb = get embedding from db by user_id
        # img = frame from the video
        face_check = self._verify_face(emb, img)

        # Check audio for key word
        audio_check = self.sv.validate_pronunciation(video_path, task["phrase"])

        # Check video for actions or places
        video_check = self.vd.verify_description(video_path, task["description"])
        
        self.current_video_path = video_path
        MultithreadHandler.run_in_threads([deep_fake, face_check, audio_check, video_check], self.on_fi_done)
    
  
    def on_fi_done(self, func_id, result):
        print(f"Function {func_id} finished with result {result}")
        
        self.tasks_done.append({
            "task_id": func_id,
            "result": result
        })
        
        if(self.current_video_path):
            os.remove(self.current_video_path)
            
        if(len(self.tasks_done) == 4):
            print("All checks done!")
            print(self.tasks_done)
            
            if self.telegram_callback:
                self.telegram_callback(self.tasks_done)
                
            self.tasks_done = []
            self.current_video_path = None

    
    
    def get_tasks(self, user_id: int):
        return self.tasks
    
    def get_embedings(self, image_base64: str):
       return self.fa.extract_embedding_b64(image_base64)
