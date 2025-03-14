import os
import subprocess
import sys
import tempfile
import time

import cv2
import numpy as np
import open_clip
import requests
import torch
from open_clip import tokenizer
from PIL import Image
from speech_model.model import YandexSpeechKit


class VideoDescriptionMatcher:
    """
    A class for verifying if a video matches a given challenge description.
    Uses CLIP model for embedding and similarity comparison.
    """
    
    # Default thresholds for similarity checks
    VIDEO_SIMILARITY_THRESHOLD = 0.6
    RECOGNIZED_TEXT_SIMILARITY_THRESHOLD = 0.27
    
    def __init__(self, model_name='ViT-H-14-378-quickgelu', pretrained='dfn5b'):
        """
        Initialize the VideoDescriptionMatcher with a specified model.
        
        Args:
            model_name (str): Name of the CLIP model to use.
            pretrained (str): Pretrained weights to use.
        """
        # Check if CUDA is available and set device accordingly
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Set the environment variable to use the first GPU if CUDA is available
        if self.device.type == "cuda":
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
        
        # Initialize model
        print("Creating model and preprocessing transforms...")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained)
        
        # Move model to the appropriate device (GPU or CPU)
        self.model = self.model.to(self.device)
    
    def verify_description(self, video_path, challenge_description):
        """
        Verifies if the video and recognized text match the challenge description.
        
        Args:
            video_path (str): Path to the video file.
            challenge_description (str): Description of the challenge in Russian.
            
        Returns:
            bool: True if the video or recognized text matches the description, False otherwise.
        """
        print("Extracting and translating recognized text from the video...")
        recognized_text = self._extract_text_from_speech(video_path)
        translated_description = self._translate_text_yandex(challenge_description)
        
        print("Getting embeddings for video, description, and recognized text...")
        video_embedding = self._get_video_embedding(video_path)
        description_embedding = self._get_text_embedding(translated_description)
        recognized_text_embedding = self._get_text_embedding(recognized_text)
        
        print("Calculating similarities...")
        video_similarity = video_embedding @ description_embedding.T
        recognized_text_similarity = video_embedding @ recognized_text_embedding.T
        
        print(f"Video similarity: {video_similarity.max()}")
        print(f"Recognized text similarity: {recognized_text_similarity.max()}")
        
        # Check if any similarity exceeds the threshold
        if (video_similarity.max() >= self.VIDEO_SIMILARITY_THRESHOLD or 
            recognized_text_similarity.max() >= self.RECOGNIZED_TEXT_SIMILARITY_THRESHOLD):
            return True
        else:
            return False
    
    def _translate_text_yandex(self, text, source_lang="ru", target_lang="en"):
        """
        Translates text from source language to target language using Yandex Translate API.
        
        Args:
            text (str): Text to be translated.
            source_lang (str): Source language code.
            target_lang (str): Target language code.
            
        Returns:
            str: Translated text.
            
        Raises:
            ValueError: If API key or folder ID is not set.
            Exception: If translation fails.
        """
        # Get API key and folder ID from environment variables
        api_key = os.getenv("YC_API_KEY")
        folder_id = os.getenv("YC_FOLDER_ID")
        
        if not api_key or not folder_id:
            raise ValueError("YС_API_KEY and YС_FOLDER_ID environment variables must be set")
        
        url = "https://translate.api.cloud.yandex.net/translate/v2/translate"
        
        headers = {
            "Authorization": f"Api-Key {api_key}",
            "Content-Type": "application/json",
        }
        
        data = {
            "folder_id": folder_id,
            "texts": [text],
            "sourceLanguageCode": source_lang,
            "targetLanguageCode": target_lang,
        }
        
        response = requests.post(url, json=data, headers=headers)
        
        if response.status_code == 200:
            return response.json()["translations"][0]["text"]
        else:
            raise Exception(f"Translation error: {response.text}")
    
    def _extract_text_from_speech(self, video_path):
        """
        Extracts and translates text from speech in a video file.
        
        Args:
            video_path (str): Path to the video file.
            
        Returns:
            str: Translated recognized text.
        """
        # Extract audio from video
        opus_audio = self._extract_opus_audio(video_path)
        
        # Create a temporary file to save the audio data
        with tempfile.NamedTemporaryFile(delete=False, suffix=".opus") as temp_audio_file:
            temp_audio_file.write(opus_audio)
            temp_audio_file_path = temp_audio_file.name
        
        try:
            # Initialize the recognizer
            recognizer = YandexSpeechKit()
            
            # Process the audio file
            recognition_result = recognizer.process_audio(temp_audio_file_path)
            recognition_result = recognition_result['ru-RU']['text']
            recognition_result = self._translate_text_yandex(recognition_result)
            
            return recognition_result
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_audio_file_path):
                os.remove(temp_audio_file_path)
    
    def _get_text_embedding(self, text):
        """
        Computes the text embedding using the model.
        
        Args:
            text (str): Text to be embedded.
            
        Returns:
            torch.Tensor: Normalized text embedding.
        """
        text_input = tokenizer.tokenize([text]).to(self.device)
        
        with torch.no_grad():
            text_features = self.model.encode_text(text_input).float()
        
        text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features
    
    def _get_video_embedding(self, video_path):
        """
        Computes the video embedding using the model.
        
        Args:
            video_path (str): Path to the video file.
            
        Returns:
            torch.Tensor: Normalized video embedding.
        """
        print("Extracting frames from video...")
        images = self._get_frames(video_path)
        
        print("Preprocessing images...")
        images_prepoc1 = [Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)) for image in images]
        images_preproc2 = [self.preprocess(image) for image in images_prepoc1]
        images_input = torch.tensor(np.stack(images_preproc2)).to(self.device)
        
        print("Computing image embeddings...")
        start_time = time.time()
        with torch.no_grad():
            image_features = self.model.encode_image(images_input).float()
        end_time = time.time()
        print(f"Image embeddings computed in {end_time - start_time:.2f} seconds.")
        
        image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features
    
    def _extract_opus_audio(self, input_video):
        """
        Extracts Opus audio from a video file using FFmpeg.
        
        Args:
            input_video (str): Path to the video file.
            
        Returns:
            bytes: Extracted audio data.
            
        Raises:
            Exception: If FFmpeg fails to extract audio.
        """
        command = [
            "ffmpeg",
            "-i", input_video,
            "-vn",                   # no video
            "-acodec", "libopus",    # Opus encoder
            "-f", "opus",            # Opus format
            "-"
        ]
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        audio_data, error = process.communicate()
        if process.returncode != 0:
            raise Exception("FFmpeg error: " + error.decode())
        return audio_data
    
    def _get_frames(self, video_path):
        """
        Extracts frames from a video file.
        
        Args:
            video_path (str): Path to the video file.
            
        Returns:
            list: List of frames extracted from the video.
        """
        images = []
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            return images
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            print("Error: Could not determine video framerate")
            return images
            
        # Get total frame count and duration
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        # If video is longer than 30 seconds, only process first 30 seconds
        max_frames = total_frames
        if duration > 30:
            max_frames = int(30 * fps)
            
        # Calculate step size to get 3 frames per second
        step_size = int(fps / 3)
        
        frame_count = 0
        while frame_count < max_frames:
            ret, frame = cap.read()
            
            if not ret:
                print("End of video or error reading frame.")
                break
            
            if frame_count % step_size == 0:
                images.append(frame)
            
            frame_count += 1
        
        cap.release()
        
        if not images:
            print("No frames were extracted from the video.")
        
        return images


if __name__ == "__main__":
    # Example usage
    video_path = "/Users/theother_archee/CursorProjects/Parimate/speech_model/parimate_test_videos/2.mp4"
    challenge_description = "Человеку надо поделиться прочитанной главе в книге"
    
    matcher = VideoDescriptionMatcher()
    result = matcher.verify_description(video_path, challenge_description)
    print(result)