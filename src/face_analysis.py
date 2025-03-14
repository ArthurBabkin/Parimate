import base64
import cv2
import numpy as np
from deepface import DeepFace

class FaceAnalysis:
    def __init__(self, detector: str = "opencv", model_name: str = "Facenet"):
        self.detector = detector
        self.model_name = model_name

    def extract_embedding(self, image: np.ndarray) -> list[float]:
        """Extracts face embedding from an image."""
        embeddings = DeepFace.represent(
            image, detector_backend=self.detector, model_name=self.model_name
        )
        if len(embeddings) != 1:
            raise ValueError("There should be exactly one face in the reference image.")
        return embeddings[0]['embedding']
    
    def extract_embedding_b64(self, image: str) -> list[float]:
        """
        Converts a base64-encoded image to a NumPy array and extracts its face embedding.
        """
        return self.extract_embedding(convert_base64_to_np(image))

    def verify_face(self, image: np.ndarray, reference: list[float]) -> bool:
        """Verifies if two face images belong to the same person."""
        result = DeepFace.verify(
            image, reference, detector_backend=self.detector, model_name=self.model_name, silent=True
        )
        return result.get('verified', False)
    
    def verify_face_b64(self, image: str, reference: list[float]):
        return self.verify_face(convert_base64_to_np(image), reference)

def convert_base64_to_np(img_b64: str) -> np.ndarray:
    """
    Converts a base64-encoded image string into a NumPy array.
    """
    return cv2.imdecode(np.frombuffer(base64.b64decode(img_b64), dtype=np.uint8), -1)
