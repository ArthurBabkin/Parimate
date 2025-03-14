import numpy as np
from deepface import DeepFace

class FaceAnalysis:
    def __init__(self, detector: str = "opencv", model_name: str = "Facenet"):
        self.detector = detector
        self.model_name = model_name

    def extract_embedding(self, image: np.ndarray) -> np.ndarray:
        """Extracts face embedding from an image."""
        embeddings = DeepFace.represent(
            image, detector_backend=self.detector, model_name=self.model_name
        )
        if len(embeddings) != 1:
            raise ValueError("There should be exactly one face in the reference image.")
        return embeddings[0]['embedding']

    def verify_face(self, image: np.ndarray, reference: np.ndarray) -> bool:
        """Verifies if two face images belong to the same person."""
        result = DeepFace.verify(
            image, reference, detector_backend=self.detector, model_name=self.model_name, silent=True
        )
        return result.get('verified', False)

