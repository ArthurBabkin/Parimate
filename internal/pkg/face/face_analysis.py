import base64

import cv2
import numpy as np
from deepface import DeepFace


def convert_base64_to_np(img_b64):
    return cv2.imdecode(np.frombuffer(base64.b64decode(img_b64), dtype=np.uint8), -1)

def extract_embedding(image):
    embs = DeepFace.represent(convert_base64_to_np(image))

    if len(embs) != 1:
        raise ValueError("There should be 1 face on the reference image")

    return embs[0]['embedding']


def check_face(image, reference):
    """image - new photo
    reference - reference embedding/photo

    Args:
        image (np.array): new photo
        reference (np.array): reference embedding/photo
    """
    result = DeepFace.verify(
        image, reference
    )
    return result['verified']
