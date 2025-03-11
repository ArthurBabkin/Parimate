from deepface import DeepFace
import cv2
import itertools

def extract_embedding(image):
    try:
        embs = DeepFace.represent(image)
    except ValueError as e:
        return None

    assert len(embs) == 1, "There should be 1 face on the reference image"

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

