from deepface import DeepFace
import cv2

def check_face(image, reference, kwargs):
    """image - new photo
    reference - reference embedding/photo

    Args:
        image (np.array): new photo
        reference (np.array): reference embedding/photo
    """
    result = DeepFace.verify(
        image, reference, **kwargs
    )
    return result['verified']


if __name__ == "__main__":
    f = cv2.imread('./test_img/1.jpg')
    ref = cv2.imread('./test_img/2.jpg')

    print(check_face(f, ref))
