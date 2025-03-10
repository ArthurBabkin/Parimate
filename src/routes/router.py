from fastapi import APIRouter, UploadFile, File, Form

router = APIRouter()

@router.post('/upload_image')
def upload_image(
        img: UploadFile = File()
):
    """upload image of the user

    Args:
        img (UploadFile, optional): _description_. Defaults to File().
    """
    pass

@router.post('/get_analysis')
def get_analysis(
    video: UploadFile = File(),
    description: str = Form(...),
    key_word: str = Form(...),
):
    """
    Get analysis for the video. The steps are:
    1) check if the person is the user
    2) check if the video provided match the description
    3) check if the key_word was pronounced in the video

    Args:
        video (UploadFile, optional): video to analyse.
        description: text description of the challenge
        key_word: key word to verify
    """
    pass