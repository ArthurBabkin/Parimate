import cv2
import numpy as np
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from sqlalchemy.orm import Session
from src.services.face_analysis import *

router = APIRouter()

@router.post('/upload_image')
def upload_image(
    tg_id: int,
    img: UploadFile = File(),
):
    """Upload image of the user

    Args:
        img (UploadFile, optional): Image file from user. Defaults to File().
        db (Session, optional): SQLAlchemy DB session from dependency.
    """
    emb = extract_embedding(
        cv2.imdecode(np.frombuffer(img.file.read(), dtype=np.uint8), -1)
        )
    if emb is None:
        raise HTTPException(401, "There is no face on the image")
    return {'user_embedding': emb}



@router.post('/get_analysis')
def get_analysis(
    tg_id: int,
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