from typing import List

import cv2
import numpy as np
from deepface import DeepFace


def extract_frames_from_video(video_path: str, step: int,
                              by_time: bool = True) -> List[np.ndarray]:
    """
    Извлечь фреймы из видео.
    Args:
        video_path (str): путь к видео.
        step (int): шаг извлечения фреймов.
        by_time (bool): флаг для извлечения по времени или нет.

    Returns:
        frames (list): лист фреймов.
    """
    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = int(vid.get(cv2.CAP_PROP_FPS))
    frame_step = step * fps if by_time else step

    frames = []
    frame_idx = 0

    while True:
        ret, frame = vid.read()
        if not ret:
            break

        if frame_idx % frame_step == 0:
            frames.append(frame)

        frame_idx += 1

    vid.release()

    print(len(frames))
    frames = check_laplacian(frames)
    print(len(frames))
    frames = check_face(frames)
    print(len(frames))
    return frames


def check_laplacian(frames: List[np.ndarray]) -> List[np.ndarray]:
    variance_laplacians = [
        cv2.Laplacian(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                      cv2.CV_64F).var() for frame in frames]

    p = 0.3

    frame_with_laplacian = list(zip(variance_laplacians, frames))
    frame_with_laplacian.sort(reverse=True)

    return [t[1] for t in
            frame_with_laplacian[:int(p * len(frame_with_laplacian))]]


def check_face(frames: List[np.ndarray]) -> List[np.ndarray]:
    analyze_frames = []
    for frame in frames:
        try:
            res = DeepFace.analyze(img_path=frame, actions=[])
        except:
            res = []
        analyze_frames.append(res)

    frames_with_face = []
    for i, frame in enumerate(frames):
        if len(analyze_frames[i]):
            frames_with_face.append(frame)

    return frames_with_face
