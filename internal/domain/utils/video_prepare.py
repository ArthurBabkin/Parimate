from typing import List

import cv2
import numpy as np


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
    return frames
