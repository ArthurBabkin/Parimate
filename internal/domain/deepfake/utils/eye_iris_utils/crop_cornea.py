from typing import List, Tuple

import cv2
import numpy as np


def cornea_convex_hull(left_eye_img: np.ndarray, right_eye_img: np.ndarray,
                       new_eyes_position_list: List) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Вырезает роговицу из правого и левого глаза.
    Args:
        left_eye_img (np.ndarray): Фичи левого глаза.
        right_eye_img (np.ndarray): Фичи правого глаза.
        new_eyes_position_list (list): Лист с двумя подлистами, первый лист
        содержит новые позиции правого глаза, второй - левого глаза.

    Returns:
        left_cornea (np.ndarray): Фичи левой роговицы (gray image).
        right_cornea (np.ndarray): Фичи правой роговицы (gray image).
        left_cornea_matrix (np.ndarray): Маска левой роговицы (binary).
        right_cornea_matrix (np.ndarray): Маска левой роговицы (binary).
    """
    left_cornea = np.zeros((left_eye_img.shape[0],
                            left_eye_img.shape[1],
                            3), np.uint8)
    left_cornea_matrix = np.zeros((left_eye_img.shape[0],
                                   left_eye_img.shape[1]),
                                  np.uint8)
    cv2.fillConvexPoly(left_cornea, new_eyes_position_list[0], (255, 255, 255))

    right_cornea = np.zeros((right_eye_img.shape[0],
                             right_eye_img.shape[1],
                             3), np.uint8)
    right_cornea_matrix = np.zeros((right_eye_img.shape[0],
                                    right_eye_img.shape[1]),
                                   np.uint8)
    cv2.fillConvexPoly(right_cornea,
                       new_eyes_position_list[1],
                       (255, 255, 255))

    left_cornea_img = left_cornea
    right_cornea_img = right_cornea

    for i in range(left_cornea_img.shape[0]):
        for j in range(left_cornea_img.shape[1]):
            if (left_cornea_img[i][j][0] == 255 and
                    left_cornea_img[i][j][1] == 255 and
                    left_cornea_img[i][j][2] == 255):
                left_cornea_matrix[i][j] = 1

    for i in range(right_cornea_img.shape[0]):
        for j in range(right_cornea_img.shape[1]):
            if (right_cornea_img[i][j][0] == 255 and
                    right_cornea_img[i][j][1] == 255 and
                    right_cornea_img[i][j][2] == 255):
                right_cornea_matrix[i][j] = 1
    return left_cornea, right_cornea, left_cornea_matrix, right_cornea_matrix
