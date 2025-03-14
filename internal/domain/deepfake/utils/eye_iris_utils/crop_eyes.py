from typing import List, Tuple

import dlib
import numpy as np


def crop_eye(img: np.ndarray, left: np.ndarray, right: np.ndarray) \
        -> Tuple[List, List, np.ndarray, np.ndarray]:
    """
    Вырезает глаза с изображения лица.
    Args:
        img (np.ndarray): Фичи лица.
        left (np.ndarray): Кординаты левого глаза.
        right (np.ndarray): Координаты правого глаза.

    Returns:
        eyes_list (list): Фичи левого и правого глаза.
        new_eyes_position_list (list): Лист с двумя подлистами, первый лист
        содержит новые позиции левого глаза, второй - правого глаза.
        double_eye_list (list): Последовательные элементы в области двойных
        глаз, взятые с лица.
        double_eye_position_difference_list (np.ndarray): Расстояние между
        списком new_eyes_position_list и списком double_eye_list
    """
    eyes_list = []
    new_eyes_position_list = []
    rescale_position_list = []
    left_eye = left
    right_eye = right
    eyes = [left_eye, right_eye]
    lp_min = float("inf")
    rp_max = -float("inf")
    tp_min = float("inf")
    bp_max = -float("inf")
    for j in range(len(eyes)):
        lp = np.min(eyes[j][:, 0])
        rp = np.max(eyes[j][:, 0])
        tp = np.min(eyes[j][:, -1])
        bp = np.max(eyes[j][:, -1])
        if lp < lp_min:
            lp_min = lp
        if rp > rp_max:
            rp_max = rp
        if tp < tp_min:
            tp_min = tp
        if bp > bp_max:
            bp_max = bp
        w = rp - lp
        h = bp - tp
        lp_ = int(np.maximum(0, lp - 0.25 * w))
        rp_ = int(np.minimum(img.shape[1], rp + 0.25 * w))  # 0.25
        tp_ = int(np.maximum(0, tp - 1.75 * h))
        bp_ = int(np.minimum(img.shape[0], bp + 1.75 * h))  # 1.75

        eyes_list.append(img[tp_:bp_, lp_:rp_, :])
        new_eye = eyes[j] - [lp_, tp_]
        new_eyes_position_list.append(new_eye)
        rescale_position_list.append([lp_, tp_])
    double_eye_list = img[tp_min - 5:bp_max + 5, lp_min - 1:rp_max + 1, :]
    double_eye_position_difference_list = np.asarray(
        rescale_position_list) - np.asarray([lp_min - 1, tp_min - 5])
    return eyes_list, new_eyes_position_list, \
           double_eye_list, double_eye_position_difference_list


def drawPoints(faceLandmarks, startpoint: int, endpoint: int) -> np.ndarray:
    """
    Получить координаты глаза на лице.
    Args:
        faceLandmarks (): Ориентиры/детали для лица.
        startpoint (int): Начальная точка глаз.
        endpoint (int): Конечная точка глаз.

    Returns:
        points (np.ndarray): Координаты глаза определяются на основе
        ориентиров лица.
    """
    points = []
    for i in range(startpoint, endpoint + 1):
        point = [faceLandmarks.part(i).x, faceLandmarks.part(i).y]
        points.append(point)
    points = np.array(points, dtype=np.int32)
    return points


def eye_detection(img: np.ndarray, predictor_path: str) \
        -> Tuple[np.ndarray, np.ndarray, List, int, np.ndarray, np.ndarray]:
    """
    Возвращает снимок лица для детектирования глаз.
    Args:
        data (str): Путь к изображению лица.
        predictor_path (str): Путь к dlib знаковому предсказателю.

    Returns:
        left_eye_image (list): Фичи левого глаза.
        right_eye_image (list): Фичи правого глаза.
        new_eyes_position_list (list): Лист с двумя подлистами, первый лист
        содержит новые позиции правого глаза, второй - левого глаза.
        len(dets) (int): Количество лиц на изображении.
        double_eye_list (list): Последовательные элементы в области глаз,
        взятые с лица.
        double_eye_position_difference_list (np.ndarray): Расстояние между
        списком new_eyes_position_list и списком double_eye_list
    """
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    # Попросите детектор найти ограничивающие рамки каждой грани.
    # Цифра 1 во втором аргументе указывает на то, что мы должны выполнить
    # повторную выборку изображения 1 раз. Этот увеличит изображение и
    # позволит нам распознавать больше лиц.
    dets = detector(img, 1)
    d = dlib.rectangle(int(dets[0].left()), int(dets[0].top()),
                       int(dets[0].right()), int(dets[0].bottom()))
    # Найдите ориентиры/детали для лица в блоке d.
    shape = predictor(img, d)
    left_eye = drawPoints(shape, 36, 41)
    right_eye = drawPoints(shape, 42, 47)

    eyes_list, \
    new_eyes_position_list, \
    double_eye_list, \
    double_eye_position_difference_list = \
        crop_eye(img, left_eye, right_eye)

    left_eye_image = eyes_list[0]
    right_eye_image = eyes_list[1]
    return left_eye_image, right_eye_image, new_eyes_position_list, \
           len(dets), double_eye_list, double_eye_position_difference_list
