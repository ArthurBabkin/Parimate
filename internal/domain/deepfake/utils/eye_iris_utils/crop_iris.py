import cv2
import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion
from skimage import exposure
from skimage.feature import canny
from skimage.transform import hough_circle, hough_circle_peaks
from typing import Tuple


def extract_reflection(img: np.ndarray, mask: np.ndarray) \
        -> Tuple[np.ndarray, np.ndarray]:
    """
    Извлечь блики и зрачок, удалив пиксели с высокой цветопередачей.
    Args:
        img (np.ndarray): Изображение роговицы.
        mask (np.ndarray): Маска радужной оболочки (boolean).

    Returns:
        highlights (np.ndarray): Маска из бликов и зрачка (boolean).
        num_refl (np.ndarray): Количество пикселей от выделенных участков.
    """
    negative_mask = np.logical_not(mask)
    roi_HSV = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    roi_V = roi_HSV[..., 2]
    roi_V = exposure.rescale_intensity(roi_V, in_range=(0, 255))
    roi_V[negative_mask] = 0
    highlights = roi_V >= 150
    num_refl = np.sum(highlights)

    pupil = roi_V <= 50
    pupil[negative_mask] = 0
    highlights = np.logical_or(highlights, pupil)

    return highlights, num_refl


def segment_iris(face_crop: np.ndarray, eye_mask: np.ndarray,
                 radius_min_para: np.ndarray, radius_max_para: np.ndarray) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                 Tuple[int, int], int, Tuple[int, int],
                 np.ndarray, np.ndarray, bool]:
    """
    Обрезайте радужную оболочку в соответствии с пересечением роговицы
    и круга Хафа.
    Args:
        face_crop (np.ndarray): Фичи глаз.
        eye_mask (np.ndarray): Маска роговицы (boolean).
        radius_min_para (np.ndarray): Масштаб минимального радиуса
        окружности Хафа.
        radius_max_para (np.ndarray): Масштаб масимального радиуса
        окружности Хафа.

    Returns:
        img_ori_copy (np.ndarray): Изображение глаза с белым
        обведенным кружком.
        img_copy (np.ndarray): Изображение радужной оболочки глаза (фон белый).
        iris_mask (np.ndarray): Маска радужной оболочки (boolean).
        (cx_glob, cy_glob) (tuple): Центр наилучшего круга Хафа.
        radius_glob (int): Наилучший радиус окружности Хафа.
        (eye_cx, eye_cy) (tuple): Среднее значение координаты роговицы.
        num_refl (np.ndarray):
        highlights_global (np.ndarray): Маска из бликов (boolean).
        valid (boolean): Флажок указывает на то, существует круг или нет.
    """

    img_copy = face_crop.copy()

    mask_coords = np.where(eye_mask != 0)
    mask_min_y = np.min(mask_coords[0])
    mask_max_y = np.max(mask_coords[0])
    mask_min_x = np.min(mask_coords[1])
    mask_max_x = np.max(mask_coords[1])

    roi_top = np.clip(mask_min_y, 0, face_crop.shape[0])
    roi_bottom = np.clip(mask_max_y, 0, face_crop.shape[0])
    roit_left = np.clip(mask_min_x, 0, face_crop.shape[1])
    roi_right = np.clip(mask_max_x, 0, face_crop.shape[1])

    roi_image = img_copy[roi_top:roi_bottom, roit_left:roi_right, :]

    roi_mask = eye_mask[roi_top:roi_bottom, roit_left:roi_right]

    roi_gray = cv2.cvtColor(roi_image, cv2.COLOR_RGB2LAB)
    roi_gray = roi_gray[..., 0]
    roi_gray = exposure.rescale_intensity(roi_gray, in_range=(0, 255))

    negative_mask = np.logical_not(roi_mask)
    roi_gray[negative_mask] = 0.0
    edges = canny(roi_gray, sigma=2.0, low_threshold=40, high_threshold=70)

    edges_mask = canny(roi_mask.astype(np.float64) * 255, sigma=1.5,
                       low_threshold=1,
                       high_threshold=240)
    edges_mask = binary_erosion(edges_mask)
    edges_mask = binary_dilation(edges_mask)
    edges_mask = np.logical_not(edges_mask)

    # определение окружностей в пределах радиуса действия на основе ориентиров
    edges = np.logical_and(edges, edges_mask)
    diam = mask_max_x - mask_min_x
    radius_min = int(diam / radius_min_para)
    radius_max = int(diam / radius_max_para)
    hough_radii = np.arange(radius_min, radius_max, 1)
    hough_res = hough_circle(edges, hough_radii)
    # выбор наилучшего обнаружения
    accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii,
                                               total_num_peaks=1,
                                               normalize=True)

    # выбор центральной точки и diam/4 в качестве запасного варианта
    if radii is None or radii.size == 0:
        cx_glob = int(np.mean(mask_coords[1]))
        cy_glob = int(np.mean(mask_coords[0]))
        radius_glob = int(diam / 4.0)
        valid = False
    else:
        cx_glob = cx[0] + mask_min_x
        cy_glob = cy[0] + mask_min_y
        radius_glob = radii[0]
        valid = True

    # создание маски для радужной оболочки
    iris_mask = np.zeros_like(eye_mask, dtype=np.uint8)
    cv2.circle(iris_mask, (cx_glob, cy_glob), radius_glob, 255, -1)
    img_ori_copy = face_crop.copy()
    cv2.circle(img_ori_copy, (cx_glob, cy_glob), radius_glob,
               (255, 255, 255), 1)

    iris_mask = np.logical_and(iris_mask, eye_mask)

    for i in range(img_copy.shape[0]):
        for j in range(img_copy.shape[1]):
            if iris_mask[i][j] == False:
                img_copy[i][j] = np.asarray([255, 255, 255])
    roi_iris = iris_mask[roi_top:roi_bottom, roit_left:roi_right]

    highlights, num_refl = extract_reflection(roi_image, roi_iris)
    highlights_global = np.zeros_like(eye_mask)
    highlights_coord = np.where(highlights != 0)
    highlights_coord[0][:] += mask_min_y
    highlights_coord[1][:] += mask_min_x
    highlights_global[highlights_coord] = 1

    eye_cx = int(np.mean(mask_coords[1]))
    eye_cy = int(np.mean(mask_coords[0]))
    return img_ori_copy, img_copy, iris_mask, (cx_glob, cy_glob), \
           radius_glob, (eye_cx, eye_cy), highlights_global, num_refl, valid
