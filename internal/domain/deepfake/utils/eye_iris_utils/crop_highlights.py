import math

import cv2
import numpy as np
import skimage.filters as filter
from typing import Tuple, List


def matrix_reduce(iris_left_matrix: np.ndarray,
                  iris_right_matrix: np.ndarray) \
        -> Tuple[np.ndarray, np.ndarray]:
    """
    Уменьшить радужную оболочку.
    Args:
        iris_left_matrix (np.ndarray): Маска левой радужной оболочки (boolean).
        iris_right_matrix (np.ndarray): Маска правой радужной
        оболочки (boolean).

    Returns:
        reduced_iris_left_matrix (np.ndarray): Уменьшенная маска левой
        радужной оболочки (boolean).
        reduced_iris_right_matrix (np.ndarray): Уменьшенная маска правой
        радужной оболочки (boolean).
    """
    reduced_iris_left_matrix = np.zeros(
        (iris_left_matrix.shape[0], iris_left_matrix.shape[1]), dtype=int)
    reduced_iris_right_matrix = np.zeros(
        (iris_right_matrix.shape[0], iris_right_matrix.shape[1]), dtype=int)

    for i in range(iris_left_matrix.shape[0]):
        for j in range(iris_left_matrix.shape[1]):
            if iris_left_matrix[i][j] == 1:
                if iris_left_matrix[i - 1][j] == 0 or \
                        iris_left_matrix[i + 1][j] == 0 or \
                        iris_left_matrix[i][j - 1] == 0 or \
                        iris_left_matrix[i][j + 1] == 0:
                    reduced_iris_left_matrix[i][j] = 0
                else:
                    reduced_iris_left_matrix[i][j] = 1
            else:
                reduced_iris_left_matrix[i][j] = 0

    for i in range(iris_right_matrix.shape[0]):
        for j in range(iris_right_matrix.shape[1]):
            if iris_right_matrix[i][j] == 1:
                if iris_right_matrix[i - 1][j] == 0 or \
                        iris_right_matrix[i + 1][j] == 0 or \
                        iris_right_matrix[i][j - 1] == 0 or \
                        iris_right_matrix[i][j + 1] == 0:
                    reduced_iris_right_matrix[i][j] = 0
                else:
                    reduced_iris_right_matrix[i][j] = 1
            else:
                reduced_iris_right_matrix[i][j] = 0

    return reduced_iris_left_matrix, reduced_iris_right_matrix


def shiftbits(template: np.ndarray, noshifts: int,
              matrix: bool = False) -> np.ndarray:
    """
    Сдвинуть побитовые схемы выделения.
    Args:
        template (np.ndarray): Маска для выделения (boolean).
        noshifts (int): Размер шага и направление перемещения.
        matrix (bool): Заполнять пустой элемент в маске или нет.

    Returns:
        templatenew (np.ndarray): Сдвинутая маска бликов (boolean).
    """
    templatenew = np.zeros(template.shape)
    width = template.shape[1]
    s = np.abs(noshifts)
    p = width - s

    if noshifts == 0:
        templatenew = template

    elif noshifts < 0:
        x = np.arange(p)
        templatenew[:, x] = template[:, s + x]
        x = np.arange(p, width)
        if matrix:
            templatenew[:, x] = 0
        else:
            templatenew[:, x] = template[:, x - p]

    else:
        x = np.arange(s, width)
        templatenew[:, x] = template[:, x - s]
        x = np.arange(s)
        if matrix:
            templatenew[:, x] = 0
        else:
            templatenew[:, x] = template[:, p + x]

    return templatenew


def shift(img_left_matrix: np.ndarray, img_right_matrix: np.ndarray,
          negative_lr_shift: float, positive_lr_shift: float,
          negative_ud_shift: float, positive_ud_shift: float) \
        -> Tuple[np.ndarray, float, List, float]:
    """
    Переместить маску левых бликов вверх, вниз, влево и вправо, чтобы
    максимально приблизить ее к маске правых бликов, чтобы получить
    наилучший результат IOU.
    Args:
        img_left_matrix (np.ndarray): Маска из левых бликов (boolean).
        img_right_matrix (np.ndarray): Маска из правых бликов (boolean).
        negative_lr_shift (float): Максимальный размер шага перемещения влево.
        positive_lr_shift (float): Максимальный размер шага перемещения вправо.
        negative_ud_shift (float): Максимальный размер шага при перемещении
        вверх.
        positive_ud_shift (float): Максимальный размер шага при перемещении
        вниз.

    Returns:
        opt_img_left_shift (np.ndarray): Оптимальная маска для оставленных
        бликов после перемещения.
        max_overlap (float): Максимальное количество перекрывающихся пикселей.
        opt_shift (list): Оптимальные этапы перемещения.
        IOU_score (float): Лучший IOU.
    """
    max_overlap = -math.inf
    IOU_score = 0
    opt_shift = []
    opt_img_left_shift = []
    for shifts_lr in range(negative_lr_shift, positive_lr_shift):
        for shift_ud in range(negative_ud_shift, positive_ud_shift):
            img_left_ud_shift = shiftbits(img_left_matrix, shift_ud,
                                          matrix=True)
            img_left_lr_shift = np.transpose(
                shiftbits(np.transpose(img_left_ud_shift), shifts_lr,
                          matrix=True))
            m = np.sum(
                np.logical_and(img_left_lr_shift, img_right_matrix).astype(
                    int))
            union_individual = np.sum(
                np.logical_or(img_left_lr_shift, img_right_matrix).astype(int))
            if m >= max_overlap:
                max_overlap = m
                if union_individual == 0:
                    IOU_score = 0.
                else:
                    IOU_score = m / union_individual
                opt_shift = [shift_ud, shifts_lr]
                opt_img_left_shift = img_left_lr_shift
    return opt_img_left_shift, max_overlap, opt_shift, IOU_score


def process_aligned_image(iris_left: np.ndarray, iris_right: np.ndarray,
                          iris_left_matrix: np.ndarray,
                          iris_right_matrix: np.ndarray,
                          l_highlights: np.ndarray, r_highlights: np.ndarray,
                          left_eye_image: np.ndarray,
                          right_eye_image: np.ndarray,
                          double_eye_img: np.ndarray,
                          double_eye_position_difference_list: np.ndarray,
                          reduce: bool = True, reduce_size: int = 2,
                          threshold_scale_left: int = 1,
                          threshold_scale_right: int = 1) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                 np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
    """
    Обрезать блики на левой и правой радужной оболочке.
    Args:
        iris_left (np.ndarray): Изображение левой радужной оболочки
        (фон белый).
        iris_right (np.ndarray): Изображение правой радужной оболочки
        (фон белый).
        iris_left_matrix (np.ndarray): Маска левой радужной оболочки (boolean).
        iris_right_matrix (np.ndarray): Маска правой радужной оболочки
        (boolean).
        l_highlights (np.ndarray): Маска левых бликов (boolean).
        r_highlights (np.ndarray): Маска правых бликов (boolean).
        left_eye_image (np.ndarray): Фичи левого глаза.
        right_eye_image (np.ndarray): Фичи правого глаза.
        double_eye_img (np.ndarray): Последовательные элементы в области глаз,
        взятые с лица.
        double_eye_position_difference_list (np.ndarray): Расстояние между
        списком new_eyes_position_list и списком double_eye_list.
        reduce (bool): Уменьшить радужку или нет.
        reduce_size (int): Размер шага должен уменьшаться от края к
        внутренней стороне.
        threshold_scale_left (int): Значение для увеличения или уменьшения
        порогового значения для левой радужной оболочки.
        threshold_scale_right (int): Значение для увеличения или уменьшения
        порогового значения для правой радужной оболочки.

    Returns:
        iris_left (np.ndarray): Измененное изображение левой радужной
        оболочки.
        iris_right (np.ndarray): Измененное изображение правой радужной
        оболочки.
        left_recolor (np.ndarray): Отображает только блики (черным цветом) в
         левой радужной оболочке на белом фоне.
        right_recolor (np.ndarray): Отображает только блики (черным цветом) в
        правой радужной оболочке на белом фоне.
        left_recolor_resize (np.ndarray): Изменяет размер изображения левой
        радужной оболочки и выделите основные моменты зеленым цветом.
        right_recolor_resize (np.ndarray): Изменяет размер изображения правой
        радужной оболочки и выделите основные моменты зеленым цветом.
        IOU_score (float): Рассчитывает свой результат, основываясь на
        совпадении левых и правых бликов.
        double_eye_img_modified (np.ndarray): Показывает блики (зеленым
        цветом слева и красным справа) на обоих глазах в double_eye_img.
    """

    # уменьшение границы радужной оболочки
    double_eye_img_modified = double_eye_img.copy()
    if reduce:
        for i in range(reduce_size):
            iris_left_matrix, iris_right_matrix = matrix_reduce(
                iris_left_matrix, iris_right_matrix)
        left_matrix = iris_left_matrix
        right_matrix = iris_right_matrix
        for i in range(left_matrix.shape[0]):
            for j in range(left_matrix.shape[1]):
                if left_matrix[i][j] != 1:
                    iris_left[i][j] = np.asarray([255, 255, 255])

        for i in range(right_matrix.shape[0]):
            for j in range(right_matrix.shape[1]):
                if right_matrix[i][j] != 1:
                    iris_right[i][j] = np.asarray([255, 255, 255])
    else:
        left_matrix = iris_left_matrix
        right_matrix = iris_right_matrix

    left_matrix_new = np.logical_xor(left_matrix, l_highlights)
    right_matrix_new = np.logical_xor(right_matrix, r_highlights)
    l_iris_vals = left_eye_image[left_matrix_new, :]
    r_iris_vals = right_eye_image[right_matrix_new, :]
    lIrisMean = np.mean(l_iris_vals, axis=0).astype(int)
    rIrisMean = np.mean(r_iris_vals, axis=0).astype(int)

    iris_left_ori_reduce_iris_color = iris_left.astype(int) - lIrisMean
    iris_right_ori_reduce_iris_color = iris_right.astype(int) - rIrisMean
    iris_left_ori_reduce_iris_color[iris_left_ori_reduce_iris_color < 0] = 0
    iris_right_ori_reduce_iris_color[iris_right_ori_reduce_iris_color < 0] = 0
    iris_left_ori_reduce_iris_color = iris_left_ori_reduce_iris_color.astype(
        np.uint8)
    iris_right_ori_reduce_iris_color = iris_right_ori_reduce_iris_color.astype(
        np.uint8)

    # вычислиние порогов
    iris_left_HSV = cv2.cvtColor(iris_left_ori_reduce_iris_color,
                                 cv2.COLOR_BGR2HSV)
    iris_right_HSV = cv2.cvtColor(iris_right_ori_reduce_iris_color,
                                  cv2.COLOR_BGR2HSV)

    left_color_list = []
    for i in range(left_matrix.shape[0]):
        for j in range(left_matrix.shape[1]):
            if left_matrix[i][j] == 1:
                left_color_list.append(iris_left_HSV[i][j])

    right_color_list = []
    for i in range(right_matrix.shape[0]):
        for j in range(right_matrix.shape[1]):
            if right_matrix[i][j] == 1:
                right_color_list.append(iris_right_HSV[i][j])

    the_left_V = filter.threshold_yen(
        np.asarray(left_color_list)[:, 2]) * threshold_scale_left
    the_right_V = filter.threshold_yen(
        np.asarray(right_color_list)[:, 2]) * threshold_scale_right

    # извлечение бликов
    left_recolor = np.zeros((iris_left.shape[0], iris_left.shape[1], 3),
                            dtype=np.uint8)
    right_recolor = np.zeros((iris_right.shape[0], iris_right.shape[1], 3),
                             dtype=np.uint8)
    left_recolor_matrix = np.zeros((iris_left.shape[0], iris_left.shape[1]),
                                   dtype=int)
    right_recolor_matrix = np.zeros((iris_right.shape[0], iris_right.shape[1]),
                                    dtype=int)

    for i in range(left_matrix.shape[0]):
        for j in range(left_matrix.shape[1]):
            if left_matrix[i][j] == 1:
                if iris_left_HSV[i][j][2] > the_left_V:
                    left_recolor[i][j] = np.asarray([0, 0, 0])
                    left_recolor_matrix[i][j] = 1
                    double_eye_img_modified[
                        i + double_eye_position_difference_list[0][1]][
                        j + double_eye_position_difference_list[0][0]] \
                        = np.asarray([0, 255, 0])
                else:
                    left_recolor[i][j] = np.asarray([255, 255, 255])
                    left_recolor_matrix[i][j] = 0
            else:
                left_recolor[i][j] = np.asarray([255, 255, 255])
                left_recolor_matrix[i][j] = 0

    for i in range(right_matrix.shape[0]):
        for j in range(right_matrix.shape[1]):
            if right_matrix[i][j] == 1:
                if iris_right_HSV[i][j][2] > the_right_V:
                    right_recolor[i][j] = np.asarray([0, 0, 0])
                    right_recolor_matrix[i][j] = 1
                    double_eye_img_modified[
                        i + double_eye_position_difference_list[1][1]][
                        j + double_eye_position_difference_list[1][
                            0]] = np.asarray([255, 0, 0])
                else:
                    right_recolor[i][j] = np.asarray([255, 255, 255])
                    right_recolor_matrix[i][j] = 0
            else:
                right_recolor[i][j] = np.asarray([255, 255, 255])
                right_recolor_matrix[i][j] = 0

    # создание двух согласованных изображений и матриц
    max_x_axis = max(iris_left.shape[0], iris_right.shape[0])
    max_y_axis = max(iris_left.shape[1], iris_right.shape[1])
    left_ori_resize = np.full((max_x_axis, max_y_axis, 3), 255, dtype=np.uint8)
    right_ori_resize = np.full((max_x_axis, max_y_axis, 3), 255,
                               dtype=np.uint8)
    left_recolor_resize = np.full((max_x_axis, max_y_axis, 3), 255,
                                  dtype=np.uint8)
    right_recolor_resize = np.full((max_x_axis, max_y_axis, 3), 255,
                                   dtype=np.uint8)
    left_recolor_matrix_resize = np.zeros((max_x_axis, max_y_axis), dtype=int)
    right_recolor_matrix_resize = np.zeros((max_x_axis, max_y_axis), dtype=int)
    left_matrix_resize = np.zeros((max_x_axis, max_y_axis), dtype=int)
    right_matrix_resize = np.zeros((max_x_axis, max_y_axis), dtype=int)

    for i in range(left_recolor.shape[0]):
        for j in range(left_recolor.shape[1]):
            left_ori_resize[i][j] = iris_left[i][j]
            left_recolor_matrix_resize[i][j] = left_recolor_matrix[i][j]
            left_matrix_resize[i][j] = left_matrix[i][j]

    for i in range(right_recolor.shape[0]):
        for j in range(right_recolor.shape[1]):
            right_ori_resize[i][j] = iris_right[i][j]
            right_recolor_matrix_resize[i][j] = right_recolor_matrix[i][j]
            right_matrix_resize[i][j] = right_matrix[i][j]

    # сдвиг
    opt_img_left_shift, \
    max_overlap, \
    opt_shift, \
    IOU_score = \
        shift(left_recolor_matrix_resize, right_recolor_matrix_resize,
              -int(max_x_axis / 6), int(max_x_axis / 6),
              -int(max_y_axis / 5), int(max_y_axis / 5))

    left_matrix_ud_resize = shiftbits(left_matrix_resize, opt_shift[0],
                                      matrix=True)
    left_matrix_lr_resize = np.transpose(
        shiftbits(np.transpose(left_matrix_ud_resize), opt_shift[1],
                  matrix=True))
    for i in range(opt_img_left_shift.shape[0]):
        for j in range(opt_img_left_shift.shape[1]):
            if left_matrix_lr_resize[i][j] == 1:
                if opt_img_left_shift[i][j] == 1:
                    left_recolor_resize[i][j] = np.asarray([0, 255, 0])
                else:
                    left_recolor_resize[i][j] = np.asarray([255, 255, 255])
            else:
                left_recolor_resize[i][j] = np.asarray([255, 255, 255])
            if right_matrix_resize[i][j] == 1:
                if right_recolor_matrix_resize[i][j] == 1:
                    right_recolor_resize[i][j] = np.asarray([255, 0, 0])
                else:
                    right_recolor_resize[i][j] = np.asarray([255, 255, 255])
            else:
                right_recolor_resize[i][j] = np.asarray([255, 255, 255])
    return iris_left, iris_right, left_recolor, right_recolor, \
           left_recolor_resize, right_recolor_resize, IOU_score, \
           double_eye_img_modified
