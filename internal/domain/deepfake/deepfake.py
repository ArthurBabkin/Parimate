from datetime import datetime
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from exiftool import ExifToolHelper
from omegaconf import DictConfig
import tensorflow as tf

from ..utils import extract_frames_from_video
from .utils.eye_iris_utils import (cornea_convex_hull, eye_detection,
                                   process_aligned_image, segment_iris)
from .utils.mesonet import Meso4


class DeepFakeMetadata:
    """
    Класс для проверки верификации метаданных в видео на
    предмет редактирования.
    """

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg.metadata

    def extract_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        Извлечь метаданные из видео.
        Args:
            file_path (str): путь к видео.

        Returns:
            metadata (dict): словарь метаданных.
        """
        try:
            with ExifToolHelper(common_args=["-G0", "-a", "-u", "-ee"]) as et:
                metadata = et.get_metadata(file_path)[0]
                return metadata
        except Exception as e:
            raise RuntimeError(f"Ошибка извлечения метаданных: {str(e)}")

    def parse_duration(self, duration_str: str) -> Optional[float]:
        """
        Спарсить длительность.
        Args:
            duration_str (str): длительность в строковом формате.

        Returns:
            Значение длительности в float формате.
        """
        try:
            if isinstance(duration_str, (int, float)):
                return float(duration_str)

            clean_str = duration_str.split('+')[0].split('.')[0]

            parts = list(map(float, clean_str.replace(':', ' ').split()))
            if len(parts) == 3:
                return parts[0] * 3600 + parts[1] * 60 + parts[2]
            elif len(parts) == 1:
                return parts[0]
            return None
        except:
            return None

    def _check_software(self, report: Dict[str, List],
                        metadata: Dict[str, Any]) -> None:
        """
        Проверить метаданные на наличие следов редактирования с помощью ПО.
        Args:
            report (dict): репорт для верификации видео.
            metadata (dict): метаданные видео.

        Returns:
            Функция добавляет результат в report.
        """
        check_name = "_check_software"

        software_fields = {
            'Software': 'Software',
            'EncodingTool': 'Encoding Tool',
            'EncodedBy': 'Encoded By',
            'WritingLibrary': 'Writing Library',
            'MuxingApp': 'Muxing App',
            'HistorySoftwareAgent': 'История изменений в XMP',
            'ProcessingTool': 'Инструмент обработки',
        }

        for field, name in software_fields.items():
            if value := metadata.get(field):
                report[check_name].append(f"{field} {value}")

    def _check_time(self, report: Dict[str, List],
                    metadata: Dict[str, Any]) -> None:
        """
        Проверить метаданные на несовпадение длительности.
        Args:
            report (dict): репорт для верификации видео.
            metadata (dict): метаданные видео.

        Returns:
            Функция добавляет результат в report.
        """
        check_name = "_check_time"

        date_formats = [
            '%Y:%m:%d %H:%M:%S',
            '%Y-%m-%dT%H:%M:%S',
            '%Y-%m-%d %H:%M:%S'
        ]

        date_pairs = [
            ('CreateDate', 'ModifyDate'),
            ('DateTimeOriginal', 'FileModifyDate'),
            ('MediaCreateDate', 'MediaModifyDate')
        ]

        for create_field, modify_field in date_pairs:
            if create_field in metadata and modify_field in metadata:
                try:
                    for fmt in date_formats:
                        try:
                            create_date = datetime.strptime(
                                metadata[create_field].split('')[0], fmt)
                            modify_date = datetime.strptime(
                                metadata[modify_field].split('.')[0], fmt)
                        except ValueError:
                            continue

                    if modify_date != create_date:
                        report[check_name].append(f"{modify_date} "
                                                  f"{create_date}")
                except Exception:
                    pass

    def _check_audio_duration(self, report: Dict[str, List],
                              metadata: Dict[str, Any]) -> None:
        """
        Проверить метаданные на несовпадение длительности для аудио.
        Args:
            report (dict): репорт для верификации видео.
            metadata (dict): метаданные видео.

        Returns:
            Функция добавляет результат в report.
        """
        check_name = "_check_audio_duration"

        duration_fields = ['Duration', 'MediaDuration',
                           'TrackDuration', 'AudioDuration']
        durations = {}

        for field in duration_fields:
            if value := metadata.get(field):
                if dur := self.parse_duration(value):
                    durations[field] = dur

        threshold = self.cfg.audio_threshold
        for main_field in ['Duration', 'MediaDuration']:
            for compare_field in ['TrackDuration', 'AudioDuration']:
                if main_field in durations and compare_field in durations:
                    delta_dur = abs(durations[main_field] -
                                    durations[compare_field])
                    if delta_dur > threshold:
                        report[check_name].append(
                            f"{main_field}={durations[main_field]} vs "
                            f"{compare_field}={durations[compare_field]})"
                        )

    def _check_original_parameters(self, report: Dict[str, List],
                                   metadata: Dict[str, Any]) -> None:
        """
        Проверить метаданные на несовпадение оригинальных значений и текущих.
        Args:
            report (dict): репорт для верификации видео.
            metadata (dict): метаданные видео.

        Returns:
            Функция добавляет результат в report.
        """
        check_name = "_check_original_parameters"

        video_checks = {
            'FrameRate': 'OriginalFrameRate',
            'ImageWidth': 'OriginalImageWidth',
            'ImageHeight': 'OriginalImageHeight'
        }

        threshold = self.cfg.original_threshold
        for current, original in video_checks.items():
            if current in metadata and original in metadata:
                try:
                    current_val = float(metadata[current])
                    original_val = float(metadata[original])
                    if abs(current_val - original_val) > threshold:
                        report[check_name].append(
                            f"{current}={current_val} vs "
                            f"{original}={original_val}"
                        )
                except:
                    pass

    def _check_comment(self, report: Dict[str, List],
                       metadata: Dict[str, Any]) -> None:
        """
        Проверить комментарии на наличие следов редактирования.
        Args:
            report (dict): репорт для верификации видео.
            metadata (dict): метаданные видео.

        Returns:
            Функция добавляет результат в report.
        """
        check_name = "_check_comment"

        field = 'Comment'

        editors = [
            'Adobe', 'Premiere', 'Final Cut', 'DaVinci',
            'Blender', 'After Effects', 'Fusion', 'CyberLink',
            'DeepFaceLab', 'FakeApp', 'FaceSwap'
        ]
        change = ['Edited', 'Trimmed', 'Cropped', 'Color Graded', 'Stabilized',
                  'Transitions Added', 'Effects Applied', 'Audio Enhanced',
                  'Subtitles Added', 'Watermark Applied', 'Mixed', 'DeepFake',
                  'Fake', 'Edit', 'Adjustment'
                  ]

        comment_words = [*editors, *change]

        if value := metadata.get(field):
            if any([word.lower() in value.lower() for word in comment_words]):
                report[check_name].append(str(value))

    def _check_model_device(self, report: Dict[str, List],
                            metadata: Dict[str, Any]) -> None:
        """
        Проверить метаданные на несовпадение технический возможностей модели и
        параметров видео.
        Args:
            report (dict): репорт для верификации видео.
            metadata (dict): метаданные видео.

        Returns:
            Функция добавляет результат в report.
        """
        check_name = "_check_model_device"

        models = {
            'iphone': [(3840, 2160), (1920, 1080), (1280, 720)]
        }

        make = metadata.get('Make')
        model = metadata.get('Model', '')

        if make or model:
            if ((model.lower() in models.keys() or
                 make.lower() in models.keys()) and
                    'ImageWidth' in metadata and
                    'ImageHeight' in metadata):
                width = int(metadata['ImageWidth'])
                height = int(metadata['ImageHeight'])
                if not any(
                        (width == model_width and height == model_height) for
                        (model_width, model_height) in models[model]):
                    report[check_name].append(f"{width} {height}")

    def analyze_metadata(self, metadata: Dict[str, Any]) \
            -> Dict[str, List[str]]:
        """
        Запустить все возможные проверки для видео и вернуть отчет с
        результатом верификации.
        Args:
            metadata (dict): метаданные видео.

        Returns:
            report (dict): репорт для верификации видео.
        """
        report = {}

        check_functions = [self._check_software, self._check_time,
                           self._check_audio_duration,
                           self._check_original_parameters,
                           self._check_comment,
                           self._check_model_device]

        for check_func in check_functions:
            report[check_func.__name__] = []
            check_func(report, metadata)

        return report

    def analyze_video_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        Верифицировать видео по метаданным и вернуть отчет с
        результатом верификации.
        Args:
            file_path (str): путь к видео.

        Returns:
            report (dict): репорт для верификации видео.
        """
        try:
            metadata = self.extract_metadata(file_path)
            report = self.analyze_metadata(metadata)
            return report
        except Exception as e:
            return {'error': str(e)}


class DeepFakeEyeIris:
    """
    Класс для верификации видео с помощью различий бликов в глазах.
    """

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg.eye_detection

    def analyze_eye_iris(self, frame: np.ndarray,
                         shrink: bool = True, shrink_size: int = 2):

        try:
            left_eye_image, \
            right_eye_image, \
            new_eyes_position_list, \
            number_face, \
            double_eye_img, \
            double_eye_position_difference_list = \
                eye_detection(frame, self.cfg.shape_predictor_path)

        except:
            return "eye_detection error"

        if number_face != 1:
            return "number_face error"

        try:
            left_cornea, \
            right_cornea, \
            left_cornea_matrix, \
            right_cornea_matrix = \
                cornea_convex_hull(left_eye_image, right_eye_image,
                                   new_eyes_position_list)
        except:
            return "cornea_convex_hull error"

        try:
            img_left, \
            iris_left, \
            l_iris, \
            l_iris_center, \
            l_radius, \
            l_eye_center, \
            l_highlights, \
            l_num_refl, \
            l_valid = \
                segment_iris(left_eye_image, left_cornea_matrix.astype(bool),
                             self.cfg.radius_min_para,
                             self.cfg.radius_max_para)

            img_right, \
            iris_right, \
            r_iris, \
            r_iris_center, \
            r_radius, \
            r_eye_center, \
            r_highlights, \
            r_num_refl, \
            r_valid = \
                segment_iris(right_eye_image, right_cornea_matrix.astype(bool),
                             self.cfg.radius_min_para,
                             self.cfg.radius_max_para)

            if l_num_refl == 0 and r_num_refl == 0:
                return "l_num_refl == 0 and r_num_refl == 0 error"

        except Exception as e:
            return "segment_iris error"

        try:
            double_eye_img_ori = double_eye_img.copy()

            new_left_eye = l_iris_center + double_eye_position_difference_list[
                0]
            new_right_eye = r_iris_center + \
                            double_eye_position_difference_list[1]
            cv2.circle(double_eye_img, (new_left_eye[0], new_left_eye[1]),
                       l_radius, (0, 0, 255), 2)  # left
            cv2.circle(double_eye_img, (new_right_eye[0], new_right_eye[1]),
                       r_radius, (0, 0, 255), 2)  # right

        except:
            return "circle error"

        try:
            iris_left_resize, \
            iris_right_resize, \
            left_recolor, \
            right_recolor, \
            left_recolor_resize, \
            right_recolor_resize, \
            IOU_score, \
            double_eye_img_modified = \
                process_aligned_image(iris_left, iris_right, l_iris, r_iris,
                                      l_highlights, r_highlights,
                                      left_eye_image,
                                      right_eye_image,
                                      double_eye_img,
                                      double_eye_position_difference_list,
                                      reduce=shrink,
                                      reduce_size=shrink_size,
                                      threshold_scale_left=
                                      self.cfg.threshold_scale_left,
                                      threshold_scale_right=
                                      self.cfg.threshold_scale_right)

        except:
            return "process_aligned_image error"

        print("IOU:{}".format(f'{IOU_score:.4f}'))

        return IOU_score > self.cfg.threshold_iou


class DeepFakeNN:
    def __init__(self, cfg: DictConfig):
        self.model = Meso4()
        self.model.load(cfg.nn_detection.mesonet_path)

    def transform_frames(self, frames: List[np.ndarray] | np.ndarray):
        if not isinstance(frames, list):
            frames = [frames]

        for i in range(len(frames)):
            frame = frames[i]
            frame = tf.convert_to_tensor(frame)
            resized_image = tf.image.resize(frame,
                                            [256, 256])
            rescaled_image = resized_image / 255.0
            frames[i] = rescaled_image

        return tf.convert_to_tensor(frames)

    def analyze_frame(self, frame: str | np.ndarray) -> np.ndarray:
        frames = self.transform_frames(frame)
        result = self.model.predict(frames)
        return result

    def analyze_video(self, frames: List[np.ndarray]) -> np.ndarray:
        frames = self.transform_frames(frames)
        results = self.model.predict(frames)
        return results


class DeepFake:
    """
    Класс объеденяющий все возможные верификации видео.
    """

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.df1 = DeepFakeMetadata(cfg)
        self.df2 = DeepFakeEyeIris(cfg)
        self.df3 = DeepFakeNN(cfg)

    def check_video(self, video_path: str) -> str:
        report = self.df1.analyze_video_metadata(video_path)

        if 'error' in report:
            return 'error'

        for check_func, check_list in report.items():
            if len(check_list):
                return 'fake'

        frames = extract_frames_from_video(video_path, step=self.cfg.step)
        if len(frames) == 0:
            return 'face not found'

        iou_m = [self.df2.analyze_eye_iris(frame) for frame in frames]
        iou_m = list(filter(lambda x: isinstance(x, int), iou_m))
        if len(iou_m):
            iou_m = sum(iou_m) / len(iou_m)

            if iou_m < 0.5:
                return 'fake'

        deepfake_conf = self.df3.analyze_video(frames)

        if sum(deepfake_conf) / len(deepfake_conf) < self.cfg.nn_detection.threshold_conf:
            return 'fake'

        return 'correct'
