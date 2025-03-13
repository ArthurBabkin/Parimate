from datetime import datetime
from typing import Any, Dict, List, Optional

from exiftool import ExifToolHelper
from omegaconf import DictConfig


class DeepFake:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

    def extract_metadata(self, file_path: str) -> Dict[str, Any]:
        try:
            with ExifToolHelper(common_args=["-G0", "-a", "-u", "-ee"]) as et:
                metadata = et.get_metadata(file_path)[0]
                return metadata
        except Exception as e:
            raise RuntimeError(f"Ошибка извлечения метаданных: {str(e)}")

    def parse_duration(self, duration_str: str) -> Optional[float]:
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
        check_name = "_check_software"

        editors = [
            'Adobe', 'Premiere', 'Final Cut', 'DaVinci',
            'Blender', 'After Effects', 'Fusion', 'CyberLink',
            'DeepFaceLab', 'FakeApp', 'FaceSwap'
        ]

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
                                metadata[create_field].split('.')[0], fmt)
                            modify_date = datetime.strptime(
                                metadata[modify_field].split('.')[0], fmt)
                            break
                        except ValueError:
                            continue

                    if modify_date > create_date:
                        report[check_name].append(
                            f"{modify_date} {create_date}")
                except Exception:
                    pass

    def _check_audio_duration(self, report: Dict[str, List],
                              metadata: Dict[str, Any]) -> None:
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
                    delta_dur = abs(
                        durations[main_field] - durations[compare_field])
                    if delta_dur > threshold:
                        report[check_name].append(
                            f"{main_field}={durations[main_field]} vs "
                            f"{compare_field}={durations[compare_field]})"
                        )

    def _check_original_parameters(self, report: Dict[str, List],
                                   metadata: Dict[str, Any]) -> None:
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
        check_name = "_check_comment"

        field = 'Comment'

        if value := metadata.get(field):
            if 'adjustment' in value.lower() or 'edit' in value.lower():
                report[check_name].append(str(value))

    def _check_model_device(self, report: Dict[str, List],
                            metadata: Dict[str, Any]) -> None:
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
        try:
            metadata = self.extract_metadata(file_path)
            return self.analyze_metadata(metadata)
        except Exception as e:
            return {'error': str(e)}

    def check_video(self, file_path: str) -> bool:
        report = self.analyze_video_metadata(file_path)

        if 'error' in report:
            return True

        for check_func, check_list in report.items():
            if len(check_list):
                return False
