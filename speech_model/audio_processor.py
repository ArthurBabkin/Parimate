import os
import tempfile
import subprocess

class AudioProcessor:
    """
    Класс для обработки аудиофайлов: проверка формата, длительности, сжатие.
    """
    
    # Поддерживаемые форматы аудио
    SUPPORTED_FORMATS = {'lpcm', 'mp3', 'oggopus'}
    
    # Сопоставление расширений файлов с форматами Yandex SpeechKit
    FORMAT_MAPPING = {
        'wav': 'lpcm',
        'mp3': 'mp3',
        'ogg': 'oggopus',
        'opus': 'oggopus',
        'lpcm': 'lpcm'
    }
    
    # Сопоставление форматов Yandex с форматами FFmpeg
    FFMPEG_FORMAT_MAPPING = {
        'lpcm': 'wav',
        'mp3': 'mp3',
        'oggopus': 'opus'
    }
    
    @classmethod
    def validate_audio_format(cls, audio_file_path):
        """
        Проверка формата аудиофайла на поддержку.
        
        Args:
            audio_file_path (str): Путь к аудиофайлу.
            
        Returns:
            str: Название формата для Yandex SpeechKit.
            
        Raises:
            ValueError: Если формат не поддерживается.
        """
        file_extension = os.path.splitext(audio_file_path)[1].lower().lstrip('.')
        
        if file_extension not in cls.FORMAT_MAPPING:
            raise ValueError(f"Неподдерживаемый формат аудио: {file_extension}. Поддерживаемые форматы: WAV (LPCM), MP3, OGG/OPUS.")
        
        return cls.FORMAT_MAPPING[file_extension]
    
    @classmethod
    def get_audio_duration(cls, audio_path):
        """
        Получение длительности аудиофайла в секундах с помощью FFmpeg.
        
        Args:
            audio_path (str): Путь к аудиофайлу.
            
        Returns:
            float: Длительность аудиофайла в секундах.
        """
        cmd = [
            'ffprobe',
            '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            audio_path
        ]
        
        try:
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
            duration = float(result.stdout.strip())
            return duration
        except (subprocess.SubprocessError, ValueError) as e:
            print(f"Ошибка при определении длительности для {audio_path}: {e}")
            return 0.0
    
    @classmethod
    def process_audio_duration(cls, input_path, output_path, format_name, duration):
        """
        Обработка аудиофайла с учетом ограничений по длительности (29 секунд).
        
        Args:
            input_path (str): Путь к входному аудиофайлу.
            output_path (str): Путь для сохранения обработанного аудиофайла.
            format_name (str): Формат аудио.
            duration (float): Длительность аудиофайла в секундах.
            
        Returns:
            str: Путь к обработанному аудиофайлу.
        """
        max_duration = 29
        if duration > max_duration:
            print(f"Аудио длительностью {duration:.2f} секунд, обрезаем до {max_duration} секунд")
            cls.convert_audio(
                input_path, 
                output_path,
                format_name=format_name,
                max_duration=max_duration
            )
        else:
            # Просто копируем файл, возможно потребуется сжатие на следующем шаге
            cls.convert_audio(
                input_path, 
                output_path,
                format_name=format_name
            )
        
        return output_path
    
    @classmethod
    def optimize_audio_size(cls, audio_path, format_name):
        """
        Оптимизация размера аудиофайла при необходимости (<=1Мб).
        
        Args:
            audio_path (str): Путь к аудиофайлу.
            format_name (str): Формат аудио.
            
        Returns:
            str: Путь к оптимизированному аудиофайлу.
        """
        file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
        max_size_mb = 1.0
        
        if file_size_mb > max_size_mb:
            print(f"Размер аудиофайла {file_size_mb:.2f} MB, сжимаем для уменьшения размера")
            temp_dir = tempfile.gettempdir()
            compressed_audio_path = os.path.join(temp_dir, f"compressed_{os.path.basename(audio_path)}")
            
            # Сжимаем, уменьшая битрейт и частоту дискретизации
            cls.compress_audio(
                audio_path,
                compressed_audio_path,
                format_name=format_name
            )
            
            # Заменяем обработанный файл сжатым
            os.remove(audio_path)
            return compressed_audio_path
        
        return audio_path
    
    @classmethod
    def convert_audio(cls, input_path, output_path, format_name='lpcm', sample_rate=16000, max_duration=None):
        """
        Конвертация аудиофайла в другой формат с помощью FFmpeg.
        
        Args:
            input_path (str): Путь к входному аудиофайлу.
            output_path (str): Путь для сохранения выходного аудиофайла.
            format_name (str, optional): Целевой формат аудио. По умолчанию 'lpcm'.
            sample_rate (int, optional): Частота дискретизации в Гц. По умолчанию 16000.
            max_duration (int, optional): Максимальная длительность в секундах. Если None, обрезка не выполняется.
            
        Returns:
            bool: True, если конвертация успешна, False в противном случае.
        """
        ffmpeg_format = cls.FFMPEG_FORMAT_MAPPING.get(format_name, 'wav')
        
        # Формируем команду FFmpeg
        cmd = [
            'ffmpeg',
            '-i', input_path,
            '-ar', str(sample_rate),  # Частота дискретизации
            '-ac', '1',  # Моно
        ]
        
        # Добавляем ограничение длительности, если указано
        if max_duration is not None:
            cmd.extend(['-t', str(max_duration)])
        
        # Добавляем формат вывода и путь
        cmd.extend([
            '-f', ffmpeg_format,
            output_path
        ])
        
        try:
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            return True
        except subprocess.SubprocessError as e:
            print(f"Ошибка при конвертации {input_path}: {e}")
            return False
    
    @classmethod
    def compress_audio(cls, input_path, output_path, format_name='lpcm'):
        """
        Сжатие аудиофайла для уменьшения размера с помощью FFmpeg (надо <= 1Мб).
        
        Args:
            input_path (str): Путь к входному аудиофайлу.
            output_path (str): Путь для сохранения сжатого аудиофайла.
            format_name (str, optional): Формат аудио. По умолчанию 'lpcm'.
            
        Returns:
            bool: True, если сжатие успешно, False в противном случае.
        """
        ffmpeg_format = cls.FFMPEG_FORMAT_MAPPING.get(format_name, 'wav')
        
        # Настройки сжатия в зависимости от формата
        if format_name == 'mp3':
            # Для MP3 используем пониженный битрейт
            cmd = [
                'ffmpeg',
                '-i', input_path,
                '-ar', '16000',  # Пониженная частота дискретизации
                '-ac', '1',      # Моно
                '-b:a', '32k',   # Пониженный битрейт
                '-f', ffmpeg_format,
                output_path
            ]
        elif format_name == 'oggopus':
            # Для Opus используем пониженный битрейт
            cmd = [
                'ffmpeg',
                '-i', input_path,
                '-ar', '16000',  # Пониженная частота дискретизации
                '-ac', '1',      # Моно
                '-b:a', '24k',   # Пониженный битрейт
                '-f', ffmpeg_format,
                output_path
            ]
        else:  # lpcm/wav
            # Для WAV используем пониженную частоту дискретизации и битовую глубину
            cmd = [
                'ffmpeg',
                '-i', input_path,
                '-ar', '16000',  # Пониженная частота дискретизации
                '-ac', '1',      # Моно
                '-acodec', 'pcm_s16le',  # 16-битный PCM
                '-f', ffmpeg_format,
                output_path
            ]
        
        try:
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            return True
        except subprocess.SubprocessError as e:
            print(f"Ошибка при сжатии {input_path}: {e}")
            return False 