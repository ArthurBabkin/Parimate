import os
import tempfile
from dotenv import load_dotenv

# Пытаемся использовать относительные импорты, если файл импортируется как часть пакета
try:
    from .audio_processor import AudioProcessor
    from .speech_recognizer import SpeechRecognizer
except ImportError:
    # Если относительные импорты не работают, используем абсолютные
    # Это позволяет запускать файл напрямую
    import sys
    from pathlib import Path
    
    # Добавляем родительскую директорию в путь импорта
    parent_dir = str(Path(__file__).parent.parent)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    
    from speech_model.audio_processor import AudioProcessor
    from speech_model.speech_recognizer import SpeechRecognizer

# Load environment variables from .env file
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path)

class YandexSpeechKit:
    """
    A class to interact with Yandex SpeechKit API for speech-to-text recognition.
    https://yandex.cloud/ru/docs/speechkit/quickstart/stt-quickstart-v1
    """
    
    # Поддерживаемые языки Yandex SpeechKit
    SUPPORTED_LANGUAGES = {
        'ru-RU': 'Русский',
        'en-US': 'Английский'
    }
    
    def __init__(self, api_key=None, folder_id=None):
        """
        Initialize the YandexSpeechKit class with API credentials.
        
        Args:
            api_key (str, optional): Yandex SpeechKit API key. If not provided, will look for YC_API_KEY env variable.
            folder_id (str, optional): Yandex Cloud folder ID. If not provided, will look for YC_FOLDER_ID env variable.
        """
        self.api_key = api_key or os.environ.get('YC_API_KEY')
        self.folder_id = folder_id or os.environ.get('YC_FOLDER_ID')
        
        if not self.api_key:
            raise ValueError("API key is required. Provide it as a parameter or set YC_API_KEY environment variable.")
        
        if not self.folder_id:
            raise ValueError("Folder ID is required. Provide it as a parameter or set YC_FOLDER_ID environment variable.")
        
        # Инициализация распознавателя речи
        self.recognizer = SpeechRecognizer(self.api_key, self.folder_id)
    
    def process_audio(self, audio_file_path):
        """
        Комплексная обработка аудиофайла для распознавания речи:
        1. Проверка формата (поддерживаются LPCM, OggOpus, MP3)
        2. Проверка длительности (≤ 29 секунд, обрезка при необходимости)
        3. Проверка размера (сжатие, если > 1MB)
        4. Определение языка (русский или английский)
        5. Транскрибация (в обоих языках, если уверенность близка)
        
        Args:
            audio_file_path (str): Путь к аудиофайлу для обработки.
            
        Returns:
            dict: Словарь с результатами распознавания.
        """
        # Сохраняем оригинальный путь к файлу
        original_audio_path = audio_file_path
        
        # Шаг 1: Проверка формата аудио и конвертация при необходимости
        format_name, audio_file_path = AudioProcessor.validate_audio_format(audio_file_path)
        
        # Шаг 2: Проверка длительности аудио
        duration = AudioProcessor.get_audio_duration(audio_file_path)
        if duration <= 0:
            raise ValueError(f"Не удалось определить длительность для {audio_file_path}")
        
        # Создаем временный файл для обработанного аудио
        temp_dir = tempfile.gettempdir()
        processed_audio_path = os.path.join(temp_dir, f"processed_{os.path.basename(audio_file_path)}")
        
        # Обрабатываем аудио (обрезка при необходимости)
        processed_audio_path = AudioProcessor.process_audio_duration(
            audio_file_path, 
            processed_audio_path, 
            format_name, 
            duration
        )
        
        # Шаг 3: Проверка размера файла и сжатие при необходимости
        processed_audio_path = AudioProcessor.optimize_audio_size(processed_audio_path, format_name)
        
        try:
            # Шаг 4: Определение языка и распознавание речи
            result = self.recognizer.recognize_with_language_detection(
                audio_file_path=processed_audio_path,
                format=format_name,
                sample_rate_hertz=16000
            )
            
            # Шаг 5: Формирование результата с учетом определения языка
            return self.recognizer.prepare_recognition_result(result)
            
        finally:
            # Очищаем временные файлы
            if os.path.exists(processed_audio_path):
                os.remove(processed_audio_path)
            
            # Если был создан временный файл при конвертации, тоже удаляем его
            if audio_file_path != processed_audio_path and audio_file_path != original_audio_path:
                if os.path.exists(audio_file_path):
                    os.remove(audio_file_path)


# Пример использования
if __name__ == "__main__":
    # Инициализация объекта класса
    speech_model = YandexSpeechKit()
    
    try:
        # Используем абсолютный путь к файлу в директории speech_model
        script_dir = os.path.dirname(os.path.abspath(__file__))
        audio_file_path = os.path.join(script_dir, "1.mp3")
        
        # Проверяем существование файла перед обработкой
        if not os.path.exists(audio_file_path):
            print(f"Ошибка: Файл {audio_file_path} не найден")
        else:
            print(f"Обрабатываем файл: {audio_file_path}")
            
            result = speech_model.process_audio(audio_file_path)
            print("Результат распознавания:", result)
    except Exception as e:
        print(f"Ошибка: {e}")
