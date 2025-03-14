# Файл для тестирования функциональности распознавания речи
import os
from pathlib import Path

# Пытаемся использовать относительные импорты, если файл импортируется как часть пакета
try:
    from speech_model import YandexSpeechKit
except ImportError:
    # Если относительные импорты не работают, используем абсолютные
    import sys
    
    # Добавляем родительскую директорию в путь импорта
    parent_dir = str(Path(__file__).parent.parent)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    
    from speech_model import YandexSpeechKit


def main():
    # Инициализация модели распознавания речи
    speech_model = YandexSpeechKit()
    
    # Получаем абсолютный путь к тестовому аудиофайлу
    script_dir = os.path.dirname(os.path.abspath(__file__))
    test_file = os.path.join(script_dir, "1.mp3")
    
    # Проверяем, существует ли файл
    if not os.path.exists(test_file):
        print(f"Ошибка: Файл {test_file} не найден")
        return
    
    # Пример обработки аудиофайла
    try:
        print(f"Обработка файла: {test_file}")
        result = speech_model.process_audio(test_file)
        print("Результат распознавания:")
        print(f"Обнаружено языков: {result['number_languages_detected']}")
        
        for lang_code, data in result.items():
            if lang_code != 'number_languages_detected':
                print(f"{data['language']} ({lang_code}) [уверенность: {data['confidence']}]:")
                print(f"{data['text'][:1000]}..." if len(data['text']) > 1000 else data['text'])
                print()
    except Exception as e:
        print(f"Ошибка: {e}")

if __name__ == "__main__":
    main()