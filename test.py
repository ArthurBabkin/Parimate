#!/usr/bin/env python3
"""
Тестовый скрипт для демонстрации работы пакета speech_model
"""

from speech_model import YandexSpeechKit

def main():
    # Инициализация модели распознавания речи
    speech_model = YandexSpeechKit()
    
    # Пример обработки аудиофайла
    try:
        result = speech_model.process_audio("speech_model/test_audio.ogg")
        print("Результат распознавания:")
        print(f"Обнаружено языков: {result['number_languages_detected']}")
        
        for lang_code, data in result.items():
            if lang_code != 'number_languages_detected':
                print(f"{data['language']} ({lang_code}) [уверенность: {data['confidence']}]:")
                print(f"{data['text'][:100]}..." if len(data['text']) > 100 else data['text'])
                print()
    except Exception as e:
        print(f"Ошибка: {e}")

if __name__ == "__main__":
    main() 