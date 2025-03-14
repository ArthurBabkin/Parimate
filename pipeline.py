# Файл для обработки всего пайплайна Speech Recognition части (Транскрибация через SR и проверка наличия слова через NLP)
import os
import sys
from pathlib import Path

parent_dir = str(Path(__file__).parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from speech_model import YandexSpeechKit
from nlp_model.nlp_processor import NLP_analysis

def validate_pronunciation(audio_file_path: str, target_word: str) -> bool:
    """
    Проверка произношения целевого слова в аудиофайле
    
    Args:
        audio_file_path (str): Путь к аудиофайлу
        target_word (str): Целевое слово для проверки
        
    Returns:
        bool: True, если целевое слово найдено в транскрипции, False в противном случае
    """
    try:
        print(f"Processing file: {audio_file_path}")
        print(f"Target word: {target_word}")
        
        # Инициализация объектов моделей
        speech_model = YandexSpeechKit()
        nlp_model = NLP_analysis()
        
        # Транскрибируем аудио
        transcription_result = speech_model.process_audio(audio_file_path)
        
        if not transcription_result:
            print("Error: Empty transcription result")
            return False

        transcriptions = []

        
        # Check for language-specific results (ru-RU, en-US)
        for lang in ['ru-RU', 'en-US']:
            if lang in transcription_result:
                lang_result = transcription_result[lang]
                
                if isinstance(lang_result, dict) and 'text' in lang_result:
                    transcriptions.append({
                        'text': lang_result['text'],
                        'language': lang,
                        'confidence': lang_result.get('confidence', 0)
                    })
                elif isinstance(lang_result, str) and lang_result:
                    transcriptions.append({
                        'text': lang_result,
                        'language': lang,
                        'confidence': 1.0
                    })

        # If no language-specific results, check for 'text' key
        if not transcriptions and 'text' in transcription_result:
            transcriptions.append({
                'text': transcription_result['text'],
                'language': transcription_result.get('language', 'unknown'),
                'confidence': transcription_result.get('confidence', 0)
            })
        
        # Сортируем по confidence 
        transcriptions.sort(key=lambda x: x['confidence'], reverse=True)
        
        if not transcriptions:
            print("Error: No transcribed text found")
            return False
    
        # Check if target word is in any transcription
        for t in transcriptions:
            if nlp_model.process_corpus(t['text'], target_word):
                print(f"Target word '{target_word}' found in transcription")
                return True
        
        print(f"Target word '{target_word}' not found in any transcription")
        return False
        
    except Exception as e:
        print(f"Error: {e}")
        return False

# Пример использования
if __name__ == "__main__":
    audio_path = "..."  # Замените на реальный путь к аудиофайлу
    word = "..."  # Замените на целевое слово
    
    result = validate_pronunciation(audio_path, word)
    print(f"Pronunciation validation result: {result}") 