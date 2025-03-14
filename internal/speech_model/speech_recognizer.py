import requests


class SpeechRecognizer:
    """
    Класс для распознавания речи с использованием Yandex SpeechKit API.
    """
    
    # Поддерживаемые языки Yandex SpeechKit
    SUPPORTED_LANGUAGES = {
        'ru-RU': 'Русский',
        'en-US': 'Английский'
    }
    
    def __init__(self, api_key, folder_id):
        """
        Инициализация распознавателя речи.
        
        Args:
            api_key (str): API-ключ Yandex SpeechKit.
            folder_id (str): ID папки в Yandex Cloud.
        """
        self.api_key = api_key
        self.folder_id = folder_id
    
    def recognize_with_language_detection(self, audio_file_path, format='lpcm', sample_rate_hertz=16000):
        """
        Автоматическое определение языка и распознавание речи.
        
        Args:
            audio_file_path (str): Путь к аудиофайлу.
            format (str, optional): Формат аудио. По умолчанию 'lpcm'.
            sample_rate_hertz (int, optional): Частота дискретизации в Гц. По умолчанию 16000.
            
        Returns:
            dict: Словарь, содержащий определенный язык и распознанный текст.
        """
        results = {}
        confidence_scores = {}
        
        # Пробуем распознать речь на каждом поддерживаемом языке
        for lang_code in self.SUPPORTED_LANGUAGES.keys():
            try:
                response = self.recognize_speech_http(
                    audio_file_path=audio_file_path,
                    format=format,
                    sample_rate_hertz=sample_rate_hertz,
                    language_code=lang_code
                )
                text = response.get('result', '')
                
                # Сохраняем результат
                results[lang_code] = text
                
                # Рассчитываем простую оценку уверенности на основе длины распознанного текста
                # Это очень простая эвристика - более длинный распознанный текст может указывать на лучшее распознавание
                confidence_scores[lang_code] = len(text.split())
                
            except Exception as e:
                print(f"Ошибка при распознавании речи на {lang_code}: {e}")
                results[lang_code] = ""
                confidence_scores[lang_code] = 0
        
        # Находим язык с наивысшей оценкой уверенности
        if confidence_scores:
            best_lang = max(confidence_scores.items(), key=lambda x: x[1])[0]
            
            return {
                'detected_language': best_lang,
                'language_name': self.SUPPORTED_LANGUAGES.get(best_lang, 'Неизвестный'),
                'text': results[best_lang],
                'all_results': results,
                'confidence_scores': confidence_scores
            }
        else:
            return {
                'detected_language': None,
                'language_name': 'Неизвестный',
                'text': '',
                'all_results': results,
                'confidence_scores': confidence_scores
            }
    
    def recognize_speech_http(self, audio_file_path, format='lpcm', sample_rate_hertz=16000, language_code='ru-RU'):
        """
        Распознавание речи с использованием прямого HTTP-запроса к API.
        
        Args:
            audio_file_path (str): Путь к аудиофайлу для распознавания.
            format (str, optional): Формат аудио. По умолчанию 'lpcm'.
            sample_rate_hertz (int, optional): Частота дискретизации в Гц. По умолчанию 16000.
            language_code (str, optional): Код языка. По умолчанию 'ru-RU'.
            
        Returns:
            dict: Ответ API с результатами распознавания.
        """
        url = "https://stt.api.cloud.yandex.net/speech/v1/stt:recognize"
        
        headers = {
            "Authorization": f"Api-Key {self.api_key}"
        }
        
        params = {
            "lang": language_code,
            "format": format,
            "sampleRateHertz": sample_rate_hertz,
            "folderId": self.folder_id
        }
        
        with open(audio_file_path, 'rb') as f:
            audio_data = f.read()
        
        response = requests.post(url, headers=headers, params=params, data=audio_data)
        
        if response.status_code != 200:
            raise Exception(f"Ошибка распознавания речи: {response.text}")
        
        return response.json()
    
    def prepare_recognition_result(self, result):
        """
        Подготовка результата распознавания с учетом определения языка.
        
        Args:
            result (dict): Результат распознавания речи.
            
        Returns:
            dict: Структурированный результат распознавания.
        """
        # Проверяем, нужно ли транскрибировать на обоих языках
        confidence_scores = result['confidence_scores']
        ru_score = confidence_scores.get('ru-RU', 0)
        en_score = confidence_scores.get('en-US', 0)
        
        # Если оценки близки (в пределах 45% друг от друга), транскрибируем на обоих языках
        transcribe_both = False
        if ru_score > 0 and en_score > 0:
            max_score = max(ru_score, en_score)
            min_score = min(ru_score, en_score)
            if min_score / max_score > 0.55:  # В пределах 45% друг от друга
                transcribe_both = True
        
        output = {
            'number_languages_detected': 2 if transcribe_both else 1
        }
        
        main_lang = result['detected_language']
        output[main_lang] = {
            'language': self.SUPPORTED_LANGUAGES.get(main_lang, 'Неизвестный'),
            'confidence': confidence_scores.get(main_lang, 0),
            'text': result['text']
        }
        
        # Добавляем транскрипцию на другом языке, если нужно
        if transcribe_both:
            other_lang = 'en-US' if main_lang == 'ru-RU' else 'ru-RU'
            output[other_lang] = {
                'language': self.SUPPORTED_LANGUAGES.get(other_lang, 'Неизвестный'),
                'confidence': confidence_scores.get(other_lang, 0),
                'text': result['all_results'][other_lang]
            }
        
        return output 