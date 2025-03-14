# Файл для определения наличия слова в тексте, используя FuzzySearch (fuzzywuzzy) и Jaccard Similarity Score

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

from Levenshtein import distance 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import jaccard_score
from fuzzywuzzy import fuzz

class NLP_analysis:
    LEVENSHTEIN_THRESHOLD = 3
    JACCARD_THRESHOLD = 0.2
    FUZZY_THRESHOLD = 55

    def __init__(self):
        pass

    def get_jaccard_similarity(self, str1: str, str2: str) -> int:
        vectorizer = CountVectorizer(analyzer='char', ngram_range=(2, 2))
        X = vectorizer.fit_transform([str1, str2])

        return jaccard_score(X.toarray()[0], X.toarray()[1], average='macro')
    
    def get_fuzzywuzzy_score(self, str1: str, str2: str) -> int:
        fuzzy_score = fuzz.ratio(str1, str2)
        
        return fuzzy_score
    
    def get_levenshtein_distance(self, str1: str, str2: str) -> int:
        levenshtein_distance = distance(str1, str2)
        
        return levenshtein_distance

    def is_within_levenshtein_threshold(self, str1: str, str2: str) -> bool:
        levenshtein_distance = self.get_levenshtein_distance(str1, str2)
        levenshtein_match = levenshtein_distance <= self.LEVENSHTEIN_THRESHOLD

        return levenshtein_match
    
    def is_within_jaccard_score(self, str1: str, str2: str) -> bool:
        jaccard_score_value = self.get_jaccard_similarity(str1, str2)
        jaccard_match = jaccard_score_value >= self.JACCARD_THRESHOLD

        return jaccard_match
    
    def is_within_fuzzywuzzy_score(self, str1: str, str2: str) -> bool:
        fuzzy_score = self.get_fuzzywuzzy_score(str1, str2)
        fuzzy_match = fuzzy_score >= self.FUZZY_THRESHOLD

        return fuzzy_match
    
    def process_corpus(self, text: str, target_word: str) -> bool:
        words = text.split()
        
        if not words:
            raise ValueError("No text recieved.")
        
        matches = [self.is_within_fuzzywuzzy_score(target_word, word) for word in words]
        
        for i in range(len(words)):
            matches[i] = False 
            
            if self.is_within_fuzzywuzzy_score(target_word, words[i]) and self.is_within_jaccard_score(target_word, words[i]):
                matches[i] = True
            

        print("РОНАЛДУУУ")
        for i in range(len(words)):
            if matches[i]:
                print(words[i])
                print(self.get_jaccard_similarity(target_word, words[i]))

        if True in matches:
            return True
        
        return False

    

# Пример использования
if __name__ == "__main__":
    # Инициализация объекта класса
    nlp_model = NLP_analysis()
    try:
        result = nlp_model.process_corpus("Кодовое слово харизма я начала читать новую книгу это книга по переговорам сначала скажите нет джима кемпа и я прочитала 1 главу и начала читать 2 и в этих главах Специалист рассказывает про то что нам в переговорах важно противнику не показывать свою нужду и ее контролировать потому что как только мы начинаем показывать в том что мы нуждаемся в сделке то это значит что противник может начать", "Харизма")
        print(result)
    except Exception as e:
        print(f"Ошибка: {e}")