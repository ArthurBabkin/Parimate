
# 📌 Parimate — Telegram-based Media Verification Bot

**Parimate** — это ML-система, позволяющая пользователям проверять подлинность аудио- и видеоконтента через Telegram. Система анализирует лицо, речь и метаданные, предоставляя понятный и объяснимый результат в чате.

---

## 🚀 Возможности

- 🔍 Распознавание лиц и речи
- 🧠 Сравнение с эталонной фразой и изображением
- 📁 Проверка метаданных (время, устройство, геолокация)
- 📩 Ответ в Telegram с интерпретацией результата
- ⚙️ Бэкенд на FastAPI, ML-инференс, PostgreSQL, Docker

---

## 🛠️ Стек технологий

- **Язык:** Python
- **Модели:** Whisper, FaceNet, ffmpeg/ffprobe
- **API:** FastAPI + Telegram Bot API
- **Хранилище:** MinIO (S3 совместимое)
- **Инфраструктура:** Docker + docker-compose

---

## 📦 Быстрый старт

```bash
git clone https://github.com/<your-org>/Parimate.git
cd Parimate
cp .env.example .env
docker-compose up --build
```

---

## 📷 Пример ответа

```json
{
  "face_match": true,
  "speech_match": false,
  "metadata_valid": true,
  "explanation": "Лицо совпадает, но фраза не соответствует ожидаемой. Видео снято недавно на смартфон."
}
```

---

## 🧠 Принципы дизайна

Проект следует подходам Human-Centered AI:

- **Explainability** — система объясняет решения модели
- **User-Centered** — Telegram-интерфейс без барьеров входа
- **Fairness** — учёт инклюзивности и прозрачности
- **Mixed-Initiative** — бот может уточнять ввод

---

## 👥 Команда

- Дарья — ML (распознавание лиц)
- Артур — backend, ASR
- Илья — дипфейк, метаданные
- Никита — scene analysis, UX
- Ильяс - документация, описание архитектуры
- Все — DevOps

---

## 📄 Лицензия

MIT © 2025 Parimate Team
