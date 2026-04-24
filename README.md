# Emotion AI Live Analyzer

Готовое desktop-приложение для мультимодального анализа эмоций по камере и видеофайлам.

Система объединяет:

- эмоцию лица по видеокадрам
- распознавание речи из аудио (VOSK)
- эмоцию текста по распознанной речи
- итоговую агрегированную эмоцию

## Что входит в поставку

- готовый чекпоинт: `checkpoints_production/best_emotion_model.pt`
- готовая директория модели VOSK: `vosk-model-small-ru-0.22/`
- графический интерфейс: `scripts/run_ui.py`
- CLI-инференс: `scripts/run_inference.py`

## Требования

- Windows 10/11 или Linux
- Python 3.9+
- `ffmpeg` в PATH (обязателен для извлечения аудио из видео)

## Установка

```bash
pip install -r requirements.txt
```

## Быстрый старт (UI)

```bash
python scripts/run_ui.py
```

В интерфейсе доступны:

- режим камеры для анализа в реальном времени
- загрузка видео с офлайн-предобработкой
- субтитры и плашка эмоций во время воспроизведения видео

## Обработка видео

После загрузки видео приложение выполняет предобработку перед воспроизведением:

1. Извлекает аудиодорожку через `ffmpeg`
2. Строит субтитры с таймкодами через VOSK
3. Определяет эмоцию текста для реплик
4. Предвычисляет эмоции лица по кадрам
5. Запускает воспроизведение с синхронизированными оверлеями

Если модель анализа текста недоступна, воспроизведение все равно работает, а эмоция текста устанавливается в `NEUTRAL`.

## CLI-инференс

Видео или изображение:

```bash
python scripts/run_inference.py <input_path> --model checkpoints_production/best_emotion_model.pt
```

Примеры:

```bash
python scripts/run_inference.py video.mp4 --model checkpoints_production/best_emotion_model.pt
python scripts/run_inference.py photo.jpg --model checkpoints_production/best_emotion_model.pt
python scripts/run_inference.py 0 --model checkpoints_production/best_emotion_model.pt
```

Многопоточный запуск с нескольких камер:

```bash
python scripts/run_stream.py 0 1 --model checkpoints_production/best_emotion_model.pt
```

## Структура проекта

```text
face/
├── checkpoints_production/
│   └── best_emotion_model.pt
├── config/
│   └── default.yaml
├── scripts/
│   ├── run_ui.py
│   ├── run_inference.py
│   ├── run_stream.py
│   └── eval_accuracy.py
├── src/
│   ├── face_detection/
│   ├── emotion/
│   ├── audio/
│   ├── pipeline/
│   └── streams/
└── vosk-model-small-ru-0.22/
```

## Устранение проблем

- Ошибка предобработки видео:
  - проверьте, что `ffmpeg` установлен и доступен в терминале
  - проверьте, что в видео есть аудиодорожка
- Модель VOSK не определяется:
  - держите папку `vosk-model-small-ru-0.22` в корне проекта
- Камера недоступна:
  - закройте другие приложения, которые используют камеру

## Лицензия

Для внутреннего и учебного использования, если иная политика лицензирования не задана в вашей организации.
