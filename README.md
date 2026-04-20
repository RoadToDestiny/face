# Комплекс распознавания эмоций в реальном времени

Программный комплекс на основе ИИ для распознавания эмоций человека в видеопотоке. Архитектура: **MTCNN** (детекция лиц, PyTorch) → **ResNet18** (классификация эмоций).

## Архитектура

```
Camera 1 ─┐
Camera 2 ─┤
Camera 3 ─┼─→ Video Stream Manager
          ↓
     Face Detection (MTCNN)
          ↓
     Face Crop + Align + Preprocess
          ↓
     Emotion CNN (ResNet18)
          ↓
     Emotion Prediction → Dashboard / Logs
```

## Установка

```bash
cd face
pip install -r requirements.txt
```

**Системные требования:**
- Python 3.9+
- CUDA (опционально, для GPU)
- FER2013 скачивается автоматически при обучении

## Обучение модели эмоций

**Скачайте FER2013 перед обучением:**
1. Зарегистрируйтесь на [Kaggle](https://www.kaggle.com) и примите правила конкурса [FER2013](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge)
2. Настройте [Kaggle API](https://github.com/Kaggle/kaggle-api#api-credentials)
3. Выполните: `pip install opendatasets && python scripts/download_fer2013.py --data-dir ./data`
4. Или вручную скачайте `fer2013.csv` и поместите в `./data/fer2013/`

Затем запустите обучение:
```bash
python scripts/train_emotion.py --data-dir ./data --output ./checkpoints --epochs 80
```

Для повышения точности:
- Увеличить `--epochs` до 100–120
- Использовать `--backbone resnet50`
- Добавить RAF-DB для дообучения (требует ручной загрузки датасета)

**Ожидаемая точность:** FER2013 ~68–72% (сложный датасет). Для 95%+ рекомендуется дообучение на RAF-DB или AffectNet.

## Запуск

### Графический интерфейс (выбор источника)
```bash
python scripts/run_ui.py
```

В интерфейсе можно выбрать:
- изображение
- видеофайл
- веб-камеру

Предпросмотр отображается прямо в окне интерфейса.
Для видео/веб-камеры используйте кнопку `Start` для запуска и `Stop` для остановки.

### Распознавание речи с микрофона (VOSK)

В UI добавлена кнопка `Voice: On/Off`.

1. Установите зависимости:
```bash
pip install -r requirements.txt
```
2. Скачайте модель VOSK (например `vosk-model-small-ru-0.22` или `vosk-model-small-en-us-0.15`).
3. Распакуйте модель в корень проекта или в папку `models/`.
4. В UI откройте `Settings` и укажите путь в поле `VOSK model directory` (если не определился автоматически).
5. Нажмите `Voice: On`.

Распознавание голоса работает в отдельном потоке и не блокирует распознавание эмоций по видео. Текст отображается как субтитры поверх видео и в правой панели.

### Видеофайл или изображение
```bash
python scripts/run_inference.py video.mp4 --model checkpoints/best_emotion_model.pt
python scripts/run_inference.py photo.jpg --model checkpoints/best_emotion_model.pt
```

### Веб-камера (реальное время)
```bash
python scripts/run_inference.py 0 --model checkpoints/best_emotion_model.pt
```

### Несколько камер
```bash
python scripts/run_stream.py 0 1 2 --model checkpoints/best_emotion_model.pt
```

### Оценка точности на тестовом наборе
```bash
python scripts/eval_accuracy.py --checkpoint checkpoints/best_emotion_model.pt --data-dir ./data
```

## Технические требования к медиаконтенту

| Параметр | Минимум |
|----------|---------|
| Разрешение | 640×480 |
| Частота кадров | 15 fps |
| Освещение | Достаточное, без сильных теней |
| Лицо в кадре | ≥48×48 пикселей |

## Структура проекта

```
face/
├── config/default.yaml    # Конфигурация
├── src/
│   ├── face_detection/    # RetinaFace
│   ├── emotion/           # ResNet, препроцессинг, обучение
│   ├── pipeline/          # Полный пайплайн
│   └── streams/           # Управление видеопотоками
├── scripts/
│   ├── train_emotion.py   # Обучение на FER2013
│   ├── train_custom.py    # Дообучение на своём датасете
│   ├── run_inference.py   # Инференс (видео/фото/камера)
│   ├── run_stream.py      # Поток с нескольких камер
│   └── eval_accuracy.py   # Оценка точности
└── checkpoints/           # Сохранённые модели
```

## Дообучение на российском контенте

Для работы с российским медиаконтентом (ТВ, стримы, видеоконференции) модель нужно дообучить на локальных данных.

### 1. Подготовка датасета

Соберите изображения лиц с размеченными эмоциями. Структура папок (ImageFolder):

```
data/russian/
├── train/
│   ├── anger/      # злость
│   ├── disgust/    # отвращение
│   ├── fear/       # страх
│   ├── happy/      # радость
│   ├── sad/        # грусть
│   ├── surprise/   # удивление
│   └── neutral/    # нейтральное
└── test/
    └── (те же 7 папок)
```

**Требования к изображениям:**
- Формат: JPG или PNG
- Размер лица: не менее 48×48 пикселей (рекомендуется 112×112)
- Желательно: вырезанные кадры лиц из видео (используйте `run_inference.py` для детекции, затем сохранение кропов)
- Рекомендуемый объём: от 500 изображений на класс в train, от 50 на класс в test

**Источники данных:**
- Кадры из российских фильмов, сериалов, новостей
- Записи видеозвонков (с согласия участников)
- Стримы, подкасты
- Собственные записи с веб-камеры с разными эмоциями

**Разметка:** изображения раскладываются по папкам с названием эмоции. Названия папок должны точно совпадать: `anger`, `disgust`, `fear`, `happy`, `sad`, `surprise`, `neutral`.

### 2. Первичное обучение (если ещё не обучена)

Сначала обучите базовую модель на FER2013:

```bash
python scripts/train_emotion.py --data-dir ./data --output ./checkpoints --epochs 80 --backbone resnet50
```

### 3. Дообучение на российском датасете

```bash
python scripts/train_custom.py \
  --data-dir ./data/russian \
  --checkpoint ./checkpoints/best_emotion_model.pt \
  --output ./checkpoints \
  --epochs 30 \
  --lr 0.0001
```

Параметры:
- `--checkpoint` — путь к обученной модели (FER2013)
- `--lr 0.0001` — меньший learning rate при дообучении
- `--epochs 30` — обычно достаточно 20–40 эпох для дообучения

### 4. Оценка результата

```bash
python scripts/eval_accuracy.py --checkpoint ./checkpoints/best_emotion_model.pt --data-dir ./data/russian
```

---

## Улучшение качества распознавания

1. **RAF-DB / AffectNet** — более качественные датасеты для дообучения.
3. **ResNet50** — `--backbone resnet50` при обучении.
4. **Ансамбль** — объединение нескольких моделей.
5. **Аудио** — добавление анализа тона голоса (будущее расширение).

## Лицензия

Проект создан в учебных целях.
