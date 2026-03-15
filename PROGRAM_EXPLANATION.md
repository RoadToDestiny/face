# Полное объяснение работы программы

Этот документ описывает, как устроен проект распознавания эмоций по лицу: от загрузки данных и обучения модели до инференса на фото, видео, веб-камере и нескольких потоках.

## 1. Что делает программа

Программа определяет эмоцию человека на кадре:

1. Находит лица в кадре.
2. Для каждого лица делает выравнивание и предобработку.
3. Передает лицо в нейросеть классификации эмоций.
4. Возвращает класс эмоции и вероятность.
5. Рисует рамку и подпись на изображении.

Поддерживаемые эмоции (7 классов):
- anger
- disgust
- fear
- happy
- sad
- surprise
- neutral

## 2. Общая архитектура

Основные подсистемы:

- Детекция лиц: `src/face_detection/retinaface_detector.py`
- Предобработка лица: `src/emotion/preprocessing.py`
- Модель эмоций (ResNet): `src/emotion/model.py`
- End-to-end пайплайн: `src/pipeline/emotion_pipeline.py`
- Управление потоками видео: `src/streams/video_manager.py`
- Скрипты запуска/обучения/оценки: папка `scripts/`

Важно: в файле детектора класс называется `RetinaFaceDetector`, но фактически внутри используется MTCNN из `facenet-pytorch` (PyTorch-native детектор, без TensorFlow).

Для чего здесь используется MTCNN:
- локализует лицо (bbox), чтобы модель эмоций анализировала лицо, а не весь кадр;
- находит landmarks (глаза/нос), которые нужны для геометрического выравнивания лица;
- отсекает слабые детекции по confidence threshold, снижая число ложных срабатываний;
- стабилизирует качество в реальном времени при разных ракурсах, освещении и масштабе лица.

## 3. Поток данных в режиме инференса

### 3.1 Точка входа

Основной скрипт для инференса: `scripts/run_inference.py`.

Он принимает:
- путь к видео
- путь к изображению
- индекс камеры (например, `0`)

Скрипт создает объект `EmotionPipeline`, читает кадры и для каждого кадра делает:

1. `pipeline.process_frame(frame)`
2. `pipeline.draw_results(frame, results)`

### 3.2 Детекция и выравнивание лица

Файл: `src/face_detection/retinaface_detector.py`

Логика:

1. Ленивая инициализация MTCNN (`_get_mtcnn()` и `_ensure_mtcnn()`).
2. Конвертация входа в RGB.
3. Запуск `self._mtcnn.detect(..., landmarks=True)`.
4. Фильтрация по порогу confidence.
5. Формирование результата:
   - `bbox` (координаты лица)
   - `confidence`
   - `landmarks` (глаза/нос)
6. Если включен `align`, выполняется `_align_face(...)`:
   - угол вычисляется по линии глаз
   - лицо поворачивается так, чтобы глаза были ближе к горизонтали

Зачем это нужно:
- уменьшает влияние наклона головы;
- повышает стабильность распознавания эмоций.

Почему именно MTCNN важен для качества:
- без MTCNN модель часто получает «грязный» вход (фон, часть головы, случайные области), и точность падает;
- landmarks от MTCNN позволяют нормализовать положение глаз, что уменьшает разброс признаков между кадрами;
- детектор работает как фильтр качества входа: если лицо не найдено уверенно, пайплайн не делает сомнительный прогноз.

### 3.3 Предобработка лица

Файл: `src/emotion/preprocessing.py`, класс `EmotionPreprocessor`.

Шаги предобработки:

1. Приведение к grayscale (по умолчанию).
2. Resize к фиксированному размеру (`input_size`, обычно 112x112).
3. Histogram equalization (для усиления контраста в серых изображениях).
4. Масштабирование в [0, 1].
5. Репликация в 3 канала (ResNet ожидает 3-канальный вход).
6. Нормализация mean/std ImageNet.

Итог: тензор формата `(C, H, W)` типа `float32`, готовый для модели.

### 3.4 Классификация эмоции

Файл: `src/emotion/model.py`

`create_emotion_model(...)` создает `EmotionResNet`:
- backbone: `resnet18`/`resnet34`/`resnet50`
- можно загрузить checkpoint

В `EmotionResNet`:
- берется pretrained ResNet (ImageNet)
- удаляется исходный финальный FC слой
- добавляется собственная голова:
  - Flatten
  - Dropout
  - Linear -> 256
  - ReLU
  - Dropout
  - Linear -> 7 классов

В инференсе используется `predict_proba(...)`, затем `argmax` для выбора класса и confidence.

### 3.5 Сборка результатов и визуализация

Файл: `src/pipeline/emotion_pipeline.py`

`EmotionPipeline.process_frame(...)`:
1. Находит лица через детектор.
2. Для каждого лица получает crop (`aligned_face` либо вырезка по bbox).
3. Вызывает `_predict_emotion(...)`.
4. Возвращает список словарей:
   - `bbox`
   - `emotion`
   - `confidence`
   - `landmarks`

`EmotionPipeline.draw_results(...)`:
- рисует рамки разных цветов для разных эмоций;
- добавляет текст вида `happy 0.87`.

## 4. Обучение модели

### 4.1 Обучение на FER2013

Скрипт: `scripts/train_emotion.py`

Что делает:

1. Создает модель (`create_emotion_model`).
2. Запускает функцию `train(...)` из `src/emotion/trainer.py`.

Файл `src/emotion/trainer.py`:

- Проверяет наличие файла датасета в `data/fer2013/`.
- Формирует train/test датасеты через `get_fer2013_dataset(...)`.
- Применяет трансформации:
  - grayscale -> 3 канала
  - resize
  - normalize
  - аугментации для train (flip, rotation/affine, jitter)
- Обучает по эпохам:
  - loss: `CrossEntropyLoss(label_smoothing=0.1)`
  - optimizer: `AdamW`
  - scheduler: `CosineAnnealingLR`
- Сохраняет лучший checkpoint по test accuracy в:
  - `checkpoints/best_emotion_model.pt`

### 4.2 Дообучение на собственном датасете

Скрипт: `scripts/train_custom.py`

Ожидает структуру ImageFolder:
- `data_dir/train/<class_name>/...`
- `data_dir/test/<class_name>/...` (опционально)

Особенность:
- классы папок ImageFolder идут в алфавитном порядке;
- в проекте фиксирован порядок FER2013.

Поэтому строится `label_map`, который переводит индексы ImageFolder в правильные индексы FER2013. Это важно для совместимости с `EmotionPipeline.LABELS`.

Если есть `test`, лучшая модель сохраняется по test accuracy; если `test` нет, по train accuracy.

### 4.3 Загрузка FER2013

Скрипт: `scripts/download_fer2013.py`

- Пытается скачать данные через `opendatasets` с Kaggle.
- Копирует CSV в `data/fer2013/`.
- Если автоматический путь не сработал, печатает инструкцию для ручной загрузки.

## 5. Оценка качества

Скрипт: `scripts/eval_accuracy.py`

Поддерживает 2 режима:

1. Если существует `data_dir/test`, использует ваш кастомный `ImageFolder` test.
2. Иначе оценивает на FER2013 test split.

На выходе:
- общая accuracy
- accuracy по каждому классу

## 6. Работа с несколькими видеопотоками

Скрипт: `scripts/run_stream.py`

Использует `VideoStreamManager` из `src/streams/video_manager.py`.

`VideoStreamManager` умеет:
- открыть несколько источников (`open`)
- читать кадры по очереди (`iter_frames`)
- пропускать кадры (`process_every_n`) для ускорения
- корректно освобождать ресурсы (`close`)

Дальше каждый кадр проходит тот же `EmotionPipeline`.

## 7. Конфигурация проекта

Файл: `config/default.yaml`

Там описаны параметры по умолчанию:
- порог детекции лица
- размер входа эмоц. модели
- число классов
- гиперпараметры обучения
- параметры потока (fps, очереди, skip)

На текущий момент CLI-скрипты задают часть параметров напрямую аргументами, но YAML можно использовать как единый источник значений по умолчанию при дальнейшем расширении.

## 8. Роли ключевых файлов

- `scripts/run_inference.py`: инференс на фото/видео/камере
- `scripts/run_stream.py`: инференс на нескольких камерах
- `scripts/train_emotion.py`: базовое обучение на FER2013
- `scripts/train_custom.py`: дообучение на пользовательских данных
- `scripts/eval_accuracy.py`: подсчет метрик
- `scripts/download_fer2013.py`: загрузка FER2013
- `src/pipeline/emotion_pipeline.py`: объединяет детекцию + предобработку + классификацию
- `src/face_detection/retinaface_detector.py`: детектор лиц + alignment
- `src/emotion/model.py`: архитектура и загрузка весов
- `src/emotion/preprocessing.py`: подготовка входных данных
- `src/emotion/trainer.py`: цикл обучения и валидации
- `src/streams/video_manager.py`: итерация по кадрам из одного/нескольких источников

## 9. Типовой сценарий работы

1. Подготовить FER2013 (`scripts/download_fer2013.py`).
2. Обучить модель (`scripts/train_emotion.py`).
3. Проверить качество (`scripts/eval_accuracy.py`).
4. Запустить инференс (`scripts/run_inference.py` или `scripts/run_stream.py`).
5. При необходимости дообучить под домен (`scripts/train_custom.py`).

## 10. Важные практические замечания

- Без обученного checkpoint инференс возможен, но качество будет низким (модель с ImageNet-весами без эмоц. fine-tuning).
- На качество сильно влияют освещение, размер лица, поворот головы и окклюзии.
- Для более высокой точности обычно требуется доменное дообучение и улучшение датасета.
- Если нужен realtime на слабом железе, используйте:
  - `--skip` в `run_stream.py`
  - меньший input size (компромисс между скоростью и качеством)
  - GPU при наличии.

---

## 11. Архитектура (дополнительно)

Ниже три представления архитектуры: слои, компоненты и runtime.

### 11.1 Слойная архитектура

```
+----------------------------------------------------------+
|                   Presentation Layer                     |
| scripts/run_inference.py, scripts/run_stream.py         |
| (CLI, окно OpenCV, визуализация рамок и эмоций)         |
+----------------------------------------------------------+
        |
        v
+----------------------------------------------------------+
|                   Application Layer                      |
| src/pipeline/emotion_pipeline.py                         |
| (оркестрация: detect -> preprocess -> predict -> render) |
+----------------------------------------------------------+
        |
        v
+---------------------------+   +--------------------------+
| Domain: Face Detection    |   | Domain: Emotion Model    |
| src/face_detection/...    |   | src/emotion/...          |
| MTCNN + align             |   | preprocessing + ResNet   |
+---------------------------+   +--------------------------+
        |
        v
+----------------------------------------------------------+
|                   Infrastructure Layer                   |
| PyTorch, torchvision, facenet-pytorch, OpenCV, CUDA     |
+----------------------------------------------------------+
        |
        v
+----------------------------------------------------------+
|                        Data Layer                        |
| data/fer2013, custom ImageFolder, checkpoints/*.pt      |
+----------------------------------------------------------+
```

Смысл разделения:
- слой CLI ничего не знает о деталях нейросети;
- `EmotionPipeline` изолирует бизнес-логику обработки кадра;
- доменные модули можно переиспользовать отдельно (например, только обучение);
- инфраструктурные зависимости (GPU, OpenCV, torch) не «протекают» в верхний уровень API.

### 11.2 Компонентная архитектура (inference)

```
InputSource (image/video/camera)
    |
    v
Frame Reader (OpenCV)
    |
    v
EmotionPipeline
  |- FaceDetector (RetinaFaceDetector -> MTCNN backend)
  |    |- detect faces
  |    |- detect landmarks
  |    `- optional face alignment
  |
  |- EmotionPreprocessor
  |    |- grayscale/resize/hist_eq
  |    `- normalization to model tensor
  |
  `- EmotionResNet
   |- forward pass
   `- softmax -> label + confidence

Result Renderer (bbox + text)
    |
    v
Output (window/video file)
```

Граница ответственности:
- Детектор отвечает только за геометрию лица и landmarks.
- На практике эту роль выполняет MTCNN: он определяет bbox и ключевые точки, которые затем используются для align.
- Препроцессор отвечает только за преобразование пикселей.
- Модель отвечает только за вероятности классов.
- Пайплайн связывает все этапы и формирует итоговую структуру результата.

### 11.3 Runtime-архитектура

#### Одиночный источник (`run_inference.py`)

```
Main Thread
  -> cv2.VideoCapture.read()
  -> EmotionPipeline.process_frame()
   -> detector.detect()
   -> preprocessor.preprocess()
   -> model.predict_proba()
  -> EmotionPipeline.draw_results()
  -> cv2.imshow()
```

Особенность: это последовательный цикл. Простой и предсказуемый, но максимальный FPS ограничен суммой времени всех этапов.

#### Несколько источников (`run_stream.py` + `VideoStreamManager`)

```
Main Thread
  -> manager.iter_frames()  # круговой опрос источников
  -> process(frame, idx)
   -> EmotionPipeline.process_frame()
   -> cv2.imshow(f"Camera {idx}")
```

Особенность: источники обрабатываются по очереди в одном цикле. Параметр `process_every_n` снижает вычислительную нагрузку пропуском кадров.

### 11.4 Архитектура обучения

```
Dataset (FER2013 / ImageFolder)
  |
  v
DataLoader (batching, shuffle, workers)
  |
  v
Model (EmotionResNet)
  |
  v
Loss (CrossEntropy + label_smoothing)
  |
  v
Optimizer (AdamW) + Scheduler (CosineAnnealingLR)
  |
  v
Validation Loop
  |
  v
Checkpoint Manager (save best_emotion_model.pt)
```

Ключевая идея: checkpoint сохраняется по лучшему качеству на валидации/тесте, что защищает от деградации на поздних эпохах.

### 11.5 Артефакты и контракты между модулями

- Вход пайплайна: `np.ndarray` кадр (обычно BGR от OpenCV).
- Выход детектора: список словарей c `bbox`, `confidence`, опционально `landmarks`, `aligned_face`.
- Вход модели: тензор `(N, C, H, W)` после нормализации.
- Выход модели: логиты/вероятности по 7 классам.
- Выход пайплайна: список результатов (`emotion`, `confidence`, `bbox`) для рендеринга/логирования.

Именно эти контракты позволяют менять реализацию внутренних узлов (например, заменить MTCNN на другой детектор) без переписывания всего приложения.

---

Если нужно, могу следующим шагом добавить визуальные Mermaid-диаграммы прямо в этот же файл (для удобства просмотра в Markdown-рендерах, которые поддерживают Mermaid).
