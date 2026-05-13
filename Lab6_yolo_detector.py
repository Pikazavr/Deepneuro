# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 17:27:20 2024

@author: AM4
"""
# Импортируем библиотеки
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import ultralytics
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import random

# Проверяем что доступно из оборудования
ultralytics.checks()

# Создаем модель из файла с предобученными весами. 
#!!! При первом вызове загружает веса, требуется интернет/
model = YOLO("yolov8s.pt")

test_dir = "D:/Conda_labs/data/test"

# собираем список изображений
test_files = [
    f for f in os.listdir(test_dir)
    if f.lower().endswith((".jpg", ".png", ".jpeg"))
]

random.seed()  # ← добавлено для случайного выбора при каждом запуске
random.shuffle(test_files)
random_files = test_files[:5]### добавил 5 случайныйх фото для угадывания вместо одного(они случайны)

for random_file in random_files:
    random_path = os.path.join(test_dir, random_file)
    print("Выбрано изображение:", random_path)

    # Запускаем модель на случайном изображении
    results = model(random_path)

    # Достаем результаты модели
    result = results[0]

    # Выводим результаты на экран при помощи OpenCV
    frame = result.plot()
    cv2.namedWindow("YOLOv8", cv2.WINDOW_NORMAL)
    cv2.imshow("YOLOv8", frame)
    cv2.waitKey(0)

# В результатах содержится много полезной информации:
boxes = result.boxes.cpu()       # Рамки объектов по умолчанию в формате YOLO
print(boxes)
boxes = result.boxes.xyxy.cpu()  # Координаты рамок можно преобразовать в пиксели
print(boxes)

# Вероятности для каждого из обнаруженных объектов
# чем больше, тем сеть увереннее что это именно этот объект
confidences = result.boxes.conf
print(confidences)

classes = result.boxes.cls # Номера классов для каждого объекта
print(classes)
class_names = result.names # Сами названия классов  
print(class_names)

# Даже исходное изображение
img = result.orig_img
plt.imshow(img[:, :, ::-1])

# напишем свою функцию для отрисовки прямоугольников на изображении
def draw_bboxes(image, results):
    boxes = results[0].boxes.cpu()
    orig_h, orig_w = results[0].orig_shape
    class_names = results[0].names
    for box in boxes:
        class_idx = box.cls
        confidence = box.conf
        if confidence > 0.7:
            x1, y1, x2, y2 = box.xyxy[0].numpy()
            cv2.rectangle(
                image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2, cv2.LINE_AA
            )
            cv2.putText(
                image, class_names[int(class_idx)], (int(x1), int(y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
            )
    return image

# вызовем функцию и выведем на экран то чот получилось
annotated_img = draw_bboxes(img.copy(), results)
cv2.namedWindow("YOLOv8", cv2.WINDOW_NORMAL)### вывод изображения через отдельно окно 
cv2.imshow("YOLOv8", annotated_img)
cv2.waitKey(0)

# Теперь попробуем обучить на собственном датасете
# он доступен по ссылке https://drive.google.com/file/d/1qS_yGj3vkmEuv9Fc3Xffqg6fE5C8m3AG/view?usp=drive_link

# перед обучением необходимо скорректировать пути в файле masked.yaml
# пути должны быть абсолютными
# обучение может занять много времени, особенно на CPU

model = YOLO("yolov8s.pt")

results = model.train(
    data="D:/Conda_labs/data.yaml",
    epochs=10,
    batch=8,
    imgsz=512,
    amp=False,
    workers=0,### крашило процессы добавил чтобы нормально запускалось 
    project='masks',
    val=True,
    verbose=True
)

# После обучения посмотрим несколько изображений из test
test_files = [
    f for f in os.listdir(test_dir)
    if f.lower().endswith((".jpg", ".png", ".jpeg"))
]

random.seed()  # ← добавлено для случайного выбора при каждом запуске
random.shuffle(test_files)### добавил просмотр изображений после обучения
random_files = test_files[:5]

for random_file in random_files:### вывод 5 изображений в конце для ручной проверки
    random_path = os.path.join(test_dir, random_file)
    print("Проверка изображения:", random_path)

    results = model(random_path)
    result = results[0]

    frame = result.plot()
    #cv2.namedWindow("YOLOv8 - проверка", cv2.WINDOW_NORMAL)
    cv2.imshow("YOLOv8 - проверка", frame)
    cv2.waitKey(0)

# Проверка модели на изображении
results = model(random_path)

# посмотрим что получилось
result = results[0]
frame = result.plot()
cv2.namedWindow("YOLOv8", cv2.WINDOW_NORMAL)
cv2.imshow("YOLOv8", frame)
cv2.waitKey(0)

# Попробуем обработать видео

# Открываем видеофайл
video_path = "masktrack.mp4"
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    success, frame = cap.read()
    if success:
        results = model(frame)
        result = results[0]
        cv2.imshow("YOLOv8", result.plot())
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
