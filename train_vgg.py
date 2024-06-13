# USAGE

# python train_vgg.py --dataset ../dataset/TrashBox/TrashBox_train_set --model output/smallvggnet1.model --label-bin output/smallvggnet_lb1.pickle --plot output/smallvggnet_plot1.png

# импортируем серверную часть Agg из matplotlib, чтобы сохранить графики на диск.
import matplotlib
matplotlib.use("Agg")

# импортируем необходимые пакеты
from pyimagesearch.smallvggnet import SmallVGGNet
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import pickle
import cv2
import os

# создаем парсер аргументов
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset of images")
ap.add_argument("-m", "--model", required=True,
	help="path to output trained model")
ap.add_argument("-l", "--label-bin", required=True,
	help="path to output label binarizer")
ap.add_argument("-p", "--plot", required=True,
	help="path to output accuracy/loss plot")
args = vars(ap.parse_args())

# инициализация изображений и меток
print("[:Информация:] Загрузка изображений...")
data = []
labels = []

# получение пути к изображениям и рандомное смешивание данных
imagePaths = sorted(list(paths.list_images(args["dataset"])))
random.seed(42)
random.shuffle(imagePaths)

# цикл загрузки изображнии из директории
for imagePath in imagePaths:
	# загрузка изображения, изменения размера на 64х64 и добавление к массиву
	image = cv2.imread(imagePath)
	image = cv2.resize(image, (64, 64))
	data.append(image)

	# извлекаем метку класса из пути к изображению и обновляем список тегов
	label = imagePath.split(os.path.sep)[-2]
	labels.append(label)

# масштабируем интенсивность пикселей по диапазону [0, 1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# разделение полученных изображений на тестовую и обучающую составляющую соотношением 25%/75%
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)

# конвертировать метки из целых чисел в векторы
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# увеличиваем количество данных с помощью аугментации данных
aug = ImageDataGenerator(
	rotation_range=40,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.2,
	zoom_range=0.2,
	horizontal_flip=True,
	fill_mode="nearest")

# Загрузка и создание модели нейронной сети из файла smallvggnet.py
model = SmallVGGNet.build(width=64, height=64, depth=3,	classes=len(lb.classes_))

print("[INFO]: Model information:")
model.summary()
# инициализировать скорость обучения, общее количество эпох и размер пакета
INIT_LR = 0.01
EPOCHS = 10
BS = 32
decay = 0.001

# скомпилируем модель с помощью SGD
print("[INFO] training network...")
opt = SGD(lr=INIT_LR, weight_decay=decay, nesterov=True)
text = model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])


# обучение модели нейронной сети
H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),
	validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
	epochs=EPOCHS)

# оценка нейронной сети по тестовым данным
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=BS)
print(classification_report(testY.argmax(axis=1),
	predictions.argmax(axis=1), target_names=lb.classes_))

# построение графиков итогов обучения
N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.subplot(1, 2, 1)
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["accuracy"], label="train_acc")
plt.title('Тренировочные значения')
plt.xlabel("Эпоха №")
plt.ylabel("Потери/Точность")
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.plot(N, H.history["val_accuracy"], label="val_acc")
plt.title("Тренировочные значения")
plt.xlabel("Эпоха №")
plt.ylabel("Потери/Точность")
plt.legend()
plt.savefig(args["plot"])

# сохраняем полученные модели обучения и меток
print("[:Информация:] сериализация сети и бинаризатор меток...")
model.save(args["model"], save_format='h5')
f = open(args["label_bin"], "wb")
f.write(pickle.dumps(lb))
f.close()