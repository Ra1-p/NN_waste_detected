# USAGE
# python train_simple_nn.py --dataset animals --model output/simple_nn.model --label-bin output/simple_nn_lb.pickle --plot output/simple_nn_plot.png

# импортируем бэкенд Agg из matplotlib для сохранения графиков на диск
import matplotlib
matplotlib.use("Agg")

# подключаем необходимые пакеты
import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers import Dense
from keras import layers
from keras.optimizers import SGD
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import pickle
import cv2
import os


# создаём парсер аргументов и передаём их
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

# инициализируем данные и метки
print("[INFO] loading images...")
data = []
labels = []
image_width = 64
image_height = 64
batch_size = 8

# берём пути к изображениям и рандомно перемешиваем
imagePaths = sorted(list(paths.list_images(args["dataset"])))
random.seed(42)
random.shuffle(imagePaths)

# цикл по изображениям
for imagePath in imagePaths:
	# загружаем изображение, меняем размер на 32x32 пикселей (без учета
	# соотношения сторон), сглаживаем его в 32x32x3=3072 пикселей и
	# добавляем в список
	try:
		image = cv2.imread(imagePath)
		# image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		image = cv2.resize(image, (image_height, image_width)).flatten()
		data.append(image)
		# извлекаем метку класса из пути к изображению и обновляем
		# список меток
		label = imagePath.split(os.path.sep)[-2]
		labels.append(label)
	except Exception as e:
		print(f'Error an image: {imagePath}. With error: {e}')
		os.remove(imagePath)
		print(f'Delete image: {imagePath}')


# масштабируем интенсивности пикселей в диапазон [0, 1]
data = np.array(data, dtype="float")/ 255.0
labels = np.array(labels)

# разбиваем данные на обучающую и тестовую выборки, используя 80%
# данных для обучения и оставшиеся 20% для тестирования
(trainX, testX, trainY, testY) = train_test_split(data,
	labels, test_size=0.20, random_state=42)

# конвертируем метки из целых чисел в векторы (для 2х классов при
# бинарной классификации вам следует использовать функцию Keras
# “to_categorical” вместо “LabelBinarizer” из scikit-learn, которая
# не возвращает вектор)
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

print(trainX, trainY)

# # определим архитектуру 12288-4064-1024-512-256-128-7 с помощью Keras
model = Sequential()
model.add(Dense(4064, input_shape=(12288,), activation="sigmoid"))
model.add(Dense(1024, activation="sigmoid"))
model.add(Dense(512, activation="sigmoid"))
model.add(Dense(256, activation="sigmoid"))
model.add(Dense(128, activation="sigmoid"))
model.add(Dense(len(lb.classes_), activation="softmax"))

# инициализируем скорость обучения и общее число эпох
INIT_LR = 0.01
EPOCHS = 75
decay = 0.001

# компилируем модель, используя SGD как оптимизатор и категориальную
# кросс-энтропию в качестве функции потерь (для бинарной классификации
# следует использовать binary_crossentropy)
print("[INFO] training network...")
opt = SGD(lr=INIT_LR, weight_decay=decay)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])


# обучаем нейросеть
H = model.fit(trainX, trainY,
			  validation_data=(testX, testY),
			  epochs=EPOCHS,
			  batch_size=batch_size)

# оцениваем нейросеть
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=batch_size)
print(classification_report(testY.argmax(axis=1),
	predictions.argmax(axis=1), target_names=lb.classes_))

# строим графики потерь и точности
N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.subplot(1, 2, 1)
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["accuracy"], label="train_acc")
plt.title('Тренеровчные значения')
plt.xlabel("Эпоха №")
plt.ylabel("Потери/Точность")
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.plot(N, H.history["val_accuracy"], label="val_acc")
plt.title("Тестовое значения")
plt.xlabel("Эпоха №")
plt.ylabel("Потери/Точность")
plt.legend()
plt.savefig(args["plot"])

# сохраняем модель и бинаризатор меток на диск
print("[INFO] serializing network and label binarizer...")
model.save(args["model"], save_format="h5")
f = open(args["label_bin"], "wb")
f.write(pickle.dumps(labels))
f.close()