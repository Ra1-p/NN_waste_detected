# python predict.py --image images/cardboard.jpg --model output/VGGnet_SGD_default/smallvggnet4.model --label-bin output/VGGnet_SGD_default/smallvggnet_lb4.pickle --width 64 --height 64

# импортируем необходимые библиотеки
from keras.models import load_model
import pickle
import cv2
import os

# подключаемся к вебкамере(по умолчанию 0 это веб камера ноутбука который подключен к 0 порту)
cap = cv2.VideoCapture(0)

# Создаем переменные для константного значения высоты и ширины изображения
width = 64
height = 64

# Создание выходной переменной для записи видеоряда выходных изображений
out = cv2.image('output.mp4', -1, 20.0, (640,480))

while True:
        ret, image = cap.read()  # Получаем фреймы с помощью функции чтения видео
        output = image.copy()  # Сохраняем копию

        image = cv2.resize(image, (width, height))

        # Масштабируем значение пикселей в диапазон [0, 1]
        image = image.astype("float") / 255.0

        # проверяем, нужно ли сгладить изображение и добавить пакет размеров
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

        # загрузить модель и метки
        print("[Информация!] сеть загрузки и бинаризатор меток...")
        model = load_model('output/VGGnet_SGD_default/smallvggnet4.model')
        lb = pickle.loads(open('output/VGGnet_SGD_default/smallvggnet_lb4.pickle', 'rb').read())

        # сделать прогноз по изображению
        predicts = model.predict(image)
        print(predicts, lb.classes_)

        # найти индекс метки класса с наибольшей вероятностью совпадения
        i = predicts.argmax(axis=1)[0]
        label = lb.classes_[i]

        # рисуем метку класса + вероятность на выходном изображении
        text = "{}: {:.2f}%".format(label, predicts[0][i] * 100)
        cv2.putText(output, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        # out.write(output)

        # Вывод на экран выходного значения
        # cv2.imshow("Image", output)
        output_dir = '/output/image'
        output_filename = 'output_1.png'
        output_path = os.path.join(output_dir, output_filename)
        success = cv2.imwrite(output_path, output)


        # Остановка изображения экрана кнопкой «q»
        if cv2.waitKey(50) & 0xFF == ord('q'):
                break

out.release()