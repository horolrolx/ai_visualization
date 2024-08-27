from django.shortcuts import render
from django.conf import settings
import tensorflow as tf
from django.core.files.storage import default_storage
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
import os
import numpy as np
from PIL import Image


def cifar10(request):
    file = request.FILES.get("image")
    if not file:
        return render(request, "index.html")

    else:
        file_name = file.name
        file_path = os.path.join(settings.MEDIA_ROOT, "images", file_name)

        # Save the uploaded file
        with default_storage.open(file_path, "wb+") as destination:
            for chunk in file.chunks():
                destination.write(chunk)

        # 테스트 데이터 준비
        # test_image = Image.open(file_path)
        # test_image = test_image.resize((32, 32))
        # test_image = np.asarray(test_image) / 255.0
        # test_image = test_image.reshape(1, 32, 32, 3)
        test_image = image.load_img(file_path, target_size=(32, 32))
        test_image = img_to_array(test_image) / 255.0
        test_image = np.expand_dims(test_image, axis=0)

        # 사전학습 모델 가져오기
        model = load_model("cifar10/models/vgg4_model.h5")

        # 라벨이름 저장
        label_names = [
            "airplane",
            "automobile",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        ]
        pred = model.predict(test_image)
        objs_num = np.argmax(pred, axis=1)[0]
        objs = label_names[objs_num]
        probs = pred[0][int(objs_num)]

        return render(request, "index.html", {"objs": objs, "probs": probs})
