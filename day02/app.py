import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model

model = load_model("mnist_model.h5")

st.title("MNIST 이미지 분류")

file = st.file_uploader("이미지를 올려주세요", type=["png", "jpeg", "jpg"])

if file is None:
    st.text("이미지를 올려주세요")
else:
    img = Image.open(file)
    st.image(img, caption="Uploaded Image")

    # 이미지 변환
    img = img.convert("L")  # 회색변환
    img = np.array(img)  # 배열변환
    img = img.reshape(1, 28, 28, 1)  # 사이즈변환
    pred = np.argmax(model.predict(img), axis=1)  # 예측
    st.success("올린 손글씨는 {} 입니다.".format(pred[0]))
