# %% [markdown]
# # FASHION MNIST EXAMPLE
# 참조: [https://www.datacamp.com/tutorial/convolutional-neural-networks-python](https://www.datacamp.com/tutorial/convolutional-neural-networks-python)

# %%
# 필요한 라이브러리 로딩
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from keras.datasets import fashion_mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
import visualkeras
from tensorflow.keras.utils import plot_model
from PIL import Image


# %%
def load_and_preprocess_data():

    # 패션이미지를 가지고 와서 훈련, 테스트세트로 분리
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    # 훈련, 테스트 사이즈 확인
    print(x_train.shape, x_test.shape)  # (60000, 28, 28), (10000, 28, 28)

    # 라벨 이름
    label_names = [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ]

    # 훈련 데이터 시각화
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(x_train[i])
        plt.title(label_names[y_train[i]])
        plt.axis("off")
    plt.tight_layout()
    plt.show()

    # 테스트 데이터 시각화
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(x_test[i])
        plt.title(label_names[y_test[i]])
        plt.axis("off")
    plt.tight_layout()
    plt.show()

    # 이미지 사이즈 변환
    x_train = x_train.reshape(60000, 28, 28, 1)
    x_test = x_test.reshape(10000, 28, 28, 1)

    # 이미지 정규화
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # 라벨 인코딩
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    return (x_train, y_train), (x_test, y_test)


# %%
def build_model(input_shape):
    model = Sequential()
    model.add(Conv2D(32, 3, activation="relu", input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, 3, activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, 3, activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(10, activation="softmax"))

    return model


# %%
def train_model(model, x_train, y_train, epochs, batch_size):
    # 컴파일 모델
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    # 체크 포인트 객체 생성

    ckpt_model = "fashion_mnist.keras"
    checkpoint = ModelCheckpoint(
        ckpt_model, monitor="val_accuracy", verbose=1, save_best_only=True, mode="max"
    )

    # 모델 훈련
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_test, y_test),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[checkpoint],
        verbose=1,
    )

    return model, history


# %%
def predict_model(model, input_data):
    # 예측
    y_pred = model.predict(x_test)
    y_pred = np.argmax(y_pred, axis=1)
    return y_pred


# %%
def evaluate_model(model, x_test, y_test, y_pred, history):
    # 모델평가
    score = model.evaluate(x_test, y_test)
    print(model.metrics_names[0], score[0])
    print(model.metrics_names[1], score[1])

    # 정확도
    print("accuracy", accuracy_score(np.argmax(y_test, axis=1), y_pred))

    # 히스토리 그래프
    plt.figure(figsize=(15, 5))
    plt.subplot(121)
    plt.plot(history.history["loss"], label="train")
    plt.plot(history.history["val_loss"], label="test")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("loss")
    plt.legend()
    plt.subplot(122)
    plt.plot(history.history["accuracy"], label="train")
    plt.plot(history.history["val_accuracy"], label="test")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.title("accuracy")
    plt.legend()
    plt.show()

    # 혼동행렬
    cfm = confusion_matrix(np.argmax(y_test, axis=1), y_pred)
    sns.heatmap(cfm, annot=True, fmt="g")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    # 분류리포트
    print(classification_report(np.argmax(y_test, axis=1), y_pred))


# %%
def prepare_input(image):
    # 테스트 데이터 가져오기
    image = Image.open(image).convert("L")

    # 사이즈 확인
    print("이미지 사이즈", image.size)

    # 이미지 자르고 사이즈 변환, 좌우반전
    image = image.crop((100, 400, 600, 900))
    image = image.resize((28, 28))
    image = image.transpose(Image.FLIP_LEFT_RIGHT)

    # 배열변환
    image = np.asarray(image)

    plt.imshow(image)
    plt.show()

    # 이미지 반전
    image = 255 - image

    plt.imshow(image)
    plt.show()

    # 이미지 전처리
    image = image.reshape(1, 28, 28, 1)
    image = image / 255.0

    return image


# %%
def main():
    # 데이터 로드 및 전처리 함수 호출
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()

    model = build_model((28, 28, 1))  # 모델 훈련
    model, history = train_model(model, x_train, y_train, 10, 64)
    print(model.summary())
    visualkeras.layered_view(model, legend=True)
    plot_model(model, "model.png", show_shapes=True)

    # 모델 저장
    model.save("fashion_mnist1.keras")

    y_pred = predict_model(model, x_test)

    evaluate_model(model, x_test, y_test, y_pred, history)

    image = prepare_input("ankleboot.jpg")
    print(np.argmax(model.predict(image), axis=1))


if "__name__" == "__main__":
    main()
