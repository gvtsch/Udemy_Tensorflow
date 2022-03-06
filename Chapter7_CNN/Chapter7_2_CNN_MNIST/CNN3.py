#! path\to\interpreter\python.exe

# %%
import os # Arbeiten mit Pfaden
import numpy as np
from typing import Tuple
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.initializers import Constant
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import MaxPooling2D

# %%
LOGS_DIR = os.path.abspath('C:/Selbststudium/Udemy/Udemy_Tensorflow/logs')
if not os.path.exists(LOGS_DIR):
    os.mkdir(LOGS_DIR)
MODEL_LOG_DIR = os.path.join(LOGS_DIR, 'mnist_cnn3')

# %%
def get_dataset(img_shape: int, num_classes: int) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    x_train = x_train.astype(np.float32)
    x_test  = x_test.astype(np.float32)
    
    y_train = to_categorical(y_train, num_classes=num_classes, dtype=np.float32)
    y_test  = to_categorical(y_test, num_classes=num_classes, dtype=np.float32)

    return (x_train, y_train), (x_test, y_test)

# %%
def build_model(img_shape: int, num_classes: int) -> Sequential:
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=3, input_shape=img_shape))
    model.add(Activation('relu'))
    model.add(Conv2D(filters=32, kernel_size=3, input_shape=img_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D())


    model.add(Conv2D(filters=64, kernel_size=3))
    model.add(Activation('relu'))
    model.add(Conv2D(filters=64, kernel_size=3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D())

    model.add(Flatten())
    model.add(Dense(units=num_classes))
    model.add(Activation('softmax'))
    
    model.summary()
    return model

# %%
def main() -> None:
    img_shape = (28, 28, 1) # Bild hat 28*28 Pixel
    num_classes = 10 # 10 Ziffern möglich
    
    (x_train, y_train), (x_test, y_test) = get_dataset(img_shape, num_classes)
    
    model = build_model(img_shape, num_classes)

    opt = Adam(learning_rate=0.001)

    model.compile(
        loss='categorical_crossentropy', # wird bei Kategorie-Problemen mit mehr als 2 Klassen genommen
        optimizer=opt,
        metrics=['accuracy']
    )

    tb_callback = TensorBoard(
        log_dir=MODEL_LOG_DIR,
        histogram_freq=1,
        write_graph=True
    )

    model.fit(
        x=x_train,
        y=y_train,
        epochs=20,
        batch_size=256,
        verbose=1,
        validation_data=(x_test, y_test),
        callbacks=[tb_callback]
    )

    scores = model.evaluate(
        x=x_test, 
        y=y_test, 
        verbose=0
    )

# %%
if __name__=='__main__':
    main()


