import os
import numpy as np
import cv2
from skimage import transform

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers.experimental.preprocessing import RandomRotation # Rotieren eines Bildes (Data Augmentation)
from tensorflow.keras.layers.experimental.preprocessing import RandomTranslation # Verschieben eines Bildes (Data Augmentation)
from tensorflow.keras.layers.experimental.preprocessing import RandomZoom # Zoom Bild (Data Augmentation)
from tensorflow.keras.layers.experimental.preprocessing import Rescaling # Quasi MinMaxScaling
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential

DATA_DIR = os.path.join("C:/Selbststudium/Zubehoer/Cats_and_dogs")
X_FILE_PATH = os.path.join(DATA_DIR, "x.npy")
Y_FILE_PATH = os.path.join(DATA_DIR, "y.npy")
IMG_SIZE = 64
IMG_DEPTH = 3
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, IMG_DEPTH)

# %%
def extract_cats_vs_dogs():
    # Bilder einlesen und preprocessing
    cats_dir = os.path.join(DATA_DIR, "Cat")
    dogs_dir = os.path.join(DATA_DIR, "Dog")
    dirs = [cats_dir, dogs_dir]
    class_names = ["cat", "dog"]
    # Was kein Bild ist, muss raus:
    for d in dirs:
        for f in os.listdir(d): # Alle Dateien, die in d liegen
            if f.split(".")[-1] != "jpg": # An der letzten Stelle der Liste, also die Dateiendung
                print(f"Removing file: {f}")
                os.remove(os.path.join(D, f))
    num_cats = len(os.listdir(cats_dir))
    num_dogs = len(os.listdir(dogs_dir))
    num_images = num_cats + num_dogs
    x = np.zeros(
        shape=(num_images, IMG_SIZE, IMG_SIZE, IMG_DEPTH),
        dtype=np.float32
    )
    y = np.zeros(
        shape=(num_images,),
        dtype=np.float32
    )

    cnt = 0
    for d, class_name in zip(dirs, class_names):
        for f in os.listdir(d):
            img_file_path = os.path.join(d, f)
            try:
                img = cv2.imread(img_file_path, cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                x[cnt] = transform.resize(
                    image=img,
                    output_shape=IMG_SHAPE
                )
                if class_name == "cat":
                    y[cnt] = 0
                elif class_name == "dog":
                    y[cnt] = 1
                else:
                    print(f"Invalid classname")
                cnt+=1
            except: # noqa: E722
                print(f"Image could not be read")
                os.remove(img_file_path)

    # Dropping not readable img_idxs
    x = x[:cnt]
    y = y[:cnt]

    np.save(X_FILE_PATH, x)
    np.save(Y_FILE_PATH, y)

# %%
class DOGSCATS:
    def __init__(self, test_size: float = 0.2, validation_size = 0.33) -> None:
        # User-defined constants
        self.num_classes = 10
        self.batch_size = 128
        # Load dataset
        x = np.load(X_FILE_PATH)
        y = np.load(Y_FILE_PATH)
        # Split dataset
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=validation_size)
        # Preprocess x
        self.x_train = x_train.astype(np.float32)
        self.x_test = x_test.astype(np.float32)
        self.x_val = x_val.astype(np.float32)
        # Preprocess y
        self.y_train = to_categorical(y_train, num_classes=self.num_classes)
        self.y_test  = to_categorical(y_test, num_classes=self.num_classes)
        self.y_val  = to_categorical(y_val, num_classes=self.num_classes)
        # Dataset attributes
        self.train_size = self.x_train.shape[0]
        self.test_size = self.x_test.shape[0]
        self.tval_size = self.x_val.shape[0]
        self.width = self.x_train.shape[1]
        self.height = self.x_train.shape[2]
        self.depth = self.x_train.shape[3]
        self.img_shape = (self.width, self.height, self.depth)
        # tf.data.Datasets: Daten aus den Numpy-Arrays holen und ins Dataset packen
        self.train_dataset = tf.data.Dataset.from_tensor_slices((self.x_train, self.y_train))
        self.test_dataset = tf.data.Dataset.from_tensor_slices((self.x_test, self.y_test))
        self.val_dataset = tf.data.Dataset.from_tensor_slices((self.x_val, self.y_val))
        # Data Augmentation
        self.train_dataset = self._prepare_dataset(self.train_dataset, shuffle=True, augment=True)
        self.test_dataset = self._prepare_dataset(self.test_dataset) # Kein Data Augmentation 
        self.val_dataset = self._prepare_dataset(self.val_dataset) # Kein Data Augmentation 

    def get_train_set(self) -> tf.data.Dataset:
        return self.train_dataset
    
    def get_test_set(self) -> tf.data.Dataset:
        return self.test_dataset

    def get_val_set(self) -> tf.data.Dataset:
        return self.val_dataset
    
    @staticmethod
    def _build_data_augmentation() -> Sequential:
        model = Sequential()
        model.add(RandomRotation(factor=0.08))
        model.add(RandomTranslation(height_factor=0.08, width_factor=0.08))
        model.add(RandomZoom(height_factor=0.08, width_factor=0.08))
        return model

    def _prepare_dataset(
        self,
        dataset: tf.data.Dataset,
        shuffle: bool = False,
        augment: bool = False
    ) -> tf.data.Dataset:
        if shuffle:
            dataset = dataset.shuffle(buffer_size=1_000)
        dataset = dataset.batch(batch_size=self.batch_size)
        if augment:
            data_augmentation_model = self._build_data_augmentation()
            dataset = dataset.map(
                map_func=lambda x, y: (data_augmentation_model(x, training=False), y),
                num_parallel_calls=tf.data.experimental.AUTOTUNE
            )
        return dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

data = DOGSCATS()