import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import cifar10
from typing import Tuple
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

class MNIST:
    def __init__(self, with_normalization: bool = True):
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        self.x_train_: np.ndarray = None
        self.y_train_: np.ndarray = None
        self.x_vali_: np.ndarray = None
        self.y_vali_: np.ndarray = None
        self.val_size = 0
        self.train_splitted_size = 0
        # Preprocess x
        self.x_train = x_train.astype(np.float32)
        self.x_test  = x_test.astype(np.float32)
        if with_normalization:
            self.x_train = self.x_train / 255.0
            self.x_test = self.x_test / 255.0
        # Dataset attributes
        self.train_size = self.x_train.shape[0]
        self.test_size = self.x_test.shape[0]
        self.width = self.x_train.shape[1]
        self.height = self.x_train.shape[2]
        self.depth = self.x_train.shape[3]
        self.img_shape = (self.width, self.height, self.depth)
        self.num_classes = len(np.unique(y_train))
        # Preprocess y
        self.y_train = to_categorical(y_train, num_classes=self.num_classes, dtype=np.float32)
        self.y_test  = to_categorical(y_test, num_classes=self.num_classes, dtype=np.float32)

    def get_train_set(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.x_train, self.y_train
    
    def get_test_set(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.x_test, self.y_test

    def get_splitted_train_validation_set(self, validation_size: float = 0.33) -> Tuple:
        self.x_train_, self.x_val_, self.y_train_, self.y_val_ = train_test_split(
            self.x_train, 
            self.y_train, 
            test_size=validation_size
            )
        self.val_size = self.x_val_.shape[0]
        self.train_splitted_size = self.x_train_.shape[0]
        return self.x_train_, self.x_val_, self.y_train_, self.y_val_

    def data_augmentation(self, augment_size: int = 5_000) -> None:
        image_generator = ImageDataGenerator(
            width_shift_range=0.08, # 0.08 -> 8% -> Verschiebung bis zu 2 Pixel 
            height_shift_range=0.08,
            zoom_range=0.05, # %
            rotation_range=5
        )
        # Fit the data generator
        image_generator.fit(self.x_train, augment=True)
        # Get rnd Train Images for data augmentation
        rand_idx = np.random.randint(self.train_size, size=augment_size)
        x_augmented = self.x_train[rand_idx].copy()
        y_augmented = self.y_train[rand_idx].copy()
        x_augmented = image_generator.flow(
            x_augmented, 
            np.zeros(augment_size), 
            batch_size=augment_size, 
            shuffle=False).next()[0]
        # Append images to train set
        self.x_train = np.concatenate((self.x_train, x_augmented))
        self.y_train = np.concatenate((self.y_train, y_augmented))
        self.train_size = self.x_train.shape[0]