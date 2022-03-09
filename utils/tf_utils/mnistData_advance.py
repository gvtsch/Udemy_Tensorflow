import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist 
from tensorflow.keras.layers.experimental.preprocessing import RandomRotation # Rotieren eines Bildes (Data Augmentation)
from tensorflow.keras.layers.experimental.preprocessing import RandomTranslation # Verschieben eines Bildes (Data Augmentation)
from tensorflow.keras.layers.experimental.preprocessing import RandomZoom # Zoom Bild (Data Augmentation)
from tensorflow.keras.layers.experimental.preprocessing import Rescaling # Quasi MinMaxScaling
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential

# %%
class MNIST:
    def __init__(self, validation_size: float = 0.33) -> None:
        # User-defined constants
        self.num_classes = 10
        self.batch_size = 128
        # Load dataset
        (self.x_train, self.y_train), (self.x_test, self.y_test) = mnist.load_data()
        # Split dataset
        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(
            self.x_train,
            self.y_train,
            test_size=validation_size
        )
        # Preprocess x
        self.x_train = np.expand_dims(self.x_train, axis=-1).astype(np.float32)
        self.x_test = np.expand_dims(self.x_test, axis=-1).astype(np.float32)
        self.x_val = np.expand_dims(self.x_val, axis=-1).astype(np.float32)
        # Preprocess y
        self.y_train = to_categorical(self.y_train, num_classes=self.num_classes)
        self.y_test  = to_categorical(self.y_test, num_classes=self.num_classes)
        self.y_val  = to_categorical(self.y_val, num_classes=self.num_classes)
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
    def _build_preprocessing() -> Sequential:
        model = Sequential()
        model.add(Rescaling(scale=(1./255.), offset=0.0))
        return model
    
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
        preprocessing_model = self._build_preprocessing()
        dataset = dataset.map(
            map_func=lambda x, y: (preprocessing_model(x, training=False), y),
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
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