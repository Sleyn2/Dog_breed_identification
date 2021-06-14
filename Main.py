import pandas
import sklearn
import numpy as np
import tensorflow as tf
import cv2

from os.path import join
from os import listdir
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import *
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau


def process_image(image_path):
    #czytanie obrazu
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, size=[IMAGE_SIZE, IMAGE_SIZE])
    return image

def build_model():
    inputs = Input((IMAGE_SIZE, IMAGE_SIZE, 3))
    backbone = MobileNetV2(input_tensor=inputs, include_top=False, weights="imagenet")
    backbone.trainable = True
    x = backbone.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.2)(x)
    x = Dense(1024, activation="relu")(x)
    x = Dense(NUM_CLASSES, activation="softmax")(x)
    model = tf.keras.Model(inputs, x)
    return model

def read_image(path, size):
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (size, size))
    image = image / 255.0
    image = image.astype(np.float32)
    return image

def parse_data(x, y):
    x = x.decode()
    image = read_image(x, NUM_IMAGES)
    label = [0] * NUM_CLASSES
    label[y] = 1
    label = np.array(label)
    label = label.astype(np.int32)
    return image, label

def tf_parse(x, y):
    x, y = tf.numpy_function(parse_data, [x, y], [tf.float32, tf.int32])
    x.set_shape((IMAGE_SIZE, IMAGE_SIZE, 3))
    y.set_shape(NUM_CLASSES)
    return x, y

def tf_dataset(x, y):
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.map(tf_parse)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.repeat()
    return dataset


# 1 Parameters
# TODO: set parameters

# 2 Importing dataset
NUM_CLASSES = 10
NUM_IMAGES = 1000
IMAGE_SIZE = 224
BATCH_SIZE = 32
LR = 1e-4
NUM_EPOCHS = 10

data_dir = 'input/'
labels = pandas.read_csv(join(data_dir, 'labels.csv'))
print("Calkowita ilosc obrazow treningowych: {}".format(len(listdir(join(data_dir, 'train')))))
# Wyswietlanie Num_classes - ilości ras
print("Pierwsze {} ras".format(NUM_CLASSES))
print(labels
      .groupby("breed")
      .count()
      .sort_values("id", ascending=False)
      .head(NUM_CLASSES)
      )

# TODO: add cross

# lista ras
breed_names = np.array(labels.breed)
# lista unikalnych ras
unique_breed_names = np.unique(breed_names)

boolean_breed_names = [label == np.array(unique_breed_names) for label in breed_names]

X = ['input/train/' + file + '.jpg' for file in labels.id]
y = boolean_breed_names

# walidacja krzyżowa
X_train, X_valid, y_train, y_valid = train_test_split(X[:NUM_IMAGES], y[:NUM_IMAGES], test_size=0.2)

model = build_model()
model.compile(loss="categorical_crossentropy", optimizer=Adam(LR), metrics=["acc"])
train_dataset = tf_dataset(X_train, y_train)
valid_dataset = tf_dataset(X_valid, y_valid)

callbacks = [
        ModelCheckpoint("model.h5", verbose=1, save_best_only=True),
        ReduceLROnPlateau(factor=0.1, patience=5, min_lr=1e-6)]

train_steps = (len(X_train)//BATCH_SIZE) + 1
valid_steps = (len(X_valid)//BATCH_SIZE) + 1
model.fit(train_dataset,
    steps_per_epoch=train_steps,
    validation_steps=valid_steps,
    validation_data=valid_dataset,
    epochs=NUM_EPOCHS,
    callbacks=callbacks)
