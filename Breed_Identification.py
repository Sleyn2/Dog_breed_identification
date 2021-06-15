import numpy as np
import pandas
import os

import matplotlib.pyplot as plt

from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator as Imgen
from os.path import join
from keras.applications.nasnet import NASNetLarge
from keras.applications.inception_resnet_v2 import InceptionResNetV2

from keras.models import Sequential,load_model
from keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,GlobalAveragePooling2D,Dropout

from keras.preprocessing import image

import cv2

import pickle

data_dir = 'input/'
labels = pandas.read_csv(join(data_dir, 'labels.csv'))
sample = pandas.read_csv(join(data_dir, 'sample_submission.csv'))


# print(labels.head())

def app_jpg(id):
    return id + ".jpg"


def plot_images(img,labels):
    plt.figure(figsize=[15,10])
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.imshow(img[i])
        plt.title(class_names[np.argmax(labels[i])])
        plt.axis('off')

labels['id'] = labels['id'].apply(app_jpg)
sample['id'] = sample['id'].apply(app_jpg)

datagen = Imgen(preprocessing_function=keras.applications.nasnet.preprocess_input,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                validation_split=0.2
                )
train_ds = datagen.flow_from_dataframe(
    labels,
    directory='input/train',
    x_col='id',
    y_col='breed',
    subset="training",
    color_mode="rgb",
    target_size=(331, 331),
    class_mode="categorical",
    batch_size=32,
    shuffle=True,
    seed=123,
)

val_ds = datagen.flow_from_dataframe(
    labels,
    directory='input/train',
    x_col='id',
    y_col='breed',
    subset="validation",
    color_mode="rgb",
    target_size=(331, 331),
    class_mode="categorical",
    batch_size=32,
    shuffle=True,
    seed=123,
)

a = train_ds.class_indices
class_names = list(a.keys())
class_names[:10]

x,y = next(train_ds)
x.shape
plot_images(x,y)

base_model = InceptionResNetV2(include_top=False,
                     weights='imagenet',
                     input_shape=(331,331,3)
                     )
base_model.trainable = False

model = Sequential([
    base_model,

    GlobalAveragePooling2D(),

    Dense(256, activation='relu'),
    Dropout(0.5),

    Dense(120, activation='softmax')
])
model.summary()

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
my_calls = [keras.callbacks.EarlyStopping(monitor='val_accuracy',patience=2),
            keras.callbacks.ModelCheckpoint("Model.h5",verbose=1,save_best_only=True)]

hist = model.fit(train_ds,epochs=10,validation_data=val_ds,callbacks=my_calls)

plt.figure(figsize=(15,6))

plt.subplot(1,2,1)
plt.plot(hist.epoch,hist.history['accuracy'],label = 'Training')
plt.plot(hist.epoch,hist.history['val_accuracy'],label = 'validation')

plt.title("Accuracy")
plt.legend()

plt.subplot(1,2,2)
plt.plot(hist.epoch,hist.history['loss'],label = 'Training')
plt.plot(hist.epoch,hist.history['val_loss'],label = 'validation')

plt.title("Loss")
plt.legend()
plt.show()