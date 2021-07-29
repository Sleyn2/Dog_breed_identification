import matplotlib.pyplot as plt
import numpy as np
import pandas
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator as Imgen
from tensorflow import keras

NUM_CLASSES = 120
IMAGE_SIZE = 200
BATCH_SIZE = 32
NUM_EPOCHS = 80
VALIDATION_SPLIT = 0.2


def app_jpg(id):
    return id + ".jpg"


def plot_images(img, labels):
    plt.figure(figsize=[15, 10])
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.imshow(img[i])
        plt.title(class_names[np.argmax(labels[i])])
        plt.axis('off')


# def check_images(img):
#     plt.figure(figsize=[25, 10])
#     for i in range(25):
#         plt.subplot(5, 5, i + 1)
#         plt.imshow(img[i])
#         plt.title(class_names[pred[i]])
#         plt.axis('off')


test_path = 'input/test'
labels_path = 'input/labels.csv'
train_path = 'input/train'
sample_path = 'input/sample_submission.csv'

samples = pandas.read_csv(sample_path)
labels = pandas.read_csv(labels_path)
labels['id'] = labels['id'].apply(app_jpg)
samples['id'] = samples['id'].apply(app_jpg)

datagen = Imgen(preprocessing_function=keras.applications.nasnet.preprocess_input,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                validation_split=VALIDATION_SPLIT
                )

train_dataset = datagen.flow_from_dataframe(
    labels,
    directory=train_path,
    x_col='id',
    y_col='breed',
    subset="training",
    color_mode="rgb",
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    class_mode="categorical",
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=123,
)

validation_dataset = datagen.flow_from_dataframe(
    labels,
    directory=train_path,
    x_col='id',
    y_col='breed',
    subset="validation",
    color_mode="rgb",
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    class_mode="categorical",
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=123,
)

a = train_dataset.class_indices
class_names = list(a.keys())
class_names[:NUM_CLASSES]

x, y = next(train_dataset)
x.shape
plot_images(x, y)

base_model = InceptionResNetV2(include_top=False,
                               weights=None,
                               input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)
                               )
base_model.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(120, activation='softmax')
])
model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
my_calls = [keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=7),
            keras.callbacks.ModelCheckpoint("Model.h5", verbose=1, save_best_only=True)]

hist = model.fit(train_dataset, epochs=NUM_EPOCHS, validation_data=validation_dataset, callbacks=my_calls)

plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
plt.plot(hist.epoch, hist.history['accuracy'], label='Training')
plt.plot(hist.epoch, hist.history['val_accuracy'], label='validation')

plt.title("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(hist.epoch, hist.history['loss'], label='Training')
plt.plot(hist.epoch, hist.history['val_loss'], label='validation')

plt.title("Loss")
plt.legend()
plt.show()

# model = load_model("./Model.h5")
#
# testgen = Imgen(preprocessing_function=keras.applications.inception_resnet_v2.preprocess_input)
#
# test_dataset = testgen.flow_from_dataframe(
#     samples,
#     directory=test_path,
#     x_col='id',
#     y_col=None,
#     target_size=(IMAGE_SIZE, IMAGE_SIZE),
#     class_mode=None,
#     batch_size=BATCH_SIZE,
#     shuffle=False
# )
#
# predictions = model.predict(test_dataset, verbose=1)
# pred = [np.argmax(i) for i in predictions]
#
# X = next(test_dataset)
# X.shape
#
# check_images(X)
