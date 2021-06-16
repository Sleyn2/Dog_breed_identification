import matplotlib.pyplot as plt
import numpy as np
import pandas
from keras.preprocessing.image import ImageDataGenerator as Imgen
from tensorflow import keras
from Breed_Identification import class_names, BATCH_SIZE, IMAGE_SIZE
from keras.models import load_model


def check_images(img):
    plt.figure(figsize=[25, 10])
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.imshow(img[i])
        plt.title(class_names[pred[i]])
        plt.axis('off')

test_path = 'input/test'
sample_path = 'input/sample_submission.csv'

samples = pandas.read_csv(sample_path)
model = load_model("./Model.h5")

testgen = Imgen(preprocessing_function=keras.applications.inception_resnet_v2.preprocess_input)

test_dataset = testgen.flow_from_dataframe(
    samples,
    directory=test_path,
    x_col='id',
    y_col=None,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    class_mode=None,
    batch_size=BATCH_SIZE,
    shuffle=False
)

predictions = model.predict(test_dataset, verbose=1)
pred = [np.argmax(i) for i in predictions]

X = next(test_dataset)
X.shape

check_images(X)
