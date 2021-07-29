# Make predictions on the validation data
import keras.models
import numpy as np
import pandas
import tensorflow as tf
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator as Imgen

# Set number of images to use
NUM_IMAGES = 1000
NUM_CLASSES = 120
IMAGE_SIZE = 200
BATCH_SIZE = 32
NUM_EPOCHS = 200
VALIDATION_SPLIT = 0.2


def app_jpg(id):
    return id + ".jpg"


# Creating function to return a tuple (image, label)
def image_label(path, label):
    image = process(path)
    return image, label


def process(image_path):
    # Read in image file
    image = tf.io.read_file(image_path)

    # Turn the jpeg image into numerical Tensor with 3 colour channels (Red, Green, Blue)
    image = tf.image.decode_jpeg(image, channels=3)

    # Convert the colour channel values from 0-225 values to 0-1 values
    image = tf.image.convert_image_dtype(image, tf.float32)

    # Resize the image to our desired size
    image = tf.image.resize(image, size=[IMAGE_SIZE, IMAGE_SIZE])
    return image


def data_batches(x, y=None, batch_size=BATCH_SIZE, valid_data=False, test_data=False):
    # If the data is a test dataset
    if test_data:
        print("Test data batches created")
        data = tf.data.Dataset.from_tensor_slices((tf.constant(x)))  # only filepaths
        data_batch = data.map(process).batch(BATCH_SIZE)
        return data_batch

    # If the data if a valid dataset
    elif valid_data:

        data = tf.data.Dataset.from_tensor_slices((tf.constant(x),  # filepaths
                                                   tf.constant(y)))  # labels
        data_batch = data.map(image_label).batch(BATCH_SIZE)
        print("Validation data batches created")
        return data_batch

    else:
        # If the data is a training dataset, we shuffle it

        # Turn filepaths and labels into Tensors
        data = tf.data.Dataset.from_tensor_slices((tf.constant(x),  # filepaths
                                                   tf.constant(y)))  # labels

        # Shuffling pathnames and labels
        data = data.shuffle(buffer_size=len(x))

        # Create (image, label) tuples
        data = data.map(image_label)

        # Turn the data into batches
        data_batch = data.batch(BATCH_SIZE)
        print("Training data batches created")
    return data_batch

train_path = 'input/train'
labels_path = 'input/labels.csv'

labels = pandas.read_csv(labels_path)
label = labels["breed"].to_numpy()  # convert labels column to NumPy array

# Find the number of unique label
unique = np.unique(label)
len(unique)

filenames = []
for name in labels["id"]:
    filenames.append("input/train/" + name + ".jpg")
boolean_label = [element == np.array(unique) for element in label]


# Setup X & y variables
x = filenames
y = boolean_label

# Split them into training and validation using NUM_IMAGES
x_train, x_val, y_train, y_val = train_test_split(x[:NUM_IMAGES],
                                                  y[:NUM_IMAGES],
                                                  test_size=0.2,
                                                  random_state=42)
labels = pandas.read_csv(labels_path)
labels['id'] = labels['id'].apply(app_jpg)
train_data = data_batches(x_train, y_train)
val_data = data_batches(x_val, y_val, valid_data=True)

# attributes of our data batches
train_data.element_spec, val_data.element_spec

datagen = Imgen(preprocessing_function=keras.applications.nasnet.preprocess_input,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                validation_split=VALIDATION_SPLIT
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

predictions = keras.models.load_model("Model.h5").predict(validation_dataset,
                                                          verbose=1)  # verbose shows us how long there is to go
predictions
predictions.shape


# First prediction
# print(predictions[0])
# print(
#     f"Max value (probability of prediction): {np.max(predictions[0])}")  # the max probability value predicted by the model
# print(f"Sum: {np.sum(predictions[0])}")
# print(f"Max index: {np.argmax(predictions[0])}")  # the index of where the max value in predictions[0] occurs
# print(f"Predicted label: {unique[np.argmax(predictions[0])]}")  # the predicted label

def pred_label(prediction_probabilities):
    return unique[np.argmax(prediction_probabilities)]


prediction_label = pred_label(predictions[0])
prediction_label

# data for batch

# Turn filepaths and labels into Tensors
data = tf.data.Dataset.from_tensor_slices((tf.constant(x),  # filepaths
                                           tf.constant(y)))  # labels

# Shuffling pathnames and labels
data = data.shuffle(buffer_size=len(x))

# Create (image, label) tuples
data = data.map(image_label)

val_images, val_labels = next(val_data.as_numpy_iterator())

# function to unbatch a batched dataset
def unbatchify(data):
    images = []
    labels = []

    # Loop through unbatched data
    for image, label in data.unbatch().as_numpy_iterator():
        images.append(image)
        labels.append(unique[np.argmax(label)])
    return images, labels


# Unbatchify the validation data
val_images, val_labels = unbatchify(val_data)
val_images[0], val_labels[0]


def plot_pred(prediction_probabilities, labels, images, n=1):
    pred_prob, true_label, image = prediction_probabilities[n], labels[n], images[n]

    # Get the pred label
    prediction_label = pred_label(pred_prob)

    # Plot image & remove ticks
    plt.imshow(image)
    plt.xticks([])
    plt.yticks([])

    # Change the color of the title depending on if the prediction is right or wrong
    if prediction_label == true_label:
        color = "green"
    else:
        color = "red"

    plt.title("{} {:2.0f}% ({})".format(prediction_label,
                                        np.max(pred_prob) * 100,
                                        true_label),
              color=color)


def plot_pred_conf(prediction_probabilities, labels, n=1):
    pred_prob, true_label = prediction_probabilities[n], labels[n]

    # Get the predicted label
    prediction_label = pred_label(pred_prob)

    # Find the top 7 prediction confidence indexes
    pred_indexes_7 = pred_prob.argsort()[-7:][::-1]

    # Find the top 7 prediction confidence values
    pred_values_7 = pred_prob[pred_indexes_7]

    # Find the top 7 prediction labels
    pred_labels_7 = unique[pred_indexes_7]

    # Setup plot
    top_plot = plt.bar(np.arange(len(pred_labels_7)),
                       pred_values_7,
                       color="purple")
    plt.xticks(np.arange(len(pred_labels_7)),
               labels=pred_labels_7,
               rotation="vertical")

    # Change color of true label
    if np.isin(true_label, pred_labels_7):
        top_plot[np.argmax(pred_labels_7 == true_label)].set_color("orange")
    else:
        pass


plot_pred_conf(prediction_probabilities=predictions,
               labels=val_labels,
               n=10)

# few predictions and their different values
mult = 0
rows = 2
cols = 3
num_images = rows * cols
plt.figure(figsize=(5 * 1.5 * cols, 5 * rows))
for i in range(num_images):
    plt.subplot(rows, 2 * cols, 2 * i + 1)
    plot_pred(prediction_probabilities=predictions,
              labels=val_labels,
              images=val_images,
              n=i + mult)
    plt.subplot(rows, 2 * cols, 2 * i + 2)
    plot_pred_conf(prediction_probabilities=predictions,
                   labels=val_labels,
                   n=i + mult)
plt.tight_layout(h_pad=1.0)
plt.show()
