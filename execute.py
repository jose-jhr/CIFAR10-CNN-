import random
from dataclasses import dataclass

import numpy as np
import tensorflow as tf
from keras.saving.experimental.saving_lib import load_model
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from tensorflow.keras.datasets import cifar10
from tensorflow import keras
from tensorflow.python.keras.layers import Conv2D, Dense, Dropout, MaxPooling2D, Flatten
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.utils.np_utils import to_categorical

# get data and split in train and test
(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
Y_test = to_categorical(Y_test)

# get model
model = keras.models.load_model('model/modelcnn.h5')


def evualuate_model(dataset, model):
    class_names = ['airplane',
                   'automobile',
                   'bird',
                   'cat',
                   'deer',
                   'dog',
                   'frog',
                   'horse',
                   'ship',
                   'truck']

    num_rows = 3
    num_col = 6

    # retrieve a number of images from the dataset
    data_batch = dataset[0:num_col * num_rows]

    # get prediction model
    predictions = model.predict(data_batch)

    # figure
    plt.figure(figsize=(20, 8))
    num_matches = 0

    for idx in range(num_rows * num_col):
        ax = plt.subplot(num_rows, num_col, idx + 1)
        plt.axis("off")
        plt.imshow(data_batch[idx])

        pred_idx = tf.argmax(predictions[idx]).numpy()
        truth_idx = np.nonzero(Y_test[idx])

        title = str(class_names[truth_idx[0][0]] + " : " + str(class_names[pred_idx]))
        title_obj = plt.title(title, fontdict={'fontsize': 13})
        if pred_idx == truth_idx:
            num_matches += 1
            plt.setp(title_obj, color='g')
        else:
            plt.setp(title_obj, color='r')
        acc = num_matches / (idx + 1)
        print("Prediction Accuracy", int(100 * acc) / 100)
        return


evualuate_model(X_test, model)

# generate prediction dataset
predictions = model.predict(X_test)
# for each sample image in the test dataset
predict_labels = [np.argmax(i) for i in predictions]

# convert one-hot encoded labels to integer
y_test_integer_labels = tf.argmax(Y_test, axis=1)
# generate matrix confution for the test dataset
cm = tf.math.confusion_matrix(labels=y_test_integer_labels, predictions=predict_labels)
# plot the confusion matriz as  a heatmap
plt.figure(figsize=(12, 6))
import seaborn as sn

sn.heatmap(cm, annot=True, fmt='d', annot_kws={"size": 12})
plt.title('Confusion matrix')
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()
