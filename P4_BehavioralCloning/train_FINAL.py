import csv
import os
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Flatten, Dense, Activation, MaxPooling2D, Conv2D, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.backend import tf as ktf
import cv2
import numpy as np
from tensorflow.contrib.layers import flatten
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from keras import backend as K
import collections

lines = []

dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path = os.path.normpath(dir_path)

# separator = os.path.sep

# for windows, ignore if mac
separator = '\\'

with open(dir_path+'/data3/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
measurements = []

input_size = (66,200) # height, width

for line in lines:
    source_path = line[0]

    # split based on slashes, then take final item (filename)
    filename = source_path.split('\\')[-1]
    current_path = dir_path + separator + 'data3{}IMG{}'.format(separator,separator,separator)+ filename

    image = cv2.imread(current_path)
    images.append(image)

    measurement = float(line[3])
    measurements.append(measurement)
    images.append(cv2.flip(image,1))
    measurements.append(-measurement)


def delete_straight(images, measurements):
    nbins = 23
    avg_samples_per_bin = len(measurements)/nbins
    hist, bins = np.histogram(measurements, nbins)

    # plt.hist(measurements,bins)
    # plt.show()

    # determine keep probability for each bin: if below avg_samples_per_bin, keep all; otherwise keep prob is proportional to number of samples above the average, so as to bring the number of samples for that bin down to the average

    keep_probs = []
    target = avg_samples_per_bin * 0.8
    for i in range(nbins):
        if hist[i] < target:
            keep_probs.append(1.)
        else:
            keep_probs.append(1./(hist[i]/target))
    remove_list = []

    # delete straight driving data to make sure car doesn't tend to drive straight even in corners
    for i in range(len(measurements)):
        if measurements[i]<=0.1 and measurements[i]>=-0.1:
            if np.random.uniform(0,1) > 0.5:
                remove_list.append(i)

    print(remove_list)
    images = np.delete(images, remove_list, axis=0)
    measurements = np.delete(measurements, remove_list)
    # plt.hist(measurements,bins)
    # plt.show()
    return images, measurements

images, measurements = delete_straight(images,measurements)

x = np.array(images)
y = np.array(measurements)

x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.33, random_state=42)

def alexnet(width, height, depth, classes, weightsPath=None):
    # initialize the model
    model = Sequential()

    #normalize
    model.add(Lambda(lambda x:x/255.0-0.5,input_shape=(height,width,depth)))

    # ((top_crop, bottom_crop), (left_crop, right_crop))
    model.add(Cropping2D(cropping=((28,25),(0,0))))

    # new dim = 160-53 = 107
    # first set of CONV => RELU => POOL
    model.add(Conv2D(96,11, strides=2))
    # out (107-11)/4 = 24
    # model.add(Convolution2D(6, 5, 5, border_mode="valid",
    #     input_shape=(depth, height, width)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

    # out 8
    # second set of CONV => RELU => POOL
    model.add(Conv2D(256,5, strides=1))
    # out 3
    # model.add(Convolution2D(16, 5, 5, border_mode="valid"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    # 2
    # third set of CONV => RELU => POOL
    model.add(Conv2D(384,3, strides =1))
    # model.add(Convolution2D(6, 5, 5, border_mode="valid",
    #     input_shape=(depth, height, width)))
    model.add(Activation("relu"))
    # model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # 4th set of CONV => RELU => POOL
    model.add(Conv2D(384,3))
    # model.add(Convolution2D(6, 5, 5, border_mode="valid",
    #     input_shape=(depth, height, width)))
    model.add(Activation("relu"))
    # model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # 5th set of CONV => RELU => POOL
    model.add(Conv2D(256,3))
    # model.add(Convolution2D(6, 5, 5, border_mode="valid",
    #     input_shape=(depth, height, width)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    # set of FC => RELU layers
    model.add(Flatten())

    # output shape: 120 neurons
    model.add(Dense(4096))
    # Dense is just fully connected layers (y = wx +b )
    model.add(Activation("relu"))

    model.add(Dense(4096))
    model.add(Activation("relu"))
    # softmax classifier

    model.add(Dense(1000))
    model.add(Activation("relu"))

    model.add(Dense(classes))

    # output 1 class
    # don't activate cuz regression
    # model.add(Activation("softmax"))

    # if weightsPath is specified load the weights
    if weightsPath is not None:
        model.load_weights(weightsPath)
    return model

def nvidia(width, height, depth, classes, weightsPath=None):
    # width=320, height=160, depth=3
    # initialize the model
    model = Sequential()
    #normalize
    model.add(Lambda(lambda x:x/255.0-0.5,input_shape=(height,width,depth)))

    # IMPORTANT: Must do all resizing in the network itself, because in our case we're loading a fixed image width and height (unless we can specify it when loading, but this generalizes better)

    # # ((top_crop, bottom_crop), (left_crop, right_crop))
    model.add(Cropping2D(cropping=((70,24),(0,0))))
    # new height: 160-94 = 66
    # new size

    model.add(Lambda(lambda x: K.tf.image.resize_images(x, input_size), input_shape=(66, 320, 3)))
    # matches network input layer requirements

    model.add(Conv2D(24,5, strides=2))
    model.add(Activation("relu"))

    model.add(Conv2D(36,5, strides=2))
    model.add(Activation("relu"))

    model.add(Conv2D(48,5, strides =2))
    model.add(Activation("relu"))

    model.add(Conv2D(64,3))
    model.add(Activation("relu"))

    model.add(Conv2D(64,3))
    model.add(Activation("relu"))

    model.add(Dropout(0.2))
    model.add(Flatten())

    model.add(Dense(100))
    model.add(Activation("relu"))

    model.add(Dense(50))
    model.add(Activation("relu"))

    model.add(Dense(10))
    model.add(Activation("relu"))

    model.add(Dense(classes))

    # output 1 class
    # don't activate because regression

    # if weightsPath is specified load the weights
    if weightsPath is not None:
        model.load_weights(weightsPath)

    return model


model = nvidia(width=320, height=160, depth=3, classes=1)
# model = alexnet(width=320, height=160, depth=3, classes=1)
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5, batch_size=128, verbose=1)
# evaluate the model
scores = model.evaluate(x_valid, y_valid, verbose=1)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
model.save("nvidiafinal.h5")
