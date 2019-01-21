import csv
import os
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Flatten, Dense, Activation, MaxPooling2D, Conv2D, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
import cv2
import numpy as np
from tensorflow.contrib.layers import flatten
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
lines = []

dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path = os.path.normpath(dir_path)
print(dir_path)
# separator = os.path.sep
separator = '\\'


print(separator)


# def replace_backlash(text):
#     for i in text:
#         if i =='\'

with open(dir_path+'/data2/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
measurements = []

for line in lines:
    source_path = line[0]
    print(line[0])
    # split based on slashes, then take final item (filename)
    filename = source_path.split('\\')[-1]
    print("filename",filename)
    current_path = dir_path + separator + 'data2{}IMG{}'.format(separator,separator,separator)+ filename
    print("current", current_path)
    image = cv2.imread(current_path)
    #print(image)
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)
    print(measurement)
x = np.array(images)
# print(x_train)
y = np.array(measurements)

x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.33, random_state=42)
# print(y_train)
def lenet_nvidia(width, height, depth, classes, weightsPath=None):
    # initialize the model
    model = Sequential()

    #normalize
    model.add(Lambda(lambda x:x/255.0-0.5,input_shape=(height,width,depth)))

    model.add(Cropping2D(cropping=((70,25),(0,0))))
    # first set of CONV => RELU => POOL
    model.add(Conv2D(24,5, strides=2))
    # model.add(Convolution2D(6, 5, 5, border_mode="valid",
    #     input_shape=(depth, height, width)))
    model.add(Activation("relu"))
    # model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # second set of CONV => RELU => POOL
    model.add(Conv2D(36,5, strides=2))
    # model.add(Convolution2D(16, 5, 5, border_mode="valid"))
    model.add(Activation("relu"))
    # model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # third set of CONV => RELU => POOL
    model.add(Conv2D(48,5, strides =2))
    # model.add(Convolution2D(6, 5, 5, border_mode="valid",
    #     input_shape=(depth, height, width)))
    model.add(Activation("relu"))
    # model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # 4th set of CONV => RELU => POOL
    model.add(Conv2D(64,3))
    # model.add(Convolution2D(6, 5, 5, border_mode="valid",
    #     input_shape=(depth, height, width)))
    model.add(Activation("relu"))
    # model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # 5th set of CONV => RELU => POOL
    model.add(Conv2D(64,3))
    # model.add(Convolution2D(6, 5, 5, border_mode="valid",
    #     input_shape=(depth, height, width)))
    model.add(Activation("relu"))
    # model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    # set of FC => RELU layers
    model.add(Flatten())

    # output shape: 120 neurons
    model.add(Dense(100))
    # Dense is just fully connected layers (y = wx +b )
    model.add(Activation("relu"))

    model.add(Dense(50))
    model.add(Activation("relu"))
    # softmax classifier

    model.add(Dense(10))
    model.add(Activation("relu"))

    model.add(Dense(classes))

    #output 1 class

    # don't activate cuz regression
    # model.add(Activation("softmax"))

    # if weightsPath is specified load the weights
    if weightsPath is not None:
        model.load_weights(weightsPath)

    return model

# note: when use 40 and 20 works pretty well
def lenet_keras(width, height, depth, classes, weightsPath=None):
    # initialize the model
    model = Sequential()

    # first set of CONV => RELU => POOL
    model.add(Conv2D(8,5))
    # model.add(Convolution2D(6, 5, 5, border_mode="valid",
    #     input_shape=(depth, height, width)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # second set of CONV => RELU => POOL
    model.add(Conv2D(23,5))
    # model.add(Convolution2D(16, 5, 5, border_mode="valid"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # set of FC => RELU layers
    model.add(Flatten())

    # output shape: 120 neurons
    model.add(Dense(120))
    # Dense is just fully connected layers (y = wx +b )
    model.add(Activation("relu"))

    model.add(Dense(84))
    model.add(Activation("relu"))
    # softmax classifier
    model.add(Dense(classes))

    #output 1 class

    # don't activate cuz regression
    # model.add(Activation("softmax"))

    # if weightsPath is specified load the weights
    if weightsPath is not None:
        model.load_weights(weightsPath)

    return model

model = lenet_nvidia(width=320, height=160, depth=3, classes=1)

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10, batch_size=128, verbose=1)
# evaluate the model
scores = model.evaluate(x_valid, y_valid, verbose=1)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
model.save("drive_nvidia2.h5")



#
# def LeNet(x):
#     # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
#     mu = 0
#     sigma = 0.1
#
#     # SOLUTION: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
#     # SOLUTION: Layer 1: Convolutional. Input = 160x320x3. Output = 155x315x6.
#     conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 3, 6), mean = mu, stddev = sigma))
#     conv1_b = tf.Variable(tf.zeros(6))
#     conv1   = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b
#
#     # SOLUTION: Activation.
#     conv1 = tf.nn.relu(conv1)
#
#     # 155x315x6. out:
#     # SOLUTION: Pooling. Input = 28x28x6. Output = 14x14x6.
#     conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
#
#     #
# SOLUTION: Layer 2: Convolutional. Output = 10x10x16.
#     conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean = mu, stddev = sigma))
#     conv2_b = tf.Variable(tf.zeros(16))
#     conv2   = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
#
#     # SOLUTION: Activation.
#     conv2 = tf.nn.relu(conv2)
#
#     # SOLUTION: Pooling. Input = 10x10x16. Output = 5x5x16.
#     conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
#
#     # SOLUTION: Flatten. Input = 5x5x16. Output = 400.
#     fc0   = flatten(conv2)
#
#     # SOLUTION: Layer 3: Fully Connected. Input = 400. Output = 120.
#     fc1_W = tf.Variable(tf.truncated_normal(shape=(45584, 120), mean = mu, stddev = sigma))
#     fc1_b = tf.Variable(tf.zeros(120))
#     fc1   = tf.matmul(fc0, fc1_W) + fc1_b
#
#     # SOLUTION: Activation.
#     fc1    = tf.nn.relu(fc1)
#
#     # SOLUTION: Layer 4: Fully Connected. Input = 120. Output = 84.
#     fc2_W  = tf.Variable(tf.truncated_normal(shape=(120, 84), mean = mu, stddev = sigma))
#     fc2_b  = tf.Variable(tf.zeros(84))
#     fc2    = tf.matmul(fc1, fc2_W) + fc2_b
#
#     # SOLUTION: Activation.
#     fc2    = tf.nn.relu(fc2)
#
#     # SOLUTION: Layer 5: Fully Connected. Input = 84. Output = 10.
#     fc3_W  = tf.Variable(tf.truncated_normal(shape=(84, 1), mean = mu, stddev = sigma))
#     fc3_b  = tf.Variable(tf.zeros(1))
#     logits = tf.matmul(fc2, fc3_W) + fc3_b
#
#     return logits
#
#
# x = tf.placeholder(tf.float32, (None, 160, 320, 3))
# y = tf.placeholder(tf.float32, (None))
#
#
#
# # ## Training Pipeline
# # Create a training pipeline that uses the model to classify MNIST data.
# #
# # You do not need to modify this section.
#
# # In[ ]:
# BATCH_SIZE = 128
# EPOCHS = 10
# rate = 0.001
#
# with tf.name_scope("archiecture"):
#     logits = LeNet(x)
# # loss function
# # with tf.name_scope("cross_entropy"):
# #     cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits)
# #     #tf.summary.scalar("cross entropy", cross_entropy)
# with tf.name_scope("loss"):
#     loss_operation = tf.losses.mean_squared_error(labels=y,predictions=logits)
#
#
# with tf.name_scope("train"):
#     optimizer = tf.train.AdamOptimizer(learning_rate = rate)
#     training_operation = optimizer.minimize(loss_operation)
#
#
# # ## Model Evaluation
# # Evaluate how well the loss and accuracy of the model for a given dataset.
# #
# # You do not need to modify this section.
#
# # In[ ]:
#
# with tf.name_scope("accuracy"):
#     # not tf equal cus we are doing regression problem
#     # correct_prediction = tf.equal(logits, y)
#     accuracy_operation = tf.reduce_mean(tf.cast(loss_operation, tf.float32))
#
# saver = tf.train.Saver()
# # tf.summary.histogram("cross entropy",cross_entropy)
# tf.summary.scalar("reduce mean",loss_operation)
#
# tf.summary.scalar("acc",accuracy_operation)
# def evaluate(X_data, y_data):
#     num_examples = len(X_data)
#     print("examples:",num_examples)
#     total_accuracy = 0
#     sess = tf.get_default_session()
#     for offset in range(0, num_examples, BATCH_SIZE):
#         batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
#         accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
#         total_accuracy += (accuracy * len(batch_x))
#     return total_accuracy / num_examples
#
#
# # ## Train the Model
# # Run the training data through the training pipeline to train the model.
# #
# # Before each epoch, shuffle the training set.
# #
# # After each epoch, measure the loss and accuracy of the validation set.
# #
# # Save the model after training.
# #
# # You do not need to modify this section.
#
# # In[ ]:
#
#
#
# with tf.Session() as sess:
#
#
#     # remember not executed until you hit sess.run
#     merged_summary = tf.summary.merge_all()
#     writer = tf.summary.FileWriter("graphs")
#     writer.add_graph(sess.graph)
#     num_examples = len(x_train)
#     sess.run(tf.global_variables_initializer())
#     print("Training...")
#     for i in range(EPOCHS):
#         x_train, y_train = shuffle(x_train, y_train)
#         j=0
#         for offset in range(0, num_examples, BATCH_SIZE):
#             end = offset + BATCH_SIZE
#             batch_x, batch_y = x_train[offset:end], y_train[offset:end]
#             summary = sess.run(merged_summary, feed_dict={x: batch_x, y: batch_y})
#             training = sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})
#             writer.add_summary(summary,j)
#             j+=1
#
#
#         validation_accuracy = evaluate(x_valid, y_valid)
#         print("EPOCH {} ...".format(i+1))
#         print("Validation Accuracy = {:.3f}".format(validation_accuracy))
#
#     saver.save(sess, './car')
#     print("Model saved")
# #
# model = Sequential()
# model.add(Flatten(input_shape=(160,320,3)))
# model.add(Dense(1))
#
# model.compile(loss='mse',optimizer = 'adam')
# model.fit(x_train,y_train, validation_split=0.2, shuffle=True)
#
# model.save('model.h5')
# print("done")
