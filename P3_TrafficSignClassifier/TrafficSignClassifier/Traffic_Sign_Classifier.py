import matplotlib.pyplot as plt
import pickle
import os
import numpy as np
from sklearn.utils import shuffle
import tensorflow as tf
import cv2
from tensorflow.contrib.layers import flatten

EPOCHS = 100
BATCH_SIZE = 128
GRAY_MODE = False
VISUALS = False

dir_path = os.path.dirname(os.path.abspath(__file__))
data_path = dir_path + "/data"

training_file = data_path + "/train.p"
validation_file = data_path + "/test.p"
testing_file = data_path + "/valid.p"

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

x_train, y_train = train['features'], train['labels']
x_train, y_train= shuffle(x_train, y_train)
x_valid, y_valid = valid['features'], valid['labels']
x_test, y_test = test['features'], test['labels']

# 'features' is a 4D array containing raw pixel data of the traffic sign images, (num examples, width, height, channels).
# 'labels' is a 1D array containing the label/class id of the traffic sign. The file signnames.csv contains id -> name mappings for each id.
# 'sizes' is a list containing tuples, (width, height) representing the original width and height the image.
# 'coords' is a list containing tuples, (x1, y1, x2, y2) representing coordinates of a bounding box around the sign in the image. THESE COORDINATES ASSUME THE ORIGINAL IMAGE. THE PICKLED DATA CONTAINS RESIZED VERSIONS (32 by 32) OF THESE IMAGES


# crucial step in understanding shape of the dataset
print(x_train[0][0][0])
print(x_train[0])
print(x_train[0].shape)
# shape of x_train[0] is 32,32,3 means 1 image
print(len(x_train[0]))

n_train = len(x_train)
n_validation = len(x_valid)
n_test = len(x_test)
image_shape = [x_train.shape[1], x_train.shape[2]]
n_classes = len(np.unique(y_train))

def rgbtogray(rgbdata):
    for i in range(n_train):
        graydata = np.zeros(shape=[n_train,32,32,1])
        # need to have same shape as the network architecture
        graydata[i]=np.reshape(cv2.cvtColor(rgbdata[i], cv2.COLOR_RGB2GRAY),(32,32,1))
        return graydata

# provide option to train only grayscale model
if GRAY_MODE==True:
    x_train = rgbtogray(x_train)
    x_test = rgbtogray(x_test)
    x_valid = rgbtogray(x_valid)

n_channels = x_train.shape[3]

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)
print("Number of color channels = ", n_channels)

print(x_train[0])
print(x_train.shape)

def visualize(data):
    n = 0
    for i in range(25):
        image = data[i] #don't need to reshape
        plt.subplot(5,5,n+1)
        # np.reshape(image, (32,32))
        # assign 25 plots for 25 images
        plt.imshow(image)
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.gca().get_yaxis().set_visible(False)
        n+=1
    plt.tight_layout()
    plt.show()

if VISUALS==True:
    visualize(x_train)

def LeNet(x):
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1

    # Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, n_channels, 6), mean = mu, stddev = sigma))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1   = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b

    # Activation.
    conv1 = tf.nn.relu(conv1)

    # Pooling. Input = 28x28x6. Output = 14x14x6.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Layer 2: Convolutional. Output = 10x10x16.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean = mu, stddev = sigma))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2   = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b

    # Activation.
    conv2 = tf.nn.relu(conv2)

    # Pooling. Input = 10x10x16. Output = 5x5x16.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Flatten. Input = 5x5x16. Output = 400.
    fc0   = flatten(conv2)

    # Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(400, 120), mean = mu, stddev = sigma))
    fc1_b = tf.Variable(tf.zeros(120))
    fc1   = tf.matmul(fc0, fc1_W) + fc1_b

    # Activation.
    fc1    = tf.nn.relu(fc1)

    # Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2_W  = tf.Variable(tf.truncated_normal(shape=(120, 84), mean = mu, stddev = sigma))
    fc2_b  = tf.Variable(tf.zeros(84))
    fc2    = tf.matmul(fc1, fc2_W) + fc2_b

    # Activation.
    fc2    = tf.nn.relu(fc2)

    # Layer 5: Fully Connected. Input = 84. Output = 10.
    fc3_W  = tf.Variable(tf.truncated_normal(shape=(84, n_classes), mean = mu, stddev = sigma))
    fc3_b  = tf.Variable(tf.zeros(n_classes))
    logits = tf.matmul(fc2, fc3_W) + fc3_b

    return logits


x = tf.placeholder(tf.float32, (None, 32, 32, n_channels))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, n_classes)

rate = 0.001

with tf.name_scope("archiecture"):
    logits = LeNet(x)
# loss function
with tf.name_scope("cross_entropy"):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
    #tf.summary.scalar("cross entropy", cross_entropy)
with tf.name_scope("loss"):
    loss_operation = tf.reduce_mean(cross_entropy)
with tf.name_scope("train"):
    optimizer = tf.train.AdamOptimizer(learning_rate = rate)
    training_operation = optimizer.minimize(loss_operation)
with tf.name_scope("accuracy"):
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
    accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver()

# create summaries for visualizing loss using tensorboard
tf.summary.histogram("cross entropy",cross_entropy)
tf.summary.scalar("reduce mean",loss_operation)
tf.summary.scalar("acc",accuracy_operation)

def evaluate(x_data, y_data):
    num_examples = len(x_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = x_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

with tf.Session() as sess:
    # remember not executed until you hit sess.run
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter("graphs")
    writer.add_graph(sess.graph)
    num_examples = len(x_train)
    sess.run(tf.global_variables_initializer())
    print("Training...")
    print()
    for i in range(EPOCHS):
        x_train, y_train = shuffle(x_train, y_train)
        j=0
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = x_train[offset:end], y_train[offset:end]
            summary = sess.run(merged_summary, feed_dict={x: batch_x, y: batch_y})
            training = sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})
            writer.add_summary(summary,j)
            j+=1
        validation_accuracy = evaluate(x_valid, y_valid)
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))

    saver.save(sess, './lenet')
    print("Model saved")
