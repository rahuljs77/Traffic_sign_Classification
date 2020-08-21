from sklearn.model_selection import train_test_split
from tensorflow.contrib.layers import flatten
from sklearn.utils import shuffle
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pickle
import random
import glob
import csv
import cv2
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

training_file = 'traffic_signs/train.p'
validation_file = 'traffic_signs/valid.p'
testing_file = 'traffic_signs/test.p'
print()
print('Loading Data...')
with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

# print(y_train[10])
print("Data Loading Successful!")

# index = np.random.randint(0, len(X_train))
# image = X_train[index]
# plt.imshow(image)
# plt.show()


n_train = len(X_train)
n_validation = len(X_valid)
n_test = len(X_test)

image_shape = X_train[0].shape
n_classes = max(y_train) + 1
print()
print("--------------DATA SUMMARY-------------")
print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)
print("---------------------------------------")
print()

#  DATA VISUALIZATION

all_labels = []
with open('signnames.csv', 'r') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        all_labels += [row[1]]

rows = 11
columns = 4
labels = []
for i in range(0, len(X_train)):
    image = (X_train[i])
    label = y_train[i]
    if label in labels:
        pass
    else:
        labels.append(label)
        # plt.imshow(image)
        # plt.title(all_labels[label + 1])
        # plt.show()

# END OF DATA VISUALIZATION

print('Normalizing Images...')
def gray_image(image_set):
    new_shape = image_shape[0:2] + (1,)

    prep_image_set = np.empty(shape=(len(image_set),) + new_shape, dtype=int)

    for ind in range(0, len(image_set)):
        norm_img = cv2.normalize(image_set[ind], np.zeros(image_shape[0:2]), 0, 255, cv2.NORM_MINMAX)
        norm_img = norm_img.astype(np.float32)

        # grayscale
        gray_img = cv2.cvtColor(norm_img, cv2.COLOR_BGR2GRAY)
        prep_image_set[ind] = np.reshape(gray_img, new_shape)

    return prep_image_set


X_train = gray_image(X_train)
X_valid = gray_image(X_valid)
X_test = gray_image(X_test)
print('Normalizing Done!')
print()

################################# DEEP LEARNING ARCHITECTURE #############################

BATCH_SIZE = 128
EPOCHS = 20
keep_prob = tf.placeholder(tf.float32, name='keep_prob')


def LeNet(X):
    #     keep_prob = tf.placeholder(tf.float32)
    mu = 0
    sig = 0.1
    # Convoluted layer 1
    cn1_w = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 12), mean=mu, stddev=sig))
    cn1_b = tf.Variable(tf.zeros(12))
    cn1 = tf.nn.conv2d(X, cn1_w, strides=[1, 1, 1, 1], padding='VALID') + cn1_b
    # Activation layer
    cn1 = tf.nn.relu(cn1)
    # Maxpool
    cn1 = tf.nn.max_pool(cn1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    cn2_w = tf.Variable(tf.truncated_normal(shape=(5, 5, 12, 25), mean=mu, stddev=sig))
    cn2_b = tf.Variable(tf.zeros(25))
    cn2 = tf.nn.conv2d(cn1, cn2_w, strides=[1, 1, 1, 1], padding='VALID') + cn2_b

    cn2 = tf.nn.relu(cn2)

    cn2 = tf.nn.max_pool(cn2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    f0 = flatten(cn2)
    f0 = tf.nn.dropout(f0, keep_prob)
    dn1_w = tf.Variable(tf.truncated_normal(shape=(625, 300), mean=mu, stddev=sig))
    dn1_b = tf.Variable(tf.zeros(300))
    dn1 = tf.matmul(f0, dn1_w) + dn1_b

    dn1 = tf.nn.relu(dn1)

    dn2_w = tf.Variable(tf.truncated_normal(shape=(300, 100), mean=mu, stddev=sig))
    dn2_b = tf.Variable(tf.zeros(100))
    dn2 = tf.matmul(dn1, dn2_w) + dn2_b

    dn2 = tf.nn.relu(dn2)

    dn3_w = tf.Variable(tf.truncated_normal(shape=(100, 43), mean=mu, stddev=sig))
    dn3_b = tf.Variable(tf.zeros(43))
    dn3 = tf.matmul(dn2, dn3_w) + dn3_b

    logits = dn3

    return logits

##################### TRAINING MODEL ###########################


x = tf.placeholder(tf.float32, (None, 32, 32, 1), name='x')   # name your placeholder
y = tf.placeholder(tf.int32, (None), name='y')
one_hot_y = tf.one_hot(y, 43)

rate = 0.001
logits = LeNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=rate)
training_operation = optimizer.minimize(loss_operation)

################ EVALUATION #############################

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()


def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, keep_prob : 1.0})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples


X_train, X_tst, y_train, y_tst = train_test_split(X_train, y_train, test_size=0.2, random_state=0)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)

    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 0.8})

        validation_accuracy = evaluate(X_tst, y_tst)
        print("EPOCH {} ...".format(i + 1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()

    saver.save(sess, './lenet')
    print("Model saved")

################ Testing Five images #########################

# test_set = []
# test_images = glob.glob("test images/test*.jpeg")
# for fname in test_images:
#     image = mpimg.imread(fname)
#     image = cv2.resize(image, (32, 32))
#     plt.imshow(image)
#     plt.show()
#     test_set.append(image)
# print(len(test_set))

# test_set = gray_image(test_set)
# labels = []
# with tf.Session() as sess:
#     saver.restore(sess, tf.train.latest_checkpoint('.'))
#     logits_predicted = sess.run(logits, feed_dict={x: test_set, keep_prob: 1.0})
#     print(np.argmax(logits_predicted, axis = 1))

# for i in labels:
#     print('The traffic sign is: ', y[i + 1])














