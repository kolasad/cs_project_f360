import numpy as np
import tensorflow as tf
import sklearn.datasets
import keras_preprocessing.image as img
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageFilter


TRAIN_PATH = "fruits-360/Training"
TEST_PATH = "fruits-360/Test"
TRAIN_PERCENT = 0.7
PIXEL_DEPTH = 255
CLASSIFIER = "CNN"  # or DNN
IMG_SIZE = 100

LEARNING_RATE = 0.001
BATCH_SIZE = 64
EPOCHS = 2


def plot_history(epochs, y, line, ylabel):
    ep = np.arange(0, epochs)
    for i in range(len(y)):
        plt.plot(ep, y[i], line[i])

    plt.xlabel("EPOCHS")
    plt.ylabel(ylabel)
    plt.show()


def load(path):
    dataset = sklearn.datasets.load_files(path)
    return np.array(dataset["filenames"]), \
           np.array(dataset["target"])


def one_hot(labels, num_classes):
    return np.eye(num_classes, dtype=np.float32)[labels]


def load_images(paths):
    # transform
    for path in paths:
        imageObject = Image.open(path)
        # apply filter
        imageObject.filter(ImageFilter.SHARPEN)
        imageObject.save(path)

    return np.array([img.img_to_array(img.load_img(p)) for p in paths],
                    dtype=np.float32)


def norm(data):
    return data / PIXEL_DEPTH


def batch(data, labels):
    for i in range(0, len(labels), BATCH_SIZE):
        if i + BATCH_SIZE >= len(labels):
            yield data[i:], labels[i:]
        else:
            yield data[i:i + BATCH_SIZE], labels[i:i + BATCH_SIZE]


print("Loading data paths")
train_data, train_labels = load(TRAIN_PATH)
perm = np.random.permutation(len(train_labels))
train_data, train_labels = train_data[perm], train_labels[perm]

test_data, test_labels = load(TEST_PATH)
perm = np.random.permutation((len(test_labels)))
test_data, test_labels = test_data[perm], test_labels[perm]

print("One-hot encoding")
num_classes = len(np.unique(train_labels))
if num_classes < len(np.unique(test_labels)):
    raise Exception("Too many classes in test labels")
train_labels = one_hot(train_labels, num_classes)
test_labels = one_hot(test_labels, num_classes)

print("Loading validation set")
train_size = int(TRAIN_PERCENT * train_labels.shape[0])
valid_data = norm(load_images(train_data[train_size:]))
valid_labels = train_labels[train_size:]
print("Loading training set")
train_data = norm(load_images(train_data[:train_size]))
train_labels = train_labels[:train_size]
print("Loading test set")
test_data = norm(load_images(test_data))

print("Train size {}, Valid size {}, Test size {}"
      .format(train_data.shape, valid_data.shape, test_data.shape))


def cnn(num_classes):
    print("Building CNN")
    graph = tf.Graph()
    with graph.as_default():
        x = tf.placeholder(tf.float32, shape=(None, IMG_SIZE, IMG_SIZE, 3))
        y = tf.placeholder(tf.float32, shape=(None, num_classes))

        def conv(prev, prev_channels, filter_size, num_filters):
            w = tf.Variable(
                tf.truncated_normal(shape=[filter_size,
                                           filter_size,
                                           prev_channels,
                                           num_filters]))
            b = tf.Variable(tf.constant(0.0, shape=[num_filters]))

            conv = tf.nn.conv2d(input=prev,
                                filter=w,
                                strides=[1, 1, 1, 1],
                                padding="SAME")
            conv = tf.nn.relu(conv + b)
            return tf.nn.max_pool(value=conv,
                                  ksize=[1, 2, 2, 1],
                                  strides=[1, 2, 2, 1],
                                  padding="SAME")

        def dense(prev, prev_size, out_size):
            w = tf.Variable(
                tf.truncated_normal(shape=[prev_size, out_size])
            )
            b = tf.Variable(
                tf.constant(0.0, shape=[out_size])
            )
            return tf.matmul(prev, w) + b

        net = conv(x, prev_channels=3, filter_size=5, num_filters=16)
        net = conv(net, prev_channels=16, filter_size=5, num_filters=32)
        dense_size = int(net.shape[1] * net.shape[2] * net.shape[3])
        net = tf.reshape(net, [-1, dense_size])
        net = dense(net, prev_size=dense_size, out_size=num_classes)

        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(logits=net, labels=y)
        )
        accuracy = tf.reduce_mean(tf.cast(tf.equal(
            tf.argmax(tf.nn.softmax(net), axis=1),
            tf.argmax(y, axis=1)
        ), dtype=np.float32))
        global_step = tf.Variable(0)
        optimizer = tf.train.AdamOptimizer(LEARNING_RATE) \
            .minimize(loss, global_step=global_step)

        def batch_run(data, labels):
            valid_loss = 0
            valid_acc = 0
            i = 0

            for batch_data, batch_labels in batch(data, labels):
                feed_dict = {x: batch_data, y: batch_labels}

                v_loss, v_acc = session.run([loss, accuracy], feed_dict=feed_dict)
                valid_loss += v_loss
                valid_acc += v_acc
                i += 1
            valid_acc /= i
            valid_loss /= i
            return valid_loss, valid_acc

        session = tf.Session()
        with session.as_default():
            session.run(tf.global_variables_initializer())
            train_loss_history = []
            train_acc_history = []
            valid_loss_history = []
            valid_acc_history = []
            total = int(np.ceil(len(train_labels) / BATCH_SIZE))
            for ep in range(EPOCHS):
                print("EPOCH: ", ep + 1)
                i = 0
                train_loss = 0
                train_acc = 0

                for batch_data, batch_labels in batch(train_data, train_labels):
                    feed_dict = {x: batch_data, y: batch_labels}

                    _, t_loss, t_acc = session.run([optimizer, loss, accuracy], feed_dict=feed_dict)
                    train_loss += t_loss
                    train_acc += t_acc
                    if (i + 1) % 10 == 0 or i + 1 == total:
                        print("Batch {}/{} loss {}, batch acc {}".format(i + 1, total, t_loss, t_acc))
                    i += 1
                train_acc /= i
                train_loss /= i
                train_loss_history.append(train_loss)
                train_acc_history.append(train_acc)
                print("EPOCH: ", ep + 1)
                print("Train loss {}, train acc {}".format(train_loss, train_acc))
                # VALIDATION
                valid_loss, valid_acc = batch_run(valid_data, valid_labels)
                valid_loss_history.append(valid_loss)
                valid_acc_history.append(valid_acc)
                print("Valid loss {}, valid acc {}".format(valid_loss, valid_acc))
            test_loss, test_acc = batch_run(test_data, test_labels)
            print("Test loss {}, test acc {}".format(test_loss, test_acc))
            plot_history(EPOCHS, [train_loss, valid_loss], ["r--", "b--"], "LOSS")
            plot_history(EPOCHS, [train_acc, valid_acc], ["r:", "b:"], "ACCURACY")


cnn(num_classes)
