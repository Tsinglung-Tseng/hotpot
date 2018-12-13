import tensorflow as tf
import tables
import matplotlib.pyplot as plt
from dxl.learn.model.cnns import Conv2D
import itertools
import functools
import numpy as np


BATCH_SIZE = 32
F = tables.open_file("/home/qinglong/flickr256Data/25k22000train")

x = F.root.train['bicubic']
y_true = F.root.train['img']

x_test = F.root.test['bicubic']
y_true_test = F.root.test['img']


x_ph = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE,256,256,3])
y_true_ph = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE,256,256,3])


class DataIterFactory:
    class _DataGen(object):
        def __init__(self, data_gen):
            self.data_gen = data_gen

        def __call__(self):
            return next(self.data_gen)

    def _gen(self):
        for i in itertools.count(self.batch_size, self.batch_size):
            try:
                if self.x_data[i - self.batch_size:i].shape[0] == self.batch_size:
                    yield (self.x_data[i - self.batch_size:i], self.y_data[i - self.batch_size:i])
                else:
                    break
            except IndexError:
                print("Run out of data...")

    def __init__(self, x_data, y_data, batch_size, num_epochs=1):
        self.x_data = x_data
        self.y_data = y_data
        self.batch_size = batch_size
        self.num_epochs = num_epochs

    @property
    def x_data_shape(self):
        return [self.batch_size, self.x_data.shape[0], self.x_data.shape[1], self.x_data.shape[2]]

    @property
    def y_data_shape(self):
        return [self.batch_size, self.y_data.shape[0], self.y_data.shape[1], self.y_data.shape[2]]

    def __call__(self):
        gen = self._gen()
        data_gen = self._DataGen(data_gen=gen)

        data_set_train = (tf.data.Dataset
                         .from_generator(generator=data_gen,
                                         output_types=(tf.int64, tf.int64),  #TODO get data type out of here
                                         output_shapes=(self.x_data_shape,
                                                        self.y_data_shape))
                         .repeat(self.num_epochs))
        iter_train = data_set_train.make_initializable_iterator()
        return iter_train



        # def gen(batch_size, x, y_true):
#     for i in itertools.count(batch_size, batch_size):
#         #         print(f"Mini-batch {i} under going")
#         try:
#             if x[i - batch_size:i].shape[0] == BATCH_SIZE:
#                 yield (x[i - batch_size:i], y_true[i - batch_size:i])
#             else:
#                 break
#         except IndexError:
#             print("Run out of data...")






gen_with_batch = functools.partial(gen, BATCH_SIZE)

data_gen = gen_with_batch(x, y_true)
validate_data_gen = gen_with_batch(x_test, y_true_test)


def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))


def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))


def new_conv_layer(input,  # The previous layer.
                   num_input_channels,  # Num. channels in prev. layer.
                   filter_size,  # Width and height of each filter.
                   num_filters,  # Number of filters.
                   use_pooling=False):  # Use 2x2 max-pooling.

    shape = [filter_size, filter_size, num_input_channels, num_filters]

    weights = new_weights(shape=shape)
    biases = new_biases(length=num_filters)

    layer = tf.nn.conv2d(input=input,
                         filter=weights,
                         strides=[1, 1, 1, 1],
                         padding='SAME')

    layer += biases

    if use_pooling:
        layer = tf.nn.max_pool(value=layer,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME')

    layer = tf.nn.relu(layer)

    return layer, weights


def get_infer(x, i):
    infer = (sess.run(layer_conv3,feed_dict={x_ph: x[(i//BATCH_SIZE)*BATCH_SIZE: (i//BATCH_SIZE+1)*BATCH_SIZE]}))[i%BATCH_SIZE,:,:,:]
    infer = infer/np.max(infer)*255
    return infer.astype(np.int64)


get_train_infer = functools.partial(get_infer, x)
get_test_infer = functools.partial(get_infer, x_test)


def get_psnr(get_infer, x, y, i):
    psnr_infer = tf.image.psnr(get_infer(i), y[i], max_val=255)
    psnr_interpolate = tf.image.psnr(x[i], y[i], max_val=255)
    return sess.run(psnr_infer), sess.run(psnr_interpolate)


get_train_psnr = functools.partial(get_psnr, get_train_infer, x, y_true)
get_test_psnr = functools.partial(get_psnr, get_test_infer, x_test, y_true_test)


def aligned_show(_get_psnr, _get_infer, x, y, i):
    fig, axes = plt.subplots(1, 3, figsize=(15, 15))

    psnr = _get_psnr(i)

    axes[0].imshow(x[i])
    axes[0].set_title(f"Interpolate\n  {psnr[1]}")

    axes[1].imshow(np.reshape(_get_infer(i), [256, 256, 3]))
    axes[1].set_title(f"Inference\n  {psnr[0]}")

    axes[2].imshow(y[i])
    axes[2].set_title("GroundTruth")

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])


aligned_show_train = functools.partial(aligned_show, get_train_psnr, get_train_infer, x, y_true)
aligned_show_test = functools.partial(aligned_show, get_test_psnr, get_test_infer, x_test, y_true_test)


def feeded_avg():
    tmp = []
    def avg(i):
        tmp.append(i)
        return np.mean(tmp)
    return avg


def avg_interpolate_psnr(avger):
    tmp = []
    for i in range(len(x_test)):
        infer = get_infer(i, x_test)
        psnr = sess.run(tf.image.psnr(infer, y_true_test[i], max_val=255))
        tmp.append(avger(psnr))
    return tmp


layer_conv1, weights_conv1 = new_conv_layer(input=x_ph, num_input_channels=3, filter_size=5, num_filters=64)
layer_conv2, weights_conv2 = new_conv_layer(input=layer_conv1, num_input_channels=64, filter_size=1, num_filters=16)
layer_conv3, weights_conv3 = new_conv_layer(input=layer_conv2, num_input_channels=16, filter_size=5, num_filters=3)

mse = tf.losses.mean_squared_error(labels=y_true_ph, predictions=layer_conv3)
cost = tf.reduce_mean(mse)
optimizer = tf.train.AdamOptimizer(learning_rate=0.005).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())


def optimize(num_batches):
    for _ in range(num_batches):
        train_input, train_label = next(data_gen)
        feed_dict_train = {x_ph: train_input, y_true_ph: train_label}
        sess.run(optimizer, feed_dict = feed_dict_train)


y_true_test_ph = tf.placeholder(dtype=tf.int32, shape=[32, 256, 256, 3])
infer_psnr = tf.image.psnr(layer_conv3, y_true_test_ph, max_val=255)


def validation(num_batches):
    psnr_buffer = []
    for _ in range(num_batches):
        test_input, test_label = next(validate_data_gen)
        feed_dict_train = {x_ph: test_input, y_true_ph: test_label, y_true_test_ph: test_label}
        psnr_buffer.append(sess.run(infer_psnr, feed_dict=feed_dict_train))
    return psnr_buffer