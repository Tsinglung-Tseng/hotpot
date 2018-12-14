import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import functools
# import tables


def get_infer(batch_size, x, i):
    infer = (sess.run(layer_conv3,
                      feed_dict={x_ph: x[(i//batch_size)*batch_size: (i//batch_size+1)*batch_size]}))[i%batch_size,:,:,:]
    infer = infer/np.max(infer)*255
    return infer.astype(np.int64)


get_train_infer = functools.partial(get_infer, x_train)
get_test_infer = functools.partial(get_infer, x_test)


def get_psnr(get_infer, x, y, i):
    psnr_infer = tf.image.psnr(get_infer(i), y[i], max_val=255)
    psnr_interpolate = tf.image.psnr(x[i], y[i], max_val=255)
    return sess.run(psnr_infer), sess.run(psnr_interpolate)


get_train_psnr = functools.partial(get_psnr, get_train_infer, x_train, y_train)
get_test_psnr = functools.partial(get_psnr, get_test_infer, x_test, y_test)


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


aligned_show_train = functools.partial(aligned_show, get_train_psnr, get_train_infer, x_train, y_train)
aligned_show_test = functools.partial(aligned_show, get_test_psnr, get_test_infer, x_test, y_test)


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


# if __name__=="__main__":
#     BATCH_SIZE = 32
#     DEFAULT_DATA_FILE = "/home/qinglong/flickr256Data/25k22000train"
#
#     F = tables.open_file(DEFAULT_DATA_FILE)
#
#     x_train = F.root.config['bicubic']
#     y_train = F.root.config['img']
#
#     x_test = F.root.test['bicubic']
#     y_test = F.root.test['img']
#
#     train_iter = DataIterFactory(x_data=x_train, y_data=y_train, batch_size=32, num_epochs=1)
#     test_iter = DataIterFactory(x_data=x_test, y_data=y_test, batch_size=32, num_epochs=1)