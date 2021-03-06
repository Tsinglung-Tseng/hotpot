# from dxl.learn.function.crop import random_crop_offset, random_crop, boundary_crop, align_crop, shape_as_list
import tensorflow as tf
# import numpy as np
import itertools
# import tables
from typing import Iterable
# import math


# class DataIterFactory:
#     class _DataGen(object):
#         def __init__(self, data_gen):
#             self.data_gen = data_gen
#
#         def __call__(self):
#             return next(self.data_gen)
#
#     def _gen(self):
#         for i in itertools.count(0):
#             try:
#                 if self.x_data[i - self.batch_size:i].shape[0] == self.batch_size:
#                     yield (self.x_data[i], self.y_data[i])
#                 else:
#                     break
#             except IndexError:
#                 print("Run out of data...")
#
#     def __init__(self, x_data, y_data, batch_size, num_epochs=1):
#         self.x_data = x_data
#         self.y_data = y_data
#         self.batch_size = batch_size
#         self.num_epochs = num_epochs
#         gen = self._gen
#         data_gen = self._DataGen(data_gen=gen)
#         self.next_batch = (tf.data.Dataset
#                            .from_generator(generator=data_gen,
#                                            output_types=(tf.int64, tf.int64),  #TODO get data type out of here
#                                            output_shapes=(self.x_data_shape,
#                                                           self.y_data_shape))
#                            .batch(self.batch_size)
#                            .repeat(self.num_epochs)
#                            .make_one_shot_iterator()
#                            .get_next())
#
#     @property
#     def x(self):
#         return tf.reshape(self.next_batch[0], [self.batch_size, 256, 256, 3])
#
#     @property
#     def y(self):
#         return tf.reshape(self.next_batch[1], [self.batch_size, 256, 256, 3])
#
#     @property
#     def x_data_shape(self):
#         return [self.x_data.shape[1], self.x_data.shape[2], self.x_data.shape[3]]
#
#     @property
#     def y_data_shape(self):
#         return [self.y_data.shape[1], self.y_data.shape[2], self.y_data.shape[3]]



    # def __call__(self):
    #     gen = self._gen
    #     data_gen = self._DataGen(data_gen=gen)
    #
    #     return
    #
    #     # iter_train = data_set_train.make_one_shot_iterator()
    #     # return iter_train


# class DataGen:
#     """
#     A data generator for phantom and sinogram dataset.
#     g = DataGen(file)
#
#     with tf.Session() as sess:
#         sess.run(g.next_batch)
#     """
#     def __init__(self, source:Iterable[tables.array.Array], batch_size=32, buffer_size=1):
#         self.source = source
#         self.batch_size = batch_size
#         self.next_batch = (
#             tf.data.Dataset
#             .from_generator(self._gen,
#                             (tf.int64, tf.int64),
#                             (tf.TensorShape(self.output_shape),
#                              tf.TensorShape(self.output_shape))
#                             .shuffle(buffer_size=buffer_size)
#                             .batch(batch_size)
#                             .make_one_shot_iterator()
#                             .get_next()
#
#     def _gen(self):
#         for i in itertools.count(0):
#             try:
#                 tmp = []
#                 for s in self.source[i]:
#                     tmp.append(s)
#                 yield tmp
#
#
#                 #
#                 #
#                 # yield (self._file.root.phantom[i][0],
#                 #        self._file.root.phantom[i][1],
#                 #        self._file.root.phantom[i][2],
#                 #        self._file.root.phantom[i][3])
#             except IndexError:
#                 break
#
#     @property
#     def source_shape(self):
#         return self.source[0].shape[1:]
#
#     @property
#     def output_shape(self):
#         tmp = list(self.source_shape)
#         return tmp.insert(0, self.batch_size)
#
#     @property
#     def x(self):
#         return tf.reshape(self.next_batch[0], self.output_shape)
#
#     @property
#     def y(self):
#         return tf.reshape(self.next_batch[1], self.output_shape)


# def show():
#     pass
#
#
# def psnr(inference, label, pix_max=255.0, idx=0):
#     mse = np.mean((rescale_single(inference[idx]) - rescale_single(label[idx])) ** 2)
#     if mse == 0:
#         return 100
#     return 20 * math.log10(pix_max / math.sqrt(mse))
#
#
# def rescale_single(inputs, bin_size=255):
#     window = inputs.max() - inputs.min()
#     scale_rate = bin_size / window
#     return inputs*scale_rate - inputs.min()*scale_rate
#
#
# def rescale_batch(images):
#     return np.array([rescale_single(img) for img in images])
