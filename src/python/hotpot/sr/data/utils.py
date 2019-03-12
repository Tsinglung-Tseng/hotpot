import tensorflow as tf


class DownSampler:
    """
    d = DownSampler(input, down_sample_ratios, batch_size)
    with tf.Session() as sess:
        sess.run(d())
    """
    def __init__(self, input_, down_sample_ratios):
        self.input_ = input_
        self.down_sample_ratios = down_sample_ratios

    @property
    def batch_size(self):
        if self.input_dim == 4:
            return self.input_shape[0]
        if self.input_dim == 3:
            return 1
        else:
            raise ValueError(f'The shape of input {input_shape} is not accepted. Use 3D or 4D tensor.')

    @property
    def input_shape(self):
        return shape_as_list(self.input_)

    @property
    def input_dim(self):
        return len(shape_as_list(self.input_))

    @property
    def output_shape(self):
        return [self.input_shape[0]] + [self.input_shape[x]
                                        // self.down_sample_ratios
                                        for x in range(1, 3)] + [self.input_shape[3]]

    def __call__(self):
        return tf.image.resize_images(self.input_, tf.convert_to_tensor(self.output_shape[1:3], dtype=tf.int32))


class AlignSampler:
    def __init__(self, low, high, target_low_shape):
        self.low = low
        self.high = high
        self.target_low_shape = target_low_shape

    @property
    def low_shape(self):
        return shape_as_list(self.low)

    @property
    def high_shape(self):
        return shape_as_list(self.high)

    @property
    def target_high_shape(self):
        return list(np.multiply(self.target_low_shape, self.scale))

    @property
    def scale(self):
        return [x//y for x, y in zip(self.high_shape, self.low_shape)]

    @property
    def offsets(self):
        offset_low = random_crop_offset(self.low_shape, self.target_low_shape)
        offset_high = np.multiply(self.scale, offset_low)
        return list(offset_low), list(offset_high)

    def __call__(self):
        offset_low, offset_high = self.offsets
        return (tf.slice(self.low, offset_low, self.target_low_shape),
                tf.slice(self.high, offset_high, self.target_high_shape))