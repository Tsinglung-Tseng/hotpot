from dxl.learn.model import Model, Stack
from dxl.learn.model.cnns import Conv2D
from dxl.learn.function.activation import relu
import tensorflow as tf


# def inference_blk(num_layer ,layer):
#     l = []
#     for _ in range(num_layer):
#         l.append(layer)
#
#     return Stack(name='stack', models=l)


class ResidualBlock(Model):
    class KEYS(Model.KEYS):
        class CONFIG:
            FILTERS = 'filters'
            KERNEL_SIZE = 'kernel_size'
            STRIDES = 'strides'
            PADDING = 'padding'

    def __init__(self, name, filters=None, strides=None, padding=None):
        super().__init__(name)
        self.config.update(self.KEYS.CONFIG.FILTERS, filters)
        self.config.update_value_and_default(self.KEYS.CONFIG.STRIDES, strides, (1, 1))
        self.config.update_value_and_default(self.KEYS.CONFIG.PADDING, padding, 'same')
        self.model = None

    def build(self, x):
        if isinstance(x, tf.Tensor):

            def model(x):
                x_in = x

                x = relu(x_in)

                x_a = Conv2D(name="resi_blk_xa", filters=self.config[self.KEYS.CONFIG.FILTERS], kernel_size=(1, 1))(x)

                x_b = Conv2D(name="resi_blk_xb0", filters=self.config[self.KEYS.CONFIG.FILTERS], kernel_size=(1, 1))(x)
                x_b = relu(x_b)
                x_b = Conv2D(name="resi_blk_xb1", filters=self.config[self.KEYS.CONFIG.FILTERS], kernel_size=(3, 3))(x_b)

                x_c = Conv2D(name="resi_blk_xc0", filters=self.config[self.KEYS.CONFIG.FILTERS], kernel_size=(1, 1))(x)
                x_c = relu(x_c)
                x_c = Conv2D(name="resi_blk_xc1", filters=self.config[self.KEYS.CONFIG.FILTERS], kernel_size=(3, 3))(x_c)
                x_c = relu(x_c)
                x_c = Conv2D(name="resi_blk_xc2", filters=self.config[self.KEYS.CONFIG.FILTERS], kernel_size=(3, 3))(x_c)

                x = tf.concat([x_a, x_b, x_c], axis=-1)
                x = relu(x)
                x = Conv2D(name="resi_concat", filters=self.config[self.KEYS.CONFIG.FILTERS], kernel_size=(1, 1))(x)

                out = x_in + x * 0.3
                return out
            self.model = model
            return
        raise TypeError(f"Not support tensor type: {type(x)}.")

    def kernel(self, x_in):
        return self.model(x_in)

    @property
    def parameters(self):
        return self.model.weights


class SuperResolutionBlock(Model):
    class KEYS(Model.KEYS):
        class CONFIG:
            FILTERS = 'filters'
            KERNEL_SIZE = 'kernel_size'
            NUM_BLOCKS = 'num_blocks'
            BOUNDARY_CROP = 'boundary_crop'
            NUM_INFER_BLK = 'num_inference_blocks'

    def __init__(self, name, num_blocks=None, kernel_size=None, filters=None, boundary_crop=None,
                 num_inference_block=None):
        super().__init__(name)
        self.config.update(self.KEYS.CONFIG.FILTERS, filters)
        self.config.update(self.KEYS.CONFIG.KERNEL_SIZE, kernel_size)
        self.config.update(self.KEYS.CONFIG.NUM_BLOCKS, num_blocks)
        self.config.update(self.KEYS.CONFIG.BOUNDARY_CROP, boundary_crop)
        self.config.update(self.KEYS.CONFIG.NUM_INFER_BLK, num_inference_block)

    def _inference_blk(self, num_inference_blocks, layer):
        l = []
        for _ in range(num_inference_blocks):
            l.append(layer)

        return Stack(name='inference_blk', models=l)

    def _short_cut(self, name):
        conv2d_ins = Conv2D(name=name,
                            filters=self.config[self.KEYS.CONFIG.FILTERS],
                            kernel_size=self.config[self.KEYS.CONFIG.KERNEL_SIZE])

        return Stack(name='short_cut',
                     models=[conv2d_ins,
                             self._inference_blk(self.config[self.KEYS.CONFIG.NUM_INFER_BLK],
                                                 ResidualBlock(name='res_blk',
                                                               filters=self.config[self.KEYS.CONFIG.FILTERS]))])

    @property
    def parameters(self):
        return parameters(self.models)

    def kernel(self, x):
        return self.model(x)

    def build(self, x):
        if isinstance(x, tf.Tensor):
            self.model = self._short_cut(name="sr")

def inference_blk(num_layer ,layer):
    l = []
    for _ in range(num_layer):
        l.append(layer)

    return Stack(name='stack', models=l)
