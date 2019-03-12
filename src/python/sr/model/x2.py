import tensorflow as tf
import tables
import itertools

from hotpot.superesolution.data.util import AlignSampler

from dxl.learn.model import Model, Stack
from dxl.learn.model.cnns import Conv2D
from dxl.learn.function.losses import mean_square_error
from dxl.learn.model.super_resolution.super_resolution import ResidualBlock


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




def gen():
    for i in itertools.count(0):
        try:
            yield data_file.root.phantom[i][2].reshape(1600,320,320,1), data_file.root.phantom[i][3].reshape(1600,320,320,1)
        except IndexError:
            break


def inference_blk(num_layer ,layer):
    l = []
    for _ in range(num_layer):
        l.append(layer)

    return Stack(name='stack', models=l)


data_file = tables.open_file("/home/qinglong/drssrn.dev/simu_phantom.h5")

ds = tf.data.Dataset.from_generator(
    gen,
    (tf.float32, tf.float32),
    (tf.TensorShape([1600,320,320,1]), tf.TensorShape([1600,320,320,1]))
)

sino_x1, sino_x2 = ds.make_one_shot_iterator().get_next()
# aspr = AlignSampler(sino_x1, sino_x2, [1600,64,64,1])
# low, high = aspr()
low = tf.image.central_crop(sino_x1, 0.2)
high = tf.image.central_crop(sino_x2, 0.2)


ls_residual = SuperResolutionBlock(name = "sr",
                                   num_blocks=5,
                                   kernel_size=(3,3),
                                   filters=32,
                                   boundary_crop=(4,4),
                                   num_inference_block=5)(low)

SRS = Conv2D(name="reshape", kernel_size=(1,1), filters=1)(ls_residual)
# SRS = ls_residual + low


psnr_SRS_high = tf.image.psnr(SRS[100], high[100], max_val=1)
psnr_low_high = tf.image.psnr(low[100], high[100], max_val=1)

loss = mean_square_error(high, SRS)
train_op = tf.train.AdamOptimizer(0.00001).minimize(loss)


tf.summary.scalar("loss", loss)
tf.summary.scalar("psnr_SRS_high", psnr_SRS_high)
tf.summary.scalar("psnr_low_high", psnr_low_high)

tf.summary.image("high", tf.reshape(high[10], [1,64,64,1]))
tf.summary.image("low", tf.reshape(low[10], [1,64,64,1]))
tf.summary.image("SRS", tf.reshape(SRS[10], [1,64,64,1]))
merged_summary = tf.summary.merge_all()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True

sess = tf.Session(config=config)
writer = tf.summary.FileWriter('/home/qinglong/drssrn.dev/summary/run-2018-12-11-3rd', sess.graph)
sess.run(tf.global_variables_initializer())


print("Training2x...")
counter = 0
while True:
    try:
        (_,
         summary,
         loss_temp,
         inference,
         aligned_label) = sess.run([train_op,
                             merged_summary,
                             loss,
                             low,
                             high])
        writer.add_summary(summary, counter)

        counter += 1
        if counter % 2 == 0:
            print(f'Loss after {counter} batch is {loss_temp}')
#             temp_psnr = psnr(inference, aligned_label)
#             show_subplot(interp, inference, aligned_label, psnr=temp_psnr, counter=counter)

    except tf.errors.OutOfRangeError:
        print('Done')
        break



