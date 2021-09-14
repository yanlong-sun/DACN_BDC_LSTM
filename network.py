import tensorflow as tf
import ops


class UNet(object):
    def __init__(self, sess, conf, is_train):
        self.sess = sess
        self.conf = conf
        self.is_train = is_train

        self.axis = (2, 3)
        self.channel_axis = 4
        self.input_shape = [conf.batch, conf.depth, conf.height, conf.width, conf.channel]
        self.output_shape = [conf.batch, conf.height, conf.width]
        self.start_slice = conf.start_slice_num

    def inference(self, inputs):

        print('-------------------- LSTM UNet --------------------')

        # input
        outputs = inputs
        print('input:   ', outputs.get_shape())

        # down sampling 1
        outputs = ops.convlstm2d(outputs, filters=32, kernel_size=3)
        down1 = ops.convlstm2d(outputs, 32, 3)
        outputs = ops.max_pool_3d(down1)
        print('down_sampling1:  ', outputs.get_shape())

        # down sampling 2
        outputs = ops.convlstm2d(outputs, 64, 3)
        down2 = ops.convlstm2d(outputs, 64, 3)
        outputs = ops.max_pool_3d(down2)
        print('down_sampling2:  ', outputs.get_shape())

        # down sampling 3
        outputs = ops.convlstm2d(outputs, 128, 3)
        down3 = ops.convlstm2d(outputs, 128, 3)
        outputs = ops.max_pool_3d(down3)
        print('down_sampling3:  ', outputs.get_shape())

        # down sampling 4
        outputs = ops.convlstm2d(outputs, 256, 3)
        down4 = ops.convlstm2d(outputs, 256, 3)
        outputs = ops.max_pool_3d(down4)
        print('down_sampling4:  ', outputs.get_shape())

        # down sampling 5
        outputs = ops.convlstm2d(outputs, 512, 3)
        outputs = ops.convlstm2d(outputs, 512, 3)
        print('down_sampling5:  ', outputs.get_shape())

        # up sampling 1
        outputs = ops.upsamping3d(outputs)
        up1 = ops.convlstm2d(outputs, 256, 3)
        outputs = up1 + down4
        outputs = ops.convlstm2d(outputs, 256, 3)
        print('up_sampling1:    ', outputs.get_shape())

        # up sampling 2
        outputs = ops.upsamping3d(outputs)
        up2 = ops.convlstm2d(outputs, 128, 3)
        outputs = up2 + down3
        outputs = ops.convlstm2d(outputs, 128, 3)
        print('up_sampling2:    ', outputs.get_shape())

        # up sampling 3
        outputs = ops.upsamping3d(outputs)
        up3 = ops.convlstm2d(outputs, 64, 3)
        outputs = up3 + down2
        outputs = ops.convlstm2d(outputs, 64, 3)
        print('up_sampling3:    ', outputs.get_shape())

        # up sampling 4
        outputs = ops.upsamping3d(outputs)
        up4 = ops.convlstm2d(outputs, 32, 3)
        outputs = up4 + down1
        outputs = ops.convlstm2d(outputs, 32, 3)
        print('up_sampling4:    ', outputs.get_shape())

        # outputs

        branch1 = ops.convlstm2d(outputs, 1, 3, return_sequences=False)
        branch2 = ops.convlstm2d(outputs, 1, 3, return_sequences=False)
        branch3 = ops.convlstm2d(outputs, 2, 3, return_sequences=False)
        outputs = tf.concat([branch1, branch2, branch3], 3)
        print('outputs_shape:   ', outputs.get_shape())

        return outputs
