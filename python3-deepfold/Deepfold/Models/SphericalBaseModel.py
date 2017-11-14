import tensorflow as tf

from .BaseModel import BaseModel
from Deepfold.Utils import tf_pad_wrap


class SphericalBaseModel(BaseModel):
    @staticmethod
    def create_maxpool_layer(index,
                             input,
                             ksize,
                             strides):
        # Pad input with periodic image
        padded_input = tf_pad_wrap.tf_pad_wrap(input, [(0, 0), (0, 0), (0, 0), (ksize[3] / 2, ksize[3] / 2), (0, 0)])

        return {'pool': tf.nn.max_pool3d(padded_input,
                                         ksize=ksize,
                                         strides=strides,
                                         padding='VALID')}

    @staticmethod
    def create_avgpool_layer(index,
                             input,
                             ksize,
                             strides):
        # Pad input with periodic image
        padded_input = tf_pad_wrap.tf_pad_wrap(input, [(0, 0), (0, 0), (0, 0), (ksize[3] / 2, ksize[3] / 2), (0, 0)])

        return {'pool': tf.nn.avg_pool3d(padded_input,
                                         ksize=ksize,
                                         strides=strides,
                                         padding='VALID')}

    @staticmethod
    def create_conv3D_layer(index,
                            input,
                            window_size_r,
                            window_size_theta,
                            window_size_phi,
                            channels_out,
                            stride_r=1,
                            stride_theta=1,
                            stride_phi=1,
                            use_r_padding=True):

        # Pad input with periodic image
        if use_r_padding:
            r_padding = (window_size_r / 2, window_size_r / 2)
        else:
            r_padding = (0, 0)

        input = tf.pad(input,
                       [(0, 0), r_padding, (window_size_theta / 2, window_size_theta / 2),
                        (0, 0), (0, 0)], "CONSTANT")

        # Pad input with periodic image - only in phi
        padded_input = tf_pad_wrap.tf_pad_wrap(input,
                                               [(0, 0), (0, 0), (0, 0), (window_size_phi / 2, window_size_phi / 2),
                                                (0, 0)])

        filter_shape = [window_size_r, window_size_theta, window_size_phi, padded_input.get_shape().as_list()[-1],
                        channels_out]
        W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W_%d" % index)
        b = tf.Variable(tf.truncated_normal(filter_shape[-1:], stddev=0.1), name="bias_%d" % index)
        conv = tf.nn.bias_add(
            tf.nn.conv3d(padded_input,
                         W,
                         strides=[1, stride_r, stride_theta, stride_phi, 1],
                         padding="VALID",
                         name="conv_%d" % (index)),
            b)
        return {'W': W, 'b': b, 'conv': conv}

    @staticmethod
    def create_conv3D_separate_r_layer(index,
                                       input,
                                       ksize_r,
                                       ksize_theta,
                                       ksize_phi,
                                       channels_out,
                                       stride_r=1,
                                       stride_theta=1,
                                       stride_phi=1,
                                       use_r_padding=False):

        # Pad the input perodic in theta and phi
        if use_r_padding:
            r_padding = (ksize_r / 2, ksize_r / 2)
        else:
            r_padding = (0, 0)

        input = tf.pad(input, [(0, 0), r_padding, (ksize_theta / 2, ksize_theta / 2), (0, 0), (0, 0)], "CONSTANT")

        # Create convolutions for each r value
        convs = []

        for i in range(ksize_r / 2, input.shape[1] - ksize_r / 2, stride_r):
            input_fixed_r = input[:, i - ksize_r / 2:i + ksize_r / 2 + 1, :, :, :]

            # Pad input with periodic imag
            padded_input = tf_pad_wrap.tf_pad_wrap(input_fixed_r, [(0, 0), (0, 0), (ksize_theta / 2, ksize_theta / 2),
                                                                   (ksize_phi / 2, ksize_phi / 2), (0, 0)])

            filter_shape = [ksize_r, ksize_theta, ksize_phi, padded_input.get_shape().as_list()[-1], channels_out]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W_%d_%d" % (i, index))
            b = tf.Variable(tf.truncated_normal(filter_shape[-1:], stddev=0.1), name="b%d" % i)
            convs.append(
                tf.nn.bias_add(
                    tf.nn.conv3d(padded_input,
                                 W,
                                 strides=[1, 1, stride_theta, stride_phi, 1],
                                 padding="VALID",
                                 name="conv_%d_%d" % (i, index)),
                    b))
        conv = tf.concat(convs, axis=1)
        return {'W': W, 'b': b, 'conv': conv}

        # @staticmethod
        # def create_spherical_conv_banded_r_layer(index,
        #                                             input,
        #                                             ksize_r,
        #                                             ksize_theta,
        #                                             ksize_phi,
        #                                             channels_out,
        #                                             kstride_r,
        #                                             kstride_theta,
        #                                             kstride_phi,
        #                                             window_size_r,
        #                                             window_stride_r,
        #                                             use_r_padding):

        #     # Pad the input perodic in theta and phi
        #     if use_r_padding:
        #         r_padding = (ksize_r/2, ksize_r/2)
        #     else:
        #         r_padding = (0, 0)

        #     input = tf.pad(input, [(0,0), r_padding, (ksize_theta/2, ksize_theta/2), (0,0), (0,0)], "CONSTANT")

        #     # Create convolutions for each r value
        #     convs_r = []

        #     for i in range(window_size_r/2, input.shape[1]-window_size_r/2, window_stride_r):

        #         filter_shape = [ksize_r, ksize_theta, ksize_phi, padded_input.get_shape().as_list()[-1], channels_out]
        #         W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W_%d_r%d" % (index, i))
        #         b = tf.Variable(tf.truncated_normal(filter_shape[-1:], stddev=0.1), name="b_%d_r%d" % (index, i))

        #         conv_r = tf.nn.bias_add(
        #             tf.nn.conv3d(padded_input,
        #                          W,
        #                          strides=[1, stride_r, stride_theta, stride_phi, 1],
        #                          padding="VALID",
        #                          name="conv_%d"%(index)),
        #             b)
        #         convs_r.append(conv_r)

        #     return {'conv': convs_r} ### HERE IS A BUG!!!
