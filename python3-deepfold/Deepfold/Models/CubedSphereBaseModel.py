import tensorflow as tf

from .BaseModel import BaseModel


class CubedSphereBaseModel(BaseModel):
    @staticmethod
    def pad_cubed_sphere_grid(tensor, r_padding=(0, 0), xi_padding=(0, 0), eta_padding=(0, 0), name=None):
        assert (xi_padding[0] > 0)
        assert (xi_padding[1] > 0)
        assert (eta_padding[0] > 0)
        assert (eta_padding[1] > 0)

        # Zero pad the tensor in the r dimension
        tensor = tf.pad(tensor, [(0, 0), (0, 0), r_padding, (0, 0), (0, 0), (0, 0)], "CONSTANT")

        # Transpose xi and eta axis
        tensorT = tf.transpose(tensor, [0, 1, 2, 4, 3, 5])

        # Pad xi left (0) and right (1)
        wrap_chunk0 = tf.stack([tensor[:, 3, :, -xi_padding[0]:, :, :],  # Patch 0
                                tensor[:, 0, :, -xi_padding[0]:, :, :],  # Patch 1
                                tensor[:, 1, :, -xi_padding[0]:, :, :],  # Patch 2
                                tensor[:, 2, :, -xi_padding[0]:, :, :],  # Patch 3
                                tf.reverse(tensorT[:, 3, :, -xi_padding[0]:, :, :], axis=[3]),  # Patch 4
                                tf.reverse(tensorT[:, 3, :, :xi_padding[0], :, :], axis=[2])],  # Patch 5
                               axis=1)

        wrap_chunk1 = tf.stack([tensor[:, 1, :, :xi_padding[1], :, :],  # Patch 0
                                tensor[:, 2, :, :xi_padding[1], :, :],  # Patch 1
                                tensor[:, 3, :, :xi_padding[1], :, :],  # Patch 2
                                tensor[:, 0, :, :xi_padding[1], :, :],  # Patch 3
                                tf.reverse(tensorT[:, 1, :, -xi_padding[1]:, :, :], axis=[2]),  # Patch 4
                                tf.reverse(tensorT[:, 1, :, :xi_padding[1], :, :], axis=[3])],  # Patch 5
                               axis=1)

        padded_tensor = tf.concat([wrap_chunk0, tensor, wrap_chunk1], axis=3)

        # Pad eta bottom (0) and top (1)
        wrap_chunk0 = tf.stack([tensor[:, 5, :, :, -eta_padding[0]:, :],  # Patch 0
                                tf.reverse(tensorT[:, 5, :, :, -eta_padding[0]:, :], axis=[2]),  # Patch 1
                                tf.reverse(tensor[:, 5, :, :, :eta_padding[0], :], axis=[2, 3]),  # Patch 2
                                tf.reverse(tensorT[:, 5, :, :, :eta_padding[0], :], axis=[3]),  # Patch 3
                                tensor[:, 0, :, :, -eta_padding[0]:, :],  # Patch 4
                                tf.reverse(tensor[:, 2, :, :, :eta_padding[0], :], axis=[2, 3])],  # Patch 5
                               axis=1)

        wrap_chunk1 = tf.stack([tensor[:, 4, :, :, :eta_padding[1], :],  # Patch 0
                                tf.reverse(tensorT[:, 4, :, :, -eta_padding[1]:, :], axis=[3]),  # Patch 1
                                tf.reverse(tensor[:, 4, :, :, -eta_padding[1]:, :], axis=[2, 3]),  # Patch 2
                                tf.reverse(tensorT[:, 4, :, :, :eta_padding[1], :], axis=[2]),  # Patch 3
                                tf.reverse(tensor[:, 2, :, :, -eta_padding[1]:, :], axis=[2, 3]),  # Patch 4
                                tensor[:, 0, :, :, :eta_padding[1], :]],  # Patch 5
                               axis=1)

        wrap_chunk0_padded = tf.pad(wrap_chunk0, [(0, 0), (0, 0), (0, 0), xi_padding, (0, 0), (0, 0)], "CONSTANT")
        wrap_chunk1_padded = tf.pad(wrap_chunk1, [(0, 0), (0, 0), (0, 0), xi_padding, (0, 0), (0, 0)], "CONSTANT")

        padded_tensor = tf.concat([wrap_chunk0_padded, padded_tensor, wrap_chunk1_padded], axis=4, name=name)

        return padded_tensor

    @staticmethod
    def create_cubed_sphere_conv_layer(index,
                                       input,
                                       ksize_r,
                                       ksize_xi,
                                       ksize_eta,
                                       channels_out,
                                       stride_r=1,
                                       stride_xi=1,
                                       stride_eta=1,
                                       use_r_padding=True):

        # Pad input with periodic image
        if use_r_padding:
            r_padding = (ksize_r / 2, ksize_r / 2)
        else:
            r_padding = (0, 0)

        if r_padding == (0, 0) and ksize_xi==1 and ksize_eta==1:
            padded_input = input
        else:
            padded_input = CubedSphereBaseModel.pad_cubed_sphere_grid(input,
                                                                      r_padding=r_padding,
                                                                      xi_padding=(ksize_xi / 2, ksize_xi / 2),
                                                                      eta_padding=(ksize_eta / 2, ksize_eta / 2))

        filter_shape = [ksize_r, ksize_xi, ksize_eta, padded_input.get_shape().as_list()[-1], channels_out]
        W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W%d" % index)
        b = tf.Variable(tf.truncated_normal(filter_shape[-1:], stddev=0.1), name="b%d" % index)

        convs = []

        for patch in range(padded_input.get_shape().as_list()[1]):
            convs.append(tf.nn.bias_add(
                tf.nn.conv3d(padded_input[:, patch, :, :, :, :],
                             W,
                             strides=[1, stride_r, stride_xi, stride_eta, 1],
                             padding="VALID",
                             name="cubed_sphere_conv%d_p%d" % (index, patch)),
                b))

        conv = tf.stack(convs, axis=1, name="cubed_sphere_conv%d" % (index))

        return {'W': W, 'b': b, 'conv': conv}

    @staticmethod
    def create_cubed_sphere_conv_banded_disjoint_layer(index,
                                                       input,
                                                       ksize_r,
                                                       ksize_xi,
                                                       ksize_eta,
                                                       channels_out,
                                                       stride_r=1,
                                                       stride_xi=1,
                                                       stride_eta=1,
                                                       use_r_padding=True):
        return CubedSphereBaseModel.create_cubed_sphere_conv_banded_layer(index,
                                                                          input,
                                                                          ksize_r=ksize_r,
                                                                          ksize_xi=ksize_xi,
                                                                          ksize_eta=ksize_eta,
                                                                          channels_out=channels_out,
                                                                          kstride_r=1,
                                                                          kstride_xi=stride_xi,
                                                                          kstride_eta=stride_eta,
                                                                          window_size_r=ksize_r,
                                                                          window_stride_r=stride_r,
                                                                          use_r_padding=use_r_padding)

    @staticmethod
    def create_cubed_sphere_conv_banded_layer(index,
                                              input,
                                              ksize_r,
                                              ksize_xi,
                                              ksize_eta,
                                              channels_out,
                                              kstride_r,
                                              kstride_xi,
                                              kstride_eta,
                                              window_size_r,
                                              window_stride_r,
                                              use_r_padding):

        # Pad input with periodic image
        if use_r_padding:
            r_padding = (ksize_r / 2, ksize_r / 2)
        else:
            r_padding = (0, 0)

        padded_input = CubedSphereBaseModel.pad_cubed_sphere_grid(input,
                                                                  r_padding=r_padding,
                                                                  xi_padding=(ksize_xi / 2, ksize_xi / 2),
                                                                  eta_padding=(ksize_eta / 2, ksize_eta / 2))

        # Create convolutions for each r value
        convs_r = []

        for i in range(0, padded_input.shape[2] - ksize_r + 1, window_stride_r):

            padded_input_r_band = padded_input[:, :, i:i+window_size_r, :, :, :]

            filter_shape = [ksize_r, ksize_xi, ksize_eta, padded_input_r_band.get_shape().as_list()[-1], channels_out]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W_%d_r%d" % (index, i))
            b = tf.Variable(tf.truncated_normal(filter_shape[-1:], stddev=0.1), name="b_%d_r%d" % (index, i))

            convs_patches = []

            for patch in range(padded_input_r_band.get_shape().as_list()[1]):
                convs_patches.append(tf.nn.bias_add(
                    tf.nn.conv3d(padded_input_r_band[:, patch, :, :, :, :],
                                 W,
                                 strides=[1, kstride_r, kstride_xi, kstride_eta, 1],
                                 padding="VALID",
                                 name="cubed_sphere_conv_%d_r%d_p%d" % (index, i, patch)),
                    b))

            conv_r = tf.stack(convs_patches, axis=1, name="cubed_sphere_conv_%d_r%d" % (index, i))
            convs_r.append(conv_r)

        conv = tf.concat(convs_r, axis=2, name="cubed_sphere_conv%d" % (index))

        return {'conv': conv}

    @staticmethod
    def create_cubed_sphere_avgpool_layer(index,
                                          input,
                                          ksize_r,
                                          ksize_xi,
                                          ksize_eta,
                                          stride_r,
                                          stride_xi,
                                          stride_eta,
                                          use_r_padding=True):

        # Pad input with periodic image
        if use_r_padding:
            r_padding = (ksize_r / 2, ksize_r / 2)
        else:
            r_padding = (0, 0)

        padded_input = CubedSphereBaseModel.pad_cubed_sphere_grid(input,
                                                                  r_padding=r_padding,
                                                                  xi_padding=(ksize_xi / 2, ksize_xi / 2),
                                                                  eta_padding=(ksize_eta / 2, ksize_eta / 2))

        pools = []

        for patch in range(padded_input.get_shape().as_list()[1]):
            pools.append(tf.nn.avg_pool3d(padded_input[:, patch, :, :, :, :],
                                          ksize=[1, ksize_r, ksize_xi, ksize_eta, 1],
                                          strides=[1, stride_r, stride_xi, stride_eta, 1],
                                          padding='VALID'))

        pool = tf.stack(pools, axis=1, name="cubed_sphere_pool%d" % (index))

        return {'pool': pool}
