import tensorflow as tf

from .CartesianBaseModel import CartesianBaseModel


class CartesianHighres(CartesianBaseModel):
    """Cartesian model mimicking the design of cubed sphere model without the seperate r layers"""

    def _init_model(self,
                    x_size_high_res,
                    y_size_high_res,
                    z_size_high_res,
                    channels_high_res,
                    output_size):

        self.output_size = output_size

        self.x_high_res = tf.placeholder(tf.float32, [None, x_size_high_res, y_size_high_res, z_size_high_res, channels_high_res])

        self.y = tf.placeholder(tf.float32, [None, output_size])

        self.dropout_keep_prob = tf.placeholder(tf.float32)

        ### LAYER 0 ###
        self.layers = [{'input': self.x_high_res}]
        self.print_layer(self.layers, -1, 'input')

        ### LAYER 1 ###
        self.layers.append({})
        self.layers[-1].update(self.create_conv3d_layer(len(self.layers)-1,
                                                        self.layers[-2]['input'],
                                                        ksize_x=5,
                                                        ksize_y=5,
                                                        ksize_z=5,
                                                        channels_out=16,
                                                        stride_x=2,
                                                        stride_y=2,
                                                        stride_z=2,
                                                        use_padding=True))
        self.layers[-1]['activation'] = tf.nn.relu(self.layers[-1]['conv'])
        self.layers[-1].update(self.create_avgpool_layer(len(self.layers)-1,
                                                         self.layers[-1]['activation'],
                                                         ksize_x=3,
                                                         ksize_y=3,
                                                         ksize_z=3,
                                                         stride_x=1,
                                                         stride_y=1,
                                                         stride_z=1,
                                                         use_padding=False))
        self.print_layer(self.layers, -1, 'activation')
        self.print_layer(self.layers, -1, 'pool')

        ### LAYER 2 ###
        self.layers.append({})
        self.layers[-1].update(self.create_conv3d_layer(len(self.layers)-1,
                                                        self.layers[-2]['pool'],
                                                        ksize_x=3,
                                                        ksize_y=3,
                                                        ksize_z=3,
                                                        channels_out=32,
                                                        stride_x=1,
                                                        stride_y=1,
                                                        stride_z=1,
                                                        use_padding=False))
        self.layers[-1]['activation'] = tf.nn.relu(self.layers[-1]['conv'])
        self.layers[-1].update(self.create_avgpool_layer(len(self.layers)-1,
                                                         self.layers[-1]['activation'],
                                                         ksize_x=3,
                                                         ksize_y=3,
                                                         ksize_z=3,
                                                         stride_x=2,
                                                         stride_y=2,
                                                         stride_z=2,
                                                         use_padding=False))
        self.print_layer(self.layers, -1, 'activation')
        self.print_layer(self.layers, -1, 'pool')


        ### LAYER 3 ###
        self.layers.append({})
        self.layers[-1].update(self.create_conv3d_layer(len(self.layers)-1,
                                                        self.layers[-2]['pool'],
                                                        ksize_x=3,
                                                        ksize_y=3,
                                                        ksize_z=3,
                                                        channels_out=64,
                                                        stride_x=1,
                                                        stride_y=1,
                                                        stride_z=1,
                                                        use_padding=False))
        self.layers[-1]['activation'] = tf.nn.relu(self.layers[-1]['conv'])
        self.layers[-1].update(self.create_avgpool_layer(len(self.layers)-1,
                                                         self.layers[-1]['activation'],
                                                         ksize_x=3,
                                                         ksize_y=3,
                                                         ksize_z=3,
                                                         stride_x=1,
                                                         stride_y=1,
                                                         stride_z=1,
                                                         use_padding=False))
        self.print_layer(self.layers, -1, 'activation')
        self.print_layer(self.layers, -1, 'pool')


        ### LAYER 4 ###
        self.layers.append({})
        self.layers[-1].update(self.create_conv3d_layer(len(self.layers)-1,
                                                        self.layers[-2]['pool'],
                                                        ksize_x=3,
                                                        ksize_y=3,
                                                        ksize_z=3,
                                                        channels_out=128,
                                                        stride_x=1,
                                                        stride_y=1,
                                                        stride_z=1,
                                                        use_padding=False))
        self.layers[-1]['activation'] = tf.nn.relu(self.layers[-1]['conv'])
        self.layers[-1].update(self.create_avgpool_layer(len(self.layers)-1,
                                                         self.layers[-1]['activation'],
                                                         ksize_x=3,
                                                         ksize_y=3,
                                                         ksize_z=3,
                                                         stride_x=1,
                                                         stride_y=1,
                                                         stride_z=1,
                                                         use_padding=True))
        self.print_layer(self.layers, -1, 'activation')
        self.print_layer(self.layers, -1, 'pool')

        ### LAYER 5 ###
        self.layers.append({})
        self.layers[-1].update(self.create_dense_layer(len(self.layers)-1,
                                                       self.layers[-2]['pool'],
                                                       output_size=2048))
        self.layers[-1]['activation'] = tf.nn.relu(self.layers[-1]['dense'])
        self.layers[-1]['dropout'] = tf.nn.dropout(self.layers[-1]['activation'], self.dropout_keep_prob)
        self.print_layer(self.layers, -1, 'W')
        self.print_layer(self.layers, -1, 'activation')

        ### LAYER 6 ###
        self.layers.append({})
        self.layers[-1].update(self.create_dense_layer(len(self.layers)-1,
                                                       self.layers[-2]['dropout'],
                                                       output_size=-1))
        self.layers[-1]['activation'] = tf.nn.relu(self.layers[-1]['dense'])
        self.layers[-1]['dropout'] = tf.nn.dropout(self.layers[-1]['activation'], self.dropout_keep_prob)
        self.print_layer(self.layers, -1, 'W')
        self.print_layer(self.layers, -1, 'activation')

        ### LAYER 7 ###
        self.layers.append({})
        self.layers[-1].update(self.create_dense_layer(len(self.layers)-1,
                                                       self.layers[-2]['dropout'],
                                                       output_size=output_size))
        self.layers[-1]['activation'] = tf.nn.softmax(self.layers[-1]['dense'])
        self.print_layer(self.layers, -1, 'W')
        self.print_layer(self.layers, -1, 'activation')
