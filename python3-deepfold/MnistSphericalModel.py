import tensorflow as tf

from MnistBaseModel import MnistBaseModel
from Deepfold.Models.BaseModel import BaseModel
from Deepfold.Models.SphericalBaseModel import SphericalBaseModel


class MnistSphericalModel(MnistBaseModel):

    def _init_model(self,
                    r_shape,
                    theta_shape,
                    phi_shape,
                    output_size):

        self.output_size = output_size

        # Setup model
        self.X = tf.placeholder(tf.float32, [None, r_shape, theta_shape, phi_shape, 1])
        self.y = tf.placeholder(tf.float32, [None, output_size])

        self.dropout_keep_prob = tf.placeholder(tf.float32)

        ### LAYER 0 ###
        self.layers = [{'input': self.X}]
        self.print_layer(self.layers, -1, 'input')
        
        ### LAYER 1 ###
        self.layers.append({})
        self.layers[-1].update(SphericalBaseModel.create_conv3D_layer(len(self.layers)-1,
                                                                      self.layers[-2]['input'],
                                                                      window_size_r=1,
                                                                      window_size_theta=3,
                                                                      window_size_phi=3,
                                                                      channels_out=32,
                                                                      stride_r=1,
                                                                      stride_theta=1,
                                                                      stride_phi=1,
                                                                      use_r_padding=False))
        self.layers[-1]['activation'] = tf.nn.relu(self.layers[-1]['conv'])
        self.print_layer(self.layers, -1, 'conv')
        self.print_layer(self.layers, -1, 'activation')

        ### LAYER 2 ###
        self.layers.append({})
        self.layers[-1].update(SphericalBaseModel.create_conv3D_layer(len(self.layers)-1,
                                                                      self.layers[-2]['activation'],
                                                                      window_size_r=1,
                                                                      window_size_theta=3,
                                                                      window_size_phi=3,
                                                                      channels_out=4,
                                                                      stride_r=1,
                                                                      stride_theta=1,
                                                                      stride_phi=1,
                                                                      use_r_padding=False))
        self.layers[-1]['activation'] = tf.nn.relu(self.layers[-1]['conv'])
        self.layers[-1].update(SphericalBaseModel.create_avgpool_layer(len(self.layers) - 1,
                                                                       self.layers[-1]['activation'],
                                                                       ksize=[1, 1, 2, 2, 1],
                                                                       strides=[1, 1, 2, 2, 1]))
        self.print_layer(self.layers, -1, 'conv')
        self.print_layer(self.layers, -1, 'activation')
        self.print_layer(self.layers, -1, 'pool')


        ### LAYER 3 ###
        self.layers.append({})
        self.layers[-1].update(BaseModel.create_dense_layer(len(self.layers) - 1,
                                                            self.layers[-2]['pool'],
                                                            output_size=128))
        self.layers[-1]['activation'] = tf.nn.relu(self.layers[-1]['dense'])
        self.layers[-1]['dropout'] = tf.nn.dropout(self.layers[-1]['activation'], self.dropout_keep_prob)
        self.print_layer(self.layers, -1, 'W')
        self.print_layer(self.layers, -1, 'activation')

        ### LAYER 4 ###
        self.layers.append({})
        self.layers[-1].update(BaseModel.create_dense_layer(len(self.layers) - 1,
                                                            self.layers[-2]['dropout'],
                                                            output_size=output_size))
        self.layers[-1]['activation'] = tf.nn.softmax(self.layers[-1]['dense'])
        self.print_layer(self.layers, -1, 'W')
        self.print_layer(self.layers, -1, 'activation')
