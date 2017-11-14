import tensorflow as tf

from MnistBaseModel import MnistBaseModel
from Deepfold.Models.BaseModel import BaseModel
from Deepfold.Models.CubedSphereBandedModel import CubedSphereBaseModel


class MnistCubedSphereModel(MnistBaseModel):

    def _init_model(self,
                    patch_shape,
                    r_shape,
                    xi_shape,
                    eta_shape,
                    output_size):

        self.output_size = output_size

        # Setup model
        self.X = tf.placeholder(tf.float32, [None, patch_shape, r_shape, xi_shape, eta_shape, 1])
        self.y = tf.placeholder(tf.float32, [None, output_size])

        self.dropout_keep_prob = tf.placeholder(tf.float32)

        ### LAYER 0 ###
        self.layers = [{'input': self.X}]
        self.print_layer(self.layers, -1, 'input')
        
        ### LAYER 1 ###
        self.layers.append({})
        self.layers[-1].update(CubedSphereBaseModel.create_cubed_sphere_conv_layer(len(self.layers) - 1,
                                                                                   self.layers[-2]['input'],
                                                                                   ksize_r=1,
                                                                                   ksize_xi=3,
                                                                                   ksize_eta=3,
                                                                                   channels_out=32,
                                                                                   stride_r=1,
                                                                                   stride_xi=1,
                                                                                   stride_eta=1,
                                                                                   use_r_padding=False))
        self.layers[-1]['activation'] = tf.nn.relu(self.layers[-1]['conv'])
        self.print_layer(self.layers, -1, 'conv')
        self.print_layer(self.layers, -1, 'activation')

        ### LAYER 2 ###
        self.layers.append({})
        self.layers[-1].update(CubedSphereBaseModel.create_cubed_sphere_conv_layer(len(self.layers) - 1,
                                                                                   self.layers[-2]['activation'],
                                                                                   ksize_r=1,
                                                                                   ksize_xi=3,
                                                                                   ksize_eta=3,
                                                                                   channels_out=4,
                                                                                   stride_r=1,
                                                                                   stride_xi=1,
                                                                                   stride_eta=1,
                                                                                   use_r_padding=False))
        self.layers[-1]['activation'] = tf.nn.relu(self.layers[-1]['conv'])
        self.layers[-1].update(CubedSphereBaseModel.create_cubed_sphere_avgpool_layer(len(self.layers) - 1,
                                                                                      self.layers[-1]['activation'],
                                                                                      ksize_r=1,
                                                                                      ksize_xi=2,
                                                                                      ksize_eta=2,
                                                                                      stride_r=1,
                                                                                      stride_xi=2,
                                                                                      stride_eta=2,
                                                                                      use_r_padding=False))
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
