# Copyright 2017 Jes Frellsen and Wouter Boomsma. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

import tensorflow as tf
from . import pad_cubed_sphere
# import pad_wrap

def conv_spherical(input, filter, strides, padding, name=None):
  r"""Computes a spherical convolution (actually a cross-correlation) given 5-D
  `input` and `filter` tensors, using spherical coordinates.

  The input is assumed to be in spherical coordinates, in the order
  (r, theta, phi), where r denotes the radial component, theta the polar
  angle ([0;pi]) and phi is the azimuthal angle ([0;2*pi]).

  Note that this representation will have a grid distortion around the poles.
  The `conv_spherical_cubed_sphere` is often a better alternative.

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`.
      Shape `[batch, in_r, in_theta, in_phi, in_channels]`.
    filter: A `Tensor`. Must have the same type as `input`.
      Shape `[filter_r, filter_theta, filter_phi, in_channels,
      out_channels]`. `in_channels` must match between `input` and `filter`.
    strides: A list of `ints` that has length `>= 5`.
      1-D tensor of length 5. The stride of the sliding window for each
      dimension of `input`. Must have `strides[0] = strides[4] = 1`.
    padding: A `string` from: `"SAME", "VALID"`.
      The type of padding algorithm to use for the radial and polar dimensions.
      Note that the azimuthal dimension will always use periodic padding.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """

  filter_size_r, filter_size_theta, filter_size_phi = filter.shape.as_list()[1:-1]

  # Standard wrapping in r and theta dimensions
  padded_input = input
  if padding == "SAME":
      padded_input = tf.pad(input,
                            [(0, 0),
                             (filter_size_r // 2, filter_size_r // 2),
                             (filter_size_theta // 2, filter_size_theta // 2),
                             (0, 0),
                             (0, 0)], "CONSTANT")

  # Pad input with periodic image - only in phi
  # padded_input = pad_wrap.pad_wrap(padded_input,
  #                                  [(0, 0), (0, 0), (0, 0),
  #                                   (filter_size_phi / 2, filter_size_phi / 2),
  #                                   (0, 0)])

  return tf.nn.conv3d(padded_input,
                      filter,
                      strides=strides,
                      padding="VALID",
                      name=name)


def conv_spherical_cubed_sphere(input, filter, strides, padding, name=None):
  r"""Computes a spherical convolution (actually a cross-correlation) given 6-D
  `input` and `filter` tensors, using spherical coordinates.

  The input is assumed to be in cubed-sphere coordinates, in the order
  (patch, r, xi, eta, where patch is the cube-side ({0,..6}), r denotes the
  radial component, and xi and eta are angular variables in [-pi,pi]. See:

  Ronchi, C., R. Iacono, and Pier S. Paolucci. The "cubed sphere": a new method for
  the solution of partial differential equations in spherical geometry.
  Journal of Computational Physics 124.1 (1996): 93-114.

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`.
      Shape `[batch, in_patch, in_r, in_theta, in_phi, in_channels]`, where patch
      denotes the 6 faces of the cube.
    filter: A `Tensor`. Must have the same type as `input`.
      Shape `[filter_r, filter_xi, filter_eta, in_channels,
      out_channels]`. `in_channels` must match between `input` and `filter`.
    strides: A list of `ints` that has length `>= 5`.
      1-D tensor of length 5. The stride of the sliding window for each dimension
      of `input`. Must have `strides[0] = strides[1] = strides[4] = 1`.
    padding: A `string` from: `"SAME", "VALID"`.
      The type of padding algorithm to use for the radial dimension.
      Note that padding of the other dimensions is given by the wrapping of the
      cubed sphere.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """

  filter_size_r, filter_size_xi, filter_size_eta = filter.shape.as_list()[0:3]



  radial_padding_size = (0,0)
  if padding == "SAME":
    radial_padding_size = (filter_size_r // 2, filter_size_r // 2)

  padded_input = pad_cubed_sphere.pad_cubed_sphere_grid(input,
                                                        r_padding=radial_padding_size,
                                                        xi_padding=(filter_size_xi // 2, filter_size_xi // 2),
                                                        eta_padding=(filter_size_eta // 2, filter_size_eta // 2))
  convs = []
  for patch in range(padded_input.get_shape().as_list()[1]):
    convs.append(
      tf.nn.conv3d(padded_input[:, patch, :, :, :, :],
                   filter,
                   strides=strides,
                   padding="VALID",
                   name=name))

  conv = tf.stack(convs, axis=1, name=name)

  return conv

def avg_pool_spherical_cubed_sphere(value, ksize, strides, padding, name=None):
  r"""Performs average pooling of the input, using cubed sphere coordinates.
  Args:
    value: A `Tensor`. Must be one of the following types: `float32`, `float64`.
      Shape `[batch, in_r, in_xi, in_eta, in_channels]`.
    ksize: A list of ints that has length >= 5.
      The size of the window for each dimension of the input tensor.
      Must have `ksize[0] = ksize[4] = 1`.
    strides: A list of `ints` that has length `>= 5`.
      1-D tensor of length 5. The stride of the sliding window for each
      dimension of `input`. Must have `strides[0] = strides[4] = 1`.
    padding: A `string` from: `"SAME", "VALID"`.
      The type of padding algorithm to use for the radial and polar dimensions.
      Note that the azimuthal dimension will always use periodic padding.
    name: A name for the operation (optional).
  Returns:
    A `Tensor`. Has the same type as `input`.
  """

  ksize_r, ksize_xi, ksize_eta = ksize[1:4]

  radial_padding_size = (0,0)
  if padding == "SAME":
    radial_padding_size = (filter_size_r // 2, filter_size_r // 2)

  padded_input = pad_cubed_sphere.pad_cubed_sphere_grid(value,
                                                        r_padding=radial_padding_size,
                                                        xi_padding=(ksize_xi // 2, ksize_xi // 2),
                                                        eta_padding=(ksize_eta // 2, ksize_eta // 2))

  pools = []
  for patch in range(padded_input.get_shape().as_list()[1]):
    pools.append(tf.nn.avg_pool3d(padded_input[:, patch, :, :, :, :],
                                  ksize=[1, ksize_r, ksize_xi, ksize_eta, 1],
                                  strides=strides,
                                  padding='VALID'))

  return tf.stack(pools, axis=1, name=name)

if __name__ == '__main__':

  import numpy as np

  #### Spherical coordinates ####
  x = tf.placeholder(tf.float32, [None, 24, 76, 151, 2])

  # Filter shape
  filter_shape = [3, 5, 5, 2, 16]

  # Filter
  W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")

  # Strides
  strides = [1,1,1,1,1]

  # Spherical convolution
  conv_spherical = conv_spherical(x, W, strides, padding='VALID', name="conv")

  # Add Bias
  b = tf.Variable(tf.truncated_normal(filter_shape[-1:], stddev=0.1), name="b")
  layer = tf.nn.bias_add(conv_spherical, b)



  #### Cubed sphere coordinates ####
  x = tf.placeholder(tf.float32, [None, 6, 24, 38, 38, 2])

  # Filter shape
  filter_shape = [3, 5, 5, 2, 16]

  # Filter
  W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")

  # Strides
  strides = [1,1,1,1,1]

  # Spherical convolution
  conv_spherical_cs = conv_spherical_cubed_sphere(x, W, strides, padding='VALID', name="conv_cs")

  # Add Bias
  b = tf.Variable(tf.truncated_normal(filter_shape[-1:], stddev=0.1), name="b")
  layer = tf.nn.bias_add(conv_spherical_cs, b)

