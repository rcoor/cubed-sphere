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
"""
Module for supporting directional data in tensorflow, in particular spherical
convolutions.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from directional.python.ops.conv import conv_spherical
from directional.python.ops.conv import conv_spherical_cubed_sphere
from directional.python.ops.conv import avg_pool_spherical_cubed_sphere
