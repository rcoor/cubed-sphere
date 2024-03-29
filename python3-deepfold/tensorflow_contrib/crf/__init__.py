# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Linear-chain CRF layer.

See the @{$python/contrib.crf} guide.

@@crf_sequence_score
@@crf_log_norm
@@crf_log_likelihood
@@crf_unary_score
@@crf_binary_score
@@CrfForwardRnnCell
@@viterbi_decode
"""





from tensorflow.contrib.crf.python.ops.crf import _lengths_to_masks
from tensorflow.contrib.crf.python.ops.crf import crf_binary_score
from tensorflow.contrib.crf.python.ops.crf import crf_log_likelihood
from tensorflow.contrib.crf.python.ops.crf import crf_log_norm
from tensorflow.contrib.crf.python.ops.crf import crf_sequence_score
from tensorflow.contrib.crf.python.ops.crf import crf_unary_score
from tensorflow.contrib.crf.python.ops.crf import CrfForwardRnnCell
from tensorflow.contrib.crf.python.ops.crf import viterbi_decode

# from tensorflow.python.util.all_util import remove_undocumented

# remove_undocumented(__name__)
