# Copyright 2023 The medical_research_foundations Authors.
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

"""Library for computing BiT embeddings."""

import tensorflow.compat.v1 as tf
import tensorflow_hub as hub

BIT_MODULE_PATH = {
    'ResNet-50x1': 'https://tfhub.dev/google/bit/m-r50x1/1',
    'ResNet-152x1': 'https://tfhub.dev/google/bit/m-r152x2/1',
}


def bit_embedding(
    images,  # pylint: disable=invalid-name
    model_name='ResNet-50x1',
    trainable=True,
    verify_input_range=True,
):
  """Tensorflow function that computes BiT embeddings from hub module.

  Args:
    images: a 4-D RGB images tensor scaled to [0, 1] range of the shape N x H x
      W x C, where N is the batch size, H and W are images' height and width, C
      is the number of input channels. The number of input channels should be
      exactly 3, H and W are flexible, but each of them should be at least 32
      pixels and for the best performance they should be close to 224 pixels.
    model_name: name of the model. Currently, we support 5 ResNet models:
      `ResNet-50x1`, `ResNet-50x3`, `ResNet-101x1`, `ResNet-101x3` and
      `ResNet-152x4`, where the first number indicates the network's depth and
      the second number, e.g. 'x3', indicates the network's width multiplier.
      Multipliers 'x1', 'x3' and 'x4' correspond to 2048, 6144 and
      8192-dimensional embeddings respectively.
    trainable: if True, the model variables are treated as trainable and are
      automatically added to the TRAINABLE_VARIABLES collection.
    verify_input_range: if True, the function will verify that input images are
      scaled to range [0, 1].

  Returns:
    A tuple, where the first item is a tensor with BiT embeddings and the second
    item is a dictionary with various end_points of the BiT model.
  """

  model = hub.Module(model_name, trainable=trainable)

  assert_range_ops = []
  if verify_input_range:
    # Verify [0, 1] range.
    assert_range_ops = [
        tf.assert_greater_equal(tf.reduce_min(images), 0.0),
        tf.assert_less_equal(tf.reduce_max(images), 1.0),
    ]

  # Scale to range [-1, 1].
  images = 2.0 * images - 1.0

  # Apply BiT model.
  with tf.control_dependencies(assert_range_ops):
    end_points = model(images, signature='representation', as_dict=True)

  return end_points['pre_logits'], end_points
