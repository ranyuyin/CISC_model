from __future__ import absolute_import
from __future__ import division
# from __future__ import google_type_annotations
from __future__ import print_function

import math

import tensorflow as tf
from typing import Any, Dict, List, Optional, Text, Tuple

# from tensorflow.python.keras.layers.preprocessing import image_preprocessing as image_ops
import official.vision.image_classification.augment as aug


class RSRandAugment(aug.ImageAugment):
  """Applies the RandAugment policy to images.

  RandAugment is from the paper https://arxiv.org/abs/1909.13719,
  """

  def __init__(self,
               num_layers: int = 2,
               magnitude: float = 10.,
               cutout_const: float = 40.,
               translate_const: float = 100.):
    """Applies the RandAugment policy to images.

    Args:
      num_layers: Integer, the number of augmentation transformations to apply
        sequentially to an image. Represented as (N) in the paper. Usually best
        values will be in the range [1, 3].
      magnitude: Integer, shared magnitude across all augmentation operations.
        Represented as (M) in the paper. Usually best values are in the range
        [5, 10].
      cutout_const: multiplier for applying cutout.
      translate_const: multiplier for applying translation.
    """
    super(RSRandAugment, self).__init__()

    self.num_layers = num_layers
    self.magnitude = float(magnitude)
    self.cutout_const = float(cutout_const)
    self.translate_const = float(translate_const)
    self.available_ops = [
        'AutoContrast', 'Equalize', 'Color', 'Contrast', 'Brightness', 'Sharpness', 'Cutout'
    ]

  def distort(self, image: tf.Tensor) -> tf.Tensor:
    """Applies the RandAugment policy to `image`.

    Args:
      image: `Tensor` of shape [height, width, 3] representing an image.

    Returns:
      The augmented version of `image`.
    """
    input_image_type = image.dtype

    if input_image_type != tf.uint8:
      image = tf.clip_by_value(image, 0.0, 255.0)
      image = tf.cast(image, dtype=tf.uint8)

    replace_value = [255] * 3
    min_prob, max_prob = 0.2, 0.8

    for _ in range(self.num_layers):
      op_to_select = tf.random.uniform([],
                                       maxval=len(self.available_ops) + 1,
                                       dtype=tf.int32)

      branch_fns = []
      for (i, op_name) in enumerate(self.available_ops):
        prob = tf.random.uniform([],
                                 minval=min_prob,
                                 maxval=max_prob,
                                 dtype=tf.float32)
        func, _, args = aug._parse_policy_info(op_name, prob, self.magnitude,
                                           replace_value, self.cutout_const,
                                           self.translate_const)
        branch_fns.append((
            i,
            # pylint:disable=g-long-lambda
            lambda selected_func=func, selected_args=args: selected_func(
                image, *selected_args)))
        # pylint:enable=g-long-lambda

      image = tf.switch_case(
          branch_index=op_to_select,
          branch_fns=branch_fns,
          default=lambda: tf.identity(image))

    image = tf.cast(image, dtype=input_image_type)
    return image