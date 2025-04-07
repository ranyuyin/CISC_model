"""Configuration definitions for ResNet losses, learning rates, and optimizers."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import dataclasses

from official.modeling.hyperparams import base_config
from official.vision.image_classification.configs import base_configs



@dataclasses.dataclass
class gf_ResNetModelConfig(base_configs.ModelConfig):
    """Configuration for the RS Multi-header input model based on ResNet."""
    name: str = 'gf_resnet'
    num_classes: int = 4
    model_params: base_config.Config = dataclasses.field(
        default_factory=lambda: {
            'num_classes': 4,
            'batch_size': None,
            'use_l2_regularizer': True,
            'rescale_inputs': False,
            'input_size':121
        })
    loss: base_configs.LossConfig = base_configs.LossConfig(
        name='sparse_categorical_crossentropy')
    optimizer: base_configs.OptimizerConfig = base_configs.OptimizerConfig(
        name='momentum',
        decay=0.9,
        epsilon=0.001,
        momentum=0.9,
        moving_average_decay=None)
    learning_rate: base_configs.LearningRateConfig = (
        base_configs.LearningRateConfig(
            name='stepwise',
            initial_lr=0.1,
            examples_per_epoch=180000,
            boundaries=[30, 60, 80],
            # boundaries=[150, 300, 400],
            warmup_epochs=5,
            scale_by_batch_size=1. / 256.,
            # multipliers=[0.15 / 256, 0.015 / 256, 0.0015 / 256, 0.00015 / 256]))
            multipliers=[0.1 / 256, 0.01 / 256, 0.001 / 256, 0.0001 / 256]))


@dataclasses.dataclass
class gfldbanddem_ResNetANNModelConfig(base_configs.ModelConfig):
    """Configuration for the RS Multi-header input model based on ResNet."""
    name: str = 'gfldbanddem_resnetann'
    num_classes: int = 4
    model_params: base_config.Config = dataclasses.field(
        default_factory=lambda: {
            'num_classes': 4,
            'batch_size': None,
            'use_l2_regularizer': True,
            'rescale_inputs': False,
            'input_size':121,
            'n_hiden':60,
        })
    loss: base_configs.LossConfig = base_configs.LossConfig(
        name='sparse_categorical_crossentropy')
    optimizer: base_configs.OptimizerConfig = base_configs.OptimizerConfig(
        name='momentum',
        decay=0.9,
        epsilon=0.001,
        momentum=0.9,
        moving_average_decay=None)
    learning_rate: base_configs.LearningRateConfig = (
        base_configs.LearningRateConfig(
            name='stepwise',
            initial_lr=0.1,
            examples_per_epoch=180000,
            boundaries=[30, 60, 80],
            # boundaries=[150, 300, 400],
            warmup_epochs=5,
            scale_by_batch_size=1. / 256.,
            # multipliers=[0.15 / 256, 0.015 / 256, 0.0015 / 256, 0.00015 / 256]))
            multipliers=[0.1 / 256, 0.01 / 256, 0.001 / 256, 0.0001 / 256]))
