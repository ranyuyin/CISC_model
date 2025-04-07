# with reference to
### from official.vision.image_classification.configs import configs
"""Configuration utils for RS multi-header input classification experiments."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import dataclasses
import RSdataset_factory

from official.vision.image_classification import dataset_factory
from official.vision.image_classification.configs import base_configs
from official.modeling.hyperparams import base_config
# from official.vision.image_classification.efficientnet import efficientnet_config
# from official.vision.image_classification.resnet import resnet_config
from RSModel_config import *

@dataclasses.dataclass
class RsTrainConfig(base_configs.TrainConfig):
    resume_checkpoint_epoches: int = None

@dataclasses.dataclass
class RuntimeConfigServer(base_configs.RuntimeConfig):
    skipgpu: int = 0

@dataclasses.dataclass
class GF_ResNetConfig(base_configs.ExperimentConfig):
    """Base configuration to train combination of resnet-50 and ANN on Multi-header RS dataset."""
    export: base_configs.ExportConfig = base_configs.ExportConfig()
    runtime: base_configs.RuntimeConfig = RuntimeConfigServer()
    evaluation: base_configs.EvalConfig = base_configs.EvalConfig(
        epochs_between_evals=1, steps=None)

@dataclasses.dataclass
class gfldbanddem_ResNetANNConfig(GF_ResNetConfig):
    train_dataset: dataset_factory.DatasetConfig = \
        RSdataset_factory.RSTrainDatasetConfig_gfldbanddem(split='train')
    validation_dataset: dataset_factory.DatasetConfig = \
        RSdataset_factory.RSEvalDatasetConfig_gfldbanddem(split='validation')
    model: base_configs.ModelConfig = gfldbanddem_ResNetANNModelConfig()
    train: RsTrainConfig = RsTrainConfig(
        resume_checkpoint=True,
        resume_checkpoint_epoches=None,
        epochs=90,
        steps=None,
        callbacks=base_configs.CallbacksConfig(
            enable_checkpoint_and_export=True, enable_tensorboard=True),
        metrics=['accuracy','accreduce'],
        time_history=base_configs.TimeHistoryConfig(log_steps=200),
        tensorboard=base_configs.TensorBoardConfig(
            track_lr=True, write_model_weights=False),
        set_epoch_loop=False)

@dataclasses.dataclass
class geo_meta_config(base_config.Config):
    ncols: int = 0
    nrows: int = 0
    res: int = 0
    xmin: int = 0
    ymin: int = 0
    xmax: int = 0
    ymax: int = 0
    xmin_new: int = 0
    ymin_new: int = 0
    xmax_new: int = 0
    ymax_new: int = 0
    win_size: int = 128
    crs_string: str = ''  # Make a CRS from an EPSG, PROJ, or WKT string


@dataclasses.dataclass
class RS_Pred_RuntimeConfig(base_configs.RuntimeConfig):
    storage_root: str = ''
    out_dir: str = ''
    tile_name: str = ''
    skipgpu: int = 0


# >>>>>>>>>>>>>>>>>PREDICT CONFIG >>>>>>>>>>>>>>>>>>>
@dataclasses.dataclass
class RS_Pred_Base_Config(base_configs.ExperimentConfig):
    geo_meta: geo_meta_config = geo_meta_config()
    runtime: RS_Pred_RuntimeConfig = RS_Pred_RuntimeConfig()
    
    out_dtype: str = 'uint8'
    name: str = ''
    model_weights_dict: dict = None

    model_important_weight: list = None
    batch_size: int = 512
    temporalWindow:int = None


@dataclasses.dataclass
class gfldbanddem_ResNetANN_Pred_Config(RS_Pred_Base_Config):
    validation_dataset: dataset_factory.DatasetConfig = \
        RSdataset_factory.RSEvalDatasetConfig_gfldbanddem(split='validation')
    export: base_configs.ExportConfig = base_configs.ExportConfig()
    model: base_configs.ModelConfig = gfldbanddem_ResNetANNModelConfig()
    out_dtype: str = 'uint8'


# PREDICT CONFIG <<<<<<<<<<<<<<<<<<<<


def get_config(model: str, dataset: str) -> base_configs.ExperimentConfig:
    """Given model and dataset names, return the ExperimentConfig."""
    dataset_model_config_map = {
        'gfldbanddem':{
            'gfldbanddem_resnetann': gfldbanddem_ResNetANNConfig(),
        },
    }
    try:
        return dataset_model_config_map[dataset][model]
    except KeyError:
        if dataset not in dataset_model_config_map:
            raise KeyError('Invalid dataset received. Received: {}. Supported '
                           'datasets include: {}'.format(
                               dataset, ', '.join(dataset_model_config_map.keys())))
        raise KeyError('Invalid model received. Received: {}. Supported models '
                       'include: {}'.format(
                           model,
                           ', '.join(dataset_model_config_map.keys())))


def get_RSpred_config(model: str, dataset: str) -> base_configs.ExperimentConfig:
    dataset_model_config_map = {
        'gfldbanddem':{
            'gfldbanddem_resnetann': gfldbanddem_ResNetANN_Pred_Config(),
        },
    }
    try:
        return dataset_model_config_map[dataset][model]
    except KeyError:
        if dataset not in dataset_model_config_map:
            raise KeyError('Invalid dataset received. Received: {}. Supported '
                           'datasets include: {}'.format(
                               dataset, ', '.join(dataset_model_config_map.keys())))
        raise KeyError('Invalid model received. Received: {}. Supported models '
                       'include: {}'.format(
                           model,
                           ', '.join(dataset_model_config_map.keys())))
