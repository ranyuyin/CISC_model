import os,sys
from os import path
module_path = os.path.abspath(path.join('.'))
if module_path not in sys.path:
    sys.path.insert(0, module_path)
import pprint
import numpy as np
from typing import Any, Tuple, Text, Optional, Mapping
import geopandas as gpd
from absl import app
from absl import flags
from absl import logging
import tensorflow as tf
from official.common import distribute_utils
from official.modeling import hyperparams
from official.modeling import performance
from official.utils import hyperparams_flags
from official.utils.misc import keras_utils
from functools import partial
from official.vision.image_classification.configs import base_configs
import configs
import RSdataset
import RS_dataset_dict
import rasterio as rio
from math import floor, ceil
import yaml
import trainer
from threading import Thread


def define_predict_flags():
    """Defines common flags for image classification."""
    hyperparams_flags.initialize_common_flags()
    flags.DEFINE_string(
        'dataset', default=None, help='The name of the input dataset.')
    flags.DEFINE_bool(
        'run_eagerly',
        default=False,
        help='Use eager execution and disable autograph for debugging.')
    flags.DEFINE_string(
        'model_type',
        default=None,
        help='The type of the model, e.g. resnet, etc.')
    flags.DEFINE_string(
        'out_dir',
        default=None,
        help='The location of the out_dir.')
    flags.DEFINE_string(
        'tile_name',
        default=None,
        help='The location of the tile_name.')
    flags.DEFINE_string(
        'data_dir',
        default=None,
        help='The location of the input data.')
    flags.DEFINE_string(
        'storage_root',
        default=None,
        help='The location of the storage_root.')
    flags.DEFINE_float(
        'xmin',
        default=None,
        help='xmin')
    flags.DEFINE_float(
        'xmax',
        default=None,
        help='xmax')
    flags.DEFINE_float(
        'ymin',
        default=None,
        help='ymin')
    flags.DEFINE_float(
        'ymax',
        default=None,
        help='ymax')
    flags.DEFINE_float(
        'res',
        default=None,
        help='resolution')
    flags.DEFINE_integer(
        'win_size',
        default=128,
        help='win_size')
    flags.DEFINE_list(
        'model_important_weight',
        default=[1],
        help='model_important_weight')
    flags.DEFINE_string(
        'prj_path',
        default=None,
        help='The location of the prj_path.')
    flags.DEFINE_string(
        'models_file',
        default=None,
        help='The location of the multi models config file.') 
    flags.DEFINE_integer(
        name="batch_size",
        short_name="bs",
        default=512,
        help="Batch size for training and evaluation.")
    flags.DEFINE_integer(
        name="skip_gpu",
        default=0,
        help="skip gpu numbers.")
        


def _get_params_from_flags(flags_obj: flags.FlagValues):
    xmin, ymin, xmax, ymax = flags_obj.xmin, flags_obj.ymin, flags_obj.xmax, flags_obj.ymax
    res = flags_obj.res
    xmin_new = floor(xmin/res)*res
    ymin_new = floor(ymin/res)*res
    xmax_new = ceil(xmax/res)*res
    ymax_new = ceil(ymax/res)*res
    nrows = round((ymax_new-ymin_new)/res)
    ncols = round((xmax_new-xmin_new)/res)
    model = flags_obj.model_type.lower()
    params = configs.get_RSpred_config(model=model, dataset=flags_obj.dataset)
    multi_model_dict = yaml.safe_load(open(flags_obj.models_file))
    # print('eager config:{}'.format(flags_obj.run_eagerly))
    flags_overrides = {
        'geo_meta': {
            'ncols': ncols,
            'nrows': nrows,
            'res': res,
            'xmin': xmin,
            'ymin': ymin,
            'xmax': xmax,
            'ymax': ymax,
            'xmin_new': xmin_new,
            'ymin_new': ymin_new,
            'xmax_new': xmax_new,
            'ymax_new': ymax_new,
            'win_size': flags_obj.win_size,
            'crs_string': open(flags_obj.prj_path).read(),
        },
        'runtime': {
            'run_eagerly': flags_obj.run_eagerly,
            'tpu': flags_obj.tpu,
            'num_gpus': flags_obj.num_gpus,
            'storage_root': flags_obj.storage_root,
            'out_dir': flags_obj.out_dir,
            'tile_name': flags_obj.tile_name,
            'skipgpu': flags_obj.skip_gpu
        },
        'name': flags_obj.dataset,
        'model_weights_dict': multi_model_dict,
        'model_important_weight': [float(i) for i in flags_obj.model_important_weight],
        'batch_size':flags_obj.batch_size
    }
    overriding_configs = (flags_obj.config_file, flags_obj.params_override,
                          flags_overrides)
    pp = pprint.PrettyPrinter()

    logging.debug('Base params: %s', pp.pformat(params.as_dict()))

    for param in overriding_configs:
        logging.debug('Overriding params: %s', param)
        params = hyperparams.override_params_dict(
            params, param, is_strict=True)
    params.validate()
    assert len(params.model_weights_dict.as_dict().keys())<=len(params.model_important_weight)
    params.lock()
    logging.info('Final model parameters: %s', pp.pformat(params.as_dict()))
    return params


def serialize_config(params: base_configs.ExperimentConfig, out_dir: str):
    """Serializes and saves the experiment config."""
    params_save_path = os.path.join(out_dir, 'params.yaml')
    logging.info('Saving experiment configuration to %s', params_save_path)
    tf.io.gfile.makedirs(out_dir)
    hyperparams.save_params_dict_to_yaml(params, params_save_path)


def initialize(params: base_configs.ExperimentConfig):
    """Initializes backend related initializations."""
    keras_utils.set_session_config(enable_xla=params.runtime.enable_xla)
    performance.set_mixed_precision_policy(tf.float32)
    if params.runtime.run_eagerly:
        # Enable eager execution to allow step-by-step debugging
        tf.config.run_functions_eagerly(True)
        print('eager {}.'.format(True))
    else:
        tf.config.run_functions_eagerly(False)
        print('eager {}.'.format(False))
    if tf.config.list_physical_devices('GPU'):
        if params.runtime.gpu_thread_mode:
            keras_utils.set_gpu_thread_mode_and_count(
                per_gpu_thread_count=params.runtime.per_gpu_thread_count,
                gpu_thread_mode=params.runtime.gpu_thread_mode,
                num_gpus=params.runtime.num_gpus,
                datasets_num_private_threads=params.runtime
                .dataset_num_private_threads)  # pylint:disable=line-too-long
        if params.runtime.batchnorm_spatial_persistent:
            os.environ['TF_USE_CUDNN_BATCHNORM_SPATIAL_PERSISTENT'] = '1'

def getTemporalWinDataset(input, start_i=None, temporalWindow=None):
    # print(input)
    win_slice_featrues = ['lds']
    for n in win_slice_featrues:
        input[n] = tf.slice(input[n],[0+start_i,0,0,0],[temporalWindow, -1, -1, -1])
    return input

class RSDataThread(Thread):
    def __init__(self, func, args, ngpu):
        '''
        :param func: 可调用的对象
        :param args: 可调用对象的参数
        '''
        Thread.__init__(self)
        self.func = func
        self.args = args
        self.ngpu = ngpu
        self.result = None

    def run(self):
        with tf.device("/gpu:{}".format(self.ngpu)):
            self.result = self.func(*self.args)#.prefetch(1)

    def getResult(self):
        return self.result

def RS_predict(params: configs.RS_Pred_Base_Config,
               strategy_override: tf.distribute.Strategy):
    dataset_dict = {
        'gfldbanddem':['ldband','dem','gf']
    }
    logging.info('Running RS Predict.')
    distribute_utils.configure_cluster(params.runtime.worker_hosts,
                                       params.runtime.task_index)
    strategy = strategy_override or distribute_utils.get_distribution_strategy(
        distribution_strategy=params.runtime.distribution_strategy,
        all_reduce_alg=params.runtime.all_reduce_alg,
        num_gpus=params.runtime.num_gpus,
        tpu_address=params.runtime.tpu)
    strategy_scope = distribute_utils.get_strategy_scope(strategy)

    logging.debug('Detected %d devices.',
                 strategy.num_replicas_in_sync if strategy else 1)

    ds_name_list = dataset_dict[params.name]
    Rsdataset_dict = RS_dataset_dict.get(params.runtime.storage_root)
    mrsd = RSdataset.MultiHeadDatasets(
        {
            k: RSdataset.rsDataset(
                **Rsdataset_dict[k]
            ) for k in ds_name_list
        }
    )
    # print(mrsd.subDatasets['lds'].rescaleFunc)
    tile_transform = rio.transform.from_bounds(
        params.geo_meta.xmin_new,
        params.geo_meta.ymin_new,
        params.geo_meta.xmax_new,
        params.geo_meta.ymax_new,
        params.geo_meta.ncols,
        params.geo_meta.nrows
    )

    rowids = np.arange(params.geo_meta.nrows)
    colids = np.arange(params.geo_meta.ncols)

    grid_results = []
    n_grid_row = ceil(params.geo_meta.nrows/params.geo_meta.win_size)

    n_grid_col = ceil(params.geo_meta.ncols/params.geo_meta.win_size)

    row_groups = np.array_split(rowids, n_grid_row)
    col_groups = np.array_split(colids, n_grid_col)
    initialize(params)
    logging.info('Global batch size: %d', params.batch_size)
    with tf.device("/gpu:{}".format(params.runtime.skipgpu)):
        boundsList,nrowsList,ncolList = [],[],[]
        for _, row_block_ids in enumerate(row_groups):
            for _, col_block_ids in enumerate(col_groups):
                this_bound = (
                    params.geo_meta.xmin_new +
                    col_block_ids[0]*params.geo_meta.res,
                    params.geo_meta.ymax_new -
                    (row_block_ids[-1]+1)*params.geo_meta.res,
                    params.geo_meta.xmin_new +
                    (col_block_ids[-1]+1)*params.geo_meta.res,
                    params.geo_meta.ymax_new -
                    (row_block_ids[0])*params.geo_meta.res,
                )
                nrowsList.append(len(row_block_ids))
                ncolList.append(len(col_block_ids))
                boundsList.append(this_bound)

        next_block_dataset = None
        for i, row_block_ids in enumerate(row_groups):
            this_row_results = []
            for j, col_block_ids in enumerate(col_groups):
                thisID = i*len(col_groups)+ j
                logging.info('predicting %d of %d',thisID+1,len(boundsList))
                this_bound = boundsList[thisID]
                nrows, ncols = nrowsList[thisID],ncolList[thisID]
                if thisID == 0:
                    block_dataset = mrsd.predict_dataset_rect(
                        this_bound, params.geo_meta.res)#.cache()
                elif next_block_dataset is not None:
                    # del block_dataset
                    block_dataset = next_block_dataset
                if (thisID+1) < len(boundsList):
                    tdata_next = RSDataThread(mrsd.predict_dataset_rect, [boundsList[thisID+1], params.geo_meta.res], params.runtime.skipgpu)
                    tdata_next.start()

                if params.temporalWindow is not None and 'lds' in ds_name_list:
                    temporalWindow = params.temporalWindow
                    totalN = len(Rsdataset_dict['lds']['imNames'])
                    lastWinStart = totalN-temporalWindow
                    nMid = int(lastWinStart/temporalWindow*2) + 1
                    stepLen = ceil(lastWinStart/nMid)
                    winStarts = [ii*stepLen for ii in range(nMid)] + [lastWinStart]
                    result = predict_block_multi_model_temporal(
                        params, 
                        block_dataset, 
                        strategy_scope=None, 
                        winStarts=winStarts, 
                        temporalWindow=temporalWindow, 
                        totalN=totalN, 
                        batch_size=2000)
                else:
                    block_dataset = block_dataset.batch(params.batch_size).prefetch(10)
                    result = predict_block_multi_model(params, block_dataset, strategy_scope)       
                this_row_results.append(result.reshape((nrows,ncols)))
                if (thisID+1) < len(boundsList):
                    tdata_next.join()
                    next_block_dataset = tdata_next.getResult()    
                    del block_dataset,tdata_next
            grid_results.append(np.concatenate(this_row_results,axis=1))
        del next_block_dataset

    final_result = np.concatenate(grid_results)
    out_path = path.join(params.runtime.out_dir, params.runtime.tile_name)
    out_meta = {
        'driver': 'GTiff',
        'width': params.geo_meta.ncols,
        'height': params.geo_meta.nrows,
        'count': 1,
        'dtype': params.out_dtype,
        'crs': rio.crs.CRS.from_string(params.geo_meta.crs_string),
        'transform': tile_transform,
        'compress': 'lzw',
    }
    with rio.open(out_path, 'w', **out_meta) as dst:
        dst.write(final_result.astype(params.out_dtype), 1)
    serialize_config(params=params, out_dir=params.runtime.out_dir)


def predict_block_multi_model_temporal(params, block_dataset, winStarts=[0], strategy_scope=None, temporalWindow=20, totalN=20, batch_size=512):
    resultlist = []
    # with strategy_scope:
    model_params = params.model.model_params.as_dict()
    # logging.info('DEBUG: %s', params.model_weights_dict)
    tf.keras.backend.clear_session()
    model = trainer.get_models()[params.model.name](**model_params)
    # assert len(params.model_weights_dict.as_dict().keys())<len(params.model_important_weight)
    for k in params.model_weights_dict.as_dict().keys():
        if params.model_weights_dict.as_dict()[k] is None:
            continue
        if params.model_important_weight[int(k)] < 0.01:
            continue
        model.load_weights(params.model_weights_dict.as_dict()[k])
        logging.info('loaded model: %s', k)
        predList = []
        for start_i in winStarts:
            block_temporal_dataset = block_dataset.map(
                partial(getTemporalWinDataset, start_i=start_i, temporalWindow=temporalWindow),
                num_parallel_calls=tf.data.AUTOTUNE
                ).batch(batch_size).prefetch(5)
            logging.debug('loaded temporal dataset: %d', start_i)
            # for itemmm in block_temporal_dataset.unbatch().take(1):
            #     print(itemmm)
            thislogits = model.predict(block_temporal_dataset,verbose=0)
            padding = [[0,0],[totalN - start_i - temporalWindow, start_i]]
            # 两侧的logits向外平均
            thislogits[:,-1] /= start_i+1
            thislogits[:,0] /= totalN - start_i - temporalWindow + 1
            predList.append(np.pad(thislogits, padding, mode='edge'))
        pred = np.stack(predList,2)
        pred = np.mean(pred, 2)
        resultlist.append(pred*params.model_important_weight[int(k)])
    all_result = np.stack(resultlist)
    return all_result.sum(axis=0).argmax(axis=1)

def predict_block_multi_model(params, block_dataset, strategy_scope=None):
    resultlist = []
    # with strategy_scope:
    model_params = params.model.model_params.as_dict()
    # logging.info('DEBUG: %s', params.model_weights_dict)
    tf.keras.backend.clear_session()
    model = trainer.get_models()[params.model.name](**model_params)

    model_importants = np.array(params.model_important_weight)
    if (model_importants[0]>0.99) and (model_importants.sum()>2.1):
        model_importants[1:]=0
    for k in params.model_weights_dict.as_dict().keys():
        if params.model_weights_dict.as_dict()[k] is None:
            continue
        if model_importants[int(k)] < 0.01:
            continue
        logging.info('Predicting %d th Model', int(k))
        model.load_weights(params.model_weights_dict.as_dict()[k])
        this_result = model.predict(block_dataset)
        resultlist.append(this_result*model_importants[int(k)])
    all_result = np.stack(resultlist)
    return all_result.sum(axis=0).argmax(axis=1)


def run(flags_obj: flags.FlagValues,
        strategy_override: tf.distribute.Strategy = None) -> Mapping[str, Any]:
    """Runs Image Classification model using native Keras APIs.
    Args:
      flags_obj: An object containing parsed flag values.
      strategy_override: A `tf.distribute.Strategy` object to use for model.
    Returns:
      Dictionary of training/eval stats
    """
    params = _get_params_from_flags(flags_obj)
    return RS_predict(params, strategy_override)


def main(_):
    stats = run(flags.FLAGS)
    if stats:
        logging.info('Run stats:\n%s', stats)


if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
        except RuntimeError as e:
            print(e)
    define_predict_flags()
    flags.mark_flag_as_required('dataset')
    flags.mark_flag_as_required('out_dir')
    flags.mark_flag_as_required('model_type')
    flags.mark_flag_as_required('tile_name')
    flags.mark_flag_as_required('prj_path')
    flags.mark_flag_as_required('xmin')
    flags.mark_flag_as_required('xmax')
    flags.mark_flag_as_required('ymin')
    flags.mark_flag_as_required('ymax')
    flags.mark_flag_as_required('res')
    flags.mark_flag_as_required('storage_root')
    
    app.run(main)
