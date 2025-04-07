## 生成每个分区的训练参数，调用trainer进行训练
import geopandas as gpd
from glob import glob
import os
from os import path
import sys
py_dir = path.split(sys.argv[0])[0]
module_path = os.path.abspath(path.join(py_dir,'..'))
if module_path not in sys.path:
    sys.path.insert(0, module_path)

import yaml
import platform
import tensorflow as tf

n_gpus = 2
storage_root = 'path_to_satellite_data_root'
model_root = 'path_to_model_root'
tfrecoed_dir = 'path_to_tfrecord_root'
tfr_dict = {
    'gfldbanddem': path.join(tfrecoed_dir,'GF_L20Familly'),
}
zonal_pattern = 'validation_zone{:d}_*'
general_pattern = 'validation_*'

weightPaths = {
    'refine4ISCover': ["refine4ISCover_sampleSample.csv"]
}

def genConfigFile(
    val_tfrs, model_dir, 
    train_weight_df_name=None, train_class_df_name=None,
    val_weight_df_name=None, val_class_df_name=None,
    agname=None,n_gpu_skip=0,
    n_epoch = None, ckpt_epoch=None
    ):
    config_dict = {
        'train_dataset':{
            'weight_df_name': train_weight_df_name,
            'class_df_name': train_class_df_name,
            'augmenter':{
                'name': agname,
            },
            'mean_subtract': True,
            'standardize': True,
        },
        'validation_dataset': {
            'filenames':val_tfrs,
            'weight_df_name': val_weight_df_name,
            'class_df_name': val_class_df_name,
            'mean_subtract': True,
            'standardize': True,
        },
        'runtime':{
            'num_gpus': n_gpus,
            'skipgpu': n_gpu_skip,
        },
        'train':{     
        }

    }
    if n_epoch is not None:
        nd = {
                'epochs':n_epoch
            }
        config_dict['train'].update(nd)
    if ckpt_epoch is not None:
        nd = {
                'resume_checkpoint_epoches':ckpt_epoch
            }
        config_dict['train'].update(nd)
    config_path = path.join(model_dir, 'config.yaml')
    with open(config_path, 'w') as f:
        yaml.safe_dump(config_dict,f)
    return config_path
    
def do_zonal(
    dataset_name, 
    model_type, 
    zoneID=None, 
    append_zone_weight=2, 
    weightdfkey=None, 
    agname=None, 
    additionalComments=None, 
    labeldfkey=None,
    n_gpu_skip=0,
    debug=False,
    finetuning = False,
    model_dir = None,
    config_file=None
    ):
    # 核心操作在于生成每个分区的训练权重和验证集
    # model_type = ''
    comments = '{}_{}_z{}_ag{}_{}'.format(dataset_name, model_type, zoneID, agname, additionalComments)
    if model_dir is None:
        model_dir = path.join(model_root, comments)
    if not path.exists(model_dir):
        os.makedirs(model_dir)
    data_dir = tfr_dict[dataset_name]

    val_tfrs = glob(path.join(data_dir,general_pattern))
    if weightdfkey is not None:
        weight_df_name = path.join(storage_root, *weightPaths[weightdfkey])
    else: 
        weight_df_name = None
    valWeight = path.join(storage_root, *weightPaths['refine4ISCover'])
    if config_file is None:
        config_file = genConfigFile(
            val_tfrs=val_tfrs,
            model_dir=model_dir, 
            train_weight_df_name=weight_df_name,
            val_weight_df_name=valWeight,
            val_class_df_name=None,
            agname=agname,
            n_gpu_skip=n_gpu_skip
            )
        #  --num_gpus=1
    command = 'python {} \
        --dataset={} --data_dir={} --model_type={} --model_dir={} \
            --mode=train_and_eval --config_file={} --run_eagerly={}'.format(path.join(py_dir,'trainer.py'),
                dataset_name, data_dir, model_type, model_dir, config_file, debug
                )

    os.system(command)

if __name__ == '__main__':
    model_dir = None
    config_file = None
    labeldfkey = None
    ft = False
    comments = ''
    dataset_name = 'gfldbanddem'
    model_type = 'gfldbanddem_resnetann'
    weightdfkey='refine4ISCover'

    agname = 'rsrandaugment'
    data_dir = tfr_dict[dataset_name]
    
    debug = False
    if debug:
        n_gpus = 1

    try:
        n_gpu_skip = int(sys.argv[1])
    except:
        n_gpu_skip = 0

    do_zonal(
        dataset_name=dataset_name, 
        model_type=model_type, 
        weightdfkey=weightdfkey, 
        labeldfkey=labeldfkey,
        agname=agname, 
        additionalComments=comments, 
        n_gpu_skip=n_gpu_skip, debug=debug,
        finetuning = ft,
        model_dir=model_dir,
        config_file = config_file,
        )
