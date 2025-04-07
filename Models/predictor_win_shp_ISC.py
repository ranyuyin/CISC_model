## 读取定义"预测目标窗口"的矢量文件
## （暂定）矢量文件中可使用逗号分隔字符串定义每个模型参数的权重，如‘0.5,0,0.5’
#       权重小于0.01的模型参数将跳过
# 需指定模型参数路径列表文件yaml
## 
import geopandas as gpd
from glob import glob
import os
from os import path
from tqdm import tqdm
import sys
from functools import partial
import time
py_dir = path.split(sys.argv[0])[0]
module_path = os.path.abspath(path.join(py_dir,'.'))
if module_path not in sys.path:
    sys.path.insert(0, module_path)
import yaml

import platform
from multiprocessing import Pool

storage_root = 'path_to_satellite_data_root'
model_root = 'path_to_model_root'
pred_root = 'path_to_save_pred_result'

n_gpus = 1
res = 30
res_dict = {
    'gfldbanddem': 30,
}
prj_dict = {
    'gfldbanddem': 'path of the prj',
}
multiModelDict = {
    'gfldbanddem':{
        '0':path.join(
            model_root,
            'gfldbanddem_gfldbanddem_resnetann_zNone_agrsrandaugment_', 
            'model.ckpt-0068'
            ),
        '1': path.join(
            model_root,
            'gfldbanddem_gfldbanddem_resnetann_ReT_noNormal_pureMulti_zone1', 
            'model.ckpt-0073'
            ),
        '2': path.join(
            model_root,
            'gfldbanddem_gfldbanddem_resnetann_ReT_noNormal_pureMulti_zone2', 
            'model.ckpt-0080'
            ),
        '3': path.join(
            model_root,
            'gfldbanddem_gfldbanddem_resnetann_ReT_noNormal_pureMulti_zone3', 
            'model.ckpt-0074'
            ),
        '4': path.join(
            model_root,
            'gfldbanddem_gfldbanddem_resnetann_ReT_noNormal_pureMulti_zone4', 
            'model.ckpt-0087'
            ),
        '5': path.join(
            model_root,
            'gfldbanddem_gfldbanddem_resnetann_ReT_noNormal_pureMulti_zone5', 
            'model.ckpt-0085'
            ),
    }
}

def parseNameLds(item):
    xmin,ymin,xmax,ymax = item.geometry.bounds
    n = 'N{:.0f}_E{:.0f}.tif'.format(ymax,xmin)
    return n

def genConfigFile(eval_result_dir):
    config_dict = {
        'validation_dataset': {
            'mean_subtract': True,
            'standardize': True,
        },
        }
    config_path = path.join(eval_result_dir, 'config.yaml')
    with open(config_path, 'w') as f:
        yaml.safe_dump(config_dict,f)
    return config_path

def pred_row(row,out_dir,nameParser,dataset_name,model_type,models_dict_file,config_file=None,n_gpu_skip=0,n_split=-1):
    
    i,item = row
    xmin,ymin,xmax,ymax = item.geometry.bounds
    weightList = [str(item['model_{}'.format(i)]) for i in range(6)]
    model_important_weight = ','.join(weightList)
    # xmax = xmin+0.2
    # ymax = ymin+0.2
    if n_split != -1:
        if i%4 != n_split:
            return
    this_name = nameParser(item)
    if path.exists(path.join(out_dir,this_name)):
        print('skip {}'.format(this_name))
        return
    debug = False
    eagerly = '--run_eagerly' if debug else ''
    command = "python {} --dataset {} --model_type {} --prj_path {} \
        --xmin {} --ymin {} --xmax {} --ymax {} --res {} --storage_root {} \
            --models_file {} -model_important_weight {} -batch_size {}\
                --out_dir {} --tile_name {} --num_gpus {} --win_size {} --skip_gpu {} {} \
                    ".format(
                    path.join(py_dir,'predictor.py'),
                    dataset_name,
                    model_type,
                    prj_dict[dataset_name],
                    xmin, ymin, xmax, ymax, res_dict[dataset_name],
                    storage_root, models_dict_file, model_important_weight, 512,
                    out_dir, this_name, n_gpus, 1000 ,n_gpu_skip,eagerly
                )
    print(command)
    os.system(command)
    time.sleep(2)
    
if __name__ == '__main__':
    out_comments = 'gfldbanddem'
    dataset_name = 'gfldbanddem'
    model_type = 'gfldbanddem_resnetann'

    gwin_name = 'path_to_gridwindow_shpfile'

    gdf_gwin = gpd.read_file(gwin_name)
    out_dir = path.join(pred_root,out_comments)
    if not path.exists(out_dir):
        os.makedirs(out_dir)


    models_dict = multiModelDict[dataset_name]
    models_dict_file = path.join(out_dir,'models_dict_{}.yaml'.format(platform.system()))
    with open(models_dict_file, 'w') as f:
        yaml.safe_dump(models_dict, f)
    
    try:
        n_gpu_skip = int(sys.argv[1])
    except:
        n_gpu_skip = 0

    try:
        n_split = int(sys.argv[2])
    except:
        n_split = -1

    for row in tqdm(gdf_gwin.iterrows(),total=len(gdf_gwin)):
        pred_row(
            row,out_dir=out_dir,
            nameParser=parseNameLds,
            dataset_name=dataset_name,
            model_type=model_type,
            models_dict_file=models_dict_file,
            n_split=n_split,
            n_gpu_skip=n_gpu_skip,
            )
        