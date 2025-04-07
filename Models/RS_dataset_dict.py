from os import path
from pathlib import Path
import numpy as np
from RSdataset_factory import MEAN_BAND,MEAN_RGB,STDDEV_BAND,STDDEV_RGB
import platform


def parselds(x):
    x = x[:6]
    x = x*0.0000275-0.2
    x = x-np.array(MEAN_BAND['ldband']).reshape(-1,1,1)
    x = x/np.array(STDDEV_BAND['ldband']).reshape(-1,1,1)
    return x

def parsedem(x):
    x = x[:3]
    x = x-np.array(MEAN_BAND['dem']).reshape(-1,1,1)
    x = x/np.array(STDDEV_BAND['dem']).reshape(-1,1,1)
    return x

def parsegf(x):
    x = x-np.array(MEAN_RGB['gf']).reshape(-1,1,1)
    x = x/np.array(STDDEV_RGB['gf']).reshape(-1,1,1)
    return x
def get(storage_root):
    Rsdataset_dict = {
        'gf':{
            # 'imNames':[path.join(storage_root, 'ChinaMap2m','2020','2020.vrt')],
            'imNames':[path.join(storage_root, 'ChinaMap2m','2022','2022.vrt')],
            'nNeighber':60,
            'rescaleFunc': parsegf,
            'name':'gf',
        },
        'ldband':{
            'imNames': [
                # path.join(storage_root,'ChinaYearlyLandsat','2020','2020_albers'+'.vrt')
                path.join(storage_root,'ChinaYearlyLandsat','2022','2022_albers'+'.vrt') 
                ],
            'rescaleFunc': parselds,
            'name':'ldband',
        },
        'dem':{
            'imNames':[
                path.join(storage_root,'srtm_products','demproductsR0_albers'+'.vrt')
            ],
            'rescaleFunc':parsedem,
            'name':'dem',
        },
    }
    return Rsdataset_dict