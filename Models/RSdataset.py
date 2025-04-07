import geopandas as gpd
import rasterio as rio
import numpy as np
# import numba
# import sys,math
import os
from os import path, read
from rasterio.transform import rowcol
from rasterio.warp import transform
from multiprocessing import Pool
from tqdm import tqdm
from functools import partial
import cv2
from shapely.geometry import Point


class rsDataset():
    def __init__(
        self, 
        imNames, 
        nNeighber=0, 
        rescaleFunc=None,
        multiSorceWeightDataset=None, 
        name=None, 
        class_dict=None, 
        tf_dtype=None,
        outCount=None
        ) -> None:
        if type(imNames) is not list:
            imNames = [imNames]
        self.rescaleFunc = rescaleFunc if rescaleFunc is not None else lambda x: x
        self.weightDS = multiSorceWeightDataset
        self.class_dict = class_dict
        self.name = name
        ntime = len(imNames)
        self.srcs = [rio.open(im) for im in imNames]
        self.crs = self.srcs[0].crs
        self.tr = self.srcs[0].res[0]
        self.transform = self.srcs[0].transform
        self.nNeighber = nNeighber 
        self.sampleWidth, self.sampleHeight = 2*nNeighber+1, 2*nNeighber+1
        self.count = self.srcs[0].count
        self.ntime = ntime
        outCount = self.count if outCount is None else outCount
        shape = np.array(
            [ntime, self.sampleHeight, self.sampleWidth, outCount])
        # self.sampleShape = tuple(shape[shape > 1])
        try:
            import tensorflow as tf
        except:
            print("need tensorflow")
            return
        tf_dtype = tf_dtype if tf_dtype is not None else tf.float32
        self.FeatureSpec = tf.TensorSpec(
            shape=shape, dtype=tf_dtype)

    # def readGeoWin(self, geoBounds):
    #     xmin, ymin, xmax, ymax = geoBounds
    #     ioWin = rowcol(self.transform, [xmin, xmax], [ymax, ymin])
    #     ar = self.src.read(window=ioWin)
    #     return ar
    def read(self, imWin):
        # return self.s.read(window=imWin).transpose([1,2,0])
        return np.stack([self.rescaleFunc(s.read(window=imWin,boundless=True,fill_value=0)).transpose([1, 2, 0]) for s in self.srcs])

    def getImPointWin(self, p: Point):
        [x], [y] = transform(rio.crs.CRS.from_epsg(4326),
                             self.crs, [p.x], [p.y])
        rowC, colC = rowcol(self.transform, x, y)
        imWin = ((rowC-self.nNeighber, rowC+self.nNeighber+1),
                 (colC-self.nNeighber, colC+self.nNeighber+1))
        return imWin

    def getPointFeature(self, p: Point):
        # outshp = [times, rows, cols, bands]
        imWin = self.getImPointWin(p)
        ars = []
        for src in self.srcs:
            # print(imWin)
            ars.append(np.squeeze(self.rescaleFunc(
                src.read(window=imWin)).transpose((1, 2, 0))))
        return np.squeeze(np.stack(ars))

    def fromShapefile(self, shpfileName, shuffle=False, BATCH_SIZE=1, cache=False, split='train', cleanFunction=None):
        # from Point geometry
        # using Function<cleanFunction> to pre-process featureList, which will modify the shp list inplace
        self.featureList = gpd.read_file(shpfileName)
        if cleanFunction is not None:
            cleanFunction(self.featureList)
        self.featureList = self.featureList.loc[self.featureList['split'] == split]
        if shuffle:
            self.featureList = self.featureList.sample(frac=1.0)
        if self.class_dict is None:
            self.class_names = self.featureList['class'].unique()
            self.class_dict = {n: i for i, n in enumerate(self.class_names)}
        dataset = tf.data.Dataset.from_generator(
            self.genSampleShapefile,
            output_signature=(
                self.FeatureSpec,
                tf.TensorSpec(shape=(), dtype=tf.uint8),
                tf.TensorSpec(shape=(), dtype=tf.float32)
            ))
        cacheName = self.name if self.name is not None else path.splitext(
            path.basename(shpfileName))[0]

        return dataset  # .prefetch(200)

    def genSampleShapefile(self):
        for i, row in self.featureList.iterrows():
            geoP = row['geometry']
            classID = row['class']
            yield (
                self.getPointFeature(geoP),
                self.class_dict[classID],
                1 if self.weightDS is None else self.weightDS.getWeight(geoP)
            )


    def predict_dataset_rect(self, bound, operating_res, batch_size=None):
        
        xmin, ymin, xmax, ymax = bound
        outWin = (
            (xmin+operating_res/2, xmax-operating_res /
             2), (ymax-operating_res/2, ymin+operating_res/2)
        )
        expand = self.tr*self.nNeighber
        imWin = ((xmin+self.tr/2-expand, xmax+self.tr/2+expand),
                 (ymax-self.tr/2+expand, ymin-self.tr/2-expand))
        im_window = rowcol(self.transform, imWin[0], imWin[1])

        ar_wins =  np.stack(self.read(imWin=im_window))
        
        off = (im_window[0][0], im_window[1][0])
        eps = np.finfo(np.float32).eps
        xs = np.arange(outWin[0][0], outWin[0][1]+eps, operating_res)
        ys = np.arange(outWin[1][0], outWin[1][1]-eps, -operating_res)
        xv, yv = np.meshgrid(xs, ys)
        xys = np.stack([xv,yv],axis=2).reshape(-1,2)
        rowcols = np.stack(rowcol(self.transform, xys[:,0], xys[:,1]),axis=1)-off
        del xys, xv, yv, xs, ys
        rowcols = rowcols.astype(np.int32)
        dataset_rowcol = tf.data.Dataset.from_tensor_slices((rowcols[:,0],rowcols[:,1]))
        dataset_refs = tf.data.Dataset.from_tensors((ar_wins,np.int32(self.nNeighber))).repeat().take(len(rowcols))
        dataset = tf.data.Dataset.zip((dataset_rowcol,dataset_refs))
        def func(rowcol, refs):
            this_row, this_col = rowcol
            ar_wins,nNeighber = refs
            return tf.slice(ar_wins,[0,this_row-nNeighber,this_col-nNeighber,0],[-1,nNeighber*2+1,nNeighber*2+1,-1])
        dataset = dataset.map(func,num_parallel_calls=tf.data.AUTOTUNE)
        return dataset


class DataYearly(rsDataset):
    def __init__(self, vrtName=None, **kwarg) -> None:
        super().__init__(vrtName,  **kwarg)


class DataGF(rsDataset):
    def __init__(self, vrtName=None, nNeighber=60, **kwarg) -> None:
        super().__init__(vrtName, nNeighber, **kwarg)

    def genOne(self):
        for _, item in self.featureList.iterrows():
            imWin = self.getImPointWin(item['geometry'])
            im = self.srcs.read(window=imWin)
            yield im


class MultiHeadDatasets():
    def __init__(self, datasetDict, multiSorceWeightDataset=None, class_dict=None, name=None) -> None:
        self.subDatasetNames = datasetDict.keys()
        self.subDatasets: list[rsDataset] = datasetDict
        self.weightDS = multiSorceWeightDataset
        self.name = name
        self.class_dict = class_dict

    def fromShapefile(self, shpfileName, shuffle=False, BATCH_SIZE=1, cache=False, split='train', cleanFunction=None):
        self.featureList = gpd.read_file(shpfileName)
        if self.featureList.crs.to_epsg() != 4326:
            raise ValueError('shape file need projection of epsg:4326')
        self.featureList = self.featureList.loc[self.featureList['split'] == split]
        if shuffle:
            self.featureList = self.featureList.sample(frac=1.0)
        if self.class_dict is None:
            self.class_names = self.featureList['class'].unique()
            self.class_dict = {n: i for i, n in enumerate(self.class_names)}
        dataset = tf.data.Dataset.from_generator(
            self.genSampleShapefile,
            output_signature=(
                {k: self.subDatasets[k].FeatureSpec for k in self.subDatasetNames},
                tf.TensorSpec(shape=(), dtype=tf.uint8),
                tf.TensorSpec(shape=(), dtype=tf.float32)
            ))

        return dataset  # .prefetch(200)

    def genSampleShapefile(self, featureList=None, return_property=False, classmap=True):
        # featureList 为单独调用使用
        if featureList is not None:
            self.featureList = featureList
        for i, row in self.featureList.iterrows():
            geoP = row['geometry']
            classID = row['class']
            weight = 1
            if 'weight' in row.keys():
                weight = row['weight']
            if return_property:
                yield (
                    {k: self.subDatasets[k].getPointFeature(
                        geoP) for k in self.subDatasetNames},
                    self.class_dict[classID] if classmap else classID,
                    weight if self.weightDS is None else self.weightDS.getWeight(
                        geoP),
                    row['GID'], row
                )
            else:
                yield (
                    {k: self.subDatasets[k].getPointFeature(
                        geoP) for k in self.subDatasetNames},
                )


    def predict_dataset_rect(self, bound, operating_res, batch_size=None):
        # print('test>>>>>>>>>>>>>>>>>>>')
        def merge(*x):
            oo = {
                list(self.subDatasetNames)[i]: tf.squeeze(x[i]) for i in range(len(list(self.subDatasetNames)))
            }
            return oo
        datasets = tuple(
            self.subDatasets[k].predict_dataset_rect(bound, operating_res, batch_size) for k in self.subDatasetNames
        )
        dataset = tf.data.Dataset.zip(datasets)
        dataset = dataset.map(merge, tf.data.AUTOTUNE)
        if batch_size is not None:
            dataset = dataset.batch(batch_size)
        return dataset


    def fromSlides(self, geoBounds):
        pass

    def drillByLongLat(self):
        pass

    def drillByXY(self):
        pass

def getImWinFromCenter(src, p: Point, halfPatchWin):
    [x], [y] = transform(rio.crs.CRS.from_epsg(4326), src.crs, [p.x], [p.y])
    rowC, colC = rowcol(src.transform, x, y)
    imWin = ((rowC-halfPatchWin, rowC+halfPatchWin+1),
             (colC-halfPatchWin, colC+halfPatchWin+1))
    return imWin


def drillPointOne(gdfrow, imPath=None):
    i, gdfrow = gdfrow
    src = rio.open(imPath)
    imWin = getImWinFromCenter(src, gdfrow['geometry'], 0)
    im = src.read(window=imWin).flatten()
    return im

def getTemporalWinDataset(x_dict, start_i=None, temporalWindow=None):
    # x_dict.shape = [times, rows, cols, bands]
    return x_dict[start_i:start_i+temporalWindow]