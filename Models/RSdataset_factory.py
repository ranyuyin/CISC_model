from __future__ import absolute_import
from __future__ import division
# from __future__ import google_type_annotations
from __future__ import print_function
from absl import logging
from dataclasses import dataclass, field, asdict
import tensorflow as tf
import tensorflow_datasets as tfds
from official.vision.image_classification import dataset_factory
from official.vision.image_classification.dataset_factory import DatasetConfig
from official.vision.image_classification import preprocessing
from official.vision.image_classification import augment
import RSaugment
from typing import Any, List, Optional, Tuple, Mapping, Union
import RSdataset as rsd
import pandas as pd
from functools import partial
import numpy as np

MEAN_RGB = {'gf':(82.42709, 85.72802, 71.39721)}
STDDEV_RGB = {'gf':(44.3095678, 42.571138,  42.624498)}

MEAN_BAND = {
    'ldband':(0.08301989, 0.11843648, 0.13163911, 0.21340215, 0.20781338,
       0.16904424),
    'dem':(1062.1642,6.22126,171.19586)
}
STDDEV_BAND = {
    'ldband':(0.06089173, 0.0668469 , 0.08094209, 0.08961518, 0.10137873,
       0.09523851),
    'dem':(1496.6609,8.689172,247.09402)
}
import os

feature_dataset_map = {
    'gf': rsd.DataGF,
}

RSAUGMENTERS = {
    'rsrandaugment': RSaugment.RSRandAugment,
}
@dataclass
class RSAugmentConfig(dataset_factory.AugmentConfig):
  def build(self) -> augment.ImageAugment:
    """Build the augmenter using this config."""
    params = self.params or {}
    augmenter = RSAUGMENTERS.get(self.name, None)
    return augmenter(**params) if augmenter is not None else None


@dataclass
class MultiHeaderConfig(DatasetConfig):
    """The base RS MultiHeader dataset config."""
    name: str = 'MultiHeader'
    # Note: for large datasets like ImageNet, using records is faster than tfds
    builder: str = 'generator'
    batch_size: int = 100
    sample_path: Optional[str] = None
    input_features: Optional[list] = None
    rs_srcs: Optional[dict] = None



###-----TRAIN Dataset------>>>>>>>>>
@dataclass
class RSTrainDatasetConfig(DatasetConfig):
    augmenter: RSAugmentConfig = RSAugmentConfig()
    builder: str = 'records'
    batch_size: int = 300
    # is_shuffle: bool = False
    weight_df_name: Optional[str] = None
    class_df_name: Optional[str] = None
    use_weight:Optional[bool] = False
    class_weight: Optional[dict] = None

@dataclass
class RSTrainDatasetConfig_gf(RSTrainDatasetConfig):
    name: str = 'gf'
    shuffle_buffer_size: int = 20000
    num_examples: int = 180000
    num_classes: int = 4

@dataclass
class RSTrainDatasetConfig_gfldbanddem(RSTrainDatasetConfig_gf):
    name: str = 'gfldbanddem'

###-----TRAIN Dataset------<<<<<<<<<<<


#####----------------EVAL Dataset------->>>>>>>>>>>>>
@dataclass
class RSEvalDatasetConfig_gf(RSTrainDatasetConfig):
    name: str = 'gf'
    num_examples: int = 20000
    num_classes: int = 4

@dataclass
class RSEvalDatasetConfig_gfldbanddem(RSEvalDatasetConfig_gf):
    name: str = 'gfldbanddem'

#####------------------EVAL Dataset-------<<<<<<<<<<<


def parse_id(feature_dict, **kwargs)-> tf.Tensor:
    return feature_dict['id']

def parse_gf(
    feature_dict,
    is_training: bool = False,
    augmenter: Optional[augment.ImageAugment] = None,
    mean_subtract: bool = False,
    standardize: bool = False,
    dtype: tf.dtypes.DType = tf.float32
    ) -> tf.Tensor:
    encoded = feature_dict['image_raw_GF']
    im_size = feature_dict['GF_im_size']
    if im_size > 0:
        height, width = im_size, im_size
    else:
        height, width = feature_dict['GF_height'], feature_dict['GF_width']
    decoded = tf.reshape(tf.io.decode_raw(
        encoded, tf.uint8), [height, width, 3])
    if is_training:
        decoded = tf.image.random_flip_left_right(decoded)
        decoded = tf.image.random_flip_up_down(decoded)
        if augmenter is not None:
            decoded = augmenter.distort(decoded)
    decoded = tf.cast(decoded,tf.float32)
    if mean_subtract:
        decoded = preprocessing.mean_image_subtraction(image_bytes=decoded, means=MEAN_RGB['gf'])
    if standardize:
        decoded = preprocessing.standardize_image(image_bytes=decoded, stddev=STDDEV_RGB['gf'])
    # decoded = decoded/255
    if dtype is not None:
        decoded = tf.image.convert_image_dtype(decoded, dtype)
    return decoded

def parse_ldband(
    feature_dict,
    is_training: bool = False,
    augmenter: Optional[augment.ImageAugment] = None,
    mean_subtract: bool = False,
    standardize: bool = False,
    dtype: tf.dtypes.DType = tf.float32
    ) -> tf.Tensor:
    encoded = feature_dict['image_raw_ld20']
    decoded = tf.io.decode_raw(encoded, tf.uint16)
    decoded = tf.reshape(decoded,[-1])
    decoded = tf.slice(tf.cast(decoded, dtype),[0],[6])*tf.constant(0.0000275)-tf.constant(0.2)
    if mean_subtract:
        decoded = decoded-tf.constant(MEAN_BAND['ldband'])
    if standardize:
        decoded = decoded / STDDEV_BAND['ldband']
    return decoded

def parse_dem(
    feature_dict,
    is_training: bool = False,
    augmenter: Optional[augment.ImageAugment] = None,
    mean_subtract: bool = False,
    standardize: bool = False,
    dtype: tf.dtypes.DType = tf.float32
    ) -> tf.Tensor:
    encoded = feature_dict['image_raw_dem']
    decoded = tf.io.decode_raw(encoded, tf.int16)
    decoded = tf.reshape(decoded,[-1])
    decoded = tf.slice(tf.cast(decoded, dtype),[0],[3])
    if mean_subtract:
        decoded = decoded-tf.constant(MEAN_BAND['dem'])
    if standardize:
        decoded = decoded / STDDEV_BAND['dem']
    return decoded

class RSDatasetBuilder:
    """An object for building datasets.

    Allows building various pipelines fetching examples, preprocessing, etc.
    Maintains additional state information calculated from the dataset, i.e.,
    training set split, batch size, and number of steps (batches).
    """
    keys_to_features_dict = {
        'gfldbanddem':{
            'id': tf.io.FixedLenFeature([], tf.int64, -1),
            'x': tf.io.FixedLenFeature([], tf.float32, -1),
            'y': tf.io.FixedLenFeature([], tf.float32, -1),
            'image_raw_GF': tf.io.FixedLenFeature((), tf.string, ''),
            'image_raw_ld20': tf.io.FixedLenFeature((), tf.string, ''),
            'image_raw_dem': tf.io.FixedLenFeature((), tf.string, ''),
            'class': tf.io.FixedLenFeature([], tf.int64, -1),
            'weight': tf.io.FixedLenFeature([], tf.float32, 1),
            'GF_height': tf.io.FixedLenFeature([], tf.int64, -1),
            'GF_width': tf.io.FixedLenFeature([], tf.int64, -1),
            'GF_im_size': tf.io.FixedLenFeature([], tf.int64, -1),
        },

    }
    featrues_function_dict = {
        'gfldbanddem':{
            'gf':parse_gf,
            'ldband':parse_ldband,
            'dem': parse_dem,
            'id':parse_id
        },
    }
    augmenter_todolist = ['gf']
    weight_df = None
    class_df = None
    
    def __init__(
        self, config: DatasetConfig, 
        # weight_df_name:str=None, 
        # class_df_name:str=None, 
        **overrides: Any
        ):
        """Initialize the builder from the config."""
        self.config = config.replace(**overrides)
        self.builder_info = None
        if self.config.weight_df_name is not None:
            self.weight_df = pd.read_csv(self.config.weight_df_name,index_col='GID')
        if self.config.class_df_name is not None:
            self.class_df = pd.read_csv(self.config.class_df_name, index_col='GID')
        if config.augmenter is not None:
            logging.info('Using augmentation: %s', self.config.augmenter.name)
            self.augmenter = config.augmenter.build()
        else:
            self.augmenter = None
    
    @property
    def is_training(self) -> bool:
        """Whether this is the training set."""
        return self.config.split == 'train'

    @property
    def batch_size(self) -> int:
        """The batch size, multiplied by the number of replicas (if configured)."""
        if self.config.use_per_replica_batch_size:
            return self.config.batch_size * self.config.num_devices
        else:
            return self.config.batch_size

    @property
    def global_batch_size(self):
        """The global batch size across all replicas."""
        return self.batch_size

    @property
    def local_batch_size(self):
        """The base unscaled batch size."""
        if self.config.use_per_replica_batch_size:
            return self.config.batch_size
        else:
            return self.config.batch_size // self.config.num_devices

    @property
    def num_steps(self) -> int:
        """The number of steps (batches) to exhaust this dataset."""
        # Always divide by the global batch size to get the correct # of steps
        return self.num_examples // self.global_batch_size

    @property
    def dtype(self) -> tf.dtypes.DType:
        """Converts the config's dtype string to a tf dtype.

        Returns:
          A mapping from string representation of a dtype to the `tf.dtypes.DType`.

        Raises:
          ValueError if the config's dtype is not supported.

        """
        dtype_map = {
            'float32': tf.float32,
            'bfloat16': tf.bfloat16,
            'float16': tf.float16,
            'fp32': tf.float32,
            'bf16': tf.bfloat16,
        }
        try:
            return dtype_map[self.config.dtype]
        except:
            raise ValueError('Invalid DType provided. Supported types: {}'.format(
                dtype_map.keys()))

    @property
    def image_size(self) -> int:
        """The size of each image (can be inferred from the dataset)."""

        if self.config.image_size == 'infer':
            return self.info.features['image'].shape[0]
        else:
            return int(self.config.image_size)

    @property
    def num_channels(self) -> int:
        """The number of image channels (can be inferred from the dataset)."""
        if self.config.num_channels == 'infer':
            return self.info.features['image'].shape[-1]
        else:
            return int(self.config.num_channels)

    @property
    def num_examples(self) -> int:
        """The number of examples (can be inferred from the dataset)."""
        if self.config.num_examples == 'infer':
            return self.info.splits[self.config.split].num_examples
        else:
            return int(self.config.num_examples)

    @property
    def num_classes(self) -> int:
        """The number of classes (can be inferred from the dataset)."""
        if self.config.num_classes == 'infer':
            return self.info.features['label'].num_classes
        else:
            return int(self.config.num_classes)

    @property
    def info(self) -> tfds.core.DatasetInfo:
        """The TFDS dataset info, if available."""
        try:
            if self.builder_info is None:
                self.builder_info = tfds.builder(self.config.name).info
        except ConnectionError as e:
            logging.error('Failed to use TFDS to load info. Please set dataset info '
                          '(image_size, num_channels, num_examples, num_classes) in '
                          'the dataset config.')
            raise e
        return self.builder_info

    def build(
            self,
            strategy: Optional[tf.distribute.Strategy] = None) -> tf.data.Dataset:
        """Construct a dataset end-to-end and return it using an optional strategy.

        Args:
          strategy: a strategy that, if passed, will distribute the dataset
            according to that strategy. If passed and `num_devices > 1`,
            `use_per_replica_batch_size` must be set to `True`.

        Returns:
          A TensorFlow dataset outputting batched images and labels.
        """
        if strategy:
            if strategy.num_replicas_in_sync != self.config.num_devices:
                logging.warn(
                    'Passed a strategy with %d devices, but expected'
                    '%d devices.', strategy.num_replicas_in_sync,
                    self.config.num_devices)
            dataset = strategy.distribute_datasets_from_function(self._build)
        else:
            dataset = self._build()

        return dataset

    def _build(
        self,
        input_context: Optional[tf.distribute.InputContext] = None
    ) -> tf.data.Dataset:
        """Construct a dataset end-to-end and return it.

        Args:
          input_context: An optional context provided by `tf.distribute` for
            cross-replica training.

        Returns:
          A TensorFlow dataset outputting batched images and labels.
        """
        builders = {
            'tfds': self.load_tfds,
            'records': self.load_records,
            'synthetic': self.load_synthetic,
            'generator': self.load_generator,
        }

        builder = builders.get(self.config.builder, None)

        if builder is None:
            raise ValueError(
                'Unknown builder type {}'.format(self.config.builder))

        self.input_context = input_context
        dataset = builder()
        dataset = self.pipeline(dataset)

        return dataset

    def load_tfds(self) -> tf.data.Dataset:
        """Return a dataset loading files from TFDS."""

        logging.info('Using TFDS to load data.')

        builder = tfds.builder(self.config.name, data_dir=self.config.data_dir)

        if self.config.download:
            builder.download_and_prepare()

        decoders = {}

        if self.config.skip_decoding:
            decoders['image'] = tfds.decode.SkipDecoding()

        read_config = tfds.ReadConfig(
            interleave_cycle_length=10,
            interleave_block_length=1,
            input_context=self.input_context)

        dataset = builder.as_dataset(
            split=self.config.split,
            as_supervised=True,
            shuffle_files=True,
            decoders=decoders,
            read_config=read_config)

        return dataset

    def load_records(self) -> tf.data.Dataset:
        """Return a dataset loading files with TFRecords."""
        logging.info('Using TFRecords to load data.')
        if self.config.filenames is None:
            if self.config.data_dir is None:
                raise ValueError(
                    'Dataset must specify a path for the data files.')

            file_pattern = os.path.join(self.config.data_dir,
                                        '{}*'.format(self.config.split))
            dataset = tf.data.Dataset.list_files(file_pattern, shuffle=False)
        else:
            dataset = tf.data.Dataset.from_tensor_slices(self.config.filenames)

        return dataset

    def load_synthetic(self) -> tf.data.Dataset:
        """Return a dataset generating dummy synthetic data."""
        logging.info('Generating a synthetic dataset.')

        def generate_data(_):
            image = tf.zeros([self.image_size, self.image_size, self.num_channels],
                             dtype=self.dtype)
            label = tf.zeros([1], dtype=tf.int32)
            return image, label

        dataset = tf.data.Dataset.range(1)
        dataset = dataset.repeat()
        dataset = dataset.map(
            generate_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        return dataset

    def load_generator(self) -> tf.data.Dataset:
        """Return a dataset from generator."""
        logging.info('Using a generator to load dataset.')
        dsd = {
            k:feature_dataset_map[k](self.config.rs_srcs.as_dict()[k],rescaleFunc=lambda x:x/255) \
                for k in self.config.input_features}
        MDS = rsd.MultiHeadDatasets(dsd)
        return MDS.fromShapefile(self.config.sample_path, split=self.config.split)

    def pipeline(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        """Build a pipeline fetching, shuffling, and preprocessing the dataset.

        Args:
          dataset: A `tf.data.Dataset` that loads raw files.

        Returns:
          A TensorFlow dataset outputting batched images and labels.
        """
        if (self.config.builder != 'tfds' and self.input_context and
                self.input_context.num_input_pipelines > 1):
            dataset = dataset.shard(self.input_context.num_input_pipelines,
                                    self.input_context.input_pipeline_id)
            logging.info(
                'Sharding the dataset: input_pipeline_id=%d '
                'num_input_pipelines=%d', self.input_context.num_input_pipelines,
                self.input_context.input_pipeline_id)

        if self.is_training and self.config.builder == 'records':
            # Shuffle the input files.
            dataset.shuffle(buffer_size=self.config.file_shuffle_buffer_size)
            logging.info('shuffle tfrecords.')

        if self.is_training and not self.config.cache:
        # if not self.config.cache:
            dataset = dataset.repeat()

        if self.config.builder == 'records':
            # Read the data from disk in parallel
            dataset = dataset.interleave(
                tf.data.TFRecordDataset,
                cycle_length=10,
                block_length=1,
                num_parallel_calls=tf.data.experimental.AUTOTUNE)

        if self.config.cache:
            dataset = dataset.cache()

        if self.is_training:
            dataset = dataset.shuffle(self.config.shuffle_buffer_size)
            logging.info('shuffle samples.')
            dataset = dataset.repeat()

        # Parse, pre-process, and batch the data in parallel
        if self.config.builder == 'records':
            preprocess = self.parse_record
            dataset = dataset.map(
                preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        if self.input_context and self.config.num_devices > 1:
            if not self.config.use_per_replica_batch_size:
                raise ValueError(
                    'The builder does not support a global batch size with more than '
                    'one replica. Got {} replicas. Please set a '
                    '`per_replica_batch_size` and enable '
                    '`use_per_replica_batch_size=True`.'.format(
                        self.config.num_devices))

            # The batch size of the dataset will be multiplied by the number of
            # replicas automatically when strategy.distribute_datasets_from_function
            # is called, so we use local batch size here.
            dataset = dataset.batch(
                self.local_batch_size, drop_remainder=self.is_training)
        else:
            dataset = dataset.batch(
                self.global_batch_size, drop_remainder=self.is_training)

        # Prefetch overlaps in-feed with training
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        if self.config.tf_data_service:
            if not hasattr(tf.data.experimental, 'service'):
                raise ValueError('The tf_data_service flag requires Tensorflow version '
                                 '>= 2.3.0, but the version is {}'.format(
                                     tf.__version__))
            dataset = dataset.apply(
                tf.data.experimental.service.distribute(
                    processing_mode='parallel_epochs',
                    service=self.config.tf_data_service,
                    job_name='resnet_train'))
            dataset = dataset.prefetch(
                buffer_size=tf.data.experimental.AUTOTUNE)

        return dataset

    def parse_record(self, record: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Parse an ImageNet record from a serialized string Tensor."""
        keys_to_features = self.keys_to_features_dict[self.config.name]
        parsed = tf.io.parse_single_example(record, keys_to_features)
        features = {
            k:f(
                parsed,
                is_training=self.is_training,
                augmenter=self.augmenter if k in self.augmenter_todolist else None,
                mean_subtract=self.config.mean_subtract,
                standardize=self.config.standardize,
                ) for k,f in self.featrues_function_dict[self.config.name].items()}

        def _weightByid(gid):
            try:
                w = self.weight_df.loc[gid,'weight'].astype('float32')
            except:
                w = np.float32(0.0)
            return w
        @tf.function(input_signature=[tf.TensorSpec(None, tf.int64)])
        def weightByid(gid):
            return tf.numpy_function(_weightByid, [gid], tf.float32)
        def _labelByid(gid):
            try:
                l = self.class_df.loc[gid,'class'].astype('int32')
            except:
                l = np.int32(0)
            return l
        @tf.function(input_signature=[tf.TensorSpec(None, tf.int64)])
        def labelByid(gid):
            return tf.numpy_function(_labelByid, [gid], tf.int32)    

        this_id = parsed['id']
        if self.weight_df is not None:
            weight = parsed['weight'] * weightByid(this_id)
        else:
            weight = parsed['weight']
        if self.class_df is not None:
            label = labelByid(this_id)
        else:
            label = tf.cast(parsed['class'], tf.int32)
        label = tf.reshape(label, shape=[1])
        weight = tf.reshape(weight, shape=[1])


        return features, label, weight


    @classmethod
    def from_params(cls, *args, **kwargs):
        """Construct a dataset builder from a default config and any overrides."""
        config = DatasetConfig.from_args(*args, **kwargs)
        return cls(config)
