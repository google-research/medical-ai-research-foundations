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

"""The main training pipeline."""
import json
import math
import os

from absl import app
from absl import flags
from . import data as data_lib
from . import data_util
from . import model as model_lib
from . import model_util
from . import resnet
import tensorflow.compat.v1 as tf
import tensorflow_datasets as tfds
from tensorflow_estimator.compat.v1 import estimator as tf_estimator  # pylint: disable=g-deprecated-tf-checker
import tensorflow_hub as hub

FLAGS = flags.FLAGS

_LEARNING_RATE = flags.DEFINE_float(
    'learning_rate', 0.3, 'Initial learning rate per batch size of 256.'
)

_WARMUP_EPOCHS = flags.DEFINE_float(
    'warmup_epochs', 10, 'Number of epochs of warmup.'
)

_WEIGHT_DECAY = flags.DEFINE_float(
    'weight_decay', 1e-6, 'Amount of weight decay to use.'
)

_BATCH_NORM_DECAY = flags.DEFINE_float(
    'batch_norm_decay', 0.9, 'Batch norm decay parameter.'
)

_TRAIN_BATCH_SIZE = flags.DEFINE_integer(
    'train_batch_size', 512, 'Batch size for training.'
)

_TRAIN_SPLIT = flags.DEFINE_string(
    'train_split', 'train', 'Split for training.'
)

_TRAIN_EPOCHS = flags.DEFINE_integer(
    'train_epochs', 100, 'Number of epochs to train for.'
)

_TRAIN_STEPS = flags.DEFINE_integer(
    'train_steps',
    0,
    'Number of steps to train for. If provided, overrides train_epochs.',
)

_EVAL_BATCH_SIZE = flags.DEFINE_integer(
    'eval_batch_size', 256, 'Batch size for eval.'
)

_TRAIN_SUMMARY_STEPS = flags.DEFINE_integer(
    'train_summary_steps',
    100,
    'Steps before saving training summaries. If 0, will not save.',
)

_CHECKPOINT_EPOCHS = flags.DEFINE_integer(
    'checkpoint_epochs', 1, 'Number of epochs between checkpoints/summaries.'
)

_CHECKPOINT_STEPS = flags.DEFINE_integer(
    'checkpoint_steps',
    0,
    (
        'Number of steps between checkpoints/summaries. If provided, overrides '
        'checkpoint_epochs.'
    ),
)

_EVAL_SPLIT = flags.DEFINE_string(
    'eval_split', 'validation', 'Split for evaluation.'
)

_DATASET = flags.DEFINE_string(
    'dataset', 'cifar10', 'Name of a dataset to load from a TFDS builder.'
)

_CACHE_DATASET = flags.DEFINE_bool(
    'cache_dataset',
    False,
    (
        'Whether to cache the entire dataset in memory. If the dataset is '
        'ImageNet, this is a very bad idea, but for smaller datasets it can '
        'improve performance.'
    ),
)

_MODE = flags.DEFINE_enum(
    'mode',
    'train',
    ['train', 'eval', 'train_then_eval'],
    'Whether to perform training or evaluation.',
)

_TRAIN_MODE = flags.DEFINE_enum(
    'train_mode',
    'pretrain',
    ['pretrain', 'finetune'],
    'The train mode controls different objectives and trainable components.',
)

_CHECKPOINT = flags.DEFINE_string(
    'checkpoint',
    None,
    'Loading from the given checkpoint for continued training or fine-tuning.',
)

_VARIABLE_SCHEMA = flags.DEFINE_string(
    'variable_schema',
    '?!global_step',
    'This defines whether some variable from the checkpoint should be loaded.',
)

_ZERO_INIT_LOGITS_LAYER = flags.DEFINE_bool(
    'zero_init_logits_layer',
    False,
    'If True, zero initialize layers after avg_pool for supervised learning.',
)

_FINE_TUNE_AFTER_BLOCK = flags.DEFINE_integer(
    'fine_tune_after_block',
    -1,
    (
        'The layers after which block that we will fine-tune. -1 means'
        ' fine-tuning everything. 0 means fine-tuning after stem block. 4 means'
        ' fine-tuning just the linear head.'
    ),
)

_TF_RUNNER = flags.DEFINE_string(
    'tf_runner',
    None,
    (
        'Address/name of the TensorFlow runner to use. By default, use an '
        'in-process runner.'
    ),
)

_MODEL_DIR = flags.DEFINE_string(
    'model_dir', None, 'Model directory for training.'
)

_DATA_DIR = flags.DEFINE_string(
    'data_dir', None, 'Directory where dataset is stored.'
)

_USE_TPU = flags.DEFINE_bool('use_tpu', True, 'Whether to run on TPU.')

tf.flags.DEFINE_string(
    'tpu_name', None,
    'The Cloud TPU to use for training. This should be either the name '
    'used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 '
    'url.')

tf.flags.DEFINE_string(
    'tpu_zone', None,
    '[Optional] GCE zone where the Cloud TPU is located in. If not '
    'specified, we will attempt to automatically detect the GCE project from '
    'metadata.')

tf.flags.DEFINE_string(
    'gcp_project', None,
    '[Optional] Project name for the Cloud TPU-enabled project. If not '
    'specified, we will attempt to automatically detect the GCE project from '
    'metadata.')

_OPTIMIZER = flags.DEFINE_enum(
    'optimizer', 'lars', ['momentum', 'adam', 'lars'], 'Optimizer to use.'
)

_MOMENTUM = flags.DEFINE_float('momentum', 0.9, 'Momentum parameter.')

_EVAL_NAME = flags.DEFINE_string('eval_name', None, 'Name for eval.')

_KEEP_CHECKPOINT_MAX = flags.DEFINE_integer(
    'keep_checkpoint_max', 5, 'Maximum number of checkpoints to keep.'
)

_KEEP_HUB_MODULE_MAX = flags.DEFINE_integer(
    'keep_hub_module_max', 1, 'Maximum number of Hub modules to keep.'
)

_TEMPERATURE = flags.DEFINE_float(
    'temperature', 0.1, 'Temperature parameter for contrastive loss.'
)

_HIDDEN_NORM = flags.DEFINE_boolean(
    'hidden_norm', True, 'Temperature parameter for contrastive loss.'
)

_HEAD_PROJ_MODE = flags.DEFINE_enum(
    'head_proj_mode',
    'nonlinear',
    ['none', 'linear', 'nonlinear'],
    'How the head projection is done.',
)

_HEAD_PROJ_DIM = flags.DEFINE_integer(
    'head_proj_dim', 128, 'Number of head projection dimension.'
)

_NUM_NLH_LAYERS = flags.DEFINE_integer(
    'num_nlh_layers', 1, 'Number of non-linear head layers.'
)

_GLOBAL_BN = flags.DEFINE_boolean(
    'global_bn',
    True,
    'Whether to aggregate BN statistics across distributed cores.',
)

_WIDTH_MULTIPLIER = flags.DEFINE_integer(
    'width_multiplier', 1, 'Multiplier to change width of network.'
)

_RESNET_DEPTH = flags.DEFINE_integer('resnet_depth', 50, 'Depth of ResNet.')

_USE_BIT = flags.DEFINE_boolean('use_bit', False, 'Whether to use BiT.')

_IMAGE_SIZE = flags.DEFINE_integer('image_size', 224, 'Input image size.')

_COLOR_JITTER_STRENGTH = flags.DEFINE_float(
    'color_jitter_strength', 1.0, 'The strength of color jittering.'
)

_USE_ELASTIC_DEFORM = flags.DEFINE_boolean(
    'use_elastic_deform', False, 'Whether or not to use elastic deformation.'
)

_EQUALIZE_HISTOGRAM = flags.DEFINE_boolean(
    'equalize_histogram',
    False,
    'Whether to equalize a histogram. This is only for greyscale images.',
)

_USE_BLUR = flags.DEFINE_boolean(
    'use_blur',
    True,
    'Whether or not to use Gaussian blur for augmentation during pretraining.',
)

_ROTATION_RANGE = flags.DEFINE_float(
    'rotation_range', 0.0, 'The rotation range.'
)

_MAX_CONTRAST_DISTORT = flags.DEFINE_float(
    'max_contrast_distort', 0.8, 'The amount of contrast to use.'
)

_MAX_BRIGHTNESS_DISTORT = flags.DEFINE_float(
    'max_brightness_distort',
    0.8,
    'The maximum amount of brightness to distort.',
)

_IMAGENET_CKPT = flags.DEFINE_string(
    'imagenet_ckpt',
    None,
    'to continue the training from an imagenet checkpoint.',
)

_RESCALE_AT_INIT = flags.DEFINE_float(
    'rescale_at_init',
    1.0,
    'Amount to rescale kernel weights at the initialization.',
)

_VERBOSE = flags.DEFINE_boolean(
    'verbose',
    True,
    'Whether or not to output extralogging info such as input images.',
)

_TF_DATA_SERVICE_ADDRESS = flags.DEFINE_string(
    'tf_data_service_address',
    '',
    'If given, will use the data service to distribute data import.',
)

_MULTI_INSTANCE = flags.DEFINE_boolean(
    'multi_instance',
    False,
    'If True, MICLe is activated for the positive pair selection.',
)


def build_hub_module(model, num_classes, global_step, checkpoint_path):
  """Create TF-Hub module."""

  tags_and_args = [
      # The default graph is built with batch_norm, dropout etc. in inference
      # mode. This graph version is good for inference, not training.
      ([], {'is_training': False}),
      # A separate "train" graph builds batch_norm, dropout etc. in training
      # mode.
      (['train'], {'is_training': True}),
  ]

  def module_fn_simclr(is_training):
    """Function that builds TF-Hub module for SimCLR models."""
    endpoints = {}
    inputs = tf.placeholder(
        tf.float32, [None, _IMAGE_SIZE.value, _IMAGE_SIZE.value, 3]
    )
    with tf.variable_scope('base_model', reuse=tf.AUTO_REUSE):
      hiddens = model(inputs, is_training)
      for v in [
          'initial_conv',
          'initial_max_pool',
          'block_group1',
          'block_group2',
          'block_group3',
          'block_group4',
          'final_avg_pool',
      ]:
        endpoints[v] = tf.get_default_graph().get_tensor_by_name(
            'base_model/{}:0'.format(v))
    if _TRAIN_MODE.value == 'pretrain':
      hiddens_proj = model_util.projection_head(hiddens, is_training)
      endpoints['proj_head_input'] = hiddens
      endpoints['proj_head_output'] = hiddens_proj
    else:
      logits_sup = model_util.supervised_head(hiddens, num_classes, is_training)
      endpoints['logits_sup'] = logits_sup
    hub.add_signature(
        inputs=dict(images=inputs), outputs=dict(endpoints, default=hiddens)
    )

  def module_fn_simclr_bit(is_training):
    """Function that builds TF-Hub module for SimCLR+BiT models."""
    endpoints = {}
    inputs = tf.placeholder(
        tf.float32, [None, _IMAGE_SIZE.value, _IMAGE_SIZE.value, 3]
    )

    def bit_generator(depth, width_multiplier):
      bit_model_archs = [
          (50, 1),
          (50, 3),
          (101, 1),
          (101, 3),
          (152, 2),
          (152, 4),
      ]
      assert (depth, width_multiplier) in bit_model_archs, (
          'There is no SimCLR+BiT model architecture for '
          'the requested model depth and width multiplier: '
          '({},{}). Valid model combinations are: {}'.format(
              depth, width_multiplier, bit_model_archs
          )
      )
      model_name = (
          f'https://tfhub.dev/google/bit/m-r{depth}x{width_multiplier}/1'
      )

      pre_logits, endpoints = resnet.bit.bit_embedding(
          inputs, model_name=model_name, trainable=is_training, scope_name='bit'
      )
      return pre_logits, endpoints

    with tf.variable_scope('base_model', reuse=tf.AUTO_REUSE):
      hiddens, endpoints = bit_generator(
          _RESNET_DEPTH.value, _WIDTH_MULTIPLIER.value
      )
    if _TRAIN_MODE.value == 'pretrain':
      hiddens_proj = model_util.projection_head(hiddens, is_training)
      endpoints['proj_head_input'] = hiddens
      endpoints['proj_head_output'] = hiddens_proj
    else:
      logits_sup = model_util.supervised_head(hiddens, num_classes, is_training)
      endpoints['logits_sup'] = logits_sup
    hub.add_signature(
        inputs=dict(images=inputs), outputs=dict(endpoints, default=hiddens)
    )

  # Drop the non-supported non-standard graph collection.
  drop_collections = ['trainable_variables_inblock_%d' % d for d in range(6)]

  if _USE_BIT.value:
    spec = hub.create_module_spec(
        module_fn_simclr_bit, tags_and_args, drop_collections
    )
  else:
    spec = hub.create_module_spec(
        module_fn_simclr, tags_and_args, drop_collections
    )
  hub_export_dir = os.path.join(_MODEL_DIR.value, 'hub')
  checkpoint_export_dir = os.path.join(hub_export_dir, str(global_step))
  if tf.io.gfile.exists(checkpoint_export_dir):
    # Do not save if checkpoint already saved.
    tf.io.gfile.rmtree(checkpoint_export_dir)
  spec.export(
      checkpoint_export_dir,
      checkpoint_path=checkpoint_path,
      name_transform_fn=None)

  if _KEEP_HUB_MODULE_MAX.value > 0:
    # Delete old exported Hub modules.
    exported_steps = []
    for subdir in tf.io.gfile.listdir(hub_export_dir):
      if not subdir.isdigit():
        continue
      exported_steps.append(int(subdir))
    exported_steps.sort()
    for step_to_delete in exported_steps[: -_KEEP_HUB_MODULE_MAX.value]:
      tf.io.gfile.rmtree(os.path.join(hub_export_dir, str(step_to_delete)))


def perform_evaluation(
    estimator, input_fn, eval_steps, model, num_classes, checkpoint_path=None
):
  """Perform evaluation.

  Args:
    estimator: TPUEstimator instance.
    input_fn: Input function for estimator.
    eval_steps: Number of steps for evaluation.
    model: Instance of transfer_learning.models.Model.
    num_classes: Number of classes to build model for.
    checkpoint_path: Path of checkpoint to evaluate.

  Returns:
    result: A Dict of metrics and their values.
  """
  if not checkpoint_path:
    checkpoint_path = estimator.latest_checkpoint()
  result = estimator.evaluate(
      input_fn,
      eval_steps,
      checkpoint_path=checkpoint_path,
      name=_EVAL_NAME.value,
  )

  # Record results as JSON.
  result_json_path = os.path.join(_MODEL_DIR.value, 'result.json')
  with tf.io.gfile.GFile(result_json_path, 'w') as f:
    json.dump({k: float(v) for k, v in result.items()}, f)
  result_json_path = os.path.join(
      _MODEL_DIR.value, 'result_%d.json' % result['global_step']
  )
  with tf.io.gfile.GFile(result_json_path, 'w') as f:
    json.dump({k: float(v) for k, v in result.items()}, f)
  flag_json_path = os.path.join(_MODEL_DIR.value, 'flags.json')
  with tf.io.gfile.GFile(flag_json_path, 'w') as f:
    json.dump(FLAGS.flag_values_dict(), f)

  # Save Hub module.
  build_hub_module(
      model,
      num_classes,
      global_step=result['global_step'],
      checkpoint_path=checkpoint_path,
  )

  return result


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  # Enable training summary.
  if _TRAIN_SUMMARY_STEPS.value > 0:
    tf.config.set_soft_device_placement(True)

  # Use TFDS builder
  builder = tfds.builder(_DATASET.value, data_dir=_DATA_DIR.value)
  builder.download_and_prepare()
  num_train_examples = builder.info.splits[_TRAIN_SPLIT.value].num_examples
  num_eval_examples = builder.info.splits[_EVAL_SPLIT.value].num_examples
  num_classes = builder.info.features['label'].num_classes

  train_steps = model_util.get_train_steps(num_train_examples)
  eval_steps = int(math.ceil(num_eval_examples / _EVAL_BATCH_SIZE.value))
  epoch_steps = int(round(num_train_examples / _TRAIN_BATCH_SIZE.value))

  if _VERBOSE.value:
    tf.logging.info('===== Dataset Stats =====')
    tf.logging.info('Total number of examples:  %d', num_train_examples)
    tf.logging.info('Total number of classes: %d', num_classes)
    tf.logging.info('Total number of training steps: %d', train_steps)
    tf.logging.info('Total number of steps in an epoch: %d', epoch_steps)

  options = data_util.DistortionOptions()
  options.use_elastic_deform = _USE_ELASTIC_DEFORM.value
  options.max_brightness_distort = _MAX_BRIGHTNESS_DISTORT.value
  options.max_contrast_distort = _MAX_CONTRAST_DISTORT.value
  options.equalize_histogram = _EQUALIZE_HISTOGRAM.value
  options.use_blur = _USE_BLUR.value

  resnet.BATCH_NORM_DECAY = _BATCH_NORM_DECAY.value
  if _USE_BIT.value:
    model_ = resnet.resnet_v2(
        depth=_RESNET_DEPTH.value,
        width_multiplier=_WIDTH_MULTIPLIER.value,
        verify_input_range=False,
    )
    avg_pool_key = 'pre_logits'
  else:
    model_ = resnet.resnet_v1(
        resnet_depth=_RESNET_DEPTH.value,
        width_multiplier=_WIDTH_MULTIPLIER.value,
        cifar_stem=_IMAGE_SIZE.value <= 32,
        train_mode=_TRAIN_MODE.value,
        fine_tune_after_block=_FINE_TUNE_AFTER_BLOCK.value,
        global_bn=_GLOBAL_BN.value,
        batch_norm_decay=_BATCH_NORM_DECAY.value,
    )
    avg_pool_key = 'final_avg_pool'
  model = lambda x, is_training: model_(x, is_training)[avg_pool_key]

  checkpoint_steps = _CHECKPOINT_STEPS.value or (
      _CHECKPOINT_EPOCHS.value * epoch_steps
  )

  cluster = None
  if _USE_TPU.value and _TF_RUNNER.value is None:
    if FLAGS.tpu_name:
      cluster = tf.distribute.cluster_resolver.TPUClusterResolver(
          FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)
    else:
      cluster = tf.distribute.cluster_resolver.TPUClusterResolver()
      tf.config.experimental_connect_to_cluster(cluster)
      tf.tpu.experimental.initialize_tpu_system(cluster)

  default_eval_mode = tf_estimator.tpu.InputPipelineConfig.PER_HOST_V1
  sliced_eval_mode = tf_estimator.tpu.InputPipelineConfig.SLICED
  run_config = tf_estimator.tpu.RunConfig(
      tpu_config=tf_estimator.tpu.TPUConfig(
          iterations_per_loop=checkpoint_steps,
          eval_training_input_configuration=sliced_eval_mode
          if _USE_TPU.value
          else default_eval_mode,
      ),
      model_dir=_MODEL_DIR.value,
      save_summary_steps=checkpoint_steps,
      save_checkpoints_steps=checkpoint_steps,
      keep_checkpoint_max=_KEEP_CHECKPOINT_MAX.value,
      master=_TF_RUNNER.value,
      cluster=cluster,
  )
  estimator = tf_estimator.tpu.TPUEstimator(
      model_lib.build_model_fn(model, num_classes, num_train_examples),
      config=run_config,
      train_batch_size=_TRAIN_BATCH_SIZE.value,
      eval_batch_size=_EVAL_BATCH_SIZE.value,
      use_tpu=_USE_TPU.value,
  )

  if _MODE.value == 'eval':
    for ckpt in tf.train.checkpoints_iterator(
        run_config.model_dir, min_interval_secs=15):
      try:
        result = perform_evaluation(
            estimator=estimator,
            input_fn=data_lib.build_input_fn_for_builder(
                builder,
                False,
                cache_dataset=_CACHE_DATASET.value,
                image_size=_IMAGE_SIZE.value,
                color_jitter_strength=_COLOR_JITTER_STRENGTH.value,
                rotation_range=_ROTATION_RANGE.value,
                multi_instance=_MULTI_INSTANCE.value,
                options=options,
            ),
            eval_steps=eval_steps,
            model=model,
            num_classes=num_classes,
            checkpoint_path=ckpt,
        )
      except tf.errors.NotFoundError:
        continue
      if result['global_step'] >= train_steps:
        return
  else:  # Pretrain mode
    train_input_fn = data_lib.build_input_fn_for_builder(
        builder,
        True,
        cache_dataset=_CACHE_DATASET.value,
        image_size=_IMAGE_SIZE.value,
        color_jitter_strength=_COLOR_JITTER_STRENGTH.value,
        rotation_range=_ROTATION_RANGE.value,
        options=options,
    )
    estimator.train(train_input_fn, max_steps=train_steps)
    if _MODE.value == 'train_then_eval':
      perform_evaluation(
          estimator=estimator,
          input_fn=data_lib.build_input_fn_for_builder(
              builder,
              False,
              cache_dataset=_CACHE_DATASET.value,
              image_size=_IMAGE_SIZE.value,
              color_jitter_strength=_COLOR_JITTER_STRENGTH.value,
              options=options,
          ),
          eval_steps=eval_steps,
          model=model,
          num_classes=num_classes,
      )


if __name__ == '__main__':
  tf.disable_eager_execution()  # Disable eager mode when running with TF2.
  app.run(main)
