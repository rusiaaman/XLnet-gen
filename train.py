"""Pretraining on TPUs."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import app
from absl import flags
import absl.logging as _logging  # pylint: disable=unused-import

import math
import numpy as np

import tensorflow as tf
import model_utils
import tpu_estimator
import function_builder
import data_utils

# TPU parameters
flags.DEFINE_string("master", default=None,
      help="master")
flags.DEFINE_string("tpu", default=None,
      help="The Cloud TPU to use for training. This should be either the name "
      "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 url.")
flags.DEFINE_string("gcp_project", default=None,
      help="Project name for the Cloud TPU-enabled project. If not specified, "
      "we will attempt to automatically detect the GCE project from metadata.")
flags.DEFINE_string("tpu_zone",default=None,
      help="GCE zone where the Cloud TPU is located in. If not specified, we "
      "will attempt to automatically detect the GCE project from metadata.")
flags.DEFINE_bool("use_tpu", default=True,
      help="Use TPUs rather than plain CPUs.")
flags.DEFINE_integer("num_hosts", default=1,
      help="number of TPU hosts")
flags.DEFINE_integer("num_core_per_host", default=8,
      help="number of cores per host")
flags.DEFINE_bool("track_mean", default=False,
      help="Whether to track mean loss.")

# Experiment (data/checkpoint/directory) config
flags.DEFINE_integer("num_passes", default=1,
      help="Number of passed used for training.")
flags.DEFINE_string("record_info_dir", default=None,
      help="Path to local directory containing `record_info-lm.json`.")
flags.DEFINE_string("model_dir", default=None,
      help="Estimator model_dir.")
flags.DEFINE_string("init_checkpoint", default=None,
      help="Checkpoint path for initializing the model.")

# Optimization config
flags.DEFINE_float("learning_rate", default=1e-4,
      help="Maximum learning rate.")
flags.DEFINE_float("clip", default=1.0,
      help="Gradient clipping value.")
# lr decay
flags.DEFINE_float("min_lr_ratio", default=0.001,
      help="Minimum ratio learning rate.")
flags.DEFINE_integer("warmup_steps", default=0,
      help="Number of steps for linear lr warmup.")
flags.DEFINE_float("adam_epsilon", default=1e-8,
      help="Adam epsilon.")
flags.DEFINE_string("decay_method", default="poly",
      help="Poly or cos.")
flags.DEFINE_float("weight_decay", default=0.0,
      help="Weight decay rate.")

# Training config
flags.DEFINE_integer("train_batch_size", default=16,
      help="Size of the train batch across all hosts.")
flags.DEFINE_integer("train_steps", default=100000,
      help="Total number of training steps.")
flags.DEFINE_integer("iterations", default=1000,
      help="Number of iterations per repeat loop.")
flags.DEFINE_integer("save_steps", default=None,
      help="Number of steps for model checkpointing. "
      "None for not saving checkpoints")
flags.DEFINE_integer("max_save", default=100000,
      help="Maximum number of checkpoints to save.")

# Data config
flags.DEFINE_integer("seq_len", default=0,
      help="Sequence length for pretraining.")
flags.DEFINE_integer("reuse_len", default=0,
      help="How many tokens to be reused in the next batch. ")
flags.DEFINE_bool("uncased", False,
      help="Use uncased inputs or not.")
flags.DEFINE_integer("perm_size", 0,
      help="Window size of permutation.")
flags.DEFINE_bool("bi_data", default=True,
      help="Use bidirectional data streams, i.e., forward & backward.")
flags.DEFINE_integer("mask_alpha", default=6,
      help="How many tokens to form a group.")
flags.DEFINE_integer("mask_beta", default=1,
      help="How many tokens to mask within each group.")
flags.DEFINE_integer("num_predict", default=None,
      help="Number of tokens to predict in partial prediction.")
flags.DEFINE_integer("n_token", 32000, help="Vocab size")

# Model config
flags.DEFINE_integer("mem_len", default=0,
      help="Number of steps to cache")
flags.DEFINE_bool("same_length", default=False,
      help="Same length attention")
flags.DEFINE_integer("clamp_len", default=-1,
      help="Clamp length")

flags.DEFINE_integer("n_layer", default=6,
      help="Number of layers.")
flags.DEFINE_integer("d_model", default=32,
      help="Dimension of the model.")
flags.DEFINE_integer("d_embed", default=32,
      help="Dimension of the embeddings.")
flags.DEFINE_integer("n_head", default=4,
      help="Number of attention heads.")
flags.DEFINE_integer("d_head", default=8,
      help="Dimension of each attention head.")
flags.DEFINE_integer("d_inner", default=32,
      help="Dimension of inner hidden size in positionwise feed-forward.")
flags.DEFINE_float("dropout", default=0.0,
      help="Dropout rate.")
flags.DEFINE_float("dropatt", default=0.0,
      help="Attention dropout rate.")
flags.DEFINE_bool("untie_r", default=False,
      help="Untie r_w_bias and r_r_bias")
flags.DEFINE_string("summary_type", default="last",
      help="Method used to summarize a sequence into a compact vector.")
flags.DEFINE_string("ff_activation", default="relu",
      help="Activation type used in position-wise feed-forward.")
flags.DEFINE_bool("use_bfloat16", False,
      help="Whether to use bfloat16.")

# Parameter initialization
flags.DEFINE_enum("init", default="normal",
      enum_values=["normal", "uniform"],
      help="Initialization method.")
flags.DEFINE_float("init_std", default=0.02,
      help="Initialization std when init is normal.")
flags.DEFINE_float("init_range", default=0.1,
      help="Initialization std when init is uniform.")

# For generative lm
flags.DEFINE_bool("generative",False,
                  help="Whether to use generative language modelling.")
flags.DEFINE_integer("gen_alpha",9,
                  help="Length of context to consider for genreative masking.")
flags.DEFINE_integer("gen_beta",1,
                  help="Threshold for number of tokens in context for non-zero prob")
flags.DEFINE_float("gen_gamma",0.5,
                  help="Scaling factor for generative masking. Lower it "
                  "for harder optimization but more bi-directionality")
flags.DEFINE_integer("max_seeds",5,
                  help="Maximum number of seed tokens")
flags.DEFINE_bool("mem_drop",False,
                  help="randomly drop memory from some batches")
flags.DEFINE_float("mem_drop_scale",2.,
                  help="scale of dropping of memory")
# Evaluation
flags.DEFINE_bool("eval",False,help="Whether to generate masks for evaluation."
                  "This will use only generate causal masking")
flags.DEFINE_integer("eval_batch_size", default=16,
      help="Size of the train batch across all hosts.")
flags.DEFINE_bool("do_eval_only",False,help="Do not train only evaluate")
flags.DEFINE_string("eval_ckpt_path", default=None,
      help="Checkpoint path to directly evaluate on.")
flags.DEFINE_integer("max_eval_batch", default=-1,
      help="Set -1 to turn off. Only used in test mode.")
flags.DEFINE_integer("start_eval_steps", default=10000,
      help="Which checkpoint to start with in `do_eval_only` mode.")
flags.DEFINE_enum("eval_split", "valid", ["train","valid","test"],
      help="Which data split to evaluate.")
flags.DEFINE_string("record_info_dir_eval", default=None,
      help="Path to local directory containing `record_info-lm.json`.")


FLAGS = flags.FLAGS

def metric_fn(loss):
  """Evaluation metric Fn which runs on CPU."""
  perplexity = tf.exp(tf.reduce_mean(loss))
  bpc = tf.reduce_mean(loss) / tf.constant(math.log(2))
  return {
      "perplexity": tf.metrics.mean(perplexity),
      "bpc": tf.metrics.mean(bpc),
  }

def get_drop_mask(batch_size):
  """Randomly drops few instances in a batch according
  to a probability distribution which insures some indices of
  the batch are never dropped"""
  tf_float = tf.bfloat16 if FLAGS.use_bfloat16 else tf.float32
  indices = tf.cast(tf.range(1,batch_size+1),dtype=tf_float)
  p_notdrop = tf.math.tanh(indices*32/(batch_size*FLAGS.mem_drop_scale))
  to_logits = tf.concat([1-p_notdrop[:,None],p_notdrop[:,None]],axis=-1)
  to_logits = tf.log(to_logits)
  return tf.random.categorical(to_logits,num_samples=1,dtype=tf_float)[:,0]


def get_model_fn():
  """doc."""
  def model_fn(features, labels, mode, params):
    """doc."""
    #### Training or Evaluation
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    #assert is_training

    #### Retrieve `mems` from `params["cache"]`
    mems = {}
    idx = 0
    if FLAGS.mem_len > 0:
      mems["mems"] = params["cache"]

    #### Get loss from inputs
    total_loss, new_mems, monitor_dict = function_builder.get_loss(
        FLAGS, features, labels, mems, is_training)

    #### Turn `new_mems` into `new_cache`
    new_cache = []
    if FLAGS.mem_len > 0:
      if not (is_training and FLAGS.mem_drop):
        new_cache += new_mems["mems"]
      else:
        # Dropping some memory
        bsz = tf.shape(new_mems['mems'][0])[1]
        bsz = tf.cast(bsz,dtype=tf.in32)
        drop_masks = get_drop_mask(bsz)
        new_cache = [[drop_masks[None,:,None]*mem 
                          for mem in new_mems['mems']]]

    #### Check model parameters
    num_params = sum([np.prod(v.shape) for v in tf.trainable_variables()])
    tf.logging.info("#params: {}".format(num_params))

    #### Configuring the optimizer
    if is_training:
      train_op, learning_rate, gnorm = model_utils.get_train_op(
          FLAGS, total_loss)
      monitor_dict["lr"] = learning_rate
      monitor_dict["gnorm"] = gnorm

    #### Customized initial checkpoint
    scaffold_fn = model_utils.init_from_checkpoint(FLAGS, global_vars=True)


    if mode == tf.estimator.ModeKeys.EVAL:
      if FLAGS.use_tpu:
        with tf.colocate_with(total_loss):
          total_loss = tf.contrib.tpu.cross_replica_sum(total_loss) \
                     / FLAGS.num_hosts / FLAGS.num_core_per_host
      metric_loss = tf.tile(tf.reshape(total_loss, [1, 1]), [params["batch_size"], 1])
      eval_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          eval_metrics=(metric_fn, [metric_loss]))

      eval_spec.cache = new_cache

      return eval_spec

    #### Creating host calls
    host_call = function_builder.construct_scalar_host_call(
        monitor_dict=monitor_dict,
        model_dir=FLAGS.model_dir,
        prefix="train/",
        reduce_fn=tf.reduce_mean)

    #### Constucting training TPUEstimatorSpec with new cache.
    train_spec = tf.contrib.tpu.TPUEstimatorSpec(
        mode=mode, loss=total_loss, train_op=train_op, host_call=host_call,
        scaffold_fn=scaffold_fn)

    train_spec.cache = new_cache

    return train_spec

  return model_fn


def get_cache_fn(mem_len):
  """doc."""
  tf_float = tf.bfloat16 if FLAGS.use_bfloat16 else tf.float32
  def cache_fn(batch_size):
    mems = []
    if FLAGS.mem_len > 0:
      for _ in range(FLAGS.n_layer):
        zeros = tf.zeros(
            [mem_len, batch_size, FLAGS.d_model],
            dtype=tf_float)
        mems.append(zeros)

    return mems

  if mem_len > 0:
    return cache_fn
  else:
    return None


def get_input_fn(split,toeval=False,
                 batch_size=None,
                 tfrecord_dir=None):
  """doc."""
  #assert split == "train"
  if batch_size is None: 
    batch_size = FLAGS.train_batch_size
  if tfrecord_dir is None:
    tfrecord_dir = FLAGS.record_info_dir
  reuse_len = FLAGS.reuse_len if not toeval else FLAGS.seq_len
  input_fn, record_info_dict = data_utils.get_input_fn(
      tfrecord_dir=tfrecord_dir,
      split=split,
      bsz_per_host=batch_size // FLAGS.num_hosts,
      seq_len=FLAGS.seq_len,
      reuse_len=reuse_len,
      bi_data=FLAGS.bi_data,
      num_hosts=FLAGS.num_hosts,
      num_core_per_host=FLAGS.num_core_per_host,
      perm_size=FLAGS.perm_size,
      mask_alpha=FLAGS.mask_alpha,
      mask_beta=FLAGS.mask_beta,
      uncased=FLAGS.uncased,
      num_passes=FLAGS.num_passes,
      use_bfloat16=FLAGS.use_bfloat16,
      num_predict=FLAGS.num_predict,
      generative=FLAGS.generative,
      gen_alpha=FLAGS.gen_alpha,
      gen_beta=FLAGS.gen_beta,
      gen_gamma=FLAGS.gen_gamma,
      max_seeds=FLAGS.max_seeds,
      toeval=toeval)

  return input_fn, record_info_dict


def main(unused_argv):
  del unused_argv  # Unused

  tf.logging.set_verbosity(tf.logging.INFO)

  assert FLAGS.seq_len > 0
  #assert FLAGS.perm_size > 0

  # Evaluation not supported yet for non-generative lm
  toeval = FLAGS.eval or FLAGS.do_eval_only
  assert not toeval or (toeval and FLAGS.generative), ("Evaluation not"
        "supproted for non-generative language modelling")

  if FLAGS.record_info_dir_eval is None:
    FLAGS.record_info_dir_eval = FLAGS.record_info_dir

  FLAGS.n_token = data_utils.VOCAB_SIZE
  tf.logging.info("n_token {}".format(FLAGS.n_token))

  if not tf.gfile.Exists(FLAGS.model_dir):
    tf.gfile.MakeDirs(FLAGS.model_dir)

  if not FLAGS.do_eval_only:
    # Get train input function
    train_input_fn, train_record_info_dict = get_input_fn('train')
    num_train_batch = train_record_info_dict["num_batch"]
    tf.logging.info("num of train batches {}".format(
                    num_train_batch))
  if toeval:
    assert FLAGS.num_hosts == 1
    # Get eval input function
    eval_input_fn, eval_record_info_dict = \
                                get_input_fn(FLAGS.eval_split,
                                            toeval=True,
                                            batch_size=FLAGS.eval_batch_size,
                                            tfrecord_dir=FLAGS.record_info_dir_eval)
    num_eval_batch = eval_record_info_dict["num_batch"]
    if FLAGS.max_eval_batch > 0:
      num_eval_batch = min(FLAGS.max_eval_batch, num_eval_batch)
    tf.logging.info("num of eval batches {}".format(
                    num_eval_batch))

  if toeval:
    eval_cache_fn = get_cache_fn(FLAGS.mem_len)
  else:
    eval_cache_fn = None

  if not FLAGS.do_eval_only:
    # Get train cache function
    train_cache_fn = get_cache_fn(FLAGS.mem_len)
  else:
    train_cache_fn = None

  ##### Get model function
  model_fn = get_model_fn()

  ##### Create TPUEstimator
  # TPU Configuration
  run_config = model_utils.configure_tpu(FLAGS)

  # TPU Estimator
  estimator = tpu_estimator.TPUEstimator(
      model_fn=model_fn,
      train_cache_fn=train_cache_fn,
      eval_cache_fn=eval_cache_fn,
      use_tpu=FLAGS.use_tpu,
      config=run_config,
      params={"track_mean": FLAGS.track_mean},
      train_batch_size=FLAGS.train_batch_size,
      eval_batch_size=FLAGS.eval_batch_size,
      eval_on_tpu=FLAGS.use_tpu)

  if FLAGS.do_eval_only:
    if FLAGS.eval_ckpt_path is not None:
      ret = estimator.evaluate(input_fn=eval_input_fn, steps=num_eval_batch,
                               checkpoint_path=FLAGS.eval_ckpt_path)
      tf.logging.info("=" * 200)
      log_str = "Eval results | "
      for key, val in ret.items():
        log_str += "{} {} | ".format(key, val)
      tf.logging.info(log_str)
      tf.logging.info("=" * 200)
    else:
      ckpt_state = tf.train.get_checkpoint_state(FLAGS.model_dir)
      eval_results = []
      for eval_checkpoint in ckpt_state.all_model_checkpoint_paths:
        if not exists(eval_checkpoint + ".index"): continue
        global_step = int(eval_checkpoint.split("-")[-1])
        if global_step < FLAGS.start_eval_steps or global_step > FLAGS.train_steps:
          continue
        ret = estimator.evaluate(input_fn=eval_input_fn, steps=num_eval_batch,
                                 checkpoint_path=eval_checkpoint)
        eval_results.append(ret)

      eval_results.sort(key = lambda x: x["perplexity"])

      tf.logging.info("=" * 200)
      log_str = "Best results | "
      for key, val in eval_results[0].items():
        log_str += "{} {} | ".format(key, val)
      tf.logging.info(log_str)
      tf.logging.info("=" * 200)
  else:
    if not FLAGS.eval:
      estimator.train(input_fn=train_input_fn, steps=FLAGS.train_steps)
    else:
      # Evaluate every epoch
      for step in range(0, FLAGS.train_steps, num_train_batch):
        train_steps = min(FLAGS.train_steps - step, num_train_batch)
        estimator.train(input_fn=train_input_fn, steps=FLAGS.train_steps)
        estimator.evaluate(input_fn=eval_input_fn, steps=num_eval_batch)


if __name__ == "__main__":
  app.run(main)
