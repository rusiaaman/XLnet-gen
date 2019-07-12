"""Pretraining on GPUs."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys
import math
import json
import time
import numpy as np

from absl import flags
import absl.logging as _logging  # pylint: disable=unused-import

import tensorflow as tf

import data_utils
import model_utils
from gpu_utils import assign_to_gpu, average_grads_and_vars
import function_builder


# GPU config
flags.DEFINE_integer("num_hosts", default=1,
      help="Number of hosts")
flags.DEFINE_integer("num_core_per_host", default=8,
      help="Number of cores per host")
flags.DEFINE_bool("use_tpu", default=False,
      help="Whether to use TPUs for training.")

# Experiment (data/checkpoint/directory) config
flags.DEFINE_integer("num_passes", default=1,
      help="Number of passed used for training.")
flags.DEFINE_string("record_info_dir", default=None,
      help="Path to local directory containing `record_info-lm.json`.")
flags.DEFINE_string("model_dir", default=None,
      help="Estimator model_dir.")
flags.DEFINE_string("init_checkpoint", default=None,
      help="checkpoint path for initializing the model.")

# Optimization config
flags.DEFINE_float("learning_rate", default=2.5e-4,
      help="Maximum learning rate.")
flags.DEFINE_float("clip", default=0.25,
      help="Gradient clipping value.")
# for cosine decay
flags.DEFINE_float("min_lr_ratio", default=0.004,
      help="Minimum ratio learning rate.")
flags.DEFINE_integer("warmup_steps", default=0,
      help="Number of steps for linear lr warmup.")
flags.DEFINE_float("adam_epsilon", default=1e-8,
      help="Adam epsilon")
flags.DEFINE_string("decay_method", default="poly",
      help="poly or cos")
flags.DEFINE_float("weight_decay", default=0.0,
      help="weight decay")

# Training config
flags.DEFINE_integer("train_batch_size", default=60,
      help="Size of train batch.")
flags.DEFINE_integer("train_steps", default=100000,
      help="Total number of training steps.")
flags.DEFINE_integer("iterations", default=500,
      help="Number of iterations per repeat loop.")
flags.DEFINE_integer("save_steps", default=10000,
      help="number of steps for model checkpointing.")

# Data config
flags.DEFINE_integer('seq_len', default=0,
      help='Sequence length for pretraining.')
flags.DEFINE_integer('reuse_len', default=0,
      help="How many tokens to be reused in the next batch. "
      "Could be half of seq_len")
flags.DEFINE_bool("bi_data", default=True,
      help="Use bidirectional data streams, i.e., forward & backward.")
flags.DEFINE_integer("mask_alpha", default=2,
      help="How many tokens to form a group.")
flags.DEFINE_integer("mask_beta", default=1,
      help="How many tokens to mask within each group.")
flags.DEFINE_integer("num_predict", default=None,
      help="Number of tokens to predict in partial prediction.")
flags.DEFINE_integer('perm_size', default=0,
  help='perm size.')
flags.DEFINE_bool("uncased", False,
      help="Use uncased inputs or not.")
flags.DEFINE_integer("n_token", 32000, help="Vocab size")

# Model config
flags.DEFINE_integer("mem_len", default=70,
      help="Number of steps to cache")
flags.DEFINE_bool("same_length", default=False,
      help="Same length attention")
flags.DEFINE_integer("clamp_len", default=-1,
      help="Clamp length")

flags.DEFINE_integer("n_layer", default=6,
      help="Number of layers.")
flags.DEFINE_integer("d_model", default=500,
      help="Dimension of the model.")
flags.DEFINE_integer("d_embed", default=500,
      help="Dimension of the embeddings.")
flags.DEFINE_integer("n_head", default=10,
      help="Number of attention heads.")
flags.DEFINE_integer("d_head", default=50,
      help="Dimension of each attention head.")
flags.DEFINE_integer("d_inner", default=1000,
      help="Dimension of inner hidden size in positionwise feed-forward.")
flags.DEFINE_float("dropout", default=0.1,
      help="Dropout rate.")
flags.DEFINE_float("dropatt", default=0.1,
      help="Attention dropout rate.")
flags.DEFINE_bool("untie_r", default=False,
      help="Untie r_w_bias and r_r_bias")
flags.DEFINE_string("summary_type", default="attn",
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
                  help="Maximum number of seed tokens")# Evaluation
flags.DEFINE_bool("eval",False,help="Whether to generate masks for evaluation."
                  "This will use only generate causal masking")
flags.DEFINE_bool("do_eval_only",False,help="Do not train only evaluate")
flags.DEFINE_integer("max_eval_batch", default=-1,
      help="Set -1 to turn off. Only used in test mode.")
flags.DEFINE_integer("start_eval_steps", default=10000,
      help="Which checkpoint to start with in `do_eval_only` mode.")
flags.DEFINE_enum("eval_split", "valid", ["train","valid","test"],
      help="Which data split to evaluate.")
flags.DEFINE_integer("eval_batch_size", default=16,
      help="Size of the train batch across all hosts.")

FLAGS = flags.FLAGS


def get_model_fn():
  def model_fn(features, labels, mems, is_training):
    #### Get loss from inputs
    total_loss, new_mems, monitor_dict = function_builder.get_loss(
        FLAGS, features, labels, mems, is_training)

    #### Check model parameters
    num_params = sum([np.prod(v.shape) for v in tf.trainable_variables()])
    tf.logging.info('#params: {}'.format(num_params))

    # GPU
    #assert is_training
    if is_training:
      all_vars = tf.trainable_variables()
      grads = tf.gradients(total_loss, all_vars)
      grads_and_vars = list(zip(grads, all_vars))
    else:
      grads_and_vars = None

    return total_loss, new_mems, grads_and_vars

  return model_fn


def single_core_graph(is_training, features, mems):
  model_fn = get_model_fn()

  model_ret = model_fn(
      features=features,
      labels=None,
      mems=mems,
      is_training=is_training)

  return model_ret


def create_mems_tf(bsz_per_core):
  mems = [tf.placeholder(dtype=tf.float32,
                         shape=[FLAGS.mem_len, bsz_per_core, FLAGS.d_model],
                         name='mems')
          for layer in range(FLAGS.n_layer)]

  return mems


def initialize_mems_np(bsz_per_core):
  mems_np = [np.zeros(shape=[FLAGS.mem_len, bsz_per_core, FLAGS.d_model],
                      dtype=np.float32)
             for layer in range(FLAGS.n_layer)]

  return mems_np



def get_input_fn(split,toeval=False,
                 batch_size=None):
  """doc."""
  #assert split == "train"
  if batch_size is None: 
    batch_size = FLAGS.train_batch_size
  reuse_len = FLAGS.reuse_len if not toeval else FLAGS.seq_len
  input_fn, record_info_dict = data_utils.get_input_fn(
      tfrecord_dir=FLAGS.record_info_dir,
      split=split,
      bsz_per_host=batch_size,
      seq_len=FLAGS.seq_len,
      reuse_len=reuse_len,
      bi_data=FLAGS.bi_data,
      num_hosts=1,
      num_core_per_host=1, # set to one no matter how many GPUs
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

def build_graph(ps_device,example,bsz_per_core,is_training):

  if FLAGS.num_core_per_host > 1:
    examples = [{} for _ in range(FLAGS.num_core_per_host)]
    for key in example.keys():
      vals = tf.split(example[key], FLAGS.num_core_per_host, 0)
      for device_id in range(FLAGS.num_core_per_host):
        examples[device_id][key] = vals[device_id]
  else:
    examples = [example]

  ##### Create computational graph
  tower_mems, tower_losses, tower_new_mems, tower_grads_and_vars = [], [], [], []

  for i in range(FLAGS.num_core_per_host):
    reuse = True if i > 0 else None
    with tf.device(assign_to_gpu(i, ps_device)), \
        tf.variable_scope(tf.get_variable_scope(), reuse=reuse):

      # The mems for each tower is a dictionary
      mems_i = {}
      if FLAGS.mem_len:
        mems_i["mems"] = create_mems_tf(bsz_per_core)

      loss_i, new_mems_i, grads_and_vars_i = single_core_graph(
          is_training=is_training,
          features=examples[i],
          mems=mems_i)

      tower_mems.append(mems_i)
      tower_losses.append(loss_i)
      tower_new_mems.append(new_mems_i)
      tower_grads_and_vars.append(grads_and_vars_i)

  ## average losses and gradients across towers
  if len(tower_losses) > 1:
    loss = tf.add_n(tower_losses) / len(tower_losses)
    if is_training:
      grads_and_vars = average_grads_and_vars(tower_grads_and_vars)
  else:
    loss = tower_losses[0]
    if is_training:
      grads_and_vars = tower_grads_and_vars[0]

  ## get train op
  if is_training:
    train_op, learning_rate, gnorm = model_utils.get_train_op(FLAGS, None,
        grads_and_vars=grads_and_vars)
  global_step = tf.train.get_global_step()

  # initialize mems
  tower_mems_np = []
  for i in range(FLAGS.num_core_per_host):
    mems_i_np = {}
    for key in tower_mems[i].keys():
      mems_i_np[key] = initialize_mems_np(bsz_per_core)
    tower_mems_np.append(mems_i_np)

  saver = tf.train.Saver()

  model_utils.init_from_checkpoint(FLAGS, global_vars=True)
  if is_training:
    fetches = [loss, tower_new_mems, global_step, gnorm, learning_rate, train_op]
  else:
    fetches = [loss, tower_new_mems]
  return fetches,tower_mems,tower_mems_np,saver


def train_step(sess, fetches, feed_dict, train_example, example_placeholder,saver,
               total_loss, prev_step):
  input_example = sess.run(train_example)

  feed_dict.update({v:input_example[k] for k,v in example_placeholder.items()})

  fetched = sess.run(fetches, feed_dict=feed_dict)

  loss_np, tower_mems_np, curr_step = fetched[:3]
  total_loss += loss_np

  if curr_step > 0 and curr_step % FLAGS.iterations == 0:
    curr_loss = total_loss / (curr_step - prev_step)
    tf.logging.info("[{}] | gnorm {:.2f} lr {:8.6f} "
        "| loss {:.2f} | pplx {:>7.2f}, bpc {:>7.4f}".format(
        curr_step, fetched[-3], fetched[-2],
        curr_loss, math.exp(curr_loss), curr_loss / math.log(2)))
    total_loss, prev_step = 0., curr_step

  if curr_step > 0 and curr_step % FLAGS.save_steps == 0:
    save_path = os.path.join(FLAGS.model_dir, "model.ckpt")
    saver.save(sess, save_path)
    tf.logging.info("Model saved in path: {}".format(save_path))

  done = False
  if curr_step >= FLAGS.train_steps:
      done = True

  return done, tower_mems_np, total_loss, prev_step, curr_step

def evaluate(sess, fetches, tower_mems, eval_example, example_placeholder, num_eval_batch, tower_mems_np):
  tf.logging.info("="*40+"Evaluating"+"="*40)
  losses = []
  tower_mems_np_eval = tower_mems_np
  for i in range(num_eval_batch):
    feed_dict = {a:b for i in range(FLAGS.num_core_per_host)\
                   for k,v in tower_mems_np_eval[i].items() \
                   for a,b in zip(tower_mems[i][k],v)}

    input_example = sess.run(eval_example)

    feed_dict.update({v:input_example[k] for k,v in example_placeholder.items()})

    loss_np,tower_mems_np_eval = sess.run(fetches[:2], feed_dict = feed_dict)

    losses.append(loss_np)

    tf.logging.info(f"{i}/{num_eval_batch} loss: {loss_np}")


  tf.logging.info(f"Evaluation result Avg loss {np.mean(losses)}"
                   f"ppl {np.exp(np.mean(losses))}")

def get_placeholder_example():
  return {
        "input_k": tf.placeholder(tf.int32, (None, None),name='input_k'),
        "seg_id": tf.placeholder(tf.int32, (None, None),name='seg_id'),
        "input_q": tf.placeholder(tf.float32, (None, None),name='input_q'),
        "perm_mask": tf.placeholder(tf.float32, (None, None, None),name='perm_mask'),
        "target_mask": tf.placeholder(tf.float32, (None, None),name='target_mask'),
        "target": tf.placeholder(tf.int32, (None, None),name='target')
    }

def train(ps_device):
  ##### Get input function and model function
  toeval = FLAGS.eval or FLAGS.do_eval_only
  assert not toeval or (toeval and FLAGS.generative), ("Evaluation not"
        "supproted for non-generative language modelling")

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
                                            batch_size=FLAGS.eval_batch_size)
    num_eval_batch = eval_record_info_dict["num_batch"]
    if FLAGS.max_eval_batch > 0:
      num_eval_batch = min(FLAGS.max_eval_batch, num_eval_batch)
    tf.logging.info("num of eval batches {}".format(
                    num_eval_batch))

  #ToDo: handle different eval and train BS
  bsz_per_core = FLAGS.train_batch_size // FLAGS.num_core_per_host

  ##### Create input tensors / placeholders
  placeholder_example = get_placeholder_example()
  if toeval:
    
    params = {
        "batch_size": FLAGS.eval_batch_size # the whole batch
    }
    eval_set = eval_input_fn(params)
    eval_example = eval_set.make_one_shot_iterator().get_next()
    fetches_e,tower_mems_e,tower_mems_np_e,_ = \
            build_graph(ps_device,placeholder_example,bsz_per_core,False)

  if not FLAGS.do_eval_only:
    params = {
        "batch_size": FLAGS.eval_batch_size # the whole batch
    }
    train_set = train_input_fn(params)
    train_example = train_set.make_one_shot_iterator().get_next()
    fetches,tower_mems,tower_mems_np_train,saver = \
            build_graph(ps_device,placeholder_example,bsz_per_core,True)


  

  gpu_options = tf.GPUOptions(allow_growth=True)

  curr_step = 0

  with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
      gpu_options=gpu_options)) as sess:
    sess.run(tf.global_variables_initializer())

    total_loss, prev_step = 0., -1
    while True:
      if FLAGS.do_eval_only or (curr_step%num_train_batch and toeval):
        evaluate(sess,fetches_e,tower_mems_e,eval_example,placeholder_example,num_eval_batch,tower_mems_np_e)
        if FLAGS.do_eval_only:
          break

      feed_dict = {a:b for i in range(FLAGS.num_core_per_host)\
                   for k,v in tower_mems_np_train[i].items() \
                   for a,b in zip(tower_mems[i][k],v)}

      done,tower_mems_np_train, total_loss, prev_step, curr_step\
                                train_step(sess,
                                            fetches,
                                            feed_dict,
                                            train_example,
                                            placeholder_example,
                                            tower_mems_np_train,
                                            saver,
                                            total_loss,
                                            prev_step)

      if done:
        break
      


def main(unused_argv):
  del unused_argv  # Unused

  tf.logging.set_verbosity(tf.logging.INFO)

  # Get corpus info
  FLAGS.n_token = data_utils.VOCAB_SIZE
  tf.logging.info("n_token {}".format(FLAGS.n_token))

  if not tf.gfile.Exists(FLAGS.model_dir):
    tf.gfile.MakeDirs(FLAGS.model_dir)

  FLAGS.eval_batch_size = FLAGS.train_batch_size #ToDo: handle different eval and train BS

  train("/gpu:0")


if __name__ == "__main__":
  tf.app.run()
