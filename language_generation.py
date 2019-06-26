from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join
from absl import flags
import os
import sys
import csv
import collections
import numpy as np
import time
import math
import json
import random
from copy import copy
from collections import defaultdict as dd

import absl.logging as _logging  # pylint: disable=unused-import
import tensorflow as tf

import sentencepiece as spm

from data_utils import SEP_ID, VOCAB_SIZE, CLS_ID
import model_utils
import function_builder
from classifier_utils import PaddingInputExample
from classifier_utils import convert_single_example
from prepro_utils import preprocess_text, encode_ids
from modeling import _create_mask

CLS_ID = 3 #[CLS] token ID

# Model
flags.DEFINE_string("model_config_path", default=None,
      help="Model config path.")
flags.DEFINE_integer("clamp_len", default=-1,
      help="Clamp length")
flags.DEFINE_bool("same_length", default=False,
      help="Same length attention")

# Data and memory
flags.DEFINE_integer("batch_size",default=1,help='batch size')
flags.DEFINE_integer("max_mem_length", default=128, 
      help="Max sequence length for cached hidden states"
           " which each predicted token is conditioned upon"
           ". Directly increases the memory requirement")
flags.DEFINE_bool("uncased", False,
      help="Use uncased inputs or not.")

# I/O paths
flags.DEFINE_string("init_checkpoint", default=None,
      help="checkpoint path for initializing the model. "
      "Could be a pretrained model or a finetuned model.")
flags.DEFINE_string("spiece_model_file", default="",
      help="Sentence Piece model path.")
flags.DEFINE_string("input_file", default="",
      help="File containing new line separated prompts "
            "for conditional sampling")

# prediction
flags.DEFINE_integer("num_samples",default=1,
      help="Number of samples to predict per instance")
flags.DEFINE_bool("interactive", default=True,
      help="Flag for interactive prediction through command line")
flags.DEFINE_bool("unconditional", default=True,
      help="Prints samples wihtout any prompt")
flags.DEFINE_float("top_p", default=0,
      help="Top-p coverage to use. Set 0 to use top_k sampling")
flags.DEFINE_integer("top_k", default=40,
      help="Top-k sampling strategy parameter. Use only when top-p is zero. Set"
            "-1 to use all the samples")
flags.DEFINE_integer("temperature", default=1,
      help="Scaling factor for logits")
flags.DEFINE_integer("num_toks_pred", default=1024,
      help="Number of tokens to predict")

# Static flags, do not change
flags.DEFINE_bool("use_tpu", default=False,
      help="TPU can't be used for inference. Do not set")
flags.DEFINE_bool("use_bfloat16", False,
      help="Whether to use bfloat16. Not implemented")
flags.DEFINE_float("dropout", default=0,
      help="Dropout rate. Do not change.")
flags.DEFINE_float("dropatt", default=0,
      help="Attention dropout rate. Do not change.")


flags.DEFINE_enum("init", default="normal",
      enum_values=["normal", "uniform"],
      help="Initialization method.")
flags.DEFINE_float("init_std", default=0.02,
      help="Initialization std when init is normal.")
flags.DEFINE_float("init_range", default=0.1,
      help="Initialization std when init is uniform.")


FLAGS = flags.FLAGS



def _int64_feature(values):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

def convert_examples_to_features(examples,
                                 tokenize_fn):
    """
    Input:
    examples: [List[str]] input texts
    tokenize_fn: [function] encodes text into IDs
    Output:
    tf input features
    """
    featurized_examples = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10 == 0:
            tf.logging.info("Featuring example {} of {}".format(ex_index,
                                                        len(examples)))
        tokens = tokenize_fn(example)
        feature = {
          "input_k": _int64_feature(tokens),
        }

        featurized_examples.append(tf.train.Example(features=tf.train.Features(feature=feature)))
    
    tf.logging.info("Featurized %s examples",len(featurized_examples))

    return featurized_examples

def get_input_dataset(featurized_examples):
    """Returns tf.data.Dataset for input"""
    batch_size = FLAGS.batch_size
    max_mem_length = FLAGS.max_mem_length
    
    def mask(example):
        input_k = example['input_k'][-max_mem_length:]
        seq_len = len(input_k)
        input_mask = tf.convert_to_tensor([0]*len(seq_len))
        if seq_len<max_mem_length:
            pad_len = max_mem_length-seq_len
            input_k = tf.concat([tf.zeros(pad_len),input_k],axis=0)
            input_mask = tf.concat([[1]*pad_len,input_mask],axis=0)
        example['input_mask'] = input_mask
        example['input_k'] = input_k
        example['seg_id'] = tf.convert_to_tensor([0]*max_mem_length)
        

    dataset = tf.data.Dataset.from_tensor_slices(featurized_examples)
    dataset = dataset.map(mask)
    dataset = dataset.batch(batch_size,
                            drop_remainder=False)
    dataset.prefetch(1)
    return dataset

def inputs_and_mask(latest_tokens,batch_size):
    """Computes inputs and masks for prediction loop.
    A dummy token ([CLS]) is appended at the at of the previous
    tokens

    Input:
    latest_tokens: Tensor [batch_size,1] or None
            If None then last dimension is 1 in the returned tensors

    output:
    input_k: [batch_size,2] latest_tokens with a dummy
            token appened at the end of the sequence
    seg_id: [batch_size,2]
    attn_masks: [batch_size,2,2]
    input_q: [batch_size,2]
            masks the tokens to predict. In this case the last token
    """
    input_k = tf.tile([[CLS_ID]],[batch_size,1])
    seg_id = tf.tile([[0]],[batch_size,1])
    input_q = tf.tile([[1]],[batch_size,1])

    if not latest_tokens is None:
        input_k = tf.concat([latest_tokens,input_k],axis=-1)
        seg_id = tf.tile(seg_id,[1,2])
        input_q_0 = tf.tile([[0]],[batch_size,1])
        input_q = tf.concat([input_q_0,input_q],axis=-1)

        attn_masks = tf.convert_to_tensor([[0,1],[0,1]])
    else:
        attn_masks = tf.convert_to_tensor([[1]])

    return input_k,seg_id, attn_masks, input_q


def get_logits(xlnet_model):
    lookup_table = xlnet_model.get_embedding_table()
    tie_weight=False

    with tf.variable_scope("model", reuse=tf.AUTO_REUSE):
        initializer = xlnet_model.get_initializer()
        hidden = xlnet_model.get_sequence_output()[-1:-2,:,:]
        n_token = xlnet_config.n_token 
        d_model = xlnet_config.d_model

        with tf.variable_scope('lm_loss'):
            if tie_weight:
              assert lookup_table is not None, \
                  'lookup_table cannot be None for tie_weight'
              softmax_w = lookup_table
            else:
              softmax_w = tf.get_variable('weight', [n_token, d_model],
                                          dtype=hidden.dtype, initializer=initializer)

            softmax_b = tf.get_variable('bias', [n_token], dtype=hidden.dtype,
                                        initializer=tf.zeros_initializer())

            logits = tf.einsum('ibd,nd->ibn', hidden, softmax_w) + softmax_b

    return logits

def sampling_strategy():
    if FLAGS.top_p!=0:
        return 'top_p'

    return 'top_k'

def sample_token(logits):
    """
    Inputs:
    logits: tf.Tensor([batch_size,len,num_tokens])
    Outpus:
    samples: tf.Tensor([batch_size,len])
    """
    #credits: https://github.com/nshepperd/gpt-2
    
    logits/=FLAGS.temperature

    batch_size,seq_len,num_toks = tf.shape(logits)
    
    if sampling_strategy()=='top_p':
        logits_sorted = tf.sort(logits,
                                direction="DESCENDING",
                                axis=-1)
        probs = tf.nn.softmax(logits_sorted,axis=-1)
        cum_probs = tf.math.cumsum(probs,
                                   axis=-1,
                                   exclusive=True)
        logits_masked = tf.where(cum_probs<FLAGS.top_p,
                                 logits_sorted,
                                 tf.ones_like(logits_sorted)*100)
        min_logits = tf.reduce_min(logits_masked,axis=-1)

        logits = tf.where(logits<min_logits,
                          tf.ones_like(logits)*-1e10,
                          logits)

    elif sampling_strategy()=="top_k":
        if FLAGS.top_k == 0:
            return logits

        values, _ = tf.nn.top_k(logits, k=k)
        min_values = values[:,:,-1]
        logits = tf.where(
            logits < min_values,
            tf.ones_like(logits, dtype=logits.dtype) * -1e10,
            logits,
        )
    else:
        raise NotImplementedError("Invalid sampling strategy")

    logits = tf.reshape(logits,(-1,num_toks))

    samples = tf.random.categorical(logits,
                          num_samples=1,
                          output_dtype=tf.int32)

    return tf.reshape(samples,(batch_size,seq_len))




def prediction_graph(features):
    """Gets features and
    return predicted tokens
    features: Dict[str:tf.train.features] Contains following features:
              input_k
              seg_id
              input_mask
    """

    # Building prediction graph
    ## Transforming features for batch channel on last axis
    inp = tf.transpose(features["input_k"], [1, 0])
    seg_id = tf.transpose(features["seg_id"], [1, 0])
    inp_mask = tf.transpose(features["input_mask"], [1, 0])

    ## Model config
    xlnet_config = xlnet.XLNetConfig(json_path=FLAGS.model_config_path)
    run_config = xlnet.create_run_config(False, True, FLAGS)
    run_config.mem_len = max_mem_length

    perm_mask = _create_mask(tf.shape(inp)[0],tf.shape(inp)[0])

    # Getting the hidden states for the prompts
    xlnet_model = xlnet.XLNetModel(
      xlnet_config=xlnet_config,
      run_config=run_config,
      input_ids=inp,
      seg_ids=seg_id,
      input_mask=inp_mask,
      perm_mask=perm_mask)


    # getting memory
    mems = xlnet_model.get_new_memory()

    latest_tokens = None
    prev_tokens = None
    batch_size = tf.shape(mems[0])[1]

    def cond(*args):
        return True

    def body(mems,latest_tokens,prev_tokens):
        """The main body of sampling loop.
        mem: cache memory--calculated hidden states
             of previous tokens
        latest_tokens: latest sampled tokens
        prev_tokens: all the previous tokens including latest_tokens
        """

        # get dummy input token and permutation mask
        input_k, seg_id, perm_mask, input_q = \
                                inputs_and_mask(latest_tokens,
                                                batch_size)

        input_k = tf.transpose(input_k,(1,0))
        input_q = tf.transpose(input_q,(1,0))
        seg_id = tf.transpose(seg_id,(1,0))
        perm_mask = tf.transpose(perm_mask,(1,2,0))
        # Get logits
        with tf.variable_scope("",reuse=tf.AUTO_REUSE):
            xlnet_model = xlnet.XLNetModel(
              xlnet_config=xlnet_config,
              run_config=run_config,
              input_ids=input_k,
              seg_ids=seg_id,
              perm_mask=perm_mask,
              mems=mems)

            logits = get_logits(xlnet_model)

        # Getting new memory
        new_mems = xlnet_model.get_new_memory()

        # sample a token
        logits = tf.transpose(logits,(1,0,2))
        sampled_tokens = sample_token(logits)

        prev_tokens = sampled_tokens if prev_tokens is None \
                        else tf.concat([prev_tokens,sampled_tokens],axis=1)
        # Cache the memory of the the last latest_tokens
        if latest_tokens:
            mems = [tf.concat(_mem[1:],_new_mem[:1]) 
                    for _mem,_new_mem in zip(mems,new_mems)]
        return mems, sampled_tokens, prev_tokens

    mems,latest_tokens,prev_tokens=body(mems,latest_tokens,prev_tokens)

    _,_,predicted_tokens = tf.while_loop(
        cond=cond,
        body=body,
        maximum_iterations=FLAGS.num_toks_pred-1,
        loop_vars=[mems,latest_tokens,prev_tokens],
        shape_invariants=[
                list(map(lambda x: x.shape,mems)),
                tf.shape(latest_tokens),
                tf.TensorShape([batch_size,None])
            ]
        )

    return predicted_tokens   




def main(unused_argv):
    """Main function routine"""

    del unused_argv #unncessary args like file name

    # Fixed flags
    if FLAGS.use_tpu:
        raise Exception("Inference can't run on TPU")
    assert FLAGS.use_bfloat16 == False, "Do not change this flag"
    assert FLAGS.dropout == 0, "Do not change this flag"
    assert FLAGS.dropatt == 0, "Do not change this flag"

    tf.logging.set_verbosity(tf.logging.INFO)

    #Text encoding
    sp = spm.SentencePieceProcessor()
    sp.Load(FLAGS.spiece_model_file)
    def tokenize_fn(text):
        text = preprocess_text(text, lower=FLAGS.uncased)
        return encode_ids(sp, text)

    def predict(examples):
        """Given a list of texts in examples
        return the result"""
        num_examples = len(examples)
        num_batches = np.ceil(num_examples/FLAGS.batch_size)
        features = convert_examples_to_features(examples,
                                                tokenize_fn)
        dataset = get_input_dataset(features)
        example = dataset.make_one_shot_iterator().get_next()
        predicted_tokens = prediction_graph(example)

        saver = tf.train.Saver()
        gpu_options = tf.GPUOptions(allow_growth=True)
        model_utils.init_from_checkpoint(FLAGS, global_vars=False)
        outputs = []
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                        gpu_options=gpu_options)) as sess:
            sess.run(tf.global_variables_initializer())
            for _ in range(num_batches):
                output = sess.run(predicted_tokens)
                outputs.extend(output)
        return outputs

    if FLAGS.interactive:
        text = input("----PROMPT---\n")
        outputs = predict([text]*FLAGS.num_samples)
        for i,output in enumerate(outputs):
            print("--------SAMPLE No. {}---------\n".format(i))
            print(sp.decode_ids(output))
    else:
        raise NotImplementedError("WIP file reading")


if __name__=="__main__":

    tf.app.run()