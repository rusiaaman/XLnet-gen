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



# Model
flags.DEFINE_string("model_config_path", default=None,
      help="Model config path.")
flags.DEFINE_integer("clamp_len", default=-1,
      help="Clamp length")
# I/O paths
flags.DEFINE_string("init_checkpoint", default=None,
      help="checkpoint path for initializing the model. "
      "Could be a pretrained model or a finetuned model.")
flags.DEFINE_string("spiece_model_file", default="",
      help="Sentence Piece model path.")
flags.DEFINE_string("model_dir", default="",
      help="Directory for saving the finetuned model.")
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
flags.DEFINE_float("top-p", default=0,
      help="Top-p coverage to use. Set 0 to use top_k sampling")
flags.DEFINE_float("top-k", default=40,
      help="Top-k sampling strategy parameter. Use only when top-p is zero. Set"
            "-1 to use all the samples")
flags.DEFINE_bool("temperature", default=1,
      help="Scaling factor for logits")
flags.DEFINE_bool("num_toks_pred", default=1024,
      help="Number of tokens to predict")
flags.DEFINE_integer("max_mem_length", default=128, 
      help="Max sequence length for cached hidden states"
           " which each predicted token is conditioned upon"
           ". Directly increases the memory requirement")

FLAGS = flags.FLAGS


def convert_examples_to_features(examples,
                                 max_seq_length,
                                 tokenize_fn):
    """
    Input:
    examples: [List[str]] input texts
    max_seq_length: [int] maximum length of context. 
                Truncated on left so that only most
                recent context is taken
    tokenize_fn: [function] encodes text into IDs
    Output:
    tf input features
    """
    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10 == 0:
            tf.logging.info("Featuring example {} of {}".format(ex_index,
                                                        len(examples)))
        tokens = tokenize_fn(example)
        is_masked = [False]*len(tokens)
        seg_id = [0]*len(tokens)
        feature = {
          "input": _int64_feature(tokens),
          "is_masked": _int64_feature(is_masked),
        }
        features.append(feature)

    featurized_examples = tf.train.Example(features=tf.train.Features(feature=features))

    tf.logging.info("Featurized %s examples",len(featurized_examples))

    return featurized_examples

def inputs_and_mask(latest_tokens):
    """Computes inputs and masks for prediction loop.
    latest_tokens: Tensor [batch_size,num_tokens]

    output:
    input_k: [batch_size,num_tokens+1] latest_tokens with a dummy
            token appened at the end of the sequence
    attn_masks: [batch_size,num_tokens+1,num_tokens+1]
    input_q: [batch_size,num_tokens+1]
            masks the tokens to predict. In this case the last token
    """

    return input_k,attn_masks,input_q


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

def prediction_graph(features,mems):
    """Gets features and mems and
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

    perm_mask = get_causal_attention_mask(tf.shape(inp)[0],tf.shape(inp)[0])

    # Getting the hidden states for the prompts
    xlnet_model = xlnet.XLNetModel(
      xlnet_config=xlnet_config,
      run_config=run_config,
      input_ids=inp,
      seg_ids=seg_id,
      input_mask=inp_mask,
      perm_mask=perm_mask,
      mems=mems)

    logits = get_logits(xlnet_model)
    
    # getting new memory
    mems = xlnet_model.get_new_memory()

    latest_tokens = None
    prev_tokens = None
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
        input_k, seg_id, perm_mask, input_q = inputs_and_mask(latest_tokens)
        # Get logits
        with tf.variable_scope("",reuse=tf.AUTO_REUSE):
            xlnet_model = xlnet.XLNetModel(
              xlnet_config=xlnet_config,
              run_config=run_config,
              input_ids=input_k,
              seg_ids=seg_id,
              perm_mask=perm_mask,
              mems=mems) 

        
        

        with tf.variable_scope("model", reuse=tf.AUTO_REUSE):


        # Getting new memory
        new_mems = xlnet_model.get_new_memory()

        # sample a token
        sampled_tokens =
        prev_tokens = sampled_tokens if prev_tokens is None \
                        else tf.concat([prev_tokens,sampled_tokens],axis=1)
        # Cache the memory of the the last lateset_token
        if latest_tokens:
            mem = 
        else:
            mem = 

        return mem, sampled_tokens, prev_tokens

    




def main():
    """Main function routine"""
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




if __name__=="__main__":
    # Fixed flags
    FLAGS.use_tpu = False #TPU won't be used for prediction
    FLAGS.use_bfloat16 = False
    FLAGS.dropout = 0
    FLAGS.dropatt = 0
    FLAGS.init = 'uniform' #doesn't matter
    FLAGS.init_std=0.02
    FLAGS.init_range=0.1

    tf.app.run()