"""Generate language using XLNet"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
import re

from tqdm import tqdm
import absl.logging as _logging  # pylint: disable=unused-import
import tensorflow as tf
import sentencepiece as spm

import model_utils
from data_utils import CLS_ID, special_symbols, EOD_ID

import xlnet
from prepro_utils import preprocess_text, encode_ids



EOP_ID = special_symbols["<eop>"]

parser = argparse.ArgumentParser()
# Model
parser.add_argument("--model_config_path", default=None,
                    help="Model config path.", type=str)
parser.add_argument("--clamp_len", default=-1,
                    help="Clamp length", type=int)
parser.add_argument("--same_length", default=False,
                    help="Same length attention", action='store_true')

# Data and memory
parser.add_argument("--batch_size", default=1, help='batch size', type=int)
parser.add_argument("--max_mem_length", default=128,
                    help="Max sequence length for cached hidden states"
                    " which each predicted token is conditioned upon"
                    ". Directly increases the memory requirement", type=int)
parser.add_argument("--uncased", default=False,
                    help="Use uncased inputs or not.", action='store_true')

# I/O paths
parser.add_argument("--init_checkpoint", default=None,
                    help="checkpoint path for initializing the model. "
                    "Could be a pretrained model or a finetuned model.")
parser.add_argument("--spiece_model_file", default="",
                    help="Sentence Piece model path.")
parser.add_argument("--input_file", default="",
                    help="File containing prompts separated by empty new line "
                    "for conditional sampling")

# prediction
parser.add_argument("--num_samples", default=1,
                    help="Number of samples to predict per instance", type=int)
parser.add_argument(
    "--interactive",
    default=False,
    help="Flag for interactive prediction through command line",
    action='store_true')
parser.add_argument(
    "--unconditional",
    default=False,
    help="Prints samples wihtout any prompt",
    action='store_true')
parser.add_argument(
    "--top_p",
    default=0,
    help="Top-p coverage to use. Set 0 to use top_k sampling",
    type=float)
parser.add_argument(
    "--top_k",
    default=40,
    help="Top-k sampling strategy parameter. Use only when top-p is zero. Set"
    "-1 to use all the samples",
    type=int)
parser.add_argument("--temperature", default=1,
                    help="Scaling factor for logits", type=int)
parser.add_argument("--num_toks_pred", default=1024,
                    help="Number of tokens to predict", type=int)
parser.add_argument("--bidirectional_eachstep", help="Compute bidirectional"
                    "attention every step. Consumes a lot of time but better results",
                    action='store_true')

FLAGS = parser.parse_args()


def _create_mask(qlen, mlen):
    """Simple bi-directional attention mask. Attend
    to all token in sequence and memory"""
    klen = qlen + mlen
    return tf.zeros((qlen, klen))

def get_preprocessor(examples, tokenize_fn, pad_ids):
    """
    Input:
    examples: [List[str]] input texts
    tokenize_fn: [function] encodes text into IDs
    Output:
    tf input features
    """
    def generator():
        for example in examples:
            tokens = tokenize_fn(example)
            yield pad_ids + tokens

    return generator


def get_input_dataset(preprocessor):
    """Returns tf.data.Dataset for input"""
    batch_size = FLAGS.batch_size
    max_mem_length = FLAGS.max_mem_length

    def mask(ids):
        example = {'input_k': ids}
        input_k = example['input_k'][-max_mem_length:]
        seq_len = tf.shape(input_k)[0]
        input_mask = tf.tile(
            tf.convert_to_tensor(
                [0],
                dtype=tf.float32),
            [seq_len])
        pad_len = tf.maximum(0, max_mem_length - seq_len)
        pad_tensor = tf.concat([[[pad_len]], [[0]]], axis=-1)
        input_k = tf.pad(input_k, pad_tensor, constant_values=0)
        input_mask = tf.pad(input_mask, pad_tensor, constant_values=1)
        example['input_mask'] = input_mask
        example['input_k'] = input_k
        example['seg_id'] = tf.convert_to_tensor([0] * max_mem_length)
        return example

    dataset = tf.data.Dataset.from_generator(preprocessor,
                                             output_types=tf.int32)
    dataset = dataset.map(mask)

    dataset = dataset.batch(batch_size,
                            drop_remainder=False)
    dataset.prefetch(1)
    return dataset



def inputs_and_mask(latest_tokens, batch_size):
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
    input_k = tf.tile([[CLS_ID]], [batch_size, 1])
    seg_id = tf.tile([[0]], [batch_size, 1])
    input_q = tf.tile([[1]], [batch_size, 1])

    if latest_tokens is not None:
        input_k = tf.concat([latest_tokens, input_k], axis=-1)
        seg_id = tf.tile(seg_id, [1, 2])
        input_q_0 = tf.tile([[0]], [batch_size, 1])
        input_q = tf.concat([input_q_0, input_q], axis=-1)
        target_mapping = tf.tile(tf.constant(
            [[[0], [1]]], dtype=tf.float32), [1, 1, batch_size])
        attn_masks = tf.convert_to_tensor([[0, 1], [0, 1]], dtype=tf.float32)
    else:
        attn_masks = tf.convert_to_tensor([[1]], dtype=tf.float32)
        target_mapping = tf.tile(tf.constant(
            [[[1]]], dtype=tf.float32), [1, 1, batch_size])

    attn_masks = tf.tile(attn_masks[None, :, :], [batch_size, 1, 1])
    input_q = tf.cast(input_q, tf.float32)

    return input_k, seg_id, attn_masks, input_q, target_mapping


def get_logits(xlnet_model, xlnet_config):
    """Builds the graph for calculating the final logits"""
    lookup_table = xlnet_model.get_embedding_table()
    tie_weight = True

    with tf.variable_scope("model", reuse=tf.AUTO_REUSE):
        initializer = xlnet_model.get_initializer()
        hidden = xlnet_model.get_sequence_output()[-1:, :, :]
        n_token = xlnet_config.n_token
        d_model = xlnet_config.d_model

        with tf.variable_scope('lm_loss'):
            if tie_weight:
                assert lookup_table is not None, \
                    'lookup_table cannot be None for tie_weight'
                softmax_w = lookup_table
            else:
                softmax_w = tf.get_variable(
                    'weight', [
                        n_token, d_model], dtype=hidden.dtype, initializer=initializer)

            softmax_b = tf.get_variable('bias', [n_token], dtype=hidden.dtype,
                                        initializer=tf.zeros_initializer())

            logits = tf.einsum('ibd,nd->ibn', hidden, softmax_w) + softmax_b

    return logits


def sampling_strategy():
    """Based on flags return either top_k or
    top_p strategy."""
    if FLAGS.top_p != 0:
        return 'top_p'

    return 'top_k'


def sample_token(logits):
    """
    Inputs:
    logits: tf.Tensor([batch_size,len,num_tokens])
    Outpus:
    samples: tf.Tensor([batch_size,len,1])
    """
    # credits: https://github.com/nshepperd/gpt-2

    logits /= FLAGS.temperature

    batch_size = tf.shape(logits)[0]
    seq_len = tf.shape(logits)[1]
    num_toks = tf.shape(logits)[2]

    if sampling_strategy() == 'top_p':
        logits_sorted = tf.sort(logits,
                                direction="DESCENDING",
                                axis=-1)
        probs = tf.nn.softmax(logits_sorted, axis=-1)
        cum_probs = tf.math.cumsum(probs,
                                   axis=-1,
                                   exclusive=True)
        logits_masked = tf.where(cum_probs < FLAGS.top_p,
                                 logits_sorted,
                                 tf.ones_like(logits_sorted) * 100)
        min_logits = tf.reduce_min(logits_masked, axis=-1)

        logits_masked = tf.where(logits < min_logits,
                                 tf.ones_like(logits) * -1e10,
                                 logits)

    elif sampling_strategy() == "top_k":
        if FLAGS.top_k != 0:
            values, _ = tf.nn.top_k(logits, k=FLAGS.top_k)
            min_values = values[:, :, -1:]
            logits_masked = tf.where(
                logits < min_values,
                tf.ones_like(logits, dtype=logits.dtype) * -1e10,
                logits,
            )
    else:
        raise NotImplementedError("Invalid sampling strategy")

    logits_masked = tf.reshape(logits_masked, (-1, num_toks))

    samples = tf.random.categorical(logits_masked,
                                    num_samples=1,
                                    dtype=tf.int32)

    probs = tf.nn.softmax(tf.reshape(logits, (-1, num_toks)), axis=-1)
    confidences = tf.gather_nd(params=probs, batch_dims=1, indices=samples)

    return tf.reshape(samples, (batch_size, seq_len, 1)),\
        tf.reshape(confidences, (batch_size, seq_len, 1))

def prediction_graph_memory():
    """Gets features and
    return predicted tokens)
    features: Dict[str:tf.train.features] Contains following features:
              input_k
              seg_id
              input_mask
    """

    features = {
        "input_k": tf.placeholder(tf.int32, (None, None)),
        "seg_id": tf.placeholder(tf.int32, (None, None)),
        "input_mask": tf.placeholder(tf.float32, (None, None))
    }

    # Building prediction graph
    # Transforming features for batch channel on last axis
    inp = tf.transpose(features["input_k"], [1, 0])
    seg_id = tf.transpose(features["seg_id"], [1, 0])
    inp_mask = tf.transpose(features["input_mask"], [1, 0])

    # Model config
    xlnet_config = xlnet.XLNetConfig(json_path=FLAGS.model_config_path)
    run_config = xlnet.create_run_config(False, True, FLAGS)
    run_config.mem_len = FLAGS.max_mem_length

    perm_mask = _create_mask(tf.shape(inp)[0], 0)[:, :, None]
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
    prev_confs = None
    batch_size = tf.shape(mems[0])[1]

    def cond(*_):
        """Dummy condition since we stop based on iteration"""
        return True

    def body(mems, latest_tokens, mem_mask, prev_tokens, prev_confs):
        """The main body of sampling loop.
        mem: cache memory--calculated hidden states
             of previous tokens
        latest_tokens: latest sampled tokens
        mem_mask: masking for setting previous memory zero. Used for padding
        prev_tokens: all the previous tokens including latest_tokens
        prev_confs: confidences of respective tokens in prev_tokens
        """

        # get dummy input token and permutation mask
        input_k, seg_id, perm_mask, input_q, target_mapping = \
            inputs_and_mask(latest_tokens,
                            batch_size)

        input_k = tf.transpose(input_k, (1, 0))
        input_q = tf.transpose(input_q, (1, 0))
        seg_id = tf.transpose(seg_id, (1, 0))
        perm_mask = tf.transpose(perm_mask, (1, 2, 0))
        # Set the hidden state of the padded tokens to be zero[
        for i, mem in enumerate(mems):
            mems[i] = (1 - mem_mask[:, :, None]) * mems[i]
        # Get logits
        xlnet_model = xlnet.XLNetModel(
            xlnet_config=xlnet_config,
            run_config=run_config,
            input_ids=input_k,
            seg_ids=seg_id,
            perm_mask=perm_mask,
            mems=mems,
            input_mask=None,
            inp_q=input_q,
            target_mapping=target_mapping)

        logits = get_logits(xlnet_model, xlnet_config)

        # Getting new memory
        new_mems = xlnet_model.get_new_memory()

        # sample a token
        logits = tf.transpose(logits, (1, 0, 2))
        sampled_tokens, confs = sample_token(logits)
        sampled_tokens = sampled_tokens[:, -1, :]  # Last token
        confs = confs[:, -1, :]  # Last token
        prev_tokens = sampled_tokens if prev_tokens is None \
            else tf.concat([prev_tokens, sampled_tokens], axis=1)
        prev_confs = confs if prev_confs is None \
            else tf.concat([prev_confs, confs], axis=1)
        # Cache the memory of the the last latest_tokens
        if latest_tokens is not None:
            merged_mems = []

            for i, mem in enumerate(mems):
                merged_mems.append(
                    tf.concat([mems[i][1:], new_mems[i][-2:-1]], axis=0))
            mem_mask = tf.concat(
                [mem_mask[1:], tf.zeros_like(mem_mask[:1])], axis=0)
            return [
                merged_mems,
                sampled_tokens,
                mem_mask,
                prev_tokens,
                prev_confs]

        return [mems, sampled_tokens, mem_mask, prev_tokens, prev_confs]

    mems, latest_tokens, mem_mask, prev_tokens, prev_confs = body(
        mems, latest_tokens, inp_mask, prev_tokens, prev_confs)

    args = tf.while_loop(
        cond=cond,
        body=body,
        maximum_iterations=FLAGS.num_toks_pred - 1,
        loop_vars=[mems, latest_tokens, mem_mask, prev_tokens, prev_confs],
        shape_invariants=[
            [tf.TensorShape([None, None, None]) for _ in range(len(mems))],
            tf.TensorShape([None, None]),
            tf.TensorShape([None, None]),
            tf.TensorShape([None, None]),
            tf.TensorShape([None, None])
        ]
    )

    predicted_tokens, predicted_confs = args[-2:]
    return (predicted_tokens, predicted_confs), features

def prediction_graph_no_memory():
    """Builds graphs and returns prediction and input features.
    Output:
    predictions: Tuple(Tensors) Currently returns sampled tokens and confidences
    features: Dict[str:tf.train.features] Contains following features:
              input_k
              seg_id
              input_mask
    """

    features = {
        "input_k": tf.placeholder(tf.int32, (None, None)),
        "seg_id": tf.placeholder(tf.int32, (None, None)),
        "input_mask": tf.placeholder(tf.float32, (None, None))
    }

    # Building prediction graph
    # Transforming features for batch channel on last axis
    inp = tf.transpose(features["input_k"], [1, 0])
    seg_id = tf.transpose(features["seg_id"], [1, 0])
    inp_mask = tf.transpose(features["input_mask"], [1, 0])

    # Model config
    xlnet_config = xlnet.XLNetConfig(json_path=FLAGS.model_config_path)
    run_config = xlnet.create_run_config(False, True, FLAGS)
    run_config.mem_len = FLAGS.max_mem_length

    perm_mask = _create_mask(tf.shape(inp)[0], 0)[:, :, None]
    # Getting the hidden states for the prompts

    prev_tokens = None
    prev_conf = None
    # target mapping
    seq_len = tf.shape(inp)[0]
    batch_size = tf.shape(inp)[-1]
    target_mapping = tf.concat(
        [tf.zeros((1, seq_len - 1, batch_size)), tf.ones((1, 1, batch_size))], axis=1)

    def cond(*_):
        """Dummy condition since we stop based on iteration"""
        return True

    def recalc(inp, inp_mask, seg_id, perm_mask):
        """Augment the inputs for the new token. Appends 1 row or columns accordingly"""
        input_q = tf.zeros_like(inp, dtype=tf.float32)
        inp = tf.pad(inp, tf.convert_to_tensor(
            [[0, 1], [0, 0]]), constant_values=0)
        inp_mask = tf.pad(inp_mask, tf.convert_to_tensor(
            [[0, 1], [0, 0]]), constant_values=0)
        seg_id = tf.pad(seg_id, tf.convert_to_tensor(
            [[0, 1], [0, 0]]), constant_values=0)
        col = tf.ones(tf.shape(perm_mask)[0:1], dtype=tf.float32)
        perm_mask = tf.concat([perm_mask, col[:, None, None]], axis=1)
        row = tf.concat([tf.zeros(tf.shape(perm_mask)[1:2] - 1, dtype=tf.float32),
                         tf.ones([1], dtype=tf.float32)], axis=0)
        perm_mask = tf.concat([perm_mask, row[None, :, None]], axis=0)
        input_q = tf.pad(input_q, tf.convert_to_tensor(
            [[0, 1], [0, 0]]), constant_values=1)

        return inp[1:], inp_mask[1:], perm_mask[1:, 1:], input_q[1:], seg_id[1:]

    def body(inp, inp_mask, seg_id, perm_mask, prev_tokens, prev_conf):
        """The main body of sampling loop.
        inp: input ids
        inp_mask: input masks for paddings, etc.
        seg_id: segment id. Zeros here.
        perm_mask: permutation mask to pass to transformer
        prev_tokens: all the previous tokens including latest_tokens
        prev_conf: confidences of respective tokens in prev_tokens
        """

        # get dummy input token and permutation mask
        input_k, input_mask, perm_mask, input_q, seg_id = recalc(
            inp, inp_mask, seg_id, perm_mask)
        # Get logits
        xlnet_model = xlnet.XLNetModel(
            xlnet_config=xlnet_config,
            run_config=run_config,
            input_ids=input_k,
            seg_ids=seg_id,
            input_mask=inp_mask,
            perm_mask=perm_mask,
            inp_q=input_q,
            target_mapping=target_mapping)

        logits = get_logits(xlnet_model, xlnet_config)

        # sample a token
        logits = tf.transpose(logits, (1, 0, 2))
        sampled_tokens, confidences = sample_token(logits)
        sampled_tokens = sampled_tokens[:, -1, :]  # Last token
        confidences = confidences[:, -1, :]
        prev_tokens = sampled_tokens if prev_tokens is None \
            else tf.concat([prev_tokens, sampled_tokens], axis=1)
        prev_conf = confidences if prev_conf is None \
            else tf.concat([prev_conf, confidences], axis=1)

        input_k = tf.concat(
            [input_k[:-1], tf.transpose(sampled_tokens, (1, 0))], axis=0)
        perm_mask = _create_mask(tf.shape(input_k)[0], 0)[:, :, None]
        return [input_k, input_mask, seg_id, perm_mask, prev_tokens, prev_conf]

    inp, inp_mask, seg_id, perm_mask, prev_tokens, prev_conf = body(
        inp, inp_mask, seg_id, perm_mask, prev_tokens, prev_conf)
    args = tf.while_loop(
        cond=cond,
        body=body,
        maximum_iterations=FLAGS.num_toks_pred - 1,
        loop_vars=[inp, inp_mask, seg_id, perm_mask, prev_tokens, prev_conf],
        shape_invariants=[
            tf.TensorShape([None, None]),
            tf.TensorShape([None, None]),
            tf.TensorShape([None, None]),
            tf.TensorShape([None, None, None]),
            tf.TensorShape([None, None]),
            tf.TensorShape([None, None]),
        ]
    )
    predicted_tokens, predicted_confs = args[-2:]
    return (predicted_tokens, predicted_confs), features


def main():
    """Main function routine"""

    tf.logging.set_verbosity(tf.logging.INFO)

    # Text encoding
    sp = spm.SentencePieceProcessor()
    sp.Load(FLAGS.spiece_model_file)

    def tokenize_fn(text):
        text = preprocess_text(text, lower=FLAGS.uncased)
        return encode_ids(sp, text)

    # Temporary fix for context problem.
    pad_txt = """In 1991, the remains of Russian Tsar Nicholas II and his family
                (except for Alexei and Maria) are discovered.
                The voice of Nicholas's young son, Tsarevich Alexei Nikolaevich, narrates the
                remainder of the story. 1883 Western Siberia,
                a young Grigori Rasputin is asked by his father and a group of men to perform magic.
                Rasputin has a vision and denounces one of the men as a horse thief. Although his
                father initially slaps him for making such an accusation, Rasputin watches as the
                man is chased outside and beaten. Twenty years later, Rasputin sees a vision of
                the Virgin Mary, prompting him to become a priest. Rasputin quickly becomes famous,
                 with people, even a bishop, begging for his blessing. """
    pad_ids = tokenize_fn(pad_txt)
    pad_ids.append(EOD_ID)

    to_special_symbol = {v:k for k,v in special_symbols.items()}
    def parse_ids(toks):
        """Uses sentencepiece to conver to text. Subsitute
        EOP_ID and EOD_ID with new lines, and rest with their names"""
        start = 0
        sent = ""
        for i in range(len(toks)):
          if toks[i] in to_special_symbol:
            if start<i:
              sent+=sp.decode_ids(toks[start:i])
            if toks[i] in [EOD_ID,EOP_ID]:
                replace_by = "\n\n"
            else:
                replace_by = to_special_symbol[toks[i]]

            sent+=f" {replace_by} "
            start=i+1
        if start<len(toks):
          sent+=sp.decode_ids(toks[start:])

        return sent

    if not FLAGS.bidirectional_eachstep:
        prediction_graph = prediction_graph_memory
    else:
        prediction_graph = prediction_graph_no_memory

    predictions, features = prediction_graph()
    gpu_options = tf.GPUOptions(allow_growth=True)
    model_utils.init_from_checkpoint(FLAGS, global_vars=False)

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                          gpu_options=gpu_options)) as sess:
        sess.run(tf.global_variables_initializer())

        def predict(examples):
            """Given a list of texts in examples
            return the result"""
            preprocessor = get_preprocessor(examples,
                                            tokenize_fn, pad_ids)
            dataset = get_input_dataset(preprocessor)
            example = dataset.make_one_shot_iterator().get_next()

            num_examples = len(examples)
            num_batches = int(np.ceil(num_examples / FLAGS.batch_size))

            for _ in tqdm(range(num_batches)):
                inputs = sess.run(example)
                output, conf = sess.run(
                    predictions, feed_dict={
                        features[k]: v for k, v in inputs.items()})
                for _output,_conf in zip(output,conf):
                    yield _output,_conf

        if FLAGS.unconditional or FLAGS.interactive:
            tf.logging.info("Interactive flag received."
                            " Ignoring input files if any.")
            while True:
                if FLAGS.unconditional:
                    text = ""
                else:
                    text = input("----PROMPT----\n")
                outputs = predict([text] * FLAGS.num_samples)
                for i, (output,_) in enumerate(outputs):
                    out = parse_ids(output.tolist())
                    print("======SAMPLE {}======".format(i))
                    print(out)
                    print("=====================")
                if FLAGS.unconditional:
                    break
        else:
            assert FLAGS.input_file!="", "Please provide either an"\
            " input file or set interactive flag for command line input"
            assert os.path.exists(FLAGS.input_file), FLAGS.input_file+\
            " does not exists"

            with open(FLAGS.input_file) as f:
                texts = []
                text = ""
                for line in f:
                    if line.strip()=="":
                        if text!="":
                            # Removing the last <eop> of prompt
                            # since it is not desired
                            if text.endswith("<eop>"):
                                text=text[:-5]
                            texts.extend([text]*FLAGS.num_samples)
                            text=""
                        continue
                    text+=re.sub(r'\n','<eop>',line)
                if text!="":
                    texts.extend([text]*FLAGS.num_samples)

            tf.logging.info("Got %s lines in the input file",
                            len(texts)//FLAGS.num_samples)
            tf.logging.info("Sampling each line %s times",FLAGS.num_samples)

            outputs = iter(predict(texts))
            with open(os.path.join(FLAGS.input_file+".xlnet"),'w') as f:
                for i in range(0,len(texts),FLAGS.num_samples):
                    f.write("\n======Example {}=================\n".format(i))
                    f.write(texts[i])
                    for j in range(FLAGS.num_samples):
                        output,_ = next(outputs)
                        out = parse_ids(output.tolist())
                        f.write("\n======Example {} SAMPLE {}======\n".format(i,j))
                        f.write(out)
                        f.write("\n==================================\n")



if __name__ == "__main__":

    # Fixed flags
    FLAGS.use_tpu = False
    FLAGS.use_bfloat16 = False
    FLAGS.dropout = 0
    FLAGS.dropatt = 0
    FLAGS.init = "normal"
    FLAGS.init_std = 0.02
    FLAGS.init_range = 0.1
    print("Args: {}".format(vars(FLAGS)))
    main()
