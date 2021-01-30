# Update 30-01-2021
This repository is archived. Please use https://github.com/huggingface/transformers which supports XLNet language generation in both pytorch and tensorflow

# XLnet-gen


Generate language using [XLNet](https://github.com/zihangdai/xlnet/). This is not an official implementation. Samples are included at the end of this README as well as in the `samples` folder.

Medium article as a summary of this effort: https://medium.com/@amanrusia/xlnet-speaks-comparison-to-gpt-2-ea1a4e9ba39e

Colab notebook where you can give prompts:  https://colab.research.google.com/drive/12u-CmB9evMIASNOqJtDW26gmNvSgepBv

# Usage
* Step 1: Download and install requirements (change tensorflow to tensorflow-gpu in requirements.txt if needed)
  ```
  git clone https://github.com/rusiaaman/XLnet-gen.git && cd XLnet-gen
  pip install -r requirements.txt
  ```
 * Step 2:
   Download and unzip pretrained XLNet model from https://github.com/zihangdai/xlnet/
   ```
   wget https://storage.googleapis.com/xlnet/released_models/cased_L-24_H-1024_A-16.zip
   unzip cased_L-24_H-1024_A-16.zip
   ```
 * Step 3:
   Either run in interactive mode using `--interactive` flag or pass an input file using `--input_file` argument as described later. Use `--unconditional` for generating text without any conditioned text.
   ```
   python language_generation.py\
        --model_config_path=xlnet_cased_L-24_H-1024_A-16/xlnet_config.json\
        --init_checkpoint=xlnet_cased_L-24_H-1024_A-16/xlnet_model.ckpt\
        --spiece_model_file=xlnet_cased_L-24_H-1024_A-16/spiece.model\
        --interactive\
        --max_mem_length=256\
        --num_toks_pred=256\
        --num_samples=1\
        --top_p=0.9\
        --bidirectional_eachstep
   ```
# Important Notes
## Methodology
XLNet is a novel permutation based language model. In current implementation of XLNet-gen, we generate texts from left to right.

XLNet is trained using `num_predict=85`, which means 85 tokens out of 512 in a single example are predicted at a time. **More importantly rest of the 512-85 = 427 tokens can attend to each other in the attention mechanism (bidrectional attention)**. This creates problems with conventional causal attention mechanism during language generation. Following problems were faced:

  * Use of small context leads to gibberish predictions. Currently a hard-coded random text is included as a leading text followed by `<eod>`, the end of document token, along with the desired context. This helps with small prompts.
  * Due to the nature of pretraining, context tokens attend to each other in bi-directional way. And the context is spread throughout the input of the model. Because of this generating tokens left to right in causal way leads to suboptimal output. Recalculating hidden states each step allows us to have bidirectional attention to each new generated token which substantially improve the generation. To do the same use `--bidirectional_eachstep` flag
 
## Explanation of flags (specific to XLNet-gen)

* `--max_mem_length` Max sequence length used for prediction. NOTE: number of tokens to be predicted can be greater than this, but the context gets truncated at the beginning. For `--autoregressive` case, this sets the size of the 'memory'.
* `--num_toks_pred` Number of tokens to predict. This can be as large as we want, however the context is truncated if longer than `max_mem_length` for the default case.
* `--num_samples` For each prompt the number of samples to generate.
* `--interactive` Command line prompt input.
* `--input_file` path to the file which is used for conditional prompts. Prompts are separted by an empty line. The output is generated in the same location in a new file with the same file name appended with ".xlnet".
* `--top_p` top_p paramter for nucleus sampling. Set this 0 if you want to use top_k sampling process.
* `--top_k` top_k parameter for top_k sampling. Only top_k most probable tokens are considered for sampling. Set `top_p=0` if you want to use this.
* `--unconditional` Generates unconditional samples. Ignores `--interactive` and `--input_file` flags.
* `--bidirectional_eachstep` leads to much better output at the expense of computation. Explanation in methodology.

## Sampling schemes
- [x] top-k sampling: use `--top_k` flag, ensure `--top_p=0`
- [x] Nucleus sampling: use `--top_p` flag
- [ ] Permutation sampling

## Notes on quality of the samples
- There is a vast difference in quality with and without `bidirectional_eachstep` flag, which turns on re-calculation of hidden states with bidirectional attention everytime a new token is generated. This is probably due to the way XLNet was pretrained--with sparse masks and bidrectional context. However, I am currently investigating this issue and this could be an area of improvement for XLNet.
- Generation of artifacts like empty quotes `""`, `" "`, multiple hyphens `---`, and combination of them `""-"` can all be attributed to bad training data. **Specifically, there seems to be bugs in https://github.com/attardi/wikiextractor which leads to generation of empty quotes and other such artifacts.** This is probably the same library that was used by the authors.
- Wikipedia has a lot of ellipses in its articles which is reflected in the generation. The wiki data dump has it in the form with and without spaces: both `. . .`, and `...`. 
- The XLNet can only predict end of paragraph and end of documents, but not new line characters or tabs, so it doesn't generate good structure of the documents
- Vocabulary is limited to English and not all Unicode characters are in the vocabulary. Other language characters and emojis can't be generated are decoded as <unk>. 

# Samples
**We’ve trained a large-scale unsupervised language model which generates coherent paragraphs of text, achieves state-of-the-art performance on many language modeling benchmarks, and performs rudimentary reading comprehension, machine translation, question answering, and summarization** tasks in our lab using automated translation/text analysis with an automated computer system, Pro (Pro Text Analysis). From this training we have developed an automated translation tool, Pro Translation. Our system is known as the Pro Translation Suite, and is designed for translation between text, computer documents, and web pages. All of the tools in the Pro Translation Suite provide both text and "real time" translation. The program also features extensive user-friendly interfaces for user-directed development and customization of the software. The Pro Translation Suite features a number of features which offer new and innovative translation tasks. In addition, the Pro Translation Suite offers enhanced support for "realtime" translation systems, such as translation for Web pages, "real time" translation of language models, and machine translation.

We currently have a highly optimized robot in the development stage and support for this robot is currently being increased to include a (possibly) real-time translation engine, "The Trans-To-Trans". The Trans-To-Trans robot has been optimized, optimized and (and perhaps) may become a real-time translation engine, " The Trans-To-Trans". As one of our main goals, we will also be testing this robot against real time translation standards and benchmarks. Additionally, this robot has been made available publicly to evaluate and use, at no cost to the public.

The Trans-To-Trans robot has been built to meet a "real time" translation requirement (which is a requirement of English translation methods), which is the language to which all other robot translation will be converted. It has been designed for trans-lingual translation, such as translation between English and other popular languages. We expect to use this robot to do such translation in the future, and have been working on a translation tool, which we will be releasing near the end of the year.
The Trans-To-Trans robot has been optimized to meet a "real time" translation requirement. This is a requirement of English translation methods. We have been working on a translation tool, which will be released near the end of the year. We have been working on a translation tool, which will be released near the end of the year.

---
**Before boarding your rocket to Mars, remember to pack these items**. First, you must pack a rocket case, or booster, for your rocket. The launcher is a special product developed by the World Space Program, which is a government agency of the United States. When you get the launcher, the rocket will be built to you. And it will take only 3 days! Another important item you should pack is the rocket engine. The rocket engine is a component of the rocket, that is made from two parts. The engine consists of two "core" chambers. The main chamber is constructed of a ceramic material. The second chamber is made of stainless steel. A solid core of the second chamber, called the "fire pit", is made from carbon fiber. The "fire pit" is sealed with seal-on plastic and then put into a hollow box, or “spar case". The spar case contains all of the other components, such as the engines, and the components inside the spar case are then assembled at the launch site. While waiting for the rocket to be assembled, you can rest and drink your milk or water. At the launch site, you will be given some kind of an instrument, or scope, and guided with the scope. The mission is to launch the rocket. The rocket will leave the launch site. The rocket will travel approximately 5.5 hours.

When the rocket arrives, you will be given a helmet, and then the rocket will launch. As you are lifting off off, keep your eyes open, and try to keep on track. It is important that you stay open and focused on the mission. If you are able to do this, then you will be able to fly away safely. Also, remember to drink water and drink fresh milk. Then, try your best to keep your body from over heating up while on the flight of your rocket.

There are many things that come into use in a restaurant kitchen. A dish is a component of the cook and it serves a food in a particular form or a certain way. Other types of dishes are prepared according to the needs of the customer or the guest. There are also different types of food that comes into use in the food service company


# Todo
* [ ] Comparison with GPT-2.
* [ ] Permutation based decoding instead of left-to-right only.
