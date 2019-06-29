# XLnet-gen
Generate language using [XLNet](https://github.com/zihangdai/xlnet/). This is not an official implementation. Samples are included at the end of this README as well as in the `samples` folder.

# Usage
* Step 1:
  ```
  pip install -r requirements.txt
  ```
 * Step 2:
   Download and unzip pretrained XLNet model from https://github.com/zihangdai/xlnet/
   ```
   wget https://storage.googleapis.com/xlnet/released_models/cased_L-24_H-1024_A-16.zip
   ```
 * Step 3:
   Either run in interactive mode using `--interactive` flag or pass an input file using `--input_file` argument as described later. Use `--unconditional` for generating text without any conditioned text.
   ```
   python language_generation.py\
        --model_config_path=xlnet_cased_L-24_H-1024_A-16/xlnet_config.json\
        --init_checkpoint=xlnet_cased_L-24_H-1024_A-16/xlnet_model.ckpt\
        --spiece_model_file=xlnet_cased_L-24_H-1024_A-16/spiece.model\
        --interactive\
        --max_mem_length 512\
        --num_toks_pred=512\
        --num_samples=1\
        --top_p=0.9
   ```
# Notes

### Sampling schemes
- [x] top-k sampling: use `--top_k` flag, ensure `--top_p=0`
- [x] top-p sampling: use `--top_p` flag
- [ ] Permutation sampling

# Samples
**We’ve trained a large-scale unsupervised language model which generates coherent paragraphs of text, achieves state-of-the-art performance on many language modeling benchmarks, and performs rudimentary reading comprehension, machine translation, question answering, and summarization*** tasks in our lab using automated translation/text analysis with an automated computer system, Pro (Pro Text Analysis). From this training we have developed an automated translation tool, Pro Translation. Our system is known as the Pro Translation Suite, and is designed for translation between text, computer documents, and web pages. All of the tools in the Pro Translation Suite provide both text and "real time" translation. The program also features extensive user-friendly interfaces for user-directed development and customization of the software. The Pro Translation Suite features a number of features which offer new and innovative translation tasks. In addition, the Pro Translation Suite offers enhanced support for "realtime" translation systems, such as translation for Web pages, "real time" translation of language models, and machine translation.
We currently have a highly optimized robot in the development stage and support for this robot is currently being increased to include a (possibly) real-time translation engine, "The Trans-To-Trans". The Trans-To-Trans robot has been optimized, optimized and (and perhaps) may become a real-time translation engine, " The Trans-To-Trans". As one of our main goals, we will also be testing this robot against real time translation standards and benchmarks. Additionally, this robot has been made available publicly to evaluate and use, at no cost to the public.
The Trans-To-Trans robot has been built to meet a "real time" translation requirement (which is a requirement of English translation methods), which is the language to which all other robot translation will be converted. It has been designed for trans-lingual translation, such as translation between English and other popular languages. We expect to use this robot to do such translation in the future, and have been working on a translation tool, which we will be releasing near the end of the year.
The Trans-To-Trans robot has been optimized to meet a "real time" translation requirement. This is a requirement of English translation methods. We have been working on a translation tool, which will be released near the end of the year. We have been working on a translation tool, which will be released near the end of the year.

---
**Before boarding your rocket to Mars, remember to pack these items**. First, you must pack a rocket case, or booster, for your rocket. The launcher is a special product developed by the World Space Program, which is a government agency of the United States. When you get the launcher, the rocket will be built to you. And it will take only 3 days! Another important item you should pack is the rocket engine. The rocket engine is a component of the rocket, that is made from two parts. The engine consists of two "core" chambers. The main chamber is constructed of a ceramic material. The second chamber is made of stainless steel. A solid core of the second chamber, called the "fire pit", is made from carbon fiber. The "fire pit" is sealed with seal-on plastic and then put into a hollow box, or “spar case". The spar case contains all of the other components, such as the engines, and the components inside the spar case are then assembled at the launch site. While waiting for the rocket to be assembled, you can rest and drink your milk or water. At the launch site, you will be given some kind of an instrument, or scope, and guided with the scope. The mission is to launch the rocket. The rocket will leave the launch site. The rocket will travel approximately 5.5 hours.

When the rocket arrives, you will be given a helmet, and then the rocket will launch. As you are lifting off off, keep your eyes open, and try to keep on track. It is important that you stay open and focused on the mission. If you are able to do this, then you will be able to fly away safely. Also, remember to drink water and drink fresh milk. Then, try your best to keep your body from over heating up while on the flight of your rocket.

There are many things that come into use in a restaurant kitchen. A dish is a component of the cook and it serves a food in a particular form or a certain way. Other types of dishes are prepared according to the needs of the customer or the guest. There are also different types of food that comes into use in the food service company
