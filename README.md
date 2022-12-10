#  Music Separation Enhancement With Generative Modeling

This is the official implementation of the Make it Sound Good (MSG) model from our 2022 ISMIR paper "Music Separation Enhancement with Generative Modeling" [\[paper\]](https://arxiv.org/pdf/2208.12387.pdf)[ \[website\]](https://interactiveaudiolab.github.io/project/msg.html)

We introduce Make it Sound Good (MSG), a post-processor that enhances the output quality of source separation systems like Demucs, Wavenet, Spleeter, and OpenUnmix
![](https://interactiveaudiolab.github.io/assets/images/projects/MSG-hero-image.png)

## Table of Contents
- [Setup](#Setup)
- [Training](#Training)
- [Inference](#Inference)
- [Citation](#Citation)

## Setup
1. We train our model using salient source samples from the training data. To get the salient source samples, our training loop uses [nussl's](https://github.com/nussl/nussl/tree/salient_mixsrc2/nussl) SalientExcerptMixSourceFolder class from the salient_mixsrc2 branch. The specific branch of the repo can be downloaded using the steps below:
```
$ git clone https://github.com/nussl/nussl.git
$ cd nussl
$ git checkout salient_mixsrc2
$ pip install -e .
```
2. Download our repo from github.
```
$ pip install https://github.com/interactiveaudiolab/MSG.git
```
3. Download the requirements.txt.
```
$ pip install -r requirements.txt
```
4. If you would like to use our [pretrained checkpoints on huggingface download](https://huggingface.co/boazcogan/MSG_pretrained_checkpoints/tree/main) the model.

## Training



## Inference
1. Our inference script passes an audio file through MSG loaded from a specified checkpoint. Note: The inference script is currently set to work with parameters specified in ```training_config_template.yml```. If you have a checkpoint with different parameters, you will need to modify the definition of the generator within the script. 

2. To call the inference script, use the following command

```
$ python Run_Inference.py -a <path to audio file> -g <path to checkpoint>
```

3. The inference script will write to the directory ```msg_output``` with the file name ```msg_output/<input file name>```


## Citation

```
@inproceedings{schaffer2022music,
title={Music Separation Enhancement with Generative Modeling},
author={Schaffer, Noah and Cogan, Boaz and Manilow, Ethan and Morrison, Max and Seetharaman, Prem and Pardo, Bryan},
booktitle={International Society for Music Information Retrieval (ISMIR)},
month={December},
year={2022}
}
```
