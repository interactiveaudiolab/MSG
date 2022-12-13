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
$ git clone https://github.com/interactiveaudiolab/MSG.git
```
3. Change to the MSG repo folder and download the requirements.txt.
```
$ cd MSG
$ pip install -r requirements.txt
```
4. If you would like to use our [pretrained checkpoints on huggingface download](https://huggingface.co/boazcogan/MSG_pretrained_checkpoints/tree/main) the model.

## Training

The directory format needs to be as follows


```
BaseFolder/
    train/
        vocals/
            song1.wav
        drums/
            ...
        bass/
            ...
        other/
            ...
    valid/
        ...
    test/
        ...

```

If you are training on the output of multiple separators, data from all separators must be in the same directory.

If you would like to use weights and biases, you'll need to login. (Weights and biases is integrated into our code, it should still be runable without logging in.)

To run the training loop follow the steps below:

1. Copy the config template to a new file and change the parameters for file paths to match your actual file paths.
2. In addition to file paths the following paramters need to be updated in the config:
<br><t>a.<t> source: list string name of the source/s getting separated for logging and dataset loading purposes. e.g: ['bass', 'drums']
<br><t>b.<t> validation_song: the file path to a specific song, this parameter allows us to see how the model's behavior changes during training using a specific example. If using weights and biases this example will be uploaded to the service. 
<br><t>c.<t> song_names: Specific MSG output examples that the user would like to listen to, written locally with the corresponding song name during testing.
<br><t>d.<t> evaluation_models: Model checkpoint/s to use during evaluation if you would like to compare multiple models directly.
3. Run the python file main.py using the config. e.g.:
    ```
    $ python main.py -e <MY_CONFIG.yml>
    ```

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
