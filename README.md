## Debiasing the BottomUpTopDown Model for Visual Question Answering
This repo contains code to run the VQA-CP experiments from our paper ["Don’t Take the Easy Way Out: Ensemble Based Methods for Avoiding Known Dataset Biases"](https://arxiv.org/abs/1909.03683).
In particular, it contains code to a train VQA model so that it does
not make use of question-type priors when answering questions, and evaluate it on [VQA-CP v2.0](https://www.cc.gatech.edu/~aagrawal307/vqa-cp/).

This repo is a fork of [this](https://github.com/hengyuan-hu/bottom-up-attention-vqa/) 
implementation of the [BottomUpTopDown VQA model](https://arxiv.org/abs/1707.07998). This fork extends the implementation so it can be used
on [VQA-CP v2.0](https://www.cc.gatech.edu/~aagrawal307/vqa-cp/), and supports the debiasing methods from our paper. 


### Prerequisites

Make sure you are on a machine with a NVIDIA GPU and Python 2 with about 100 GB disk space.

1. Install [PyTorch v0.3](http://pytorch.org/) with CUDA and Python 2.7.
2. Install h5py, pillow, and tqdm           

### Data Setup

All data should be downloaded to a 'data/' directory in the root
directory of this repository.

The easiest way to download the data is to run the provided script
`tools/download.sh` from the repository root. The features are
provided by and downloaded from the original authors'
[repo](https://github.com/peteanderson80/bottom-up-attention). If the
script does not work, it should be easy to examine the script and
modify the steps outlined in it according to your needs. Then run
`tools/process.sh` from the repository root to process the data to the
correct format.

### Setup Example
On a fresh machine with Ubuntu 18.04, I was able to setup everything by installing [Cuda 10.0](https://developer.nvidia.com/cuda-10.0-download-archive), then running:

```
sudo apt update
sudo apt install unzip
sudo apt install python-pip
pip2 install torch==0.3.1
pip2 install h5py, tqdm, pillow 
bash tools/download.sh
bash tools/process.sh
```

### Training

Run `python main.py --output_dir /path/to/output --seed 0` to start training our learned-mixin +H VQA-CP model, see the command line options
for how to use other ensemble method, or how to train on non-changing priors VQA 2.0.


### Code Changes
In general we have tried to minimizes changes to the original codebase to reduce the risk of adding bugs, the main changes are:

1. The download and preprocessing script also setup [VQA-CP 2.0](https://www.cc.gatech.edu/~aagrawal307/vqa-cp/)
2. We use the filesystem, instead of HDF5, to store image feature. On my machine this is about a 1.5-3.0x speed up
3. Support dynamically loading the image features from disk during training so models can be trained
on machines with less RAM
4. Debiasing objectives are added in `vqa_debiasing_objectives.py`
5. Some additional arguments are added to `main.py` that control the debiasing objective
6. Minor quality of life improvements and tqdm progress monitoring
