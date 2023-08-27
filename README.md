Dataset for Contour Detection and 3D Reconstruction
====================================

Understanding the finer details of a 3D object, its contours, is the first step toward a physical understanding of an object. Many real-world application domains require adaptable 3D object shape recognition models, usually with little training data. For this purpose, we develop the first automatically generated contour labeled dataset, bypassing manual human labeling. Using this dataset, we study the performance of current state-of-the-art instance segmentation algorithms on detecting and labeling the contours. We produce promising visual results with accurate contour prediction and labeling. We demonstrate that our finely labeled contours can help downstream tasks in computer vision, such as 3D reconstruction from a 2D image.

<img width="834" alt="pipeline" src="https://github.com/santhanamhari/Automated-Line-Labelling-Dataset/assets/40223805/c84e2ae4-1ba4-4b05-a81a-db25aa1518fe">




The code to generate the dataset is broken into the following sections:

  * [Setup and Dependencies](#setup-and-dependencies)
  * [Download Data](#download-data)
  * [Training](#training)
  * [Evaluation](#evaluation)
  * [Pretrained Checkpoint](#pretrained-checkpoint)

If you use this code in your research, please consider citing:

```text
@inproceedings{santhanam2023automated,
  title={Automated Line Labelling: Dataset for Contour Detection and 3D Reconstruction},
  author={Santhanam, Hari and Doiphode, Nehal and Shi, Jianbo},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={3136--3145},
  year={2023}
}
```


Setup and Dependencies
----------------------

### Anaconda or Miniconda

1. Install Anaconda or Miniconda distribution based on Python3+ from their [downloads' site][2].
2. Clone this repository and create an environment:

```sh
git clone https://www.github.com/batra-mlp-lab/visdial-challenge-starter-pytorch
conda create -n visdialch python=3.6

# activate the environment and install all dependencies
conda activate visdialch
cd visdial-challenge-starter-pytorch/
pip install -r requirements.txt

# install this codebase as a package in development version
python setup.py develop
```

**Note:** Docker setup is necessary if you wish to extract image features using Detectron.

### Docker

We provide a Dockerfile which creates a light-weight image with all the dependencies installed.

1. Install [nvidia-docker][18], which enables usage of GPUs from inside a container.
2. Build the image as:

```sh
cd docker
docker build -t visdialch .
```

3. Run this image in a container by setting user+group, attaching project root (this codebase) as a volume and setting shared memory size according to your requirements (depends on the memory usage of your model).

```sh
nvidia-docker run -u $(id -u):$(id -g) \
                  -v $PROJECT_ROOT:/workspace \
                  --shm-size 16G visdialch /bin/bash
```

We recommend this development workflow, attaching the codebase as a volume would immediately reflect source code changes inside the container environment. We also recommend containing all the source code for data loading, models and other utilities inside `visdialch` directory. Since it is a setuptools-style package, it makes handling of absolute/relative imports and module resolving less painful. Scripts using `visdialch` can be created anywhere in the filesystem, as far as the current conda environment is active.


Download Data
-------------

1. Download the VisDial v1.0 dialog json files from [here][7] and keep it under `$PROJECT_ROOT/data` directory, for default arguments to work effectively.

2. Get the word counts for VisDial v1.0 train split [here][9]. They are used to build the vocabulary.

3. We also provide pre-extracted image features of VisDial v1.0 images, using a Faster-RCNN pre-trained on Visual Genome. If you wish to extract your own image features, skip this step and download VIsDial v1.0 images from [here][7] instead. Extracted features for v1.0 train, val and test are available for download at these links.

  * [`features_faster_rcnn_x101_train.h5`](https://s3.amazonaws.com/visual-dialog/data/v1.0/2019/features_faster_rcnn_x101_train.h5): Bottom-up features of 36 proposals from images of `train` split.
  * [`features_faster_rcnn_x101_val.h5`](https://s3.amazonaws.com/visual-dialog/data/v1.0/2019/features_faster_rcnn_x101_val.h5): Bottom-up features of 36 proposals from images of `val` split.
  * [`features_faster_rcnn_x101_test.h5`](https://s3.amazonaws.com/visual-dialog/data/v1.0/2019/features_faster_rcnn_x101_test.h5): Bottom-up features of 36 proposals from images of `test` split.

4. We also provide pre-extracted FC7 features from VGG16, although the `v2019` of this codebase does not use them anymore.

  * [`features_vgg16_fc7_train.h5`](https://s3.amazonaws.com/visual-dialog/data/v1.0/2019/features_vgg16_fc7_train.h5): VGG16 FC7 features from images of `train` split.
  * [`features_vgg16_fc7_val.h5`](https://s3.amazonaws.com/visual-dialog/data/v1.0/2019/features_vgg16_fc7_val.h5): VGG16 FC7 features from images of `val` split.
  * [`features_vgg16_fc7_test.h5`](https://s3.amazonaws.com/visual-dialog/data/v1.0/2019/features_vgg16_fc7_test.h5): VGG16 FC7 features from images of `test` split.


Training
--------

This codebase supports both generative and discriminative decoding; read more [here][16]. For reference, we have Late Fusion Encoder from the Visual Dialog paper.

We provide a training script which accepts arguments as config files. The config file should contain arguments which are specific to a particular experiment, such as those defining model architecture, or optimization hyperparameters. Other arguments such as GPU ids, or number of CPU workers should be declared in the script and passed in as argparse-style arguments.

Train the baseline model provided in this repository as:

```sh
python train.py --config-yml configs/lf_disc_faster_rcnn_x101.yml --gpu-ids 0 1 # provide more ids for multi-GPU execution other args...
```

To extend this starter code, add your own encoder/decoder modules into their respective directories and include their names as choices in your config file. We have an `--overfit` flag, which can be useful for rapid debugging. It takes a batch of 5 examples and overfits the model on them.

### Saving model checkpoints

This script will save model checkpoints at every epoch as per path specified by `--save-dirpath`. Refer [visdialch/utils/checkpointing.py][19] for more details on how checkpointing is managed.

### Logging

We use [Tensorboard][5] for logging training progress. Recommended: execute `tensorboard --logdir /path/to/save_dir --port 8008` and visit `localhost:8008` in the browser.


Evaluation
----------

Evaluation of a trained model checkpoint can be done as follows:

```sh
python evaluate.py --config-yml /path/to/config.yml --load-pthpath /path/to/checkpoint.pth --split val --gpu-ids 0
```

This will generate an EvalAI submission file, and report metrics from the [Visual Dialog paper][13] (Mean reciprocal rank, R@{1, 5, 10}, Mean rank), and Normalized Discounted Cumulative Gain (NDCG), introduced in the first Visual Dialog Challenge (in 2018).

The metrics reported here would be the same as those reported through EvalAI by making a submission in `val` phase. To generate a submission file for `test-std` or `test-challenge` phase, replace `--split val` with `--split test`.


Results and pretrained checkpoints
----------------------------------

Performance on `v1.0 test-std` (trained on `v1.0` train + val):

  Model  |  R@1   |  R@5   |  R@10  | MeanR  |  MRR   |  NDCG  |
 ------- | ------ | ------ | ------ | ------ | ------ | ------ |
[lf-disc-faster-rcnn-x101][12] | 0.4617 | 0.7780 | 0.8730 |  4.7545| 0.6041 | 0.5162 |
[lf-gen-faster-rcnn-x101][20]  | 0.3620 | 0.5640 | 0.6340 | 19.4458| 0.4657 | 0.5421 |

