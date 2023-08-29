Dataset for Contour Detection and 3D Reconstruction
====================================

Understanding the finer details of a 3D object, its contours, is the first step toward a physical understanding of an object. Many real-world application domains require adaptable 3D object shape recognition models, usually with little training data. For this purpose, we develop the first automatically generated contour labeled dataset, bypassing manual human labeling. Using this dataset, we study the performance of current state-of-the-art instance segmentation algorithms on detecting and labeling the contours. We produce promising visual results with accurate contour prediction and labeling. We demonstrate that our finely labeled contours can help downstream tasks in computer vision, such as 3D reconstruction from a 2D image.

<img width="834" alt="pipeline" src="https://github.com/santhanamhari/Automated-Line-Labelling-Dataset/assets/40223805/c84e2ae4-1ba4-4b05-a81a-db25aa1518fe">


The code to generate the dataset is broken into the following sections:

  * [Setup and Dependencies](#setup-and-dependencies)
  * [Usage](#download-data)
  * [Training](#training)
  * [Evaluation](#evaluation)


Setup and Dependencies
----------------------

Clone this repository, create an environment, and install dependencies:

```sh
git clone https://github.com/santhanamhari/Automated-Line-Labelling-Dataset.git
conda create -n line_label python=3.10

# activate environment
conda activate line_label

# install dependencies
pip install -r requirements.txt
```

Usage
-------------
We provide the code for generating our dataset in the `code` folder. 

- `main.py` processes the stl meshes from the `raw_meshes` folder that are prescribed in `genus/total_genus.txt`

- `edge_detection.py` extracts the contours and also classifies them as obscuring, concave, or convex. 

- `group.py` performs initial grouping and fine-tuned grouping based on a heap's priority for grouping. 


To generate the dataset, run the following line of code:

```sh
python3 code/main.py
```



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


Results
----------------------------------


Publication
----------------------------------
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