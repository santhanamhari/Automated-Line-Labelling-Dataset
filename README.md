Dataset for Contour Detection and 3D Reconstruction
====================================

Understanding the finer details of a 3D object, its contours, is the first step toward a physical understanding of an object. Many real-world application domains require adaptable 3D object shape recognition models, usually with little training data. For this purpose, we develop the first automatically generated contour labeled dataset, bypassing manual human labeling. Using this dataset, we study the performance of current state-of-the-art instance segmentation algorithms on detecting and labeling the contours. We produce promising visual results with accurate contour prediction and labeling. We demonstrate that our finely labeled contours can help downstream tasks in computer vision, such as 3D reconstruction from a 2D image.

<img width="834" alt="pipeline" src="https://github.com/santhanamhari/Automated-Line-Labelling-Dataset/assets/40223805/c84e2ae4-1ba4-4b05-a81a-db25aa1518fe">


The code to generate the dataset is broken into the following sections:

  * [Setup and Dependencies](#setup-and-dependencies)
  * [Usage](#download-data)
  * [Results](#training)



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

This will generate the rendered images (`output/images`), extracted/classified contours (`output/gt_edges`), and grouped contours (`output/groups`). This may take some time to render, so the full results are in the `results` folder. 


Results
----------




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