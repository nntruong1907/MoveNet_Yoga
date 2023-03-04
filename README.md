# MoveNet_Yoga
This project identifies and evaluates several yoga movements.

## Related Library

- Python (programming language): <a href="https://www.python.org/downloads/release/python-31010/" alt="Python"><img src="https://img.shields.io/badge/python-v3.10.10-blue?logo=python" /></a>

- The PyPA recommended tool for installing Python packages: <a href="https://pypi.org/project/pip/" alt="pip"><img src="https://img.shields.io/badge/pypi-v23.0.1-blue?logo=pypi" /></a>

- Fundamental package for array computing in Python: <a href="https://numpy.org/" alt="numpy"><img src="https://img.shields.io/badge/numpy-v1.23.4-blue?logo=numpy" /></a>

- Powerful data structures for data analysis, time series, and statistics: <a href="https://pandas.pydata.org/" alt="pandas"><img src="https://img.shields.io/badge/pandas-v1.5.1-blue?logo=pandas" /></a>

- Fast, Extensible Progress Meter: <a href="https://tqdm.github.io/" alt="tqdm"><img src="https://img.shields.io/badge/tqdm-v4.64.1-blue?logo=tqdm" /></a>

- Pure python download utility: <a href="https://pypi.org/project/wget/" alt="wget"><img src="https://img.shields.io/badge/wget-v3.2-blue?logo=wget" /></a>

- Source machine learning framework for everyone: <a href="https://www.tensorflow.org/" alt="tensorflow"><img src="https://img.shields.io/badge/tensorflow-v2.10.0-blue?logo=tensorflow" /></a>

- Python plotting package: <a href="https://matplotlib.org/" alt="matplotlib"><img src="https://img.shields.io/badge/matplotlib-v3.6.1-blue?logo=matplotlib" /></a>

- Wrapper package for OpenCV python bindings: <a href="https://github.com/opencv/opencv-python" alt="opencv-python"><img src="https://img.shields.io/badge/opencv python-v4.6.0.66-blue?logo=opencv" /></a>

- A set of python modules for machine learning and data mining: <a href="https://scikit-learn.org/stable/" alt="scikit learn"><img src="https://img.shields.io/badge/scikit learn-v1.1.3-blue?logo=scikitlearn" /></a>

- Parser for command-line options, arguments and sub-commands: <a href="https://docs.python.org/3/library/argparse.html" alt="argparse"><img src="https://img.shields.io/badge/argparse-v1.4.0-blue?logo=argparse" /></a>

## Recommended IDE Setup

[VSCode](https://code.visualstudio.com/) + [Python](https://www.python.org/downloads/release/python-31010/)



## Project Setup
 
<details><summary> <b>Create a virtual environment for the project</b> </summary>

Virtual environment `venv`
```sh
python -m venv venv
```
Activate virtual environment:
```sh
venv\Scripts\activate
```
Update to the latest pip version:
```sh
python.exe -m pip install --upgrade pip
```

</details>


<b>- Install the external dependencies needed for the project:</b>
```sh
pip install -r setup.txt
```

## Make data 

Skeleton is extracted like below FIG

<div align="center">
    <a href="./">
        <img src="./images/draw_skeleton.png" width="60%"/>
    </a>
</div>

``` shell
python make_csv.py --source yoga_cg --data data
```

## Training

Make model `.h5` to detect yoga poses 

``` shell
python train.py --train data/train_data.csv --test data/test_data.csv --output models/model_yoga_LSTM.h5 --epochs 200 --batch 64 --patience 20
```

## Inference detection 
Detect several pose of yoga include: chair, cobra, dog, tree, and warrior

``` shell
python test.py --model models/model_yoga_LSTM.h5 --data images/tree3.jpg
```


# Deactivate env

Once you’re done working with this virtual environment, you can deactivate it:
```sh
deactivate
```
