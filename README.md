# Project Title

Fault diagnosis.

## Table of Contents

1. [Introduction](#introduction)
2. [Features](#features)
3. [Requirements](#requirements)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Project Structure](#project-structure)
7. [Data](#data)
8. [Models](#models)
9. [Evaluation](#evaluation)
10. [Timeline](#timeline)
11. [License](#license)
12. [Acknowledgments](#acknowledgments)

## Introduction

Provide a more detailed introduction to the project. Explain its purpose, goals, and any relevant background information.

## Features

List the key features or functionalities of the project.

## Requirements
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Machine Learning](https://img.shields.io/badge/Machine_Learning-blue?style=for-the-badge&logo=python&logoColor=ffdd54)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)
![Windows Terminal](https://img.shields.io/badge/Windows%20Terminal-%234D4D4D.svg?style=for-the-badge&logo=windows-terminal&logoColor=white)
![Shell Script](https://img.shields.io/badge/Bash-%23121011.svg?style=for-the-badge&logo=gnu-bash&logoColor=white)
![Django](https://img.shields.io/badge/django-%23092E20.svg?style=for-the-badge&logo=django&logoColor=white)
![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)
![JS](https://img.shields.io/badge/logo-javascript-blue?logo=javascript)


## Installation

Provide step-by-step instructions on how to install the project and its dependencies. This may include setting up a virtual environment, installing libraries, or configuring specific settings.

```bash
# Example installation steps
git clone https://github.com/yourusername/yourproject.git
cd yourproject
pip install -r requirements.txt
```

## Usage

Explain how to use the project. Include examples of command-line commands or code snippets.

```bash
# Example usage
python main.py --input data/input.csv --output results/output.csv
```

## Project Structure

Explain the organization of the project's directories and files. Highlight the purpose of important files or folders.

```
project-root/
|-- dataset/
|   |-- raw/
|   |-- processed/
|-- notebooks/
|   |-- tep-fault-diagnosis-tree-classification.ipynb
|-- web-dashboard/
|   |-- static/
|       |-- style.css
|       |-- script.js
|   |-- templates/
|       |-- index.html 
|       |-- layout.html
|   |-- main.py
|   |-- utlis.py
|   |-- requirements.txt
|-- README.md
|-- requirements.txt
```

## Data
Dataset link: [kaggle source](https://www.kaggle.com/datasets/averkij/tennessee-eastman-process-simulation-dataset)

This dataverse contains the data referenced in Rieth et al. (2017). Issues and Advances in Anomaly Detection Evaluation for Joint Human-Automated Systems. To be presented at Applied Human Factors and Ergonomics 2017.

Each .RData file is an external representation of an R dataframe that can be read into an R environment with the 'load' function. The variables loaded are named ‘fault_free_training’, ‘fault_free_testing’, ‘faulty_testing’, and ‘faulty_training’, corresponding to the RData files.

**Each dataframe contains 55 columns:**

* Column 1 ('faultNumber') ranges from 1 to 20 in the “Faulty” datasets and represents the fault type in the TEP. The “FaultFree” datasets only contain fault 0 (i.e. normal operating conditions).

* Column 2 ('simulationRun') ranges from 1 to 500 and represents a different random number generator state from which a full TEP dataset was generated (Note: the actual seeds used to generate training and testing datasets were non-overlapping).

* Column 3 ('sample') ranges either from 1 to 500 (“Training” datasets) or 1 to 960 (“Testing” datasets). The TEP variables (columns 4 to 55) were sampled every 3 minutes for a total duration of 25 hours and 48 hours respectively. Note that the faults were introduced 1 and 8 hours into the Faulty Training and Faulty Testing datasets, respectively.

* Columns 4 to 55 contain the process variables; the column names retain the original variable names.

for data preprocessing refer:
1. [TEP data analysis](tep-fault-diagnosis-tree-classification.ipynb)
```
#.rdata file reader
import pyreadr

#zipfile extract
import zipfile as zp

#extract from zipfile
with zp.ZipFile("dataset\TEP_FaultFree_Training.RData (1).zip", 'r') as zip_ref:
    zip_ref.extractall("dataset")

# pyreadr converts rData into dictionary type object.
result = pyreadr.read_r("dataset/TEP_Faulty_Training.RData")
result1 = pyreadr.read_r("dataset/TEP_FaultFree_Training.RData")

#dataframes
df_train = result['faulty_training']
df_ff = result1['fault_free_training']


#save in csv file
df_ff.to_csv("dataset/fault_free_training.csv")
df_train.to_csv("dataset/faulty_training.csv")
```

## Models

| Method                                    |Accuracy  |
|-----------------------------------------  |----------|
| XG Boost                                  |  0.924  |
| Random Forest                           |  0.895   |
| Naive Bayes                                   |  0.652   |
| KNN                               |  0.464   |
| Decision Tree                                  |  0.827   |
| Logistic Regression                                  |  0.695   |

## Evaluation

Explain how the project's performance is evaluated. Include metrics or criteria used to measure success.

## Timeline

## License

Specify the project's license to clarify how others can use and contribute to it.

## Acknowledgments

This Dataset was sponsored by the Office of Naval Research, Human & Bioengineered Systems (ONR 341), program officer Dr. Jeffrey G. Morrison under contract N00014-15-C-5003. The views expressed are those of the authors and do not reflect the official policy or position of the Office of Naval Research, Department of Defense, or US Government.

