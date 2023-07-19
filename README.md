# 23HIDERatArthritis

# Hi-DER
This is the official PyTorch implementation of “[Classification models for osteoarthritis grades of multiple joints based on continual learning]()” (RSNA Radiology 2023)

We developed a hierarchical osteoarthritis (OA) classification system for multiple joints that can be continuously updated by using a continual learning and hierarchical labeling strategy, hierarchical DER (Hi-DER), for classification model training. Please refer to our paper for more details.

## Table of Contents
- [Introduction](#introduction)
- [Proposed Architecture](#proposed-architecture)
- [Pre-requisites](#pre-requisites)
- [Hyperparameters](#hyperparameters)
- [Run Experiment](#run-experiment)
- [Acknowledgement](#acknowledgement)
- [Citation](#citation)

## Introduction
Training medical AI in a fixed range of settings, for instance, applying individualized training process to each of the multiple joints with various morphologies, is time-consuming and resource-intensive. Moreover, while AI model training for OA typically requires vast amounts of data, some joints or OA grades might lack sufficient data. Although some studies have applied continual learning to update the classification range of medical AI models, research for superior continual learning methods that utilize strategies such as dynamically expandable representation ([DER](https://arxiv.org/abs/2103.16788)) is lacking.

DER is a method that improves the efficiency of multi-class prediction by enabling the simultaneous handling of multiple tasks and adaptation of new classifications. However, it assumes a dataset with no connection between the outcomes, while most outcomes in the medical domain in contrast have a hierarchical structure. Therefore, we propose a three-stage architecture that utilizes a hierarchical labeling approach within the incremental concept, to incorporate hierarchical information between the outcomes, and expand the applicability of DER.

## Proposed Architecture
![Figure_1](https://github.com/DigitalHealthcareLab/23HIDERatArthritis/assets/61937818/a47212ce-9fcc-43f7-bf90-b17dbb797a92)
- Image set with corresponding hierarchical labels (tree-based architecture) is used as the input at each incremental step (in our study, each step represents a different anatomical location).
- The model is incrementally trained by expanding through multiple steps (multiple anatomies).
- It accumulates knowledge from the previous steps, while incorporating new information from the current step.
- Hierarchy-specific FC layers and classifiers are introduced at each step, and the model is trained by computing the hierarchical loss at each level of the classification hierarchy.

## Pre-requisites
Run the following commands to clone this repository and install the required packages.

```bash
git clone (...)
pip install -r requirements.txt
```

### Hyperparameters

---

- **memory_size**: The total number of preserved exemplar in the incremental learning process.
- **memory_per_class**: The number of preserved exemplar per class ($\frac{memory-size}{K-classes}$).
- **shuffle**: Whether to shuffle the class order or not.
- **init_cls**: The number of classes in the initial incremental step.
- **increment**: The number of classes in each incremental step $t$ ($t$ > 1).

### Run Experiment

---

- Edit the hider.json file for global settings.
- Edit the structure of hierarchical labels in the base.py, Hi_DER.py, and hierarchical_loss.py file.
- Run the following command to run the experiment.

```bash
python main.py --config=./exps/hider.json
```

### Acknowledgement

---

Our code is based on [PyCIL](https://github.com/G-U-N/PyCIL). We thank the authors for providing the great base code.

### Citation

---

If you find this code useful, please consider citing our paper.

(…)
