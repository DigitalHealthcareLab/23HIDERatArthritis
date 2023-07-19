'''Helper functions.
'''


# This is a modified version of original  https://github.com/pytorch/vision/blob/master/torchvision/datasets/cifar.py
# Coarse labels is added for cifar100 as an option 

from __future__ import print_function
from PIL import Image
import os
import os.path
import errno
import numpy as np
import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

import torch.utils.data as data
from torchvision.datasets.utils import download_url, check_integrity
import torch


def unpickle(file):
    '''Unpickle the given file
    '''

    with open(file, 'rb') as f:
        res = pickle.load(f, encoding='bytes')
    return res

def read_meta(metafile):
    '''Read the meta file and return the coarse and fine labels.
    '''
    meta_data = unpickle('/home/bongkyung/Deep_Hierarchical_Classification/dataset/pickle_files/meta')
    fine_label_names = [t.decode('utf8') for t in meta_data[b'fine_label_names']]
    coarse_label_names = [t.decode('utf8') for t in meta_data[b'coarse_label_names']]
    return coarse_label_names, fine_label_names


def calculate_accuracy(predictions, labels):
    '''Calculates the accuracy of the prediction.
    '''

    num_data = labels.size()[0]
    predicted = torch.argmax(predictions, dim=1)

    correct_pred = torch.sum(predicted == labels)

    accuracy = correct_pred*(100/num_data)

    return accuracy.item()
