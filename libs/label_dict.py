# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
from collections import OrderedDict
from config import Config

net_config = Config()

if net_config.DATASET_NAME == 'tct':
    NAME_LABEL_MAP = OrderedDict({
        "back_ground": 0,
        'ascus': 1,
        'asch': 2,
        'lsil': 3,
        'hsil': 4,
        'scc': 5,
        'agc': 6,
        'trichomonas': 7,
        'candida': 8,
        'flora': 9,
        'herps': 10,
        'actinomyces': 11,

    })
elif net_config.DATASET_NAME == 'pascal':
    NAME_LABEL_MAP = OrderedDict({
        'back_ground': 0,
        'aeroplane': 1,
        'bicycle': 2,
        'bird': 3,
        'boat': 4,
        'bottle': 5,
        'bus': 6,
        'car': 7,
        'cat': 8,
        'chair': 9,
        'cow': 10,
        'diningtable': 11,
        'dog': 12,
        'horse': 13,
        'motorbike': 14,
        'person': 15,
        'pottedplant': 16,
        'sheep': 17,
        'sofa': 18,
        'train': 19,
        'tvmonitor': 20
    })
else:
    assert 'please set label dict!'

def get_label_name_map():
    reverse_dict = {}
    for name, label in NAME_LABEL_MAP.items():
        reverse_dict[label] = name
    return reverse_dict

LABEl_NAME_MAP = get_label_name_map()