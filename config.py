import numpy as np
import math
from collections import OrderedDict

class Config(object):

    ##############################
    # Data And Dataset
    ##############################
    CHECKPOINT_DIR = "/root/userfolder/kuku/resnet_imagenet_v2_fp32_20181001"
    NUM_CLASS = 0
    NUM_ITEM_DATASET = 0
    DATASET_NAME = None
    DATA_DIR = None
    MODLE_DIR = None
    TARGET_SIDE = None
    NAME_TO_LABEL = None
    IMAGE_MAX_INSTANCES = 100
    PIXEL_MEANS = np.array([115.2, 118.8, 123.0])

    ###################################
    # Network config
    ###################################
    # Anchor stride
    # If 1 then anchors are created for each cell in the backbone feature map.
    # If 2, then anchors are created for every other cell, and so on.
    RPN_ANCHOR_STRIDE = 1
    BASE_ANCHOR_SIZE_LIST = [32, 64, 128, 256, 512]
    LEVEL = ['P2', 'P3', 'P4', 'P5', "P6"]
    BACKBONE_STRIDES = [4, 8, 16, 32, 64]


    ###################################
    # Training Config
    ###################################
    GPU_GROUPS = ["/gpu:0", "/gpu:1"]
    NUM_GPUS = len(GPU_GROUPS)
    COMPUTE_TIME = False
    CLIP_GRADIENT_NORM = 5.0
    EPOCH_BOUNDARY = [25]
    EPOCH = 30
    WEIGHT_DECAY = 0.0001
    EPSILON = 1e-5
    MOMENTUM = 0.9
    LEARNING_RATE = 0.001
    PER_GPU_IMAGE = 2
    

    ###################################
    # RPN
    ###################################
    ANCHOR_RATIOS = [0.5, 1, 2]
    # The iou between two rpn proposals is bigger RPN_NMS_IOU_THRESHOLD
    # the lower scores of rpn proposal will be move
    RPN_NMS_IOU_THRESHOLD = 0.7
    RPN_IOU_POSITIVE_THRESHOLD = 0.7
    RPN_IOU_NEGATIVE_THRESHOLD = 0.3
    RPN_MINIBATCH_SIZE = 256
    RPN_POSITIVE_RATE = 0.5
    RPN_TOP_K_NMS = 12000
    MAX_PROPOSAL_NUM_TRAINING = 2000
    MAX_PROPOSAL_NUM_INFERENCE = 1000
    RPN_BBOX_STD_DEV = [0.1, 0.1, 0.25, 0.27]
    BBOX_STD_DEV = [0.13, 0.13, 0.27, 0.26]

    ###################################
    # Fast_RCNN
    ###################################
    ROI_SIZE = 7
    # the iou between different detections is
    # less than HEAD_NMS_IOU_THRESHOLD
    HEAD_NMS_IOU_THRESHOLD = 0.3
    FINAL_SCORE_THRESHOLD = 0.7
    HEAD_IOU_POSITIVE_THRESHOLD = 0.5
    HEAD_IOU_LOW_NEG_THRESHOLD = 0.1
    HEAD_MINIBATCH_SIZE = 200
    HEAD_POSITIVE_RATE = 0.33
    DETECTION_MAX_INSTANCES = 200

    def __init__(self):

        self.BATCH_SIZE = self.NUM_GPUS * self.PER_GPU_IMAGE
        self.BOUNDARY = [self.NUM_ITEM_DATASET * i // self.BATCH_SIZE for i in self.EPOCH_BOUNDARY]
        self.SAVE_EVERY_N_STEP = int(self.NUM_ITEM_DATASET/self.BATCH_SIZE)
        # (h ,w)
        self.BACKBONE_SHAPES = np.array(
            [[int(math.ceil(self.TARGET_SIDE / stride)),
              int(math.ceil(self.TARGET_SIDE / stride))]
             for stride in self.BACKBONE_STRIDES])
        self.LABEL_TO_NAME = self.get_label_name_map()

    def get_label_name_map(self):
        reverse_dict = {}
        for name, label in self.NAME_TO_LABEL.items():
            reverse_dict[label] = name
        return reverse_dict


class TCTConfig(Config):

    NUM_CLASS = 11 + 1
    NUM_ITEM_DATASET = 5714
    # The exact location of data be located by fllowing four variables
    DATA_DIR = "/root/userfolder/kuku/tfdata"
    DATASET_NAME = 'tct'
    TRAIN_DATASET_NAME = "train.tfrecord"
    EVAL_DATASET_NAME = "test.tfrecord"

    # the summary and model will be saved in this location
    DEBUG =False
    MODLE_DIR = "./logs"
    BACKBONE_NET = "resnet_model"
    NET_NAME = "ResNet_FPN"
    # resize and padding the image shape to (1024, 1024)
    TARGET_SIDE = 1024
    PIXEL_MEANS = np.array([115.2, 118.8, 123.0])
    NAME_TO_LABEL = OrderedDict({
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
                                'actinomyces': 11})
    LABEL_TO_NAME = OrderedDict({v:k for k, v in NAME_TO_LABEL.items()})
    

    def __init__(self):
        Config.__init__(self)
