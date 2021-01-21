
# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from yacs.config import CfgNode as CN


_C = CN()

_C.OUTPUT_DIR = ''
_C.LOG_DIR = ''
_C.DATA_DIR = ''
_C.GPUS = (0,)
_C.WORKERS = 4
_C.PRINT_FREQ = 20
_C.AUTO_RESUME = False
_C.PIN_MEMORY = True
_C.RANK = 0

# Cudnn related params
_C.CUDNN = CN()
_C.CUDNN.BENCHMARK = True
_C.CUDNN.DETERMINISTIC = False
_C.CUDNN.ENABLED = True

# common params for NETWORK
_C.MODEL = CN()
_C.MODEL.NAME = 'pose_hrnet'
_C.MODEL.ARCH = 'PoseNAS-A'
_C.MODEL.IN_CHANNELS = 16
_C.MODEL.INIT_WEIGHTS = True
_C.MODEL.PRETRAINED = ''
_C.MODEL.NUM_JOINTS = 17
_C.MODEL.NUM_CLASSES = 1000
_C.MODEL.TAG_PER_JOINT = True
_C.MODEL.TARGET_TYPE = 'gaussian'
_C.MODEL.IMAGE_SIZE = [256, 256]  # width * height, ex: 192 * 256
_C.MODEL.HEATMAP_SIZE = [64, 64]  # width * height, ex: 24 * 32
_C.MODEL.SIGMA = 2
_C.MODEL.EXTRA = CN(new_allowed=True)

_C.LOSS = CN()
_C.LOSS.USE_OHKM = False
_C.LOSS.TOPK = 8
_C.LOSS.USE_TARGET_WEIGHT = True
_C.LOSS.USE_DIFFERENT_JOINTS_WEIGHT = False

# DATASET related params
_C.DATASET = CN()
_C.DATASET.ROOT = ''
_C.DATASET.DATASET = 'mpii'
_C.DATASET.TRAIN_SET = 'train'
_C.DATASET.TEST_SET = 'valid'
_C.DATASET.DATA_FORMAT = 'jpg'
_C.DATASET.HYBRID_JOINTS_TYPE = ''
_C.DATASET.SELECT_DATA = False

# training data augmentation
_C.DATASET.FLIP = True
_C.DATASET.SCALE_FACTOR = 0.25
_C.DATASET.ROT_FACTOR = 30
_C.DATASET.PROB_HALF_BODY = 0.0
_C.DATASET.NUM_JOINTS_HALF_BODY = 8
_C.DATASET.COLOR_RGB = False

# train
_C.TRAIN = CN()

_C.TRAIN.LR_FACTOR = 0.1
_C.TRAIN.LR_STEP = [90, 110]
_C.TRAIN.LR = 0.001

_C.TRAIN.OPTIMIZER = 'adam'
_C.TRAIN.MOMENTUM = 0.9
_C.TRAIN.FNA = False
_C.TRAIN.WD = 0.0001
_C.TRAIN.NESTEROV = False
_C.TRAIN.GAMMA1 = 0.99
_C.TRAIN.GAMMA2 = 0.0

_C.TRAIN.BEGIN_EPOCH = 0
_C.TRAIN.END_EPOCH = 140

_C.TRAIN.RESUME = False
_C.TRAIN.CHECKPOINT = ''

_C.TRAIN.BATCH_SIZE_PER_GPU = 32
_C.TRAIN.SHUFFLE = True

# testing
_C.TEST = CN()

# size of images for each device
_C.TEST.BATCH_SIZE_PER_GPU = 32
# Test Model Epoch
_C.TEST.FLIP_TEST = False
_C.TEST.POST_PROCESS = False
_C.TEST.SHIFT_HEATMAP = False

_C.TEST.USE_GT_BBOX = False

# nms
_C.TEST.IMAGE_THRE = 0.1
_C.TEST.NMS_THRE = 0.6
_C.TEST.SOFT_NMS = False
_C.TEST.OKS_THRE = 0.5
_C.TEST.IN_VIS_THRE = 0.0
_C.TEST.COCO_BBOX_FILE = ''
_C.TEST.BBOX_THRE = 1.0
_C.TEST.MODEL_FILE = ''

# debug
_C.DEBUG = CN()
_C.DEBUG.DEBUG = False
_C.DEBUG.SAVE_BATCH_IMAGES_GT = False
_C.DEBUG.SAVE_BATCH_IMAGES_PRED = False
_C.DEBUG.SAVE_HEATMAPS_GT = False
_C.DEBUG.SAVE_HEATMAPS_PRED = False


_F = CN()

_F.OUTPUT_DIR = ''
_F.VERSION = 'large'
_F.LOG_DIR = ''
_F.DATA_DIR = ''
_F.GPUS = (0,)
_F.WORKERS = 4
_F.PRINT_FREQ = 20
_F.AUTO_RESUME = False
_F.PIN_MEMORY = True
_F.RANK = 0

# Cudnn related params
_F.CUDNN = CN()
_F.CUDNN.BENCHMARK = True
_F.CUDNN.DETERMINISTIC = False
_F.CUDNN.ENABLED = True

# common params for NETWORK
_F.MODEL = CN()
_F.MODEL.NAME = 'posenas'
_F.MODEL.PRETRAINED = ''
_F.MODEL.NUM_JOINTS = 17
_F.MODEL.TAG_PER_JOINT = True
_F.MODEL.TARGET_TYPE = 'gaussian'
_F.MODEL.IMAGE_SIZE = [256, 256]  # width * height, ex: 192 * 256
_F.MODEL.HEATMAP_SIZE = [64, 64]  # width * height, ex: 24 * 32
_F.MODEL.SIGMA = 2
_F.MODEL.EXTRA = CN(new_allowed=True)

_F.LOSS = CN()
_F.LOSS.USE_OHKM = False
_F.LOSS.TOPK = 8
_F.LOSS.USE_TARGET_WEIGHT = True
_F.LOSS.USE_DIFFERENT_JOINTS_WEIGHT = False

# DATASET related params
_F.DATASET = CN()
_F.DATASET.ROOT = ''
_F.DATASET.DATASET = 'mpii'
_F.DATASET.TRAIN_SET = 'train'
_F.DATASET.VALID_SET = 'train'
_F.DATASET.TEST_SET = 'valid'
_F.DATASET.DATA_FORMAT = 'jpg'
_F.DATASET.HYBRID_JOINTS_TYPE = ''
_F.DATASET.SELECT_DATA = False

# training data augmentation
_F.DATASET.FLIP = True
_F.DATASET.SCALE_FACTOR = 0.25
_F.DATASET.ROT_FACTOR = 30
_F.DATASET.PROB_HALF_BODY = 0.0
_F.DATASET.NUM_JOINTS_HALF_BODY = 8
_F.DATASET.COLOR_RGB = False

# train
_F.TRAIN = CN()

_F.TRAIN.CAL_TYPE = "gpu"
_F.TRAIN.RECAL = False
_F.TRAIN.FNA = True
_F.TRAIN.RUNS = 10000
_F.TRAIN.WARM_UP_EPOCH = 100
_F.TRAIN.LR = 0.001
_F.TRAIN.ARCH_LR = 0.0005
_F.TRAIN.TEMP = 5.0
_F.TRAIN.WARM_UP = False
_F.TRAIN.LUT_FILE = ''
_F.TRAIN.OPTIMIZER = 'adam'
_F.TRAIN.MOMENTUM = 0.9
_F.TRAIN.WD = 0.0001
_F.TRAIN.ARCH_WD = 0.0005
_F.TRAIN.NESTEROV = False
_F.TRAIN.GAMMA1 = 0.99
_F.TRAIN.GAMMA2 = 0.0

_F.TRAIN.BEGIN_EPOCH = 0
_F.TRAIN.END_EPOCH = 140

_F.TRAIN.RESUME = False
_F.TRAIN.CHECKPOINT = ''

_F.TRAIN.BATCH_SIZE_PER_GPU = 32
_F.TRAIN.SHUFFLE = True

# testing
_F.TEST = CN()

# size of images for each device
_F.TEST.BATCH_SIZE_PER_GPU = 32
# Test Model Epoch
_F.TEST.FLIP_TEST = False
_F.TEST.POST_PROCESS = False
_F.TEST.SHIFT_HEATMAP = False

_F.TEST.USE_GT_BBOX = False

# nms
_F.TEST.IMAGE_THRE = 0.1
_F.TEST.NMS_THRE = 0.6
_F.TEST.SOFT_NMS = False
_F.TEST.OKS_THRE = 0.5
_F.TEST.IN_VIS_THRE = 0.0
_F.TEST.COCO_BBOX_FILE = ''
_F.TEST.BBOX_THRE = 1.0
_F.TEST.MODEL_FILE = ''

# debug
_F.DEBUG = CN()
_F.DEBUG.DEBUG = False
_F.DEBUG.SAVE_BATCH_IMAGES_GT = False
_F.DEBUG.SAVE_BATCH_IMAGES_PRED = False
_F.DEBUG.SAVE_HEATMAPS_GT = False
_F.DEBUG.SAVE_HEATMAPS_PRED = False

def update_config(cfg, args):
    cfg.defrost()
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)

    if args.modelDir:
        cfg.OUTPUT_DIR = args.modelDir

    if args.logDir:
        cfg.LOG_DIR = args.logDir

    if args.dataDir:
        cfg.DATA_DIR = args.dataDir

    cfg.DATASET.ROOT = os.path.join(
        cfg.DATA_DIR, cfg.DATASET.ROOT
    )

    cfg.MODEL.PRETRAINED = os.path.join(
        cfg.DATA_DIR, cfg.MODEL.PRETRAINED
    )

    if cfg.TEST.MODEL_FILE:
        cfg.TEST.MODEL_FILE = os.path.join(
            cfg.DATA_DIR, cfg.TEST.MODEL_FILE
        )

    cfg.freeze()


if __name__ == '__main__':
    import sys
    print("test.....")
    with open(sys.argv[1], 'w') as f:
        print(_C, file=f)

