# ------------------------------------------------------------------------------
# pose.pytorch
# Copyright (c) 2018-present Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint

import cv2
import torch
import numpy as np
import torchvision
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

import _init_paths
from config import cfg
from config import update_config
from core.loss import JointsMSELoss1
from core.function import validate
from utils.utils import create_logger

import models
import dataset


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--prevModelDir',
                        help='prev Model directory',
                        type=str,
                        default='')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    update_config(cfg, args)

    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, args.cfg, 'valid')

    logger.info(pprint.pformat(args))
    logger.info(cfg)

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
        cfg, is_train=False
    )
    model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()
    
    if cfg.TEST.MODEL_FILE:
        logger.info('=> loading model from {}'.format(cfg.TEST.MODEL_FILE))
        model.module.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=True)
    else:
        model_state_file = os.path.join(
            final_output_dir, 'final_state.pth'
        )
        logger.info('=> loading model from {}'.format(model_state_file))
        model.module.load_state_dict(torch.load(model_state_file))
    

    # define loss function (criterion) and optimizer
    criterion = JointsMSELoss1(
        use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT
    ).cuda()

    # Data loading code
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    valid_dataset = eval('dataset.'+cfg.DATASET.DATASET)(
        cfg, cfg.DATASET.ROOT, cfg.DATASET.TEST_SET, False,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=True
    )
    
    # from utils.transforms import get_affine_transform
    # from core.inference import get_final_preds, get_max_preds
    # for i, (input, target, target_weight, meta) in enumerate(valid_loader):
    #     features_blobs_2 = []
    #     features_blobs = []
    #     def hook_feature(module, output):
    #         features_blobs.append(output[0].data.cpu().numpy())
    #     def hook_feature2(module, output):
    #         features_blobs_2.append(output[0].data.cpu().numpy())
        
    #     for name, m in model.named_modules():
    #         if name == 'final':
    #             m.register_forward_pre_hook(hook_feature)
    #         if name == 'final.2':
    #             m.register_forward_pre_hook(hook_feature2)
    #     b, c, h, w = input.shape
    #     y = model(input)
    #     img_name = meta['image']
    #     center = meta['center']
    #     scale = meta['scale']
    #     joints_vis = meta['joints_vis']
    #     joints = meta['joints']
    #     preds, maxvals = get_max_preds(y.detach().cpu().numpy())
    #     for img_id in range(b):
    #         data_numpy = cv2.imread(
    #             img_name[img_id], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
    #         )
    #         trans = get_affine_transform(center[img_id].numpy(), scale[img_id].numpy(), 0, np.array([h, w]))
    #         row_img = cv2.warpAffine(
    #             data_numpy,
    #             trans,
    #             (int(h), int(w)),
    #             flags=cv2.INTER_LINEAR)
    #         cv2.imwrite('vis/raw{}_{}.jpg'.format(i, img_id), row_img)
    #         img = row_img.copy()
    #         for joint, joint_vis in zip(joints[img_id], joints_vis[img_id]):
    #             if joint_vis[0]:
    #                 cv2.circle(img, (int(joint[0]), int(joint[1])), 2, [0, 0, 255], 2)
    #         cv2.imwrite("vis/GT{}_{}.jpg".format(i, img_id), img)
    #         ################################### vis pred ###################################
    #         img1 = row_img.copy()
    #         for j in range(preds.shape[1]):
    #             cv2.circle(
    #                 img1, 
    #                 (int(preds[img_id][j][0]*4), int(preds[img_id][j][1]*4)),
    #                 2, [0, 0, 255], 2)
    #         cv2.imwrite('vis/pred{}_{}.jpg'.format(i, img_id), img1)
    #         ################################################################################
            
            
    #         cam = features_blobs[0][img_id][18]
    #         cam = cam - np.min(cam)
    #         cam_img = cam / np.max(cam)
    #         cam_img = np.uint8(256 * cam_img)
    #         heatmap1 = cv2.applyColorMap(cam_img, cv2.COLORMAP_JET)
    #         heatmap1 = cv2.resize(heatmap1, (256, 256))
    #         heatmap = np.uint8(0.7*heatmap1 + 0.3*row_img)
    #         cv2.imwrite('vis/pred_before{}_{}.jpg'.format(i, img_id), heatmap)
            
    #         cam = features_blobs_2[0][img_id][12]
    #         cam = cam - np.min(cam)
    #         cam_img = cam / np.max(cam)
    #         cam_img = np.uint8(256 * cam_img)
    #         heatmap2 = cv2.applyColorMap(cam_img, cv2.COLORMAP_JET)
    #         heatmap2 = cv2.resize(heatmap2, (256, 256))
    #         ht = np.uint8(0.7*heatmap2 + 0.3*row_img)
    #         cv2.imwrite('vis/pred_after{}_{}.jpg'.format(i, img_id), ht)
    #     # exit(0)
    # '''
    validate(cfg, valid_loader, valid_dataset, model, criterion,
             final_output_dir, tb_log_dir)

    # # input = torch.Tensor(1, 3, 256, 256).cuda()
    
    # # for name, m in model.named_modules():
    # #     print(name)
    # #     if name == 'module.final':
    # #         m.register_forward_pre_hook(hook_feature)
    # # for i, (input, target, target_weight, meta) in enumerate(valid_loader):
    # #     y = model(input)
    # #     bz, nc, h, w = features_blobs[0].shape
    # #     cam = features_blobs[0][0][0]
    # #     cam = cam - np.min(cam)
    # #     cam_img = cam / np.max(cam)
    # #     cam_img = np.uint8(255 * cam_img)
    # #     heatmap = cv2.applyColorMap(cam_img, cv2.COLORMAP_JET)
    # #     cv2.imwrite('cam.jpg', heatmap)
    # #     save_vis_map(input, meta['joints'], meta['joints_vis'])
    
    # # exit(0)


if __name__ == '__main__':
    main()