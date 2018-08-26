#!/usr/bin/env python

# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen, based on code from Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
from model.config import cfg
from model.test import im_detect
from model.nms_wrapper import nms

from utils.timer import Timer
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import numpy as np
import os, cv2
import argparse

from nets.vgg16 import vgg16
from nets.resnet_v1 import resnetv1


import glob
import os

SAVE_FIG = False
# CLASSES = ('__background__',
#            'aeroplane', 'bicycle', 'bird', 'boat',
#            'bottle', 'bus', 'car', 'cat', 'chair',
#            'cow', 'diningtable', 'dog', 'horse',
#            'motorbike', 'person', 'pottedplant',
#            'sheep', 'sofa', 'train', 'tvmonitor')

CLASSES = ('__background__',  # always index 0
        'polyp','dyed-lifted-polyp','dyed-resection-margin', 'normal-z-line')

NETS = {'vgg16': ('vgg16_faster_rcnn_iter_70000.ckpt',),'res101': ('res101_faster_rcnn_iter_5000.ckpt',)}
# NETS = {'vgg16': ('vgg16_faster_rcnn_iter_70000.ckpt',),'res101': ('res101_faster_rcnn_iter_110000.ckpt','res101_faster_rcnn_iter_5000,ckpt')}
DATASETS= {'pascal_voc': ('voc_2007_trainval',),'pascal_voc_0712': ('voc_2007_trainval+voc_2012_trainval',),'medico_2018':('medico_2018_trainval',)}

def vis_detections(ax, class_name, dets, annot_file, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]
        # print(class_name+' '+str(score)+' '+str(bbox[0])+' '+str(bbox[2])+' '+str(bbox[2])+' '+str(bbox[3])+'\n')
        annot_file.write(class_name+' '+str(score)+' '+str(bbox[0])+' '+str(bbox[1])+' '+str(bbox[2])+' '+str(bbox[3])+'\n')
        if not SAVE_FIG:
            continue
        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    # ax.set_title(('{} detections with '
    #               'p({} | box) >= {:.1f}').format(class_name, class_name,
    #                                               thresh),
    #               fontsize=14)
   

def demo(sess, net, image_name, CONF_THRESH):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = image_name
    # im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(sess, net, im)
    timer.toc()
    #print('Detection took {:.3f}s for {:d} object proposals'.format(timer.total_time, boxes.shape[0]))

    # Visualize detections for each class
    NMS_THRESH = 0.3
 
    out_file = OUTPUT_DIR+'result_'+os.path.basename(im_name)
    fo = open(out_file.replace('.jpg','.txt'),"w")
    
    im = im[:, :, (2, 1, 0)]
    ax = None
    if SAVE_FIG:
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.imshow(im, aspect='equal')

    for cls_ind, cls in enumerate(CLASSES[1:]):
       
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        vis_detections(ax, cls, dets, fo, thresh=CONF_THRESH)

    if SAVE_FIG:
        plt.axis('off')
        plt.tight_layout()
        plt.draw()
        print('Save img result to: '+out_file)
        plt.savefig(out_file)
        fo.close()
        plt.close('all')
        

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN demo')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16 res101]',
                        choices=NETS.keys(), default='res101')
    parser.add_argument('--dataset', dest='dataset', help='Trained dataset [pascal_voc pascal_voc_0712]',
                        choices=DATASETS.keys(), default='medico_2018')
    parser.add_argument('--inpdir', dest='inpdir')
    parser.add_argument('--conf', dest='conf', default='0.9')
    parser.add_argument('--savefig', dest='savefig', default = True)
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    args = parse_args()
    #INPUT AND OUTPUT DIRECTORY
    INPUT_DIR = args.inpdir
    OUTPUT_DIR = 'demo_result/' + os.path.basename(os.path.dirname(args.inpdir))+'_'+str(args.conf)+'/'
    print('OUTPUT DIR = '+ OUTPUT_DIR)

    #CONFIDENT THRESH
    CONF_THRESH = float(args.conf)
    SAVE_FIG = bool(args.savefig)
    # model path
    demonet = args.demo_net
    dataset = args.dataset
    tfmodel = os.path.join('output', demonet, DATASETS[dataset][0], 'default',
                              NETS[demonet][0])


    if not os.path.isfile(tfmodel + '.meta'):
        raise IOError(('{:s} not found.\nDid you download the proper networks from '
                       'our server and place them properly?').format(tfmodel + '.meta'))

    # set config
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth=True

    # init session
    sess = tf.Session(config=tfconfig)
    # load network
    if demonet == 'vgg16':
        net = vgg16()
    elif demonet == 'res101':
        net = resnetv1(num_layers=101)
    else:
        raise NotImplementedError
    net.create_architecture("TEST", 5,
                          tag='default', anchor_scales=[4, 8, 16, 32])
    saver = tf.train.Saver()
    saver.restore(sess, tfmodel)

    print('Loaded network {:s}'.format(tfmodel))
    
    if (not os.path.exists(OUTPUT_DIR)):
        os.makedirs(OUTPUT_DIR)
    print(INPUT_DIR+'*.jpg')
    im_names = glob.glob(INPUT_DIR+'*.jpg')
    print('Total demo imgs = '+str(len(im_names)))

    total = len(im_names)
    for i,im_name in enumerate(im_names):
        # print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        # print('Demo for data/demo/{}'.format(im_name))
        if (i%10 == 0):
            print(str(i)+'/'+str(total))
        demo(sess, net, im_name,CONF_THRESH)
    # plt.show()

