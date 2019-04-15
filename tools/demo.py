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
        'Guitar', 'Ice_cream', 'French_fries', 'Bread')

NETS = {'vgg16': ('vgg16_faster_rcnn_iter_70000.ckpt',),'res101': ('res101_faster_rcnn_iter_35000.ckpt',)}
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

def demo(sess, net, image_name, CONF_THRESH,INPUT_DIR, OUTPUT_DIR):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = image_name    
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(sess, net, im)
    timer.toc()
    #print('Detection took {:.3f}s for {:d} object proposals'.format(timer.total_time, boxes.shape[0]))

    # Visualize detections for each class
    NMS_THRESH = 0.3
 
    out_file = os.path.join(OUTPUT_DIR,im_name.replace(INPUT_DIR,''))
    out_dir  = os.path.dirname(out_file)
    if (not os.path.exists(out_dir)):
        os.makedirs(out_dir)
    
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
    parser.add_argument('--outdir', default = 'demo_result/')
    parser.add_argument('--conf', dest='conf', default='0.5')
    parser.add_argument('--savefig', dest='savefig', default = True)
    parser.add_argument('--gpu_id', default = "0")
    parser.add_argument('--classes', default = None)
    parser.add_argument('--checkpoint', default = None)
    parser.add_argument('--pattern', default= "*.jpg")
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    args = parse_args()
    #GPU CONFIG
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]= str(args.gpu_id)
    
    #INPUT AND OUTPUT DIRECTORY
    INPUT_DIR = args.inpdir
    OUTPUT_DIR = os.path.join(args.outdir, os.path.basename(os.path.dirname(args.inpdir))+'_'+str(args.conf)+'/')
    
    

    #CONFIDENT THRESH
    CONF_THRESH = float(args.conf)
    SAVE_FIG = bool(args.savefig)
    
    #CLASSES
    if (args.classes != None):  
        with open(args.classes) as fi:
            all_classes = [x.strip() for x in fi.readlines()]
        CLASSES = ('__background__',) + tuple(all_classes)
    
    print(">>> NUM_CLASSES: \t",len(CLASSES))
    
    # model path
    demonet = args.demo_net
    if (args.checkpoint !=None):
        tfmodel = args.checkpoint
    else:
        dataset = args.dataset
        tfmodel = os.path.join('output', demonet, DATASETS[dataset][0], 'default', NETS[demonet][0])
        
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
    
    net.create_architecture("TEST", len(CLASSES),
                          tag='default', anchor_scales=[4, 8, 16, 32])
    saver = tf.train.Saver()
    saver.restore(sess, tfmodel)

    print('>>> LOADES NETWORK: \t{:s}'.format(tfmodel))
    print('>>> OUTPUT DIR: \t', OUTPUT_DIR)
        
    find_imgs = os.path.join(INPUT_DIR, args.pattern)
    print('>>> INPUT DIR: \t', find_imgs)
    im_names = glob.glob(find_imgs)
#     im_names += glob.glob(INPUT_DIR+'*.JPG')
    
    print('>>> COUNT_IMGS: \t= '+str(len(im_names)))

    total = len(im_names)
    for i,im_name in enumerate(im_names):
        if (i%10 == 0):
            print(str(i)+'/'+str(total))
        demo(sess, net, im_name, CONF_THRESH, INPUT_DIR, OUTPUT_DIR)

