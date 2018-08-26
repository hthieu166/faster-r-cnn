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
import datetime
import pickle
import itertools

CLASSES = ('__background__',  # always index 0
           'normal',
           'polyp','dyed-lifted-polyp','dyed-resection-margin')

NETS = {'vgg16': ('vgg16_faster_rcnn_iter_70000.ckpt',),'res101': ('res101_faster_rcnn_iter_5000.ckpt',)}
DATASETS= {'pascal_voc': ('voc_2007_trainval',),'pascal_voc_0712': ('voc_2007_trainval+voc_2012_trainval',),'medico_2018':('medico_2018_trainval',)}


def plot_confusion_matrix(cm, classes,
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Reds):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else '.2f'
    thresh = cm.max() / 2.
    
    n_classes = len(classes)
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\n')

def class_max_conf(dets, thresh=0.5):
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return 0.0
    tmp = np.argmax(dets[:,-1])
    return dets[tmp,-1]

def demo(log_out,sess, net, image_name, gt, cfs_mat, INP_DIR, CONF_THRESH):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the input image
    im = cv2.imread(image_name)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(sess, net, im)
    timer.toc()

    # Visualize detections for each class
    NMS_THRESH = 0.3
 
    res_cls = CLASSES[1]
    res_conf = 0.0
    for cls_ind, cls in enumerate(CLASSES[2:]): 
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        tmp = class_max_conf(dets,CONF_THRESH)
        
        if (tmp>res_conf):
            res_conf = tmp
            res_cls = cls
    
    cfs_mat[gt][res_cls] += 1
    correct = (gt == res_cls)

    img_id = image_name.replace(INP_DIR,'')

    log_out.write(img_id+','+str(correct)+','+gt+','+res_cls+','+'{:3f},{:3f}'.format(res_conf,timer.total_time)+'\n')
    return correct

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN demo')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16 res101]',
                        choices=NETS.keys(), default='res101')
    parser.add_argument('--dataset', dest='dataset', help='Trained dataset [pascal_voc pascal_voc_0712]',
                        choices=DATASETS.keys(), default='medico_2018')
    parser.add_argument('--inpdir', dest='inpdir')
    parser.add_argument('--testlist', dest='testlist')
    parser.add_argument('--conf', dest='conf', default='0.9')
    parser.add_argument('--outdir', dest='outdir', default = 'result')
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    
    #CREATE TIME STAMP ID
    time_stamp = str(datetime.datetime.now())
    #INPUT AND OUTPUT DIRECTORY
    args = parse_args()
    INPUT_DIR = args.inpdir
    OUTPUT_DIR = os.path.join('cls_result',args.outdir+'_'+time_stamp+'/')
    OUTPUT_LOG = OUTPUT_DIR + 'log_'+time_stamp+'.csv'
    TEST_LIST = args.testlist

    #SAVE LOG FILE
    print('Save log to = '+ OUTPUT_LOG)

    if not os.path.exists(os.path.dirname(OUTPUT_LOG)):
        os.makedirs(os.path.dirname(OUTPUT_LOG))
   
    flog = open(OUTPUT_LOG,"w")
    flog.write('id,correct,gt_cls,predict_cls,conf,time\n')

    #CONFIDENT THRESH
    CONF_THRESH = float(args.conf)
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
    
    fi = open(TEST_LIST)
    lines = fi.readlines()

    print('Total input imgs = '+str(len(lines)))
    
    num_of_test = len(lines)

    cfs_mat = {}
    for i_class in CLASSES[1:]:
        cfs_mat[i_class] = {}
        for j_class in CLASSES[1:]:
            cfs_mat[i_class][j_class] = 0

    for i,line in enumerate(lines):
        # print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        # print('Demo for data/demo/{}'.format(im_name))
        im_name, gt = line.strip().split(' ')
        im_name = os.path.join(INPUT_DIR,im_name)

        if (i%10 == 0):
            print(str(i) + '/' + str(num_of_test))
        if (i%100 == 0 and i>0) or (i == len(im_name)-1):
            c = '{:25s}'.format('')
            for i_class in CLASSES[1:]:
                c+= '{:25s}'.format(i_class)
            print(c)
            for i_class in CLASSES[1:]:
                c = '{:25s}'.format(i_class)
                for j_class in CLASSES[1:]:
                    c+= '{:25s}'.format(str(cfs_mat[i_class][j_class]))
                print(c+'\n')
            print('-------------------')
        crr = demo(flog, sess, net, im_name, gt, cfs_mat, INPUT_DIR, CONF_THRESH)
    flog.close()

    #SAVE cvs_mat
    fo = open(OUTPUT_DIR+'confusion_matrix.pickle',"wb")
    pickle.dump((CLASSES,cfs_mat),fo)
    fo.close()

    #PRINT result
    fo = open(OUTPUT_DIR+'confusion_matrix.txt',"w")
    print('--------FINAL RESULT-----------')
    print('Total = ' + str(num_of_test))
    print('Confusion matrix: ')
    c = '{:25s}'.format('')
    for i_class in CLASSES[1:]:
        c+= '{:25s}'.format(i_class)
    print(c)
    fo.write(c + '\n')
    for i_class in CLASSES[1:]:
        c = '{:25s}'.format(i_class)
        for j_class in CLASSES[1:]:
            c+= '{:25s}'.format(str(cfs_mat[i_class][j_class]))
        print(c+'\n')
        fo.write(c + '\n')
    fo.close()

    #SAVE RES IMG
    n_cls = len(CLASSES[1:])
    cm = np.zeros((n_cls,n_cls))
    for i,i_class in enumerate(CLASSES[1:]):
        for j,j_class in enumerate(CLASSES[1:]):
            cm[i][j] = int(cfs_mat[i_class][j_class])

    plt.figure()
    plot_confusion_matrix(cm,CLASSES[1:], title = 'Confusion matrix normalized')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR+'confusion_matrix_normalized.png', dpi = 600)
    print('Confusion matrix normalize saved!')

    plt.figure()
    plot_confusion_matrix(cm,CLASSES[1:],normalize=False)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR+'confusion_matrix.png', dpi = 600)
    print('Confusion matrix saved!')

