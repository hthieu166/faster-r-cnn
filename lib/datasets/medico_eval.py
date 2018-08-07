# --------------------------------------------------------
# Fast/er R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Bharath Hariharan
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import xml.etree.ElementTree as ET
import os
import pickle
import numpy as np
from model.config import cfg

def parse_rec(filename):
    """ Parse a Medico txt file """
    
    fi = open(filename,"r")
    lines = fi.readlines()
    num_objs = len(lines)

    objects = []
    # Load object bounding boxes into a data frame.
    for i,line in enumerate(lines):
        obj_struct = {}
        info = line.split(' ') 
        # Make pixel indexes 0-based
        x1 = int(info[0]) - 1
        y1 = int(info[1]) - 1
        x2 = int(info[2]) - 1
        y2 = int(info[3]) - 1
        obj_struct['name'] = info[4].strip()
        obj_struct['bbox'] = [x1,y1,x2,y2]
        objects.append(obj_struct)
        # print('obj_struct',str(obj_struct))

    # print('Objects: '+str(objects))
    # for obj in tree.findall('object'):
    #     obj_struct = {}
    #     obj_struct['name'] = obj.find('name').text
    #     obj_struct['pose'] = obj.find('pose').text
    #     obj_struct['truncated'] = int(obj.find('truncated').text)
    #     obj_struct['difficult'] = int(obj.find('difficult').text)
    #     bbox = obj.find('bndbox')
    #     obj_struct['bbox'] = [int(bbox.find('xmin').text),
    #                         int(bbox.find('ymin').text),
    #                         int(bbox.find('xmax').text),
    #                         int(bbox.find('ymax').text)]
    #     objects.append(obj_struct)

    return objects


def voc_ap(rec, prec, use_07_metric=False):
  """ ap = voc_ap(rec, prec, [use_07_metric])
  Compute VOC AP given precision and recall.
  If use_07_metric is true, uses the
  VOC 07 11 point method (default:False).
  """
  if use_07_metric:
    # 11 point metric
    ap = 0.
    for t in np.arange(0., 1.1, 0.1):
      if np.sum(rec >= t) == 0:
        p = 0
      else:
        p = np.max(prec[rec >= t])
      ap = ap + p / 11.
  else:
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
      mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
  return ap


def medico_eval(detpath,
             annopath,
             imagesetfile,
             classname,
             cachedir,
             ovthresh=0.5,
             use_07_metric=False,
             use_diff=False):
    """rec, prec, ap = voc_eval(detpath,
                                annopath,
                                imagesetfile,
                                classname,
                                [ovthresh],
                                [use_07_metric])

    Top level function that does the PASCAL VOC evaluation.

    detpath: Path to detections
        detpath.format(classname) should produce the detection results file.
    annopath: Path to annotations
        annopath.format(imagename) should be the xml annotations file.
    imagesetfile: Text file containing the list of images, one image per line.
    classname: Category name (duh)
    cachedir: Directory for caching the annotations
    [ovthresh]: Overlap threshold (default = 0.5)
    [use_07_metric]: Whether to use VOC07's 11 point AP computation
        (default False)
    """
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name
    # cachedir caches the annotations in a pickle file

    # first load gt

    if not os.path.isdir(cachedir):
        os.mkdir(cachedir)
    
    cachefile = os.path.join(cachedir, '%s_annots.pkl' % os.path.basename(imagesetfile))
    # print('>> cachedir '+cachedir)
    # print('>> cachefile ' + cachefile)
    # print('>> exist '+ str(os.path.exists(cachefile)))
    # read list of images
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]

    # print('>> is file '+ str(os.path.isfile(cachefile)))    
    if True:
    # if not os.path.isfile(cachefile):
        # load annotations
        # print('no cache file, create!')
        recs = {}
        for i, imagename in enumerate(imagenames):
            recs[imagename] = parse_rec(annopath.format(imagename))
            if i % 100 == 0:
                print('Reading annotation for {:d}/{:d}'.format(
                i + 1, len(imagenames)))
        # save
        print('Saving cached annotations to {:s}'.format(cachefile))
        with open(cachefile, 'w') as f:
            pickle.dump(recs, f)
    else:
        # load
        with open(cachefile, 'rb') as f:
            try:
                recs = pickle.load(f)
            except:
                recs = pickle.load(f, encoding='bytes')

    # extract gt objects for this class
    class_recs = {}
    
    # print('recs = '+str(recs))
    #========== READ OUTPUT FROM GROUND-TRUTH FILES ================
    total_gt = 0
    for imagename in imagenames:
        # print('img_name = '+ imagename)
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
        # print(imagename+ ' R: ',str(R))
        total_gt += len(R)
        # print('classname: '+classname)
        bbox = np.array([x['bbox'] for x in R])

        difficult = np.array([False for x in R]).astype(np.bool)

        det = [False] * len(R)

        class_recs[imagename] = {'bbox': bbox,
                                'det': det}
    # print('Class_recs: ')
    # print(class_recs)
    # read dets
    #========== READ OUTPUT FROM FASTER-RCNN RESULT FILES ================
    detfile = detpath.format(classname)
    # print('detfile ' + detfile)
    with open(detfile, 'r') as f:
        lines = f.readlines()
    splitlines = [x.strip().split(' ') for x in lines]   #split by line <img name>  <confident> <x1> <y1> <x2> <y2>
    tmp = []
    
    bbox_out_folder = cfg.TEST_RESULT_DIR + 'bbox/'
    res_out_folder = cfg.TEST_RESULT_DIR + 'res/'
    
    if not os.path.exists(bbox_out_folder ):
        os.makedirs(bbox_out_folder )
    
    if not os.path.exists(res_out_folder):
        os.makedirs(res_out_folder)
        
    for i in xrange(len(splitlines)):
        if float(splitlines[i][1])>=0.85:
            tmp.append(splitlines[i])
            file_name = bbox_out_folder +splitlines[i][0]+'.txt'
            if not os.path.exists(file_name):
                fo = open(file_name,"w")
            else:
                fo = open(file_name,"a")
            fo.write(classname+' '+splitlines[i][1]+' '+splitlines[i][2]+' '+splitlines[i][3]+' '+splitlines[i][4]+' '+splitlines[i][5]+'\n')
            fo.close()
    
    splitlines = tmp
    
    # print('splitlines: ')
    # for i in splitlines:
    #     print(i)
    # print('----------')
    
    npos = len(tmp)
    image_ids = [x[0] for x in splitlines] #list of img name 
    confidence = np.array([float(x[1]) for x in splitlines]) #list of conf
    BB = np.array([[float(z) for z in x[2:]] for x in splitlines]) #list of bbox

    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    # print('BB:')
    # print(BB)
    fo = open(res_out_folder+classname+'.txt',"w")
    if BB.shape[0] > 0:
        # sort by confidence
        sorted_ind = np.argsort(-confidence) #short 
        sorted_scores = np.sort(-confidence)
        # print('sorted_Scores')
        # print(sorted_scores)
        # print('sorted_ind')
        # print(sorted_ind)

        BB = BB[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]

        # go down dets and mark TPs and FPs
        for d in range(nd):
            R = class_recs[image_ids[d]]
            # print('R: ',str(R))

            bb = BB[d, :].astype(float)
            ovmax = -np.inf
            BBGT = R['bbox'].astype(float)

            if BBGT.size > 0:
                # compute overlaps
                # intersection
                ixmin = np.maximum(BBGT[:, 0], bb[0])
                iymin = np.maximum(BBGT[:, 1], bb[1])
                ixmax = np.minimum(BBGT[:, 2], bb[2])
                iymax = np.minimum(BBGT[:, 3], bb[3])
                iw = np.maximum(ixmax - ixmin + 1., 0.)
                ih = np.maximum(iymax - iymin + 1., 0.)
                inters = iw * ih

                # union
                uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                    (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                    (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

                overlaps = inters / uni
                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)

            if ovmax > ovthresh and not R['det'][jmax]:
                tp[d] = 1.
                R['det'][jmax] = 1
                fo.write('correct '+ str(image_ids[d])+'\n')
                # print('correct '+ str(image_ids[d]))
            else:
                fp[d] = 1.
                fo.write('false-positive '+image_ids[d]+'\n')
    for i in class_recs:
        for x in class_recs[i]['det']:
            if x == False:
                fo.write('false-negative '+ i+'\n')
                # print('false-negative '+ i)
                break
    fo.close()

    # compute precision recall
    c_fp = np.sum(fp)
    c_tp = np.sum(tp)
    c_prec = c_tp / float(np.maximum(c_tp + c_fp, np.finfo(np.float64).eps))
    c_rec = c_tp / float(total_gt)

    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(total_gt)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)
    
    #print result
    file_out_name = res_out_folder + "res_sum.csv"
    if not(os.path.exists(file_out_name)):
        fo = open(file_out_name,"w")
        fo.write('class,total_gt,detected,true_positive,false_positive,false_negative,precision,recall\n')
    else:
        fo = open(file_out_name,"a")

    print('------------------------------')
    print('>>>Ground truth bbox: '+str(total_gt))
    print('>>>Detected bbox: '+str(npos))
    print('>>>True positive: '+str(c_tp))
    print('>>>False positive: '+str(c_fp))
    print('>>>Precision: '+str(c_prec))
    print('>>>Recall: '+str(c_rec))
    fo.write(classname+','+str(total_gt)+','+str(npos)+','+str(c_tp)+','+str(c_fp)+','+str(c_prec)+','+str(c_rec)+'\n')
    fo.close()
    return rec, prec, ap
