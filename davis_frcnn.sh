CLASSES="/home/hthieu/FRCNN/tf-faster-rcnn-medico/coco_classes.txt"
CKPT="/home/hthieu/FRCNN/tf-faster-rcnn-medico/output/res101/coco_900-1190/coco_2014_train+coco_2014_valminusminival/res101_faster_rcnn_iter_1190000.ckpt"
INP_DIR="/home/hthieu/DAVIS2019/dataset/test-challenge/DAVIS/JPEGImages/480p/"
OUT_DIR="/home/hthieu/DAVIS2019/frcnn_test-challenge/"
PATTERN="*/*.jpg"

./tools/demo.py \
    --gpu_id 1 \
    --inpdir $INP_DIR \
    --outdir $OUT_DIR \
    --net 'res101' \
    --checkpoint $CKPT \
    --classes $CLASSES \
    --pattern $PATTERN \
    --conf 0.6


    