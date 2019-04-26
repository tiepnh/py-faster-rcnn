#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse

CLASSES = ('__background__', # always index 0
           'motorbike', 'car', 'bus', 'truck', 'van')
COLOR = {'__background__': (0, 0, 0),
         'motorbike': (0, 255, 0),
         'car': (0, 0, 255),
         'bus': (255, 0, 0),
         'truck': (255, 255, 0),
         'van': (0, 255, 255)}

this_dir = os.path.dirname(__file__)


def draw_result(im, det_res, waitKey=0):
    for res in det_res:
        class_name = res['class_name']
        score = res['score']
        bbox = res['bbox']
        cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), COLOR[class_name], 1)
        cv2.putText(im, "{}: {:.2f}".format(class_name, score), (bbox[0], bbox[1] + 20), cv2.FONT_HERSHEY_COMPLEX, 0.7,
                    COLOR[class_name])
    cv2.imshow("results", im)
    return im, cv2.waitKey(waitKey) & 0XFF


def detection(net, img, CONF_THRESH=0.8, NMS_THRESH=0.3):
    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, img)
    timer.toc()
    print((('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])))

    # Visualize detections for each class
    det_res = []
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
        if len(inds) == 0:
            continue
        for i in inds:
            bbox = dets[i, :4]
            bbox = [int(c) for c in bbox]
            score = dets[i, -1]
            det_res.append({"class_name": cls, "score": score, "bbox": bbox})

    return det_res


def demo(net, data_file_path, save_data_path):
    list_img_file_path = []
    if os.path.isdir(data_file_path):
        import glob
        list_img_file_path = []
        for ext in ['jpg', 'png', 'mp4', 'MOV']:
            list_img_file_path += glob.glob(data_file_path + '/*.{}'.format(ext))
    else:
        list_img_file_path.append(data_file_path)

    for data_file_path in list_img_file_path:
        img = cv2.imread(data_file_path)
        if img is None:
            import exiftool
            video_rotation = 0
            with exiftool.ExifTool() as et:
                metadata = et.get_metadata(data_file_path)
                print (metadata)
                video_rotation = metadata['Composite:Rotation']

            cap = cv2.VideoCapture(data_file_path)
            save_video_path = data_file_path + '_res.mp4'
            print('----------------------------')
            print(save_video_path)
            print('----------------------------')
            fourcc = cv2.VideoWriter_fourcc(*'MP4V')
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            if video_rotation == 90:
                size = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
            else:
                size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            writer = cv2.VideoWriter(save_video_path, fourcc, fps, size)
            res_img = np.zeros((100, 100), np.int8)

            # is video
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if video_rotation == 90:
                    # need to rotate
                    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                det_res = detection(net, frame)
                res_img, key = draw_result(frame, det_res, waitKey=1)
                writer.write(res_img)
                if key == ord('q'):
                    break
            cap.release()
            writer.release()
        else:
            det_res = detection(net, img)
            res_img, _ = draw_result(img, det_res, waitKey=1)
            file_name = os.path.basename(data_file_path)
            save_path = os.path.join(save_data_path, file_name)
            cv2.imwrite(save_path, res_img)
    return 1


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--prototxt', dest='prototxt', help='prototxt path', default='')
    parser.add_argument('--caffemodel', dest='caffemodel', help='caffemodel path', default='')
    parser.add_argument('--input_data_path', dest='input_data_path', help='folder/image/video path', required=False)
    parser.add_argument('--save_data_path', dest='save_data_path', help='save folder', required=False)

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()

    if args.prototxt == '':
        args.prototxt = os.path.join(this_dir, '..', "models/pascal_voc/Vehicle/faster_rcnn_end2end/test.prototxt")

    if args.caffemodel == '':
        args.caffemodel = os.path.join(this_dir, '..', "output/faster_rcnn_end2end/voc_2007_trainval/vehicle_faster_rcnn_iter_70000.caffemodel")

    caffemodel = args.caffemodel
    prototxt = args.prototxt
    args.input_data_path = '/media/mvn/Data/Dataset/Image/ITS/VehicleDataset/darknet/Test'
    args.input_data_path = '/media/mvn/Data/Dataset/Image/ITS/Video'

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print(('\n\nLoaded network {:s}'.format(caffemodel)))

    # Warmup on a dummy image
    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    for i in range(2):
        _, _= im_detect(net, im)

    demo(net, args.input_data_path, args.save_data_path)
