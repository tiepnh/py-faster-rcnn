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
from PIL import ImageFont, ImageDraw, Image

CLASSES = ('__background__',  # always index 0
           'motorbike', 'car', 'bus', 'truck', 'van')
COLOR = {'__background__': (0, 0, 0),
         'motorbike': (0, 255, 0),
         'car': (0, 0, 255),
         'bus': (255, 0, 0),
         'truck': (255, 255, 0),
         'van': (0, 255, 255)}

this_dir = os.path.dirname(__file__)


class VehicleDetection():
    def __init__(self, prototxt, caffemodel, gpu_id=0, conf_thresh=0.8, nms_thresh=0.3, fontpath="arial.ttf"):
        cfg.TEST.HAS_RPN = True  # Use RPN for proposals

        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh

        if not os.path.isfile(caffemodel):
            raise IOError(('{:s} not found.\nDid you run ./data/script/'
                           'fetch_faster_rcnn_models.sh?').format(caffemodel))

        if gpu_id == -1:
            caffe.set_mode_cpu()
        else:
            caffe.set_mode_gpu()
            caffe.set_device(gpu_id)
            cfg.GPU_ID = gpu_id
        self.detect_net = caffe.Net(prototxt, caffemodel, caffe.TEST)

        self.font = ImageFont.truetype(fontpath, 14)

    def draw_result(self, im, det_res, detection_time=0):
        # convert to pil image
        img_pil = Image.fromarray(im)
        draw = ImageDraw.Draw(img_pil)

        for res in det_res:
            class_name = res['class_name']
            bbox = res['bbox']
            draw.rectangle([(bbox[0], bbox[1]), (bbox[2], bbox[3])], outline=COLOR[class_name])
            # score = res['score']
            # "{}: {:.2f}".format(class_name, score)
        # draw information board
        # draw.rectangle([(10, 10), (160, 70)], fill=(100, 100, 100, 10))
        if detection_time > 0:
            draw.text((15, 15), "FPS: {:.1f}".format(1000/detection_time), font=self.font,
                      fill=(255, 255, 255, 255))
        del draw
        im = np.array(img_pil)
        return im

    def detection(self, img):
        # Detect all object classes and regress object bounds
        timer = Timer()
        timer.tic()
        scores, boxes = im_detect(self.detect_net, img)
        timer.toc()
        elapsed_time = timer.total_time * 1000
        print((('Detection took {:.3f}ms for '
                '{:d} object proposals').format(elapsed_time, boxes.shape[0])))

        # Visualize detections for each class
        det_res = []
        for cls_ind, cls in enumerate(CLASSES[1:]):
            cls_ind += 1  # because we skipped background
            cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
            cls_scores = scores[:, cls_ind]
            dets = np.hstack((cls_boxes,
                              cls_scores[:, np.newaxis])).astype(np.float32)
            keep = nms(dets, self.nms_thresh)
            dets = dets[keep, :]
            inds = np.where(dets[:, -1] >= self.conf_thresh)[0]
            if len(inds) == 0:
                continue
            for i in inds:
                bbox = dets[i, :4]
                bbox = [int(c) for c in bbox]
                score = dets[i, -1]
                det_res.append({"class_name": cls, "score": score, "bbox": bbox, "color": COLOR[cls]})

        return det_res, elapsed_time


def demo(vehicle_detection, data_file_path, save_data_path, scale=1.0):
    list_img_file_path = []
    if os.path.isdir(data_file_path):
        import glob
        list_img_file_path = []
        for ext in ['jpg', 'png', 'mp4', 'MOV']:
            list_img_file_path += glob.glob(data_file_path + '/*.{}'.format(ext))
    else:
        list_img_file_path.append(data_file_path)

    for data_file_path in list_img_file_path:
        if data_file_path.__contains__('_res.mp4'):
            continue
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
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * scale)
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * scale)
            if video_rotation == 90:
                size = (height, width)
            else:
                size = (width, height)
            writer = cv2.VideoWriter(save_video_path, fourcc, fps, size)

            # is video, loop each frame
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                # resize image if needed
                if scale != 1:
                    frame = cv2.resize(frame, (width, height))
                if video_rotation == 90:
                    # need to rotate
                    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                det_res, elapsed_time = vehicle_detection.detection(frame)
                res_img = vehicle_detection.draw_result(frame, det_res, elapsed_time)
                writer.write(res_img)

                cv2.imshow("res", res_img)
                if cv2.waitKey(1) & 0XFF == ord('q'):
                    break
            cap.release()
            writer.release()
        else:
            if scale != 1:
                img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
            det_res, elapsed_time = vehicle_detection.detection(img)
            res_img = vehicle_detection.draw_result(img, det_res)
            file_name = os.path.basename(data_file_path)
            save_path = os.path.join(save_data_path, file_name)
            cv2.imwrite(save_path, res_img)
            cv2.imshow("res", res_img)
            cv2.waitKey()
    return 1


if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    prototxt = os.path.join(this_dir, '..', "models/pascal_voc/Vehicle/faster_rcnn_end2end/test.prototxt")

    caffemodel = os.path.join(this_dir, '..',
                                       "output/faster_rcnn_end2end/voc_2007_trainval/vehicle_faster_rcnn_iter_70000.caffemodel")

    input_data_path_ = '/media/mvn/Data/Dataset/Image/ITS/VehicleDataset/darknet/Test/RCNN_4000177_Special_0721_TDH_b_1s.avi_1401.jpg'
    input_data_path_ = '/media/mvn/Data/Dataset/Image/ITS/Video'
    save_data_path_ = ''

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))

    vehicle_detection_ = VehicleDetection(prototxt, caffemodel, gpu_id=0,
                                          fontpath='/home/mvn/Desktop/Deeplearning/object_detection/py-faster-rcnn/arial.ttf')
    demo(vehicle_detection_, input_data_path_, save_data_path_, scale=0.5)
