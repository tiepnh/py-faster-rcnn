# -*- coding: utf-8 -*-

from vehicle_detection import VehicleDetection
from obj_tracking import MultiObjectTracking
from object_counting import ObjectCounting
import os
import cv2
import exiftool
import numpy as np

this_dir = os.path.dirname(__file__)

mouse_clicked_point = None
mouse_move_point = None


def handle_mouse(event, x, y, flags, params):
    global mouse_move_point
    global mouse_clicked_point
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_clicked_point = (x, y)

    if event == cv2.EVENT_MOUSEMOVE:
        mouse_move_point = (x, y)


# config display result
main_window_name = "main_window"
cv2.namedWindow(main_window_name)
cv2.setMouseCallback(main_window_name, handle_mouse)


def demo(object_detector, multi_obj_tracker, data_file_path, scale=1.0):
    global mouse_clicked_point

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
            # init object counter
            counter_area = []
            object_counter = ObjectCounting(object_detector, multi_obj_tracker, counter_area)

            # Read video / init video write
            with exiftool.ExifTool() as et:
                metadata = et.get_metadata(data_file_path)
                print (metadata)
                video_rotation = metadata['Composite:Rotation']

            cap = cv2.VideoCapture(data_file_path)
            save_video_path = data_file_path + '_res.mp4'
            fourcc = cv2.VideoWriter_fourcc(*'MP4V')
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * scale)
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * scale)
            if video_rotation == 90:
                temp = height
                height = width
                width = temp

            size = (width, height)
            writer = cv2.VideoWriter(save_video_path, fourcc, fps, size)
            frame_id = 0
            total_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)

            # set area at 1/2 of frame
            mouse_clicked_point = (size[0]//2, size[1]*2//5)

            # is video, loop each frame
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # just run with 80% video
                frame_id += 1
                if frame_id < total_frame * 0.15:
                    continue
                if frame_id > total_frame * 0.9:
                    break

                if video_rotation == 90:
                    # need to rotate
                    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

                # resize image if needed
                if scale != 1:
                    frame = cv2.resize(frame, (width, height))

                # make counter area
                if mouse_clicked_point is not None:
                    counter_area = [(0, mouse_clicked_point[1]), (frame.shape[1], mouse_clicked_point[1]),
                                    (frame.shape[1], frame.shape[0]), (0, frame.shape[0])]
                    object_counter.update_counter_area(counter_area)

                # count by frame
                res_img = object_counter.process_frame(frame)

                # draw counter area
                if mouse_clicked_point is not None:
                    cv2.line(res_img, (0, mouse_clicked_point[1]), (frame.shape[1], mouse_clicked_point[1]),
                             (255, 255, 255), 2)
                # if mouse_move_point is not None:
                #     cv2.line(res_img, (0, mouse_move_point[1]), (frame.shape[1], mouse_move_point[1]), (255, 255, 0), 1)

                writer.write(res_img)

                cv2.imshow(main_window_name, res_img)
                if cv2.waitKey(1) & 0XFF == ord('q'):
                    break

            # Close video
            cap.release()
            writer.release()
        # break
    return 1


def main():
    prototxt = os.path.join(this_dir, '..', "models/pascal_voc/Vehicle/faster_rcnn_end2end/test.prototxt")

    caffemodel = os.path.join(this_dir, '..',
                              "models/pascal_voc/Vehicle/faster_rcnn_end2end/vehicle_faster_rcnn_iter_70000.caffemodel")

    input_data_path_ = '/media/mvn/Data/Dataset/Image/ITS/Video'

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))

    object_detector = VehicleDetection(prototxt, caffemodel, gpu_id=0,
                                          fontpath='/home/mvn/Desktop/Deeplearning/object_detection/py-faster-rcnn/arial.ttf')

    multi_obj_tracker = MultiObjectTracking(max_lost=30, max_relative_distance=0.3)
    demo(object_detector, multi_obj_tracker, input_data_path_, scale=0.5)

    return 1


if __name__ == '__main__':
    main()
