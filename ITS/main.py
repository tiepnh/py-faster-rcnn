from vehicle_detection import VehicleDetection
from obj_tracking import MultiObjectTracking
import os
import cv2
import exiftool

this_dir = os.path.dirname(__file__)


def demo_video(vehicle_detection, multi_obj_tracker, data_file_path, save_data_path, scale=1.0):
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
                size = (height, width)
            else:
                size = (width, height)
            writer = cv2.VideoWriter(save_video_path, fourcc, fps, size)
            frame_id = 0
            total_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            # is video, loop each frame
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                # just run with 80% video
                frame_id += 1
                if frame_id < total_frame * 0.1:
                    continue
                if frame_id > total_frame * 0.9:
                    break

                # resize image if needed
                if scale != 1:
                    frame = cv2.resize(frame, (width, height))
                if video_rotation == 90:
                    # need to rotate
                    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

                # detect object
                det_res, elapsed_time = vehicle_detection.detection(frame)
                res_img = vehicle_detection.draw_result(frame, det_res, elapsed_time)

                # tracking and draw tracking
                multi_obj_tracker.update(frame, det_res)
                res_img = multi_obj_tracker.draw_tracking(res_img)

                writer.write(res_img)

                cv2.imshow("res", res_img)
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
                              "output/faster_rcnn_end2end/voc_2007_trainval/vehicle_faster_rcnn_iter_70000.caffemodel")

    input_data_path_ = '/media/mvn/Data/Dataset/Image/ITS/Video'
    save_data_path_ = ''

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))

    vehicle_detection_ = VehicleDetection(prototxt, caffemodel, gpu_id=0,
                                          fontpath='/home/mvn/Desktop/Deeplearning/object_detection/py-faster-rcnn/arial.ttf')

    multi_obj_tracker = MultiObjectTracking(max_lost=30, max_relative_distance=0.3)
    demo_video(vehicle_detection_, multi_obj_tracker, input_data_path_, save_data_path_, scale=0.5)

    return 1


if __name__ == '__main__':
    main()