import os

this_dir = os.path.dirname(__file__)


class ObjectCounting():
    def __init__(self, object_detector, multi_obj_tracker, counter_area):
        self.counter_area = counter_area
        self.object_detector = object_detector
        self.multi_obj_tracker = multi_obj_tracker
        # init count result base on object_detector
        self.count_result = {}
        for idx in range(1, len(object_detector.CLASSES)):
            self.count_result[object_detector.CLASSES[idx]] = 0
        print("obj counting")

    def update_counter_area(self, counter_area):
        self.counter_area = counter_area

    def process_frame(self, frame):
        # detect object
        det_res, elapsed_time = self.object_detector.detection(frame)

        # tracking and draw tracking
        self.multi_obj_tracker.update(frame, det_res)

        # counting
        self.count_result = self.multi_obj_tracker.count_object(self.counter_area, self.count_result)

        # drawing object detection, tracking, counting results
        res_img = self.object_detector.draw_result(frame, det_res, elapsed_time, self.count_result)
        res_img = self.multi_obj_tracker.draw_tracking(res_img)
        return res_img
