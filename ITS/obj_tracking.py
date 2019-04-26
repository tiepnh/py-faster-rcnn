import cv2
import math
import numpy as np
from collections import OrderedDict
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon


class ObjectTracker():
    def __init__(self, id, detect_obj, max_lost):
        # detect_obj: {"class_name": cls, "score": score, "bbox": bbox, "color": COLOR[cls]}
        bbox = detect_obj['bbox']
        color = detect_obj['color']
        cls = detect_obj['class_name']
        self.bboxes = []
        self.centers = []
        self.lost_time = 0
        self.max_lost = max_lost
        self.tracker_id = id
        self.color = color
        self.class_name = cls
        self.update_loc(bbox)
        self.is_counted = 0
        print('Create ObjectTracker')

    def update_loc(self, bbox):
        if len(bbox) == 0:
            self.lost_time += 1
        else:
            self.bboxes.append(bbox)
            self.centers.append([(bbox[0] + bbox[2])//2, (bbox[1] + bbox[3])//2])
            self.lost_time = 0

    def draw_tracking(self, img):
        bbox = self.bboxes[-1]
        if self.lost_time == 0:
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), self.color)
            cv2.putText(img, "{}".format(self.tracker_id), (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        self.color)
        # else:
        #     cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 0))

        return img

    def distance(self, img, detect_obj):
        # detect_obj: {"class_name": cls, "score": score, "bbox": bbox, "color": COLOR[cls]}
        d = float("INF")
        color = detect_obj['color']
        if color == self.color:
            bbox = detect_obj['bbox']
            center = ([(bbox[0] + bbox[2])//2, (bbox[1] + bbox[3])//2])
            dx = (center[0] - self.centers[-1][0])*1.0/img.shape[1]
            dy = (center[1] - self.centers[-1][1])*1.0/img.shape[0]
            d = math.fabs(dx) + 2 * math.fabs(dy)
        return d

    def lost(self, img=None):
        # for vehicle counting only
        if img is not None:
            if self.bboxes[-1][3] >= img.shape[0] * 0.9:
                self.lost_time = self.max_lost + 10
        return self.lost_time >= self.max_lost

    def do_count(self, counter_area):
        if self.is_counted or self.lost_time > 0 or len(counter_area) == 0:
            return self.class_name, 0
        obj_point = Point((self.bboxes[-1][0] + self.bboxes[-1][2])//2, self.bboxes[-1][3])
        polygon = Polygon(counter_area)
        if polygon.contains(obj_point):
            self.is_counted = True
            return self.class_name, 1
        else:
            return self.class_name, 0

    @staticmethod
    def create_tracker_by_name(tracker_type):
        tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
        # Create a tracker based on tracker name
        if tracker_type == tracker_types[0]:
            tracker = cv2.TrackerBoosting_create()
        elif tracker_type == tracker_types[1]:
            tracker = cv2.TrackerMIL_create()
        elif tracker_type == tracker_types[2]:
            tracker = cv2.TrackerKCF_create()
        elif tracker_type == tracker_types[3]:
            tracker = cv2.TrackerTLD_create()
        elif tracker_type == tracker_types[4]:
            tracker = cv2.TrackerMedianFlow_create()
        elif tracker_type == tracker_types[5]:
            tracker = cv2.TrackerGOTURN_create()
        elif tracker_type == tracker_types[6]:
            tracker = cv2.TrackerMOSSE_create()
        elif tracker_type == tracker_types[7]:
            tracker = cv2.TrackerCSRT_create()
        else:
            tracker = None
            print('Incorrect tracker name')
            print('Available trackers are:')
            for t in tracker_types:
                print(t)

        return tracker


class MultiObjectTracking():
    def __init__(self, max_lost=30, max_relative_distance=0.1):
        """
        Create multi object tracker
        :param max_lost: Number frame the object did not appear before delete
        :param max_relative_distance: The max distance of same object (relative with width, height)
        """
        self.list_tracker = OrderedDict()
        self.next_tracker_id = 0
        self.max_lost = max_lost
        self.max_relative_distance = max_relative_distance
        print('MultiObjectTracking')

    def update(self, img, list_det_res):
        # if there is no detected object --> all tracker lost
        if len(list_det_res) == 0:
            for tracker_id, tracker in self.list_tracker.items():
                tracker.update_loc([])
                if tracker.lost(img):
                    self.de_register(tracker_id)
            return self.list_tracker

        # if the first time --> register all detected object as new tracker
        if len(self.list_tracker) == 0:
            for detect_obj in list_det_res:
                self.register(detect_obj)
            return self.list_tracker

        list_tracker_id = list(self.list_tracker.keys())

        # Get distance between tracker and detected object
        D = np.ones((len(self.list_tracker), len(list_det_res)), dtype=np.float)
        for y, tracker_id in enumerate(self.list_tracker.keys()):
            for x, detect_obj in enumerate(list_det_res):
                D[y][x] = self.list_tracker[tracker_id].distance(img, detect_obj)

        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        used_rows = set()
        used_cols = set()

        # loop over the combination of the (row, column) index tuples
        for (row, col) in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue

            # if distance between object and tracker too far --> ignore matching
            if D[row][col] >= self.max_relative_distance:
                continue

            obj_bbox = list_det_res[col]['bbox']
            tracker_id = list_tracker_id[row]
            self.list_tracker[tracker_id].update_loc(obj_bbox)
            used_rows.add(row)
            used_cols.add(col)

        # get all row(tracker) and column(object) index which are not examined (lost/ new tracker)
        un_used_tracker = set(range(0, len(self.list_tracker))).difference(used_rows)
        un_used_object = set(range(0, len(list_det_res))).difference(used_cols)

        # update and check lost tracker
        for index in un_used_tracker:
            tracker_id = list_tracker_id[index]
            self.list_tracker[tracker_id].update_loc([])
            if self.list_tracker[tracker_id].lost(img):
                self.de_register(tracker_id)

        # register new tracker
        for index in un_used_object:
            self.register(list_det_res[index])

        return self.list_tracker

    def register(self, detect_obj):
        tracker = ObjectTracker(self.next_tracker_id, detect_obj, self.max_lost)
        self.list_tracker[self.next_tracker_id] = tracker
        self.next_tracker_id += 1

    def de_register(self, tracker_id):
        del self.list_tracker[tracker_id]

    def draw_tracking(self, img):
        for tracker_id, tracker in self.list_tracker.items():
            img = tracker.draw_tracking(img)
        return img

    def count_object(self, counter_area, count_result):
        for tracker_id, tracker in self.list_tracker.items():
            cls, should_count = tracker.do_count(counter_area)
            if count_result.__contains__(cls):
                count_result[cls] += should_count
            else:
                count_result[cls] = should_count
        return count_result


