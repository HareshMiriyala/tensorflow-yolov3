#!/usr/bin/python3.6

"""
email : haresh.miriyala@utexas.edu
Name : Haresh Karnan
"""
import rospy
import cv2
import time
import numpy as np
import tensorflow as tf
import sys
import math
from PIL import Image
from openpose_ros_msgs.msg import OpenPoseHumanList
from core import utils
from sensor_msgs.msg import Image as Img
from openni import openni2
from cv_bridge import CvBridge, CvBridgeError
from ros_numpy import numpify
from PIL import ImageFont, ImageDraw

import os
print('current path is :: ',os.getcwd())
IMAGE_H, IMAGE_W = 416, 416
video_path = 0 # use camera

classes = utils.read_coco_names('./data/coco.names')
num_classes = len(classes)
input_tensor, output_tensors = utils.read_pb_return_tensors(tf.get_default_graph(),
                                                            "./checkpoint/yolov3_gpu_nms.pb",
                                                            ["Placeholder:0", "concat_9:0", "mul_6:0"])

class find_pointing_bag():
    def __init__(self):
        self.r_elbow = 0;self.r_shoulder=0;self.r_wrist=0;
        self.l_elbow = 0;self.l_shoulder=0;self.l_wrist=0;
        self.num_humans=0;
        self.openpose_id = {
            "RShoulder":2,
            "RElbow":3,
            "RWrist":4,
            "LShoulder":5,
            "LElbow":6,
            "LWrist":7
        }
        # self.openpose_listener()
        rospy.Subscriber("/openpose_ros/human_list", OpenPoseHumanList, self.callback)

    def callback(self, data):
        # rospy.loginfo(rospy.get_caller_id() + "I heard %s", data)
        # if (data.num_humans == 1): print('Only 1 human found in the camera')
        # print(self.openpose_id)
        self.num_humans = data.num_humans
        self.r_elbow = data.human_list[0].body_key_points_with_prob[self.openpose_id["RElbow"]]
        self.l_elbow = data.human_list[0].body_key_points_with_prob[self.openpose_id["LElbow"]]
        self.r_shoulder = data.human_list[0].body_key_points_with_prob[self.openpose_id["RShoulder"]]
        self.l_shoulder = data.human_list[0].body_key_points_with_prob[self.openpose_id["LShoulder"]]
        self.r_wrist = data.human_list[0].body_key_points_with_prob[self.openpose_id["RWrist"]]
        self.l_wrist = data.human_list[0].body_key_points_with_prob[self.openpose_id["LWrist"]]
        # print(self.l_shoulder)


class openni_camera():
    def __init__(self,display=True):
        self.cv_image = None
        self.bridge = CvBridge()
        self.display = display # display bool
        rospy.Subscriber("/camera/rgb/image_raw", Img, self.camera_callback)

    def camera_callback(self,data):
        cv_image = numpify(data)
        self.cv_image = cv_image
        if self.display:
            cv2.imshow('openni_input_image',self.cv_image)
            cv2.waitKey(1)

    def get_frame(self):
        return self.cv_image

def tr_area(a,b,c):
    # returns the area of a triangle of 3 sides a,b,c
    # a=abs(a);b=abs(b);c=abs(c);
    s = (a+b+c)/2.0
    return math.sqrt(s*(s-a)*(s-b)*(s-c))

def len_side(x1,y1,x2,y2):
    return abs(math.sqrt((x2-x1)**2+(y2-y1)**2))

def concave_polygon(box,shoulder,wrist):

    print('Box : ')
    print(box)
    print('Shoulder : ')
    print(shoulder.x,shoulder.y)
    print('Wrist')
    print(wrist.x,wrist.y)

    diag = len_side(box[0],box[1],box[2],box[3])
    big_side_1 = len_side(shoulder.x,shoulder.y,box[0],box[1])
    big_side_2 = len_side(shoulder.x,shoulder.y,box[2],box[3])
    area_total = tr_area(diag,big_side_1,big_side_2)
    print('area_total: ',area_total)

    s0 = len_side(shoulder.x,shoulder.y,wrist.x,wrist.y)
    s1 = len_side(wrist.x,wrist.y,box[0],box[1])
    s2 = len_side(wrist.x,wrist.y,box[2],box[3])

    area_tr_1 = tr_area(big_side_1,s0,s1)
    area_tr_2 = tr_area(big_side_2,s0,s2)
    area_tr_3 = tr_area(s2,s1,diag)

    triangle_sum_area = area_tr_1+area_tr_2+area_tr_3

    print('triangle sum area : ',triangle_sum_area)
    if abs(area_total-triangle_sum_area)<1000:
        return True
    else:
        return False

if __name__ == '__main__':
    rospy.init_node('bag_finder', anonymous=True)

    with tf.Session() as sess:

        bag_finder = find_pointing_bag()
        camera = openni_camera(display=False)

        while True:

            frame = camera.get_frame()
            if frame is None: continue

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # cv2.imshow('dis', frame)
            # cv2.waitKey(1)

            frame = Image.fromarray(frame)
            img_resized = np.array(frame.resize((IMAGE_H, IMAGE_W)), dtype=np.float32)
            img_resized = img_resized / 255.

            # cv2.imshow('image_resized', np.asarray(img_resized))
            # cv2.waitKey(1)

            prev_time = time.time()
            #
            boxes, scores = sess.run(output_tensors, feed_dict={input_tensor: np.expand_dims(img_resized, axis=0)})
            boxes, scores, labels = utils.cpu_nms(boxes, scores, num_classes, score_thresh=0.4, iou_thresh=0.5)
            #
            idx_backpack = [i for i, x in enumerate(labels) if x == 56]
            boxes_backpack = [box for i, box in enumerate(boxes) if i in idx_backpack]
            scores_backpack = [score for i, score in enumerate(scores) if i in idx_backpack]
            labels_backpack = [label for i, label in enumerate(labels) if i in idx_backpack]
            frame_save = frame.copy()

            yolo_chair, bbox_rescaled = utils.draw_boxes(frame, boxes_backpack, scores_backpack, labels_backpack,
                                                         classes, (IMAGE_H, IMAGE_W), show=False)
            yolo_chair = np.asarray(yolo_chair)

            # yolo_chair = cv2.cvtColor(yolo_chair, cv2.COLOR_BGR2RGB)

            # cv2.line(yolo_chair, (int(bag_finder.r_wrist.x), int(bag_finder.r_wrist.y)), (int(bbox_rescaled[0][0]), int(bbox_rescaled[0][1])), (255, 0, 0), 5)

            # cv2.imshow("yolo_result", yolo_chair)
            # cv2.waitKey(1)

            if labels is None: continue
            if not 56 in labels: continue  # filtering chairs
            if not 0 in labels: continue  # filtering no humans

            # idx_backpack = labels.tolist().index(56)
            idx_backpack = [i for i, x in enumerate(labels) if x == 56]
            box_id = None

            for i in range(len(bbox_rescaled)):
                # first check if the person is definitely pointing to something
                # with either hands
                if (abs(bag_finder.l_shoulder.x - bag_finder.l_wrist.x) < 50 and abs(
                        bag_finder.r_shoulder.x - bag_finder.r_wrist.x) < 20):
                    print('not pointing to anything')
                    break

                if (bbox_rescaled[i][0] > 600):
                    # box to the left of the operator
                    bbox_rescaled[i][0], bbox_rescaled[i][1], bbox_rescaled[i][2], bbox_rescaled[i][3] = \
                        bbox_rescaled[i][0], bbox_rescaled[i][3], bbox_rescaled[i][2], bbox_rescaled[i][1]

                    if (concave_polygon(bbox_rescaled[i], bag_finder.l_shoulder, bag_finder.l_wrist)):
                        # left hand pointing
                        box_id = i
                        print('CONCAVE POLYGON FOUND !')
                        bbox_rescaled[i][0], bbox_rescaled[i][3], bbox_rescaled[i][2], bbox_rescaled[i][1] = \
                            bbox_rescaled[i][0], bbox_rescaled[i][1], bbox_rescaled[i][2], bbox_rescaled[i][3]
                        break
                    elif (concave_polygon(bbox_rescaled[i], bag_finder.r_shoulder, bag_finder.r_wrist)):
                        # right hand pointing
                        box_id = i
                        print('CONCAVE POLYGON FOUND !')
                        bbox_rescaled[i][0], bbox_rescaled[i][3], bbox_rescaled[i][2], bbox_rescaled[i][1] = \
                            bbox_rescaled[i][0], bbox_rescaled[i][1], bbox_rescaled[i][2], bbox_rescaled[i][3]
                        break
                else:
                    # box to the right of the operator
                    if (concave_polygon(bbox_rescaled[i], bag_finder.l_shoulder, bag_finder.l_wrist)):
                        # left hand pointing.
                        box_id = i
                        print('CONCAVE POLYGON FOUND !')
                        break
                    elif (concave_polygon(bbox_rescaled[i], bag_finder.r_shoulder, bag_finder.r_wrist)):
                        # right hand pointing
                        box_id = i
                        print('CONCAVE POLYGON FOUND !')
                        break

            if box_id is None:
                image = np.asarray(frame_save)
            else:
                draw = ImageDraw.Draw(frame_save)
                draw.rectangle(bbox_rescaled[box_id], outline=(255, 0, 0), width=3)

                # p1 = (int(bbox_rescaled[box_id][0]),int(bbox_rescaled[box_id][1]))
                # p2 = (int(bbox_rescaled[box_id][2]),int(bbox_rescaled[box_id][3]))
                # print('points are : ',p1,p2)
                image = np.asarray(frame_save)

            cv2.imshow('result', image)
            cv2.waitKey(1)

            # image,_ = utils.draw_boxes(frame, [boxes[idx_backpack]], [scores[idx_backpack]], [labels[idx_backpack]], classes, (IMAGE_H, IMAGE_W), show=False)
            #
