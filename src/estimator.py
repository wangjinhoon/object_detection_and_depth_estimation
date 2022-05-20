#!/usr/bin/env python2

from abc import abstractmethod
import rospy
from ppb_ros.msg import BoundingBox, BoundingBoxes
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from variables import *

import cv2
import numpy as np
import math
import matplotlib.pyplot as plt


xycar_image = None

class Estimator:
    def __init__(self, node_name):
        #rospy.init_node(node_name)
        #self.sub = rospy.Subscriber('/yolov3_trt_ros/detections', BoundingBoxes, self.callback, queue_size=1)
        # self.sub_img = rospy.Subscriber('/yolov3_trt_ros/detections', BoundingBoxes, self.callback, queue_size=1)
        self.bridge = CvBridge()

    @abstractmethod
    def callback(self, msg):
        pass

    # def img_callback(msg):
    #     global xycar_image
    #     xycar_image = CvBridge.imgmsg_to_cv2(msg, "bgr8")


    def run(self):
        rospy.spin()


class GeometricEstimator(Estimator):
    def callback(self, msg):
        #print(msg)
      # calibrated image
        # img = xycar_image
        dis = []
        # detected bounding boxes
        for bounding_box in msg.bounding_boxes:
            class_id = bounding_box.id
            xmin = int(bounding_box.xmin)
            ymin = int(bounding_box.ymin)
            xmax = int(bounding_box.xmax)
            ymax = int(bounding_box.ymax)
            print(xmin, ymin, xmax, ymax)
            width = 640
            # height = ymax - ymin

            CAMERA_HEIGHT = 0.1475
            FOVh = (135.4-42)/2

            # cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0,0,255), 1)
            # cv2.putText(img, str(class_id), (xmin, ymin), 1, 1, (0,0,0), 2)

            x_center = (xmin+xmax)/2.
            y_norm = (ymax - CAMERA_METRIX[1][2]) / CAMERA_METRIX[1][1]
            deltax = (x_center - width/2.)
            azimuth = (deltax/(width/2.) * FOVh)

            dz = (1 * CAMERA_HEIGHT / y_norm * 100.)
            dx = dz * math.tan(math.pi*(azimuth/180.))
            # if deltax > 0:
            #     dx *= -1
            d = dz / math.cos(math.pi * (azimuth/180.))
            if azimuth < 0:
                azimuth *= -1

            t = (dx,dz)
            dis.append(t) 
            print("x_center", x_center)
            print("y_norm", y_norm)
            print("deltax", deltax)
            print("azimuth", azimuth)
        
        mask = cv2.imread("/home/nvidia/xycar_ws/src/PPB-detection/src/mask.png")
        cv2.circle(mask, (160, 320), 5, (255, 255, 255), -1)

        for list in dis:
            print("dx/dz",int(list[0]),int(list[1]))
            x = list[0] * (320./640.) + 160
            if x >= 160:
                x += -(160-x) * 4
            else:
                x += (x-160) * 4
            y = 320 - round(list[1] * (320./480.)) * 3
            print("x/y : ",x, y)

            # cv2.circle(mask, (int(x),int(y)), 10, (255, 255, 255), -1)
            # cv2.putText(mask, "({}, {})".format(int(list[0]), int(list[1])), (int(x - 20),int(y + 30)), 1, 1, (255, 255, 255), 1)
            cv2.circle(mask, (int(x),int(y)), 10, (0, 0, 255), -1)
            cv2.putText(mask, "({}, {}, d :{})".format(int(dx), int(dz), int(d)), (int(x) - 20,int(y) + 30), 1, 1, (255, 255, 255), 1)
        cv2.imshow("mask", mask)
        cv2.waitKey(1)
       


class HomographyEstimator(Estimator):
     def callback(self, msg):
      # calibrated image
        # img = msg.img

        # detected bounding boxes
        for bounding_box in msg.bounding_boxes:
            class_id = bounding_box.id
            xmin, ymin, xmax, ymax = bounding_box.xmin, bounding_box.ymin, bounding_box.xmax, bounding_box.ymax

        # TODO: geometric or homography ->
        # TODO: get distance & coordinate
        # TODO: visualize 2D map 


