#!/usr/bin/env python2

from abc import abstractmethod
import rospy
from yolov3_trt_ros.msg import BoundingBox, BoundingBoxes
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from variables import *

import cv2
import numpy as np
import math
import matplotlib.pyplot as plt


class Estimator:
    def __call__(self, node_name):
        rospy.init_node(node_name)
        self.sub = rospy.Subscriber('/yolov3_trt_ros/detections', BoundingBoxes, self.callback, queue_size=1)
        self.bridge = CvBridge()

    @abstractmethod
    def callback(msg):
        pass


class GeometricEstimator(Estimator):
    def callback(msg):
      # calibrated image
        img = msg.img
        dis = []
        # detected bounding boxes
        for bounding_box in msg.bounding_boxes:
            class_id = bounding_box.id
            xmin = int(bounding_box.xmin)
            ymin = int(bounding_box.ymin)
            xmax = int(bounding_box.xmax)
            ymax = int(bounding_box.ymax)
            width = xmax - xmin
            height = ymax - ymin

            CAMERA_HEIGHT = 0.1475
            FOVh = (135.4-42)/2

            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0,0,255), 1)
            cv2.putText(img, f"{i}", (xmin, ymin), 1, 1, (0,0,0), 2)

            x_center = (xmin+xmax)/2
            y_norm = (ymax - CAMERA_METRIX[1][2]) / CAMERA_METRIX[1][1]
            deltax = (x_center - width/2)
            azimuth = FOVh - (deltax/320*FOVh)

            dz = (1 * CAMERA_HEIGHT / y_norm * 100)
            dx = dz * math.tan(math.pi*(azimuth/180))
            if deltax > 0:
                dx *= -1
            d = dz / math.cos(math.pi * (azimuth/180))
            if azimuth < 0:
                azimuth *= -1

            t = (dx,dz)
            dis.append(t) 
        
        mask = cv2.imread("mask.png")

        for list in dis:
            x = round(list[0]) + 160
            if x >= 160:
                x += -(160-x) * 3
            else:
                x += (x-160) * 3
            y = 320 - round(list[1]) * 2
            # y = 320 - round(list[1]) => y -= (320 - (320 - round(list[1])))
            
            cv2.circle(mask, (int(x),int(y)), 10, (255, 255, 255), -1)
            cv2.putText(mask, f"({int(list[0])}, {int(list[1])})", (int(x - 20),int(y + 30)), 1, 1, (255, 255, 255), 1)
        
        cv2.imshow("mask",mask)
        cv2.imshow("img", img)
        cv2.waitKey(0)
       


class HomographyEstimator(Estimator):
     def callback(msg):
      # calibrated image
        img = msg.img

        # detected bounding boxes
        for bounding_box in msg.bounding_boxes:
            class_id = bounding_box.id
            xmin, ymin, xmax, ymax = bounding_box.xmin, bounding_box.ymin, bounding_box.xmax, bounding_box.ymax

        # TODO: geometric or homography ->
        # TODO: get distance & coordinate
        # TODO: visualize 2D map 

        