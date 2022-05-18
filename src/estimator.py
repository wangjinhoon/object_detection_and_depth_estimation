#!/usr/bin/env python2

from abc import abstractmethod
import rospy
from yolov3_trt_ros.msg import BoundingBox, BoundingBoxes
from cv_bridge import CvBridge
from sensor_msgs.msg import Image


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

        # detected bounding boxes
        for bounding_box in msg.bounding_boxes:
            class_id = bounding_box.id
            xmin, ymin, xmax, ymax = bounding_box.xmin, bounding_box.ymin, bounding_box.xmax, bounding_box.ymax

        # TODO: geometric or homography ->
        # TODO: get distance & coordinate
        # TODO: visualize 2D map 


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

        