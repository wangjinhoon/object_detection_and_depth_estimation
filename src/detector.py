#!/usr/bin/env python2
#
# Copyright 1993-2019 NVIDIA Corporation.  All rights reserved.
#
# NOTICE TO LICENSEE:
#
# This source code and/or documentation ("Licensed Deliverables") are
# subject to NVIDIA intellectual property rights under U.S. and
# international Copyright laws.
#
# These Licensed Deliverables contained herein is PROPRIETARY and
# CONFIDENTIAL to NVIDIA and is being provided under the terms and
# conditions of a form of NVIDIA software license agreement by and
# between NVIDIA and Licensee ("License Agreement") or electronically
# accepted by Licensee.  Notwithstanding any terms or conditions to
# the contrary in the License Agreement, reproduction or disclosure
# of the Licensed Deliverables to any third party without the express
# written consent of NVIDIA is prohibited.
#
# NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
# LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
# SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
# PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
# NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
# DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
# NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
# NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
# LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
# SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
# DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
# WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
# ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
# OF THESE LICENSED DELIVERABLES.
#
# U.S. Government End Users.  These Licensed Deliverables are a
# "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
# 1995), consisting of "commercial computer software" and "commercial
# computer software documentation" as such terms are used in 48
# C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
# only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
# 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
# U.S. Government End Users acquire the Licensed Deliverables with
# only those rights set forth herein.
#
# Any use of the Licensed Deliverables in individual and commercial
# software must include, in the user documentation and internal
# comments to the code, the above Disclaimer and U.S. Government End
# Users Notice.
#

import sys, os
import time
import numpy as np
import cv2
import tensorrt as trt
from PIL import Image,ImageDraw
import rospy

from std_msgs.msg import String
from ppb_ros.msg import BoundingBox, BoundingBoxes

from cv_bridge import CvBridge
from sensor_msgs.msg import Image as Imageros

from data_processing import PreprocessYOLO, PostprocessYOLO, ALL_CATEGORIES
import common
from variables import *
from estimator import GeometricEstimator


TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

CFG = "/home/nvidia/xycar_ws/src/PPB-detection/src/yolov3-tiny_ppb_416.cfg"
TRT = '/home/nvidia/xycar_ws/src/PPB-detection/src/model_epoch2650.trt'
NUM_CLASS = 1

bridge = CvBridge()
xycar_image = np.empty(shape=[0])

class yolov3_trt(object):
    def __init__(self):
        self.cfg_file_path = CFG
        self.num_class = NUM_CLASS
        width, height, masks, anchors = parse_cfg_wh(self.cfg_file_path)
        self.engine_file_path = TRT
        self.show_img = True

        # Two-dimensional tuple with the target network's (spatial) input resolution in HW ordered
        input_resolution_yolov3_WH = (width, height)
        # Create a pre-processor object by specifying the required input resolution for YOLOv3
        self.preprocessor = PreprocessYOLO(input_resolution_yolov3_WH)

        # Output shapes expected by the post-processor
        output_channels = (self.num_class + 5) * 3
        if len(masks) == 2:
            self.output_shapes = [(1, output_channels, height//32, width//32), (1, output_channels, height//16, width//16)]
        else:
            self.output_shapes = [(1, output_channels, height//32, width//32), (1, output_channels, height//16, width//16), (1, output_channels, height//8, width//8)]

        postprocessor_args = {"yolo_masks": masks,                    # A list of 3 three-dimensional tuples for the YOLO masks
                              "yolo_anchors": anchors,
                              "obj_threshold": 0.5,                                               # Threshold for object coverage, float value between 0 and 1
                              "nms_threshold": 0.3,                                               # Threshold for non-max suppression algorithm, float value between 0 and 1
                              "yolo_input_resolution": input_resolution_yolov3_WH,
                              "num_class": self.num_class}

        self.postprocessor = PostprocessYOLO(**postprocessor_args)

        self.engine = get_engine(self.engine_file_path)

        self.context = self.engine.create_execution_context()
        
        # self.image_pub = rospy.Publisher('/yolov3_trt_ros/image', Imageros, queue_size=1)
        self.detection_pub = rospy.Publisher('/yolov3_trt_ros/detections', BoundingBoxes, queue_size=1)
        self.mapx, self.mapy = cv2.initUndistortRectifyMap(CAMERA_METRIX, DIST_COEFFS, None, None, (640, 480), cv2.CV_32FC1)
        self.estimator =  GeometricEstimator("trt_driver")
        self.w_rate = 640.0/416.0
        self.h_rate = 480.0/416.0

    def detect(self):
        rate = rospy.Rate(10)
        image_sub = rospy.Subscriber("/usb_cam/image_raw", Imageros, self.img_callback)
        while not rospy.is_shutdown():
            rate.sleep()

            # Do inference with TensorRT

            inputs, outputs, bindings, stream = common.allocate_buffers(self.engine)

            # if xycar_image is empty, skip inference
            if xycar_image.shape[0] == 0:
                continue
            
            if self.show_img:
                show_img = xycar_image.copy()
                #print(show_img.shape)
                #cv2.imshow("show_trt",xycar_image)
                #cv2.waitKey(1)

            image = self.preprocessor.process(xycar_image)
            # Store the shape of the original input image in WH format, we will need it for later
            shape_orig_WH = (image.shape[3], image.shape[2])
            # Set host input to the image. The common.do_inference function will copy the input to the GPU before executing.
            start_time = time.time()
            inputs[0].host = image
            trt_outputs = common.do_inference(self.context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
            #print("trt_output:", len(trt_outputs))
            #print("element shape", trt_outputs[0].shape)

            # Before doing post-processing, we need to reshape the outputs as the common.do_inference will give us flat arrays.
            trt_outputs = [output.reshape(shape) for output, shape in zip(trt_outputs, self.output_shapes)]
            # Run the post-processing algorithms on the TensorRT outputs and get the bounding box details of detected objects
            boxes, classes, scores = self.postprocessor.process(trt_outputs, shape_orig_WH)

            latency = time.time() - start_time
            fps = 1 / latency

            #publish detected objects boxes and classes
            self.publish(boxes, scores, classes)

            # Draw the bounding boxes onto the original input image and save it as a PNG file
            # print(boxes, classes, scores)
            if self.show_img:
                img_show = np.array(np.transpose(image[0], (1,2,0)) * 255, dtype=np.uint8)
                obj_detected_img = draw_bboxes(Image.fromarray(img_show), boxes, scores, classes, ALL_CATEGORIES)
                obj_detected_img_np = np.array(obj_detected_img)    
                #show_img = cv2.cvtColor(obj_detected_img_np, cv2.COLOR_RGB2BGR)
        
                if boxes is not None:
                    for index, box in enumerate(boxes):
                        #print("box: ", box)
                        w_rate = 640.0/416.0
                        h_rate = 480.0/416.0
                        xmin = int(box[0] * self.w_rate)
                        ymin = int(box[1] * self.h_rate)
                        xmax = int((box[0] + box[2]) * self.w_rate)
                        ymax = int((box[1] + box[3]) * self.h_rate)
                        
                        cv2.rectangle(show_img, (xmin, ymin), (xmax, ymax), (0,0,255), 2)
                        cv2.putText(show_img, str(index), (xmin, ymin), 1, 1, (0,0,0), 2)

                cv2.putText(show_img, "FPS:"+str(int(fps)), (10,50),cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0),2,1)
                cv2.imshow("result",show_img)
                cv2.waitKey(1)

    def _write_message(self, detection_results, boxes, scores, classes):
        """ populate output message with input header and bounding boxes information """

        if boxes is None:
            return None
        for box, score, category in zip(boxes, scores, classes):
            # Populate darknet message
            minx, miny, width, height = box
            detection_msg = BoundingBox()
            detection_msg.xmin = int(minx * self.w_rate)
            detection_msg.ymin = int(miny * self.h_rate)
            detection_msg.xmax = int((minx + width) * self.w_rate)
            detection_msg.ymax = int((miny + height) * self.h_rate)
            detection_msg.probability = score
            detection_msg.id = int(category)
            detection_results.bounding_boxes.append(detection_msg)
        return detection_results

    def publish(self, boxes, confs, classes):
        """ Publishes to detector_msgs
        Parameters:
        boxes (List(List(int))) : Bounding boxes of all objects
        confs (List(double))	: Probability scores of all objects
        classes  (List(int))	: Class ID of all classes
        """
        detection_results = BoundingBoxes()
        self._write_message(detection_results, boxes, confs, classes)
        self.estimator.callback(detection_results)
        #self.detection_pub.publish(detection_results)
        #calibrated_image = Imageros()
        #calibrated_image.data = xycar_image
        #self.image_pub.publish(CvBridge.cv2_to_imgmsg(calibrated_image, "bgr8"))

    def img_callback(self, data):
        global xycar_image
        img = bridge.imgmsg_to_cv2(data, "bgr8")
        xycar_image = cv2.remap(img, self.mapx, self.mapy, cv2.INTER_LINEAR)


#parse width, height, masks and anchors from cfg file
def parse_cfg_wh(cfg):
    masks = []
    with open(cfg, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if 'width' in line:
                w = int(line[line.find('=')+1:].replace('\n',''))
            elif 'height' in line:
                h = int(line[line.find('=')+1:].replace('\n',''))
            elif 'anchors' in line:
                anchor = line.split('=')[1].replace('\n','')
                anc = [int(a) for a in anchor.split(',')]
                anchors = [(anc[i*2], anc[i*2+1]) for i in range(len(anc) // 2)]
            elif 'mask' in line:
                mask = line.split('=')[1].replace('\n','')
                m = tuple(int(a) for a in mask.split(','))
                masks.append(m)
    return w, h, masks, anchors

def draw_bboxes(image_raw, bboxes, confidences, categories, all_categories, bbox_color='blue'):
    """Draw the bounding boxes on the original input image and return it.

    Keyword arguments:
    image_raw -- a raw PIL Image
    bboxes -- NumPy array containing the bounding box coordinates of N objects, with shape (N,4).
    categories -- NumPy array containing the corresponding category for each object,
    with shape (N,)
    confidences -- NumPy array containing the corresponding confidence for each object,
    with shape (N,)
    all_categories -- a list of all categories in the correct ordered (required for looking up
    the category name)
    bbox_color -- an optional string specifying the color of the bounding boxes (default: 'blue')
    """

    draw = ImageDraw.Draw(image_raw)
    if bboxes is None :
        return image_raw
    # for index, box in enumerate(bboxes):
    #     # x_coord, y_coord, width, height = box
    #     # left = max(0, np.floor(x_coord + 0.5).astype(int))
    #     # top = max(0, np.floor(y_coord + 0.5).astype(int))
    #     # right = min(image_raw.width, np.floor(x_coord + width + 0.5).astype(int))
    #     # bottom = min(image_raw.height, np.floor(y_coord + height + 0.5).astype(int))

    #     xmin = int(box[0])
    #     ymin = int(box[1])
    #     xmax = int(xmin + box[2])
    #     ymax = int(ymin + box[3])
    #     width = xmax - xmin
    #     height = ymax - ymin

    #     cv2.rectangle(image_raw, (xmin, ymin), (xmax, ymax), (0,0,255), 1)
    #     cv2.putText(image_raw, index, (xmin, ymin), 1, 1, (0,0,0), 2)

    #     # draw.rectangle((xmin, ymin), (xmax, ymax), outline=bbox_color)
    #     # draw.text((xmin, ymax), '{0}'.format(index), fill=bbox_color)

    return image_raw

def get_engine(engine_file_path=""):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""
    if os.path.exists(engine_file_path):
        # If a serialized engine exists, use it instead of building an engine.
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        print("no trt model")
        sys.exit(1)

if __name__ == '__main__':
    yolo = yolov3_trt()
    rospy.init_node('yolov3_trt_ros', anonymous=True)
    yolo.detect()

