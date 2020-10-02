#!/usr/bin/env python
import rospy
import rospkg

import time
from math import floor, ceil, sqrt
import cv2
import numpy as np
from os.path import join
import torch

import message_filters
from cv_bridge import CvBridge, CvBridgeError

from sensor_msgs.msg import Image
from wm_od_interface_msgs.msg import ira_dets
from wm_od_interface_msgs.msg import KeypointDetection
from wm_od_interface_msgs.msg import KeypointDetections

from models import SmallStackedHourglass
from models import LargeStackedHourglass

class KeypointDetectorNode(object):
    ''' This node uses a stacked hourglass network implemented in pytorch to
    find the locations of keypoints in an image.
    '''
    def __init__(self):
        rospy.init_node('keypoint_detector_node')
        self.heatmap_pub = rospy.Publisher('pose_estimator/heatmap', Image,
                                           queue_size=1)
        self.heatmap_test_pub = rospy.Publisher('pose_estimator/heatmap_test',
                                                Image, queue_size=1)
        self.keypoints_pub = rospy.Publisher('pose_estimator/keypoints',
                                             KeypointDetections, queue_size=5)

        self.img_keypoints_pub = rospy.Publisher('pose_estimator/img_keypoints',
                                             KeypointDetections, queue_size=50)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        rospy.loginfo("Using %s to run stacked hourglass", self.device)

        num_hourglasses = rospy.get_param('~num_hourglasses', 4)
        hourglass_channels = rospy.get_param('~hourglass_channels', 256)
        self.img_size = rospy.get_param('~img_size', 256)
        self.detection_thresh = rospy.get_param('~detection_threshold', 0.01)
        

        rospack = rospkg.RosPack()
        model_base_path = join(rospack.get_path('rcta_object_pose_detection'),
                               'models',
                               'keypoint_localization')

        num_keypoints_file_path = rospy.get_param('~num_keypoints_file',
                                                  'num_keypoints.txt')
        num_keypoints_file_path = join(model_base_path, num_keypoints_file_path)
        model_path = rospy.get_param('~model_path', "keypoints.pt")
        model_path = join(model_base_path, model_path)

        self.keypoints_indices = dict()
        start_index = 0
        
        with open(num_keypoints_file_path, 'r') as num_keypoints_file:
            for line in num_keypoints_file:
                split = line.split(' ')
                if len(split) == 2:
                    self.keypoints_indices[split[0]] = \
                        (start_index, start_index + int(split[1]))
                    start_index += int(split[1])
        print "Keypoint indices:", self.keypoints_indices


        model_type = rospy.get_param('~model_type', 'small')
        if model_type == 'small':
            self.model = SmallStackedHourglass(
                num_hg=num_hourglasses,
                hg_channels=hourglass_channels,
                out_channels=start_index)
        elif model_type == 'large':
            self.model = LargeStackedHourglass(out_channels=start_index)
        else:
            rospy.logerr("%s is an invalid model type", model_type)
            return
        self.model.to(self.device)

        self.model.load_state_dict(torch.load(model_path, map_location=self.device)['stacked_hg'])
        self.model.eval()
        rospy.loginfo("Loaded model")
        self.bridge = CvBridge()

        detections_sub = message_filters.Subscriber('detected_objects',
                                                    ira_dets)
        image_sub = message_filters.Subscriber('/image', Image)
        combined_sub = message_filters.TimeSynchronizer(
            [detections_sub, image_sub], 100)
        combined_sub.registerCallback(self.detect_keypoints)
        rospy.loginfo("Spinning")
        rospy.spin()

    def detect_keypoints(self, detections_msg, image_msg):
        rospy.loginfo("Got %d detections in a %d x %d image",
                      len(detections_msg.dets),
                      image_msg.width,
                      image_msg.height)

        if not detections_msg.dets:
            return

        before_time = time.clock()
        try:
            image = self.bridge.imgmsg_to_cv2(image_msg, "32FC3")
        except CvBridgeError as error:
            rospy.logerr(error)
            return
        output_image = np.copy(image)
        output_image = output_image.astype(np.uint8)

        # Rescale image to be in range [0, 1]
        image /= 255.0

	# Rescale the image to be in range [-1, 1]
	image = 2 * (image - 0.5)


        patches, bounds = self.get_patches_and_bounds(detections_msg.dets,
                                                      image)

        pred_keypoints = self.get_keypoints(patches)

        keypoint_detections = KeypointDetections()
        keypoint_detections.header = image_msg.header
        
        img_keypoint_detections = KeypointDetections()
        img_keypoint_detections.header = image_msg.header
        

        for i, detection in enumerate(detections_msg.dets):
            detection_msg = KeypointDetection()
            detection_msg.obj_name = detection.obj_name
            predictions = pred_keypoints[i, :, :, :]
            
            img_detection_msg = KeypointDetection()
            img_detection_msg.obj_name = detection.obj_name

            if detection.obj_name not in self.keypoints_indices:
                keypoint_detections.detections.append(detection_msg)
                img_keypoint_detections.detections.append(img_detection_msg)
                continue


            for j in range(self.keypoints_indices[detection.obj_name][0],
                           self.keypoints_indices[detection.obj_name][1]):
                coords = np.unravel_index(np.argmax(predictions[j]),
                                          predictions[j, :, :].shape)
                detection_msg.x.append(coords[1])
                detection_msg.y.append(coords[0])
                detection_msg.probabilities.append(predictions[j,
                                                               coords[0],
                                                               coords[1]])

                

                img_coords = [0, 0]
                img_coords[0] = bounds[i][0] + \
                                int(1.0 * coords[1] / predictions.shape[-2] * \
                                    (bounds[i][1] - bounds[i][0]) + 0.5)
                img_coords[1] = bounds[i][2] + \
                                int(1.0 * coords[0] / predictions.shape[-1] * \
                                    (bounds[i][3] - bounds[i][2]) + 0.5)

                img_detection_msg.x.append(img_coords[0])
                img_detection_msg.y.append(img_coords[1])
                img_detection_msg.probabilities.append(predictions[j,
                                                                   coords[0],
                                                                   coords[1]])

                # Only draw the keypoint if it is more probable than the
                # threshold
                if predictions[j, coords[0], coords[1]] < self.detection_thresh:
                    continue

                cv2.circle(output_image, (img_coords[0], img_coords[1]), 5,
                           (255, 0, 0), thickness=-1)
            keypoint_detections.detections.append(detection_msg)
            img_keypoint_detections.detections.append(img_detection_msg)


        self.keypoints_pub.publish(keypoint_detections)
        self.img_keypoints_pub.publish(img_keypoint_detections)


        image_msg = self.bridge.cv2_to_imgmsg(output_image, encoding="rgb8")
        self.heatmap_pub.publish(image_msg)

        rospy.loginfo("Found keypoints for %d objects in %f seconds",
                      len(detections_msg.dets), time.clock() - before_time)


    def get_patches_and_bounds(self, detections, image):
        ''' Uses the detections to get the patches of the image that contain
        the detected objects.
        '''
        patches = np.zeros((len(detections), 3, self.img_size, self.img_size))
        bounds = []

        for i, detection in enumerate(detections):
            x_min = int(floor(detection.bbox.points[0].x))
            y_min = int(floor(detection.bbox.points[0].y))

            x_max = int(ceil(detection.bbox.points[2].x))
            y_max = int(ceil(detection.bbox.points[2].y))

            # Increases the size of the patch to ensure the entire
            # object is included
            width = x_max - x_min
            height = y_max - y_min
            x_min = max(0, x_min - width / 6)
            x_max = min(image.shape[1], x_max + width / 6)
            y_min = max(0, y_min - height / 6)
            y_max = min(image.shape[0], y_max + height / 6)

            patch = image[y_min:y_max, x_min:x_max, :]

            resized_patch = cv2.resize(patch, (self.img_size, self.img_size))

            resized_patch = np.moveaxis(resized_patch, 2, 0)
            patches[i, :, : :] = resized_patch
            bounds.append([x_min, x_max, y_min, y_max])
        return patches, bounds

    def get_keypoints(self, patches):
        ''' Runs the images through the network
        '''
        patches_tensor = torch.from_numpy(patches)
        patches_tensor = patches_tensor.to(self.device, dtype=torch.float32)
        with torch.no_grad():
            heatmaps = self.model(patches_tensor)
        # The network returns a list of the outputs of all of the hourglasses
        # in the stack.  We want the output of the final hourglass
        return heatmaps[-1].cpu().numpy()

    def generate_heatmap_grid(self, keypoints, object_type):
        ''' Generates a grid of heatmaps for a single detection
        '''
        num_images = self.keypoints_indices[object_type][1] - \
            self.keypoints_indices[object_type][0]

        grid_size = int(ceil(sqrt(num_images)))
        combined_keypoints = np.zeros((64*grid_size, 64*grid_size),
                                      dtype=np.float32)
        for i in range(grid_size):
            for j in range(grid_size):
                index = i * grid_size + j + self.keypoints_indices[object_type][0]
                if index >= self.keypoints_indices[object_type][1]:
                    continue
                print keypoints[index].shape
                combined_keypoints[i*64:(i+1)*64, j*64:(j+1)*64] = keypoints[index]
        return combined_keypoints

if __name__ == '__main__':
    KeypointDetectorNode()
