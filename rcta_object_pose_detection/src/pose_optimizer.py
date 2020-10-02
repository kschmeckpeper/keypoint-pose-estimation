#!/usr/bin/env python
import rospy
import rospkg
import tf
import tf2_ros
import tf2_geometry_msgs
import message_filters

import os

import objectPose
import matlab

import numpy as np

from sensor_msgs.msg import CameraInfo
from std_msgs.msg import String
from std_msgs.msg import Float32MultiArray
from wm_od_interface_msgs.msg import *
from rcta_object_pose_detection.srv import *

from rcta_perception_msgs.msg import DetectedObject
from rcta_perception_msgs.msg import DetectedObjectArray

from geometry_msgs.msg import Pose
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import PoseArray
from geometry_msgs.msg import Polygon, Point32
from geometry_msgs.msg import TransformStamped

class pose_optimizer_node():

    def __init__(self):
        self.myPose = objectPose.initialize()

        rospy.init_node('pose_optimizer')

        self.goal_objects = []
        self.K = None

        rospack = rospkg.RosPack()
        self.model_path = os.path.join(rospack.get_path('rcta_object_pose_detection'), 'models', 'pose_optimizer')

        

        self.pub_ = rospy.Publisher("detected_object_pose", ira_dets, queue_size=1)
        self.pub_debug = rospy.Publisher("detected_object_pose_debug", PoseStamped, queue_size=1)
        self.pub_pre_icp_ = rospy.Publisher("before_icp_object_pose", PoseArray, queue_size=1)

        self.static_object_pub_ = rospy.Publisher("/static_object_detections", DetectedObjectArray, queue_size=10)
        
        self.refinement_req_pub_ = rospy.Publisher("/ira5/pose_req", ira_dets, queue_size=1)

        self.pose_response_ = []
        self.response_received_ = False
        rospy.Subscriber("/ira5/pose_response", ira_dets, self.poseRequestCb)

        self.camerainfo_sub = rospy.Subscriber("/camera/rgb/camera_info", CameraInfo, self.getCameraInfoCb, queue_size=1)

        detections_sub = message_filters.Subscriber('detected_objects', ira_dets)
        keypoints_sub = message_filters.Subscriber('pose_estimator/keypoints', KeypointDetections)
        combined_sub = message_filters.ApproximateTimeSynchronizer([detections_sub, keypoints_sub], 10, 0.1)
        combined_sub.registerCallback(self.detectionsAndKeypointsCb)

        self.keypoint_indices = {'gascan':(0, 10), 'cabinet':(10, 19), 'pelican_case':(19, 30)}

        rospy.loginfo("Launched node for pose optimization")
        rospy.spin()

    def detectionsAndKeypointsCb(self, detection_msg, keypoint_msg):
        print "Got detectionsAndKeypointsCb"
        if self.K is None:
            rospy.logwarn("The pose optimizer needs the camera params before it can calculate the poses")

        self.goal_objects = []
        scores = []
        locations = []
        for detection in keypoint_msg.detections:
            self.goal_objects.append(detection.obj_name)
            scores.append(detection.probabilities)
            locations.append([detection.x, detection.y])

        centers = []
        scales = []
        self.detections = []

        for detection in detection_msg.dets:
            x_center = (detection.bbox.points[0].x + detection.bbox.points[1].x) / 2
            y_center = (detection.bbox.points[0].y + detection.bbox.points[2].y) / 2
            width = detection.bbox.points[1].x - detection.bbox.points[0].x
            height = detection.bbox.points[2].y - detection.bbox.points[0].y
            self.detections.append(detection)
            centers.append([x_center, y_center])
            scales.append(max(width, height) / 200 * 1.3)


        if scores and len(scores) == len(centers):
            poses = self.poseOptimization(scores, locations, centers, scales)
            detected_object_array = DetectedObjectArray()
            detected_object_array.header = detection_msg.dets[0].header
            for i in range(len(poses)):
                detection_6dof = DetectedObject()

                detection_6dof.pose.header = detection_msg.dets[i].header
                detection_6dof.pose.pose = poses[i]

                detection_6dof.classification.time = detection_msg.dets[i].header.stamp
                detection_6dof.classification.confidence = detection_msg.dets[i].confidence
                detection_6dof.classification.type.name = detection_msg.dets[i].obj_name
                detection_6dof.classification.type.id = detection_msg.dets[i].id

                detection_6dof.bounds.header = detection_msg.dets[i].header
                inner_point = Point32()
                inner_point.x = poses[i].position.x - 0.1
                inner_point.y = poses[i].position.y - 0.1
                inner_point.z = poses[i].position.z - 0.1
                detection_6dof.bounds.polygon.points.append(inner_point)

                outer_point = Point32()
                outer_point.x = poses[i].position.x + 0.1
                outer_point.y = poses[i].position.y + 0.1
                outer_point.z = poses[i].position.z + 0.1
                detection_6dof.bounds.polygon.points.append(outer_point)
                detected_object_array.objects.append(detection_6dof)

            self.static_object_pub_.publish(detected_object_array)
        else:
            print "Not matched:", len(scores), len(centers)


    def poseRequestCb(self, data):
        self.pose_response_ = data
        self.response_received_ = True

        rospy.loginfo("Got optimized pose")
        if self.pose_response_.n_dets > 0:
            T_resp = [self.pose_response_.dets[0].pose.position.x,
                      self.pose_response_.dets[0].pose.position.y,
                      self.pose_response_.dets[0].pose.position.z]
            quat_resp = [self.pose_response_.dets[0].pose.orientation.x,
                     self.pose_response_.dets[0].pose.orientation.y,
                     self.pose_response_.dets[0].pose.orientation.z,
                     self.pose_response_.dets[0].pose.orientation.w]
        else:
            rospy.logwarn("Optimized pose has no detections")
            return

        #Publish the pose
        pose = Pose()
        pose.position.x = T_resp[0]
        pose.position.y = T_resp[1]
        pose.position.z = T_resp[2]
        pose.orientation.x = quat_resp[0]
        pose.orientation.y = quat_resp[1]
        pose.orientation.z = quat_resp[2]
        pose.orientation.w = quat_resp[3]

        detection_results = ira_dets()
        detector_out = []

        object_pose = ira_det()
        object_pose = self.detections[0]
        object_pose.pose = pose
        object_pose.obj_name = self.goal_objects[0]


        gascan_pose = TransformStamped()
        gascan_pose.header.frame_id = "camera_rgb_optical_frame"
        gascan_pose.header.stamp = rospy.Time()
        gascan_pose.transform.translation = pose.position
        gascan_pose.transform.rotation = pose.orientation

        #Publish the bbox points
        #Starting with top left width side open for nozzle
        width = 0.18
        length = 0.24
        height = 0.24

        bbox = []
        gascan_bbox = PoseStamped()
        gascan_bbox.header.frame_id = "object_frame"
        gascan_bbox.header.stamp = rospy.Time()

        # TODO: read in from mesh
        # TODO: Remove assumption that the origin is in the center of the object
        topleftback = Point32(x=-width/2, y=-length/2, z=height/2)
        toprightback = Point32(x=-width/2, y=length/2, z=height/2)
        toprightfront = Point32(x=width/2, y=length/2, z=height/2)
        topleftfront = Point32(x=width/2, y=-length/2, z=height/2)
        botleftback = Point32(x=-width/2, y=-length/2, z=-height/2)
        botrightback = Point32(x=-width/2, y=length/2, z=-height/2)
        botrightfront = Point32(x=width/2, y=length/2, z=-height/2)
        botleftfront = Point32(x=width/2, y=-length/2, z=-height/2)

        bbox_points = [topleftback, toprightback, toprightfront, topleftfront, botleftback, botrightback, botrightfront, botleftfront]

        for ind in range(0, 8):
            out = PoseStamped()
            gascan_bbox.pose.position = bbox_points[ind]
            gascan_bbox.pose.orientation.x = 0
            gascan_bbox.pose.orientation.y = 0
            gascan_bbox.pose.orientation.z = 0
            gascan_bbox.pose.orientation.w = 1
            out = tf2_geometry_msgs.do_transform_pose(gascan_bbox, gascan_pose)
            bbox.append(out.pose.position)

        object_pose.bbox.points = [bbox[0], bbox[1], bbox[2], bbox[3], bbox[7], bbox[4], bbox[0], bbox[1], bbox[5], bbox[4], bbox[6], bbox[7], bbox[5], bbox[6], bbox[2]]

        detector_out.append(object_pose)
        detection_results.dets = detector_out
        detection_results.n_dets = len(detector_out)
        self.pub_.publish(detection_results)

        #For debugging purpose
        pose_debug = PoseStamped()
        pose_debug.pose = pose
        pose_debug.header.frame_id = "camera_rgb_optical_frame"
        self.pub_debug.publish(pose_debug)


    def getCameraInfoCb(self, data):
        self.K = data.K;
        self.K = [self.K[0:3], self.K[3:6], self.K[6:9]]
        self.camerainfo_sub.unregister();


    def poseOptimization(self, scores, locations, centers, scales):
        poses = []
        for i in range(len(scores)):

            K = matlab.double(self.K)
            W_hp = matlab.double(locations[i])
            score = matlab.double(scores[i])
            scale = matlab.double([scales[i]])
            center = matlab.double(centers[i])

            S,R,T = self.myPose.objectPose(W_hp, score, center, scale, K, self.model_path, self.goal_objects[i], nargout=3)


            R = np.array(R)
            T = np.array(T)
            R2 = np.array([0, 0, 0])
            R3 = np.array([0, 0, 0, 1])
            R = np.column_stack((R, R2))
            R = np.row_stack((R, R3))
            quaternion = tf.transformations.quaternion_from_matrix(R)

            curr_time = rospy.Time.now()

            ira_msg = ira_det();
            ira_msg.header.stamp = curr_time
            ira_msg.obj_name = self.goal_objects[i]
            top_left = Point32()
            top_left.x = centers[i][0] - scales[i] * 100.
            top_left.y = centers[i][1] - scales[i] * 100.
            top_left.z = 1.
            bottom_right = Point32()
            bottom_right.x = centers[i][0] + scales[i] * 100.
            bottom_right.y = centers[i][1] + scales[i] * 100.
            bottom_right.z = 1.
            ira_msg.bbox.points = [top_left, bottom_right]
            ira_msg.pose = Pose()

            # Hack to fix the optimizer sometimes returning a pose behind the camera
            if T[2] < 0:
                T = -1 * T
            ira_msg.pose.position.x = T[0]
            ira_msg.pose.position.y = T[1]
            ira_msg.pose.position.z = T[2]
            ira_msg.pose.orientation.x = quaternion[0]
            ira_msg.pose.orientation.y = quaternion[1]
            ira_msg.pose.orientation.z = quaternion[2]
            ira_msg.pose.orientation.w = quaternion[3]

            ira_msg.confidence = 1

            ira_msg_list = ira_dets()
            ira_msg_list.n_dets = 0
            ira_msg_list.header.stamp = curr_time

            ira_msg_list.dets.append(ira_msg)
            ira_msg_list.n_dets += 1

            
            poses.append(ira_msg.pose)

            self.refinement_req_pub_.publish(ira_msg_list)

            # print(ira_msg_list)

            rospy.loginfo("Sent pose for refinement")

        pre_icp_poses = PoseArray()
        pre_icp_poses.header.frame_id = "camera_rgb_optical_frame"
        pre_icp_poses.header.stamp = rospy.Time.now()
        pre_icp_poses.poses = poses
        self.pub_pre_icp_.publish(pre_icp_poses)
        return poses


if __name__ == '__main__':
    try:
        pose_optimizer_node()
    except rospy.ROSInterruptException:
        pass

