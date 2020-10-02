#!/usr/bin/env python
import rospy
import tf.transformations as transformations
from wm_od_interface_msgs.msg import ira_dets

from geometry_msgs.msg import Point
from geometry_msgs.msg import Pose
from geometry_msgs.msg import PoseArray

class BoundingBoxCreator(object):
    def __init__(self):
        ira_sub = rospy.Subscriber('/ira5/pose_req', ira_dets, self.calc_bounding_box)

        self.bounding_box_pub = rospy.Publisher('bounding_box', ira_dets, queue_size=10)

        self.pose_array_pub = rospy.Publisher('bounding_box_debug', PoseArray, queue_size=10)
        self.base_frame = rospy.get_param('~base_frame', "/map")

        rospy.loginfo("Inited")
        rospy.spin()

    def calc_bounding_box(self, msg):
        out_msg = ira_dets()
        out_msg.header = msg.header
        out_msg.n_dets = 0

        out_poses = PoseArray()
        out_poses.header = msg.header


        for detection in msg.dets:
            if detection.obj_name != "pelican_case":
                continue



            quaternion = [detection.pose.orientation.x,
                          detection.pose.orientation.y,
                          detection.pose.orientation.z,
                          detection.pose.orientation.w]
            matrix = transformations.quaternion_matrix(quaternion)

            handle_pos = Point()
            handle_pos.x = detection.pose.position.x + 0.075 * matrix[0, 1] + 0.025 * matrix[0, 2]
            handle_pos.y = detection.pose.position.y + 0.075 * matrix[1, 1] + 0.025 * matrix[1, 2]
            handle_pos.z = detection.pose.position.z + 0.075 * matrix[2, 1] + 0.025 * matrix[2, 2]

            bbox_1_x = detection.pose.position.x + 0
            bbox_1_y = detection.pose.position.y + 0
            bbox_1_z = detection.pose.position.z + 0

            bbox_2_x = handle_pos.x - 0.05 * matrix[0, 0] + 0.075 * matrix[0, 1] + 0.025 * matrix[0, 2]
            bbox_2_y = handle_pos.y - 0.05 * matrix[1, 0] + 0.075 * matrix[1, 1] + 0.025 * matrix[1, 2] 
            bbox_2_z = handle_pos.z - 0.05 * matrix[2, 0] + 0.075 * matrix[2, 1] + 0.025 * matrix[2, 2] 

            detection.bbox.points[0].x = min(bbox_1_x, bbox_2_x)
            detection.bbox.points[0].y = min(bbox_1_y, bbox_2_y)
            detection.bbox.points[0].z = min(bbox_1_z, bbox_2_z)

            detection.bbox.points[1].x = max(bbox_1_x, bbox_2_x)
            detection.bbox.points[1].y = max(bbox_1_y, bbox_2_y)
            detection.bbox.points[1].z = max(bbox_1_z, bbox_2_z)

            detection.pose.position = handle_pos

            out_msg.dets.append(detection)
            out_poses.poses.append(detection.pose)

            test_pose_1 = Pose()
            test_pose_1.position = detection.bbox.points[0]
            test_pose_1.orientation.w = 1
            out_poses.poses.append(test_pose_1)
            test_pose_2 = Pose()
            test_pose_2.position = detection.bbox.points[1]
            test_pose_2.orientation.w = 1
            out_poses.poses.append(test_pose_2)


        self.pose_array_pub.publish(out_poses)
        self.bounding_box_pub.publish(out_msg)
        rospy.loginfo("Pubed bounding_box")

if __name__ == '__main__':
    rospy.init_node('bounding_box_creator')
    creator = BoundingBoxCreator()
