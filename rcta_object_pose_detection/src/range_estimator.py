#!/usr/bin/env python
""" This node calculates the 3D position of an object that was detected by vision
This node combines the point cloud and the bounding box to get the location.
"""
import rospy

import numpy as np
import tf

from timeit import default_timer
import message_filters
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import PointField
from geometry_msgs.msg import Polygon
from geometry_msgs.msg import Point32

from wm_od_interface_msgs.msg import ira_dets

from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray


from rcta_perception_msgs.msg import DetectedObject
from rcta_perception_msgs.msg import DetectedObjectArray

from rcta_worldmodel_msgs.msg import Object
from rcta_worldmodel_msgs.srv import AddObjects, AddObjectsRequest, AddObjectsResponse
from rcta_worldmodel_msgs.srv import AddClass, AddClassRequest, AddClassResponse
from rcta_worldmodel_msgs.srv import LookupClasses, LookupClassesRequest, LookupClassesResponse

import random

class RangeEstimator(object):
    def __init__(self):
        self.listener = tf.TransformListener()

        self.fx = None

        self.show_debug = rospy.get_param('~show_debug', True)

        if self.show_debug:
            self.marker_pub = rospy.Publisher('~markers', MarkerArray, queue_size=1)
            self.synced_cloud = rospy.Publisher('~synced_cloud', PointCloud2, queue_size=1)

        self.use_service = rospy.get_param('~use_service', True)
        self.use_class_service = rospy.get_param("~use_class_service", True)

        self.added_classes = dict()

        if not self.use_service:
            self.detected_object_publisher = rospy.Publisher('static_object_detections', DetectedObjectArray, queue_size=1)
        else:
            rospy.loginfo('Waiting for connection to the world model')
            rospy.wait_for_service('/add_objects')
            self.add_objects_srv_proxy = rospy.ServiceProxy('/add_objects', AddObjects)            

        if self.use_class_service:
            rospy.wait_for_service('/add_class')
            rospy.wait_for_service('/lookup_classes')
            self.add_class_srv_proxy = rospy.ServiceProxy('/add_class', AddClass)
            self.lookup_classes_srv_proxy = rospy.ServiceProxy('/lookup_classes', LookupClasses)

        self.camera_info_topic = rospy.get_param("~camera_info_topic", "/camera/color/camera_info")
        self.detection_topic = rospy.get_param("~detection_topic", "detected_objects_in_image")
        self.pointcloud_topic = rospy.get_param("~pointcloud_topic", "fast_assembled_cloud_lowres")

        rospy.logwarn("Subscribing to %s", self.camera_info_topic)
        self.camerainfo_sub = rospy.Subscriber(self.camera_info_topic, \
                                               CameraInfo, \
                                               self.get_camera_info, \
                                               queue_size=1)
        
        detections_sub = message_filters.Subscriber(self.detection_topic, ira_dets)
        point_cloud_sub = message_filters.Subscriber(self.pointcloud_topic, PointCloud2)

        combined_sub = message_filters.ApproximateTimeSynchronizer(
            [detections_sub, point_cloud_sub], 100, 100.0)
        combined_sub.registerCallback(self.estimate_ranges)
        
        rospy.loginfo("Starting point_cloud_estimator")
        rospy.spin()

    def get_camera_info(self, info):
        # rospy.logwarn("The camera params in the bag file seem to be wrong so we are manually modifying them")
        self.fx = info.K[0]
        self.fy = info.K[4]
        self.cx = info.K[2]
        self.cy = info.K[5]
        # self.camerainfo_sub.unregister()

    def estimate_ranges(self, detections, point_cloud):
        if self.fx is None:
            rospy.logwarn("No camera params recieved")
            return

        start = default_timer()
        points_converted = np.fromstring(point_cloud.data, dtype='<f4')
        x = points_converted[0::4]
        y = points_converted[1::4]
        z = points_converted[2::4]

        points = np.array([x, y, z, np.ones(len(x))])
        try:
            (trans,rot) = self.listener.lookupTransform(detections.header.frame_id,  # Target frame
                                                        point_cloud.header.frame_id, # Source frame
                                                        point_cloud.header.stamp)    # Time

        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as error:
            rospy.logerr("Failed to get transform {}".format(error))
            return
        matrix = tf.transformations.quaternion_matrix(rot)
        matrix[:3, 3] = trans
        points = np.matmul(matrix, points)

        points = points[:, points[2, :] > 0]
        image_coords = np.array([self.cx + self.fx * points[0, :] / points[2, :],
                                 self.cy + self.fy * points[1, :] / points[2, :]])
        
        if self.show_debug:
            end = default_timer()
            rospy.loginfo("Time to image_coords: {:.3f}".format(end - start))
            start = default_timer()

        marker_array = MarkerArray()
        in_front_of_camera = (points[2, :] > 0)
        
        detected_object_array = DetectedObjectArray()
        detected_object_array.header = detections.header

        objects_to_add = AddObjectsRequest()
        objects_to_add.client = "range_estimator"

        for i, detection in enumerate(detections.dets):
            min_x = detection.bbox.points[0].x
            max_x = detection.bbox.points[1].x
            min_y = detection.bbox.points[0].y
            max_y = detection.bbox.points[2].y
            # min_x = 0
            # max_x = 2752
            # min_y = 0
            # max_y = 2200
            x_bounds = (image_coords[0, :] > min_x) & (image_coords[0, :] < max_x)
            y_bounds = (image_coords[1, :] > min_y) & (image_coords[1, :] < max_y)
            points_in_box = points[:3, x_bounds & y_bounds & in_front_of_camera]
            if points_in_box.shape[1] == 0: # Skips boxes that have no points in them 
                rospy.loginfo("Skipping {}: couldn't find any points".format(detection.obj_name.lower()))
                continue
            box_size = [1.0, 1.0, 1.0]

            min_distance = np.min(points_in_box[2, :])
            points_in_box = points_in_box[:, points_in_box[2, :] < min_distance + box_size[2]]
            box_center = np.mean(points_in_box, axis=1)
            # print box_center

            marker = Marker()
            marker.header = detection.header
            marker.id = i
            marker.type = 1
            marker.action = 0
            marker.pose.position.x = box_center[0]
            marker.pose.position.y = box_center[1]
            marker.pose.position.z = box_center[2]
            marker.pose.orientation.w = 1

            marker.scale.x = box_size[0]
            marker.scale.y = box_size[1]
            marker.scale.z = box_size[2]

            marker.color.r = 1
            marker.color.g = 0
            marker.color.b = 0
            marker.color.a = 0.75
            marker_array.markers.append(marker)

            if self.use_class_service:
                if detection.obj_name not in self.added_classes:
                    entered_classes = self.lookup_classes_srv_proxy(LookupClassesRequest())
                    for c in entered_classes.classes:
                        if c.name.lower() == detection.obj_name.lower():
                            self.added_classes[detection.obj_name] = c.id
                            break
                    if detection.obj_name not in self.added_classes:

                        rospy.loginfo("Adding class {}".format(detection.obj_name))
                        new_class = AddClassRequest()
                        new_class.client = 'range_estimator'
                        new_class.object_class.name = detection.obj_name
                        new_class.object_class.id = 0
                        new_class.object_class.color.r = random.random()
                        new_class.object_class.color.g = random.random()
                        new_class.object_class.color.b = random.random()
                        new_class.object_class.color.a = 1.0

                        returned_class = self.add_class_srv_proxy(new_class)
                        self.added_classes[detection.obj_name] = returned_class.object_class.id


            detection_3dof = DetectedObject()

            detection_3dof.pose.header = detection.header
            
            detection_3dof.pose.pose.position.x = box_center[0]
            detection_3dof.pose.pose.position.y = box_center[1]
            detection_3dof.pose.pose.position.z = box_center[2]
            detection_3dof.pose.pose.orientation.w = 1

            detection_3dof.classification.time = detection.header.stamp
            detection_3dof.classification.confidence = detection.confidence
            detection_3dof.classification.tracking_id = detection.id
            detection_3dof.classification.type.name = detection.obj_name
            try:
                detection_3dof.classification.type.id = self.added_classes[detection.obj_name]
            except Exception as e:
                # @TODO: assign a useful class ID when we aren't using the service
                detection_3dof.classification.type.id = 0

            detection_3dof.bounds.header = detection.header
            detection_3dof.bounds.header.frame_id = "" # Indicate that bounds are in object frame
            inner_point = Point32()
            inner_point.x = - box_size[0] / 2
            inner_point.y = - box_size[1] / 2
            inner_point.z = - box_size[2] / 2
            detection_3dof.bounds.polygon.points.append(inner_point)

            outer_point = Point32()
            outer_point.x = box_size[0] / 2
            outer_point.y = box_size[1] / 2
            outer_point.z = box_size[2] / 2
            detection_3dof.bounds.polygon.points.append(outer_point)
            detected_object_array.objects.append(detection_3dof)

            obj_msg = Object()
            obj_msg.base = detection_3dof
            #obj_msg.id = self.added_classes[detection.obj_name]
            obj_msg.name = detection.obj_name
            objects_to_add.objects.append(obj_msg)
            # rospy.loginfo(obj_msg)

        if not self.use_service:
            self.detected_object_publisher.publish(detected_object_array)
        else:
            self.add_objects_srv_proxy(objects_to_add)

        if self.show_debug:
            end = default_timer()
            rospy.loginfo("Time to get {} boxes: {:.3f}".format(len(detections.dets), end - start))
            self.marker_pub.publish(marker_array)

            point_cloud.header.stamp = detections.header.stamp
            self.synced_cloud.publish(point_cloud)

if __name__ == '__main__':
    rospy.init_node('range_estimator')
    RangeEstimator()
