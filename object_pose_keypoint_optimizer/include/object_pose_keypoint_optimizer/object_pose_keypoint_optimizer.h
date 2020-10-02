#ifndef OBJECT_POSE_KEYPOINT_OPTIMIZER_NODE
#define OBJECT_POSE_KEYPOINT_OPTIMIZER_NODE

#include <ros/ros.h>
#include <ros/package.h>

#include <Eigen/Core>

#include <boost/filesystem.hpp>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/eigen.hpp>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/sync_policies/exact_time.h>

#include <sensor_msgs/CameraInfo.h>
#include <geometry_msgs/PoseArray.h>

#include "wm_od_interface_msgs/ira_dets.h"
#include "wm_od_interface_msgs/KeypointDetections.h"


#include "rcta_perception_msgs/DetectedObject.h"
#include "rcta_perception_msgs/DetectedObjectArray.h"

#include "object_pose_keypoint_optimizer/geometry.h"

class KeypointOptimizer
{
  public:
    KeypointOptimizer(ros::NodeHandle& n);

  private:
    void KeypointAndDetectionCallback(const wm_od_interface_msgs::ira_dets::ConstPtr& detections,
                                      const wm_od_interface_msgs::KeypointDetections::ConstPtr& keypoints );

    void CameraParamsCallback(const sensor_msgs::CameraInfo::ConstPtr& msg);
    Eigen::MatrixXd normalize_coords(const std::vector<int>& x, const std::vector<int>& y);
    void LoadModelFiles(std::string& path);

    std::string optimization_method_;
    std::string model_path_;

    ros::Subscriber camera_params_sub_;
    sensor_msgs::CameraInfo::ConstPtr camera_params_;

    ros::Publisher pose_publisher_, detected_object_pub_;

    cv::Mat_<float> distortion_params_;
    cv::Mat_<float> camera_intrinsics_; 

    std::map<std::string, geometry::ObjectModelBasis> loaded_models_;
};


#endif // OBJECT_POSE_KEYPOINT_OPTIMIZER_NODE
