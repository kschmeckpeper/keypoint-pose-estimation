#include "object_pose_keypoint_optimizer/object_pose_keypoint_optimizer.h"

KeypointOptimizer::KeypointOptimizer(ros::NodeHandle& n)
{
    n.param("method", optimization_method_, std::string("LevenbergMarquardt"));

    camera_intrinsics_ = cv::Mat_<float>::zeros(3, 3);
    std::string project_path = ros::package::getPath("object_pose_keypoint_optimizer");

    std::string path = project_path + "/models/";

    LoadModelFiles(path);
    
    // Prints contents of model files
    //for(auto& x : loaded_models_)
    //{
    //    std::cout << x.first << std::endl << x.second.mu << std::endl << std::endl;
    //}

    camera_params_sub_ = n.subscribe("/camera/rgb/camera_info",
                                     10,
                                     &KeypointOptimizer::CameraParamsCallback,
                                     this);

    message_filters::Subscriber<wm_od_interface_msgs::ira_dets> detections_sub(n, "detected_objects", 1);
    message_filters::Subscriber<wm_od_interface_msgs::KeypointDetections> keypoints_sub(n, "pose_estimator/img_keypoints", 1);
    typedef message_filters::sync_policies::ExactTime<wm_od_interface_msgs::ira_dets, wm_od_interface_msgs::KeypointDetections> sync_policy;
    
    message_filters::Synchronizer<sync_policy> sync(sync_policy(10),
                                                    detections_sub,
                                                    keypoints_sub);
    sync.registerCallback(boost::bind(&KeypointOptimizer::KeypointAndDetectionCallback,
                                      this,
                                      _1,
                                      _2));

    pose_publisher_ = n.advertise<geometry_msgs::PoseArray>("object_poses", 10);
    detected_object_pub_ = n.advertise<rcta_perception_msgs::DetectedObjectArray>("static_object_detections", 10);

    ros::spin();

}

void KeypointOptimizer::LoadModelFiles(std::string& path )
{
    boost::filesystem::path model_path(path);
    for (auto i = boost::filesystem::directory_iterator(model_path); i != boost::filesystem::directory_iterator(); i++)
    {
        if (!boost::filesystem::is_directory(i->path()))
        {
            ROS_INFO("Loading %s", i->path().filename().string().c_str());
            // Key is file name without extension
            std::string file_name = i->path().filename().string();
            std::string class_name = file_name.substr(0, file_name.find('.'));
            
            loaded_models_[class_name] = geometry::readModelFile(i->path().string());
        }
        else
            continue;
    }
    ROS_INFO("Loaded %d models", (int)(loaded_models_.size()));
}

void KeypointOptimizer::CameraParamsCallback(const sensor_msgs::CameraInfo::ConstPtr& msg)
{
    std::vector<double> dist_params = msg->D;
    distortion_params_ = cv::Mat_<double>(dist_params);

    camera_intrinsics_(0, 0) = msg->K[0];
    camera_intrinsics_(0, 2) = msg->K[2];
    camera_intrinsics_(1, 1) = msg->K[4];
    camera_intrinsics_(1, 2) = msg->K[5];
    camera_intrinsics_(2, 2) = 1;
}

Eigen::MatrixXd KeypointOptimizer::normalize_coords(const std::vector<int>& x,
                                                    const std::vector<int>& y)
{
    std::vector<cv::Point2f> input_points;
    for (int i = 0; i < x.size(); i++) {
        input_points.push_back(cv::Point2f(x[i], y[i]));
    }

    std::vector<cv::Point2f> output_points;
    cv::undistortPoints(input_points, output_points, camera_intrinsics_, distortion_params_);

    Eigen::MatrixXd normalized_coords(3, output_points.size());
    for (int i = 0; i < output_points.size(); i++) {
        
        normalized_coords(0, i) = output_points[i].x;
        normalized_coords(1, i) = output_points[i].y;
        normalized_coords(2, i) = 1;
    }
    return normalized_coords;
}

void KeypointOptimizer::KeypointAndDetectionCallback(const wm_od_interface_msgs::ira_dets::ConstPtr& detections,
                                                     const wm_od_interface_msgs::KeypointDetections::ConstPtr& keypoints)
{
    ros::Time start_time = ros::Time::now();
    if (camera_intrinsics_(2, 2) < 0.5) {
        ROS_WARN("KeypointOptimizer has not received CameraInfo. Returning.");
        return;
    }

    geometry_msgs::PoseArray poses;
    poses.header = detections->header;
    rcta_perception_msgs::DetectedObjectArray objects;
    objects.header = detections->header;
    
    
    for(int i = 0; i<detections->dets.size(); i++) {
        std::string object_class = detections->dets[i].obj_name;

        if (loaded_models_.count(object_class) == 0) {
            ROS_WARN("Pose estimator has no model for class %s", object_class.c_str());
            continue;
        }

        geometry::ObjectModelBasis model = loaded_models_.find(object_class)->second;
        
        if (object_class.compare(keypoints->detections[i].obj_name) != 0) {
            ROS_ERROR("Detection has class %s while keypoint has class %s. Exiting",
                      object_class.c_str(),
                      keypoints->detections[i].obj_name.c_str());
            return;
        }


        Eigen::MatrixXd normalized_coords = normalize_coords(keypoints->detections[i].x,
                                                             keypoints->detections[i].y);
        std::vector<float> probabilities = keypoints->detections[i].probabilities;
        Eigen::Map<Eigen::VectorXf> weight_floats(probabilities.data(), probabilities.size());
        Eigen::VectorXd weights = weight_floats.cast<double>();

        geometry::StructureResult result;
        if (optimization_method_.compare("LevenbergMarquardt") == 0) {
            result = geometry::optimizeStructureFromProjection(normalized_coords, model, weights);
        } else if (optimization_method_.compare("CoordinateDescent") == 0) {
            result = geometry::optimizeStructureFromProjectionCoordinateDescent(normalized_coords,
                                                                                model,
                                                                                weights);
        }

        geometry_msgs::Pose pose;
        pose.position.x = result.t[0];
        pose.position.y = result.t[1];
        pose.position.z = result.t[2];
        Eigen::Quaterniond q(result.R);
        pose.orientation.w = q.w();
        pose.orientation.x = q.x();
        pose.orientation.y = q.y();
        pose.orientation.z = q.z();
        std::cout << result.C << std::endl;

        rcta_perception_msgs::DetectedObject object;
        object.classification.time = detections->header.stamp;
        object.classification.confidence = detections->dets[i].confidence;
        object.classification.type.name = detections->dets[i].obj_name;
        object.classification.tracking_id = detections->dets[i].id;
        object.classification.type.id = detections->dets[i].id;
        // TODO Add color?

        object.pose.header = poses.header;
        object.pose.pose = pose;

        // TODO: Populate bounding box

        objects.objects.push_back(object);
        poses.poses.push_back(pose);
    }
    

    pose_publisher_.publish(poses);
    detected_object_pub_.publish(objects);
    
    ROS_INFO("Pose optimization took %f seconds for %d objects",
             (ros::Time::now() - start_time).toSec(),
             (int)(detections->dets.size()));
}


int main(int argc, char **argv) {
    ros::init(argc, argv, "object_pose_keypoint_optimizer");
    ros::NodeHandle n;

    KeypointOptimizer optimizer(n);
}
