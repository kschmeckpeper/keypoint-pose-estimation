# Pose Estimation from Keypoints

## Usage
Run 

```
roslaunch rcta_object_pose_detection rcta_object_pose_detection.launch
```
to start the pose estimation.

## Installation
For the keypoint base pose estimation, pytorch must be installed.  
I run with version 1.7.1.

Pip seems to only offers up to version 1.4 for python2.7, so the keypoint detector node is set to run with python 3.

### Optimization
The pose optimization depends on gtsam.  

gtsam does not appear to be able to be built with ```catkin_make```.

Instead, you should use ``` catkin_make_isolated``` or ```catkin build```.

