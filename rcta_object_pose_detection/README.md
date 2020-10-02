# rcta_object_pose_detection
## Overview
This package is responsible for localizing known objects.  The full pipeline takes in the object detections, locates semantic keypoints on them, and then optimizes the poses of the objects using known models of their shapes.
* Object detections - Currently come from py_faster_rcnn_ros
* Keypoint detections - Uses `keypoint_detector.py` in this package
* Keypoint Optimization - Uses the main node in object_pose_keypoint_optimizer

This code is based on work from [this](https://www.seas.upenn.edu/~pavlakos/projects/object3d/) paper.


## Relevant Files:
* Slides, old models, old datasets: [dropbox](https://www.dropbox.com/sh/ksllbfvv611rlxg/AAC64Veaikn8Z4xbXTt8qQv1a?dl=0)
* Object Detection Net: [Faster-RCNN](https://github.com/rbgirshick/py-faster-rcnn.git)

## Installation

### Pose Estimation using Stacked Hourglass Network
For pose estimation, we use a stacked hourglass network with an additional optimization step. 

The stacked hourglass network is implemented using pytorch.

#### PyTorch Installation

Install pytorch from `rcta_externals_src` if `pip install` does not work on your machine.


## Usage

### 1. Camera bringup
Start whatever camera you are using.  The pipeline only requires monocular rgb

### 2. Start pose estimation pipeline

```
$ roslaunch rcta_object_pose_detection rcta_object_pose_detection.launch
```

This will start the object detector, the keypoint estimator, and the keypoint pose optimizer. Make sure that the arguments in the launch file are correctly mapped to your camera name.


