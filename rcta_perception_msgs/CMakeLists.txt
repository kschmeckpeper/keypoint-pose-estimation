cmake_minimum_required(VERSION 2.8.3)
project(rcta_perception_msgs)

find_package(catkin REQUIRED 
  COMPONENTS
  roscpp
  message_generation
  std_msgs
  geometry_msgs
)

add_message_files(
  FILES
  Classification.msg
  DetectedObject.msg
  DetectedObjectArray.msg 
  ObjectClass.msg
  ObjectClassArray.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES rcta_perception_msgs
  CATKIN_DEPENDS message_runtime std_msgs geometry_msgs
#  DEPENDS system_lib
)
