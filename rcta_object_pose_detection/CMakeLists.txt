cmake_minimum_required(VERSION 2.8.3)
project(rcta_object_pose_detection)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DEIGEN_NO_DEBUG")
SET(CMAKE_BUILD_TYPE "Release") # Set to Debug or Release

#IF(DEFINED ENV{ARM_ARCHITECTURE})
#   SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -O3 -march=native") # arm
#ELSE()
#  SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -O3 -msse4.2 -msse2 -msse") # intel architecture optimizations
#ENDIF()
SET(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -g") # gdb support for backtracing

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
  cv_bridge
  # pose refinement dependencies 12/07/17
  roscpp
  pcl_ros
  image_transport
  image_geometry
  wm_od_interface_msgs
  #pytorch
  )

# Pose refinement packages 12/07/17
#find_package(Eigen3 REQUIRED)
#set(OpenCV_DIR "~/Programs/opencv-2.4.9/build/")
find_package(OpenCV REQUIRED)
find_package(Boost REQUIRED COMPONENTS signals regex system thread filesystem)
find_package(PCL REQUIRED)

include_directories(include ${Boost_INCLUDE_DIR} ${catkin_INCLUDE_DIRS} ${PCL_INCLUDE_DIRS})
link_directories(${catkin_LIBRARY_DIRS} ${PCL_LIBRARY_DIRS})
link_libraries(${catkin_LIBRARIES} ${PCL_LIBRARIES})

# find_package(OpenMP REQUIRED)
# set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
# set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
# End pose refinement packages

# JF 2018-02-19: Commenting this out since python code is not in
# correct location (i.e., src/rcta_object_pose_detection)
# catkin_python_setup()

## Generate services in the 'srv' folder
# add_service_files(
#   FILES
#   ImageFrame.srv
#   ObjectLabel.srv
# )

## Generate added messages and services with any dependencies listed here
# generate_messages(
#   DEPENDENCIES
#   std_msgs
#   sensor_msgs
#   geometry_msgs
#   rcta_object_pose_detection
# )

###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if you package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(

#  INCLUDE_DIRS include
  LIBRARIES rcta_object_pose_detection
  CATKIN_DEPENDS rospy std_msgs message_runtime cv_bridge pcl_ros image_transport image_geometry
  DEPENDS #Eigen3 boost opencv PCL libqhull libvtk libvtk-qt

)

#catkin_package(
#  CATKIN_DEPENDS roscpp cv_bridge laser_assembler image_transport image_geometry #trooper_msgs
#  DEPENDS  Eigen3 boost opencv PCL libqhull libvtk libvtk-qt
#)



###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)


# compile libs and bins
#add_executable(obj_detection_xtion src/detect_pc_xtion.cpp)
#target_link_libraries(obj_detection_xtion ${OpenCV_LIBRARIES} ${Boost_LIBRARIES}) # ${VTK_LIBRARIES}


## Mark executable scripts (Python etc.) for installation
## in contrast to setup.py, you can choose the destination
install(PROGRAMS
  scripts/run_keypoint_localization.sh
  src/pose_optimizer.py
  src/keypoint_detector.py
  src/range_estimator.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY
    src/models
#    object_ros/objectPose/for_redistribution_files_only/objectPose
    DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
    )

## Mark executables and/or libraries for installation
#install(TARGETS obj_detection_xtion # ${PROJECT_NAME} ${PROJECT_NAME}_node
#  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
#)


## Mark other files for installation (e.g. launch and bag files, etc.)
install(FILES
  launch/camera.launch
  launch/rcta_object_pose_detection.launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)

install(DIRECTORY
    models
    DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
    )

