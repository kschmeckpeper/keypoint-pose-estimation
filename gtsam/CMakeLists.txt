cmake_minimum_required(VERSION 2.8.3)
project(gtsam)

###########################################################################
# Configure our GTSAM options before including directory


# Set default build type
set(CMAKE_BUILD_TYPE "Release" CACHE STRING "")

# Always build shared library
set(GTSAM_BUILD_STATIC_LIBRARY OFF CACHE BOOL "")
# Postfix release level on library name
set(GTSAM_BUILD_TYPE_POSTFIXES ON CACHE BOOL "")

# Do NOT build Matlab wrapper
set(GTSAM_BUILD_WRAP OFF CACHE BOOL "")
set(GTSAM_INSTALL_WRAP OFF CACHE BOOL "")

# Disable tests
set(GTSAM_BUILD_TESTS OFF CACHE BOOL "")

# Disable examples
set(GTSAM_BUILD_EXAMPLES OFF CACHE BOOL "")

set(GEOGRAPHICLIB_LIB_TYPE "SHARED" CACHE STRING "")

#We don't have MKL, though it might be good
set(GTSAM_WITH_EIGEN_MKL OFF CACHE BOOL "")
###########################################################################

set(GTSAM_USE_SYSTEM_EIGEN OFF CACHE BOOL "")

set(GTSAM_TYPEDEF_POINTS_TO_VECTORS OFF CACHE BOOL "")
set(GTSAM_ALLOW_DEPRECATED_SINCE_V4 ON CACHE BOOL "")

find_package(catkin REQUIRED)

include_directories(
  )

catkin_package(
#  INCLUDE_DIRS include
  DEPENDS
  )

#add_subdirectory(gtsam-3.1.0)
#add_subdirectory(gtsam-3.2.0)
#add_subdirectory(gtsam-3.2.1)
add_subdirectory(gtsam-4.0.0)

install(FILES package.xml DESTINATION share/gtsam)
