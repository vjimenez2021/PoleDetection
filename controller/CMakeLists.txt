cmake_minimum_required(VERSION 3.8)
project(controller)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(PCL REQUIRED COMPONENTS segmentation kdtree search)

set(dependencies
  rclcpp
  sensor_msgs
  pcl_conversions
)

include_directories(
  include
  ${PCL_INCLUDE_DIRS}
)

# Link directories para PCL
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

# Executable
add_executable(pole_detector_main
  src/pole_detector_main.cpp
  src/pole_detector.cpp
)

ament_target_dependencies(pole_detector_main ${dependencies})
target_link_libraries(pole_detector_main
  ${PCL_LIBRARIES}
)

# Install
install(TARGETS
  pole_detector_main
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include
)

ament_package()