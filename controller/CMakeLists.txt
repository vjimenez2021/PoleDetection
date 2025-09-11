cmake_minimum_required(VERSION 3.8)
project(controller)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(PCL REQUIRED)

set(dependencies
  rclcpp
  sensor_msgs
  pcl_conversions
)

include_directories(
  include
  ${PCL_INCLUDE_DIRS}
)

# Library
add_library(${PROJECT_NAME} SHARED
  src/pole_detector.cpp
)
ament_target_dependencies(${PROJECT_NAME} ${dependencies})
target_link_libraries(${PROJECT_NAME} ${PCL_LIBRARIES})

# Executable
add_executable(pole_detector_main src/pole_detector_main.cpp)
ament_target_dependencies(pole_detector_main ${dependencies})
target_link_libraries(pole_detector_main ${PROJECT_NAME} ${PCL_LIBRARIES})

# Install
install(TARGETS
  ${PROJECT_NAME}
  pole_detector_main
  ARCHIVE DESTINATION lib/${PROJECT_NAME}
  LIBRARY DESTINATION lib/${PROJECT_NAME}
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include/
)

ament_package()
