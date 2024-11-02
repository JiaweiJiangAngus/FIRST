cmake_minimum_required(VERSION 3.5)
project(auto_shoot)


find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)
find_package(tdt_interface REQUIRED)

include_directories(include/auto_shoot)


file(GLOB SOURCES src/*.cpp)


add_executable(auto_shoot_node ${SOURCES})


ament_target_dependencies(auto_shoot_node rclcpp nav_msgs geometry_msgs sensor_msgs cv_bridge OpenCV tdt_interface)


install(TARGETS
  auto_shoot_node
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()

