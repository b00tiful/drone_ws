cmake_minimum_required(VERSION 3.8)
project(aerostrike_pkg)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(yaml_cpp_vendor REQUIRED)

option(AEROSTRIKE_ENABLE_ONNXRUNTIME "Build policy_node with ONNX Runtime support" ON)

if(AEROSTRIKE_ENABLE_ONNXRUNTIME)
  list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
  find_package(ONNXRuntime QUIET)
endif()

add_executable(policy_node src/policy_node.cpp)
target_compile_features(policy_node PRIVATE cxx_std_17)
ament_target_dependencies(policy_node rclcpp std_msgs)
target_link_libraries(policy_node yaml-cpp)

if(ONNXRuntime_FOUND)
  target_link_libraries(policy_node ONNXRuntime::ONNXRuntime)
  target_compile_definitions(policy_node PRIVATE AEROSTRIKE_HAS_ONNXRUNTIME=1)
else()
  target_compile_definitions(policy_node PRIVATE AEROSTRIKE_HAS_ONNXRUNTIME=0)
  message(WARNING "ONNX Runtime C++ SDK not found; policy_node will build but inference is disabled. Set ONNXRUNTIME_ROOT to the SDK prefix to enable runtime inference.")
endif()

install(TARGETS policy_node DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY config launch DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
