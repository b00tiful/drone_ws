cmake_minimum_required(VERSION 3.8)
project(aerostrike_pkg)

if(POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(yaml_cpp_vendor REQUIRED)

option(AEROSTRIKE_ENABLE_ONNXRUNTIME "Build policy_node with ONNX Runtime support" ON)

if(AEROSTRIKE_ENABLE_ONNXRUNTIME)
  list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
  find_package(ONNXRUNTIME QUIET)
endif()

add_library(observation_builder_core src/observation_builder_core.cpp)
target_compile_features(observation_builder_core PUBLIC cxx_std_17)
target_include_directories(observation_builder_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

add_library(command_adapter_core src/command_adapter_core.cpp)
target_compile_features(command_adapter_core PUBLIC cxx_std_17)
target_include_directories(command_adapter_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

add_library(metrics_logger_core src/metrics_logger_core.cpp)
target_compile_features(metrics_logger_core PUBLIC cxx_std_17)
target_include_directories(metrics_logger_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

add_executable(policy_node src/policy_node.cpp)
target_compile_features(policy_node PRIVATE cxx_std_17)
ament_target_dependencies(policy_node rclcpp std_msgs)
target_link_libraries(policy_node yaml-cpp)

add_executable(goal_publisher src/goal_publisher.cpp)
target_compile_features(goal_publisher PRIVATE cxx_std_17)
ament_target_dependencies(goal_publisher rclcpp geometry_msgs)

add_executable(observation_builder src/observation_builder.cpp)
target_compile_features(observation_builder PRIVATE cxx_std_17)
ament_target_dependencies(observation_builder rclcpp geometry_msgs nav_msgs std_msgs)
target_link_libraries(observation_builder observation_builder_core)

add_executable(command_adapter src/command_adapter.cpp)
target_compile_features(command_adapter PRIVATE cxx_std_17)
ament_target_dependencies(command_adapter rclcpp geometry_msgs std_msgs)
target_link_libraries(command_adapter command_adapter_core)

add_executable(metrics_logger src/metrics_logger.cpp)
target_compile_features(metrics_logger PRIVATE cxx_std_17)
ament_target_dependencies(metrics_logger rclcpp geometry_msgs nav_msgs std_msgs)
target_link_libraries(metrics_logger metrics_logger_core)

if(ONNXRUNTIME_FOUND)
  target_link_libraries(policy_node ONNXRuntime::ONNXRuntime)
  target_compile_definitions(policy_node PRIVATE AEROSTRIKE_HAS_ONNXRUNTIME=1)
  get_filename_component(ONNXRUNTIME_LIBRARY_DIR "${ONNXRUNTIME_LIBRARY}" DIRECTORY)
  set_target_properties(policy_node PROPERTIES
    BUILD_RPATH "${ONNXRUNTIME_LIBRARY_DIR}"
    INSTALL_RPATH "${ONNXRUNTIME_LIBRARY_DIR}"
  )
  add_executable(policy_inference_smoke test/policy_inference_smoke.cpp)
  target_compile_features(policy_inference_smoke PRIVATE cxx_std_17)
  target_link_libraries(policy_inference_smoke ONNXRuntime::ONNXRuntime)
  set_target_properties(policy_inference_smoke PROPERTIES
    BUILD_RPATH "${ONNXRUNTIME_LIBRARY_DIR}"
    INSTALL_RPATH "${ONNXRUNTIME_LIBRARY_DIR}"
  )
else()
  target_compile_definitions(policy_node PRIVATE AEROSTRIKE_HAS_ONNXRUNTIME=0)
  message(WARNING
    "ONNX Runtime C++ SDK not found; policy_node will build but inference is disabled. "
    "Set ONNXRUNTIME_ROOT to the SDK prefix to enable runtime inference."
  )
endif()

install(TARGETS
  policy_node
  goal_publisher
  observation_builder
  observation_builder_core
  command_adapter
  command_adapter_core
  metrics_logger
  metrics_logger_core
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)
if(TARGET policy_inference_smoke)
  install(TARGETS policy_inference_smoke DESTINATION lib/${PROJECT_NAME})
endif()
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY config launch DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()

  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(observation_builder_core_test test/observation_builder_core_test.cpp)
  target_link_libraries(observation_builder_core_test observation_builder_core)

  ament_add_gtest(command_adapter_core_test test/command_adapter_core_test.cpp)
  target_link_libraries(command_adapter_core_test command_adapter_core)

  ament_add_gtest(metrics_logger_core_test test/metrics_logger_core_test.cpp)
  target_link_libraries(metrics_logger_core_test metrics_logger_core)
endif()

ament_package()
