// Copyright 2026 AeroStrike
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#if AEROSTRIKE_HAS_ONNXRUNTIME
#include <onnxruntime_cxx_api.h>
#endif
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>

namespace aerostrike_pkg
{
namespace
{
using std_msgs::msg::Float32MultiArray;

std::string declare_string_parameter(
  rclcpp::Node & node, const std::string & name, const std::string & default_value)
{
  node.declare_parameter<std::string>(name, default_value);
  return node.get_parameter(name).as_string();
}

int declare_int_parameter(rclcpp::Node & node, const std::string & name, const int default_value)
{
  node.declare_parameter<int>(name, default_value);
  return static_cast<int>(node.get_parameter(name).as_int());
}

bool declare_bool_parameter(rclcpp::Node & node, const std::string & name, const bool default_value)
{
  node.declare_parameter<bool>(name, default_value);
  return node.get_parameter(name).as_bool();
}
}  // namespace

class PolicyNode final : public rclcpp::Node
{
public:
  PolicyNode()
  : Node("policy_node")
  {
    policy_path_ = declare_string_parameter(
      *this, "policy_path",
      "checkpoints/aerostrike_policy.onnx");
    metadata_path_ = declare_string_parameter(
      *this, "metadata_path", "checkpoints/aerostrike_policy.yaml");
    const auto observation_topic = declare_string_parameter(
      *this, "observation_topic", "/aerostrike/policy_observation");
    const auto action_topic = declare_string_parameter(
      *this, "action_topic", "/aerostrike/policy_action");
    expected_observation_size_ = declare_int_parameter(*this, "expected_observation_size", 41);
    expected_action_size_ = declare_int_parameter(*this, "expected_action_size", 3);
    publish_zero_on_error_ = declare_bool_parameter(*this, "publish_zero_on_error", false);

    action_pub_ = create_publisher<Float32MultiArray>(action_topic, rclcpp::QoS(1).reliable());
    observation_sub_ = create_subscription<Float32MultiArray>(
      observation_topic,
      rclcpp::SensorDataQoS(),
      [this](const Float32MultiArray::SharedPtr msg) {handle_observation(msg);});

    configure_runtime();

    RCLCPP_INFO(
      get_logger(),
      "policy_node ready: %s[%d] -> %s[%d]",
      observation_topic.c_str(),
      expected_observation_size_,
      action_topic.c_str(),
      expected_action_size_);
    RCLCPP_INFO(get_logger(), "policy metadata: %s", metadata_path_.c_str());
  }

private:
  void configure_runtime()
  {
    validate_metadata();

    if (expected_observation_size_ <= 0 || expected_action_size_ <= 0) {
      RCLCPP_ERROR(
        get_logger(),
        "Invalid policy dimensions: observation=%d action=%d",
        expected_observation_size_,
        expected_action_size_);
      runtime_ready_ = false;
      return;
    }

#if AEROSTRIKE_HAS_ONNXRUNTIME
    if (!std::filesystem::exists(policy_path_)) {
      RCLCPP_ERROR(get_logger(), "Policy file does not exist: %s", policy_path_.c_str());
      runtime_ready_ = false;
      return;
    }

    try {
      session_options_.SetIntraOpNumThreads(1);
      session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
      session_ = std::make_unique<Ort::Session>(env_, policy_path_.c_str(), session_options_);

      input_names_ = read_io_names(true);
      output_names_ = read_io_names(false);
      input_name_ptrs_ = make_name_ptrs(input_names_);
      output_name_ptrs_ = make_name_ptrs(output_names_);

      if (input_name_ptrs_.empty() || output_name_ptrs_.empty()) {
        RCLCPP_ERROR(get_logger(), "ONNX policy must expose at least one input and one output");
        runtime_ready_ = false;
        return;
      }

      runtime_ready_ = true;
      RCLCPP_INFO(get_logger(), "ONNX Runtime loaded policy: %s", policy_path_.c_str());
    } catch (const Ort::Exception & error) {
      RCLCPP_ERROR(get_logger(), "Failed to initialize ONNX Runtime: %s", error.what());
      runtime_ready_ = false;
    }
#else
    RCLCPP_ERROR(
      get_logger(),
      "policy_node was built without ONNX Runtime; install the C++ SDK and rebuild with "
      "ONNXRUNTIME_ROOT set to enable inference.");
    runtime_ready_ = false;
#endif
  }

  void validate_metadata() const
  {
    if (!std::filesystem::exists(metadata_path_)) {
      RCLCPP_WARN(get_logger(), "Policy metadata file does not exist: %s", metadata_path_.c_str());
      return;
    }

    try {
      const auto metadata = YAML::LoadFile(metadata_path_);
      const int metadata_observation_size = metadata["observation"]["dimension"].as<int>();
      const int metadata_action_size = metadata["action"]["dimension"].as<int>();

      if (metadata_observation_size != expected_observation_size_ ||
        metadata_action_size != expected_action_size_)
      {
        RCLCPP_WARN(
          get_logger(),
          "Configured policy dimensions [%d -> %d] differ from metadata [%d -> %d]",
          expected_observation_size_,
          expected_action_size_,
          metadata_observation_size,
          metadata_action_size);
      }
    } catch (const YAML::Exception & error) {
      RCLCPP_WARN(
        get_logger(),
        "Could not parse policy metadata %s: %s",
        metadata_path_.c_str(),
        error.what());
    }
  }

  void handle_observation(const Float32MultiArray::SharedPtr msg)
  {
    if (msg->data.size() != static_cast<std::size_t>(expected_observation_size_)) {
      RCLCPP_WARN_THROTTLE(
        get_logger(),
        *get_clock(),
        2000,
        "Ignoring observation with %zu values; expected %d",
        msg->data.size(),
        expected_observation_size_);
      publish_zero_action_if_configured();
      return;
    }

    if (!runtime_ready_) {
      RCLCPP_WARN_THROTTLE(
        get_logger(),
        *get_clock(),
        5000,
        "Policy runtime is not ready; no action will be published");
      publish_zero_action_if_configured();
      return;
    }

    const auto action = run_policy(msg->data);
    if (action.size() != static_cast<std::size_t>(expected_action_size_)) {
      RCLCPP_ERROR_THROTTLE(
        get_logger(),
        *get_clock(),
        2000,
        "Policy returned %zu actions; expected %d",
        action.size(),
        expected_action_size_);
      publish_zero_action_if_configured();
      return;
    }

    publish_action(action);
  }

  std::vector<float> run_policy(const std::vector<float> & observation)
  {
#if AEROSTRIKE_HAS_ONNXRUNTIME
    try {
      std::array<int64_t, 2> input_shape{1, expected_observation_size_};
      auto input_tensor = Ort::Value::CreateTensor<float>(
        memory_info_,
        const_cast<float *>(observation.data()),
        observation.size(),
        input_shape.data(),
        input_shape.size());

      auto outputs = session_->Run(
        Ort::RunOptions{nullptr},
        input_name_ptrs_.data(),
        &input_tensor,
        1,
        output_name_ptrs_.data(),
        1);

      if (outputs.empty() || !outputs.front().IsTensor()) {
        return {};
      }

      auto & output = outputs.front();
      const auto shape_info = output.GetTensorTypeAndShapeInfo();
      const auto output_count = shape_info.GetElementCount();
      const auto * output_data = output.GetTensorData<float>();

      std::vector<float> action(output_data, output_data + output_count);
      for (auto & value : action) {
        value = std::clamp(value, -1.0F, 1.0F);
      }
      return action;
    } catch (const Ort::Exception & error) {
      RCLCPP_ERROR_THROTTLE(
        get_logger(),
        *get_clock(),
        2000,
        "ONNX inference failed: %s",
        error.what());
      return {};
    }
#else
    (void)observation;
    return {};
#endif
  }

  void publish_zero_action_if_configured()
  {
    if (!publish_zero_on_error_) {
      return;
    }

    publish_action(std::vector<float>(static_cast<std::size_t>(expected_action_size_), 0.0F));
  }

  void publish_action(const std::vector<float> & action)
  {
    Float32MultiArray msg;
    msg.data = action;
    action_pub_->publish(msg);
  }

#if AEROSTRIKE_HAS_ONNXRUNTIME
  std::vector<std::string> read_io_names(const bool read_inputs)
  {
    Ort::AllocatorWithDefaultOptions allocator;
    const auto count = read_inputs ? session_->GetInputCount() : session_->GetOutputCount();

    std::vector<std::string> names;
    names.reserve(count);
    for (std::size_t index = 0; index < count; ++index) {
      auto name = read_inputs ?
        session_->GetInputNameAllocated(index, allocator) :
        session_->GetOutputNameAllocated(index, allocator);
      names.emplace_back(name.get());
    }
    return names;
  }

  static std::vector<const char *> make_name_ptrs(const std::vector<std::string> & names)
  {
    std::vector<const char *> ptrs;
    ptrs.reserve(names.size());
    for (const auto & name : names) {
      ptrs.push_back(name.c_str());
    }
    return ptrs;
  }
#endif

  std::string policy_path_;
  std::string metadata_path_;
  int expected_observation_size_{0};
  int expected_action_size_{0};
  bool publish_zero_on_error_{false};
  bool runtime_ready_{false};

  rclcpp::Subscription<Float32MultiArray>::SharedPtr observation_sub_;
  rclcpp::Publisher<Float32MultiArray>::SharedPtr action_pub_;

#if AEROSTRIKE_HAS_ONNXRUNTIME
  Ort::Env env_{ORT_LOGGING_LEVEL_WARNING, "aerostrike_policy_node"};
  Ort::SessionOptions session_options_;
  Ort::MemoryInfo memory_info_{Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)};
  std::unique_ptr<Ort::Session> session_;
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
  std::vector<const char *> input_name_ptrs_;
  std::vector<const char *> output_name_ptrs_;
#endif
};
}  // namespace aerostrike_pkg

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<aerostrike_pkg::PolicyNode>());
  rclcpp::shutdown();
  return 0;
}
