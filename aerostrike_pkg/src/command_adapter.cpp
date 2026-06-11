// Copyright 2026 AeroStrike
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "aerostrike_pkg/command_adapter_core.hpp"

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <geometry_msgs/msg/twist_stamped.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>

namespace aerostrike_pkg
{
namespace
{
using geometry_msgs::msg::TwistStamped;
using std_msgs::msg::Float32MultiArray;

double declare_double_parameter(
  rclcpp::Node & node, const std::string & name, const double default_value)
{
  node.declare_parameter<double>(name, default_value);
  return node.get_parameter(name).as_double();
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

std::string declare_string_parameter(
  rclcpp::Node & node, const std::string & name, const std::string & default_value)
{
  node.declare_parameter<std::string>(name, default_value);
  return node.get_parameter(name).as_string();
}
}  // namespace

class CommandAdapter final : public rclcpp::Node
{
public:
  CommandAdapter()
  : Node("command_adapter")
  {
    const auto action_topic = declare_string_parameter(
      *this, "action_topic", "/aerostrike/policy_action");
    const auto command_topic = declare_string_parameter(
      *this, "command_topic", "/aerostrike/body_velocity_cmd");
    frame_id_ = declare_string_parameter(*this, "frame_id", "base_link");
    publish_zero_on_invalid_ = declare_bool_parameter(*this, "publish_zero_on_invalid", true);

    config_.action_size = static_cast<std::size_t>(
      declare_int_parameter(*this, "expected_action_size", 3));
    config_.horizontal_velocity_limit_mps = declare_double_parameter(
      *this, "horizontal_velocity_limit_mps", 5.0);
    config_.vertical_velocity_limit_mps = declare_double_parameter(
      *this, "vertical_velocity_limit_mps", 1.0);
    config_.clamp_normalized_action = declare_bool_parameter(
      *this, "clamp_normalized_action", true);
    validate_command_adapter_config(config_);

    command_pub_ = create_publisher<TwistStamped>(command_topic, rclcpp::QoS(1).reliable());
    action_sub_ = create_subscription<Float32MultiArray>(
      action_topic,
      rclcpp::QoS(1).reliable(),
      [this](const Float32MultiArray::SharedPtr msg) {handle_policy_action(msg);});

    RCLCPP_INFO(
      get_logger(),
      "command_adapter ready: %s[%zu] -> %s TwistStamped frame=%s limits=(%.2f, %.2f) m/s",
      action_topic.c_str(),
      config_.action_size,
      command_topic.c_str(),
      frame_id_.c_str(),
      config_.horizontal_velocity_limit_mps,
      config_.vertical_velocity_limit_mps);
  }

  ~CommandAdapter() override
  {
    if (command_pub_) {
      publish_command(VelocityCommand{});
    }
  }

private:
  void handle_policy_action(const Float32MultiArray::SharedPtr msg)
  {
    try {
      publish_command(scale_policy_action(config_, msg->data));
    } catch (const std::exception & error) {
      RCLCPP_WARN_THROTTLE(
        get_logger(),
        *get_clock(),
        2000,
        "Ignoring policy action: %s",
        error.what());
      if (publish_zero_on_invalid_) {
        publish_command(VelocityCommand{});
      }
    }
  }

  void publish_command(const VelocityCommand & command)
  {
    TwistStamped msg;
    msg.header.stamp = get_clock()->now();
    msg.header.frame_id = frame_id_;
    msg.twist.linear.x = command.vx_body_mps;
    msg.twist.linear.y = command.vy_body_mps;
    msg.twist.linear.z = command.vz_body_mps;
    command_pub_->publish(msg);
  }

  CommandAdapterConfig config_;
  bool publish_zero_on_invalid_{true};
  std::string frame_id_;

  rclcpp::Publisher<TwistStamped>::SharedPtr command_pub_;
  rclcpp::Subscription<Float32MultiArray>::SharedPtr action_sub_;
};
}  // namespace aerostrike_pkg

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  try {
    rclcpp::spin(std::make_shared<aerostrike_pkg::CommandAdapter>());
  } catch (const std::exception & error) {
    RCLCPP_FATAL(rclcpp::get_logger("command_adapter"), "%s", error.what());
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}
