// Copyright 2026 AeroStrike
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "aerostrike_pkg/observation_builder_core.hpp"

#include <chrono>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <geometry_msgs/msg/point_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>

namespace aerostrike_pkg
{
namespace
{
using geometry_msgs::msg::PointStamped;
using nav_msgs::msg::Odometry;
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

class ObservationBuilder final : public rclcpp::Node
{
public:
  ObservationBuilder()
  : Node("observation_builder")
  {
    const auto odometry_topic =
      declare_string_parameter(*this, "odometry_topic", "/aerostrike/odom");
    const auto ray_distances_topic = declare_string_parameter(
      *this, "ray_distances_topic", "/aerostrike/ray_distances");
    const auto goal_topic = declare_string_parameter(*this, "goal_topic", "/aerostrike/goal");
    const auto previous_action_topic = declare_string_parameter(
      *this, "previous_action_topic", "/aerostrike/policy_action");
    const auto observation_topic = declare_string_parameter(
      *this, "observation_topic", "/aerostrike/policy_observation");
    const double publish_rate_hz = declare_double_parameter(*this, "publish_rate_hz", 50.0);

    config_.ray_count = static_cast<std::size_t>(declare_int_parameter(*this, "ray_count", 24));
    config_.ray_min_range_m = declare_double_parameter(*this, "ray_min_range_m", 0.2);
    config_.ray_max_range_m = declare_double_parameter(*this, "ray_max_range_m", 10.0);
    config_.goal_distance_normalizer_m = declare_double_parameter(
      *this, "goal_distance_normalizer_m", 20.0);
    config_.max_height_m = declare_double_parameter(*this, "max_height_m", 4.0);
    config_.ray_distances_are_normalized = declare_bool_parameter(
      *this, "ray_distances_are_normalized", false);
    config_.twist_is_body_frame = declare_bool_parameter(*this, "twist_is_body_frame", true);

    if (publish_rate_hz <= 0.0) {
      throw std::invalid_argument("publish_rate_hz must be positive");
    }

    observation_pub_ = create_publisher<Float32MultiArray>(
      observation_topic,
      rclcpp::SensorDataQoS());
    odometry_sub_ = create_subscription<Odometry>(
      odometry_topic,
      rclcpp::SensorDataQoS(),
      [this](const Odometry::SharedPtr msg) {handle_odometry(msg);});
    ray_distances_sub_ = create_subscription<Float32MultiArray>(
      ray_distances_topic,
      rclcpp::SensorDataQoS(),
      [this](const Float32MultiArray::SharedPtr msg) {handle_ray_distances(msg);});
    goal_sub_ = create_subscription<PointStamped>(
      goal_topic,
      rclcpp::QoS(1).reliable().transient_local(),
      [this](const PointStamped::SharedPtr msg) {handle_goal(msg);});
    previous_action_sub_ = create_subscription<Float32MultiArray>(
      previous_action_topic,
      rclcpp::QoS(1).reliable(),
      [this](const Float32MultiArray::SharedPtr msg) {handle_previous_action(msg);});

    timer_ = create_wall_timer(
      std::chrono::duration<double>(1.0 / publish_rate_hz),
      [this]() {publish_observation();});

    RCLCPP_INFO(
      get_logger(),
      "observation_builder ready: %s + %s + %s -> %s[%zu] at %.2f Hz",
      odometry_topic.c_str(),
      ray_distances_topic.c_str(),
      goal_topic.c_str(),
      observation_topic.c_str(),
      expected_observation_size(),
      publish_rate_hz);
  }

private:
  void handle_odometry(const Odometry::SharedPtr msg)
  {
    ObservationBuilderInput next = latest_input_.value_or(ObservationBuilderInput{});
    next.position_w = Vector3{
      msg->pose.pose.position.x,
      msg->pose.pose.position.y,
      msg->pose.pose.position.z,
    };
    next.orientation_wb = Quaternion{
      msg->pose.pose.orientation.x,
      msg->pose.pose.orientation.y,
      msg->pose.pose.orientation.z,
      msg->pose.pose.orientation.w,
    };
    next.linear_velocity = Vector3{
      msg->twist.twist.linear.x,
      msg->twist.twist.linear.y,
      msg->twist.twist.linear.z,
    };
    next.angular_velocity = Vector3{
      msg->twist.twist.angular.x,
      msg->twist.twist.angular.y,
      msg->twist.twist.angular.z,
    };
    latest_input_ = next;
    have_odometry_ = true;
  }

  void handle_ray_distances(const Float32MultiArray::SharedPtr msg)
  {
    if (msg->data.size() != config_.ray_count) {
      RCLCPP_WARN_THROTTLE(
        get_logger(),
        *get_clock(),
        2000,
        "Ignoring ray distance message with %zu values; expected %zu",
        msg->data.size(),
        config_.ray_count);
      return;
    }

    ObservationBuilderInput next = latest_input_.value_or(ObservationBuilderInput{});
    next.ray_distances = msg->data;
    latest_input_ = next;
    have_ray_distances_ = true;
  }

  void handle_goal(const PointStamped::SharedPtr msg)
  {
    ObservationBuilderInput next = latest_input_.value_or(ObservationBuilderInput{});
    next.goal_w = Vector3{msg->point.x, msg->point.y, msg->point.z};
    latest_input_ = next;
    have_goal_ = true;
  }

  void handle_previous_action(const Float32MultiArray::SharedPtr msg)
  {
    if (msg->data.size() != 3U) {
      RCLCPP_WARN_THROTTLE(
        get_logger(),
        *get_clock(),
        2000,
        "Ignoring previous action message with %zu values; expected 3",
        msg->data.size());
      return;
    }

    ObservationBuilderInput next = latest_input_.value_or(ObservationBuilderInput{});
    next.previous_action = msg->data;
    latest_input_ = next;
  }

  void publish_observation()
  {
    if (!have_odometry_ || !have_ray_distances_ || !have_goal_ || !latest_input_.has_value()) {
      RCLCPP_WARN_THROTTLE(
        get_logger(),
        *get_clock(),
        2000,
        "Waiting for odometry=%s rays=%s goal=%s before publishing policy observations",
        have_odometry_ ? "ready" : "missing",
        have_ray_distances_ ? "ready" : "missing",
        have_goal_ ? "ready" : "missing");
      return;
    }

    try {
      Float32MultiArray msg;
      msg.data = build_policy_observation(config_, latest_input_.value());
      observation_pub_->publish(msg);
    } catch (const std::exception & error) {
      RCLCPP_ERROR_THROTTLE(
        get_logger(),
        *get_clock(),
        2000,
        "Failed to build policy observation: %s",
        error.what());
    }
  }

  std::size_t expected_observation_size() const
  {
    return config_.ray_count + 17U;
  }

  ObservationBuilderConfig config_;
  std::optional<ObservationBuilderInput> latest_input_;
  bool have_odometry_{false};
  bool have_ray_distances_{false};
  bool have_goal_{false};

  rclcpp::Publisher<Float32MultiArray>::SharedPtr observation_pub_;
  rclcpp::Subscription<Odometry>::SharedPtr odometry_sub_;
  rclcpp::Subscription<Float32MultiArray>::SharedPtr ray_distances_sub_;
  rclcpp::Subscription<PointStamped>::SharedPtr goal_sub_;
  rclcpp::Subscription<Float32MultiArray>::SharedPtr previous_action_sub_;
  rclcpp::TimerBase::SharedPtr timer_;
};
}  // namespace aerostrike_pkg

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  try {
    rclcpp::spin(std::make_shared<aerostrike_pkg::ObservationBuilder>());
  } catch (const std::exception & error) {
    RCLCPP_FATAL(rclcpp::get_logger("observation_builder"), "%s", error.what());
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}
