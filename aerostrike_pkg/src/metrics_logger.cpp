// Copyright 2026 AeroStrike
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "aerostrike_pkg/metrics_logger_core.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>

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

std::string declare_string_parameter(
  rclcpp::Node & node, const std::string & name, const std::string & default_value)
{
  node.declare_parameter<std::string>(name, default_value);
  return node.get_parameter(name).as_string();
}

MetricsVector3 point_to_vector(const PointStamped & msg)
{
  return MetricsVector3{msg.point.x, msg.point.y, msg.point.z};
}

MetricsVector3 position_to_vector(const Odometry & msg)
{
  return MetricsVector3{
    msg.pose.pose.position.x,
    msg.pose.pose.position.y,
    msg.pose.pose.position.z,
  };
}

MetricsVector3 linear_velocity_to_vector(const Odometry & msg)
{
  return MetricsVector3{
    msg.twist.twist.linear.x,
    msg.twist.twist.linear.y,
    msg.twist.twist.linear.z,
  };
}
}  // namespace

class MetricsLogger final : public rclcpp::Node
{
public:
  MetricsLogger()
  : Node("metrics_logger")
  {
    const auto odometry_topic =
      declare_string_parameter(*this, "odometry_topic", "/aerostrike/odom");
    const auto ray_distances_topic = declare_string_parameter(
      *this, "ray_distances_topic", "/aerostrike/ray_distances");
    const auto goal_topic = declare_string_parameter(*this, "goal_topic", "/aerostrike/goal");
    const auto terminal_metrics_topic = declare_string_parameter(
      *this, "terminal_metrics_topic", "/aerostrike/terminal_metrics");
    output_path_ = declare_string_parameter(*this, "output_path", "logs/ros_metrics/latest.csv");
    const double log_period_s = declare_double_parameter(*this, "log_period_s", 1.0);

    config_.ray_count = static_cast<std::size_t>(declare_int_parameter(*this, "ray_count", 24));
    config_.success_radius_m = declare_double_parameter(*this, "success_radius_m", 0.75);
    config_.collision_radius_m = declare_double_parameter(*this, "collision_radius_m", 0.35);
    config_.proximity_radius_m = declare_double_parameter(*this, "proximity_radius_m", 1.5);
    validate_metrics_logger_config(config_);
    if (log_period_s <= 0.0) {
      throw std::invalid_argument("log_period_s must be positive");
    }

    open_output_file();

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
    terminal_metrics_sub_ = create_subscription<Float32MultiArray>(
      terminal_metrics_topic,
      rclcpp::QoS(1).reliable(),
      [this](const Float32MultiArray::SharedPtr msg) {handle_terminal_metrics(msg);});

    timer_ = create_wall_timer(
      std::chrono::duration<double>(log_period_s),
      [this]() {write_periodic_summary();});

    RCLCPP_INFO(
      get_logger(),
      "metrics_logger ready: odom=%s rays=%s goal=%s terminal=%s output=%s period=%.2fs",
      odometry_topic.c_str(),
      ray_distances_topic.c_str(),
      goal_topic.c_str(),
      terminal_metrics_topic.c_str(),
      output_path_.empty() ? "<disabled>" : output_path_.c_str(),
      log_period_s);
  }

  ~MetricsLogger() override
  {
    try {
      write_snapshot("final");
      log_snapshot("final", metrics_.snapshot());
    } catch (const std::exception &) {
    }
  }

private:
  void open_output_file()
  {
    if (output_path_.empty()) {
      return;
    }

    const std::filesystem::path path(output_path_);
    if (!path.parent_path().empty()) {
      std::filesystem::create_directories(path.parent_path());
    }
    output_.open(path, std::ios::out | std::ios::trunc);
    if (!output_) {
      throw std::runtime_error("Failed to open metrics output file: " + output_path_);
    }

    output_
      << "write_time_s,event,run_duration_s,odometry_samples,ray_samples,success,collision,"
      << "timeout,in_proximity,current_collision,terminal_samples,"
      << "proximity_samples,collision_samples,"
      << "proximity_sample_ratio,collision_sample_ratio,proximity_time_s,collision_time_s,"
      << "average_speed_mps,latest_speed_mps,max_speed_mps,final_goal_distance_m,"
      << "min_goal_distance_m,latest_min_ray_distance_m,min_ray_distance_m\n";
  }

  void handle_goal(const PointStamped::SharedPtr msg)
  {
    latest_goal_w_ = point_to_vector(*msg);
  }

  void handle_ray_distances(const Float32MultiArray::SharedPtr msg)
  {
    try {
      metrics_.observe_ray_distances(config_, msg->data);
    } catch (const std::exception & error) {
      RCLCPP_WARN_THROTTLE(
        get_logger(),
        *get_clock(),
        2000,
        "Ignoring ray distance metrics sample: %s",
        error.what());
    }
  }

  void handle_odometry(const Odometry::SharedPtr msg)
  {
    if (!latest_goal_w_.has_value()) {
      RCLCPP_WARN_THROTTLE(
        get_logger(),
        *get_clock(),
        2000,
        "Waiting for goal before recording metrics");
      return;
    }

    try {
      metrics_.observe_odometry(
        config_,
        message_time_s(*msg),
        position_to_vector(*msg),
        linear_velocity_to_vector(*msg),
        latest_goal_w_.value());
    } catch (const std::exception & error) {
      RCLCPP_WARN_THROTTLE(
        get_logger(),
        *get_clock(),
        2000,
        "Ignoring odometry metrics sample: %s",
        error.what());
    }
  }

  void handle_terminal_metrics(const Float32MultiArray::SharedPtr msg)
  {
    if (msg->data.size() < 5U) {
      RCLCPP_WARN_THROTTLE(
        get_logger(),
        *get_clock(),
        2000,
        "Ignoring terminal metrics with %zu values; expected at least 5",
        msg->data.size());
      return;
    }

    metrics_.observe_terminal_metrics(
      TerminalMetrics{
        msg->data[0] > 0.5F,
        msg->data[1] > 0.5F,
        msg->data[2] > 0.5F,
        static_cast<double>(msg->data[3]),
        static_cast<double>(msg->data[4]),
      });
  }

  double message_time_s(const Odometry & msg)
  {
    if (msg.header.stamp.sec == 0 && msg.header.stamp.nanosec == 0U) {
      return get_clock()->now().seconds();
    }
    return rclcpp::Time(msg.header.stamp).seconds();
  }

  void write_periodic_summary()
  {
    const MetricsSnapshot snapshot = metrics_.snapshot();
    if (!snapshot.started) {
      return;
    }
    write_snapshot("periodic");
    log_snapshot("periodic", snapshot);
  }

  void write_snapshot(const std::string & event)
  {
    const MetricsSnapshot snapshot = metrics_.snapshot();
    if (!snapshot.started || !output_) {
      return;
    }

    output_ << std::fixed << std::setprecision(6)
            << get_clock()->now().seconds() << ','
            << event << ','
            << snapshot.run_duration_s << ','
            << snapshot.odometry_samples << ','
            << snapshot.ray_samples << ','
            << (snapshot.success ? 1 : 0) << ','
            << (snapshot.collision ? 1 : 0) << ','
            << (snapshot.timeout ? 1 : 0) << ','
            << (snapshot.in_proximity ? 1 : 0) << ','
            << (snapshot.current_collision ? 1 : 0) << ','
            << snapshot.terminal_samples << ','
            << snapshot.proximity_samples << ','
            << snapshot.collision_samples << ','
            << snapshot.proximity_sample_ratio << ','
            << snapshot.collision_sample_ratio << ','
            << snapshot.proximity_time_s << ','
            << snapshot.collision_time_s << ','
            << snapshot.average_speed_mps << ','
            << snapshot.latest_speed_mps << ','
            << snapshot.max_speed_mps << ','
            << snapshot.final_goal_distance_m << ','
            << snapshot.min_goal_distance_m << ','
            << snapshot.latest_min_ray_distance_m << ','
            << snapshot.min_ray_distance_m << '\n';
    output_.flush();
  }

  void log_snapshot(const std::string & event, const MetricsSnapshot & snapshot)
  {
    if (!snapshot.started) {
      return;
    }

    RCLCPP_INFO(
      get_logger(),
      "metrics[%s]: duration=%.2fs success=%s collision=%s timeout=%s avg_speed=%.3f m/s "
      "goal=%.3f m min_ray=%.3f m proximity_ratio=%.3f",
      event.c_str(),
      snapshot.run_duration_s,
      snapshot.success ? "yes" : "no",
      snapshot.collision ? "yes" : "no",
      snapshot.timeout ? "yes" : "no",
      snapshot.average_speed_mps,
      snapshot.final_goal_distance_m,
      snapshot.min_ray_distance_m,
      snapshot.proximity_sample_ratio);
  }

  MetricsLoggerConfig config_;
  MetricsAccumulator metrics_;
  std::optional<MetricsVector3> latest_goal_w_;
  std::string output_path_;
  std::ofstream output_;

  rclcpp::Subscription<Odometry>::SharedPtr odometry_sub_;
  rclcpp::Subscription<Float32MultiArray>::SharedPtr ray_distances_sub_;
  rclcpp::Subscription<PointStamped>::SharedPtr goal_sub_;
  rclcpp::Subscription<Float32MultiArray>::SharedPtr terminal_metrics_sub_;
  rclcpp::TimerBase::SharedPtr timer_;
};
}  // namespace aerostrike_pkg

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  try {
    rclcpp::spin(std::make_shared<aerostrike_pkg::MetricsLogger>());
  } catch (const std::exception & error) {
    RCLCPP_FATAL(rclcpp::get_logger("metrics_logger"), "%s", error.what());
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}
