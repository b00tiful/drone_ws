#include <chrono>
#include <memory>
#include <stdexcept>
#include <string>

#include <geometry_msgs/msg/point_stamped.hpp>
#include <rclcpp/rclcpp.hpp>

namespace aerostrike_pkg
{
namespace
{
double declare_double_parameter(
  rclcpp::Node & node, const std::string & name, const double default_value)
{
  node.declare_parameter<double>(name, default_value);
  return node.get_parameter(name).as_double();
}

std::string declare_string_parameter(
  rclcpp::Node & node, const std::string & name, const std::string & default_value)
{
  node.declare_parameter<std::string>(name, default_value);
  return node.get_parameter(name).as_string();
}
}  // namespace

class GoalPublisher final : public rclcpp::Node
{
public:
  GoalPublisher()
  : Node("goal_publisher")
  {
    const auto topic = declare_string_parameter(*this, "goal_topic", "/aerostrike/goal");
    frame_id_ = declare_string_parameter(*this, "frame_id", "world");
    goal_x_ = declare_double_parameter(*this, "goal_x", 0.0);
    goal_y_ = declare_double_parameter(*this, "goal_y", 5.0);
    goal_z_ = declare_double_parameter(*this, "goal_z", 1.5);
    const double publish_rate_hz = declare_double_parameter(*this, "publish_rate_hz", 2.0);

    if (publish_rate_hz <= 0.0) {
      throw std::invalid_argument("publish_rate_hz must be positive");
    }

    goal_pub_ = create_publisher<geometry_msgs::msg::PointStamped>(
      topic, rclcpp::QoS(1).reliable().transient_local());
    timer_ = create_wall_timer(
      std::chrono::duration<double>(1.0 / publish_rate_hz),
      [this]() { publish_goal(); });

    RCLCPP_INFO(
      get_logger(),
      "goal_publisher ready: %s goal=(%.3f, %.3f, %.3f) frame=%s rate=%.2f Hz",
      topic.c_str(),
      goal_x_,
      goal_y_,
      goal_z_,
      frame_id_.c_str(),
      publish_rate_hz);
  }

private:
  void publish_goal()
  {
    geometry_msgs::msg::PointStamped msg;
    msg.header.stamp = get_clock()->now();
    msg.header.frame_id = frame_id_;
    msg.point.x = goal_x_;
    msg.point.y = goal_y_;
    msg.point.z = goal_z_;
    goal_pub_->publish(msg);
  }

  rclcpp::Publisher<geometry_msgs::msg::PointStamped>::SharedPtr goal_pub_;
  rclcpp::TimerBase::SharedPtr timer_;
  std::string frame_id_;
  double goal_x_{0.0};
  double goal_y_{0.0};
  double goal_z_{0.0};
};
}  // namespace aerostrike_pkg

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  try {
    rclcpp::spin(std::make_shared<aerostrike_pkg::GoalPublisher>());
  } catch (const std::exception & error) {
    RCLCPP_FATAL(rclcpp::get_logger("goal_publisher"), "%s", error.what());
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}
