// Copyright 2026 AeroStrike
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#pragma once

#include <cstddef>
#include <limits>
#include <vector>

namespace aerostrike_pkg
{
struct MetricsVector3
{
  double x{0.0};
  double y{0.0};
  double z{0.0};
};

struct MetricsLoggerConfig
{
  std::size_t ray_count{24};
  double success_radius_m{0.75};
  double collision_radius_m{0.35};
  double proximity_radius_m{1.5};
};

struct MetricsSnapshot
{
  bool started{false};
  bool success{false};
  bool collision{false};
  bool timeout{false};
  bool in_proximity{false};
  bool current_collision{false};
  std::size_t odometry_samples{0};
  std::size_t ray_samples{0};
  std::size_t terminal_samples{0};
  std::size_t proximity_samples{0};
  std::size_t collision_samples{0};
  double run_duration_s{0.0};
  double average_speed_mps{0.0};
  double latest_speed_mps{0.0};
  double max_speed_mps{0.0};
  double final_goal_distance_m{std::numeric_limits<double>::infinity()};
  double min_goal_distance_m{std::numeric_limits<double>::infinity()};
  double latest_min_ray_distance_m{std::numeric_limits<double>::infinity()};
  double min_ray_distance_m{std::numeric_limits<double>::infinity()};
  double proximity_sample_ratio{0.0};
  double collision_sample_ratio{0.0};
  double proximity_time_s{0.0};
  double collision_time_s{0.0};
};

struct TerminalMetrics
{
  bool success{false};
  bool collision{false};
  bool timeout{false};
  double final_goal_distance_m{std::numeric_limits<double>::infinity()};
  double min_ray_distance_m{std::numeric_limits<double>::infinity()};
};

void validate_metrics_logger_config(const MetricsLoggerConfig & config);

double min_ray_distance_m(const std::vector<float> & ray_distances);

class MetricsAccumulator
{
public:
  void observe_ray_distances(
    const MetricsLoggerConfig & config,
    const std::vector<float> & ray_distances);

  void observe_odometry(
    const MetricsLoggerConfig & config,
    double timestamp_s,
    const MetricsVector3 & position_w,
    const MetricsVector3 & linear_velocity_mps,
    const MetricsVector3 & goal_w);

  void observe_terminal_metrics(const TerminalMetrics & terminal_metrics);

  MetricsSnapshot snapshot() const;

private:
  bool started_{false};
  bool success_{false};
  bool collision_{false};
  bool timeout_{false};
  bool terminal_metrics_received_{false};
  bool in_proximity_{false};
  bool current_collision_{false};
  std::size_t odometry_samples_{0};
  std::size_t ray_samples_{0};
  std::size_t terminal_samples_{0};
  std::size_t proximity_samples_{0};
  std::size_t collision_samples_{0};
  double start_time_s_{0.0};
  double last_time_s_{0.0};
  double last_speed_mps_{0.0};
  double speed_time_integral_{0.0};
  double latest_speed_mps_{0.0};
  double max_speed_mps_{0.0};
  double final_goal_distance_m_{std::numeric_limits<double>::infinity()};
  double min_goal_distance_m_{std::numeric_limits<double>::infinity()};
  double latest_min_ray_distance_m_{std::numeric_limits<double>::infinity()};
  double min_ray_distance_m_{std::numeric_limits<double>::infinity()};
  double proximity_time_s_{0.0};
  double collision_time_s_{0.0};
};
}  // namespace aerostrike_pkg
