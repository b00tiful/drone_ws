// Copyright 2026 AeroStrike
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "aerostrike_pkg/metrics_logger_core.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace aerostrike_pkg
{
namespace
{
bool is_finite_vector(const MetricsVector3 & value)
{
  return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
}

double norm(const MetricsVector3 & value)
{
  return std::sqrt(value.x * value.x + value.y * value.y + value.z * value.z);
}

double distance(const MetricsVector3 & lhs, const MetricsVector3 & rhs)
{
  return norm(MetricsVector3{lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z});
}
}  // namespace

void validate_metrics_logger_config(const MetricsLoggerConfig & config)
{
  if (config.ray_count == 0U) {
    throw std::invalid_argument("ray_count must be positive");
  }
  if (config.success_radius_m <= 0.0) {
    throw std::invalid_argument("success_radius_m must be positive");
  }
  if (config.collision_radius_m <= 0.0) {
    throw std::invalid_argument("collision_radius_m must be positive");
  }
  if (config.proximity_radius_m <= 0.0) {
    throw std::invalid_argument("proximity_radius_m must be positive");
  }
  if (config.collision_radius_m > config.proximity_radius_m) {
    throw std::invalid_argument("collision_radius_m must be <= proximity_radius_m");
  }
}

double min_ray_distance_m(const std::vector<float> & ray_distances)
{
  double nearest = std::numeric_limits<double>::infinity();
  for (const float value : ray_distances) {
    if (!std::isfinite(value)) {
      continue;
    }
    nearest = std::min(nearest, std::max(0.0, static_cast<double>(value)));
  }
  return nearest;
}

void MetricsAccumulator::observe_ray_distances(
  const MetricsLoggerConfig & config,
  const std::vector<float> & ray_distances)
{
  validate_metrics_logger_config(config);
  if (ray_distances.size() != config.ray_count) {
    throw std::invalid_argument(
            "ray_distances size " + std::to_string(ray_distances.size()) +
            " does not match configured ray_count " + std::to_string(config.ray_count));
  }

  latest_min_ray_distance_m_ = min_ray_distance_m(ray_distances);
  ++ray_samples_;

  current_collision_ = std::isfinite(latest_min_ray_distance_m_) &&
    latest_min_ray_distance_m_ <= config.collision_radius_m;
  in_proximity_ = std::isfinite(latest_min_ray_distance_m_) &&
    latest_min_ray_distance_m_ <= config.proximity_radius_m;

  if (std::isfinite(latest_min_ray_distance_m_)) {
    min_ray_distance_m_ = std::min(min_ray_distance_m_, latest_min_ray_distance_m_);
  }
  if (in_proximity_) {
    ++proximity_samples_;
  }
  if (current_collision_) {
    collision_ = true;
    ++collision_samples_;
  }
}

void MetricsAccumulator::observe_odometry(
  const MetricsLoggerConfig & config,
  const double timestamp_s,
  const MetricsVector3 & position_w,
  const MetricsVector3 & linear_velocity_mps,
  const MetricsVector3 & goal_w)
{
  validate_metrics_logger_config(config);
  if (!std::isfinite(timestamp_s)) {
    throw std::invalid_argument("timestamp_s must be finite");
  }
  if (!is_finite_vector(position_w)) {
    throw std::invalid_argument("position_w must contain finite values");
  }
  if (!is_finite_vector(linear_velocity_mps)) {
    throw std::invalid_argument("linear_velocity_mps must contain finite values");
  }
  if (!is_finite_vector(goal_w)) {
    throw std::invalid_argument("goal_w must contain finite values");
  }

  const double speed_mps = norm(linear_velocity_mps);
  const double goal_distance_m = distance(position_w, goal_w);

  if (!started_) {
    started_ = true;
    start_time_s_ = timestamp_s;
    last_time_s_ = timestamp_s;
    last_speed_mps_ = speed_mps;
  } else if (timestamp_s > last_time_s_) {
    const double dt_s = timestamp_s - last_time_s_;
    speed_time_integral_ += 0.5 * (last_speed_mps_ + speed_mps) * dt_s;
    if (in_proximity_) {
      proximity_time_s_ += dt_s;
    }
    if (current_collision_) {
      collision_time_s_ += dt_s;
    }
    last_time_s_ = timestamp_s;
    last_speed_mps_ = speed_mps;
  } else {
    last_speed_mps_ = speed_mps;
  }

  ++odometry_samples_;
  latest_speed_mps_ = speed_mps;
  max_speed_mps_ = std::max(max_speed_mps_, speed_mps);
  if (!terminal_metrics_received_) {
    final_goal_distance_m_ = goal_distance_m;
    min_goal_distance_m_ = std::min(min_goal_distance_m_, goal_distance_m);
  }
  if (goal_distance_m <= config.success_radius_m) {
    success_ = true;
  }
}

void MetricsAccumulator::observe_terminal_metrics(const TerminalMetrics & terminal_metrics)
{
  ++terminal_samples_;
  terminal_metrics_received_ = true;
  success_ = success_ || terminal_metrics.success;
  collision_ = collision_ || terminal_metrics.collision;
  timeout_ = timeout_ || terminal_metrics.timeout;

  if (std::isfinite(terminal_metrics.final_goal_distance_m)) {
    final_goal_distance_m_ = terminal_metrics.final_goal_distance_m;
    min_goal_distance_m_ = std::min(min_goal_distance_m_, terminal_metrics.final_goal_distance_m);
  }
  if (std::isfinite(terminal_metrics.min_ray_distance_m)) {
    latest_min_ray_distance_m_ = terminal_metrics.min_ray_distance_m;
    min_ray_distance_m_ = std::min(min_ray_distance_m_, terminal_metrics.min_ray_distance_m);
  }
}

MetricsSnapshot MetricsAccumulator::snapshot() const
{
  MetricsSnapshot result;
  result.started = started_;
  result.success = success_;
  result.collision = collision_;
  result.timeout = timeout_;
  result.in_proximity = in_proximity_;
  result.current_collision = current_collision_;
  result.odometry_samples = odometry_samples_;
  result.ray_samples = ray_samples_;
  result.terminal_samples = terminal_samples_;
  result.proximity_samples = proximity_samples_;
  result.collision_samples = collision_samples_;
  result.run_duration_s = started_ ? std::max(0.0, last_time_s_ - start_time_s_) : 0.0;
  result.average_speed_mps = result.run_duration_s > 0.0 ?
    speed_time_integral_ / result.run_duration_s :
    latest_speed_mps_;
  result.latest_speed_mps = latest_speed_mps_;
  result.max_speed_mps = max_speed_mps_;
  result.final_goal_distance_m = final_goal_distance_m_;
  result.min_goal_distance_m = min_goal_distance_m_;
  result.latest_min_ray_distance_m = latest_min_ray_distance_m_;
  result.min_ray_distance_m = min_ray_distance_m_;
  result.proximity_sample_ratio = ray_samples_ > 0U ?
    static_cast<double>(proximity_samples_) / static_cast<double>(ray_samples_) :
    0.0;
  result.collision_sample_ratio = ray_samples_ > 0U ?
    static_cast<double>(collision_samples_) / static_cast<double>(ray_samples_) :
    0.0;
  result.proximity_time_s = proximity_time_s_;
  result.collision_time_s = collision_time_s_;
  return result;
}
}  // namespace aerostrike_pkg
