// Copyright 2026 AeroStrike
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "aerostrike_pkg/observation_builder_core.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace aerostrike_pkg
{
namespace
{
constexpr std::size_t kVelocitySize = 3;
constexpr std::size_t kGravitySize = 3;
constexpr std::size_t kGoalDirectionSize = 3;
constexpr std::size_t kGoalDistanceSize = 1;
constexpr std::size_t kPreviousActionSize = 3;
constexpr std::size_t kHeightSize = 1;
constexpr double kEpsilon = 1.0e-6;

double clamp(const double value, const double low, const double high)
{
  return std::max(low, std::min(value, high));
}

Quaternion normalized(const Quaternion & q)
{
  const double norm = std::sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
  if (norm <= kEpsilon) {
    return Quaternion{};
  }
  return Quaternion{q.x / norm, q.y / norm, q.z / norm, q.w / norm};
}

Quaternion conjugate(const Quaternion & q)
{
  return Quaternion{-q.x, -q.y, -q.z, q.w};
}

Vector3 rotate_vector(const Quaternion & q, const Vector3 & v)
{
  const Quaternion unit = normalized(q);
  const Vector3 qv{unit.x, unit.y, unit.z};
  const Vector3 t{
    2.0 * (qv.y * v.z - qv.z * v.y),
    2.0 * (qv.z * v.x - qv.x * v.z),
    2.0 * (qv.x * v.y - qv.y * v.x),
  };

  return Vector3{
    v.x + unit.w * t.x + (qv.y * t.z - qv.z * t.y),
    v.y + unit.w * t.y + (qv.z * t.x - qv.x * t.z),
    v.z + unit.w * t.z + (qv.x * t.y - qv.y * t.x),
  };
}

void append_vector(std::vector<float> & observation, const Vector3 & value)
{
  observation.push_back(static_cast<float>(value.x));
  observation.push_back(static_cast<float>(value.y));
  observation.push_back(static_cast<float>(value.z));
}
}  // namespace

Vector3 rotate_world_to_body(const Quaternion & orientation_wb, const Vector3 & vector_w)
{
  return rotate_vector(conjugate(normalized(orientation_wb)), vector_w);
}

std::vector<float> build_policy_observation(
  const ObservationBuilderConfig & config,
  const ObservationBuilderInput & input)
{
  if (config.ray_count == 0U) {
    throw std::invalid_argument("ray_count must be positive");
  }
  if (config.ray_max_range_m <= 0.0) {
    throw std::invalid_argument("ray_max_range_m must be positive");
  }
  if (config.goal_distance_normalizer_m <= 0.0) {
    throw std::invalid_argument("goal_distance_normalizer_m must be positive");
  }
  if (config.max_height_m <= 0.0) {
    throw std::invalid_argument("max_height_m must be positive");
  }
  if (input.ray_distances.size() != config.ray_count) {
    throw std::invalid_argument(
            "ray_distances size " + std::to_string(input.ray_distances.size()) +
            " does not match configured ray_count " + std::to_string(config.ray_count));
  }
  if (input.previous_action.size() != kPreviousActionSize) {
    throw std::invalid_argument("previous_action must contain exactly 3 values");
  }

  std::vector<float> observation;
  observation.reserve(
    config.ray_count + (2U * kVelocitySize) + kGravitySize + kGoalDirectionSize +
    kGoalDistanceSize + kPreviousActionSize + kHeightSize);

  for (const float ray : input.ray_distances) {
    if (config.ray_distances_are_normalized) {
      observation.push_back(static_cast<float>(clamp(ray, 0.0, 1.0)));
      continue;
    }

    const double distance_m =
      std::isfinite(ray) ? static_cast<double>(ray) : config.ray_max_range_m;
    const double clamped_distance_m = clamp(
      distance_m, config.ray_min_range_m,
      config.ray_max_range_m);
    observation.push_back(static_cast<float>(clamped_distance_m / config.ray_max_range_m));
  }

  const Vector3 linear_velocity_b = config.twist_is_body_frame ?
    input.linear_velocity :
    rotate_world_to_body(input.orientation_wb, input.linear_velocity);
  const Vector3 angular_velocity_b = config.twist_is_body_frame ?
    input.angular_velocity :
    rotate_world_to_body(input.orientation_wb, input.angular_velocity);
  append_vector(observation, linear_velocity_b);
  append_vector(observation, angular_velocity_b);

  append_vector(observation, rotate_world_to_body(input.orientation_wb, Vector3{0.0, 0.0, -1.0}));

  const Vector3 goal_delta_w{
    input.goal_w.x - input.position_w.x,
    input.goal_w.y - input.position_w.y,
    input.goal_w.z - input.position_w.z,
  };
  const Vector3 goal_delta_b = rotate_world_to_body(input.orientation_wb, goal_delta_w);
  const double goal_distance = std::sqrt(
    goal_delta_b.x * goal_delta_b.x + goal_delta_b.y * goal_delta_b.y + goal_delta_b.z *
    goal_delta_b.z);
  const double goal_direction_scale = goal_distance > kEpsilon ? 1.0 / goal_distance : 0.0;
  append_vector(
    observation,
    Vector3{
      goal_delta_b.x * goal_direction_scale,
      goal_delta_b.y * goal_direction_scale,
      goal_delta_b.z * goal_direction_scale,
    });

  observation.push_back(
    static_cast<float>(clamp(goal_distance / config.goal_distance_normalizer_m, 0.0, 1.0)));

  for (const float value : input.previous_action) {
    observation.push_back(static_cast<float>(clamp(value, -1.0, 1.0)));
  }

  observation.push_back(
    static_cast<float>(clamp(
      input.position_w.z / config.max_height_m, 0.0,
      1.0)));
  return observation;
}
}  // namespace aerostrike_pkg
