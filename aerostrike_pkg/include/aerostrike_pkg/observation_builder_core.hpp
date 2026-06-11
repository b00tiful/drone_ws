// Copyright 2026 AeroStrike
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#pragma once

#include <cstddef>
#include <vector>

namespace aerostrike_pkg
{
struct Vector3
{
  double x{0.0};
  double y{0.0};
  double z{0.0};
};

struct Quaternion
{
  double x{0.0};
  double y{0.0};
  double z{0.0};
  double w{1.0};
};

struct ObservationBuilderConfig
{
  std::size_t ray_count{24};
  double ray_min_range_m{0.2};
  double ray_max_range_m{10.0};
  double goal_distance_normalizer_m{20.0};
  double max_height_m{4.0};
  bool ray_distances_are_normalized{false};
  bool twist_is_body_frame{true};
};

struct ObservationBuilderInput
{
  Vector3 position_w;
  Quaternion orientation_wb;
  Vector3 linear_velocity;
  Vector3 angular_velocity;
  Vector3 goal_w;
  std::vector<float> ray_distances;
  std::vector<float> previous_action{0.0F, 0.0F, 0.0F};
};

Vector3 rotate_world_to_body(const Quaternion & orientation_wb, const Vector3 & vector_w);

std::vector<float> build_policy_observation(
  const ObservationBuilderConfig & config,
  const ObservationBuilderInput & input);
}  // namespace aerostrike_pkg
