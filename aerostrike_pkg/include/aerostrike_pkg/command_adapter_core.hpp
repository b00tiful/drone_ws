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
struct CommandAdapterConfig
{
  std::size_t action_size{3};
  double horizontal_velocity_limit_mps{5.0};
  double vertical_velocity_limit_mps{1.0};
  bool clamp_normalized_action{true};
};

struct VelocityCommand
{
  double vx_body_mps{0.0};
  double vy_body_mps{0.0};
  double vz_body_mps{0.0};
};

void validate_command_adapter_config(const CommandAdapterConfig & config);

VelocityCommand scale_policy_action(
  const CommandAdapterConfig & config,
  const std::vector<float> & normalized_action);
}  // namespace aerostrike_pkg
