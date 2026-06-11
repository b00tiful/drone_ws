// Copyright 2026 AeroStrike
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "aerostrike_pkg/command_adapter_core.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace aerostrike_pkg
{
namespace
{
constexpr std::size_t kPolicyActionSize = 3;
}

void validate_command_adapter_config(const CommandAdapterConfig & config)
{
  if (config.action_size != kPolicyActionSize) {
    throw std::invalid_argument("action_size must be 3");
  }
  if (config.horizontal_velocity_limit_mps <= 0.0) {
    throw std::invalid_argument("horizontal_velocity_limit_mps must be positive");
  }
  if (config.vertical_velocity_limit_mps <= 0.0) {
    throw std::invalid_argument("vertical_velocity_limit_mps must be positive");
  }
}

VelocityCommand scale_policy_action(
  const CommandAdapterConfig & config,
  const std::vector<float> & normalized_action)
{
  validate_command_adapter_config(config);

  if (normalized_action.size() != config.action_size) {
    throw std::invalid_argument(
            "policy action size " + std::to_string(normalized_action.size()) +
            " does not match configured action_size " + std::to_string(config.action_size));
  }

  double action[3];
  for (std::size_t index = 0; index < kPolicyActionSize; ++index) {
    if (!std::isfinite(normalized_action[index])) {
      throw std::invalid_argument("policy action contains non-finite value");
    }

    action[index] = static_cast<double>(normalized_action[index]);
    if (config.clamp_normalized_action) {
      action[index] = std::clamp(action[index], -1.0, 1.0);
    }
  }

  return VelocityCommand{
    action[0] * config.horizontal_velocity_limit_mps,
    action[1] * config.horizontal_velocity_limit_mps,
    action[2] * config.vertical_velocity_limit_mps,
  };
}
}  // namespace aerostrike_pkg
