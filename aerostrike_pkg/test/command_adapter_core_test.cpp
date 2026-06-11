// Copyright 2026 AeroStrike
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "aerostrike_pkg/command_adapter_core.hpp"

#include <gtest/gtest.h>

#include <limits>
#include <stdexcept>
#include <vector>

namespace aerostrike_pkg
{
namespace
{
constexpr double kTolerance = 1.0e-6;
}

TEST(CommandAdapterCore, ScalesAndClipsPolicyAction)
{
  CommandAdapterConfig config;

  const VelocityCommand command = scale_policy_action(
    config,
    std::vector<float>{1.0F, -0.5F, 2.0F});

  EXPECT_NEAR(command.vx_body_mps, 5.0, kTolerance);
  EXPECT_NEAR(command.vy_body_mps, -2.5, kTolerance);
  EXPECT_NEAR(command.vz_body_mps, 1.0, kTolerance);
}

TEST(CommandAdapterCore, CanDisableActionClamping)
{
  CommandAdapterConfig config;
  config.clamp_normalized_action = false;

  const VelocityCommand command = scale_policy_action(
    config,
    std::vector<float>{1.2F, -1.5F, -0.5F});

  EXPECT_NEAR(command.vx_body_mps, 6.0, kTolerance);
  EXPECT_NEAR(command.vy_body_mps, -7.5, kTolerance);
  EXPECT_NEAR(command.vz_body_mps, -0.5, kTolerance);
}

TEST(CommandAdapterCore, RejectsMalformedOrNonfiniteAction)
{
  CommandAdapterConfig config;

  EXPECT_THROW(
    scale_policy_action(config, std::vector<float>{0.0F, 0.0F}),
    std::invalid_argument);
  EXPECT_THROW(
    scale_policy_action(
      config,
      std::vector<float>{0.0F, std::numeric_limits<float>::quiet_NaN(), 0.0F}),
    std::invalid_argument);
}

TEST(CommandAdapterCore, RejectsInvalidConfig)
{
  CommandAdapterConfig config;

  config.action_size = 2;
  EXPECT_THROW(validate_command_adapter_config(config), std::invalid_argument);

  config = CommandAdapterConfig{};
  config.horizontal_velocity_limit_mps = 0.0;
  EXPECT_THROW(validate_command_adapter_config(config), std::invalid_argument);

  config = CommandAdapterConfig{};
  config.vertical_velocity_limit_mps = -1.0;
  EXPECT_THROW(validate_command_adapter_config(config), std::invalid_argument);
}
}  // namespace aerostrike_pkg
