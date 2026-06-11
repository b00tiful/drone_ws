// Copyright 2026 AeroStrike
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "aerostrike_pkg/observation_builder_core.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace aerostrike_pkg
{
namespace
{
constexpr double kTolerance = 1.0e-5;
constexpr double kPi = 3.14159265358979323846;

Quaternion yaw_quaternion(const double radians)
{
  return Quaternion{0.0, 0.0, std::sin(radians / 2.0), std::cos(radians / 2.0)};
}

ObservationBuilderInput make_base_input()
{
  ObservationBuilderInput input;
  input.position_w = Vector3{0.0, 0.0, 2.0};
  input.orientation_wb = Quaternion{};
  input.linear_velocity = Vector3{1.0, 2.0, 3.0};
  input.angular_velocity = Vector3{0.1, 0.2, 0.3};
  input.goal_w = Vector3{10.0, 0.0, 2.0};
  input.ray_distances = std::vector<float>(24, 5.0F);
  input.previous_action = std::vector<float>{0.1F, -0.2F, 0.3F};
  return input;
}
}  // namespace

TEST(ObservationBuilderCore, BuildsExpectedFortyOneValueLayout)
{
  ObservationBuilderConfig config;
  ObservationBuilderInput input = make_base_input();
  input.orientation_wb = yaw_quaternion(kPi / 2.0);
  input.goal_w = Vector3{0.0, 10.0, 2.0};

  const std::vector<float> observation = build_policy_observation(config, input);

  ASSERT_EQ(observation.size(), 41U);
  EXPECT_NEAR(observation[0], 0.5F, kTolerance);
  EXPECT_NEAR(observation[23], 0.5F, kTolerance);

  EXPECT_NEAR(observation[24], 1.0F, kTolerance);
  EXPECT_NEAR(observation[25], 2.0F, kTolerance);
  EXPECT_NEAR(observation[26], 3.0F, kTolerance);
  EXPECT_NEAR(observation[27], 0.1F, kTolerance);
  EXPECT_NEAR(observation[28], 0.2F, kTolerance);
  EXPECT_NEAR(observation[29], 0.3F, kTolerance);

  EXPECT_NEAR(observation[30], 0.0F, kTolerance);
  EXPECT_NEAR(observation[31], 0.0F, kTolerance);
  EXPECT_NEAR(observation[32], -1.0F, kTolerance);

  EXPECT_NEAR(observation[33], 1.0F, kTolerance);
  EXPECT_NEAR(observation[34], 0.0F, kTolerance);
  EXPECT_NEAR(observation[35], 0.0F, kTolerance);
  EXPECT_NEAR(observation[36], 0.5F, kTolerance);

  EXPECT_NEAR(observation[37], 0.1F, kTolerance);
  EXPECT_NEAR(observation[38], -0.2F, kTolerance);
  EXPECT_NEAR(observation[39], 0.3F, kTolerance);
  EXPECT_NEAR(observation[40], 0.5F, kTolerance);
}

TEST(ObservationBuilderCore, ConvertsWorldTwistToBodyWhenConfigured)
{
  ObservationBuilderConfig config;
  config.twist_is_body_frame = false;
  ObservationBuilderInput input = make_base_input();
  input.orientation_wb = yaw_quaternion(kPi / 2.0);
  input.linear_velocity = Vector3{0.0, 2.0, 0.0};
  input.angular_velocity = Vector3{-3.0, 0.0, 0.0};

  const std::vector<float> observation = build_policy_observation(config, input);

  EXPECT_NEAR(observation[24], 2.0F, kTolerance);
  EXPECT_NEAR(observation[25], 0.0F, kTolerance);
  EXPECT_NEAR(observation[26], 0.0F, kTolerance);
  EXPECT_NEAR(observation[27], 0.0F, kTolerance);
  EXPECT_NEAR(observation[28], 3.0F, kTolerance);
  EXPECT_NEAR(observation[29], 0.0F, kTolerance);
}

TEST(ObservationBuilderCore, SanitizesRayDistancesAndClipsActions)
{
  ObservationBuilderConfig config;
  ObservationBuilderInput input = make_base_input();
  input.ray_distances[0] = -1.0F;
  input.ray_distances[1] = std::numeric_limits<float>::quiet_NaN();
  input.ray_distances[2] = 100.0F;
  input.previous_action = std::vector<float>{-2.0F, 0.5F, 2.0F};

  const std::vector<float> observation = build_policy_observation(config, input);

  EXPECT_NEAR(observation[0], 0.02F, kTolerance);
  EXPECT_NEAR(observation[1], 1.0F, kTolerance);
  EXPECT_NEAR(observation[2], 1.0F, kTolerance);
  EXPECT_NEAR(observation[37], -1.0F, kTolerance);
  EXPECT_NEAR(observation[38], 0.5F, kTolerance);
  EXPECT_NEAR(observation[39], 1.0F, kTolerance);
}

TEST(ObservationBuilderCore, RejectsMalformedInputSizes)
{
  ObservationBuilderConfig config;
  ObservationBuilderInput input = make_base_input();

  input.ray_distances.pop_back();
  EXPECT_THROW(build_policy_observation(config, input), std::invalid_argument);

  input = make_base_input();
  input.previous_action.pop_back();
  EXPECT_THROW(build_policy_observation(config, input), std::invalid_argument);
}
}  // namespace aerostrike_pkg
