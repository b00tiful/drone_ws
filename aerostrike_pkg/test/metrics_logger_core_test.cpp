// Copyright 2026 AeroStrike
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "aerostrike_pkg/metrics_logger_core.hpp"

#include <gtest/gtest.h>

#include <limits>
#include <stdexcept>
#include <vector>

namespace aerostrike_pkg
{
namespace
{
constexpr double kTolerance = 1.0e-6;

MetricsLoggerConfig test_config()
{
  MetricsLoggerConfig config;
  config.ray_count = 3;
  return config;
}
}  // namespace

TEST(MetricsLoggerCore, TracksSuccessDurationAndAverageSpeed)
{
  const MetricsLoggerConfig config = test_config();
  MetricsAccumulator metrics;

  metrics.observe_odometry(
    config,
    10.0,
    MetricsVector3{0.0, 0.0, 0.0},
    MetricsVector3{1.0, 0.0, 0.0},
    MetricsVector3{2.0, 0.0, 0.0});
  metrics.observe_odometry(
    config,
    11.0,
    MetricsVector3{1.0, 0.0, 0.0},
    MetricsVector3{3.0, 0.0, 0.0},
    MetricsVector3{2.0, 0.0, 0.0});
  metrics.observe_odometry(
    config,
    12.0,
    MetricsVector3{1.4, 0.0, 0.0},
    MetricsVector3{1.0, 0.0, 0.0},
    MetricsVector3{2.0, 0.0, 0.0});

  const MetricsSnapshot snapshot = metrics.snapshot();

  EXPECT_TRUE(snapshot.started);
  EXPECT_TRUE(snapshot.success);
  EXPECT_EQ(snapshot.odometry_samples, 3U);
  EXPECT_NEAR(snapshot.run_duration_s, 2.0, kTolerance);
  EXPECT_NEAR(snapshot.average_speed_mps, 2.0, kTolerance);
  EXPECT_NEAR(snapshot.max_speed_mps, 3.0, kTolerance);
  EXPECT_NEAR(snapshot.final_goal_distance_m, 0.6, kTolerance);
  EXPECT_NEAR(snapshot.min_goal_distance_m, 0.6, kTolerance);
}

TEST(MetricsLoggerCore, TracksCollisionAndProximityFromRaySamples)
{
  const MetricsLoggerConfig config = test_config();
  MetricsAccumulator metrics;

  metrics.observe_ray_distances(config, std::vector<float>{2.0F, 1.0F, 3.0F});
  metrics.observe_odometry(
    config,
    0.0,
    MetricsVector3{0.0, 0.0, 0.0},
    MetricsVector3{2.0, 0.0, 0.0},
    MetricsVector3{10.0, 0.0, 0.0});
  metrics.observe_odometry(
    config,
    1.0,
    MetricsVector3{1.0, 0.0, 0.0},
    MetricsVector3{2.0, 0.0, 0.0},
    MetricsVector3{10.0, 0.0, 0.0});
  metrics.observe_ray_distances(config, std::vector<float>{0.3F, 4.0F, 2.0F});
  metrics.observe_odometry(
    config,
    2.0,
    MetricsVector3{2.0, 0.0, 0.0},
    MetricsVector3{2.0, 0.0, 0.0},
    MetricsVector3{10.0, 0.0, 0.0});

  const MetricsSnapshot snapshot = metrics.snapshot();

  EXPECT_TRUE(snapshot.collision);
  EXPECT_TRUE(snapshot.current_collision);
  EXPECT_TRUE(snapshot.in_proximity);
  EXPECT_EQ(snapshot.ray_samples, 2U);
  EXPECT_EQ(snapshot.proximity_samples, 2U);
  EXPECT_EQ(snapshot.collision_samples, 1U);
  EXPECT_NEAR(snapshot.proximity_sample_ratio, 1.0, kTolerance);
  EXPECT_NEAR(snapshot.collision_sample_ratio, 0.5, kTolerance);
  EXPECT_NEAR(snapshot.proximity_time_s, 2.0, kTolerance);
  EXPECT_NEAR(snapshot.collision_time_s, 1.0, kTolerance);
  EXPECT_NEAR(snapshot.min_ray_distance_m, 0.3, kTolerance);
}

TEST(MetricsLoggerCore, IgnoresNonfiniteRayDistances)
{
  const MetricsLoggerConfig config = test_config();
  MetricsAccumulator metrics;

  metrics.observe_ray_distances(
    config,
    std::vector<float>{
      std::numeric_limits<float>::infinity(),
      std::numeric_limits<float>::quiet_NaN(),
      2.0F});

  const MetricsSnapshot snapshot = metrics.snapshot();

  EXPECT_FALSE(snapshot.collision);
  EXPECT_FALSE(snapshot.in_proximity);
  EXPECT_NEAR(snapshot.latest_min_ray_distance_m, 2.0, kTolerance);
  EXPECT_NEAR(snapshot.min_ray_distance_m, 2.0, kTolerance);
}

TEST(MetricsLoggerCore, RejectsMalformedInputOrInvalidConfig)
{
  MetricsLoggerConfig config = test_config();
  MetricsAccumulator metrics;

  EXPECT_THROW(
    metrics.observe_ray_distances(config, std::vector<float>{1.0F, 2.0F}),
    std::invalid_argument);
  EXPECT_THROW(
    metrics.observe_odometry(
      config,
      std::numeric_limits<double>::quiet_NaN(),
      MetricsVector3{},
      MetricsVector3{},
      MetricsVector3{}),
    std::invalid_argument);

  config.ray_count = 0;
  EXPECT_THROW(validate_metrics_logger_config(config), std::invalid_argument);

  config = test_config();
  config.collision_radius_m = 2.0;
  config.proximity_radius_m = 1.0;
  EXPECT_THROW(validate_metrics_logger_config(config), std::invalid_argument);
}
}  // namespace aerostrike_pkg
