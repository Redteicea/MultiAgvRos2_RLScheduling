// TEB 规划器 RViz 演示节点。
//
// 不接任何实际机器人，只用参数里的起终点和障碍物做周期性重规划，
// 把轨迹和障碍物发成 Path / MarkerArray，在 RViz 里即可看到效果。
// 用 RViz 工具栏的 "2D Pose Estimate" / "2D Goal Pose" 可以实时拖动起终点。

#include <cmath>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <ackermann_msgs/msg/ackermann_drive_stamped.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <nav_msgs/msg/path.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include <teb_planner/ros_bridge.h>
#include <teb_planner/teb_planner.h>

using g2o::teb::Obstacle;
using g2o::teb::Pose2D;

class TebDemoNode : public rclcpp::Node {
 public:
  TebDemoNode() : Node("teb_planner_demo") {
    frame_ = declare_parameter<std::string>("frame_id", "map");
    const double rate = declare_parameter<double>("rate_hz", 2.0);

    start_ = poseParam("start", {0., 0., 0.});
    goal_ = poseParam("goal", {5., 2., 0.});

    // ---- 底盘运动学 ----
    const std::string kin =
        declare_parameter<std::string>("kinematics", "differential");
    if (kin == "ackermann" || kin == "carlike") {
      config_.kinematics = g2o::teb::Kinematics::Ackermann;
      config_.wheelbase = declare_parameter<double>("wheelbase", 0.5);
      config_.maxSteerAngleDeg =
          declare_parameter<double>("max_steer_angle_deg", 30.0);
      RCLCPP_INFO(get_logger(),
                  "kinematics: Ackermann (wheelbase %.2f m, max steer %.1f deg, "
                  "R_min = %.2f m)",
                  config_.wheelbase, config_.maxSteerAngleDeg,
                  config_.wheelbase /
                      std::tan(config_.maxSteerAngleDeg * M_PI / 180.));
    } else if (kin != "differential") {
      throw std::invalid_argument("kinematics must be differential or ackermann");
    }

    const auto ox = declare_parameter<std::vector<double>>("obstacles_x", {2.0, 0.0});
    const auto oy = declare_parameter<std::vector<double>>("obstacles_y", {1.0, -1.5});
    const auto orad = declare_parameter<std::vector<double>>("obstacles_r", {0.5, 0.35});
    if (ox.size() != oy.size() || ox.size() != orad.size()) {
      throw std::invalid_argument("obstacles_x/y/r must have the same length");
    }
    for (size_t i = 0; i < ox.size(); ++i) {
      static_obstacles_.emplace_back(ox[i], oy[i], orad[i]);
    }

    // ---- 矩形障碍物（有向矩形）：5 个数组等长，默认两堵墙 ----
    //   rect_obstacles_x/y      中心 [m]
    //   rect_obstacles_yaw      朝向 [deg]（0 = 长边沿 x 轴）
    //   rect_obstacles_length   全长 [m]
    //   rect_obstacles_width    全宽 [m]
    const auto rx = declare_parameter<std::vector<double>>("rect_obstacles_x", {4.0, -1.0});
    const auto ry = declare_parameter<std::vector<double>>("rect_obstacles_y", {0.0, 2.2});
    const auto ryaw =
        declare_parameter<std::vector<double>>("rect_obstacles_yaw", {0.0, 45.0});
    const auto rlen =
        declare_parameter<std::vector<double>>("rect_obstacles_length", {0.4, 1.6});
    const auto rwid = declare_parameter<std::vector<double>>("rect_obstacles_width", {3.0, 0.5});
    if (rx.size() != ry.size() || rx.size() != ryaw.size() ||
        rx.size() != rlen.size() || rx.size() != rwid.size()) {
      throw std::invalid_argument(
          "rect_obstacles_x/y/yaw/length/width must have the same length");
    }
    for (size_t i = 0; i < rx.size(); ++i) {
      if (rlen[i] <= 0. || rwid[i] <= 0.) {
        throw std::invalid_argument("rect_obstacles_length/width must be > 0");
      }
      static_obstacles_.push_back(Obstacle::rect(rx[i], ry[i], 0.5 * rlen[i], 0.5 * rwid[i],
                                                 ryaw[i] * M_PI / 180.));
    }

    moving_obstacle_ = declare_parameter<bool>("moving_obstacle", true);

    config_.robotRadius = declare_parameter<double>("robot_radius", 0.32);
    config_.maxSpeed = declare_parameter<double>("max_speed", 1.0);
    config_.maxOmega = declare_parameter<double>("max_omega", 1.4);
    config_.minX = declare_parameter<double>("min_x", -8.5);
    config_.maxX = declare_parameter<double>("max_x", 8.5);
    config_.minY = declare_parameter<double>("min_y", -5.5);
    config_.maxY = declare_parameter<double>("max_y", 5.5);
    planner_ = std::make_unique<g2o::teb::TebPlanner>(config_);

    pub_path_ = create_publisher<nav_msgs::msg::Path>("~/plan", 10);
    pub_markers_ = create_publisher<visualization_msgs::msg::MarkerArray>("~/obstacles", 10);
    pub_ackermann_ =
        create_publisher<ackermann_msgs::msg::AckermannDriveStamped>("~/cmd_ackermann", 10);

    // ---- 全局地图：收到后自动替代参数里的障碍物和边界 ----
    map_range_ = declare_parameter<double>("map_range", 6.0);
    map_stride_ = declare_parameter<int>("map_stride", 2);
    sub_map_ = create_subscription<nav_msgs::msg::OccupancyGrid>(
        "map", rclcpp::QoS(1).transient_local().reliable(),
        [this](const nav_msgs::msg::OccupancyGrid::ConstSharedPtr msg) {
          map_ = msg;
          double minX, maxX, minY, maxY;
          if (g2o_teb::occupancyGridBounds(*msg, minX, maxX, minY, maxY)) {
            config_.minX = minX;
            config_.maxX = maxX;
            config_.minY = minY;
            config_.maxY = maxY;
            planner_->setConfig(config_);
          }
          RCLCPP_INFO(get_logger(), "map received: %ux%u @ %.3f m/cell -> "
                                    "边界与障碍物已切换为地图内容",
                      msg->info.width, msg->info.height, msg->info.resolution);
        });

    sub_start_ = create_subscription<geometry_msgs::msg::PoseStamped>(
        "initialpose", 10,
        [this](const geometry_msgs::msg::PoseStamped::ConstSharedPtr msg) {
          start_ = g2o_teb::fromPose(msg->pose);
          seed_.clear();
          RCLCPP_INFO(get_logger(), "start -> (%.2f, %.2f, %.2f)", start_.x, start_.y,
                      start_.theta);
        });
    sub_goal_ = create_subscription<geometry_msgs::msg::PoseStamped>(
        "goal_pose", 10,
        [this](const geometry_msgs::msg::PoseStamped::ConstSharedPtr msg) {
          goal_ = g2o_teb::fromPose(msg->pose);
          seed_.clear();
          RCLCPP_INFO(get_logger(), "goal  -> (%.2f, %.2f, %.2f)", goal_.x, goal_.y,
                      goal_.theta);
        });

    timer_ = create_wall_timer(std::chrono::duration<double>(1.0 / rate),
                               std::bind(&TebDemoNode::tick, this));
    RCLCPP_INFO(get_logger(),
                "teb_planner demo ready: 拖动 RViz 的 2D Pose Estimate / 2D Goal Pose "
                "即可改变起终点");
  }

 private:
  Pose2D poseParam(const char* name, std::vector<double> def) {
    auto v = declare_parameter<std::vector<double>>(name, std::move(def));
    if (v.size() != 3) {
      throw std::invalid_argument(std::string(name) + " must be [x, y, yaw]");
    }
    return Pose2D(v[0], v[1], v[2]);
  }

  void tick() {
    // 障碍物来源：有全局地图用地图，否则用参数里的演示障碍物
    std::vector<Obstacle> obstacles;
    if (map_) {
      obstacles = g2o_teb::obstaclesFromOccupancyGrid(*map_, start_, map_range_,
                                                      map_stride_);
    } else {
      obstacles = static_obstacles_;
    }
    if (moving_obstacle_) {
      // 往复运动的动态障碍物（位置和解析速度都给规划器，用于提前避让）
      const double t = now().seconds();
      const double amp = 1.8;
      const double w = 0.35;
      Obstacle o(3.5, amp * std::sin(w * t), 0.45, /*dynamic=*/true);
      o.velocity = g2o::Vector2(0., amp * w * std::cos(w * t));
      obstacles.push_back(o);
    }

    const g2o::teb::PlanResult result = planner_->plan(start_, goal_, obstacles, seed_);
    seed_ = result.ok ? result.trajectory : std::vector<Pose2D>();

    const rclcpp::Time stamp = now();
    pub_path_->publish(g2o_teb::toPath(result, frame_, stamp));
    pub_markers_->publish(g2o_teb::toObstacleMarkers(
        obstacles, frame_, stamp, config_.robotRadius, config_.safetyMargin));
    if (config_.kinematics == g2o::teb::Kinematics::Ackermann && result.ok) {
      pub_ackermann_->publish(g2o_teb::toAckermannDrive(result, frame_, stamp));
    }

    RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 5000,
                          "plan ok=%d poses=%zu chi2=%.3f clearance=%.3f (%.2f ms)",
                          static_cast<int>(result.ok), result.trajectory.size(), result.chi2,
                          result.minClearance, result.timeMs);
  }

  std::string frame_;
  g2o::teb::TebConfig config_;
  std::unique_ptr<g2o::teb::TebPlanner> planner_;
  Pose2D start_, goal_;
  std::vector<Pose2D> seed_;
  std::vector<Obstacle> static_obstacles_;
  bool moving_obstacle_{true};
  nav_msgs::msg::OccupancyGrid::ConstSharedPtr map_;
  double map_range_{6.0};
  int map_stride_{2};

  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pub_path_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_markers_;
  rclcpp::Publisher<ackermann_msgs::msg::AckermannDriveStamped>::SharedPtr pub_ackermann_;
  rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr sub_map_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr sub_start_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr sub_goal_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<TebDemoNode>());
  rclcpp::shutdown();
  return 0;
}
