// TEB 规划结果 <-> ROS 消息 的转换辅助。
//
// 核心库（teb_planner.h）不含任何 ROS 依赖；如果你在自己的节点里使用，
// 可以直接用这里的几个 inline 函数完成 nav_msgs/Path、MarkerArray 等转换。

#ifndef TEB_PLANNER_ROS_BRIDGE_H
#define TEB_PLANNER_ROS_BRIDGE_H

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include <ackermann_msgs/msg/ackermann_drive_stamped.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <nav_msgs/msg/path.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include <rclcpp/time.hpp>

#include <teb_planner/teb_planner.h>

namespace g2o_teb {

//! Pose2D（x, y, yaw）-> geometry_msgs/Pose（yaw 转四元数）
inline geometry_msgs::msg::Pose toPose(const g2o::teb::Pose2D& p) {
  geometry_msgs::msg::Pose msg;
  msg.position.x = p.x;
  msg.position.y = p.y;
  msg.position.z = 0.;
  msg.orientation.z = std::sin(p.theta * 0.5);
  msg.orientation.w = std::cos(p.theta * 0.5);
  return msg;
}

//! geometry_msgs/Pose -> Pose2D（四元数提取 yaw）
inline g2o::teb::Pose2D fromPose(const geometry_msgs::msg::Pose& msg) {
  const double yaw =
      std::atan2(2. * (msg.orientation.w * msg.orientation.z +
                       msg.orientation.x * msg.orientation.y),
                 1. - 2. * (msg.orientation.y * msg.orientation.y +
                            msg.orientation.z * msg.orientation.z));
  return g2o::teb::Pose2D(msg.position.x, msg.position.y, yaw);
}

inline g2o::teb::Pose2D fromPoseStamped(const geometry_msgs::msg::PoseStamped& msg) {
  return fromPose(msg.pose);
}

//! 规划结果 -> nav_msgs/Path（RViz 可直接显示）
inline nav_msgs::msg::Path toPath(const g2o::teb::PlanResult& result,
                                  const std::string& frame_id,
                                  const rclcpp::Time& stamp) {
  nav_msgs::msg::Path path;
  path.header.frame_id = frame_id;
  path.header.stamp = stamp;
  path.poses.reserve(result.trajectory.size());
  for (const g2o::teb::Pose2D& p : result.trajectory) {
    geometry_msgs::msg::PoseStamped ps;
    ps.header = path.header;
    ps.pose = toPose(p);
    path.poses.push_back(ps);
  }
  return path;
}

//! 障碍物列表 -> MarkerArray（本体圆柱 + 膨胀圈 + 动态障碍速度箭头）
inline visualization_msgs::msg::MarkerArray toObstacleMarkers(
    const std::vector<g2o::teb::Obstacle>& obstacles, const std::string& frame_id,
    const rclcpp::Time& stamp, double robot_radius, double safety_margin) {
  using visualization_msgs::msg::Marker;
  visualization_msgs::msg::MarkerArray array;

  Marker delete_all;
  delete_all.action = Marker::DELETEALL;
  array.markers.push_back(delete_all);

  for (size_t i = 0; i < obstacles.size(); ++i) {
    const g2o::teb::Obstacle& o = obstacles[i];
    const bool rect = (o.shape == g2o::teb::Shape::Rectangle);
    const double clr = robot_radius + safety_margin;  // 规划器实际使用的 clearance

    Marker body;
    body.header.frame_id = frame_id;
    body.header.stamp = stamp;
    body.ns = "obstacles";
    body.id = static_cast<int>(i);
    body.type = rect ? Marker::CUBE : Marker::CYLINDER;
    body.action = Marker::ADD;
    body.pose.position.x = o.center[0];
    body.pose.position.y = o.center[1];
    body.pose.position.z = 0.05;
    if (rect) {
      body.pose.orientation.z = std::sin(0.5 * o.yaw);
      body.pose.orientation.w = std::cos(0.5 * o.yaw);
      body.scale.x = 2. * o.halfLength;
      body.scale.y = 2. * o.halfWidth;
    } else {
      body.pose.orientation.w = 1.;
      body.scale.x = body.scale.y = 2. * o.radius;
    }
    body.scale.z = 0.1;
    body.color.r = o.dynamic ? 1.0f : 0.75f;
    body.color.g = o.dynamic ? 0.4f : 0.75f;
    body.color.b = 0.2f;
    body.color.a = 0.9f;
    array.markers.push_back(body);

    if (rect) {
      // 矩形：直接画规划器真正约束的那条边界 —— 矩形按
      // (车体半径 + 安全余量) 膨胀后的圆角矩形轮廓
      const std::vector<g2o::Vector2> outline = g2o::teb::roundedRectOutline(
          o.center, o.yaw, o.halfLength + clr, o.halfWidth + clr, clr);
      Marker contour;
      contour.header.frame_id = frame_id;
      contour.header.stamp = stamp;
      contour.ns = "inflation";
      contour.id = static_cast<int>(i);
      contour.type = Marker::LINE_STRIP;
      contour.action = Marker::ADD;
      contour.pose.orientation.w = 1.;
      contour.scale.x = 0.03;
      contour.color.r = 1.0f;
      contour.color.g = 0.6f;
      contour.color.b = 0.1f;
      contour.color.a = 0.9f;
      contour.points.resize(outline.size() + 1);  // 末点回到起点，闭合成环
      for (size_t k = 0; k < outline.size(); ++k) {
        contour.points[k].x = outline[k][0];
        contour.points[k].y = outline[k][1];
      }
      contour.points[outline.size()] = contour.points[0];
      array.markers.push_back(contour);
    } else {
      Marker inflation = body;
      inflation.ns = "inflation";
      inflation.scale.x = inflation.scale.y = 2. * (o.radius + clr);
      inflation.color.r = 1.0f;
      inflation.color.g = 0.6f;
      inflation.color.b = 0.1f;
      inflation.color.a = 0.15f;
      array.markers.push_back(inflation);
    }

    if (o.dynamic) {
      Marker arrow;
      arrow.header.frame_id = frame_id;
      arrow.header.stamp = stamp;
      arrow.ns = "velocity";
      arrow.id = static_cast<int>(i);
      arrow.type = Marker::ARROW;
      arrow.action = Marker::ADD;
      arrow.pose.orientation.w = 1.;
      arrow.scale.x = 0.05;  // 箭杆半径
      arrow.scale.y = 0.12;  // 箭头半径
      arrow.scale.z = 0.12;
      arrow.color.b = 1.0f;
      arrow.color.a = 0.9f;
      arrow.points.resize(2);
      arrow.points[0].x = o.center[0];
      arrow.points[0].y = o.center[1];
      arrow.points[1].x = o.center[0] + o.velocity[0];
      arrow.points[1].y = o.center[1] + o.velocity[1];
      array.markers.push_back(arrow);
    }
  }
  return array;
}

/**
 * \brief 全局地图（nav_msgs/OccupancyGrid）-> 规划器可用的障碍物列表
 *
 * 这是"把全局地图喂给 TEB"的标准入口：取机器人周围 maxRange 范围内
 * 所有占据栅格（data[i] > occupiedThreshold，-1 表示未知、跳过），
 * 每个格子生成一个圆障碍（半径为格子的外接圆半径）。
 *
 * \param grid            地图（通常来自 map_server 或 nav2 的 global costmap）
 * \param robot           机器人当前位姿（用于截取局部窗口）
 * \param maxRange        只提取该范围内的障碍 [m]（局部规划不需要全图）
 * \param stride          栅格采样步长（2 = 每隔一个格子取一个，控制障碍数量）
 * \param maxObstacles    障碍数量上限（取距离最近的，保证规划耗时可控）
 */
inline std::vector<g2o::teb::Obstacle> obstaclesFromOccupancyGrid(
    const nav_msgs::msg::OccupancyGrid& grid, const g2o::teb::Pose2D& robot,
    double maxRange = 6.0, int stride = 2, int maxObstacles = 250,
    int occupiedThreshold = 50) {
  std::vector<g2o::teb::Obstacle> obstacles;
  if (grid.data.empty() || grid.info.resolution <= 0.) return obstacles;

  const double res = grid.info.resolution;
  const unsigned int w = grid.info.width;
  const unsigned int h = grid.info.height;

  // 地图原点位姿（含朝向；大多数地图朝向为 0）
  const double ox = grid.info.origin.position.x;
  const double oy = grid.info.origin.position.y;
  const double c = std::cos(g2o_teb::fromPose(grid.info.origin).theta);
  const double s = std::sin(g2o_teb::fromPose(grid.info.origin).theta);

  // 机器人世界坐标 -> 地图坐标（格子中心对齐）
  const double dx = robot.x - ox;
  const double dy = robot.y - oy;
  const double localX = c * dx + s * dy;
  const double localY = -s * dx + c * dy;
  const int col0 = static_cast<int>(localX / res) - 0;
  const int row0 = static_cast<int>(localY / res) - 0;
  const int reach = static_cast<int>(maxRange / res);

  struct Candidate {
    g2o::teb::Obstacle obstacle;
    double distance;
  };
  std::vector<Candidate> candidates;

  for (int row = row0 - reach; row <= row0 + reach; row += stride) {
    if (row < 0 || row >= static_cast<int>(h)) continue;
    for (int col = col0 - reach; col <= col0 + reach; col += stride) {
      if (col < 0 || col >= static_cast<int>(w)) continue;
      const int8_t cell = grid.data[static_cast<size_t>(row) * w + col];
      if (cell <= static_cast<int8_t>(occupiedThreshold)) continue;  // 空闲/未知
      const double gx = (col + 0.5) * res;
      const double gy = (row + 0.5) * res;
      Candidate cand;
      cand.obstacle = g2o::teb::Obstacle(ox + c * gx - s * gy,
                                         oy + s * gx + c * gy,
                                         0.7071 * res);  // 格子外接圆半径
      cand.distance = std::hypot(cand.obstacle.center[0] - robot.x,
                                 cand.obstacle.center[1] - robot.y);
      if (cand.distance > maxRange) continue;
      candidates.push_back(cand);
    }
  }

  // 距离最近者优先，保证规划耗时可控
  if (static_cast<int>(candidates.size()) > maxObstacles) {
    std::partial_sort(candidates.begin(), candidates.begin() + maxObstacles,
                      candidates.end(),
                      [](const Candidate& a, const Candidate& b) {
                        return a.distance < b.distance;
                      });
    candidates.resize(maxObstacles);
  }
  obstacles.reserve(candidates.size());
  for (const Candidate& cand : candidates) obstacles.push_back(cand.obstacle);
  return obstacles;
}

/**
 * \brief 地图的世界坐标范围（用于设置 TebConfig 的 min/max 边界）
 * \return 是否成功（空地图返回 false）
 */
inline bool occupancyGridBounds(const nav_msgs::msg::OccupancyGrid& grid,
                                double& minX, double& maxX, double& minY,
                                double& maxY) {
  if (grid.data.empty() || grid.info.resolution <= 0.) return false;
  const double res = grid.info.resolution;
  const double ox = grid.info.origin.position.x;
  const double oy = grid.info.origin.position.y;
  const double theta = g2o_teb::fromPose(grid.info.origin).theta;
  const double c = std::cos(theta);
  const double s = std::sin(theta);
  const double corners[4][2] = {{0., 0.},
                                {grid.info.width * res, 0.},
                                {0., grid.info.height * res},
                                {grid.info.width * res, grid.info.height * res}};
  minX = minY = 1e18;
  maxX = maxY = -1e18;
  for (const auto& corner : corners) {
    const double wx = ox + c * corner[0] - s * corner[1];
    const double wy = oy + s * corner[0] + c * corner[1];
    minX = std::min(minX, wx);
    maxX = std::max(maxX, wx);
    minY = std::min(minY, wy);
    maxY = std::max(maxY, wy);
  }
  return true;
}

//! 规划结果的第一段控制量 -> ackermann_msgs/AckermannDriveStamped
//! （index 可取后续段，实现前瞻控制）
inline ackermann_msgs::msg::AckermannDriveStamped toAckermannDrive(
    const g2o::teb::PlanResult& result, const std::string& frame_id,
    const rclcpp::Time& stamp, size_t index = 0) {
  ackermann_msgs::msg::AckermannDriveStamped msg;
  msg.header.frame_id = frame_id;
  msg.header.stamp = stamp;
  if (index < result.cmdV.size()) {
    msg.drive.speed = result.cmdV[index];
    msg.drive.steering_angle =
        index < result.cmdSteer.size() ? result.cmdSteer[index] : 0.;
  }
  return msg;
}

}  // namespace g2o_teb

#endif  // TEB_PLANNER_ROS_BRIDGE_H
