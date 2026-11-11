// Copyright 2026
// A* 路径规划（8 邻域）+ 视线平滑：
// 在静态地图 + 动态障碍集合上进行寻路，并对栅格路径做贪心视线简化，
// 使差分驱动 AGV 获得由长直线段组成的航点序列，便于跟踪。
#ifndef FLEET_MANAGER__ASTAR_PLANNER_HPP_
#define FLEET_MANAGER__ASTAR_PLANNER_HPP_

#include "warehouse_map.hpp"

#include <functional>
#include <queue>
#include <unordered_map>
#include <vector>
#include <cmath>
#include <algorithm>

namespace fleet_manager
{
/// @brief A* 全局路径规划
class AStarPlanner
{
public:
  /// 可通行判断：返回 cell 是否可进入（通常为 静态地图 + 动态障碍）
  using FreeCheck = std::function<bool(const Cell &)>;

  /// 在 start 到 goal 间规划 8 邻域路径（含起终点），失败返回空。
  /// near_goal：若终点被占，可放宽到终点附近自由栅格（默认不启用）。
  static std::vector<Cell> plan(
    const FreeCheck & free, const Cell & start, const Cell & goal,
    int max_iter = 30000, double near_goal_radius = 0.0)
  {
    if (free(start) && free(goal)) {
      return planCore(free, start, goal, max_iter);
    }
    // 起点被占：仅当非常接近时忽略（由调用方保证起点可通行）
    if (!free(start)) {
      return {};
    }
    // 终点被占：搜索附近可通行的栅格作为替代终点
    if (near_goal_radius > 0.0) {
      int r = std::max(1, static_cast<int>(std::ceil(near_goal_radius)));
      Cell best;
      double best_dist = 1e9;
      for (int dx = -r; dx <= r; ++dx) {
        for (int dy = -r; dy <= r; ++dy) {
          Cell c{goal.x + dx, goal.y + dy};
          if (free(c)) {
            double d = std::hypot(dx, dy);
            if (d < best_dist) { best_dist = d; best = c; }
          }
        }
      }
      if (best_dist < 1e8) {
        return planCore(free, start, best, max_iter); 
      }
    }
    return {};
  }

  /// 贪心视线平滑：以直线可达的最远栅格为下一航点，输出航点(栅格)序列。
  static std::vector<Cell> simplify(const FreeCheck & free, const std::vector<Cell> & path)
  {
    if (path.size() <= 2) { return path; }
    std::vector<Cell> out;
    std::size_t anchor = 0;
    out.push_back(path[anchor]);
    while (anchor < path.size() - 1) {
      std::size_t far = path.size() - 1;
      while (far > anchor + 1 && !lineClear(free, path[anchor], path[far])) {
        --far;
      }
      out.push_back(path[far]);
      anchor = far;
    }
    return out;
  }

  /// @brief 搜索 c 附近（Chebyshev 半径 ≤ max_r）最近的可通行格；找不到返回 {-1,-1}。
  /// 用于“起点/终点落在膨胀区内”时放宽到最近可通行格（膨胀是虚拟的，车物理上可移动）。
  static Cell nearestFree(const FreeCheck & free, const Cell & c, int max_r = 8)
  {
    if (free(c)) { return c; }
    for (int r = 1; r <= max_r; ++r) {
      Cell best{-1, -1};
      double best_d = 1e9;
      for (int dx = -r; dx <= r; ++dx) {
        for (int dy = -r; dy <= r; ++dy) {
          if (std::max(std::abs(dx), std::abs(dy)) != r) { continue; }  // 只扫当前环
          Cell n{c.x + dx, c.y + dy};
          if (free(n)) {
            double d = std::hypot(static_cast<double>(dx), static_cast<double>(dy));
            if (d < best_d) { best_d = d; best = n; }
          }
        }
      }
      if (best.x >= 0) { return best; }
    }
    return {-1, -1};
  }

private:
  /// @brief 规划函数
  /// @param free 
  /// @param start 
  /// @param goal 
  /// @param max_iter 
  /// @return 
  static std::vector<Cell> planCore(
    const FreeCheck & free, const Cell & start, const Cell & goal, int max_iter)
  {
    using Node = std::pair<double, Cell>;  // (f, cell) 
    std::priority_queue<Node, std::vector<Node>, std::greater<Node>> open;  // 从当前格子到终点的预估代价
    std::unordered_map<Cell, double, CellHash> g_score; // 从起点走到当前格子的实际代价
    std::unordered_map<Cell, Cell, CellHash> came_from; // 终点,起点

    open.push({heuristic(start, goal), start});
    g_score[start] = 0.0;

    int iter = 0;
    while (!open.empty() && iter++ < max_iter) {
      auto [f, cur] = open.top();
      open.pop();
      if (cur == goal) {
        return reconstruct(came_from, start, goal);
      }
      double g_cur = g_score[cur];
      for (int dx = -1; dx <= 1; ++dx) {
        for (int dy = -1; dy <= 1; ++dy) {
          if (dx == 0 && dy == 0) { continue; }
          Cell n{cur.x + dx, cur.y + dy};
          if (!free(n)) {
            continue; 
          }
          // 禁止贴角穿越（两对角均被占时视为不可通行）
          if (dx != 0 && dy != 0) {
            if (!free({cur.x + dx, cur.y}) && !free({cur.x, cur.y + dy})) { continue; }
          }
          double step = (dx != 0 && dy != 0) ? 1.414213562 : 1.0;
          double g_new = g_cur + step;
          auto it = g_score.find(n);
          if (it == g_score.end() || g_new < it->second) {
            g_score[n] = g_new;
            came_from[n] = cur;
            open.push({g_new + heuristic(n, goal), n});
          }
        }
      }
    }
    return {};
  }

  /// @brief 启发函数 。h = max(dx, dy) + (√2 − 1) × min(dx, dy)
  /// @param a 
  /// @param b 
  /// @return 从格子 a 到格子 b 的预估代价
  static double heuristic(const Cell & a, const Cell & b)
  {
    double dx = std::abs(a.x - b.x);
    double dy = std::abs(a.y - b.y);
    return std::max(dx, dy) + (1.414213562 - 1.0) * std::min(dx, dy);
  }

  /// @brief 
  /// @param came_from 
  /// @param start 
  /// @param goal 
  /// @return 
  static std::vector<Cell> reconstruct(
    const std::unordered_map<Cell, Cell, CellHash> & came_from,
    const Cell & start, const Cell & goal)
  {
    std::vector<Cell> path;
    Cell cur = goal;
    while (true) {
      path.push_back(cur);
      if (cur == start) { break; }
      auto it = came_from.find(cur);
      if (it == came_from.end()) { break; }
      cur = it->second;
    }
    std::reverse(path.begin(), path.end());
    return path;
  }

  /// @brief 栅格中心连线是否全程可通行（采样检查）
  /// @param came_from 
  /// @param start 
  /// @param goal 
  /// @return 
  static bool lineClear(const FreeCheck & free, const Cell & a, const Cell & b)
  {
    double ax = a.x + 0.5, ay = a.y + 0.5;
    double bx = b.x + 0.5, by = b.y + 0.5;
    int steps = std::max(std::abs(b.x - a.x), std::abs(b.y - a.y));
    if (steps <= 1) { return true; }
    // 以 0.5 栅格为采样步长，防止贴角穿越
    int samples = steps * 2;
    for (int k = 1; k < samples; ++k) {
      double t = static_cast<double>(k) / samples;
      double wx = ax + t * (bx - ax);
      double wy = ay + t * (by - ay);
      Cell c{static_cast<int>(std::floor(wx)), static_cast<int>(std::floor(wy))};
      if (!free(c)) { return false; }
    }
    return true;
  }
};

}  // namespace fleet_manager
#endif  // FLEET_MANAGER__ASTAR_PLANNER_HPP_
