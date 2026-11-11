// Copyright 2026
// 路权管理（TrafficControl）：
// - 栅格占用：每台 AGV 的膨胀足迹所在栅格 => 所属 AGV
// - 前方预留：移动中 AGV 沿路径向前预留 LOOKAHEAD 个栅格
// - 冲突判定：canEnter 检查目标栅格是否被其它 AGV 占用/预留
// - 动态障碍：为死锁重规划提供 其它 AGV 占用+预留 的障碍集合
#ifndef FLEET_MANAGER__TRAFFIC_CONTROL_HPP_
#define FLEET_MANAGER__TRAFFIC_CONTROL_HPP_

#include "warehouse_map.hpp"
#include "robot_agent.hpp"

#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace fleet_manager
{
  /// @brief 路权管理
  class TrafficControl
  {
  public:
    /// radius_cells：占用膨胀半径（栅格数）
    explicit TrafficControl(double radius_cells) : radius_cells_(radius_cells) {}

    /// 每周期刷新：重建占用表与预留表
    void update(const std::vector<std::shared_ptr<RobotAgent>> & robots)
    {
      occupancy_.clear();
      claims_.clear();
      for (const auto & r : robots) {
        if (!r->has_pose) { continue; }
        if (r->state == RobotState::FAULT || r->state == RobotState::OFFLINE) { continue; }

        // ---- 占用 ----
        std::vector<Cell> cells;
        // 膨胀
        this->inflateCells(r->current_cell, cells);
        for (const auto & c : cells) {
          // 标记机器人占用的栅格
          occupancy_[c] = r->id;
        }

        // ---- 预留（仅移动中的 AGV；YIELD/FAULT 不预留，便于让行）----
        if (r->isMovingState() && !r->path.empty()) {
          std::size_t idx = TrafficControl::pathIndexOf(r->path, r->current_cell);
          for (std::size_t k = idx + 1; k <= idx + kLookahead && k < r->path.size(); ++k) {
            // 向前预留，提前标记机器人路径即将通过的栅格
            claims_[r->path[k]] = r->id;
          }
        }
      }
    }

    /// 其它 AGV 是否占用/预留了栅格 c（自身 id 除外）
    bool isBlockedByOther(int id, const Cell & c) const
    {
      auto it = occupancy_.find(c);
      if (it != occupancy_.end() && it->second != id) { return true; }
      auto jt = claims_.find(c);
      if (jt != claims_.end() && jt->second != id) { return true; }
      return false;
    }

    /// 获得栅格 c 的占用者（-1 表示空闲/仅自身）
    int occupantOf(const Cell & c) const
    {
      auto it = occupancy_.find(c);
      if (it != occupancy_.end()) { return it->second; }
      auto jt = claims_.find(c);
      if (jt != claims_.end()) { return jt->second; }
      return -1;
    }

    /// 其它 AGV 的动态障碍（占用 + 预留）集合
    void dynamicObstacles(int id, std::unordered_set<Cell, CellHash> & out) const
    {
      out.clear();
      for (const auto & kv : occupancy_) {
        if (kv.second != id) { out.insert(kv.first); }
      }
      for (const auto & kv : claims_) {
        if (kv.second != id) { out.insert(kv.first); }
      }
    }

    /// 机器人足迹膨胀（以栅格为单位，包含自身栅格）
    void inflateCells(const Cell & c, std::vector<Cell> & out) const
    {
      out.clear();
      int r = static_cast<int>(std::ceil(radius_cells_));
      for (int dx = -r; dx <= r; ++dx) {
        for (int dy = -r; dy <= r; ++dy) {
          if (std::hypot(dx, dy) <= radius_cells_ + 1e-6) {
            out.push_back({c.x + dx, c.y + dy});
          }
        }
      }
    }

    /// 在路径中定位与给定栅格最近的索引（线性扫描）
    static std::size_t pathIndexOf(const std::deque<Cell> & path, const Cell & c)
    {
      std::size_t best = 0;
      double best_d = 1e9;
      for (std::size_t i = 0; i < path.size(); ++i) {
        double d = std::hypot(static_cast<double>(path[i].x - c.x), static_cast<double>(path[i].y - c.y));
        if (d < best_d) { 
          best_d = d; 
          best = i; 
        }
      }
      return best;
    }

  private:
    double radius_cells_;
    std::unordered_map<Cell, int, CellHash> occupancy_; // 占用表（当前时刻谁占着哪儿）
    std::unordered_map<Cell, int, CellHash> claims_;    // 预留表（未来一段谁将经过哪儿）
    static constexpr std::size_t kLookahead = 2;  // 预留前方栅格数
  };

}  // namespace fleet_manager
#endif  // FLEET_MANAGER__TRAFFIC_CONTROL_HPP_
