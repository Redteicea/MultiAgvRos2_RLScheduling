// Copyright 2026
// 仓储栅格地图：将 warehouse.world 中的墙体/货架建模为障碍栅格，
// 提供 世界坐标 <-> 栅格坐标 转换与障碍查询。
#ifndef FLEET_MANAGER__WAREHOUSE_MAP_HPP_
#define FLEET_MANAGER__WAREHOUSE_MAP_HPP_

#include "data_type/grid_hash_map.hpp"

#include <cstdint>
#include <algorithm>
#include <cmath>
#include <vector>
#include <string>
#include <iostream>

namespace fleet_manager
{
  /// @brief 仓储栅格化地图
  class WarehouseMap
  {
  public:
    /// 地图原点(左下角世界坐标)、栅格分辨率、栅格数量
    WarehouseMap(double origin_x, double origin_y, double res, int nx, int ny)
    : origin_x_(origin_x), origin_y_(origin_y), res_(res), nx_(nx), ny_(ny),
      occ_(static_cast<std::size_t>(nx) * ny, 0)
    {}

    int nx() const { return nx_; }
    int ny() const { return ny_; }
    double res() const { return res_; }
    double originX() const { return origin_x_; }
    double originY() const { return origin_y_; }

    /// 世界坐标 -> 栅格（越界会 clamp 到边界，调用方需自行用 inBounds/isFree 判断）
    Cell worldToCell(double wx, double wy) const
    {
      Cell c;
      c.x = static_cast<int>(std::floor((wx - origin_x_) / res_));
      c.y = static_cast<int>(std::floor((wy - origin_y_) / res_));
      c.x = std::clamp(c.x, 0, nx_ - 1);
      c.y = std::clamp(c.y, 0, ny_ - 1);
      return c;
    }

    /// 栅格中心 -> 世界坐标
    double cellCenterX(const Cell & c) const { return origin_x_ + (c.x + 0.5) * res_; }
    double cellCenterY(const Cell & c) const { return origin_y_ + (c.y + 0.5) * res_; }

    /// 是否在栅格化区域内
    bool inBounds(const Cell & c) const
    {
      return c.x >= 0 && c.x < nx_ && c.y >= 0 && c.y < ny_;
    }

    /// 静态栅格是否可通行
    bool isFree(const Cell & c) const
    {
      if (!inBounds(c)) { return false; }
      return occ_[static_cast<std::size_t>(c.y) * nx_ + c.x] == 0;
    }

    /// 添加矩形障碍（世界坐标），可选膨胀量（米）
    void addRectObstacle(double x0, double y0, double x1, double y1, double inflate = 0.0)
    {
      Cell c0 = worldToCell(x0 - inflate, y0 - inflate);
      Cell c1 = worldToCell(x1 + inflate, y1 + inflate);
      for (int j = c0.y; j <= c1.y; ++j) {
        for (int i = c0.x; i <= c1.x; ++i) {
          if (inBounds({i, j})) {
            /// 标记为不可行
            occ_[static_cast<std::size_t>(j) * nx_ + i] = 1;
          }
        }
      }
      /// 标记为障碍物
      obstacles_.push_back({x0, y0, x1, y1, "rect"});
    }

    /// 清空矩形区域（世界坐标）——用于"物理上可停但被膨胀覆盖"的口袋，
    /// 如充电停车位：目标点紧贴墙，0.6 膨胀把目标栅格吞进障碍区（问题 29）
    void clearRect(double x0, double y0, double x1, double y1)
    {
      Cell c0 = worldToCell(x0, y0);
      Cell c1 = worldToCell(x1, y1);
      for (int j = c0.y; j <= c1.y; ++j) {
        for (int i = c0.x; i <= c1.x; ++i) {
          if (inBounds({i, j})) {
            occ_[static_cast<std::size_t>(j) * nx_ + i] = 0;
          }
        }
      }
    }

    /// 4 邻域自由邻居
    void neighbors4(const Cell & c, std::vector<std::pair<Cell, double>> & out) const
    {
      out.clear();
      static const int dx[4] = {1, -1, 0, 0};
      static const int dy[4] = {0, 0, 1, -1};
      for (int k = 0; k < 4; ++k) {
        Cell n{c.x + dx[k], c.y + dy[k]};
        if (isFree(n)) { 
          out.push_back({n, 1.0}); 
        }
      }
    }

    /// 8 邻域自由邻居（对角代价 sqrt(2)）
    void neighbors8(const Cell & c, std::vector<std::pair<Cell, double>> & out) const
    {
      out.clear();
      for (int dx = -1; dx <= 1; ++dx) {
        for (int dy = -1; dy <= 1; ++dy) {
          if (dx == 0 && dy == 0) { continue; }
          Cell n{c.x + dx, c.y + dy};
          if (isFree(n)) {
            out.push_back({n, (dx != 0 && dy != 0) ? 1.414213562 : 1.0});
          }
        }
      }
    }

    /// 在给定动态障碍集合下判断可通行（静态地图 + 动态障碍）
    template<typename BlockedSet>
    bool isFreeDynamic(const Cell & c, const BlockedSet & blocked) const
    {
      if (!isFree(c)) { return false; }
      return blocked.find(c) == blocked.end();
    }

    /// 打印障碍统计（调试用）
    void printStats() const
    {
      std::size_t n = 0;
      for (auto v : occ_) { n += (v != 0); }
      std::cout << "[WarehouseMap] grid " << nx_ << "x" << ny_
                << " res=" << res_ << " occupied_cells=" << n << std::endl;
    }

  private:
    double origin_x_, origin_y_, res_;      // 坐标原点，分辨率
    int nx_, ny_;                           // x和y的栅格化数量
    std::vector<int8_t> occ_;               // 栅格化列表，0（可行）和1（不可行）
    std::vector<RectObstacle> obstacles_;   // 障碍物列表
  };

}  // namespace fleet_manager
#endif  // FLEET_MANAGER__WAREHOUSE_MAP_HPP_
