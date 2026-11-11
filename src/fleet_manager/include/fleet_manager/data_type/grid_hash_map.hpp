#ifndef FLEET_MANAGER_GRID_HASH_MAP_HPP_
#define FLEET_MANAGER_GRID_HASH_MAP_HPP_

#include <string>

namespace fleet_manager
{

    /// 栅格单元（整数坐标）
    struct Cell
    {
        int x = 0;
        int y = 0;
        bool operator==(const Cell & o) const { return x == o.x && y == o.y; }
        bool operator!=(const Cell & o) const { return !(*this == o); }
        bool operator<(const Cell & o) const
        {
            return (y != o.y) ? (y < o.y) : (x < o.x);
        }
    };

    struct CellHash
    {
        std::size_t operator()(const Cell & c) const
        {
            return (static_cast<std::size_t>(c.x) << 32) ^ static_cast<std::size_t>(c.y);
        }
    };

    /// 简单矩形障碍物
    struct RectObstacle
    {
        double x0, y0, x1, y1;
        std::string name;
    };

}

#endif  // FLEET_MANAGER_GRID_HASH_MAP_HPP_