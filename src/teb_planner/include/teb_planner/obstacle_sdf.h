// 障碍物几何：有向矩形（OBB）的符号距离、梯度与轮廓采样。纯头文件、无 ROS 依赖。
//
// 本包的避障统一表述为「位姿点到障碍物表面的符号距离 >= clearance」：
//   - 圆形障碍物：d = |p - c|
//   - 矩形障碍物：d = 点到 OBB 的距离（外部为正、内部为负）
// 机器人本体一律按半径 robotRadius 的圆处理（robotRadius + safetyMargin 并入
// clearance），因此矩形约束的物理含义是「矩形按圆膨胀后的圆角矩形之外」——
// 这正是 RViz 里那条膨胀轮廓线。

#ifndef TEB_PLANNER_OBSTACLE_SDF_H
#define TEB_PLANNER_OBSTACLE_SDF_H

#include <algorithm>
#include <cmath>
#include <vector>

#include <teb_planner/g2o_compat.h>

namespace g2o {
namespace teb {

//! 点到有向矩形的符号距离 [m]：外部为正，内部为负
//! \param center      矩形中心
//! \param yaw         矩形朝向 [rad]（局部 x 轴为"长边"方向）
//! \param halfLength  沿局部 x 轴的半长 [m]
//! \param halfWidth   沿局部 y 轴的半宽 [m]
inline number_t rectDistance(const Vector2& point, const Vector2& center, number_t yaw,
                             number_t halfLength, number_t halfWidth) {
  const number_t c = std::cos(yaw);
  const number_t s = std::sin(yaw);
  const number_t dx = point[0] - center[0];
  const number_t dy = point[1] - center[1];
  const number_t px = c * dx + s * dy;  // 旋到矩形局部坐标系
  const number_t py = -s * dx + c * dy;
  const number_t qx = std::fabs(px) - halfLength;
  const number_t qy = std::fabs(py) - halfWidth;
  const number_t ox = std::max(qx, cst(0.));
  const number_t oy = std::max(qy, cst(0.));
  const number_t outside = std::sqrt(ox * ox + oy * oy);   // 角点区/面外向距离
  const number_t inside = std::min(std::max(qx, qy), cst(0.));  // 内部为负
  return outside + inside;
}

//! rectDistance 对 point 的梯度（单位向量，指向"离开矩形"的方向）
//! 内部点取最近面的法向；与 rectDistance 的分段一致，可直接用于解析雅可比。
inline Vector2 rectDistanceGradient(const Vector2& point, const Vector2& center,
                                    number_t yaw, number_t halfLength,
                                    number_t halfWidth) {
  const number_t c = std::cos(yaw);
  const number_t s = std::sin(yaw);
  const number_t dx = point[0] - center[0];
  const number_t dy = point[1] - center[1];
  const number_t px = c * dx + s * dy;
  const number_t py = -s * dx + c * dy;
  const number_t qx = std::fabs(px) - halfLength;
  const number_t qy = std::fabs(py) - halfWidth;

  number_t gx = 0.;
  number_t gy = 0.;
  if (qx > 0. && qy > 0.) {  // 角点区域：沿角点的径向外向
    const number_t n = std::sqrt(qx * qx + qy * qy);
    gx = (px >= 0. ? qx : -qx) / n;
    gy = (py >= 0. ? qy : -qy) / n;
  } else if (qx > 0.) {  // 只有 x 向在外
    gx = (px >= 0. ? 1. : -1.);
  } else if (qy > 0.) {  // 只有 y 向在外
    gy = (py >= 0. ? 1. : -1.);
  } else if (qx > qy) {  // 内部：取最近的面
    gx = (px >= 0. ? 1. : -1.);
  } else {
    gy = (py >= 0. ? 1. : -1.);
  }

  return Vector2(c * gx - s * gy, s * gx + c * gy);  // 转回世界系
}

//! 矩形按半径 cornerRadius 膨胀后的轮廓（圆角矩形），用于 RViz 可视化。
//! 每段圆弧采样 samplesPerCorner + 1 个点，四个角依次首尾相接。
inline std::vector<Vector2> roundedRectOutline(const Vector2& center, number_t yaw,
                                               number_t halfLength, number_t halfWidth,
                                               number_t cornerRadius,
                                               int samplesPerCorner = 8) {
  const number_t kPi = std::acos(-1.);
  const int n = std::max(2, samplesPerCorner);
  const number_t cornerX[4] = {halfLength, -halfLength, -halfLength, halfLength};
  const number_t cornerY[4] = {halfWidth, halfWidth, -halfWidth, -halfWidth};
  const number_t baseAngle[4] = {0., kPi / 2., kPi, 3. * kPi / 2.};
  const number_t c = std::cos(yaw);
  const number_t s = std::sin(yaw);

  std::vector<Vector2> outline;
  outline.reserve(static_cast<size_t>(4 * (n + 1)));
  for (int ci = 0; ci < 4; ++ci) {
    for (int k = 0; k <= n; ++k) {
      const number_t a =
          baseAngle[ci] + (kPi / 2.) * static_cast<number_t>(k) / static_cast<number_t>(n);
      const number_t lx = cornerX[ci] + cornerRadius * std::cos(a);
      const number_t ly = cornerY[ci] + cornerRadius * std::sin(a);
      outline.push_back(Vector2(center[0] + c * lx - s * ly, center[1] + s * lx + c * ly));
    }
  }
  return outline;
}

}  // namespace teb
}  // namespace g2o

#endif  // TEB_PLANNER_OBSTACLE_SDF_H
