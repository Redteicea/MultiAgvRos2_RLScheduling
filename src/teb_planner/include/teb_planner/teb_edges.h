// TEB (Timed Elastic Band) 风格的局部轨迹规划约束。
// 每个约束都以 g2o 的边的形式实现，位姿顶点使用下方自定义的 VertexSE2
// (x, y, theta)，其增量参数为直接相加，因此雅可比可以解析求出。

#ifndef TEB_PLANNER_EDGES_H
#define TEB_PLANNER_EDGES_H

#include <cmath>
#include <iostream>

#include <teb_planner/g2o_compat.h>
#include <teb_planner/obstacle_sdf.h>

#include "g2o/core/base_binary_edge.h"
#include "g2o/core/base_multi_edge.h"
#include "g2o/core/base_unary_edge.h"
#include "g2o/core/base_vertex.h"
#include "g2o/types/slam2d/se2.h"

namespace g2o {
namespace teb {

using Vector1 = VectorN<1>;

/**
 * \brief TEB 专用位姿顶点：SE2 + 加法增量
 *
 * 上游 g2o 的 VertexSE2 用乘法（流形）更新 _estimate = _estimate * SE2(d)，
 * 而本包所有约束边的解析雅可比都按 (dx, dy, dtheta) 直接加推导。
 * 因此这里定义自己的顶点，不依赖任何对 g2o 源码的修改，
 * 在原版 g2o（>= 2020.5，apt 的 libg2o-dev）上即可得到正确的优化结果。
 */
class VertexSE2 : public BaseVertex<3, SE2> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  void setToOriginImpl() override { _estimate = SE2(); }

  void oplusImpl(const number_t* update) override {
    Vector2 t = _estimate.translation();
    t += Eigen::Map<const Vector2>(update);
    const number_t angle = normalize_theta(_estimate.rotation().angle() + update[2]);
    _estimate.setTranslation(t);
    _estimate.setRotation(Rotation2D(angle));
  }

  bool read(std::istream&) override { return false; }
  bool write(std::ostream&) const override { return false; }
};

/**
 * \brief 平滑 / 最短路约束（相邻两个位姿）
 *
 * 误差 = 后一位姿 - 前一位姿（measurement 为 0），即把轨迹"拉紧"，
 * 配合目标点约束后得到的就是一条平滑且尽量短的轨迹。
 */
class EdgeSmooth : public BaseBinaryEdge<3, Vector3, VertexSE2, VertexSE2> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  EdgeSmooth() { _measurement.setZero(); }

  void computeError() override {
    const VertexSE2* from = static_cast<const VertexSE2*>(_vertices[0]);
    const VertexSE2* to = static_cast<const VertexSE2*>(_vertices[1]);
    Vector3 delta = to->estimate().toVector() - from->estimate().toVector();
    delta[2] = normalize_theta(delta[2]);
    _error = delta - _measurement;
  }

  void linearizeOplus() override {
    _jacobianOplusXi = -Matrix3::Identity();
    _jacobianOplusXj = Matrix3::Identity();
  }

  bool read(std::istream&) override { return false; }
  bool write(std::ostream&) const override { return false; }
};

/**
 * \brief 速度 / 角速度上限约束（相邻两个位姿）
 *
 * 误差 = [max(0, 位移 - vmax*dt), max(0, |转角| - wmax*dt)]
 * 属于单边（hinge）约束，满足限制时误差与雅可比均为 0。
 */
class EdgeVelocityLimit : public BaseBinaryEdge<2, Vector2, VertexSE2, VertexSE2> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  EdgeVelocityLimit() {
    _measurement.setZero();
    maxTranslation = 0.;
    maxRotation = 0.;
  }

  void computeError() override {
    const VertexSE2* from = static_cast<const VertexSE2*>(_vertices[0]);
    const VertexSE2* to = static_cast<const VertexSE2*>(_vertices[1]);
    const Vector3 p1 = from->estimate().toVector();
    const Vector3 p2 = to->estimate().toVector();

    const number_t dx = p2[0] - p1[0];
    const number_t dy = p2[1] - p1[1];
    const number_t dist = std::sqrt(dx * dx + dy * dy);
    const number_t dTheta = normalize_theta(p2[2] - p1[2]);

    _error[0] = std::max(cst(0.), dist - maxTranslation);
    _error[1] = std::max(cst(0.), std::fabs(dTheta) - maxRotation);
    _error -= _measurement;
  }

  void linearizeOplus() override {
    const VertexSE2* from = static_cast<const VertexSE2*>(_vertices[0]);
    const VertexSE2* to = static_cast<const VertexSE2*>(_vertices[1]);
    const Vector3 p1 = from->estimate().toVector();
    const Vector3 p2 = to->estimate().toVector();

    _jacobianOplusXi.setZero();
    _jacobianOplusXj.setZero();

    const number_t dx = p2[0] - p1[0];
    const number_t dy = p2[1] - p1[1];
    const number_t dist = std::sqrt(dx * dx + dy * dy);
    if (dist > cst(1e-6) && dist - maxTranslation > 0.) {
      const number_t ux = dx / dist;
      const number_t uy = dy / dist;
      _jacobianOplusXi(0, 0) = -ux;
      _jacobianOplusXi(0, 1) = -uy;
      _jacobianOplusXj(0, 0) = ux;
      _jacobianOplusXj(0, 1) = uy;
    }

    const number_t dTheta = normalize_theta(p2[2] - p1[2]);
    if (std::fabs(dTheta) - maxRotation > 0.) {
      const number_t s = dTheta >= 0. ? cst(1.) : cst(-1.);
      _jacobianOplusXi(1, 2) = -s;
      _jacobianOplusXj(1, 2) = s;
    }
  }

  bool read(std::istream&) override { return false; }
  bool write(std::ostream&) const override { return false; }

  number_t maxTranslation;  //!< vmax * dt
  number_t maxRotation;     //!< wmax * dt
};

/**
 * \brief 圆形障碍物避碰约束（单个位姿）
 *
 * 误差 = max(0, clearance - d)，其中 clearance = 障碍半径 + 车体半径 + 安全余量，
 * d 为位姿到障碍圆心的距离。进入 clearance 内才会产生误差。
 */
class EdgeObstacle : public BaseUnaryEdge<1, Vector1, VertexSE2> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  EdgeObstacle() {
    _measurement.setZero();
    center.setZero();
    clearance = 0.;
  }

  void computeError() override {
    const VertexSE2* v = static_cast<const VertexSE2*>(_vertices[0]);
    const Vector2 p = v->estimate().translation();
    const number_t d = (p - center).norm();
    _error[0] = std::max(cst(0.), clearance - d) - _measurement[0];
  }

  void linearizeOplus() override {
    const VertexSE2* v = static_cast<const VertexSE2*>(_vertices[0]);
    const Vector2 p = v->estimate().translation();
    const Vector2 diff = p - center;
    number_t d = diff.norm();
    _jacobianOplusXi.setZero();
    if (d < clearance) {
      if (d < cst(1e-6)) d = cst(1e-6);
      _jacobianOplusXi(0, 0) = -diff[0] / d;
      _jacobianOplusXi(0, 1) = -diff[1] / d;
    }
  }

  bool read(std::istream&) override { return false; }
  bool write(std::ostream&) const override { return false; }

  Vector2 center;      //!< 障碍物圆心
  number_t clearance;  //!< 允许的最小间距
};

/**
 * \brief 矩形（有向包围盒 / OBB）障碍物约束（单个位姿）
 *
 * 误差 = max(0, clearance - d)，其中 d 是位姿点到矩形的符号距离
 * （外部为正、内部为负），clearance = 车体半径 + 安全余量。
 * 等价于要求位姿点落在"矩形按车体半径膨胀后的圆角矩形"之外。
 *
 * 与 EdgeObstacle 完全同构，只是把"到圆心距离"换成"到 OBB 的符号距离"，
 * 梯度取 rectDistanceGradient()，仍为解析雅可比。
 * 机器人本体按圆形处理（圆角矩形的圆角半径即 clearance）。
 */
class EdgeObstacleRect : public BaseUnaryEdge<1, Vector1, VertexSE2> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  EdgeObstacleRect() {
    _measurement.setZero();
    center.setZero();
    yaw = 0.;
    halfLength = 0.;
    halfWidth = 0.;
    clearance = 0.;
  }

  void computeError() override {
    const VertexSE2* v = static_cast<const VertexSE2*>(_vertices[0]);
    const Vector2 p = v->estimate().translation();
    const number_t d = rectDistance(p, center, yaw, halfLength, halfWidth);
    _error[0] = std::max(cst(0.), clearance - d) - _measurement[0];
  }

  void linearizeOplus() override {
    const VertexSE2* v = static_cast<const VertexSE2*>(_vertices[0]);
    const Vector2 p = v->estimate().translation();
    _jacobianOplusXi.setZero();
    // 只在铰链的"起作用区间"（进入安全距离内）内产生梯度
    if (rectDistance(p, center, yaw, halfLength, halfWidth) < clearance) {
      const Vector2 g = rectDistanceGradient(p, center, yaw, halfLength, halfWidth);
      _jacobianOplusXi(0, 0) = -g[0];
      _jacobianOplusXi(0, 1) = -g[1];
    }
  }

  bool read(std::istream&) override { return false; }
  bool write(std::ostream&) const override { return false; }

  Vector2 center;      //!< 矩形中心
  number_t yaw;        //!< 矩形朝向 [rad]
  number_t halfLength; //!< 沿局部 x 轴的半长 [m]
  number_t halfWidth;  //!< 沿局部 y 轴的半宽 [m]
  number_t clearance;  //!< 允许的最小间距
};

/**
 * \brief 地图边界约束（单个位姿）
 *
 * 误差 = [xmin - x, x - xmax, ymin - y, y - ymax] 中大于 0 的部分，
 * 保证规划轨迹始终留在可行区域内。
 */
class EdgeBounds : public BaseUnaryEdge<4, Vector4, VertexSE2> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  EdgeBounds() { _measurement.setZero(); }

  void computeError() override {
    const VertexSE2* v = static_cast<const VertexSE2*>(_vertices[0]);
    const Vector2 p = v->estimate().translation();
    _error[0] = std::max(cst(0.), minX - p[0]);
    _error[1] = std::max(cst(0.), p[0] - maxX);
    _error[2] = std::max(cst(0.), minY - p[1]);
    _error[3] = std::max(cst(0.), p[1] - maxY);
    _error -= _measurement;
  }

  void linearizeOplus() override {
    const VertexSE2* v = static_cast<const VertexSE2*>(_vertices[0]);
    const Vector2 p = v->estimate().translation();
    _jacobianOplusXi.setZero();
    if (minX - p[0] > 0.) _jacobianOplusXi(0, 0) = -1.;
    if (p[0] - maxX > 0.) _jacobianOplusXi(1, 0) = 1.;
    if (minY - p[1] > 0.) _jacobianOplusXi(2, 1) = -1.;
    if (p[1] - maxY > 0.) _jacobianOplusXi(3, 1) = 1.;
  }

  bool read(std::istream&) override { return false; }
  bool write(std::ostream&) const override { return false; }

  number_t minX = 0., maxX = 0., minY = 0., maxY = 0.;
};

/**
 * \brief 目标点约束（作用在最后一个位姿上）
 *
 * 误差 = 目标位姿 - 当前位姿（角度做归一化）。用软约束而非固定顶点，
 * 这样避障约束可以把终点稍微推开，避免无解时的刚性冲突。
 */
class EdgeGoal : public BaseUnaryEdge<3, Vector3, VertexSE2> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  EdgeGoal() { _measurement.setZero(); }

  void computeError() override {
    const VertexSE2* v = static_cast<const VertexSE2*>(_vertices[0]);
    Vector3 e = _measurement - v->estimate().toVector();
    e[2] = normalize_theta(e[2]);
    _error = e;
  }

  void linearizeOplus() override { _jacobianOplusXi = -Matrix3::Identity(); }

  bool read(std::istream&) override { return false; }
  bool write(std::ostream&) const override { return false; }
};

/**
 * \brief 非完整约束：抑制侧向滑动（相邻两个位姿）
 *
 * 误差 = 位移在"车体侧向"上的投影，使轨迹更接近真实差速/阿克曼车辆的运动。
 */
class EdgeNonHolonomic : public BaseBinaryEdge<1, Vector1, VertexSE2, VertexSE2> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  EdgeNonHolonomic() { _measurement.setZero(); }

  void computeError() override {
    const VertexSE2* from = static_cast<const VertexSE2*>(_vertices[0]);
    const VertexSE2* to = static_cast<const VertexSE2*>(_vertices[1]);
    const Vector3 p1 = from->estimate().toVector();
    const Vector3 p2 = to->estimate().toVector();
    const number_t s = std::sin(p1[2]);
    const number_t c = std::cos(p1[2]);
    const number_t lateral = -s * (p2[0] - p1[0]) + c * (p2[1] - p1[1]);
    _error[0] = lateral - _measurement[0];
  }

  void linearizeOplus() override {
    const VertexSE2* from = static_cast<const VertexSE2*>(_vertices[0]);
    const VertexSE2* to = static_cast<const VertexSE2*>(_vertices[1]);
    const Vector3 p1 = from->estimate().toVector();
    const Vector3 p2 = to->estimate().toVector();
    const number_t s = std::sin(p1[2]);
    const number_t c = std::cos(p1[2]);

    _jacobianOplusXi(0, 0) = s;
    _jacobianOplusXi(0, 1) = -c;
    _jacobianOplusXi(0, 2) = -(c * (p2[0] - p1[0]) + s * (p2[1] - p1[1]));
    _jacobianOplusXj(0, 0) = -s;
    _jacobianOplusXj(0, 1) = c;
    _jacobianOplusXj(0, 2) = 0.;
  }

  bool read(std::istream&) override { return false; }
  bool write(std::ostream&) const override { return false; }
};

/**
 * \brief 最小转弯半径约束（阿克曼 / carlike，连续三个位姿）
 *
 * 阿克曼底盘的转弯半径有下限 R_min = L / tan(δ_max)（L 为轴距，δ_max 为
 * 最大前轮转角），不能像差速车那样原地转。取连续三个位姿确定的外接圆
 * 半径作为转弯半径的度量：
 *
 *   R = |AB|·|BC|·|CA| / (4·S)，S 为三点围成的三角形面积
 *
 * 误差 = max(0, R_min - R)，单边（hinge）约束：转弯半径足够大时为 0，
 * 弯过急时把三个位姿"掰开"。三点近似共线时 R -> ∞，误差自然为 0。
 *
 * 雅可比用数值差分（三点外接圆的解析导数过于繁琐，且轨迹只有几十个
 * 位姿，数值差分的开销可忽略）。
 */
class EdgeTurnRadius : public BaseMultiEdge<1, Vector1> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  EdgeTurnRadius() {
    _measurement.setZero();
    minRadius = 0.;
    resize(3);  // 连接三个位姿顶点
  }

  void computeError() override {
    const VertexSE2* a = static_cast<const VertexSE2*>(_vertices[0]);
    const VertexSE2* b = static_cast<const VertexSE2*>(_vertices[1]);
    const VertexSE2* c = static_cast<const VertexSE2*>(_vertices[2]);
    const Vector2 pa = a->estimate().translation();
    const Vector2 pb = b->estimate().translation();
    const Vector2 pc = c->estimate().translation();

    const number_t ab = (pb - pa).norm();
    const number_t bc = (pc - pb).norm();
    const number_t ca = (pa - pc).norm();
    if (ab < cst(1e-6) || bc < cst(1e-6) || ca < cst(1e-6)) {
      _error[0] = -_measurement[0];
      return;
    }
    const Vector2 u = pb - pa;
    const Vector2 v = pc - pa;
    const number_t area2 = std::fabs(u[0] * v[1] - u[1] * v[0]);  // 2*面积
    if (area2 < cst(1e-9)) {
      // 三点共线：转弯半径无穷大，约束满足
      _error[0] = -_measurement[0];
      return;
    }
    const number_t radius = ab * bc * ca / (2. * area2);
    _error[0] = std::max(cst(0.), minRadius - radius) - _measurement[0];
  }

  bool read(std::istream&) override { return false; }
  bool write(std::ostream&) const override { return false; }

  number_t minRadius;  //!< 最小转弯半径 [m]
};

}  // namespace teb
}  // namespace g2o

#endif  // TEB_PLANNER_EDGES_H
