// 基于 g2o 的 TEB 风格局部轨迹规划器实现

#include <teb_planner/teb_planner.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>

#include <teb_planner/teb_edges.h>

#include "g2o/core/block_solver.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/core/solver.h"
#include "g2o/solvers/csparse/linear_solver_csparse.h"
#include "g2o/stuff/misc.h"
#include "g2o/types/slam2d/se2.h"

namespace g2o {
namespace teb {

namespace {
double clampValue(double value, double lo, double hi) {
  return std::max(lo, std::min(hi, value));
}

/**
 * 生成初始猜测：优先用上一轮轨迹热启动，否则线性插值；
 * 之后把落在障碍物内的点沿径向推出，并夹到可行区域内。
 * 这一步能避免"直线穿过障碍中心"这种让优化器很难收敛的初值。
 */
Pose2D initialPose(const Pose2D& start, const Pose2D& goal,
                   const std::vector<Pose2D>& seed, int index, int count,
                   const std::vector<Obstacle>& obstacles, const TebConfig& config) {
  Pose2D p;
  if (static_cast<int>(seed.size()) == count) {
    p = seed[index];
  } else {
    const number_t t = count > 1 ? static_cast<number_t>(index) / (count - 1) : 0.;
    p = Pose2D(start.x + t * (goal.x - start.x), start.y + t * (goal.y - start.y),
               start.theta + t * normalize_theta(goal.theta - start.theta));
  }

  // 沿"直线方向的法向"把点推到障碍物的同一侧，形成一个初步的绕行弧线；
  // 比逐个径向推出更容易让优化器收敛。
  Vector2 lineDirection(goal.x - start.x, goal.y - start.y);
  const double lineLength = lineDirection.norm();
  if (lineLength > 1e-6) {
    lineDirection /= lineLength;
  } else {
    lineDirection = Vector2(1., 0.);
  }
  const Vector2 perpendicular(-lineDirection[1], lineDirection[0]);

  for (size_t k = 0; k < obstacles.size(); ++k) {
    const Obstacle& obstacle = obstacles[k];

    // 矩形障碍物：沿"到矩形表面的最速下降方向"（SDF 梯度）推出。
    // 长墙、横向挡板都能被正确推开，而"沿直线法向推"对横向挡板无效。
    if (obstacle.shape == Shape::Rectangle) {
      const double minDistance = config.robotRadius + config.safetyMargin + 0.05;
      const Vector2 here(p.x, p.y);
      const double d = rectDistance(here, obstacle.center, obstacle.yaw,
                                    obstacle.halfLength, obstacle.halfWidth);
      if (d < minDistance) {
        const Vector2 g = rectDistanceGradient(here, obstacle.center, obstacle.yaw,
                                               obstacle.halfLength, obstacle.halfWidth);
        const double push = minDistance - d;
        p.x += g[0] * push;
        p.y += g[1] * push;
      }
      continue;
    }

    const Vector2 diff(p.x - obstacle.center[0], p.y - obstacle.center[1]);
    const double d = diff.norm();
    const double minDistance =
        obstacle.radius + config.robotRadius + config.safetyMargin + 0.05;
    if (d < minDistance) {
      // 障碍物位于直线的哪一侧，就把轨迹推到另一侧
      const Vector2 toObstacle = obstacle.center - Vector2(start.x, start.y);
      double side = toObstacle[0] * perpendicular[0] + toObstacle[1] * perpendicular[1];
      if (std::fabs(side) < 1e-6) side = 1.;
      const double push = (minDistance - d) * (side > 0. ? -1. : 1.);
      p.x += perpendicular[0] * push;
      p.y += perpendicular[1] * push;
    }
  }

  // 注意：起点（index 0）不走这条路径——见 plan() 中的说明。
  p.x = clampValue(p.x, config.minX, config.maxX);
  p.y = clampValue(p.y, config.minY, config.maxY);
  return p;
}
}  // namespace

TebPlanner::TebPlanner(const TebConfig& config) : config_(config) {
  // 阿克曼参数的"单位陷阱"防护：maxSteerAngleDeg 必须填【角度】。
  // 若误填弧度（例如 0.52 代表 30°），R_min 会算成几十米，轨迹被压成直线
  // 却不会报错——这里显式告警，避免静默地把规划结果做废。
  if (config_.kinematics == Kinematics::Ackermann) {
    const double kPi = std::acos(-1.);
    if (config_.wheelbase <= 0.) {
      std::cerr << "[teb_planner] WARNING: wheelbase must be > 0 m, got "
                << config_.wheelbase << " m\n";
    }
    if (config_.maxSteerAngleDeg <= 0. || config_.maxSteerAngleDeg >= 90.) {
      std::cerr << "[teb_planner] WARNING: maxSteerAngleDeg must be in (0, 90) deg, got "
                << config_.maxSteerAngleDeg
                << " deg. Did you pass radians? (30 deg = 0.5236 rad)\n";
    } else if (config_.maxSteerAngleDeg < 5.) {
      std::cerr << "[teb_planner] WARNING: maxSteerAngleDeg = "
                << config_.maxSteerAngleDeg
                << " deg looks too small - it should be a DEGREE value "
                   "(30 deg = 0.5236 rad), giving R_min = "
                << config_.wheelbase / std::tan(config_.maxSteerAngleDeg * kPi / 180.)
                << " m\n";
    }
  }
}

int TebPlanner::poseCount(double distanceToGoal) const {
  // 每个位姿最多前进 maxSpeed*dt，再留余量用于绕障
  const double spacing = std::max(1e-3, config_.maxSpeed * config_.dt);
  const int estimated =
      static_cast<int>(std::ceil(distanceToGoal / spacing * config_.horizonSlack)) + 2;
  return std::max(config_.minPoses, std::min(config_.poses, estimated));
}

PlanResult TebPlanner::plan(const Pose2D& start, const Pose2D& goal,
                            const std::vector<Obstacle>& obstacles,
                            const std::vector<Pose2D>& seed) {
  PlanResult result;
  result.iterations = config_.iterations;
  const std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();

  // 线性求解器 -> 块求解器 -> LM 算法
  auto linearSolver = g2o::make_unique<LinearSolverCSparse<BlockSolverX::PoseMatrixType>>();
  auto blockSolver = g2o::make_unique<BlockSolverX>(std::move(linearSolver));
  OptimizationAlgorithmLevenberg* algorithm =
      new OptimizationAlgorithmLevenberg(std::move(blockSolver));

  SparseOptimizer optimizer;
  optimizer.setVerbose(false);
  optimizer.setAlgorithm(algorithm);

  // 0) 自适应时间带：让轨迹长度略大于到目标的距离，留出绕障的余量。
  //    若时间带过短（小于到目标的距离），轨迹会被拉成一条无法绕障的直线。
  const double distanceToGoal =
      std::sqrt((goal.x - start.x) * (goal.x - start.x) +
                (goal.y - start.y) * (goal.y - start.y));

  const int n = poseCount(distanceToGoal);
  const double dt = config_.dt;
  result.dt = dt;

  // 1) 位姿顶点
  std::vector<VertexSE2*> vertices;
  vertices.reserve(n);
  for (int i = 0; i < n; ++i) {
    VertexSE2* v = new VertexSE2();
    v->setId(i);
    if (i == 0) {
      // 起点是固定顶点，必须严格等于车辆当前位姿：
      // 一旦它也被"推出障碍物"或夹到边界，整条轨迹就会整体平移，
      // 跟踪时车辆实际上跑在轨迹起点之外，很容易擦到障碍物。
      v->setEstimate(SE2(start.x, start.y, start.theta));
    } else {
      const Pose2D p = initialPose(start, goal, seed, i, n, obstacles, config_);
      v->setEstimate(SE2(p.x, p.y, p.theta));
    }
    optimizer.addVertex(v);
    vertices.push_back(v);
  }
  vertices.front()->setFixed(true);  // 起点固定为车辆当前位姿

  // 2) 相邻位姿约束：平滑 + 速度限制 + 非完整
  const bool carlike = config_.kinematics == Kinematics::Ackermann;
  const number_t maxTranslation = config_.maxSpeed * dt;
  const number_t maxRotation = config_.maxOmega * dt;
  // 阿克曼的最小转弯半径（自行车模型：R = L / tan(δ_max)）
  const double kPi = std::acos(-1.);
  const double minTurnRadius =
      config_.wheelbase / std::tan(config_.maxSteerAngleDeg * kPi / 180.);

  Matrix3 infoSmooth = Matrix3::Zero();
  infoSmooth(0, 0) = config_.weightSmooth;
  infoSmooth(1, 1) = config_.weightSmooth;
  infoSmooth(2, 2) = config_.weightSmoothRot;

  Matrix2 infoVelocity = Matrix2::Zero();
  infoVelocity(0, 0) = config_.weightVelocity;
  // 差速车才用"角速度上限"约束；阿克曼的正确约束是最小转弯半径边，
  // 若同时保留 ω 限幅，低速大曲率的合理轨迹会被误罚。
  infoVelocity(1, 1) = carlike ? 0. : config_.weightVelocity;

  for (int i = 0; i + 1 < n; ++i) {
    EdgeSmooth* smooth = new EdgeSmooth();
    smooth->setMeasurement(Vector3::Zero());
    smooth->setInformation(infoSmooth);
    smooth->vertices()[0] = vertices[i];
    smooth->vertices()[1] = vertices[i + 1];
    optimizer.addEdge(smooth);

    EdgeVelocityLimit* velocity = new EdgeVelocityLimit();
    velocity->maxTranslation = maxTranslation;
    velocity->maxRotation = maxRotation;
    velocity->setMeasurement(Vector2::Zero());
    velocity->setInformation(infoVelocity);
    velocity->vertices()[0] = vertices[i];
    velocity->vertices()[1] = vertices[i + 1];
    optimizer.addEdge(velocity);

    if (config_.weightNonHolonomic > 0.) {
      EdgeNonHolonomic* nonHolo = new EdgeNonHolonomic();
      nonHolo->setMeasurement(Vector1::Zero());
      Eigen::Matrix<number_t, 1, 1> infoNonHolo;
      infoNonHolo(0, 0) = config_.weightNonHolonomic;
      nonHolo->setInformation(infoNonHolo);
      nonHolo->vertices()[0] = vertices[i];
      nonHolo->vertices()[1] = vertices[i + 1];
      optimizer.addEdge(nonHolo);
    }
  }

  // 2.5) 阿克曼：最小转弯半径约束（连续三个位姿的外接圆半径 >= R_min）
  if (carlike && config_.weightTurnRadius > 0.) {
    Eigen::Matrix<number_t, 1, 1> infoTurn;
    infoTurn(0, 0) = config_.weightTurnRadius;
    for (int i = 0; i + 2 < n; ++i) {
      EdgeTurnRadius* turn = new EdgeTurnRadius();
      turn->minRadius = minTurnRadius;
      turn->setMeasurement(Vector1::Zero());
      turn->setInformation(infoTurn);
      turn->vertices()[0] = vertices[i];
      turn->vertices()[1] = vertices[i + 1];
      turn->vertices()[2] = vertices[i + 2];
      optimizer.addEdge(turn);
    }
  }

  // 3) 避障约束：每个位姿对每种障碍物
  //    动态障碍物按各自的速度（矩形还有角速度）外推到该位姿对应的时刻，
  //    实现"提前避让"
  for (int i = 1; i < n; ++i) {  // 起点固定，无需避障边
    const number_t predictionTime = std::min(i * dt, config_.predictionHorizon);
    for (size_t k = 0; k < obstacles.size(); ++k) {
      const Obstacle& obstacle = obstacles[k];
      // 软障碍（其它车辆）权重更低：宁可互相靠近，也不要被挤进真正的障碍物
      Eigen::Matrix<number_t, 1, 1> infoObstacle;
      infoObstacle(0, 0) =
          config_.weightObstacle * (obstacle.soft ? config_.weightSoftScale : 1.);

      if (obstacle.shape == Shape::Rectangle) {
        EdgeObstacleRect* edge = new EdgeObstacleRect();
        edge->center = obstacle.predictCenter(predictionTime);
        edge->yaw = obstacle.predictYaw(predictionTime);
        edge->halfLength = obstacle.halfLength;
        edge->halfWidth = obstacle.halfWidth;
        edge->clearance = config_.robotRadius + config_.safetyMargin;
        edge->setMeasurement(Vector1::Zero());
        edge->setInformation(infoObstacle);
        edge->vertices()[0] = vertices[i];
        optimizer.addEdge(edge);
        continue;
      }

      EdgeObstacle* edge = new EdgeObstacle();
      edge->center = obstacle.predictCenter(predictionTime);
      edge->clearance = obstacle.radius + config_.robotRadius + config_.safetyMargin;
      edge->setMeasurement(Vector1::Zero());
      edge->setInformation(infoObstacle);
      edge->vertices()[0] = vertices[i];
      optimizer.addEdge(edge);
    }
  }

  // 4) 边界约束：把轨迹限制在可行区域内
  Eigen::Matrix<number_t, 4, 4> infoBounds = Eigen::Matrix<number_t, 4, 4>::Zero();
  infoBounds(0, 0) = config_.weightBounds;
  infoBounds(1, 1) = config_.weightBounds;
  infoBounds(2, 2) = config_.weightBounds;
  infoBounds(3, 3) = config_.weightBounds;
  if (config_.weightBounds > 0.) {
    for (int i = 1; i < n; ++i) {
      EdgeBounds* bounds = new EdgeBounds();
      bounds->minX = config_.minX;
      bounds->maxX = config_.maxX;
      bounds->minY = config_.minY;
      bounds->maxY = config_.maxY;
      bounds->setMeasurement(Vector4::Zero());
      bounds->setInformation(infoBounds);
      bounds->vertices()[0] = vertices[i];
      optimizer.addEdge(bounds);
    }
  }

  // 5) 目标点约束（软约束，作用在最后一个位姿）
  Matrix3 infoGoal = Matrix3::Zero();
  infoGoal(0, 0) = config_.weightGoal;
  infoGoal(1, 1) = config_.weightGoal;
  infoGoal(2, 2) = config_.weightGoalRot;
  EdgeGoal* goalEdge = new EdgeGoal();
  goalEdge->setMeasurement(Vector3(goal.x, goal.y, goal.theta));
  goalEdge->setInformation(infoGoal);
  goalEdge->vertices()[0] = vertices.back();
  optimizer.addEdge(goalEdge);

  // 6) 求解
  optimizer.initializeOptimization();
  const int performed = optimizer.optimize(config_.iterations);
  result.chi2 = optimizer.chi2();
  result.iterations = performed;

  // 7) 提取轨迹并检查可行性
  result.trajectory.reserve(n);
  bool finite = true;
  for (int i = 0; i < n; ++i) {
    const Vector3 p = vertices[i]->estimate().toVector();
    if (!std::isfinite(p[0]) || !std::isfinite(p[1]) || !std::isfinite(p[2])) finite = false;
    result.trajectory.push_back(Pose2D(p[0], p[1], normalize_theta(p[2])));
  }

  // 可行性只看"硬"障碍（静态 / 动态障碍物）；其它车辆属于软约束，
  // 车辆对顶时本来就可能无解，不应该判定为规划失败。
  // 第 0 个位姿是固定不可优化的当前位姿，不计入指标：
  // 车辆已经贴近障碍物时它必然是负值，但那不是这次规划的"决策"。
  result.minClearance = std::numeric_limits<double>::max();
  for (size_t i = 1; i < result.trajectory.size(); ++i) {
    const Pose2D& p = result.trajectory[i];
    const Vector2 point(p.x, p.y);
    for (size_t k = 0; k < obstacles.size(); ++k) {
      if (obstacles[k].soft) continue;
      const double predictionTime = std::min(i * dt, config_.predictionHorizon);
      double clearance = 0.;
      if (obstacles[k].shape == Shape::Rectangle) {
        // 矩形：符号距离 - 车体半径（= 到"圆角矩形"边界的余量）
        clearance = rectDistance(point, obstacles[k].predictCenter(predictionTime),
                                 obstacles[k].predictYaw(predictionTime),
                                 obstacles[k].halfLength, obstacles[k].halfWidth) -
                    config_.robotRadius;
      } else {
        const Vector2 predicted = obstacles[k].predictCenter(predictionTime);
        const Vector2 diff(p.x - predicted[0], p.y - predicted[1]);
        clearance =
            diff.norm() - obstacles[k].radius - config_.robotRadius;
      }
      result.minClearance = std::min(result.minClearance, clearance);
    }
  }
  if (result.minClearance == std::numeric_limits<double>::max()) result.minClearance = 0.;

  // 8) 逐段控制量：v = 段长 / dt；阿克曼模式下按自行车模型反解前轮转角
  //    δ = atan(L · ω / v)，并夹到 ±最大转角。下游控制器可直接使用。
  const double maxSteerRad = config_.maxSteerAngleDeg * kPi / 180.;
  result.cmdV.clear();
  result.cmdSteer.clear();
  for (size_t i = 0; i + 1 < result.trajectory.size(); ++i) {
    const Pose2D& a = result.trajectory[i];
    const Pose2D& b = result.trajectory[i + 1];
    const double dist =
        std::sqrt((b.x - a.x) * (b.x - a.x) + (b.y - a.y) * (b.y - a.y));
    const double v = dist / dt;
    result.cmdV.push_back(v);
    if (carlike) {
      const double omega = normalize_theta(b.theta - a.theta) / dt;
      double steer =
          std::atan2(config_.wheelbase * omega, std::max(v, 0.05));
      steer = clampValue(steer, -maxSteerRad, maxSteerRad);
      result.cmdSteer.push_back(steer);
    }
  }

  const std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
  result.timeMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
  result.ok = finite && !result.trajectory.empty();

  return result;
}

}  // namespace teb
}  // namespace g2o
