// 基于 g2o 的 TEB 风格局部轨迹规划器（对外主头文件）。
//
// 轨迹由一串 SE2 位姿构成，通过 smooth / velocity / obstacle / goal 等约束
// 在图优化框架下求解，得到一条平滑、满足速度限制且避开障碍物的局部轨迹。
//
// 本头文件与 ROS 无关；ROS 消息转换见 ros_bridge.h。

#ifndef TEB_PLANNER_PLANNER_H
#define TEB_PLANNER_PLANNER_H

#include <vector>

#include <teb_planner/g2o_compat.h>
#include <teb_planner/obstacle_sdf.h>

#include "g2o/core/sparse_optimizer.h"

namespace g2o {
namespace teb {

//! 二维位姿
struct Pose2D {
  Pose2D() : x(0.), y(0.), theta(0.) {}
  Pose2D(double x_, double y_, double theta_) : x(x_), y(y_), theta(theta_) {}
  double x;
  double y;
  double theta;
};

//! 障碍物形状
enum class Shape {
  Circle,     //!< 圆形：用 center + radius 描述
  Rectangle,  //!< 有向矩形（OBB）：用 center + yaw + halfLength/halfWidth 描述
};

//! 障碍物（圆形或有向矩形，均可为动态、均可为软约束）
struct Obstacle {
  Obstacle()
      : shape(Shape::Circle), center(Vector2::Zero()), velocity(Vector2::Zero()),
        radius(0.), yaw(0.), yawRate(0.), halfLength(0.), halfWidth(0.), dynamic(false),
        soft(false) {}
  //! 圆形障碍物
  Obstacle(double x, double y, double r, bool dyn = false, bool softObstacle = false)
      : shape(Shape::Circle), center(x, y), velocity(Vector2::Zero()), radius(r), yaw(0.),
        yawRate(0.), halfLength(0.), halfWidth(0.), dynamic(dyn), soft(softObstacle) {}

  //! 有向矩形障碍物。注意 halfLength / halfWidth 是**半长 / 半宽**，不是全长 / 全宽
  static Obstacle rect(double x, double y, double halfLength_, double halfWidth_,
                       double yaw_ = 0., bool dyn = false, bool softObstacle = false) {
    Obstacle o;
    o.shape = Shape::Rectangle;
    o.center = Vector2(x, y);
    o.halfLength = halfLength_;
    o.halfWidth = halfWidth_;
    o.yaw = yaw_;
    o.dynamic = dyn;
    o.soft = softObstacle;
    return o;
  }

  //! 按当前速度外推 t 秒后的中心位置
  Vector2 predictCenter(double t) const { return center + velocity * t; }
  //! 按当前角速度外推 t 秒后的朝向
  double predictYaw(double t) const { return yaw + yawRate * t; }

  Shape shape;        //!< 形状
  Vector2 center;     //!< 中心位置
  Vector2 velocity;   //!< 障碍物速度 [m/s]，用于按轨迹时间点预测其位置
  double radius;      //!< 圆形半径 [m]（shape == Circle 时使用）
  double yaw;         //!< 矩形朝向 [rad]（shape == Rectangle 时使用）
  double yawRate;     //!< 矩形角速度 [rad/s]，用于预测朝向
  double halfLength;  //!< 矩形沿局部 x 轴的半长 [m]
  double halfWidth;   //!< 矩形沿局部 y 轴的半宽 [m]
  bool dynamic;       //!< 是否为动态障碍物
  bool soft;          //!< 软障碍（例如其它车辆）：参与优化但不计入可行性判定
};

/**
 * 把一个矩形用一列内接圆覆盖（矩形长边方向等距排布）。
 *
 * 用途：需要把矩形障碍物喂给只支持圆形的下游（或不想改动调用方代码）时，
 * 用 n 个圆近似矩形。返回的圆半径 = sqrt(halfWidth² + (s/2)²)，s = 2·halfLength/n，
 * 保证完整覆盖矩形本身（外扩量约为 s/2 以内）。
 * n 越大越贴合；长条形（长宽比大）建议 n >= 4。
 */
inline std::vector<Obstacle> rectToCircles(double x, double y, double halfLength,
                                           double halfWidth, double yaw = 0., int n = 4,
                                           bool dyn = false, bool softObstacle = false) {
  std::vector<Obstacle> circles;
  if (n < 1) n = 1;
  if (halfLength <= 0. || halfWidth <= 0.) return circles;
  const double cell = 2. * halfLength / n;         // 每个圆负责的格宽
  const double radius = std::sqrt(halfWidth * halfWidth + 0.25 * cell * cell);
  const double c = std::cos(yaw);
  const double s = std::sin(yaw);
  for (int i = 0; i < n; ++i) {
    // 圆心放在每个格的中间，保证两端也只多外扩一点点
    const double localX = -halfLength + (static_cast<double>(i) + 0.5) * cell;
    Obstacle o(x + c * localX, y + s * localX, radius, dyn, softObstacle);
    o.velocity = Vector2::Zero();
    circles.push_back(o);
  }
  return circles;
}

//! 底盘运动学模型
enum class Kinematics {
  Differential,  //!< 差速车：可原地旋转（用 maxOmega 约束角速度）
  Ackermann,     //!< 阿克曼：有最小转弯半径（用轴距 + 最大前轮转角约束曲率）
};

//! 规划器参数
struct TebConfig {
  int poses = 60;              //!< 轨迹上的位姿个数上限（实际个数按目标距离自适应）
  int minPoses = 14;           //!< 轨迹上的位姿个数下限
  double dt = 0.25;            //!< 相邻位姿的时间间隔 [s]
  double maxSpeed = 1.0;       //!< 线速度上限 [m/s]
  double maxOmega = 1.4;       //!< 角速度上限 [rad/s]（仅差速模式生效）
  double robotRadius = 0.32;   //!< 车体半径 [m]
  double safetyMargin = 0.26;  //!< 避障安全余量 [m]

  //! ---- 阿克曼（carlike）参数：kinematics = Ackermann 时生效 ----
  Kinematics kinematics = Kinematics::Differential;  //!< 底盘运动学模型
  double wheelbase = 0.5;          //!< 轴距 [m]
  double maxSteerAngleDeg = 30.0;  //!< 最大前轮转角 [deg]
  //! 最小转弯半径 = 轴距 / tan(最大转角)（求解时按此计算，无需手动填）

  double weightSmooth = 1.0;      //!< 平滑 / 最短路权重
  double weightSmoothRot = 0.5;   //!< 平滑权重（朝向部分）
  double weightVelocity = 20.0;   //!< 速度限制权重
  double weightObstacle = 45.0;   //!< 避障权重
  double weightGoal = 25.0;       //!< 目标点权重
  double weightGoalRot = 2.0;     //!< 目标点权重（朝向部分）
  double weightNonHolonomic = 1.5;//!< 非完整约束权重
  double weightTurnRadius = 40.0; //!< 最小转弯半径权重（阿克曼模式）
  double weightBounds = 15.0;     //!< 地图边界权重
  double weightSoftScale = 0.15;  //!< 软障碍（其它车辆）相对硬障碍的权重系数

  //! 可行区域（规划轨迹会被约束在该矩形内）；不需要边界约束时把权重设 0 即可
  double minX = -8.5, maxX = 8.5, minY = -5.5, maxY = 5.5;

  //! 动态障碍物的预测时长上限 [s]（避免长时间线性外推失真）
  double predictionHorizon = 1.5;

  //! 时间带相对直线距离的余量系数（越大越"松"，绕障空间越大）
  double horizonSlack = 1.35;

  int iterations = 40;         //!< 每次规划的优化迭代次数
};

//! 规划结果
struct PlanResult {
  PlanResult() : ok(false), chi2(0.), minClearance(0.), timeMs(0.), iterations(0), dt(0.) {}
  bool ok;                        //!< 是否得到了可用的轨迹
  std::vector<Pose2D> trajectory; //!< 规划轨迹（第一个点为当前位姿）
  double chi2;                    //!< 优化后的残差
  double minClearance;            //!< 轨迹上最小的避障间距（<0 表示侵入）
  double timeMs;                  //!< 本次规划耗时 [ms]
  int iterations;                 //!< 优化迭代次数
  double dt;                      //!< 本次实际使用的位姿时间间隔 [s]

  //! 逐段控制量（第 i 项对应 trajectory[i] -> trajectory[i+1]）：
  //! 差速 / 阿克曼均填充 cmdV；cmdSteer 仅阿克曼模式填充。
  //! 阿克曼换算：δ = atan(轴距 · ω / v)，已夹到 ±maxSteerAngle。
  std::vector<double> cmdV;      //!< 每段线速度 [m/s]
  std::vector<double> cmdSteer;  //!< 每段前轮转角 [rad]（阿克曼）
};

/**
 * \brief TEB 轨迹规划器
 *
 * 每次调用 plan() 都会构建一个新的 g2o 图并求解。轨迹规模很小（几十个位姿），
 * 重建图比维护图更省心，也避免了上一轮状态污染。单次规划典型耗时 2~5 ms。
 */
class TebPlanner {
 public:
  explicit TebPlanner(const TebConfig& config = TebConfig());

  const TebConfig& config() const { return config_; }
  void setConfig(const TebConfig& config) { config_ = config; }

  //! 根据到目标的距离估算需要的位姿个数（保证位姿足够密集，能描述小障碍物）
  int poseCount(double distanceToGoal) const;

  /**
   * \brief 规划一条从 start 到 goal 的局部轨迹
   * \param start    当前位姿（轨迹第一个点，固定不动）
   * \param goal     目标位姿
   * \param obstacles 需要考虑的障碍物（静态 / 动态 / 其它车辆）
   * \param seed     可选的初始猜测（上一轮轨迹，用于热启动）
   */
  PlanResult plan(const Pose2D& start, const Pose2D& goal,
                  const std::vector<Obstacle>& obstacles,
                  const std::vector<Pose2D>& seed = std::vector<Pose2D>());

 private:
  TebConfig config_;
};

}  // namespace teb
}  // namespace g2o

#endif  // TEB_PLANNER_PLANNER_H
