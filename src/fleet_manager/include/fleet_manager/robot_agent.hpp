// Copyright 2026
// 单个 AGV 智能体：状态机 + 当前任务 + 规划路径 + 控制器输出。
#ifndef FLEET_MANAGER__ROBOT_AGENT_HPP_
#define FLEET_MANAGER__ROBOT_AGENT_HPP_

#include "warehouse_map.hpp"
#include "data_type/robot_state.hpp"

#include <memory>
#include <deque>
#include <string>
#include <vector>

namespace fleet_manager
{
  /// @brief 机器人智能体
  class RobotAgent
  {
  public:
    int id = -1;
    std::string name;

    // ---- 世界系位姿（来自 /fleet_sim/model_states）----
    double x = 0.0, y = 0.0, theta = 0.0;
    bool has_pose = false;        // 是否已收到过有效位姿数据定义
    double last_pose_time = 0.0;

    // ---- 电量（模拟）----
    double battery = 100.0;

    // ---- 状态机 ----
    RobotState state = RobotState::OFFLINE;
    RobotState resume_state = RobotState::IDLE;  // YIELD 结束后恢复的状态
    double state_time = 0.0;      // 当前状态持续时间
    int charger_id = -1;          // 分配的充电桩

    // ---- 当前导航目标 ----
    double goal_x = 0.0, goal_y = 0.0;
    bool has_new_goal = false;

    // ---- 任务 ----
    std::shared_ptr<TaskSpec> task;   // 当前任务（任务列表中的共享指针）

    // ---- 规划路径 ----
    std::deque<Cell> path;                    // 完整栅格路径（含起终点）
    std::vector<std::pair<double, double>> waypoints;  // 平滑后航点（世界坐标）
    std::size_t wp_idx = 0;                   // 当前航点下标
    Cell current_cell;

    // ---- 行驶状态 ----
    double leg_start_x = 0.0, leg_start_y = 0.0;  // 当前导航段起点
    double last_x = 0.0, last_y = 0.0;            // 上一控制周期的实际位置（用于卡死检测）
    bool have_last = false;                       // 实际位置采样是否已初始化
    double blocked_time = 0.0;                    // 连续被阻挡时间
    double blocked_start_x = 0.0, blocked_start_y = 0.0;  // 被阻挡起始位置
    bool stuck = false;                           // 死锁标记（待解算）
    double yield_distance = 0.0;                  // 避让已倒退距离
    double yield_target = 0.0;                    // 避让目标倒退距离

    // ---- 实际速度（位姿差分，供监控显示）----
    double lin_vel = 0.0, ang_vel = 0.0;          // 实际线速度(m/s) / 角速度(rad/s)
    double vx = 0.0, vy = 0.0;                   // 世界系线速度分量(m/s)——TEB 动态障碍预测用
    double prev_x = 0.0, prev_y = 0.0, prev_theta = 0.0;  // 上一周期位姿
    bool has_prev_pose = false;                   // 位姿差分是否已初始化

    // ---- 三点掉头（四驱阿克曼不能原地旋转）----
    int turn_phase = 0;                           // 0=否 1=前进转向 2=倒车反向 3=前进摆正
    double turn_start_yaw = 0.0;                  // 掉头起点 yaw
    double turn_sign = 1.0;                       // 车头转向方向：+1 左转 / -1 右转

    // ---- 控制输出 ----
    double cmd_v = 0.0, cmd_w = 0.0;

    bool isMovingState() const
    {
      return state == RobotState::NAV_PICKUP ||
            state == RobotState::NAV_DROPOFF ||
            state == RobotState::TO_CHARGER;
    }

    bool hasTask() const { return task != nullptr && task->active; }

    double distanceTo(double px, double py) const
    {
      return std::hypot(x - px, y - py);
    }

    /// 当前任务优先级（无任务视为最低优先级）
    int priority() const
    {
      if (task != nullptr && task->active) { return task->priority; }
      return 999;
    }
  };

}  // namespace fleet_manager
#endif  // FLEET_MANAGER__ROBOT_AGENT_HPP_
