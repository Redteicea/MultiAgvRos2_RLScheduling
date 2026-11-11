// Copyright 2026
// RL 任务分配推理器（Phase 7）：LibTorch 加载 policy_jit.pt，复刻 Python 训练侧的
// 观测编码与动作掩码，做 argmax 确定性分配。
//
// 与 rl_training/rl_policy.py 的 FleetAllocator.allocate 一一对应：
//   输入  cars[N] + pending[M] + 当前时间
//   输出  actions[N]（0=等待；k(1..M_max)=接 pending[k-1]）
// 观测布局 / 掩码规则见本文件注释与 fleet_env.py 的 _obs_one / get_masks。
#ifndef FLEET_MANAGER__RL_POLICY_HPP_
#define FLEET_MANAGER__RL_POLICY_HPP_

#include <torch/script.h>

#include <string>
#include <vector>

namespace fleet_manager
{

/// @brief 单台 AGV 的推理输入状态（由 FleetManagerNode 从 RobotAgent 映射而来）
struct RLCarState
{
  bool idle = true;        // 是否空闲可接单
  double charge = 100.0;   // 电量 0~100
  double eta = 0.0;        // 完成当前剩余工作还需的秒数（截断 ≤600）
  int target_code = 0;     // 0=无任务 · 1..24=去第 k 个站点(12 港 + 12 货架位) · 25=充电
};

/// @brief 待分配任务（队首窗口用）
struct RLPendingTask
{
  int priority = 2;        // 1 最急
  double wait = 0.0;       // 队列等待秒数
  double deadline = 200.0; // 队列等待上限秒数
  int src_code = 0;        // canonical 站点索引 0..23（0..11 港、12..23 货架位）
  int dst_code = 0;        // canonical 站点索引 0..23（同 src）
};

/// @brief MAPPO 策略推理器
class RLPolicy
{
public:
  // 与训练 env 维度一致（policy.pt 元数据已固化，这里冗余常量便于离线校验）
  // ★ v3 拓扑（2026-09-22）：24 站点（12 港 + 12 货架位）→ 单槽 4+2×24 = 52
  //   (NUM_AGENTS+4) + M_MAX×52 + (NUM_AGENTS-1)×4 + 4 = 7+312+8+4 = **331**
  //   （v2.1 为 235、v2 为 91；单槽 36→52：取/卸各 24 位 canonical one-hot）
  static constexpr int OBS_DIM = 331;
  static constexpr int ACT_DIM = 7;
  static constexpr int NUM_AGENTS = 3;
  static constexpr int M_MAX = 6;

  /// @brief 加载 TorchScript 模型
  /// @param model_path policy_jit.pt 路径
  /// @param n_agents 车队规模（须 = NUM_AGENTS，否则策略维度不匹配）
  explicit RLPolicy(const std::string & model_path, int n_agents = NUM_AGENTS);

  /// @brief 单步决策：外部车/任务状态 → 每车动作
  /// @param cars 每台车状态（顺序即车编号）
  /// @param pending 待分配任务（按 优先级→id 排好序的队首窗口）
  /// @param sim_time 当前仿真/节点时间（秒，用于 time/600 归一化）
  /// @return actions[cars.size()]：0=等待；k≥1 接 pending[k-1]
  std::vector<int> allocate(const std::vector<RLCarState> & cars,
                            const std::vector<RLPendingTask> & pending,
                            double sim_time);

private:
  torch::jit::script::Module model_;
  int n_agents_ = NUM_AGENTS;
};

}  // namespace fleet_manager

#endif  // FLEET_MANAGER__RL_POLICY_HPP_
