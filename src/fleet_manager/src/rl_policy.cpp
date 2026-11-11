// RL 任务分配推理器实现：LibTorch 前向 + 训练一致的观测/掩码编码。
// 观测布局与 rl_training/env/fleet_env.py 的 _obs_one / get_masks 严格一致：
//   A.自身块(N+4) | B.队首任务窗口(M_max×36) | C.他车块((N-1)×4) | D.全局块(4)
//   合计 = 7 + 216 + 8 + 4 = 235 = OBS_DIM
//   ★ v2.1 货架语义化：任务点 8（4 港取/4 港卸）→ 16（8 港 + 8 货架位），
//     单槽 one-hot 12 → 36（取/卸各用 16 位 canonical 索引，与 target_code 同源），
//     目标码域 9 → 17（16 站点 + 充电桩）。改这里必须同步改
//     env/fleet_env.py 与 rl_policy.hpp，否则策略输入错位。
#include "fleet_manager/rl_policy.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace fleet_manager
{

RLPolicy::RLPolicy(const std::string & model_path, int n_agents)
  : n_agents_(n_agents)
{
  if (n_agents_ != NUM_AGENTS) {
    throw std::runtime_error(
      "RLPolicy: 车队规模须为 " + std::to_string(NUM_AGENTS) +
      "（策略按 3 车训练），实际 " + std::to_string(n_agents_));
  }
  try {
    model_ = torch::jit::load(model_path);
  } catch (const c10::Error & e) {
    throw std::runtime_error(
      "RLPolicy: 加载模型失败 " + std::string(e.what()));
  }
  model_.eval();
}

// ★ 与 fleet_env.SITE_COORD 的顺序严格一致（0..11 港、12..23 货架位）
static constexpr int N_SITES_C = 24;                    // 站点数 = len(SITE_COORD)
static constexpr int SLOT_C = 3 + 2 * N_SITES_C + 1;    // 每任务槽 = 52

std::vector<int> RLPolicy::allocate(
  const std::vector<RLCarState> & cars,
  const std::vector<RLPendingTask> & pending,
  double sim_time)
{
  const int N = static_cast<int>(cars.size());
  if (N != n_agents_) {
    throw std::runtime_error("RLPolicy::allocate: 车数不匹配");
  }

  const int M = static_cast<int>(pending.size());
  const int view = std::min(M, M_MAX);   // 队首窗口（最多 M_MAX 个）

  // 全局量
  int n_idle = 0;
  for (const auto & c : cars) {
    if (c.idle) { ++n_idle; }
  }
  double oldest = 0.0;
  for (const auto & t : pending) {
    oldest = std::max(oldest, t.wait);
  }

  // ---- 观测编码（与 _obs_one 严格一致）----
  std::vector<float> obs(static_cast<std::size_t>(N) * OBS_DIM, 0.0f);
  for (int i = 0; i < N; ++i) {
    float * o = obs.data() + static_cast<std::size_t>(i) * OBS_DIM;
    const RLCarState & car = cars[i];

    // A. 自身块：身份 one-hot + idle + 电量 + eta + 目标编码
    o[i] = 1.0f;
    o[NUM_AGENTS + 0] = car.idle ? 1.0f : 0.0f;
    o[NUM_AGENTS + 1] = static_cast<float>(car.charge / 100.0);
    o[NUM_AGENTS + 2] = static_cast<float>(std::min(car.eta, 600.0) / 600.0);
    o[NUM_AGENTS + 3] = static_cast<float>(car.target_code / 25.0);  // 目标码域 [0,25]（= N_SITES+1，N_SITES=24）

    // B. 队首任务窗口（★ v3：每槽 52 维，不足补 0）
    //    [0]存在 [1]等待/600 [2]优先级
    //    [3..26]  取货点 canonical one-hot(24)  [27..50] 卸货点 canonical one-hot(24)
    //    [51] 临近超期
    //    src_code/dst_code 直接是 canonical 索引(0..23)，与 target_code 同源，零映射。
    float * t = o + (NUM_AGENTS + 4);
    for (int k = 0; k < M_MAX; ++k) {
      if (k < view) {
        const RLPendingTask & tk = pending[k];
        t[0] = 1.0f;
        t[1] = static_cast<float>(std::min(tk.wait / 600.0, 1.0));
        t[2] = static_cast<float>((3 - tk.priority) / 2.0);
        for (int s = 0; s < N_SITES_C; ++s) {
          t[3 + s] = (tk.src_code == s) ? 1.0f : 0.0f;
        }
        for (int s = 0; s < N_SITES_C; ++s) {
          t[3 + N_SITES_C + s] = (tk.dst_code == s) ? 1.0f : 0.0f;
        }
        t[3 + 2 * N_SITES_C] = (tk.wait > 0.6 * tk.deadline) ? 1.0f : 0.0f;
      }
      t += SLOT_C;
    }

    // C. 他车块（j != i）
    float * c = t;
    for (int j = 0; j < N; ++j) {
      if (j == i) { continue; }
      const RLCarState & cj = cars[j];
      c[0] = cj.idle ? 0.0f : 1.0f;
      c[1] = static_cast<float>(cj.charge / 100.0);
      c[2] = static_cast<float>(std::min(cj.eta, 600.0) / 600.0);
      c[3] = static_cast<float>(cj.target_code / 25.0);
      c += 4;
    }

    // D. 全局块
    c[0] = static_cast<float>(std::min(sim_time / 600.0, 1.0));
    c[1] = static_cast<float>(n_idle) / static_cast<float>(N);
    c[2] = std::min(static_cast<float>(M) / 12.0f, 1.0f);
    c[3] = static_cast<float>(std::min(oldest / 600.0, 1.0));
  }

  // ---- 动作掩码（与 get_masks 一致）----
  std::vector<float> mask(static_cast<std::size_t>(N) * ACT_DIM, 0.0f);
  for (int i = 0; i < N; ++i) {
    float * m = mask.data() + static_cast<std::size_t>(i) * ACT_DIM;
    m[0] = 1.0f;                       // 等待永远合法
    if (cars[i].idle) {
      for (int k = 0; k < view; ++k) {
        m[k + 1] = 1.0f;               // 仅空闲车可接队首任务
      }
    }
  }

  // ---- 前向 + 掩码 + argmax ----
  auto obs_t = torch::from_blob(
    obs.data(), {static_cast<int64_t>(N), static_cast<int64_t>(OBS_DIM)},
    torch::kFloat32);
  auto mask_t = torch::from_blob(
    mask.data(), {static_cast<int64_t>(N), static_cast<int64_t>(ACT_DIM)},
    torch::kFloat32);

  torch::Tensor logits;
  {
    torch::NoGradGuard no_grad;
    logits = model_.forward({obs_t}).toTensor();
  }
  logits = logits.masked_fill(mask_t < 0.5f, -1e8f);
  auto act = logits.argmax(/*dim=*/1);

  std::vector<int> actions(N);
  auto acc = act.accessor<int64_t, 1>();
  for (int i = 0; i < N; ++i) {
    actions[i] = static_cast<int>(acc[i]);
  }
  return actions;
}

}  // namespace fleet_manager
