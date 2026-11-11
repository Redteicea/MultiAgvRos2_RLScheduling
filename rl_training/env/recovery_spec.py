# -*- coding: utf-8 -*-
"""
recovery_spec.py —— 「卡死恢复原语选择器」的观测/动作规范（训练与部署的**唯一来源**）
================================================================================

为什么单独成文
--------------
本仓库反复强调"训练/推理零偏差"（见 rl_policy.py 头部）。恢复策略的观测向量若在
训练侧（recovery_env.py）与部署侧（src/fleet_sim/scripts/rl_recovery.py）各写一份，
迟早漂移，且漂移后表现为"策略在现场乱选原语"，极难排查。
故把三件事集中在此文件，两侧都 import 它：

  1. PRIMITIVES  —— 离散原语清单（动作空间的定义）
  2. FEATURES    —— 观测段定义（顺序 + 归一化）
  3. build_obs() —— 由"特征字典"构造观测向量

设计要点
--------
· **纯标准库**（不依赖 numpy）：nav 节点一侧刻意保持"无 numpy"（见 custom_nav.py
  头部"依赖：rclpy + 标准库"），本文件若用 numpy 会把这个性质破坏掉。
  返回的是普通 list，torch.as_tensor(list) 可直接吃。
· **归一化**：所有量都压到大致 [-1,1]/[0,1]，让小 MLP 收敛快、对量纲不敏感。
· **观测只含"从 ros 侧便宜可得"的量**：全部可由 custom_nav 现有状态/工具函数算出
  （_obstacle_bins / _clear_at / _first_seg_turn / _neighbors / _stuck_accum ...），
  不引入新的感知依赖。

为什么是"离散原语"而不是"连续 v,ω"
-----------------------------------
实测结论（见 DEVELOPMENT_NOTES §7.9/§8.6）：在本仓库"阿克曼 + 窄巷道 + 频繁掉头"
的现场，失败模式**不是"速度不够好"，而是"卡住/不会掉头"**——即"该选哪个动作"的
离散决策。而 custom_nav 的控制循环本身就是一棵**手写策略树**
（_control_loop 的 10 级优先级链），RL 在这里是"**学那棵树**"，
而不是替代几何：原语本身仍是有界、可验证的几何动作。
"""

# ---------------------------------------------------------------------------
# 一、动作空间：6 个离散原语
# ---------------------------------------------------------------------------
# 索引与含义（与 rl_recovery.py / recovery_env.py 严格一致，勿改顺序）：
#   0 follow       继续跟随（不干预，交回主流程）——"其实没卡住"时的正解
#   1 wait         原地停车等待让行（前车会走 / 障碍会移开）
#   2 back         低速直退 0.9s（基类已有的 _start_escape mode='back'）
#   3 spin         原地转向 0.9s（★ 阿克曼上**无效**：插件 steer=atan(w·L/v)，
#                  v=0 → speed=0，车根本不动。保留它是为了让策略**学会别选它**，
#                  并在日志里暴露"手写规则一直在浪费这个原语"的证据）
#   4 three_point  ★ 三点掉头（前弧→倒弧→前弧），本仓库 nav 侧此前**完全没有**，
#                  由 rl_recovery.py 新增；这正是"巷道里掉头 180°"的唯一解
#   5 replan       请求局部 A* 重规划（注入邻车圆盘 + auto_via 分段绕行）
PRIMITIVES = ["follow", "wait", "back", "spin", "three_point", "replan"]
N_ACTIONS = len(PRIMITIVES)
IDX_FOLLOW, IDX_WAIT, IDX_BACK, IDX_SPIN, IDX_TP, IDX_REPLAN = range(N_ACTIONS)

# ---------------------------------------------------------------------------
# 二、观测空间：19 维
# ---------------------------------------------------------------------------
# 段布局（顺序即拼接顺序，两侧必须一致）：
#   [0]     stuck_t     连续"有指令无位移"累计秒  /3.0
#   [1]     cmd_v       上次下发线速度            /v_max(0.6)
#   [2]     cmd_w       上次下发角速度            /w_max(1.5)
#   [3]     goal_sin    目标方向(车体系) sin      —— 是否需要掉头的核心信息
#   [4]     goal_cos    目标方向(车体系) cos
#   [5]     front       正前方净空(m)             /2.0
#   [6]     back        正后方净空(m)             /2.0  —— 决定"能否倒车"
#   [7]     left        左侧(±90°)净空            /2.0
#   [8]     right       右侧(±90°)净空            /2.0
#   [9]     turn        path_world 首段相对车头夹角 /π   —— >100° 即"需要掉头"
#   [10]    nb_dist     最近邻车距离(m)           /3.0（无邻车记 1.0）
#   [11]    remain      到目标剩余距离(m)         /12.0
#   [12]    need_uturn  0/1：|turn|>100°（路径要求近乎掉头）
#   [13:19] prev_onehot 上一 tick 所选原语 one-hot(6) —— 允许学"升级/别反复横跳"
#   [19]    repeat      同一原语"已连续被选"的次数 /2（截 1.0）：0=首拍、0.5=第 2 拍
#                      依据：现场影子数据 138 个卡死事件里，规则与策略都出现
#                      "同一个动作连选 6 拍"（follow×6 62 次 / back×6 103 次 /
#                      three_point×6 39 次），75% 的事件以"6 拍上限"收尾 = 谁都没脱困。
FEATURES = [
    "stuck_t", "cmd_v", "cmd_w", "goal_sin", "goal_cos",
    "front", "back", "left", "right", "turn",
    "nb_dist", "remain", "need_uturn",
    "prev_follow", "prev_wait", "prev_back", "prev_spin", "prev_tp", "prev_replan",
    "repeat",
]
OBS_DIM = len(FEATURES)          # = 19

# 归一化除数（与上面注释一一对应；改这里等于改观测定义，须同时重训）
SCALE = {
    "stuck_t": 3.0, "cmd_v": 0.6, "cmd_w": 1.5,
    "front": 2.0, "back": 2.0, "left": 2.0, "right": 2.0,
    "nb_dist": 3.0, "remain": 12.0, "turn": 3.141592653589793,
}
# 路径首段转角超过该值 → 判定"需要掉头"（与 custom_nav._neighbor_detour 里
# "首段 >100° 判定为掉头不可执行"同一判据，保持语义一致）
UTURN_RAD = 1.7453292519943296   # 100°


def _clip(v, lo=-1.0, hi=1.0):
    return lo if v < lo else (hi if v > hi else v)


def empty_features():
    """构造一份"什么都没发生"的特征字典（缺省值），便于两侧兜底。"""
    f = {k: 0.0 for k in FEATURES}
    f["nb_dist"] = 1.0        # 无邻车 → 归一化后为 1.0（"很空"）
    f["prev_follow"] = 1.0    # 默认"上一动作=继续跟随"
    return f


def build_obs(feat):
    """特征字典 → 观测 list（长度 OBS_DIM）。

    feat 缺项按 empty_features() 的默认值补（保证部署期异常不会崩维度）。
    返回普通 list；torch.as_tensor(obs, dtype=float32) 可直接使用。
    """
    base = empty_features()
    base.update({k: float(v) for k, v in feat.items() if k in base})
    out = []
    for k in FEATURES:
        v = base[k]
        if k in SCALE:
            v = v / SCALE[k]
        out.append(_clip(v))
    return out


def build_mask(feat):
    """动作合法性掩码（长度 N_ACTIONS，1=可选）——**训练侧与部署侧共用**。

    这不是"为了让训练变简单"，而是**安全约束**：现场绝不允许策略选出一个
    几何上必然把车顶进墙里的原语。判据取现场实测值：
      · back / three_point：后方净空 < 0.45 m → 不可用（三点掉头要退弧）
      · three_point：还需侧向 ≥0.55 m（弧线要往外甩）
      · replan：前后都顶死(<0.30) → 无意义（A* 也解不出）
      · follow：永远合法（"其实没卡住"是最常见的正解）
    """
    m = [1] * N_ACTIONS
    front = feat.get("front", 0.0)
    back = feat.get("back", 0.0)
    side = min(feat.get("left", 0.0), feat.get("right", 0.0))
    # ★ spin 恒被掩码。理由（实测 + 代码双重确认）：阿克曼插件 steer=atan(w·L/v)，
    #   v≈0 → speed=0 → **车根本不动**，它是**空操作**而不是"原地转向"。
    #   实测证据：未掩码时 MAPPO 有 21% 的决策选了 spin —— 因为"不动"既能躲开
    #   "顶死惩罚"，又不付出任何代价，成了策略的避风港，最终恢复率(0.90)反而
    #   不如规则基线(1.00)。索引保留只为日志兼容（"若策略仍在选它，说明有 bug"）。
    m[IDX_SPIN] = 0
    if back < 0.45:
        m[IDX_BACK] = 0
    if back < 0.45 or side < 0.55:
        m[IDX_TP] = 0
    if front < 0.30 and back < 0.30:
        m[IDX_REPLAN] = 0
    # ★ 重复即升级：同一原语**已连续选了 2 拍**（repeat>=0.5）且仍未脱困
    #   → 把它掩掉，逼着换一个（follow→back→three_point→replan→wait 这条升级链）。
    #   依据：现场影子数据显示双方都会"同一动作连选 6 拍"直到上限收尾
    #   （138 事件里 75% 如此），即缺一条"发现没用就换招"的硬规则。
    #   安全：follow/wait/back 不可能同时被掩（各自条件互斥），至少留一条路。
    if feat.get("repeat", 0.0) >= 0.5:
        for i, name in enumerate(PRIMITIVES):
            if feat.get("prev_" + name, 0.0) >= 0.5:
                m[i] = 0
    if not any(m):                      # 极端兜底：绝不能出现"无动作可选"
        m[IDX_WAIT] = 1
    return m


def rule_action(feat):
    """**规则基线**（= 手写策略树）：部署侧的 `recovery_mode=rule` 与训练时的
    "MAPPO vs 规则"对比都用它。语义与 custom_nav._control_loop 的优先级链一致，
    但**补上了它缺的两步**：

      ① 需要掉头（目标在后方 >100°）时走**三点掉头**（原来只会倒车/原地转向，
         而 spin 在阿克曼上是空操作）；
      ② **按卡死时长升级**（`stuck_t`）：先 follow/重规划，无效再退，再无效掉头
         —— 基类的链路是靠 return 逐级"下沉"，这里显式用时间做升级判据。

    返回原语索引；调用方仍须过一遍 build_mask。
    """
    front = feat.get("front", 0.0)
    back = feat.get("back", 0.0)
    stuck = feat.get("stuck_t", 0.0)
    # 1) 前方开阔且目标在前 → 根本不需要干预
    if front >= 0.70 and abs(feat.get("turn", 0.0)) < 1.0:
        return IDX_FOLLOW
    # 2a) 上一拍已经打过三点掉头仍然卡着 → 说明**一把空间不够**（前弧被挡），
    #     先把车退开腾出 ≈R_min 的前进空间，再打（这是"两步规划"）。
    #     给规则补这一条是为了让基线足够强——否则基线在 escape_then_turn 族上
    #     会陷入"tp→被顶→再 tp"的循环，对照实验就失去意义了。
    if (feat.get("need_uturn", 0.0) >= 0.5 and feat.get("prev_tp", 0.0) >= 0.5
            and stuck > 1.6 and back >= 0.90):
        return IDX_BACK
    # 2) 目标在后方 >100° → 只能掉头（★ 基类完全没有的能力）
    if feat.get("need_uturn", 0.0) >= 0.5 and back >= 0.45:
        return IDX_TP
    # 3) 前后夹击 → 停车等让行
    if front < 0.45 and back < 0.45:
        return IDX_WAIT
    # 4) 刚开始卡（<2 拍）且邻车近 → 先试图绕行（换道比重规划便宜）
    if feat.get("nb_dist", 9.9) < 1.20 and stuck <= 2.0 * 0.8:
        return IDX_REPLAN
    # 5) 前方被堵、后方有空间 → 退开换视角（基类的 back）
    if front < 0.45 and back >= 0.45:
        return IDX_BACK
    # 6) 兜底：卡久了就掉头换朝向（比反复小退有用），没空间则等让行
    return IDX_TP if back >= 0.45 else IDX_WAIT


def masked_rule_action(feat):
    """规则基线 + 掩码兜底 → 最终可执行的**合法**原语索引。

    部署侧（rl_recovery.py）与训练侧评估都要用："规则说 back 但后方只有 0.2 m"
    这种组合必须被掩码拦下，否则规则基线自己就会撞墙，对照实验失去意义。
    兜底顺序体现优先级：先"什么都不做/等"，再"退"，最后才是"重规划/掉头"。
    """
    m = build_mask(feat)
    a = rule_action(feat)
    if m[a]:
        return a
    # 兜底顺序：**先"退开/等让行"，最后才是"什么都不做"**。
    # 早期版本把 IDX_FOLLOW 排第一，于是在"朝向贴墙的口袋"场景（三点掉头被掩码
    # 否决）里规则退化成 follow —— 车顶着墙"继续跟随"，等于什么都不做，
    # 实测让规则基线的恢复率从 0.57 掉到 0.17。这不是小事：**"显然不动"的兜底
    # 会让对照实验失真**，也会掩盖策略真正学到的差异。
    for cand in (IDX_BACK, IDX_WAIT, IDX_REPLAN, IDX_FOLLOW, IDX_TP, IDX_SPIN):
        if m[cand]:
            return cand
    return IDX_FOLLOW


def tp_sign(feat):
    """三点掉头往哪边打（+1=左 / -1=右）——**训练侧与部署侧共用**的确定性规则。

    为什么把"方向"做成规则而不是动作：动作空间已定为 6 个离散原语（P0 的
    最小改动面）。方向按几何选，依据两条（都可解释、也可复现）：
      1) 优先朝**目标所在一侧**转（目标是掉头的动机：goal_sin>0 说明目标在左）；
      2) 若该侧空间不足（<0.55 m，转了会蹭），改朝另一侧。
    实测场景里巷道宽 ≥2.4 m，两侧空间都够时即等价于"朝目标侧转"。
    """
    side = 1.0 if feat.get("goal_sin", 0.0) >= 0.0 else -1.0
    left = feat.get("left", 0.0)
    right = feat.get("right", 0.0)
    if side > 0 and left < 0.55 and right >= 0.55:
        return -1.0
    if side < 0 and right < 0.55 and left >= 0.55:
        return 1.0
    return side


def rule_action_from_obs(obs):
    """从**观测向量**还原规则动作（PPO 锚定用）。

    为什么能做到：`rule_action()` 只用到 front/back/left/right/turn/need_uturn/
    nb_dist/stuck_t/prev_tp 这几项，而它们**全在观测里**（见 FEATURES）。
    于是可以在 PPO 更新后，用同一批 obs 直接算出"规则会怎么选"，做一步小监督，
    把策略锚在"不差于手写规则"的水平上。

    注意：观测经过 build_obs 的归一化+裁剪，这里按 SCALE 反归一化还原量纲
    （stuck_t 等被裁到 1.0 的项，还原后为 3.0，仍满足规则里 >1.6 的判据）。
    """
    idx = {k: i for i, k in enumerate(FEATURES)}
    feat = empty_features()
    for k, sc in SCALE.items():
        if k in idx:
            feat[k] = float(obs[idx[k]]) * sc
    if "need_uturn" in idx:
        feat["need_uturn"] = 1.0 if obs[idx["need_uturn"]] > 0.5 else 0.0
    for name in PRIMITIVES:
        key = "prev_" + name
        if key in idx:
            feat[key] = 1.0 if obs[idx[key]] > 0.5 else 0.0
    return masked_rule_action(feat)
