# -*- coding: utf-8 -*-
"""
recovery_env.py —— 「卡死恢复原语选择器」的训练环境（P0）
================================================================================

定位
----
这是 DEVELOPMENT_NOTES §7.9/§8.6 得出的结论的直接产物：本仓库"阿克曼 + 窄巷道 +
频繁掉头"的现场，失败模式是**卡住/不会掉头**——一个**离散恢复决策**问题，
而不是"v/ω 取多少"的连续控制问题。因此本 env 采样"卡死场景"，让策略在
6 个**几何原语**里选一个，而不是学连续速度。

为什么不用现成的 FleetEnv
-------------------------
`env/fleet_env.py` 是**图/队列抽象**：只有节点与边，没有任何几何、动力学、激光。
恢复决策必须看到"车身周围的空间"，所以本文件新写一个**几何 + 阿克曼动力学**的
小 env。但它**不留私货**：

  · 地图复用 `custom_nav.MapGrid`（同一份 map.pgm / 同一套膨胀语义）；
  · 运动学**复刻** `gazebo_ros_ackermann_drive` 插件语义（见 _twist_step）；
  · 观测/动作规范复用 `recovery_spec.py`（与部署节点同一来源）。

关键语义复刻（这三条决定了原语是否有效，必须与真机一致）
------------------------------------------------------
1. **Twist → 阿克曼**：插件内部 `speed = linear.x`；`steer = atan(angular.z·L/v)`
   并限幅 ±max_steer（0.6 rad）。于是：
     · `|v| ≥ 0.05` 时，实际偏航率 = v/L·tan(steer) ≈ angular.z（未限幅时精确相等）；
     · `|v| ≈ 0` 时 **speed=0 → 车根本不动** —— 这正是基类 `spin` 原语
       （_send(0.0, w_max·0.9)）在阿克曼上**空操作**的原因，也是"必须新增
       三点掉头"的依据；
     · 转向限幅 ⇒ 最小转弯半径 R_min = L/tan(0.6) = 0.32/0.684 = **0.468 m**，
       命令曲率超过它时车只能按 R_min 走（"绕圈"现象的来源）。
2. **碰撞判据用内切半径**：车体 0.5×0.38 → 内切 0.19 m。占用栅格的欧氏距离场
   (EDT) < 0.19 m 即判"压上障碍"（与 nav2 的 253 内切带同源）。
3. **邻车/托盘按圆盘注入**：与 `custom_nav._inject_neighbor_blocks` 一致
   （激光 0.325 m 平面高于车体 ≤0.29 m，"看不见"邻车，故靠 odom 圆盘协作感知）。

接口（与 MAPPO 期望一致，N=1 单智能体）
--------------------------------------
  reset() -> obs[1, OBS_DIM]
  step(actions) -> (obs[1,D], reward_float, done_bool, info)
  get_masks() -> mask[1, N_ACTIONS]
"""

import math
import os
import sys

import numpy as np

# ---------------------------------------------------------------------------
# 复用导航侧的同一份地图/工具（避免"两套地图语义"漂移）
# ---------------------------------------------------------------------------
_NAV_DIR = os.environ.get(
    "RECOVERY_NAV_DIR",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..",
                                 "src", "fleet_sim", "scripts")))
if _NAV_DIR not in sys.path:
    sys.path.insert(0, _NAV_DIR)
from custom_nav import MapGrid, wrap_pi          # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from recovery_spec import (                       # noqa: E402
    PRIMITIVES, N_ACTIONS, OBS_DIM, FEATURES, build_obs, empty_features,
    build_mask, rule_action, tp_sign,
    IDX_FOLLOW, IDX_WAIT, IDX_BACK, IDX_SPIN, IDX_TP, IDX_REPLAN,
)

try:
    from scipy import ndimage
    _HAS_SCIPY = True
except Exception:                                  # 兜底：无 scipy 用 BFS 距离场
    _HAS_SCIPY = False

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_MAP = os.path.abspath(os.path.join(
    _HERE, "..", "..", "src", "fleet_sim", "maps", "map.yaml"))

# ---- 底盘与几何常量（= URDF agv_core.urdf.xacro / nav2 参数里的实测值）----
WHEELBASE = 0.32          # 轴距(m)
MAX_STEER = 0.6           # 前轮最大转向角(rad)
V_MAX = 0.6               # 巡航线速度上限(m/s)
W_MAX = 1.5               # 角速度指令上限(rad/s)
HALF_LEN, HALF_WID = 0.25, 0.19      # 车体半长/半宽 → 内切半径 0.19
R_MIN = WHEELBASE / math.tan(MAX_STEER)      # 0.468 m
NB_RADIUS = 0.27          # 邻车等效圆盘（与 custom_nav 参数一致）
# ★ 与部署侧 custom_nav 的参数**逐值一致**（判据对齐的前提，勿改）
D_SAFE = 0.70             # VFH 通行净空（custom_nav d_safe）
EMERGENCY = 0.45          # 紧急停障距离（custom_nav emergency）
EP_DT = 0.1               # 物理步进（插件 update_rate=50，此处 10Hz 足够）
DECIDE_DT = 1.0           # 每个原语的"最短执行时长"步长基准

# ---- 原语时长方案（秒）----
T_WAIT, T_BACK, T_SPIN = 1.0, 0.9, 0.9
TP_SEG_T = (1.6, 1.6, 1.0)          # 三点掉头三段：前弧 / 倒弧 / 前弧
TP_V = 0.35                          # 弧线线速度（±）
TP_W = 1.5                           # 弧线角速度指令（会被 R_min 限幅到极限弧）
TP_TOTAL = sum(TP_SEG_T)             # 4.2 s


def ackermann_steer(v, w, wheelbase=WHEELBASE, max_steer=MAX_STEER):
    """复刻插件的 Twist→转向角：steer = atan(w·L/v)，限幅 ±max_steer。

    |v| < 0.05 视为停车（插件 speed=0 → 车不动），返回 0 转向角。
    返回 (speed, steer)：speed 是**插件实际执行**的线速度（v≈0 时为 0）。
    """
    if abs(v) < 0.05:
        return 0.0, 0.0
    return v, max(-max_steer, min(max_steer, math.atan(w * wheelbase / v)))


# 各原语占用的**真实时间**(s)：三点掉头 4.2s，直退 0.9s，等待 1.0s……
# 用途：把"时间惩罚"改成**按时长计价**。早期每步统一扣 0.05，等于让 4.2s 的
# 三点掉头与 0.9s 的直退一样"便宜" → 策略倾向乱试长原语（实测 MAPPO 平均
# 4.43 步 vs 规则 2.75 步，恢复更慢）。现场关心的是**墙钟恢复时间**，
# 所以惩罚必须与原语时长成正比。
_PRIM_SECONDS = {
    IDX_FOLLOW: 0.6, IDX_WAIT: T_WAIT, IDX_BACK: T_BACK,
    IDX_SPIN: T_SPIN, IDX_TP: TP_TOTAL, IDX_REPLAN: 0.3,
}

# 各原语"若不受阻"应产生的行驶里程(m) —— 用于判定"是否被顶死"
_EXP_TRAVEL = {
    IDX_FOLLOW: 0.5 * 0.6,
    IDX_WAIT: 0.0,
    IDX_BACK: 0.5 * T_BACK,
    IDX_SPIN: 0.0,
    IDX_TP: TP_V * sum(TP_SEG_T),
    IDX_REPLAN: 0.0,
}


class RecoveryEnv:
    """卡死恢复 env：单智能体、离散 6 原语、情景由 4 类"卡死族"随机采样。"""

    def __init__(self, map_yaml=None, seed=0, max_steps=12,
                 scenarios=("head_on", "uturn", "sandwich", "pocket", "escape_then_turn"),
                 coarse_stride=3, max_range=2.5):
        self.grid = MapGrid(map_yaml or _DEFAULT_MAP)
        self.nx, self.ny, self.res = self.grid.nx, self.grid.ny, self.grid.res
        self.max_steps, self.scenarios = max_steps, list(scenarios)
        self.coarse_stride, self.max_range = coarse_stride, max_range
        self.rng = np.random.RandomState(seed)
        # ---- 占用栅格的欧氏距离场（米）：EDT 在栅格空间算，最后乘 res ----
        occ = np.array(self.grid.occ, dtype=bool).reshape(self.ny, self.nx)
        if _HAS_SCIPY:
            self.dist = ndimage.distance_transform_edt(~occ) * self.res
        else:                                        # 退化：只有 0/1（精度差）
            self.dist = (~occ).astype(np.float32) * self.res
        # ---- 可停放格（blocked=False，已含 0.4 m 膨胀 → 车心安全区）----
        free = np.array([not b for b in self.grid.blocked],
                        dtype=bool).reshape(self.ny, self.nx)
        rr, cc = np.nonzero(free)
        self._free_rc = np.stack([rr, cc], axis=1)
        # 场景状态
        self.x = self.y = self.yaw = 0.0
        self.gx = self.gy = 0.0
        self.step_i = 0
        self.prev_prim = IDX_FOLLOW
        self.kind = '-'              # 本集场景族（诊断用）
        self.dyn = []                # 动态障碍圆盘 [(x, y, r)]
        self.collided = False        # 本集是否出现过"被顶死"
        self.blocked = False         # 上一步原语是否被顶死
        self._travel = 0.0           # 上一原语的实际里程（判"顶死"用）
        self._d_prev = 0.0
        self._rel_abs = 0.0          # |目标方位|（rad）
        self._repeat = 0             # 同一原语连续被选的次数（"重复即升级"用）
        self.episode_prim = []

    # ===================== 几何查询 =====================
    def clearance(self, x, y):
        """(x,y) 处到最近**物理障碍**的距离(m)；同时计入动态圆盘。

        动态圆盘是 4 元组 (x, y, r, until)：
          until=None → 永久存在（墙/固定托盘）；
          until=int  → 到第 until 个决策步后消失 —— 用来建模**邻车终会让行**
                       （没有它，"前后夹击"这一族在 6 个原语里根本无解，
                        实测 0/13 恢复、并靠 12 步 wait 霸占样本）。
        """
        r, c = self.grid.world_to_rc(x, y)
        if not (0 <= r < self.ny and 0 <= c < self.nx):
            return 0.0
        d = float(self.dist[r, c])
        for (dx, dy, dr, _u) in self.dyn:
            d = min(d, math.hypot(x - dx, y - dy) - dr)
        return max(0.0, d)

    def ray_clear(self, x, y, ang, max_r=None):
        """沿 ang 方向步进，返回**到障碍表面的距离**(m) —— 与激光语义同义。

        ⚠️ 约定必须与部署侧一致（否则规则/策略的阈值全错位）：
        custom_nav 的 `_clear_at()` 用的是**激光读数**＝从车体到障碍**表面**的距离，
        现场阈值 emergency=0.45 / d_safe=0.70 都按这个口径标定。
        因此这里也返回"表面距离"：步进到"车心距障碍 < 内切半径(0.19)"的点，
        再加回 0.19 作为表面距离。**不要**返回"车心可走距离"，那是另一套口径。
        """
        max_r = max_r or self.max_range
        step = 0.05
        n = int(max_r / step)
        for k in range(1, n + 1):
            d = k * step
            if self.clearance(x + d * math.cos(ang), y + d * math.sin(ang)) \
                    < HALF_WID:
                return max(0.0, d - step + HALF_WID)
        return max_r

    def _front_clear(self):
        return self.ray_clear(self.x, self.y, self.yaw)

    def _path_executable(self):
        """★ 与部署侧 `custom_nav._path_executable` **同判据**——"恢复成功"的准绳。

        现场判据（custom_nav._try_after_escape 调用的就是它）：
          ① 原始正前净空 < emergency(0.45) → 直接判不可执行（硬刹保险）；
          ② 前瞻方向 ±50° 内净空 ≥ d_safe(0.70) → 可执行；
          ③ 否则 VFH 在**目标方向 ±70°** 内找到净空 ≥ d_safe 的方向 → 可执行。

        ⚠️ 为什么必须对齐：早期 env 用宽松代理（front≥0.50 且 turn<1.0），
           结果是"env 恢复率 0.983 / 现场 0.14"的巨大落差——判据不对齐，
           等于**训练目标与部署目标不是同一个**，再好也是自嗨。
        """
        if self._front_clear() < EMERGENCY:
            return False
        rel = wrap_pi(math.atan2(self.gy - self.y, self.gx - self.x) - self.yaw)
        if self.ray_clear(self.x, self.y, wrap_pi(self.yaw + rel), 2.5) >= D_SAFE:
            return True
        for k in range(-7, 8):                       # VFH：±70°，10° 一档
            if self.ray_clear(self.x, self.y,
                              wrap_pi(self.yaw + rel + k * math.radians(10.0)),
                              2.5) >= D_SAFE:
                return True
        return False

    def _dir_clear(self, rel_ang):
        return self.ray_clear(self.x, self.y, wrap_pi(self.yaw + rel_ang))

    def _remain(self):
        return math.hypot(self.gx - self.x, self.gy - self.y)

    # ===================== 运动学（复刻插件） =====================
    def _twist_step(self, v, w):
        """按插件语义推进一步（dt=EP_DT）。返回 True = 本步被**顶住**。

        **碰撞用"顶住不动"建模**（与现场实测一致）：车顶死在货架柱/托盘上时，
        轮子在转但车身不动（实测指令 0.30 m/s、实际车身速度 0.005 m/s）。
        因此若下一步会侵入障碍 → **撤销该步**并标记 blocked，而不是"穿进去"
        再判碰撞。这样"往前顶"的后果是**原地不动**（拿不到进度奖励 + 顶死惩罚），
        与真机行为同构。
        """
        speed, steer = ackermann_steer(v, w)
        if speed == 0.0:
            return False                     # ★ v≈0 → speed=0，无位移（非"顶住"）
        nx = self.x + speed * math.cos(self.yaw) * EP_DT
        ny = self.y + speed * math.sin(self.yaw) * EP_DT
        if self.clearance(nx, ny) < HALF_WID:
            return True                      # 顶住：撤销本步
        self.x, self.y = nx, ny
        self.yaw = wrap_pi(self.yaw
                           + speed / WHEELBASE * math.tan(steer) * EP_DT)
        self._travel += abs(speed) * EP_DT
        return False

    # ===================== 原语执行 =====================
    def _seg(self, v, w, dur):
        """执行一段定速指令；中途被顶住即提前结束（真机也是这样顶着的）。"""
        for _ in range(int(dur / EP_DT)):
            if self._twist_step(v, w):
                return True
        return False

    def _exec(self, idx, feat):
        """执行一个原语；返回 (是否被顶住, 是否成功恢复)。

        注意 `follow` 在 env 里 = "沿当前朝向试着前进"：受阻就顶住不动，
        这正是"其实没卡住→能走"与"真卡住→白顶 0.6s"的分水岭。
        """
        if idx == IDX_FOLLOW:
            return self._seg(0.5, 0.0, 0.6), False
        if idx == IDX_WAIT:
            return self._seg(0.0, 0.0, T_WAIT), False
        if idx == IDX_BACK:
            return self._seg(-0.5, 0.0, T_BACK), False
        if idx == IDX_SPIN:
            return self._seg(0.0, W_MAX * 0.9 * (1 if feat["goal_sin"] >= 0 else -1),
                             T_SPIN), False
        if idx == IDX_TP:
            s = tp_sign(feat)
            for k, dur in enumerate(TP_SEG_T):
                v = TP_V if k != 1 else -TP_V
                if self._seg(v, s * TP_W, dur):
                    return True, False
            return False, False
        if idx == IDX_REPLAN:
            return False, self._replan_ok()
        return False, False

    def _replan_ok(self):
        """'局部 A* 重规划能否解出**可执行**路径' 的**廉价代理**。

        真实实现见 custom_nav._local_replan/_neighbor_detour（完整 A* + auto_via）。
        训练里每步都跑 240×160 的 Python A* 太慢，这里用**降采样 BFS**
        （stride=3 → 0.3 m）+ 首段转角判据近似，语义对齐：
          可达 且 首段 |转角| < 100° → 视为"重规划有效"。
        """
        st = self.coarse_stride
        nb = self.grid.nx // st + 1
        # 起点/终点落到粗格
        sr, sc = self.grid.world_to_rc(self.x, self.y)
        gr, gc = self.grid.world_to_rc(self.gx, self.gy)
        sr, sc, gr, gc = sr // st, sc // st, gr // st, gc // st
        dirty = np.array(self.grid.blocked, dtype=bool).reshape(self.ny, self.nx)
        # 注入动态圆盘（与 _inject_neighbor_blocks 同义），再降采样：
        # 粗格只要有一个细格不可走就判不可走（保守）
        for (dx, dy, dr, _u) in self.dyn:
            rr, cc = self.grid.world_to_rc(dx, dy)
            rad = int((dr + HALF_WID) / self.res) + 1
            for dr_ in range(-rad, rad + 1):
                for dc in range(-rad, rad + 1):
                    if dr_ * dr_ + dc * dc > rad * rad:
                        continue
                    r2, c2 = rr + dr_, cc + dc
                    if 0 <= r2 < self.ny and 0 <= c2 < self.nx:
                        dirty[r2, c2] = True

        import collections
        seen = {(sr, sc)}
        q = collections.deque([(sr, sc)])
        reach = False
        while q:
            r, c = q.popleft()
            if abs(r - gr) <= 1 and abs(c - gc) <= 1:
                reach = True
                break
            for dr_, dc_ in ((1, 0), (-1, 0), (0, 1), (0, -1),
                             (1, 1), (1, -1), (-1, 1), (-1, -1)):
                r2, c2 = r + dr_, c + dc_
                if not (0 <= r2 < nb and 0 <= c2 < nb) or (r2, c2) in seen:
                    continue
                # 粗格内任一细格不可走 → 该粗格视为不可走
                blocked = False
                for k in range(st):
                    for l in range(st):
                        rr2, cc2 = r2 * st + k, c2 * st + l
                        if rr2 < self.ny and cc2 < self.nx and dirty[rr2, cc2]:
                            blocked = True
                            break
                    if blocked:
                        break
                if blocked:
                    continue
                seen.add((r2, c2))
                q.append((r2, c2))
        if not reach:
            return False
        # 首段转角：粗解方向 vs 车头（>100° 判定"掉头不可执行"）
        wr, wc = self.grid.rc_to_world(gr * st, gc * st)
        rel = abs(wrap_pi(math.atan2(wc - self.y, wr - self.x) - self.yaw))
        return rel < 1.7453292519943296

    # ===================== 场景采样 =====================
    def _rand_free(self):
        r, c = self._free_rc[self.rng.randint(len(self._free_rc))]
        return self.grid.rc_to_world(int(r), int(c))

    def _axis_clears(self, x, y, mx=3.0):
        """四轴方向的 (角度, 净空) 列表（0/90/180/270°，与地图轴对齐）。"""
        return [(k * math.pi / 2.0, self.ray_clear(x, y, k * math.pi / 2.0, mx))
                for k in range(4)]

    def _set_goal(self, gx, gy):
        """目标点必须是可停放格（否则"恢复成功"无从定义）。"""
        gr, gc = self.grid.world_to_rc(gx, gy)
        if not self.grid.in_bounds(gr, gc) or self.grid.blocked[gr * self.nx + gc]:
            return False
        self.gx, self.gy = gx, gy
        return True

    def _pick_corridor_dir(self):
        """挑一个"沿通道"的朝向：四轴方向里净空最大者。"""
        return max(self._axis_clears(self.x, self.y), key=lambda t: t[1])[0]

    # ---- 4 个卡死族的"构造器"（族内拒绝采样，不是"随机撞运气") ----
    #
    # ★ 2026-09-19 重要修正：原实现在**重试循环里反复重抽 kind**，而各族接受条件
    #   宽严不一（head_on/pocket 在新口径下几乎不可满足）→ 结果是条件最宽松的
    #   sandwich **垄断了全部样本**（实测 6/6 全是 sandwich，其余三族 0 次被采到，
    #   训练分布与"现场分布"完全脱节）。现改为：**每次 reset 只抽一次 kind**，
    #   再在该族内构造场景；构造失败才退到 fallback。
    def _build_head_on(self):
        """正前方 0.5m 有邻车/托盘，后方通畅 → 该退或绕（最常见的"顶牛"）。"""
        for _ in range(200):
            x, y = self._rand_free()
            self.x, self.y = x, y
            a, fwd = max(self._axis_clears(x, y), key=lambda t: t[1])
            if fwd < 0.90:
                continue
            self.yaw = a
            if self.ray_clear(x, y, wrap_pi(a + math.pi), 3.0) < 0.80:
                continue
            self.dyn = [(x + 0.55 * math.cos(a), y + 0.55 * math.sin(a), NB_RADIUS, None)]
            gd = self.rng.uniform(3.0, 6.0)
            if self._set_goal(x + gd * math.cos(a), y + gd * math.sin(a)):
                return True
        return False

    def _build_sandwich(self):
        """前后各 0.5m 被夹（前后夹击）→ 只能等让行或掉头换侧。"""
        for _ in range(200):
            x, y = self._rand_free()
            self.x, self.y = x, y
            a, fwd = max(self._axis_clears(x, y), key=lambda t: t[1])
            if fwd < 1.30:
                continue
            self.yaw = a
            # 前车"等一会儿就让行"（2~4 步后消失）→ 正确解是 wait 而不是硬顶
            self.dyn = [(x + 0.50 * math.cos(a), y + 0.50 * math.sin(a), NB_RADIUS,
                         self.rng.randint(2, 5)),
                        (x - 0.55 * math.cos(a), y - 0.55 * math.sin(a), NB_RADIUS, None)]
            gd = self.rng.uniform(2.5, 5.0)
            if self._set_goal(x + gd * math.cos(a), y + gd * math.sin(a)):
                return True
        return False

    def _build_uturn(self):
        """目标在**车后方 >100°**，通道够宽 → 必须掉头（★ 唯一非它不可的族）。"""
        for _ in range(200):
            x, y = self._rand_free()
            self.x, self.y = x, y
            cs = self._axis_clears(x, y)
            a, fwd = max(cs, key=lambda t: t[1])
            if fwd < 1.20:
                continue
            self.yaw = a
            # 侧向用 **max** 而不是 min：掉头方向可以朝宽敞的一侧打
            side = max(self.ray_clear(x, y, wrap_pi(a + math.pi / 2), 2.5),
                       self.ray_clear(x, y, wrap_pi(a - math.pi / 2), 2.5))
            if side < 0.90:
                continue
            self.dyn = []
            gd = self.rng.uniform(2.0, 4.0)
            if self._set_goal(x - gd * math.cos(a), y - gd * math.sin(a)):
                return True
        return False

    def _build_pocket(self):
        """贴着货架/墙的口袋：**朝向最近的障碍**（构造出来的死角，不是撞运气）。"""
        for _ in range(200):
            x, y = self._rand_free()
            self.x, self.y = x, y
            a, fwd = min(self._axis_clears(x, y), key=lambda t: t[1])
            if fwd > 0.60 or fwd < 0.30:        # 要"很近但不贴死"
                continue
            self.yaw = a
            if self.ray_clear(x, y, wrap_pi(a + math.pi), 3.0) < 0.90:
                continue
            self.dyn = []
            gd = self.rng.uniform(1.5, 3.5)
            if self._set_goal(x - gd * math.cos(a), y - gd * math.sin(a)):
                return True
        return False

    def _build_escape_then_turn(self):
        """★ 必须**两步规划**的族：先退一点腾出甩弧空间，再掉头。

        构造：正前方 0.35~0.5m 被挡（口袋），目标在车后方 >100°（须掉头），
        但侧向余量**刚好不够一把打完**（0.55~0.75m）→ 一上来就 three_point
        会在前弧中途顶住、白耗 1.6s 且只转一部分，之后仍需再打；
        正解是 back（0.9s 腾出空间）→ three_point（4.2s 一次到位）= **2 步**。

        为什么加这一族：规则基线是"单拍启发式"（看到 need_uturn 就 tp），
        在"一把打不完"时必然多花一步。这是 RL 相对规则**真正能赢**的地方，
        也是把评测口径从"恢复率"扩展到"**墙钟恢复时间**"的原因。
        """
        for _ in range(200):
            x, y = self._rand_free()
            self.x, self.y = x, y
            a, fwd = min(self._axis_clears(x, y), key=lambda t: t[1])
            # ★ 判据的几何本质：三点掉头的**前弧**必须先向前走 ≈R_min(0.468m)
            #   才能把车头转出去。所以"前方 < R_min" ⇒ 一把弧必然中途顶住。
            #   （早期写成"侧向 0.55~0.80"是错的：车头朝墙时，侧向正是**沿墙**
            #    方向，在仓库里几乎总是很宽 → 该族 28/28 全部回落，等于没采到。）
            if not (0.22 <= fwd < 0.45):
                continue
            back = self.ray_clear(x, y, wrap_pi(a + math.pi), 3.0)
            if back < 1.10:                     # 后面要退得出去（这才是解法的前提）
                continue
            left = self.ray_clear(x, y, wrap_pi(a + math.pi / 2), 2.5)
            right = self.ray_clear(x, y, wrap_pi(a - math.pi / 2), 2.5)
            if max(left, right) < 0.55:         # 侧向够（tp 本身合法，只是"一把打不完"）
                continue
            self.yaw = a
            self.dyn = []
            gd = self.rng.uniform(2.0, 3.5)
            if self._set_goal(x - gd * math.cos(a), y - gd * math.sin(a)):
                return True
        return False

    def _build_fallback(self):
        """兜底：把圆盘放在"放得下的最近处"，保证样本仍是一个卡死场景。"""
        for _ in range(200):
            x, y = self._rand_free()
            self.x, self.y = x, y
            a, fwd = max(self._axis_clears(x, y), key=lambda t: t[1])
            if fwd < 0.70:
                continue
            self.yaw = a
            d = min(0.55, max(0.35, fwd - 0.30))
            self.dyn = [(x + d * math.cos(a), y + d * math.sin(a), NB_RADIUS, None)]
            if self._set_goal(x + 4.0 * math.cos(a), y + 4.0 * math.sin(a)):
                return True
        self.x, self.y, self.yaw = 0.0, 0.0, 0.0      # 最后兜底：主通道
        self.dyn = []
        self.gx, self.gy = 3.0, 0.0
        return True

    def _sample_scene(self):
        """每次 reset 抽**一个族**，族内构造场景。"""
        kind = self.scenarios[self.rng.randint(len(self.scenarios))]
        self.kind = kind
        if getattr(self, "_build_" + kind)():
            return True
        self.kind = kind + "→fallback"
        return self._build_fallback()

    # ===================== 观测 =====================
    def _feat(self):
        """采集与部署侧**同名同义**的特征字典（两侧共用 recovery_spec.build_obs）。"""
        f = empty_features()
        # "已在卡死中多久"：随决策步单调增长（部署侧由 rl_recovery 累计触发次数）。
        # 它是策略/规则**升级原语**（follow → back → three_point）的依据，
        # 因此**不能是常数**——早期把它写死 1.5 会让"升级"逻辑永远走同一个分支。
        f["stuck_t"] = 0.8 + self.step_i * 1.0
        f["cmd_v"] = 0.0
        f["cmd_w"] = 0.0
        rel = wrap_pi(math.atan2(self.gy - self.y, self.gx - self.x) - self.yaw)
        self._rel_abs = abs(rel)          # 朝向误差（rad），供 step 的 shaping 用
        f["goal_sin"] = math.sin(rel)
        f["goal_cos"] = math.cos(rel)
        f["front"] = self._front_clear()
        f["back"] = self._dir_clear(math.pi)
        f["left"] = self._dir_clear(math.pi / 2.0)
        f["right"] = self._dir_clear(-math.pi / 2.0)
        f["turn"] = abs(rel)                  # env 无 path_world，用目标方位等效
        f["need_uturn"] = 1.0 if abs(rel) > 1.7453292519943296 else 0.0
        f["nb_dist"] = (min([math.hypot(self.x - d[0], self.y - d[1])
                             for d in self.dyn]) if self.dyn else 3.0)
        f["remain"] = self._remain()
        for k, name in enumerate(PRIMITIVES):
            f["prev_" + name] = 1.0 if self.prev_prim == k else 0.0
        # repeat：同一原语已连续被选的次数 /2（截 1.0），供"重复即升级"掩码使用
        rep = self._repeat if self.prev_prim is not None else 0
        f["repeat"] = min(1.0, rep / 2.0)
        return f

    def _obs(self):
        return np.asarray([build_obs(self._feat())], dtype=np.float32)

    def get_masks(self):
        """动作掩码：**直接调用 recovery_spec.build_mask**（与部署侧同源）。

        安全约束：现场绝不允许策略选出几何上必然把车顶进墙里的原语。
        """
        return np.asarray([build_mask(self._feat())], dtype=np.float32)

    # ===================== 主接口 =====================
    def reset(self):
        if not self._sample_scene():
            # 极端兜底：用地图中心 + 正前方目标（几乎不会发生）
            self.x, self.y, self.yaw = 0.0, 0.0, 0.0
            self.gx, self.gy = 3.0, 0.0
            self.dyn = []
        self.step_i = 0
        self.collided = False
        self.blocked = False
        self._travel = 0.0
        self._repeat = 1             # 每集首个原语从 1 计（无历史）
        self.prev_prim = IDX_FOLLOW
        self.episode_prim = []
        self._d_prev = self._remain()
        self.info = {"recovered": 0, "collided": 0, "prim": []}
        return self._obs()

    def step(self, actions):
        """actions: 长度 1 的数组（或 int）→ 执行一个原语并推进一个决策步。"""
        idx = int(np.asarray(actions).reshape(-1)[0])
        self.step_i += 1
        feat = self._feat()
        rel0 = self._rel_abs
        self._travel = 0.0
        _, recovered = self._exec(idx, feat)
        # 邻车让行：到期的动态圆盘消失（"等一会儿就通了"）——wait 原语的回报来源
        if self.dyn:
            alive = [d for d in self.dyn if d[3] is None or d[3] > self.step_i]
            if len(alive) != len(self.dyn):
                self.dyn = alive
        exp = _EXP_TRAVEL.get(idx, 0.0)
        # "被顶死"：该原语本应产生里程，但实际走了不到 40% → 说明被挡住了
        self.blocked = bool(exp > 0.05 and self._travel < 0.4 * exp)
        self.episode_prim.append(idx)
        self.info["prim"].append(idx)
        if self.blocked:
            self.collided = True
            self.info["collided"] += 1

        feat2 = self._feat()                   # 执行后的状态（一次采集，多处复用）
        d_now = self._remain()
        # ---- 奖励 ----
        # ① 朝目标靠近；② ★朝向对齐★：掉头的第一步必然"远离目标"，只用距离
        #    shaping 会把 three_point 判成坏动作（实测 MAPPO 因此几乎不用它，
        #    三点占比 1% vs 规则 29%，恢复率也卡在 0.90）。加上"朝向误差减小"
        #    这一项后，掉头终于有正回报。
        r = 3.0 * (self._d_prev - d_now) + 0.6 * (rel0 - self._rel_abs)
        r -= 0.06 * _PRIM_SECONDS.get(idx, 1.0)  # ③ 时间惩罚：**按时长计价**
        if idx != self.prev_prim:
            r -= 0.10                          # ④ 轻微惩罚反复横跳
        if self.blocked:
            r -= 0.50                          # ⑤ 顶死：别硬顶（现场就是被顶死的）
        # ---- "恢复"判据：★ 与现场 custom_nav._path_executable 完全一致 ★ ----
        if not recovered:
            recovered = self._path_executable()
        if recovered:
            r += 8.0
            self.info["recovered"] = 1
        self._d_prev = d_now
        self._repeat = self._repeat + 1 if idx == self.prev_prim else 1
        self.prev_prim = idx

        done = bool(recovered or self.step_i >= self.max_steps)
        if done and not recovered:
            r -= 2.0                           # 到步数上限仍未恢复
        return self._obs(), float(r), done, self.info


def make_env(**kw):
    """与 env/fleet_env.py 的 make_env 同风格：便于 train_recovery.py 统一调用。"""
    return RecoveryEnv(**kw)
