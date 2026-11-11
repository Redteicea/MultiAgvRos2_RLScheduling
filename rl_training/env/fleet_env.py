"""
fleet_env.py - 多 AGV 仓储调度环境（离散时间调度仿真器）

Phase 6 (Task 6.2) 重写版。替代旧版“一步结算假环境”。

拓扑对齐真实仿真仓库 (warehouse.world / fleet_manager.task_generator)：
  - pickup_w(-9.6, 0) / pickup_e(9.6, 0)：取货站（y=0 主通道两端）
  - 卸货站 dropoff_nw/ne/sw/se（四角外围）；共 8 个站点，见 SITE_COORD（地图 v2）
  - 主通道 PW–PE（y=0，宽 2.4 m）→ **全图唯一窄道**，串行占用：同向跟车、对向等空
  - 北/南横向通道 y=±3.25（宽 1.9m）与 x=±9.6 纵向通道可并行

执行模型：
  - 每决策 tick = 1 s；速度 1.0 m/s；装卸 dwell 各 3 s
  - AGV 沿图最短路径（内建 Dijkstra）行驶；站点容量 1（先到先装卸）
  - 任务带优先级（1=最急）与 deadline；队列等待超时判失败
  - 电量（可选）：自动回充最低位，充满回待命

事件奖励（每 step 结算增量）：
  - 每 tick 时间成本 -0.02；任务完成 +25*prio + 时效加成；失败 -14*prio
  - 窄道互斥等待事件 -0.4 / 站前排队 -0.2 / 抢单无效动作 -0.5 / 清空 +15
状态奖励（每 tick 依当前状态结算，见 _state_reward；为高负载/资源紧张调优）：
  - 有活却空闲：-0.06 ×(空闲车数)，队内有 pending 时才施加（抑制“挑单空转”）
  - 临近超期：-0.08 ×Σ max(0, wait/deadline − 0.5)（截止感知的紧迫度塑形）

接口（纯 Python，无 gym 依赖）：
  env.reset()/step(actions)/get_masks()/episode_stats()
  属性: num_agvs, act_dim(=M_max+1), obs_dim, max_steps
"""
import heapq
import math
import random

import numpy as np

# ---------------------------------------------------------------------------
# 仓库拓扑（★ 2026-09-19 地图 v2 全面重排；与真实 world / Nav2 地图严格一致）
# ---------------------------------------------------------------------------
# 几何定义在以下三处（改动必须同步，否则策略学到的是另一个仓库）：
#   · src/fleet_sim/scripts/generate_warehouse_world.py  RACK_ROWS/COLS + 8 站点
#   · src/fleet_sim/scripts/generate_hand_map.py         maps/map.pgm
#   · src/fleet_manager/src/fleet_manager_node.cpp       buildMap + stations/stops
#
# 布局（24 m × 16 m）：
#   · 货架 4 排 y=±2.0/±5.2，**行内连续（无死端口袋）** → 主通道 y∈[-1.2,1.2] 宽 2.4 m
#   · 东/西纵向通道 x∈[±9.0,±11.75] 宽 2.75 m（站点/充电桩所在）
#   · 中部横向通道 y=±3.6（跨库捷径）；南/北外围通道 y=±6.8
#   · **8 个环形分布的站点（4 取 4 卸）+ 4 个充电桩（纵向通道中段贴墙）**
# ---------------------------------------------------------------------------
# 24 个任务点 = 12 个港站台 + 12 个货架存取位（★ v3 地图：开放货架间通道）
# 字典顺序即"规范站点索引"：0..11 = 港、12..23 = 货架位。
#   → target_code = 1 + 该索引（1..24），充电桩 = 25；归一化 /25（见 _target_code：
#     它用 N_SITES+1 自动归一，所以**扩站点数不需要改编码逻辑**）。
SITE_COORD = {
    # ---- 港站台 12 ----
    # 4 入库港（= 图节点 PW/PE/NM/SM）
    "pickup_w": (-10.4, 0.0), "pickup_e": (10.4, 0.0),
    "pickup_n": (0.0, 6.8), "pickup_s": (0.0, -6.8),
    # 4 出库港（= 图节点 NW/NE/SW/SE）
    "dropoff_nw": (-10.4, 6.8), "dropoff_ne": (10.4, 6.8),
    "dropoff_sw": (-10.4, -6.8), "dropoff_se": (10.4, -6.8),
    # ★ v3 新增 4 个外港：南/北外围通道的**中段**（把 NW-NM / NM-NE 拆成链），
    #   让"四周"从 8 个点变成 12 个点 → 增大了四周→四周、货架→四周的随机性。
    "port_nw": (-6.0, 6.8), "port_ne": (6.0, 6.8),
    "port_sw": (-6.0, -6.8), "port_se": (6.0, -6.8),
    # ---- 货架拣货/补货位 12 ----
    # ★ v3：对齐新列中心 x=±1.4/±4.2/±7.0（原 ±2/±6 在新布局下落在**通道**里 →
    #   会"对着空地取货"）。北 6 个在 y=+3.6、南 6 个在 y=-3.6。
    "bay_n1": (-7.0, 3.6), "bay_n2": (-4.2, 3.6), "bay_n3": (-1.4, 3.6),
    "bay_n4": (1.4, 3.6), "bay_n5": (4.2, 3.6), "bay_n6": (7.0, 3.6),
    "bay_s1": (-7.0, -3.6), "bay_s2": (-4.2, -3.6), "bay_s3": (-1.4, -3.6),
    "bay_s4": (1.4, -3.6), "bay_s5": (4.2, -3.6), "bay_s6": (7.0, -3.6),
}
# ★ v3：**取/卸都用全部 24 个点** —— 用户要求"从货架取货到四周，也允许四周取货到四周"，
#   四类任务（港↔港 / 货架→港 / 港→货架 / 货架↔货架）由随机配对自然产生。
#   任务生成侧会跳过 src==dst（见 _spawn 与 C++ task_generator 的 same-site 保护）。
SITE_NAMES = list(SITE_COORD)
BAY_SITES = [s for s in SITE_COORD if s.startswith("bay")]
PORT_SITES = [s for s in SITE_COORD if s.startswith("pickup") or s.startswith("dropoff")]
PICKUP_SITES = list(SITE_NAMES)
DROPOFF_SITES = list(SITE_NAMES)
# 站点 → 图上节点
SITE_NODE = {"pickup_w": "PW", "pickup_e": "PE", "pickup_n": "NM", "pickup_s": "SM",
             "dropoff_nw": "NW", "dropoff_ne": "NE",
             "dropoff_sw": "SW", "dropoff_se": "SE"}
for _b in BAY_SITES + [s for s in SITE_COORD if s.startswith("port_")]:
    SITE_NODE[_b] = _b
N_SITES = len(SITE_COORD)                          # 24（target_code 归一化用）
SITE_IDX = {s: k for k, s in enumerate(SITE_COORD)}

# 节点坐标（28 个）：
#   PW/PE        主通道两端（= 入库港 pickup_w/e）
#   WN/WS/EN/ES  纵向 × 横向通道交叉点（= 4 个充电桩）
#   NW/NE/SW/SE  四角外围（= 出库港 dropoff_nw/ne/sw/se）
#   NM/SM        南/北外围中点（= 入库港 pickup_n/s）
#   port_*       ★ v3 外围通道中段（= 新增 4 个外港）
#   bay_*        ★ v3 12 个货架存取位（把 WN-EN / WS-ES 拆成 6 段链）
NODE_COORD = {
    "PW": (-10.4, 0.0), "PE": (10.4, 0.0),
    "WN": (-10.4, 3.6), "WS": (-10.4, -3.6),
    "EN": (10.4, 3.6), "ES": (10.4, -3.6),
    "NW": (-10.4, 6.8), "NE": (10.4, 6.8),
    "SW": (-10.4, -6.8), "SE": (10.4, -6.8),
    "NM": (0.0, 6.8), "SM": (0.0, -6.8),
}
for _s in [s for s in SITE_COORD if s.startswith("port_") or s.startswith("bay")]:
    NODE_COORD[_s] = SITE_COORD[_s]

# 无向边表：(u, v, [可选标志]) —— 共 31 条（v3）。无标志 = 普通可并行通道；
# 带 "narrow" 的 PW–PE 即 y=0 主通道（对向互斥、同向跟车）：
# **全图唯一的窄道，也是 RL 必须学会协调的地方**。
EDGES = [
    # 东/西纵向通道（各 4 段）
    ("PW", "WN"), ("WN", "NW"), ("PW", "WS"), ("WS", "SW"),
    ("PE", "EN"), ("EN", "NE"), ("PE", "ES"), ("ES", "SE"),
    # 南/北外围通道：★ v3 新增 4 个外港，各由 2 段拆成 4 段
    ("NW", "port_nw"), ("port_nw", "NM"), ("NM", "port_ne"), ("port_ne", "NE"),
    ("SW", "port_sw"), ("port_sw", "SM"), ("SM", "port_se"), ("port_se", "SE"),
    # 北横向通道（y=3.6）：WN → bay_n1..n6 → EN（6 段链）
    ("WN", "bay_n1"), ("bay_n1", "bay_n2"), ("bay_n2", "bay_n3"),
    ("bay_n3", "bay_n4"), ("bay_n4", "bay_n5"), ("bay_n5", "bay_n6"),
    ("bay_n6", "EN"),
    # 南横向通道（y=-3.6）：WS → bay_s1..s6 → ES
    ("WS", "bay_s1"), ("bay_s1", "bay_s2"), ("bay_s2", "bay_s3"),
    ("bay_s3", "bay_s4"), ("bay_s4", "bay_s5"), ("bay_s5", "bay_s6"),
    ("bay_s6", "ES"),
    # 主通道（唯一窄道）
    ("PW", "PE", "narrow"),
]
_NARROW = ("PW", "PE")               # 窄通道有序端点元组（西→东）

# ---------------------------------------------------------------------------
# 执行模型常量（单位：决策 tick = 1 s；速度 1 m/s → 每 tick 行驶 1 m）
# ---------------------------------------------------------------------------
V = 1.0            # 巡航速度 m/s（每决策 tick 内可行驶距离 = V × dt = 1 m）
DWELL_PICK = 3.0   # 取货站装卸停留时间 s
DWELL_DROP = 3.0   # 卸货站装卸停留时间 s
FOLLOW_GAP = 2.0   # 窄道 y=0 内“同向跟车”的最小安全间隔 s：
                   # 前车进入窄道后，后车须等其先行 ≥ 2 s 才允许进入
CHARGERS = ["WN", "EN", "WS", "ES"]     # 充电桩节点（东/西纵向通道中段贴墙）
CHARGE_LOW = 18.0  # 电量下限 %：idle 掉电 ≤ 18% → 触发自动回充
CHARGE_FULL = 95.0 # 回充完成阈值 %：充至 ≥ 95% 才回到 idle 待命

# ---------------------------------------------------------------------------
# 奖励常数（每 step 只结算“本步新发生事件”的增量，见 _event_reward）
#   · 时间成本 -0.02/tick：全程施加，鼓励尽早完工
#   · 完成任务：+25×prio（prio=1 最急 → 25；prio=2 → 50）+ 时效加成
#       max(0, 10 − latency/6)：越快完成加成越高，latency ≤ 60 s 内线性衰减
#   · 失败 -14×prio / 窄道等待 -0.4 / 站前排队 -0.2 / 无效动作 -0.5
#   · 收官奖励 +15：无失败且全部完成时于终止步一次性发放
# 状态奖励常数（每 tick 依状态，见 _state_reward；2026-09-14 为高负载调优）
#   · 有活却空闲 R_IDLE_LOAD：抑制“挑单空转”，把资源压到满负荷
#   · 临近超期 R_URGENT：截止感知的紧迫度塑形（贪心看不见剩余松弛度）
# ---------------------------------------------------------------------------
R_TIME = -0.02      # 每 tick 时间成本（贯穿整个 episode）
R_COMPLETE = 25.0   # 完成任务基准奖励（×prio）
R_DONE_BONUS = 15.0 # 整场无失败完成的收官奖励（终止步发放）
R_FAIL = -14.0      # 任务超期失败惩罚（×prio）
R_NARROW = -0.4     # 窄道入口等待（同向跟车/对向互斥被挡）单次惩罚
R_SITE_WAIT = -0.2  # 站点被占、站前排队重试单次惩罚
R_INVALID = -0.5    # 无效动作（越界槽位/抢同一单落选）单次惩罚
R_IDLE_LOAD = -0.06 # 有 pending 时每台空闲车每 tick 的惩罚（抑制空转）
R_URGENT = -0.08    # 临近超期塑形系数（×Σ max(0, wait/deadline − 0.5)）


def _euclid(a, b):
    """两节点 a、b 的欧氏直线距离（用于边权与窄道通行时间）。"""
    x1, y1 = NODE_COORD[a]
    x2, y2 = NODE_COORD[b]
    return math.hypot(x1 - x2, y1 - y2)


def _build_graph():
    """由 EDGES 建邻接表：g[node] = [(neighbor, 距离, 是否窄道), ...]
    无向化处理，并解析第三位 "narrow" 标志。"""
    g = {n: [] for n in NODE_COORD}
    for e in EDGES:
        a, b, *tail = e
        narrow = bool(tail and tail[0] == "narrow")
        d = _euclid(a, b)
        g[a].append((b, d, narrow))
        g[b].append((a, d, narrow))
    return g


GRAPH = _build_graph()


def shortest_route(src, dst):
    """Dijkstra 求 src → dst 最短路径。

    参数:
      src, dst: 节点名（如 "W0"、"DN"）
    返回:
      (总距离, 路径段列表)
      路径段每项 = (u, v, w, narrow)：起点/终点节点、边长(m)、是否窄道段，
      按行驶顺序排列（策略层用其估算 ETA / 智能体用其推进行驶）。
    """
    dist = {n: math.inf for n in GRAPH}
    prev = {}
    dist[src] = 0.0
    pq = [(0.0, src)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        if u == dst:
            break
        for (v, w, narrow) in GRAPH[u]:
            nd = d + w
            if nd < dist[v] - 1e-9:
                dist[v] = nd
                prev[v] = (u, w, narrow)
                heapq.heappush(pq, (nd, v))
    segs = []
    cur = dst
    while cur != src:
        pu, w, narrow = prev[cur]
        segs.append((pu, cur, w, narrow))
        cur = pu
    segs.reverse()
    total = sum(s[2] for s in segs)
    return total, segs


class _Task:
    """单任务数据结构（__slots__ 省内存）。

    字段:
      tid          任务编号（0..task_total-1，也是任务 id）
      site         {"src": 取货站, "dst": 卸货站, "sx","sy","dx","dy": 坐标}
      priority     优先级（1=最急, 2=次急；prio 越小越急）
      deadline     允许的“队列等待”上限 s（超期即失败）
      arr_time     到达时间（生成任务的 t）
      enter_time   进入 pending 队列的时间（初始=arr_time，被延迟处理时更新）
      status       pending | assigned | done | failed
      assign_time / done_time / latency  接单/完成时刻、总时延（完成时填）
    """
    __slots__ = ("tid", "site", "priority", "deadline", "arr_time",
                 "enter_time", "status", "assign_time", "done_time", "latency")

    def __init__(self, tid, site, priority, deadline, arr_time):
        self.tid = tid
        self.site = site
        self.priority = priority
        self.deadline = deadline
        self.arr_time = arr_time
        self.enter_time = arr_time
        self.status = "pending"     # pending | assigned | done | failed
        self.assign_time = None
        self.done_time = None
        self.latency = None


class FleetEnv:
    def __init__(self, num_agvs=3, M_max=5, task_total=8,
                 interval=(10.0, 22.0), prio2_ratio=0.65,
                 deadline_hi=(240, 340), deadline_lo=(150, 210),
                 battery_enabled=True, max_steps=600, seed=None,
                 speedup=1.0):
        # ============================================================
        # 构造参数说明（课程 lite/mid/high 三档须保持 num_agvs 与 M_max
        # 不变，obs/act 维度才一致 → 才能热启迁移权重，见 make_env）
        #   num_agvs        AGV 数量 = 车队规模 N（决定 obs 中 one-hot 维度）
        #   M_max           每车可见/可选的“队首任务槽”上限；
        #                   动作空间 act_dim = M_max + 1（0=等待，1..M=接队首第 k 个）
        #   task_total      单集生成的任务总数（负载强度）
        #   interval        (min,max) 相邻任务到达间隔的均匀采样区间 s；
        #                   越小 → 任务到得越密 → 负载越高
        #   prio2_ratio     生成任务中“较高优先级 prio=2”的占比，
        #                   其余 1−prio2_ratio 为 prio=1（最急档）
        #   deadline_hi     prio=1（最急）任务 deadline 的采样区间 s
        #   deadline_lo     prio=2（次急）任务 deadline 的采样区间 s
        #   battery_enabled 是否启用“低电回充”电量模型（训练默认 True）
        #   max_steps       单集最大步数上限（超过则强制终止）
        #   seed            随机种子（决定任务流/到达间隔，可复现实验）
        #   speedup         加速倍率：只影响 idle 掉电速率（0.02/speedup），
        #                   让仿真决策可以比真实时钟更快
        # ============================================================
        self.num_agvs = num_agvs
        self.M_max = M_max
        self.task_total = task_total
        self.interval = interval
        self.prio2_ratio = prio2_ratio
        self.deadline_hi = deadline_hi
        self.deadline_lo = deadline_lo
        self.battery_enabled = battery_enabled
        self.max_steps = int(max_steps)
        self.speedup = speedup
        self.rng = random.Random(seed)
        # 维度：act = 等待 + M_max 个槽；obs 布局见 _obs_one docstring
        # ★ v2.1 货架语义化：单槽 = 4 标量 + 取货 canonical one-hot(N_SITES=16)
        #   + 卸货 canonical one-hot(16) = 4 + 32 = **36**
        #   （one-hot 用 16 位 canonical 索引，取/卸同一套索引 → C++ 侧零映射、
        #   与 target_code 同源，最不易错位）
        #   obs_dim = (num_agvs+4) + M_max*36 + (num_agvs-1)*4 + 4
        #   3 车 / M_max=6 → 7 + 216 + 8 + 4 = **235**（v2 为 91，v1 为 67）
        self.act_dim = M_max + 1
        self.obs_dim = (num_agvs + 4) + M_max * (4 + 2 * N_SITES) + (num_agvs - 1) * 4 + 4
        self.dwell_pick = DWELL_PICK
        self.dwell_drop = DWELL_DROP
        self._event = 0.0            # 事件累加器（step 用快照差量，不用累计值）
        self.reset()

    # ============================ 重置 ============================
    def reset(self, seed=None):
        """重置一集并返回初始联合观测 obs[N, obs_dim]。

        参数:
          seed: 覆盖随机种子（None=沿用构造时的 rng，便于课程续训）
        重置项：时钟/统计计数器/任务队列/到达表/车辆/站点与窄道占用。
        车辆初始位置从起始节点列表轮询分配（W0, E0, Em3, ...）。
        """
        if seed is not None:
            self.rng = random.Random(seed)
        self.time = 0.0
        self.completed = 0
        self.failed = 0
        self.narrow_events = 0
        self.site_waits = 0
        self.invalid_acts = 0
        self.latencies = []
        self.pending = []
        self.assigned = []
        self.done_log = []          # 完成任务 dict 记录（含 prio/lat/deadline）
        self._arrivals = []
        self._arr_idx = 0
        self._gen_arrivals()
        self._enter_arrivals()
        self.cars = []
        # ★ 地图 v2：起始节点换成新拓扑名（PW/PE = 主通道两端，其余为纵向通道交叉点）
        start_nodes = ["PW", "PE", "ES", "WN", "WS", "EN"]
        for i in range(self.num_agvs):
            self.cars.append(self._new_car(i, start_nodes[i % len(start_nodes)]))
        self.site_occ = {s: None for s in SITE_COORD}   # site -> car_id
        # 窄道 y=0：同向跟车（2s 间距），对向等整道空出
        self.narrow_dir = None        # 道内行驶方向 (+1 西→东 / -1)
        self.narrow_ins = {}          # car_id -> enter_time
        self._event = 0.0
        return self._obs()

    def _new_car(self, i, node):
        """新建第 i 台车状态字典（起点 node）。

        字段: id/node/phase（idle|to_pick|pick|to_drop|drop|charge）/
              task(当前任务)/route(剩余路径段)/seg(段索引)/prog(段内进度 m)/
              wait(等待计时)/wait_kind(等待原因:narrow|site)/charge(电量%)/
              drain(idle 掉电速率 %/tick，受 speedup 缩放)
        """
        return {"id": i, "node": node, "phase": "idle",
                "task": None, "route": [], "seg": 0, "prog": 0.0,
                "wait": 0.0, "wait_kind": None, "charge": 100.0,
                "drain": 0.02 / self.speedup}

    def _gen_arrivals(self):
        """按任务流参数生成全集的到达表 self._arrivals。

        每任务: 抽优先级（prio2_ratio 决定 prio=2 比例）→ 依优先级选
        deadline 采样区间(deadline_lo/hi) → 随机取/卸货站点对 → 累加
        随机间隔（interval 均匀采样）得到到达时刻 t。间隔越小任务越密。
        """
        t = 0.0
        for k in range(self.task_total):
            if self.rng.random() < self.prio2_ratio:
                prio, (dl0, dl1) = 2, self.deadline_lo
            else:
                prio, (dl0, dl1) = 1, self.deadline_hi
            src = self.rng.choice(PICKUP_SITES)
            dst = self.rng.choice(DROPOFF_SITES)
            # ★ v2.1：货架位同时出现在取/卸列表里，避免 src==dst（同一点自取自送）
            while dst == src:
                dst = self.rng.choice(DROPOFF_SITES)
            dl = int(round(self.rng.uniform(dl0, dl1)))
            self._arrivals.append((t, _Task(k, {"src": src, "dst": dst,
                                                "sx": SITE_COORD[src][0],
                                                "sy": SITE_COORD[src][1],
                                                "dx": SITE_COORD[dst][0],
                                                "dy": SITE_COORD[dst][1]},
                                            prio, dl, t)))
            t += self.rng.uniform(*self.interval)

    def _enter_arrivals(self):
        """把到达时刻 ≤ 当前 time 的任务全部转入 pending 队（先到先排）。"""
        while self._arr_idx < len(self._arrivals) and \
                self._arrivals[self._arr_idx][0] <= self.time:
            tk = self._arrivals[self._arr_idx][1]
            tk.enter_time = self.time
            self.pending.append(tk)
            self._arr_idx += 1

    # ============================ 观测（obs 布局见 _obs_one） ============================
    def _onehot(self, n, k):
        """n 维 one-hot 向量（第 k 位=1），用于 AGV 身份/站点编码。"""
        v = np.zeros(n, dtype=np.float32)
        v[k] = 1.0
        return v

    def _phase_code(self, car):
        """相位枚举编码：idle/to_pick/pick/to_drop/drop/charge → 0..5"""
        return {"idle": 0, "to_pick": 1, "pick": 2, "to_drop": 3,
                "drop": 4, "charge": 5}[car["phase"]]

    def _target_code(self, car):
        """目标位置编码（归一化后入 obs；★ 地图 v2 已扩到 8 个站点）。

        0            = 无任务(空闲)
        1..N_SITES   = 去第 k 个站点（顺序见 SITE_COORD：8 港 + 8 货架位）
        N_SITES+1    = 去充电桩
        取值域 [0, N_SITES+1]；入 obs 时除以 (N_SITES+1) 归一化。
        """
        if car["phase"] == "charge":
            return float(N_SITES + 1)
        t = car["task"]
        if t is None:
            return 0.0
        if car["phase"] in ("to_pick", "pick"):
            return float(1 + SITE_IDX[t.site["src"]])
        return float(1 + SITE_IDX[t.site["dst"]])

    def _eta(self, car):
        """估算该车完成“当前全部剩余工作”还需的秒数（截断 ≤600 归一化用）。

        计算口径（与执行模型一一对应）：
          idle    → 0（随时可接单）
          charge  → 剩余充电量（充速 1%/s）
          pick    → 剩余取货停留 + 之后还要一趟卸货(dwell_drop)
          drop    → 剩余卸货停留
          to_pick / to_drop → 剩余路程/v + 到站后的装卸停留(dwell_pick/drop)
        这个量让 critic/策略“看见”每车多久能腾出，是排队与派单权衡的关键信号。
        """
        ph = car["phase"]
        if ph == "idle":
            return 0.0
        if ph == "charge":
            return (100.0 - car["charge"])
        rem = 0.0
        if ph == "pick":
            rem = car["wait"] + self.dwell_drop
        elif ph == "drop":
            rem = car["wait"]
        elif ph in ("to_pick", "to_drop"):
            total = sum(s[2] for s in car["route"])       # 全程边长和
            walked = sum(s[2] for s in car["route"][:car["seg"]]) + car["prog"]
            rem = (total - walked) / V
            if ph == "to_pick":
                rem += self.dwell_pick + self.dwell_drop  # 还要取+送两趟装卸
            else:
                rem += self.dwell_drop
        return min(rem, 600.0)

    def _obs_one(self, i):
        """构造第 i 台 AGV 的局部观测向量（长度 obs_dim，各段维度与拼接顺序如下）。

        观测 = 自身块 + 队首任务窗口块 + 他车块 + 全局块，全部归一化到 [0,1]，
        使策略对量纲不敏感、利于 MLP 收敛。段布局（与 __init__ 的 obs_dim 对应）：
        ┌─ A. 自身块: num_agvs + 4
        │    [0, num_agvs)   AGV 身份 one-hot（告诉共享策略“我是几号车”）
        │    [+0] 是否 idle（空闲可接单）
        │    [+1] 电量/100
        │    [+2] ETA/600（腾空还需多久）
        │    [+3] target_code/(N_SITES+1)（当前目标类别，见 _target_code）
        ├─ B. 队首任务窗口: M_max × 36（每槽 36 维，不足补 0） ← v2.1：12→36
        │    槽 k 对应 pending[k]（先到先排的队首 M_max 个）：
        │    [+0] 槽存在标志(1.0/0.0)  [+1] 队列等待 wait/600
        │    [+2] 优先级归一化 (3-prio)/2（1 最急→1.0）
        │    [+3..18]  取货点 canonical one-hot（16：8 港 + 8 货架位）
        │    [+19..34] 卸货点 canonical one-hot（16）
        │    [+35]     是否已接近超期（wait > 0.6×deadline）
        ├─ C. 他车块: (num_agvs-1) × 4（按 j≠i 顺序排布）
        │    [+0] 忙闲(非 idle=1) [+1] 电量/100 [+2] ETA/600 [+3] target_code/(N_SITES+1)
        └─ D. 全局块: 4
            [+0] 时钟 time/600  [+1] 空闲车占比 n_idle/num_agvs
            [+2] 队长度量 min(len(pending)/12, 1)  [+3] 队内最老任务等待/600
        """
        o = []
        car = self.cars[i]
        o.extend(self._onehot(self.num_agvs, i))
        o.append(1.0 if car["phase"] == "idle" else 0.0)
        o.append(car["charge"] / 100.0)
        o.append(self._eta(car) / 600.0)
        o.append(self._target_code(car) / float(N_SITES + 1))
        view = self.pending[:self.M_max]
        for k in range(self.M_max):
            if k < len(view):
                tk = view[k]
                wait = self.time - tk.enter_time
                o.append(1.0)
                o.append(min(wait / 600.0, 1.0))
                o.append((3 - tk.priority) / 2.0)
                for s in SITE_COORD:             # 取货点 canonical one-hot（16 位）
                    o.append(1.0 if tk.site["src"] == s else 0.0)
                for s in SITE_COORD:             # 卸货点 canonical one-hot（16 位）
                    o.append(1.0 if tk.site["dst"] == s else 0.0)
                o.append(1.0 if wait > 0.6 * tk.deadline else 0.0)
            else:
                o.extend([0.0] * (4 + 2 * N_SITES))  # 无任务：整槽填 0（36 维）
        for j in range(self.num_agvs):
            if j == i:
                continue
            cj = self.cars[j]
            o.append(0.0 if cj["phase"] == "idle" else 1.0)
            o.append(cj["charge"] / 100.0)
            o.append(self._eta(cj) / 600.0)
            o.append(self._target_code(cj) / float(N_SITES + 1))
        n_idle = sum(1 for c in self.cars if c["phase"] == "idle")
        oldest = max([self.time - t.enter_time for t in self.pending],
                     default=0.0)
        o.append(self.time / 600.0)
        o.append(n_idle / self.num_agvs)
        o.append(min(len(self.pending) / 12.0, 1.0))
        o.append(min(oldest / 600.0, 1.0))
        return np.asarray(o, dtype=np.float32)

    def _obs(self):
        """联合观测：把 N 台车各自的局部 obs 叠成 [N, obs_dim] 矩阵
        （MAPPO 的“分散观测、集中训练”输入格式）。"""
        return np.stack([self._obs_one(i) for i in range(self.num_agvs)])

    def get_masks(self):
        """动作合法性掩码 [N, act_dim]：1=该槽位合法，0=非法。
        规则：a=0（等待）永远合法；仅 idle 车可接单（接 pending 队首 M_max 个）；
        忙车只能选 0（等当前工作结束）。mask 让策略无法选到无效动作 → 加速收敛。"""
        masks = np.zeros((self.num_agvs, self.act_dim), dtype=np.float32)
        for i, car in enumerate(self.cars):
            masks[i, 0] = 1.0
            if car["phase"] == "idle":
                for k in range(min(len(self.pending), self.M_max)):
                    masks[i, k + 1] = 1.0
        return masks

    # ============================ 主循环 ============================
    def step(self, actions):
        """执行一步联合动作并推进 1 决策 tick（1 s）。

        参数:
          actions: [N] int 数组/列表。车 i 的动作：0=等待；a(1..M)=接
                   pending 队首第 a 个任务（需先经 get_masks 保证合法）。
        流程: 派单 _dispatch（空闲车抢单）→ 推进一 tick _tick（行驶/装卸/
              窄道互斥/掉电）→ 检查超期 _check_overdue → 结算事件奖励。
        返回: (obs[N,obs_dim], reward标量, done bool, info dict)
        说明: 奖励是**共享标量**（每车同值）——MAPPO 集中式训练把所有车当
              一个团队优化，不做个体奖励分配，天然鼓励协作不抢功。
        """
        dl0 = len(self.done_log)
        f0, n0, s0, i0 = (self.failed, self.narrow_events,
                          self.site_waits, self.invalid_acts)
        self._dispatch(actions)
        self._tick(1.0)
        self._check_overdue()
        r = R_TIME                       # 每步基础时间成本
        r += self._event_reward(dl0, f0, n0, s0, i0)   # 本步事件增量奖励
        r += self._state_reward()        # 本步状态奖励（空转/临期塑形）
        done = self._is_done()
        if done and self.completed == self.task_total and self.failed == 0:
            r += R_DONE_BONUS            # 完美收官：终止步一次性 +15
        return self._obs(), r, done, {"time": self.time}

    def _dispatch(self, actions):
        """把空闲车的“选单动作”解析为正式派单（本步第一件事）。

        裁决顺序：只有 idle 车能抢；动作 >len(pending) 或指向非 pending 任务
        记为无效；同一任务多车抢 → 只派给 id 最小者，其余记无效（索引快照式，
        先收集 wanted 再统一派发，避免边派边删队列导致错配）。
        """
        # 索引快照式裁决：先锁定“想选哪个任务”，冲突只保留 id 最小车，
        # 避免中途 pop 造成“拿错任务”的错配。
        wanted = {}
        for i, car in enumerate(self.cars):
            if car["phase"] != "idle":
                continue
            a = int(actions[i])
            if a <= 0 or a > len(self.pending):
                if a > self.act_dim - 1 or a > len(self.pending):
                    self.invalid_acts += 1  # 越界：等待/空槽动作
                continue
            tk = self.pending[a - 1]
            if tk.status != "pending":
                self.invalid_acts += 1
                continue
            wanted.setdefault(tk, []).append(i)
        for tk, ids in wanted.items():
            ids.sort()
            if len(ids) > 1:
                self.invalid_acts += len(ids) - 1   # 抢同一单：其余车无效
            i = ids[0]
            car = self.cars[i]
            tk.status = "assigned"
            tk.assign_time = self.time
            self.pending.remove(tk)
            self.assigned.append(tk)
            car["task"] = tk
            self._set_route(car, SITE_NODE[tk.site["src"]], "to_pick")

    def _set_route(self, car, dst_node, phase):
        """给车规划到 dst_node 的最短路径并进入行驶相位。
        重算整条路线（含任务切换/到站后去下一站），重置段游标/进度/等待。"""
        _, segs = shortest_route(car["node"], dst_node)
        car["route"] = segs
        car["seg"] = 0
        car["prog"] = 0.0
        car["wait"] = 0.0
        car["wait_kind"] = None
        car["phase"] = phase

    def _tick(self, dt):
        """时间前推 dt 秒：先把新到任务入队，再逐车推进。"""
        self.time += dt
        self._enter_arrivals()
        for car in self.cars:
            self._advance_car(car, dt)

    def _advance_car(self, car, dt):
        """单台车在一个 tick 内的状态机推进（dt 通常=1 s，可被窄道等待截断）。

        相位机流转：
          idle ──(掉电至 CHARGE_LOW)──→ charge ──(充满)──→ idle
          接单 ──→ to_pick ──到站且站点空──→ pick ──(dwell 结束)──→ to_drop
               ──到站且站点空──→ drop ──(dwell 结束)──→ idle（任务完成）
        途中任意行驶段若进窄道被拒则 wait>0 原地等待（wait_kind=narrow/site）。
        """
        ph = car["phase"]
        if ph == "idle":
            # idle 才掉电（干活不掉电）：掉到 CHARGE_LOW 触发回充
            if self.battery_enabled:
                car["charge"] = max(0.0, car["charge"] - 0.015 * dt)
                if car["charge"] <= CHARGE_LOW:
                    self._go_charge(car)
            return
        if ph == "charge":
            car["charge"] = min(100.0, car["charge"] + 1.0 * dt)  # 充速 1%/s
            if car["charge"] >= CHARGE_FULL:
                car["phase"] = "idle"
            return
        if ph in ("pick", "drop"):
            car["wait"] -= dt            # 装卸倒计时
            if car["wait"] > 0.0:
                return
            if ph == "pick":
                self.site_occ[car["task"].site["src"]] = None   # 释放源站
                car["phase"] = "to_drop"
                self._set_route(car, SITE_NODE[car["task"].site["dst"]],
                                "to_drop")
            else:
                tk = car["task"]
                self.site_occ[tk.site["dst"]] = None
                tk.status = "done"       # 落账：完成时刻/时延/统计
                tk.done_time = self.time
                tk.latency = self.time - tk.arr_time
                self.completed += 1
                self.latencies.append(tk.latency)
                self.done_log.append({"prio": tk.priority,
                                      "lat": tk.latency,
                                      "deadline": tk.deadline})
                if tk in self.assigned:
                    self.assigned.remove(tk)
                car["task"] = None
                car["phase"] = "idle"    # 车回到空闲，等待下一单
            return
        # to_pick / to_drop：沿路推进
        if car["wait"] > 0.0:
            car["wait"] -= dt            # 处于窄道/站点等待中
            return
        while dt > 1e-9 and car["phase"] in ("to_pick", "to_drop"):
            segs = car["route"]
            if car["seg"] >= len(segs):
                self._arrive_site(car)   # 当前段走完 → 到站（或排队）
                return
            u, v, w, narrow = segs[car["seg"]]
            if narrow and car["prog"] <= 0.0:
                # 段起点是窄道入口：先请求进入许可（同向跟车/对向互斥）
                wait_s = self._try_enter_narrow(car, u, v)
                if wait_s > 0.0:
                    car["wait"] = wait_s
                    car["wait_kind"] = "narrow"
                    self.narrow_events += 1
                    return
            step = V * dt                # 本 tick 可走距离
            rem = w - car["prog"]        # 当前段剩余距离
            if step >= rem - 1e-9:       # 本 tick 能走完当前段 → 跨过该段
                dt -= rem / V
                car["prog"] = w
                car["seg"] += 1
                car["prog"] = 0.0
                car["node"] = v          # 车到达段终点节点
                if narrow:
                    self.narrow_ins.pop(car["id"], None)   # 出窄道销账
                    if not self.narrow_ins:
                        self.narrow_dir = None             # 道空 → 清占用方向
            else:                        # 本 tick 走不完 → 段内推进后收工
                car["prog"] += step
                dt = 0.0

    def _try_enter_narrow(self, car, u, v):
        """窄道 y=0 入场控制（车位于窄道入口 (u,v)，请求进入）。

        规则（与真实仓库调度语义一致）：
          同向（沿道内既有方向走）→ 与前车保持 FOLLOW_GAP=2s 间距可跟入；
          对向（与道内当前方向相反）→ 须等道内全部清空才放行（保守串行）。
        返回需等待秒数（0=立即放行并登记进入时刻 self.narrow_ins）。
        """
        my_dir = 1.0 if (u, v) == _NARROW else -1.0   # 西→东=+1，东→西=-1
        dur = _euclid(*_NARROW) / V                   # 单程穿越时长 ≈19.2s
        if not self.narrow_ins:
            self.narrow_dir = my_dir                  # 空道：锁定方向直接进
            self.narrow_ins[car["id"]] = self.time
            return 0.0
        if self.narrow_dir != my_dir:
            # 对向：等最后进入者驶出窄道
            last = max(self.narrow_ins.values())
            wait = (last + dur) - self.time
            return max(0.0, wait)
        # 同向：保持跟车间距
        newest = max(self.narrow_ins.values())
        gap = FOLLOW_GAP - (self.time - newest)
        if gap > 0.0:
            return gap
        self.narrow_ins[car["id"]] = self.time
        return 0.0

    def _arrive_site(self, car):
        """行驶相位走完到达站点：站空则停靠装卸，被占则 0.5s 后重试（排队）。
        site_occ 容量=1：站被占的车原地等待（记一次 site_waits），
        释放后由下一次 tick 的 _arrive_site 接手。"""
        task = car["task"]
        if car["phase"] == "to_pick":
            site = task.site["src"]
            hold = self.dwell_pick
        else:
            site = task.site["dst"]
            hold = self.dwell_drop
        if self.site_occ[site] is not None:
            if car["wait_kind"] != "site":      # 首次排队才计数
                car["wait_kind"] = "site"
                self.site_waits += 1
            car["wait"] = 0.5                    # 0.5s 后重试入场
            return
        car["wait_kind"] = None
        self.site_occ[site] = car["id"]          # 占用站点（容量1）
        car["node"] = SITE_NODE[site]
        car["phase"] = "pick" if car["phase"] == "to_pick" else "drop"
        car["wait"] = hold                       # 装卸倒计时开始

    def _go_charge(self, car):
        """低电触发：选最近充电桩，瞬移挂充（充电期间不参与调度，obs 可见）。"""
        best, bd = None, math.inf
        for ch in CHARGERS:
            d, _ = shortest_route(car["node"], ch)
            if d < bd:
                bd, best = d, ch
        car["phase"] = "charge"
        car["node"] = best

    # ============================ 结算 ============================
    def _check_overdue(self):
        """逐任务查 pending 队列：队列等待 > deadline 即判失败（超期不候补）。
        注意口径是“等待时长”而非“总时延”——只要被及时接走执行就不算超期；
        失败的元凶是“无人接单/接了太晚”，与行驶中时间无关。"""
        keep = []
        for tk in self.pending:
            wait = self.time - tk.enter_time
            if wait > tk.deadline:
                tk.status = "failed"
                self.failed += 1
            else:
                keep.append(tk)
        self.pending = keep

    def _state_reward(self):
        """每 tick 的**状态**奖励（区别于 _event_reward 的事件增量），
        专为“高需求 + 资源紧张”场景塑形，两个可解释的引导信号：

          ① 有活却空闲（idle-when-work）：
             队内有 pending 时，每台空闲车 -|R_IDLE_LOAD|。高负载下空闲=浪费
             吞吐；该惩罚把策略推向“有单必派”（对齐贪心基线的天然优势），
             同时保留 RL 的全局协调（选哪一单）优势。
          ② 临近超期（deadline proximity）：
             对每台仍在 pending 的任务惩罚 max(0, wait/deadline − 0.5)，
             即“等待进度过半”后才线性施压。贪心用**静态** prio 惩罚、看不见
             剩余松弛度；该信号让 RL 学会按**紧迫度**而非仅优先级排序，
             提前清掉临期单，减少超期失败。

        返回标量（与 _event_reward 同口径累加进 step 奖励）。
        """
        r = 0.0
        if self.pending:
            n_idle = sum(1 for c in self.cars if c["phase"] == "idle")
            r += R_IDLE_LOAD * n_idle
            urgent = 0.0
            for tk in self.pending:
                frac = (self.time - tk.enter_time) / max(float(tk.deadline), 1.0)
                if frac > 0.5:
                    urgent += (frac - 0.5)
            r += R_URGENT * urgent
        return r

    def _event_reward(self, dl0, f0, n0, s0, i0):
        """结算“本 step 内新发生事件”的奖励增量（用计数器差量，杜绝重复计账）。

        参数 dl0/f0/n0/s0/i0 = step 开始时各计数器的快照。增量来源：
          · 完成任务 +R_COMPLETE×prio + 时效加成 max(0, 10−latency/6)
          · 任务失败 / 窄道等待 / 站前排队 / 无效动作 → 对应负项
        每 tick 的固定时间成本 R_TIME 由 step() 单独加，不在此处。"""
        r = 0.0
        # 本步新完成任务：按优先级 + 时效加成（时延越小奖励越高）
        for rec in self.done_log[dl0:]:
            r += R_COMPLETE * rec["prio"]
            r += max(0.0, 10.0 - rec["lat"] / 6.0)
        r += (self.failed - f0) * R_FAIL
        r += (self.narrow_events - n0) * R_NARROW
        r += (self.site_waits - s0) * R_SITE_WAIT
        r += (self.invalid_acts - i0) * R_INVALID
        return r

    def _is_done(self):
        """终止条件（三选一）：
        ① 超时 time ≥ max_steps；② 出现任务失败（提前止损，不浪费步数）；
        ③ 到达表耗尽且无 pending/assigned（自然收官：清空全场）。"""
        if self.time >= self.max_steps:
            return True
        if self.failed > 0:
            return True
        if self._arr_idx >= len(self._arrivals) and not self.pending \
                and not self.assigned:
            return True
        return False

    def episode_stats(self):
        """单集汇总统计（训练日志/评估报告的数据源）：
        completed/failed/total、完成率 completion、平均时延 avg_latency（无完成
        任务时为 None）、窄道等待/站前排队/无效动作次数、耗时 time。"""
        comp = self.completed / self.task_total if self.task_total else 0.0
        return {
            "completed": self.completed,
            "failed": self.failed,
            "total": self.task_total,
            "completion": comp,
            "avg_latency": (float(np.mean(self.latencies))
                            if self.latencies else None),
            "narrow_events": self.narrow_events,
            "site_waits": self.site_waits,
            "invalid_acts": self.invalid_acts,
            "time": self.time,
        }

    def render(self):
        """预留可视化钩子（当前无 GUI 渲染，训练用 CSV/曲线代替）。"""
        pass


def make_env(scenario="lite", seed=None, battery=False):
    """课程式递增难度场景工厂（架构统一：3 车 / M_max 6 → obs/act 维度一致，
    训练可按 lite→mid→high 顺序热启动迁移权重，见 train.py --curriculum）。

    场景参数表（task_total 越大、interval 越短 → 负载越高；deadline 越小 → 期限越紧）：
      ┌─────────┬────────────┬──────────────┬──────────────┬─────────┬──────────────┐
      │ scenario│ task_total │ interval(s)  │ deadline hi/lo│max_steps│ 语义         │
      ├─────────┼────────────┼──────────────┼──────────────┼─────────┼──────────────┤
      │ lite    │  5 稀疏    │ 20~34 慢到    │ 260~340/160~220 │ 500   │ 低负载宽裕   │
      │ mid     │  9         │ 12~24        │ 200~280/130~180 │ 600   │ 中负载       │
      │ high    │ 13 密集    │  8~14 快到    │ 150~220/ 90~140 │ 700   │ 高负载紧期限 │
      │ extreme │ 20 过载    │  4~7 更密     │ 100~150/ 60~100  │ 800   │ 高竞争(协调) │
      └─────────┴────────────┴──────────────┴──────────────┴─────────┴──────────────┘
    其余 FleetEnv 默认参数（prio2_ratio=0.65 等）三档共用。
    """
    cfg = {
        "lite": dict(num_agvs=3, M_max=6, task_total=5,
                     interval=(20.0, 34.0), max_steps=500,
                     deadline_hi=(260, 340), deadline_lo=(160, 220)),
        "mid": dict(num_agvs=3, M_max=6, task_total=9,
                    interval=(12.0, 24.0), max_steps=600,
                    deadline_hi=(200, 280), deadline_lo=(130, 180)),
        "high": dict(num_agvs=3, M_max=6, task_total=13,
                     interval=(8.0, 14.0), max_steps=700,
                     deadline_hi=(150, 220), deadline_lo=(90, 140)),
        # 2026-09-07 新增：高竞争/过载场景。到达速率(~0.2/s)超过车队服务速率，
        # 队列持续堆积 + 紧期限 → 贪心的局部最近派单会因重复抢单/不均衡而漏单，
        # 考验 RL 的全局协调（去重、负载均衡、急单优先），见 DEVELOPMENT_NOTES §7.5。
        "extreme": dict(num_agvs=3, M_max=6, task_total=20,
                        interval=(4.0, 7.0), max_steps=800,
                        deadline_hi=(100, 150), deadline_lo=(60, 100)),
        # 2026-09-14 新增：高需求 + 资源紧张（“临界饱和 + 紧期限”）。
        # 到达率≈0.118/s 约为 3 车服务容量（~0.086/s）的 1.4 倍，且期限远紧于
        # high（70~100 / 45~70 s）。实测贪心完成率仅 0.542（high 为 1.00）：
        # 队列持续积压 + 临期单密集，逼出“按剩余松弛度排序”的截止感知派单——
        # 贪心只有静态 prio 惩罚、看不见剩余时间，会为了近单放过临期急单；
        # RL 用 obs 里的 wait/deadline 可提前清临期单 → 更少超期、更高完成率。
        "surge": dict(num_agvs=3, M_max=6, task_total=18,
                      interval=(7.0, 10.0), max_steps=800,
                      deadline_hi=(70, 100), deadline_lo=(45, 70)),
    }
    return FleetEnv(**cfg[scenario], battery_enabled=battery, seed=seed)
