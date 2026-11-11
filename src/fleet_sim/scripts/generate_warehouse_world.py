"""
generate_warehouse_world.py
生成用于多 AGV 仓储调度仿真的 warehouse.world（24m × 16m 现代化智能仓库）。

地图要素：
  - 库房围墙 + 顶部照明
  - 4 排 × 8 列 双面立体货架（立柱 + 层板 + 彩色货箱），部分货架有空位/空架
  - 主通道与横向通道的虚线车道、安全黄线、双向行驶箭头
  - 装货（绿色）/ 卸货（蓝色）工作站：停车区 + 防撞柱 + 待装托盘货物
  - 4 处充电桩（黄色充电位 + 充电桩 + 状态灯）
  - 角落调度控制室（墙 + 办公桌 + 显示器）
  - 消防器材、通道安全立柱

使用方法：python3 scripts/generate_warehouse_world.py
输出：worlds/warehouse.world
"""

import os

WORLD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "worlds")
OUT_PATH = os.path.join(WORLD_DIR, "warehouse.world")

# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------

def pose(x, y, z, r=0, p=0, yaw=0):
    return f"{x} {y} {z} {r} {p} {yaw}"


def _geometry(gtype, gparams):
    if gtype == "box":
        return f'<geometry><box><size>{gparams[0]} {gparams[1]} {gparams[2]}</size></box></geometry>'
    if gtype == "cylinder":
        return (f'<geometry><cylinder><radius>{gparams[0]}</radius>'
                f'<length>{gparams[1]}</length></cylinder></geometry>')
    if gtype == "sphere":
        return f'<geometry><sphere><radius>{gparams[0]}</radius></sphere></geometry>'
    raise ValueError(f"unknown geometry: {gtype}")


def _visual(name, gtype, gparams, p, rgba):
    r, g, b, a = rgba
    return (
        f'      <visual name="{name}">\n'
        f'        <pose>{p}</pose>\n'
        f'        {_geometry(gtype, gparams)}\n'
        f'        <material>\n'
        f'          <ambient>{r} {g} {b} {a}</ambient>\n'
        f'          <diffuse>{r} {g} {b} {a}</diffuse>\n'
        f'          <specular>0.1 0.1 0.1 1</specular>\n'
        f'        </material>\n'
        f'      </visual>\n'
    )


def _collision(name, gtype, gparams, p):
    return (
        f'      <collision name="{name}">\n'
        f'        <pose>{p}</pose>\n'
        f'        {_geometry(gtype, gparams)}\n'
        f'      </collision>\n'
    )


def _link(lname, geoms):
    """geoms: (mode, gtype, gparams, pose, rgba)，mode: 'v' 仅视觉 / 'c' 仅碰撞 / 'b' 两者"""
    s = f'    <link name="{lname}">\n'
    for mode, gtype, gparams, p, rgba in geoms:
        if mode in ("v", "b"):
            s += _visual(lname + "_v", gtype, gparams, p, rgba)
        if mode in ("c", "b"):
            s += _collision(lname + "_c", gtype, gparams, p)
    s += "    </link>\n"
    return s


def model(name, p, links):
    return (
        f'  <model name="{name}">\n'
        f'    <static>true</static>\n'
        f'    <pose>{p}</pose>\n'
        + "\n".join(links)
        + f'  </model>\n'
    )


# ---------------------------------------------------------------------------
# 颜色
# ---------------------------------------------------------------------------

CONCRETE   = (0.72, 0.72, 0.75, 1.0)
FRAME      = (0.35, 0.37, 0.42, 1.0)
STEEL_BLUE = (0.28, 0.36, 0.50, 1.0)
DECK       = (0.45, 0.45, 0.50, 1.0)
WHITE      = (0.95, 0.95, 0.95, 1.0)
YELLOW     = (0.92, 0.75, 0.12, 1.0)
SAFETY_Y   = (0.90, 0.70, 0.10, 1.0)
GREEN      = (0.12, 0.68, 0.32, 1.0)
BLUE       = (0.15, 0.42, 0.85, 1.0)
CHARGER_Y  = (0.95, 0.80, 0.15, 1.0)
WOOD       = (0.55, 0.35, 0.15, 1.0)
RED        = (0.85, 0.20, 0.20, 1.0)
DARK       = (0.12, 0.12, 0.15, 1.0)
WALL_GRAY  = (0.78, 0.78, 0.80, 1.0)

BOX_PALETTE = [
    (0.90, 0.35, 0.10, 1.0),  # 橙色
    (0.20, 0.50, 0.90, 1.0),  # 蓝色
    (0.85, 0.20, 0.20, 1.0),  # 红色
    (0.25, 0.75, 0.35, 1.0),  # 绿色
    (0.95, 0.75, 0.10, 1.0),  # 黄色
    (0.55, 0.30, 0.75, 1.0),  # 紫色
]


# ---------------------------------------------------------------------------
# 货架（4 排 × 8 列，坐标与原地图一致）
# ---------------------------------------------------------------------------

# ===========================================================================
# ★ 地图 v2（2026-09-19）：为"阿克曼底盘 + 现有导航方法"重排布局
# ---------------------------------------------------------------------------
# 本文件的 RACK_ROWS/RACK_COLS 必须与以下两处**完全一致**（三处同步）：
#   · src/fleet_sim/scripts/generate_hand_map.py   SHELF_ROWS/SHELF_COLS
#   · src/fleet_manager/src/fleet_manager_node.cpp buildMap()
#
# 设计依据（全部来自实测，不是拍脑袋）：
#   · 阿克曼最小转弯半径 R_min = wheelbase/tan(max_steer) = 0.32/tan(0.6) = 0.468 m
#   · 掉头 180° 需净宽 ≥ 2·R_min + 车宽 = 1.32 m  → 主通道取 **2.4 m**
#   · 内切膨胀带(253) 要求车心离障碍 ≥ 0.51 m     → 2.4 m 给出 ±1.01 m 余量
#   · **死端口袋一律消除**：货架在"行内连续"（碰撞盒宽 = 列间距），
#     否则 AGV 会开进 1.6 m 缝隙、被相邻货架柱顶死
#     （实测：指令 |vx|=0.300 m/s 而实际车身速度 0.005 m/s，轮子空转）
#   · 竖向通道（东西两侧）保持 ≥2.75 m：站点/充电桩都在那里，且要能原地调头
RACK_ROWS = [5.2, 2.0, -2.0, -5.2]      # y 中心；主通道 = -1.2..1.2 → 2.4 m
# ★ 列中心再内缩 0.4 m：货架行端部由 x=±9.0 → **±8.6**，纵向通道 2.75 → 3.15 m。
#   原因（实测 lethal=46）：行端 x=±9.0 与主通道 y=±1.2 构成 90° 内角，
#   RPP 的大前视（L_d≈0.95~1.3，为保证曲率可行必须这么大）在这里切弯，
#   使车心距货架仅 0.14 m < 内切半径 0.19 m → 车自身格变成 253
#   → Hybrid-A* 报 `Starting point in lethal space` → 拒绝规划 → 永久卡死。
RACK_COLS = [-7.0, -4.2, -1.4, 1.4, 4.2, 7.0]
                                        # 8 列 × 2.25 间距 → 连续覆盖 x∈[-9.0, 9.0]
RACK_W, RACK_D, RACK_H = 0.6, 1.6, 1.5  # 视觉尺寸（保持"一个个货架"的观感）
RACK_PITCH = 2.8                        # ★ v3 列间距（8 列→6 列）
RACK_COLLIDE_W = RACK_W                 # ★ v3: 碰撞盒 = 视觉宽 0.6 → 留出 2.8-0.6 = **2.2m 通道**
#   为什么改回来：v2 把碰撞盒做成"连续屏障"是为了消除 1.6m 死端口袋；
#   现改为 6 列 × 2.8m 间距 → 通道 2.2m，**成了可穿行的过道**（不是死端）：
#     · 车宽 0.38、内切 0.19 → 2.2m 余量充足；
#     · A*(膨胀 0.4m) 自由带 2.2-0.8 = 1.4m；Nav2(膨胀 0.55) 剩 1.1m；
#     · 阿克曼 R_min=0.468 → 进出通道的 90° 转弯需 ≈0.66m 侧向 → 够。
#   行-列总数 4×6 = 24（原 4×8 = 32）→ 适当减少货架。
# ★ 端部货架在 y 向的收窄量（每侧）：给"主通道 ↔ 纵向通道"的 90° 大拐角做退让。
#   实测（未收窄时）：agv_0 停在 (-8.78, 1.01)，距行端内角 (-8.6, 1.1) 仅 **0.198 m**
#   ≈ 内切半径 0.19 → 自身格被判 253 → Hybrid-A* 报 `Starting point in lethal space`
#   （10 次）→ 拒绝出图 → 永久卡死。收窄 0.30 m/侧后该处到障碍余量升到 ~0.5 m。
#   注意：收窄只作用于**碰撞盒**，视觉货架尺寸不变（观感不受影响）。
END_SHRINK = 0.30


def rack_model(ri, ci):
    """双面立体货架：立柱 + 两层层板 + 顶梁 + 货箱（部分货位为空）"""
    x, y = RACK_COLS[ci], RACK_ROWS[ri]
    idx = ri * 8 + ci
    links = []

    # ★ 整体碰撞盒（仅碰撞，无视觉）：宽度取 RACK_COLLIDE_W（=列间距），
    #   使同排相邻货架**首尾相接**→整排成为连续屏障，不再留有 1.6 m 死端口袋。
    #   视觉仍用 RACK_W=0.6 的离散货架（观感与复杂度完全保留）。
    # ★ 两端货架在 y 向收窄 END_SHRINK×2，给走廊大拐角退让（见常量处说明）。
    coll_d = RACK_D - (END_SHRINK * 2.0
                       if ci in (0, len(RACK_COLS) - 1) else 0.0)
    links.append(_link("col", [("c", "box", (RACK_COLLIDE_W, coll_d, RACK_H),
                                pose(0, 0, RACK_H / 2), FRAME)]))

    # 四根立柱（纯视觉，物理由外层整体碰撞盒承担）
    for dx, dy in ((0.25, 0.75), (0.25, -0.75), (-0.25, 0.75), (-0.25, -0.75)):
        post_name = f"post_{str(dx).replace('.', '_').replace('-', 'm')}_{str(dy).replace('.', '_').replace('-', 'm')}"
        links.append(_link(post_name,
                           [("v", "box", (0.06, 0.06, RACK_H), pose(dx, dy, RACK_H / 2), STEEL_BLUE)]))

    # 层板与顶梁（纯视觉）
    links.append(_link("deck_low", [("v", "box", (RACK_W + 0.04, RACK_D + 0.04, 0.04), pose(0, 0, 0.25), DECK)]))
    links.append(_link("deck_mid", [("v", "box", (RACK_W + 0.04, RACK_D + 0.04, 0.04), pose(0, 0, 0.85), DECK)]))
    links.append(_link("top_beam", [("v", "box", (RACK_W + 0.04, RACK_D + 0.04, 0.03), pose(0, 0, 1.45), DECK)]))

    # 货箱（纯视觉）：默认两层各两箱；每 5 个货架空一个，每 5 个只放下层
    empty = (idx % 5 == 4)
    low_only = (idx % 5 == 3)
    if not empty:
        links.append(_link("box_l1", [("v", "box", (0.45, 0.45, 0.4), pose(0, 0.4, 0.47), BOX_PALETTE[idx % 6])]))
        links.append(_link("box_l2", [("v", "box", (0.45, 0.45, 0.4), pose(0, -0.4, 0.47), BOX_PALETTE[(idx + 2) % 6])]))
        if not low_only:
            links.append(_link("box_h1", [("v", "box", (0.45, 0.45, 0.4), pose(0, 0.4, 1.07), BOX_PALETTE[(idx + 4) % 6])]))
            links.append(_link("box_h2", [("v", "box", (0.45, 0.45, 0.4), pose(0, -0.4, 1.07), BOX_PALETTE[(idx + 5) % 6])]))

    return model(f"rack_r{ri}_c{ci}", pose(x, y, 0), links)


# ---------------------------------------------------------------------------
# 工作站（装货 / 卸货）
# ---------------------------------------------------------------------------

def station_model(name, cx, cy, zone_color, yaw=0.0, pallet_side=0.85):
    """工作站：停车区 + 停车线 + 防撞柱 + 待装托盘货物

    2026-09-19 地图 v2 修正三处（都对应实测故障）：
      1) `zone` 地贴 / `stopline` 停车线 → 由 "b"(两者) 改为 **"v"(纯视觉)**：
         它们是 1.2~2 cm 的薄板，作为碰撞体会卡住车轮（万向轮离地仅 0.035 m）；
         地面标线本就不该有碰撞。
      2) 托盘**不再自动朝 +x 伸出**：由调用方显式给 `pallet_side`，且应**朝墙**。
         原实现（`-0.85 if cx>0 else 0.85`）让西侧站点的托盘伸进主通道
         （实测占住 x≈-9.95），并且把旧停靠点 (±9.6, ±5) 套进了托盘物理盒里。
      3) 新增 `yaw`：南北侧站点旋转 90°，使托盘朝南北墙、停车位正对通道。
    """
    links = []
    links.append(_link("zone", [("v", "box", (1.2, 1.2, 0.02), pose(0, 0, 0.01), zone_color)]))
    links.append(_link("stopline", [("v", "box", (1.1, 0.06, 0.012), pose(0, 0.28, 0.011), WHITE)]))
    # 四角防撞柱 —— 2026-09-19 改为**纯视觉**（原因见下）
    # 原为真实碰撞体（0.5 m 高），但它们在停车位四周 0.6 m 处形成"柱阵"：
    # 阿克曼底盘 + RPP 前视 L_d≈1.0 m 时，进站必然切弯刮到柱子
    # （实测：agv_2 停在 (-9.67,-0.40)，距 bollard(-9.8,-0.6) 仅 0.13 m，
    #   指令 vx=+0.5 而实际车身速度 0.010 m/s = 顶在柱子上）。
    # 保留视觉以维持场景观感；站点真正需要避让的障碍由**托盘+货物**提供
    # （托盘仍为真实碰撞体，且故意不画进静态地图，交给雷达实时检测）。
    for dx, dy in ((0.6, 0.6), (0.6, -0.6), (-0.6, 0.6), (-0.6, -0.6)):
        bname = f"bollard_{str(dx).replace('.', '_').replace('-', 'm')}_{str(dy).replace('.', '_').replace('-', 'm')}"
        links.append(_link(bname, [("v", "cylinder", (0.035, 0.5), pose(dx, dy, 0.25), YELLOW)]))
    # 待装托盘货物（真实碰撞体；故意不画进静态地图，交给雷达实时检测）
    links.append(_link("pallet", [("b", "box", (0.9, 0.7, 0.1), pose(pallet_side, 0, 0.05), WOOD)]))
    links.append(_link("goods1", [("b", "box", (0.4, 0.4, 0.3), pose(pallet_side, 0, 0.25), BOX_PALETTE[1])]))
    links.append(_link("goods2", [("b", "box", (0.4, 0.4, 0.3), pose(pallet_side, 0, 0.55), BOX_PALETTE[2])]))
    return model(name, pose(cx, cy, 0, 0, 0, yaw), links)


# ---------------------------------------------------------------------------
# 充电桩
# ---------------------------------------------------------------------------

def charger_model(name, cx, cy, yaw=0.0, back=0.55):
    """充电桩：地贴(pad) + 桩体(body) + 指示灯。

    2026-09-19：① pad 由 "b" 改 **"v"**（2 cm 碰撞薄板会卡轮）；
               ② 桩体沿本地 +x 后退 `back` 米 → 把**停车位让出来**。
                  原实现桩体正压在停靠点上（车被要求停在桩体里）。
    """
    links = []
    links.append(_link("pad", [("v", "box", (0.4, 0.4, 0.02), pose(0, 0, 0.01), CHARGER_Y)]))
    links.append(_link("body", [("b", "box", (0.24, 0.24, 0.5), pose(back, 0, 0.25), STEEL_BLUE)]))
    links.append(_link("led", [("b", "cylinder", (0.02, 0.03), pose(back, 0, 0.53), GREEN)]))
    return model(name, pose(cx, cy, 0, 0, 0, yaw), links)


# ---------------------------------------------------------------------------
# 控制室
# ---------------------------------------------------------------------------

def control_room():
    links = []
    links.append(_link("floor", [("b", "box", (2.0, 1.5, 0.02), pose(0, 0, 0.01), (0.5, 0.5, 0.55, 1.0))]))
    links.append(_link("wall_east", [("b", "box", (0.1, 1.5, 1.2), pose(1.0, 0, 0.6), WALL_GRAY)]))
    links.append(_link("wall_north", [("b", "box", (2.0, 0.1, 1.2), pose(0, 0.75, 0.6), WALL_GRAY)]))
    links.append(_link("wall_south", [("b", "box", (2.0, 0.1, 1.2), pose(0, -0.75, 0.6), WALL_GRAY)]))
    # 办公桌椅
    links.append(_link("desk", [("b", "box", (1.2, 0.4, 0.4), pose(0, -0.35, 0.2), WOOD)]))
    links.append(_link("monitor", [("b", "box", (0.4, 0.02, 0.35), pose(0, -0.27, 0.55), DARK)]))
    links.append(_link("chair", [("b", "box", (0.4, 0.4, 0.45), pose(0.15, 0.05, 0.225), (0.2, 0.25, 0.35, 1.0))]))
    # ★ 地图 v2：原位置 (10.0, 6.2) 与新增的 dropoff_ne (10.4, 6.8) **完全重叠**
    #   （控制室占 x∈[9,11]、y∈[5.45,6.95]，把站点整个包在里面 → 车根本进不去）。
    #   移到货架屏障内部：该处在地图上本就是障碍，
    #   因此**不需要改调度器**（自然规避三处一致性问题）。
    # ★ 2026-09-27 修正：5.0 → 5.6。v3 地架半宽收窄为 0.30 后，c4/c5 列间走廊为
    #   x∈[4.5,6.7]（中心 5.6），原 5.0 时控制室西半边 (4.0~4.5) 压进 rack_r0_c4
    #   (3.9~4.5)，与货架物理重叠（激光/地图都能看出来）。平移后两侧各留 0.1m。
    #   注意：buildMap（fleet_manager_node_teb.cpp）须同步此包络 (4.6,4.4)-(6.65,6.0)。
    return model("control_room", pose(5.6, 5.2, 0), links)


# ---------------------------------------------------------------------------
# 消防器材
# ---------------------------------------------------------------------------

def fire_extinguisher(name, cx, cy):
    links = [_link("body", [("b", "cylinder", (0.05, 0.5), pose(0, 0, 0.25), RED)]),
             _link("top", [("b", "cylinder", (0.02, 0.05), pose(0, 0, 0.52), DARK)])]
    return model(name, pose(cx, cy, 0), links)


# ---------------------------------------------------------------------------
# 地面标线
# ---------------------------------------------------------------------------

def dashed_line(name, y, x_start, x_end, step=1.0, color=WHITE):
    """沿 X 方向的虚线车道标线"""
    links = []
    i, x = 0, x_start
    while x + 0.5 <= x_end:
        links.append(_link(f"dash_{i}", [("v", "box", (0.5, 0.1, 0.005), pose(x, y, 0.005), color)]))
        x += step
        i += 1
    return model(name, pose(0, 0, 0), links)


def solid_line(name, y, length, color=SAFETY_Y):
    return model(name, pose(0, 0, 0),
                 [_link("line", [("v", "box", (length, 0.08, 0.005), pose(0, y, 0.005), color)])])


def arrow_model(name, x, y, direction, color=WHITE):
    """地面方向箭头，direction=1 指向 +x，-1 指向 -x"""
    links = [_link("shaft", [("v", "box", (0.56, 0.14, 0.005), pose(x, y, 0.005), color)]),
             _link("fin1", [("v", "box", (0.18, 0.07, 0.005),
                            pose(x + direction * 0.30, y - 0.08, 0.005, 0, 0, direction * 0.785), color)]),
             _link("fin2", [("v", "box", (0.18, 0.07, 0.005),
                            pose(x + direction * 0.30, y + 0.08, 0.005, 0, 0, -direction * 0.785), color)])]
    return model(name, pose(0, 0, 0), links)


# ---------------------------------------------------------------------------
# 组装世界
# ---------------------------------------------------------------------------

def build_world():
    parts = ['<?xml version="1.0"?>']
    parts.append('<sdf version="1.6">')
    parts.append('  <world name="warehouse">')

    # 场景与照明
    parts.append('    <scene>')
    parts.append('      <ambient>0.72 0.72 0.75 1</ambient>')
    parts.append('      <background>0.88 0.88 0.92 1</background>')
    parts.append('      <shadows>true</shadows>')
    parts.append('    </scene>')

    # 地面与阳光
    parts.append('    <include><uri>model://ground_plane</uri></include>')
    parts.append('    <include><uri>model://sun</uri></include>')

    # 状态插件（供上层调度读取模型位姿）
    parts.append('    <plugin name="gazebo_ros_state" filename="libgazebo_ros_state.so">')
    parts.append('      <ros><namespace>/fleet_sim</namespace></ros>')
    parts.append('      <update_rate>10</update_rate>')
    parts.append('    </plugin>')

    # 顶部照明
    for i, (lx, ly) in enumerate(((-6, 3.5), (6, 3.5), (-6, -3.5), (6, -3.5))):
        parts.append(f'    <light type="point" name="light_{i}">')
        parts.append(f'      <pose>{lx} {ly} 4.5 0 0 0</pose>')
        parts.append('      <diffuse>0.85 0.85 0.85 1</diffuse>')
        parts.append('      <specular>0.2 0.2 0.2 1</specular>')
        parts.append('      <attenuation><range>14</range><constant>0.9</constant><linear>0.01</linear><quadratic>0.01</quadratic></attenuation>')
        parts.append('      <cast_shadows>false</cast_shadows>')
        parts.append('    </light>')

    # ---- 库房围墙 ----
    walls = [
        ("wall_east", (0.5, 16.5, 1.5), pose(12.0, 0, 0.75)),
        ("wall_west", (0.5, 16.5, 1.5), pose(-12.0, 0, 0.75)),
        ("wall_north", (24.5, 0.5, 1.5), pose(0, 8.0, 0.75)),
        ("wall_south", (24.5, 0.5, 1.5), pose(0, -8.0, 0.75)),
    ]
    for wname, size, p in walls:
        parts.append(model(wname, p, [_link("wall", [("b", "box", size, pose(0, 0, 0), CONCRETE)])]))

    # ---- 货架 ----
    for ri in range(len(RACK_ROWS)):
        for ci in range(len(RACK_COLS)):
            parts.append(rack_model(ri, ci))

    # ---- 工作站：装货（绿）×4 / 卸货（蓝）×4 —— 沿外围环线分布 ----
    # ★ 地图 v2：2 取 2 卸 → **4 取 4 卸**，均匀布置在"东西纵向通道 + 南北外围通道"
    #   上。效果：任务天然要求跨库行驶（例如北外围取货 → 西南卸货），
    #   并且必然争抢主通道(y=0, 2.4m)与横向通道 → 这才是多机协调的复杂度来源。
    #   `yaw` 让站点旋转到"托盘朝墙、停车位正对通道"；`pallet_side` 指向最近的墙。
    #   站点中心 = 停车位（与 fleet_manager 的 stops 表一致）。
    parts.append(station_model("station_pickup_w", -10.4, 0.0, GREEN, yaw=0.0, pallet_side=-0.85))
    parts.append(station_model("station_pickup_e", 10.4, 0.0, GREEN, yaw=0.0, pallet_side=+0.85))
    parts.append(station_model("station_pickup_n", 0.0, 6.8, GREEN, yaw=1.5707963, pallet_side=+0.85))
    parts.append(station_model("station_pickup_s", 0.0, -6.8, GREEN, yaw=1.5707963, pallet_side=-0.85))
    parts.append(station_model("station_dropoff_nw", -10.4, 6.8, BLUE, yaw=0.0, pallet_side=-0.85))
    parts.append(station_model("station_dropoff_ne", 10.4, 6.8, BLUE, yaw=0.0, pallet_side=+0.85))
    parts.append(station_model("station_dropoff_sw", -10.4, -6.8, BLUE, yaw=0.0, pallet_side=-0.85))
    parts.append(station_model("station_dropoff_se", 10.4, -6.8, BLUE, yaw=0.0, pallet_side=+0.85))

    # ---- 充电桩：移到东西纵向通道中段、贴墙 ----
    # 原四角位置让给了卸货站。桩体沿本地 +x 后退 0.55 m（西侧 yaw=π 使其朝西墙），
    # 停车位（=桩体坐标）空出来给 AGV。
    parts.append(charger_model("charger_0", -11.0, 3.6, yaw=3.1415927))
    parts.append(charger_model("charger_1", 11.0, 3.6, yaw=0.0))
    parts.append(charger_model("charger_2", 11.0, -3.6, yaw=0.0))
    parts.append(charger_model("charger_3", -11.0, -3.6, yaw=3.1415927))

    # ---- 货架拣货/补货位（★ 地图 v2.1：货架语义化）----
    # 横向通道 y=±3.6、x=-6/-2/2/6 处的地面存取位标记（**纯视觉，无碰撞**）。
    # 任务由"港→港穿梭"扩展为四类（货架→港 拣货 / 港→货架 补货 / 货架→货架 移库），
    # AGV 会真正开到货架跟前停靠。不新增碰撞几何，故不会重新引入死端口袋。
    BAY_C = (0.35, 0.55, 0.75, 1.0)   # 钢蓝：存取位标记
    for bi, (bx, by) in enumerate((
            # ★ 地图 v3：货架位对齐到**新列中心** x=±1.4 / ±4.2
            #   （v2.1 的 ±2/±6 在新布局下正好落在**通道**里 → 会"对着空地取货"）
            (-7.0, 3.6), (-4.2, 3.6), (-1.4, 3.6), (1.4, 3.6), (4.2, 3.6), (7.0, 3.6),
            (-7.0, -3.6), (-4.2, -3.6), (-1.4, -3.6), (1.4, -3.6), (4.2, -3.6), (7.0, -3.6))):
        parts.append(model(
            f"rack_bay_{bi}", pose(bx, by, 0),
            [_link("pad", [("v", "box", (0.6, 0.4, 0.005), pose(0, 0, 0.005), BAY_C)])]))

    # ---- ★ v3 新增 4 个外港地贴：南/北外围通道中段 ----
    #   刻意用**纯视觉 pad（"v"，无碰撞）**：外围通道宽仅 1.6 m，若按普通站点给
    #   四角防撞柱（±0.6 m）会把通道收窄到 1.2 m，A*(膨胀 0.4) 与 Nav2(0.55)
    #   都可能过不去 → **会堵死整条外围通道**。这里只画地贴，不产生碰撞体。
    for pi, (px_, py_) in enumerate(((-6.0, 6.8), (6.0, 6.8),
                                     (-6.0, -6.8), (6.0, -6.8))):
        parts.append(model(
            f"port_pad_{pi}", pose(px_, py_, 0),
            [_link("pad", [("v", "box", (1.2, 0.5, 0.005), pose(0, 0, 0.005), BAY_C)])]))

    # ---- 控制室 / 消防 ----
    parts.append(control_room())
    parts.append(fire_extinguisher("fire_ext_1", 0, 7.5))
    parts.append(fire_extinguisher("fire_ext_2", 0, -7.5))

    # ---- 地面标线（随地图 v2 更新）----
    # 主通道（y=0）与两条横向通道（y=±3.6）虚线车道
    parts.append(dashed_line("lane_main", 0.0, -11.4, 11.4, 1.0))
    parts.append(dashed_line("lane_north", 3.6, -11.4, 11.4, 1.0))
    parts.append(dashed_line("lane_south", -3.6, -11.4, 11.4, 1.0))
    # 安全黄线：主通道边缘 ±1.05（主通道 -1.2..1.2，宽 2.4 m）
    #            横向通道边缘 ±2.95 / ±4.25（横向通道 2.8..4.4）
    parts.append(solid_line("safe_line_y105", 1.05, 23.0))
    parts.append(solid_line("safe_line_ym105", -1.05, 23.0))
    parts.append(solid_line("safe_line_y295", 2.95, 23.0))
    parts.append(solid_line("safe_line_y425", 4.25, 23.0))
    parts.append(solid_line("safe_line_ym295", -2.95, 23.0))
    parts.append(solid_line("safe_line_ym425", -4.25, 23.0))
    # 双向行驶箭头
    parts.append(arrow_model("arrow_1", -6.0, 0.0, 1))
    parts.append(arrow_model("arrow_2", -3.0, 0.0, -1))
    parts.append(arrow_model("arrow_3", 3.0, 0.0, 1))
    parts.append(arrow_model("arrow_4", 6.0, 0.0, -1))

    parts.append('  </world>')
    parts.append('</sdf>')
    return "\n".join(parts)


if __name__ == "__main__":
    os.makedirs(WORLD_DIR, exist_ok=True)
    content = build_world()
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"world file written: {OUT_PATH} ({len(content.splitlines())} lines)")
