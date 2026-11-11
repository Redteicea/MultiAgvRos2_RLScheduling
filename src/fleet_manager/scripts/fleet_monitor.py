#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""fleet_monitor.py — AGV 车队实时监控终端界面（curses TUI，单文件便于维护）

订阅 `/fleet/status`（fleet_common/FleetStatus），实时显示：
  1. 调度概览：AGV 总数 / 各状态计数 / 任务统计 / 平均电量
  2. AGV 状态表：车号 · 状态 · 当前任务 · 位置(x,y) · 朝向 · 电量条
  3. 任务列表：任务号 · 优先级 · 取货点 · 卸货点 · 状态 · 归属车
  4. 状态栏：刷新周期 / 数据新鲜度 / 按键提示

用法
----
  # 方式一（推荐）：直接运行
  python3 src/fleet_manager/scripts/fleet_monitor.py
  # 方式二：安装后 ros2 run
  ros2 run fleet_manager fleet_monitor

按键
----
  q / ESC : 退出
  r       : 立即刷新
  + / -   : 调整刷新周期（0.2s ~ 5s）

依赖：仅 Python 标准库（curses）+ rclpy + fleet_common.msgs，无需第三方库。

设计说明（便于维护）
--------------------
  - `MonitorState`  : 纯数据缓存（最新一帧 status），与界面解耦
  - `FleetMonitorNode` : 只负责订阅 /fleet/status 并更新缓存
  - `draw_*` 系列   : 只负责把缓存画到 curses 窗口（纯渲染，无状态）
  - `dwidth/pad`    : 中文等宽字符的显示宽度计算（保证表格对齐）
  改样式只需动 `draw_*`；改数据源只需动 `FleetMonitorNode`。
"""
import curses
import math
import sys
import time
import unicodedata

import rclpy
from rclpy.node import Node
from fleet_common.msg import FleetStatus

# ----------------------------- 常量 / 配色 ----------------------------- #

STATUS_ORDER = [
    "OFFLINE", "IDLE", "NAV_PICKUP", "LOADING", "NAV_DROPOFF",
    "UNLOADING", "TO_CHARGER", "CHARGING", "YIELD", "FAULT",
]

# 状态 -> (中文简称, curses 颜色 pair 编号)
STATUS_STYLE = {
    "OFFLINE":     ("离线", 5),
    "IDLE":        ("空闲", 4),
    "NAV_PICKUP":  ("去取货", 1),
    "LOADING":     ("装货", 1),
    "NAV_DROPOFF": ("去卸货", 1),
    "UNLOADING":   ("卸货", 1),
    "TO_CHARGER":  ("去充电", 3),
    "CHARGING":    ("充电中", 3),
    "YIELD":       ("避让", 3),
    "FAULT":       ("故障", 2),
}

C_NORMAL, C_RUN, C_ERR, C_WARN, C_IDLE, C_DIM = 1, 2, 3, 4, 5, 6
C_TITLE, C_HEAD, C_BORDER = 7, 8, 9

STALE_SEC = 3.0          # 超过该秒数未收到 status 视为断流


# ----------------------------- 宽度工具 ----------------------------- #

def dwidth(s):
    """字符串显示宽度（东亚宽/全角字符算 2，其余算 1）。"""
    return sum(2 if unicodedata.east_asian_width(c) in ("W", "F") else 1 for c in s)


def pad(s, width, align="left"):
    """按显示宽度补齐/截断到 width，保证表格对齐。"""
    if dwidth(s) > width:
        out, w = "", 0
        for ch in s:
            cw = 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
            if w + cw > width:
                break
            out, w = out + ch, w + cw
        s, w = out, w
        return s + " " * (width - w)
    extra = width - dwidth(s)
    return (" " * extra + s) if align == "right" else (s + " " * extra)


def fnum(v, width=6, dec=1):
    """定宽浮点（右对齐）。"""
    return pad(f"{v:.{dec}f}", width, "right")


# ----------------------------- 数据缓存 ----------------------------- #

class MonitorState:
    """最新一帧 status 的缓存（与界面解耦）。"""

    def __init__(self):
        self.robots = []          # list[(robot_id, status, x, y, theta, battery, task_id)]
        self.tasks = []           # list[(task_id, priority, px, py, dx, dy, active)]
        self.last_stamp = 0.0     # 最近一次收到 status 的本地时间
        self.frames = 0           # 累计收到帧数

    def update(self, msg: FleetStatus):
        self.robots = [
            (r.robot_id, r.status, r.x, r.y, r.theta, r.battery, r.task_id,
             r.linear_vel, r.angular_vel)
            for r in msg.robots
        ]
        self.tasks = [
            (t.task_id, t.priority, t.pickup_x, t.pickup_y,
             t.dropoff_x, t.dropoff_y, t.is_active)
            for t in msg.tasks
        ]
        self.last_stamp = time.time()
        self.frames += 1

    @property
    def age(self):
        return time.time() - self.last_stamp if self.last_stamp else float("inf")


class FleetMonitorNode(Node):
    """只负责订阅 /fleet/status 并写入缓存。"""

    def __init__(self, state: MonitorState):
        super().__init__("fleet_monitor")
        self._state = state
        self.create_subscription(FleetStatus, "/fleet/status", self._on_status, 10)

    def _on_status(self, msg: FleetStatus):
        self._state.update(msg)


# ----------------------------- 渲染 ----------------------------- #

def init_colors():
    curses.start_color()
    try:
        curses.use_default_colors()
        bg = -1
    except curses.error:
        bg = curses.COLOR_BLACK
    curses.init_pair(C_NORMAL, curses.COLOR_WHITE, bg)
    curses.init_pair(C_RUN, curses.COLOR_CYAN, bg)
    curses.init_pair(C_ERR, curses.COLOR_RED, bg)
    curses.init_pair(C_WARN, curses.COLOR_YELLOW, bg)
    curses.init_pair(C_IDLE, curses.COLOR_GREEN, bg)
    curses.init_pair(C_DIM, curses.COLOR_BLUE, bg)
    curses.init_pair(C_TITLE, curses.COLOR_BLACK, curses.COLOR_CYAN)
    curses.init_pair(C_HEAD, curses.COLOR_YELLOW, bg)
    curses.init_pair(C_BORDER, curses.COLOR_BLUE, bg)


def add(W, y, x, s, attr=0):
    """安全写字符串（越界/异常忽略）。"""
    try:
        W.addstr(y, x, s, attr)
    except curses.error:
        pass


def hline(W, y, x, width, ch="─"):
    add(W, y, x, ch * width, curses.color_pair(C_BORDER))


def battery_bar(pct, width=10):
    """电量条：满格 █，空格 ░。"""
    filled = int(round(max(0.0, min(100.0, pct)) / 100.0 * width))
    return "█" * filled + "░" * (width - filled)


def status_attr(status):
    _, pair = STATUS_STYLE.get(status, ("?", C_NORMAL))
    return curses.color_pair(pair)


def draw_header(W, width, state, interval):
    """标题栏 + 数据新鲜度。"""
    title = " AGV 车队实时监控 "
    t = time.strftime("%Y-%m-%d %H:%M:%S")
    add(W, 0, 0, " " * width, curses.color_pair(C_TITLE))
    add(W, 0, 1, title, curses.color_pair(C_TITLE) | curses.A_BOLD)
    add(W, 0, 1 + dwidth(title), f" {t}", curses.color_pair(C_TITLE))

    age = state.age
    if age > STALE_SEC:
        txt, attr = f"⚠ 数据断流 {age:.1f}s", curses.color_pair(C_ERR) | curses.A_BOLD
    else:
        txt, attr = f"● 在线 {1000*interval:.0f}ms  帧#{state.frames}", curses.color_pair(C_TITLE)
    add(W, 0, max(1, width - dwidth(txt) - 2), txt, attr)


def draw_overview(W, y, width, state):
    """调度概览：AGV 计数 / 任务统计 / 平均电量。"""
    add(W, y, 2, "调度概览", curses.color_pair(C_HEAD) | curses.A_BOLD)

    n = len(state.robots)
    cnt = {}
    for r in state.robots:
        cnt[r[1]] = cnt.get(r[1], 0) + 1
    online = n - cnt.get("OFFLINE", 0)
    busy = sum(cnt.get(s, 0) for s in
               ("NAV_PICKUP", "LOADING", "NAV_DROPOFF", "UNLOADING"))
    idle = cnt.get("IDLE", 0)
    charge = cnt.get("CHARGING", 0) + cnt.get("TO_CHARGER", 0)
    fault = cnt.get("FAULT", 0)

    active = sum(1 for t in state.tasks if t[6])
    pending = len(state.tasks) - active
    done = sum(1 for r in state.robots if False)  # 占位：done 计数由 fleet_manager 不发布，忽略

    avg_b = (sum(r[5] for r in state.robots) / n) if n else 0.0

    parts = [
        ("车队", f"{online}/{n} 在线", C_NORMAL),
        ("执行中", str(busy), C_RUN),
        ("空闲", str(idle), C_IDLE),
        ("充电", str(charge), C_WARN),
        ("故障", str(fault), C_ERR if fault else C_NORMAL),
        ("在途任务", str(active), C_RUN),
        ("待分配", str(pending), C_WARN),
        ("平均电量", f"{avg_b:.0f}%", C_NORMAL),
    ]
    x = 14
    for label, val, cp in parts:
        seg = f"{label} {val}"
        add(W, y, x, seg, curses.color_pair(cp) | curses.A_BOLD)
        x += dwidth(seg) + 4
        if x > width - 18:
            y += 1
            x = 14
    return y


def draw_agents(W, y, width, state):
    """AGV 状态表。"""
    add(W, y, 2, "AGV 状态", curses.color_pair(C_HEAD) | curses.A_BOLD)
    y += 1

    # 列宽（显示宽度）
    cols = [("车号", 7), ("状态", 8), ("任务", 8), ("位置 (x, y)", 18),
            ("朝向", 7), ("线速度", 10), ("角速度", 10), ("电量", 16)]
    hdr = "  " + "".join(pad(f" {c[0]} ", c[1]) for c in cols)
    add(W, y, 0, pad(hdr, width), curses.color_pair(C_HEAD) | curses.A_BOLD)
    y += 1
    hline(W, y, 2, width - 4)
    y += 1

    for r in state.robots:
        rid, st, x, yy, th, bat, tid, lv, av = r
        cname, _ = STATUS_STYLE.get(st, (st, C_NORMAL))
        task_s = f"#{tid}" if tid and tid >= 0 else "—"
        line = (
            "  "
            + pad(f" agv_{rid} ", 7)
            + pad(f" {cname} ", 8)
            + pad(f" {task_s} ", 8)
            + pad(f" ({x:6.2f},{yy:6.2f}) ", 18)
            + pad(f" {math.degrees(th):5.0f}° ", 7)
            + pad(f" {lv:5.2f} m/s ", 10)
            + pad(f" {av:+5.2f}r/s ", 10)
        )
        add(W, y, 0, line, curses.color_pair(C_NORMAL))
        # 电量条（按电量配色）
        bcp = C_ERR if bat < 25 else (C_WARN if bat < 50 else C_IDLE)
        add(W, y, dwidth(line), f" {bat:5.1f}% ", curses.color_pair(bcp) | curses.A_BOLD)
        add(W, y, dwidth(line) + 9, battery_bar(bat, 10), curses.color_pair(bcp))
        # 状态列着色（覆盖）
        add(W, y, 2 + 7 + 1, f"{cname}", status_attr(st) | curses.A_BOLD)
        y += 1
    return y


def draw_tasks(W, y, width, state):
    """任务列表（含归属车）。"""
    add(W, y, 2, "任务列表", curses.color_pair(C_HEAD) | curses.A_BOLD)
    y += 1

    owner = {r[6]: f"agv_{r[0]}" for r in state.robots if r[6] is not None and r[6] >= 0}

    cols = [("任务", 7), ("优先级", 9), ("取货点", 17), ("卸货点", 17), ("状态", 10), ("执行车", 9)]
    hdr = "  " + "".join(pad(f" {c[0]} ", c[1]) for c in cols)
    add(W, y, 0, pad(hdr, width), curses.color_pair(C_HEAD) | curses.A_BOLD)
    y += 1
    hline(W, y, 2, width - 4)
    y += 1

    if not state.tasks:
        add(W, y, 4, "(当前无任务)", curses.color_pair(C_DIM))
        return y + 1

    for t in sorted(state.tasks, key=lambda z: (z[1], z[0])):
        tid, pri, px, py, dx, dy, act = t
        st_s, cp = ("执行中", C_RUN) if act else ("待分配", C_WARN)
        own = owner.get(tid, "—")
        pri_s = f"P{pri}" + ("!" if pri == 1 else "")
        line = (
            "  "
            + pad(f" #{tid} ", 7)
            + pad(f" {pri_s} ", 9)
            + pad(f" ({px:5.1f},{py:5.1f}) ", 17)
            + pad(f" ({dx:5.1f},{dy:5.1f}) ", 17)
            + pad(f" {st_s} ", 10)
            + pad(f" {own} ", 9)
        )
        add(W, y, 0, line, curses.color_pair(C_NORMAL))
        add(W, y, 2 + 7 + 9 + 17 + 17 + 1, f"{st_s}", curses.color_pair(cp) | curses.A_BOLD)
        y += 1
    return y


def draw_footer(W, height, width, state, interval):
    """底部状态栏。"""
    y = height - 1
    add(W, y, 0, " " * width, curses.color_pair(C_TITLE))
    help_s = " q 退出 | r 刷新 | + / - 周期 "
    add(W, y, 1, help_s, curses.color_pair(C_TITLE) | curses.A_BOLD)
    info = f"周期 {interval:.2f}s | 刷新率 {1.0/max(interval,1e-3):.1f}Hz | 在途 {len(state.tasks)} 任务 "
    add(W, y, max(dwidth(help_s) + 3, width - dwidth(info) - 2), info, curses.color_pair(C_TITLE))


def draw(stdscr, state, interval):
    """整屏重绘。"""
    stdscr.erase()
    h, w = stdscr.getmaxyx()
    if h < 14 or w < 60:
        add(stdscr, 0, 0, "终端窗口太小（需 ≥ 60x14），请放大。", curses.color_pair(C_ERR))
        stdscr.refresh()
        return

    draw_header(stdscr, w, state, interval)
    y = 2
    y = draw_overview(stdscr, y, w, state)
    y += 1
    hline(stdscr, y, 2, w - 4)
    y += 1
    y = draw_agents(stdscr, y, w, state)
    y += 1
    hline(stdscr, y, 2, w - 4)
    y += 1
    draw_tasks(stdscr, y, w, state)
    draw_footer(stdscr, h, w, state, interval)
    stdscr.refresh()


# ----------------------------- 主循环 ----------------------------- #

def curses_main(stdscr, node, state):
    curses.curs_set(0)          # 隐藏光标
    stdscr.nodelay(True)        # 非阻塞 getch
    stdscr.keypad(True)
    init_colors()

    interval = 0.3              # 刷新周期（秒）
    next_draw = 0.0

    while True:
        # 1) 按键
        try:
            ch = stdscr.getch()
        except curses.error:
            ch = -1
        if ch in (ord("q"), 27):         # q / ESC
            break
        elif ch == ord("r"):
            next_draw = 0.0
        elif ch in (ord("+"), ord("=")):
            interval = max(0.2, interval - 0.2)
        elif ch in (ord("-"), ord("_")):
            interval = min(5.0, interval + 0.2)

        # 2) 推进 ROS（非阻塞）
        rclpy.spin_once(node, timeout_sec=0.0)

        # 3) 定时重绘
        now = time.time()
        if now >= next_draw:
            next_draw = now + interval
            draw(stdscr, state, interval)

        time.sleep(0.03)


def main():
    rclpy.init()
    state = MonitorState()
    node = FleetMonitorNode(state)

    # 等首帧（最多 3s），避免开局空白
    t0 = time.time()
    try:
        while state.frames == 0 and time.time() - t0 < 3.0:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass

    try:
        curses.wrapper(curses_main, node, state)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
