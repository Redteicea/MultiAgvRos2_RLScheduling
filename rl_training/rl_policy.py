"""
rl_policy.py - Phase 6 (Task 6.6) 策略部署接口（离线可运行）
================================================================================
加载训练导出的 policy.pt（仅 actor），把外部传入的“车辆状态 + 待分配任务”
编码成与训练**完全一致**的观测 → 门控掩码 → argmax 输出每台 AGV 的任务分配。

设计要点：
  · 不依赖 ROS：供 Phase 7 FleetManager(C++/ROS) 通过子进程/HTTP/Python 桥调用；
  · 复用 FleetEnv 的 _obs()/get_masks() 做编码 —— 状态表示唯一来源，
    训练/推理零偏差（这是 sim2real 可靠性的前提）；
  · 推理 = act_greedy（argmax 确定性），结果可复现。

用法：
  python3 rl_policy.py --policy policy.pt            # 运行内置演示
  或作为模块：
    from rl_policy import FleetAllocator
    alloc = FleetAllocator("policy.pt", scenario="high")
    assign = alloc.allocate(cars, pending)   # cars/pending 见 allocate 注释
"""
import argparse
import json

import numpy as np
import torch

from env.fleet_env import make_env, _Task, SITE_NODE, PICKUP_SITES, \
    DROPOFF_SITES
from algorithms.mappo import MAPPO


class FleetAllocator:
    """策略推理器：外部车/任务状态 → 每 AGV 动作（0=等待 / k=接 pending[k-1]）。

    一次构造后可持续调用 allocate（每次内部按需建临时 env 编码状态）。
    scenario 只用于取“与训练一致”的拓扑/维度（3 车 6 槽），默认 high。"""

    def __init__(self, policy_pt, scenario="high"):
        """加载部署策略。校验 policy.pt 元数据维度与所选 scenario 的
        env 维度一致 —— 不一致说明训练场景配错，直接断言失败（fail-fast）。"""
        env0 = make_env(scenario)
        ck = torch.load(policy_pt, map_location="cpu")
        assert ck["obs_dim"] == env0.obs_dim and \
            ck["act_dim"] == env0.act_dim, \
            "policy 维度与 scenario 不匹配，请确认训练场景配置一致"
        self.ppo = MAPPO(env0.obs_dim, env0.act_dim, env0.num_agvs)
        self.ppo.actor.load_state_dict(ck["actor"])
        self.ppo.actor.eval()          # 推理模式（无 dropout/bn 语义）
        self.scenario = scenario
        self.env0 = env0               # 维度参考（预留复用）

    def allocate(self, cars, pending):
        """单步决策接口：编码外部状态 → 门控掩码 → argmax 动作。

        输入约定（与 ROS FleetManager 上报字段一一对应）：
          cars[N]: 每车 dict——
            node    所在图节点名（W0/E0/Em3/W3/Wm3/E3/DN/DS）
            phase   idle/to_pick/pick/to_drop/drop/charge（忙车无法接单）
            charge  电量 %（默认 100）
            task    当前任务 {src,dst} 或 None（有任务=忙，需带阶段）
          pending: 待分配任务 dict 列表（顺序即队首顺序）——
            src/dst 取/卸货站名（pickup_w/e、dropoff_n/s）
            priority 1 最急 / 2 次急；deadline 队列等待上限(s)
            wait    已等待秒数（enter_time=-wait 换算，供 obs 归一化）
          返回: actions[N] int（0=等待；k≥1 → 接 pending[k-1] 号任务）
        """
        env = make_env(self.scenario)
        # —— 用外部状态覆写 env 内部车/任务（其余字段保持训练默认）——
        # ★ 地图 v2 新拓扑（与 fleet_env.start_nodes 保持一致）
        st_nodes = ["PW", "PE", "ES", "WN", "WS", "EN"]
        env.cars = []
        for i, cs in enumerate(cars):
            task = cs.get("task")
            t_obj = None
            if task is not None:
                t_obj = _Task(i, {"src": task["src"], "dst": task["dst"],
                                  "sx": 0.0, "sy": 0.0, "dx": 0.0, "dy": 0.0},
                              task.get("priority", 1), 99999, 0.0)
                t_obj.status = "assigned"
            env.cars.append({
                "id": i, "node": cs.get("node", st_nodes[i % 6]),
                "phase": cs.get("phase", "idle"),
                "task": t_obj,
                "route": [], "seg": 0, "prog": 0.0, "wait": 0.0,
                "wait_kind": None,
                "charge": float(cs.get("charge", 100.0)),
                "drain": 0.02, "charge_node": None,
            })
        env.pending = []
        for k, ts in enumerate(pending):
            src = ts["src"] if ts["src"] in PICKUP_SITES else "pickup_w"
            dst = ts["dst"] if ts["dst"] in DROPOFF_SITES else "dropoff_nw"
            t = _Task(k, {"src": src, "dst": dst,
                          "sx": 0.0, "sy": 0.0, "dx": 0.0, "dy": 0.0},
                      int(ts.get("priority", 2)),
                      int(ts.get("deadline", 200)),
                      float(ts.get("wait", 0.0)) * -1.0)  # enter_time=-wait
            t.enter_time = float(ts.get("enter_time", 0.0))
            env.pending.append(t)
        env.time = float(max((p.get("wait", 0.0) for p in pending),
                             default=0.0))
        obs = env._obs()              # 复用训练观测编码
        mask = env.get_masks()        # 忙车/越界槽自动屏蔽
        return self.ppo.act_greedy(obs, mask)


def demo(policy_pt="policy.pt"):
    """内置演示：3 车（2 空闲 + 1 在途）× 3 待派任务 → 打印策略建议与 JSON。
    构造的样例刻意覆盖：不同节点/电量/忙闲、prio1 紧急件已等 30s ——
    便于肉眼核对策略是否学会“就近 + 急单优先”。"""
    alloc = FleetAllocator(policy_pt, scenario="high")
    cars = [
        {"node": "W0", "phase": "idle", "charge": 96.0, "task": None},
        {"node": "E0", "phase": "idle", "charge": 88.0, "task": None},
        {"node": "Em3", "phase": "to_drop", "charge": 72.0,
         "task": {"src": "pickup_e", "dst": "dropoff_s"}},
    ]
    pending = [
        {"src": "pickup_w", "dst": "dropoff_s", "priority": 1, "wait": 30.0,
         "deadline": 200, "enter_time": 30.0},
        {"src": "pickup_e", "dst": "dropoff_n", "priority": 2, "wait": 12.0,
         "deadline": 150, "enter_time": 12.0},
        {"src": "pickup_w", "dst": "dropoff_n", "priority": 2, "wait": 3.0,
         "deadline": 180, "enter_time": 3.0},
    ]
    act = alloc.allocate(cars, pending)
    print("== 策略分配建议 ==")
    for i, a in enumerate(act):
        if a == 0:
            print(f"  AGV_{i}: 等待")
        else:
            t = pending[a - 1]
            print(f"  AGV_{i}: 接任务#{a} "
                  f"({t['src']} -> {t['dst']}, prio {t['priority']}, "
                  f"已等 {t['wait']:.0f}s)")
    # 输出 JSON 供外部系统使用
    out = {"assign": [{"agv": i, "task": int(a)} for i, a in enumerate(act)]}
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="policy.pt")
    args = ap.parse_args()
    demo(args.policy)
