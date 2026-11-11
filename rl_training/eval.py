"""
eval.py - Phase 6 (Task 6.4/6.6) 策略评估与规则基线对比
================================================================================
加载导出的部署策略 policy.pt（仅 actor + 维度元数据），在 lite/mid/high
三场景下以**确定性动作**（act_greedy / greedy_actions）跑同一批新种子，
生成 runs/eval_report.csv —— README §Phase 6 验收表的官方数据来源。

评估口径说明（与训练期区分）：
  · 用与训练不同的 seed_base（默认 5000）→ 看到的是“没见过”的任务流，
    反映泛化能力而非过拟合训练分布；
  · 两策略用**完全相同**的种子序列 → 同场景同任务流，可比性公平；
  · MAPPO 用 argmax（无采样噪声）；Greedy 用最近任务启发式（train.py）。

用法（在 rl_training/ 下）：
  python3 eval.py --policy policy.pt
  python3 eval.py --policy policy.pt --scenarios lite,mid,high --eps 20
"""
import argparse
import csv
import os

import numpy as np
import torch

from env.fleet_env import make_env
from algorithms.mappo import MAPPO
from train import greedy_actions, evaluate


def load_deploy_policy(path, env):
    """从部署 checkpoint（仅 actor）重建可推理策略实例。

    policy.pt 只含 actor state_dict + obs_dim/act_dim/num_agents 元数据，
    不含 critic/opt —— 按元数据构造 MAPPO 骨架后灌入 actor 权重即可
    做 argmax 推理（critic 留空无影响）。env 仅用于形状校验/预留。"""
    ck = torch.load(path, map_location="cpu")
    p = MAPPO(ck["obs_dim"], ck["act_dim"], ck["num_agents"])
    p.actor.load_state_dict(ck["actor"])
    return p


def main():
    # CLI 参数：
    #   --policy      部署策略文件（train.py --export 的产物）
    #   --scenarios   逗号分隔场景列表（默认三档全评）
    #   --eps         每场景评估集数（越多越稳；10~20 即可，CPU 秒级）
    #   --seed-base   评估种子起点（与训练种子错开；保证 MAPPO/Greedy 同流）
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="policy.pt")
    ap.add_argument("--scenarios", default="lite,mid,high")
    ap.add_argument("--eps", type=int, default=10)
    ap.add_argument("--seed-base", type=int, default=5000)
    args = ap.parse_args()

    env0 = make_env(args.scenarios.split(",")[0])    # 建首个场景 env 以拿维度
    policy = load_deploy_policy(args.policy, env0)
    print(f"loaded deploy policy: obs={policy.obs_dim} "
          f"act={policy.act_dim} agents={policy.num_agents}")

    rows = [["scenario", "policy", "completion", "failed",
             "avg_latency", "narrow", "ep_reward", "steps"]]
    for sc in args.scenarios.split(","):
        # 同一种子序列下分别跑 MAPPO（确定性）与 Greedy 基线
        learned = evaluate(policy, sc, num_eps=args.eps,
                           seed_base=args.seed_base, greedy=False)
        base = evaluate(policy, sc, num_eps=args.eps,
                        seed_base=args.seed_base, greedy=True)
        print(f"\n===== {sc} ({args.eps} eps) =====")
        print(f"  {'MAPPO':6s}: completion={learned['completion']:.3f} "
              f"fail={learned['failed']:.2f} lat={learned['avg_latency']:.1f} "
              f"narrow={learned['narrow']:.2f} epR={learned['ep_reward']:.1f}")
        print(f"  {'Greedy':6s}: completion={base['completion']:.3f} "
              f"fail={base['failed']:.2f} lat={base['avg_latency']:.1f} "
              f"narrow={base['narrow']:.2f} epR={base['ep_reward']:.1f}")
        for name, d in (("MAPPO", learned), ("Greedy", base)):
            rows.append([sc, name, round(d["completion"], 3),
                         round(d["failed"], 2),
                         round(d["avg_latency"], 1) if d["avg_latency"] else "",
                         round(d["narrow"], 2), round(d["ep_reward"], 1),
                         round(d["steps"], 1)])

    os.makedirs("runs", exist_ok=True)
    with open("runs/eval_report.csv", "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print("\nreport -> runs/eval_report.csv")


if __name__ == "__main__":
    main()
