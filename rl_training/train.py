"""
train.py - Phase 6 (Task 6.4) 多 AGV 调度 MAPPO 训练入口
================================================================================
职责：单场景训练 / 课程学习（lite→mid→high 热启迁移）/ 续训 / 评估 /
      greedy 基线对比 / policy.pt 导出。

训练管线（一次迭代）：
  1) rollout_episode：env 与策略交互采样一集（只在实际决策点采样，加速）
  2) gae_on_buffer：批量价值估计 + GAE(λ) 优势/回报（团队标量，广播到每车）
  3) 攒够 UPDATE_N=6 集 → policy.update 做 PPO 更新（epochs 轮）
  4) 每 10 集写日志 / 每 eval_freq 集跑一次快速评估
  每阶段结束保存 ckpt/{tag}_{scenario}.pt；课程模式相邻阶段热启迁移。

用法（在 rl_training/ 目录下执行）：
  # 单场景训练
  python3 train.py --scenario lite --episodes 300 --tag run1
  # 课程学习（lite→mid→high 热启迁移，架构统一）
  python3 train.py --curriculum "lite@300,mid@400,high@600" --tag cur1
  # 续训 / 评估导出
  python3 train.py --scenario high --episodes 200 --warm-start <ckpt.pt> --tag cur2
  python3 train.py --scenario high --episodes 0 --warm-start <ckpt.pt> --export policy.pt --eval-only
"""
import argparse
import csv
import os
import random
import time

import numpy as np

from env.fleet_env import FleetEnv, make_env, SITE_NODE, shortest_route
from algorithms.mappo import MAPPO

# 日志 CSV 列（每行 = 一集）：ep=集号、episode_reward=本集总奖励、
# steps=耗时(s)、completion=完成率、failed=失败数、avg_latency=平均时延、
# narrow=窄道等待次数、policy_loss/value_loss/entropy=最近一次 PPO 更新的统计、
# time_s=阶段累计墙钟(s)
LOG_COLS = ["ep", "episode_reward", "steps", "completion", "failed",
            "avg_latency", "narrow", "policy_loss", "value_loss", "entropy",
            "time_s"]


def greedy_actions(env, prio_penalty=25.0):
    """
    两大核心原则：省空驶（控成本）+ 急单优先（保履约）
    贪心基线（规则调度，评估对照组）：每台 idle 车取“到源距离 + 源→目的
    距离”最小的队首任务；prio=2（次急）加 prio_penalty=25 惩罚使其被让位给
    更急的 prio=1。返回 [N] 动作（0=等待 / k+1=选队首第 k 个）。
    
    cost = d1（空车到取货点的距离）+ d2（取货点到卸货点的距离）+ 急单修正

    启发式依据：就近派单省空驶，急单优先防超期 —— 是无学习能力的强基线，
    MAPPO 须在完成率/时延上不弱于它才算“学到了有用调度”（见 eval_report）。"""
    a = np.zeros(env.num_agvs, dtype=np.int64)
    for i, c in enumerate(env.cars):
        if c["phase"] != "idle":
            continue
        best, bd = None, 1e18
        for k, t in enumerate(env.pending[:env.M_max]):
            d1, _ = shortest_route(c["node"], SITE_NODE[t.site["src"]])
            d2, _ = shortest_route(SITE_NODE[t.site["src"]],
                                   SITE_NODE[t.site["dst"]])
            cost = d1 + d2 + (0.0 if t.priority == 1 else prio_penalty)
            if cost < bd:
                bd, best = cost, k
        if best is not None:
            a[i] = best + 1     # 接第best + 1个任务，best是索引,+1是序号. 0为等待
    return a


def evaluate(policy, scenario, num_eps=8, seed_base=2000, greedy=False):
    """确定性策略评估：policy 或 greedy 在新种子集上跑 num_eps 集，
    返回平均指标 dict。policy 用 argmax（act_greedy，无探索噪声），
    保证指标可复现、可与基线公平对比。"""
    from algorithms.mappo import MAPPO  # noqa
    rec = {"completion": [], "failed": [], "latency": [], "narrow": [], "steps": [], "ep_reward": []}
    for e in range(num_eps):
        env = make_env(scenario, seed=seed_base + e)   # 每集独立种子
        obs = env.reset()
        done, ep_r = False, 0.0
        while not done:
            mask = env.get_masks()
            if greedy:
                act = greedy_actions(env)
            else:
                act = policy.act_greedy(obs, mask)
            obs, r, done, info = env.step(act)
            ep_r += r
        st = env.episode_stats()
        rec["completion"].append(st["completion"])
        rec["failed"].append(st["failed"])
        if st["avg_latency"] is not None:
            rec["latency"].append(st["avg_latency"])   # 无完成任务的集跳过
        rec["narrow"].append(st["narrow_events"])
        rec["steps"].append(int(info["time"]))
        rec["ep_reward"].append(ep_r)
    return {
        "completion": float(np.mean(rec["completion"])),
        "failed": float(np.mean(rec["failed"])),
        "avg_latency": float(np.mean(rec["latency"])) if rec["latency"] else None,
        "narrow": float(np.mean(rec["narrow"])),
        "steps": float(np.mean(rec["steps"])),
        "ep_reward": float(np.mean(rec["ep_reward"])),
    }


def compute_gae(rewards, values, dones, gamma, lam):
    """
    Generalized Advantage Estimation : 广义优势估计
    标准 GAE(λ) 优势估计（团队标量序列，见 mappo 模块说明）。

    参数: rewards[T] 每步共享奖励 · values[T+1] 价值（尾随一个 bootstrap）·
          dones[T] 终止标志（终止步不向后传播）· gamma/lam 折扣/GAE 系数
    返回: (adv[T], ret[T]) —— adv 为 GAE 优势，ret=adv+V 为折扣回报
          （critic 的回归目标）。
    """
    T = len(rewards)
    adv = np.zeros(T)
    gae = 0.0
    next_v = 0.0
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * (0.0 if dones[t] else values[t + 1]) - values[t]
        gae = delta + gamma * lam * (0.0 if dones[t] else gae)
        adv[t] = gae
        next_v = values[t]
    ret = adv + np.asarray(values[:T])
    return adv, ret


def rollout_episode(env, policy):
    """单集 rollout 采样（策略随机采样，含探索），返回 episode buffer dict。

    性能关键优化：只有“存在 idle 车且队内有可选项”（mask 的 1.. 列有 1）的
    步才真正调策略采样；纯等待步直接填 0 动作/0 logp —— Categorical.sample
    有 ~10ms 固定开销，跳过无决策步可提速 10x+，且对梯度无影响
    （等待步 logp=0 不影响 importance ratio 的有意义样本）。
    返回: obs/act/logp/rew/done/mask 各 [T, ...] 的 list + ep_reward 标量。
    """
    obs = env.reset()
    done = False
    obs_buf, act_buf, lp_buf, rew_buf, don_buf, msk_buf = \
        [], [], [], [], [], []
    ep_r = 0.0
    while not done:
        mask = env.get_masks()
        need = bool(mask[:, 1:].any())     # 有 idle 车且队内有可选项
        if need:
            act, lp = policy.select_actions(obs, mask)
        else:
            act = np.zeros(env.num_agvs, dtype=np.int64)    # 全车等待
            lp = np.zeros(env.num_agvs, dtype=np.float32)   # logp=0
        obs2, r, done, _ = env.step(act)
        obs_buf.append(obs)
        msk_buf.append(mask)
        act_buf.append(act)
        lp_buf.append(lp)
        rew_buf.append(r)
        don_buf.append(done)
        obs = obs2
        ep_r += r
    return {"obs": obs_buf, "act": act_buf, "logp": lp_buf,
            "rew": rew_buf, "done": don_buf, "mask": msk_buf,
            "ep_reward": ep_r}


def gae_on_buffer(buf, policy, gamma, lam):
    """把 rollout buffer 加工成带 advantage/return 的批量训练数组。

    步骤：批量预测每帧团队价值 → compute_gae 出标量 adv/ret →
    沿智能体维 tile 广播（每帧 N 台车共享同一优势/回报，团队协作口径）。
    返回元组 (obs, act, logp, adv, ret, mask)，各为 [T,N,...] 的 np 数组，
    可直接喂给 MAPPO.update（其内部再扁平化为 T*N 样本）。"""
    obs_seq = np.asarray(buf["obs"])          # [T,N,obs]
    vals = policy.predict_values(obs_seq)      # [T]（critic 批量前向）
    rews = np.asarray(buf["rew"], dtype=np.float64)
    dones = np.asarray(buf["done"], dtype=bool)
    adv, ret = compute_gae(rews, vals, dones, gamma, lam)
    T, N = obs_seq.shape[0], obs_seq.shape[1]
    # 团队标量 → 复制到每智能体样本
    adv_a = np.tile(adv[:, None], (1, N))
    ret_a = np.tile(ret[:, None], (1, N))
    return (obs_seq, np.asarray(buf["act"]), np.asarray(buf["logp"]),
            adv_a, ret_a, np.asarray(buf["mask"]))


def run_stage(scenario, episodes, policy, tag, seed=42, export=None,
              resume=None, eval_freq=50, compare=True):
    """跑一个训练阶段：连续 rollout episodes 并做间歇性 PPO 更新。

    参数:
      scenario  场景名 lite/mid/high（构造 env）
      episodes  本阶段训练集数
      policy    MAPPO 实例（可热启，跨阶段沿用）
      tag       目录 runs/<tag> 与检查点 ckpt/<tag>_<scenario>.pt 前缀
      seed      阶段随机种子（env 内部 rng 由 make_env 播种）
      export    若给出路径，阶段结束后导出 policy.pt
      eval_freq 每隔 N 集打印一次快速评估（4 集）；0=关
      compare   阶段结束是否跑 8 集 MAPPO vs Greedy 对比并写 compare_*.txt
    过程: 每集 rollout → 算 adv/ret 入缓存 pend；攒满 UPDATE_N=6 集合并
    拼接后 update 一次（减小每次 update 的固定开销）；日志每集一行 CSV。"""
    log_dir = os.path.join("runs", tag)
    os.makedirs(log_dir, exist_ok=True)
    log_f = open(os.path.join(log_dir, f"log_{scenario}.csv"), "a", newline="")
    writer = csv.writer(log_f)
    if os.path.getsize(os.path.join(log_dir, f"log_{scenario}.csv")) == 0:
        writer.writerow(LOG_COLS)

    env = make_env(scenario)
    UPDATE_N = 6          # 每 N 集合并做一次 PPO 更新（减小 update 开销）
    t0 = time.time()
    pend = []             # 待合并更新的批量缓存
    st = None             # 最近一次 update 统计（未更新前为 None）
    for ep in range(1, episodes + 1):
        buf = rollout_episode(env, policy)
        d = gae_on_buffer(buf, policy, policy.gamma, policy.lam)
        pend.append(d)
        if len(pend) >= UPDATE_N or ep == episodes:
            # 把缓存批沿时间维拼接 → 一次 PPO 更新
            obs_a = np.concatenate([x[0] for x in pend])
            act_a = np.concatenate([x[1] for x in pend])
            lp_a = np.concatenate([x[2] for x in pend])
            adv_a = np.concatenate([x[3] for x in pend])
            ret_a = np.concatenate([x[4] for x in pend])
            msk_a = np.concatenate([x[5] for x in pend])
            st = policy.update(obs_a, act_a, lp_a, adv_a, ret_a, msk_a)
            pend = []
        env_stats = env.episode_stats()
        row = [ep, round(buf["ep_reward"], 2), int(env_stats["time"]),
               round(env_stats["completion"], 3), env_stats["failed"],
               round(env_stats["avg_latency"], 2) if env_stats["avg_latency"] else "",
               env_stats["narrow_events"],
               round(st["policy_loss"], 4) if st else "",
               round(st["value_loss"], 4) if st else "",
               round(st["entropy"], 4) if st else "",
               round(time.time() - t0, 1)]
        writer.writerow(row)
        if ep % 10 == 0:
            log_f.flush()
            print(f"[{tag}|{scenario}] ep {ep:4d} R={row[1]:7.2f} "
                  f"steps={row[2]:4d} comp={row[3]:.2f} "
                  f"fail={row[4]} lat={row[5]}", flush=True)
        if eval_freq and ep % eval_freq == 0:
            ev = evaluate(policy, scenario, num_eps=4)
            print(f"   -- eval: completion={ev['completion']:.2f} "
                  f"fail={ev['failed']:.2f} lat={ev['avg_latency']} "
                  f"narrow={ev['narrow']:.2f}", flush=True)
    log_f.close()

    if compare:
        ev_learned = evaluate(policy, scenario, num_eps=8)
        ev_greedy = evaluate(policy, scenario, num_eps=8, greedy=True)
        print("\n===== 对比 (scenario=%s) =====" % scenario)
        for name, d in (("MAPPO", ev_learned), ("Greedy", ev_greedy)):
            print(f"  {name:6s}: completion={d['completion']:.3f} "
                  f"fail={d['failed']:.2f} lat={d['avg_latency']} "
                  f"narrow={d['narrow']:.2f} epR={d['ep_reward']:.1f}")
        with open(os.path.join(log_dir, f"compare_{scenario}.txt"), "w") as f:
            f.write(f"scenario={scenario}\n")
            for name, d in (("MAPPO", ev_learned), ("Greedy", ev_greedy)):
                f.write(f"{name} {d}\n")

    if export:
        policy.export_policy(export)
        print(f"policy exported -> {export}")
    return env_stats


def parse_curriculum(s):
    """解析课程串 "lite@300,mid@400,high@600" → [(scenario, episodes), ...]"""
    out = []
    for tok in s.split(","):
        sc, _, eps = tok.partition("@")
        out.append((sc.strip(), int(eps)))
    return out


def main():
    # CLI 参数说明（默认值=本仓库正式训练 cur_v2 用的值）：
    #   --scenario  lite/mid/high 单场景训练（与 --curriculum 二选一）
    #   --episodes  单场景训练集数；0 = 不训练只做 warm-start 评估
    #   --curriculum 课程串 "lite@300,mid@400,high@600"：逐阶段热启迁移，
    #               每阶段结束存 ckpt/{tag}_{scenario}.pt
    #   --seed      阶段随机种子（env 构造用，任务流可复现）
    #   --lr/--gamma/--lam/--hidden/--ent-coef 透传给 MAPPO（见 mappo.py 超参表）
    #   --tag       输出前缀：runs/<tag>/log_*.csv、ckpt/<tag>_*.pt
    #   --warm-start 从 ckpt.pt 恢复（维度须一致）
    #   --export    结束导出 policy.pt（仅 actor）
    #   --eval-only 只评估（--episodes 0 亦可），配合 --warm-start 用
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="lite", help="lite/mid/high")
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--curriculum", default=None, help='例如 "lite@300,mid@400,high@600"')
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--ent-coef", type=float, default=0.02)
    ap.add_argument("--tag", default=f"run_{int(time.time())}")
    ap.add_argument("--warm-start", default=None)
    ap.add_argument("--export", default=None, help="训练结束导出 policy.pt 路径")
    ap.add_argument("--eval-only", action="store_true")
    args = ap.parse_args()

    os.makedirs("ckpt", exist_ok=True)
    os.makedirs("runs", exist_ok=True)

    if args.curriculum:
        # ---- 课程学习模式：架构统一（维度一致）→ 前阶段权重直接续训后阶段 ----
        stages = parse_curriculum(args.curriculum)
        scenario0 = stages[0][0]
        env0 = make_env(scenario0)
        ppo = MAPPO(env0.obs_dim, env0.act_dim, env0.num_agvs,
                    hidden=args.hidden, lr=args.lr, gamma=args.gamma,
                    lam=args.lam, ent_coef=args.ent_coef)
        if args.warm_start:
            ppo.load(args.warm_start)
        for sc, eps in stages:
            print(f"\n######## 课程阶段: {sc} × {eps} ########")
            # 阶段内不做与 greedy 的完整对比（各阶段间快速评估已够），
            # 最终在最难场景统一对比一次
            run_stage(sc, eps, ppo, args.tag, seed=args.seed,
                      eval_freq=max(20, eps // 10), compare=False)
            ck = f"ckpt/{args.tag}_{sc}.pt"
            ppo.save(ck)
            print(f"checkpoint -> {ck}")
        # 最终在 hardest 场景对比基线
        hardest = stages[-1][0]
        ev1 = evaluate(ppo, hardest, num_eps=10)
        ev2 = evaluate(ppo, hardest, num_eps=10, greedy=True)
        print("\n===== 最终对比 (scenario=%s) =====" % hardest)
        for name, d in (("MAPPO", ev1), ("Greedy", ev2)):
            print(f"  {name:6s}: completion={d['completion']:.3f} "
                  f"fail={d['failed']:.2f} lat={d['avg_latency']} "
                  f"narrow={d['narrow']:.2f} epR={d['ep_reward']:.1f}")
        if args.export:
            ppo.export_policy(args.export)
            print("policy exported ->", args.export)
        return

    # ---- 单场景模式 ----
    env = make_env(args.scenario)
    ppo = MAPPO(env.obs_dim, env.act_dim, env.num_agvs,
                hidden=args.hidden, lr=args.lr, gamma=args.gamma,
                lam=args.lam, ent_coef=args.ent_coef)
    if args.warm_start:
        ppo.load(args.warm_start)
        print("warm-start from", args.warm_start)
    if args.eval_only or args.episodes == 0:   # 纯评估路径
        ev = evaluate(ppo, args.scenario, num_eps=10)
        print("EVAL", ev)
        return
    run_stage(args.scenario, args.episodes, ppo, args.tag, seed=args.seed,
              export=args.export, eval_freq=args.episodes // 10)
    ck = f"ckpt/{args.tag}_{args.scenario}.pt"
    ppo.save(ck)
    print("checkpoint ->", ck)


if __name__ == "__main__":
    main()
