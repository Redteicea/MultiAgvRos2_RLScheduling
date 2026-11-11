#!/usr/bin/env bash
# =============================================================================
# clean_ros.sh — 启动 fleet_system 前的环境清理
#
# 【为什么需要】
#   ros2 launch 链路很长：Gazebo → spawn 3 台 AGV(+robot_state_publisher)
#   → 自定义导航 → fleet_manager/task_generator → RViz。任何一段崩溃时，
#   launch 父进程与子节点可能"半死不活"地残留下来（本仓库曾累积 14 个
#   ros2 launch + 3 个 robot_state_publisher，最老的跑了 13h56m）。
#
#   每个存活的 ROS 进程都持有一个 FastDDS 共享内存 participant，用 flock
#   独占锁锁住 /dev/shm/fastrtps_portXXXX。新 launch 申请端口撞上这些锁时
#   就会报：
#       [RTPS_TRANSPORT_SHM Error] Failed init_port fastrtps_port7415:
#       open_and_lock_file failed -> Function open_port_internal
#   （端口号一路递增 7415→7425→7429… 就是它在逐个换端口重试）
#
#   该报错本身通常非致命（DDS 会回退 UDP），但残留进程会带来端口耗尽、
#   /dev/shm 持续增长、旧 task_generator 与新实例 topic 串扰等问题。
#
# 【用法】
#   ./scripts/clean_ros.sh           # 清理 + 校验（每次启动前跑一次）
#   ./scripts/clean_ros.sh --purge   # 额外删除 /dev/shm 里陈旧的 SHM 文件
#
# 【注意】必须"以文件形式"执行，不要改成 bash -c 内联！
#   pkill -f 的模式串会出现在外层 shell 的 cmdline 里，导致 pkill 把外层
#   shell 自己杀掉（返回码 143）。脚本文件的内容不在任何进程 cmdline 中，
#   所以安全。
# =============================================================================
set -u

PURGE=0
[[ "${1:-}" == "--purge" ]] && PURGE=1

# 要终止的进程：从"上层（父）"到"下层（子）"排列。
# 仅 kill 'ros2 launch' 就能级联带走大部分子节点，其余为兜底。
PATTERNS=(
  'ros2 launch'            # 所有 launch 父进程（fleet_system / fleet_nav / spawn_robots …）
  'gzserver'               # Gazebo 服务端
  'gzclient'               # Gazebo 客户端
  'robot_state_publisher'  # spawn_robots 为每台 AGV 起的 RSP
  'rviz2'
  'fleet_manager_node'     # 调度主节点
  'task_generator'         # 任务生成器
  'custom_nav'             # 自研导航节点（launch 级联关闭带不走时会残留）
  'fleet_monitor'          # 监控节点（ros2 run 起的，不带在 launch 树里）
  # ---- Nav2 模式（nav:=nav2）会额外起这些；实测 launch 的级联关闭带不走它们 ----
  'lib/nav2_'                    # map_server / planner / controller / behavior / bt_navigator / lifecycle_manager
  'nav2_goal_bridge'             # goal_pose → navigate_to_pose 桥接
  'static_transform_publisher'   # nav2_fleet 补的 map→map_i / map_i→odom_i 恒等 TF
)

echo "[clean_ros] 1/3  终止残留 ROS/Gazebo 进程"
for p in "${PATTERNS[@]}"; do
  if pkill -f -- "$p" 2>/dev/null; then
    echo "    已终止: $p"
  fi
done

# 给 launch 的级联关闭留时间；仍顽固的补一刀 -9
sleep 2
for p in "${PATTERNS[@]}"; do
  pkill -9 -f -- "$p" 2>/dev/null
done

# ros2 CLI 守护进程缓存了旧节点图，停掉可避免后续 ros2 node/topic list 看到幽灵条目
if ros2 daemon stop >/dev/null 2>&1; then
  echo "    已停止 ros2 daemon"
fi

echo "[clean_ros] 2/3  校验残留"
# 注意 grep 用 [r]os2 这种"括号技巧"，避免 grep 匹配到自己这条命令
LEFT=$(ps -eo args | grep -cE '[r]os2 launch|[g]zserver|[g]zclient|[r]obot_state_publisher|[r]viz2|[l]ib/nav2_|[n]av2_goal_bridge|[s]tatic_transform_publisher')
if [[ "$LEFT" -eq 0 ]]; then
  echo "    ✓ 无残留进程"
else
  echo "    ✗ 仍有 $LEFT 个残留："
  ps -eo pid,etime,args | grep -E '[r]os2 launch|[g]zserver|[g]zclient|[r]obot_state_publisher|[r]viz2|[l]ib/nav2_|[n]av2_goal_bridge|[s]tatic_transform_publisher' | cut -c1-120
fi

echo "[clean_ros] 3/3  FastDDS 共享内存（/dev/shm）"
NSHM=$(ls /dev/shm 2>/dev/null | grep -c fastrtps)
if [[ "$PURGE" -eq 1 ]]; then
  rm -f /dev/shm/fastrtps_* /dev/shm/sem.fastrtps_* 2>/dev/null
  echo "    已清理陈旧 SHM 文件（原 $NSHM 个）"
else
  echo "    保留 $NSHM 个陈旧 SHM 文件"
  echo "    （无进程持锁时 FastDDS 会自行复用，不影响启动；要彻底清空加 --purge）"
fi

echo ""
echo "[clean_ros] 完成。现在可以执行："
echo "    ros2 launch fleet_manager fleet_system.launch.py"
