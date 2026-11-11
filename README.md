# AGV Fleet RL — 智能仓储多机协同调度系统

> Gazebo 仓储环境多 AGV 仿真 + C++ FleetManager 调度 + 自研轻量导航 + MAPPO 强化学习
> 调度训练，覆盖「感知 → 决策 → 调度 → 导航 → 控制 → 仿真」全栈链路。
>
> 技术栈：ROS 2 Humble / Gazebo Classic / C++(LibTorch 推理) / Python(PyTorch, CPU 可训)。
>
> 本文档只保留**项目介绍、结构、运行步骤、进度**；各阶段**实现思路、重难点、踩坑记录**
> 见 [`docs/DEVELOPMENT_NOTES.md`](docs/DEVELOPMENT_NOTES.md)，调试排障手册见
> [`DEBUG_NOTES.md`](DEBUG_NOTES.md)，TF/坐标系转换关系见
> [`TF_FRAMES.md`](TF_FRAMES.md)。

## 1. 项目介绍

系统共五层，逐层解耦：

```text
┌─────────────────────────────────────────────────────────────────┐
│ 任务 / 环境层：WMS 订单 · 随机任务生成器 · Gazebo 仓储世界         │
└────────────────────┬────────────────────────────────────────────┘
                     │ 任务流
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│ RL 决策层 (FleetManager)：状态感知 → 策略推理 → 任务分配决策       │
│     MAPPO / QMIX | C++ LibTorch 推理节点                        │
└────────────────────┬────────────────────────────────────────────┘
                     │ 调度指令
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│ 多机调度层：/agv_0/ /agv_1/ /agv_2/ · 冲突消解 · 死锁检测 · 路权  │
└────────────────────┬────────────────────────────────────────────┘
                     │ goal
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│ 导航层（自研，已替代 Nav2）：A* 全局 + Pure Pursuit + VFH 避障    │
└────────────────────┬────────────────────────────────────────────┘
                     │ Twist
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│ 控制/执行层：阿克曼四驱（前轮转向 + 四轮驱动）→ 物理引擎积分      │
└─────────────────────────────────────────────────────────────────┘
```


## 2. 项目结构

```text
.
├── src/
│   ├── fleet_common/          # 公共接口定义
│   │   ├── msg/               # RobotState / Task / FleetStatus.msg
│   │   └── srv/               # AddTask.srv
│   ├── fleet_sim/             # Gazebo 仿真 + 自研导航
│   │   ├── urdf/              # agv_core 四轮宏 + agv0/1/2 变体（xacro）
│   │   ├── worlds/            # warehouse.world（脚本生成）
│   │   ├── maps/              # map.pgm 240×160@0.1m（自研导航加载）
│   │   ├── config/            # (Legacy) Nav2 参数模板
│   │   ├── launch/            # spawn_robots / fleet_nav / fleet_rviz / fleet_system / nav2_agv
│   │   ├── scripts/           # custom_nav / send_goal / check_fleet / 地图与世界生成脚本
│   │   └── rviz/              # RViz 配置
│   └── fleet_manager/         # C++ 调度（fleet_manager_node + fleet_manager_node_teb + task_generator）
│       ├── launch/            # fleet_system（一键）/ fleet_manager / fleet_system_switch（nav:=custom|nav2|teb）
│       ├── config/            # fleet_manager.yaml
│       ├── rviz/              # fleet_manager.rviz
│       └── src/ include/      # 调度逻辑（A* / 路权 / 状态机 / 卡死与死锁处理 / RL 推理 rl_policy / TEB 导航：协调脱困·优先级让行·楔死点记忆）
├── rl_training/              # Phase 6 RL 训练（纯 Python + PyTorch）
│   ├── env/fleet_env.py      # 离散时间调度仿真器（拓扑对齐真实仓库）
│   ├── algorithms/mappo.py   # MAPPO（集中式 critic + 分散共享 actor）
│   ├── train.py              # 训练 / 课程学习 / 贪心基线 / 导出
│   ├── eval.py               # policy.pt vs 贪心 对比评估
│   ├── plot_curves.py        # 训练曲线绘图
│   ├── rl_policy.py          # 部署策略离线推理接口（无 ROS 依赖）
│   ├── export_jit.py         # policy.pt → policy_jit.pt（TorchScript，供 C++ 加载）
│   ├── ckpt/                 # 训练检查点（cur_v2 课程链等）
│   ├── runs/                 # 训练日志 + eval_report.csv
│   └── policy.pt             # 已导出部署策略（仅 actor + 元数据）
│   └── policy_jit.pt         # TorchScript 版（C++ LibTorch 加载用，export_jit.py 生成）
├── docs/                     # 开发笔记 / 安装验证脚本 / 历史执行命令
├── DEBUG_NOTES.md            # 调试排障手册（问题→根因→修复）
├── TF_FRAMES.md               # TF/坐标系转换关系手册（帧清单·三模式 TF 树·话题 frame_id 对照）
├── PHASE5_NAV2_PLAN.md       # (历史) Phase 5 计划
├── PHASE6_RL_PLAN.md         # (历史) Phase 6 计划
├── README.md                 # 本文档（介绍 / 结构 / 运行 / 进度）
├── build/ install/ log/      # colcon 产物与运行日志（symlink install）
```

## 3. 仿真环境介绍

### 3.1 AGV 模型（`fleet_sim/urdf/agv_core.urdf.xacro`）

模拟仓储搬运 AGV（载货式 AMR，无机械臂）；`agv0/1/2` 配色为蓝/橙/绿：

- **阿克曼四驱**：`ackermann_drive` 插件（前轮转向 + 四轮驱动），订阅 `/agv_X/cmd_vel`
  （Twist → 内部转 speed/steering_angle）；前轮绕 Z 转向（steering joint）、四轮绕 Y 驱动，
  轴距 0.32m / 轮距 0.34m；`max_steer=0.6rad`（最小转弯半径 0.47m），**不能原地旋转**，
  窄道 180° 掉头由上层“三点掉头”实现；**车底无万向球**
- 4 个轮子 `<dynamics damping="0.01"/>`（2026-09-13 修复：原 `continuous` 关节无阻尼导致
  落地弹跳后无阻力匀速滑行、直线漂移——即“左摇右晃”，详见 `DEBUG_NOTES.md` §2.9）
- 车体 500×380mm（20kg），前后保险杠、白色/红色状态灯、侧向反光条
- 车顶承载平台 + 可选货物（`with_cargo` 参数，`true` 显示载货）
- 车顶后角 **360° 激光雷达**（10Hz/12m），输出 `/agv_X/scan`，同时服务定位与避障；
  偏置安装（不在车体中心）用于防止定位 yaw 反向（180° 反锁）
- 里程计取 WORLD 物理真值（`odometry_source=1`），轮子打滑/被顶时依然真实

### 3.2 仓储地图（`fleet_sim/worlds/warehouse.world`）

24m×16m 库房，由 `scripts/generate_warehouse_world.py` 生成（改布局重跑脚本即可）：

- 4 排 × 8 列双面立体货架（立柱+层板+顶梁+彩色货箱）
- 装货站 ×2（绿）/ 卸货站 ×2（蓝）：停车区 + 防撞柱 + 托盘货物
- 充电桩 ×4；地面标线（通道虚线车道 / 安全黄线 / 双向箭头）；控制室、消防器材、照明

## 4. 编译与运行

### 4.1 编译

```bash
cd <项目根目录>        # 含 src/ 与 install/
colcon build --symlink-install
source install/setup.bash
```

> symlink install 下改 `src/` 内 launch / xacro / yaml / python 脚本**无需重新 build**，
> 重启对应 launch 即生效；只有改 C++ / msg / srv 才需重编。

### 4.2 启动前清理（重要，每次 launch 前先跑）

启动链路很长（Gazebo → spawn 3 台 AGV + robot_state_publisher → 导航 → 调度 → RViz）

```bash
cd <项目根目录>
./scripts/clean_ros.sh          # 终止残留进程 + 校验 + 停 ros2 daemon
./scripts/clean_ros.sh --purge  # 额外清空 /dev/shm 里陈旧的 SHM 文件
```

> ⚠️ **不要把这套清理写成 `bash -c "..."` 内联命令**：`pkill -f` 的模式串会出现在外层 shell
> 的 cmdline 里，导致它**把外层 shell 自己杀掉**（实测返回码 143）。脚本必须以**文件形式**
> 执行——脚本内容不在任何进程 cmdline 中，所以安全。详见 `scripts/clean_ros.sh` 顶部注释。

### 4.3 一键启动（仿真 + 调度 + 任务 + RViz）

```bash
./scripts/clean_ros.sh                            # ① 先清理
ros2 launch fleet_manager fleet_system_action.launch.py  # ② 再启动
```

即：Gazebo 仓库 + 3 台 AGV（出生点 x=-2.0/0.5/3.0, y=0）→ FleetManager C++ 调度
主节点 + task_generator 任务生成器 → 5s 后 RViz（含 world→odom 静态 TF）。

```bash
./scripts/clean_ros.sh

# 推荐（三种模式通用）：headless:=true —— 只起 gzserver，不开 Gazebo 图形窗口
ros2 launch fleet_manager fleet_system_action.launch.py headless:=true
```

> ⚠️ **TEB 模式（`fleet_manager_node_teb`）编译必须带 Release**，否则 g2o/Eigen 在 -O0 下
> 慢 200~500 倍、控制环从 20Hz 坍缩到 0.15Hz（"转过头"的元凶，见 `DEBUG_NOTES.md` §2.18）：
>
> ```bash
> colcon build --packages-select fleet_common fleet_manager --cmake-args -DCMAKE_BUILD_TYPE=Release
> ```
>
> 该模式集成 A* 全局 + 自研 TEB 局部规划 + 平滑纯跟踪执行 + **协调脱困让行**
> （位移窗口卡死检测 → 低优先级让行 / 同优先级 id 大让行；无邻车时三阶段摆尾盲脱困；
> 楔死点障碍记忆，详见 `DEBUG_NOTES.md` §2.19~2.22）。2026-09-25 联调完成，随机任务流
> 实测吞吐 7 分钟 7 单（三车），为三种模式中表现最好。

`headless:=true` 的原理是让 gzclient **根本不启动**（在根描述里先把 Gazebo 的 `gui`
配置置为 `false`；`DeclareLaunchArgument` 对已存在的配置不覆盖，而 gzclient 的启动条件
正是 `IfCondition(LaunchConfiguration('gui'))`）。仿真本身完全不受影响。
需要目视物理世界时随时挂上、看完关掉即可：

### 4.3.1 调度/导航分离架构

**调度器只派单，导航由 action 服务器实现**，接口走标准 ROS2 action，支持状态查询、
超时强制、主动取消：

```bash
./scripts/clean_ros.sh                                          # ① 清理
colcon build --packages-select fleet_common fleet_manager \
  --cmake-args -DCMAKE_BUILD_TYPE=Release                       # ② action/消息有更新，必须重编
ros2 launch fleet_manager fleet_system_action.launch.py headless:=true   # ③ v2 一键启动
```

| 组件 | 节点 | 职责 |
|---|---|---|
| 调度器 | `fleet_scheduler` | 任务管理/电量/RL 分配/停靠点吸附；导航仅发 action goal |
| 导航服务器 | `fleet_nav_server` | A* + TEB + 纯跟踪 + 让行仲裁/盲脱困/楔死点/穿模看门狗 全量导航引擎 |

- **Action**：`/navigate_to_pose`（`fleet_common/action/NavigateToPose`），goal 带
  `robot_name/x/y/timeout_sec/priority`；终态四种 `status`：SUCCEEDED(0) / ABORTED(1) /
  CANCELED(2) / TIMEOUT(3)，超时由服务器强制执行、调度器收到 TIMEOUT 自动 requeue 重派；
- **状态查询**：`/fleet/nav_status`（`FleetNavStatus`，1Hz 快照）+ feedback 2Hz；
  `ros2 action send_goal` 可直接手动派单调试；
- **验证**（2026-10-19 随机任务流实测）：五条路径全绿——多任务 SUCCEEDED、车忙 REJECT、
  客户端 CANCELED（终态 2）、服务器 TIMEOUT（4s/20s 均验证，终态 3）、调度器超时 requeue.

> ⚠️ `task_generator` 的 `enable` 参数为**启动时一次性读取**，运行时
> `ros2 param set` 不生效；要停随机任务流直接 `pkill -x task_generator`。


```bash
ros2 launch gazebo_ros gzclient.launch.py      # 挂到正在跑的 gzserver 上

### 4.4 分步启动与验证

```bash
# 终端 1：仅 Gazebo + 3 台 AGV
ros2 launch fleet_sim spawn_robots.launch.py

# 终端 2：自研导航栈（每车一个 custom_nav 节点）
ros2 launch fleet_sim fleet_nav.launch.py

# 终端 3：验证（话题频率 / 车队状态）
ros2 topic hz /agv_0/odom
bash src/fleet_sim/scripts/check_fleet.sh
```

手动控制 / 发目标：

```bash
# 前进
ros2 topic pub --rate 10 /agv_0/cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.2}}"

# 旋转
ros2 topic pub --rate 10 /agv_0/cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.2}, angular: {z: 1.0}}"

# 发导航目标（x y [yaw°]）
ros2 run fleet_sim send_goal 5.0 0.0 --ros-args -r __ns:=/agv_0
```

RViz：`ros2 launch fleet_sim fleet_rviz.launch.py`（导航路径/代价栅格可视化）。

### 4.5 RL 任务分配

`fleet_manager_node` 支持用训练好的 MAPPO 策略替换默认的贪心最近分配：

```bash
# 先导出 TorchScript（一次即可，产物 rl_training/policy_jit.pt）
cd rl_training && python3 export_jit.py --policy policy.pt --out policy_jit.pt

# 启动 C++ 调度，use_rl=true 启用 RL 分配
ros2 run fleet_manager fleet_manager_node --ros-args \
  -p use_rl:=true -p policy_path:=$PWD/rl_training/policy_jit.pt
```

- 依赖：LibTorch 复用 Python `torch` 自带的 C++ 头文件/库（CMake 自动探测，RPATH 已内嵌，无需 `LD_LIBRARY_PATH`）；可用 `-DTORCH_HOME=<路径>` 或环境变量 `TORCH_HOME` 覆盖。
- 分配逻辑：`rl_policy.{hpp,cpp}` 用与训练**完全一致**的观测编码（`_obs_one`/`get_masks`）+ argmax 确定性推理，C++ 输出已与 Python `rl_policy.py` 逐位对齐校验。
- 开关：`use_rl:=false` 时回退贪心最近分配（原有行为）。

### 4.6 RL 训练 / 评估 / 部署复现

```bash
cd rl_training
python3 train.py --curriculum "lite@300,mid@400,high@600" --tag cur_v2 --export policy.pt  # 课程训练
python3 eval.py --policy policy.pt --eps 20                      # 评估 → runs/eval_report.csv
python3 plot_curves.py --tag cur_v2 --out runs/cur_v2.png        # 训练曲线
python3 rl_policy.py --policy policy.pt                          # 部署推理演示
```

### 4.7 实时监控界面（车队看板）

单文件 curses 终端看板（`src/fleet_manager/scripts/fleet_monitor.py`），订阅 `/fleet/status`
实时显示：

- **调度概览**：在线数 / 执行中 / 空闲 / 充电 / 故障 / 在途任务 / 待分配 / 平均电量
- **AGV 状态表**：车号 · 状态 · **当前任务** · 位置 (x,y) · 朝向 · **线速度/角速度** · 电量条
- **任务列表**：任务号 · 优先级 · 取货点 · 卸货点 · 状态 · **执行车**

**前置：仿真 + 调度已在运行（spawn_robots.launch.py + fleet_manager.launch.py）**

**或直接：python3 src/fleet_manager/scripts/fleet_monitor.py**


```bash
ros2 run fleet_manager fleet_monitor
```
**按键：q 退出 · r 立即刷新 · + / - 调刷新周期(0.2~5s)**

依赖仅 Python 标准库（curses）+ rclpy，无需第三方库；改样式只需动 `draw_*`，改数据源只需动 `FleetMonitorNode`（便于维护）。
为此 `RobotState.msg` 新增 `int32 task_id`（当前任务号，-1=无）、`float64 linear_vel / angular_vel`（实际速度，取自 `model_states` 的 `twist`）。