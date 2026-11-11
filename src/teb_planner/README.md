# teb_planner —— ROS 2 最小集成包

基于 g2o 图优化的 TEB（Timed Elastic Band）风格局部轨迹规划器。
核心库是**纯 C++、零 ROS 依赖**，可被任何 ROS 2 节点直接调用；
另附一个 RViz 演示节点用于可视化验证。

从 `g2o_teb/g2o/examples/teb_multi_vehicle` 演示工程剥离而来，
算法与调参（240 s 压力场景、接触率 0.2~0.4 次/(车·分钟)）与原工程完全一致。

## 目录结构

```
teb_planner/
├── CMakeLists.txt            # ament_cmake 构建
├── package.xml
├── cmake/FindG2O.cmake       # 定位系统 g2o
├── include/teb_planner/
│   ├── g2o_compat.h          # g2o 跨版本兼容层（>= 2020.5 开箱即用）
│   ├── obstacle_sdf.h        # 矩形(OBB)符号距离 + 梯度 + 圆角轮廓采样（纯几何）
│   ├── teb_edges.h           # 8 种约束边 + 自带加法更新顶点（见下）
│   ├── teb_planner.h         # ★ 对外 API：TebPlanner / TebConfig / PlanResult
│   └── ros_bridge.h          # Path / MarkerArray 等 ROS 消息转换
└── src/
    ├── teb_planner.cpp       # 核心实现（无 ROS 依赖）
    └── demo_node.cpp         # RViz 演示节点（可 -DTEB_BUILD_DEMO=OFF 关闭）
```

## 快速开始

```bash
# 1. 装依赖（Ubuntu 20.04+；g2o 用 apt 的即可，无需源码编译）
#    libsuitesparse-dev 提供 cs.h —— g2o 的 csparse 头文件会 include 它，必装
sudo apt install ros-${ROS_DISTRO}-desktop libg2o-dev libsuitesparse-dev

# 2. 拷进工作空间并构建
cp -r teb_planner ~/ros2_ws/src/
cd ~/ros2_ws
colcon build --packages-select teb_planner
source install/setup.bash

# 3. 跑演示
ros2 run teb_planner teb_planner_demo
# 另开一个终端起 RViz，Fixed Frame 填 map，Add:
#   /teb_planner_demo/plan      (Path)      —— 规划出的局部轨迹
#   /teb_planner_demo/obstacles (MarkerArray) —— 障碍物 + 膨胀圈 + 速度箭头
# 用工具栏的 "2D Pose Estimate" / "2D Goal Pose" 拖动起点和目标点，
# 松手即重规划；动态障碍物在往复运动，可观察避让。
```

## 在你自己的节点里调用（最小代码）

```cpp
#include <teb_planner/teb_planner.h>

g2o::teb::TebConfig cfg;              // 全部参数有默认值，按需覆盖
cfg.robotRadius = 0.2;                // 车体半径 [m]
cfg.maxSpeed = 1.2;                   // 线速度上限 [m/s]
cfg.safetyMargin = 0.3;               // 避障安全余量 [m]
cfg.weightBounds = 0.;                // 例如：没有全局代价地图时关掉边界约束

// ---- 阿克曼底盘：三行切换运动学模型 ----
cfg.kinematics = g2o::teb::Kinematics::Ackermann;
cfg.wheelbase = 0.5;                  // 轴距 [m]
cfg.maxSteerAngleDeg = 30.0;          // 最大前轮转角 [deg] -> R_min = 0.87 m

g2o::teb::TebPlanner planner(cfg);

// 每个 replan 周期调用一次（典型 2~10 Hz，单次 2~5 ms）
std::vector<g2o::teb::Obstacle> obstacles;
obstacles.emplace_back(2.0, 0.5, 0.4);            // 静态圆形障碍
// 矩形障碍（墙/货架/车辆外框）：半长、半宽、朝向[rad] —— 详见「矩形障碍物」一节
obstacles.push_back(g2o::teb::Obstacle::rect(3.5, 0.0, 1.0, 0.2, 0.0));
g2o::teb::Obstacle dyn(3.0, 0.0, 0.3, /*dynamic=*/true);
dyn.velocity = g2o::Vector2(0.0, 0.3);            // 动态障碍：给速度，规划器按时间外推
obstacles.push_back(dyn);

const g2o::teb::PlanResult r =
    planner.plan(/*start=*/{0., 0., 0.}, /*goal=*/{5., 2., 0.}, obstacles,
                 /*seed=*/lastTrajectory);        // 上一轮轨迹热启动

if (r.ok) {
  // r.trajectory —— SE2 位姿序列，第一个点就是当前位姿
  // r.dt        —— 相邻位姿时间间隔，配合起来就是带时间戳的轨迹
  // r.minClearance —— 最小避障间距（<0 表示侵入，可用来做降速/急停判断）
  // r.cmdV      —— 逐段线速度（差速/阿克曼都有）
  // r.cmdSteer  —— 逐段前轮转角 [rad]（阿克曼模式才有），已夹到 ±最大转角
  lastTrajectory = r.trajectory;
}
```

CMake 侧（你的包）：

```cmake
find_package(teb_planner REQUIRED)
target_link_libraries(your_node teb_planner::teb_planner_core)
```

需要发 ROS 消息时用 `#include <teb_planner/ros_bridge.h>` 里的
`g2o_teb::toPath / toObstacleMarkers / toAckermannDrive / toPose / fromPose`。

## 矩形障碍物（墙、货架、车辆轮廓）

除了圆形，障碍物还支持**有向矩形（OBB）**——墙、货架、集装箱、其它车辆的
外接框这类"长条形/带朝向"的东西用矩形描述比用一堆小圆省事得多，也精确得多。

### 三种设置方式

**① 直接用 `Obstacle::rect()`（推荐）**

```cpp
#include <teb_planner/teb_planner.h>

std::vector<g2o::teb::Obstacle> obstacles;

// 圆形：Obstacle(x, y, r)                       —— 接口保持不变
obstacles.emplace_back(2.0, 0.5, 0.4);

// 矩形：中心 + 半长 + 半宽 + 朝向
//   halfLength = 全长/2，halfWidth = 全宽/2（不是全长/全宽！）
//   朝向 yaw 单位是 弧度，0 = 长边沿世界 x 轴
//   下面这个 = 3.0 m 长 × 0.4 m 宽、横在路中间的一堵墙
obstacles.push_back(g2o::teb::Obstacle::rect(
    /*x=*/4.0, /*y=*/0.0,
    /*halfLength=*/1.5,   // 沿矩形自身长轴（局部 x 轴）的半长
    /*halfWidth =*/0.2,   // 沿矩形自身短轴（局部 y 轴）的半宽
    /*yaw=*/0.0));        // [rad]

// 45° 斜放的矩形（0.7854 rad = 45°）
obstacles.push_back(g2o::teb::Obstacle::rect(-1.0, 2.2, 0.8, 0.25, 0.7854));
```

> `yaw` 只是"矩形绕自己中心转了多少"，长边方向 = `yaw`。
> `halfLength` 与 `halfWidth` 谁大谁小无所谓，不需要保证 `halfLength >= halfWidth`；
> 沿 y 轴立一堵 0.4 × 3.0 m 的墙就填 `rect(x, y, 0.2, 1.5)`，完全合法。

**② 动态矩形**：矩形也能给速度，规划器按轨迹时间点线性外推它的中心与朝向。

```cpp
g2o::teb::Obstacle truck = g2o::teb::Obstacle::rect(3.0, 0.0, 2.0, 0.9,
                                                    90.0 * M_PI / 180.0,
                                                    /*dynamic=*/true);
truck.velocity = g2o::Vector2(0.0, -0.5);   // 中心速度 [m/s]
truck.yawRate  = 0.05;                      // 朝向角速度 [rad/s]（不需要可留 0）
obstacles.push_back(truck);
```

外推上限由 `cfg.predictionHorizon`（默认 1.5 s）截断，避免长时间线性外推失真。
注意：外推用的是**轨迹时刻**（`t_i = i * dt`），不是当前时刻 `t = 0`。

**③ 只支持圆形的下游 / 不想改调用方代码**：用 `rectToCircles()` 把矩形拆成一列内接圆。

```cpp
// n 个圆沿长轴等距排布，半径 = sqrt(halfWidth² + (s/2)²)，s = 2·halfLength/n
auto circles = g2o::teb::rectToCircles(/*x=*/4.0, /*y=*/0.0,
                                       /*halfLength=*/1.5, /*halfWidth=*/0.2,
                                       /*yaw=*/0.0,
                                       /*n=*/6);
obstacles.insert(obstacles.end(), circles.begin(), circles.end());
```

长条形（长宽比大）建议 `n >= 4`，`n` 越大越贴合；代价是优化变量翻倍、
单次规划耗时随圆的数量线性上升。

### 单位与小陷阱

| 量 | 字段 | 单位 / 约定 |
| --- | --- | --- |
| 中心 | `center`（`Obstacle::rect` 的 `x, y`） | 米 [m] |
| 半长 | `halfLength` | 米 [m]，**全长的一半** |
| 半宽 | `halfWidth` | 米 [m]，**全宽的一半** |
| 朝向 | `yaw` | **弧度 [rad]**（演示节点的 `rect_obstacles_yaw` 参数是**角度**，别搞混） |
| 速度 / 角速度 | `velocity` / `yawRate` | m/s / rad/s |

⚠️ 最容易踩的两个坑：

1. **半长 ≠ 全长**。`Obstacle::rect(x, y, 1.5, 0.2)` 的尺寸是 **3.0 × 0.4 m**。
   ROS 参数 `rect_obstacles_length/width` 则填**全长/全宽**（内部会 `×0.5`），
   这是为了让参数更贴近你量出来的实际尺寸。
2. **朝向单位不一致**。C++ API 全用弧度，ROS 参数用角度。二者都遵循同一套
   内部约定（见下面「障碍物约束的物理含义」），但输入口不同。

### 障碍物约束的物理含义

本包的避障约束统一表述成一句话：

> **位姿点到障碍物表面的符号距离 ≥ `robotRadius + safetyMargin`**

- 圆形：`d = |p − c|`（原样保留，行为与旧版完全一致）
- 矩形：`d =` 点到 OBB 的距离（外部为正、内部为负，即 SDF）

把机器人按半径 `robotRadius` 的圆处理，于是矩形约束的真实含义是
**「矩形按圆膨胀后的圆角矩形之外」**。RViz 里画的那条膨胀轮廓线就是这条
边界本身（不是近似），`minClearance` 也按同一公式计算，因此**显示值与
`minClearance` 严格自洽**：

```
minClearance = rectDistance(位姿, 矩形) − robotRadius
```

> 关于"为什么是圆角矩形而不是完整 Minkowski 和"：机器人被建模成圆，
> 圆的 Minkowski 和本身就是圆角矩形，`+robotRadius` 这一步已经包含在
> 约束里了；`safetyMargin` 是额外的余量，不参与形状膨胀。

### 静态矩形建议把 `weightObstacle` 调大

矩形有"面"而不是"点"，轨迹贴着长边滑行时接触机会远多于圆形障碍。
本地实测（40° 斜放矩形，其余参数默认）：

| `weightObstacle` | 最小间距 | 绕行幅度 |
| --- | --- | --- |
| 45（默认） | −0.067 m（**轻微擦边**） | 0.564 m |
| **90** | **+0.036 m** | 0.550 m |
| 150 | +0.105 m | 0.551 m |
| 250 | +0.158 m | 0.569 m |

也就是说：**默认权重下静态矩形会允许几厘米的贴边接触**（这是权重之间的
代价折衷，不是 bug——把迭代次数从 40 提到 120、或把位姿加密 `dt` 0.25→0.12
都毫无改善，只有提权有效）。想让矩形硬性避让，设 `weightObstacle ≥ 90`；
绕行幅度几乎不变（0.550 vs 0.564 m），代价只是收敛更"用力"一点。
对**动态**矩形别调太高，否则避让动作会过于生猛。

### RViz 里的显示

`toObstacleMarkers()` 对矩形的渲染：

- `CUBE` 标记：`scale.x = 2·halfLength`、`scale.y = 2·halfWidth`，
  用四元数 `(0, 0, sin(yaw/2), cos(yaw/2))` 表示朝向；
- `LINE_STRIP` 轮廓：`roundedRectOutline(center, yaw, halfLength + clr, halfWidth + clr, clr)`，
  其中 `clr = robot_radius + safety_margin`，即**规划器实际使用的约束边界**；
- 圆形障碍物的渲染保持不变。

> `scale.z` 给的是一个很小的值（薄板），只是为了避免 RViz 里出现高瘦的立方体，
> 不影响任何计算。

### 性能

矩形边（`EdgeObstacleRect`）与圆形边（`EdgeObstacle`）结构相同，都是
1 维一元边、解析雅可比，单次误差/雅可比计算只多几次 `cos/sin`。
本地基准：单次规划 2~5 ms（与圆形障碍同量级），矩形数量在几十个以内
不会成为瓶颈。已验证解析雅可比与中心差分数值微分**逐项一致（误差 0.0000）**。

## 接入全局地图

规划器接受"圆形 / 矩形障碍物列表 + 矩形可行区域"，全局地图要经一步转换。
`ros_bridge.h` 提供了现成函数（**目前从栅格提取出的仍是圆形障碍**，
需要矩形的话请自行对占据区做最小外接矩形拟合后调 `Obstacle::rect()`）：

```cpp
#include <teb_planner/ros_bridge.h>

// 订阅 map（map_server / nav2 都会发布），回调里存下来：
nav_msgs::msg::OccupancyGrid::ConstSharedPtr map;  // ...

// 规划前：取机器人周围 6 m 内的占据栅格 -> 障碍列表（自动按距离截断）
auto obstacles = g2o_teb::obstaclesFromOccupancyGrid(*map, robotPose,
                                                     /*maxRange=*/6.0,
                                                     /*stride=*/2);

// 顺手用地图外沿设置可行区域（也可以换成代价地图的 lethal 区）
double minX, maxX, minY, maxY;
g2o_teb::occupancyGridBounds(*map, minX, maxX, minY, maxY);
cfg.minX = minX; cfg.maxX = maxX; cfg.minY = minY; cfg.maxY = maxY;
```

要点：

- **局部规划不需要全图**：只提取机器人周围几米内的障碍，`stride` 控制采样
  密度、`maxObstacles`（默认 250）上限保证单次规划耗时可控。
- **推荐架构**（与 ROS 导航栈一致）：全局规划器（NavFn / Smac，或你自己的
  A*）出全局路径 → 局部规划器（本包）在每个周期取"全局路径上视野最远的
  点"作为 `goal` 传给 `plan()`，实现滚动式局部避障。直接拿最终目标点也
  能用，但距离超过十几米时轨迹离散密度下降，建议逐段推进。
- **接 nav2 代价地图**时，把 `occupiedThreshold` 调低（如 90 = lethal），
  或自行把 costmap 聚类成障碍圆后走 `Obstacle` 列表（`soft` 字段可标记
  动态物体）。

## 阿克曼模式说明

- `kinematics = Ackermann` 后：差速的"角速度上限"约束自动关闭，改为
  **最小转弯半径约束**（连续三个位姿的外接圆半径 ≥ L/tan(δ_max)），
  轨迹保证可被阿克曼底盘执行。
- `PlanResult::cmdSteer` 给出逐段前轮转角 `δ = atan(L·ω/v)`，
  已夹到 ±最大转角；`toAckermannDrive()` 一行转成 `AckermannDriveStamped`。
- 默认参数（L=0.5 m、δ_max=30° → R_min≈0.87 m）仅为演示值，
  **务必换成你底盘的实际参数**。

### 单位（容易搞反，务必确认）

| 量 | 字段 | 单位 |
| --- | --- | --- |
| 轴距 | `wheelbase` | **米 [m]** |
| 最大前轮转角（输入） | `maxSteerAngleDeg` | **角度 [deg]**（名字带 `Deg`，内部 `×π/180` 转弧度） |
| 前轮转角（输出） | `PlanResult::cmdSteer` | **弧度 [rad]** |
| 线速度 | `PlanResult::cmdV` | m/s |

> 输入用**角度**、输出用**弧度**——不要把 `cmdSteer` 回填给 `maxSteerAngleDeg`。

构造 `TebPlanner` 时会做单位体检：若 `maxSteerAngleDeg` 不在 `(0, 90)` 内，
或小于 5°（典型症状是把 0.5236 rad 当角度填了进去），会向 `std::cerr` 打印
一条 `WARNING`，并给出该值对应的 `R_min`。这个警告只是提醒，不会中断程序——
所以**调试时留意终端输出**，别让"车开得笔直"这种症状蒙混过关。

如果你的规格书只给**最小转弯半径 R_min**（更常见），反推转角：

```
δ_max = atan(L / R_min)          # 结果单位是 rad，再 ×180/π 填进去
```

换算速查（后轴中心半径定义，R = L/tan δ）：

| R_min | L = 0.5 m | L = 0.65 m | L = 1.0 m |
| --- | --- | --- | --- |
| 1.0 m | 26.6° | 33.0° | 45.0° |
| 1.5 m | 18.4° | 23.4° | 33.7° |
| 2.0 m | 14.0° | 18.0° | 26.6° |
| 3.0 m | 9.5° | 12.2° | 18.4° |

⚠️ 两点提醒：
1. 车厂标的"最小转弯半径"常指**车身最外侧划出的半径**，而本模型用的是
   **后轴中心**半径。若手册给的是外廓半径，先减去 `√((B/2)² + L_f²)`
   （B=车宽，L_f=后轴到车头的距离）再代入。
2. 这里填的是**单轨（自行车）模型的等效转角**。真实阿克曼梯形里内轮转角
   大于外轮，若要严格对应，用**外轮转角**偏保守（R 略大），用内轮偏激进。
- 已验证（本地 11+3 项测试）：绕障时最小转弯半径 1.08 m > 0.87 m 限值；
  90° 急转弯场景半径 0.862 m，精确压在 0.866 m 极限上。
- 跟踪控制建议用纯跟踪（pure pursuit）或 MPC——轨迹本身已满足曲率约束。

## 接口速查

| 类型 | 字段/方法 | 说明 |
| --- | --- | --- |
| `TebConfig` | `poses/minPoses/dt` | 轨迹离散化（个数自适应目标距离） |
| | `maxSpeed/maxOmega` | 速度/角速度上限（maxOmega 仅差速模式） |
| | `kinematics` | `Differential`（默认）或 `Ackermann` |
| | `wheelbase/maxSteerAngleDeg` | 阿克曼轴距与最大前轮转角 -> 最小转弯半径 |
| | `robotRadius/safetyMargin` | 车体半径与安全余量 |
| | `weight*` | 各约束权重（保持默认，除非做消融实验） |
| | `minX/maxX/minY/maxY` | 可行区域矩形（`weightBounds=0` 关闭） |
| | `predictionHorizon` | 动态障碍速度外推的上限时长 |
| `Obstacle` | `center/radius` | 圆形障碍（从代价地图聚类出的障碍圆给这里） |
| | `Obstacle::rect(x,y,halfL,halfW,yaw)` | **矩形障碍**：半长/半宽（全长全宽的一半），`yaw` 单位弧度 |
| | `shape` | `Shape::Circle`（默认）/ `Shape::Rectangle`，由 `rect()` 自动置位 |
| | `halfLength/halfWidth/yaw/yawRate` | 矩形尺寸、朝向 [rad]、朝向角速度 [rad/s] |
| | `velocity/dynamic` | 动态障碍：速度向量，规划器按轨迹时刻外推 |
| | `predictCenter(t)/predictYaw(t)` | 手动取外推后的中心 / 朝向（调试用） |
| | `soft` | 软障碍（其它机器人）：参与优化但不计可行性 |
| `rectToCircles` | `(x,y,halfL,halfW,yaw,n)` | 矩形 → n 个内接圆（喂给只支持圆形的下游） |
| `rectDistance` | `(p,center,yaw,halfL,halfW)` | 点到 OBB 的符号距离 [m]，外部正/内部负 |
| `PlanResult` | `trajectory` | 位姿序列，`[0]` 为当前位姿 |
| | `dt` | 相邻位姿时间间隔 → `t_i = i * dt` |
| | `cmdV / cmdSteer` | 逐段线速度 / 前轮转角（steer 仅阿克曼） |
| | `minClearance` | 最小间距，<0 侵入（软障碍不计入） |
| | `timeMs / chi2 / iterations` | 耗时、残差、迭代数 |

## 与原工程的关系 / 为什么自带顶点类

- 上游 g2o 的 `VertexSE2` 用**乘法（流形）**更新，而本规划器全部解析雅可比
  按**加法增量**推导。原演示工程改了 `g2o/types/slam2d/vertex_se2.h` 源码，
  本包改为在 `teb_edges.h` 自带一个加法更新的顶点，**不修改 g2o 源码**，
  因此 apt 安装的原版 g2o（≥ 2020.5）可直接使用。
- `g2o_compat.h` 兜底了 g2o < 2020.5 缺少 `eigen_types.h` 的情况。
- 链接依赖：`g2o_core / g2o_stuff / g2o_types_slam2d / g2o_solver_csparse /
  g2o_csparse_extension`（`FindG2O.cmake` 已处理）。

## 演示节点参数（ros2 param）

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `frame_id` | `map` | 输出消息的坐标系 |
| `rate_hz` | `2.0` | 重规划频率 |
| `kinematics` | `differential` | `differential` 或 `ackermann` |
| `wheelbase` | `0.5` | 轴距 [m]（阿克曼） |
| `max_steer_angle_deg` | `30.0` | 最大前轮转角 [deg]（阿克曼） |
| `start` / `goal` | `[0,0,0]` / `[5,2,0]` | 起终点 `[x, y, yaw]`，也可用 RViz 工具拖 |
| `obstacles_x/y/r` | 两组静态障碍 | 同长度数组（无地图时使用） |
| `rect_obstacles_x/y` | `[4.0, -1.0]` / `[0.0, 2.2]` | 矩形中心 [m]，五个 `rect_obstacles_*` 数组必须等长 |
| `rect_obstacles_yaw` | `[0.0, 45.0]` | 矩形朝向 [**deg**]，0 = 长边沿世界 x 轴 |
| `rect_obstacles_length` | `[0.4, 1.6]` | 矩形**全长** [m]（内部 ×0.5 转半长） |
| `rect_obstacles_width` | `[3.0, 0.5]` | 矩形**全宽** [m]（内部 ×0.5 转半宽） |
| `moving_obstacle` | `true` | 是否加一个往复运动的动态障碍 |
| `map_range` / `map_stride` | `6.0 / 2` | 从 `/map` 提取障碍的范围与采样步长 |
| `robot_radius` / `max_speed` / `max_omega` | `0.32 / 1.0 / 1.4` | 车辆参数 |

演示节点订阅 `map`（`nav_msgs/OccupancyGrid`，transient local）：一旦收到
地图，障碍物来源和可行区域边界**自动切换为地图内容**，参数里的演示障碍
不再生效。用 `map_server` 起一张地图即可复现真实场景：

```bash
# 阿克曼模式 + 地图
ros2 run teb_planner teb_planner_demo --ros-args \
  -p kinematics:=ackermann -p wheelbase:=0.65 -p max_steer_angle_deg:=28 \
  -p map_range:=8.0
ros2 run nav2_map_server map_server --ros-args -p yaml_filename:=my_map.yaml
```

```bash
ros2 run teb_planner teb_planner_demo --ros-args -p goal:="[7.0,0.0,1.57]"
```

## 已知边界

- 反应式局部规划：窄通道迎头会车仍可能贴边/轻微接触（与原工程一致，
  演示工程里用"迎头减速让行"缓解，属上层策略，不在此库职责内）。
- 障碍输入支持**圆形与有向矩形**；仍不支持任意多边形/栅格直接入图——
  复杂形状请自行用几个矩形拼，或走 `rectToCircles()` 近似成圆。
- 轨迹不含倒车（正向运动假设，差速/阿克曼一致）；需要窄巷倒车调头的场景
  要另行扩展双向 TEB。
