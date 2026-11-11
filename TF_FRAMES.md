# MultiAgv TF 坐标转换关系手册

> 整理日期：2026-10-18。所有条目均按源码逐文件核对，`文件:行号` 可直接回溯。
> 适用版本：fleet_sim / fleet_manager / teb_planner 当前工作区（nav::=custom | nav2 | teb 三模式）。

---

## 0. 一页速览（先记住这 3 条）

1. **整个系统只有一个"数值真值系"——Gazebo 世界系。** 激光里程计插件
   `odometry_source=WORLD`（`agv_core.urdf.xacro:131`），所以每台车的
   `odom_X` 帧原点与世界系**数值重合**，`/agv_X/odom` 就是真值。
2. **"真值定位" = 恒等 TF。** 因为 odom 数值==世界系，把 `map`（或 `map_X`）
   锚到 `odom_X` 只需要恒等变换，零漂移（`nav2_fleet.launch.py:21-24` 注释）。
3. **自研栈（custom / teb）不走 TF 查询。** custom_nav 直接读 odom 数值
   （`custom_nav.py:622-626`），fleet_manager_node_teb 直接读
   `/fleet_sim/model_states` 真值。TF 树在这两种模式里**只为 RViz 渲染服务**；
   只有 Nav2 模式才真正消费 TF。

---

## 1. 帧清单（Frame Registry）

| 帧名 | X = | 定义处 | 数值性质 |
|---|---|---|---|
| *（世界系，无帧名）* | — | `warehouse.world` 所有模型 pose、`/fleet_sim/model_states` | 全系统唯一的物理真值系 |
| `map` | — | fleet_manager 全系消息的统一 `frame_id`（`fleet_manager_node_teb.cpp:1714` 等） | 数值=世界系（恒等链保证） |
| `map_X` | 0..2 | Nav2 每车隔离的 `global_frame`（`nav2_params_agv.yaml:33,64,389`） | 数值=世界系 |
| `odom_X` | 0..2 | Gazebo ackermann 插件 `<odometry_frame>`（`agv_core.urdf.xacro:470`） | 数值=世界系（WORLD 源） |
| `base_link_X` | 0..2 | URDF 车体根 link（`agv_core.urdf.xacro:62`），spawn z=0.08（轮心，`spawn_robots.launch.py:60`） | — |
| `lidar_link_X` | 0..2 | URDF 雷达 link（`agv_core.urdf.xacro:333`），相对 base 偏移 **(-0.14, 0.12, 0.325)**（joint origin，`:361-366`） | — |
| `beacon_X` / `top_platform_X` / `cargo_X` | 0..2 | URDF 固定子链（`:300,384,410`） | 装饰/可见性柱，不影响导航 |
| `front_left_steer_link_X` 等 | 0..2 | URDF 动态关节（转向/车轮，`:143,187,...`） | **不广播**：URDF 无 joint_state 插件、无 /joint_states → robot_state_publisher 只发固定关节；不影响导航链 |
| `world` | — | `fleet_manager_node_inter_nav.cpp`（`:1366,1415,1437`） | 数值与 `map` 相同，仅帧名不同；RViz 配置默认不启用其显示（`nav2_fleet.rviz:144` 注释：避免 TF 报错） |

---

## 2. TF 树（按模式）

### 2.1 `nav:=teb`（当前主用模式）

```
                    ┌─────────────── 每台车 X = 0,1,2 各一条链 ───────────────┐
                    │                                                            │
  map ──恒等(静态)──> odom_X ──Gazebo ackermann──> base_link_X ──robot_state_pub──> lidar_link_X
  (1)                 (2)         WORLD真值TF          (3)        /tf_static       (4)
```

| 边 | 发布者 | 频率 | 性质 |
|---|---|---|---|
| (1) `map→odom_X` | `static_transform_publisher` ×3，`fleet_custom_viz.launch.py:48-57`（`with_map_tf:=true`，由 `fleet_system_switch.launch.py:125-127` 指定） | 静态 | 恒等 0,0,0,0,0,0,1 |
| (2) `odom_X→base_link_X` | Gazebo `libgazebo_ros_ackermann_drive`（`publish_odom_tf=true`，`agv_core.urdf.xacro:468-471`） | 随控制周期 | WORLD 真值 |
| (3)(4) `base_link_X→lidar_link_X/beacon_X/...` | `robot_state_publisher` ×3（`spawn_robots.launch.py:39-48`，全局 /tf、/tf_static） | 启动即静态 | URDF 固定关节 |

**关键**：teb 模式**不起任何导航栈**（`fleet_system_switch.launch.py:100-102`），
`fleet_manager_node_teb` 位姿全部来自 `/fleet_sim/model_states` 真值（`onModelStates`），
**不读 TF**；它发布的一切话题 `frame_id="map"`，数值=世界系。TF 树在此模式
**纯粹给 RViz 渲染**（固定帧 `map`，`fleet_custom.rviz:717`）。

### 2.2 `nav:=custom`（自研 custom_nav）

链结构与 2.1 完全相同，唯一区别是边 (1) 的发布者换成：

| 边 | 发布者 | 出处 |
|---|---|---|
| (1) `map→odom_X` | `static_transform_publisher` ×3，随 `fleet_nav.launch.py:62-72` 启动 | 恒等；注释见 `:7-9`（"WORLD 里程计，恒等变换使 RViz 固定帧 map 下可显示全局路径/代价地图"） |

（`fleet_custom_viz.launch.py` 的补发项默认 `with_map_tf:=true`，与 fleet_nav 重复时
同值静态 TF 无冲突，也可显式关掉避免重复。）

**custom_nav 不做 TF lookup**：odom 即真值自己 `quat_to_yaw`（`custom_nav.py:622-626`），
scan 用极坐标 + 自身 yaw 直接进 VFH 直方图；所有发布 `frame_id='map'`（`:341,790,832...`）。

### 2.3 `nav:=nav2`（Nav2 多机栈）

```
  map ──恒等(静态)──> map_X ──A──> odom_X ──Gazebo──> base_link_X ──rsp──> lidar_link_X
  (nav2_fleet:90)      │
                       ├─ A = 恒等静态 TF（use_amcl:=false，默认，nav2_fleet:92-95）
                       └─ A = AMCL 动态 TF（use_amcl:=true，激光定位，nav2_agv.launch.py:109-119）
```

- **(1) `map→map_X` 恒等**：把 fleet_manager 发的 `goal.frame_id="map"` 与 Nav2 的
  `global_frame="map_X"` 锚接到一起（`nav2_fleet.launch.py:17-20,89-90`）。
  `nav2_goal_bridge.py:38-41` 因此**原样透传** frame_id，不改写。
- **(2) `map_X→odom_X`**：
  - `use_amcl:=false`（默认真值模式）：恒等静态 TF，"真值定位"（`nav2_fleet.launch.py:92-95`）；
  - `use_amcl:=true`：AMCL 发布动态 TF，帧参数 `map_X / odom_X / base_link_X`
    （`nav2_params_agv.yaml:28,33,42`）。
  - 另有备选"假 AMCL"：`sim_pose_publisher.py` 订阅 model_states、以 10Hz 把真值
    pose 作为 `map_X→odom_X` 动态发布（`:72-80`）——本仿真里与恒等 TF 数值等价，
    留作以后无缝替换真 AMCL 的路径（`nav2_agv.launch.py:6-7` 注释）。

**Nav2 内部帧使用**（`nav2_params_agv.yaml`）：

| 组件 | global_frame | robot/base 帧 | scan 源 |
|---|---|---|---|
| amcl | `map_X`(:33) | `base_link_X`(:28) | `scan`(:59) |
| bt_navigator | `map_X`(:64) | `base_link_X`(:65) | odom 话题 `:66` |
| local_costmap | `odom_X`(:325) | `base_link_X`(:326) | `scan`(:362-363) |
| global_costmap | `map_X`(:389) | `base_link_X`(:390) | `scan`(:409-410) |
| behavior_server | `odom_X`(:513) | `base_link_X`(:514) | local_costmap 话题 |

scan 的 `frame_id=lidar_link_X` 由 ray 插件写入（`agv_core.urdf.xacro:505`），
costmap 靠 TF 链 `lidar_link_X→base_link_X→odom_X→map_X` 完成坐标变换——
**这是全系统唯一真正消费 TF 的用户**。

---

## 3. 话题 ↔ frame_id 对照表

### 3.1 生产者

| 话题 | 发布者 | frame_id | 数值系 |
|---|---|---|---|
| `/fleet_sim/model_states` | Gazebo gazebo_ros_state 插件 | （无） | 世界真值 |
| `/agv_X/odom` | ackermann 插件 | `odom_X` → child `base_link_X` | 世界真值（WORLD 源） |
| `/agv_X/scan` | ray 插件（xacro:499-506） | `lidar_link_X` | 雷达系极坐标 |
| `/agv_X/goal_pose` | fleet_manager（custom/nav2 模式下发） | `map` | 世界数值 |
| `/agv_X/global_map` | fleet_manager_node_teb `publishGlobalMap()` | `map` | 世界数值 |
| `/agv_X/plan` | 同上 `pubGlobalPlan()`（`:1714`） | `map` | 世界数值 |
| `/agv_X/teb_poses` | `sendGoalToAGV()`（`:2119`） | `map` | 世界数值 |
| `/agv_X/teb_obstacles` | 同上（`:2178`） | `map` | 世界数值 |
| 任务/目标 MarkerArray | 同上（`:2208`） | `map` | 世界数值 |
| global_map / plan / dynamic_obstacles / 文本 | custom_nav.py（`:341,790,891...`） | `map` | 世界数值 |
| 规划/标记 | fleet_manager_node.cpp（`:1063,1118`） | `map` | 世界数值 |
| 规划/标记 | fleet_manager_node_inter_nav.cpp（`:1366,1415,1437`） | `world` | 世界数值（帧名不同） |

### 3.2 消费者（谁走 TF、谁绕过）

| 消费者 | 用 TF？ | 说明 |
|---|---|---|
| Nav2 costmap/amcl/bt | **是** | 全系统唯一重度 TF 用户（见 §2.3 表） |
| RViz（custom 配置，固定帧 `map`，`fleet_custom.rviz:717`） | 是 | 仅渲染：恒等链 map→odom_X→base_link_X→lidar_link_X |
| RViz（nav2 配置，固定帧 `map_${AGV_ID}`，`nav2_fleet.rviz:162`） | 是 | 同上 |
| custom_nav | **否** | odom 即真值；scan 极坐标+自身 yaw（`custom_nav.py:622-626`） |
| fleet_manager_node / node_teb | **否** | 位姿全部来自 model_states 真值 |
| nav2_goal_bridge | 否 | frame_id 原样透传（`nav2_goal_bridge.py:157-158` 仅兜底空值） |
| `_dbg/scan_map_check.py` | 否（手动变换） | 真值 pose + 雷达偏移 (-0.14,0.12) 手动把 scan 转世界系 |
| `_dbg/teb_view_check.py` / `truth.py` | 否 | 同理，直接用真值数值 |

---

## 4. 关键约定与踩坑（改代码前必读）

1. **`odometry_source=WORLD` 是整棵恒等树的地基**（`agv_core.urdf.xacro:131`、
   `custom_nav.py:72-73`）。若以后换成轮式积分（WHEEL odometry），odom≠世界，
   所有恒等 TF 必须换成真定位（AMCL / sim_pose_publisher），否则 RViz 与规划数值会分裂。
2. **`frame_id="map"` 的消息数值一律是世界系**——fleet_manager 三个变体（node /
   node_teb / inter_nav）发的所有话题都直接用世界数值，只贴帧名标签。
   切忌把 "map" 理解成某个被定位结果修正过的局部系。
3. **`map` vs `map_X` vs `world` 的命名分裂**是历史包袱：同一数值、三个名字，
   靠恒等 TF 锚接。新增发布代码时：fleet_manager 系沿用 `"map"`；
   新 RViz 显示 inter_nav 的 marker 需先补 `map→world` 恒等（或像
   `nav2_fleet.rviz:144` 那样默认不启用）。
4. **`map→odom_X` 恒等 TF 有三处可能发布**（fleet_nav / fleet_custom_viz /
   nav2_fleet 的 map_X→odom_X），重复发布同值静态 TF 无害，但排障时注意
   `ros2 run tf2_ros tf_monitor` 里可能出现多条同边，属预期。
5. **`use_amcl:=true` 时 nav2_fleet 不再补 `map_X→odom_X` 恒等**（`nav2_fleet.launch.py:93`），
   AMCL 接管；`map→map_X` 恒等仍在。
6. **lidar_joint 的偏移必须写在 joint origin**（`agv_core.urdf.xacro:361-365`）：
   fixed 零位姿子 link 会被 libsdformat 折叠、sensor pose 丢失——曾导致
   雷达坐标双倍偏移/归零的坑，影响 `lidar_link_X` 在 TF 链中的真实位置。
7. **base_link 出生 z=0.08**（轮心，`spawn_robots.launch.py:60`）——TF 的 z 分量
   与 2D 平面显示无冲突，但任何手动算"车心高度"时记得用 0.08 而不是 0。
8. **`sim_pose_publisher.py` vs 恒等静态 TF**：两者数值等价（都通向真值），
   区别在于前者是动态 TransformBroadcaster（10Hz、可平滑替换 AMCL 的接口形态），
   后者是静态节点。调试 TF 问题时别把两者混启（等价无害，但拓扑更乱）。

---

## 5. 验证命令（VM 上直接跑）

```bash
# 1) 出整棵 TF 树的 PDF/文本（当前激活模式的真实拓扑）
ros2 run tf2_tools view_frames

# 2) 关键边的数值抽查（恒等则输出 0,0,0 + 单位四元数）
ros2 run tf2_ros tf2_echo map odom_0          # teb/custom 锚接边
ros2 run tf2_ros tf2_echo map_0 odom_0        # nav2 锚接边
ros2 run tf2_ros tf2_echo base_link_0 lidar_link_0   # URDF 固定链（应为 -0.14, 0.12, 0.325）

# 3) 各话题 frame_id 抽查
ros2 topic echo /agv_0/odom --field header --once
ros2 topic echo /agv_0/scan --field header --once
ros2 topic echo /agv_0/teb_poses --field header --once

# 4) TF 链健康度（重复边/断链都会在这里显示）
ros2 run tf2_ros tf_monitor
```

预期（teb 模式，headless 起法）：
`view_frames` 应画出 `map → odom_{0,1,2} → base_link_{0,1,2} → {lidar_link,beacon,top_platform,cargo}_{0,1,2}`，
共 3×5=15 个叶子帧 + 7 个中间帧，无断链、无告警。
