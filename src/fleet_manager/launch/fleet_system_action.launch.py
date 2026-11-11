# -*- coding: utf-8 -*-
"""
fleet_system_action.launch.py — 调度/导航分离架构一键启动
================================================================================
调度与导航剥离：调度器只发目标点（NavigateToPose action），导航由独立
action 服务器实现（支持超时取消、状态查询、进度反馈）。

    ros2 launch fleet_manager fleet_system_action.launch.py
    ros2 launch fleet_manager fleet_system_action.launch.py headless:=true

启动内容：
  ① Gazebo 仓库世界 + spawn 3 台 AGV + robot_state_publisher（fleet_sim）
  ② 闸门：等 3 台车 /agv_X/odom 就绪后放行（避免导航抢跑）
  ③ fleet_scheduler     —— 纯调度（任务/电量/RL 分配；action 客户端）
  ④ fleet_nav_server    —— 导航 action 服务器（/fleet/navigate_to_pose）
  ⑤ task_generator      —— 随机任务流（demo）
  ⑥ RViz                —— fleet_custom_viz（话题名与 teb 模式一致，
                           plan/teb_poses/teb_obstacles/global_map 直接可见）

排障入口：
  ros2 action list / ros2 action info /fleet/navigate_to_pose
  ros2 action send_goal /fleet/navigate_to_pose \
    fleet_common/action/NavigateToPose "{robot_name: agv_0, x: 10.4, y: 0.0, timeout_sec: 60.0, priority: 1}"
  ros2 topic echo /fleet/nav_status          # 任意时刻查询各车导航状态
"""

import os

import launch
import launch_ros
from ament_index_python.packages import get_package_share_directory
from launch.actions import (DeclareLaunchArgument, IncludeLaunchDescription,
                            OpaqueFunction, SetLaunchConfiguration, TimerAction)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration

NUM_AGVS = 3
NEEDED_TOPICS = [f'/agv_{i}/odom' for i in range(NUM_AGVS)]


def _topic_exists(name):
    import subprocess
    try:
        out = subprocess.check_output(['ros2', 'topic', 'list'],
                                      stderr=subprocess.DEVNULL,
                                      text=True, timeout=3)
    except Exception:
        return False
    # out.split() 按空白字符（空格、换行符）将其分割成一个 Python 列表
    return name in out.split()


def generate_launch_description():
    fleet_sim_dir = get_package_share_directory('fleet_sim')
    fleet_mgr_dir = get_package_share_directory('fleet_manager')
    params_file = os.path.join(fleet_mgr_dir, 'config', 'fleet_manager.yaml')

    def gate(context):
        """等 3 台车 odom 就绪 → 起 调度器+导航服务器+任务流；未就绪 1s 后重查。"""
        if all(_topic_exists(t) for t in NEEDED_TOPICS):
            with_rviz = context.launch_configurations.get('with_rviz', 'true')
            nodes = [
                # 纯调度（节点名 fleet_manager → 继承 fleet_manager.yaml 的
                # 任务/电量/RL 参数；use_rl/policy_path 生效）
                launch_ros.actions.Node(
                    package='fleet_manager', executable='fleet_scheduler',
                    name='fleet_manager', output='screen',
                    parameters=[params_file]),
                # 导航 action 服务器（节点名 fleet_nav_server；
                # 导航参数用 declare 默认值，与 yaml 数值一致）
                launch_ros.actions.Node(
                    package='fleet_manager', executable='fleet_nav_server',
                    name='fleet_nav_server', output='screen'),
                # 随机任务流（demo；参数在 yaml 的 task_generator 节下）
                launch_ros.actions.Node(
                    package='fleet_manager', executable='task_generator',
                    name='task_generator', output='screen',
                    parameters=[params_file]),
            ]
            if with_rviz.strip().lower() in ('true', '1', 'yes'):
                # 话题名/帧名与 teb 模式一致 → 现有 RViz 配置直接复用；
                # with_map_tf=true 补 map→odom_X 恒等 TF（TF_FRAMES.md §2.1）
                nodes.append(TimerAction(period=2.0, actions=[
                    IncludeLaunchDescription(
                        PythonLaunchDescriptionSource(os.path.join(
                            fleet_sim_dir, 'launch', 'fleet_custom_viz.launch.py')),
                        launch_arguments=[('with_map_tf', 'true')])]))
            return nodes
        return [TimerAction(period=1.0, actions=[OpaqueFunction(function=gate)])]

    return launch.LaunchDescription([
        DeclareLaunchArgument(
            'headless', default_value='false', choices=['true', 'false'],
            description='true=不启动 Gazebo GUI(gzclient)，省约 2 核'),
        DeclareLaunchArgument(
            'with_rviz', default_value='true',
            description='是否启动 RViz（fleet_custom_viz）'),
        # headless:=true → gazebo.launch.py 的 gzclient 启动条件 IfCondition(gui) 失效
        SetLaunchConfiguration('gui', 'false',
                               condition=IfCondition(LaunchConfiguration('headless'))),
        # ① Gazebo + spawn + robot_state_publisher
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(
                fleet_sim_dir, 'launch', 'spawn_robots.launch.py'))),
        # ② 就绪闸门 → ③④⑤⑥
        OpaqueFunction(function=gate),
    ])
