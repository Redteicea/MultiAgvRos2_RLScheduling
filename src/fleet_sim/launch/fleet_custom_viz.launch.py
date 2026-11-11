# -*- coding: utf-8 -*-
"""
fleet_custom_viz.launch.py —— 自研导航栈（custom_nav）实时可视化
==================================================================
单窗口 RViz（固定坐标系 map）同时显示：
  1) 规划路径      —— /agv_X/plan（A* 全局/绕行路径，custom_nav 周期发布）
  2) 静态障碍物    —— /agv_X/global_map（墙/货架的 blocked OccupancyGrid）
  3) 动态障碍物    —— /agv_X/dynamic_obstacles（其他 AGV 等效圆盘 MarkerArray，
                      橙色=正前挡道触发跟停，黄色=一般邻车）
  4) 3 台 AGV 模型 —— /agv_X/robot_description（需 map→odom_X TF 齐全）

前置条件：Gazebo 已启动并 spawn 好 AGV。
  a) 已跑 fleet_nav.launch.py（同时启动 3 台 custom_nav 与 map→odom_X TF）：
       ros2 launch fleet_sim fleet_custom_viz.launch.py
  b) 手动 ros2 run custom_nav（未跑 fleet_nav）→ 本 launch 默认补发
       map→odom_0..2 恒等 TF（with_map_tf:=false 可关，避免与 fleet_nav 重复）

参数：
  agv_id:=0    观察对象（决定 global_map/plan/scan/dynamic_obstacles 读哪台车）
  num_agvs:=3  补发 TF 的 AGV 数量
  with_map_tf:=true  是否补发 map→odom_X 恒等 TF
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchContext, LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch_ros.actions import Node


def _launch_rviz(context: LaunchContext):
    """替换 ${AGV_ID} 占位符 → 写临时 RViz 配置并启动。"""
    agv_id = context.launch_configurations["agv_id"]
    num = int(context.launch_configurations["num_agvs"])
    with_tf = context.launch_configurations["with_map_tf"].lower() in ("true", "1", "yes")
    pkg_share = get_package_share_directory("fleet_sim")
    with open(os.path.join(pkg_share, "rviz", "fleet_custom.rviz")) as f:
        content = f.read()
    content = content.replace("${AGV_ID}", agv_id)
    tmp_rviz = f"/tmp/fleet_custom_{agv_id}.rviz"
    with open(tmp_rviz, "w") as f:
        f.write(content)

    nodes = []
    # map 与 odom_X 恒等（WORLD 里程计，map/odom/world 数值重合），
    # 兜底补发：确保 RViz 固定坐标系 map 存在、3 台 AGV 模型都能显示
    if with_tf:
        for i in range(num):
            nodes.append(Node(
                package="tf2_ros",
                executable="static_transform_publisher",
                name=f"viz_map_to_odom_{i}",
                output="log",
                arguments=["0", "0", "0", "0", "0", "0", "1",
                           "map", f"odom_{i}"],
            ))
    nodes.append(Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", tmp_rviz],
    ))
    return nodes


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            name="agv_id",
            default_value="0",
            description="观察对象：global_map/plan/scan/dynamic_obstacles 读 agv_0/1/2 哪台",
        ),
        DeclareLaunchArgument(
            name="num_agvs",
            default_value="3",
            description="补发 map→odom_X 恒等 TF 的数量（需与 spawn 数量一致）",
        ),
        DeclareLaunchArgument(
            name="with_map_tf",
            default_value="true",
            description="是否补发 map→odom_X 恒等 TF",
        ),
        OpaqueFunction(function=_launch_rviz),
    ])
