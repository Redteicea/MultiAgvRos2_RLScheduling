import launch
import launch_ros
from ament_index_python.packages import get_package_share_directory
from launch.launch_description_sources import PythonLaunchDescriptionSource
import os

# AGV 配置（3台）—— 均出生在主通道 y=0 上，与货架区无碰撞
AGVS = [
    {"id": 0, "name": "agv_0", "x": -2.0, "y": 0.0, "urdf": "agv0.urdf.xacro"},
    {"id": 1, "name": "agv_1", "x": 0.5, "y": 0.0, "urdf": "agv1.urdf.xacro"},
    {"id": 2, "name": "agv_2", "x": 3.0, "y": 0.0, "urdf": "agv2.urdf.xacro"},
]


def generate_launch_description():
    robot_name_in_model = "AGV"
    pkg_path = get_package_share_directory('fleet_sim')
    default_world_path = os.path.join(pkg_path, 'worlds', 'warehouse.world')

    actions = []

    # 1. Gazebo 仿真世界
    # 通过 IncludeLaunchDescription 包含另外一个 launch 文件
    launch_gazebo = launch.actions.IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(get_package_share_directory('gazebo_ros'), 'launch', 'gazebo.launch.py')),
      	# 传递参数
        launch_arguments=[('world', default_world_path),('verbose','true')]
    )
    actions.append(launch_gazebo)

    # 3. 逐个 spawn 3台AGV
    for agv in AGVS:
        urdf_path = os.path.join(pkg_path, 'urdf', agv['urdf'])

        robot_description = {
            'robot_description': launch.substitutions.Command(['xacro ', urdf_path])
        }

        # 3.1. robot_state_publisher（全局）
        actions.append(launch_ros.actions.Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            namespace=agv['name'],
            name='robot_state_publisher',
            parameters=[robot_description],
            remappings=[('tf', '/tf'), ('tf_static', '/tf_static')],
            output='screen'
        ))

        # 3.2. 请求 Gazebo 加载机器人
        spawn_entity_node = launch_ros.actions.Node(
            package='gazebo_ros',
            executable='spawn_entity.py',
            namespace=agv['name'],
            arguments=[
                '-entity', agv['name'],                    # 指定唯一名字
                '-topic', 'robot_description',             # 从 robot_state_publisher 发布的 topic 读 URDF
                '-x', str(agv['x']),                       # 指定坐标
                '-y', str(agv['y']),
                '-z', '0.08',   # 轮子半径 0.08，base_link 原点在轮心 → 贴地不嵌入不悬空
                '-robot_namespace', agv['name']            # 指定namespace
            ],
            output='screen'
        )
        
        actions.append(spawn_entity_node)

    return launch.LaunchDescription(actions)
