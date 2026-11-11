import os 
import launch
import launch_ros
from ament_index_python.packages import get_package_share_directory
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
                                      text=True,
                                      timeout=3)
    except Exception:
        return False
    return name in out.split()

def generate_launch_description():
    fleet_sim_dir = get_package_share_directory('fleet_sim')
    fleet_mgr_dir = get_package_share_directory('fleet_manager')
    params_file = os.path.join(fleet_mgr_dir, 'config', 'fleet_manager.yaml')

    def gate(context: launch.LaunchContext):
        if all(_topic_exists(t) for t in NEEDED_TOPICS):
            with_rivz = context.launch_configurations.get('with_rivz', 'true')
            nodes = []

            nodes.append(
                launch_ros.actions.Node(
                    package='fleet_manager',
                    executable='fleet_schduler',
                    name='fleet_schduler',
                    output='screen',
                    parameters=[params_file]
                )
            )

            nodes.append(
                launch_ros.actions.Node(
                    package='fleet_manager',
                    executable='fleet_nav_server',
                    name='fleet_nav_server',
                    output='screen'
                )
            )

            nodes.append(
                launch_ros.actions.Node(
                    package='fleet_manager',
                    executable='task_generator',
                    name='task_generator',
                    output='screen',
                    parameters=[params_file]
                )
            )

            if with_rivz.strip().lower() in ('true', '1', 'yes'):
                nodes.append(
                    launch.actions.TimerAction(period=2.0, actions=[
                        launch.actions.IncludeLaunchDescription(
                            PythonLaunchDescriptionSource(os.path.join(fleet_sim_dir, 'launch', 'fleet_custom_viz.launch.py')),
                            launch_arguments=[('with_map_tf', 'true')]
                            )]
                        )
                    )
                
            return nodes
        return launch.actions.TimerAction(
            period=1.0,
            actions=[launch.actions.OpaqueFunction(function=gate)]
        )

    return launch.LaunchDescription([
        launch.actions.DeclareLaunchArgument(
            'headless', default_value='false', choices=['true', 'false']
        ),
        launch.actions.DeclareLaunchArgument(
            'with_rivz', default_value='true', choices=['true', 'false']
        ),
        launch.actions.SetLaunchConfiguration(
            'gui', 'false', condition=IfCondition(LaunchConfiguration('headless'))
        ),
        launch.actions.IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(fleet_sim_dir, 'launch', 'spawn_robots.launch.py'))
        ),
        launch.actions.OpaqueFunction(function=gate)
    ])