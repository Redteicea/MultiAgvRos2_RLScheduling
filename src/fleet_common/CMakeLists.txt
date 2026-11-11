cmake_minimum_required(VERSION 3.8)
project(fleet_common)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(geometry_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/RobotState.msg"
  "msg/Task.msg"
  "msg/FleetStatus.msg"
  "msg/NavStatus.msg"
  "msg/FleetNavStatus.msg"
  "srv/AddTask.srv"
  # ★ 2026-10-19（v2 调度/导航分离）：多车导航 action——调度器只发目标点，
  #   导航（A*+TEB+跟踪+让行+脱困）整体由 fleet_nav_server 实现，
  #   支持超时/取消/状态反馈（详见 MultiAgv/README.md §2.4）。
  "action/NavigateToPose.action"
  DEPENDENCIES geometry_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
