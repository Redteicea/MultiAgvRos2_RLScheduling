cmake_minimum_required(VERSION 3.8)
project(fleet_manager)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(gazebo_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(fleet_common REQUIRED)
find_package(teb_planner REQUIRED)

# ---- LibTorch（Phase 7 RL 推理）----
# 复用 Python torch 自带的 C++ 头文件/库，无需单独下载 libtorch。
# 覆盖方式：-DTORCH_HOME=<路径> 或环境变量 TORCH_HOME；否则自动探测 python3 torch。
if(DEFINED ENV{TORCH_HOME} AND NOT DEFINED TORCH_HOME)
  set(TORCH_HOME $ENV{TORCH_HOME})
endif()
if(NOT TORCH_HOME)
  execute_process(
    COMMAND python3 -c "import torch,os;print(os.path.dirname(torch.__file__))"
    OUTPUT_VARIABLE _torch_py_dir OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _torch_py_rc)
  if(_torch_py_rc EQUAL 0 AND _torch_py_dir)
    set(TORCH_HOME ${_torch_py_dir})
  endif()
endif()
if(TORCH_HOME)
  list(APPEND CMAKE_PREFIX_PATH ${TORCH_HOME})
endif()
find_package(Torch REQUIRED)

# task_generator / fleet_nav_server / fleet_scheduler

add_executable(task_generator
  src/task_generator.cpp
)
ament_target_dependencies(task_generator
  rclcpp std_msgs fleet_common
)
install(TARGETS task_generator
  DESTINATION lib/${PROJECT_NAME})

# fleet_nav_server   : 导航 action 服务器（A*+TEB+跟踪+让行+脱困 全量迁移自 node_teb）
add_executable(fleet_nav_server
  src/fleet_nav_server_node.cpp
)
target_include_directories(fleet_nav_server PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
ament_target_dependencies(fleet_nav_server
  rclcpp rclcpp_action std_msgs geometry_msgs visualization_msgs gazebo_msgs
  nav_msgs fleet_common teb_planner
)
install(TARGETS fleet_nav_server
  DESTINATION lib/${PROJECT_NAME})

# fleet_scheduler    : 纯调度节点（任务/电量/RL 分配 + NavigateToPose action 客户端）
add_executable(fleet_scheduler
  src/fleet_scheduler_node.cpp
  src/rl_policy.cpp
)
target_include_directories(fleet_scheduler PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
ament_target_dependencies(fleet_scheduler
  rclcpp rclcpp_action std_msgs geometry_msgs visualization_msgs gazebo_msgs
  std_srvs nav_msgs fleet_common
)
target_link_libraries(fleet_scheduler ${TORCH_LIBRARIES})
target_include_directories(fleet_scheduler PUBLIC ${TORCH_INCLUDE_DIRS})
if(TORCH_HOME)
  set(_torch_lib_dir "${TORCH_HOME}/lib")
  set_target_properties(fleet_scheduler PROPERTIES
    BUILD_RPATH "${_torch_lib_dir}"
    INSTALL_RPATH "${_torch_lib_dir}")
endif()
install(TARGETS fleet_scheduler
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME})

# 实时监控界面（curses TUI，单文件便于维护）
# 用法：ros2 run fleet_manager fleet_monitor
install(PROGRAMS scripts/fleet_monitor.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME fleet_monitor)

ament_package()
