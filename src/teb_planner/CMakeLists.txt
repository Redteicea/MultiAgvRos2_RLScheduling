cmake_minimum_required(VERSION 3.10)
project(teb_planner)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(ackermann_msgs REQUIRED)
find_package(Eigen3 REQUIRED)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
find_package(G2O REQUIRED)  # apt install libg2o-dev

# ---------------------------------------------------------------------------
# glog：g2o 内部大量使用 CHECK/DLOG 等宏，共享库允许带未定义符号编出来，
# 但下游可执行文件链接时必须能解析到这些符号，因此显式链接 glog。
# ---------------------------------------------------------------------------
find_package(glog REQUIRED)  # apt install libgoogle-glog-dev

# ---------------------------------------------------------------------------
# SuiteSparse / CSparse：g2o 的 csparse 求解器头文件（csparse_extension.h 等）
# 会 #include <cs.h>，而 cs.h 属于 SuiteSparse —— libg2o-dev 本身不带它，
# Ubuntu 需额外安装 libsuitesparse-dev（头文件位于 /usr/include/suitesparse/）。
# ---------------------------------------------------------------------------
find_path(CSPARSE_INCLUDE_DIR NAMES cs.h
  PATHS /usr/include/suitesparse /usr/include/csparse
        /usr/local/include/suitesparse /usr/local/include
  PATH_SUFFIXES suitesparse csparse)
if(NOT CSPARSE_INCLUDE_DIR)
  message(FATAL_ERROR
    "cs.h (SuiteSparse) not found. Install it with:\n"
    "  sudo apt install libsuitesparse-dev")
endif()
message(STATUS "CSparse include: ${CSPARSE_INCLUDE_DIR}")

# ---------------------------------------------------------------------------
# 核心库：纯 C++，不含任何 ROS 依赖 —— 这就是给外部直接调用的接口。
# 下游节点：find_package(teb_planner) + target_link_libraries(your_node
#           teb_planner::teb_planner_core) 即可。
# ---------------------------------------------------------------------------
add_library(teb_planner_core SHARED src/teb_planner.cpp)
add_library(teb_planner::teb_planner_core ALIAS teb_planner_core)
target_include_directories(teb_planner_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${G2O_INCLUDE_DIRS}>
  $<BUILD_INTERFACE:${CSPARSE_INCLUDE_DIR}>
  $<BUILD_INTERFACE:${EIGEN3_INCLUDE_DIR}>
  $<INSTALL_INTERFACE:include>
  # 公开头文件（teb_planner.h / g2o_compat.h）直接 include 了 g2o 与 Eigen
  # 的头文件，这两个路径必须随 imported target 一起导出给下游。
  $<INSTALL_INTERFACE:${G2O_INCLUDE_DIRS}>
  $<INSTALL_INTERFACE:${EIGEN3_INCLUDE_DIR}>)
target_link_libraries(teb_planner_core PUBLIC ${G2O_LIBRARIES} glog::glog)

# 公开头文件 ros_bridge.h 直接 include 了这些 ROS 消息包的头文件。
# 用 INTERFACE 关键字：只向下游传递 include/link 需求，不会把 rclcpp 等
# 链进核心库本身——核心库保持"纯 C++、零 ROS 依赖"的定位不变。
ament_target_dependencies(teb_planner_core INTERFACE
  rclcpp geometry_msgs nav_msgs visualization_msgs ackermann_msgs)

# ---------------------------------------------------------------------------
# RViz 演示节点（不想要可 -DTEB_BUILD_DEMO=OFF）
# ---------------------------------------------------------------------------
option(TEB_BUILD_DEMO "Build the RViz demo node" ON)
if(TEB_BUILD_DEMO)
  add_executable(teb_planner_demo src/demo_node.cpp)
  target_link_libraries(teb_planner_demo teb_planner::teb_planner_core)
  ament_target_dependencies(teb_planner_demo
    rclcpp geometry_msgs nav_msgs visualization_msgs ackermann_msgs)
  install(TARGETS teb_planner_demo RUNTIME DESTINATION lib/${PROJECT_NAME})
endif()

install(TARGETS teb_planner_core
  EXPORT export_teb_planner
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)
install(DIRECTORY include/ DESTINATION include)

# 公开头文件（teb_planner.h / g2o_compat.h）直接 include 了 g2o 的头文件，
# 所以除自身 include/ 外还要把 g2o 的头文件路径一并导出给下游。
ament_export_include_directories(include ${G2O_INCLUDE_DIRS})
ament_export_targets(export_teb_planner HAS_LIBRARY_TARGET)

# 公开头文件（ros_bridge.h / teb_planner.h）直接 include 了这些包的头文件，
# 下游 find_package(teb_planner) 时必须能同时拿到它们，否则会报找不到头文件。
ament_export_dependencies(Eigen3
  rclcpp geometry_msgs nav_msgs visualization_msgs ackermann_msgs)

# 导出目标的 INTERFACE_LINK_LIBRARIES 中含 glog::glog，
# 下游需先 find_package(glog) 才能解析该 imported target。
ament_export_dependencies(glog)

ament_package()

