# 定位系统安装的 g2o（apt install libg2o-dev，或源码 make install 到 /usr/local）。
# 参考 teb_local_planner 等成熟包的做法：按头文件 + 逐个库文件查找，
# 不依赖 g2o 官方并未提供的 CMake config。

find_path(G2O_INCLUDE_DIR g2o/core/sparse_optimizer.h
  PATHS /usr/include /usr/local/include /opt/ros/${ROS_DISTRO}/include)

find_library(G2O_CORE_LIB NAMES g2o_core)
find_library(G2O_STUFF_LIB NAMES g2o_stuff)
find_library(G2O_TYPES_SLAM2D_LIB NAMES g2o_types_slam2d)
find_library(G2O_SOLVER_CSPARSE_LIB NAMES g2o_solver_csparse)
find_library(G2O_CSPARSE_EXTENSION_LIB NAMES g2o_csparse_extension)

set(G2O_INCLUDE_DIRS ${G2O_INCLUDE_DIR})
set(G2O_LIBRARIES
  ${G2O_CORE_LIB}
  ${G2O_STUFF_LIB}
  ${G2O_TYPES_SLAM2D_LIB}
  ${G2O_SOLVER_CSPARSE_LIB}
  ${G2O_CSPARSE_EXTENSION_LIB})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(G2O
  REQUIRED_VARS
    G2O_INCLUDE_DIR
    G2O_CORE_LIB
    G2O_STUFF_LIB
    G2O_TYPES_SLAM2D_LIB
    G2O_SOLVER_CSPARSE_LIB
    G2O_CSPARSE_EXTENSION_LIB)

mark_as_advanced(
  G2O_INCLUDE_DIR
  G2O_CORE_LIB
  G2O_STUFF_LIB
  G2O_TYPES_SLAM2D_LIB
  G2O_SOLVER_CSPARSE_LIB
  G2O_CSPARSE_EXTENSION_LIB)
