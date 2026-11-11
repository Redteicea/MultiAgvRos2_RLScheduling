// teb_planner 的 g2o 跨版本兼容层。
//
// 本包只需要 g2o 的：头文件（core / stuff / types_slam2d / solvers/csparse）
// 和动态库（g2o_core, g2o_stuff, g2o_types_slam2d, g2o_solver_csparse）。
// Ubuntu 20.04+ / Debian 12+ 的 libg2o-dev（g2o >= 2020.5）开箱即用；
// 更老的 g2o 没有 eigen_types.h 时，这里手动补齐本包用到的别名。
//
// 注意：本包不依赖 g2o::VertexSE2（上游是乘法更新，而 TEB 约束的解析
// 雅可比按加法增量推导），顶点类在 teb_edges.h 中自带，见那里的说明。

#ifndef TEB_PLANNER_G2O_COMPAT_H
#define TEB_PLANNER_G2O_COMPAT_H

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "g2o/stuff/misc.h"  // number_t / normalize_theta / cst / make_unique

#if __has_include("g2o/core/eigen_types.h")
#include "g2o/core/eigen_types.h"
#else
// g2o < 2020.5：没有统一的 eigen_types.h，补齐本包用到的别名
namespace g2o {
template <int N>
using VectorN = Eigen::Matrix<number_t, N, 1, Eigen::ColMajor>;
using Vector2 = VectorN<2>;
using Vector3 = VectorN<3>;
using Vector4 = VectorN<4>;
template <int N>
using MatrixN = Eigen::Matrix<number_t, N, N, Eigen::ColMajor>;
using Matrix2 = MatrixN<2>;
using Matrix3 = MatrixN<3>;
using Rotation2D = Eigen::Rotation2D<number_t>;
}  // namespace g2o
#endif

#endif  // TEB_PLANNER_G2O_COMPAT_H
