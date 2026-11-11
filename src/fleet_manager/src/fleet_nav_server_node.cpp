// =====================================================================
// FleetNavServer — 多车导航 Action 服务器
// ---------------------------------------------------------------------
//   - Action   : /fleet/navigate_to_pose（多车单服务器，goal 带 robot_name）
//   - 终态     : SUCCEEDED / ABORTED / CANCELED / TIMEOUT（status 码见 action 定义）
//   - 超时     : goal.timeout_sec > 0 时由服务器强制（到点停车 + TIMEOUT 结果）
//   - 取消     : 标准 action cancel（本拍停车 + CANCELED 结果）
//   - 反馈     : 2Hz feedback（PLANNING/MOVING/YIELD/RECOVERY + 剩余距离/耗时）
//   - 查询     : /fleet/nav_status（FleetNavStatus，无需 goal handle 即可查询）
// =====================================================================
#include "fleet_manager/astar_planner.hpp"
#include "fleet_manager/robot_agent.hpp"
#include "fleet_manager/traffic_control.hpp"
#include "fleet_manager/warehouse_map.hpp"

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <gazebo_msgs/msg/model_states.hpp>
#include <gazebo_msgs/srv/set_entity_state.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <nav_msgs/msg/path.hpp>
#include <fleet_common/action/navigate_to_pose.hpp>
#include <fleet_common/msg/fleet_nav_status.hpp>
#include <teb_planner/ros_bridge.h>
#include <teb_planner/teb_planner.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
using namespace std::chrono_literals;

namespace fleet_manager
{

inline double angleDiff(double a, double b)
{
  double d = a - b;
  while (d > M_PI) { d -= 2.0 * M_PI; }
  while (d < -M_PI) { d += 2.0 * M_PI; }
  return d;
}

inline double quatToYaw(double qx, double qy, double qz, double qw)
{
  (void)qx; (void)qy;
  return std::atan2(2.0 * (qw * qz), 1.0 - 2.0 * qz * qz);
}

inline const char * stageName(int s)
{
  switch (s) {
    case 0: return "空闲";
    case 1: return "navTick（导航状态机）";
    case 2: return "路权刷新 traffic->update";
    case 10: return "planPath（全局 A*）";
    case 11: return "TEB 规划 planner_.plan";
    case 12: return "跟踪解算（方向/前视/阻尼）";
    case 13: return "发布 cmd_vel";
    case 14: return "障碍构建 getRobotNearbyObstacles";
    default: return "未知";
  }
}

class FleetNavServerNode : public rclcpp::Node
{
public:
  using NavigateToPose = fleet_common::action::NavigateToPose;
  using GoalHandleNav = rclcpp_action::ServerGoalHandle<NavigateToPose>;

  FleetNavServerNode() : Node("fleet_nav_server")
  {
    declareParams();
    loadParams();
    buildMap();
    initTebLocalPlanner();
    initRobots();
    initPublishers();
    initSubscribers();
    initActionServer();

    control_timer_ = create_wall_timer(
      std::chrono::duration<double>(1.0 / control_rate_), [this]() { controlTick(); });
    status_timer_ = create_wall_timer(
      std::chrono::duration<double>(status_period_),
      [this]() { publishFeedbackAndStatus(); });
    marker_timer_ = create_wall_timer(
      std::chrono::duration<double>(marker_period_),
      [this]() {
        for (const auto & robot : this->robots_) { this->pubGlobalPlan(robot); }
        if (++this->map_pub_cnt_ % 4 == 1) { this->publishGlobalMap(); }
      });

    // 独立线程看门狗（迁移自 node_teb：单线程执行器里 timer 自检救不了自己）
    watchdog_ = std::thread([this]() {
      while (watchdog_run_.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        if (!watchdog_run_.load()) { break; }
        const double t = secNow();
        if (tick_begin_.load() > 0.0 && (t - tick_begin_.load()) > 3.0) {
          RCLCPP_ERROR(get_logger(),
            "看门狗：controlTick 已卡住 %.1f s！当前相位 = %s（该相位已耗时 %.1f s，"
            "上一拍 TEB 规划 %.0f ms）",
            t - tick_begin_.load(), stageName(stage_.load()),
            t - stage_t0_.load(), plan_ms_.load());
        }
      }
    });

    RCLCPP_INFO(get_logger(), "FleetNavServer started: %zu robots, control %.1f Hz, action=/fleet/navigate_to_pose",
      robots_.size(), control_rate_);
  }

  ~FleetNavServerNode()
  {
    watchdog_run_.store(false);
    if (watchdog_.joinable()) { watchdog_.join(); }
  }

private:
  // ===================== NavJob（★ 必须先于所有使用它的方法声明） =====================
  struct NavJob
  {
    std::shared_ptr<GoalHandleNav> gh;
    double x = 0.0, y = 0.0;         // 目标点（登记用；实际跟踪走 robot->goal_*）
    double timeout = 0.0;            // 0 = 不限时
    int priority = 999;
    double t0 = 0.0;                 // 开始时刻（steady 秒）
    double yield_t0 = 0.0;           // 进入让行时刻
    uint8_t state = 0;               // Feedback 的 NAV_* 常量
    std::string message;             // 最近状态说明
    bool cancel_requested = false;
    int plan_fails = 0;              // A* 连续失败计数
    double last_plan_fail = 0.0;     // 上次 A* 失败时刻（重试间隔控制）

    bool active() const { return gh != nullptr; }
  };
  std::unordered_map<std::string, NavJob> job_by_robot_;

  // ===================== 参数 =====================
  void declareParams()
  {
    declare_parameter("robot_names", std::vector<std::string>{"agv_0", "agv_1", "agv_2"});
    declare_parameter("max_linear_speed", 0.5);
    declare_parameter("max_angular_speed", 1.0);
    declare_parameter("goal_tolerance", 0.18);
    declare_parameter("waypoint_tolerance", 0.10);
    declare_parameter("turn_stop_factor", 1.2);
    declare_parameter("max_steer", 0.6);
    declare_parameter("wheelbase", 0.32);
    declare_parameter("min_turn_speed", 0.12);
    declare_parameter("robot_radius", 0.30);
    declare_parameter("traffic_stop_distance", 0.55);
    declare_parameter("stuck_timeout", 5.0);
    declare_parameter("yield_backup", 0.60);
    declare_parameter("stuck_window", 4.0);
    declare_parameter("stuck_disp", 0.10);
    declare_parameter("yield_scope", 2.0);
    declare_parameter("yield_speed", 0.15);
    declare_parameter("yield_timeout", 12.0);
    declare_parameter("yield_clear_dist", 2.2);
    declare_parameter("charger_x", std::vector<double>{-11.0, 11.0, 11.0, -11.0});
    declare_parameter("charger_y", std::vector<double>{3.6, 3.6, -3.6, -3.6});
    declare_parameter("status_period", 0.5);
    declare_parameter("marker_period", 0.5);
    declare_parameter("control_rate", 30.0);
    declare_parameter("plan_retry_max", 3);          // ★ 新增：A* 失败重试次数（耗尽→ABORTED）
    declare_parameter("plan_retry_interval", 1.5);   // ★ 新增：重试间隔 [s]
  }

  void loadParams()
  {
    robot_names_ = get_parameter("robot_names").as_string_array();
    traffic_ = std::make_unique<TrafficControl>(get_parameter("robot_radius").as_double() / 0.2);
    v_max_ = get_parameter("max_linear_speed").as_double();
    w_max_ = get_parameter("max_angular_speed").as_double();
    goal_tol_ = get_parameter("goal_tolerance").as_double();
    wp_tol_ = get_parameter("waypoint_tolerance").as_double();
    turn_stop_factor_ = get_parameter("turn_stop_factor").as_double();
    max_steer_ = get_parameter("max_steer").as_double();
    wheelbase_ = get_parameter("wheelbase").as_double();
    min_turn_speed_ = get_parameter("min_turn_speed").as_double();
    min_turn_radius_ = wheelbase_ / std::tan(max_steer_);
    robot_radius_ = get_parameter("robot_radius").as_double();
    stop_dist_ = get_parameter("traffic_stop_distance").as_double();
    stuck_timeout_ = get_parameter("stuck_timeout").as_double();
    yield_backup_ = get_parameter("yield_backup").as_double();
    stuck_window_ = get_parameter("stuck_window").as_double();
    stuck_disp_ = get_parameter("stuck_disp").as_double();
    yield_scope_ = get_parameter("yield_scope").as_double();
    yield_speed_ = get_parameter("yield_speed").as_double();
    yield_timeout_ = get_parameter("yield_timeout").as_double();
    yield_clear_dist_ = get_parameter("yield_clear_dist").as_double();
    charger_x_ = get_parameter("charger_x").as_double_array();
    charger_y_ = get_parameter("charger_y").as_double_array();
    status_period_ = get_parameter("status_period").as_double();
    marker_period_ = get_parameter("marker_period").as_double();
    control_rate_ = get_parameter("control_rate").as_double();
    plan_retry_max_ = get_parameter("plan_retry_max").as_int();
    plan_retry_interval_ = get_parameter("plan_retry_interval").as_double();
  }

  // ===================== 地图 =====================
  void buildMap()
  {
    const double ox = -12.0, oy = -8.0;
    // ★ 双图分工 ：A* 用膨胀 0.60 图，TEB 盘采样用原始图（无膨胀），
    //   消除"双重膨胀→通道内无可行走廊→幽灵掉头"（详见 node_teb 注释）。
    auto buildRects = [&](WarehouseMap & m, double inflate) {
      // 墙体（内边界）
      m.addRectObstacle(-12.0, -8.0, -11.6, 8.0, inflate);   // 西墙
      m.addRectObstacle(11.6, -8.0, 12.0, 8.0, inflate);     // 东墙
      m.addRectObstacle(-12.0, -8.0, 12.0, -7.6, inflate);   // 南墙
      m.addRectObstacle(-12.0, 7.6, 12.0, 8.0, inflate);     // 北墙
      // 货架：4 排 × 6 列（v3，与 generate_warehouse_world.py 一致）
      const double rows[4] = {5.2, 2.0, -2.0, -5.2};
      const double cols[6] = {-7.0, -4.2, -1.4, 1.4, 4.2, 7.0};
      const double rack_half_w = 0.30;    // v3 视觉半宽 → 通道 2.2m
      for (double ry : rows) {
        for (int ci = 0; ci < 6; ++ci) {
          const double cx = cols[ci];
          const double half_l = (ci == 0 || ci == 5) ? 0.5 : 0.8;
          m.addRectObstacle(cx - rack_half_w, ry - half_l,
                            cx + rack_half_w, ry + half_l, inflate);
        }
      }
      // 地图漏画但 Gazebo 真实存在的立体障碍（§2.24）：控制室整块封禁。
      //   控制室已平移至 (5.6,5.2)（c4/c5 走廊正中），包络含东墙。
      m.addRectObstacle(4.6, 4.4, 6.65, 6.0, inflate);
      // 8 个站台货箱
      const double goods[8][2] = {
        {-11.25, 0.0}, {11.25, 0.0}, {0.85, 6.8}, {-0.85, -6.8},
        {-11.25, 6.8}, {11.25, 6.8}, {-11.25, -6.8}, {11.25, -6.8}};
      for (const auto & g : goods) {
        m.addRectObstacle(g[0] - 0.2, g[1] - 0.2, g[0] + 0.2, g[1] + 0.2, inflate);
      }
      // 消防栓（map_ 中本就被墙膨胀覆盖，raw_map_ 需要——TEB 盘采样源）
      m.addRectObstacle(-0.2, 7.3, 0.2, 7.7, inflate);
      m.addRectObstacle(-0.2, -7.7, 0.2, -7.3, inflate);
    };
    map_ = std::make_unique<WarehouseMap>(ox, oy, 0.2, 120, 80);
    buildRects(*map_, 0.60);              // A* / global_map / 楔死点注入用
    // 充电停车口袋（问题 29）：充电目标 (±11,±3.6) 栅格中心被墙膨胀吞掉，
    // 在 A* 图上凿口袋；raw_map_ 不动，TEB 盘仍按真实墙约束。
    for (std::size_t i = 0; i < charger_x_.size() && i < charger_y_.size(); ++i) {
      map_->clearRect(charger_x_[i] - 0.3, charger_y_[i] - 0.5,
                      charger_x_[i] + 0.3, charger_y_[i] + 0.5);
    }
    raw_map_ = std::make_unique<WarehouseMap>(ox, oy, 0.2, 120, 80);
    buildRects(*raw_map_, 0.0);
    map_->printStats();
  }

  // ===================== TEB 局部规划器 =====================
  void initTebLocalPlanner()
  {
    g2o::teb::TebConfig cfg;
    cfg.robotRadius = 0.3;
    cfg.maxSpeed = 0.6;
    cfg.safetyMargin = 0.10;
    cfg.weightBounds = 0.;
    cfg.kinematics = g2o::teb::Kinematics::Ackermann;
    cfg.wheelbase = this->wheelbase_;
    cfg.maxSteerAngleDeg = 34.37;
    cfg.weightObstacle = 120.0;
    cfg.minX = -12.0;
    cfg.maxX = 12.0;
    cfg.minY = -8.0;
    cfg.maxY = 8.0;
    cfg.weightBounds = 15.0;
    this->planner_ = g2o::teb::TebPlanner(cfg);
    RCLCPP_INFO(get_logger(), "teb_planner ready");
  }

  void initRobots()
  {
    for (std::size_t i = 0; i < robot_names_.size(); ++i) {
      auto r = std::make_shared<RobotAgent>();
      r->id = static_cast<int>(i);
      r->name = robot_names_[i];
      robots_.push_back(r);
    }
    for (const auto & r : robots_) { name_to_robot_[r->name] = r; }
  }

  // ===================== ROS 接口 =====================
  void initPublishers()
  {
    for (const auto & r : robots_) {
      goal_pubs_[r->name] = this->create_publisher<geometry_msgs::msg::Twist>("/" + r->name + "/cmd_vel", rclcpp::QoS(10));
      local_path_pubs_[r->name] = this->create_publisher<nav_msgs::msg::Path>("/" + r->name + "/plan", rclcpp::QoS(10));
      global_path_pubs_[r->name] = this->create_publisher<nav_msgs::msg::Path>("/" + r->name + "/global_plan", rclcpp::QoS(10));
      // TEB 视角可视化（诊断"幽灵掉头/横穿货架"用，话题名与 node_teb 一致 →
      // 现有 fleet_custom.rviz 无需任何改动）
      teb_pose_pubs_[r->name] = this->create_publisher<geometry_msgs::msg::PoseArray>("/" + r->name + "/teb_poses", rclcpp::QoS(10));
      teb_obstacle_pubs_[r->name] = this->create_publisher<visualization_msgs::msg::MarkerArray>("/" + r->name + "/teb_obstacles", rclcpp::QoS(10));
    }
    // 静态障碍栅格（RViz 货架/墙体显示源；TRANSIENT_LOCAL + 周期重发）
    global_map_pub_ = this->create_publisher<nav_msgs::msg::OccupancyGrid>(
      "/agv_0/global_map", rclcpp::QoS(1).reliable().transient_local());
    // 导航状态查询话题（2Hz，见 publishFeedbackAndStatus）
    nav_status_pub_ = this->create_publisher<fleet_common::msg::FleetNavStatus>(
      "/fleet/nav_status", rclcpp::QoS(10));
  }

  void initSubscribers()
  {
    model_states_sub_ = create_subscription<gazebo_msgs::msg::ModelStates>(
      "/fleet_sim/model_states", rclcpp::SensorDataQoS(),
      [this](const gazebo_msgs::msg::ModelStates::SharedPtr msg) { onModelStates(msg); });

    // 穿模救援传送服务（问题 28）：world 的 gazebo_ros_state 插件命名空间 /fleet_sim
    tp_client_ = create_client<gazebo_msgs::srv::SetEntityState>(
      "/fleet_sim/set_entity_state");

    // 回读自己发布的 cmd_vel（盲脱困需要"最后一次指令"的符号）
    for (const auto & robot : robots_) {
      vec_subs_[robot->name] = create_subscription<geometry_msgs::msg::Twist>(
        "/" + robot->name + "/cmd_vel", rclcpp::QoS(10),
        [this, robot](const geometry_msgs::msg::Twist msg) {
          robot->cmd_v = msg.linear.x;
          robot->cmd_w = msg.angular.z;
        });
    }
  }

  // ===================== 多车导航 Action =====================
  void initActionServer()
  {
    action_server_ = rclcpp_action::create_server<NavigateToPose>(
      this, "navigate_to_pose",
      // 目标请求：校验车名 + 该车是否空闲
      [this](const rclcpp_action::GoalUUID &,
             std::shared_ptr<const NavigateToPose::Goal> goal)
        -> rclcpp_action::GoalResponse {
        auto it = name_to_robot_.find(goal->robot_name);
        if (it == name_to_robot_.end()) {
          RCLCPP_WARN(get_logger(), "action goal 拒绝：未知机器人 '%s'", goal->robot_name.c_str());
          return rclcpp_action::GoalResponse::REJECT;
        }
        if (jobByRobot(it->second)) {
          RCLCPP_WARN(get_logger(), "action goal 拒绝：%s 正在导航中（先取消或等待完成）",
            goal->robot_name.c_str());
          return rclcpp_action::GoalResponse::REJECT;
        }
        return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
      },
      // 取消请求：一律接受；按 goal handle 找到对应 NavJob 置取消标志，
      // 停车 + CANCELED 结果由下一控制拍的 navTick ① 统一执行（避免回调内改状态）。
      [this](const std::shared_ptr<GoalHandleNav> gh)
        -> rclcpp_action::CancelResponse {
        for (auto & kv : job_by_robot_) {
          NavJob & job = kv.second;
          if (job.active() && job.gh == gh) {
            job.cancel_requested = true;
            RCLCPP_INFO(get_logger(), "[%s] 收到取消请求 → 下一控制拍停车",
              kv.first.c_str());
            return rclcpp_action::CancelResponse::ACCEPT;
          }
        }
        // 找不到（已终态）→ 也接受，结果回调会按已完结处理
        return rclcpp_action::CancelResponse::ACCEPT;
      },
      // 目标被接受：登记 NavJob，导航从下一控制拍开始
      [this](const std::shared_ptr<GoalHandleNav> gh) {
        const auto & goal = gh->get_goal();
        auto it = name_to_robot_.find(goal->robot_name);
        if (it == name_to_robot_.end()) { return; }   // 不会发生（handle_goal 已挡）
        const auto & robot = it->second;

        NavJob job;
        job.gh = gh;
        job.x = goal->x;
        job.y = goal->y;
        job.timeout = std::max(0.0, goal->timeout_sec);
        job.priority = goal->priority;
        job.t0 = secNow();
        job.state = NavigateToPose::Feedback::NAV_PLANNING;
        job_by_robot_[robot->name] = job;

        // 复位该车全部导航簿记（换目标 = 新航段）
        robot->goal_x = goal->x;
        robot->goal_y = goal->y;
        robot->has_new_goal = true;
        robot->path.clear();
        robot->waypoints.clear();
        robot->wp_idx = 0;
        last_traj_by_robot_[robot->name].clear();
        reverse_by_robot_[robot->name] = false;
        pursuit_by_robot_[robot->name] = PursuitState{};
        stuck_by_robot_[robot->name] = StuckGuard{};
        watch_by_robot_[robot->name] = StuckWatch{};
        robot->state = RobotState::NAV_DROPOFF;   // TrafficControl 语义：移动中

        RCLCPP_INFO(get_logger(),
          "[%s] 接受导航目标 (%.2f, %.2f) timeout=%.0fs priority=%d",
          robot->name.c_str(), goal->x, goal->y, job.timeout, job.priority);
      });
  }

  /// 该车当前 NavJob（无则 nullptr）
  NavJob * jobByRobot(const std::shared_ptr<RobotAgent> & r)
  {
    auto it = job_by_robot_.find(r->name);
    return (it != job_by_robot_.end() && it->second.active()) ? &it->second : nullptr;
  }

  /// 该车当前导航优先级（无任务 = 999，与 node_teb 的 RobotAgent::priority() 同语义）
  int navPriority(const std::shared_ptr<RobotAgent> & r) const
  {
    auto it = job_by_robot_.find(r->name);
    return (it != job_by_robot_.end() && it->second.active()) ? it->second.priority : 999;
  }

  // ===================== 主控制周期 =====================
  void controlTick()
  {
    double now = secNow();
    tick_begin_.store(now);
    double dt = std::min(now - last_control_, 0.1);
    last_control_ = now;
    if (dt <= 0.0) { return; }
    sim_time_ += dt;

    setStage(2);
    traffic_->update(robots_);

    setStage(1);
    for (auto & robot : robots_) {
      robot->current_cell = map_->worldToCell(robot->x, robot->y);
      if (robot->has_pose) { checkPenetration(robot); }
      navTick(robot, dt);
    }
    setStage(0);
  }

  /// 单车导航状态机：取消 / 超时 / 规划（含重试）/ 行驶 / 让行 → 终态上报
  void navTick(const std::shared_ptr<RobotAgent> & robot, double dt)
  {
    NavJob * job = jobByRobot(robot);
    if (!job) { return; }
    // 尚未收到该车真值位姿（外部客户端抢跑）：不规划不控制，等位姿到达
    if (!robot->has_pose) { return; }

    // ---- ① 客户端取消：立即停车 + CANCELED ----
    if (job->cancel_requested) {
      this->publishZero(robot->name);
      finishJob(robot, job, NavigateToPose::Result::STATUS_CANCELED, "客户端取消");
      return;
    }

    // ---- ② 超时：停车 + TIMEOUT（0 = 不限时）----
    if (job->timeout > 0.0 && (secNow() - job->t0) > job->timeout) {
      this->publishZero(robot->name);
      finishJob(robot, job, NavigateToPose::Result::STATUS_TIMEOUT,
        "超时（>" + std::to_string(job->timeout) + "s）");
      return;
    }

    // ---- ③ 规划（新目标 / 让行恢复 / 重规划要求）----
    if (robot->has_new_goal) {
      job->state = NavigateToPose::Feedback::NAV_PLANNING;
      const double t = secNow();
      // 重试间隔门（上次失败后间隔未到 → 停车等待，has_new_goal 保持 true）
      if (job->last_plan_fail > 0.0 && t - job->last_plan_fail < plan_retry_interval_) {
        this->publishZero(robot->name);
        return;
      }
      const bool ok = this->planPath(robot);
      if (!ok) {
        // ★ 有意差异（相对 node_teb 的"无限滞留"）：A* 失败按预算重试，
        //   耗尽 → ABORTED 上报，调度器可重新分配而不是干等。
        job->last_plan_fail = t;
        if (++job->plan_fails > plan_retry_max_) {
          finishJob(robot, job, NavigateToPose::Result::STATUS_ABORTED,
            "A* 不可达（重试 " + std::to_string(plan_retry_max_) + " 次耗尽）");
          return;
        }
        RCLCPP_WARN(get_logger(), "[%s] A* 失败，%.1fs 后重试 (%d/%d)",
          robot->name.c_str(), plan_retry_interval_, job->plan_fails, plan_retry_max_);
        return;   // has_new_goal 保持 true → 下个间隔自动重试
      }
      robot->has_new_goal = false;
      job->plan_fails = 0;
      job->last_plan_fail = 0.0;
      job->state = NavigateToPose::Feedback::NAV_MOVING;
    }

    // ---- ④ 行驶中：卡死监测（仲裁让行 / 盲脱困在内部触发）----
    // 覆盖所有移动态（NAV_PICKUP / NAV_DROPOFF / TO_CHARGER）
    if (robot->isMovingState()) {
      this->updateStuckWatch(robot, dt);
    } else if (robot->state != RobotState::YIELD) {
      this->watch_by_robot_[robot->name].t0 = 0.0;
    }

    // ---- ⑤ 阶段执行 ----
    if (robot->state == RobotState::YIELD) {
      // 让行中（迁移自 node_teb 的 YIELD case：goal 保持 EXECUTING）
      job->state = NavigateToPose::Feedback::NAV_YIELD;
      StuckWatch & sw = this->watch_by_robot_[robot->name];
      sw.yield_backed += std::fabs(robot->lin_vel) * dt;
      geometry_msgs::msg::Twist msg;
      if (sw.yield_backed < sw.yield_target) {
        msg.linear.x = sw.yield_vsign * this->yield_speed_;
        msg.angular.z = 0.0;
      } else {
        msg.linear.x = 0.0;
        msg.angular.z = 0.0;
      }
      this->goal_pubs_[robot->name]->publish(msg);

      const std::shared_ptr<RobotAgent> winner = this->robotById(sw.yield_winner);
      const bool clear = !winner || !winner->has_pose ||
        robot->distanceTo(winner->x, winner->y) > this->yield_clear_dist_;
      if (clear || (secNow() - job->yield_t0) > this->yield_timeout_) {
        RCLCPP_INFO(get_logger(), "[%s] 让行结束（%s）→ 恢复导航重新规划",
          robot->name.c_str(), clear ? "赢家已通过" : "超时");
        sw.t0 = 0.0;
        sw.yield_winner = -1;
        sw.yield_backed = 0.0;
        sw.yield_target = 0.0;
        this->stuck_by_robot_[robot->name] = StuckGuard{};   // 复位盲解救
        robot->state = RobotState::NAV_DROPOFF;
        robot->has_new_goal = true;      // 从当前位姿全量重规划
        job->state = NavigateToPose::Feedback::NAV_MOVING;
      }
      return;
    }

    if (this->stuck_by_robot_[robot->name].phase >= 1) {
      job->state = NavigateToPose::Feedback::NAV_RECOVERY;   // 盲脱困相位由 publishPursuitCommand 执行
    } else {
      job->state = NavigateToPose::Feedback::NAV_MOVING;
    }

    // ---- ⑥ 行驶：TEB + 纯跟踪（sendNavDrive 内含停车安全）----
    this->sendNavDrive(robot);

    // ---- ⑦ 到点判定：SUCCEEDED ----
    if (robot->distanceTo(robot->goal_x, robot->goal_y) < goal_tol_) {
      this->publishZero(robot->name);
      auto res = std::make_shared<NavigateToPose::Result>();
      res->status = NavigateToPose::Result::STATUS_SUCCEEDED;
      res->final_x = robot->x;
      res->final_y = robot->y;
      res->distance_remaining = robot->distanceTo(robot->goal_x, robot->goal_y);
      res->message = "到达目标点";
      RCLCPP_INFO(get_logger(), "[%s] 到达 (%.2f, %.2f)，耗时 %.1fs",
        robot->name.c_str(), robot->x, robot->y, secNow() - job->t0);
      job->gh->succeed(res);
      job_by_robot_.erase(robot->name);
      robot->state = RobotState::IDLE;   // TrafficControl 语义：驻留
    }
  }

  /// 终态收尾：停车 + 对应结果 + 簿记清理
  void finishJob(const std::shared_ptr<RobotAgent> & robot, NavJob * job,
                 uint8_t status, const std::string & message)
  {
    auto res = std::make_shared<NavigateToPose::Result>();
    res->status = status;
    res->final_x = robot->x;
    res->final_y = robot->y;
    res->distance_remaining = robot->distanceTo(robot->goal_x, robot->goal_y);
    res->message = message;
    RCLCPP_WARN(get_logger(), "[%s] 导航终态(%d)：%s",
      robot->name.c_str(), static_cast<int>(status), message.c_str());
    if (status == NavigateToPose::Result::STATUS_CANCELED) {
      job->gh->canceled(res);
    } else {
      job->gh->abort(res);
    }
    job_by_robot_.erase(robot->name);
    robot->state = RobotState::IDLE;
    this->stuck_by_robot_[robot->name] = StuckGuard{};
    this->watch_by_robot_[robot->name] = StuckWatch{};
  }

  // ===================== 位姿订阅（迁移自 node_teb.onModelStates） =====================
  void onModelStates(const gazebo_msgs::msg::ModelStates::SharedPtr msg)
  {
    if (msg->name.size() != msg->pose.size()) { return; }
    double now = secNow();
    for (std::size_t i = 0; i < msg->name.size(); ++i) {
      auto it = name_to_robot_.find(msg->name[i]);
      if (it == name_to_robot_.end()) { continue; }
      auto r = it->second;
      r->x = msg->pose[i].position.x;
      r->y = msg->pose[i].position.y;
      r->theta = quatToYaw(
        msg->pose[i].orientation.x, msg->pose[i].orientation.y,
        msg->pose[i].orientation.z, msg->pose[i].orientation.w);
      r->has_pose = true;
      r->last_pose_time = now;
      r->vx = msg->twist[i].linear.x;
      r->vy = msg->twist[i].linear.y;
      r->lin_vel = std::hypot(msg->twist[i].linear.x, msg->twist[i].linear.y);
      r->ang_vel = msg->twist[i].angular.z;
    }
  }

  // ===================== 穿模看门狗（问题 28 · 迁移自 node_teb，逐字保留） =====================
  void checkPenetration(const std::shared_ptr<RobotAgent> & robot)
  {
    const Cell c = raw_map_->worldToCell(robot->x, robot->y);
    if (raw_map_->isFree(c)) {
      pen_cnt_[robot->name] = 0;
      return;
    }
    if (++pen_cnt_[robot->name] < 5) { return; }
    pen_cnt_[robot->name] = 0;

    const double now = secNow();
    if (now - last_tp_[robot->name] < 5.0) { return; }   // 冷却，防连续传送

    Cell best{-1, -1};
    for (int rad = 1; rad <= 12 && best.x < 0; ++rad) {
      for (int dx = -rad; dx <= rad && best.x < 0; ++dx) {
        for (int dy = -rad; dy <= rad; ++dy) {
          if (std::max(std::abs(dx), std::abs(dy)) != rad) { continue; }
          const Cell cc{c.x + dx, c.y + dy};
          if (!raw_map_->inBounds(cc)) { continue; }
          bool ok = true;
          for (int ox = -2; ox <= 2 && ok; ++ox) {
            for (int oy = -2; oy <= 2 && ok; ++oy) {
              const Cell n{cc.x + ox, cc.y + oy};
              if (!raw_map_->inBounds(n) || !raw_map_->isFree(n)) { ok = false; }
            }
          }
          if (ok) { best = cc; break; }
        }
      }
    }
    if (best.x < 0) {
      RCLCPP_ERROR(get_logger(),
        "%s 穿模看门狗：车心 (%.2f,%.2f) 在实体内，但 2.4m 内无净空格 → 放弃传送",
        robot->name.c_str(), robot->x, robot->y);
      return;
    }
    const double tx = raw_map_->cellCenterX(best), ty = raw_map_->cellCenterY(best);
    RCLCPP_ERROR(get_logger(),
      "%s 穿模看门狗：车心 (%.2f,%.2f) 深入实体障碍（抛飞/数值爆炸类）→ 传送到 (%.2f,%.2f)",
      robot->name.c_str(), robot->x, robot->y, tx, ty);
    teleportEntity(robot->name, tx, ty, robot->theta);
    last_tp_[robot->name] = now;
  }

  void teleportEntity(const std::string & name, double x, double y, double yaw)
  {
    if (!tp_client_ || !tp_client_->service_is_ready()) {
      RCLCPP_ERROR(get_logger(),
        "%s 穿模救援失败：/fleet_sim/set_entity_state 服务不可用", name.c_str());
      return;
    }
    gazebo_msgs::srv::SetEntityState::Request req;
    req.state.name = name;
    req.state.pose.position.x = x;
    req.state.pose.position.y = y;
    req.state.pose.position.z = 0.08;                 // 与 spawn 高度一致（轮半径 0.08）
    req.state.pose.orientation.z = std::sin(0.5 * yaw);
    req.state.pose.orientation.w = std::cos(0.5 * yaw);
    tp_client_->async_send_request(
      std::make_shared<gazebo_msgs::srv::SetEntityState::Request>(req));
  }

  // ===================== 卡死检测与让行仲裁 =====================
  // job priority——仲裁全序（数值大让行、同值 id 大让行）语义不变。
  // 每隔 stuck_window_ 秒，记一次车的位置（sw.x/sw.y），和窗口起点位置比位移。
  void updateStuckWatch(const std::shared_ptr<RobotAgent> & robot, double /*dt*/)
  {
    // 临近目标蠕行不参与卡死判定
    if (robot->distanceTo(robot->goal_x, robot->goal_y) < 1.5 * this->goal_tol_) {
      this->watch_by_robot_[robot->name].t0 = 0.0;   
      return;
    }
    StuckWatch & sw = this->watch_by_robot_[robot->name];
    if (sw.t0 == 0.0) {
      sw.t0 = this->sim_time_;
      sw.x = robot->x; sw.y = robot->y;
      return;
    }
    if (this->sim_time_ - sw.t0 < this->stuck_window_) { return; }

    const double disp = std::hypot(robot->x - sw.x, robot->y - sw.y);
    sw.t0 = this->sim_time_;
    sw.x = robot->x; sw.y = robot->y;
    // 如果这个窗口里车移动的距离达到了阈值 stuck_disp，说明车是活的,不执行脱困策略
    if (disp >= this->stuck_disp_) {
      auto & att = this->stuck_by_robot_[robot->name].attempts;
      att = (att > 0) ? att - 1 : 0;
      sw.yield_count = 0; // 对同一赢家连续让行次数
      return;
    }

    const std::shared_ptr<RobotAgent> near = this->nearestOther(robot, this->yield_scope_);
    // ================= 分支 A：无邻车盲脱困（三阶段摆尾） =================
    // 触发前提：位移窗口判定卡死（stuck_window 秒内位移 < stuck_disp），且
    // yield_scope 半径内没有任何邻车 → 排除了"被别人堵住"的可能，
    // 判定为物理楔死（货架角 / 墙角 / 斜角卡死），只能靠自己挪姿态脱困。
    if (!near) {
      StuckGuard & sg = this->stuck_by_robot_[robot->name];
      // phase==0 表示当前没有脱困动作在执行，这里负责"点火"一次：
      // 选定本阶段动作并设置初速度；真正的执行器在 publishPursuitCommand
      // 的"盲脱困执行器"段（sg.phase>=1 时按计时接管 cmd_vel，到时 phase 清零）。
      if (sg.phase == 0) {
        // ---- ① 按 attempts % 3 轮换三阶段动作，摆尾换姿态 ----
        //   尝试#0/3/6... → phase 1 直退（最温和，先试能不能直接退出来）
        //   尝试#1/4/7... → phase 2 倒车 + 右舵 0.5（右摆尾）
        //   尝试#2/5/8... → phase 3 倒车 + 左舵 -0.5（左摆尾）
        // 斜角楔死时纯直退退不出（轮子顶死货架边），必须倒车带转角
        // 让车尾偏摆、改变车体姿态角度才有空间脱出——所以三阶段交替。
        const int k = sg.attempts % 3;
        sg.phase = (k == 0) ? 1 : ((k == 1) ? 2 : 3);
        sg.steer = (sg.phase == 2) ? 0.5 : ((sg.phase == 3) ? -0.5 : 0.0);
        sg.t0 = this->now().seconds();   // 本阶段动作计时起点（执行器按 dur 切断）
        // ---- ② 计算脱困初速 rec_v：方向取"上一条指令的反向" ----
        // 上一拍还在往前顶（last_cmd>0.02）就倒着退；在往后倒就顶着出；
        // 几乎为零则一律取 -mag（默认倒车起步）。
        const double last_cmd = robot->cmd_v;
        // ---- ③ 力度递进：前 3 次温和 0.18 m/s，3 次还没出来加大到 0.26 m/s ----
        // （每次成功脱困后 attempts 只减 1 不清零，见 disp>=stuck_disp 分支，
        //   所以"挪一点又楔回去"不会重置力度档位。）
        const double mag = (sg.attempts >= 3) ? 0.26 : 0.18;
        sg.rec_v = (std::fabs(last_cmd) > 0.02) ? -mag * (last_cmd > 0.0 ? 1.0 : -1.0)
                                                : -mag;
        RCLCPP_WARN(get_logger(),
          "[%s] 卡死 %.1fs 位移 %.2fm（无邻车，末指令 %+.2f）→ 盲脱困#%d %s",
          robot->name.c_str(), this->stuck_window_, disp, last_cmd, sg.attempts + 1,
          sg.phase == 1 ? "直退" : (sg.phase == 2 ? "倒车右摆尾" : "倒车左摆尾"));
        // ---- ④ attempts>=3：除了摆尾，还要换路 —— 强制全量重规划 ----
        // 摆尾只能解决"这次怎么出来"；连续 3 次楔死说明 A* 给的还是同一条
        // 穿楔角的死路，必须连"下次走哪"一起换：
        //   has_new_goal = true       → 下拍强制 A* 全量重规划
        //   清空 last_traj_by_robot_  → 丢弃热启动 seed（旧 seed 会把 TEB 再次
        //                               拉回同一条穿楔角的轨迹，即"毒 seed"）
        //   addWedgePoint(robot)      → 把车头前方 0.6m 接触点记为临时障碍
        //                               （A* 与 TEB 都绕开，25s TTL 自动过期）
        // 这是打破"脱困 → 跟同一条死路 → 再楔死"恶性循环的关键（吞吐量质变点）。
        if (sg.attempts >= 3) {
          robot->has_new_goal = true;
          this->last_traj_by_robot_[robot->name].clear();   // 清毒 seed
          this->addWedgePoint(robot);                        // 楔死点记障碍
          RCLCPP_WARN(get_logger(), "[%s] 连续 %d 次楔死 → 强制全量重规划（清 seed + 记楔死点）",
            robot->name.c_str(), sg.attempts + 1);
        }
      }
      // 无论 phase 是否为 0 都 return：phase>=1 期间由执行器接管，本函数
      // 不再产生新的判定（等脱困动作做完、phase 清零后再回到正常监测）。
      return;
    }

    // 有邻车：优先级仲裁（job priority 替代任务 priority，全序比较不变）
    const int my_p = this->navPriority(robot), op_p = this->navPriority(near);
    const bool i_lose = (my_p > op_p) || (my_p == op_p && robot->id > near->id);
    if (!i_lose) {
      RCLCPP_INFO(get_logger(),
        "[%s] 卡死但仲裁胜出（p%d#%d vs p%d#%d）→ 等待 %s 让行",
        robot->name.c_str(), my_p, robot->id, op_p, near->id, near->name.c_str());
      return;
    }

    // ================= 分支 B：让行循环保护 → 转自己的盲脱困 =================
    // 到这里说明：卡死判定成立 + 有邻车 + 优先级仲裁我输了（本来该我让行）。
    // 但如果"对同一个赢家连续让行 >=3 次自己仍无位移"，说明赢家大概率自己
    // 楔死了、永远过不去——继续让行只会无限空等。于是放弃让行，改走自己的
    // 三阶段摆尾脱困（把姿态挪开，哪怕赢家不动我也先能退出去）。
    // 注意：yield_count 由 disp>=stuck_disp（车动起来）或换赢家时清零/重置，
    // 只有"同一个赢家 + 我每次让完仍卡死"才会累加到 3。
    if (near->id == sw.last_winner) { ++sw.yield_count; } else { sw.yield_count = 1; }
    sw.last_winner = near->id;
    if (sw.yield_count >= 3) {
      RCLCPP_WARN(get_logger(),
        "[%s] 对 %s 连续 %d 次让行仍卡死（赢家疑楔死）→ 转盲脱困摆尾",
        robot->name.c_str(), near->name.c_str(), sw.yield_count);
      StuckGuard & sg = this->stuck_by_robot_[robot->name];
      // 与分支 A 的点火逻辑完全一致（三阶段轮换 / 反向初速 / 力度递进），
      // 区别只有两点：
      // ① 不带 addWedgePoint / 清 seed：楔死点在赢家那边而不是我这边，
      //    我的位置未必是死路，不该把这里永久记成障碍；
      // ② 也不强制全量重规划（不置清 seed）——我的全局路径本身没错，
      //    只是被赢家挡住；脱困挪开姿态后继续让行仲裁即可。
      // 真正的执行同样由 publishPursuitCommand 的盲脱困执行器完成。
      if (sg.phase == 0) {
        const int k = sg.attempts % 3;
        sg.phase = (k == 0) ? 1 : ((k == 1) ? 2 : 3);
        sg.steer = (sg.phase == 2) ? 0.5 : ((sg.phase == 3) ? -0.5 : 0.0);
        sg.t0 = this->now().seconds();
        const double last_cmd = robot->cmd_v;
        const double mag = (sg.attempts >= 3) ? 0.26 : 0.18;
        sg.rec_v = (std::fabs(last_cmd) > 0.02) ? -mag * (last_cmd > 0.0 ? 1.0 : -1.0)
                                                : -mag;
        if (sg.attempts >= 3) { robot->has_new_goal = true; }
      }
      return;
    }

    // 我让行：退让方向 = 远离赢家
    const double fx = std::cos(robot->theta), fy = std::sin(robot->theta);
    const double proj = (near->x - robot->x) * fx + (near->y - robot->y) * fy;
    sw.yield_vsign = (proj > 0.0) ? -1 : +1;
    sw.yield_target = std::max(0.40, this->yield_backup_);
    sw.yield_backed = 0.0;
    sw.yield_winner = near->id;
    sw.t0 = 0.0;
    this->stuck_by_robot_[robot->name] = StuckGuard{};
    RCLCPP_WARN(get_logger(),
      "[%s] 卡死 %.1fs 位移 %.2fm，邻车 %s（p%d#%d）→ 仲裁让行（%s %.2fm）",
      robot->name.c_str(), this->stuck_window_, disp, near->name.c_str(),
      op_p, near->id,
      sw.yield_vsign < 0 ? "倒退" : "前进", sw.yield_target);
    // 进入让行（goal 保持 EXECUTING；恢复时间戳记在 job 上）
    robot->state = RobotState::YIELD;
    auto * job = jobByRobot(robot);
    if (job) { job->yield_t0 = secNow(); }
  }

  /// @brief 查找最近的邻车
  /// @param self 
  /// @param radius 
  /// @return 
  std::shared_ptr<RobotAgent> nearestOther(const std::shared_ptr<RobotAgent> & self, double radius)
  {
    std::shared_ptr<RobotAgent> best;
    double bd = radius;
    for (const auto & r : robots_) {
      if (r.get() == self.get() || !r->has_pose) { continue; }
      const double d = self->distanceTo(r->x, r->y);
      if (d < bd) { bd = d; best = r; }
    }
    return best;
  }

  /// @brief 根据机器人id获取机器人对象
  /// @param id 
  /// @return 
  std::shared_ptr<RobotAgent> robotById(int id)
  {
    for (const auto & r : robots_) {
      if (r->id == id) { return r; }
    }
    return nullptr;
  }

  // ===================== 楔死点记忆 =====================
  struct WedgePoint { double x, y; double expire; };
  std::vector<WedgePoint> wedge_points_;

  void addWedgePoint(const std::shared_ptr<RobotAgent> & robot)
  {
    const double wx = robot->x + 0.6 * std::cos(robot->theta);
    const double wy = robot->y + 0.6 * std::sin(robot->theta);
    const double ttl = 25.0;
    for (auto & w : this->wedge_points_) {
      if (std::hypot(w.x - wx, w.y - wy) < 0.5) {
        w.expire = this->sim_time_ + ttl;
        return;
      }
    }
    this->wedge_points_.push_back({wx, wy, this->sim_time_ + ttl});
  }

  std::vector<std::pair<double, double>> activeWedgePoints() const
  {
    std::vector<std::pair<double, double>> out;
    for (const auto & w : this->wedge_points_) {
      if (w.expire > this->sim_time_) { out.emplace_back(w.x, w.y); }
    }
    return out;
  }

  // ===================== 全局规划 =====================
  bool planPath(const std::shared_ptr<RobotAgent> & r)
  {
    std::unordered_set<Cell, CellHash> dyn;
    std::unordered_set<Cell, CellHash> dyn_traffic;

    double gx = r->goal_x, gy = r->goal_y;
    Cell start = map_->worldToCell(r->x, r->y);
    Cell goal = map_->worldToCell(gx, gy);

    // 楔死点注入动态障碍（3×3 格）
    for (const auto & wp : this->activeWedgePoints()) {
      const Cell wc = map_->worldToCell(wp.first, wp.second);
      for (int dx = -1; dx <= 1; ++dx) {
        for (int dy = -1; dy <= 1; ++dy) {
          const Cell nc = {wc.x + dx, wc.y + dy};
          if (map_->inBounds(nc)) { dyn.insert(nc); }
        }
      }
    }

    // 路权信息接入全局规划：他车足迹/预留 → A* 硬障碍
    this->traffic_->dynamicObstacles(r->id, dyn_traffic);
    for (const auto & c : dyn_traffic) { dyn.insert(c); }
    if (!dyn_traffic.empty()) {
      RCLCPP_INFO(get_logger(), "%s: 规划含路权避让（他车足迹/预留 %zu 格）",
                  r->name.c_str(), dyn_traffic.size());
    }

    AStarPlanner::FreeCheck free = [this, &dyn](const Cell & c) {
      if (!map_->isFree(c)) { return false; }
      return dyn.find(c) == dyn.end();
    };

    // 起点/目标放宽（膨胀区是虚拟障碍，nearestFree 到最近可通行格）
    if (!free(start)) {
      Cell alt = AStarPlanner::nearestFree(free, start, 8);
      if (alt.x < 0) {
        RCLCPP_WARN(get_logger(), "%s: start cell occupied（附近无可通行格）", r->name.c_str());
        return false;
      }
      RCLCPP_INFO(get_logger(), "%s: 起点(%d,%d)在膨胀区 → 放宽到(%d,%d)",
                  r->name.c_str(), start.x, start.y, alt.x, alt.y);
      start = alt;
    }
    if (!free(goal)) {
      Cell alt = AStarPlanner::nearestFree(free, goal, 8);
      if (alt.x < 0) {
        RCLCPP_WARN(get_logger(), "%s: goal cell occupied（附近无可通行格）(%d,%d)",
                    r->name.c_str(), goal.x, goal.y);
        return false;
      }
      RCLCPP_INFO(get_logger(), "%s: 目标栅格(%d,%d)在膨胀区 → 放宽到(%d,%d)",
                  r->name.c_str(), goal.x, goal.y, alt.x, alt.y);
      goal = alt;
    }

    std::vector<Cell> path = AStarPlanner::plan(free, start, goal, 30000, 0.3);
    if (path.empty() && !dyn_traffic.empty()) {
      // 路权障碍造成"全局不可达" → 忽略他车足迹重试（TEB 软代价 + 让行仲裁兜底）
      RCLCPP_WARN(get_logger(), "%s: A*(含路权避让)不可达 → 忽略他车足迹重试",
                  r->name.c_str());
      for (const auto & c : dyn_traffic) { dyn.erase(c); }
      path = AStarPlanner::plan(free, start, goal, 30000, 0.3);
    }
    if (path.empty()) {
      RCLCPP_WARN(get_logger(), "%s: A* failed to (%.2f, %.2f)", r->name.c_str(), gx, gy);
      return false;
    }
    std::vector<Cell> wps = AStarPlanner::simplify(free, path);
    r->path.assign(path.begin(), path.end());
    r->waypoints.clear();
    for (const auto & c : wps) {
      r->waypoints.push_back({map_->cellCenterX(c), map_->cellCenterY(c)});
    }
    if (r->waypoints.empty() ||
        std::hypot(r->waypoints.back().first - gx, r->waypoints.back().second - gy) > 0.05)
    {
      r->waypoints.push_back({gx, gy});
    }
    r->wp_idx = 0;
    r->leg_start_x = r->x;
    r->leg_start_y = r->y;

    // 新航段方向初始化：一律前进起步（倒车航段在 Gazebo 实测不可行）
    if (!r->waypoints.empty()) {
      this->reverse_by_robot_[r->name] = false;
      this->rev_proj_ema_[r->name] = 0.0;
      this->rev_cand_ts_[r->name] = 0.0;
      RCLCPP_INFO(get_logger(), "[%s] 新航段方向初始化 -> 前进", r->name.c_str());
    }
    return true;
  }

  // ===================== TEB 障碍收集 =====================
  std::vector<g2o::teb::Obstacle> getRobotNearbyObstacles(
    const std::shared_ptr<RobotAgent> & robot,
    const g2o::teb::Pose2D & start, const g2o::teb::Pose2D & goal)
  {
    std::vector<g2o::teb::Obstacle> obstacles;
    const int nx = 120, ny = 80;
    const int iMaxObstacles = 120;
    const int iStride = 2;
    const double res = 0.2;
    const double disc_r = 0.30;
    const double iMaxRange = 3.0;

    const double sx = start.x, sy = start.y;
    const double gx = goal.x,  gy = goal.y;
    const double seg_dx = gx - sx, seg_dy = gy - sy;
    const double seg_len2 = seg_dx * seg_dx + seg_dy * seg_dy;
    auto distToBand = [&](double px, double py) -> double {
      double t = seg_len2 > 1e-9 ? ((px - sx) * seg_dx + (py - sy) * seg_dy) / seg_len2 : 0.0;
      t = std::max(0.0, std::min(1.0, t));
      return std::hypot(px - (sx + t * seg_dx), py - (sy + t * seg_dy));
    };

    const double margin = 1.5;
    const double wx0 = std::min(robot->x, gx) - margin, wx1 = std::max(robot->x, gx) + margin;
    const double wy0 = std::min(robot->y, gy) - margin, wy1 = std::max(robot->y, gy) + margin;
    const int col0 = std::max(0, static_cast<int>(std::floor((wx0 + 12.0) / res)));
    const int col1 = std::min(nx - 1, static_cast<int>(std::floor((wx1 + 12.0) / res)));
    const int row0 = std::max(0, static_cast<int>(std::floor((wy0 + 8.0) / res)));
    const int row1 = std::min(ny - 1, static_cast<int>(std::floor((wy1 + 8.0) / res)));

    struct Candidate { g2o::teb::Obstacle obstacle; double distance; };
    std::vector<Candidate> candidates;

    for (int col = col0; col <= col1; col += iStride) {
      for (int row = row0; row <= row1; row += iStride) {
        Cell curCell = {col, row};
        if (this->raw_map_->inBounds(curCell) && this->raw_map_->isFree(curCell)) {
          continue;
        }
        Candidate cand;
        cand.obstacle = g2o::teb::Obstacle(this->raw_map_->cellCenterX(curCell),
                                          this->raw_map_->cellCenterY(curCell),
                                          disc_r);
        cand.distance = distToBand(cand.obstacle.center[0],
                                   cand.obstacle.center[1]);
        candidates.push_back(cand);
      }
    }

    for (const std::shared_ptr<RobotAgent> & otherRobot : this->robots_) {
      if (otherRobot == robot) { continue; }
      const double d_robot = std::hypot(otherRobot->x - robot->x, otherRobot->y - robot->y);
      if (d_robot > iMaxRange) { continue; }
      Candidate cand;
      cand.distance = std::min(distToBand(otherRobot->x, otherRobot->y), d_robot);
      cand.obstacle = g2o::teb::Obstacle(otherRobot->x, otherRobot->y, this->robot_radius_, true);
      // §2.27 修复：velocity = 世界系二维线速度；角速度走 yawRate 字段
      cand.obstacle.velocity = g2o::Vector2(otherRobot->vx, otherRobot->vy);
      cand.obstacle.yawRate = otherRobot->ang_vel;
      candidates.push_back(cand);
    }

    for (const auto & wp : this->activeWedgePoints()) {
      const double d_robot = std::hypot(wp.first - robot->x, wp.second - robot->y);
      if (d_robot > iMaxRange) { continue; }
      Candidate cand;
      cand.distance = std::min(distToBand(wp.first, wp.second), d_robot);
      cand.obstacle = g2o::teb::Obstacle(wp.first, wp.second,
                                         0.8 * this->robot_radius_, true);
      candidates.push_back(cand);
    }

    if (static_cast<int>(candidates.size()) > iMaxObstacles) {
      std::partial_sort(candidates.begin(), candidates.begin() + iMaxObstacles,
                        candidates.end(),
                        [](const Candidate& a, const Candidate& b) {
                          return a.distance < b.distance;
                        });
      candidates.resize(iMaxObstacles);
    }
    obstacles.reserve(candidates.size());
    for (const Candidate& cand : candidates) obstacles.push_back(cand.obstacle);
    return obstacles;
  }

  /// NaN 净化③：剔除非有限障碍物
  void purgeNonFiniteObstacles(std::vector<g2o::teb::Obstacle> & obstacles,
                               const std::string & name)
  {
    const std::size_t before = obstacles.size();
    obstacles.erase(std::remove_if(obstacles.begin(), obstacles.end(),
      [](const g2o::teb::Obstacle & o) {
        return !std::isfinite(o.center.x()) || !std::isfinite(o.center.y()) ||
               !std::isfinite(o.radius) || !std::isfinite(o.yaw) ||
               !std::isfinite(o.velocity.x()) || !std::isfinite(o.velocity.y());
      }), obstacles.end());
    if (obstacles.size() != before) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
        "%s 剔除了 %zu 个非有限障碍物（NaN/Inf，会挂死优化器）",
        name.c_str(), before - obstacles.size());
    }
  }

  /// @brief 热启动 seed 净化（NaN ① + 端点对齐 F）
  /// @param name 
  /// @param goal 
  /// @param seed 
  void sanitizeSeed(const std::string & name, const g2o::teb::Pose2D & goal,
                   std::vector<g2o::teb::Pose2D> & seed)
  {
    if (seed.empty()) { return; }
    for (const auto & p : seed) {
      if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.theta)) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
          "%s 热启动轨迹含非有限位姿 → 已丢弃该 seed（否则优化器每拍必挂）",
          name.c_str());
        seed.clear();
        return;
      }
    }
    // 判断上一拍存下的热启动轨迹，其终点是否指向当前要去的目标
    // 如果偏离超过 0.6m，说明这条旧轨迹已经过时（目标换了/车挪了），作废清空
    const auto & tail = seed.back();
    if (std::hypot(tail.x - goal.x, tail.y - goal.y) > 0.6) { seed.clear(); }
  }

  /// @brief 剩余行驶距离（沿航点序列）
  /// @param robot 
  /// @return 
  double remainingDistance(const std::shared_ptr<RobotAgent> & robot) const
  {
    double d = 0.0;
    double px = robot->x, py = robot->y;
    for (std::size_t i = robot->wp_idx; i < robot->waypoints.size(); ++i) {
      d += std::hypot(robot->waypoints[i].first - px, robot->waypoints[i].second - py);
      px = robot->waypoints[i].first;
      py = robot->waypoints[i].second;
    }
    return d;
  }

  // ===================== 行驶 =====================
  void publishZero(const std::string & name)
  {
    geometry_msgs::msg::Twist zero;
    this->goal_pubs_[name]->publish(zero);
  }

  bool advanceWaypoint(RobotAgent & robot)
  {
    if (robot.waypoints.empty()) { return false; }
    while (robot.wp_idx + 1 < robot.waypoints.size()) {
      const double d = std::hypot(robot.waypoints[robot.wp_idx].first - robot.x,
                                  robot.waypoints[robot.wp_idx].second - robot.y);
      if (d >= std::max(this->wp_tol_, 0.5)) { break; }
      ++robot.wp_idx;
    }
    return true;
  }

  /// @brief 局部目标朝向(目标点的朝向角 θ),倒车航段 +π；末航点用进入段方向
  /// @param robot 
  /// @return 
  double localGoalHeading(RobotAgent & robot)
  {
    // 现局部规划默认采用前进方式，不用倒退. default : false
    const bool rev = this->reverse_by_robot_[robot.name];
    double th = robot.theta;
    bool have = false;
    if (robot.wp_idx + 1 < robot.waypoints.size()) {
      const auto & p0 = robot.waypoints[robot.wp_idx];
      const auto & p1 = robot.waypoints[robot.wp_idx + 1];
      const double seg = std::hypot(p1.first - p0.first, p1.second - p0.second);
      if (seg > 1e-3) {
        th = std::atan2(p1.second - p0.second, p1.first - p0.first);
        have = true;
      }
    } else {
      const double seg = std::hypot(robot.goal_x - robot.x, robot.goal_y - robot.y);
      if (seg > 1e-3) {
        th = std::atan2(robot.goal_y - robot.y, robot.goal_x - robot.x);
        have = true;
      }
    }
    if (!have) { return robot.theta; }
    return rev ? (th + M_PI) : th;
  }

  /// @brief 
  /// @param robot 
  void sendNavDrive(const std::shared_ptr<RobotAgent> & robot)
  {
    // ② 航点推进；无可行路径 → 安全停车（等待规划重试）
    if (!this->advanceWaypoint(*robot)) {
      this->publishZero(robot->name);
      return;
    }

    // ④ 局部规划输入（NaN 净化②）,检查起点终点每个值都必须是有效值
    setStage(11);
    const g2o::teb::Pose2D start = g2o::teb::Pose2D{robot->x, robot->y, robot->theta};
    const g2o::teb::Pose2D goal = g2o::teb::Pose2D{
      robot->waypoints[robot->wp_idx].first,
      robot->waypoints[robot->wp_idx].second,
      this->localGoalHeading(*robot)};
    if (!std::isfinite(start.x) || !std::isfinite(start.y) || !std::isfinite(start.theta) ||
        !std::isfinite(goal.x) || !std::isfinite(goal.y)) {
      RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 2000,
        "%s 位姿/目标非有限（start=(%.2f,%.2f,%.2f) goal=(%.2f,%.2f)），本拍停车",
        robot->name.c_str(), start.x, start.y, start.theta, goal.x, goal.y);
      this->publishZero(robot->name);
      return;
    }

    // ⑤ 沿带障碍 + NaN 净化③
    setStage(14);
    std::vector<g2o::teb::Obstacle> obstacles = this->getRobotNearbyObstacles(robot, start, goal);
    this->purgeNonFiniteObstacles(obstacles, robot->name);

    // ⑥ 热启动 seed + 直线初值重建
    auto & seed = this->last_traj_by_robot_[robot->name];
    this->sanitizeSeed(robot->name, goal, seed);
    if (seed.size() < 2) {
      const bool rev = this->reverse_by_robot_[robot->name];
      const double dir_th = std::atan2(goal.y - start.y, goal.x - start.x) + (rev ? M_PI : 0.0);
      constexpr int N = 12;
      for (int i = 0; i <= N; ++i) {
        const double s = static_cast<double>(i) / N;
        seed.push_back(g2o::teb::Pose2D{start.x + s * (goal.x - start.x),
                                        start.y + s * (goal.y - start.y), dir_th});
      }
    }

    // ⑦ TEB 局部规划（典型 2~5 ms）
    const double t_plan0 = secNow();
    const g2o::teb::PlanResult tebResult = this->planner_.plan(start, goal, obstacles, seed);
    plan_ms_.store((secNow() - t_plan0) * 1000.0);
    if (plan_ms_.load() > 50.0) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 3000,
        "%s TEB 规划 %0.f ms（典型 2~5 ms）｜位姿数 = %zu，障碍数 = %zu",
        robot->name.c_str(), plan_ms_.load(), tebResult.trajectory.size(),
        obstacles.size());
    }

    // ⑧ 无解 / 轨迹过短 → 显式发零（修复 G）
    if (!tebResult.ok || tebResult.trajectory.size() < 2) {
      this->publishZero(robot->name);
      return;
    }

    // ⑨ 纯跟踪 + 发布（方向航段内锁定，联调②）
    setStage(12);
    const auto & traj = tebResult.trajectory;
    this->publishPursuitCommand(*robot, traj, this->reverse_by_robot_[robot->name], false);
    this->local_path_pubs_[robot->name]->publish(
      g2o_teb::toPath(tebResult, "map", now()));

    // TEB 视角可视化（teb_poses / teb_obstacles）
    {
      geometry_msgs::msg::PoseArray pa;
      pa.header.frame_id = "map";
      pa.header.stamp = now();
      pa.poses.reserve(traj.size() + 1);
      for (const auto & p : traj) { pa.poses.push_back(g2o_teb::toPose(p)); }
      pa.poses.push_back(g2o_teb::toPose(goal));
      this->teb_pose_pubs_[robot->name]->publish(pa);
      this->teb_obstacle_pubs_[robot->name]->publish(
        g2o_teb::toObstacleMarkers(obstacles, "map", now(),
                                   this->robot_radius_, 0.30));
    }
    seed = traj;
  }

  /// @brief 纯跟踪解算并发布 cmd_vel
  /// @param robot 
  /// @param traj 
  /// @param rev 
  /// @param dir_pending 
  void publishPursuitCommand(RobotAgent & robot,
                            const std::vector<g2o::teb::Pose2D> & traj,
                            bool rev, bool dir_pending)
  {
      // ==========================================
      // 1. 动态计算预瞄距离 (Look-ahead Distance)
      // ==========================================
      // 取当前线速度和 0.05 的最大值，防止静止时速度过小导致预瞄距离过短。
      // 公式：ld = 0.5 + 1.0 * v，并限制在 [0.50, 1.30] 米之间。
      const double v_meas = std::max(robot.lin_vel, 0.05);
      const double lookahead = std::max(0.50, std::min(1.30, 0.50 + 1.00 * v_meas));
      
      // ==========================================
      // 2. 在轨迹上寻找前视点 (Look-ahead Point)
      // ==========================================
      // 默认取轨迹的最后一个点（终点）。
      double lx = traj.back().x, ly = traj.back().y;
      // 从轨迹的第 2 个点开始遍历（假设第 1 个点是车辆当前位置），
      // 找到第一个距离车辆 >= lookahead 的轨迹点，作为当前预瞄点。
      for (std::size_t i = 1; i < traj.size(); ++i) {
          if (std::hypot(traj[i].x - robot.x, traj[i].y - robot.y) >= lookahead) {
              lx = traj[i].x; ly = traj[i].y; break;
          }
      }

      // ==========================================
      // 3. 计算方位角 alpha (含倒车坐标系翻转)
      // ==========================================
      // 如果是倒车(rev=true)，虚拟车头朝向翻转 180 度，这样预瞄点就会在"车头"前方。
      const double th_ref = rev ? (robot.theta + M_PI) : robot.theta;
      
      // 计算预瞄点相对于车辆当前朝向的角度差 (alpha)。
      // 使用 std::remainder 将角度规整到 [-pi, pi] 之间，防止跳变。
      double alpha = std::atan2(ly - robot.y, lx - robot.x) - th_ref;
      alpha = std::remainder(alpha, 2.0 * M_PI);

      // 角度低通滤波 (EMA: 指数移动平均)。
      // 防止 TEB 轨迹突变或定位跳变导致前轮疯狂打死（画龙）。权重 50:50 平滑。
      PursuitState & ps = this->pursuit_by_robot_[robot.name];
      alpha = ps.alpha_init ? (0.50 * ps.alpha_ema + 0.50 * alpha)
                            : (ps.alpha_init = true, alpha);
      ps.alpha_ema = alpha;

      // ==========================================
      // 4. 计算纯跟踪前轮转角 (Pure Pursuit Steering)
      // ==========================================
      // 核心公式: delta = atan(2 * L * sin(alpha) / ld)
      const double delta_pp = std::atan2(2.0 * this->wheelbase_ * std::sin(alpha), lookahead);

      // ==========================================
      // 5. 速度规划 (含各种减速策略)
      // ==========================================
      StuckGuard & sg = this->stuck_by_robot_[robot.name];
      const double t_now = this->now().seconds();

      // 倒车限速 0.30，前进限速 0.60。
      const double v_max = rev ? 0.30 : 0.60; 
      // 转角越大，速度越慢（最大降速 65%），保证转弯时的稳定性。
      double v_mag = v_max * (1.0 - 0.65 * std::min(1.0, std::fabs(alpha) / (M_PI / 2.0))); 
      
      // 接近终点时平滑减速（距离小于 0.8 米时减速，最低降至原速度的 15%）。
      if (robot.wp_idx + 1 >= robot.waypoints.size()) {
          const double remain = std::hypot(robot.goal_x - robot.x, robot.goal_y - robot.y);
          if (remain < 0.80) { v_mag *= std::max(0.15, remain / 0.80); }
      }
      
      // 陷入卡死/脱困状态时，限制最大速度为 0.20，蠕行寻找出路。
      if (sg.attempts > 0) { v_mag = std::min(v_mag, 0.20); }  
      
      // 组合最终线速度，并处理方向切换指令。
      double v_cmd = rev ? -v_mag : v_mag; 
      // 如果方向待定（例如正在换挡间隙），强制停车。
      if (dir_pending) { v_cmd = 0.0; }

      // ==========================================
      // 6. 脱困盲打逻辑 (Stuck Recovery)
      // ==========================================
      // 如果卡死超过一定次数，触发预设的"三阶段摆尾"盲打。
      if (sg.phase >= 1) {
          const double dur = (sg.phase == 1) ? 2.5 : 2.0; // 各阶段持续时间
          if (t_now - sg.t0 < dur) {
              // 直接发布预设的脱困速度 (sg.rec_v) 和转角 (sg.steer)，忽略正常计算。
              geometry_msgs::msg::Twist msg;
              msg.linear.x  = sg.rec_v;
              msg.angular.z = sg.rec_v / this->wheelbase_ * std::tan(sg.steer);
              this->goal_pubs_[robot.name]->publish(msg);
              return;
          }
          sg.phase = 0; // 阶段结束，重置
          ++sg.attempts;
          sg.ts = t_now;
      } else {
          sg.ts = 0.0;
      }

      // ==========================================
      // 7. 转角修偏与平滑 (阻尼控制)
      // ==========================================
      // 将期望转角转化为期望角速度, 阿克曼（自行车）运动学模型
      const double omega_des = v_cmd / this->wheelbase_ * std::tan(delta_pp);
      const double k_dd = 0.20; // 阻尼系数,抑制震荡,防止'画龙'
      const double d_lim = 0.10; // 阻尼修正上限 0.1 弧度
      
      // 根据当前实际角速度(robot.ang_vel)与期望角速度的误差，对转角进行微调。
      // (omega_des - robot.ang_vel) 期望转角与实际转角的误差
      const double d_corr = std::max(-d_lim, std::min(d_lim,
                              k_dd * (omega_des - robot.ang_vel)));
      // 根据误差增大或减小转角
      double delta = std::max(-0.6, std::min(0.6, delta_pp + d_corr)); // 限制最大转角 +-0.6 rad
      
      // 关键工程保护：限制前轮转角单次变化率 (kDeltaMaxStep = 0.125 rad)。
      // 防止相邻控制周期转角突变导致底盘机械结构损坏或车辆打滑。
      constexpr double kDeltaMaxStep = 0.125;
      // 限制转角幅度
      delta = std::max(ps.delta_prev - kDeltaMaxStep,
                      std::min(ps.delta_prev + kDeltaMaxStep, delta));
      ps.delta_prev = delta; // 更新历史转角

      // ==========================================
      // 8. 计算最终角速度，并做 NaN 净化保护
      // ==========================================
      double omega_cmd = v_cmd / this->wheelbase_ * std::tan(delta);
      
      // 如果因为除零或数学异常导致输出了 NaN/Inf，强制刹车保护底盘。
      if (!std::isfinite(v_cmd) || !std::isfinite(omega_cmd)) {
          RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 2000,
              "%s 解出非有限指令（v=%.3f ω=%.3f）→ 发 0 保护", robot.name.c_str(),
              v_cmd, omega_cmd);
          v_cmd = 0.0;
          omega_cmd = 0.0;
      }

      // ==========================================
      // 9. 发布控制指令
      // ==========================================
      geometry_msgs::msg::Twist msg;
      msg.linear.x = v_cmd;
      msg.angular.z = omega_cmd;
      this->goal_pubs_[robot.name]->publish(msg);
  }

  // ===================== 可视化发布 =====================
  void pubGlobalPlan(const std::shared_ptr<RobotAgent> & robot)
  {
    if (robot->waypoints.empty()) { return; }
    nav_msgs::msg::Path path;
    path.header.frame_id = "map";
    path.header.stamp = now();
    path.poses.reserve(robot->waypoints.size());
    for (const auto& p : robot->waypoints) {
      geometry_msgs::msg::PoseStamped ps;
      ps.header = path.header;
      ps.pose.position.x = p.first;
      ps.pose.position.y = p.second;
      ps.pose.position.z = 0.;
      ps.pose.orientation.w = 1;      // 单位四元数（z=0,w=0 会被 RViz 丢弃）
      path.poses.push_back(ps);
    }
    this->global_path_pubs_[robot->name]->publish(path);
  }

  void publishGlobalMap()
  {
    if (!map_) { return; }
    nav_msgs::msg::OccupancyGrid msg;
    msg.header.frame_id = "map";
    msg.header.stamp = this->now();
    msg.info.resolution = map_->res();
    msg.info.width = static_cast<std::uint32_t>(map_->nx());
    msg.info.height = static_cast<std::uint32_t>(map_->ny());
    msg.info.origin.position.x = map_->originX();
    msg.info.origin.position.y = map_->originY();
    msg.info.origin.orientation.w = 1.0;
    msg.data.resize(static_cast<std::size_t>(map_->nx()) * static_cast<std::size_t>(map_->ny()));
    for (int y = 0; y < map_->ny(); ++y) {
      for (int x = 0; x < map_->nx(); ++x) {
        const Cell c{x, y};
        msg.data[static_cast<std::size_t>(y) * map_->nx() + x] = map_->isFree(c) ? 0 : 100;
      }
    }
    global_map_pub_->publish(msg);
  }

  // ===================== Action 反馈 + 状态查询话题（2Hz） =====================
  void publishFeedbackAndStatus()
  {
    fleet_common::msg::FleetNavStatus fns;
    for (const auto & robot : robots_) {
      fleet_common::msg::NavStatus ns;
      ns.robot_name = robot->name;
      NavJob * job = jobByRobot(robot);
      if (job && job->gh) {
        const double elapsed = secNow() - job->t0;
        const double dist = robot->distanceTo(robot->goal_x, robot->goal_y);
        ns.has_goal = true;
        ns.state = job->state;
        ns.goal_x = robot->goal_x;
        ns.goal_y = robot->goal_y;
        ns.distance_remaining = dist;
        ns.elapsed_sec = elapsed;
        ns.timeout_sec = job->timeout;
        ns.message = job->message;

        auto fb = std::make_shared<NavigateToPose::Feedback>();
        fb->state = job->state;
        fb->distance_remaining = dist;
        fb->elapsed_sec = elapsed;
        fb->message = job->message;
        try {
          job->gh->publish_feedback(fb);
        } catch (const std::exception &) {
          // 客户端掉线等：反馈失败不影响导航本身
        }
      } else {
        ns.has_goal = false;
      }
      fns.robots.push_back(ns);
    }
    nav_status_pub_->publish(fns);
  }

  // ===================== 成员 =====================
  std::unique_ptr<WarehouseMap> map_;
  std::unique_ptr<WarehouseMap> raw_map_;
  std::unique_ptr<TrafficControl> traffic_;

  g2o::teb::TebPlanner planner_;
  std::unordered_map<std::string, std::vector<g2o::teb::Pose2D>> last_traj_by_robot_;
  std::unordered_map<std::string, bool> reverse_by_robot_;
  std::unordered_map<std::string, double> rev_cand_ts_;
  std::unordered_map<std::string, double> rev_proj_ema_;

  struct StuckGuard
  {
    int phase = 0;
    double t0 = 0.0;
    double ts = 0.0;
    double rec_v = 0.0;
    int attempts = 0;
    double steer = 0.0;
  };
  std::unordered_map<std::string, StuckGuard> stuck_by_robot_;

  struct StuckWatch
  {
    double t0 = 0.0;
    double x = 0.0, y = 0.0;
    int yield_winner = -1;
    double yield_backed = 0.0;
    double yield_target = 0.0;
    int yield_vsign = -1;
    int last_winner = -1;
    int yield_count = 0;
  };
  std::unordered_map<std::string, StuckWatch> watch_by_robot_;

  struct PursuitState
  {
    bool alpha_init = false;
    double alpha_ema = 0.0;
    double delta_prev = 0.0;
  };
  std::unordered_map<std::string, PursuitState> pursuit_by_robot_;

  std::vector<std::string> robot_names_;
  std::vector<std::shared_ptr<RobotAgent>> robots_;
  std::unordered_map<std::string, std::shared_ptr<RobotAgent>> name_to_robot_;

  double v_max_, w_max_, goal_tol_, wp_tol_, turn_stop_factor_;
  double max_steer_, wheelbase_, min_turn_speed_, min_turn_radius_;
  double robot_radius_, stop_dist_, stuck_timeout_, yield_backup_;
  double stuck_window_, stuck_disp_, yield_scope_, yield_speed_, yield_timeout_, yield_clear_dist_;
  double status_period_, marker_period_, control_rate_;
  double plan_retry_interval_;
  int plan_retry_max_;
  std::vector<double> charger_x_, charger_y_;

  std::thread watchdog_;
  std::atomic<bool> watchdog_run_{true};
  std::atomic<int> stage_{0};
  std::atomic<double> tick_begin_{0.0};
  std::atomic<double> stage_t0_{0.0};
  std::atomic<double> plan_ms_{0.0};
  void setStage(int s) { stage_.store(s); stage_t0_.store(secNow()); }

  double sim_time_ = 0.0;
  double last_control_ = 0.0;

  std::unordered_map<std::string, rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr> goal_pubs_;
  std::unordered_map<std::string, rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr> global_path_pubs_;
  std::unordered_map<std::string, rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr> local_path_pubs_;
  std::unordered_map<std::string, rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr> teb_pose_pubs_;
  std::unordered_map<std::string, rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr> teb_obstacle_pubs_;
  rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr global_map_pub_;
  rclcpp::Publisher<fleet_common::msg::FleetNavStatus>::SharedPtr nav_status_pub_;
  std::uint32_t map_pub_cnt_ = 0;

  rclcpp::Subscription<gazebo_msgs::msg::ModelStates>::SharedPtr model_states_sub_;
  rclcpp::Client<gazebo_msgs::srv::SetEntityState>::SharedPtr tp_client_;
  std::unordered_map<std::string, int> pen_cnt_;
  std::unordered_map<std::string, double> last_tp_;
  std::unordered_map<std::string, rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr> vec_subs_;

  rclcpp_action::Server<NavigateToPose>::SharedPtr action_server_;
  rclcpp::TimerBase::SharedPtr control_timer_, status_timer_, marker_timer_;

  double secNow() const { return rclcpp::Clock(RCL_STEADY_TIME).now().seconds(); }
};

}  // namespace fleet_manager

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<fleet_manager::FleetNavServerNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
