// Copyright 2026
// =====================================================================
// FleetScheduler — 纯调度节点（v2 调度/导航分离 · 2026-10-19）
// ---------------------------------------------------------------------
// 从 fleet_manager_node_teb.cpp 剥离导航层后的任务引擎：
//   任务管理（/fleet/task + add_task 服务）· RL/贪心任务分配 ·
//   停靠点吸附 snapToStop（互斥 + 沿通道错开）· 电量/充电管理 ·
//   任务状态机 · /fleet/status + /fleet/markers · pause/resume
//
// 调度器与导航的唯一通道 = NavigateToPose action（/fleet/navigate_to_pose）：
//   - 状态机每进入一个导航阶段，只发一次目标点（robot_name/x/y/timeout/priority）
//   - 到达 / 失败 / 超时 / 取消 由 action 结果回调驱动状态机推进
//   - 剩余距离（RL 的 ETA 观测）从 /fleet/nav_status 订阅（服务器 2Hz 广播）
//   - 电量告急 → async_cancel_request + 任务重新排队 + 转充电
//
// 客户端健壮性（★ 多车 action 的关键工程细节）：
//   - goal 代次 gen：每次发新目标 +1；accepted/result/feedback 回调先比对代次，
//     陈旧回调直接丢弃（否则"取消后收到旧结果"会串扰新目标的推进）
//   - 服务器未就绪：目标保留（has_new_goal 不清），下拍自动重试
//   - 服务器拒绝（车忙/未知车）：视为该次导航失败 → 任务重新排队
// =====================================================================
#include "fleet_manager/astar_planner.hpp"
#include "fleet_manager/robot_agent.hpp"
#include "fleet_manager/rl_policy.hpp"
#include "fleet_manager/warehouse_map.hpp"

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <gazebo_msgs/msg/model_states.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <fleet_common/msg/fleet_status.hpp>
#include <fleet_common/msg/robot_state.hpp>
#include <fleet_common/msg/task.hpp>
#include <fleet_common/msg/fleet_nav_status.hpp>
#include <fleet_common/srv/add_task.hpp>
#include <fleet_common/action/navigate_to_pose.hpp>
#include <std_srvs/srv/trigger.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <list>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
using namespace std::chrono_literals;

namespace fleet_manager
{

inline double quatToYaw(double qx, double qy, double qz, double qw)
{
  (void)qx; (void)qy;
  return std::atan2(2.0 * (qw * qz), 1.0 - 2.0 * qz * qz);
}

class FleetSchedulerNode : public rclcpp::Node
{
public:
  using NavigateToPose = fleet_common::action::NavigateToPose;
  using GoalHandleNav = rclcpp_action::ClientGoalHandle<NavigateToPose>;

  /// @brief 构造函数
  FleetSchedulerNode() : Node("fleet_manager")
  {
    declareParams();
    loadParams();
    initRLPolicy();
    buildValidationMap();
    initRobots();
    initChargers();
    initPublishers();
    initSubscribers();
    initServices();
    initActionClient();

    control_timer_ = create_wall_timer(
      std::chrono::duration<double>(1.0 / control_rate_), [this]() { controlTick(); });
    status_timer_ = create_wall_timer(
      std::chrono::duration<double>(status_period_), [this]() { publishStatus(); });
    marker_timer_ = create_wall_timer(
      std::chrono::duration<double>(marker_period_), [this]() { publishMarkers(); });

    RCLCPP_INFO(get_logger(), "FleetScheduler started: %zu robots, control %.1f Hz "
      "（导航已剥离 → /fleet/navigate_to_pose action）",
      robots_.size(), control_rate_);
  }

private:
  // ===================== 参数（任务/电量/RL 与 node_teb 同名同默认） =====================
  void declareParams()
  {
    declare_parameter("robot_names", std::vector<std::string>{"agv_0", "agv_1", "agv_2"});
    declare_parameter("dwell_time_pickup", 3.0);
    declare_parameter("dwell_time_dropoff", 3.0);
    declare_parameter("allocator_interval", 0.5);
    declare_parameter("battery_drain_move", 0.05);
    declare_parameter("battery_drain_idle", 0.005);
    declare_parameter("battery_charge_rate", 1.0);
    declare_parameter("charge_threshold", 25.0);
    declare_parameter("charge_full", 95.0);
    declare_parameter("battery_critical", 12.0);
    declare_parameter("charger_x", std::vector<double>{-11.0, 11.0, 11.0, -11.0});
    declare_parameter("charger_y", std::vector<double>{3.6, 3.6, -3.6, -3.6});
    declare_parameter("status_period", 0.5);
    declare_parameter("marker_period", 0.5);
    declare_parameter("control_rate", 20.0);
    declare_parameter("use_rl", true);
    declare_parameter("policy_path", std::string("rl_training/policy_v5.pt"));
    // ★ 新增：导航 action 超时下发（服务器强制执行）
    declare_parameter("nav_timeout", 180.0);        // 任务腿（取货/卸货）[s]
    declare_parameter("nav_timeout_charge", 240.0); // 充电腿 [s]
  }

  void loadParams()
  {
    robot_names_ = get_parameter("robot_names").as_string_array();
    dwell_pickup_ = get_parameter("dwell_time_pickup").as_double();
    dwell_dropoff_ = get_parameter("dwell_time_dropoff").as_double();
    alloc_interval_ = get_parameter("allocator_interval").as_double();
    drain_move_ = get_parameter("battery_drain_move").as_double();
    drain_idle_ = get_parameter("battery_drain_idle").as_double();
    charge_rate_ = get_parameter("battery_charge_rate").as_double();
    charge_threshold_ = get_parameter("charge_threshold").as_double();
    charge_full_ = get_parameter("charge_full").as_double();
    battery_critical_ = get_parameter("battery_critical").as_double();
    charger_x_ = get_parameter("charger_x").as_double_array();
    charger_y_ = get_parameter("charger_y").as_double_array();
    status_period_ = get_parameter("status_period").as_double();
    marker_period_ = get_parameter("marker_period").as_double();
    control_rate_ = get_parameter("control_rate").as_double();
    use_rl_ = get_parameter("use_rl").as_bool();
    policy_path_ = get_parameter("policy_path").as_string();
    nav_timeout_ = get_parameter("nav_timeout").as_double();
    nav_timeout_charge_ = get_parameter("nav_timeout_charge").as_double();
    v_max_ = 0.5;   // ETA 折算用（与 fleet_manager.yaml 的 max_linear_speed 一致）
  }

  // ===================== RL 推理（迁移自 node_teb，逐字保留） =====================
  void initRLPolicy()
  {
    if (!use_rl_) { return; }
    std::string path = policy_path_;
    if (path.empty()) {
      path = "rl_training/policy_v8.pt";
    }
    try {
      rl_policy_ = std::make_shared<RLPolicy>(path, static_cast<int>(robot_names_.size()));
      RCLCPP_INFO(get_logger(), "RL 策略已加载：%s", path.c_str());
    } catch (const std::exception & e) {
      RCLCPP_ERROR(get_logger(), "RL 策略加载失败，回退贪心分配：%s", e.what());
      use_rl_ = false;
      rl_policy_.reset();
    }
  }

  /// @brief 站点编码（24 站点最近匹配，顺序与 rl_policy 同源）
  /// @param x 
  /// @param y 
  /// @param is_pickup 
  /// @return 
  static int siteCode(double x, double y, bool is_pickup)
  {
    (void)is_pickup;
    static const double S[24][2] = {
      {-10.4, 0.0}, {10.4, 0.0}, {0.0, 6.8}, {0.0, -6.8},
      {-10.4, 6.8}, {10.4, 6.8}, {-10.4, -6.8}, {10.4, -6.8},
      {-6.0, 6.8}, {6.0, 6.8}, {-6.0, -6.8}, {6.0, -6.8},
      {-7.0, 3.6}, {-4.2, 3.6}, {-1.4, 3.6},
      {1.4, 3.6}, {4.2, 3.6}, {7.0, 3.6},
      {-7.0, -3.6}, {-4.2, -3.6}, {-1.4, -3.6},
      {1.4, -3.6}, {4.2, -3.6}, {7.0, -3.6}};
    int best = 0;
    double bd = 1e18;
    for (int k = 0; k < 24; ++k) {
      const double d = std::hypot(x - S[k][0], y - S[k][1]);
      if (d < bd) { bd = d; best = k; }
    }
    return best;
  }

  /// @brief ETA（RL obs 用）：导航中的车用 /fleet/nav_status 的剩余距离折算
  /// @param robot 
  /// @return 
  double etaFor(const std::shared_ptr<RobotAgent> & robot) const
  {
    switch (robot->state) {
      case RobotState::IDLE: return 0.0;
      case RobotState::CHARGING: return (100.0 - robot->battery);
      case RobotState::LOADING:
        return std::max(0.0, dwell_pickup_ - robot->state_time) + dwell_dropoff_;
      case RobotState::UNLOADING:
        return std::max(0.0, dwell_dropoff_ - robot->state_time);
      case RobotState::NAV_PICKUP:
        return navBook(robot).last_dist / v_max_ + dwell_pickup_ + dwell_dropoff_;
      case RobotState::NAV_DROPOFF:
        return navBook(robot).last_dist / v_max_ + dwell_dropoff_;
      default: return 0.0;   // OFFLINE/TO_CHARGER/YIELD/FAULT
    }
  }

  /// @brief 目标位置编码（0=无 · 1..16=站点 · 17=充电）
  /// @param robot 
  /// @return 
  int targetCodeFor(const std::shared_ptr<RobotAgent> & robot) const
  {
    if (robot->state == RobotState::CHARGING || robot->state == RobotState::TO_CHARGER) {
      return 17;
    }
    if (!robot->task) { return 0; }
    if (robot->state == RobotState::NAV_PICKUP || robot->state == RobotState::LOADING) {
      return 1 + siteCode(robot->task->pickup_x, robot->task->pickup_y, true);
    }
    return 1 + siteCode(robot->task->dropoff_x, robot->task->dropoff_y, false);
  }

  /// @brief 任务描述排序，根据优先级从小到大排序，同优先级id小的在前
  /// @return 
  std::vector<std::shared_ptr<TaskSpec>> sortedPending() const
  {
    std::vector<std::shared_ptr<TaskSpec>> pend;
    for (const auto & t : tasks_) {
      if (!t->active && !t->done) { pend.push_back(t); }
    }
    std::sort(pend.begin(), pend.end(),
      [](const auto & a, const auto & b) {
        if (a->priority != b->priority) { return a->priority < b->priority; }
        return a->id < b->id;
      });
    return pend;
  }

  // ===================== 校验用地图（仅 addTask 合法性检查） =====================
  
  /// @brief 与 fleet_nav_server 的 buildMap 同源常量；改 world 布局时,两处必须同步改
  void buildValidationMap()
  {
    const double ox = -12.0, oy = -8.0;
    auto buildRects = [&](WarehouseMap & m, double inflate) {
      m.addRectObstacle(-12.0, -8.0, -11.6, 8.0, inflate);
      m.addRectObstacle(11.6, -8.0, 12.0, 8.0, inflate);
      m.addRectObstacle(-12.0, -8.0, 12.0, -7.6, inflate);
      m.addRectObstacle(-12.0, 7.6, 12.0, 8.0, inflate);
      const double rows[4] = {5.2, 2.0, -2.0, -5.2};
      const double cols[6] = {-7.0, -4.2, -1.4, 1.4, 4.2, 7.0};
      const double rack_half_w = 0.30;
      for (double ry : rows) {
        for (int ci = 0; ci < 6; ++ci) {
          const double cx = cols[ci];
          const double half_l = (ci == 0 || ci == 5) ? 0.5 : 0.8;
          m.addRectObstacle(cx - rack_half_w, ry - half_l,
                            cx + rack_half_w, ry + half_l, inflate);
        }
      }
      m.addRectObstacle(4.6, 4.4, 6.65, 6.0, inflate);       // 控制室
      const double goods[8][2] = {
        {-11.25, 0.0}, {11.25, 0.0}, {0.85, 6.8}, {-0.85, -6.8},
        {-11.25, 6.8}, {11.25, 6.8}, {-11.25, -6.8}, {11.25, -6.8}};
      for (const auto & g : goods) {
        m.addRectObstacle(g[0] - 0.2, g[1] - 0.2, g[0] + 0.2, g[1] + 0.2, inflate);
      }
      m.addRectObstacle(-0.2, 7.3, 0.2, 7.7, inflate);
      m.addRectObstacle(-0.2, -7.7, 0.2, -7.3, inflate);
    };
    map_ = std::make_unique<WarehouseMap>(ox, oy, 0.2, 120, 80);
    buildRects(*map_, 0.60);
    for (std::size_t i = 0; i < charger_x_.size() && i < charger_y_.size(); ++i) {
      map_->clearRect(charger_x_[i] - 0.3, charger_y_[i] - 0.5,
                      charger_x_[i] + 0.3, charger_y_[i] + 0.5);
    }
  }

  // ===================== 机器人 / 充电桩 =====================
  
  /// @brief 初始化机器人
  void initRobots()
  {
    for (std::size_t i = 0; i < robot_names_.size(); ++i) {
      auto r = std::make_shared<RobotAgent>();
      r->id = static_cast<int>(i);
      r->name = robot_names_[i];
      robots_.push_back(r);
      book_by_robot_[r->name] = NavBook{};
    }
    for (const auto & r : robots_) { name_to_robot_[r->name] = r; }
  }

  /// @brief 初始化充电站
  void initChargers()
  {
    std::size_t n = std::min(charger_x_.size(), charger_y_.size());
    for (std::size_t i = 0; i < n; ++i) {
      chargers_.push_back({charger_x_[i], charger_y_[i]});
    }
    RCLCPP_INFO(get_logger(), "%zu chargers", chargers_.size());
  }

  // ===================== ROS 接口 =====================
  void initPublishers()
  {
    status_pub_ = this->create_publisher<fleet_common::msg::FleetStatus>("/fleet/status", rclcpp::QoS(10));
    markers_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("/fleet/markers", rclcpp::QoS(10));
  }

  void initSubscribers()
  {
    model_states_sub_ = create_subscription<gazebo_msgs::msg::ModelStates>(
      "/fleet_sim/model_states", rclcpp::SensorDataQoS(),
      [this](const gazebo_msgs::msg::ModelStates::SharedPtr msg) { onModelStates(msg); });

    task_sub_ = create_subscription<fleet_common::msg::Task>(
      "/fleet/task", rclcpp::QoS(10),
      [this](const fleet_common::msg::Task::SharedPtr msg) { onTaskMsg(msg); });

    // ★ 导航状态回读（ETA 用）：fleet_nav_server 2Hz 广播
    nav_status_sub_ = create_subscription<fleet_common::msg::FleetNavStatus>(
      "/fleet/nav_status", rclcpp::QoS(10),
      [this](const fleet_common::msg::FleetNavStatus::SharedPtr msg) {
        for (const auto & ns : msg->robots) {
          auto it = name_to_robot_.find(ns.robot_name);
          if (it == name_to_robot_.end()) { continue; }
          NavBook & b = book_by_robot_[ns.robot_name];
          b.last_dist = ns.distance_remaining;
          b.nav_state = ns.state;
        }
      });
  }

  void initServices()
  {
    add_task_srv_ = create_service<fleet_common::srv::AddTask>(
      "/fleet/add_task",
      [this](const fleet_common::srv::AddTask::Request::SharedPtr req,
             fleet_common::srv::AddTask::Response::SharedPtr res) {
        res->success = addTask(req->priority, req->pickup_x, req->pickup_y,
                               req->dropoff_x, req->dropoff_y, res->task_id);
        res->message = res->success ? "ok" : "failed";
      });
      
    pause_srv_ = create_service<std_srvs::srv::Trigger>(
      "/fleet/pause",
      [this](const std::shared_ptr<std_srvs::srv::Trigger::Request>,
             std::shared_ptr<std_srvs::srv::Trigger::Response> res) {
        paused_ = true;
        res->success = true;
        res->message = "paused";
      });

    resume_srv_ = create_service<std_srvs::srv::Trigger>(
      "/fleet/resume",
      [this](const std::shared_ptr<std_srvs::srv::Trigger::Request>,
             std::shared_ptr<std_srvs::srv::Trigger::Response> res) {
        paused_ = false;
        res->success = true;
        res->message = "resumed";
      });
  }

  /// @brief 初始化action,用于发布任务并激活导航
  void initActionClient()
  {
    nav_client_ = rclcpp_action::create_client<NavigateToPose>(this, "navigate_to_pose");
  }

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
      r->lin_vel = std::hypot(msg->twist[i].linear.x, msg->twist[i].linear.y);
    }
  }

  void onTaskMsg(const fleet_common::msg::Task::SharedPtr msg)
  {
    if (!msg->is_active) { return; }
    int tid = -1;
    addTask(msg->priority, msg->pickup_x, msg->pickup_y,
            msg->dropoff_x, msg->dropoff_y, tid);
  }

  // ===================== 任务管理 =====================
  bool addTask(int priority, double px, double py, double dx, double dy, int & tid)
  {
    Cell pc = map_->worldToCell(px, py);
    Cell dc = map_->worldToCell(dx, dy);
    if (!map_->isFree(pc)) {
      RCLCPP_WARN(get_logger(), "pickup point (%.2f,%.2f) is occupied", px, py);
      return false;
    }
    if (!map_->isFree(dc)) {
      RCLCPP_WARN(get_logger(), "dropoff point (%.2f,%.2f) is occupied", dx, dy);
      return false;
    }

    auto t = std::make_shared<TaskSpec>();
    t->id = next_task_id_++;
    t->priority = std::max(1, priority);
    t->pickup_x = px; t->pickup_y = py;
    t->dropoff_x = dx; t->dropoff_y = dy;
    t->active = false; t->done = false;
    t->status = "PENDING";
    t->enter_time = secNow();
    t->deadline = 200.0;
    tasks_.push_back(t);
    tid = t->id;
    RCLCPP_INFO(get_logger(), "Task#%d added: P(%.1f,%.1f)->D(%.1f,%.1f) pri=%d",
      t->id, px, py, dx, dy, t->priority);
    if (tasks_.size() > 200) {
      for (auto it = tasks_.begin(); it != tasks_.end();) {
        if ((*it)->done) { it = tasks_.erase(it); }
        else { ++it; }
      }
    }
    return true;
  }

  // ===================== 任务分配 =====================
  
  /// @brief 分配任务，有RL策略就用RL.没有则选离取货点最近的空闲机器人
  void allocateTasks()
  {
    if (use_rl_ && rl_policy_) {
      allocateTasksRL();
      return;
    }
    auto pending = sortedPending();
    if (pending.empty()) { return; }
    for (auto & t : pending) {
      std::shared_ptr<RobotAgent> best;
      double best_d = std::numeric_limits<double>::max();
      for (auto & robot : robots_) {
        if (robot->state != RobotState::IDLE) { continue; }
        double d = robot->distanceTo(t->pickup_x, t->pickup_y);
        if (d < best_d) { best_d = d; best = robot; }
      }
      if (best) {
        assignTaskToRobot(t, best);
      }
    }
  }

  /// @brief RL分配任务
  void allocateTasksRL()
  {
    auto pend = sortedPending();
    if (pend.empty()) { return; }

    const double now = secNow();
    std::vector<RLCarState> cars;
    cars.reserve(robots_.size());
    for (const auto & robot : robots_) {
      RLCarState c;
      c.idle = (robot->state == RobotState::IDLE);
      c.charge = robot->battery;
      c.eta = etaFor(robot);
      c.target_code = targetCodeFor(robot);
      cars.push_back(c);
    }

    std::vector<RLPendingTask> rl_pend;
    rl_pend.reserve(pend.size());
    for (const auto & task : pend) {
      RLPendingTask pendingTask;
      pendingTask.priority = task->priority;
      pendingTask.wait = std::max(0.0, now - task->enter_time);
      pendingTask.deadline = task->deadline;
      pendingTask.src_code = siteCode(task->pickup_x, task->pickup_y, true);
      pendingTask.dst_code = siteCode(task->dropoff_x, task->dropoff_y, false);
      rl_pend.push_back(pendingTask);
    }

    std::vector<int> actions = rl_policy_->allocate(cars, rl_pend, sim_time_);

    for (std::size_t i = 0; i < robots_.size(); ++i) {
      const int a = actions[i];
      if (a <= 0 || a > static_cast<int>(pend.size())) { continue; }
      auto & robot = robots_[i];
      if (robot->state != RobotState::IDLE) { continue; }
      auto task = pend[static_cast<std::size_t>(a - 1)];
      if (task->active) { continue; }
      assignTaskToRobot(task, robot);
    }
  }

  /// @brief 接单：置目标点与状态（发送在 updateRobot 的对应阶段完成, 登记 has_new_goal）
  /// @param t 任务
  /// @param robot 机器人
  void assignTaskToRobot(const std::shared_ptr<TaskSpec> & t,
                         const std::shared_ptr<RobotAgent> & robot)
  {
    robot->task = t;
    t->active = true;
    t->status = "ACTIVE";
    double gx = t->pickup_x, gy = t->pickup_y;
    snapToStop(gx, gy, robot);
    robot->goal_x = gx;
    robot->goal_y = gy;
    robot->has_new_goal = true;
    robot->resume_state = RobotState::NAV_PICKUP;
    robot->state = RobotState::NAV_PICKUP;
    robot->state_time = 0.0;
    robot->blocked_time = 0.0;
    RCLCPP_INFO(get_logger(), "Task#%d assigned to %s", t->id, robot->name.c_str());
  }

  // ===================== 站点停靠点映射（迁移自 node_teb，逐字保留） =====================
  void snapToStop(double & x, double & y,
                  const std::shared_ptr<RobotAgent> & self = nullptr)
  {
    constexpr double stations[12][2] = {
      {-10.4, 0.0}, {10.4, 0.0}, {0.0, 6.8}, {0.0, -6.8},
      {-10.4, 6.8}, {10.4, 6.8}, {-10.4, -6.8}, {10.4, -6.8},
      {-6.0, 6.8}, {6.0, 6.8}, {-6.0, -6.8}, {6.0, -6.8}};
    constexpr double stops[12][2] = {
      {-10.4, 0.0}, {10.4, 0.0}, {0.0, 6.8}, {0.0, -6.8},
      {-10.4, 6.8}, {10.4, 6.8}, {-10.4, -6.8}, {10.4, -6.8},
      {-6.0, 6.8}, {6.0, 6.8}, {-6.0, -6.8}, {6.0, -6.8}};
    constexpr double off_dir[12][2] = {
      {0.0, +1.0}, {0.0, +1.0},
      {+1.0, 0.0}, {+1.0, 0.0},
      {0.0, -1.0}, {0.0, -1.0},
      {0.0, +1.0}, {0.0, +1.0},
      {+1.0, 0.0}, {-1.0, 0.0},
      {+1.0, 0.0}, {-1.0, 0.0}};
    constexpr double SNAP_R  = 1.5;
    constexpr double STEP    = 0.70;
    constexpr double KEEP_R  = 0.60;
    constexpr int    MAX_OFF = 3;

    for (int i = 0; i < 12; ++i) {
      if (std::hypot(x - stations[i][0], y - stations[i][1]) >= SNAP_R) { continue; }
      for (int k = 0; k <= MAX_OFF; ++k) {
        const double cx = stops[i][0] + off_dir[i][0] * STEP * k;
        const double cy = stops[i][1] + off_dir[i][1] * STEP * k;
        if (!stopTaken(cx, cy, KEEP_R, self)) {
          if (k > 0) {
            RCLCPP_INFO(get_logger(), "停靠点%d 已被占用 → %s 错开 %d 格到 (%.2f, %.2f)",
                        i, self ? self->name.c_str() : "?", k, cx, cy);
          }
          x = cx;
          y = cy;
          return;
        }
      }
      RCLCPP_WARN(get_logger(), "停靠点%d 的 %d 个错开位均已占用，%s 回落首选位 (%.2f, %.2f)",
                  i, MAX_OFF, self ? self->name.c_str() : "?", stops[i][0], stops[i][1]);
      x = stops[i][0];
      y = stops[i][1];
      return;
    }

    constexpr double bays[12][2] = {
      {-7.0, 3.6}, {-4.2, 3.6}, {-1.4, 3.6}, {1.4, 3.6}, {4.2, 3.6}, {7.0, 3.6},
      {-7.0, -3.6}, {-4.2, -3.6}, {-1.4, -3.6}, {1.4, -3.6}, {4.2, -3.6}, {7.0, -3.6}};
    for (int bi = 0; bi < 12; ++bi) {
      if (std::hypot(x - bays[bi][0], y - bays[bi][1]) >= SNAP_R) { continue; }
      for (int k = 0; k <= MAX_OFF; ++k) {
        const double cx = bays[bi][0] + STEP * k;
        const double cy = bays[bi][1];
        if (!stopTaken(cx, cy, KEEP_R, self)) {
          if (k > 0) {
            RCLCPP_INFO(get_logger(), "货架位%d 已被占用 → %s 错开 %d 格到 (%.2f, %.2f)",
                        bi, self ? self->name.c_str() : "?", k, cx, cy);
          }
          x = cx;
          y = cy;
          return;
        }
      }
      RCLCPP_WARN(get_logger(), "货架位%d 的 %d 个错开位均已占用，%s 回落 (%.2f, %.2f)",
                  bi, MAX_OFF, self ? self->name.c_str() : "?", bays[bi][0], bays[bi][1]);
      x = bays[bi][0];
      y = bays[bi][1];
      return;
    }
  }

  bool stopTaken(double x, double y, double r,
                 const std::shared_ptr<RobotAgent> & self) const
  {
    const auto own = self ? self->task : nullptr;
    for (const auto & other : robots_) {
      if (other == self) { continue; }
      if (!other->task || other->task == own) { continue; }
      if (std::hypot(x - other->goal_x, y - other->goal_y) < r) { return true; }
      if (std::hypot(x - other->x, y - other->y) < r) { return true; }
    }
    for (const auto & t : tasks_) {
      if (!t->active || t->done || t == own) { continue; }
      if (std::hypot(x - t->pickup_x, y - t->pickup_y) < r) { return true; }
      if (std::hypot(x - t->dropoff_x, y - t->dropoff_y) < r) { return true; }
    }
    return false;
  }

  // ===================== 导航 Action 客户端 =====================
  struct NavBook
  {
    typename GoalHandleNav::SharedPtr gh;
    int gen = 0;               // goal 代次（陈旧回调过滤）
    bool in_flight = false;    // 已发送且未出结果
    bool finished = false;     // 结果已到（待状态机消费）
    uint8_t result_status = 255;
    std::string result_msg;
    double last_dist = 0.0;    // '/fleet/nav_status' 回读（ETA 用）
    uint8_t nav_state = 0;     // 服务器内部阶段（同 action Feedback 常量）
    double send_t0 = 0.0;      // 发送时刻（服务器无响应看护用）
  };
  std::unordered_map<std::string, NavBook> book_by_robot_;

  NavBook & navBook(const std::shared_ptr<RobotAgent> & r) const
  {
    return const_cast<NavBook &>(book_by_robot_.at(r->name));
  }

  /// @brief 发送导航目标（只发一次；服务器未就绪 → 保留 has_new_goal 下拍重试）
  /// @param robot 
  /// @param timeout 
  /// @param priority 
  /// @return 
  bool sendNavGoal(const std::shared_ptr<RobotAgent> & robot, double timeout, int priority)
  {
    if (!nav_client_->action_server_is_ready()) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
        "%s: 导航服务器未就绪，目标暂缓下发并自动重试", robot->name.c_str());
      return false;
    }
    auto goal = NavigateToPose::Goal();
    goal.robot_name = robot->name;
    goal.x = robot->goal_x;
    goal.y = robot->goal_y;
    goal.timeout_sec = timeout;
    goal.priority = priority;

    NavBook & b = book_by_robot_[robot->name];
    const int mygen = ++b.gen;          // 新代次：旧回调全部作废
    b.in_flight = true;
    b.finished = false;
    b.result_status = 255;
    b.result_msg.clear();
    b.last_dist = robot->distanceTo(robot->goal_x, robot->goal_y);
    b.send_t0 = secNow();

    rclcpp_action::Client<NavigateToPose>::SendGoalOptions opts;
    // goal_response_callback 只收 GoalHandle（nullptr = 被拒绝）
    opts.goal_response_callback =
      [this, robot, mygen](typename GoalHandleNav::SharedPtr goalHandle) {
        NavBook & b = book_by_robot_[robot->name];
        if (mygen != b.gen) { return; }             // 陈旧回调
        if (!goalHandle) {
          b.in_flight = false;
          b.finished = true;
          b.result_status = NavigateToPose::Result::STATUS_ABORTED;
          b.result_msg = "服务器拒绝（车忙/未知车名）";
          RCLCPP_WARN(get_logger(), "[%s] 导航 goal 被拒绝（%s）",
            robot->name.c_str(), b.result_msg.c_str());
        } else {
          b.gh = goalHandle;                                 // 供取消请求用
        }
      };
    opts.feedback_callback =
      [this, robot, mygen](typename GoalHandleNav::SharedPtr,
                           std::shared_ptr<const NavigateToPose::Feedback> fb) {
        NavBook & b = book_by_robot_[robot->name];
        if (mygen != b.gen) { return; }
        b.last_dist = fb->distance_remaining;
        b.nav_state = fb->state;
      };
    opts.result_callback =
      [this, robot, mygen](const typename GoalHandleNav::WrappedResult & wr) {
        NavBook & b = book_by_robot_[robot->name];
        if (mygen != b.gen) { return; }             // 陈旧回调（已被取消/替换）
        b.in_flight = false;
        b.finished = true;
        b.result_status = wr.result->status;
        b.result_msg = wr.result->message;
        RCLCPP_INFO(get_logger(), "[%s] 导航终态 status=%d：%s",
          robot->name.c_str(), static_cast<int>(b.result_status),
          b.result_msg.c_str());
      };

    nav_client_->async_send_goal(goal, opts);
    return true;
  }

  /// 取消当前导航（电量中断等）。代次 +1 使旧回调作废；
  /// 取消结果由服务器侧收尾，状态机以本地"已取消"为准直接推进。
  void cancelNavGoal(const std::shared_ptr<RobotAgent> & robot)
  {
    NavBook & b = book_by_robot_[robot->name];
    if (b.gh && b.in_flight && !b.finished) {
      RCLCPP_WARN(get_logger(), "[%s] 取消进行中的导航目标", robot->name.c_str());
      try {
        // Humble API：async_cancel_goal。goal 已终态时会抛 UnknownGoalHandleError，
        // 此时结果回调已带终态，忽略即可。
        nav_client_->async_cancel_goal(b.gh);
      } catch (const std::exception & e) {
        RCLCPP_WARN(get_logger(), "[%s] 取消请求未受理（%s）——目标可能已终态",
          robot->name.c_str(), e.what());
      }
    }
    ++b.gen;
    b.in_flight = false;
    b.finished = false;
  }

  // ===================== 主控制周期 =====================
  
  /// @brief 主控制循环，分配任务、更新电池、更新机器人状态
  void controlTick()
  {
    double now = secNow();
    double dt = std::min(now - last_control_, 0.1);
    last_control_ = now;
    if (dt <= 0.0) { return; }
    sim_time_ += dt;

    if (!paused_) {
      alloc_timer_ += dt;
      if (alloc_timer_ >= alloc_interval_) {
        alloc_timer_ = 0.0;
        allocateTasks();
      }
    }

    for (auto & robot : robots_) {
      updateBattery(robot, dt);
      updateRobot(robot, dt);
    }
  }

  /// @brief 任务状态机（导航结果驱动）
  /// @param robot 
  /// @param dt 
  void updateRobot(const std::shared_ptr<RobotAgent> & robot, double dt)
  {
    robot->state_time += dt;
    NavBook & b = book_by_robot_[robot->name];

    // ★ 服务器无响应看护：goal 发出后"服务器超时上限 + 30s"仍无终态结果
    //   （action 服务器崩溃/DDS 断链等极端情况）→ 本地判失败，任务重新排队。
    //   正常超时由服务器强制执行并回 TIMEOUT 结果，这里只是最后的保险丝。
    if (b.in_flight && !b.finished && robot->isMovingState()) {
      const double to = (robot->state == RobotState::TO_CHARGER)
                        ? nav_timeout_charge_ : nav_timeout_;
      const double budget = to + 30.0;
      if (secNow() - b.send_t0 > budget) {
        RCLCPP_ERROR(get_logger(),
          "[%s] 导航服务器 %.0fs 无终态响应 → 本地判失败（服务器崩溃？）",
          robot->name.c_str(), budget);
        b.in_flight = false;
        b.finished = true;
        b.result_status = NavigateToPose::Result::STATUS_ABORTED;
        b.result_msg = "服务器无响应（客户端看护触发）";
      }
    }

    switch (robot->state) {
      case RobotState::OFFLINE: {
        if (robot->has_pose) {
          robot->state = RobotState::IDLE;
          RCLCPP_INFO(get_logger(), "%s online at (%.2f, %.2f)",
            robot->name.c_str(), robot->x, robot->y);
        }
        break;
      }
      case RobotState::IDLE: {
        if (robot->battery <= charge_threshold_) { assignChargerAndGo(robot); }
        break;
      }
      case RobotState::NAV_PICKUP: {
        if (robot->battery <= battery_critical_) {
          cancelNavGoal(robot);
          requeueAndCharge(robot);
          break;
        }
        if (robot->has_new_goal && !b.in_flight && !b.finished) {
          if (sendNavGoal(robot, nav_timeout_, robot->priority())) {
            robot->has_new_goal = false;
          }
          break;   // 本拍先等 accepted
        }
        if (b.finished) {
          if (b.result_status == NavigateToPose::Result::STATUS_SUCCEEDED) {
            robot->state = RobotState::LOADING;
            robot->state_time = 0.0;
            b.finished = false; ++b.gen;
            RCLCPP_INFO(get_logger(), "%s arrived at pickup, loading...", robot->name.c_str());
          } else {
            RCLCPP_WARN(get_logger(), "Task#%d 取货导航失败（%s）→ 任务重新排队",
              robot->task ? robot->task->id : -1, b.result_msg.c_str());
            requeueTask(robot);
          }
        }
        break;
      }
      case RobotState::LOADING: {
        if (robot->state_time >= dwell_pickup_) {
          double gx = robot->task->dropoff_x, gy = robot->task->dropoff_y;
          snapToStop(gx, gy, robot);
          robot->goal_x = gx;
          robot->goal_y = gy;
          robot->has_new_goal = true;
          robot->resume_state = RobotState::NAV_DROPOFF;
          robot->state = RobotState::NAV_DROPOFF;
          robot->state_time = 0.0;
        }
        break;
      }
      case RobotState::NAV_DROPOFF: {
        if (robot->battery <= battery_critical_) {
          cancelNavGoal(robot);
          requeueAndCharge(robot);
          break;
        }
        if (robot->has_new_goal && !b.in_flight && !b.finished) {
          if (sendNavGoal(robot, nav_timeout_, robot->priority())) {
            robot->has_new_goal = false;
          }
          break;
        }
        if (b.finished) {
          if (b.result_status == NavigateToPose::Result::STATUS_SUCCEEDED) {
            robot->state = RobotState::UNLOADING;
            robot->state_time = 0.0;
            b.finished = false; ++b.gen;
            RCLCPP_INFO(get_logger(), "%s arrived at dropoff, unloading...", robot->name.c_str());
          } else {
            RCLCPP_WARN(get_logger(), "Task#%d 卸货导航失败（%s）→ 任务重新排队",
              robot->task ? robot->task->id : -1, b.result_msg.c_str());
            requeueTask(robot);
          }
        }
        break;
      }
      case RobotState::UNLOADING: {
        if (robot->state_time >= dwell_dropoff_) {
          if (robot->task) {
            robot->task->done = true;
            robot->task->active = false;
            robot->task->status = "DONE";
            RCLCPP_INFO(get_logger(), "Task#%d done by %s", robot->task->id, robot->name.c_str());
            robot->task.reset();
          }
          robot->state = RobotState::IDLE;
          robot->state_time = 0.0;
        }
        break;
      }
      case RobotState::TO_CHARGER: {
        if (robot->has_new_goal && !b.in_flight && !b.finished) {
          // 充电优先级 999：让行仲裁中永远让有任务的车
          if (sendNavGoal(robot, nav_timeout_charge_, 999)) {
            robot->has_new_goal = false;
          }
          break;
        }
        if (b.finished) {
          b.finished = false; ++b.gen;
          if (b.result_status == NavigateToPose::Result::STATUS_SUCCEEDED) {
            robot->state = RobotState::CHARGING;
            robot->state_time = 0.0;
            RCLCPP_INFO(get_logger(), "%s charging at charger#%d",
              robot->name.c_str(), robot->charger_id);
          } else {
            RCLCPP_WARN(get_logger(), "%s 前往充电桩失败（%s）→ 回 IDLE 待重试",
              robot->name.c_str(), b.result_msg.c_str());
            robot->state = RobotState::IDLE;
            robot->charger_id = -1;
            robot->state_time = 0.0;
          }
        }
        break;
      }
      case RobotState::CHARGING: {
        if (robot->battery >= charge_full_) {
          robot->state = RobotState::IDLE;
          robot->state_time = 0.0;
          robot->charger_id = -1;
          RCLCPP_INFO(get_logger(), "%s fully charged", robot->name.c_str());
        }
        break;
      }
      default: { break; }   // YIELD/FAULT：导航内部状态，调度器不再出现
    }
  }

  /// @brief 任务重新排队（导航失败/电量中断）+ 车回 IDLE
  /// @param robot 
  void requeueTask(const std::shared_ptr<RobotAgent> & robot)
  {
    NavBook & b = book_by_robot_[robot->name];
    b.finished = false;
    ++b.gen;
    if (robot->task) {
      robot->task->active = false;
      robot->task->status = "PENDING";
      RCLCPP_WARN(get_logger(), "Task#%d requeued", robot->task->id);
      robot->task.reset();
    }
    robot->state = RobotState::IDLE;
    robot->state_time = 0.0;
  }

  // ===================== 充电管理 =====================
  void assignChargerAndGo(const std::shared_ptr<RobotAgent> & robot)
  {
    int best = -1;
    double best_d = std::numeric_limits<double>::max();
    for (std::size_t i = 0; i < chargers_.size(); ++i) {
      bool used = false;
      for (const auto & o : robots_) {
        if (o.get() != robot.get() && o->charger_id == static_cast<int>(i) &&
            (o->state == RobotState::TO_CHARGER || o->state == RobotState::CHARGING)) {
          used = true;
          break;
        }
      }
      if (used) { continue; }
      double d = robot->distanceTo(chargers_[i].first, chargers_[i].second);
      if (d < best_d) { best_d = d; best = static_cast<int>(i); }
    }
    if (best < 0) {
      RCLCPP_WARN(get_logger(), "%s: no free charger", robot->name.c_str());
      return;
    }
    robot->charger_id = best;
    robot->goal_x = chargers_[best].first;
    robot->goal_y = chargers_[best].second;
    robot->has_new_goal = true;
    robot->resume_state = RobotState::TO_CHARGER;
    robot->state = RobotState::TO_CHARGER;
    robot->state_time = 0.0;
    RCLCPP_INFO(get_logger(), "%s -> charger#%d (battery %.1f%%)",
      robot->name.c_str(), best, robot->battery);
  }

  void requeueAndCharge(const std::shared_ptr<RobotAgent> & robot)
  {
    if (robot->task) {
      robot->task->active = false;
      robot->task->status = "PENDING";
      RCLCPP_WARN(get_logger(), "Task#%d requeued (battery low)", robot->task->id);
      robot->task.reset();
    }
    assignChargerAndGo(robot);
  }

  void updateBattery(const std::shared_ptr<RobotAgent> & robot, double dt)
  {
    if (robot->state == RobotState::CHARGING) {
      robot->battery = std::min(100.0, robot->battery + charge_rate_ * dt);
    } else if (robot->isMovingState()) {
      robot->battery = std::max(0.0, robot->battery - drain_move_ * dt);
    } else {
      robot->battery = std::max(0.0, robot->battery - drain_idle_ * dt);
    }
  }

  // ===================== 发布 =====================
  void publishStatus()
  {
    fleet_common::msg::FleetStatus fs;
    for (const auto & r : robots_) {
      fleet_common::msg::RobotState rs;
      rs.robot_id = r->id;
      rs.status = stateToString(r->state);
      rs.x = r->x;
      rs.y = r->y;
      rs.theta = r->theta;
      rs.battery = r->battery;
      rs.has_task = r->hasTask();
      rs.task_id = (r->task && r->task->active) ? r->task->id : -1;
      rs.linear_vel = r->lin_vel;
      rs.angular_vel = 0.0;
      fs.robots.push_back(rs);
    }
    for (const auto & t : tasks_) {
      if (t->done) { continue; }
      fleet_common::msg::Task tm;
      tm.task_id = t->id;
      tm.priority = t->priority;
      tm.pickup_x = t->pickup_x;
      tm.pickup_y = t->pickup_y;
      tm.dropoff_x = t->dropoff_x;
      tm.dropoff_y = t->dropoff_y;
      tm.is_active = t->active;
      fs.tasks.push_back(tm);
    }
    status_pub_->publish(fs);
  }

  void publishMarkers()
  {
    visualization_msgs::msg::MarkerArray arr;
    int tid = 0;
    for (const auto & t : tasks_) {
      if (t->done) { continue; }
      auto addSphere = [&](double x, double y, const float * c, const std::string & ns) {
        visualization_msgs::msg::Marker m;
        m.header.frame_id = "map";
        m.header.stamp = now();
        m.ns = ns;
        m.id = tid;
        m.type = visualization_msgs::msg::Marker::SPHERE;
        m.action = visualization_msgs::msg::Marker::ADD;
        m.scale.x = m.scale.y = m.scale.z = 0.35;
        m.color.r = c[0]; m.color.g = c[1]; m.color.b = c[2]; m.color.a = 0.8;
        m.pose.position.x = x; m.pose.position.y = y; m.pose.position.z = 0.15;
        m.pose.orientation.w = 1.0;
        arr.markers.push_back(m);
      };
      const float green[3] = {0.2f, 1.0f, 0.2f};
      const float red[3] = {1.0f, 0.2f, 0.2f};
      addSphere(t->pickup_x, t->pickup_y, green, "task_pickup");
      addSphere(t->dropoff_x, t->dropoff_y, red, "task_dropoff");
      ++tid;
    }
    markers_pub_->publish(arr);
  }

  // ===================== 成员 =====================
  std::unique_ptr<WarehouseMap> map_;      // 仅 addTask 合法性校验
  std::vector<std::string> robot_names_;
  std::vector<std::shared_ptr<RobotAgent>> robots_;
  std::unordered_map<std::string, std::shared_ptr<RobotAgent>> name_to_robot_;
  std::list<std::shared_ptr<TaskSpec>> tasks_;
  int next_task_id_ = 1;
  std::vector<std::pair<double, double>> chargers_;

  double dwell_pickup_, dwell_dropoff_, alloc_interval_;
  double drain_move_, drain_idle_, charge_rate_;
  double charge_threshold_, charge_full_, battery_critical_;
  double status_period_, marker_period_, control_rate_;
  double nav_timeout_, nav_timeout_charge_, v_max_;
  std::vector<double> charger_x_, charger_y_;

  bool use_rl_ = false;
  std::string policy_path_;
  std::shared_ptr<RLPolicy> rl_policy_;
  double sim_time_ = 0.0;

  rclcpp::Publisher<fleet_common::msg::FleetStatus>::SharedPtr status_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr markers_pub_;
  rclcpp::Subscription<gazebo_msgs::msg::ModelStates>::SharedPtr model_states_sub_;
  rclcpp::Subscription<fleet_common::msg::Task>::SharedPtr task_sub_;
  rclcpp::Subscription<fleet_common::msg::FleetNavStatus>::SharedPtr nav_status_sub_;
  rclcpp::Service<fleet_common::srv::AddTask>::SharedPtr add_task_srv_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr pause_srv_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr resume_srv_;
  rclcpp_action::Client<NavigateToPose>::SharedPtr nav_client_;
  rclcpp::TimerBase::SharedPtr control_timer_, status_timer_, marker_timer_;

  bool paused_ = false;
  double last_control_ = 0.0;
  double alloc_timer_ = 0.0;

  double secNow() const { return rclcpp::Clock(RCL_STEADY_TIME).now().seconds(); }
};

}  // namespace fleet_manager

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<fleet_manager::FleetSchedulerNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
