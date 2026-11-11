// 任务生成器（demo）：周期性地向 /fleet/task 注入随机搬运任务。
// 通过订阅 /fleet/status 统计在途任务数，不超过 max_active_tasks。
#include <rclcpp/rclcpp.hpp>
#include <fleet_common/msg/fleet_status.hpp>
#include <fleet_common/msg/task.hpp>

#include <chrono>
#include <random>
#include <string>
#include <vector>

namespace fleet_manager
{

class TaskGeneratorNode : public rclcpp::Node
{
public:
  TaskGeneratorNode() : Node("task_generator")
  {
    declare_parameter("enable", true);
    declare_parameter("interval_min", 8.0);
    declare_parameter("interval_max", 15.0);
    declare_parameter("max_active_tasks", 10);
    declare_parameter("max_total_tasks", 0);
    // 任务点使用站点 zone 外侧的可达停靠点（AGV 停靠后留有转向空间）
    declare_parameter("pickup_x", std::vector<double>{-9.6, 9.6});
    declare_parameter("pickup_y", std::vector<double>{0.0, 0.0});
    declare_parameter("dropoff_x", std::vector<double>{-9.6, 9.6});
    declare_parameter("dropoff_y", std::vector<double>{5.0, -5.0});
    declare_parameter("priorities", std::vector<int>{1, 1, 2});

    enable_ = get_parameter("enable").as_bool();
    interval_min_ = get_parameter("interval_min").as_double();
    interval_max_ = get_parameter("interval_max").as_double();
    max_active_ = get_parameter("max_active_tasks").as_int();
    max_total_ = get_parameter("max_total_tasks").as_int();
    pickup_x_ = get_parameter("pickup_x").as_double_array();
    pickup_y_ = get_parameter("pickup_y").as_double_array();
    dropoff_x_ = get_parameter("dropoff_x").as_double_array();
    dropoff_y_ = get_parameter("dropoff_y").as_double_array();
    auto pri = get_parameter("priorities").as_integer_array();
    for (int p : pri) { priorities_.push_back(p); }

    // 任务发布器
    task_pub_ = create_publisher<fleet_common::msg::Task>("/fleet/task", rclcpp::QoS(10));

    // 订阅AGV状态话题，更新已激活的任务数
    status_sub_ = create_subscription<fleet_common::msg::FleetStatus>(
      "/fleet/status", rclcpp::QoS(10),
      [this](const fleet_common::msg::FleetStatus::SharedPtr msg) {
        active_tasks_ = static_cast<int>(msg->tasks.size());
      });

    // 定时发布随机任务
    timer_ = create_wall_timer(std::chrono::milliseconds(500), [this]() { tick(); });
    RCLCPP_INFO(get_logger(), "task_generator started (enable=%d)", enable_);
  }

private:
  /// @brief 定时器回调函数，定时发布随机任务
  void tick()
  {
    if (!enable_) { return; }
    if (max_total_ > 0 && generated_ >= max_total_) {
      timer_->cancel();
      RCLCPP_INFO(get_logger(), "reached max_total_tasks=%d, stopped", max_total_);
      return;
    }
    double now = clock_.now().seconds();
    if (now - last_gen_time_ < next_interval_) { return; }
    if (active_tasks_ >= max_active_) { return; }

    // 随机取放货点
    //   取/卸点集合现在**是同一组 24 个站点**（四类任务由随机配对产生），
    //   因此必须**拒绝"从 A 送到 A"的自环任务** —— 否则约 4% 的任务是
    //   "原地取货再原地卸货"，既无意义，又会把一台车钉在站上白等取+卸两个 dwell。
    //   旧版取/卸集合部分重叠（8 个货架位）时同样存在该问题，只是概率更低（≈5.6%）。
    std::size_t pi = rnd_() % pickup_x_.size();
    std::size_t di = rnd_() % dropoff_x_.size();
    for (int guard = 0; guard < 20; ++guard) {
      const bool same_spot =
        std::abs(pickup_x_[pi] - dropoff_x_[di]) < 0.05 &&
        std::abs(pickup_y_[pi] - dropoff_y_[di]) < 0.05;
      if (!same_spot) { break; }
      di = rnd_() % dropoff_x_.size();
    }
    int pri = priorities_.empty() ? 1 : priorities_[rnd_() % priorities_.size()];

    fleet_common::msg::Task t;
    t.task_id = -1;
    t.priority = pri;
    t.pickup_x = pickup_x_[pi];
    t.pickup_y = pickup_y_[pi];
    t.dropoff_x = dropoff_x_[di];
    t.dropoff_y = dropoff_y_[di];
    t.is_active = true;
    task_pub_->publish(t);
    ++generated_;
    RCLCPP_INFO(get_logger(), "gen task#%d: (%.1f,%.1f)->(%.1f,%.1f) pri=%d",
      generated_, t.pickup_x, t.pickup_y, t.dropoff_x, t.dropoff_y, pri);

    last_gen_time_ = now;
    next_interval_ = interval_min_ + (rnd_() / static_cast<double>(rnd_.max())) * (interval_max_ - interval_min_);
  }

  bool enable_;
  double interval_min_, interval_max_;
  int max_active_, max_total_, active_tasks_ = 0, generated_ = 0;
  std::vector<double> pickup_x_, pickup_y_, dropoff_x_, dropoff_y_;
  std::vector<int> priorities_;
  std::mt19937 rnd_{std::random_device{}()};
  double last_gen_time_ = 0.0;
  double next_interval_ = 0.0;
  rclcpp::Clock clock_{RCL_STEADY_TIME};

  rclcpp::Publisher<fleet_common::msg::Task>::SharedPtr task_pub_;
  rclcpp::Subscription<fleet_common::msg::FleetStatus>::SharedPtr status_sub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

}  // namespace fleet_manager

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<fleet_manager::TaskGeneratorNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
