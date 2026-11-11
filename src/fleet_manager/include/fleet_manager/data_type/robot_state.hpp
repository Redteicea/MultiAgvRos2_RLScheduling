#ifndef FLEET_MANAGER_ROBOT_STATE_HPP_
#define FLEET_MANAGER_ROBOT_STATE_HPP_

#include <string>

namespace fleet_manager
{

    /// @brief 机器人状态枚举
    enum class RobotState
    {
        OFFLINE,     // 尚未收到位姿
        IDLE,        // 空闲待命
        NAV_PICKUP,  // 驶向取货点
        LOADING,     // 装货停靠
        NAV_DROPOFF, // 驶向卸货点
        UNLOADING,   // 卸货停靠
        TO_CHARGER,  // 驶向充电桩
        CHARGING,    // 充电中
        YIELD,       // 死锁避让（倒退）
        FAULT,       // 故障（等待人工处理）
    };

    /// @brief 状态转字符串
    /// @param s 状态
    /// @return 状态字符串
    inline const char * stateToString(RobotState s)
    {
        switch (s) {
            case RobotState::OFFLINE: return "OFFLINE";
            case RobotState::IDLE: return "IDLE";
            case RobotState::NAV_PICKUP: return "NAV_PICKUP";
            case RobotState::LOADING: return "LOADING";
            case RobotState::NAV_DROPOFF: return "NAV_DROPOFF";
            case RobotState::UNLOADING: return "UNLOADING";
            case RobotState::TO_CHARGER: return "TO_CHARGER";
            case RobotState::CHARGING: return "CHARGING";
            case RobotState::YIELD: return "YIELD";
            case RobotState::FAULT: return "FAULT";
        }
        return "UNKNOWN";
    }

    /// @brief 任务描述（ID / 优先级 / 取卸货坐标 / 生命周期状态）
    struct TaskSpec
    {
        int id = -1;
        int priority = 1;             // 1 = 最高
        double pickup_x = 0.0, pickup_y = 0.0;
        double dropoff_x = 0.0, dropoff_y = 0.0;
        bool active = false;          // 已分配
        bool done = false;
        std::string status;           // "PENDING" / "ACTIVE" / "DONE"
        double enter_time = 0.0;      // 入队时刻（RL obs 的队列等待 wait 用）
        double deadline = 200.0;      // 队列等待上限 s（RL obs 接近超期判定用）
    };

}
#endif // FLEET_MANAGER_ROBOT_STATE_HPP_