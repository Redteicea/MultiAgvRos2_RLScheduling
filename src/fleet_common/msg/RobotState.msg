int32 robot_id
string status         # IDLE, BUSY, CHARGING, FAULT [0 IDLE, 1 BUSY, 2 CHARGING, 3 FAULT]
float64 x
float64 y
float64 theta
float64 battery
bool has_task
int32 task_id         # 当前任务号（-1 = 无任务）
float64 linear_vel    # 实际线速度 m/s（位姿差分）
float64 angular_vel   # 实际角速度 rad/s（位姿差分）
