# NavStatus.msg — 单车导航状态快照（查询用）
# fleet_nav_server 以 2Hz 发布 /fleet/nav_status（FleetNavStatus 数组版），
# 无需持有 goal handle 即可随时查询任意车的导航状态（fleet_monitor / 排障用）。
string robot_name           # 车名（agv_X）
bool has_goal               # 是否有进行中的导航目标
uint8 state                # 与 NavigateToPose.action Feedback.state 同域：
                           # 0=PLANNING 1=MOVING 2=YIELD 3=RECOVERY
float64 goal_x              # 当前目标点
float64 goal_y
float64 distance_remaining  # 剩余距离 [m]
float64 elapsed_sec         # 已耗时 [s]
float64 timeout_sec         # 目标超时设置（0=不限时）
string message              # 最近一条状态说明
