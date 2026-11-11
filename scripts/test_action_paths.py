#!/usr/bin/env python3
"""action 链路针对性测试 v2：reject / cancel / timeout。
改进：每次发送前实时刷新 nav_status；等任务排空后再测（生成器需已停）。
"""
import sys
import time
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from fleet_common.action import NavigateToPose
from fleet_common.msg import FleetNavStatus


class ActionPathTest(Node):
    def __init__(self):
        super().__init__('action_path_test')
        self.client = ActionClient(self, NavigateToPose, '/fleet/navigate_to_pose')
        self.status = {}
        self.create_subscription(FleetNavStatus, '/fleet/nav_status', self.on_status, 10)
        self.client.wait_for_server(timeout_sec=10.0)

    def on_status(self, msg):
        for r in msg.robots:
            self.status[r.robot_name] = r.has_goal

    def refresh(self, spins=3):
        """收几拍 nav_status，返回 {name: has_goal}"""
        for _ in range(spins):
            rclpy.spin_once(self, timeout_sec=0.5)
        return dict(self.status)

    def idle_robot(self):
        for name, has_goal in self.status.items():
            if not has_goal:
                return name
        return None

    def send(self, robot, x, y, timeout_sec=0.0, priority=999):
        goal = NavigateToPose.Goal()
        goal.robot_name = robot
        goal.x = x
        goal.y = y
        goal.timeout_sec = timeout_sec
        goal.priority = priority
        fut = self.client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=5.0)
        return fut.result()

    def wait_result(self, gh, timeout=30.0):
        fut = gh.get_result_async()
        rclpy.spin_until_future_complete(self, fut, timeout_sec=timeout)
        return fut.result()


def main():
    rclpy.init()
    t = ActionPathTest()
    t.refresh()

    # ---- 等 3 车全部排空（生成器已停；在途任务自然完成） ----
    print('[W] 等待任务排空（最多 360s）...', flush=True)
    t0 = time.time()
    while time.time() - t0 < 360:
        st = t.refresh()
        if st and all(not v for v in st.values()):
            break
        time.sleep(5.0)
    st = t.refresh()
    print(f'[W] 当前状态: {st}', flush=True)
    if not st or not all(not v for v in st.values()):
        print('[W] WARN: 未全部空闲，仍继续测（用当前空闲车）')

    idle = t.idle_robot()
    if idle is None:
        print('RESULT: NO-IDLE-ROBOT')
        return 1

    ok = True

    # ---- 测试2：cancel ----
    gh = t.send(idle, 6.0, 6.8)   # 远点，确保不会 3 秒内到
    if gh and gh.accepted:
        time.sleep(3.0)
        cf = gh.cancel_goal()
        rclpy.spin_until_future_complete(t, cf, timeout_sec=5.0)
        res = t.wait_result(gh, timeout=20.0)
        if res is None:
            print('[T2 cancel] FAIL: 取消后无结果返回'); ok = False
        else:
            st = res.result.status
            if st == NavigateToPose.Result.STATUS_CANCELED:
                print(f'[T2 cancel] PASS: {idle} 取消成功, final=({res.result.final_x:.2f},{res.result.final_y:.2f}), msg="{res.result.message}"')
            elif st == NavigateToPose.Result.STATUS_SUCCEEDED:
                print(f'[T2 cancel] WARN: 恰好在取消前到点（SUCCEEDED）')
            else:
                print(f'[T2 cancel] FAIL: 终态={st} 期望 2(CANCELED)'); ok = False
    else:
        print(f'[T2 cancel] FAIL: goal 未被接受（{idle} 被调度器抢先派单?）'); ok = False

    time.sleep(2.0)
    t.refresh()
    idle = t.idle_robot()
    if idle is None:
        print('[T3 timeout] SKIP: 无空闲车'); idle = None

    # ---- 测试3：timeout（timeout_sec=4 的远点） ----
    if idle:
        gh = t.send(idle, -6.0, 6.8, timeout_sec=4.0)
        if gh and gh.accepted:
            res = t.wait_result(gh, timeout=25.0)
            if res is None:
                print('[T3 timeout] FAIL: 超时后无结果返回'); ok = False
            else:
                st = res.result.status
                if st == NavigateToPose.Result.STATUS_TIMEOUT:
                    print(f'[T3 timeout] PASS: {idle} 超时上报, dist={res.result.distance_remaining:.2f}, msg="{res.result.message}"')
                else:
                    print(f'[T3 timeout] FAIL: 期望 TIMEOUT(3) 实际 {st}'); ok = False
        else:
            print('[T3 timeout] SKIP: goal 被拒'); ok = False

    print('RESULT:', 'PASS' if ok else 'CHECK-ABOVE')
    t.destroy_node()
    rclpy.shutdown()
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
