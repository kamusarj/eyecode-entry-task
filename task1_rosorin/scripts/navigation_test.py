#!/usr/bin/env python3
"""Execute real Humble Nav2 actions and save action/TF arrival evidence.

A run passes only with action status 4, no missed waypoints, a TF arrival inside the
acceptance limits for every waypoint, and a settled final pose inside them.
"""
import argparse
import json
import math
import time
from pathlib import Path
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.time import Time
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav2_msgs.action import NavigateToPose, FollowWaypoints
from nav_msgs.msg import Odometry
from rclpy.action import ActionClient
from rcl_interfaces.srv import GetParameters
from lifecycle_msgs.msg import Transition
from lifecycle_msgs.srv import ChangeState, GetState
from tf2_ros import Buffer, TransformListener, TransformException


def wrap(v):
    return math.atan2(math.sin(v), math.cos(v))


def yaw(q):
    return math.atan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['single', 'waypoints'], default='single')
    parser.add_argument('--poses', type=Path, required=True, help='JSON list of [map_x,map_y,yaw]')
    parser.add_argument('--initial-pose', nargs=3, type=float)
    parser.add_argument('--mapping', action='store_true',
                        help='SLAM Toolbox owns map->odom; do not wait for map_server/AMCL')
    parser.add_argument('--timeout', type=float, default=120)
    parser.add_argument('--accept-xy', type=float, default=0.25,
                        help='acceptance limit (m); the vendor goal tolerance. The controller aims tighter.')
    parser.add_argument('--accept-yaw', type=float, default=0.25, help='acceptance limit (rad)')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    poses = json.loads(args.poses.read_text())
    if not poses or any(len(p) != 3 or not all(math.isfinite(float(v)) for v in p) for p in poses):
        parser.error('poses must be a nonempty list of finite [x,y,yaw]')
    if args.mode == 'single' and len(poses) != 1:
        parser.error('single mode requires exactly one goal')
    if args.mapping and args.initial_pose:
        parser.error('--initial-pose initializes AMCL and cannot be used with --mapping')
    rclpy.init(args=[])
    node = Node('part1_navigation_test', parameter_overrides=[Parameter('use_sim_time', value=True)])
    buffer = Buffer()
    listener = TransformListener(buffer, node)
    report = {'mode': args.mode, 'mapping': args.mapping, 'poses': poses, 'initial_pose': args.initial_pose,
              'lifecycle_recovery': [],
              'samples': [], 'waypoint_feedback': [], 'arrivals': {}, 'action_status': None}
    latest_odom = [None]
    node.create_subscription(Odometry, '/odom', lambda msg: latest_odom.__setitem__(0, msg), 10)
    start = time.monotonic()
    current = 0
    client = None
    handle = None
    finished = False

    def spin_future(future, seconds):
        end = time.monotonic()+seconds
        while not future.done() and rclpy.ok() and time.monotonic() < end:
            rclpy.spin_once(node, timeout_sec=.05)
        if not future.done():
            raise TimeoutError('ROS service/action response timed out')
        return future.result()

    def call(client, request, seconds=5, tries=6):
        # A request sent right after discovery can be dropped before the server
        # matches this client; resend instead of failing the whole experiment.
        for attempt in range(tries):
            try:
                return spin_future(client.call_async(request), seconds)
            except TimeoutError:
                if attempt == tries-1:
                    raise

    clients = {}

    def lifecycle_state(name):
        if name not in clients:
            clients[name] = (node.create_client(GetState, '/'+name+'/get_state'),
                             node.create_client(ChangeState, '/'+name+'/change_state'))
        if not clients[name][0].wait_for_service(timeout_sec=.5):
            return None
        try:
            return call(clients[name][0], GetState.Request(), seconds=3, tries=2).current_state.id
        except TimeoutError:
            return None

    def wait_active(names, seconds):
        end = time.monotonic()+seconds
        while time.monotonic() < end:
            if all(lifecycle_state(n) == 3 for n in names):
                return True
            for _ in range(5):
                rclpy.spin_once(node, timeout_sec=.1)
        return False

    def recover(names):
        # The launch's lifecycle managers normally do this. A Fast-DDS service response lost
        # during bringup ("failed to send response ... client will not receive response")
        # leaves a manager blocked forever with its nodes inactive; finish the standard
        # transitions in order and record that this happened.
        for name in names:
            state = lifecycle_state(name)
            steps = {1: [Transition.TRANSITION_CONFIGURE, Transition.TRANSITION_ACTIVATE],
                     2: [Transition.TRANSITION_ACTIVATE]}.get(state, [])
            for transition in steps:
                req = ChangeState.Request()
                req.transition.id = transition
                try:
                    ok = bool(call(clients[name][1], req, seconds=15, tries=1).success)
                except TimeoutError:
                    ok = None
                report['lifecycle_recovery'].append({'node': name, 'from_state': state,
                                                     'transition': transition, 'success': ok})

    def current_pose():
        t = buffer.lookup_transform('map', 'base_footprint', Time())
        now = node.get_clock().now().nanoseconds*1e-9
        ts = t.header.stamp.sec+t.header.stamp.nanosec*1e-9
        return {'x': t.transform.translation.x, 'y': t.transform.translation.y,
                'yaw': yaw(t.transform.rotation), 'stamp': ts, 'age_sim': now-ts}

    def stamped(p):
        msg = PoseStamped()
        msg.header.frame_id = 'map'
        msg.header.stamp = node.get_clock().now().to_msg()
        msg.pose.position.x, msg.pose.position.y = float(p[0]), float(p[1])
        msg.pose.orientation.z, msg.pose.orientation.w = math.sin(p[2]/2), math.cos(p[2]/2)
        return msg

    def feedback(msg):
        nonlocal current
        if args.mode == 'waypoints':
            current = int(msg.feedback.current_waypoint)
            if not report['waypoint_feedback'] or report['waypoint_feedback'][-1]['index'] != current:
                report['waypoint_feedback'].append({'wall': time.monotonic()-start, 'index': current})

    try:
        srv = node.create_client(GetParameters, '/controller_server/get_parameters')
        if not srv.wait_for_service(timeout_sec=20):
            raise RuntimeError('controller_server parameter service unavailable')
        req = GetParameters.Request()
        req.names = ['general_goal_checker.xy_goal_tolerance', 'general_goal_checker.yaw_goal_tolerance']
        values = call(srv, req).values
        if len(values) != 2 or any(v.type != 3 or v.double_value <= 0 for v in values):
            raise RuntimeError('Could not read configured goal checker tolerances')
        report['controller_goal_tolerances'] = {'xy_m': values[0].double_value, 'yaw_rad': values[1].double_value}
        xy_tol, yaw_tol = args.accept_xy, args.accept_yaw
        report['acceptance_tolerances'] = {'xy_m': xy_tol, 'yaw_rad': yaw_tol}
        localization = [] if args.mapping else ['map_server', 'amcl']
        navigation = ['controller_server', 'smoother_server', 'planner_server', 'behavior_server',
                      'bt_navigator', 'waypoint_follower', 'velocity_smoother']
        # AMCL ignores an initial pose until it is active.
        if localization and not wait_active(localization, 30):
            recover(localization)
            if not wait_active(localization, 10):
                raise RuntimeError('map_server/AMCL lifecycle activation failed')
        if args.initial_pose:
            pub = node.create_publisher(PoseWithCovarianceStamped, '/initialpose', 10)
            until = time.monotonic()+15
            while (node.get_clock().now().nanoseconds == 0 or pub.get_subscription_count() == 0) and time.monotonic() < until:
                rclpy.spin_once(node, timeout_sec=.1)
            if pub.get_subscription_count() == 0:
                raise RuntimeError('AMCL initial-pose subscriber unavailable')
            msg = PoseWithCovarianceStamped()
            base = stamped(args.initial_pose)
            msg.header, msg.pose.pose = base.header, base.pose
            msg.pose.covariance[0] = msg.pose.covariance[7] = .25
            msg.pose.covariance[35] = .0685
            # Resend only if AMCL has not produced map->odom yet (a lost first message).
            for _ in range(3):
                pub.publish(msg)
                end = time.monotonic()+3
                while time.monotonic() < end:
                    rclpy.spin_once(node, timeout_sec=.1)
                if buffer.can_transform('map', 'base_footprint', Time()):
                    break
        # Action servers can exist while their lifecycle node is still inactive.
        # Wait for activation and a fresh complete TF chain after initializing AMCL.
        ready_start = time.monotonic()
        ready_deadline = ready_start+90
        recovered = False
        while time.monotonic() < ready_deadline:
            states = {n: lifecycle_state(n) for n in localization+navigation}
            try:
                pose = current_pose()
                fresh = node.get_clock().now().nanoseconds > 0 and abs(pose['age_sim']) <= .5
            except TransformException:
                fresh = False
            if fresh and all(v == 3 for v in states.values()):
                break
            # Only nodes left inactive (not mid-transition) once TF is available.
            if fresh and not recovered and time.monotonic()-ready_start > 40:
                recover([n for n in navigation if states[n] in (1, 2)])
                recovered = True
            rclpy.spin_once(node, timeout_sec=.1)
        else:
            raise TimeoutError('Nav2 lifecycle activation or fresh map-to-base TF unavailable')
        report['start_pose'] = current_pose()
        action = NavigateToPose if args.mode == 'single' else FollowWaypoints
        name = '/navigate_to_pose' if args.mode == 'single' else '/follow_waypoints'
        client = ActionClient(node, action, name)
        if not client.wait_for_server(timeout_sec=20):
            raise RuntimeError(name+' action server unavailable')
        goal = action.Goal()
        if args.mode == 'single': goal.pose = stamped(poses[0])
        else: goal.poses = [stamped(p) for p in poses]
        handle = spin_future(client.send_goal_async(goal, feedback_callback=feedback), 10)
        report['accepted'] = bool(handle.accepted)
        if not handle.accepted:
            raise RuntimeError('Nav2 rejected the goal')
        future = handle.get_result_async()
        until = time.monotonic()+args.timeout
        last_sample = 0
        while not future.done() and rclpy.ok() and time.monotonic() < until:
            rclpy.spin_once(node, timeout_sec=.05)
            if time.monotonic()-last_sample < .1:
                continue
            last_sample = time.monotonic()
            try:
                pose = current_pose()
            except TransformException:
                continue
            pose['wall'] = time.monotonic()-start
            pose['waypoint'] = current
            report['samples'].append(pose)
            target = poses[current]
            err_xy = math.hypot(pose['x']-target[0], pose['y']-target[1])
            err_yaw = abs(wrap(pose['yaw']-target[2]))
            if abs(pose['age_sim']) <= .5 and err_xy <= xy_tol and err_yaw <= yaw_tol:
                report['arrivals'][str(current)] = dict(pose, xy_error_m=err_xy, yaw_error_rad=err_yaw)
        if not future.done():
            spin_future(handle.cancel_goal_async(), 5)
            spin_future(future, 8)  # Confirm terminal result before releasing this command owner.
            finished = True
            raise TimeoutError('Navigation exceeded wall-time limit; cancellation requested')
        outcome = future.result()
        finished = True
        report['action_status'] = int(outcome.status)
        report['missed_waypoints'] = list(outcome.result.missed_waypoints) if args.mode == 'waypoints' else []
        report['pose_at_result'] = current_pose()
        # Success arrives asynchronously while the velocity smoother is braking.
        # Measure the actual settled endpoint, with fresh odometry, without sending any motion.
        settle_sim = node.get_clock().now().nanoseconds*1e-9+.75
        settle_wall = time.monotonic()+10
        settled = False
        while time.monotonic() < settle_wall:
            rclpy.spin_once(node, timeout_sec=.05)
            odom = latest_odom[0]
            now_sim = node.get_clock().now().nanoseconds*1e-9
            if odom is not None:
                odom_stamp = odom.header.stamp.sec+odom.header.stamp.nanosec*1e-9
                v = odom.twist.twist
                settled = (now_sim >= settle_sim and abs(now_sim-odom_stamp) < .5
                           and math.hypot(v.linear.x, v.linear.y) < .01 and abs(v.angular.z) < .02)
            if settled:
                break
        report['settled_after_result'] = settled
        report['final_pose'] = current_pose()
        goal = poses[-1]
        final = report['final_pose']
        err_xy = math.hypot(final['x']-goal[0], final['y']-goal[1])
        err_yaw = abs(wrap(final['yaw']-goal[2]))
        report['final_error'] = {'xy_m': err_xy, 'yaw_rad': err_yaw}
        final_ok = settled and abs(final['age_sim']) <= .5 and err_xy <= xy_tol and err_yaw <= yaw_tol
        # Include the instantaneous final arrival, even if the result arrived between sampling ticks.
        if final_ok:
            report['arrivals'][str(len(poses)-1)] = dict(final, xy_error_m=err_xy, yaw_error_rad=err_yaw)
        report['passed'] = (outcome.status == 4 and not report['missed_waypoints'] and final_ok
                            and len(report['arrivals']) == len(poses))
    except Exception as e:
        report.update(passed=False, error=type(e).__name__+': '+str(e))
    finally:
        if handle is not None and not finished:
            try: spin_future(handle.cancel_goal_async(), 3)
            except Exception: pass
        report['wall_seconds'] = time.monotonic()-start
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        node.destroy_node()
        rclpy.shutdown()
    print(json.dumps({k: report.get(k) for k in ['passed', 'action_status', 'missed_waypoints', 'final_error', 'error']}))
    return 0 if report.get('passed') else 1


if __name__ == '__main__':
    raise SystemExit(main())
