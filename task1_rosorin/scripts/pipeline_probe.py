#!/usr/bin/env python3
"""Bounded Humble pipeline observations and optional slow command trial."""
import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, qos_profile_sensor_data
from rclpy.time import Time
from geometry_msgs.msg import Twist, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import LaserScan
from tf2_msgs.msg import TFMessage
from tf2_ros import Buffer, TransformListener, TransformException


def stamp(s):
    return s.sec + s.nanosec * 1e-9


def yaw(q):
    return math.atan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z))


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


class Probe(Node):
    def __init__(self):
        super().__init__('part1_pipeline_probe', parameter_overrides=[Parameter('use_sim_time', value=True)])
        self.start = time.monotonic()
        self.clock = []
        self.clock_last_wall = None
        self.clock_regressions = 0
        self.scan = {'messages': 0, 'valid_messages': 0}
        self.scan_last_wall = None
        self.odom = {}
        self.edges = {'/tf': Counter(), '/tf_static': Counter()}
        self.map_odom = {'/tf': [], '/tf_static': []}
        self.amcl = []
        self.buffer = Buffer()
        self.listener = TransformListener(self.buffer, self)
        static = QoSProfile(depth=100, reliability=ReliabilityPolicy.RELIABLE,
                            durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.kept = [
            self.create_subscription(Clock, '/clock', self.on_clock, qos_profile_sensor_data),
            self.create_subscription(LaserScan, '/scan', self.on_scan, qos_profile_sensor_data),
            self.create_subscription(Odometry, '/odom', self.on_odom, qos_profile_sensor_data),
            self.create_subscription(TFMessage, '/tf', lambda m: self.on_tf(m, '/tf'), 100),
            self.create_subscription(TFMessage, '/tf_static', lambda m: self.on_tf(m, '/tf_static'), static),
            self.create_subscription(PoseWithCovarianceStamped, '/amcl_pose', self.on_amcl, static),
        ]

    def on_clock(self, m):
        t = stamp(m.clock)
        now = time.monotonic()
        if self.clock and t < self.clock[-1][0]:
            self.clock_regressions += 1
        if not self.clock or t > self.clock[-1][0]:
            self.clock_last_wall = now
        self.clock.append([t, now-self.start])
        if len(self.clock) > 2:
            self.clock.pop(1)

    def on_scan(self, m):
        valid = [v for v in m.ranges if math.isfinite(v) and m.range_min <= v <= m.range_max and v > 0]
        self.scan['messages'] += 1
        self.scan['valid_messages'] += int(bool(valid))
        self.scan.update(frame=m.header.frame_id, stamp=stamp(m.header.stamp), beams=len(m.ranges),
                         finite_ranges=len(valid), range_min=m.range_min, range_max=m.range_max,
                         observed_min=min(valid) if valid else None,
                         observed_max=max(valid) if valid else None)
        self.scan_last_wall = time.monotonic()

    def on_odom(self, m):
        key = m.child_frame_id
        p = m.pose.pose.position
        s = {'wall': time.monotonic()-self.start, 'stamp': stamp(m.header.stamp),
             'frame': m.header.frame_id, 'x': p.x, 'y': p.y, 'yaw': yaw(m.pose.pose.orientation),
             'vx': m.twist.twist.linear.x, 'vy': m.twist.twist.linear.y, 'wz': m.twist.twist.angular.z}
        if key not in self.odom:
            self.odom[key] = {'count': 0, 'frames': Counter(), 'first': s, 'last': s,
                              'max_xy_step': 0., 'stamp_regressions': 0, 'samples': []}
        d = self.odom[key]
        if d['count']:
            old = d['last']
            d['max_xy_step'] = max(d['max_xy_step'], math.hypot(s['x']-old['x'], s['y']-old['y']))
            d['stamp_regressions'] += int(s['stamp'] < old['stamp'])
        d['count'] += 1
        d['frames'][m.header.frame_id] += 1
        d['last'] = s
        if not d['samples'] or s['wall']-d['samples'][-1]['wall'] >= .4:
            d['samples'].append(s)
            d['samples'] = d['samples'][-24:]

    def on_tf(self, m, topic):
        for t in m.transforms:
            edge = t.header.frame_id.lstrip('/')+' -> '+t.child_frame_id.lstrip('/')
            self.edges[topic][edge] += 1
            if edge == 'map -> odom':
                self.map_odom[topic].append([stamp(t.header.stamp), t.transform.translation.x,
                                            t.transform.translation.y, yaw(t.transform.rotation)])
                self.map_odom[topic] = self.map_odom[topic][-12:]

    def on_amcl(self, m):
        self.amcl.append({'stamp': stamp(m.header.stamp), 'x': m.pose.pose.position.x,
                          'y': m.pose.pose.position.y, 'yaw': yaw(m.pose.pose.orientation),
                          'covariance_diagonal': [m.pose.covariance[i] for i in (0, 7, 35)]})
        self.amcl = self.amcl[-12:]

    def ready(self):
        now = time.monotonic()
        return (len(self.clock) == 2 and self.clock[-1][0] > self.clock[0][0]
                and self.clock_regressions == 0 and now-self.clock_last_wall < 2
                and self.scan_last_wall is not None and now-self.scan_last_wall < 3
                and self.scan.get('finite_ranges', 0) > 0)

    def transforms(self):
        result = {}
        for target, source in [('map', 'odom'), ('odom', 'base_footprint'), ('map', 'base_footprint')]:
            try:
                m = self.buffer.lookup_transform(target, source, Time())
                result[target+' <- '+source] = {'stamp': stamp(m.header.stamp),
                    'x': m.transform.translation.x, 'y': m.transform.translation.y,
                    'yaw': yaw(m.transform.rotation)}
            except TransformException as e:
                result[target+' <- '+source] = {'error': str(e)}
        return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seconds', type=float, default=12)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--command', choices=['none', 'forward', 'rotate'], default='none')
    p.add_argument('--cmd-topic', default='/cmd_vel')
    a = p.parse_args()
    if not math.isfinite(a.seconds) or not 0 < a.seconds <= 120:
        p.error('seconds must be in (0,120]')
    rclpy.init(args=[])
    node = Probe()
    pub = node.create_publisher(Twist, a.cmd_topic, 10) if a.command != 'none' else None
    motion = {'requested': a.command, 'status': 'disabled' if pub is None else 'waiting'}
    motion_start = None
    last_sent = 0
    result = {}
    try:
        while rclpy.ok() and time.monotonic()-node.start < a.seconds:
            rclpy.spin_once(node, timeout_sec=.05)
            if pub is None or motion['status'] in ['completed', 'aborted']:
                continue
            if motion_start is None and node.ready():
                motion_start = node.clock[-1][0]
                motion.update(status='running', start_sim=motion_start, start_tf=node.transforms())
            if motion_start is not None:
                duration = 2 if a.command == 'forward' else 3
                if not node.ready() or node.clock[-1][0]-motion_start >= duration:
                    motion.update(status='completed' if node.ready() else 'aborted', end_sim=node.clock[-1][0])
                    pub.publish(Twist())
                elif time.monotonic()-last_sent >= .1:
                    m = Twist()
                    if a.command == 'forward': m.linear.x = .05
                    else: m.angular.z = .25
                    pub.publish(m)
                    last_sent = time.monotonic()
    except Exception as e:
        result['error'] = repr(e)
    finally:
        if pub is not None:
            for _ in range(3):
                pub.publish(Twist())
                rclpy.spin_once(node, timeout_sec=.03)
        for d in node.odom.values():
            f, l = d['first'], d['last']
            d['delta'] = {'x': l['x']-f['x'], 'y': l['y']-f['y'], 'yaw': wrap(l['yaw']-f['yaw'])}
        elapsed = time.monotonic()-node.start
        sim_delta = node.clock[-1][0]-node.clock[0][0] if len(node.clock) == 2 else None
        result.update(wall_elapsed=elapsed, sim_delta=sim_delta,
                      sim_wall_ratio=sim_delta/elapsed if sim_delta is not None else None,
                      clock_regressions=node.clock_regressions, scan=node.scan,
                      odometry_by_child_frame=node.odom, tf_edges_by_topic=node.edges,
                      map_to_odom_samples_by_topic=node.map_odom, amcl=node.amcl,
                      final_tf=node.transforms(), motion=motion)
        a.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
        healthy = node.ready() and bool(node.odom)
        node.destroy_node()
        rclpy.shutdown()
    return 0 if healthy and (pub is None or motion['status']=='completed') and 'error' not in result else 1


if __name__ == '__main__':
    raise SystemExit(main())
