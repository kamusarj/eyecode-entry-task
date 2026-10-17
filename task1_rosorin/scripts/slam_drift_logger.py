#!/usr/bin/env python3
"""Log SLAM pose error against Gazebo ground-truth odometry.

The Fortress OdometryPublisher derives odom->base_footprint from the model's
simulated world pose, and SLAM Toolbox starts with map == odom. During mapping,
map->base_footprint minus odom->base_footprint is therefore the SLAM pose error.
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
from tf2_ros import Buffer, TransformListener, TransformException


def pose(t):
    q = t.transform.rotation
    return (t.transform.translation.x, t.transform.translation.y,
            math.atan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--stop-file', type=Path, required=True)
    p.add_argument('--rate', type=float, default=2.0)
    a = p.parse_args()
    rclpy.init(args=[])
    node = Node('part1_slam_drift_logger', parameter_overrides=[Parameter('use_sim_time', value=True)])
    buf = Buffer()
    TransformListener(buf, node)
    samples, last = [], 0.0
    try:
        while rclpy.ok() and not a.stop_file.exists():
            rclpy.spin_once(node, timeout_sec=.05)
            if time.monotonic()-last < 1/a.rate:
                continue
            last = time.monotonic()
            try:
                m = buf.lookup_transform('map', 'base_footprint', Time())
                o = buf.lookup_transform('odom', 'base_footprint', m.header.stamp)
            except TransformException:
                continue
            (mx, my, myaw), (ox, oy, oyaw) = pose(m), pose(o)
            samples.append({'stamp': m.header.stamp.sec+m.header.stamp.nanosec*1e-9,
                            'slam': [mx, my, myaw], 'ground_truth': [ox, oy, oyaw],
                            'xy_error_m': math.hypot(mx-ox, my-oy),
                            'yaw_error_rad': abs(math.atan2(math.sin(myaw-oyaw), math.cos(myaw-oyaw)))})
    finally:
        xy = [s['xy_error_m'] for s in samples]
        yw = [s['yaw_error_rad'] for s in samples]
        path = sum(math.hypot(b['ground_truth'][0]-a_['ground_truth'][0], b['ground_truth'][1]-a_['ground_truth'][1])
                   for a_, b in zip(samples, samples[1:]))
        summary = {'samples': len(samples), 'path_length_m': path,
                   'xy_error_mean_m': sum(xy)/len(xy) if xy else None,
                   'xy_error_rmse_m': math.sqrt(sum(v*v for v in xy)/len(xy)) if xy else None,
                   'xy_error_max_m': max(xy) if xy else None, 'xy_error_final_m': xy[-1] if xy else None,
                   'yaw_error_max_rad': max(yw) if yw else None, 'yaw_error_final_rad': yw[-1] if yw else None}
        a.output.write_text(json.dumps({'summary': summary, 'samples': samples}, indent=1)+'\n')
        print(json.dumps(summary))
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
