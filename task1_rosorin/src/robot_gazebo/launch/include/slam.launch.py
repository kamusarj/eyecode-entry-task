"""SLAM Toolbox mapping; run separately from AMCL/navigation."""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from nav2_common.launch import RewrittenYaml


def generate_launch_description():
    share = get_package_share_directory('robot_gazebo')
    args = {'use_sim_time': 'true', 'use_rviz': 'true', 'map_frame': 'map',
            'odom_frame': 'odom', 'base_frame': 'base_footprint', 'scan_topic': '/scan'}
    params = RewrittenYaml(source_file=os.path.join(share, 'config', 'slam.yaml'),
                           param_rewrites={k: LaunchConfiguration(k) for k in args if k != 'use_rviz'},
                           convert_types=True)
    return LaunchDescription([
        *[DeclareLaunchArgument(k, default_value=v) for k, v in args.items()],
        Node(package='slam_toolbox', executable='sync_slam_toolbox_node', name='slam_toolbox',
             output='screen', parameters=[params]),
        Node(package='rviz2', executable='rviz2', name='rviz2',
             condition=IfCondition(LaunchConfiguration('use_rviz')),
             arguments=['-d', os.path.join(share, 'rviz', 'slam.rviz')],
             parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}]),
    ])
