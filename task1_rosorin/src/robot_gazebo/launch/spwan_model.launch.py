"""Spawn ROSOrin using Fortress's native drive and joint-state systems."""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    sim_time = LaunchConfiguration('use_sim_time')
    world = LaunchConfiguration('world')
    model = os.path.join(get_package_share_directory('robot_gazebo'), 'urdf', 'robot.gazebo.xacro')
    description = ParameterValue(Command(['xacro ', model, ' sim_ign:=true']), value_type=str)
    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('world', default_value='robot_world'),
        Node(package='robot_state_publisher', executable='robot_state_publisher',
             parameters=[{'robot_description': description,
                          'use_sim_time': ParameterValue(sim_time, value_type=bool)}], output='screen'),
        Node(package='ros_gz_sim', executable='create',
             arguments=['-world', world, '-topic', 'robot_description', '-name', 'robot',
                        '-allow_renaming', 'false', '-x', '0', '-y', '0', '-z', '0'],
             parameters=[{'use_sim_time': ParameterValue(sim_time, value_type=bool)}], output='screen'),
    ])
