"""RoboCup Home physics and sensors; omit the Gazebo GUI by default."""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def launch_setup(context):
    share = get_package_share_directory('robot_gazebo')
    world = os.path.join(share, 'worlds', 'robocup_home.sdf')
    gui = LaunchConfiguration('gui').perform(context).lower() == 'true'
    sim_time = LaunchConfiguration('use_sim_time')
    nav = LaunchConfiguration('nav')
    return [
        IncludeLaunchDescription(PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory('ros_gz_sim'), 'launch', 'gz_sim.launch.py')),
            launch_arguments={'gz_args': '-r '+('' if gui else '-s ')+world}.items()),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(os.path.join(share, 'launch', 'spawn_objects.launch.py'))),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(os.path.join(share, 'launch', 'spwan_model.launch.py')),
                                 launch_arguments={'world': 'robot_world', 'use_sim_time': sim_time}.items()),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(os.path.join(share, 'launch', 'ros_ign_bridge.launch.py')),
                                 launch_arguments={'nav': nav, 'use_sim_time': sim_time}.items()),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('gui', default_value='false'),
        DeclareLaunchArgument('nav', default_value='false'),
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        SetEnvironmentVariable('IGN_IP', os.environ.get('IGN_IP', '127.0.0.1')),
        SetEnvironmentVariable('LIBGL_ALWAYS_SOFTWARE', os.environ.get('LIBGL_ALWAYS_SOFTWARE', '1')),
        SetEnvironmentVariable('LP_NUM_THREADS', os.environ.get('LP_NUM_THREADS', '2')),
        SetEnvironmentVariable('DEPTH_CAMERA_TYPE', os.environ.get('DEPTH_CAMERA_TYPE', 'aurora')),
        OpaqueFunction(function=launch_setup),
    ])
