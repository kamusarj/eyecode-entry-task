import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'robot_gazebo'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        *[(os.path.join('share', package_name, root),
           [os.path.join(root, name) for name in files
            if not name.endswith(('.pyc', '.bak', '.swp')) and not name.startswith('.')])
          for folder in ['launch', 'urdf', 'config', 'worlds', 'rviz', 'maps', 'meshes', 'models']
          for root, dirs, files in os.walk(folder) if '__pycache__' not in root],
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ubuntu',
    maintainer_email='ubuntu@todo.todo',
    description='ROSOrin Pro Fortress simulation and Humble Nav2 bringup',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'teleop_key_control = robot_gazebo.teleop_key_control:main',
        ],
    },
)
