# Source this file in each guest terminal before running Part 1 commands.
source /opt/ros/humble/setup.bash || return $?
source "${ROSORIN_WS:-/home/ubuntu/ros2_ws}/install/setup.bash" || return $?
export IGN_IP=127.0.0.1
export LIBGL_ALWAYS_SOFTWARE=1
export LP_NUM_THREADS=2
export DEPTH_CAMERA_TYPE=aurora
