# Modified files and scope

Only the `robot_gazebo` package was changed. `rosorin_description` and the pre-existing map `map_01` are preserved. The changes below describe the submitted source relative to the original vendor workspace.

| File | Change |
|---|---|
| `config/slam.yaml` | `minimum_travel_distance` 0.0 → 0.5 and `minimum_travel_heading` 0.01 → 0.5 (SLAM Toolbox defaults). Selected by the replay experiment in `evidence/slam_replay`. |
| `config/nav2_params.yaml` | Regulated Pure Pursuit controller instead of DWB (DWB stalled at thin-wall U-turns). Goal checker 0.15 m / 0.15 rad, non-stateful XY, inside the unchanged 0.25 m / 0.25 rad acceptance limits. Global inflation 0.55 m with scaling 3.0 (was 0.3 m / 10.0). Static map layer enabled in the local costmap. 0.19 m robot radius. |
| `launch/include/navigation.launch.py` | New `use_localization` argument (default true). When false, map_server and AMCL are skipped so Nav2 can run on SLAM Toolbox's live map. |
| `launch/include/slam.launch.py` | Rewritten as a plain SLAM Toolbox launch: removed the unused `need_compile` environment dependency and the malformed RViz `ExecuteProcess`; `use_rviz`, frame and topic arguments. |
| `launch/room_worlds.launch.py`, `ros_ign_bridge.launch.py`, `spawn_objects.launch.py`, `spwan_model.launch.py` | Forwarded `nav`; spawn into the named world `robot_world`; removed the unconditional static identity `map → odom` publisher and the invalid ros2_control manager, spawners and duplicate joint publisher. |
| `urdf/robot.gazebo.xacro`, `urdf/wheel.xacro` | Shared wheel radius and track for the drive; the drive's odometry moved to `/wheel/odom` so `/odom` has a single source. |
| `rviz/nav_nav2.rviz`, `rviz/slam.rviz`, `rviz/sensors.rviz` (new) | Transient-local robot description, 15 FPS, sensor view. |
| `package.xml`, `setup.py` | Runtime dependencies and recursive asset installation. |
| `maps/part1_slam_map.{pgm,yaml}` (new) | The map built and saved by the final SLAM run. |

Added scripts in `scripts/`:

- `navigation_test.py`: Nav2 action client and acceptance checks (0.25 m / 0.25 rad by default). Its `--mapping` mode skips map_server/AMCL. It retries service calls and checks lifecycle state.
- `slam_drift_logger.py`: SLAM pose error against ground-truth odometry.
- `pipeline_probe.py`: bounded control plus clock, scan, odometry and TF checks.
- `record_desktop.py`: live X11 capture of the guest desktop.
- `env.sh`.
- Route and goal JSON files, including `uturn_waypoints.json`, the controller regression route.

The reproduction commands in the README do not require the original host-side automation.

Guest boot change, needed before any ROS work: `/etc/default/grub.d/99-eyecode-part1-rcu.cfg` and `/boot/grub/grub.cfg`. The `ubuntu` user was also added to `vboxsf` for the shared folder. `/etc/fstab` is unchanged. See `BOOT_RECOVERY.md`.
