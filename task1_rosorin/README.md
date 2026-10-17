# EyeCode Part 1 — ROSOrin Pro simulation

Part 1 of the EyeCode research entry task. The ROSOrin Pro model runs in the RoboCup Home Gazebo world. It is driven over `/cmd_vel`, shows its simulated LiDAR in RViz2, and explores the whole house while SLAM Toolbox builds a map. The map is saved as `part1_slam_map`, and Nav2 with AMCL then uses that saved map for single-goal and ordered multi-point navigation. Quantitative results come from logged runs under `evidence/`. The current Part 1 video is a command walkthrough; the historical quantitative evidence remains in `evidence/full_slam/`. `REPORT_PART1.md` explains the results, the failures and how they were fixed.

| Requirement | Where it is shown |
|---|---|
| Gazebo simulation set up | `room_worlds.launch.py`, video |
| Run and control the robot | `pipeline_probe.py --command forward/rotate`, teleop, video |
| Visualize LiDAR | RViz2 `sensors.rviz` / `nav_nav2.rviz`, video |
| SLAM and map generation | SLAM Toolbox plus Nav2 exploration, `evidence/full_slam/`, video |
| Save the map | `map_saver_cli` saves `maps/part1_slam_map.{pgm,yaml}`, video |
| Navigation on the saved map | `evidence/full_slam/single_goal.json`, `waypoints.json`; README re-run from a fresh spawn in `evidence/controller_check/`; video |
| Demo video, 3–5 min | `../videos/part1_demo.mp4` |

## Environment

The guest is Ubuntu 22.04.4 running ROS 2 Humble, Ignition Gazebo Fortress 6.16.0, Nav2 1.1.14 and SLAM Toolbox. It runs in VirtualBox 7.2.10 with 6 vCPUs and 6144 MB RAM. The host is Ubuntu 26.04 on an i5-13420H with 14 GB RAM. Rendering inside the VM uses Mesa software rendering.

Run every ROS command **inside the guest's logged-in Ubuntu desktop**. Even with `gui:=false`, the Gazebo server needs the X display to render the GPU LiDAR in software. `gui:=false` drops only the expensive 3D client; physics, the drive and the sensors keep running. Fortress's EGL `--headless-rendering` returned all-infinite LiDAR ranges in this VM, so it is not used.

The packages were already installed in the Hiwonder VM used for the experiments; that VM disk is not part of this source package. Runtime dependencies are declared in `src/robot_gazebo/package.xml`. For a clean Humble/Fortress machine, run `rosdep install --from-paths src --ignore-src --rosdistro humble -r -y`.

## Install the Part 1 source

Set `PART1` to this folder inside the guest. During the recorded runs it was the VirtualBox shared folder `/media/sf_EyeCodePart1/part1_submission`. Back up any existing package before you overwrite it.

```bash
export PART1=/path/to/eyecode-entry-task/task1_rosorin
mkdir -p /home/ubuntu/ros2_ws/src/simulations
cp -a "$PART1/src/robot_gazebo/." /home/ubuntu/ros2_ws/src/simulations/robot_gazebo/
cp -a "$PART1/src/rosorin_description/." /home/ubuntu/ros2_ws/src/simulations/rosorin_description/
source /opt/ros/humble/setup.bash
cd /home/ubuntu/ros2_ws
colcon build --packages-select rosorin_description robot_gazebo --symlink-install --executor sequential
source "$PART1/scripts/env.sh"      # in every new terminal, with PART1 set
```

`env.sh` sources Humble and the workspace, and exports `IGN_IP=127.0.0.1`, `LIBGL_ALWAYS_SOFTWARE=1` and `LP_NUM_THREADS=2`.

## 1. Start Gazebo and control the robot

Terminal 1:

```bash
ros2 launch robot_gazebo room_worlds.launch.py nav:=true gui:=false
```

`nav:=true` connects ROS `/cmd_vel` to the Fortress MecanumDrive. It does not start Nav2. To see the 3D scene, use `gui:=true`, or attach a client in another terminal with `ign gazebo -g`. Closing that client leaves the server running.

Bounded, measured motion (refuses to move unless `/clock` and finite LiDAR data are present):

```bash
python3 "$PART1/scripts/pipeline_probe.py" --command forward --seconds 16 --output /tmp/control_forward.json   # 0.05 m/s for 2 sim-s
python3 "$PART1/scripts/pipeline_probe.py" --command rotate  --seconds 16 --output /tmp/control_rotate.json    # 0.25 rad/s for 3 sim-s
```

Keyboard teleop (use it only while Nav2 is stopped): `ros2 run robot_gazebo teleop_key_control --ros-args -r controller/cmd_vel:=/cmd_vel`. `w/s` drives, `a/d` turns, Ctrl+C exits.

## 2. LiDAR in RViz2

```bash
rviz2 -d /home/ubuntu/ros2_ws/install/robot_gazebo/share/robot_gazebo/rviz/sensors.rviz --ros-args -p use_sim_time:=true
ros2 topic echo /scan --once
```

The scan has 270 beams over ±1.3626 rad (a 156° front-facing field), 0.15–12 m range and 10 Hz. Ray-casting the recorded scans against the world's wall geometry gives a mean absolute residual of 0.010 m, which matches the configured noise σ = 0.01. Best time offset was 0 s, and there was no angle offset or scale error (see `REPORT_PART1.md`).

## 3. SLAM: explore the house and save the map

Keep Gazebo running and make sure AMCL/Nav2 are **not** running: SLAM Toolbox must be the only owner of `map → odom`.

Terminal 2 starts the SLAM node:

```bash
ros2 launch robot_gazebo slam.launch.py use_sim_time:=true use_rviz:=false
```

Terminal 3 starts Nav2 on the live SLAM map, without map_server or AMCL. Its RViz window shows the growing map, the scan, the costmaps and the plan:

```bash
ros2 launch robot_gazebo navigation.launch.py use_sim_time:=true use_localization:=false
```

Terminal 4 drives the exploration. These are the four routes used in the recorded run: dining area, south-east room, living area and north-west room. They end back at the spawn pose.

```bash
for r in explore_a_dining explore_b_southeast explore_c_southwest explore_d_northwest; do
  python3 "$PART1/scripts/navigation_test.py" --mapping --mode waypoints \
      --poses "$PART1/scripts/$r.json" --timeout 420 --output /tmp/$r.json
done
```

Instead of the scripted routes, you can drive with teleop (with the Nav2 launch stopped) or click **Nav2 Goal** in RViz. Optionally, log the SLAM pose error against Gazebo ground truth while mapping:

```bash
python3 "$PART1/scripts/slam_drift_logger.py" --output /tmp/slam_drift.json --stop-file /tmp/drift.stop   # touch /tmp/drift.stop to finish
```

Save the map under a **new name**:

```bash
mkdir -p /home/ubuntu/ros2_ws/part1_maps
ros2 run nav2_map_server map_saver_cli -f /home/ubuntu/ros2_ws/part1_maps/part1_slam_map \
    --ros-args -p use_sim_time:=true -p save_map_timeout:=15.0
```

The map recorded for this submission is already in the package as `maps/part1_slam_map.{pgm,yaml}`: 203 × 162 cells at 0.05 m/cell, origin `[-5.06, -4.04]`. The pre-existing `map_01` (origin unknown, see `evidence/map_provenance.log`) is kept unchanged and is not used by the final runs. Stop the SLAM and exploration launches (Ctrl+C) before you start localization.

## 4. Navigation on the saved map

Terminal 2 starts map_server, AMCL, Nav2 and RViz on the saved map:

```bash
ros2 launch robot_gazebo navigation.launch.py use_sim_time:=true \
    map:=/home/ubuntu/ros2_ws/install/robot_gazebo/share/robot_gazebo/maps/part1_slam_map.yaml
```

Initialize AMCL **once**, either with RViz **2D Pose Estimate** or with `--initial-pose`. The map frame is the frame SLAM started in, so a freshly spawned robot is at `0 0 0`. If the robot has already moved, give its actual map pose instead.

Single goal: the north-west room, through its doorway.

```bash
python3 "$PART1/scripts/navigation_test.py" --poses "$PART1/scripts/slam_single_goal.json" \
    --initial-pose 0 0 0 --timeout 180 --output /tmp/single_goal.json
```

Ordered multi-point: `(2.1, 3.2, 0) → (4.2, 1.2, −π/2) → (0.3, −1.3, −π/2) → (2.0, −1.5, 0)`, which visits the dining area, the east side, the corridor and the south-east room. Omit `--initial-pose` if AMCL is already initialized. `scripts/uturn_waypoints.json` is a harder regression route whose third goal sits just behind the south-east room's wall; it stalled the vendor DWB controller in three runs and passes with the current one.

```bash
python3 "$PART1/scripts/navigation_test.py" --mode waypoints --poses "$PART1/scripts/slam_waypoints.json" \
    --timeout 480 --output /tmp/waypoints.json
```

You can also click **Nav2 Goal** in RViz. The helper does not treat a displayed path as success. Its acceptance limits are 0.25 m and 0.25 rad, the vendor's original goal tolerance (`--accept-xy`, `--accept-yaw`). The controller itself targets 0.15 m / 0.15 rad, so the robot stops with margin. A run passes only with all of the following:

- Nav2 action status 4.
- An empty `missed_waypoints` list.
- A TF-measured arrival inside the limits for every waypoint.
- A settled final pose inside the limits.

It exits non-zero otherwise.

## Changes to the vendor package (summary)

The full list is in `MODIFIED_FILES.md` and the evidence is in `REPORT_PART1.md`.

- **`config/slam.yaml`**: `minimum_travel_distance/heading` changed from 0.0 m / 0.01 rad to 0.5 m / 0.5 rad. This was the main SLAM fix: on the same recorded data, mean pose error dropped from 2.07 m to 0.013 m.
- **`launch/include/navigation.launch.py`**: new `use_localization` argument, so Nav2 can run on SLAM Toolbox's live map during exploration.
- **TF/odometry**: removed an unconditional static `map → odom`, and moved the drive plugin's duplicate odometry to `/wheel/odom`.
- **Drive geometry**: corrected to the model's wheel radius and track.
- **`config/nav2_params.yaml`**: Regulated Pure Pursuit replaces the DWB controller, which stalled where a plan U-turns around a thin wall. Goal checker 0.15 m / 0.15 rad, inside the 0.25 acceptance limits. Wider global inflation; static map layer in the local costmap; 0.19 m footprint radius.
- **Launch files**: removed the invalid ros2_control launch path, forwarded the `nav` argument, and spawn into the named world.

## Troubleshooting

- **One `/clock` source and `use_sim_time:=true` everywhere.** `pipeline_probe.py --seconds 12 --output /tmp/p.json` checks clock, scan, odometry and TF ownership.
- **`map → odom` has exactly one owner**: SLAM Toolbox *or* AMCL, never both.
- **World discovery**: export `IGN_IP=127.0.0.1` and allow up to 10 s for `ign service -l`. Do not start a second server.
- **Fast-DDS service responses can be lost while many nodes start in the loaded VM.** The log shows `failed to send response ... client will not receive response`. `navigation_test.py` therefore retries its service calls. If a Nav2 lifecycle manager is left waiting on a lost reply, the helper finishes the standard configure/activate transitions itself and records them under `lifecycle_recovery` in its JSON. If you start Nav2 by hand and AMCL logs "not yet in the active state", restart the navigation launch.
- **Controller.** The vendor DWB controller stalled where a plan U-turns around a thin wall: its goal-distance critics form a local minimum there. Regulated Pure Pursuit follows the plan instead. If RPP logs "collision ahead", the plan is passing too close to furniture; the wider global inflation keeps plans centred. A final in-place turn of about 180° shifts the robot 2–4 cm (its rotation centre is offset from `base_footprint`), which can trigger a re-approach. Nav2's recoveries clear this on their own.
- **Resources**: close the Gazebo 3D client while RViz and Nav2 are running. Keep the guest logged in. Stop launches with Ctrl+C.

## Layout

- `src/`: the two ROS packages, with both maps.
- `scripts/`: action/diagnostic clients, route and goal files, the desktop recorder and `env.sh`.
- `evidence/`: selected final results, controlled SLAM replay and diagnostic failures used by the report; see `evidence/README.md`.
- `figures/`: plots generated from that evidence.
- `../videos/`: the current command screen recording and its guide.
- `../report/technical_report.pdf`: the combined seven-page report; `REPORT_PART1.md` preserves the detailed Part 1 notes.
- `BOOT_RECOVERY.md`: the VM boot fix that had to come first.

The combined submission also includes Part 2 under `../task2_mobile_manipulation/`.

References: [Hiwonder ROSOrin Pro Gazebo tutorial](https://wiki.hiwonder.com/projects/rosorin-pro/en/latest/docs/9_Gazebo_Simulation.html), [Nav2](https://docs.nav2.org/), [SLAM Toolbox](https://github.com/SteveMacenski/slam_toolbox), [Humble FollowWaypoints](https://github.com/ros-navigation/navigation2/blob/humble/nav2_msgs/action/FollowWaypoints.action).

## Standalone submission layout

`src/robot_gazebo/config` and `src/robot_gazebo/maps` are the runtime copies installed by colcon. `configs/` and `maps/` expose identical files for review. The root validator checks that these copies match. The original hardware and VM measurements are preserved in the logs; packaging checks do not claim a new full ROS run.
