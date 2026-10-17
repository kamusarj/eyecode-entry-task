# Part 1 evidence used by the report

This folder contains selected raw results needed to verify the submitted measurements and debugging analysis. Desktop screenshots, recording setup files and redundant trial sessions are omitted.

| Folder / file | Purpose |
|---|---|
| `full_slam/` | Final control, four exploration routes, SLAM error, map saving, single goal and ordered waypoint results. |
| `controller_check/` | Fresh-spawn reproduction of single-goal, waypoint and U-turn navigation with the final controller. |
| `slam_replay/` | Controlled A–E comparison on the same recorded data: parameters, raw error samples, maps and loop-closure counts. |
| `debugging/` | Selected failure logs and results that support the report's root-cause analysis. |
| `map_provenance.log` | Provenance inspection of the pre-existing `map_01`, which is preserved but unused by the final runs. |

## Final results

`full_slam/control_forward.json` and `control_rotate.json` record commands, odometry, finite LiDAR returns and TF state. The four `explore_*.json` results cover 21/21 exploration waypoints. `slam_drift.json` records 52.4 m of motion with mean/max position error 0.023/0.058 m.

`single_goal.json` and `waypoints.json` contain the actual Nav2 action status, arrival samples, missed-waypoint list and settled-pose errors: 0.103 m / 0.005 rad and 0.141 m / 0.039 rad. `results.txt` summarizes them. `map_save.log` records map saving; the saved map itself is in `../maps/part1_slam_map.{pgm,yaml}`. Launch and navigation logs are retained for this final session.

`controller_check/` contains the three action results, the final Nav2 log, local-costmap plugins and the zero progress-failure count. All three routes pass.

## Selected diagnostic evidence

| Files in `debugging/` | Observed problem |
|---|---|
| `vendor_slam.log` | Loop-closure solves in the original distorted-map run. The controlled quantitative comparison is in `slam_replay/`. |
| `scan_matching_drift.json`, `odometry_weighted_drift.json` | Pose drift after disabling loop closure and after changing odometry penalties. |
| `dwb_stall.json`, `dwb_stall.log` | DWB stalls on a route around a thin wall. |
| `dwb_static_layer.json`, `dwb_static_layer.log` | Adding a static local-costmap layer does not resolve the U-turn failure. |
| `dwb_readme_retry.json` | Delayed DWB progress when leaving the north-west room. |
| `rpp_collision.json`, `rpp_collision.log` | The first RPP configuration rejects a trajectory near furniture. |
| `dds_startup.log` | Lost service responses and incomplete AMCL startup. |
| `rotation_reapproach.json`, `rotation_reapproach.log` | A pre-final configuration needs re-approach after large final rotations. |
| `boot_failure.log`, `boot_recovered.log` | VM boot diagnosis and subsequent recovered boot; see `../BOOT_RECOVERY.md`. |

Failed trials are identified as failures and are not counted as final successes. The current Part 1 video is a command walkthrough; the quantitative tables come from these JSON results rather than measurements extracted from the video.
