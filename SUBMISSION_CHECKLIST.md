# Final submission checklist

The assignment was read directly from **EyeCode Research Entry Task — Mobile Manipulation.docx**. The required final artifacts are source/GitHub, reproduction README, Part 1 video, Part 2 result/demo video where applicable, and a **5–7 page report covering both parts**. Only Part 1 has a specified **3–5 minute video** duration. Full-scale paper reproduction is explicitly not required.

| Requirement | Included deliverable / evidence | Status |
|---|---|---|
| GitHub repository **or source code** | This standalone folder; ROS and research source included | Included; source-code option |
| README with reproduction commands | [root README](README.md), [ROS README](task1_rosorin/README.md), [research README](task2_mobile_manipulation/README.md) | Included |
| Combined 5–7 page technical report | [technical_report.pdf](report/technical_report.pdf), LaTeX source and build script | 7 pages |
| Part 1 video, 3–5 min | [part1_demo.mp4](videos/part1_demo.mp4), [timeline](videos/VIDEO_GUIDE.md) | 273.828 s, within range |
| Set up ROSOrin Gazebo | [simulation package](task1_rosorin/src/robot_gazebo/), final-run logs | Included |
| Run and control robot | [control JSON](task1_rosorin/evidence/full_slam/control_forward.json), video | Recorded |
| Visualize simulated LiDAR | RViz configurations, [scan/control evidence](task1_rosorin/evidence/full_slam/control_rotate.json), video | Recorded |
| SLAM and generated/saved map | [map YAML](task1_rosorin/maps/part1_slam_map.yaml), PGM, [SLAM log](task1_rosorin/evidence/full_slam/slam_drift.json) | Recorded |
| Single-goal navigation on saved map | [single_goal.json](task1_rosorin/evidence/full_slam/single_goal.json) | Status 4; 0.103 m / 0.005 rad |
| Multi-point navigation | [waypoints.json](task1_rosorin/evidence/full_slam/waypoints.json) | Status 4, no missed waypoints |
| Part 1 setup/results/problems/solutions | Combined report pp. 1–2; [detailed notes](task1_rosorin/REPORT_PART1.md) | Included |
| Understand and compare both papers | Combined report p. 3; [paper summaries](task2_mobile_manipulation/docs/paper_summary.md) | Included |
| Official codebase or assigned-method reproduction | Vendored HoMMI, [HoMMI harness](task2_mobile_manipulation/hommi_repro/), [reduced MoManipVLA](task2_mobile_manipulation/momanipvla_repro/) | Included, scope disclosed |
| Small-scale training/fine-tuning when feasible | Official HoMMI training log/config/checkpoint; learned waypoint head | Included |
| Inference/evaluation and representative visualization | CSV/NPZ results, policy overlays, WBC and planner renders; [new inference check](task2_mobile_manipulation/results/verification/hommi_samples/metrics.json) | Included |
| At least one quantitative metric | 540 main planning episode rows, 395 validation windows, WBC per-episode errors | Included and recomputed |
| Success/failure analysis | Combined report pp. 4–6; seeds 1007/1014 and learned-policy transfer failure | Included |
| Limitation and possible remedy | Squared smoothness deviation, offline/kinematic limits, tested feasibility-first change | Included |
| Screen recording: commands and corresponding output for each task | [Part 1 commands + Gazebo/RViz](videos/part1_demo.mp4), [Part 2 command execution](videos/part2_commands.mp4), [command script](task2_mobile_manipulation/scripts/run_command_demo.sh) | Included; actual executions |
| Part 2 result/demo video | [part2_demo.mp4](videos/part2_demo.mp4) | Included; no mandated duration |

Fresh verification is recorded in [report/validation.json](report/validation.json) and [artifact_metrics_audit.json](task2_mobile_manipulation/results/verification/artifact_metrics_audit.json). Historical full-scale logs are evidence from earlier runs; assembly performed small reruns, not a second complete ROS session or a full retraining.

Remaining submission action: submit the source folder, or publish a repository with Git LFS for the large checkpoint and grant reviewer access. The report and videos are final artifacts, not placeholders. If the submission form requires personal fields beyond the name already in the report, enter those in the form.
