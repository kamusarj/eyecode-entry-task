# Demo and command screen recordings

| Requirement | Video | Content |
|---|---|---|
| Part 1: commands and results; required 3–5 minute demo | [part1_demo.mp4](part1_demo.mp4) | Current 273.828 s desktop recording: ROS commands, Gazebo, sensor/SLAM views, map saving and Nav2. |
| Part 2: representative research outputs | [part2_demo.mp4](part2_demo.mp4) | 83.2 s compilation of HoMMI policy/controller outputs and reduced MoManipVLA success/failure cases. |
| Part 2: commands being executed and corresponding output | [part2_commands.mp4](part2_commands.mp4) | 97.1 s continuous terminal screen capture of fresh HoMMI inference, held-out MoManipVLA evaluation and official WBC replay. |

The [Part 1 guide](VIDEO_GUIDE.md) gives approximate review locations. The Part 1 recording supplied in the working folder is preserved; its old subtitle file was removed because it did not match the current recording.

Part 2 command execution is reproduced by `bash scripts/run_command_demo.sh` from `task2_mobile_manipulation/`. The console log, recording metadata and fresh policy outputs are in `task2_mobile_manipulation/results/command_demo/`; planner/WBC CSVs use the `command_demo` tag. These are small checks, separate from the historical aggregate evaluation.

HoMMI policy inference is offline; the WBC and mobile planner use kinematic execution. Exact codec/duration/decode checks are recorded in [validation.json](../report/validation.json).
