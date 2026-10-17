# Demo and command screen recordings

[Google Drive: videos and report](https://drive.google.com/drive/folders/10DDnV1l6JCk7svF_ktngK9A5h_L9aIn2?usp=sharing).

| Requirement | Video | Content |
|---|---|---|
| Part 1: commands and results; required 3–5 minute demo | [part1_demo.mp4](part1_demo.mp4) · [Drive](https://drive.google.com/file/d/1d0fYTZsnEbabDSaaUSB1cGh2pADIFxfl/view?usp=drivesdk) | Current 273.828 s desktop recording: ROS commands, Gazebo, sensor/SLAM views, map saving and Nav2. |
| Part 2: commands, outputs and research demo | [part2_demo.mp4](part2_demo.mp4) · [Drive](https://drive.google.com/file/d/16TUE4Go7wBfSjiwhVOgQW2AcAaLK_1VK/view?usp=drivesdk) | Interleaved sections: HoMMI inference command/output → policy visualization; MoManipVLA command/output → planning examples; WBC command/output → controller replay. |

The interleaved Part 2 video lasts 192.3 s (3:12.3), with each command and its terminal output followed by a visualization of that component.

| Component | Command and terminal output | Visualization |
|---|---|---|
| HoMMI policy | 0:10–0:31.60 | 0:31.60–0:53.35 |
| MoManipVLA | 0:58.35–1:44.35 | 1:44.35–2:02.30 |
| HoMMI whole-body controller | 2:07.30–2:48.80 | 2:48.80–3:08.30 |

Introduction, component title cards and summary fill the remaining intervals. The three embedded terminal segments preserve all 109.1 s of the original screen recording. Visualizations are representative recordings from the full evaluation; additional failure and learned-policy examples are separate from the short terminal checks.

The [Part 1 guide](VIDEO_GUIDE.md) gives approximate review locations. The Part 1 recording supplied in the working folder is preserved; its old subtitle file was removed because it did not match the current recording.

The embedded command recordings show the visible GNOME Terminal window on this machine’s GNOME desktop. A real interactive Bash session loads the account’s normal configuration and displays `linh@linhchu`. Commands are entered automatically into that shell and executed; the video preserves the actual prompt, input and output.

The same Part 2 computations are reproduced by `bash scripts/run_command_demo.sh` from `task2_mobile_manipulation/`. The console log, recording metadata and fresh policy outputs are in `task2_mobile_manipulation/results/command_demo/`; planner/WBC CSVs use the `command_demo` tag. These are small checks, separate from the historical aggregate evaluation.

HoMMI policy inference is offline; the WBC and mobile planner use kinematic execution. Exact codec/duration/decode checks are recorded in [validation.json](../report/validation.json).
