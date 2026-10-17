# EyeCode Research Entry Task — Mobile Manipulation

Final source submission by **Bùi Hoàng Linh**. This package combines the existing ROSOrin Pro simulation work and the MoManipVLA/HoMMI research experiments. It includes source, configurations, saved maps, recorded quantitative evidence, trained checkpoints, a seven-page report, both demo videos with command execution and results.

Start with the [technical report](report/technical_report.pdf), [Part 1 video](videos/part1_demo.mp4) and [Part 2 video](videos/part2_demo.mp4), which interleaves terminal commands and outputs with the corresponding research visualizations. The [submission checklist](SUBMISSION_CHECKLIST.md) maps every requirement in the assignment to its evidence. The additional screen-recording requirement is covered by the Part 1 video and the interleaved terminal sections of the Part 2 video.

Online submission files: [Google Drive folder](https://drive.google.com/drive/folders/10DDnV1l6JCk7svF_ktngK9A5h_L9aIn2?usp=sharing).

| Artifact | Google Drive | Local copy |
|---|---|---|
| Part 1 demo | [Watch / download](https://drive.google.com/file/d/1d0fYTZsnEbabDSaaUSB1cGh2pADIFxfl/view?usp=drivesdk) | [part1_demo.mp4](videos/part1_demo.mp4) |
| Part 2 demo: interleaved commands and results | [Watch / download](https://drive.google.com/file/d/16TUE4Go7wBfSjiwhVOgQW2AcAaLK_1VK/view?usp=drivesdk) | [part2_demo.mp4](videos/part2_demo.mp4) |
| Technical report (PDF, 7 pages) | [Open PDF](https://drive.google.com/file/d/13UAg7azNqWJrK7cuvfQIVQAkQQnEdiED/view?usp=drivesdk) | [technical_report.pdf](report/technical_report.pdf) |
| Editable report (Word) | [Open Word](https://docs.google.com/document/d/1_goZUioAM8aLGFEJACo9yDmiSM4wGo1k/edit) | [technical_report.docx](report/technical_report.docx) |

```text
eyecode-entry-task/
├── README.md
├── task1_rosorin/
│   ├── README.md
│   ├── src/
│   ├── configs/
│   └── maps/
├── task2_mobile_manipulation/
│   ├── README.md
│   ├── hommi_repro/
│   ├── momanipvla_repro/
│   ├── scripts/
│   └── results/
├── report/
│   └── technical_report.pdf
└── videos/
    ├── part1_demo.mp4
    └── part2_demo.mp4
```

Additional folders contain the scripts, evidence, vendored dependencies, checkpoint, review samples, report source and validation results needed for reproduction.

## Reproduce

- **ROSOrin:** follow [task1_rosorin/README.md](task1_rosorin/README.md) in Ubuntu 22.04 / ROS 2 Humble / Gazebo Fortress, using the vendor VM or a matching installation. Commands cover build, control, LiDAR, live SLAM, map saving, AMCL single goal and ordered waypoints.
- **Research:** follow [task2_mobile_manipulation/README.md](task2_mobile_manipulation/README.md). CPU experiments use Python 3.10 in the `eyecode-p2` Conda environment; the official HoMMI policy uses the `hommi` environment. A trained checkpoint and six held-out review samples allow a small inference check before full data reconstruction.
- **Report:** install LaTeX with the packages listed in `report/technical_report.tex`, then run `bash report/build_report.sh`.

Quick integrity and metric review from this directory (Python 3, NumPy, Poppler and FFmpeg):

```bash
python3 task2_mobile_manipulation/scripts/audit_results.py
python3 report/validate_submission.py --decode-videos
```

Recorded results are preserved; new reruns use separate output tags. The [validation record](report/validation.json) describes exactly what was checked during final assembly. [SHA256SUMS](SHA256SUMS) records the packaged file contents; verify with `sha256sum -c SHA256SUMS`.

## Main results and scope

| Part | Measured result | Interpretation |
|---|---|---|
| ROSOrin | 21/21 SLAM exploration waypoints; single goal and four saved-map waypoints pass; mean SLAM error 0.023 m | Gazebo simulation with ground-truth odometry, static world |
| HoMMI policy | 30 training/validation demos; 395 held-out samples; left/right hand chunk error 6.65/8.71 cm vs 17.87/18.25 cm for hold still | Official code, reduced model, offline prediction; no physical robot success rate |
| HoMMI WBC | Laundry look-at: 0.067 mm mean hand error and 4.51° mean gaze error | Official controller, perfect kinematic tracking |
| MoManipVLA | 60 seeds; bi-level 32/60, feasibility-first 40/60 | Reduced planner study with noisy oracle waypoints and kinematic grasp checks |

MoManipVLA is a local reduced implementation; its squared smoothness penalty differs from the paper's unsquared norm. Its small learned waypoint model uses two instruction embeddings rather than OpenVLA's language model. Direct search outperforms bi-level search in these experiments, so the paper's bi-level advantage is not claimed as reproduced. The report analyzes successes, failures, ablations and the remaining limits.

## Source provenance and submission format

The original working folders were consolidated into this standalone tree. Historical logs may contain the original machine paths. Runtime scripts resolve the new directory or provide overrides. Third-party sources, robot assets, licenses and exact commits are included; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). The full raw training data and VM disks are omitted; the task README provides a pinned raw-data manifest and processing commands.

Source repository: [kamusarj/eyecode-entry-task](https://github.com/kamusarj/eyecode-entry-task). Install Git LFS before cloning the ~792 MB `.ckpt` file (`git lfs install`; `.gitattributes` is included). The videos and report are also available in the linked Google Drive folder.
