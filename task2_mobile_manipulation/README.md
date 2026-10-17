# Part 2 — Mobile manipulation

This folder contains the official HoMMI pipeline and a reduced MoManipVLA implementation, recorded experiments, a trained HoMMI checkpoint, the learned waypoint head, and representative videos. Read [paper_summary.md](docs/paper_summary.md) for both papers' problem, ideas, contributions and differences. The combined report is [technical_report.pdf](../report/technical_report.pdf); the result video is [part2_demo.mp4](../videos/part2_demo.mp4).

**Scope.** HoMMI policy evaluation is offline against held-out human demonstrations. Its official whole-body controller runs with perfect kinematic tracking. MoManipVLA uses a Panda on a holonomic base in procedural MuJoCo scenes, an oracle with noise or a small learned waypoint surrogate, and kinematic grasp/place checks. These are mechanism studies; the original papers' robot and OVMM success rates are not reproduced.

## Recorded results and evidence

| Experiment | Result | Evidence |
|---|---|---|
| HoMMI training | 30 demonstrations, 24 train / 6 validation, 60 epochs; sampled validation action MSE about 0.080 → 0.0029 | [training log](results/hommi_policy/train_log.json.txt), [training configuration](configs/hommi_training/config.yaml) |
| HoMMI offline policy | 395 validation samples; left/right hand error 6.65/8.71 cm, compared with 17.87/18.25 cm for hold-still | [offline_metrics.json](results/hommi_policy/offline_metrics.json), [per-sample errors](results/hommi_policy/errors_policy.npz) |
| Official HoMMI WBC | Laundry: look-at hand error 0.067 mm and gaze error 4.51°; 6-DoF head action: hand error 2.93 mm | [per-episode Laundry results](results/hommi_wbc/per_episode_laundry.csv) |
| MoManipVLA reduced, oracle σ=8 mm | 60 seeds per variant; bi-level 32/60, direct 36/60, fixed base 1/60 | [episodes_main.csv](results/momanipvla/episodes_main.csv) |
| Collision selection change | Bi-level success 32/60 → 40/60, 8 rescued / 0 broken; paired exact McNemar p=0.0078125 | [mcnemar_main.json](results/momanipvla/mcnemar_main.json) |
| Learned waypoint surrogate | About 5.5 cm validation error; bi-level 0/40 mobile successes | [training log](results/momanipvla/vla_training_log.csv), [held-out episodes](results/momanipvla/episodes_learned.csv) |

For the additional command-screen-recording requirement, see [part2_commands.mp4](../videos/part2_commands.mp4). Reproduce the short workflow with `bash scripts/run_command_demo.sh`; it executes trained HoMMI inference, a held-out MoManipVLA episode and a short official WBC replay. The corresponding outputs are under `results/command_demo/` and the `*_command_demo.csv` files.

All values above are historical experiment artifacts. New packaging checks are stored separately under `results/verification/`.

## 1. Dependencies

Run commands from this directory. Linux, Conda and Python 3.10 are used. CPU is sufficient for the kinematic experiments; a CUDA GPU is useful for policy inference/training. The recorded machine had an RTX 3050 laptop GPU (6 GB), 14 GB RAM and 12 CPU threads.

```bash
conda create -y -n eyecode-p2 python=3.10
conda activate eyecode-p2
python -m pip install -r requirements-sim.txt
```

Official third-party source and robot assets are already vendored under `third_party/` without Git histories. Exact commits are recorded in [third_party_versions.json](configs/third_party_versions.json); licenses are preserved. `scripts/fetch_third_party.sh` can fetch those versions when the vendored folder is absent.

For the official HoMMI policy:

```bash
# If Conda is installed outside ~/anaconda3, first set CONDA_ROOT to its prefix.
bash scripts/setup_hommi_env.sh
source scripts/hommi_env.sh
```

The recorded environment freezes are [eyecode-p2-pip-freeze.txt](configs/eyecode-p2-pip-freeze.txt) and [hommi-pip-freeze.txt](configs/hommi-pip-freeze.txt). They document the working environment; the setup scripts install the needed runtime subset.

DINOv3-S pretrained weights are subject to the model's access terms. Accept access at [facebook/dinov3-vits16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m), then use `hf auth login` in an environment containing the current Hugging Face CLI. No access token is included here.

## 2. Quick review without full data download

Recompute quantitative results directly from the recorded artifacts:

```bash
python scripts/audit_results.py
```

Run the trained official policy on six bundled validation samples (one from each held-out demonstration):

```bash
source scripts/hommi_env.sh
python scripts/infer_hommi_samples.py
# outputs: results/verification/hommi_samples/{metrics.json,predictions.npz,sample.png}
```

This check uses `checkpoints/hommi_latest.ckpt` (~792 MB), EMA weights and 16 DDIM steps. It reports its own six-sample errors, separately from the historical full evaluation. The checkpoint contains model/EMA/optimizer state; it is not merely a model URL.

Run a short simulation check without overwriting the recorded CSVs:

```bash
conda activate eyecode-p2
python momanipvla_repro/run_eval.py --episodes 1 --start-seed 1007 --workers 1 \
  --variants fixed_base_vla momanip_bilevel bilevel_feasible_select --tag submission_smoke
python hommi_repro/run_wbc_ablation.py --source data/wbc_laundry30 \
  --num-episodes 1 --offsets 0 --max-s 2 --workers 1 --tag submission_smoke
```

## 3. Reproduce the complete HoMMI run

The pinned [download manifest](configs/hommi_download_manifest.json) records 404 raw files at dataset revision `b4450274a57dd74d2df0f6ec89fb4d48d53b4c70`, downloaded for 40 demos plus calibration. Processing selected 30 complete demos. Allow several GB for downloads, the 3.75 GB replay buffer, the LMDB cache and outputs.

```bash
source scripts/hommi_env.sh
python scripts/download_recorded_subset.py
bash scripts/process_hommi_laundry.sh
CACHE_DIR="$PWD/data/hommi_cache" EPOCHS=60 bash scripts/train_hommi_small.sh
# Evaluate the bundled checkpoint on all validation samples:
python hommi_repro/eval_policy_offline.py --ckpt checkpoints/hommi_latest.ckpt \
  --dataset "$PWD/data/hommi_processed/sessions/laundry30/replay_buffer_laundry30.zarr.zip" \
  --cache-dir "$PWD/data/hommi_cache" --out results/verification/hommi_full
```

Training uses seed 42, bf16, batch 8, shared DINOv3-S/16 features, DiT 384×6, head images/point maps at 256² and 128 head tokens. See [train_hommi_small.sh](scripts/train_hommi_small.sh) for all overrides. The `.hydra` files in `configs/hommi_training/` preserve original absolute paths as provenance; executable commands override them. Diffusion sampling and GPU operations can change exact results between reruns.

Full WBC experiments:

```bash
conda activate eyecode-p2
python hommi_repro/run_wbc_ablation.py --num-episodes 25 --tag cup1_rerun
python hommi_repro/run_wbc_ablation.py --source data/wbc_laundry30 \
  --num-episodes 30 --offsets -0.15 0 0.15 --tag laundry_rerun
```

Debugging: the stream-decoding [patch](patches/hommi_stream_decode.patch) avoids retaining all full-resolution frames in RAM. DGL and PyTorch3D transform shims live in `hommi_repro/shims/`; the vendored transform implementation is reused. OpenCV-contrib 4.7.0.72 matches the code's mixed ArUco APIs. Missing editable installation, LMDB and scikit-learn are included in the setup script. Patch equivalence is an implementation intent; byte-for-byte equality of all processed outputs has not been independently established during assembly.

## 4. Reproduce MoManipVLA experiments

```bash
conda activate eyecode-p2
python momanipvla_repro/run_eval.py --episodes 60 --start-seed 1000 --workers 3 --tag main_rerun
python momanipvla_repro/run_eval.py --episodes 40 --start-seed 1000 --tag noise_rerun \
  --vla oracle:0.0 oracle:0.016 oracle:0.03 \
  --variants heuristic_base momanip_bilevel bilevel_feasible_select
MUJOCO_GL=egl python momanipvla_repro/render_episode.py --seed 1007 \
  --variants fixed_base_vla heuristic_base momanip_bilevel bilevel_feasible_select
```

Train or use the small learned waypoint surrogate:

```bash
MUJOCO_GL=egl python momanipvla_repro/vla_policy.py gen --episodes 300
source scripts/hommi_env.sh
python momanipvla_repro/vla_policy.py feats
python momanipvla_repro/vla_policy.py train --epochs 80
python momanipvla_repro/run_eval.py --episodes 40 --start-seed 1000 --vla learned --tag learned_rerun \
  --variants fixed_base_vla heuristic_base momanip_bilevel bilevel_feasible_select
```

The trained head is also bundled at the exact path used by `LearnedVLA`, `data/vla_demos/waypoint_head.pt`. This head uses DINOv3 vision features and an embedding for two instructions; it does not reproduce OpenVLA-7B or open-vocabulary language understanding. The original small-head trainer did not fix all random seeds, so a retrain is not bitwise deterministic.

**Fidelity difference:** the implementation uses squared joint/base increments for smoothness; MoManipVLA Eq. 4 uses unsquared Euclidean norms. Other reductions include analytic ESDF instead of nvblox, damped-least-squares IK instead of Pinocchio, smaller search budgets and kinematic execution. The original CSV results are kept unchanged and correspond to these choices.

The feasibility-first variant chooses the lowest-cost collision-free, IK-feasible candidate already evaluated, falling back to the original argmin when none is feasible. It is a local improvement studied here, not a claim about the authors' released implementation.

## 5. Figures and report

```bash
conda activate eyecode-p2
python scripts/make_tables.py
python scripts/make_figures.py
cd ../report
bash build_report.sh
```

The final video is already in `../videos/`. The original video builder accumulates frames in RAM and is retained for provenance; using the supplied video avoids that memory cost. Representative success/failure renders are in `results/momanipvla/`, and policy/WBC visualizations are in their respective result folders.

## Input-validation regression checks

In the `eyecode-p2` environment, run `python -m unittest discover -s tests -v`. These checks cover failed Conda/ROS setup, missing Laundry depth data and missing required video clips.
