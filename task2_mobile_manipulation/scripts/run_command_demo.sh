#!/usr/bin/env bash
# Reproduce the short executions shown in the interleaved videos/part2_demo.mp4.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1
export PYTHONWARNINGS=ignore
export OMP_NUM_THREADS=1
export MUJOCO_GL=egl

printf 'HoMMI: trained-policy inference on six samples\n'
source scripts/hommi_env.sh
python scripts/infer_hommi_samples.py --out results/command_demo/hommi

printf '\nMoManipVLA: held-out seed 1007; original selection and feasibility-first\n'
conda activate eyecode-p2
python momanipvla_repro/run_eval.py --episodes 1 --start-seed 1007 --workers 1 \
  --variants fixed_base_vla momanip_bilevel bilevel_feasible_select --tag command_demo

printf '\nHoMMI: official whole-body controller, short Laundry replay\n'
python hommi_repro/run_wbc_ablation.py --source data/wbc_laundry30 \
  --num-episodes 1 --offsets 0 --max-s 2 --workers 1 --tag command_demo

printf '\nThese short runs are separate from the aggregate evaluation.\n'
printf 'HoMMI inference is offline; controller and planner execution are kinematic.\n'
