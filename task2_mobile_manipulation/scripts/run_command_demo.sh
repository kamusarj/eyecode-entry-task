#!/usr/bin/env bash
# Short, real executions used in videos/part2_commands.mp4.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1
export PYTHONWARNINGS=ignore
export OMP_NUM_THREADS=1
export MUJOCO_GL=egl

run() {
  printf '\n$'
  printf ' %q' "$@"
  printf '\n\n'
  "$@"
}

printf 'PART 2 — actual commands and outputs\nHoMMI: trained-policy inference; MoManipVLA: one held-out episode.\n'
run source scripts/hommi_env.sh
run python scripts/infer_hommi_samples.py --out results/command_demo/hommi
sleep 6

printf '\nMoManipVLA — seed 1007; original selection vs feasibility-first\n'
run conda activate eyecode-p2
run python momanipvla_repro/run_eval.py --episodes 1 --start-seed 1007 --workers 1 \
  --variants fixed_base_vla momanip_bilevel bilevel_feasible_select --tag command_demo
sleep 6

printf '\nHoMMI — official whole-body controller, short Laundry replay\n'
run python hommi_repro/run_wbc_ablation.py --source data/wbc_laundry30 \
  --num-episodes 1 --offsets 0 --max-s 2 --workers 1 --tag command_demo
sleep 6

printf '\nThese short runs are separate from the historical aggregate evaluation.\n'
printf 'HoMMI inference is offline; controller and planner execution are kinematic.\n'
