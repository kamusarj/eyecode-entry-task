# Source me: environment for running the official HoMMI code from this repo.
export EYECODE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export HOMMI_ROOT="$EYECODE_ROOT/third_party/hommi"
export HOMMI_UMI_ROOT="$HOMMI_ROOT/deps/universal_manipulation_interface"
export HOMMI_RBY1_WBC_ROOT="$HOMMI_ROOT/deps/rby1-wbc"
export PYTHONPATH="$EYECODE_ROOT/hommi_repro/shims:$HOMMI_UMI_ROOT:$HOMMI_RBY1_WBC_ROOT:$HOMMI_ROOT:${PYTHONPATH:-}"
export WANDB_MODE=offline
if ! command -v conda >/dev/null 2>&1; then
  for prefix in "${CONDA_ROOT:-}" "$HOME/anaconda3" "$HOME/miniconda3" "$HOME/miniforge3"; do
    if [ -n "$prefix" ] && [ -f "$prefix/etc/profile.d/conda.sh" ]; then
      source "$prefix/etc/profile.d/conda.sh" || return $?
      break
    fi
  done
else
  source "$(conda info --base)/etc/profile.d/conda.sh" || return $?
fi
if ! command -v conda >/dev/null 2>&1; then
  echo "Conda not found. Set CONDA_ROOT to your conda installation." >&2
  return 1
fi
conda activate hommi || return $?
export MPLBACKEND=Agg
