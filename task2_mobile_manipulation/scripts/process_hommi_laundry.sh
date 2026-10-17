#!/usr/bin/env bash
# Official HoMMI demonstration processing on the downloaded Laundry-v3 subset.
# Reduced setting: head pointmap / head RGB stored at 256x256 instead of 512x512 (disk budget).
set -euo pipefail
source "$(dirname "$0")/hommi_env.sh"
RAW="$EYECODE_ROOT/data/hommi_raw/Laundry-v3/UMI_iPhone"
WORK="$EYECODE_ROOT/data/hommi_processed"
SESSION=${SESSION:-laundry30}
mkdir -p "$WORK"
cd "$HOMMI_ROOT/hommi/demonstration_processing"
python process_demos_iphone.py group.iphone_dir="$RAW" group.symlink=true \
  filters.session_name=Laundry-v3 demonstrations_dir="$WORK/demonstrations" \
  "skip_stages=[visualize]" detect.num_workers=4
python create_session_iphone.py demonstrations_dir="$WORK/demonstrations" sessions_dir="$WORK/sessions" \
  "input_session_filters=[Laundry-v3]" output_session_name="$SESSION" max_demos=${MAX_DEMOS:-30}
python build_umi_dataset_iphone.py session_dir="$WORK/sessions/$SESSION" \
  replay_buffer.num_workers=2 replay_buffer.out_res_pointmap=256 replay_buffer.vis_rgb=false
ls -la "$WORK/sessions/$SESSION"
