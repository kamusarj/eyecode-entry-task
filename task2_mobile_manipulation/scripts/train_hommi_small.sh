#!/usr/bin/env bash
# Small-scale HoMMI policy training with the OFFICIAL training code (hommi/train_network/train.py).
# Paper config -> our reduced config (6 GB RTX 3050 laptop GPU):
#   DINOv3 ViT-B/16 -> ViT-S/16 (shared, fine-tuned), DiT 768x10 -> 384x6, head input 512^2 -> 256^2,
#   512 -> 128 head 3D tokens, 200 -> 30 Laundry demos (24 train / 6 val), bf16 mixed precision.
set -euo pipefail
source "$(dirname "$0")/hommi_env.sh"
DATA=${DATA:-$EYECODE_ROOT/data/hommi_processed/sessions/laundry30/replay_buffer_laundry30.zarr.zip}
EPOCHS=${EPOCHS:-60}
EXP=${EXP:-small_vits}
TASK=${TASK:-ego3d_lookat_policy}
mkdir -p "$EYECODE_ROOT/outputs" && cd "$EYECODE_ROOT/outputs"
python "$HOMMI_ROOT/hommi/train_network/train.py" --config-name=umi_policy_dit model=diffusion_dit_3d task=$TASK \
  task.dataset_path="$DATA" \
  "task.shape_meta.obs.camera_head_main_rgb.shape=[3,256,256]" \
  "task.shape_meta.obs.camera_head_pointmap.shape=[3,256,256]" \
  task.dataset.val_ratio=0.2 task.dataset.normalizer_num_workers=2 task.dataset.pointmap_validity_num_workers=2 \
  task.dataset.cache_dir=${CACHE_DIR:-null} \
  model.obs_encoder.model_name=dinov3-vits16-pretrain-lvd1689m \
  model.obs_encoder.pointcloud_model_name=dinov3-vits16-pretrain-lvd1689m \
  model.obs_encoder.n_emb=384 model.obs_encoder.num_points=128 \
  model.attention_embed_dim=384 model.depth=6 model.num_heads=6 \
  dataloader.batch_size=${BS:-8} dataloader.num_workers=3 val_dataloader.batch_size=${BS:-8} val_dataloader.num_workers=1 \
  training.mixed_precision=bf16 training.num_epochs=$EPOCHS training.sample_every=5 training.val_every=5 \
  training.visualize_every=-1 checkpoint.topk.k=1 logging.mode=offline exp_name=$EXP "$@"
