#!/usr/bin/env bash
# Source your environment setup
source /etc/network_turbo

# Export additional environment variables
export HF_HOME=/root/autodl-tmp/cache/
export MUJOCO_GL=osmesa

# Run your python script with arguments
python experiments/robot/libero/run_libero_eval_object.py \
  --model_family openvla \
  --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --layer_indices "[0,8,16,24,32]" \
  --center_crop True
