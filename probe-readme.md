
1. collect data

experiments/robot/libero/run_libero_eval_spatial.py

experiments/robot/libero/run_libero_eval_object.py

Usage:

cd openvla

python experiments/robot/libero/run_libero_eval.py   --model_family openvla   --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-object   --task_suite_name libero_object   --center_crop True

first time will download finetuned checkpoint to cache somewhere

2. train probe

train_spatial_probes.py

train_object_probes.py

use parse_libero_log.py to find out success and failure

eval on failure with eval_object_probe_on_failure.py