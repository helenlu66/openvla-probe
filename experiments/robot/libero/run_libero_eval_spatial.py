
"""
run_libero_eval.py

Runs a model in a LIBERO simulation environment.

Usage:
    # OpenVLA:
    # IMPORTANT: Set center_crop=True if model is fine-tuned with augmentations
    python experiments/robot/libero/run_libero_eval.py \
        --model_family openvla \
        --pretrained_checkpoint <CHECKPOINT_PATH> \
        --task_suite_name [ libero_spatial | libero_object | libero_goal | libero_10 | libero_90 ] \
        --center_crop [ True | False ] \
        --run_id_note <OPTIONAL TAG TO INSERT INTO RUN ID FOR LOGGING> \
        --use_wandb [ True | False ] \
        --wandb_project <PROJECT> \
        --wandb_entity <ENTITY>
"""

import os
os.environ["MUJOCO_GL"] = "egl"

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import draccus
import numpy as np
import tqdm
from libero.libero import benchmark

import wandb

import torch

from detection.libero_object_object_relation_detector import LiberoObjectObjectRelationDetector as LiberoObjectRelationDetector
from detection.libero_object_action_state_subgoal_detector import LiberoObjectActionDetector

# ▲ NEW ────────────────────────────────────────────────────────────────────────
# Which hidden layers to probe. 0 = embedding, 1-32 = Llama blocks.
DESIRED_LAYER_INDICES = list(range(33))
# DESIRED_LAYER_INDICES = [0, 7, 15, 23, 31]   #   5 checkpoints through the stack

# Text files that list every symbolic key we may see (0/1 labels)
OBJREL_KEYS_FILE   = "./experiments/robot/libero/spatial_object_relations_keys.txt"
ACTSUB_KEYS_FILE   = "./experiments/robot/libero/spatial_action_states_keys.txt"

import ast
with open(OBJREL_KEYS_FILE) as f:
    ALL_OBJREL_KEYS = ast.literal_eval(f.read().strip())
with open(ACTSUB_KEYS_FILE) as f:
    ALL_ACTSUB_KEYS = ast.literal_eval(f.read().strip())

objrel_key2idx = {k: i for i, k in enumerate(ALL_OBJREL_KEYS)}
actsub_key2idx = {k: i for i, k in enumerate(ALL_ACTSUB_KEYS)}
NUM_OBJREL = len(ALL_OBJREL_KEYS)
NUM_ACTSUB = len(ALL_ACTSUB_KEYS)

# INSIDE_KEY  = "inside alphabet_soup_1 basket_1_contain_region"
# INSIDE_IDX  = objrel_key2idx[INSIDE_KEY]

EXTRA_STEPS_AFTER_SUCCESS = int(300)

def objrel_dict_to_vec(d):
    v = np.full(NUM_OBJREL, -1, dtype=np.int8)   # default “not present”
    for k, val in d.items():
        if k in objrel_key2idx:
            v[objrel_key2idx[k]] = val
    return v

def actsub_dict_to_vec(d):
    v = np.full(NUM_ACTSUB, -1, dtype=np.int8)
    for k, val in d.items():
        if k in actsub_key2idx:
            v[actsub_key2idx[k]] = val
    return v

def stack_list(v):
    if isinstance(v[0], np.ndarray):
        return torch.from_numpy(np.stack(v, axis=0))
    else:                       # assume torch.Tensor
        return torch.stack(v, dim=0)
# ───────────────────────────────────────────────────────────────────────────────


# Append current directory so that interpreter can find experiments.robot
sys.path.append("../..")
from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    quat2axisangle,
    save_rollout_video,
)
from experiments.robot.openvla_utils import get_processor
from experiments.robot.robot_utils import (
    DATE_TIME,
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)


@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family
    pretrained_checkpoint: Union[str, Path] = ""     # Pretrained checkpoint path
    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = "libero_spatial"          # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 6                    # Number of rollouts per task

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add in run ID for logging
    local_log_dir: str = "./experiments/logs"        # Local directory for eval logs

    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_project: str = "YOUR_WANDB_PROJECT"        # Name of W&B project to log to (use default!)
    wandb_entity: str = "YOUR_WANDB_ENTITY"          # Name of entity to log under

    seed: int = 7                                    # Random Seed (for reproducibility)

    # fmt: on


@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> None:
    assert cfg.pretrained_checkpoint is not None, "cfg.pretrained_checkpoint must not be None!"
    if "image_aug" in cfg.pretrained_checkpoint:
        assert cfg.center_crop, "Expecting center_crop==True because model was trained with image augmentations!"
    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"

    # Set random seed
    set_seed_everywhere(cfg.seed)

    # [OpenVLA] Set action un-normalization key
    cfg.unnorm_key = cfg.task_suite_name

    # Load model
    model = get_model(cfg)

    # [OpenVLA] Check that the model contains the action un-normalization key
    if cfg.model_family == "openvla":
        # In some cases, the key must be manually modified (e.g. after training on a modified version of the dataset
        # with the suffix "_no_noops" in the dataset name)
        if cfg.unnorm_key not in model.norm_stats and f"{cfg.unnorm_key}_no_noops" in model.norm_stats:
            cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"
        assert cfg.unnorm_key in model.norm_stats, f"Action un-norm key {cfg.unnorm_key} not found in VLA norm_stats!"

    # [OpenVLA] Get Hugging Face processor
    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)

    # Initialize local logging
    run_id = f"EVAL-{cfg.task_suite_name}-{cfg.model_family}-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    print(f"Logging to local log file: {local_log_filepath}")

    # Initialize Weights & Biases logging as well
    if cfg.use_wandb:
        wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=run_id,
        )

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    print(f"Task suite: {cfg.task_suite_name}")
    log_file.write(f"Task suite: {cfg.task_suite_name}\n")

    # Get expected image dimensions
    resize_size = get_image_resize_size(cfg)

    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        rng = np.random.default_rng(cfg.seed + task_id)      # task-specific but reproducible
        perm = rng.permutation(len(initial_states))          # new ordering
        initial_states = [initial_states[i] for i in perm]   # shuffled in-place

        # Initialize LIBERO environment and task description
        env, task_description = get_libero_env(task, cfg.model_family, resolution=256)

        core_env = env.env  # Unwrap

        # Create detectors for object relations & action subgoals
        obj_rel_detector  = LiberoObjectRelationDetector(core_env, return_int=True)
        action_det_detector = LiberoObjectActionDetector(core_env, return_int=True)

        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
            print(f"\nTask: {task_description}")
            log_file.write(f"\nTask: {task_description}\n")

            # Reset environment
            env.reset()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            replay_images = []
            done = False          # ← NEW: ensures done is always defined
            if cfg.task_suite_name == "libero_spatial":
                max_steps = 220  # longest training demo has 193 steps
            elif cfg.task_suite_name == "libero_object":
                max_steps = 280  # longest training demo has 254 steps
            elif cfg.task_suite_name == "libero_goal":
                max_steps = 300  # longest training demo has 270 steps
            elif cfg.task_suite_name == "libero_10":
                max_steps = 520  # longest training demo has 505 steps
            elif cfg.task_suite_name == "libero_90":
                max_steps = 400  # longest training demo has 373 steps

            print(f"Starting episode {task_episodes+1}...")
            log_file.write(f"Starting episode {task_episodes+1}...\n")

            # ▲ NEW (inside episode-for-loop, before while)
            episode_embeds = {L: [] for L in DESIRED_LAYER_INDICES}
            episode_objrel = []
            episode_actsub = []

            success_reached      = False
            steps_after_success  = 0

            while t < max_steps + cfg.num_steps_wait:
                try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < cfg.num_steps_wait:
                        obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                        t += 1
                        continue

                    # Get preprocessed image
                    img = get_libero_image(obs, resize_size)

                    # Save preprocessed image for replay video
                    replay_images.append(img)

                    # Prepare observations dict
                    # Note: OpenVLA does not take proprio state as input
                    observation = {
                        "full_image": img,
                        "state": np.concatenate(
                            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                        ),
                    }

                    # ▲ NEW -----------------------------------------------------------------------
                    embeds, action = get_action(
                        cfg, model, observation, task_description,
                        processor=processor,
                        return_embeddings=True,
                        layer_indices=DESIRED_LAYER_INDICES,
                    )
                    assert set(embeds) == set(DESIRED_LAYER_INDICES), "missing layer in embeds"  # ← NEW
                    # Detect symbolic states **at the SAME time-step**
                    objrel_vec = objrel_dict_to_vec(obj_rel_detector.detect_binary_states())
                    
                    # inside_val = objrel_vec[INSIDE_IDX]               # –1 / 0 / 1
                    # print(f"[t={t}] {INSIDE_KEY} = {inside_val}")
                    # log_file.write(f"[t={t}] {INSIDE_KEY} = {inside_val}\n")
                    
                    assert objrel_vec.shape[0] == NUM_OBJREL
                    assert set(np.unique(objrel_vec)).issubset({-1, 0, 1})
                    actsub_vec = actsub_dict_to_vec(action_det_detector.detect_binary_states())
                    assert actsub_vec.shape[0] == NUM_ACTSUB
                    assert set(np.unique(actsub_vec)).issubset({-1, 0, 1})

                    

                    # Append to episode buffers
                    for L, e in embeds.items():
                        episode_embeds[L].append(e)
                    episode_objrel.append(objrel_vec)
                    episode_actsub.append(actsub_vec)
                    # ----------------------------------------------------------------------------- 


                    # Normalize gripper action [0,1] -> [-1,+1] because the environment expects the latter
                    action = normalize_gripper_action(action, binarize=True)

                    # [OpenVLA] The dataloader flips the sign of the gripper action to align with other datasets
                    # (0 = close, 1 = open), so flip it back (-1 = open, +1 = close) before executing the action
                    if cfg.model_family == "openvla":
                        action = invert_gripper_action(action)

                    # Execute action in environment

                    action_to_send = action.tolist()          # always use the policy’s action

                    obs, reward, done, info = env.step(action_to_send)

                    if done and not success_reached:          # first time we see done=True
                        success_reached   = True
                        steps_after_success = 0
                        task_successes   += 1      # ← NEW
                        total_successes  += 1      # ← NEW
                    
                    if success_reached:
                        steps_after_success += 1
                        if steps_after_success >= EXTRA_STEPS_AFTER_SUCCESS:
                            break        # leave the while-loop after 3 s of extra frames
       
                    t += 1

                except Exception as e:
                    print(f"Caught exception: {e}")
                    log_file.write(f"Caught exception: {e}\n")
                    break

            task_episodes += 1
            total_episodes += 1

            # Save a replay video of the episode
            save_rollout_video(
                replay_images, total_episodes, success=done, task_description=task_description, log_file=log_file
            )

            # ▲ NEW ────────────────────────────────────────────────────────────────────────
            if any(len(v) for v in episode_embeds.values()):
                save_dict = {
                    "visual_semantic_encoding": {L: stack_list(v) for L, v in episode_embeds.items()},
                    "symbolic_state_object_relations": torch.tensor(np.array(episode_objrel)),
                    "symbolic_state_action_subgoals":   torch.tensor(np.array(episode_actsub)),
                }
                os.makedirs(cfg.local_log_dir, exist_ok=True)
                save_fp = os.path.join(cfg.local_log_dir, f"episode_{total_episodes}.pt")
                torch.save(save_dict, save_fp)
                log_file.write(f"Per-action embeddings & states saved to {save_fp}\n")
            # ───────────────────────────────────────────────────────────────────────────────

            # Log current results
            print(f"Success: {done}")
            print(f"# episodes completed so far: {total_episodes}")
            print(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")
            log_file.write(f"Success: {done}\n")
            log_file.write(f"# episodes completed so far: {total_episodes}\n")
            log_file.write(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)\n")
            log_file.flush()

        # Log final results
        print(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        print(f"Current total success rate: {float(total_successes) / float(total_episodes)}")
        log_file.write(f"Current task success rate: {float(task_successes) / float(task_episodes)}\n")
        log_file.write(f"Current total success rate: {float(total_successes) / float(total_episodes)}\n")
        log_file.flush()
        if cfg.use_wandb:
            wandb.log(
                {
                    f"success_rate/{task_description}": float(task_successes) / float(task_episodes),
                    f"num_episodes/{task_description}": task_episodes,
                }
            )

    # Save local log file
    log_file.close()

    # Push total metrics and local log file to wandb
    if cfg.use_wandb:
        wandb.log(
            {
                "success_rate/total": float(total_successes) / float(total_episodes),
                "num_episodes/total": total_episodes,
            }
        )
        wandb.save(local_log_filepath)


if __name__ == "__main__":
    eval_libero()
