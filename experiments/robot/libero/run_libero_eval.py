"""
run_libero_eval.py

Runs a model in a LIBERO simulation environment.

Usage:
    # OpenVLA:
    # IMPORTANT: Set `center_crop=True` if model is fine-tuned with augmentations
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
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import draccus
import numpy as np
import tqdm
from libero.libero import benchmark

import wandb

import json

import torch

from detection.pick_place_detector import PickPlaceDetector

def serialize_obs(obs):
    """
    Convert observations containing NumPy arrays to JSON-serializable format.
    """
    if isinstance(obs, dict):
        return {k: serialize_obs(v) for k, v in obs.items()}
    elif isinstance(obs, np.ndarray):
        return obs.tolist()
    else:
        return obs


# Append current directory so that interpreter can find experiments.robot
sys.path.append("../..")
from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    quat2axisangle,
    save_rollout_video,
)
from experiments.robot.openvla_utils import get_processor, get_vla_action
from experiments.robot.robot_utils import (
    DATE_TIME,
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)


# Define all possible symbolic state keys
ALL_SYMBOLS = [
 'grasped akita_black_bowl_1',
 'grasped akita_black_bowl_2',
 'grasped cookies_1',
 'grasped glazed_rim_porcelain_ramekin_1',
 'grasped plate_1',
 'on-table akita_black_bowl_1',
 'on-table akita_black_bowl_2',
 'on-table cookies_1',
 'on-table glazed_rim_porcelain_ramekin_1',
 'on-table plate_1',
 'on-table flat_stove_1',
 'on-table wooden_cabinet_1',
 'on akita_black_bowl_1 akita_black_bowl_1',
 'on akita_black_bowl_1 akita_black_bowl_2',
 'on akita_black_bowl_1 cookies_1',
 'on akita_black_bowl_1 glazed_rim_porcelain_ramekin_1',
 'on akita_black_bowl_1 plate_1',
 'on akita_black_bowl_1 flat_stove_1',
 'on akita_black_bowl_1 wooden_cabinet_1',
 'on akita_black_bowl_2 akita_black_bowl_1',
 'on akita_black_bowl_2 akita_black_bowl_2',
 'on akita_black_bowl_2 cookies_1',
 'on akita_black_bowl_2 glazed_rim_porcelain_ramekin_1',
 'on akita_black_bowl_2 plate_1',
 'on akita_black_bowl_2 flat_stove_1',
 'on akita_black_bowl_2 wooden_cabinet_1',
 'on cookies_1 akita_black_bowl_1',
 'on cookies_1 akita_black_bowl_2',
 'on cookies_1 cookies_1',
 'on cookies_1 glazed_rim_porcelain_ramekin_1',
 'on cookies_1 plate_1',
 'on cookies_1 flat_stove_1',
 'on cookies_1 wooden_cabinet_1',
 'on glazed_rim_porcelain_ramekin_1 akita_black_bowl_1',
 'on glazed_rim_porcelain_ramekin_1 akita_black_bowl_2',
 'on glazed_rim_porcelain_ramekin_1 cookies_1',
 'on glazed_rim_porcelain_ramekin_1 glazed_rim_porcelain_ramekin_1',
 'on glazed_rim_porcelain_ramekin_1 plate_1',
 'on glazed_rim_porcelain_ramekin_1 flat_stove_1',
 'on glazed_rim_porcelain_ramekin_1 wooden_cabinet_1',
 'on plate_1 akita_black_bowl_1',
 'on plate_1 akita_black_bowl_2',
 'on plate_1 cookies_1',
 'on plate_1 glazed_rim_porcelain_ramekin_1',
 'on plate_1 plate_1',
 'on plate_1 flat_stove_1',
 'on plate_1 wooden_cabinet_1',
 'on flat_stove_1 akita_black_bowl_1',
 'on flat_stove_1 akita_black_bowl_2',
 'on flat_stove_1 cookies_1',
 'on flat_stove_1 glazed_rim_porcelain_ramekin_1',
 'on flat_stove_1 plate_1',
 'on flat_stove_1 flat_stove_1',
 'on flat_stove_1 wooden_cabinet_1',
 'on wooden_cabinet_1 akita_black_bowl_1',
 'on wooden_cabinet_1 akita_black_bowl_2',
 'on wooden_cabinet_1 cookies_1',
 'on wooden_cabinet_1 glazed_rim_porcelain_ramekin_1',
 'on wooden_cabinet_1 plate_1',
 'on wooden_cabinet_1 flat_stove_1',
 'on wooden_cabinet_1 wooden_cabinet_1',
 'inside akita_black_bowl_1 wooden_cabinet_1_top_region',
 'inside akita_black_bowl_1 wooden_cabinet_1_middle_region',
 'inside akita_black_bowl_1 wooden_cabinet_1_bottom_region',
 'inside akita_black_bowl_2 wooden_cabinet_1_top_region',
 'inside akita_black_bowl_2 wooden_cabinet_1_middle_region',
 'inside akita_black_bowl_2 wooden_cabinet_1_bottom_region',
 'inside cookies_1 wooden_cabinet_1_top_region',
 'inside cookies_1 wooden_cabinet_1_middle_region',
 'inside cookies_1 wooden_cabinet_1_bottom_region',
 'inside glazed_rim_porcelain_ramekin_1 wooden_cabinet_1_top_region',
 'inside glazed_rim_porcelain_ramekin_1 wooden_cabinet_1_middle_region',
 'inside glazed_rim_porcelain_ramekin_1 wooden_cabinet_1_bottom_region',
 'inside plate_1 wooden_cabinet_1_top_region',
 'inside plate_1 wooden_cabinet_1_middle_region',
 'inside plate_1 wooden_cabinet_1_bottom_region',
 'inside flat_stove_1 wooden_cabinet_1_top_region',
 'inside flat_stove_1 wooden_cabinet_1_middle_region',
 'inside flat_stove_1 wooden_cabinet_1_bottom_region',
 'inside wooden_cabinet_1 wooden_cabinet_1_top_region',
 'inside wooden_cabinet_1 wooden_cabinet_1_middle_region',
 'inside wooden_cabinet_1 wooden_cabinet_1_bottom_region',
 'table-center akita_black_bowl_1',
 'table-center akita_black_bowl_2',
 'table-center cookies_1',
 'table-center glazed_rim_porcelain_ramekin_1',
 'table-center plate_1',
 'table-center flat_stove_1',
 'table-center wooden_cabinet_1',
 'pick-up-target akita_black_bowl_1',
 'pick-up-target akita_black_bowl_2',
 'pick-up-target cookies_1',
 'pick-up-target glazed_rim_porcelain_ramekin_1',
 'pick-up-target plate_1',
 'pick-up-target flat_stove_1',
 'pick-up-target wooden_cabinet_1'
]

symbol_to_index = {symbol: idx for idx, symbol in enumerate(ALL_SYMBOLS)}
num_labels = len(ALL_SYMBOLS)

def dict_to_binary_array(symbolic_state_dict):
    binary_array = np.zeros(num_labels, dtype=int)
    for symbol, value in symbolic_state_dict.items():
        if symbol in symbol_to_index:
            binary_array[symbol_to_index[symbol]] = value
        else:
            print(f"Warning: Symbol '{symbol}' not found in `ALL_SYMBOLS`. Ignoring.")
    return binary_array



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
    num_trials_per_task: int = 1                    # Number of rollouts per task

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
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"
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
        assert cfg.unnorm_key in model.norm_stats, f"Action un-norm key {cfg.unnorm_key} not found in VLA `norm_stats`!"

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

        # Initialize LIBERO environment and task description
        env, task_description = get_libero_env(task, cfg.model_family, resolution=256)
        
        core_env = env.env  # Unwrap the core environment

        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
            print(f"\nTask: {task_description}")
            log_file.write(f"\nTask: {task_description}\n")

            # Reset environment
            env.reset()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # # Log initial state predicates
            # if hasattr(core_env, "parsed_problem"):
            #     print("Evaluating initial state predicates:")
            #     for predicate in core_env.parsed_problem.get("initial_state", []):
            #         result = core_env._eval_predicate(predicate)
            #         print(f"  Initial State Predicate {predicate}: {result}")

            detector = PickPlaceDetector(env.env, return_int=True)
            
            # Setup
            t = 0
            replay_images = []
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
            
            # Containers for the episode
            episode_embeddings = []
            episode_symbolic_states = []
            
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

                    # Query model to get action and collect embeddings
                    embeddings, action = get_vla_action(
                        model,
                        processor,
                        cfg.pretrained_checkpoint,
                        observation,
                        task_description,
                        cfg.unnorm_key,
                        center_crop=cfg.center_crop,
                        log_dir=None  # Disable per-step logging
                    )

                    # Collect embeddings for this step
                    if embeddings is not None:
                        # Save the pooled embedding directly
                        episode_embeddings.append(embeddings)  # embeddings is already [4096]
                        # Optional: Check variance
                        embedding_variance = np.var(embeddings)
                        print(f"Embedding Variance: {embedding_variance}")
                    else:
                        # Handle cases where embeddings might be None
                        episode_embeddings.append(np.zeros(4096, dtype=np.float32))


                    # Normalize gripper action [0,1] -> [-1,+1] because the environment expects the latter
                    action = normalize_gripper_action(action, binarize=True)

                    # [OpenVLA] The dataloader flips the sign of the gripper action to align with other datasets
                    # (0 = close, 1 = open), so flip it back (-1 = open, +1 = close) before executing the action
                    if cfg.model_family == "openvla":
                        action = invert_gripper_action(action)

                    # Execute action in environment
                    obs, reward, done, info = env.step(action.tolist())
                    
                    # Detect the symbolic states
                    symbolic_state = detector.detect_binary_states()
                    # print("symbolic_state", symbolic_state)
                    # Convert symbolic_state dict to binary array
                    binary_label = dict_to_binary_array(symbolic_state)
                    episode_symbolic_states.append(binary_label)

                    # if hasattr(core_env, "parsed_problem"):
                    #     print("Parsed problem structure:")
                    #     for key, value in core_env.parsed_problem.items():
                    #         print(f"Key: {key}, Type: {type(value)}, Example: {repr(value)[:200]}")

#                     # Log all object states
#                     if hasattr(core_env, "object_states_dict"):
#                         print(f"Object states after step {t}:")
#                         for obj_name, obj_state in core_env.object_states_dict.items():
#                             print(f"  {obj_name}: {vars(obj_state)}")

#                     # Log goal state predicates
#                     if hasattr(core_env, "_eval_predicate"):
#                         print(f"Goal state predicate evaluations after step {t}:")
#                         for predicate in core_env.parsed_problem.get("goal_state", []):
#                             result = core_env._eval_predicate(predicate)
#                             print(f"  Goal Predicate {predicate}: {result}")
                        
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    print(f"Caught exception: {e}")
                    log_file.write(f"Caught exception: {e}\n")
                    break

            task_episodes += 1
            total_episodes += 1

            # Save per-action data for the episode
            if episode_embeddings and episode_symbolic_states:
                log_file.write(f"Saving per-action data for episode {total_episodes}...\n")
                log_file.flush()
                # Save per-action embeddings and labels
                if cfg.local_log_dir:
                    os.makedirs(cfg.local_log_dir, exist_ok=True)
                    log_file_episode = os.path.join(cfg.local_log_dir, f"episode_{total_episodes}.pt")
                    torch.save({
                        "visual_semantic_encoding": torch.FloatTensor(episode_embeddings),  # Shape: (num_actions, 4096)
                        "symbolic_state": torch.FloatTensor(episode_symbolic_states)       # Shape: (num_actions, 96)
                    }, log_file_episode)
                    print(f"Per-action embeddings and labels saved to: {log_file_episode}")
                    log_file.write(f"Per-action embeddings and labels saved to: {log_file_episode}\n")
            else:
                print("Warning: No embeddings or symbolic states collected for this episode.")
                log_file.write("Warning: No embeddings or symbolic states collected for this episode.\n")
            

            # Save a replay video of the episode
            save_rollout_video(
                replay_images, total_episodes, success=done, task_description=task_description, log_file=log_file
            )

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

    # Push total metrics and local log file to wandb
    if cfg.use_wandb:
        wandb.log(
            {
                "success_rate/total": float(total_successes) / float(total_episodes),
                "num_episodes/total": total_episodes,
            }
        )
        wandb.save(local_log_filepath)
        
    # Save local log file
    log_file.close()


if __name__ == "__main__":
    eval_libero()
