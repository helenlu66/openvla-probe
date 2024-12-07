import os
import sys
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import draccus
import numpy as np
import torch
import wandb

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from detection.pick_place_detector import PickPlaceDetector
from libero.libero import benchmark

# Append current directory to find experiments.robot
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
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)

app = FastAPI()

@dataclass
class GenerateConfig:
    # Model-specific parameters
    model_family: str = "openvla"                    # Model family
    pretrained_checkpoint: Union[str, Path] = ""     # Pretrained checkpoint path
    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)

    # LIBERO environment-specific parameters
    task_suite_name: str = "libero_spatial"          # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 1                     # Number of rollouts per task

    # Utils
    run_id_note: Optional[str] = None                # Extra note to add in run ID for logging
    local_log_dir: str = "./experiments/logs"        # Local directory for eval logs

    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_project: str = "YOUR_WANDB_PROJECT"        # Name of W&B project to log to (use default!)
    wandb_entity: str = "YOUR_WANDB_ENTITY"          # Name of entity to log under

    seed: int = 7                                    # Random Seed (for reproducibility)

# Define all possible symbolic state keys
ALL_SYMBOLS = [
    'grasped akita_black_bowl_1',
    'on akita_black_bowl_1 akita_black_bowl_1',
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

# Initialize global variables
model = None
processor = None
task_suite = None
benchmark_dict = None
resize_size = None
cfg = None
wandb_run = None
log_file = None

def get_max_steps(task_suite_name: str) -> int:
    max_steps_mapping = {
        "libero_spatial": 220,  # longest training demo has 193 steps
        "libero_object": 280,   # longest training demo has 254 steps
        "libero_goal": 300,     # longest training demo has 270 steps
        "libero_10": 520,       # longest training demo has 505 steps
        "libero_90": 400,       # longest training demo has 373 steps
    }
    return max_steps_mapping.get(task_suite_name, 300)  # default to 300 if not found

@app.on_event("startup")
def startup_event():
    global model, processor, task_suite, benchmark_dict, resize_size, cfg, wandb_run, log_file

    # Initialize configuration
    cfg = GenerateConfig(
        # You can set default values here or make them configurable via environment variables or config files
        pretrained_checkpoint="openvla/openvla-7b-finetuned-libero-spatial",
        task_suite_name="libero_spatial",
        center_crop=True,
        use_wandb=False,  # Set to True if you want to use W&B
        # Add other configurations as needed
    )

    # Set random seed
    set_seed_everywhere(cfg.seed)

    # Set action un-normalization key
    cfg.unnorm_key = cfg.task_suite_name

    # Load model
    model = get_model(cfg)

    # Check that the model contains the action un-normalization key
    if cfg.model_family == "openvla":
        if cfg.unnorm_key not in model.norm_stats and f"{cfg.unnorm_key}_no_noops" in model.norm_stats:
            cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"
        assert cfg.unnorm_key in model.norm_stats, f"Action un-norm key {cfg.unnorm_key} not found in VLA `norm_stats`!"

    # Get Hugging Face processor
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)

    # Initialize logging
    run_id = f"EVAL-{cfg.task_suite_name}-{cfg.model_family}-{DATE_TIME}"
    if cfg.run_id_note:
        run_id += f"--{cfg.run_id_note}"
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    print(f"Logging to local log file: {local_log_filepath}")

    # Initialize Weights & Biases logging
    if cfg.use_wandb:
        wandb_run = wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=run_id,
        )

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    print(f"Task suite: {cfg.task_suite_name}")
    log_file.write(f"Task suite: {cfg.task_suite_name}\n")

    # Get expected image dimensions
    resize_size = get_image_resize_size(cfg)

    print("Server startup complete.")
    log_file.write("Server startup complete.\n")
    log_file.flush()

@app.on_event("shutdown")
def shutdown_event():
    global log_file, wandb_run
    if log_file:
        log_file.write("Server shutting down.\n")
        log_file.close()
    if wandb_run:
        wandb_run.finish()
    print("Server shutdown complete.")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Client connected.")
    try:
        while True:
            # Wait for a message from the client
            data = await websocket.receive_text()
            message = json.loads(data)
            task_id = message.get("task_id")
            run_id = message.get("run_id_note", None)  # Optional run ID note

            if task_id is None:
                await websocket.send_text(json.dumps({"error": "No task_id provided."}))
                continue

            if not isinstance(task_id, int):
                await websocket.send_text(json.dumps({"error": "task_id must be an integer."}))
                continue

            # Validate task_id
            if task_id < 0 or task_id >= task_suite.n_tasks:
                await websocket.send_text(json.dumps({"error": f"Invalid task_id: {task_id}. Must be between 0 and {task_suite.n_tasks - 1}."}))
                continue

            # Optionally, update run_id_note if provided
            if run_id:
                current_run_id = f"EVAL-{cfg.task_suite_name}-{cfg.model_family}-{DATE_TIME}"
                current_run_id += f"--{run_id}"
                log_file.write(f"Run ID Note Updated: {run_id}\n")
                run_id_note = run_id
            else:
                current_run_id = f"EVAL-{cfg.task_suite_name}-{cfg.model_family}-{DATE_TIME}"

            # Get task
            task = task_suite.get_task(task_id)

            # Get initial states
            initial_states = task_suite.get_task_init_states(task_id)

            # Initialize LIBERO environment and task description
            env, task_description = get_libero_env(task, cfg.model_family, resolution=256)
            core_env = env.env  # Unwrap the core environment

            print(f"Received task_id: {task_id}, task_description: {task_description}")
            log_file.write(f"Received task_id: {task_id}, task_description: {task_description}\n")
            log_file.flush()

            # Reset environment
            env.reset()

            # Set initial states
            obs = env.set_init_state(initial_states[0])  # Assuming num_trials_per_task=1

            detector = PickPlaceDetector(env.env, return_int=True)

            # Setup
            t = 0
            replay_images = []
            max_steps = get_max_steps(cfg.task_suite_name)

            print(f"Starting task {task_id}: {task_description}")
            log_file.write(f"Starting task {task_id}: {task_description}\n")
            log_file.flush()

            # Containers for the episode
            episode_embeddings = []
            episode_symbolic_states = []
            success = False

            while t < max_steps + cfg.num_steps_wait:
                try:
                    # Do nothing for the first few timesteps
                    if t < cfg.num_steps_wait:
                        obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                        t += 1
                        continue

                    # Get preprocessed image
                    img = get_libero_image(obs, resize_size)

                    # Save preprocessed image for replay video
                    replay_images.append(img)

                    # Prepare observations dict
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
                        episode_embeddings.append(embeddings)  # embeddings is already [4096]
                        embedding_variance = np.var(embeddings)
                        print(f"Embedding Variance: {embedding_variance}")
                    else:
                        episode_embeddings.append(np.zeros(4096, dtype=np.float32))

                    # Normalize gripper action [0,1] -> [-1,+1]
                    action = normalize_gripper_action(action, binarize=True)

                    # Invert gripper action if using OpenVLA
                    if cfg.model_family == "openvla":
                        action = invert_gripper_action(action)

                    # Execute action in environment
                    obs, reward, done, info = env.step(action.tolist())

                    # Detect the symbolic states
                    symbolic_state = detector.detect_binary_states()
                    binary_label = dict_to_binary_array(symbolic_state)
                    episode_symbolic_states.append(binary_label)

                    if done:
                        success = True
                        break
                    t += 1

                except Exception as e:
                    print(f"Caught exception: {e}")
                    log_file.write(f"Caught exception: {e}\n")
                    log_file.flush()
                    await websocket.send_text(json.dumps({"error": str(e)}))
                    break

            # Save per-action data for the episode
            if episode_embeddings and episode_symbolic_states:
                log_file.write(f"Saving per-action data for episode.\n")
                log_file.flush()
                log_file_episode = os.path.join(cfg.local_log_dir, f"episode_{task_id}.pt")
                torch.save({
                    "visual_semantic_encoding": torch.FloatTensor(episode_embeddings),  # Shape: (num_actions, 4096)
                    "symbolic_state": torch.FloatTensor(episode_symbolic_states)       # Shape: (num_actions, 96)
                }, log_file_episode)
                print(f"Per-action embeddings and labels saved to: {log_file_episode}")
                log_file.write(f"Per-action embeddings and labels saved to: {log_file_episode}\n")
                log_file.flush()
            else:
                print("Warning: No embeddings or symbolic states collected for this episode.")
                log_file.write("Warning: No embeddings or symbolic states collected for this episode.\n")
                log_file.flush()

            # Save a replay video of the episode
            replay_video_path = save_rollout_video(
                replay_images, task_id, success=success, task_description=task_description, log_file=log_file
            )

            # Log current results
            print(f"Success: {success}")
            log_file.write(f"Success: {success}\n")
            log_file.flush()

            # Prepare response
            response = {
                "task_id": task_id,
                "task_description": task_description,
                "success": success,
                "embeddings_saved_at": log_file_episode if episode_embeddings else None,
                "replay_video_saved_at": replay_video_path  # Assuming save_rollout_video returns the path
            }

            # Send response back to client
            await websocket.send_text(json.dumps(response))

    except WebSocketDisconnect:
        print("Client disconnected.")
    except Exception as e:
        print(f"Error: {e}")
        log_file.write(f"Error: {e}\n")
        log_file.flush()
