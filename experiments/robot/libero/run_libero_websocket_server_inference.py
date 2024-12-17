import os
import sys
import json
import base64
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import cv2
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import uvicorn
from libero.libero import benchmark
import random
import ast
import joblib  # Added joblib import

from fastapi.middleware.cors import CORSMiddleware  # Added CORS middleware

# Append parent directory to path
sys.path.append("../..")
from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    quat2axisangle,
)
from experiments.robot.openvla_utils import get_processor, get_vla_action
from experiments.robot.robot_utils import (
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)
from detection.libero_spatial_object_relation_detector import LiberoSpatialObjectRelationDetector
from detection.libero_spatial_action_state_subgoal_detector import LiberoSpatialActionDetector

# Define the desired layer indices for embedding collection
DESIRED_LAYER_INDICES = [32]

# Define the paths to the symbolic state key files
object_relations_file = "/root/openvla/experiments/robot/libero/spatial_object_relations_keys.txt"
action_subgoals_file = "/root/openvla/experiments/robot/libero/spatial_action_states_keys.txt"  # Corrected

# Read and parse object relations symbols
with open(object_relations_file, "r") as file:
    object_relations_content = file.read().strip()
ALL_OBJECT_RELATIONS_SYMBOLS = ast.literal_eval(object_relations_content)

# Read and parse action subgoals symbols
with open(action_subgoals_file, "r") as file:
    action_subgoals_content = file.read().strip()
ALL_ACTION_SUBGOALS_SYMBOLS = ast.literal_eval(action_subgoals_content)

# Create separate symbol-to-index mappings
object_relations_symbol_to_index = {symbol: idx for idx, symbol in enumerate(ALL_OBJECT_RELATIONS_SYMBOLS)}
num_object_relations_labels = len(ALL_OBJECT_RELATIONS_SYMBOLS)

action_subgoals_symbol_to_index = {symbol: idx for idx, symbol in enumerate(ALL_ACTION_SUBGOALS_SYMBOLS)}
num_action_subgoals_labels = len(ALL_ACTION_SUBGOALS_SYMBOLS)

def dict_to_binary_array_object_relations(symbolic_state_dict):
    binary_array = np.zeros(num_object_relations_labels, dtype=int)
    for symbol, value in symbolic_state_dict.items():
        if symbol in object_relations_symbol_to_index:
            binary_array[object_relations_symbol_to_index[symbol]] = value
        else:
            print(f"Warning: Symbol '{symbol}' not found in `ALL_OBJECT_RELATIONS_SYMBOLS`. Ignoring.")
    return binary_array

def dict_to_binary_array_action_subgoals(symbolic_state_dict):
    binary_array = np.zeros(num_action_subgoals_labels, dtype=int)
    for symbol, value in symbolic_state_dict.items():
        if symbol in action_subgoals_symbol_to_index:
            binary_array[action_subgoals_symbol_to_index[symbol]] = value
        else:
            print(f"Warning: Symbol '{symbol}' not found in `ALL_ACTION_SUBGOALS_SYMBOLS`. Ignoring.")
    return binary_array

@dataclass
class ServerConfig:
    # Model parameters
    model_family: str = "openvla"
    pretrained_checkpoint: Union[str, Path] = ""
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    center_crop: bool = True

    # Environment parameters
    task_suite_name: str = "libero_spatial"
    num_steps_wait: int = 10
    resolution: int = 256

    # Server parameters
    host: str = "0.0.0.0"
    port: int = 8000
    seed: int = 7

# Define paths to your trained probes and label files
PROBE_PATHS = {
    'symbolic_state_object_relations': "/root/probe/probe_symbolic_state_object_relations_layer32.joblib",
    'symbolic_state_action_subgoals': "/root/probe/probe_symbolic_state_action_subgoals_layer32.joblib",
    # Add more detectors and their probe paths as needed
}

LABEL_FILES = {
    'symbolic_state_object_relations': "/root/openvla/experiments/robot/libero/spatial_object_relations_keys.txt",
    'symbolic_state_action_subgoals': "/root/openvla/experiments/robot/libero/spatial_action_states_keys.txt",  # Corrected
    # Add more detectors and their label files as needed
}

# Initialize FastAPI with CORS
app = FastAPI()

# Configure CORS (allow all origins for development; restrict in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this in production for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
model = None
processor = None
env = None
task_suite = None
task_description_to_task = {}
task_id_mapping = {}
object_relations_detector = None
action_detector = None

# Global dictionaries to store probes and label mappings
probes = {}
label_mappings = {}

def load_probe(probe_path):
    """
    Load a trained probe from a joblib file.
    """
    try:
        pipeline = joblib.load(probe_path)
        print(f"Successfully loaded probe from {probe_path}")
        return pipeline
    except Exception as e:
        print(f"Error loading probe from {probe_path}: {e}")
        return None

def load_label_names(detector, label_file_path):
    """
    Load label names from a specified file.
    """
    with open(label_file_path, "r") as file:
        label_content = file.read().strip()
    all_symbols = ast.literal_eval(label_content)
    return list(all_symbols)

def perform_single_inference(pipeline, embedding, label_names, target_labels):
    """
    Perform inference on a single embedding and filter for target labels.
    """
    if embedding.ndim == 1:
        embedding = embedding.reshape(1, -1)  # Reshape to (1, hidden_dim)

    try:
        prediction = pipeline.predict(embedding)  # Shape: (1, num_labels)
        prediction = prediction[0]  # Get the first (and only) prediction
    except Exception as e:
        print(f"Error during inference: {e}")
        return []

    # Map predictions to label names
    predicted_labels = [label for label, pred in zip(label_names, prediction) if pred == 1]

    # Filter for target labels
    filtered_labels = [label for label in predicted_labels if label in target_labels]

    return filtered_labels

def map_predictions_to_labels(predictions, label_names):
    """
    Map binary predictions to their corresponding label names.
    """
    predicted_labels = []
    for sample_pred in predictions:
        labels = [label for label, pred in zip(label_names, sample_pred) if pred == 1]
        predicted_labels.append(labels)
    return predicted_labels

# Define target labels
TARGET_LABELS = ["grasped akita_black_bowl_1", "grasped akita_black_bowl_2"]

def init_server(cfg: ServerConfig):
    """Initialize model, processor, probes, and label mappings."""
    global model, processor, task_suite, task_description_to_task, task_id_mapping
    global object_relations_detector, action_detector
    global probes, label_mappings

    print("\n" + "="*50)
    print("[*] Starting server initialization...")
    print("="*50)
    
    # Set random seed
    set_seed_everywhere(cfg.seed)
    print(f"[*] Random seed set to {cfg.seed}")

    # Set action un-normalization key
    cfg.unnorm_key = cfg.task_suite_name

    # Load model
    print("[*] Loading model...")
    model = get_model(cfg)

    # Load processor and check normalization key for OpenVLA
    if cfg.model_family == "openvla":
        print("[*] Initializing OpenVLA processor...")
        processor = get_processor(cfg)
        
        # Check normalization key
        if cfg.unnorm_key not in model.norm_stats and f"{cfg.unnorm_key}_no_noops" in model.norm_stats:
            cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"
        assert cfg.unnorm_key in model.norm_stats, f"Action un-norm key {cfg.unnorm_key} not found in VLA `norm_stats`!"
        print(f"[*] Using normalization key: {cfg.unnorm_key}")

    # Initialize benchmark and task suite
    print(f"\n[*] Initializing task suite: {cfg.task_suite_name}")
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks = task_suite.n_tasks
    print(f"[*] Found {num_tasks} tasks in suite")
    
    # Create mapping from task descriptions to tasks and ids
    print("\n[*] Available tasks:")
    print("-"*50)
    task_id_mapping = {}  # Keep track of task IDs
    for task_id in range(task_suite.n_tasks):
        task = task_suite.get_task(task_id)
        task_description_to_task[task.language] = task
        task_id_mapping[task.language] = task_id  # Store task ID
        print(f"Task {task_id}: {task.language}")
    print("-"*50 + "\n")
    
    # Load probes
    print("\n[*] Loading trained probes for inference...")
    for detector, probe_path in PROBE_PATHS.items():
        print(f"[*] Loading probe for detector: {detector} from {probe_path}")
        pipeline = load_probe(probe_path)
        if pipeline:
            probes[detector] = pipeline
            print(f"[+] Successfully loaded probe for detector: {detector}")
        else:
            print(f"[!] Failed to load probe for detector: {detector}")
    
    # Load label mappings
    print("\n[*] Loading label mappings...")
    for detector, label_file in LABEL_FILES.items():
        labels = load_label_names(detector, label_file)
        label_mappings[detector] = labels
        print(f"Loaded {len(labels)} labels for detector '{detector}'.")
    
    # Log loaded probes
    print("\n[*] Loaded Probes:")
    for detector in probes.keys():
        print(f"  - {detector}")
    
    # Specific check for 'symbolic_state_action_subgoals'
    if 'symbolic_state_action_subgoals' in probes:
        print("[+] 'symbolic_state_action_subgoals' probe is loaded and ready.")
    else:
        print("[!] 'symbolic_state_action_subgoals' probe is NOT loaded.")
    
    print("="*50 + "\n")

async def send_image_to_client(websocket: WebSocket, img_array, step: int, status: str = "running"):
    """Send observation image to client efficiently using OpenCV"""
    try:
        # Convert to uint8 if needed
        if img_array.dtype != np.uint8:
            img_array = (img_array * 255).astype(np.uint8)
            
        # Ensure correct color format (RGB to BGR for OpenCV)
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
        # Encode directly to JPEG bytes
        success, encoded_img = cv2.imencode('.jpg', img_array, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not success:
            raise ValueError("Failed to encode image")
            
        # Convert to base64
        img_b64 = base64.b64encode(encoded_img.tobytes()).decode()
        
        # Send to client
        await websocket.send_text(json.dumps({
            "status": status,
            "step": step,
            "image": img_b64
        }))
    except Exception as e:
        print(f"[!] Error sending image: {e}")
        await websocket.send_text(json.dumps({
            "error": f"Failed to send image: {str(e)}"
        }))

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("\n[*] Client connected")
    
    try:
        while True:
            # Receive task description from client
            data = await websocket.receive_text()
            print(f"\n[*] Received task request: {data}")
            
            # Look up task
            task = task_description_to_task.get(data)
            if not task:
                error_msg = f"Task description not found: {data}"
                print(f"[!] {error_msg}")
                print("[*] Available tasks:")
                for desc in task_description_to_task.keys():
                    print(f"  - {desc}")
                await websocket.send_text(json.dumps({
                    "error": error_msg
                }))
                continue

            # Get the task ID from the mapping
            task_id = task_id_mapping[data]
            
            print(f"\n[*] Starting task execution...")
            print("-"*50)
            print(f"Task description: {data}")
            print(f"Task ID: {task_id}")
                
            # Initialize environment and get initial state
            print("[*] Initializing environment...")
            env, task_description = get_libero_env(task, model_family=config.model_family, resolution=config.resolution)
            
            # Initialize detectors
            core_env = env.env  # Unwrap the core environment
            object_relations_detector = LiberoSpatialObjectRelationDetector(core_env, return_int=True)
            action_detector = LiberoSpatialActionDetector(core_env, return_int=True)
            
            # Get initial states for the specific task
            initial_states = task_suite.get_task_init_states(task_id)
            print(f"[*] Got {len(initial_states)} initial states for task {task_id}")
            
            # Reset and set initial state
            env.reset()
            obs = env.set_init_state(initial_states[random.randint(0, len(initial_states) - 1)])
            print("[*] Environment initialized with initial state")
            
            # Setup execution
            t = 0
            done = False
            resize_size = get_image_resize_size(config)
            
            # Get max steps based on task suite
            if config.task_suite_name == "libero_spatial":
                max_steps = 220
            elif config.task_suite_name == "libero_object":
                max_steps = 280
            elif config.task_suite_name == "libero_goal":
                max_steps = 300
            elif config.task_suite_name == "libero_10":
                max_steps = 520
            elif config.task_suite_name == "libero_90":
                max_steps = 400
            print(f"[*] Max steps for this task: {max_steps}")

            # Initialize containers for collecting data
            episode_embeddings = {layer_idx: [] for layer_idx in DESIRED_LAYER_INDICES}
            episode_symbolic_states_object_relations = []
            episode_symbolic_states_action_subgoals = []
            
            try:
                # Main execution loop
                print("\n[*] Starting execution loop...")
                print("-"*50)
                while t < max_steps + config.num_steps_wait and not done:
                    # Get image first for consistent timing
                    img = get_libero_image(obs, resize_size)
                    
                    # Initial stabilization period
                    if t < config.num_steps_wait:
                        print(f"[*] Stabilization step {t}/{config.num_steps_wait}")
                        await send_image_to_client(websocket, img, t, f"Stabilizing environment ({t}/{config.num_steps_wait})")
                        obs, reward, done, info = env.step(get_libero_dummy_action(config.model_family))
                        t += 1
                        continue

                    print(f"\n[*] Step {t}")
                    
                    # Send current observation to client
                    await send_image_to_client(websocket, img, t, f"Executing step {t}")

                    # Prepare observation for model
                    observation = {
                        "full_image": img,
                        "state": np.concatenate(
                            (obs["robot0_eef_pos"], 
                             quat2axisangle(obs["robot0_eef_quat"]), 
                             obs["robot0_gripper_qpos"])
                        ),
                    }

                    # Get action and embeddings from model
                    embeddings_dict, action = get_vla_action(
                        model,
                        processor,
                        config.pretrained_checkpoint,
                        observation,
                        task_description,
                        config.unnorm_key,
                        center_crop=config.center_crop,
                        layer_indices=DESIRED_LAYER_INDICES,
                        log_dir=None,
                        pooling_method='final_token'
                    )

                    # Collect embeddings and perform inference
                    for layer_idx, embedding in embeddings_dict.items():
                        episode_embeddings[layer_idx].append(embedding)
                        
                        # Define mapping from layer to detector
                        # Update if multiple detectors are used
                        LAYER_TO_DETECTOR = {
                            32: 'symbolic_state_action_subgoals',
                            # Add more mappings as needed
                        }
                        
                        detector = LAYER_TO_DETECTOR.get(layer_idx)
                        if not detector:
                            print(f"[!] No detector mapped for layer {layer_idx}. Skipping inference.")
                            continue
                        
                        if detector in probes:
                            pipeline = probes[detector]
                            labels = label_mappings[detector]
                            predicted_labels = perform_single_inference(pipeline, embedding, labels, TARGET_LABELS)
                            
                            # Structure inference result
                            inference_result = {
                                "detector": detector,
                                "layer": layer_idx,
                                "predicted_labels": predicted_labels
                            }
                            
                            # Print inference result on server
                            print(f"[Inference] Detector: {detector}, Layer: {layer_idx}, Predicted Labels: {predicted_labels}")
                            
                            # Send inference result to client
                            await websocket.send_text(json.dumps({
                                "inference": inference_result
                            }))
                        else:
                            print(f"[!] No probe found for detector: {detector}")

                    # Process action
                    action = normalize_gripper_action(action, binarize=True)
                    if config.model_family == "openvla":
                        action = invert_gripper_action(action)

                    print(f"Executing action: {action.tolist()}")
                    
                    # Execute action
                    obs, reward, done, info = env.step(action.tolist())
                    
                    # Detect symbolic states
                    object_relations = object_relations_detector.detect_binary_states()
                    action_subgoals = action_detector.detect_binary_states()
                    
                    # Convert to binary arrays
                    binary_label_object_relations = dict_to_binary_array_object_relations(object_relations)
                    binary_label_action_subgoals = dict_to_binary_array_action_subgoals(action_subgoals)
                    
                    # Append to lists
                    episode_symbolic_states_object_relations.append(binary_label_object_relations)
                    episode_symbolic_states_action_subgoals.append(binary_label_action_subgoals)

                    print(f"Result: done={done}, reward={reward}")
                    
                    # Check if task completed
                    if done:
                        print("\n[*] Task completed successfully!")
                        # Send final successful image
                        await send_image_to_client(
                            websocket,
                            img,
                            t,
                            status=f"Task completed successfully in {t} steps!"
                        )
                        break
                        
                    t += 1

                # If we hit max steps without completion
                if not done:
                    print("\n[!] Reached max steps without completion")
                    await send_image_to_client(
                        websocket,
                        img,
                        t,
                        status=f"Task reached max steps ({t} steps) without completion"
                    )

                # Save collected data if needed
                if episode_embeddings and episode_symbolic_states_object_relations and episode_symbolic_states_action_subgoals:
                    # Implement data saving logic here if required
                    pass

                print("-"*50)
                print("[*] Task execution finished\n")

            except Exception as e:
                print(f"[!] Error during task execution: {str(e)}")
                await websocket.send_text(json.dumps({
                    "error": f"Task execution failed: {str(e)}"
                }))
            
    except WebSocketDisconnect:
        print("[*] Client disconnected")
    except Exception as e:
        print(f"[!] Error: {str(e)}")
        await websocket.send_text(json.dumps({
            "error": str(e)
        }))

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_family", type=str, default="openvla")
    parser.add_argument("--pretrained_checkpoint", type=str, required=True)
    parser.add_argument("--task_suite_name", type=str, default="libero_spatial")
    parser.add_argument("--center_crop", type=bool, default=True)
    parser.add_argument("--host", type=str, default="0.0.0.0") 
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument("--load_in_4bit", action="store_true")
    args = parser.parse_args()
    
    if args.load_in_8bit and args.load_in_4bit:
        raise ValueError("Cannot use both 8-bit and 4-bit quantization")
        
    # Create config from args
    config = ServerConfig(
        model_family=args.model_family,
        pretrained_checkpoint=args.pretrained_checkpoint,
        task_suite_name=args.task_suite_name,
        center_crop=args.center_crop,
        host=args.host,
        port=args.port,
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit
    )
    
    # Initialize server
    init_server(config)
    
    # Start server
    print(f"\n[*] Starting server on {config.host}:{config.port}")
    uvicorn.run(app, host=config.host, port=config.port)
