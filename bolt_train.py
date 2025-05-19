import argparse
import os
import pickle
import shutil
import math
import numpy as np
import matplotlib.pyplot as plt
from importlib import metadata

try:
    try:
        if metadata.version("rsl-rl"):
            raise ImportError
    except metadata.PackageNotFoundError:
        if metadata.version("rsl-rl-lib") != "2.2.4":
            raise ImportError
except (metadata.PackageNotFoundError, ImportError) as e:
    raise ImportError("Please uninstall 'rsl_rl' and install 'rsl-rl-lib==2.2.4'.") from e
from rsl_rl.runners import OnPolicyRunner

import genesis as gs

from bolt_env import BoltEnv


def get_train_cfg(exp_name, max_iterations):
    train_cfg_dict = {
        "algorithm": {
            "class_name": "PPO",
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.02, 
            "gamma": 0.98,
            "lam": 0.95,
            "learning_rate": 0.00025,  
            "max_grad_norm": 1.0,
            "num_learning_epochs": 8, 
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "init_member_classes": {},
        "policy": {
            "activation": "elu",
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
            "init_noise_std": 1.0,
            "class_name": "ActorCritic",
        },
        "runner": {
            "checkpoint": -1,
            "experiment_name": exp_name,
            "load_run": -1,
            "log_interval": 1,
            "max_iterations": max_iterations,
            "record_interval": -1,
            "resume": False,
            "resume_path": None,
            "run_name": "",
        },
        "runner_class_name": "OnPolicyRunner",
        "num_steps_per_env": 96,  
        "save_interval": 50, 
        "empirical_normalization": None,
        "seed": 1,
    }

    return train_cfg_dict


def get_cfgs():
    env_cfg = {
        "num_actions": 6,  
        
        "default_joint_angles": {  # [rad]
            "FL_HAA": 0.0,
            "FL_HFE": 0.8,
            "FL_KFE": -1.5,
            "FR_HAA": 0.0,
            "FR_HFE": 0.8,
            "FR_KFE": -1.5,
        },
        "joint_names": [
            "FL_HAA",
            "FL_HFE",
            "FL_KFE",
            "FR_HAA",
            "FR_HFE",
            "FR_KFE",
        ],

        "kp": 200.0,
        "kd": 2.0*math.sqrt(200.0),  
 
        "termination_if_roll_greater_than": 30,  
        "termination_if_pitch_greater_than": 30,
        "base_init_pos": [0.0, 0.0, 0.5],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "episode_length_s": 20.0,
        "resampling_time_s": 4.0,
        "action_scale": 0.25,
        "simulate_action_latency": True,
        "clip_actions": 100.0,
    }
    obs_cfg = {
        "num_obs": 27,  # 3 (ang_vel) + 3 (gravity) + 3 (commands) + 6 (dof_pos) + 6 (dof_vel) + 6 (actions)
        "obs_scales": {
            "lin_vel": 2.0,
            "ang_vel": 0.25,
            "dof_pos": 1.0,
            "dof_vel": 0.05,
        },
    }
    reward_cfg = {
        "tracking_sigma": 0.25,
        "base_height_target": 0.5,  
        "reward_scales": {
            "tracking_lin_vel": 2.0,    
            "tracking_ang_vel": 0.2,      
            "lin_vel_z": -2.0,           
            "base_height": -30.0,        
            "action_rate": -0.005,         
            "similar_to_default": -0.05,   
            "orientation_stability": -0.5, 
            "survive": +0.1,  
            # "lin_acc_z": -1.0,             
        },
    }
    command_cfg = {
        "num_commands": 3,
        "lin_vel_x_range": [0.2, 0.2], 
        "lin_vel_y_range": [0, 0],
        "ang_vel_range": [0, 0],
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="bolt-walking")
    parser.add_argument("-B", "--num_envs", type=int, default=16384)
    parser.add_argument("--max_iterations", type=int, default=5000)
    args = parser.parse_args()

    gs.init(logging_level="warning")

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    train_cfg = get_train_cfg(args.exp_name, args.max_iterations)

    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    pickle.dump(
        [env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg],
        open(f"{log_dir}/cfgs.pkl", "wb"),
    )

    env = BoltEnv(
        num_envs=args.num_envs, env_cfg=env_cfg, obs_cfg=obs_cfg, reward_cfg=reward_cfg, command_cfg=command_cfg
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)

    print("=== : 0.2 m/s ===")
    first_stage_iterations = int(args.max_iterations * 0.2)
    env.command_cfg["lin_vel_x_range"] = [0.2, 0.2]
    runner.learn(num_learning_iterations=first_stage_iterations, init_at_random_ep_len=True)
    runner.save(os.path.join(log_dir, "model_stage1.pt"))

    print("=== : 0.3 m/s ===")
    second_stage_iterations = int(args.max_iterations * 0.2)
    env.command_cfg["lin_vel_x_range"] = [0.3, 0.3]
    runner.learn(num_learning_iterations=second_stage_iterations)
    runner.save(os.path.join(log_dir, "model_stage2.pt"))

    print("=== : 0.4 m/s ===")
    third_stage_iterations = int(args.max_iterations * 0.2)
    env.command_cfg["lin_vel_x_range"] = [0.4, 0.4]
    runner.learn(num_learning_iterations=third_stage_iterations)
    runner.save(os.path.join(log_dir, "model_stage3.pt"))

    print("=== : 0.5 m/s ===")
    fourth_stage_iterations = args.max_iterations - first_stage_iterations - second_stage_iterations - third_stage_iterations
    env.command_cfg["lin_vel_x_range"] = [0.5, 0.5]
    runner.learn(num_learning_iterations=fourth_stage_iterations)
    runner.save(os.path.join(log_dir, "model_final.pt"))

    print(f"===  {os.path.join(log_dir, 'model_final.pt')} ===")


if __name__ == "__main__":
    main()

"""

python bolt_train.py
"""