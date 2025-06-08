### dodo_train.py ###
import argparse
import os
import pickle
import shutil
import math
import numpy as np
from importlib import metadata
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
import genesis as gs
from rsl_rl.runners import OnPolicyRunner

from dodo_env import DodoEnv

import wandb
import os


from dataclasses import dataclass

@dataclass
class EnvWrapper:
    env_cfg:    dict
    obs_cfg:    dict
    reward_cfg: dict
    command_cfg: dict



# 全局列表，记录各个指标随迭代的变化
iters = []
val_loss = []
surrogate_loss = []
noise_std = []
total_reward = []
ep_length = []
rew_lin_vel = []
rew_ang_vel = []
rew_vel_z = []
rew_base_h = []
rew_act_rate = []
rew_sim_def = []
rew_ori_stab = []
rew_survive = []


def wandb_log(step, stats):
    wandb.log({
        "value_loss":                stats["value_loss"],
        "surrogate_loss":            stats["surrogate_loss"],
        "action_noise_std":          stats["action_noise_std"],
        "episode_reward_mean":       stats["episode_reward_mean"],
        "episode_length_mean":       stats["episode_length_mean"],
        "rew_tracking_lin_vel":      stats["rew_tracking_lin_vel"],
        "rew_tracking_ang_vel":      stats["rew_tracking_ang_vel"],
        "rew_lin_vel_z":             stats["rew_lin_vel_z"],
        "rew_base_height":           stats["rew_base_height"],
        "rew_action_rate":           stats["rew_action_rate"],
        "rew_similar_to_default":    stats["rew_similar_to_default"],
        "rew_orientation_stability": stats["rew_orientation_stability"],
        "rew_survive":               stats["rew_survive"],
    }, step=step)


def log_and_plot(log_dir, it, stats):
    iters.append(it)
    wandb_log(it, stats)
    val_loss.append(stats["value_loss"])
    surrogate_loss.append(stats["surrogate_loss"])
    noise_std.append(stats["action_noise_std"])
    total_reward.append(stats["episode_reward_mean"])
    ep_length.append(stats["episode_length_mean"])
    rew_lin_vel.append(stats["rew_tracking_lin_vel"])
    rew_ang_vel.append(stats["rew_tracking_ang_vel"])
    rew_vel_z.append(stats["rew_lin_vel_z"])
    rew_base_h.append(stats["rew_base_height"])
    rew_act_rate.append(stats["rew_action_rate"])
    rew_sim_def.append(stats["rew_similar_to_default"])
    rew_ori_stab.append(stats["rew_orientation_stability"])
    rew_survive.append(stats["rew_survive"])

    if it % 100 == 0:
        # 建立一个多子图的 figure
        fig, axes = plt.subplots(4, 4, figsize=(16, 12))
        axes = axes.flatten()

        # 每个子图画一条曲线
        axes[0].plot(iters, val_loss);           axes[0].set_title("Value Loss")
        axes[1].plot(iters, surrogate_loss);     axes[1].set_title("Surrogate Loss")
        axes[2].plot(iters, noise_std);          axes[2].set_title("Action Noise Std")
        axes[3].plot(iters, total_reward);       axes[3].set_title("Mean Total Reward")
        axes[4].plot(iters, ep_length);          axes[4].set_title("Mean Episode Length")
        axes[5].plot(iters, rew_lin_vel);        axes[5].set_title("rew_tracking_lin_vel")
        axes[6].plot(iters, rew_ang_vel);        axes[6].set_title("rew_tracking_ang_vel")
        axes[7].plot(iters, rew_vel_z);          axes[7].set_title("rew_lin_vel_z")
        axes[8].plot(iters, rew_base_h);         axes[8].set_title("rew_base_height")
        axes[9].plot(iters, rew_act_rate);       axes[9].set_title("rew_action_rate")
        axes[10].plot(iters, rew_sim_def);       axes[10].set_title("rew_similar_to_default")
        axes[11].plot(iters, rew_ori_stab);      axes[11].set_title("rew_orientation_stability")
        axes[12].plot(iters, rew_survive);       axes[12].set_title("rew_survive")

        # 如果还剩一个空子图，可以留白
        # axes[13].axis("off")

        for ax in axes:
            ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()

        # 保存（覆盖之前的同名文件）
        save_path = os.path.join(log_dir, f"metrics.png")
        fig.savefig(save_path)
        plt.close(fig)
        print(f"[Plot] saved to {save_path}")


def get_train_cfg(exp_name, max_iterations):
    train_cfg_dict = {
        "algorithm": {
            "class_name": "PPO",
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.02,
            "gamma": 0.98,
            "lam": 0.95,
            "learning_rate": 0.0002,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 8,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
            
        },
        "init_member_classes": {},
        "policy": {
            #"activation": "mish",
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
        "logger": "tensorboard",
        "tensorboard_subdir": "tb",

    }
    return train_cfg_dict


def get_cfgs():
    # Environment config
    env_cfg = {
        "num_actions": 8,
        "default_joint_angles": { name: 0.0 for name in [
            "Left_HIP_AA","Right_HIP_AA","Left_THIGH_FE","Right_THIGH_FE",
            "Left_KNEE_FE","Right_SHIN_FE","Left_FOOT_ANKLE","Right_FOOT_ANKLE"]
        },
        "joint_names": [
            "Left_HIP_AA","Right_HIP_AA","Left_THIGH_FE","Right_THIGH_FE",
            "Left_KNEE_FE","Right_SHIN_FE","Left_FOOT_ANKLE","Right_FOOT_ANKLE",
        ],
        "kp": 200.0,
        "kd": 2.0 * math.sqrt(200.0),
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
    # Observation config
    obs_cfg = {
        "num_obs": 3 + 3 + 3 + env_cfg["num_actions"] + env_cfg["num_actions"] + env_cfg["num_actions"],
        "obs_scales": {
            "lin_vel": 2.0,
            "ang_vel": 0.25,
            "dof_pos": 1.0,
            "dof_vel": 0.05,
        },
    }
    # Reward config
    reward_cfg = {
        "tracking_sigma": 0.25,
        "base_height_target": 0.38,
        "reward_scales": {
            "tracking_lin_vel": 5.0,
            "tracking_ang_vel": 3.7,
            "lin_vel_z": -3.0,
            "base_height": -180.0,
            "action_rate": -0.01,
            "similar_to_default": -0.01,
            "orientation_stability": -5.8,
            "survive": +0.15,
            "penalize_hip_aa"      : -3.5,
            "penalize_hip_fe"    : -0.00,
            "penalize_hip_fe_diff"   : -0.8,
            "penalize_knee_fe_left"   : -0.5,
            "penalize_knee_fe_right": -0.5,
            "penalize_ankle_height": -0.5,
            "step_height_consistency": 0.5,    
            "gait_regularity": 0.8,            
            "foot_orientation": 1.5, 
            "foot_contact_penalty": -2.0,
            "foot_contact_switch" : 1.0
        },
    }
    # Command config
    command_cfg = {
        "num_commands": 3,
        "lin_vel_x_range": [0.2, 0.2],
        "lin_vel_y_range": [0.0, 0.0],
        "ang_vel_range": [0.0, 0.0],
    }
    return env_cfg, obs_cfg, reward_cfg, command_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="dodo-walking")
    parser.add_argument("-B", "--num_envs", type=int, default=16384)
    parser.add_argument("--max_iterations", type=int, default=2500)
    args = parser.parse_args()

    gs.init(logging_level="warning")

    
    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    train_cfg = get_train_cfg(args.exp_name, args.max_iterations)

    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    # Save configs
    pickle.dump(

        [env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg],
        open(f"{log_dir}/cfgs.pkl", "wb"),
    )

    env = DodoEnv(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
    )
    env.cfg = EnvWrapper(env_cfg, obs_cfg, reward_cfg, command_cfg)

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)

    # Curriculum stages
    stages = [0.1, 0.3, 0.4, 0.5]
    total_it = 0
    for i, v in enumerate(stages, start=1):
        iters = int(args.max_iterations * (0.2 if i < 4 else 1 - sum(stages[:3])))
        print(f"=== Stage {i}: {v} m/s ===")
        env.command_cfg["lin_vel_x_range"] = [v, v]
        runner.learn(num_learning_iterations=iters, init_at_random_ep_len=(i==1))
        fname = f"model_stage{i}.pt" if i < 4 else "model_final.pt"
        runner.save(os.path.join(log_dir, fname))

    print(f"=== Trained model saved at {log_dir}/model_final.pt ===")


if __name__ == "__main__":
    main()