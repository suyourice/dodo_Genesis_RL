import argparse
import os
import pickle
from importlib import metadata
import torch

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="bolt-walking")
    parser.add_argument("--ckpt", type=int, default=1496)
    parser.add_argument("-v", "--vel", type=float, default=0.5, help="Target forward velocity")
    parser.add_argument("-y", "--yvel", type=float, default=0.0, help="Target side velocity")
    parser.add_argument("-r", "--rot", type=float, default=0.0, help="Target rotation velocity")
    args = parser.parse_args()

    gs.init()

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(open(f"{log_dir}/cfgs.pkl", "rb"))
    
    # Set command velocities for evaluation
    command_cfg["lin_vel_x_range"] = [args.vel, args.vel]
    command_cfg["lin_vel_y_range"] = [args.yvel, args.yvel]
    command_cfg["ang_vel_range"] = [args.rot, args.rot]
    
    # Disable rewards for evaluation
    reward_cfg["reward_scales"] = {}
    
    env = BoltEnv(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=True,
    )
    
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    resume_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")
    runner.load(resume_path)
    
    policy = runner.get_inference_policy(device=gs.device)
    
    obs, _ = env.reset()
    
    with torch.no_grad():
        while True:
            actions = policy(obs)
            obs, rews, dones, infos = env.step(actions)
            
            # Print current velocity to monitor training progress
            if env.episode_length_buf[0] % 100 == 0:
                print(f"Command vel: [{env.commands[0, 0]:.2f}, {env.commands[0, 1]:.2f}, {env.commands[0, 2]:.2f}]")
                print(f"Current vel: [{env.base_lin_vel[0, 0]:.2f}, {env.base_lin_vel[0, 1]:.2f}, {env.base_ang_vel[0, 2]:.2f}]")


if __name__ == "__main__":
    main()

"""
# evaluation
python bolt_eval.py -e bolt-walking --ckpt 100 -v 0.5
# or with custom velocities:
# python bolt_eval.py -e bolt-walking --ckpt 100 -v 0.8 -y 0.2 -r 0.3
"""