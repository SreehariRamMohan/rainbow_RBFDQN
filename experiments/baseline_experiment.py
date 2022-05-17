import gym
import numpy as np
from pathlib import Path

from stable_baselines3 import DDPG
from stable_baselines3 import TD3
from stable_baselines3 import SAC
from stable_baselines3 import PPO

from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.logger import configure

from common import utils, utils_for_q_learning
import argparse

import sys
sys.path.append("..")
sys.path.append("../scripts/")
sys.path.append("../../scripts")

from MujocoGraspEnv import MujocoGraspEnv 

'''
stable_baselines3 version 0.11.1 does not require you to call set_logger() for the output
to be properly written. In fact you cannot call set_logger because the models do not even have
a set_logger method. 

stable_baselines3 version 1.1.0 requires a call to set_logger(logger) for output to properly be written
You would do this after line 86; model.set_logger(logger)
'''

parser = argparse.ArgumentParser()

parser.add_argument("--agent",
                    required=True,
                    type=str,
                    default="DDPG")  # OpenAI gym environment name

parser.add_argument("--task",
                    required=True,
                    help="door, switch",
                    type=str,
                    default="door")  # OpenAI gym environment name

parser.add_argument("--reward_sparse",
                    required=True,
                    help="is the reward sparse?",
                    type=utils.boolify,
                    default=True) 

parser.add_argument("--seed", help="seed",
                    type=int, required=True)  # Sets Gym, PyTorch and Numpy seeds

parser.add_argument("--run_title",
                    type=str,
                    help="subdirectory for this run",
                    required=True)
                    
args, unknown = parser.parse_known_args()
env = MujocoGraspEnv(args.task, False, reward_sparse=args.reward_sparse, gravity=True) 
eval_env = MujocoGraspEnv(args.task, False, reward_sparse=args.reward_sparse, gravity=True) 

# The noise objects for DDPG
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

assert args.agent in ["DDPG", "PPO", "SAC", "TD3"]
model = None

# make eval log dir

directory_to_make = "./baseline_results/" + args.agent + "/" + args.task + "_isRewardSparse_" + str(args.reward_sparse) + "/" + args.run_title + "_seed_" + str(args.seed)

Path(directory_to_make).mkdir(parents=True, exist_ok=True)
logger = configure(directory_to_make, ["stdout", "csv", "log", "tensorboard", "json"])

if args.agent == "DDPG":
    model = DDPG("MlpPolicy", env, seed=args.seed, action_noise=action_noise, verbose=1, learning_starts=500,batch_size=512, buffer_size=500000, train_freq=(1, "step"), gradient_steps=1,)
elif args.agent == "PPO":
    model = PPO("MlpPolicy", env, seed=args.seed, verbose=1, learning_starts=500,batch_size=512, buffer_size=500000, train_freq=(1, "step"), gradient_steps=1,)
elif args.agent == "SAC":
    model = SAC("MlpPolicy", env, seed=args.seed, action_noise=action_noise, verbose=2, learning_starts=500,batch_size=512, buffer_size=500000, train_freq=(1, "step"), gradient_steps=1,)
elif args.agent == "TD3":
    model = TD3("MlpPolicy", env, seed=args.seed, action_noise=action_noise, verbose=1, learning_starts=500,batch_size=512, buffer_size=500000, train_freq=(1, "step"), gradient_steps=1,)

model.set_random_seed(args.seed)
model.set_logger(logger)
model.learn(total_timesteps=2000000, 
            eval_freq=10000, 
            n_eval_episodes=10, eval_log_path=directory_to_make, eval_env=eval_env)

model.save(directory_to_make + "/" + args.agent)