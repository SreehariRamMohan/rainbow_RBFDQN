import gym
import sys
import time
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy
import pickle
import numpy as np
from common import utils_for_q_learning, buffer_class
from common.noisy_layer import NoisyLinear
import math

from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

from common import utils_for_q_learning
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--hyper_parameter_name",
                    required=True,
                    help="0, 10, 20, etc. Corresponds to .hyper file",
                    default="0")  # OpenAI gym environment name

parser.add_argument("--seed", default=0, help="seed",
                    type=int)  # Sets Gym, PyTorch and Numpy seeds


parser.add_argument("--run_title",
                    type=str,
                    help="subdirectory for this run",
                    required=True)
                    
args, unknown = parser.parse_known_args()
    
params = utils_for_q_learning.get_hyper_parameters(args.hyper_parameter_name, "rbf")
breakpoint()
env = gym.make(params['env_name'])        

# number of steps on average per task
env_name_to_steps = {
    "Pendulum-v0": 200, 
    "LunarLanderContinuous-v2":200,
    "BipedalWalker-v3": 1600,
    "Hopper-v3": 1000,
    "HalfCheetah-v3": 1000,
    "Ant-v3": 1000,
    "Humanoid-v2":1000,
    "Walker2d-v2":1000
}

# The noise objects for DDPG
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

model = DDPG("MlpPolicy", env, action_noise=action_noise, verbose=1)

model.set_random_seed(args.seed);

model.learn(total_timesteps=params['max_episode']*env_name_to_steps[params['env_name']], eval_freq=10*env_name_to_steps[params['env_name']], n_eval_episodes=10)

#model.save(args.run_title)