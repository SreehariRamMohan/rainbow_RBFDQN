import gym
import numpy as np
from pathlib import Path

from stable_baselines3 import DDPG
from stable_baselines3 import TD3
from stable_baselines3 import SAC
from stable_baselines3 import PPO

from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.logger import configure

from common import utils_for_q_learning
import argparse

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

parser.add_argument("--hyper_parameter_name",
                    required=True,
                    help="0, 10, 20, etc. Corresponds to .hyper file",
                    default="0")  # OpenAI gym environment name

parser.add_argument("--seed", help="seed",
                    type=int, required=True)  # Sets Gym, PyTorch and Numpy seeds


parser.add_argument("--run_title",
                    type=str,
                    help="subdirectory for this run",
                    required=True)
                    
args, unknown = parser.parse_known_args()
    
params = utils_for_q_learning.get_hyper_parameters(args.hyper_parameter_name, "rbf")
env = gym.make(params['env_name'])        

# number of steps on average per episode
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

assert args.agent in ["DDPG", "PPO", "SAC", "TD3"]
model = None

# make eval log dir

directory_to_make = "./baseline_results/" + args.agent + "/" + args.run_title + "_" + args.hyper_parameter_name + "_seed_" + str(args.seed)

Path(directory_to_make).mkdir(parents=True, exist_ok=True)
logger = configure(directory_to_make, ["stdout", "csv", "log", "tensorboard", "json"])

if args.agent == "DDPG":
    model = DDPG("MlpPolicy", env, action_noise=action_noise, verbose=1)
elif args.agent == "PPO":
    model = PPO("MlpPolicy", env, verbose=1)
elif args.agent == "SAC":
    model = SAC("MlpPolicy", env, action_noise=action_noise, verbose=1)
elif args.agent == "TD3":
    model = TD3("MlpPolicy", env, action_noise=action_noise, verbose=1)

model.set_random_seed(args.seed)

model.learn(total_timesteps=params['max_episode']*env_name_to_steps[params['env_name']], 
            eval_freq=10*env_name_to_steps[params['env_name']], 
            n_eval_episodes=10) #eval_log_path=directory_to_make)