import argparse
import logging
import os
import datetime
import sys
sys.path.append("..")
sys.path.append("../..")
sys.path.append("../../../.")
sys.path.append("../../scripts")

sys.path.append("../scripts/")

from common import utils, utils_for_q_learning, buffer_class
from common.logging_utils import MetaLogger
from rainbow.RBFDQN_rainbow import Net
from rainbow.dis import Net as DistributionalNet
from stochastic_regression import StochasticRegression, load_and_plot_q
import torch
import numpy as np
import numpy
import gym
from gym import spaces
import time
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO

from MujocoGraspEnv import MujocoGraspEnv 
from stable_baselines3 import SAC


RENDER = True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hyper_param_directory",
                        required=False,
                        default="./hyper_parameters",
                        type=str)

    parser.add_argument("--hyper_parameter_name",
                        required=True,
                        help="0, 10, 20, etc. Corresponds to .hyper file",
                        default="0")  # OpenAI gym environment name

    parser.add_argument("--seed", default=0, help="seed",
                        type=int)  # Sets Gym, PyTorch and Numpy seeds

    parser.add_argument("--experiment_name",
                        type=str,
                        help="Experiment Name",
                        required=True)

    parser.add_argument("--run_title",
                        type=str,
                        help="subdirectory for this run",
                        required=True)

    parser.add_argument("--double",
                        type=utils.boolify,
                        default=False,
                        help="run using double DQN")

    parser.add_argument("--nstep",
                        type=int,
                        default=1,
                        help="run using multi-step returns of size n")

    parser.add_argument("--per",
                        type=utils.boolify,
                        default=False,
                        help="run using Priority Experience Replay (PER)")

    parser.add_argument("--dueling",
                        type=utils.boolify,
                        default=False,
                        help="run using dueling architecture")

    parser.add_argument("--mean",
                        type=utils.boolify,
                        default=False,
                        help=
                        """use the mean combine operator to
                        combine advantage and base value
                        (otherwise max is used by default""")

    parser.add_argument("--layer_normalization",
                        type=utils.boolify,
                        default=False,
                        help=
                        """apply normalization immediately
                        prior to activations in any hidden layers""")

    parser.add_argument("--noisy_layers",
                        type=utils.boolify,
                        default=False,
                        help=
                        """use noisy linear layers instead of linear
                        layers for all hidden layers""")

    parser.add_argument("--noisy_where",
                        type=str,
                        default="both",
                        choices=["both","centroid","value"],
                        help=
                        """Specify where noisy layers should be inserted:
                        centroid, value, or both""")

    parser.add_argument("--distributional",
                        type=utils.boolify,
                        default=False,
                        help=
                        """Use Distributional RBF-DQN""")

    parser.add_argument("--alpha", default=0.1, help="alpha",
                        type=float)  # alpha for PER

    parser.add_argument("--per_beta_start", default=0.4, help="beta for per",type=float)  # beta for PER

    parser.add_argument("--should_schedule_beta",
                        type=utils.boolify,
                        default=True,
                        help=
                        """Whether to anneal the value of beta from per_beta_start to 1.0 over the course of training""")
    parser.add_argument("--loss_type",
                        type=str,
                        default="MSELoss",
                        help=
                        """there are two types of loss we can use, MSELoss or HuberLoss""")

    parser.add_argument("--learning_rate",
                        type=float,
                        default=0)
    parser.add_argument("--num_points",
                        type=int,
                        default=0)
    parser.add_argument("--vmin",
                        type=float,
                        default=0)
    parser.add_argument("--vmax",
                        type=float,
                        default=0)
    parser.add_argument("--learning_rate_location_side",
                        type=float,
                        default=0)
    parser.add_argument("--target_network_learning_rate",
                        type=float,
                        default=0)
    parser.add_argument("--regularize_centroid_spread",
                        type=utils.boolify,
                        default=False)
    parser.add_argument("--regularize_centroid_spread_parameter",
                        type=float,
                        default=0)
    parser.add_argument("--regularize_centroid_central_parameter",
                        type=float,
                        default=0)
    parser.add_argument("--regularize_centroid_central",
                        type=utils.boolify,
                        default=False)

    parser.add_argument("--temperature",
                        type=float,
                        required=False)

    parser.add_argument("--log", action="store_true")

    parser.add_argument("--sigma_noise", default=0.5, help="sigma",
                        type=float) ## Noise standard deviation
    parser.add_argument("--policy_type", ## If not specified, reads the policy_type
                        type=str,        ## from the hyperparam file. Otherwise, overrides
                        required=False,  ## with the value from the commandline
                        default="unset")

    parser.add_argument('--which_gpu', '-gpu_id', default=0)

    parser.add_argument("--demo_regression", type=utils.boolify, default=False, required=False)
    parser.add_argument("--load_model", type=str, default="", required=False)

    # use randomly initialized betas for all centroids (fixed) throughout training. 
    parser.add_argument("--random_betas", 
                        type=utils.boolify, 
                        required=False,
                        default=False)

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

    parser.add_argument("--SAC", action="store_true")

    args, unknown = parser.parse_known_args()
    other_args = {(utils.remove_prefix(key, '--'), val)
                  for (key, val) in zip(unknown[::2], unknown[1::2])}

    full_experiment_name = os.path.join(args.experiment_name, args.run_title)

    utils.create_log_dir(full_experiment_name)
    hyperparams_dir = utils.create_log_dir(
        os.path.join(full_experiment_name, "hyperparams"))
    params = utils_for_q_learning.get_hyper_parameters(args.hyper_parameter_name, "rbf")
    params['hyper_parameters_name'] = args.hyper_parameter_name
    params['full_experiment_file_path'] = os.path.join(os.getcwd(), full_experiment_name)

    for arg_name, arg_value in other_args:
        utils.update_param(params, arg_name, arg_value)

    params['hyperparams_dir'] = hyperparams_dir
    params['start_time'] = str(datetime.datetime.now())
    params['seed_number'] = args.seed
    params['log'] = args.log
    params['per'] = args.per
    params['dueling'] = args.dueling
    params['distributional'] = args.distributional
    params['alpha'] = args.alpha
    params['per_beta_start'] = args.per_beta_start
    params['should_schedule_beta'] = args.should_schedule_beta
    params['loss_type'] = args.loss_type
    params['random_betas'] = args.random_betas
    params['noisy_episode_cutoff'] = params['max_step']/2

    # change hyper parameters from command line
    if args.learning_rate:
        params['learning_rate'] = args.learning_rate
    if args.num_points:
        params['num_points'] = args.num_points
    if args.vmin:
        params['vmin'] = args.vmin
    if args.vmax:
        params['vmax'] = args.vmax
    if args.learning_rate_location_side:
        params['learning_rate_location_side'] = args.learning_rate_location_side
    if args.target_network_learning_rate:
        params['target_network_learning_rate'] = args.target_network_learning_rate
    if args.regularize_centroid_spread_parameter:
        params['regularize_centroid_spread_parameter'] = args.regularize_centroid_spread_parameter

    params['regularize_centroid_spread'] = args.regularize_centroid_spread
    if args.regularize_centroid_central_parameter:
        params['regularize_centroid_central_parameter'] = args.regularize_centroid_central_parameter

    if args.temperature:
        params['temperature'] = args.temperature

    params['regularize_centroid_central'] = args.regularize_centroid_central
    #params['beta'] = args.beta
    params['sigma_noise'] = args.sigma_noise
    if args.policy_type != "unset":
        params['policy_type'] = args.policy_type
    print("Distributional:", params["distributional"])

    # Rainbow RBF-DQN improvements
    params['double'] = args.double
    print("Double:", params["double"])

    if (args.nstep != 1):
        params["nstep"] = True
        params["nstep_size"] = args.nstep
    else:
        params["nstep"] = False
        params["nstep_size"] = -1  # note using multi step returns.

    print("Nstep:", params["nstep"], "Nstep_size:", params["nstep_size"])

    print("PER:", params["per"])
    if params["per"]:
        print("PER alpha:",params['alpha'],"PER beta:",params['per_beta_start'])
        print("Scheduling PER Beta to go from per_beta_start-->1.0:", params['should_schedule_beta'])

    if (args.mean and params['dueling']):
        params["dueling_combine_operator"] = "mean"
    else:
        params["dueling_combine_operator"] = "max"

    print("Dueling:", params['dueling'], "Combine Operator:", params['dueling_combine_operator'])

    params['layer_normalization'] = args.layer_normalization
    params['noisy_layers'] = args.noisy_layers
    if params['noisy_layers']:
        params["layer_normalization"] = True
    params['noisy_where'] = args.noisy_where

    print("Layer Normalizaton: ", params['layer_normalization'], "Noisy Layers: ", params['noisy_layers'])
    print("Noisy Layers Applied to: ", params['noisy_where'])

    utils.save_hyper_parameters(params, args.seed)
    
    if torch.cuda.is_available():
        device = torch.device("cuda:" + str(args.which_gpu))
        print("Using GPU id {}".format(args.which_gpu))
    else:
        device = torch.device("cpu")
        print("Running on the CPU")

    print("Training on:", args.task, "using sparse reward scheme?", args.reward_sparse)
    env = MujocoGraspEnv(args.task, True, reward_sparse=args.reward_sparse, gravity=True) #gym.make(params["env_name"])
    test_env = MujocoGraspEnv(args.task, True, reward_sparse=args.reward_sparse, gravity=True) # gym.make(params["env_name"])

    # seed 
    utils_for_q_learning.set_random_seed({"seed_number" : args.seed, "env" : env})

    
    s0 = env.reset()

    if not params['distributional']:
        Q_object = Net(params,
                       env,
                       state_size=len(s0),
                       action_size=len(env.action_space.low),
                       device=device)
        Q_object_target = Net(params,
                              env,
                              state_size=len(s0),
                              action_size=len(env.action_space.low),
                              device=device)
    else:
        Q_object = DistributionalNet(params,
                                     env,
                                     state_size=len(s0),
                                     action_size=len(env.action_space.low),
                                     device=device)
        Q_object_target = DistributionalNet(params,
                                            env,
                                            state_size=len(s0),
                                            action_size=len(env.action_space.low),
                                            device=device)

    # here please fill in the main directory for your stored model
    # step_400000_seed_2_door_dense
    # door_sparse_saved_model_0
    # step_1100000_seed_2_switch_dense
    
    if not args.SAC:
        saved_network_dir = "/home/sreehari/Downloads/step_650000_seed_0_door_rainbow"
        # specify the model that will actually interact with the environment.
        Q_object.load_state_dict(torch.load(saved_network_dir))
        Q_object.eval()
        Q_object_target.eval()
        utils_for_q_learning.sync_networks(target=Q_object_target,
                                        online=Q_object,
                                        alpha=params['target_network_learning_rate'],
                                        copy=True)

        if args.demo_regression and args.load_model != "":
            load_and_plot_q(Q_object, args.load_model)
            exit(1)

    # switch_dense_sac.zip
    
    model = SAC.load("/home/sreehari/Downloads/sac_best_door.zip")

    
    print(f"Q_object: {Q_object}")
    print(f"Q_target: {Q_object_target}")

    # Logging with Meta Logger

    meta_logger = MetaLogger(full_experiment_name)
    logging_filename = f"seed_{args.seed}.pkl"

    meta_logger.add_field("evaluation_rewards", logging_filename)
    meta_logger.add_field("episodic_rewards", logging_filename)
    meta_logger.add_field("average_loss", logging_filename)
    meta_logger.add_field("average_q", logging_filename)
    meta_logger.add_field("average_q_star", logging_filename)
    meta_logger.add_field("task_success_rate", logging_filename)

    G_li = []
    loss_li = []
    all_times_per_steps = []
    all_times_per_updates = []

    print("using step based training.")

    # number of steps on average per episode
    env_name_to_steps = {
        "Pendulum-v0": 200, 
        "LunarLanderContinuous-v2":200,
        "BipedalWalker-v3": 1600,
        "Hopper-v3": 1000,
        "HalfCheetah-v3": 1000,
        "Ant-v3": 1000,
        "Humanoid-v2":1000,
        "Walker2d-v2":1000,
        "demo_regression":100,
        "Door":50,
        "Switch":50
    }

    steps_per_typical_episode = env_name_to_steps[params['env_name']]

    steps = 0

    rewards_per_typical_episode = 0

    loss = []
    
        # Logging with Meta Logger

    meta_logger = MetaLogger(full_experiment_name)
    logging_filename = f"seed_{args.seed}.pkl"

    meta_logger.add_field("evaluation_rewards", logging_filename)
    meta_logger.add_field("average_loss", logging_filename)

    G_li = []
    loss_li = []
    all_times_per_steps = []
    all_times_per_updates = []

    num_episodes = 20
    num_success = 0

    for j in range(num_episodes):
        print("episode:", j)
        obs = env.reset()
        obs = numpy.array(obs).reshape(1, len(s0))
        obs = torch.from_numpy(obs).float().to(device)
        
        for i in range(50):
            with torch.no_grad():
                if args.SAC:
                    action = model.predict(obs.cpu().numpy())[0].squeeze()
                else:
                    action = Q_object.get_best_qvalue_and_action(obs)[1].squeeze()
                    action = action.cpu().numpy()
            obs, reward, done, info = env.step(action)
            obs = numpy.array(obs).reshape(1, len(s0))
            obs = torch.from_numpy(obs).float().to(device)
            print("timestep:",i , "getting reward:", reward)
            if done:
                num_success += 1
                break
