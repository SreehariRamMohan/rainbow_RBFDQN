"""
This is our run file for all Rainbow RBFDQN experiments.
"""
"""
This is our run file for all Rainbow RBFDQN experiments.
"""
import argparse
import os
import datetime
import sys
from sklearn.decomposition import PCA
sys.path.append("..")
import matplotlib.pyplot as plt
from common import utils, utils_for_q_learning, buffer_class
from common.logging_utils import MetaLogger

from rainbow.RBFDQN_rainbow import Net
from rainbow.RBFDQN_dis import Net as DistributionalNet
from pathlib import Path
import glob
import torch
import numpy
import gym

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hyper_param_directory",
                        required=False,
                        default="./hyper_parameters",
                        type=str)

    parser.add_argument("--hyper_parameter_name",
                        required=False,
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

    parser.add_argument("--log", action="store_true")

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

    parser.add_argument("--distributional",
                        type=utils.boolify,
                        default=False,
                        help=
                        """Use Distributional RBF-DQN""")

    parser.add_argument("--logged_hyperparams_dir",
                        type=str,
                        help="we need that to load the hyper file we are using for the training.",
                        default="results/HalfCheetah/distributional_sweep/hyperparams/40__seed_0.hyper",
                        required=False)

    parser.add_argument("--log_centroid_location",
                        type=utils.boolify,
                        default=False,
                        help=
                        """logging centroid location""")

    args, unknown = parser.parse_known_args()
    other_args = {(utils.remove_prefix(key, '--'), val)
                  for (key, val) in zip(unknown[::2], unknown[1::2])}

    full_experiment_name = os.path.join(args.experiment_name, args.run_title)

    utils.create_log_dir(full_experiment_name)
    hyperparams_dir = utils.create_log_dir(
        os.path.join(full_experiment_name, "hyperparams"))

    params = utils_for_q_learning.get_hyper_parameters_after_training(args.logged_hyperparams_dir)

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
    # continue adding Rainbow RBFDQN flags for ablations here.

    if (args.mean and params['dueling']):
        params["dueling_combine_operator"] = "mean"
    else:
        params["dueling_combine_operator"] = "max"

    print("Dueling:", params['dueling'], "Combine Operator:", params['dueling_combine_operator'])

    params['layer_normalization'] = args.layer_normalization
    params['noisy_layers'] = args.noisy_layers

    print("Layer Normalizaton: ", params['layer_normalization'], "Noisy Layers: ", params['noisy_layers'])

    utils.save_hyper_parameters(params, args.seed)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Running on the GPU")
    else:
        device = torch.device("cpu")
        print("Running on the CPU")

    env = gym.make(params["env_name"])

    params['env'] = env

    utils_for_q_learning.set_random_seed(params)
    s0 = env.reset()
    utils_for_q_learning.action_checker(env)

    if not params['distributional']:
        Q_object = Net(params,
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
    # here please fill in the main directory for your stored model
    saved_network_dir = "./results/HalfCheetah/distributional_sweep/logs/"
    # specify the model that will actually interact with the environment.
    actor = "target_episode_1700_seed_0"
    Q_object.load_state_dict(torch.load(saved_network_dir + actor))
    Q_object.eval()


    if args.log_centroid_location:
        Q_objects = {}
        glob_pattern = saved_network_dir + "target*"
        for network in glob.glob(glob_pattern):
            key = network[network.rfind('/')+1:]
            if not params['distributional']:
                Q_objects[key] = Net(params,
                                         env,
                                         state_size=len(s0),
                                         action_size=len(env.action_space.low),
                                         device=device)
            else:
                Q_objects[key] = DistributionalNet(params,
                                                       env,
                                                       state_size=len(s0),
                                                       action_size=len(env.action_space.low),
                                                       device=device)
                Q_objects[key].load_state_dict(torch.load(network))
                Q_objects[key].eval()


    # Logging with Meta Logger

    meta_logger = MetaLogger(full_experiment_name)
    logging_filename = f"seed_{args.seed}.pkl"

    meta_logger.add_field("evaluation_rewards", logging_filename)
    meta_logger.add_field("average_loss", logging_filename)

    G_li = []
    loss_li = []
    all_times_per_steps = []
    all_times_per_updates = []

    num_episodes = 1
    num_success = 0
    for j in range(num_episodes):
        print("episode:", j)
        obs = env.reset()

        for i in range(2000):
            action = Q_object.e_greedy_policy(obs, j + 1, 'test')
            #action = env.action_space.sample()
            if args.log_centroid_location:
                obs = numpy.array(obs).reshape(1, len(s0))
                obs = torch.from_numpy(obs).float().to(device)
                for key in Q_objects:

                    with torch.no_grad():
                        all_centroids = Q_objects[key].get_centroid_locations(obs).cpu().numpy().squeeze()
                    pca = PCA(n_components=2)
                    data = pca.fit_transform(all_centroids)
                    plt.scatter(data[:,0],data[:,1] , label = key)
                    plt.legend(loc='upper left')
                    plt.savefig("centroid_graph/" +str(i)+"_step_"+key)
                    plt.clf()


            obs, reward, done, info = env.step(action)
            print("timestep:",i , "getting reward:", reward)

            #print("action taken:", action, "finished? ", done)

            #env.render()
            if done:
                num_success += 1
                break
