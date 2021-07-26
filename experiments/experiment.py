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

sys.path.append("..")

from common import utils, utils_for_q_learning, buffer_class
from common.logging_utils import MetaLogger

from rainbow.RBFDQN_rainbow import Net
from rainbow.RBFDQN_dis import Net as DistributionalNet

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
    parser.add_argument("--reward_norm",
                        type=str,
                        required=True,
                        default="clip")

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
    params['reward_norm'] = args.reward_norm

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

    print("reward normalization: ", params['reward_norm'])

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

    Q_object_target.eval()

    utils_for_q_learning.sync_networks(target=Q_object_target,
                                       online=Q_object,
                                       alpha=params['target_network_learning_rate'],
                                       copy=True)

    # Logging with Meta Logger

    meta_logger = MetaLogger(full_experiment_name)
    logging_filename = f"seed_{args.seed}.pkl"

    meta_logger.add_field("evaluation_rewards", logging_filename)
    meta_logger.add_field("average_loss", logging_filename)

    G_li = []
    loss_li = []
    all_times_per_steps = []
    all_times_per_updates = []
    for episode in range(params['max_episode']):
        print("episode {}".format(episode))

        s, done, t = env.reset(), False, 0
        while not done:
            a = Q_object.execute_policy(s, episode + 1, 'train')
            sp, r, done, _ = env.step(numpy.array(a))
            t = t + 1
            done_p = False if t == env._max_episode_steps else done
            Q_object.buffer_object.append(s, a, r, done_p, sp)
            s = sp
        # now update the Q network
        loss = []
        for count in range(params['updates_per_episode']):
            temp = Q_object.update(Q_object_target, count)
            loss.append(temp)

        loss_li.append(numpy.mean(loss))
        meta_logger.append_datapoint("average_loss", numpy.mean(loss))

        if (episode % 10 == 0) or (episode == params['max_episode'] - 1):
            temp = []
            for _ in range(10):
                s, G, done, t = env.reset(), 0, False, 0
                while done == False:
                    a = Q_object.execute_policy(s, episode + 1, 'test')
                    sp, r, done, _ = env.step(numpy.array(a))
                    s, G, t = sp, G + r, t + 1
                temp.append(G)
            print(
                "after {} episodes, learned policy collects {} average returns".format(
                    episode, numpy.mean(temp)))
            G_li.append(numpy.mean(temp))
            utils_for_q_learning.save(G_li, loss_li, params, "rbf")
            meta_logger.append_datapoint("evaluation_rewards", numpy.mean(temp), write=True)

        if (params["log"] or ((episode % 50 == 0) or episode == (params['max_episode'] + 1))):
            path = os.path.join(params["full_experiment_file_path"], "logs")
            if not os.path.exists(path):
                try:
                    os.makedirs(path, exist_ok=True)
                except OSError:
                    print("Creation of the directory %s failed" % path)
                else:
                    print("Successfully created the directory %s " % path)
            torch.save(Q_object.state_dict(), os.path.join(path, "episode_" + str(episode)))
            torch.save(Q_object_target.state_dict(), os.path.join(path, "target_episode_" + str(episode)))
