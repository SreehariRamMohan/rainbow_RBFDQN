"""
This is our run file for all Rainbow RBFDQN experiments.
"""
import argparse
import os
import datetime

from common import utils, utils_for_q_learning, buffer_class
from common.logging_utils import MetaLogger

from rainbow.RBFDQN_rainbow import Net

import torch
import numpy
import gym
import sys

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
                        action="store_true",
                        default=False,
                        help="run using double q network")

    parser.add_argument("--nstep", 
                    type=int,
                    default=-1,
                    help="run using multi-step returns of size n")


    args, unknown = parser.parse_known_args()
    other_args = {(utils.remove_prefix(key, '--'), val)
                  for (key, val) in zip(unknown[::2], unknown[1::2])}

    full_experiment_name = os.path.join(args.experiment_name, args.run_title)

    utils.create_log_dir(full_experiment_name)
    hyperparams_dir = utils.create_log_dir(
        os.path.join(full_experiment_name, "hyperparams"))

    params = utils_for_q_learning.get_hyper_parameters(args.hyper_parameter_name, "rbf")

    params['hyper_parameters_name'] = args.hyper_parameter_name

    for arg_name, arg_value in other_args:
        utils.update_param(params, arg_name, arg_value)
    params['hyperparams_dir'] = hyperparams_dir
    params['start_time'] = str(datetime.datetime.now())
    params['seed_number'] = args.seed

    # Rainbow RBF-DQN improvements
    params['double'] = args.double
    print("Double:", params["double"])

    if (args.nstep != -1):
        params["nstep"] = True
        params["nstep_size"] = args.nstep
    else:
        params["nstep"] = False
        params["nstep_size"] = -1 # note using multi step returns.
    
    print("Nstep:", params["nstep"], "Nstep_size:", params["nstep_size"])


    # continue adding Rainbow RBFDQN flags for ablations here.

    utils.save_hyper_parameters(params, args.seed)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Running on the GPU")
    else:
        device = torch.device("cpu")
        print("Running on the CPU")
   
    env = gym.make(params["env_name"])

    params['env'] = env

    if len(sys.argv) > 3:
        params['save_prepend'] = str(sys.argv[3])
        print("Save prepend is ", params['save_prepend'])

    utils_for_q_learning.set_random_seed(params)
    s0 = env.reset()
    utils_for_q_learning.action_checker(env)
    
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
    Q_object_target.eval()

    utils_for_q_learning.sync_networks(target=Q_object_target,
                                       online=Q_object,
                                       alpha=params['target_network_learning_rate'],
                                       copy=True)

    G_li = []
    loss_li = []
    all_times_per_steps = []
    all_times_per_updates = []
    for episode in range(params['max_episode']):
        print("episode {}".format(episode))

        s, done, t = env.reset(), False, 0
        while not done:
            if params['policy_type'] == 'e_greedy':
                a = Q_object.e_greedy_policy(s, episode + 1, 'train')
            elif params['policy_type'] == 'e_greedy_gaussian':
                a = Q_object.e_greedy_gaussian_policy(s, episode + 1, 'train')
            elif params['policy_type'] == 'gaussian':
                a = Q_object.gaussian_policy(s, episode + 1, 'train')
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

        if (episode % 10 == 0) or (episode == params['max_episode'] - 1):
            temp = []
            for _ in range(10):
                s, G, done, t = env.reset(), 0, False, 0
                while done == False:
                    a = Q_object.e_greedy_policy(s, episode + 1, 'test')
                    sp, r, done, _ = env.step(numpy.array(a))
                    s, G, t = sp, G + r, t + 1
                temp.append(G)
            print(
                "after {} episodes, learned policy collects {} average returns".format(
                    episode, numpy.mean(temp)))
            G_li.append(numpy.mean(temp))
            utils_for_q_learning.save(G_li, loss_li, params, "rbf")
        # TODO use meta logger to save the model state dict every 10 episodes
        # if ((episode%10==0) or episode == (params['max_episode'] + 1)):
        #     torch.save(Q_object.state_dict(), 
        #     'logs/double_hyper_' + str(params['hyper_parameters_name']) + '_' + str(episode) + "_seed_" + str(params['seed_number']))
        #     torch.save(Q_object_target.state_dict(), 
        #     'logs/double_target_hyper_' + str(params['hyper_parameters_name']) + '_' + str(episode) + "_seed_" + str(params['seed_number']))
