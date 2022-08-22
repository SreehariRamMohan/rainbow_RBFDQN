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
sys.path.insert(1, os.path.join(sys.path[0], '..'))
sys.path.insert(1, os.path.join(sys.path[0], '../scripts/'))
sys.path.insert(1, os.path.join(sys.path[0], '../../GraspInitiation/scripts/'))
sys.path.insert(1, os.path.join(sys.path[0], '../../GraspInitiation/'))

from MujocoGraspEnv import MujocoGraspEnv 

from common import utils, utils_for_q_learning, buffer_class
from common.logging_utils import MetaLogger
from common.mlp_classifier import BinaryMLPClassifier

from rainbow.RBFDQN_rainbow import Net
from rainbow.dis import Net as DistributionalNet

from stochastic_regression import StochasticRegression, load_and_plot_q

import torch
import numpy
import gym

def main():
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

    # By default this is max_steps / 2.
    # parser.add_argument("--noisy_episode_cutoff",
    #                     type=int,
    #                     default=1000,
    #                     help=
    #                     """Specify when noisy sampling should be cutoff""")

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

    parser.add_argument("--gravity",
                    required=False,
                    help="no gravity",
                    type=utils.boolify,
                    default=True) 

    parser.add_argument("--lock_gripper",
                    required=False,
                    help="should we lock the gripper during training?",
                    type=utils.boolify,
                    default=False) 

    parser.add_argument("--sample_method",
                        required=False,
                        help="random, prior, or value",
                        type=str,
                        default="random")

    parser.add_argument("--state_space_type",
                        required=False,
                        help="friendly, pure",
                        type=str,
                        default="friendly")

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

    print("Training on:", args.task, "using sparse reward scheme?", args.reward_sparse, "training with gravity:", args.gravity)

    env = MujocoGraspEnv(args.task, False, reward_sparse=args.reward_sparse, gravity=args.gravity, lock_fingers_closed=args.lock_gripper,
                         sample_method=args.sample_method, state_space=args.state_space_type)
    '''test_env = MujocoGraspEnv(args.task, False, reward_sparse=args.reward_sparse, gravity=args.gravity, lock_fingers_closed=args.lock_gripper,
                              sample_method=args.sample_method, state_space=args.state_space_type)'''

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

    # assign Q object to env for scoring grasps
    try:
        env.set_q_function(Q_object)
    except Exception as e:
        print(e)

    utils_for_q_learning.sync_networks(target=Q_object_target,
                                       online=Q_object,
                                       alpha=params['target_network_learning_rate'],
                                       copy=True)

    if args.demo_regression and args.load_model != "":
        load_and_plot_q(Q_object, args.load_model)
        exit(1)

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
    meta_logger.add_field("episodic_success_rate", logging_filename)
    meta_logger.add_field("grasp_index", logging_filename)

    if args.task == "switch":
        meta_logger.add_field("switch_state", logging_filename)
    elif args.task == "door":
        meta_logger.add_field("door_hinge_state", logging_filename)
        meta_logger.add_field("door_latch_state", logging_filename)


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
        "HumanoidStandup-v2":1000,
        "Door":50,
        "Switch":50,
        "Pitcher":50
    }

    steps_per_typical_episode = env_name_to_steps[params['env_name']]

    steps = 0

    rewards_per_typical_episode = 0

    loss = []
    # Used to train classifier when sample_method is classifier
    task_success_history, grasp_index_history = [], []

    while (steps <  params['max_step']):

        s, done, t = env.reset(), False, 0

        # Dictionary of labels, indexed by grasp index
        classifier_training_dict = {}

        while not done:
            a = Q_object.execute_policy(s, (steps + 1)/steps_per_typical_episode, 'train', steps=(steps+1))

            sp, r, done, info = env.step(numpy.array(a))
            t = t + 1
            rewards_per_typical_episode += r
            done_p = False if t == env._max_episode_steps else done
            Q_object.buffer_object.append(s, a, r, done_p, sp)
            s = sp

            if done:
                if args.sample_method == "classifier":
                    grasp_index = int(info["grasp_index"])
                    success_label = int(info["success"])
                    # Dictionary of latest grasp success label for each grasp index
                    classifier_training_dict[grasp_index] = success_label

                # Episode has terminated, record final object state and task success
                meta_logger.append_datapoint("episodic_success_rate", info["success"], write=True)
                meta_logger.append_datapoint("grasp_index", info["grasp_index"], write=True)

                if args.task == "door":
                    meta_logger.append_datapoint("door_hinge_state", info["door_hinge_state"], write=True)
                    meta_logger.append_datapoint("door_latch_state", info["door_latch_state"], write=True)
                elif args.task == "switch":
                    meta_logger.append_datapoint("switch_state", info["switch_state"], write=True)

            # we do 1 update to the Q network every update_frequency steps.
            if steps%params['update_frequency'] == 0:
                # now update the Q network
                temp, update_params = Q_object.update(Q_object_target)
                loss.append(temp)

            if steps%steps_per_typical_episode == 0:
                # clean up the buffer
                loss_li.append(numpy.mean(loss))
                meta_logger.append_datapoint("average_loss", numpy.mean(loss), write=True)

                meta_logger.append_datapoint("episodic_rewards", rewards_per_typical_episode, write=True)
                rewards_per_typical_episode = 0

                meta_logger.append_datapoint("average_q", update_params['average_q'], write=True)
                meta_logger.append_datapoint("average_q_star", update_params['average_q_star'], write=True)
                loss = []

            '''if (steps%(10000) == 0) or (steps == params['max_step'] - 1):
                temp = []

                success_rate = []

                for _ in range(10):
                    s, G, done, t = test_env.reset(), 0, False, 0
                    while done == False:
                        a = Q_object.execute_policy(s, (steps + 1)/steps_per_typical_episode, 'test', steps=(steps+1))
                        sp, r, done, _ = test_env.step(numpy.array(a))
                        s, G, t = sp, G + r, t + 1
                    temp.append(G)

                    if env.check_task_success():
                        success_rate.append(1.)
                    else:
                        success_rate.append(0.)

                print(
                    "after {} steps, learned policy collects {} average returns with success rate {}".format(
                        steps, numpy.mean(temp), numpy.mean(success_rate)))

                G_li.append(numpy.mean(temp))
                utils_for_q_learning.save(G_li, loss_li, params, "rbf")
                meta_logger.append_datapoint("evaluation_rewards", numpy.mean(temp), write=True)
                meta_logger.append_datapoint("task_success_rate", numpy.mean(success_rate), write=True)
            '''

            if (params["log"] and ((steps % (50000) == 0) or steps == (params['max_step'] - 1))):
                path = os.path.join(params["full_experiment_file_path"], "logs")
                if not os.path.exists(path):
                    try:
                        os.makedirs(path, exist_ok=True)
                    except OSError:
                        print("Creation of the directory %s failed" % path)
                    else:
                        print("Successfully created the directory %s " % path)
                torch.save(Q_object.state_dict(), os.path.join(path, "step_" + str(steps) + "_seed_" + str(args.seed)))

            steps += 1

            if (steps%1000 == 0):
                print("step {}".format(steps))
        # notify n-step that the episode has ended.
        if (params['nstep']):
            Q_object.buffer_object.storage.on_episode_end()

        if args.sample_method == "classifier":

            grasp_indices = classifier_training_dict.keys()
            # List of tensors of lists
            classifier_training_examples = env.cache_torch_state[grasp_indices]
            print("Training exampels:", classifier_training_examples)
            print("Training exampels shape:", classifier_training_examples.shape)
            # List of ints
            classifier_training_labels = [classifier_training_dict[grasp_index] for grasp_index in grasp_indices]

            '''# Convert dictionary mapping grasp indices to success labels to grasp states and success labels
            classifier_training_examples = []
            classifier_training_labels = []
            for grasp_index in classifier_training_dict:
                classifier_training_examples.append(env.cache_torch_state[grasp_index])
                classifier_training_labels.append(classifier_training_dict[grasp_index])
            print("First ex", classifier_training_examples[0])
            print("First ex shape", classifier_training_examples[0].shape)
            classifier_training_examples_tensor = torch.Tensor(len(classifier_training_examples), classifier_training_examples[0].shape[0])
            torch.cat(classifier_training_examples, out=classifier_training_examples_tensor)
            print("Combined tensor", classifier_training_examples_tensor)
            print("Combined tensor shape", classifier_training_examples_tensor.shape)'''

            '''def get_sample_weights():

                pos_egs = flatten(self.positive_examples)
                neg_egs = flatten(self.negative_examples)
                examples = pos_egs + neg_egs

                assigned_labels = np.concatenate((
                    +1 * np.ones((len(pos_egs),)),
                    -1 * np.ones((len(neg_egs),))
                ))

                # Extract what labels the current VF would have assigned
                augmented_states = np.array([eg.info["augmented_state"] for eg in examples])

                # Compute the weights based on the probability that the samples will flip
                weights = self.get_weights(torch.from_numpy(augmented_states), assigned_labels)

                return weights'''


            W = get_weights(classifier_training_examples.to(Q_object.device), classifier_training_labels)

            print("Weights", W)
            sys.exit()

            optimistic_clf = BinaryMLPClassifier

            optimistic_clf.train(classifier_training_examples.to(optimistic_clf.device), classifier_training_labels, W)

            '''training_predictions = self.optimistic_classifier.predict(X)
            positive_training_examples = X[training_predictions == 1]

            if positive_training_examples.shape[0] > 0:
                pessimistic_clf = BinaryMLPClassifier
                self.pessimistic_classifier.fit(positive_training_examples)'''

            # Set weights for agent to draw new examples
            env.classifier_probs = optimistic_clf.predict_proba(env.cache_torch_state.to(optimistic_clf.device))

def _clip(v):
    print("Clipping type", type(v), isinstance(v, np.ndarray))
    if isinstance(v, np.ndarray):
        v[v>0] = 0
        return v
    return v if v <= 0 else 0

def value2steps(value):
    """ Assuming -1 step reward, convert a value prediction to a n_step prediction. """
    gamma = .99
    clipped_value = _clip(value)
    numerator = np.log(1 + ((1-gamma) * np.abs(clipped_value)))
    denominator = np.log(gamma)
    return np.abs(numerator / denominator)

def compute_weights_unbatched(states, labels, values, threshold):
    n_states = states.shape[0]
    weights = np.zeros((n_states,))
    for i in range(n_states):
        label = labels[i]
        state_value = values[i]
        if label == 1:  # These signs are assuming that we are thresholding *steps*, not values.
            flip_mass = state_value[state_value > threshold].sum()
        else:
            flip_mass = state_value[state_value < threshold].sum()
        weights[i] = flip_mass / state_value.sum()
    return weights

 def get_weights(states, labels):
    """
    Given state, threshold, value function, compute the flipping prob for each state
    Return 1/flipping prob which is the weights
        The formula for weights is a little more complicated, see paper Akhil will send in
        channel
    Args:
      states (torch tensor): num states, state_dim
      labels (list[int]): num states
    """
    # Compute updated weights
    """ Get the value distribution for the input states. """
    # shape: (num grasps, 6)
    print("Get weights received states of type", type(states))
    actions = Q_object.get_best_qvalue_and_action(states)[1]
    # shape: (num grasps, 200)
    value_distribution = Q_object.forward(states, actions)

    # We have to mmake sure that the distribution and threshold are in the same units
    step_distribution = value2steps(value_distribution)

    # Determine the threshold. It has units of # steps.
    threshold = np.median(step_distribution)  # TODO: This should be a percentile based on class ratios
    print(f"Set the threshold to {threshold}")

    probabilities = compute_weights_unbatched(states, labels, step_distribution, threshold)
    weights = 1. / (probabilities + 1e-1)
    return weights

if __name__ == '__main__':
    main()
