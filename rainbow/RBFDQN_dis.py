'''
Passing all possible actions (centroids) into the value network
'''

import gym
import sys
import time
import numpy
import random

import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from common import utils_for_q_learning, buffer_class, utils
from common.noisy_layer import NoisyLinear

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy
import argparse
import numpy as np


def rbf_function_on_action(centroid_locations, action, beta):
    """
    centroid_locations: Tensor [batch x num_centroids (N) x a_dim (action_size)]
    action_set: Tensor [batch x a_dim (action_size)]
    beta: float
        - Parameter for RBF function
    Description: Computes the RBF function given centroid_locations and one action
    """
    assert len(centroid_locations.shape) == 3, "Must pass tensor with shape: [batch x N x a_dim]"
    assert len(action.shape) == 2, "Must pass tensor with shape: [batch x a_dim]"

    diff_norm = centroid_locations - action.unsqueeze(dim=1).expand_as(centroid_locations)  # [batch x N x a_dim]
    diff_norm = diff_norm ** 2
    diff_norm = torch.sum(diff_norm, dim=2)
    diff_norm = torch.sqrt(diff_norm + 1e-7)  # the absolute distance from centroids to actions
    diff_norm = diff_norm * beta * -1
    weights = F.softmax(diff_norm, dim=1)  # batch x N
    return weights


def rbf_function(centroid_locations, action_set, beta):
    """
    centroid_locations: Tensor [batch x num_centroids (N) x a_dim (action_size)]
    action_set: Tensor [batch x num_act x a_dim (action_size)]
        - Note: pass in num_act = 1 if you want a single action evaluated
    beta: float
        - Parameter for RBF function
    Description: Computes the RBF function given centroid_locations and some actions
    """
    assert len(centroid_locations.shape) == 3, "Must pass tensor with shape: [batch x N x a_dim]"
    assert len(action_set.shape) == 3, "Must pass tensor with shape: [batch x num_act x a_dim]"

    diff_norm = torch.cdist(centroid_locations, action_set, p=2)  # batch x N x num_act
    diff_norm = diff_norm * beta * -1
    weights = F.softmax(diff_norm, dim=2)  # batch x N x num_act
    return weights


class Net(nn.Module):
    def __init__(self, params, env, state_size, action_size, device):
        super(Net, self).__init__()

        self.env = env
        self.device = device
        self.params = params
        self.N = self.params['num_points']
        self.max_a = self.env.action_space.high[0]
        self.beta = self.params['temperature']
        self.v_min, self.v_max = self.params['vmin'], self.params['vmax']
        self.n_atoms = self.params['num_atoms']

        self.buffer_object = buffer_class.buffer_class(
            max_length=self.params['max_buffer_size'],
            env=self.env,
            seed_number=self.params['seed_number'],
            params=params)

        self.state_size, self.action_size = state_size, action_size

        def layer_norm(s):
            if self.params['layer_normalization']:
                return (nn.LayerNorm(s),)
            else:
                return tuple()

        def noisy_linear(dim_in, dim_out):
            if self.params['noisy_layers']:
                return NoisyLinear(dim_in, dim_out)
            else:
                return nn.Linear(dim_in, dim_out)

        self.value_range = torch.linspace(self.v_min, self.v_max, self.n_atoms).to(self.device)

        self.params_dic = []

        if self.params['dueling']:
            self.featureExtraction_module = nn.Sequential(
                nn.Linear(self.state_size, self.params['layer_size']),
                *layer_norm(self.params['layer_size']),
                nn.ReLU(),
                noisy_linear(self.params['layer_size'], self.params['layer_size']),
                *layer_norm(self.params['layer_size']),
                nn.ReLU(),
                noisy_linear(self.params['layer_size'], self.params['layer_size']),
                *layer_norm(self.params['layer_size']),
                nn.ReLU(),
            )
            self.baseValue_module = nn.Sequential(
                noisy_linear(self.params['layer_size'], self.n_atoms)  # [batch_size x num_atoms], needs to be expanded in dimension 1
            )
            self.advantage_module = nn.Sequential(
                noisy_linear(self.params['layer_size'], self.N * self.n_atoms)  # [batch_size x N x num_atoms]
            )
            self.params_dic.append({'params': self.featureExtraction_module.parameters(), 'lr': self.params['learning_rate']})
            self.params_dic.append({'params': self.baseValue_module.parameters(), 'lr': self.params['learning_rate']})
            self.params_dic.append({'params': self.advantage_module.parameters(), 'lr': self.params['learning_rate']})
        else:
            self.value_module = nn.Sequential(
                nn.Linear(self.state_size, self.params['layer_size']),
                *layer_norm(self.params['layer_size']),
                nn.ReLU(),
                noisy_linear(self.params['layer_size'], self.params['layer_size']),
                *layer_norm(self.params['layer_size']),
                nn.ReLU(),
                noisy_linear(self.params['layer_size'], self.params['layer_size']),
                *layer_norm(self.params['layer_size']),
                nn.ReLU(),
                noisy_linear(self.params['layer_size'], self.N * self.n_atoms),
            )
            self.params_dic.append({'params': self.value_module.parameters(), 'lr': self.params['learning_rate']})

        if self.params['num_layers_action_side'] == 1:
            self.location_module = nn.Sequential(
                nn.Linear(self.state_size, self.params['layer_size_action_side']),
                *layer_norm(self.params['layer_size_action_side']),
                nn.Dropout(p=self.params['dropout_rate']),
                nn.ReLU(),
                noisy_linear(self.params['layer_size_action_side'], self.action_size * self.N),
                utils_for_q_learning.Reshape(-1, self.N, self.action_size),
                nn.Tanh(),
            )
        elif self.params['num_layers_action_side'] == 2:
            self.location_module = nn.Sequential(
                nn.Linear(self.state_size, self.params['layer_size_action_side']),
                *layer_norm(self.params['layer_size_action_side']),
                nn.Dropout(p=self.params['dropout_rate']),
                nn.ReLU(),
                noisy_linear(self.params['layer_size_action_side'], self.params['layer_size_action_side']),
                *layer_norm(self.params['layer_size_action_side']),
                nn.Dropout(p=self.params['dropout_rate']),
                nn.ReLU(),
                noisy_linear(self.params['layer_size_action_side'], self.action_size * self.N),
                utils_for_q_learning.Reshape(-1, self.N, self.action_size),
                # shape: [batch x num_centroids x action_size]
                nn.Tanh(),
            )
        self.params_dic.append({'params': self.location_module.parameters(), 'lr': self.params['learning_rate_location_side']})

        torch.nn.init.xavier_uniform_(self.location_module[0].weight)
        torch.nn.init.zeros_(self.location_module[0].bias)

        if not self.params['noisy_layers']:
            self.location_module[3].weight.data.uniform_(-.1, .1)
            self.location_module[3].bias.data.uniform_(-1., 1.)

        self.criterion = nn.CrossEntropyLoss()
        try:
            if self.params['optimizer'] == 'RMSprop':
                self.optimizer = optim.RMSprop(self.params_dic)
            elif self.params['optimizer'] == 'Adam':
                self.optimizer = optim.Adam(self.params_dic)
            else:
                print('unknown optimizer ....')
        except:
            print("no optimizer specified ... ")

        self.to(self.device)

    def get_centroid_values(self, s):
        """
        given a batch of s, get all centroid values, [batch x N]
        """
        batch_size = s.shape[0]
        if self.params['dueling']:  # dueling needs to be computed for two situations
            features = self.featureExtraction_module(s)
            baseValue = self.baseValue_module(features).reshape(batch_size, 1, -1)  # [batch x 1 x n_atoms]
            advantages = self.advantage_module(features).reshape(batch_size, self.N, -1)  # [batch x N x n_atoms]
            if self.params["dueling_combine_operator"] == 'mean':
                logits = baseValue + (advantages - torch.mean(advantages, dim=1, keepdim=True))
            elif self.params["dueling_combine_operator"] == 'max':
                logits = baseValue + (advantages - torch.max(advantages, dim=1, keepdim=True)[0])
        else:
            logits = self.value_module(s).reshape(batch_size, self.N, -1)  # [batch x N x n_atoms]
        centroid_distributions = torch.softmax(logits, dim=2)
        centroid_values = torch.sum(self.value_range.expand(batch_size, self.N, -1) * centroid_distributions, dim=2)  # [batch x N x 1]
        return centroid_values  # [batch x N]

    def get_centroid_distributions(self, s):
        batch_size = s.shape[0]
        if self.params['dueling']:
            features = self.featureExtraction_module(s)
            baseValue = self.baseValue_module(features).reshape(batch_size, 1, -1)  # [batch x 1 x n_atoms]
            advantages = self.advantage_module(features).reshape(batch_size, self.N, -1)  # [batch x N x n_atoms]
            if self.params["dueling_combine_operator"] == 'mean':
                logits = baseValue + (advantages - torch.mean(advantages, dim=1, keepdim=True))
            elif self.params["dueling_combine_operator"] == 'max':
                logits = baseValue + (advantages - torch.max(advantages, dim=1, keepdim=True)[0])
        else:
            logits = self.value_module(s).reshape(batch_size, self.N, -1)
        centroid_distributions = torch.softmax(logits, dim=2)
        return centroid_distributions  # [batch x N x num_atoms]

    def get_centroid_locations(self, s):
        """
        given a batch of s, get all centroid_locations, [batch x N x a_dim]
        """
        centroid_locations = self.max_a * self.location_module(s)
        return centroid_locations

    def get_best_qvalue_and_action(self, s, return_all = False):
        """
        given a batch of states s, return Q(s,a), max_{a} ([batch x 1], [batch x a_dim])
        """
        all_centroids = self.get_centroid_locations(s)  # [batch x N x a_dim]
        values = self.get_centroid_values(s)  # [batch x N]
        distribution = self.get_centroid_distributions(s)
        weights = rbf_function(all_centroids, all_centroids, self.beta)  # [batch x N x N]

        allq = torch.bmm(weights, values.unsqueeze(2)).squeeze(2)  # bs x num_centroids
        alldis = torch.bmm(weights, distribution)
        # a -> all_centroids[idx] such that idx is max(dim=1) in allq
        # a = torch.gather(all_centroids, dim=1, index=indices)
        # (dim: bs x 1, dim: bs x action_dim)
        best, indices = allq.max(dim=1)
        a = torch.index_select(all_centroids, 1, indices) # This way of indexing works for both cases
        a = torch.diagonal(a, dim1=0, dim2=1).T
        dis = torch.index_select(alldis, 1, indices)
        dis = torch.diagonal(dis, dim1=0, dim2=1).T
        return best, dis, a.squeeze(0), {"alldis": alldis, "indices": indices}

    def forward(self, s, a):
        """
        given a batch of s,a , compute Q(s,a) [batch x 1]
        """
        centroid_distributions = self.get_centroid_distributions(s)  # [batch x N x num_atoms]
        centroid_locations = self.get_centroid_locations(s)
        # [batch x N]
        centroid_weights = rbf_function_on_action(centroid_locations, a, self.beta)
        output = torch.mul(centroid_weights.unsqueeze(2), centroid_distributions)  # [batch x N]
        output = output.sum(1)  # [batch x num_atoms]
        return output

    def train_noisy(self):
        """
        Serves same purpose as .train(), excepts only applies this setting to
        NoisyLayers
        """
        def train_module_noise(module):
            for m in module:
                if isinstance(m, (NoisyLinear,)):
                    m.train_noise()

        train_module_noise(self.location_module)
        if not self.params['dueling']:
            train_module_noise(self.value_module)
        else:
            train_module_noise(self.featureExtraction_module)
            train_module_noise(self.advantage_module)
            train_module_noise(self.baseValue_module)

    def eval_noisy(self):
        """
        Serves same purpose as .eval(), excepts only applies this setting to
        NoisyLayers
        """
        def eval_module_noise(module):
            for m in module:
                if isinstance(m, (NoisyLinear,)):
                    m.eval_noise()

        eval_module_noise(self.location_module)
        if not self.params['dueling']:
            eval_module_noise(self.value_module)
        else:
            eval_module_noise(self.featureExtraction_module)
            eval_module_noise(self.advantage_module)
            eval_module_noise(self.baseValue_module)

    def reset_noise(self):
        """
        Iterates through each module in the network and calls reset_noise() on any
        layer that is a NoisyLinear layer
        """
        def reset_module_noise(module):
            for m in module:
                if isinstance(m, (NoisyLinear,)):
                    m.reset_noise()

        reset_module_noise(self.location_module)
        if not self.params['dueling']:
            reset_module_noise(self.value_module)
        else:
            reset_module_noise(self.featureExtraction_module)
            reset_module_noise(self.advantage_module)
            reset_module_noise(self.baseValue_module)

    def execute_policy(self, s, episode, train_or_test):
        a = None
        if self.params['noisy_layers']:
            a = self.noisy_policy(s, episode, train_or_test)
        else:
            if self.params['policy_type'] == 'e_greedy':
                a = self.e_greedy_policy(s, episode, train_or_test)
            elif self.params['policy_type'] == 'e_greedy_gaussian':
                a = self.e_greedy_gaussian_policy(s, episode, train_or_test)
            elif self.params['policy_type'] == 'gaussian':
                a = self.gaussian_policy(s, episode, train_or_test)
        return a

    def policy(self, s, episode, train_or_test):
        '''
        Evalutes the policy
        '''
        self.eval()
        s_matrix = np.array(s).reshape(1, self.state_size)
        with torch.no_grad():
            s = torch.from_numpy(s_matrix).float().to(self.device)
            _, _, a, _ = self.get_best_qvalue_and_action(s)
            a = a.cpu().numpy()
        self.train()
        return a

    def noisy_policy(self, s, episode, train_or_test):
        '''
        Evaluates the policy, used in noisynet setup
        '''
        if train_or_test == 'train':
            self.train_noisy()  ## set self.train flags in modules
        else:
            self.eval_noisy()

        a = self.policy(s, episode, train_or_test)
        self.train_noisy()  ## set self.train flags in modules
        return a

    def e_greedy_policy(self, s, episode, train_or_test):
        """
        Given state s, at episode, take random action with p=eps if training
        Note - epsilon is determined by episode
        """
        epsilon = 1.0 / numpy.power(episode, 1.0 / self.params['policy_parameter'])
        if train_or_test == 'train' and random.random() < epsilon:
            a = self.env.action_space.sample()
            return a.tolist()
        else:
            return self.policy(s, episode, train_or_test)

    def e_greedy_gaussian_policy(self, s, episode, train_or_test):
        """
        Given state s, at episode, take random action with p=eps if training
        Note - epsilon is determined by episode
        """
        epsilon = 1.0 / numpy.power(episode, 1.0 / self.params['policy_parameter'])
        if train_or_test == 'train' and random.random() < epsilon:
            a = self.env.action_space.sample()
            return a.tolist()
        else:
            a = self.policy(s, episode, train_or_test)
            noise = numpy.random.normal(loc=0.0,
                                        scale=self.params['noise'],
                                        size=len(a))
            a = a + noise
            return a

    def gaussian_policy(self, s, episode, train_or_test):
        """
        Given state s, at episode, take random action with p=eps if training
        Note - epsilon is determined by episode
        """
        a = self.policy(s, episode, train_or_test)
        noise = numpy.random.normal(loc=0.0, scale=self.params['noise'], size=len(a))
        a = a + noise
        return a

    def update(self, target_Q, count):
        if len(self.buffer_object) < self.params['batch_size']:
            return 0
        batch_size = self.params['batch_size']

        if self.params['per']:
            s_matrix, a_matrix, r_matrix, done_matrix, sp_matrix, weights, indexes = self.buffer_object.sample(self.params['batch_size'])
        else:
            s_matrix, a_matrix, r_matrix, done_matrix, sp_matrix = self.buffer_object.sample(self.params['batch_size'])

        if self.params['reward_norm'] == "clip":
            r_matrix = numpy.clip(r_matrix, a_min=-self.params['reward_clip'], a_max=self.params['reward_clip'])
        elif self.params['reward_norm'] == "max":
            r_matrix = r_matrix * (1.0/self.params['reward_max'])
            r_matrix = numpy.clip(r_matrix, a_min=-1, a_max=1)

        s_matrix = torch.from_numpy(s_matrix).float().to(self.device)
        a_matrix = torch.from_numpy(a_matrix).float().to(self.device)
        r_matrix = torch.from_numpy(r_matrix).float().to(self.device)
        done_matrix = torch.from_numpy(done_matrix).float().to(self.device)
        sp_matrix = torch.from_numpy(sp_matrix).float().to(self.device)
        # Q_star = Q_star.reshape((self.params['batch_size'], -1))
        with torch.no_grad():
            if self.params['double']:
                _, _, _, info = self.get_best_qvalue_and_action(sp_matrix) # get indices from the online agent
                indices = info["indices"]
                _, _, _, info = target_Q.get_best_qvalue_and_action(sp_matrix)
                alldis = info["alldis"]
                dis = torch.index_select(alldis, 1, indices)
                dis = torch.diagonal(dis, dim1=0, dim2=1).T
            else:
                best, dis, a, _ = target_Q.get_best_qvalue_and_action(sp_matrix)
            if self.params['nstep']:
                next_value_range = r_matrix + self.params['gamma']**self.params['nstep_size'] * (1 - done_matrix) * self.value_range
            else:
                next_value_range = r_matrix + self.params['gamma'] * (1 - done_matrix) * self.value_range
            # compute the distribution for actions
            y = torch.zeros((batch_size, self.n_atoms)).to(self.device)
            next_v_range = torch.clip(next_value_range, self.v_min, self.v_max).to(self.device)
            next_v_pos = (next_v_range - self.v_min) / ((self.v_max - self.v_min) / (self.n_atoms - 1))
            lb = torch.floor(next_v_pos).to(torch.int64).to(self.device)
            ub = torch.ceil(next_v_pos).to(torch.int64).to(self.device)
            # handling marginal situation for lb==ub
            lb[(ub > 0) * (lb == ub)] -= 1
            ub[(lb < (self.n_atoms - 1)) * (lb == ub)] += 1
            offset = torch.linspace(0, ((batch_size - 1) * self.n_atoms), batch_size).unsqueeze(1).expand(batch_size, self.n_atoms).to(torch.int64).to(self.device)
            y.view(-1).index_add_(0, (lb + offset).view(-1), (dis * (ub.float() - next_v_pos)).view(-1))  # m_l = m_l + p(s_t+n, a*)(u - b)
            y.view(-1).index_add_(0, (ub + offset).view(-1), (dis * (next_v_pos - lb.float())).view(-1))  # m_u = m_u + p(s_t+n, a*)(b - l)

        # [s a r s_, a_,]
        y_hat = self.forward(s_matrix, a_matrix)
        if self.params['per']:
            td_error = torch.abs(y-y_hat).cpu().detach().numpy()
            self.buffer_object.storage.update_priorities(indexes, td_error)

        # loss = self.criterion(y_hat, y.type(torch.float32))
        loss = torch.sum((-y * torch.log(y_hat + 1e-8)), 1)  # (m , N_ATOM)
        loss = torch.mean(loss)
        self.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.zero_grad()
        utils_for_q_learning.sync_networks(
            target=target_Q,
            online=self,
            alpha=self.params['target_network_learning_rate'],
            copy=False)

        if self.params['noisy_layers']:
            self.reset_noise()
            target_Q.reset_noise()
        return loss.cpu().data.numpy()
