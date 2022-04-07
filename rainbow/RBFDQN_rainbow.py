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

def rbf_function_on_action(centroid_locations, action, beta):
    '''
    centroid_locations: Tensor [batch x num_centroids (N) x a_dim (action_size)]
    action_set: Tensor [batch x a_dim (action_size)]
    beta: float
        - Parameter for RBF function

    Description: Computes the RBF function given centroid_locations and one action
    '''
    assert len(centroid_locations.shape) == 3, "Must pass tensor with shape: [batch x N x a_dim]"
    assert len(action.shape) == 2, "Must pass tensor with shape: [batch x a_dim]"

    diff_norm = centroid_locations - action.unsqueeze(dim=1).expand_as(centroid_locations)
    diff_norm = diff_norm ** 2
    diff_norm = torch.sum(diff_norm, dim=2)
    diff_norm = torch.sqrt(diff_norm + 1e-7)
    diff_norm = diff_norm * beta.to(diff_norm.device) * -1
    weights = F.softmax(diff_norm, dim=1)  # batch x N
    return weights


def rbf_function(centroid_locations, action_set, beta):
    '''
    centroid_locations: Tensor [batch x num_centroids (N) x a_dim (action_size)]
    action_set: Tensor [batch x num_act x a_dim (action_size)]
        - Note: pass in num_act = 1 if you want a single action evaluated
    beta: float
        - Parameter for RBF function

    Description: Computes the RBF function given centroid_locations and some actions
    '''
    assert len(centroid_locations.shape) == 3, "Must pass tensor with shape: [batch x N x a_dim]"
    assert len(action_set.shape) == 3, "Must pass tensor with shape: [batch x num_act x a_dim]"
    diff_norm = torch.cdist(centroid_locations, action_set, p=2)  # batch x N x num_act
    diff_norm = diff_norm * beta.to(diff_norm.device) * -1
    weights = F.softmax(diff_norm, dim=2)  # batch x N x num_act
    return weights


class Net(nn.Module):
    def __init__(self, params, env, state_size, action_size, device):
        super(Net, self).__init__()

        self.env = env
        self.device = device
        self.params = params
        self.N = self.params['num_points']
        self.max_a = torch.from_numpy(self.env.action_space.high).to(self.device)
        self.beta = torch.Tensor([self.params['temperature']])

        if (self.params['random_betas']):
            # initialize random betas to be between 0 and 0.25 for each centroid.
            # the random betas are fixed for each centroid index throughout training.
            self.beta = torch.normal(mean=self.params['temperature'], std=math.sqrt(self.N)/self.N, size=(1, self.N))
            #self.beta.to(self.device)

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
                return NoisyLinear(dim_in, dim_out, sigma_init=self.params['sigma_noise'])
            else:
                return nn.Linear(dim_in, dim_out)

        ## Control for enable / disable of noisy_layers in value module vs centroid module
        old_noisy_value = self.params['noisy_layers']
        if self.params['noisy_where'] == 'centroid':
            ## Disable noisy nets for value network
            self.params['noisy_layers'] = False

        if not self.params['dueling']:
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
                noisy_linear(self.params['layer_size'], self.N),
            )
        else:

            self.advantage_module = nn.Sequential(
                nn.Linear(self.state_size, self.params['layer_size']),
                *layer_norm(self.params['layer_size']),
                nn.ReLU(),
                noisy_linear(self.params['layer_size'], self.params['layer_size']),
                *layer_norm(self.params['layer_size']),
                nn.ReLU(),
                noisy_linear(self.params['layer_size'], self.params['layer_size']),
                nn.ReLU(),
                nn.Linear(self.params['layer_size'], self.params['layer_size']),
                *layer_norm(self.params['layer_size']),
                nn.ReLU(),
                noisy_linear(self.params['layer_size'], self.N),
            )

            self.baseValue_module = nn.Sequential(
                nn.Linear(self.state_size, self.params['layer_size']),
                *layer_norm(self.params['layer_size']),
                nn.ReLU(),
                noisy_linear(self.params['layer_size'], self.params['layer_size']),
                *layer_norm(self.params['layer_size']),
                nn.ReLU(),
                noisy_linear(self.params['layer_size'], self.params['layer_size']),
                nn.ReLU(),
                nn.Linear(self.params['layer_size'], self.params['layer_size']),
                *layer_norm(self.params['layer_size']),
                nn.ReLU(),
                noisy_linear(self.params['layer_size'], 1),
            )

        ## Control for enable / disable of noisy_layers in value module vs centroid module
        self.params['noisy_layers'] = old_noisy_value
        old_noisy_value = self.params['noisy_layers']
        if self.params['noisy_where'] == 'value':
            ## Disable noisy nets for centroid network
            self.params['noisy_layers'] = False

        if self.params['num_layers_action_side'] == 1:
            self.location_module = nn.Sequential(
                nn.Linear(self.state_size, self.params['layer_size_action_side']),
                *layer_norm(self.params['layer_size_action_side']),
                nn.Dropout(p=self.params['dropout_rate']),
                nn.ReLU(),
                noisy_linear(self.params['layer_size_action_side'],
                          self.action_size * self.N),
                utils_for_q_learning.Reshape(-1, self.N, self.action_size),
                nn.Tanh(),
            )
        elif self.params['num_layers_action_side'] == 2:
            self.location_module = nn.Sequential(
                nn.Linear(self.state_size, self.params['layer_size_action_side']),
                *layer_norm(self.params['layer_size_action_side']),
                nn.Dropout(p=self.params['dropout_rate']),
                nn.ReLU(),
                noisy_linear(self.params['layer_size_action_side'],
                          self.params['layer_size_action_side']),
                *layer_norm(self.params['layer_size_action_side']),
                nn.Dropout(p=self.params['dropout_rate']),
                nn.ReLU(),
                noisy_linear(self.params['layer_size_action_side'],
                          self.action_size * self.N),
                utils_for_q_learning.Reshape(-1, self.N, self.action_size),
                nn.Tanh(),
            )
        self.params['noisy_layers'] = old_noisy_value

        torch.nn.init.xavier_uniform_(self.location_module[0].weight)
        torch.nn.init.zeros_(self.location_module[0].bias)

        if not self.params['noisy_layers']:
            self.location_module[3].weight.data.uniform_(-.1, .1)
            self.location_module[3].bias.data.uniform_(-1., 1.)

        if self.params['loss_type'] == 'MSELoss':
            self.criterion = nn.MSELoss()
        elif self.params['loss_type'] == 'HuberLoss':
            # if the torch version is smaller than 1.9.0, the huber loss is called SMOOTHL1LOSS
            self.criterion = nn.SmoothL1Loss()
            #self.criterion = nn.HuberLoss()
        else:
            raise NameError('only two kinds of loss can we use, MSELoss or HuberLoss')



        if not self.params['dueling']:
            self.params_dic = [{
                'params': self.value_module.parameters(), 'lr': self.params['learning_rate']
            },
            {
                'params': self.location_module.parameters(),
                'lr': self.params['learning_rate_location_side']
            }]
        else:
            self.params_dic = [
            {
                'params': self.advantage_module.parameters(), 'lr': self.params['learning_rate']
            },
            {
                'params': self.location_module.parameters(), 'lr': self.params['learning_rate_location_side']
            },
            {
                'params': self.baseValue_module.parameters(), 'lr': self.params['learning_rate']
            },
            ]
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

    ### dueling specific methods (below)
    def get_centroid_advantages(self, s):
        '''
        given a batch of s, get centroid advantage values, [batch x N]
        '''
        assert self.params['dueling']

        centroid_advantages = self.advantage_module(s)
        return centroid_advantages

    def get_state_value(self, s):
        '''
        given a batch of s, get the base value for each state [batch x N]
        '''
        assert self.params['dueling']

        value = self.baseValue_module(s)
        return value
    ### dueling specific methods above ^

    def get_centroid_values(self, s):
        '''
        given a batch of s, get all centroid values, [batch x N]
        '''
        centroid_values = self.value_module(s)
        return centroid_values

    def get_centroid_locations(self, s):
        '''
        given a batch of s, get all centroid_locations, [batch x N x a_dim]
        '''
        centroid_locations = self.max_a * self.location_module(s)
        return centroid_locations

    def get_best_qvalue_and_action(self, s):
        '''
        given a batch of states s, return Q(s,a), max_{a} ([batch x 1], [batch x a_dim])
        '''
        all_centroids = self.get_centroid_locations(s)
        weights = rbf_function(all_centroids, all_centroids, self.beta)  # [batch x N x N]

        if not self.params['dueling']:
            values = self.get_centroid_values(s)
        else:
            centroid_advantages = self.get_centroid_advantages(s)
            state_value = self.get_state_value(s)
            if self.params["dueling_combine_operator"] == 'mean':
                values = state_value + (centroid_advantages - torch.mean(centroid_advantages, dim=1, keepdim=True))
            elif self.params["dueling_combine_operator"] == 'max':
                values = state_value + (centroid_advantages - torch.max(centroid_advantages, dim=1, keepdim=True)[0])

        allq = torch.bmm(weights, values.unsqueeze(2)).squeeze(2)  # bs x num_centroids

        # a -> all_centroids[idx] such that idx is max(dim=1) in allq
        # a = torch.gather(all_centroids, dim=1, index=indices)
        # (dim: bs x 1, dim: bs x action_dim)

        best, indices = allq.max(dim=1)
        if s.shape[0] == 1:
            index_star = indices.item()
            a = all_centroids[0, index_star]
            return best, a
        else:
            a = all_centroids[np.arange(len(s)), indices]
            return best, a

    def forward(self, s, a):
        '''
        given a batch of s,a , compute Q(s,a) [batch x 1]
        '''
        if not self.params['dueling']:
            centroid_values = self.get_centroid_values(s)  # [batch_dim x N]
        else:
            centroid_advantages = self.get_centroid_advantages(s)
            if self.params["dueling_combine_operator"] == 'mean':
                centroid_values = self.get_state_value(s) + (centroid_advantages - torch.mean(centroid_advantages, dim=1, keepdim=True))
            elif self.params["dueling_combine_operator"] == 'max':
                centroid_values = self.get_state_value(s) + (centroid_advantages - torch.max(centroid_advantages, dim=1, keepdim=True)[0])
        centroid_locations = self.get_centroid_locations(s)
        # [batch x N]

        centroid_weights = rbf_function_on_action(centroid_locations, a, self.beta)
        output = torch.mul(centroid_weights, centroid_values)  # [batch x N]
        output = output.sum(1, keepdim=True)  # [batch x 1]
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

        if not self.params["dueling"]:
            train_module_noise(self.value_module)
            train_module_noise(self.location_module)
        else:
            train_module_noise(self.baseValue_module)
            train_module_noise(self.advantage_module)
            train_module_noise(self.location_module)

    def eval_noisy(self):
        """
        Serves same purpose as .eval(), excepts only applies this setting to
        NoisyLayers
        """
        def eval_module_noise(module):
            for m in module:
                if isinstance(m, (NoisyLinear,)):
                    m.eval_noise()

        if not self.params["dueling"]:
            eval_module_noise(self.value_module)
            eval_module_noise(self.location_module)
        else:
            eval_module_noise(self.baseValue_module)
            eval_module_noise(self.advantage_module)
            eval_module_noise(self.location_module)

    def reset_noise(self):
        """
        Iterates through each module in the network and calls reset_noise() on any
        layer that is a NoisyLinear layer
        """
        def reset_module_noise(module):
            for m in module:
                if isinstance(m, (NoisyLinear,)):
                    m.reset_noise()

        if not self.params['dueling']:
            reset_module_noise(self.value_module)
        else:
            reset_module_noise(self.advantage_module)
            reset_module_noise(self.baseValue_module)

        reset_module_noise(self.location_module)

    def execute_policy(self, s, episode, train_or_test, steps=None):
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
            _, a = self.get_best_qvalue_and_action(s)
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

        if episode >= self.params['noisy_episode_cutoff']:
            ## This effectively will disable training variance.
            self.eval_noisy()

        ## Set the policy to use here for testing (noisy / noisy with ep-greedy)
        a = None
        if self.params['policy_type'] == 'e_greedy':
            a = self.e_greedy_policy(s, episode, train_or_test)
        elif self.params['policy_type'] == 'e_greedy_gaussian':
            a = self.e_greedy_gaussian_policy(s, episode, train_or_test)
        elif self.params['policy_type'] == 'gaussian':
            a = self.gaussian_policy(s, episode, train_or_test)
        else:
            a = self.policy(s, episode, train_or_test)
        self.train_noisy()  ## set self.train flags in modules
        return a

    def e_greedy_policy(self, s, episode, train_or_test):
        '''
        Given state s, at episode, take random action with p=eps if training
        Note - epsilon is determined by episode
        '''
        epsilon = 1.0 / np.power(episode, 1.0 / self.params['policy_parameter'])
        if train_or_test == 'train' and random.random() < epsilon:
            a = self.env.action_space.sample()
            return a.tolist()
        else:
            return self.policy(s, episode, train_or_test)

    def e_greedy_gaussian_policy(self, s, episode, train_or_test):
        '''
        Given state s, at episode, take random action with p=eps if training
        Note - epsilon is determined by episode
        '''
        epsilon = 1.0 / np.power(episode, 1.0 / self.params['policy_parameter'])
        if train_or_test == 'train' and random.random() < epsilon:
            a = self.env.action_space.sample()
            return a.tolist()
        else:
            a = self.gaussian_policy(s, episode, train_or_test)
            return a

    def gaussian_policy(self, s, episode, train_or_test):
        '''
        Given state s, at episode, take random action with p=eps if training
        Note - epsilon is determined by episode
        '''
        a = self.policy(s, episode, train_or_test)
        if train_or_test == 'train':
            noise = np.random.normal(loc=0.0, scale=self.params['noise'], size=len(a))
            a = a + noise
        return a

    def update(self, target_Q):
        if len(self.buffer_object) < self.params['batch_size']:
            update_param = {}
            update_param['average_q'] = 0
            update_param['average_q_star'] = 0
            return 0, update_param

        if self.params['per']:
            s_matrix, a_matrix, r_matrix, done_matrix, sp_matrix, weights, indexes = self.buffer_object.sample(self.params['batch_size'])
        else:
            s_matrix, a_matrix, r_matrix, done_matrix, sp_matrix = self.buffer_object.sample(self.params['batch_size'])

        r_matrix = numpy.clip(r_matrix, a_min=-self.params['reward_clip'], a_max=self.params['reward_clip'])
        s_matrix = torch.from_numpy(s_matrix).float().to(self.device)
        a_matrix = torch.from_numpy(a_matrix).float().to(self.device)
        r_matrix = torch.from_numpy(r_matrix).float().to(self.device)
        done_matrix = torch.from_numpy(done_matrix).float().to(self.device)
        sp_matrix = torch.from_numpy(sp_matrix).float().to(self.device)

        if (self.params["double"]):
            if not self.params['dueling']: # ONLY DOUBLE
                _, actions = self.get_best_qvalue_and_action(sp_matrix)

                target_centroids = target_Q.get_centroid_locations(sp_matrix)

                centroid_weights = rbf_function_on_action(target_centroids, actions, self.beta)

                centroid_values = target_Q.get_centroid_values(sp_matrix)

                output = torch.mul(centroid_weights, centroid_values)  # [batch x N]
                Q_star = output.sum(1, keepdim=True)  # [batch x 1]
            else: # DOUBLE and DUELING
                _, actions = self.get_best_qvalue_and_action(sp_matrix)
                target_centroids = target_Q.get_centroid_locations(sp_matrix)
                centroid_weights = rbf_function_on_action(target_centroids, actions, self.beta)

                centroid_advantages = target_Q.get_centroid_advantages(sp_matrix)
                state_value = target_Q.get_state_value(sp_matrix)
                if self.params["dueling_combine_operator"] == 'mean':
                    centroid_values = state_value + (centroid_advantages - torch.mean(centroid_advantages, dim=1, keepdim=True))
                elif self.params["dueling_combine_operator"] == 'max':
                    centroid_values = state_value + (centroid_advantages - torch.max(centroid_advantages, dim=1, keepdim=True)[0])
                output = torch.mul(centroid_weights, centroid_values)  # [batch x N]
                Q_star = output.sum(1, keepdim=True)  # [batch x 1]
        else:
            if self.params['dueling']: # ONLY DUELING
                _, actions = target_Q.get_best_qvalue_and_action(sp_matrix)
                target_centroids = target_Q.get_centroid_locations(sp_matrix)
                centroid_weights = rbf_function_on_action(target_centroids, actions, self.beta)

                centroid_advantages = target_Q.get_centroid_advantages(sp_matrix)
                state_value = target_Q.get_state_value(sp_matrix)
                if self.params["dueling_combine_operator"] == 'mean':
                    centroid_values = state_value + (centroid_advantages - torch.mean(centroid_advantages, dim=1, keepdim=True))
                elif self.params["dueling_combine_operator"] == 'max':
                    centroid_values = state_value + (centroid_advantages - torch.max(centroid_advantages, dim=1, keepdim=True)[0])
                output = torch.mul(centroid_weights, centroid_values)  # [batch x N]
                Q_star = output.sum(1, keepdim=True)  # [batch x 1]
            else: # NO DOUBLE NO DUELING
                Q_star, _ = target_Q.get_best_qvalue_and_action(sp_matrix)

        if self.params['nstep']:
            Q_star = (self.params["gamma"]**(self.params['nstep_size']-1)) * Q_star

        Q_star = Q_star.reshape((self.params['batch_size'], -1))

        with torch.no_grad():
            y = r_matrix + self.params['gamma'] * (1 - done_matrix) * Q_star
        y_hat = self.forward(s_matrix, a_matrix)
        
        update_params = {}
        average_q_star = torch.mean(Q_star, dim=0).item()
        average_q = torch.mean(y_hat, dim=0).item()
        update_params['average_q'] = average_q
        update_params['average_q_star'] = average_q_star

        if self.params['per']:
            td_error = torch.abs(y-y_hat).cpu().detach().numpy()
            self.buffer_object.storage.update_priorities(indexes, td_error)

        loss = self.criterion(y_hat, y)
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
        return loss.cpu().data.numpy(), update_params
