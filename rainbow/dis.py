'''
Passing all possible actions (centroids) into the value network
'''

import gym
import sys
import time
import numpy as np
import random

import os

sys.path.append("..")

from common import utils_for_q_learning, buffer_class

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy
import pickle
from common.noisy_layer import NoisyLinear


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
        self.max_a = torch.from_numpy(self.env.action_space.high).to(self.device)
        self.beta = self.params['temperature']
        self.N_QUANTILE = self.params['quantiles']
        self.Prob = torch.tensor([[[1.0/self.params['quantiles']]]]).to(self.device)

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
            
        if self.params['dueling']:
            # add an extra output param to the value_module which will be the "base" support value. 
            # all the other supports are offset from this one value. 
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
                noisy_linear(self.params['layer_size'], self.N * self.N_QUANTILE),
            )

            self.base_support_value = nn.Sequential(
                nn.Linear(self.state_size, self.params['layer_size']),
                *layer_norm(self.params['layer_size']),
                nn.ReLU(),
                noisy_linear(self.params['layer_size'], self.params['layer_size']),
                *layer_norm(self.params['layer_size']),
                nn.ReLU(),
                noisy_linear(self.params['layer_size'], self.params['layer_size']),
                *layer_norm(self.params['layer_size']),
                nn.ReLU(),
                noisy_linear(self.params['layer_size'], 1),
            )
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
                noisy_linear(self.params['layer_size'], self.N * self.N_QUANTILE),
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
                noisy_linear(self.params['layer_size_action_side'], self.action_size * self.N),
                *layer_norm(self.action_size * self.N),
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
                *layer_norm(self.action_size * self.N),
                utils_for_q_learning.Reshape(-1, self.N, self.action_size),
                # shape: [batch x num_centroids x action_size]
                nn.Tanh(),
            )

        self.params['noisy_layers'] = old_noisy_value

        torch.nn.init.xavier_uniform_(self.location_module[0].weight)
        torch.nn.init.zeros_(self.location_module[0].bias)

        if not self.params['noisy_layers']:
            self.location_module[3].weight.data.uniform_(-.1, .1)
            self.location_module[3].bias.data.uniform_(-1., 1.)

        if self.params['dueling']:
            self.params_dic = [
                {
                    'params': self.value_module.parameters(),
                    'lr': self.params['learning_rate']
                },
                {
                    'params': self.base_support_value.parameters(),
                    'lr': self.params['learning_rate']
                },
                {
                    'params': self.location_module.parameters(),
                    'lr': self.params['learning_rate_location_side']
                }
            ]
        else:
            self.params_dic = [
                {
                'params': self.value_module.parameters(),
                'lr': self.params['learning_rate']
                },
                {
                    'params': self.location_module.parameters(),
                    'lr': self.params['learning_rate_location_side']
                }
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
        self.criteria = torch.nn.functional.smooth_l1_loss
        self.to(self.device)

    
    def get_centroid_values(self, s):
        """
        given a batch of s, get all centroid values, [batch x N]
        """
        batch_size = s.shape[0]
        centroid_quantiles = self.value_module(s).reshape(batch_size, self.N, -1)  # [batch x N x N_QUANTILES]

        centroid_values = (centroid_quantiles * self.Prob).sum(axis=2) 

        if self.params['dueling']:
            base_centroid_value = self.base_support_value(s)
            centroid_values = centroid_values + base_centroid_value

        return centroid_values  # [batch x N]

    def get_centroid_quantiles(self, s):
        batch_size = s.shape[0]
        centroid_quantiles = self.value_module(s).reshape(batch_size, self.N, -1)  # [batch x N x N_QUANTILES]
        
        if self.params['dueling']:
            base_centroid_quantile_value = self.base_support_value(s).reshape(-1, 1, 1)
            centroid_quantiles = centroid_quantiles + base_centroid_quantile_value

        return centroid_quantiles

    def get_centroid_locations(self, s):
        """
        given a batch of s, get all centroid_locations, [batch x N x a_dim]
        """
        centroid_locations = self.max_a * self.location_module(s)
        return centroid_locations

    def get_best_qvalue_and_action(self, s):
        """
        given a batch of states s, return Q(s,a), max_{a} ([batch x 1], [batch x a_dim])
        """
        all_centroids = self.get_centroid_locations(s)  # [batch x N x a_dim]
        values = self.get_centroid_values(s)  # [batch x N]
        quantiles = self.get_centroid_quantiles(s)  # [batch x N x n_atoms]
        weights = rbf_function(all_centroids, all_centroids, self.beta)  # [batch x N x N]

        allq = torch.bmm(weights, values.unsqueeze(2)).squeeze(2)  # bs x num_centroids
        allquantiles = torch.bmm(weights, quantiles)
        best, indices = allq.max(dim=1)
        a = torch.index_select(all_centroids, 1, indices)  # This way of indexing works for both cases
        a = torch.diagonal(a, dim1=0, dim2=1).T
        quantiles = torch.index_select(allquantiles, 1, indices)
        quantiles = torch.diagonal(quantiles, dim1=0, dim2=1).T
        return best, a, quantiles

    def forward(self, s, a):
        """
        given a batch of s,a , compute Q(s,a) [batch x 1]
        """
        centroid_quantiles = self.get_centroid_quantiles(s)  # [batch x N x num_atoms]
        centroid_locations = self.get_centroid_locations(s)
        # [batch x N]
        centroid_weights = rbf_function_on_action(centroid_locations, a, self.beta)
        output = torch.bmm(centroid_weights.unsqueeze(1), centroid_quantiles).squeeze(1)  # [batch x N]
        return output

    def execute_policy(self, s, episode, train_or_test, steps=None):
        """
        Given state s, at episode, take random action with p=eps if training
        Note - epsilon is determined by episode
        """

        if self.params['noisy_layers']:
            a = self.noisy_policy(s, episode, train_or_test, steps=steps)
            return a
        else:
            return self.e_greedy_policy(s, episode, train_or_test)

    def e_greedy_policy(self, s, episode, train_or_test):
        epsilon = 1.0 / numpy.power(episode, 1.0 / self.params['policy_parameter'])
        epsilon = self.params['train_epsilon'] if epsilon < self.params['train_epsilon'] else epsilon
        if train_or_test == 'train' and random.random() < epsilon:
            a = self.env.action_space.sample()
            return a
        else:
            self.eval()
            s_matrix = numpy.array(s).reshape(1, self.state_size)
            with torch.no_grad():
                s = torch.from_numpy(s_matrix).float().to(self.device)
                _,  a, _ = self.get_best_qvalue_and_action(s)
                a = a.cpu().numpy()
            self.train()
            return a.squeeze(0)

    def noisy_policy(self, s, episode, train_or_test, steps=0):
        '''
        Evaluates the policy, used in noisynet setup
        '''
        if train_or_test == 'train':
            self.train_noisy()  ## set self.train flags in modules
        else:
            self.eval_noisy()
	
        if steps >= self.params['noisy_episode_cutoff']:
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

    def gaussian_policy(self, s, episode, train_or_test):
        '''
        Given state s, at episode, take random action with p=eps if training
        Note - epsilon is determined by episode
        '''
        self.eval()
        s_matrix = numpy.array(s).reshape(1, self.state_size)
        with torch.no_grad():
            s = torch.from_numpy(s_matrix).float().to(self.device)
            _,  a, _ = self.get_best_qvalue_and_action(s)
            a = a.cpu().numpy()
        self.train()
        a = a.squeeze(0)
        if train_or_test == 'train':
            noise = np.random.normal(loc=0.0, scale=self.params['noise'], size=len(a))
            a = a + noise
        return a

    def softmax_policy(self, s):
        """
        Given state s, sample one action among the centroids according to the centroid values
        """
        self.eval()
        s_matrix = numpy.array(s).reshape(1, self.state_size)
        with torch.no_grad():
            s = torch.from_numpy(s_matrix).float().to(self.device)
            centroids = self.get_centroid_locations(s)
            values = self.get_centroid_values(s)
            dis = ((5*values).exp()/(5*values).exp().sum(dim=1)).reshape(-1)
            index = np.random.choice(self.N, 1, p=dis.cpu().numpy())[0]
            action = centroids[0, index, :]
        self.train()
        return action.cpu().numpy()

    def CEM_policy(self, s):
        """
        Given state s, do CEM for several runs to get the best action
        """
        self.eval()
        s_matrix = numpy.array(s).reshape(1, self.state_size)
        with torch.no_grad():
            s = torch.from_numpy(s_matrix).float().to(self.device)
            actions = self.get_centroid_locations(s)
            values = self.get_centroid_values(s)
            dis = ((10*values).exp() / (10*values).exp().sum(dim=1)).reshape(-1)
            num_action = self.N
            for _ in range(3):
                indices = np.random.choice(num_action, num_action//2, p=dis.cpu().numpy())
                actions = actions[0, indices, :].unsqueeze(dim=0)
                values = values[0, indices].unsqueeze(dim=0)
                dis = ((10 * values).exp() / (10 * values).exp().sum(dim=1)).reshape(-1)
                num_action = num_action // 2
            index = np.argmax(dis.cpu())
            action = actions[0, index, :]
        self.train()
        return action.cpu().numpy()

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
            train_module_noise(self.base_support_value)
            train_module_noise(self.value_module)
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
            eval_module_noise(self.base_support_value)
            eval_module_noise(self.value_module)
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
            reset_module_noise(self.base_support_value)
            reset_module_noise(self.value_module)
        reset_module_noise(self.location_module)

    def update(self, target_Q):
        if len(self.buffer_object) < self.params['batch_size']:
            update_param = {}
            update_param['average_q'] = 0
            update_param['average_q_star'] = 0
            return 0, update_param
        
        batch_size = self.params['batch_size']

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

        # Construct the target
        with torch.no_grad():
            if self.params['double']:
                # check this part for sure
                _, a_, _ = self.get_best_qvalue_and_action(sp_matrix)
                target_centroids, target_quantiles = target_Q.get_centroid_locations(sp_matrix), target_Q.get_centroid_quantiles(sp_matrix)
                weights = rbf_function_on_action(target_centroids, a_, self.beta)
                next_quantiles = torch.bmm(weights.unsqueeze(1), target_quantiles).squeeze()
            else:
                _, _, next_quantiles = target_Q.get_best_qvalue_and_action(sp_matrix)
            
            y = (r_matrix + self.params['gamma'] * (1 - done_matrix) * next_quantiles).unsqueeze(1)
        # Construct the prediction for the current state and action
        y_hat = self.forward(s_matrix, a_matrix).unsqueeze(2)

        # Redefine the loss
        u = y - y_hat
        QUANTS = np.linspace(0.0, 1.0, self.params['quantiles'] + 1)[1:] # Pick up the probability quantiles
        QUANTS_TARGET = (np.linspace(0.0, 1.0, self.params['quantiles'] + 1)[:-1] + QUANTS)/2
        tau = torch.FloatTensor(QUANTS_TARGET).view(1, -1, 1).to(self.device)
        weights = torch.abs(tau - u.le(0.).float())
        # loss = self.criterion(y_hat, y.type(torch.float32))
        criterion_loss = self.criteria(y_hat, y,  reduction='none')
        loss = torch.mean(weights * criterion_loss)
        self.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value_module.parameters(),0.1)
        torch.nn.utils.clip_grad_norm_(self.location_module.parameters(), 0.1)
        self.optimizer.step()
        self.zero_grad()
        utils_for_q_learning.sync_networks(
            target=target_Q,
            online=self,
            alpha=self.params['target_network_learning_rate'],
            copy=False)

        if self.params['per']:
            td_error = torch.mean(weights*criterion_loss, dim=(1, 2)).detach().cpu().numpy()
            self.buffer_object.storage.update_priorities(indexes, td_error)
        if self.params['noisy_layers']:
            self.reset_noise()
            target_Q.reset_noise()

        update_param = {}
        update_param['average_q'] = 0
        update_param['average_q_star'] = 0
        return loss.cpu().data.numpy(), update_param
