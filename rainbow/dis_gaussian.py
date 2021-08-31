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

import utils_for_q_learning, buffer_class

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy
import pickle


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
        self.v_min, self.v_max = -10, 10
        self.n_atoms = 51

        self.buffer_object = buffer_class.buffer_class(
            max_length=self.params['max_buffer_size'],
            env=self.env,
            seed_number=self.params['seed_number'])

        self.state_size, self.action_size = state_size, action_size

        self.value_range = torch.linspace(self.v_min, self.v_max, self.n_atoms).to(self.device)

        # first column is mean, second column is standard deviation
        self.value_module = nn.Sequential(
            nn.Linear(self.state_size, self.params['layer_size']),
            nn.ReLU(),
            nn.Linear(self.params['layer_size'], self.params['layer_size']),
            nn.ReLU(),
            nn.Linear(self.params['layer_size'], self.params['layer_size']),
            nn.ReLU(),
            nn.Linear(self.params['layer_size'], self.params['layer_size']),
            nn.ReLU(),
            nn.Linear(self.params['layer_size'], self.params['layer_size']),
            nn.ReLU(),
            nn.Linear(self.params['layer_size'], self.N * 2),  
        )

        if self.params['num_layers_action_side'] == 1:
            self.location_module = nn.Sequential(
                nn.Linear(self.state_size, self.params['layer_size_action_side']),
                nn.Dropout(p=self.params['dropout_rate']),
                nn.ReLU(),
                nn.Linear(self.params['layer_size_action_side'], self.action_size * self.N),
                utils_for_q_learning.Reshape(-1, self.N, self.action_size),
                nn.Tanh(),
            )
        elif self.params['num_layers_action_side'] == 2:
            self.location_module = nn.Sequential(
                nn.Linear(self.state_size, self.params['layer_size_action_side']),
                nn.Dropout(p=self.params['dropout_rate']),
                nn.ReLU(),
                nn.Linear(self.params['layer_size_action_side'], self.params['layer_size_action_side']),
                nn.Dropout(p=self.params['dropout_rate']),
                nn.ReLU(),
                nn.Linear(self.params['layer_size_action_side'], self.action_size * self.N),
                utils_for_q_learning.Reshape(-1, self.N, self.action_size),
                # shape: [batch x num_centroids x action_size]
                nn.Tanh(),
            )

        torch.nn.init.xavier_uniform_(self.location_module[0].weight)
        torch.nn.init.zeros_(self.location_module[0].bias)

        self.location_module[3].weight.data.uniform_(-.1, .1)
        self.location_module[3].bias.data.uniform_(-1., 1.)

        self.criterion = nn.CrossEntropyLoss()

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

        self.to(self.device)

    def get_centroid_values(self, s):
        """
        given a batch of s, get all centroid values, [batch x N]
        """
        batch_size = s.shape[0]
        
        # because we're using a gaussian function parameterized by mean and sigma, we just return the mean
        centroid_values = self.value_module(s).reshape(batch_size, self.N, -1)[:, :, 0]  # [batch x N x n_atoms]        
        return centroid_values  # [batch x N]

    def get_centroid_distributions(self, s):
        batch_size = s.shape[0]
        gaussian_mean_std = self.value_module(s).reshape(batch_size, self.N, -1)
        gaussian_mean_std[:, :, 1] = gaussian_mean_std[:, :, 1]**2
        return gaussian_mean_std  # [batch x N x num_atoms]

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
        mean_and_std = self.get_centroid_distributions(s)  # [batch x N x n_atoms]
        weights = rbf_function(all_centroids, all_centroids, self.beta)  # [batch x N x N]

        allq = torch.bmm(weights, values.unsqueeze(2)).squeeze(2)  # bs x num_centroids
        alldis = torch.bmm(weights, mean_and_std)
        best, indices = allq.max(dim=1)
        a = torch.index_select(all_centroids, 1, indices)  # This way of indexing works for both cases
        a = torch.diagonal(a, dim1=0, dim2=1).T
        dis = torch.index_select(alldis, 1, indices)
        dis = torch.diagonal(dis, dim1=0, dim2=1).T
        return best, dis, a.squeeze(), {"alldis": alldis, "indices": indices}

    def forward(self, s, a):
        """
        given a batch of s,a , compute Q(s,a) [batch x 1]
        """
        mean_and_std = self.get_centroid_distributions(s)  # [batch x N x num_atoms]
        centroid_locations = self.get_centroid_locations(s)
        # [batch x N]
        centroid_weights = rbf_function_on_action(centroid_locations, a, self.beta)
        output = torch.bmm(centroid_weights.unsqueeze(1), mean_and_std).squeeze(1)  # [batch x N]
        return output

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
            self.eval()
            s_matrix = numpy.array(s).reshape(1, self.state_size)
            with torch.no_grad():
                s = torch.from_numpy(s_matrix).float().to(self.device)
                _, _, a, _ = self.get_best_qvalue_and_action(s)
                a = a.cpu().numpy()
            self.train()
            return a

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
            self.eval()
            s_matrix = numpy.array(s).reshape(1, self.state_size)
            with torch.no_grad():
                s = torch.from_numpy(s_matrix).float().to(self.device)
                _, a = self.get_best_qvalue_and_action(s)
                a = a.cpu().numpy()
            self.train()
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
        self.eval()
        s_matrix = numpy.array(s).reshape(1, self.state_size)
        with torch.no_grad():
            s = torch.from_numpy(s_matrix).float().to(self.device)
            _, a = self.get_best_qvalue_and_action(s)
            a = a.cpu()
        self.train()
        noise = numpy.random.normal(loc=0.0, scale=self.params['noise'], size=len(a))
        a = a + noise
        return a

    def update(self, target_Q, count):
        if len(self.buffer_object) < self.params['batch_size']:
            return 0
        batch_size = self.params['batch_size']
        s_matrix, a_matrix, r_matrix, done_matrix, sp_matrix = self.buffer_object.sample(self.params['batch_size'])
        r_matrix = numpy.clip(r_matrix, a_min=-self.params['reward_clip'], a_max=self.params['reward_clip'])

        s_matrix = torch.from_numpy(s_matrix).float().to(self.device)
        a_matrix = torch.from_numpy(a_matrix).float().to(self.device)
        r_matrix = torch.from_numpy(r_matrix).float().to(self.device)
        done_matrix = torch.from_numpy(done_matrix).float().to(self.device)
        sp_matrix = torch.from_numpy(sp_matrix).float().to(self.device)

        # Q_star = Q_star.reshape((self.params['batch_size'], -1))
        with torch.no_grad():
            best, mean_and_std, _,  _ = target_Q.get_best_qvalue_and_action(sp_matrix).detach()
            mean_and_std[:, :, 0] = r_matrix + self.params['gamma'] * (1 - done_matrix) * mean_and_std[:, :, 0] # [batch x self.N x 2]
            
        # [s a r s_, a_,]
        y_hat = self.forward(s_matrix, a_matrix)
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
        return loss.cpu().data.numpy()
