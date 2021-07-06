import numpy
from cpprb import ReplayBuffer, PrioritizedReplayBuffer, create_env_dict, create_before_add_func
import random


class buffer_class:
	def __init__(self, max_length, seed_number, env, n_step=False, n_step_size=4):
		env_dict = create_env_dict(env)
		self.before_add = create_before_add_func(env)
		self.n_step=n_step
		self.n_step_size = n_step_size
		if self.n_step:
			self.storage = ReplayBuffer(max_length, env_dict, Nstep={
				"size": self.n_step_size,
				"gamma": 0.99,
				"rew": "rew",
				"next": "next_obs"
			})
		else:
			self.storage = ReplayBuffer(max_length, env_dict)
		

	def append(self, s, a, r, done, sp):
		if self.n_step:
			s = s.reshape(1, s.shape[0])
			a = a.reshape(1, a.shape[0])
			sp = sp.reshape(1, sp.shape[0])
			if not done:
				self.storage.add(**self.before_add(obs=s,
					act=a,
					rew=r,
					next_obs=sp,
					done=0.0))
			else:
				self.storage.on_episode_end()
		else:
			self.storage.add(**self.before_add(obs=s, act=a, rew=r, done=done, next_obs=sp))

	def sample(self, batch_size):
		batch = self.storage.sample(batch_size)
		s_matrix = batch['obs']
		a_matrix = batch['act']
		r_matrix = batch['rew']
		done_matrix = batch['done']
		sp_matrix = batch['next_obs']
		return s_matrix, a_matrix, r_matrix, done_matrix, sp_matrix

	def __len__(self):
		return self.storage.get_stored_size()

class PriorityBuffer:
	def __init__(self, max_length, seed_number, env):
		env_dict = create_env_dict(env)
		self.before_add = create_before_add_func(env)
		self.storage = PrioritizedReplayBuffer(max_length, env_dict)

	def append(self, s, a, r, done, sp):
		self.storage.add(**self.before_add(obs=s, act=a, rew=r, done=done, next_obs=sp))

	def sample(self, batch_size):
		batch = self.storage.sample(batch_size)
		s_matrix = batch['obs']
		a_matrix = batch['act']
		r_matrix = batch['rew']
		done_matrix = batch['done']
		sp_matrix = batch['next_obs']
		weights = batch['weights']
		indexes = batch['indexes']
		return s_matrix, a_matrix, r_matrix, done_matrix, sp_matrix, weights, indexes

	def __len__(self):
		return self.storage.get_stored_size()
