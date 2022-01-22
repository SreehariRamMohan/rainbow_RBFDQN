import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import gym
from gym import spaces
import torch

class StochasticRegression(gym.Env):
   
    '''
    reward is only dependent upon the action
    
    a is [2]
    s is [1] but the state is really irrelevent here. 
    '''
    def get_reward(self, s, a):
        assert s.shape == (self.s_dim, ), f"{s.shape} {self.s_dim}"
        assert a.shape == (self.a_dim, ), f"{a.shape} {self.a_dim}"
        
        # rv u for stochastic regression target.
        u = np.random.random()

        if u <= (1/3):
            return np.sin(a[0]**2 + a[1]**2)
        elif (u <= (2/3)):
            return 2
        else:
            return np.sign(a[0])*(np.abs(a[0])/(10))**(1/3) + 1.5

    def __init__(self, s_dim=0, a_dim=2, episode_length=float('inf')):
        """
        Action is what makes reward
        Observation is always the same.
        For now, episode_length is determined through the registration thing in init
        We'll have a special case where s_dim being zero makes the observation space constant and zero.
        """
        was_zero = False
        if s_dim <= 0:
            s_dim = 1
            was_zero = True

        self.s_dim = s_dim
        self.a_dim = a_dim
        self._max_episode_steps = episode_length

        print(f"sdim: {s_dim}\t adim: {a_dim}")
        self.episode_length = episode_length
        self.FREQ_SCALE = 4.

        if was_zero:
            self.observation_space = spaces.Box(1.0, 1.0, shape=(self.s_dim, ))
        else:
            self.observation_space = spaces.Box(-1.0, 1.0, shape=(self.s_dim, ))

        self.action_space = spaces.Box(-2.0, 2.0, shape=(self.a_dim, ))
        self._last_state = None
        self._t = 0

    def reset(self):
        self._last_state = self.observation_space.sample()
        self._t = 0
        return self._last_state.copy()

    def step(self, action):
        reward = self.get_reward(self._last_state, action)
        assert self.action_space.contains(action), action
        self._last_state = self.observation_space.sample()
        self._t += 1
        done = (self._t >= self.episode_length)

        return self._last_state.copy(), reward, done, {}

    def plot(self, savefile):
        assert self.a_dim == 2
        x = np.arange(-2.0, 2.0, 0.05)
        y = np.arange(-2.0, 2.0, 0.05)
        X, Y = np.meshgrid(x, y)  # grid of point
        actions = np.concatenate([X.reshape(-1, 1), Y.reshape(-1, 1)], axis=1)
        states = np.zeros((actions.shape[0], self.s_dim))
        Z = np.array([self.get_reward(states[i], actions[i]) for i in range(len(actions))])
        Z = Z.reshape(X.shape)

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        surf = ax.plot_surface(X,
                               Y,
                               Z,
                               cmap=plt.cm.jet,
                               linewidth=0,
                               antialiased=False)
        fig.colorbar(surf, shrink=0.5, aspect=5)
        #plt.show()
        plt.savefig(savefile)
        #plt.close()

    def plot_reward_functions(self, title):
        x = np.arange(-2.0, 2.0, 0.05)
        y = np.arange(-2.0, 2.0, 0.05)
        X, Y = np.meshgrid(x, y)  # grid of point
        

        f = np.sin(X**2 + Y**2)
        g = np.ones_like(X)*2
        h = np.sign(X)*(np.abs(X)/(10))**(1/3) + 1.5
        # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        # fig.set_size_inches(18.5, 10.5)

        colormap = plt.cm.viridis

        fig = plt.figure()
        fig.tight_layout()
        # fig.set_size_inches(6, 4)
        ax1 = fig.add_subplot(221, projection='3d')
        ax1.title.set_text("$f(x, y) = sin(x^2 + y^2)$")
        ax1.title.set_fontsize(20)
        surf2 = ax1.plot_surface(X,
        Y,
        f,
        cmap=colormap,
        linewidth=0,
        antialiased=False)

        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax1.set_zlabel("Z")

        ax2 = fig.add_subplot(222, projection='3d')
        ax2.title.set_text("$g(x, y) = 2$")
        ax2.title.set_fontsize(20)
        surf3 = ax2.plot_surface(X,
                               Y,
                               g,
                               cmap=colormap,
                               linewidth=0,
                               antialiased=False)

        ax2.set_xlabel("X")
        ax2.set_ylabel("Y")
        ax2.set_zlabel("Z")
        
        ax3 = fig.add_subplot(223, projection='3d')
        ax3.title.set_text("$h(x, y) = (\dfrac{x}{10}) ^ \dfrac{1}{3} + 1.5$")
        ax3.title.set_fontsize(20)
        ax3.set_xlabel("X")
        ax3.set_ylabel("Y")
        ax3.set_zlabel("Z")
        
        surf1 = ax3.plot_surface(X,
                               Y,
                               h,
                               cmap=colormap,
                               linewidth=0,
                               antialiased=False)

        ax4 = fig.add_subplot(224, projection='3d')
        ax4.title.set_text("$r_{weighted}(x, y) = \dfrac{f}{3} + \dfrac{g}{3} + \dfrac{h}{3}$")
        ax4.set_xlabel("X")
        ax4.set_ylabel("Y")
        ax4.set_zlabel("Z")
        ax4.title.set_fontsize(20)
        weighted_sum = (1/3)*f + (1/3)*g + (1/3)*h

        surf_weighted = ax4.plot_surface(X,
                               Y,
                               weighted_sum,
                               cmap=colormap,
                               linewidth=0,
                               antialiased=False)
    

        # fig.colorbar(surf1, shrink=0.5, aspect=5)
        # fig.colorbar(surf2, shrink=0.5, aspect=5)
        # fig.colorbar(surf3, shrink=0.5, aspect=5)
       
        #plt.savefig(title, bbox_inches='tight', format='pdf')
        # plt.suptitle("Stochastic Reward Function", fontsize=25)
        plt.show()

        #plt.close()
        #plt.show()


    def plot_agent(self, Q_object, savefile):
        assert self.a_dim == 2
        x = np.arange(-2.0, 2.0, 0.05)
        y = np.arange(-2.0, 2.0, 0.05)
        X, Y = np.meshgrid(x, y)  # grid of point
        actions = np.concatenate([X.reshape(-1, 1), Y.reshape(-1, 1)], axis=1)
        
        # (0, 0)
        zero_action = np.zeros((1, 2))
        zero_action_torch = torch.Tensor(zero_action).to(Q_object.device)
        
        # (0, -1.8)
        action_neg_18 = np.array([-1.8, 0]).reshape(-1, 2)
        action_neg_18_torch = torch.Tensor(action_neg_18).to(Q_object.device)

        action_pos_18 = np.array([1.8, 0]).reshape(-1, 2)
        action_pos_18_torch = torch.Tensor(action_pos_18).to(Q_object.device)
        
        actions_torch = torch.Tensor(actions).to(Q_object.device)
        #states_torch = torch.zeros_like(actions_torch)
        
        states_torch = torch.zeros(actions.shape[0], self.s_dim).to(Q_object.device)
        states_torch_zeros = torch.zeros(zero_action_torch.shape[0], self.s_dim).to(Q_object.device)

        Z_torch = Q_object.forward(states_torch, actions_torch)
        Z_output_zeros = Q_object.forward(states_torch_zeros, zero_action_torch).detach().cpu().numpy()
        Z_output_neg_18 = Q_object.forward(states_torch_zeros, action_neg_18_torch).detach().cpu().numpy()
        Z_output_pos_18 = Q_object.forward(states_torch_zeros, action_pos_18_torch).detach().cpu().numpy()
        
        Z = Z_torch.detach().cpu().numpy()

        if (Q_object.params['distributional']):
            print("mean of support distribution", Z_output_zeros.mean(axis=1))
            Z_output_zeros = Z_output_zeros.squeeze()
            Z_output_neg_18 = Z_output_neg_18.squeeze()
            Z_output_pos_18 = Z_output_pos_18.squeeze()

            from mpl_toolkits.axes_grid1 import ImageGrid

            fig = plt.figure(figsize=(6, 6))
            grid = ImageGrid(fig, 111,  # similar to subplot(111)
                nrows_ncols=(3, 1),  # creates 2x2 grid of axes
                axes_pad=0.3,  # pad between axes in inch.
                cbar_mode="single",
                cbar_location="right",
                cbar_pad=0.1
            )

            plt.title("Support Location Frequency")

            # fig, axs = plt.subplots(3)
            # fig.tight_layout()
            
            ax1 = grid[0]
            ax1.hist2d(x=Z_output_neg_18, y=np.ones((200,)), cmap=plt.cm.jet, bins=(50, 1))
            ax1.get_yaxis().set_ticks([])
            ax1.title.set_text('Supports for (-1.8, 0)')

            ax2 = grid[1]
            ax2.hist2d(x=Z_output_zeros, y=np.ones((200,)), cmap=plt.cm.jet, bins=(50, 1))
            ax2.get_yaxis().set_ticks([])
            ax2.title.set_text('Supports for (0, 0)')

            ax3 = grid[2]
            h = ax3.hist2d(x=Z_output_pos_18, y=np.ones((200,)), cmap=plt.cm.jet, bins=(50, 1))
            ax3.get_yaxis().set_ticks([])
            ax3.title.set_text('Supports for (1.8, 0)')
            ax3.set_xlabel("Support Value")

            fig.colorbar(h[3], cax=grid.cbar_axes[0], orientation='vertical')

            #axs[0].colorbar()
            #plt.yticks([])
            #plt.show()
            #plt.plot(Z_output_zeros.squeeze())
            #plt.xlabel("Support Value")

            plt.savefig("support_locations", bbox_inches='tight', format='pdf')
            #plt.show()
            plt.close()
            Z = Z.mean(axis=1)
        Z = Z.reshape(X.shape)

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        surf = ax.plot_surface(X,
                               Y,
                               Z,
                               cmap=cm.coolwarm,
                               linewidth=0,
                               antialiased=False)
        fig.colorbar(surf, shrink=0.5, aspect=5)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        #plt.show()
        #plt.savefig(savefile)
        #plt.close()

def load_and_plot_q(Q_object, saved_network_dir):
    # plot the vanilla RBF-DQN model predictions

    Q_object.load_state_dict(torch.load(saved_network_dir))
    Q_object.eval()
    Q_object.env.plot_agent(Q_object, "plotted_q_func.jpg")

    Q_object.env.plot_reward_functions("reward_function")


if __name__ == "__main__":
    env = StochasticRegression(episode_length=200)
    env.reset()
    action = np.random.random((2,))

    # stats = {'f': 0, 'g': 0, 'h': 0}
    # for _ in range(5000):
    #     a = np.array([0, 0])
    #     _, r, _, _ = env.step(a)
    #     if r == 2:
    #         stats['f'] += 1
    #     elif r == 1.5:
    #         stats['g'] += 1
    #     else:
    #         stats['h'] += 1
    # print(stats)

    
