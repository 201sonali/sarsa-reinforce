from typing import Iterable
import numpy as np
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable


class Detector1(nn.Module):
    def __init__(self, input_size):
        super(Detector1, self).__init__()
        self.input_size = input_size
        self.hidden_layer1 = nn.Linear(self.input_size, 32)
        self.hidden_layer2 = nn.Linear(32, 32)
        self.fc = nn.Linear(32, 1)
        nn.init.xavier_uniform_(self.hidden_layer1.weight)
        nn.init.xavier_uniform_(self.hidden_layer2.weight)
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        return self.fc(F.relu(self.hidden_layer2(F.relu(self.hidden_layer1(torch.from_numpy(x))))))


class Detector2(nn.Module):
    def __init__(self, input_size, output_size):
        super(Detector2, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layer1 = nn.Linear(self.input_size, 32)
        self.hidden_layer2 = nn.Linear(32, 32)
        self.hidden_layer3 = nn.Linear(32, self.output_size)
        self.Softmax = nn.Softmax(dim=1)

        nn.init.xavier_uniform_(self.hidden_layer1.weight)
        nn.init.xavier_uniform_(self.hidden_layer2.weight)
        nn.init.xavier_uniform_(self.hidden_layer3.weight)

    def forward(self, x):
        return self.Softmax(
            self.hidden_layer3(F.relu(self.hidden_layer2(F.relu(self.hidden_layer1(torch.from_numpy(x)))))))


class PiApproximationWithNN():
    def __init__(self,
                 state_dims,
                 num_actions,
                 alpha):
        """
        state_dims: the number of dimensions of state space
        action_dims: the number of possible actions
        alpha: learning rate
        """
        # TODO: implement here

        # Tips for TF users: You will need a function that collects the probability of action taken
        # actions; i.e. you need something like
        #
        # pi(.|s_t) = tf.constant([[.3,.6,.1], [.4,.4,.2]])
        # a_t = tf.constant([1, 2])
        # pi(a_t|s_t) =  [.6,.2]
        #
        # To implement this, you need a tf.gather_nd operation. You can use implement this by,
        #
        # tf.gather_nd(pi,tf.stack([tf.range(tf.shape(a_t)[0]),a_t],axis=1)),
        # assuming len(pi) == len(a_t) == batch_size
        self.num_states = state_dims
        self.detector = Detector2(state_dims, num_actions)
        self.optimizer = optim.Adam(self.detector.parameters(), lr=alpha, betas=(0.9, 0.999))
        self.loss = 0

    def __call__(self, s) -> int:
        s_reshaped = s.reshape(1, self.num_states).astype('float32')
        q_s_a = self.detector.forward(s_reshaped)
        action_probs = q_s_a.detach().numpy()[0]
        action = np.random.choice(range(len(action_probs)), p=action_probs)
        return action

    def update(self, s, a, gamma_t, delta):
        """
        s: state S_t
        a: action A_t
        gamma_t: gamma^t
        delta: G-v(S_t,w)
        """
        # TODO: implement this s

        s = s.reshape(1, self.num_states).astype('float32')

        output = self.detector(s).view(-1)
        output_log = output[a].log()
        self.loss = output_log * -1 * delta * gamma_t
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
        return self.loss.item()


class Baseline(object):
    """
    The dumbest baseline; a constant for every state
    """

    def __init__(self, b):
        self.b = b

    def __call__(self, s) -> float:
        return self.b

    def update(self, s, G):
        pass


class VApproximationWithNN(Baseline):
    def __init__(self,
                 state_dims,
                 alpha):
        self.num_states = state_dims
        self.detector = Detector1(state_dims)
        self.lossfunc = nn.MSELoss()
        self.optimizer = optim.Adam(self.detector.parameters(), lr=alpha, betas=(0.9, 0.999))
        self.loss = 0

    def __call__(self, s) -> float:
        # TODO: implement this method

        s_reshaped = s.reshape(1, self.num_states).astype('float32')
        state_value = self.detector.forward(s_reshaped)
        state_value_reshape = state_value.detach().numpy()[0][0]
        return state_value_reshape

    def update(self, s, G):
        # TODO: implement this method

        s_reshaped = s.reshape(1, self.num_states).astype('float32')
        output = self.detector(s_reshaped)
        target = torch.tensor(float(G)).view(-1, 1)
        self.loss = (self.lossfunc(output, target))
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
        return self.loss.item()


def REINFORCE(
        env,  # open-ai environment
        gamma: float,
        num_episodes: int,
        pi: PiApproximationWithNN,
        V: Baseline) -> Iterable[float]:
    """
    implement REINFORCE algorithm with and without baseline.

    input:
        env: target environment; openai gym
        gamma: discount factor
        num_episode: #episodes to iterate
        pi: policy
        V: baseline
    output:
        a list that includes the G_0 for every episodes.
    """

    reward_list = []
    action_space = np.arange(env.action_space.n)

    for i in range(num_episodes):

        state = env.reset()
        done = False
        episode = []

        while not done:
            action = action_space[pi(state)]
            next_state, reward, done, _ = env.step(action)
            episode.append((state, reward, action))
            state = next_state

        states = []
        rewards = []
        actions = []

        for each_episode in episode:
            states.append(each_episode[0])
            rewards.append(each_episode[1])
            actions.append(each_episode[2])

        for episode_number in range(len(episode)):
            G = 0
            for episode_number_next in range(episode_number + 1, len(episode)):
                G = G + gamma ** (episode_number_next - episode_number - 1) * rewards[episode_number_next]

            V.update(states[episode_number], G)
            delta = G - V(states[episode_number])
            pi.update(states[episode_number], actions[episode_number], gamma, delta)
            if episode_number == 0:
                reward_list.append(G)

    return reward_list
