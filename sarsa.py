import numpy as np
from tqdm import trange


class StateActionFeatureVectorWithTile():
    def __init__(self,
                 state_low: np.array,
                 state_high: np.array,
                 num_actions: int,
                 num_tilings: int,
                 tile_width: np.array):
        """
        state_low: possible minimum value for each dimension in state
        state_high: possible maximum value for each dimension in state
        num_actions: the number of possible actions
        num_tilings: # tilings
        tile_width: tile width for each dimension
        """
        self.num_tiles_in_dim = (np.ceil((state_high - state_low) / tile_width) + 1).astype(np.int)
        self.state_high = state_high
        self.state_low = state_low
        self.tile_width = tile_width
        self.num_tilings = num_tilings
        self.num_actions = num_actions
        self.centers = self.state_low[:, np.newaxis] - (np.arange(self.num_tilings) / self.num_tilings)[np.newaxis, :] * self.tile_width[:, np.newaxis]

    def feature_vector_len(self) -> int:
        """
        return dimension of feature_vector: d = num_actions * num_tilings * num_tiles
        """
        return self.num_actions * self.num_tilings * np.prod(self.num_tiles_in_dim)

    def __call__(self, s, done, a) -> np.array:
        """
        implement function x: S+ x A -> [0,1]^d
        if done is True, then return 0^d
        """
        if done:
            return np.zeros((self.feature_vector_len(),))

        else:
            numerator = (s[:, np.newaxis] - self.centers)
            denominator = self.tile_width[:, np.newaxis]
            coordinates = np.floor(numerator/denominator).astype(np.int)
            feature = np.zeros((self.num_actions, self.num_tilings, *self.num_tiles_in_dim))
            for i in range(self.num_tilings):
                cords = tuple(coordinates[:, i])
                feature[(a, i) + cords] = 1
            flattened_feature = feature.flatten()
            return flattened_feature


def SarsaLambda(
        env,
        gamma: float,
        lam: float,
        alpha: float,
        X: StateActionFeatureVectorWithTile,
        num_episode: int,
) -> np.array:
    """
    Implement True online Sarsa(\lambda)
    """

    def epsilon_greedy_policy(s, done, w, epsilon=.01):
        nA = env.action_space.n
        Q = []
        for a in range(nA):
            Q.append(np.dot(w, X(s, done, a)))

        if np.random.rand() < epsilon:
            return np.random.randint(nA)
        else:
            return np.argmax(Q)

    weight_vector = np.zeros((X.feature_vector_len(),))

    for i in trange(num_episode):
        initial_state, done = env.reset(), False
        greedy_action = epsilon_greedy_policy(initial_state, done, weight_vector)
        feature_vector = X(initial_state, done, greedy_action)
        elg_trace_vector = np.zeros(weight_vector.shape)
        Q_prev = 0

        while not done:
            initial_state, reward, done, _ = env.step(greedy_action)
            greedy_action = epsilon_greedy_policy(initial_state, done, weight_vector)
            feature_vector_next = X(initial_state, done, greedy_action)
            Q = np.dot(weight_vector, feature_vector)
            Q_next = np.dot(weight_vector, feature_vector_next)
            delta = reward + gamma * Q_next - Q
            elg_trace_vector = gamma * lam * elg_trace_vector + (1 - alpha * gamma * lam * np.dot(elg_trace_vector, feature_vector)) * feature_vector
            weight_vector += alpha * (delta + Q - Q_prev) * elg_trace_vector - alpha * (Q - Q_prev) * feature_vector
            Q_prev = Q_next
            feature_vector = feature_vector_next

    return weight_vector