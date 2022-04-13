
from Model import MODEL
import numpy as np
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
import pandas as pd



# ---*---

class DQNReplayer:
    def __init__(self, capacity):
        self.memory = pd.DataFrame(
            index=range(capacity),
            columns=['observation', 'action', 'reward', 'next_observation']
        )
        self.i = 0    # Row index
        self.count = 0    # Number of experience storage
        self.capacity = capacity    # Experience capacity

    def store(self, *args):
        self.memory.loc[self.i] = args
        self.i = (self.i + 1) % self.capacity    # Update row index
        self.count = min(self.count + 1, self.capacity)    # Guaranteed quantity will not exceed experience capacity

    def sample(self, size):
        indices = np.random.choice(self.count, size=size)
        return (np.stack(self.memory.loc[indices, field]) for field in self.memory.columns)

# ---*---

in_depth    = 10
in_height   = 50
in_width    = 50
in_channels = 1
outputs     = 5

# ---*---

# DoubleDQN
class Vampire_Agent:
    def __init__(
        self,
        lr = 0.01,
        batch_size = 8,
        save_weights_path = None,
        load_weights_path = None

    ):
        self.in_depth    = in_depth       # Depth of the time series
        self.in_height   = in_height
        self.in_width    = in_width
        self.in_channels = in_channels
        self.outputs     = outputs
        self.lr          = lr,            # learning

        self.gamma = 0.99

        self.min_epsilon = 0.3

        self.replay_memory_size = 10000
        self.replay_start_size = 500
        self.batch_size = batch_size    # Number of samples drawn

        self.update_freq = 100
        self.target_network_update_freq = 500

        self.save_weights_path = save_weights_path    # Specify the path to save the model weight. The default is None, do not save.
        self.load_weights_path = load_weights_path    # Specify the path to load the model weight. The default is None, do not load.

        self.evaluate_net = self.build_network()
        self.target_net = self.build_network()
        self.replayer = DQNReplayer(self.replay_memory_size)

        self.step = 0

    def build_network(self):
        model = MODEL(
            in_depth = self.in_depth,
            in_height = self.in_height,
            in_width = self.in_width,
            in_channels = self.in_channels,
            outputs = self.outputs,
            lr = self.lr,
            load_weights_path = self.load_weights_path
        )
        return model

    # 行为选择方法
    def choose_action(self, observation):

        if self.step <= self.replay_start_size or np.random.rand() < self.min_epsilon:
            q_values = np.random.rand(self.outputs)
            self.who_play = '随机探索'
        else:
            observation = observation.reshape(-1, self.in_depth, self.in_height, self.in_width, self.in_channels)
            q_values = self.evaluate_net.predict(observation)[0]
            self.who_play = '模型预测'

        action = np.argmax(q_values)

        act(action)

        return action

    def learn(self, verbose=0):

        self.step += 1

        if self.step % self.update_freq == 0:

            if self.step % self.target_network_update_freq == 0:
                self.update_target_network()

            observations, actions, rewards, next_observations = self.replayer.sample(self.batch_size)

            # Data preprocessing
            observations = observations.reshape(-1, self.in_depth, self.in_height, self.in_width, self.in_channels)
            actions = actions.astype(np.int8)
            next_observations = next_observations.reshape(-1, self.in_depth, self.in_height, self.in_width, self.in_channels)


            next_eval_qs = self.evaluate_net.predict(next_observations)
            next_actions = next_eval_qs.argmax(axis=-1)

            next_qs = self.target_net.predict(next_observations)
            next_max_qs = next_qs[np.arange(next_qs.shape[0]), next_actions]

            us = rewards + self.gamma * next_max_qs
            targets = self.evaluate_net.predict(observations)
            targets[np.arange(us.shape[0]), actions] = us


            self.evaluate_net.fit(observations, targets, batch_size=1, verbose=verbose)

            self.save_evaluate_network()

    def update_target_network(self):
        self.target_net.set_weights(self.evaluate_net.get_weights())

    def save_evaluate_network(self):
        self.evaluate_net.save_weights(self.save_weights_path)