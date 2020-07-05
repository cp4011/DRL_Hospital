from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import random, copy
import numpy as np
from config import all_bot_actions, required_slots, am_dict, nearby_hospital
import re


class DQNAgent:
    """The DQN agent that interacts with the user."""
    def __init__(self, state_size, constants):
        self.C = constants['agent']
        self.memory = []
        self.max_memory_size = self.C['max_mem_size']
        self.eps = self.C['epsilon_init']
        self.vanilla = self.C['vanilla']
        self.lr = self.C['learning_rate']
        self.gamma = self.C['gamma']
        self.batch_size = self.C['batch_size']
        self.hidden_size = self.C['dqn_hidden_size']

        self.load_weights_file_path = self.C['load_weights_file_path']
        self.save_weights_file_path = self.C['save_weights_file_path']

        if self.max_memory_size < self.batch_size:
            raise ValueError('Max memory size must be at least as great as batch size!')

        self.state_size = state_size
        self.possible_actions = all_bot_actions
        self.num_actions = len(self.possible_actions)
        self.mask = np.array([0] * self.num_actions)
        self.nearby_hospital = nearby_hospital

        self.beh_model = self._build_model()
        self.tar_model = self._build_model()
        self._load_weights()
        self.beh_model.predict(np.array([0] * self.state_size).reshape(1, self.state_size))
        self.tar_model.predict(np.array([0] * self.state_size).reshape(1, self.state_size))

    def get_action_mask(self, memory, db_search, nearby_hospital=False):
        self.mask = np.array([0] * self.num_actions)    # 每一轮对话后清空，重新收集信息
        accumulated_slots = [0] * len(required_slots)
        if not db_search:                   # not found
            self.mask[-1] = 1
            return
        print("state tracker memory:", memory)
        for k, v in memory.items():
            if v is not None:
                index = required_slots.index(k)
                accumulated_slots[index] = 1
        context = ''.join(map(str, accumulated_slots))
        # print("context:", context)

        def construct_mask(context):
            indices = am_dict[context]
            if nearby_hospital and indices[0] != 1:
                indices = self.nearby_hospital       # [2]
            for index in indices:
                self.mask[index - 1] = 1.
            return self.mask
        construct_mask(context)
        # print("action mask: ", self.mask)

    def _build_model(self):
        """Builds and returns model/graph of neural network."""

        model = Sequential()
        model.add(Dense(self.hidden_size, input_dim=self.state_size, activation='relu'))
        model.add(Dense(self.num_actions, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.lr))
        return model

    def save_weights(self):
        """Saves the weights of both models in two h5 files."""

        if not self.save_weights_file_path:
            return
        beh_save_file_path = re.sub(r'\.h5', r'_beh.h5', self.save_weights_file_path)
        self.beh_model.save_weights(beh_save_file_path)
        tar_save_file_path = re.sub(r'\.h5', r'_tar.h5', self.save_weights_file_path)
        self.tar_model.save_weights(tar_save_file_path)

    def train(self):
        # Calc. num of batches to run
        num_batches = len(self.memory) // self.batch_size
        for b in range(num_batches):
            batch = random.sample(self.memory, self.batch_size)

            states = np.array([sample[0] for sample in batch])
            next_states = np.array([sample[3] for sample in batch])

            assert states.shape == (self.batch_size, self.state_size), 'States Shape: {}'.format(states.shape)
            assert next_states.shape == states.shape

            # For leveling error
            beh_state_preds = self._dqn_predict(states)
            if not self.vanilla:
                beh_next_states_preds = self._dqn_predict(next_states)

            # For target value for DQN (& DDQN)
            tar_next_state_preds = self._dqn_predict(next_states, target=True)

            inputs = np.zeros((self.batch_size, self.state_size))
            targets = np.zeros((self.batch_size, self.num_actions))

            for i, (s, a, r, s_, d) in enumerate(batch):
                t = beh_state_preds[i]
                if not self.vanilla:

                    t[a] = r + self.gamma * tar_next_state_preds[i][np.argmax(beh_next_states_preds[i])] * (not d)
                else:
                    t[a] = r + self.gamma * np.amax(tar_next_state_preds[i]) * (not d)

                inputs[i] = s
                targets[i] = t

            self.beh_model.fit(inputs, targets, epochs=1, verbose=0)

    def get_action(self, state, use_rule=False):  # 给当前state，用rule-based或者NN的方法返回一个action_index及其action
        if self.eps > random.random():          # epsilon greedy 贪心搜索
            index = random.randint(0, self.num_actions - 1)
            action = self._map_index_to_action(index)
            return index, action
        else:
            if use_rule:
                return self._rule_action()      # 返回一个rule-based policy action
            else:
                return self._dqn_action(state)  # 给定一个state，返回behavior model输出的Q值数组中最大的那个下标action_id

    def _map_action_to_index(self, response):
        """
        Maps an action to an index from possible actions.

        Parameters:
            response (dict)

        Returns:
            int
        """

        for (i, action) in enumerate(self.possible_actions):    # 7
            if response == action:
                return i
        raise ValueError('Response: {} not found in possible actions'.format(response))

    def _dqn_action(self, state):   # behavior model输出最大Q值所在的下标Q值数组中最大Q值所在的action_index
        # print(self._dqn_predict_one(state), "*", self.mask)
        if self.mask[-1] == 1:
            return self.num_actions-1, all_bot_actions[-1]
        result = self._dqn_predict_one(state) * self.mask
        # print("result:", result)
        index = np.argmax(result)
        # index = np.argmax(self._dqn_predict_one(state))     # Q值数组中最大Q值所在的action_index
        action = self._map_index_to_action(index)
        return index, action

    def _map_index_to_action(self, index):
        for (i, action) in enumerate(self.possible_actions):
            if index == i:
                return copy.deepcopy(action)
        raise ValueError('Index: {} not in range of possible actions'.format(index))

    def _dqn_predict_one(self, state, target=False):
        """
        Returns a model prediction given a state.

        Parameters:
            state (numpy.array)
            target (bool)

        Returns:
            numpy.array     Q值数组
        """

        return self._dqn_predict(state.reshape(1, self.state_size), target=target).flatten()

    def _dqn_predict(self, states, target=False):
        """
        Returns a model prediction given an array of states.

        Parameters:
            states (numpy.array)
            target (bool)

        Returns:
            numpy.array
        """

        if target:
            return self.tar_model.predict(states)
        else:
            return self.beh_model.predict(states)

    def add_experience(self, state, action, reward, next_state, done):
        """
        Adds an experience tuple made of the parameters to the memory.

        Parameters:
            state (numpy.array)
            action (int)
            reward (int)
            next_state (numpy.array)
            done (bool)

        """

        if len(self.memory) < self.max_memory_size:
            self.memory.append(None)
        self.memory[self.memory_index] = (state, action, reward, next_state, done)
        self.memory_index = (self.memory_index + 1) % self.max_memory_size

    def empty_memory(self):
        """Empties the memory and resets the memory index."""

        self.memory = []
        self.memory_index = 0

    def is_memory_full(self):
        """Returns true if the memory is full."""

        return len(self.memory) == self.max_memory_size

    def copy(self):
        """Copies the behavior model's weights into the target model's weights."""

        self.tar_model.set_weights(self.beh_model.get_weights())

    def save_weights(self):
        """Saves the weights of both models in two h5 files."""

        if not self.save_weights_file_path:
            return
        beh_save_file_path = re.sub(r'\.h5', r'_beh.h5', self.save_weights_file_path)
        self.beh_model.save_weights(beh_save_file_path)
        tar_save_file_path = re.sub(r'\.h5', r'_tar.h5', self.save_weights_file_path)
        self.tar_model.save_weights(tar_save_file_path)

    def _load_weights(self):
        """Loads the weights of both models from two h5 files."""

        if not self.load_weights_file_path:
            return
        beh_load_file_path = re.sub(r'\.h5', r'_beh.h5', self.load_weights_file_path)
        self.beh_model.load_weights(beh_load_file_path)
        tar_load_file_path = re.sub(r'\.h5', r'_tar.h5', self.load_weights_file_path)
        self.tar_model.load_weights(tar_load_file_path)
        # print("Load_weights")
