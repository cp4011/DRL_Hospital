import numpy as np
from utils import convert_list_to_dict
import copy
from config import all_actions, required_slots, max_round
import random
from database import DB


class StateTracker:
    def __init__(self):
        self.actions_dict = convert_list_to_dict(all_actions)
        self.num_actions = len(self.actions_dict)  # 15
        self.slots_dict = convert_list_to_dict(required_slots)
        self.num_slots = len(self.slots_dict)  # 4
        self.max_round_num = max_round  # 15
        self.none_state = np.zeros(self.get_state_size())  # 36
        self.required_slots = required_slots

    def get_state_size(self):
        return 2 * self.num_actions + self.num_slots + 2    # 2 * 15 + 4 + 2 = 36

    def reset(self):
        self.accumulated_memory = {}
        # A list of the dialogues (dicts) by the agent and user so far in the conversation
        self.history = []
        self.num_turn = 0
        self.db_result = True

    def update_state(self, user_action):
        self.num_turn += 1
        if len(self.history) == 3:
            self.history.pop(0)
        self.history.append(user_action["intent"])
        slot_values = user_action["inform_slots"]
        for slot, value in slot_values.items():
            if slot in ['name', 'age', 'phone_number']:
                self.accumulated_memory['information'] = ', '.join([slot_values['name'], slot_values['age'], slot_values['gender']])
                break
            if slot in self.required_slots:
                self.accumulated_memory[slot] = value

    def get_state(self, done=False):
        if done:
            return self.none_state  # [0, 0, ...]

        user_action = self.history[-1]
        last_bot_action = self.history[-2] if len(self.history) > 1 else None
        _, self.db_result = self.db_search(self.accumulated_memory)

        user_action_rep = np.zeros((self.num_actions,))  # 15
        user_action_rep[self.actions_dict[user_action]] = 1.0

        last_bot_action_rep = np.zeros((self.num_actions,))  # 15
        if last_bot_action:
            last_bot_action_rep[self.actions_dict[last_bot_action]] = 1.0

        accumulated_memory_rep = np.zeros((self.num_slots,))  # 4
        for key in self.accumulated_memory:
            if self.accumulated_memory[key]:
                accumulated_memory_rep[self.slots_dict[key]] = 1.0

        db_result_rep = np.zeros((1,))                      # 1
        if self.db_result:
            db_result_rep = 1.0
        state_rep = np.hstack([user_action_rep, last_bot_action_rep, accumulated_memory_rep, self.num_turn, db_result_rep]).flatten()
        return state_rep

    def get_tuple_state(self, memory):
        state, bot_action, reward, state_next, done = memory

        def state_rep_fun(state):
            user_act_rep_1 = np.zeros((self.num_actions,))  # 15
            user_act_rep_1[self.actions_dict[state[0]]] = 1.0

            user_act_rep_2 = np.zeros((self.num_actions,))  # 15
            if state[1]:
                user_act_rep_2[self.actions_dict[state[1]]] = 1.0

            accumulated_memory_rep = np.zeros((self.num_slots,))  # 4
            for key in state[2]:
                if state[2][key]:
                    accumulated_memory_rep[self.slots_dict[key]] = 1.0

            db_rearch_rep = np.zeros((1,))                      # 1
            if state[4]:
                db_rearch_rep = 1.0
            state_rep = np.hstack([user_act_rep_1, user_act_rep_2, accumulated_memory_rep, state[3], db_rearch_rep]).flatten()
            return state_rep
        state_rep = state_rep_fun(state)
        bot_action_index = self.actions_dict[bot_action]
        if done:
            state_next_rep = self.none_state  # 30
        else:
            state_next_rep = state_rep_fun(state_next)
        return state_rep, bot_action_index, reward, state_next_rep, done

    def _map_index_to_action(self, index):
        for (i, action) in enumerate(all_actions):
            if index == i:
                return copy.deepcopy(action)

    def _map_action_to_index(self, response):
        for (i, action) in enumerate(all_actions):
            if response == action:
                return i

    def db_search(self, memory):
        query_success = True
        for k, v in memory.items():
            if v is not None:
                query_success = query_success and v in DB[k]

        if query_success:  # 条件与DB匹配
            # 随机选一个 slot 为空的
            slots = [(k, v) for k, v in memory.items()]
            empty_slots = list(filter(lambda slot: slot[1] is None, slots))

            # 所有的 entity 都有值了，也是成功了，只是没有下一个要问的 entity 了
            if not empty_slots:
                return "success", True

            query_entity, _ = random.choice(empty_slots)
            return query_entity, True
        else:  # 条件与DB不匹配
            return "fail", False
