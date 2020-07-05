import json, copy
import numpy as np
from dqn_agent import DQNAgent
from state_tracker import StateTracker
from user_for_platform import User
from utils import nlg, reward_function


with open("constants.json") as f:
    constants = json.load(f)

from keras import backend as K


class InteractiveSession:
    def __init__(self):
        K.clear_session()
        self.user = User(constants)
        self.state_tracker = StateTracker()
        self.dqn_agent = DQNAgent(self.state_tracker.get_state_size(), constants)
        # self.max_round = 10
        # self.episode_reward = 0
        # self.last_action = 2
        # self.done = False
        # self.experience = []
        self.state = np.zeros(self.state_tracker.get_state_size())
        self.state_tracker.reset()
        print('Testing Started...')

    def reset(self):
        # self.episode_reward = 0
        # self.last_action = 2
        # self.done = False
        self.state_tracker.reset()

    def interact(self, utterance, intent, slot_values, success='0'):
        # get input from user
        u = utterance

        # check if user wants to begin new session
        if u == 'clear' or u == 'reset' or u == 'restart':
            self.reset()
            return "reset successfully"

        # check for entrance and exit command
        elif u == 'exit' or u == 'stop' or u == 'quit' or u == 'q':
            self.reset()
            return "Thank you for using"

        elif u == 'hello' or u == 'hi':
            self.reset()
            return "what can i do for you"

        elif u == 'thank you' or u == 'thanks' or u == 'thank you very much':
            self.reset()
            return 'you are welcome'

        else:
            # reward = reward_function(int(success), self.max_round)
            # self.episode_reward += reward
            # if success in ['1', '-1']:
            #     self.done = True
            # self.last_state = copy.deepcopy(self.state)
            user_action = self.user.change_format(utterance, intent, slot_values)
            # print(user_action)
            self.state_tracker.update_state(user_action)
            self.state = self.state_tracker.get_state()
            # self.experience.append((self.last_state, copy.deepcopy(self.last_action), self.episode_reward, self.state, self.done))
            if user_action["intent"] == 'ask_nearby_hospital':
                self.dqn_agent.get_action_mask(self.state_tracker.accumulated_memory, self.state_tracker.db_result, True)
            else:
                self.dqn_agent.get_action_mask(self.state_tracker.accumulated_memory, self.state_tracker.db_result)
            agent_action_index, agent_action = self.dqn_agent.get_action(self.state)
            if agent_action_index == 0:
                slot_values = ', '.join(self.state_tracker.accumulated_memory.values())
                response = "API_CALL " + slot_values
                # Execute successfully and begin new session
                self.reset()
                return response
            return nlg[agent_action]


if __name__ == '__main__':
    from keras import backend as K
    K.clear_session()
    # create interactive session
    isess = InteractiveSession()
    # begin interaction

    utterance = "hello"
    print('User: ', utterance)
    print("Bot: ", isess.interact(utterance, 'Greeting', {}))

    utterance = 'I want to make an appointment with the fever clinic department of the hospital'
    print('User: ', utterance)
    print("Bot: ", isess.interact(utterance, 'RegisterHospital', {'department': 'fever clinic department'}))

    utterance = 'this afternoon please'
    print('User: ', utterance)
    print("Bot: ", isess.interact(utterance, 'RegisterHospital', {'time': 'this afternoon'}, '0'))

    utterance = 'I am Zhong Nanshan, 84 years old, male'
    print('User: ', utterance)
    print("Bot: ", isess.interact(utterance, 'RegisterHospital', {'name': 'Zhong Nanshan', 'age': '84 years old', 'gender': 'male'}, '0'))

    utterance = 'which hospitals are nearby around Gulou'
    print('User: ', utterance)
    print("Bot: ", isess.interact(utterance, 'RegisterHospital', {}))

    utterance = 'Nanjing Gulou Hospital please'
    print('User: ', utterance)
    print("Bot: ", isess.interact(utterance, 'RegisterHospital', {'hospital': 'Nanjing Gulou Hospital'}))

    utterance = "thanks"
    print('User: ', utterance)
    print("Bot: ", isess.interact(utterance, 'thanks', {}))

