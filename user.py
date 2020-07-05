from config import FAIL, SUCCESS, all_actions, required_slots, user_actions
from utils import reward_function
from database import DB
import random


class User:
    """Connects a real user to the conversation through the console."""

    def __init__(self, constants):
        self.max_round = constants['run']['max_round_num']
        self.user_action = {"intent": "register_hospital", "inform_slots": {slot: None for slot in required_slots}, "request_slots": {}}

    # def init(self):
    #     slot = required_slots[0]
    #     # slot = random.choice(required_slots)
    #     value = random.choice(DB[slot])
    #     self.user_action["inform_slots"] = {slot: None for slot in required_slots}
    #     self.user_action["inform_slots"][slot] = value
    #     self.user_action["intent"] = user_actions[slot]
    #     return self.user_action

    def change_format(self, uttrance, intent, slot_values, success=False):
        if intent == "NearbyHospital":
            intent = 'ask_nearby_hospital'
        else:
            intent = 'register_hospital'
        user_input = intent + '/'
        for k, v in slot_values.items():
            user_input += k + ': ' + v
        return user_input

    def reset(self):
        """
        Reset the user.
        Returns:
            dict: The user response
        """

        return self._return_response()

    def _return_response(self):     # 将控制台真实user的输入处理成格式化的user response
        """
        Asks user in console for response then receives a response as input.

        Format must be like this: request/moviename: room, date: friday/starttime, city, theater
        or inform/moviename: zootopia/
        or request//starttime
        or done//
        intents, informs keys and values, and request keys and values cannot contain / , :

        Returns:
            dict: The response of the user
        """

        response = {'intent': '', 'inform_slots': {}, 'request_slots': {}}
        while True:
            input_string = input('Response: ')
            chunks = input_string.split('/')

            intent_correct = True
            if chunks[0] not in all_actions:
                intent_correct = False
            response['intent'] = chunks[0]

            informs_correct = True
            if len(chunks[1]) > 0:
                informs_items_list = chunks[1].split('; ')
                for inf in informs_items_list:
                    inf = inf.split(': ')
                    if inf[0] not in required_slots:
                        informs_correct = False
                        break
                    response['inform_slots'][inf[0]] = inf[1]

            requests_correct = True
            if len(chunks[2]) > 0:
                requests_key_list = chunks[2].split(', ')
                for req in requests_key_list:
                    if req not in all_actions:
                        requests_correct = False
                        break
                    response['request_slots'][req] = 'UNK'

            if intent_correct and informs_correct and requests_correct:     # 当三者都输入正确后，循环结束
                break

        print(response)
        return response

    def _return_success(self):
        """
        Ask the user in console to input success (-1, 0 or 1) for (loss, neither loss nor win, win).
        Returns:
            int: Success: -1, 0 or 1
        """

        success = -2
        while success not in (-1, 0, 1):
            success = int(input('Success?: '))
        return success

    def step(self, agent_action, num_turn):
        """
        Return the user's response, reward, done and success.
        Parameters:
            agent_action (dict): The current action of the agent
        Returns:
            dict: User response
            int: Reward
            bool: Done flag
            int: Success: -1, 0 or 1 for loss, neither win nor loss, win
        """

        # Assertions ----
        # No unk in agent action informs
        # for value in agent_action['inform_slots'].values():
        #     assert value != 'UNK'
        #     assert value != 'PLACEHOLDER'
        # # No PLACEHOLDER in agent_action at all
        # for value in agent_action['request_slots'].values():
        #     assert value != 'PLACEHOLDER'
        # ---------------
        print('Agent Action: {}'.format(agent_action))
        # print('Agent Action: {}'.format(nlg.agent_action_nlg(agent_action)))

        done = False
        user_response = {'intent': '', 'request_slots': {}, 'inform_slots': {}}

        # First check round num, if equal to max then fail
        if num_turn == self.max_round:  # 如果当前已经达到max_round，success直接为fail，user_response['intent'] = 'done'
            success = FAIL
            user_response['intent'] = 'done'    # user的intent为'done'，传递给State Tracker后，ST的state全置为0
        else:
            user_response = self._return_response()
            success = self._return_success()

        if success == FAIL or success == SUCCESS:   # 有success时，done已经为true
            done = True

        assert 'UNK' not in user_response['inform_slots'].values()
        assert 'PLACEHOLDER' not in user_response['request_slots'].values()

        reward = reward_function(success, self.max_round)

        return user_response, reward, done, True if success is 1 else False

