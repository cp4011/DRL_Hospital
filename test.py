from config import WARMUP_MEM, bot_actions, all_actions, required_slots, max_round
import json
from dqn_agent import DQNAgent
from state_tracker import StateTracker
from user_sim import UserSimulator
from user import User
from rb_policy import Policy


with open("constants.json") as f:
    constants = json.load(f)

run_dict = constants['run']
USE_USERSIM = run_dict['usersim']
NUM_EP_TEST = run_dict['num_ep_run']
MAX_ROUND_NUM = run_dict['max_round_num']

# Init. Objects
if USE_USERSIM:
    user = UserSimulator()
else:
    user = User(constants)
state_tracker = StateTracker()
dqn_agent = DQNAgent(state_tracker.get_state_size(), constants)     # 已经加载已经训练的参数_load_weights()
policy = Policy()


def test_run():
    """
    Runs the loop that tests the agent.

    Tests the agent on the goal-oriented chatbot task. Only for evaluating a trained agent. Terminates when the episode
    reaches NUM_EP_TEST.

    """

    print('Testing Started...')
    episode = 0
    while episode < NUM_EP_TEST:
        print("\nLoad weights successfully...\n")
        # print("\nRegister Hospital Domain")
        episode += 1
        ep_reward = 0
        done = False
        # Get initial state from state tracker
        state_tracker.reset()
        # user_action = user.init()
        user_action = get_user_action()
        # print("User Action:", user_action)
        state_tracker.update_state(user_action)
        state = state_tracker.get_state()
        # print(dqn_agent.tar_model.get_weights())
        while not done:
            dqn_agent.get_action_mask(state_tracker.accumulated_memory, state_tracker.db_result)  # 参数nearby_hospital
            # Agent takes action given state tracker's representation of dialogue
            agent_action_index, agent_action = dqn_agent.get_action(state)

            # print("Agent information:", agent_action_index, agent_action, state_tracker.accumulated_memory)

            user_action, reward, done, success = user.step(agent_action, state_tracker.num_turn)
            if success in ['-1', '1']:
                state_tracker.reset()
                break
            ep_reward += reward
            # Update state tracker with user action
            state_tracker.update_state(user_action)
            # Grab "next state" as state
            state = state_tracker.get_state(done)
        print('Episode: {} Success: {} Reward: {}'.format(episode, success, ep_reward))
    print('...Testing Ended')


def sim_run():
    policy.reset()
    done = False
    user_action = user.init()
    print("user:  ", user_action["intent"], user_action["inform_slots"])
    while not done:
        policy.update_state(user_action)
        context = policy.policy(user_action)
        print("agent:  ", context["action"], "inform_slots: ", context["inform_slots"])
        user_action, reward, done, success = user.step(context["slot_to_fill"])
        print("user:  ", user_action["intent"], user_action["inform_slots"])


def get_user_action():
    response = {'intent': '', 'inform_slots': {}, 'request_slots': {}}
    while True:
        input_string = input('Input: ')
        if not input_string or input_string == "q":
            break
        chunks = input_string.split('/')

        intent_correct = True
        if chunks[0] not in all_actions:
            intent_correct = False
        response['intent'] = chunks[0]

        informs_correct = True
        if len(chunks[1]) > 0:
            informs_items_list = chunks[1].split(', ')
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

        if intent_correct and informs_correct and requests_correct:
            break

    return response


def return_success():
    """
    Ask the user in console to input success (-1, 0 or 1) for (loss, neither loss nor win, win).
    Returns:
        int: Success: -1, 0 or 1
    """
    success = -2
    while success not in (-1, 0, 1):
        success = int(input('Success or Fail? : '))
    return success


def real_run():
    done = False
    policy.reset()
    user_action = get_user_action()
    print("User:  ", user_action)
    while not done:
        policy.update_state(user_action)
        context = policy.policy(user_action)
        print("Agent:  ", context["action"], "    memory:", context["inform_slots"])
        # user_action = get_user_action()
        user_action, reward, done, success = user.step(context["slot_to_fill"], context["num_turn"])
        print("User  : ", user_action)
        success = return_success()
        if success != 0:
            done = True


test_run()
# real_run()

dict1 = {'I want to make an appointment with the fever clinic department of the hospital': 'inform_department/department: fever clinic department/',}
nlg = {"api_call register": 'API_CALL_Register_Hospital(<hospital>=nanjing gulou hospital, <department>=fever clinic department, \n <time>=this afternoon, <information>=zhong nanshan, 84 years old, 13333333333)',
       "api_call nearby_hospital": 'API_CALL_Nearby_Hospital(<location>=zhujiang road)',
       "ask_hospital": 'Ok which hospital do you want to make an appointment with',
       "ask_department": 'Ok which department do you want to make an appointment with',
       "ask_time": 'Ok when do you want to make an appointment',
       "ask_information": 'Can you provide me your personal information (name, age and phone number)',
       "not_found": 'no match found'
       }

"""
register_hospital//
inform_hospital/hospital: Nanjing Gulou Hospital/
inform_department/department: fever clinic department/
inform_time/time: tomorrow/
inform_information/information: Zhong Nanshan, 84 years old, male/
thanks//
"""