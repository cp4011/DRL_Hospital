from config import FAIL, SUCCESS


def convert_list_to_dict(lst):
    """    Convert list to dict where the keys are the list elements, and the values are the indices of the elements in the list
    Parameters:
        lst (list)
    Returns:
        dict
    """

    if len(lst) > len(set(lst)):
        raise ValueError('List must be unique!')
    return {k: v for v, k in enumerate(lst)}


def reward_function(success, max_round):
    """    Return the reward given the success.
    Return -1 + -max_round if success is FAIL, -1 + 2 * max_round if success is SUCCESS and -1 otherwise.
    Parameters:
        success (int)
    Returns:
        int: Reward
    """

    reward = 30
    if success == FAIL:
        reward += -max_round
    elif success == SUCCESS:
        reward = max_round
    else:
        reward -= 2
    return reward


nlg = {"api_call register": 'API_CALL_Register_Hospital(memory)',
       "api_call nearby_hospital": 'API_CALL_Nearby_Hospital(location)',
       "ask_hospital": 'Ok which hospital do you want to make an appointment with',
       "ask_department": 'Ok which department do you want to make an appointment with',
       "ask_time": 'Ok when do you want to make an appointment',
       "ask_information": 'Can you provide me your personal information (name, age and phone number)',
       "not_found": 'no match found'
       }

