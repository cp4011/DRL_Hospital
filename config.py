max_round = 10
WARMUP_MEM = 10000
FAIL = -1
NO_OUTCOME = 0
SUCCESS = 1
state_size = 36

required_slots = ["hospital", "department", "time", "information"]
required_slots_index = {slot: i for i, slot in enumerate(required_slots)}

all_user_actions = ["register_hospital", "ask_nearby_hospital", "inform_hospital", "inform_department", "inform_time", "inform_information", "thanks", "reset"]
user_actions = {            # Policy输出的slot_to_fill: 将entity映射到user的action  entity => action
    "RegisterHospital": "register_hospital",
    "NearbyHospital": "ask_nearby_hospital",
    "hospital": "inform_hospital",
    "department": "inform_department",
    "time": "inform_time",
    "information": "inform_information",
    "success": "thanks",
    "fail": "reset"
}

all_bot_actions = ["api_call register", "api_call nearby_hospital", "ask_hospital", "ask_department", "ask_time", "ask_information", "not_found"]
bot_actions = {
    "RegisterHospital": "ask_hospital",
    "NearbyHospital": "api_call nearby_hospital",
    "hospital": "ask_hospital",
    "department": "ask_department",
    "time": "ask_time",
    "information": "ask_information",
    "success": "api_call register",
    "fail": "not_found"
}

all_actions = all_bot_actions + all_user_actions
# print(all_actions)

am_dict = {
    '0000': [3, 4, 5, 6],
    '0001': [3, 4, 5],
    '0010': [3, 4, 6],
    '0011': [3, 4],
    '0100': [3, 5, 6],
    '0101': [3, 5],
    '0110': [3, 6],
    '0111': [3],
    '1000': [4, 5, 6],
    '1001': [4, 5],
    '1010': [4, 6],
    '1011': [4],
    '1100': [5, 6],
    '1101': [5],
    '1110': [6],
    '1111': [1]
}
nearby_hospital = [2]
