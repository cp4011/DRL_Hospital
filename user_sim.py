import random
import database
from rb_policy import Policy
from config import FAIL, NO_OUTCOME, SUCCESS, max_round
import config
from utils import reward_function

DB = database.DB
Required_slots = config.required_slots
User_actions = config.user_actions


class UserSimulator:
    def __init__(self):
        self.user_action = {"intent": "register_hospital", "inform_slots": {slot: None for slot in Required_slots}, "request_slots": {}}

    def init(self):
        # slot = random.choice(Required_slots)
        slot = Required_slots[0]
        value = random.choice(DB[slot])
        self.user_action["inform_slots"] = {slot: None for slot in Required_slots}
        self.user_action["inform_slots"][slot] = value
        self.user_action["intent"] = User_actions[slot]
        return self.user_action
    
    def step(self, slot, num_turn):
        # print("slot", slot)
        if slot in Required_slots:          # agent返回的slot_to_fill是必须要填值的slot
            inform_slot_value = random.choice(DB[slot])
            self.user_action["inform_slots"][slot] = inform_slot_value
        self.user_action["intent"] = User_actions[slot]
        done = False
        success = NO_OUTCOME
        if slot == "success":
            done = True
            success = SUCCESS
        if slot == "fail" or num_turn > max_round:
            done = True
            success = FAIL
        reward = reward_function(success, max_round)
        return self.user_action, reward, done, success


if __name__ == "__main__":
    sim = UserSimulator()
    p = Policy()
    p.reset()
    user_action = sim.init()
    print("user:  ", user_action["intent"], user_action["inform_slots"])
    done = False
    while not done:
        p.update_state(user_action)
        context = p.policy(user_action)
        print("agent:  ", context["action"], "slot_to_fill: ", context["slot_to_fill"], "+++", len(context["history"]), "+++", "inform_slots: ", context["inform_slots"], context["db_result"])
        user_action, reward, done, success = sim.step(context["slot_to_fill"], context["num_turn"])
        print("user:  ", user_action["intent"], user_action["inform_slots"])
