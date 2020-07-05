class User:
    """Connects a real user to the conversation through the console."""

    def __init__(self, constants):
        self.max_round = constants['run']['max_round_num']
        self.user_action = {"intent": "register_hospital", "inform_slots": {}, "request_slots": {}}

    def change_format(self, utterance, intent, slot_values):
        self.user_action = {"intent": "register_hospital", "inform_slots": {}, "request_slots": {}}
        if 'hospital' in utterance and 'near' in utterance or 'nearby' in utterance:            # 已改
            intent = 'ask_nearby_hospital'
        else:
            intent = 'register_hospital'
        self.user_action["intent"] = intent
        for k, v in slot_values.items():
                self.user_action["inform_slots"][k] = v
        # print("user_input: ", self.user_action)
        return self.user_action
