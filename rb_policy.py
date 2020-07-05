import json
import random
import database
import copy
import config

DB = database.DB
# entity => action
Bot_actions = config.bot_actions
required_slots = config.required_slots

mock_user_action = {
    "intent": "RegisterHospital",
    "inform_slots": {"hospital": None, "department": None, "time": None, "information": None},
    "request_slots": {},
}


class Policy:
    def  __init__(self):
        self.num_turn = 0
        self.history = []
        self.accumulated_memory = {}
        self.db_result = True
        self.required_slots = copy.deepcopy(required_slots)

    def reset(self):
        self.num_turn = 0
        self.history = []
        self.accumulated_memory = {}
        self.db_result = True
        self.required_slots = copy.deepcopy(required_slots)

    def db_search(self, memory):
        """ 检查 knowledgebase 中是否有对应餐厅信息，如果有返回下一个要问的 entity，如果没有返回空
        :param memory: accumulated memory
        :returns:
        :rtype:
        """
        query_success = True
        for k, v in memory.items():
            if v is not None:
                query_success = query_success and v in DB[k]

        if query_success:  # 条件与DB匹配
            # 随机选一个 slot 为空的
            slots = [(k, v) for k, v in memory.items()]
            empty_slots = list(filter(lambda slot: slot[1] is None, slots))     # 过滤出表达式为true的项

            for k, v in memory.items():
                if v is not None:
                    if k in self.required_slots:
                        self.required_slots.remove(k)

            # 所有的 entity 都有值了，也是成功了，只是没有下一个要问的 entity 了
            if len(self.required_slots) == 0:
                return "success", True

            # query_entity = random.choice(self.required_slots)
            query_entity = self.required_slots[0]
            return query_entity, True
        else:  # 条件与DB不匹配
            return "fail", False

    def update_state(self, user_action):
        self.num_turn += 1
        if len(self.history) == 3:
            self.history.pop(0)
        self.history.append(user_action["intent"])
        self.accumulated_memory.update(user_action["inform_slots"])

    def policy(self, user_action):
        query_entity, self.db_result = self.db_search(self.accumulated_memory)
        bot_action = Bot_actions[query_entity]
        if len(self.history) == 3:
            self.history.pop(0)
        self.history.append(bot_action)
    
        context = {
            # "action": {
            #     "intent": bot_action,
            #     "inform_slots": user_action["inform_slots"],  # db_search
            #     "request_slots": user_action["request_slots"],
            # },
            "action": bot_action,
            "num_turn": self.num_turn,
            "inform_slots": copy.deepcopy(self.accumulated_memory),  # accumulated
            "history": copy.deepcopy(self.history),
            "slot_to_fill": query_entity,
            "db_result": self.db_result
        }
    
        return context


if __name__ == "__main__":
    p = Policy()
    user_action = None
    print("RegisterHospital Domain")
    print("example:", json.dumps(mock_user_action))
    while True:
        user_action = input(">>> ").strip()

        if not user_action or user_action == "q":
            break

        # try:
        user_action = json.loads(user_action)
        print(user_action)
        p.update_state(user_action)
        try:
            if isinstance(p.policy(user_action), dict):
                print(p.policy(user_action)["action"])
            else:
                print(p.policy(user_action))      # “not found"
            print("accumulated_memory:", p.accumulated_memory)
        except Exception as e:
            print(e)
            print("Parse json failed, please re-input...")
