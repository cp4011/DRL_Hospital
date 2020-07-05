from rb_policy import Policy
from user_sim import UserSimulator
from config import WARMUP_MEM, bot_actions, all_actions, required_slots, max_round, state_size
import copy
import json
from dqn_agent import DQNAgent
from state_tracker import StateTracker


def warmup_run():
    total_step = 0
    memorys = []
    # 记录WARMUP_MEM=1000 rounds的经验。其中每个episode有很多round(s, a, r, s_, done)，直到done = True 即一轮episode结束
    while total_step != WARMUP_MEM:
        p.reset()
        # print("user:  ", user_action["intent"], user_action["inform_slots"])
        done = False
        user_action = sim.init()
        # print(user_action)
        p.update_state(user_action)
        # print(p.history, p.accumulated_memory, p.num_turn, p.db_result)
        s = p.history[-1], p.history[-2] if len(p.history) >= 2 else None, copy.deepcopy(p.accumulated_memory), p.num_turn, p.db_result
        while not done:
            context = p.policy(user_action)
            user_action, reward, done, success = sim.step(context["slot_to_fill"], context["num_turn"])
            # print("user:  ", user_action["intent"], user_action["inform_slots"])
            p.update_state(user_action)
            s_next = p.history[-1], p.history[-2], copy.deepcopy(p.accumulated_memory), p.num_turn, p.db_result
            bot_action = bot_actions[context["slot_to_fill"]]
            memorys.append((s, bot_action, reward, s_next, done))
            if user_action["intent"] == "thanks":
                p.reset()
            s = s_next

        total_step += 1
    print("memorys:", len(memorys))
    return memorys


def save_experience(memorys):
    with open(r'data\experience', 'w', encoding='utf-8') as f:
        for memory in memorys:
            state, agent_action_index, reward, next_state, done = memory
            state = ' '.join(map(str, state))
            next_state = ' '.join(map(str, next_state))
            memory = [state, agent_action_index, reward, next_state, done]
            memory = '||'.join(map(str, memory)) + '\n'
            f.writelines(memory)


if __name__ == "__main__":
    from keras import backend as K
    K.clear_session()

    sim = UserSimulator()
    p = Policy()
    st = StateTracker()
    ms = warmup_run()
    # for i in range(6):
    #     print(ms[i])
    # print(st.get_tuple_state(ms[4]))
    memorys = []
    print("Get experience tuple state: ...")
    for m in ms:
        memorys.append(st.get_tuple_state(m))

    save_experience(memorys)

    with open("constants.json") as f:
        constants = json.load(f)
    dqn = DQNAgent(state_size, constants)
    dqn.memory = memorys
    print("QDN training: ...")
    dqn.train()
    print("QDN training ended: ...")
    dqn.save_weights()

