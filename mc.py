import random

# reward for each state
S = [-1, -1, 2, 5, 1000]
# left, right
actions = [-1, 1]
# calculated value for each state
Vs = [[] for _ in range(len(S))]
# calculated value for each state-action pair
Qs = dict()
# discount rate
gamma = 0.9
# number of episodes to run
episodes = 100

def get_next_state(state, action):
    # if the agent is in state 3 and takes action 1 (move right),
    # it has a 70% chance to move to state 0 and a 30% chance to move to state 4
    if state == 3 and action == 1:
        return random.choices([0, 4], weights=[0.7, 0.3], k=1)[0]
    next_state = state + action
    if next_state < 0:
        next_state = 0
    elif next_state >= len(S):
        next_state = len(S) - 1
    return next_state

for _ in range(5):
    for i in range(episodes):
        # current state
        s = 0
        # first time monte carlo value estimation for state s
        ft = [0] * len(S)
        ft[s] = S[s]
        # time
        t = 0
        # first time monte carlo value estimation for q value for state s and action a
        ft_q = dict()

        while s != len(S) - 1:
            t += 1
            a = random.choice(actions)
            next_s = get_next_state(s, a)
            
            # TODO - fix bug -- v should be S[next_s] instead of gamma**t * S[next_s] inside ft[next_s]...
            # decay reward for the current state
            v = gamma**t * S[next_s]

            # we can update the value if this is not the first time we visit state s
            can_update_ft = True
            # if this is the first time we visit state s, set its value to v
            if ft[next_s] == 0:
                ft[next_s] = v
                can_update_ft = False
            
            can_update_ft_q = True
            # if this is the first time we take action a in state s, set its value to the reward of the next state
            if (s, a) not in ft_q:
                ft_q[(s, a)] = v
                can_update_ft_q = False

            # update value for all states in the history
            for tt in range(len(S)):
                if ft[tt] != 0 and (tt != next_s or can_update_ft):
                    ft[tt] += v
            
            # update value for all state-action pairs in the history
            for (ss, aa), vv in ft_q.items():
                if (ss, aa) != (s, a) or can_update_ft_q:
                    ft_q[(ss, aa)] += v

            s = next_s

        for s in range(len(S)):
            if ft[s] != 0:
                Vs[s].append(ft[s])
        for (s, a), v in ft_q.items():
            Qs[(s, a)] = Qs.get((s, a), []) + [v]

    Vs_avg = [sum(vs) / len(vs) if vs else 0 for vs in Vs]
    Qs_avg = {k: sum(vs) / len(vs) for k, vs in Qs.items()}
    print("Estimated value for each state:", Vs_avg)
    print("Estimated value for each state-action pair:", Qs_avg)

    print("\nOptimal policy:")
    s = 0
    while s != len(S) - 1:
        a = max(actions, key=lambda a: Qs_avg.get((s, a), float('-inf')))
        print(f"At state {s}, take action {a}")
        s = s + a
    print("\n" + "="*50 + "\n")
