import random

# reward for each state
S = [-1, -1, 2, 5, 100]
# left, right
actions = [-1, 1]
# calculated value for each state
Vs = [[] for _ in range(len(S))]
# calculated weight for each state
Ws = [[] for _ in range(len(S))]
# calculated value for each state-action pair
Qs = dict()
Qs_avg = dict()
# calculated weight for each state-action pair
QWs = dict()
# discount rate
gamma = 0.9
# number of episodes to run
episodes = 10
# hyperparam - probability of exploration for the on policy
# off policy will be always go right
epsilon = 0.05

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

def get_greedy_action(Qs_avg, s):
    # get the action with the highest estimated value for the current state
    return max(actions, key=lambda a: Qs_avg.get((s, a), float('-inf')))

def get_importance_sampling_weight(a, s, Qs_avg):
    # off policy
    pi = 1 if a == 1 else 0
    greedy_a = get_greedy_action(Qs_avg, s)
    b = 1 - epsilon + epsilon / len(actions) if a == greedy_a else epsilon / len(actions)
    return pi / b

for _ in range(5):
    for i in range(episodes):
        # current state
        s = 0
        # first time monte carlo value estimation for state s
        ft = [None] * len(S)
        # first time monte carlo value estimation for q value for state s and action a
        ft_q = dict()
        # history
        history = []

        while s != len(S) - 1:
            p = random.random()
            if s == 0:
                # don't wanna get stuck in state 0, always go right
                a = 1
            else:
                # get_greedy_action may return left when empty so we need to check if Qs_avg is empty
                a = random.choice(actions) if p < (epsilon / len(actions)) or len(Qs) == 0 else get_greedy_action(Qs_avg, s)
            next_s = get_next_state(s, a)
            history.append((s, a, S[next_s]))
            s = next_s

        G = 0
        # W = ∏ π(aₜ|sₜ) / b(aₜ|sₜ) for t=0 to T-1 for a single episode
        W = 1.0
        # first time weight
        ft_w = dict()
        for h in reversed(history):
            s, a, r = h
            # calculate return G for the first time we visit state s
            G = r + gamma * G
            # this is the first time we visit state s
            ft[s] = G
            ft_q[(s, a)] = G
            ft_w[s] = W
            if a != 1:
                break
            W *= get_importance_sampling_weight(a, s, Qs_avg)

        for s in range(len(S)):
            if ft[s] is not None:
                Vs[s].append(ft[s])
                Ws[s].append(ft_w[s])
        for (s, a), v in ft_q.items():
            Qs[(s, a)] = Qs.get((s, a), []) + [v]
            QWs[(s, a)] = QWs.get((s, a), []) + [ft_w[s]]

    Vs_avg = []
    for vs, ws in zip(Vs, Ws):
        gg = 0
        for i in range(len(vs)):
            gg += vs[i] * ws[i]
        Vs_avg.append(gg / sum(ws) if sum(ws) > 0 else 0)

    for (s, a), vs in Qs.items():
        gg = 0
        for i in range(len(vs)):
            gg += vs[i] * QWs[(s, a)][i]
        Qs_avg[(s, a)] = gg / sum(QWs[(s, a)]) if sum(QWs[(s, a)]) > 0 else 0

    print("Estimated value for each state:", Vs_avg)
    print("Estimated value for each state-action pair:", Qs_avg)

    print("\nOptimal policy:")
    s = 0
    while s != len(S) - 1:
        a = max(actions, key=lambda a: Qs_avg.get((s, a), float('-inf')))
        print(f"At state {s}, take action {a}")
        s = s + a
    print("\n" + "="*50 + "\n")

"""
Estimated value for each state: [77.75, 68.97628636900217, 95.0, 100.0, 0]
Estimated value for each state-action pair: {(3, 1): 100.0, (2, 1): 95.0, (1, 1): 87.5, (0, 1): 77.75, (1, -1): 68.97500000000001}

Optimal policy:
At state 0, take action 1
At state 1, take action 1
At state 2, take action 1
At state 3, take action 1

==================================================

Estimated value for each state: [77.74954822689472, 68.97628240367989, 94.11188715848283, 81.31483001746896, 0]
Estimated value for each state-action pair: {(3, 1): 81.47833744264486, (2, 1): 94.11188715848283, (1, 1): 87.49998640566616, (0, 1): 77.74954822689472, (1, -1): 68.97500000000001, (3, -1): 45.88179297500002}

Optimal policy:
At state 0, take action 1
At state 1, take action 1
At state 2, take action 1
At state 3, take action 1

==================================================

Estimated value for each state: [77.74911704298931, 68.97627892567625, 93.3709389075485, 77.01471086718648, 0]
Estimated value for each state-action pair: {(3, 1): 77.1222317974499, (2, 1): 93.3709389075485, (1, 1): 87.49997343039388, (0, 1): 77.74911704298931, (1, -1): 68.97500000000001, (3, -1): 45.88179297500002}

Optimal policy:
At state 0, take action 1
At state 1, take action 1
At state 2, take action 1
At state 3, take action 1

==================================================

Estimated value for each state: [77.74880538020939, 68.97627876386046, 92.80163958714262, 76.0311823424541, 0]
Estimated value for each state-action pair: {(3, 1): 76.28688204590956, (2, 1): 92.80163958714262, (1, 1): 87.49996405131361, (0, 1): 77.74880538020939, (1, -1): 68.97500000000001, (3, -1): 53.70868895832536}

Optimal policy:
At state 0, take action 1
At state 1, take action 1
At state 2, take action 1
At state 3, take action 1

==================================================

Estimated value for each state: [77.74798557500003, 68.97626510258175, 91.4924122341105, 69.78811791606101, 0]
Estimated value for each state-action pair: {(3, 1): 69.9852926287545, (2, 1): 91.4924122341105, (1, 1): 87.4999393803122, (0, 1): 77.74798557500003, (1, -1): 68.97500000000001, (3, -1): 53.70868895832536}

Optimal policy:
At state 0, take action 1
At state 1, take action 1
At state 2, take action 1
At state 3, take action 1

==================================================
"""