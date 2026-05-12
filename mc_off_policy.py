import random

# reward for each state
S = [-1, -1, 2, 5, 100]
# left, right
actions = [-1, 1]
# calculated value for each state
Vs = [[] for _ in range(len(S))]
# calculated value for each state-action pair
Qs = dict()
Qs_avg = dict()
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
    b = epsilon / len(actions) if a == greedy_a else 1 - epsilon + epsilon / len(actions)
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
                a = random.choice(actions) if p < (epsilon / len(actions)) else get_greedy_action(Qs, s)
            next_s = get_next_state(s, a)
            history.append((s, a, S[next_s]))
            s = next_s

        G = 0
        for h in reversed(history):
            s, a, r = h
            # calculate return G for the first time we visit state s
            G = r + gamma * G
            # this is the first time we visit state s
            ft[s] = G
            ft_q[(s, a)] = G

        for s in range(len(S)):
            if ft[s] is not None:
                Vs[s].append(ft[s])
        for (s, a), v in ft_q.items():
            Qs[(s, a)] = Qs.get((s, a), []) + [v]

    Vs_avg = [sum(vs) / len(vs) if vs else 0 for vs in Vs]
    # this can be normalized with sum of importance sampling weights if we want to get the actual expected value
    # however i am lazy
    Qs_avg = {
        (s, a): (sum(vs) / len(vs)) * get_importance_sampling_weight(a, s, Qs_avg) for (s, a), vs in Qs.items()
    }

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
Estimated value for each state: [24.885022833181814, 28.761136481313127, 30.067929423700008, 29.558582758290964, 0]
Estimated value for each state-action pair: {(3, 1): 38.75171462944419, (2, 1): 32.41305075124295, (1, 1): 29.806293827005142, (0, 1): 25.52310034172494, (1, -1): -0.0, (2, -1): 0.0, (3, -1): 0.0}

Optimal policy:
At state 0, take action 1
At state 1, take action 1
At state 2, take action 1
At state 3, take action 1

==================================================

Estimated value for each state: [27.184932934158923, 31.316592149065475, 32.74065794341546, 31.675617191995986, 0]
Estimated value for each state-action pair: {(3, 1): 1710.2931834121691, (2, 1): 1340.3222189118555, (1, 1): 1258.6636859629566, (0, 1): 1087.397317366357, (1, -1): -0.0, (2, -1): 0.0, (3, -1): 0.0}

Optimal policy:
At state 0, take action 1
At state 1, take action 1
At state 2, take action 1
At state 3, take action 1

==================================================

Estimated value for each state: [28.02335117086263, 32.24816796762514, 33.72018663070086, 32.47976109291275, 0]
Estimated value for each state-action pair: {(3, 1): 1678.6755942090488, (2, 1): 1369.271399344859, (1, 1): 1293.926718705231, (0, 1): 1120.9340468345051, (1, -1): 0.0, (2, -1): 0.0, (3, -1): 0.0}

Optimal policy:
At state 0, take action 1
At state 1, take action 1
At state 2, take action 1
At state 3, take action 1

==================================================

Estimated value for each state: [28.208097973517244, 32.453442192796935, 34.15382126767127, 32.819466702624155, 0]
Estimated value for each state-action pair: {(3, 1): 1688.1106134049592, (2, 1): 1381.5008012944695, (1, 1): 1309.5375656361657, (0, 1): 1128.3239189406897, (1, -1): 0.0, (2, -1): 0.0, (3, -1): 0.0}

Optimal policy:
At state 0, take action 1
At state 1, take action 1
At state 2, take action 1
At state 3, take action 1

==================================================

Estimated value for each state: [29.470953799276487, 33.85661533252943, 35.64956987890577, 34.39614321184238, 0]
Estimated value for each state-action pair: {(3, 1): 1800.6906184192376, (2, 1): 1438.261155626326, (1, 1): 1363.3845156406078, (0, 1): 1178.8381519710595, (1, -1): 0.0, (2, -1): 0.0, (3, -1): 0.0}

Optimal policy:
At state 0, take action 1
At state 1, take action 1
At state 2, take action 1
At state 3, take action 1

==================================================
"""