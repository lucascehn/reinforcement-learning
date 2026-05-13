import random

# reward for each state
S = [-1, -1, 2, 5, 100]
# left, right
actions = [-1, 1]
# calculated value for each state-action pair
Qs = dict()
Qs_avg = dict()
Qs_count = dict()
# discount rate
gamma = 0.9
# number of episodes to run
episodes = 1000
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
        # total return for the episode
        G = 0
        # first time weight
        ft_w = dict()

        # only do this expensive operation for the first episode
        if i == 0:
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
            for h in reversed(history):
                s, a, r = h
                G = r + gamma * G
                if a != 1:
                    break
                if ft[s] is None:
                    ft[s] = G
                if (s, a) not in ft_q:
                    ft_q[(s, a)] = G
            for (s, a), v in ft_q.items():
                Qs_avg[(s, a)] = v
                Qs_count[(s, a)] = 1
        else:
            while s != len(S) - 1:
                p = random.random()
                a = random.choice(actions) if p < epsilon else get_greedy_action(Qs_avg, s)
                next_s = get_next_state(s, a)
                r = S[next_s]
                history.append((s, a, r))

                max_a = get_greedy_action(Qs_avg, next_s)
                # q estimation
                q_estimate = r + gamma * Qs_avg.get((next_s, max_a), 0)
                n = Qs_count.get((s, a), 0) + 1
                Qs_count[(s, a)] = n
                # running average update
                Qs_avg[(s, a)] = Qs_avg.get((s, a), 0) + (q_estimate - Qs_avg.get((s, a), 0)) / n
                s = next_s

    print("Estimated value for each state-action pair:", Qs_avg)

    print("\nOptimal policy:")
    s = 0
    while s != len(S) - 1:
        a = max(actions, key=lambda a: Qs_avg.get((s, a), float('-inf')))
        print(f"At state {s}, take action {a}")
        s = s + a
    print("\n" + "="*50 + "\n")

"""
Estimated value for each state-action pair: {(3, 1): 67.50011637373545, (2, 1): 67.147993018009, (1, 1): 64.02737304734701, (0, 1): 58.311636730650356, (2, -1): 58.51281061120897, (1, -1): 53.3859772660416, (3, -1): 64.1877701313131, (0, -1): 53.2848937270039}

Optimal policy:
At state 0, take action 1
At state 1, take action 1
At state 2, take action 1
At state 3, take action 1

==================================================

Estimated value for each state-action pair: {(3, 1): 66.54236889127905, (2, 1): 66.15499999335954, (1, 1): 62.86460486129886, (0, 1): 57.11622289994014, (2, -1): 57.56984828981605, (1, -1): 52.736455955783086, (3, -1): 63.497406549777686, (0, -1): 52.806707678736494}

Optimal policy:
At state 0, take action 1
At state 1, take action 1
At state 2, take action 1
At state 3, take action 1

==================================================

Estimated value for each state-action pair: {(3, 1): 64.99102712421359, (2, 1): 63.14060517496011, (1, 1): 60.87044146593216, (0, 1): 55.835401352487494, (2, -1): 56.64843771970951, (1, -1): 51.758358109642565, (3, -1): 61.52269798006801, (0, -1): 51.91079430650375}

Optimal policy:
At state 0, take action 1
At state 1, take action 1
At state 2, take action 1
At state 3, take action 1

==================================================

Estimated value for each state-action pair: {(3, 1): 68.2811511348394, (2, 1): 67.99038062702259, (1, 1): 64.77085756682513, (0, 1): 59.039981032848736, (2, -1): 57.30761485274502, (1, -1): 52.22924863997168, (3, -1): 62.15262137362457, (0, -1): 52.47481468404011}

Optimal policy:
At state 0, take action 1
At state 1, take action 1
At state 2, take action 1
At state 3, take action 1

==================================================

Estimated value for each state-action pair: {(3, 1): 67.56823963821674, (2, 1): 67.08173609241068, (1, 1): 63.97796580435804, (0, 1): 58.376599652738626, (2, -1): 57.5211097027369, (1, -1): 52.40703532537429, (3, -1): 62.37829493415215, (0, -1): 52.623055514423164}

Optimal policy:
At state 0, take action 1
At state 1, take action 1
At state 2, take action 1
At state 3, take action 1

==================================================
"""