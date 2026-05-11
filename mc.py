import random

# reward for each state
S = [-1, -1, 2, 5, 100]
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
        ft = [None] * len(S)
        # first time monte carlo value estimation for q value for state s and action a
        ft_q = dict()
        # history
        history = []

        while s != len(S) - 1:
            a = random.choice(actions)
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

"""
Estimated value for each state: [8.399033557684474, 12.042321466751908, 17.78549147841097, 26.342888262565392, 0]
Estimated value for each state-action pair: {(3, 1): 34.85078706750553, (2, 1): 28.70859943630885, (1, 1): 18.006942330569874, (0, 1): 9.838089320076715, (0, -1): 5.7079548004335425, (1, -1): 6.8353243622176505, (2, -1): 8.926612198289709, (3, -1): 20.3827214170475}

Optimal policy:
At state 0, take action 1
At state 1, take action 1
At state 2, take action 1
At state 3, take action 1

==================================================

Estimated value for each state: [10.050505649355568, 14.059787497623702, 19.99927977829539, 27.365398344849705, 0]
Estimated value for each state-action pair: {(3, 1): 37.348412822780546, (2, 1): 29.628858510364736, (1, 1): 19.999351800465853, (0, 1): 11.65380874786133, (0, -1): 6.419792749395106, (1, -1): 6.843482356342551, (2, -1): 10.91169825513574, (3, -1): 21.625560176957055}

Optimal policy:
At state 0, take action 1
At state 1, take action 1
At state 2, take action 1
At state 3, take action 1

==================================================

Estimated value for each state: [10.115124956480754, 14.233680948617069, 20.614322740979542, 29.002354887337916, 0]
Estimated value for each state-action pair: {(3, 1): 38.79133981565156, (2, 1): 31.102119398604124, (1, 1): 20.55289046688159, (0, 1): 11.810312853755363, (0, -1): 6.770419555010871, (1, -1): 7.611321019786521, (2, -1): 11.47898261376396, (3, -1): 23.694352682210233}

Optimal policy:
At state 0, take action 1
At state 1, take action 1
At state 2, take action 1
At state 3, take action 1

==================================================

Estimated value for each state: [9.360377986957914, 13.272832049323462, 19.702508438755086, 28.165015698064035, 0]
Estimated value for each state-action pair: {(3, 1): 37.59865000664333, (2, 1): 30.34851412825763, (1, 1): 19.732257594879577, (0, 1): 10.945548844391118, (0, -1): 6.445941342464573, (1, -1): 7.284054445206835, (2, -1): 11.707916177180797, (3, -1): 22.598235928001095}

Optimal policy:
At state 0, take action 1
At state 1, take action 1
At state 2, take action 1
At state 3, take action 1

==================================================

Estimated value for each state: [9.671820135720594, 13.692740531515636, 20.169630303818558, 29.283058352176152, 0]
Estimated value for each state-action pair: {(3, 1): 38.120015911295816, (2, 1): 31.354752516958534, (1, 1): 20.152667273436702, (0, 1): 11.323466478364072, (0, -1): 6.845267329657493, (1, -1): 7.087549083240934, (2, -1): 11.746539651808533, (3, -1): 21.550772975314324}

Optimal policy:
At state 0, take action 1
At state 1, take action 1
At state 2, take action 1
At state 3, take action 1

==================================================
"""