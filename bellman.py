# reward matrix - the reward you get for taking an action in state S[i][j]
S = [
    [0, 1, 9, -3],
    [-1, -1, 2, 1],
    [-10, 4, -1, 5],
    [2, -1, 1, 15]
]
# possible actions - down, right, up, left
A = [(1, 0), (0, 1), (-1, 0), (0, -1)]
Amap = {(1, 0): '⬇️', (0, 1): '➡️', (-1, 0): '⬆️', (0, -1): '⬅️'}
# discount rate
gamma = 0.9
# state value function
Vs = [
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0]
]
# q value (state, action pair) function
Qs = dict()
# policy - the probability of an action we will take in each state
Pi = dict()

for _ in range(1000):
    for i in range(len(S)):
        for j in range(len(S[i])):
            # we are in state (i, j)
            # we will try all possible actions and see which one is the best
            Aprime = []
            for a in A:
                # the next state we will reach if we take action a
                i_next = i + a[0]
                j_next = j + a[1]
                # if the next state is out of bounds, we will stay in the same state
                if i_next < 0 or i_next >= len(S) or j_next < 0 or j_next >= len(S[i]):
                    continue
                Aprime.append(a)
                # the reward we get for taking action a in state (i, j)
                r = S[i][j]
                # the value of the next state
                v_next = Vs[i_next][j_next]
                # the q value of the state-action pair (i, j, a)
                q = r + gamma * v_next
                Qs[(i, j, a)] = q
            # the value of the state (i, j) is the maximum q value of all actions
            Vs[i][j] = max(Qs[(i, j, a)] for a in Aprime)
            # the policy is to take the action with the highest q value
            Pi[(i, j)] = max(Aprime, key=lambda a: Qs[(i, j, a)])

Vss = [
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0]
]
for k, v in Pi.items():
    Vss[k[0]][k[1]] = Amap[v]
for r in Vss:
    print(r)

"""
['➡️', '➡️', '⬇️', '⬇️']
['➡️', '⬇️', '➡️', '⬇️']
['➡️', '➡️', '➡️', '⬇️']
['➡️', '➡️', '➡️', '⬆️']
"""