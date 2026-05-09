import random

# hidden scores of arms
A = [15, 9, 6, 7, 11, 8, 2, 5, 9, 12]

for _ in range(5):
    # our total score - goal is to maximize this
    score = 0
    # we get fixed number of tries to pull arms
    tries = 100000
    # Q - our estimate of the score of each arm
    Q = [0] * len(A)
    # Qn - how many times we pulled each arm
    Qn = [0] * len(A)

    while tries > 0:
        # the arm we will pull - we choose the one with the highest estimated score
        Qa = Q.index(max(Q))
        # we pull the arm and get a score
        Qi = random.randint(0, A[Qa])
        score += Qi
        Qn[Qa] += 1
        # update our estimate of the score of the arm we pulled
        # this is a running average
        Q[Qa] += (Qi - Q[Qa]) / Qn[Qa]
        tries -= 1

    print(Q.index(max(Q)))
    print(score)

"""
0
748345
0
749036
0
749329
0
751066
0
750310
"""