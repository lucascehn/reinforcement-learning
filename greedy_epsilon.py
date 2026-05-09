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
    # hyperparam - probability of exploration
    epsilon = 0.05

    while tries > 0:
        p = random.random()
        if p < epsilon:
            # explore - pull a random arm
            Qa = random.randint(0, len(A) - 1)
        else:
            # exploit - pull the arm with the highest estimated score
            Qa = Q.index(max(Q))
        # we pull the arm and get a score
        Qi = random.randint(0, A[Qa])
        score += Qi
        Qn[Qa] += 1
        Q[Qa] += (Qi - Q[Qa]) / Qn[Qa]
        tries -= 1

    print(Q.index(max(Q)))
    print(score)

"""
0
732097
0
737317
0
732083
0
734358
0
734611
"""