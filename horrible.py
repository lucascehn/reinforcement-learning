import random

# hidden scores of arms
A = [15, 9, 6, 7, 11, 8, 2, 5, 9, 12]

for _ in range(5):
    # our total score - goal is to maximize this
    score = 0
    # we get fixed number of tries to pull arms
    tries = 100000
    Q = [0] * len(A)
    Qa = 0
    Qa_max = 0

    for i in range(len(A)):
        Q[i] = random.randint(0, A[i])
        if Q[i] > Qa_max:
            Qa_max = Q[i]
            Qa = i
        score += Q[i]
        tries -= 1

    # this horrible algorithm just pulls the arm with the highest score so far
    # we only sampled each arm once, so we have no idea which one is actually the best
    while tries > 0:
        score += random.randint(0, A[Qa])
        tries -= 1

    print(Qa)
    print(score)

"""
0
751695
4
552092
0
748352
4
550223
0
753907
"""