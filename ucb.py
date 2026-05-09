import random
import math

# UCB - Upper Confidence Bound

# hidden scores of arms
A = [15, 9, 6, 7, 11, 8, 2, 5, 9, 12]

# do this 5 times to get an idea of the distribution of results
for _ in range(5):
    # our total score - goal is to maximize this
    score = 0
    # we get fixed number of tries to pull arms
    tries = 0
    # Q - our estimate of the score of each arm
    Q = [0] * len(A)
    # Qn - how many times we pulled each arm
    Qn = [0] * len(A)

    # initialize Q and Qn by pulling each arm once
    for i in range(len(A)):
        Q[i] = random.randint(0, A[i])
        Qn[i] = 1
        score += Q[i]
        tries += 1
    
    while tries < 100000:
        # the arm we will pull - we choose the one with the highest upper confidence bound
        Qa = 0
        # the upper confidence bound of the arm we will pull
        Qa_max = 0
        # hyperparam - need to be chosen so that algo can explore enough
        # and get over typical competing arms
        # VERY IMPORTANT FOR PERFORMANCE
        c = 3

        for i in range(len(A)):
            At = Q[i] + c * math.sqrt(math.log(tries) / Qn[i])
            if At > Qa_max:
                Qa_max = At
                Qa = i

        # we pull the arm and get a score
        Qi = random.randint(0, A[Qa])
        score += Qi
        Qn[Qa] += 1
        # update our estimate of the score of the arm we pulled
        # this is a running average
        Q[Qa] += (Qi - Q[Qa]) / Qn[Qa]
        tries += 1

    print(Q.index(max(Q)))
    print(score)

"""
0
748820
0
751941
0
749226
0
749697
0
752325
"""