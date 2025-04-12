# comparison of DQN-URBE, Vanilla DQN, UBE, and robust DQN algorithms

# mdp model with changing adversary
import numpy as np
import matplotlib.pyplot as plt

class MDP(s,a,P,r):
    def next_state(s,a,P):
        #sample state from p(s'|s,a) given (s,a) from sth row and ath column of p
        c_state = s
        c_action = a
        n_state = np.random.choice(P[c_state][c_action])
        reward = r(n_state, c_state, c_action)
        return(n_state, reward)



    def action(policy, state):
        #given policy for a state s, sample action from policy
        action = np.random.choice(policy[state])
        return(action)

            








states = [0,1,2,3,4,5]
actions = [0,1,2]
P1 = np.array([
    [[0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],

    

])   



















class urbe(prior, confidence, (state, action)):
    for t in range(1, 100):
        #sample mdp from prior
        sprime, reward = MDP.get_state(state, action, p)
        #update posterior
        robust.value(phat)
        #compute a'
        #take action a'
        (state, action) = (sprime, action)
        


class ube():


class robust(setp, (state, action)):
    #robust policy

