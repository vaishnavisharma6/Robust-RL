import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


class AdversarialMDP:
    def __init__(self, p_s3=0.5):
        """
        Initialize the 7-state MDP with adversarial transitions.

        Args:
            p_s3 (float): Probability that the adversary allows transition to good state s3.
        """
        self.n_states = 7  # s0 to s6
        self.n_actions = 4  # a1 to a4
        self.state = 0  # Start from s0
        self.p_s3 = p_s3

    def reset(self):
        """Resets the environment to the initial state s0."""
        self.state = 0
        return self.state

    def step(self, action):
        """
        Execute one step in the environment.

        Args:
            action (int): Action index from 0 to 3 corresponding to a1 to a4.

        Returns:
            next_state (int): Next state.
            reward (float): Reward received.
            done (bool): Whether the episode ends (i.e., agent is sent back to s0).
        """
        s = self.state
        done = False
        reward = 0

        if s != 0:
            raise ValueError("Agent must be at s0 at start of step.")

        if action == 0:  # a1: go to s1
            next_state = 1
            reward = 0.14
            done = True

        elif action == 1:  # a2: go to s2 -> s3 or s6
            intermediate = 2
            next_state = 3 if np.random.rand() < self.p_s3 else 6
            reward = 1 if next_state == 3 else 0
            done = True

        elif action == 2:  # a3: go to s4 -> s3 or s6
            intermediate = 4
            next_state = 3 if np.random.rand() < self.p_s3 else 6
            reward = 1 if next_state == 3 else 0
            done = True

        elif action == 3:  # a4: go to s5 -> s3 or s6
            intermediate = 5
            next_state = 3 if np.random.rand() < self.p_s3 else 6
            reward = 1 if next_state == 3 else 0
            done = True

        else:
            raise ValueError("Invalid action index")

        self.state = 0 if done else next_state  # reset to s0 if terminal
        return next_state, reward, done

    def sample_action(self):
        """Random action from 0 to 3."""
        return np.random.choice(4)

    def set_adversarial_prob(self, new_p):
        """Dynamically update adversarial transition probability."""
        self.p_s3 = new_p


#-----------------------------------------------------------------------------------------------------------

class DeceptiveAdversarialMDP:
    def __init__(self, deceptive_episodes=50, p_good=0.5):
        """
        Initialize the deceptive adversarial MDP.

        Args:
            deceptive_episodes (int): Number of episodes where a0 leads to a high reward before switching.
            p_good (float): Probability of going to good state (s7) via a1/a2.
        """
        self.n_states = 8  # s0 to s7
        self.n_actions = 4  # a0 to a3
        self.state = 0  # Start at s0
        self.episode_count = 0
        self.deceptive_episodes = deceptive_episodes
        self.p_good = p_good  # p(s7) for a1/a2 paths

    def reset(self):
        """Reset the environment to initial state."""
        self.state = 0
        return self.state

    def step(self, action):
        """
        Perform a step in the environment.

        Args:
            action (int): Action index from 0 to 3.

        Returns:
            next_state (int): The next state after transition.
            reward (float): Reward for the transition.
            done (bool): Whether the episode has ended.
        """
        s = self.state
        done = False
        reward = 0

        if s != 0:
            raise ValueError("Agent must always start from s0.")

        # Action a0: deceptive high-reward path (changes behavior after a few episodes)
        if action == 0:
            next_state = 1
            if self.episode_count < self.deceptive_episodes:
                final_state = 7  # good
            else:
                final_state = 6  # trap
            reward = 1 if final_state == 7 else 0
            done = True
            self.state = 0
            self.episode_count += 1
            return final_state, reward, done

        # Action a1: goes to s2 then s3/s4 → s7 (good) or s6 (bad)
        elif action == 1:
            intermediate = np.random.choice([2, 3])
            final_state = 7 if np.random.rand() < self.p_good else 6
            reward = 1 if final_state == 7 else 0
            done = True

        # Action a2: goes to s4 then s5/s6 → s7 (good) or s6 (bad)
        elif action == 2:
            intermediate = np.random.choice([4, 5])
            final_state = 7 if np.random.rand() < self.p_good else 6
            reward = 1 if final_state == 7 else 0
            done = True

        # Action a3: direct to trap
        elif action == 3:
            final_state = 6
            reward = 0
            done = True

        else:
            raise ValueError("Invalid action index")

        self.state = 0
        self.episode_count += 1
        return final_state, reward, done

    def sample_action(self):
        """Random action from 0 to 3."""
        return np.random.choice(4)

    def set_p_good(self, new_p):
        """Dynamically change the probability of reaching good state from a1/a2 paths."""
        self.p_good = new_p


#----------------------------------------------------------------------------------------------------------

class URBEAgent:
    def __init__(self, n_states, n_actions):
        self.n_states = n_states
        self.n_actions = n_actions
        self.q_estimates = np.zeros((n_states, n_actions))  # Q̂ values
        self.counts = np.ones((n_states, n_actions))         # Visit counts (start from 1 for stability)

    def select_action(self, state):
        q = self.q_estimates[state]
        w = 1.0 / np.sqrt(self.counts[state])  # Uncertainty term w = 1/sqrt(n)
        noise = np.random.randn(self.n_actions)  # ζ ∼ N(0,1)
        return np.argmax(q + noise * np.sqrt(w))  # URBE exploration

    def update(self, state, action, reward):
        alpha = 1.0 / self.counts[state][action]
        self.q_estimates[state][action] += alpha * (reward - self.q_estimates[state][action])
        self.counts[state][action] += 1

#-----------------------------------------------------------------------------------------------
def run_urbe(env, agent, episodes, verbose=False):
    rewards = []
    cr = 0 
    for ep in range(episodes):
        state = env.reset()
        action = agent.select_action(state)
        next_state, reward, done = env.step(action)
        cr = cr + reward
        agent.update(state, action, reward)
        if ep%10 == 0:

            rewards.append(cr/10)
        
        if verbose:
            print(f"Episode {ep}: s={state}, a={action}, s'={next_state}, r={reward:.2f}")

    return rewards
#-----------------------------------------------------------------------------------------------

class RobustQLearningAgent:
    def __init__(self, env, gamma=0.95, delta=0.1):
        self.env = env
        self.gamma = gamma
        self.Q = np.zeros((env.n_states, env.n_actions))
        self.transition_counts = defaultdict(lambda: np.zeros(env.n_states))
        self.rewards = np.zeros((env.n_states, env.n_actions))
        self.visits = np.zeros((env.n_states, env.n_actions))
        self.delta = delta  # Uncertainty level

    def update(self, state, action, next_state, reward):
        self.transition_counts[(state, action)][next_state] += 1
        self.rewards[state][action] = reward
        self.visits[state][action] += 1

    def get_empirical_P(self, state, action):
        counts = self.transition_counts[(state, action)]
        total = counts.sum()
        if total == 0:
            return np.ones(self.env.n_states) / self.env.n_states
        return counts / total

    def robust_backup(self, V, P_hat, delta):
        # Approximate worst-case expected value under L1-ball
        sorted_indices = np.argsort(V)  # ascending
        worst_value = 0.0
        remaining_mass = 1.0
        for idx in sorted_indices:
            mass = min(P_hat[idx], remaining_mass)
            worst_value += mass * V[idx]
            remaining_mass -= mass
            if remaining_mass <= 0:
                break
        return worst_value - delta * np.max(np.abs(V - V.mean()))

    def robust_q_iteration(self, n_iters=10):
        for _ in range(n_iters):
            new_Q = np.copy(self.Q)
            for s in range(self.env.n_states):
                for a in range(self.env.n_actions):
                    P_hat = self.get_empirical_P(s, a)
                    V = np.max(self.Q, axis=1)
                    robust_exp = self.robust_backup(V, P_hat, self.delta)
                    new_Q[s][a] = self.rewards[s][a] + self.gamma * robust_exp
            self.Q = new_Q

    def select_action(self, state):
        return np.argmax(self.Q[state])

    def train(self, episodes=500):
        for ep in range(episodes):
            state = self.env.reset()
            action = self.select_action(state)
            next_state, reward, done = self.env.step(action)
            self.update(state, action, next_state, reward)
            self.robust_q_iteration()
#------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------

class UncertaintyAwareQLearningAgent:
    def __init__(self, env, n_states, n_actions, gamma=0.95, alpha=0.1, beta=1.0):
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma      # Discount factor
        self.alpha = alpha      # Learning rate
        self.beta = beta        # Controls exploration bonus
        self.Q = np.zeros((env.n_states, env.n_actions))
        self.visits = np.ones((env.n_states, env.n_actions))  # For uncertainty estimate

    def select_action(self, state):
        # Use visitation count to create uncertainty-aware score
        Q = self.Q[state]
        bonus = self.beta / np.sqrt(self.visits[state])
        noise = np.random.randn(self.n_actions)
        score = self.Q[state] + (noise * np.sqrt(bonus))
        return np.argmax(score)

    def update(self, state, action, reward, next_state):
        best_next = np.max(self.Q[next_state])
        td_target = reward + self.gamma * best_next
        td_error = td_target - self.Q[state, action]
        self.Q[state, action] += self.alpha * td_error
        self.visits[state, action] += 1

    def train(self, episodes=500):
        for ep in range(episodes):
            state = self.env.reset()
            action = self.select_action(state)
            next_state, reward, done = self.env.step(action)
            self.update(state, action, reward, next_state)
#----------------------------------------------------------------------------------------------------------------------            

env = AdversarialMDP(p_s3=0.001)
agent = URBEAgent(n_states=7, n_actions=4)
agent2 = RobustQLearningAgent(env, gamma = 0.95, delta = 0.1)
agent3 = UncertaintyAwareQLearningAgent(env, n_states = 7, n_actions = 4, beta=1.0)
rewards = run_urbe(env, agent, episodes=200)
print(rewards)
state = env.reset()
cr = 0
C = []
for i in range(200):
    action = env.sample_action()
    next_state, reward, done = env.step(action)
    cr = cr + reward
    if i%10 == 0:
        C.append(cr/10)
       
    print(f"Step {i}: Action a{action+1}, Next State s{next_state}, Reward: {reward}")
    if done:
        state = env.reset()


rewards = []
cu = 0 
for ep in range(200):
    state = env.reset()
    action = agent.select_action(state)
    next_state, reward, done = env.step(action)
    cu = cu + reward
    agent.update(state, action, reward)
    if ep%10 == 0:
        rewards.append(cu/10)
    

cro = 0
rrobust = []
for ep in range(200):
    state = env.reset()
    action = agent2.select_action(state)
    next_state, reward, done = agent2.env.step(action)
    cro = cro + reward
    agent2.update(state, action, next_state, reward)
    if ep%10 == 0:
        rrobust.append(cro/10)
    agent2.robust_q_iteration()    

ube = 0
uber = []
for ep in range(200):
    state = env.reset
    print(state)
    action = agent3.select_action(state)
    next_state, reward, done = agent3.env.step(action)
    ube = ube + reward
    agent3.update(state, action, next_state, reward)
    if ep%10 == 0:
        uber.append(ube/10)
        


#---------------------------------------------------------------------------------------------------
env = AdversarialMDP(p_s3 = 0.8)
state = env.reset()
agent2 = RobustQLearningAgent(env, gamma = 0.95, delta = 0.1)
agent3 = UncertaintyAwareQLearningAgent(env, beta=1.0)

for i in range(500):
    action = env.sample_action()
    next_state, reward, done = env.step(action)
    cr = cr + reward
    if i%10 == 0:
        C.append(cr/10)
       
    print(f"Step {i}: Action a{action+1}, Next State s{next_state}, Reward: {reward}")
    if done:
        state = env.reset()


#cumulative rewards for URBE

for ep in range(500):
    state = env.reset()
    action = agent.select_action(state)
    next_state, reward, done = env.step(action)
    cu = cu + reward
    agent.update(state, action, reward)
    if ep%10 == 0:
        rewards.append(cu/10)

#robust
for ep in range(500):
    state = env.reset()
    action = agent2.select_action(state)
    next_state, reward, done = agent2.env.step(action)
    cro = cro + reward
    agent2.update(state, action, next_state, reward)
    if ep%10 == 0:
        rrobust.append(cro/10)
    agent2.robust_q_iteration()       

#ube
ube = 0
uber = []
for ep in range(500):
    state = env.reset
    action = agent3.select_action(state)
    next_state, reward, done = agent3.env.step(action)
    ube = ube + reward
    agent3.update(state, action, next_state, reward)
    if ep%10 == 0:
        uber.append(ube/10)
     

#---------------------------------------------------------------------------------------------------

env = AdversarialMDP(p_s3 = 0.1)
state = env.reset()
agent2 = RobustQLearningAgent(env, gamma = 0.95, delta = 0.1)
agent3 = UncertaintyAwareQLearningAgent(env, beta=1.0)

for i in range(300):
    action = env.sample_action()
    next_state, reward, done = env.step(action)
    cr = cr + reward
    if i%10 == 0:
        C.append(cr/10)
       
    print(f"Step {i}: Action a{action+1}, Next State s{next_state}, Reward: {reward}")
    if done:
        state = env.reset()


for ep in range(300):
    state = env.reset()
    action = agent.select_action(state)
    next_state, reward, done = env.step(action)
    cu = cu + reward
    agent.update(state, action, reward)
    if ep%10 == 0:
        rewards.append(cu/10)

#robust
for ep in range(300):
    state = env.reset()
    action = agent2.select_action(state)
    next_state, reward, done = agent2.env.step(action)
    cro = cro + reward
    agent2.update(state, action, next_state, reward)
    if ep%10 == 0:
        rrobust.append(cro/10)
    agent2.robust_q_iteration()  

#ube
ube = 0
uber = []
for ep in range(300):
    state = env.reset
    action = agent3.select_action(state)
    next_state, reward, done = agent3.env.step(action)
    ube = ube + reward
    agent3.update(state, action, next_state, reward)
    if ep%10 == 0:
        uber.append(ube/10)

#----------------------------------------------------------------------------------------------------
env = AdversarialMDP(p_s3 = 0.9)
state = env.reset()
agent2 = RobustQLearningAgent(env, gamma = 0.95, delta = 0.1)
agent3 = UncertaintyAwareQLearningAgent(env, beta=1.0)

for i in range(500):
    action = env.sample_action()
    next_state, reward, done = env.step(action)
    cr = cr + reward
    if i%10 == 0:
        C.append(cr/10)
       
    print(f"Step {i}: Action a{action+1}, Next State s{next_state}, Reward: {reward}")
    if done:
        state = env.reset()



for ep in range(500):
    state = env.reset()
    action = agent.select_action(state)
    next_state, reward, done = env.step(action)
    cu = cu + reward
    agent.update(state, action, reward)
    if ep%10 == 0:
        rewards.append(cu/10)        


for ep in range(500):
    state = env.reset()
    action = agent2.select_action(state)
    next_state, reward, done = agent2.env.step(action)
    cro = cro + reward
    agent2.update(state, action, next_state, reward)
    if ep%10 == 0:
        rrobust.append(cro/10)
    agent2.robust_q_iteration()  

#ube
ube = 0
uber = []
for ep in range(500):
    state = env.reset
    action = agent3.select_action(state)
    next_state, reward, done = agent3.env.step(action)
    ube = ube + reward
    agent3.update(state, action, next_state, reward)
    if ep%10 == 0:
        uber.append(ube/10)    

#----------------------------------------------------------------------------------------------------
x = np.arange(0, len(C))
x1 = np.arange(0, len(rewards))
x2 = np.arange(0, len(rrobust))
x3 = np.arange(0, len(uber))
plt.figure(figsize=(10,6))
plt.plot(x, C, label = 'cumulative rewards with no external policy')
plt.plot(x1, rewards, label = 'URBE')
plt.plot(x2, rrobust, label = 'Robust')
plt.plot(x3, uber, label = 'UBE')
for x in [20, 70, 100, 150]:
    plt.axvline(x=x, color='gray', linestyle='--', linewidth=1)
plt.xlabel('episode*10')
plt.ylabel('cumulative reward')
plt.legend()
plt.savefig('no_policy.png')


