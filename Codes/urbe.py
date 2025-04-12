import numpy as np
import matplotlib.pyplot as plt

# States: s0 to s6
S = list(range(7))
A = [0, 1, 2, 3]  # actions a1, a2, a3, a4 mapped to 0-3
R = {1: 0.14, 3: 1.0, 6: 0.0}  # s1, s3, s6

# Transition Map for deterministic action a1 (safe)
def deterministic_transition(s, a):
    if s == 0 and a == 0:  # a1
        return 1
    return None

# Adversarial transition for a2, a3, a4
def adversarial_transition(a, p_good):
    """Returns s3 with prob p_good, else s6"""
    return 3 if np.random.rand() < p_good else 6

# URBE Agent
class URBEAgent:
    def __init__(self, confidence=1.0):
        self.confidence = confidence
        self.q_estimates = np.zeros((7, 4))  # Q-values
        self.counts = np.ones((7, 4))        # Visit counts (avoid div by zero)

    def update(self, s, a, r):
        alpha = 1 / self.counts[s][a]
        self.q_estimates[s][a] += alpha * (r - self.q_estimates[s][a])
        self.counts[s][a] += 1

    def select_action(self, s):
        q = self.q_estimates[s]
        w = 1.0 / np.sqrt(self.counts[s])  # Uncertainty ~ 1/sqrt(n)
        noise = np.random.randn(len(A))   # ζ_b ~ N(0,1)
        exploration_bonus = noise * np.sqrt(w[s])
        return np.argmax(q + exploration_bonus)

# Run MDP
def run_episode(agent, p_s3, max_steps=10):
    s = 0
    total_reward = 0
    for _ in range(max_steps):
        a = agent.select_action(s)

        if a == 0:
            s_next = deterministic_transition(s, a)
        elif a in [1, 2, 3]:
            s_next = adversarial_transition(a, p_s3)
        else:
            s_next = 0

        r = R.get(s_next, 0)
        agent.update(s, a, r)
        total_reward += r
        s = 0  # Always return to s0 after terminal state
    return total_reward

# Simulate
episodes = 100
adversary_schedule = [0.001, 0.8, 0.1, 0.9]
agent = URBEAgent()

rewards = []

for p_s3 in adversary_schedule:
    for _ in range(episodes):
        rew = run_episode(agent, p_s3)
        rewards.append(rew)

# Plotting (optional)

plt.plot(rewards)
plt.title("URBE Reward over Time with Adversarial Transitions")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.grid(True)
plt.show()
