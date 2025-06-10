import gymnasium as gym
import numpy as np
from collections import defaultdict
import pickle

bins = [
    [-2.4, 0.0, 2.4],
    [-4.0, -2.0, 0.0, 2.0, 4.0],
    [-0.2095, 0.0, 0.2095],
    [-4.0, -2.0, 0.0, 2.0, 4.0]
]

obs_space_low  = np.array([-4.8, -np.inf, -0.418, -np.inf])
obs_space_high = np.array([ 4.8,  np.inf,  0.418,  np.inf])

def observation_to_state(obs):
    state = []
    for i in range(len(obs)):
        val = np.clip(obs[i], obs_space_low[i], obs_space_high[i])
        state.append(np.digitize(val, bins[i]))
    return tuple(state)

# Load Q-table
def default_q():
    return np.zeros(2)  # 2 actions for CartPole

q_table = defaultdict(default_q)

# Load the Q-table
with open('q_table.pkl', 'rb') as file:
    q_table.update(pickle.load(file))

env = gym.make("CartPole-v1", render_mode="human")
observation, info = env.reset()
state = observation_to_state(observation)
done = False
total_reward = 0

while not done:
    action = np.argmax(q_table[state])
    observation, reward, terminated, truncated, _ = env.step(action)
    next_state = observation_to_state(observation)
    total_reward += reward
    state = next_state
    done = terminated or truncated

env.close()

print(f"Total reward using Q-table policy: {total_reward}")