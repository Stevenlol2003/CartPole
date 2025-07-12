import gymnasium as gym
import numpy as np
from collections import defaultdict
import pickle
import os

bins_per_observation = [8, 8, 20, 20]  # [Cart Position, Velocity, Pole Angle, Pole Angular Velocity]

# Observation bounds
obs_space_low  = np.array([-4.8, -5.0, -0.418, -10])
obs_space_high = np.array([4.8, 5.0, 0.418, 10])

bins = [
    np.linspace(obs_space_low[i], obs_space_high[i], bins_per_observation[i] + 1)[1:-1].tolist()
    for i in range(len(bins_per_observation))
]

def observation_to_state(obs):
    state = []
    for i in range(len(obs)):
        val = np.clip(obs[i], obs_space_low[i], obs_space_high[i])
        state.append(np.digitize(val, bins[i]))
    return tuple(state)

def default_q():
    return np.zeros(2)  # for the 2 actions

q_table = defaultdict(default_q)

# choose q-table
selected_file = "q_table_ep5000_lr0.2_noise0.0.pkl"

if not os.path.exists(selected_file):
    raise FileNotFoundError(f"Q-table file '{selected_file}' not found.")

with open(selected_file, 'rb') as file:
    q_table.update(pickle.load(file))

print(f"\nLoaded Q-table: {selected_file}")

# run using chosen q-table
# env = gym.make('CartPole-v1', render_mode="human")
env = gym.make('CartPole-v1', render_mode="rgb_array")
observation, info = env.reset()
state = observation_to_state(observation)
done = False
total_reward = 0

while not done:
    action = np.argmax(q_table[state])
    observation, reward, terminated, truncated, info = env.step(action)
    next_state = observation_to_state(observation)
    total_reward += reward
    state = next_state
    done = terminated or truncated

env.close()

print(f"\nTotal reward using Q-table policy: {total_reward}")