import gymnasium as gym
import numpy as np
from collections import defaultdict
import pickle

env = gym.make('CartPole-v1', render_mode="human")

# goal is to keep pole upright for as long as possible
# a reward of +1 is given for every step taken, including the termination step

# trying out q-learning
# intialize q-table
# choose an action
# perform action
# measure reward
# update q-table
# repeat utill termination state

# Obervations: a ndarray with shape (4,)
# Num   Obervation              Min                     Max
# 0     Cart Position           -4.8                    4.8
# 1     Cart Velocity           -Inf                    Inf
# 2     Pole Angle              -0.418 rad (-24°)       0.418 rad (24°)
# 3     Pole Angular Velocity   -Inf                    Inf

# Termination
# The episode ends if any one of the following occurs:
# 1. Pole Angle is greater than ±12°, terminates if outside [-0.2095, 0.2095]
# 2. Cart Position is greater than ±2.4 (center of the cart reaches the edge of the display)
# 3. Episode length is greater than 500 (200 for v0)       500 for my case

# observation space limits: env.observation_space
# upper limit: env.observation_space.high
# lower limit: env.observation_space.low
# action space: env.action_space        this should be 0 and 1
# all the specs: env.spec
# maximum number of steps per episode: env.spec.max_episode_steps
# reward threshold per episode: env.spec.reward_threshold

# we have have a certain number of ranges for each observartion and put each value into a bin
# [Cart Position, Cart Velocity, Pole Angle , Pole Angular Velocity]
# 4 Cart Position bins: [-4.8, -2.4], [-2.4, 0.0], [0.0, 2.4], [2.4, 4.8]
# 6 Cart Velocity: [-Inf, -4.0], [-4.0, -2.0], [-2.0, 0.0], [0.0, 2.0], [2.0, 4.0], [4.0, Inf], maybe consider 4 bins
# 4 Pole Angle bins: [-0.418, -0.2095], [-0.2095, 0.0], [0.0, 0.2095], [-0.2095, 0.41]
# 6 Pole Angular Velocity bins: [-Inf, -4.0], [-4.0, -2.0], [-2.0, 0.0], [0.0, 2.0], [2.0, 4.0], [4.0, Inf], maybe consider 4 bins and adjust velocity to be smaller
# limit cart velocity and pole angular velocity to +-4 for now to make bins

bins = [
    [-2.4, 0.0, 2.4],               # Cart Position
    [-4.0, -2.0, 0.0, 2.0, 4.0],    # Cart Velocity
    [-0.2095, 0.0, 0.2095],         # Pole Angle
    [-4.0, -2.0, 0.0, 2.0, 4.0]     # Pole Angular Velocity
]

# Observation bounds
obs_space_low  = np.array([-4.8, -np.inf, -0.418, -np.inf])
obs_space_high = np.array([ 4.8,  np.inf,  0.418,  np.inf])

def observation_to_state(obs):
    state = []
    for i in range(len(obs)):
        val = np.clip(obs[i], obs_space_low[i], obs_space_high[i])
        state.append(np.digitize(val, bins[i]))
    return tuple(state)

def default_q():
    return np.zeros(env.action_space.n)

q_table = defaultdict(default_q)

# Hyperparameters
learning_rate = 0.1     # a
discount_factor = 0.99  # y
exploration_prob = 0.8  # e
episodes = 1000

for i in range(episodes):
    print(f'Iteration:', i)
    observation, info = env.reset(seed=42)
    state = observation_to_state(observation)
    print(f'State: ',state)
    done = False
    total_reward = 0

    while not done:
        # actions: 0 = move left, 1 = move right
        # sample() uses random actions
        if np.random.rand() < exploration_prob:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])

        # original example is random action, commented out
        # action = env.action_space.sample()
        next_observation, reward, terminated, truncated, info = env.step(action)
        # print(f'Observation: ', next_observation)
        # print(f'Reward: ', reward)
        # print(f'Terminated: ', terminated)
        # print(f'Truncated: ', truncated)
        # print(f'Info: ', info)
        next_state = observation_to_state(next_observation)
        print(f'Next State: ', next_state)

        total_reward += reward

        if terminated or truncated:
            done = True
        else:
            done = False

        # formula used from https://www.geeksforgeeks.org/q-learning-in-python/
        best_next_state = np.max(q_table[next_state])
        q_table[state][action] += learning_rate * (reward + discount_factor * best_next_state - q_table[state][action])

    print(f"Episode {i+1}: Total Reward = {total_reward}")

env.close()


# save q-table as a pickle file
with open('q_table.pkl', 'wb') as file:
    pickle.dump(q_table, file)

print("\nQ-table saved as q_table.pkl")

print("\nDisplay Q-table:")
for state, actions in q_table.items():
    if np.any(actions != 0):
        print(f"State {state}: {actions}")