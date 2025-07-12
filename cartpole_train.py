import gymnasium as gym
import numpy as np
from collections import defaultdict
import pickle
import os
import matplotlib.pyplot as plt
import csv

# stat logging helper function, saves different reward and noise level combinations as csv
def save_episode_stats(rewards, mean_angles, var_angles, reward_type, noise_level):
    filename = f"episode_stats_{reward_type}_noise{noise_level}.csv"
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Episode", "TotalReward", "MeanPoleAngle", "VariancePoleAngle"])
        for i in range(len(rewards)):
            writer.writerow([i + 1, rewards[i], mean_angles[i], var_angles[i]])

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
# [Cart Position, Cart Velocity, Pole Angle, Pole Angular Velocity]
# 4 Cart Position bins: [-4.8, -2.4], [-2.4, 0.0], [0.0, 2.4], [2.4, 4.8]
# 6 Cart Velocity: [-Inf, -4.0], [-4.0, -2.0], [-2.0, 0.0], [0.0, 2.0], [2.0, 4.0], [4.0, Inf], maybe consider 4 bins
# 4 Pole Angle bins: [-0.418, -0.2095], [-0.2095, 0.0], [0.0, 0.2095], [-0.2095, 0.41]
# 6 Pole Angular Velocity bins: [-Inf, -4.0], [-4.0, -2.0], [-2.0, 0.0], [0.0, 2.0], [2.0, 4.0], [4.0, Inf], maybe consider 4 bins and adjust velocity to be smaller
# limit cart velocity and pole angular velocity to +-4 for now to make bins

# Noise not used here
# obs_noise_std = 0.0      # standard deviation for observation noise
# action_noise_prob = 0.0  # probability of action to be random instead of the chosen max reward one

# default: returns a constant reward of 1 per step
# cosine: uses cos(pole angle) to reward keeping pole closer to upright (cos(0) = 1)
reward_functions = {
    "default": lambda obs: 1.0,
    "cosine": lambda obs: np.cos(obs[2])
}

# different levels of noise to add to observations
noise_levels = [0.0, 0.01, 0.05, 0.1]

# hyperparameters
episodes = 10000
learning_rate = 0.1 # alpha
discount_factor = 0.999 # gamma

# used to track first episode of convergence: 500 total rewards reached for first time
convergence_episodes = {}

# iterate over each reward
for reward_type, reward_fn in reward_functions.items():
    mean_stats_by_noise = {}
    var_stats_by_noise = {}

    # iterate over each noise level
    for noise in noise_levels:
        # number of bins for each oberservation
        # [Cart Position, Cart Velocity, Pole Angle, Pole Angular Velocity]
        bins_per_observation = [8, 8, 20, 20]

        # min to max boundries for bins
        obs_space_low = np.array([-4.8, -5.0, -0.418, -10])
        obs_space_high = np.array([4.8, 5.0, 0.418, 10])

        # creates the bins based off above's bin count
        bins = [
            np.linspace(obs_space_low[i], obs_space_high[i], bins_per_observation[i] + 1)[1:-1].tolist()
            for i in range(len(bins_per_observation))
        ]

        # observation to state, adds noise here to
        def observation_to_state(obs):
            ranges = obs_space_high - obs_space_low
            noise_std = noise * ranges
            noisy_obs = obs + np.random.normal(0, noise_std, size=obs.shape)
            # assign oberservation to bin and store combination as state
            state = []
            for i in range(len(obs)):
                val = np.clip(noisy_obs[i], obs_space_low[i], obs_space_high[i])
                state.append(np.digitize(val, bins[i]))
            return tuple(state)

        # intialize default q-table, each state maps to 2 actions left/right 
        def default_q():
            return np.zeros(2)

        q_table = defaultdict(default_q)

        initial_exploration_prob = 1.0
        min_exploration_prob = 0.001
        exploration_decay_rate = 0.999
        exploration_prob = initial_exploration_prob

        print(f"\nTraining with reward: {reward_type}, noise: {noise}")

        all_rewards = []    # records final reward of each episode
        mean_angles = []    # mean pole angle per episode
        var_angles = []     # variance of pole angle per episode
        converged_at = None # tracks when we reach max reward 500 (499+ here for cosine reward rounding)

        # intialize CartPole environment
        # env = gym.make('CartPole-v1', render_mode="human")
        env = gym.make('CartPole-v1', render_mode="rgb_array")

        # training loop
        for i in range(episodes):
            observation, info = env.reset(seed=42)
            state = observation_to_state(observation)
            done = False
            total_reward = 0
            episode_angles = []

            while not done:
                # epsilon-green action selection
                if np.random.rand() < exploration_prob:
                    action = env.action_space.sample()
                else:
                    action = np.argmax(q_table[state])

                # 2nd variable was original reward, use custom function instead
                next_observation, _, terminated, truncated, info = env.step(action)

                # potentially use custom reward function based off observation of the pole angle, next_observation[2], such as cos(next_observation[2]) since cos(0) = 1
                reward = reward_fn(next_observation)

                episode_angles.append(next_observation[2]) # track pole angle
                next_state = observation_to_state(next_observation)

                total_reward += reward
                done = terminated or truncated

                # Q-learning update
                best_next = np.max(q_table[next_state])
                q_table[state][action] += learning_rate * (reward + discount_factor * best_next - q_table[state][action])
                state = next_state

            # if function for convergence counting
            if converged_at is None and total_reward >= 499:
                converged_at = i + 1

            # update epsilon exploration rate as episodes go on
            exploration_prob = max(min_exploration_prob, exploration_prob * exploration_decay_rate)

            all_rewards.append(total_reward)
            mean_angles.append(np.mean(episode_angles))
            var_angles.append(np.var(episode_angles))

            # display reward every 100 episodes
            if (i + 1) % 100 == 0:
                print(f"Episode {i + 1}: Total Reward = {total_reward}")

        env.close()

        # save training statistics mean and variance
        mean_stats_by_noise[noise] = mean_angles
        var_stats_by_noise[noise] = var_angles

        # save Q-table to pickle file
        filename = f"q_table_ep{episodes}_lr{learning_rate}_noise{noise}_{reward_type}.pkl"
        with open(filename, 'wb') as f:
            pickle.dump(dict(q_table), f)

        save_episode_stats(all_rewards, mean_angles, var_angles, reward_type, noise)

        # plot reward vs episodes
        plt.plot(range(1, episodes + 1), all_rewards)
        plt.xlabel("Episodes")
        plt.ylabel("Total Reward")
        plt.title(f"Episode Rewards - Noise {noise} - Reward {reward_type.capitalize()}")
        plt.savefig(f"rewards_ep{episodes}_lr{learning_rate}_noise{noise}_{reward_type}.png")
        plt.close()

        if converged_at:
            convergence_episodes[f"{reward_type}_noise_{noise}"] = converged_at
        else:
            convergence_episodes[f"{reward_type}_noise_{noise}"] = episodes

    # box plot for mean and variance of pole angles
    noise_labels = [str(n) for n in noise_levels]
    mean_data = [mean_stats_by_noise[n] for n in noise_levels]
    var_data = [var_stats_by_noise[n] for n in noise_levels]

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    axs[0].boxplot(mean_data, labels=noise_labels)
    axs[0].set_title(f"Boxplot of Mean Pole Angle ({reward_type.capitalize()} Reward)")
    axs[0].set_xlabel("Noise Level")
    axs[0].set_ylabel("Mean Pole Angle")
    axs[0].grid(True)

    axs[1].boxplot(var_data, labels=noise_labels)
    axs[1].set_title(f"Boxplot of Pole Angle Variance ({reward_type.capitalize()} Reward)")
    axs[1].set_xlabel("Noise Level")
    axs[1].set_ylabel("Variance")
    axs[1].grid(True)

    plt.tight_layout()
    plt.savefig(f"{reward_type}_reward_pole_angle_stats_boxplot_by_noise.png")
    plt.close()

# plot for first time reaching 500 total rewards
labels = list(convergence_episodes.keys())
values = list(convergence_episodes.values())
plt.figure(figsize=(12, 6))
plt.bar(labels, values)
plt.xticks(rotation=45, ha='right')
plt.ylabel("Episodes to Convergence - 500 reward")
plt.title("Convergence Comparison by Reward Function and Noise")
plt.tight_layout()
plt.savefig("episodes_to_convergence_by_reward_and_noise.png")
plt.show()