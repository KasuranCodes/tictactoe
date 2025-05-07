import gym
import random
# ---------------------------
# Step 1: Initialize Environment and Parameters
# ---------------------------

env = gym.make('FrozenLake-v1', is_slippery=False) # Disable slippingfor easier learning
num_states = env.observation_space.n
num_actions = env.action_space.n
# Hyperparameters

learning_rate = 0.2         # Alpha
discount_factor = 0.99      # Gamma
epsilon = 1.0               # Initial exploration rate
epsilon_min = 0.01          # Minimum exploration rate
epsilon_decay = 0.995       # Decay rate for epsilon
num_episodes = 5000         # Total number of episodes for training
# Initialize Q-table with small random values

Q_table = [[random.uniform(0, 0.1) for _ in range(num_actions)] for _ in range(num_states)]
# ---------------------------
# Step 2: Train the Agent Using Q-Learning
# ---------------------------
for episode in range(num_episodes):
    state, _ = env.reset()
    done = False
    total_reward = 0
    while not done:
        # ε-greedy action selection
        if random.random() < epsilon:
            action = env.action_space.sample() # Explore
        else:
            action = Q_table[state].index(max(Q_table[state])) # Exploit

        # Take the action
        next_state, reward, terminated, truncated, info = env.step(action) 
        done = terminated or truncated
        
        # Reward shaping
        if done:
            if reward == 1: # Reached the goal
                reward = 10
            else: # Fell into a hole
                reward = -10
        else:
            reward = -0.1 # Small penalty for each step

        # Update Q-value
        Q_table[state][action] += learning_rate * (reward + discount_factor * max(Q_table[next_state]) - Q_table[state][action])

        # Move to the next state
        state = next_state
        total_reward += reward

        if done:
            break
        
    # Decay epsilon
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    # Print progress every 100 episodes
    if (episode + 1) % 100 == 0:
        print(f"Episode {episode + 1}/{num_episodes}, Total Reward:{total_reward}")
# ---------------------------
# Step 3: Evaluate the Trained Agent
# ---------------------------
def evaluate_agent(env, Q_table, num_episodes=100):
    total_rewards = 0
    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        while not done:
            action = Q_table[state].index(max(Q_table[state]))
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            state = next_state
            total_rewards += reward

            if done:
                break
    average_reward = total_rewards / num_episodes
    print(f"Average reward over {num_episodes} evaluation episodes:{average_reward}")
    return average_reward
evaluate_agent(env, Q_table)
# ---------------------------
# Step 4: Visualize the Learned Policy
# ---------------------------
def print_policy(Q_table, grid_size=4):
    directions = {0: "←", 1: "↓", 2: "→", 3: "↑"}
    policy = []

    for state in range(len(Q_table)):
        best_action = Q_table[state].index(max(Q_table[state]))
        policy.append(directions[best_action])
        
        policy_grid = [policy[i:i + grid_size] for i in range(0,
        len(policy), grid_size)]
        print("\nLearned Policy:")
        for row in policy_grid:
            print(" ".join(row))
print_policy(Q_table)