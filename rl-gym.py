import gym
import numpy as np
import matplotlib.pyplot as plt

# Make the car environment - It has 3 possible actions (l,r,nothing)
env = gym.make("MountainCar-v0")

# For saving stats
ep_rewards = []
aggr_ep_rewards = {'ep': [], 'avg': [], 'max': [], 'min': []}
STATS_EVERY = 100

# Q-table settings
DISCRETE_OS_SIZE = [20, 20]
discrete_os_win_size = (env.observation_space.high-env.observation_space.low)/DISCRETE_OS_SIZE

# Q-learning settings
LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES =  25000

# Exploration settings
epsilon = 1 # not a constant, going to decay
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES//2
epsilon_decay_value = epsilon/(END_EPSILON_DECAYING - START_EPSILON_DECAYING)

# how often to render the environment (show every __ episodes)
SHOW_EVERY = 4000

# Initialize q-table
q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))

# Function to get the bucket (tuple) the state falls into
def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low)/discrete_os_win_size
    return tuple(discrete_state.astype(np.int)) # we'll use this tuple to look up the three Q values for available actions (in the q-table)
    # note to self: tuples are immutable, so initially, the state is a list, and then it's converted to a tuple to be returned

for episode in range(EPISODES):
    episode_reward = 0

    # Prep the environment
    init_state = env.reset()
    discrete_state = get_discrete_state(init_state)

    # Render the environment, if it's the right episode:
    if episode % SHOW_EVERY == 0:
        render = True
        print(episode)
    else:
        render = False

    # do the things! run the simulation:
    done = False
    while not done:
        # choose the best q value action, or randomly explore:
        if np.random.random() > epsilon:
            # get action from q table
            action  = np.argmax(q_table[discrete_state]) # 1=left, 2=right, 0=nothing
        else:
            # get action randomly
            action = np.random.randint(0, env.action_space.n)


        new_state, reward, done, _ = env.step(action)
        new_discrete_state = get_discrete_state(new_state)

        episode_reward += reward

        # don't render every time:
        if render == True:
            env.render()

        # if simulation didn't end in the last step, update q-table
        if not done:
            # get max possible q value in *next* step (for new state)
            max_future_q = np.max(q_table[new_discrete_state])

            # get current q value (for currrent state & performed action)
            current_q = q_table[discrete_state + (action,)]

            # finally, get the new q value for the current state+action:
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)

            # update q table with new q value
            q_table[discrete_state + (action,)] = new_q

        # simulation ended (for any reason):
        # if goal pos acheived, update q value with reward directly:
        elif new_state[0] >= env.unwrapped.goal_position:
            #q_table[discrete_state + (action,)] = reward
            q_table[discrete_state + (action,)] = 0

        discrete_state = new_discrete_state

    # Let's reduce epsilon (less exploration, more exploitation):
    # This decay occurs every episode, if the episode is within bounds:
    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value

    # Update stats
    ep_rewards.append(episode_reward)
    if not episode % STATS_EVERY:
        average_reward = sum(ep_rewards[-STATS_EVERY:])/STATS_EVERY
        aggr_ep_rewards['ep'].append(episode)
        aggr_ep_rewards['avg'].append(average_reward)
        aggr_ep_rewards['max'].append(max(ep_rewards[-STATS_EVERY:]))
        aggr_ep_rewards['min'].append(min(ep_rewards[-STATS_EVERY:]))
        print(f'Episode: {episode:>5d}, average reward: {average_reward:>4.1f}, current epsilon: {epsilon:>1.2f}')

env.close

plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label="average rewards")
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label="max rewards")
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label="min rewards")
plt.legend(loc=4)
plt.show()

