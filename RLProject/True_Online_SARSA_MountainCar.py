import random
import math
import numpy as np
import random as rd
import matplotlib.pyplot as plt
import statistics

actions = [-1, 0, 1]
gamma = 1
lam = 0.96
alpha = 0.0009
epsilon = 0.01


def feature_function(order):
    pos = random.uniform(-0.6, -0.4)
    vel = 0
    state_phi_values = state_features(order, pos, vel)
    d = state_phi_values.shape[0]
    state_weight_values = np.zeros((len(actions), d))
    return pos, vel, state_phi_values, state_weight_values


def state_features(order, pos, vel):
    norm_distance = (pos + 1.2) / 1.7
    norm_velocity = (vel + 0.07) / 0.14
    s_val = np.array([norm_distance, norm_velocity])
    phi_s = np.array([])
    for i in range(order + 1):
        for j in range(order + 1):
            sc = np.dot(s_val, np.array([i, j]))
            phi_s = np.append(phi_s, math.cos(math.pi * sc))
    return phi_s


def True_Online_SARSA(order, N_episodes):
    num_of_episodes = 1
    reward_list = []
    epi_count = []
    step_count_list = []
    action_count = []
    action_epi = 0
    while num_of_episodes <= N_episodes:
        if num_of_episodes==1:
            pos, vel, phi, weight = feature_function(order)
        else:
            pos, vel, phi, weight_t = feature_function(order)

        step_count = 1
        action = epsilon_greedy(epsilon, step_count, phi, weight)
        x_val = phi
        z_val = np.zeros([3, phi.shape[0]])
        cap_q_old = 0
        reward = -1
        while pos < 0.5 and step_count < 1000:
            action_epi += 1
            #next state calculation
            if action == 0:
                action_val = -1
            elif action == 1:
                action_val = 0
            elif action == 2:
                action_val = 1
            vel = vel + 0.001 * action_val - 0.0025 * math.cos(3 * pos)
            pos = pos + vel
            if pos <= -1.2:
                vel = 0
                pos = -1.2
            if vel <= -0.07:
                vel = -0.07
            if vel >= 0.07:
                vel = 0.07
            if pos < 0.5:
                reward = -1
            else:
                reward = 0
            x_dash = state_features(order, pos, vel)
            next_action = epsilon_greedy(epsilon, step_count, x_dash, weight)
            cap_q = np.dot(weight[action], x_val)
            cap_q_dash = np.dot(weight[next_action], x_dash)

            delta = reward + (gamma * cap_q_dash - cap_q)

            z_val[action] = gamma * lam * z_val[action] + (1 - alpha * gamma * lam * np.dot(z_val[action], x_val)) * x_val
            weight[action] = weight[action] + (alpha * (delta + cap_q - cap_q_old) * z_val[action]) - (alpha * (cap_q - cap_q_old) * x_val)
            cap_q_old = cap_q_dash
            x_val = x_dash
            action = next_action
            step_count += 1


        epi_count.append(num_of_episodes)
        step_count_list.append(step_count)
        action_count.append(action_epi)
        num_of_episodes += 1
    return step_count_list, epi_count, action_count


def epsilon_greedy(epsilon, step_count, phi, weight):
    actionProb = [0.0, 0.0, 0.0]
    q_val = np.array([])
    if step_count == 1:
        action_val = np.random.choice(actions, 1)[0]
    else:
        for i in range(weight.shape[0]):
            weight[i] = np.array(weight[i])
            q_val = np.append(q_val, np.dot(weight[i], phi))
        bestActions = np.max(q_val)
        bestActionsIndex = [y for y in range(3) if q_val[y] == bestActions]
        for a in range(len(actions)):
            if a in bestActionsIndex:
                actionP = ((1 - epsilon) / len(bestActionsIndex)) + (epsilon / len(actions))
                actionProb[a] = actionP
            else:
                actionP = epsilon / len(actions)
                actionProb[a] = actionP
        action_val = rd.choices(actions, actionProb)[0]
    if action_val == -1:
        action_index = 0
    elif action_val == 0:
        action_index = 1
    elif action_val == 1:
        action_index = 2
    return action_index


def main():
    episode_count = []
    step_to_goal_count_list = []
    action_arr = []
    count = 0
    N_episodes = 700
    for i in range(20):
        count += 1
        order = 10
        step_to_goal_count_list_temp, episodes, action_count = True_Online_SARSA(order, N_episodes)
        step_to_goal_count_list.append(step_to_goal_count_list_temp)
        action_arr.append(action_count)

    avg_steps_array = np.zeros((N_episodes))
    for i in range(N_episodes):
        for temp_array in action_arr:
            avg_steps_array[i] = avg_steps_array[i] + temp_array[i]

    avg_steps_goal_array = np.zeros((N_episodes))
    standard_deviation = []
    for i in range(N_episodes):
        temp_standard_dev = []
        for temp_arr in step_to_goal_count_list:
            avg_steps_goal_array[i] = avg_steps_goal_array[i] + temp_arr[i]
            temp_standard_dev.append(temp_arr[i])

        standard_deviation.append(statistics.stdev(temp_standard_dev))


    # plt.plot(average_steps_array / 20, episode_count)
    # plt.title('True Online SARSA(Mountain Car) - Time steps Vs Episodes')
    # plt.xlabel('Time steps')
    # plt.ylabel('Episodes')
    #
    # plt.xlabel('Number of episodes')
    # plt.ylabel('step count')
    # plt.plot(episodes, step_to_goal_count_list)
    # plt.show()

    # plt.figure(figsize=(10, 6))
    for i in range(N_episodes+1):
        episode_count.append(i)
    plt.errorbar(episode_count, avg_steps_goal_array / 20, yerr=standard_deviation, capsize=0.2,
                 elinewidth=0.3)
    plt.title('True Online SARSA(Mountain Car)- Episodes Vs No. of steps to reach goal')
    plt.xlabel('Episodes')
    plt.ylabel("No. of steps to reach goal")
    plt.show()

    plt.plot(avg_steps_array / 20, episode_count)
    plt.title('True Online SARSA(Mountain Car) - Time steps Vs Episodes')
    plt.xlabel('Time steps')
    plt.ylabel("Episodes")
    plt.show()




if __name__ == main():
    main()
