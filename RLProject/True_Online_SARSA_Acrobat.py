import random as rd
import math
import numpy as np
import matplotlib.pyplot as plt
import gym
import statistics

env = gym.make("Acrobot-v1")
actions = [0, 1, 2]
gamma = 1
lam = 0.95
alpha = math.exp(-5)
epsilon = 0.01


def feature_function(observation_space, order):
    cos_theta1 = observation_space[0]
    sin_theta1 = observation_space[1]
    cos_theta2 = observation_space[2]
    sin_theta2 = observation_space[3]
    ang_vel_theta1 = observation_space[4]
    ang_vel_theta2 = observation_space[5]
    state_phi_values = state_features(order, cos_theta1, sin_theta1, cos_theta2, sin_theta2, ang_vel_theta1, ang_vel_theta2)
    d = state_phi_values.shape[0]
    state_weight_values = np.zeros((len(actions), d))
    return cos_theta1, sin_theta1, cos_theta2, sin_theta2, ang_vel_theta1, ang_vel_theta2, state_phi_values, state_weight_values


def state_features(order, cos_theta1, sin_theta1, cos_theta2, sin_theta2, ang_vel_theta1, ang_vel_theta2):
    cos_theta1, sin_theta1, cos_theta2, sin_theta2, ang_vel_theta1, ang_vel_theta2
    norm_cos_theta1 = (cos_theta1 + 1) / 2
    norm_sin_theta1 = (sin_theta1 + 1) / 2
    norm_cos_theta2 = (cos_theta2 + 1) / 2
    norm_sin_theta2 = (sin_theta2 + 1) / 2
    norm_ang_vel_theta1 = (ang_vel_theta1 + 12.567) / 25.134
    norm_ang_vel_theta2 = (ang_vel_theta2 + 28.274) / 56.548
    s_val = np.array([norm_cos_theta1, norm_sin_theta1, norm_cos_theta2, norm_sin_theta2, norm_ang_vel_theta1, norm_ang_vel_theta2])
    phi_s = np.array([])
    for i in range(order + 1):
        for j in range(order + 1):
            for k in range(order + 1):
                for l in range(order + 1):
                    for m in range(order + 1):
                        for n in range(order + 1):
                            sc = np.dot(s_val, np.array([i, j, k, l, m, n]))
                            phi_s = np.append(phi_s, math.cos(math.pi * sc))
    return phi_s


def True_Online_SARSA_Acr(order, N_episodes):
    num_of_episodes = 1
    reward_list = []
    epi_count = []
    step_count_list = []
    action_count = []
    action_epi = 0
    while num_of_episodes <= N_episodes:
        observation_space = env.reset()
        if num_of_episodes == 1:
            cos_theta1, sin_theta1, cos_theta2, sin_theta2, ang_vel_theta1, ang_vel_theta2, phi, weight = feature_function(observation_space[0], order)
        # picking action according to epsilon greedy
        else:
            cos_theta1, sin_theta1, cos_theta2, sin_theta2, ang_vel_theta1, ang_vel_theta2, phi, weight = feature_function(observation_space[0], order)
        step_count = 1
        action = epsilon_greedy(epsilon, step_count, phi, weight)
        x_val = phi
        z_val = np.zeros([len(actions), phi.shape[0]])
        cap_q_old = 0
        reward = -1
        terminated = False
        while step_count <= 500 and terminated == False:
            action_epi += 1
            observation_space, r, terminated, truncated, info = env.step(action)
            #  in while (array([ 0.01759233, -0.36618477,  0.02355349,  0.64313155], dtype=float32), 1.0, False, False, {})
            reward = r
            x_dash = state_features(order, observation_space[0], observation_space[1], observation_space[2],
                                    observation_space[3], observation_space[4], observation_space[5])
            next_action = epsilon_greedy(epsilon, step_count, x_dash, weight)
            cap_q = np.dot(weight[action], x_val)
            cap_q_dash = np.dot(weight[next_action], x_dash)
            delta = reward + (gamma * cap_q_dash - cap_q)
            z_val[action] = gamma * lam * z_val[action] + (1 - alpha * gamma * lam * np.dot(z_val[action], x_val)) * x_val
            weight[action] = weight[action] + alpha * (delta + cap_q - cap_q_old) * z_val[action] - alpha * (
                    cap_q - cap_q_old) * x_val
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
        bestActionsIndex = [y for y in range(len(actions)) if q_val[y] == bestActions]
        for a in range(len(actions)):
            if a in bestActionsIndex:
                actionP = ((1 - epsilon) / len(bestActionsIndex)) + (epsilon / len(actions))
                actionProb[a] = actionP
            else:
                actionP = epsilon / len(actions)
                actionProb[a] = actionP
        action_val = rd.choices(actions, actionProb)[0]
    return action_val


def main():
    episode_count = []
    step_to_goal_count_list = []
    action_arr = []
    count = 0
    N_episodes = 700
    for i in range(20):
        count += 1
        order = 2
        step_to_goal_count_list_temp, episodes, action_count = True_Online_SARSA_Acr(order, N_episodes)
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

    # plt.figure(figsize=(10, 6))
    for i in range(N_episodes+1):
        episode_count.append(i)
    plt.errorbar(episode_count, avg_steps_goal_array / 20, yerr=standard_deviation, capsize=0.2,
                 elinewidth=0.3)
    plt.title('True Online SARSA(Acrobat)- Episodes Vs No. of steps to reach goal')
    plt.xlabel('Episodes')
    plt.ylabel("No. of steps to reach goal")
    plt.show()

    plt.plot(avg_steps_array / 20, episode_count)
    plt.title('True Online SARSA(Acrobat) - Time steps Vs Episodes')
    plt.xlabel('Time steps')
    plt.ylabel("Episodes")
    plt.show()


if __name__ == main():
    main()
