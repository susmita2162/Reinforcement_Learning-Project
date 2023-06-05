import math
import numpy as np
import random as rd
import matplotlib.pyplot as plt
from itertools import product
import gym

env = gym.make("Acrobot-v1")

def findAction(curr_state, epsilon, actions, weights, M):
    actionProb = [0.0, 0.0, 0.0]
    q_values = []
    for a in range(2):
        features = calculateFeatures(curr_state, M)
        prod = np.dot(weights[a], features)
        q_values.append(prod)
    bestActions = np.max(q_values)
    bestActionsIndex = [y for y in range(2) if q_values[y] == bestActions]
    for a in range(len(actions)):
        if a in bestActionsIndex:
            actionP = ((1 - epsilon) / len(bestActionsIndex)) + (epsilon / len(actions))
            actionProb[a] = actionP
        else:
            actionP = epsilon / len(actions)
            actionProb[a] = actionP
    return rd.choices(actions, actionProb)[0]


def calculateReturn(gamma, rewardList, tau, n, T):
    G = 0
    for i in range(tau + 1, min(tau + n, T)):
        G += (gamma ** (i - tau - 1)) * rewardList[i]
    return G


def tuple_flat(data):
    if isinstance(data, tuple):
        for x in data:
            yield from tuple_flat(x)
    else:
        yield data


def calculateFeatures(curr_state, M):
    normX1 = (curr_state[0] + 1) / 2
    normX2 = (curr_state[1] + 1) / 2
    normX3 = (curr_state[2] + 1) / 2
    normX4 = (curr_state[3] + 1) / 2
    normX5 = (curr_state[4] + 12.567) / 25.134
    normX6 = (curr_state[5] + 28.274) / 56.548
    current_state = np.array([normX1, normX2, normX3, normX4, normX5, normX6]).reshape(6, 1)
    arr = [[i for i in range(M + 1)] for var in current_state]
    arr_mult = arr[0]
    for i in range(1, len(arr)):
        arr_mult = product(arr[i], arr_mult)
    arr_mult = list(arr_mult)
    state_eq = np.array([list(tuple_flat(i)) for i in arr_mult])
    return np.cos(math.pi * np.matmul(state_eq, current_state).reshape(len(state_eq), 1))


def runEpisodes(alpha, epsilon, M, gamma, actions, episodes, weights, n):
    episodeIterationArr = []
    movingAvgIterations = [0] * (episodes + 1)

    for i in range(episodes):
        observation_space = env.reset()
        stateList = []
        actionList = []
        rewardList = [0]
        curr_state = np.array(
            [observation_space[0][0], observation_space[0][1], observation_space[0][2], observation_space[0][3],
             observation_space[0][4], observation_space[0][5]])

        a = findAction(curr_state, epsilon, actions, weights, M)

        stateList.append(curr_state)
        actionList.append(a)
        T = 1000
        t = 0
        terminated = False
        # while (True):
        while (terminated == False):
            A_t = actionList[-1]
            State_t = stateList[-1]

            if t == T:
                break

            State_t1, R_t1, terminated, truncated, info = env.step(A_t)

            stateList.append(State_t1)
            rewardList.append(R_t1)

            if State_t1[0] >= 2.4:
                T = t + 1
            else:
                A_t1 = findAction(State_t1, epsilon, actions, weights, M)
                actionList.append(A_t1)

            tau = t - n + 1
            if tau >= 0:
                G = calculateReturn(gamma, rewardList, tau, n, T)
                if tau + n < T:
                    State_tau_n = stateList[tau + n]
                    A_tau_n = actionList[tau + n]
                    features1 = calculateFeatures(State_tau_n, M)
                    qHat1 = np.dot(weights[actions.index(A_tau_n)], features1)
                    G += ((gamma ** n) * qHat1)

                State_tau = stateList[tau]
                A_tau = actionList[tau]
                for a in range(len(actions)):
                    features = calculateFeatures(State_tau, M)
                    qHat = np.dot(weights[actions.index(A_tau)], features)
                    weights[a] += (alpha * (G - qHat)) * np.transpose(features)
            t += 1

        print(t)
        episodeIterationArr.append(t)
        if i != episodes:
            movingAvgIterations[i + 1] = movingAvgIterations[i] + t
    return episodeIterationArr, movingAvgIterations


if __name__ == '__main__':
    actions = [0, 1, 2]
    alpha = 0.001
    epsilon = 0.1
    n = 4
    M = 1
    gamma = 1
    episodes = 500
    episodeTotalList = []
    movingAvgTotalList = []
    for i in range(1):
        weights = [np.array([0.0] * (M + 1) ** 6).reshape(1, (M + 1) ** 6) for i in actions]
        episodeIterationArr, movingAvgIterations = runEpisodes(alpha, epsilon, M, gamma, actions, episodes, weights, n)
        episodeTotalList.append(episodeIterationArr)
        movingAvgTotalList.append(movingAvgIterations)

    x = []
    y = []
    e = []
    for i in range(episodes):
        x.append(i + 1)

    y = np.mean(episodeTotalList, axis=0)
    e = np.std(episodeTotalList, axis=0)
    avgM = np.mean(movingAvgTotalList, axis=0)

    plt.title('Learning curve for Acrobat')
    plt.xlabel('Episodes')
    plt.ylabel('Number of steps per episode')
    plt.plot(x, y)
    plt.fill_between(x, np.asarray(y) - np.asarray(e), np.asarray(y) + np.asarray(e))
    plt.errorbar(x, y, yerr=e, ecolor='lightblue')
    plt.show()

    x1 = []
    for i in range(episodes + 1):
        x1.append(i)

    plt.title('Learning curve for Acrobat')
    plt.ylabel('Episodes')
    plt.xlabel('Time Steps')
    plt.plot(avgM, x1)
    plt.show()
