import math
import numpy as np
import random as rd
import matplotlib.pyplot as plt
from itertools import product
import gym

env = gym.make("CartPole-v1")

def findAction(x, v, angle, angular_v, epsilon, actions, weights, M):
    actionProb = [0.0, 0.0]
    q_values = []
    for a in range(2):
        features = calculateFeatures(x, v, angle, angular_v, M)
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
        G += (gamma ** (i-tau-1)) * rewardList[i]
    return G

def tuple_flat(data):
    if isinstance(data, tuple):
        for x in data:
            yield from tuple_flat(x)
    else:
        yield data

def calculateFeatures(x, v, angle, angular_v, M):
    normX = (x + 4.8) / (9.6)
    normY = (v + 4) / 8
    norm_angle = (angle + 0.418) / 0.936
    norm_angular_vel = (angular_v + 4) / 8
    current_state = np.array([normX, normY, norm_angle, norm_angular_vel]).reshape(4, 1)
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
            [observation_space[0][0], observation_space[0][1], observation_space[0][2], observation_space[0][3]])
        angle = observation_space[0][2]
        angular_v = observation_space[0][3]
        a = findAction(curr_state[0], curr_state[1], angle, angular_v, epsilon, actions, weights, M)
        stateList.append(curr_state)
        actionList.append(a)
        T = 1000
        t = 0
        terminated = False
        # while (True):
        while(terminated == False):
            A_t = actionList[-1]
            State_t = stateList[-1]
            if t >= 500:
                break
            State_t1, R_t1, terminated, truncated, info = env.step(A_t)
            stateList.append(State_t1)
            rewardList.append(R_t1)
            if State_t1[0] >= 2.4:
                T = t + 1
            else:
                A_t1 = findAction(State_t1[0], State_t1[1], State_t1[2], State_t1[3], epsilon, actions, weights, M)
                actionList.append(A_t1)
            tau = t - n + 1
            if tau >= 0:
                G = calculateReturn(gamma, rewardList, tau, n, T)
                if tau + n < T:
                    State_tau_n = stateList[tau + n]
                    A_tau_n = actionList[tau + n]
                    features1 = calculateFeatures(State_tau_n[0], State_tau_n[1], State_tau_n[2], State_tau_n[3], M)
                    qHat1 = np.dot(weights[actions.index(A_tau_n)], features1)
                    G += ((gamma ** n) * qHat1)
                State_tau = stateList[tau]
                A_tau = actionList[tau]
                for a in range(len(actions)):
                    features = calculateFeatures(State_tau[0], State_tau[1], State_tau[2], State_tau[3], M)
                    qHat = np.dot(weights[actions.index(A_tau)], features)
                    weights[a] += (alpha * (G - qHat)) * np.transpose(features)
            t += 1

        episodeIterationArr.append(t)
        print('t -- ', t)
        if i != episodes:
            movingAvgIterations[i + 1] = movingAvgIterations[i] + t
    return episodeIterationArr, movingAvgIterations


if __name__ == '__main__':
    actions = [0, 1]
    # alpha = 0.04
    # epsilon = 0.1
    alpha = 0.4
    epsilon = 0.2
    n = 4
    M = 2
    gamma = 0.9
    episodes = 1000
    episodeTotalList = []
    movingAvgTotalList = []
    for i in range(1):
        weights = [np.array([0.0] * (M+1)**4).reshape(1, (M+1)**4) for i in actions]
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

    plt.title('Learning curve for CartPole')
    plt.xlabel('Episodes')
    plt.ylabel('Number of steps per episode')
    plt.plot(x, y)
    plt.fill_between(x, np.asarray(y) - np.asarray(e), np.asarray(y) + np.asarray(e))
    plt.errorbar(x, y, yerr=e, ecolor='lightblue')
    plt.show()

    x1 = []
    for i in range(episodes + 1):
        x1.append(i)

    plt.title('Learning curve for CartPole')
    plt.ylabel('Episodes')
    plt.xlabel('Time Steps')
    plt.plot(avgM, x1)
    plt.show()
