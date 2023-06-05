import math
import numpy as np
import random as rd
import matplotlib.pyplot as plt
from itertools import product


def findAction(x, v, epsilon, actions, weights, M):
    actionProb = [0.0, 0.0, 0.0]
    q_values = []
    for a in range(3):
        features = calculateFeatures(x, v, M)
        prod = np.dot(weights[a], features)
        q_values.append(prod)
    bestActions = np.max(q_values)
    bestActionsIndex = [y for y in range(3) if q_values[y] == bestActions]
    for a in range(len(actions)):
        if a in bestActionsIndex:
            actionP = ((1 - epsilon) / len(bestActionsIndex)) + (epsilon / len(actions))
            actionProb[a] = actionP
        else:
            actionP = epsilon / len(actions)
            actionProb[a] = actionP
    return rd.choices(actions, actionProb)[0]


def calculateReward(curr_state):
    if curr_state[0] >= 0.5:
        return 1
    else:
        return -1


def findNextState(curr_state, a1, xMaxMin, vMaxMin):
    v_next = curr_state[1] + (0.001 * a1) - (0.0025 * (np.cos(3 * curr_state[0])))
    if v_next <= vMaxMin[0]:
        v_next = vMaxMin[0]
    if v_next >= vMaxMin[1]:
        v_next = vMaxMin[1]
    x_next = curr_state[0] + v_next
    if x_next <= xMaxMin[0]:
        v_next = 0
        x_next = xMaxMin[0]
    if x_next >= xMaxMin[1]:
        v_next = 0
        x_next = xMaxMin[1]
    next_state = [x_next, v_next]
    return next_state


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


def calculateFeatures(x, v, M):
    normX = (x - 1.2) / (1.7)
    normY = (v - 0.07) / (0.14)
    current_state = np.array([normX, normY]).reshape(2, 1)
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
    xMaxMin = [-1.2, 0.5]
    vMaxMin = [-0.07, 0.07]

    for i in range(episodes):
        stateList = []
        actionList = []
        rewardList = [0]

        x = rd.uniform(-1.2, 0.5)
        while (x == 0.5):
            x = rd.uniform(-1.2, 0.5)
        v = rd.uniform(-0.07, 0.07)
        curr_state = np.array([x, v])

        a = findAction(curr_state[0], curr_state[1], epsilon, actions, weights, M)

        stateList.append(curr_state)
        actionList.append(a)
        T = 1000
        t = 0
        while (True):
            A_t = actionList[-1]
            State_t = stateList[-1]

            if State_t[0] >= 0.5 or t == T:
                break

            R_t1 = calculateReward(State_t)
            State_t1 = findNextState(State_t, A_t, xMaxMin, vMaxMin)

            stateList.append(State_t1)
            rewardList.append(R_t1)

            if State_t1[0] >= 0.5:
                T = t + 1
            else:
                A_t1 = findAction(State_t1[0], State_t1[1], epsilon, actions, weights, M)
                actionList.append(A_t1)

            tau = t - n + 1
            if tau >= 0:
                G = calculateReturn(gamma, rewardList, tau, n, T)
                if tau + n < T:
                    State_tau_n = stateList[tau + n]
                    A_tau_n = actionList[tau + n]
                    features1 = calculateFeatures(State_tau_n[0], State_tau_n[1], M)
                    qHat1 = np.dot(weights[actions.index(A_tau_n)], features1)
                    G += ((gamma ** n) * qHat1)

                State_tau = stateList[tau]
                A_tau = actionList[tau]
                for a in range(len(actions)):
                    features = calculateFeatures(State_tau[0], State_tau[1], M)
                    qHat = np.dot(weights[actions.index(A_tau)], features)
                    weights[a] += (alpha * (G - qHat)) * np.transpose(features)
            t += 1

        episodeIterationArr.append(t)
        if i != episodes:
            movingAvgIterations[i + 1] = movingAvgIterations[i] + t
        print(t)
    return episodeIterationArr, movingAvgIterations


if __name__ == '__main__':
    actions = [-1, 0, 1]
    alpha = 0.04
    epsilon = 0.1
    n = 8
    M = 1
    gamma = 1
    episodes = 1000
    episodeTotalList = []
    movingAvgTotalList = []
    for i in range(20):
        weights = [np.array([0.0] * (M + 1) ** 2).reshape(1, (M + 1) ** 2) for i in actions]
        episodeIterationArr, movingAvgIterations = runEpisodes(alpha, epsilon, M, gamma, actions, episodes, weights, n)
        episodeTotalList.append(episodeIterationArr)
        movingAvgTotalList.append(movingAvgIterations)

    x = []
    y = []
    e = []
    for i in range(episodes):
        x.append(i + 1)

    print('episodeTotalList -- ', episodeTotalList)
    y = np.mean(episodeTotalList, axis=0)
    e = np.std(episodeTotalList, axis=0)
    avgM = np.mean(movingAvgTotalList, axis=0)

    plt.title('Learning curve for Mountain Car')
    plt.xlabel('Episodes')
    plt.ylabel('Number of steps per episode')
    plt.plot(x, y)
    plt.fill_between(x, np.asarray(y) - np.asarray(e), np.asarray(y) + np.asarray(e))
    plt.errorbar(x, y, yerr=e, ecolor='lightblue')
    plt.show()

    x1 = []
    for i in range(episodes + 1):
        x1.append(i)

    plt.title('Learning curve for Mountain Car')
    plt.ylabel('Episodes')
    plt.xlabel('Time Steps')
    plt.plot(avgM, x1)
    plt.show()
