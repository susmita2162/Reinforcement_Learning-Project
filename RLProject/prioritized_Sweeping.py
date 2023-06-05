import operator
import numpy as np
import random as rd
import matplotlib.pyplot as plt


def findRowCol(statesMain, d0):
    for i, s in enumerate(statesMain):
        if d0 in s:
            return i, list(np.where(s == d0)[0])[0]
    return -1, -1


def generateProbabilities(state, qHat, epsilon, actions):
    actionProb = [0, 0, 0, 0]
    q_values = qHat[state]
    bestActions = np.max(q_values)
    bestActionsIndex = [x for x in range(4) if q_values[x] == bestActions]
    for a in range(len(actions)):
        if a in bestActionsIndex:
            actionP = ((1 - epsilon) / len(bestActionsIndex)) + (epsilon / len(actions))
            actionProb[a] = actionP
        else:
            actionP = epsilon / len(actions)
            actionProb[a] = actionP

    return actionProb


# epsilon greedy policy
def determinePolicy(state, qHat, epsilon, actions):
    actionProb = generateProbabilities(state, qHat, epsilon, actions)
    return rd.choices(actions, weights=actionProb)[0]


def findAction(optimalPolicy, state):
    a = optimalPolicy[state]
    return a


def calculateReward(a, row, col, statesMain, gridsize):
    reward = 0
    if a == 'up' and row > 0:
        row -= 1
    elif a == 'down' and row < gridsize:
        row += 1
    elif a == 'left' and col > 0:
        col -= 1
    elif a == 'right' and col < gridsize:
        col += 1
    if row == (gridsize-1) and col == (gridsize-1):
        reward = 10
    # elif row == (gridsize/2) and col == (gridsize-1):
    #     reward = -10
    else:
        reward = 0
    return reward


# Make the state-space deterministic
def findNextState(a, statesMain, row, col, gridsize):
    if a == 'right':
        if col < gridsize-1:
            expectedState = statesMain[row][col + 1]
        else:
            expectedState = statesMain[row][col]
    elif a == 'left':
        if col > 0:
            expectedState = statesMain[row][col - 1]
        else:
            expectedState = statesMain[row][col]
    elif a == 'up':
        if row > 0:
            expectedState = statesMain[row - 1][col]
        else:
            expectedState = statesMain[row][col]
    elif a == 'down':
        if row < gridsize-1:
            expectedState = statesMain[row + 1][col]
        else:
            expectedState = statesMain[row][col]

    # rowE, colE = findRowCol(statesMain, expectedState)
    # if rowE == (gridsize/2) and colE == (gridsize/2):
    #     expectedState = statesMain[row][col]
    # elif rowE == ((gridsize/2)+1) and colE == (gridsize/2):
    #     expectedState = statesMain[row][col]

    return expectedState


def runEpisodes(statesMain, actions, pQueue, epsilon, model, gamma, theta, alpha, optimalPolicy, gridsize):
    qHat = np.zeros((gridsize, gridsize, 4))

    t = 0
    nStates = 0
    nUpdates = 0
    episodeIterationsMap = {}
    episodeRewardMap = {}
    r = 0
    while True:
        if t >= 5000:
            break
        predictedStateList = []
        predictedActionList = []
        predictedRewardList = []

        grid = (gridsize*gridsize)-1
        d0 = np.random.randint(1, grid)
        d0 = str(d0)
        row, col = findRowCol(statesMain, d0)
        curr_state = statesMain[row][col]
        curr_state_index = (row, col)

        a1 = determinePolicy(curr_state_index, qHat, epsilon, actions)
        a1Index = actions.index(a1)

        reward = calculateReward(a1, row, col, statesMain, gridsize)
        r += reward
        next_state = findNextState(a1, statesMain, row, col, gridsize)
        rowN, colN = findRowCol(statesMain, next_state)
        next_state_index = (rowN, colN)

        model[(curr_state, a1)] = [reward, next_state]

        p = np.abs(reward + gamma * np.max(qHat[next_state_index]) - qHat[curr_state_index][a1Index])
        # print('p -- ', p)
        if p > theta:
            pQueue[(curr_state, a1)] = p

        n = 0
        while True:
            n += 1
            if len(pQueue) == 0:
                break
            firstQ = sorted(pQueue.items(), key=operator.itemgetter(1), reverse=True)[0]
            del pQueue[(firstQ[0])]
            curr_state, a1 = firstQ[0][0], firstQ[0][1]
            a1Index = actions.index(a1)
            predictedStateList.append(curr_state)
            predictedActionList.append(a1)
            row, col = findRowCol(statesMain, curr_state)
            curr_state_index = (row, col)
            reward, next_state = model[(curr_state, a1)][0], str(model[(curr_state, a1)][1])
            predictedRewardList.append(reward)
            next_state_index = findRowCol(statesMain, next_state)
            qHat[curr_state_index][a1Index] += alpha * (reward + gamma * np.max(qHat[next_state_index]) - qHat[curr_state_index][a1Index])
            nUpdates += 1

            i = 0
            while True:
                if i > len(predictedStateList) or i < len(predictedActionList):
                    break
                r = predictedRewardList[i]
                state = predictedStateList[i]
                nStates += 1
                row, col = findRowCol(statesMain, state)
                state_index = (row, col)
                a = predictedActionList[i]
                aIndex = actions.index(a)
                p = np.abs(r + gamma * np.max(qHat[curr_state_index]) - qHat[state_index][aIndex])
                if p > theta:
                    pQueue[(state, a)] = p
                i += 1
        episodeIterationsMap[t] = nUpdates
        episodeRewardMap[t] = r
        t += 1

    return qHat, nStates, nUpdates, n, t, episodeIterationsMap, episodeRewardMap


def main():
    actions = ['up', 'down', 'right', 'left']

    pQueue = {}
    epsilon = 0.1
    model = {}
    gamma = 0.9
    theta = 0.1
    alpha = 0.3

    qAvgList = []
    x = []
    y = []
    gridsize = 6
    for i in range(20):
        x.append(gridsize*gridsize)
        #
        #Define the grid
        statesMain = np.zeros((gridsize, gridsize))
        state = 1
        for j in range(gridsize):
            for k in range(gridsize):
                statesMain[j][k] = state
                state += 1
        statesMain = ["%i" % statesMain[j][k] for j in range(gridsize) for k in range(gridsize)]
        statesMain = np.array(statesMain)
        statesMain = statesMain.reshape([gridsize, gridsize])

        #Define Optimal Policy
        optimalPolicy = np.empty([gridsize, gridsize], dtype=str)
        for j in range(gridsize):
            for k in range(gridsize):
                if j == gridsize-1 and k == gridsize-1:
                    optimalPolicy[j][k] = 'G'
                elif k == gridsize-1:
                    optimalPolicy[j][k] = 'down'
                else:
                    optimalPolicy[j][k] = 'right'
        # print("optimal policy -- ", optimalPolicy)

        for i in range(1):
            qCalc, numS, numU, n, t, episodeIterationsMap, episodeRewardMap = runEpisodes(statesMain, actions, pQueue, epsilon, model, gamma, theta, alpha, optimalPolicy, gridsize)
            qAvgList.append(qCalc)

        for k, v in episodeIterationsMap.items():
            x.append(k)
            y.append(v)

        x1 = []
        y1 = []
        for k, v in episodeRewardMap.items():
            x1.append(k)
            y1.append(v)

        policyHat = np.zeros((gridsize, gridsize, 4))
        # print('policyHat -- ', policyHat)
        for q_row in range(gridsize):
            for q_col in range(gridsize):
                state = (q_row, q_col)
                q_values = qCalc[state]
                best_action_value = np.max(q_values)
                best_action_indices = [x for x in range(len(actions)) if q_values[x] == best_action_value]

                # updating the policy
                for j in range(len(actions)):
                    if j in best_action_indices:
                        policyHat[state][j] = ((1 - epsilon) / len(best_action_indices)) + (epsilon / len(actions))
                    else:
                        policyHat[state][j] = epsilon / len(actions)

        valueHat = np.zeros((gridsize, gridsize))
        # for i in range(len(qAvgList)):
        for q_row in range(gridsize):
            for q_col in range(gridsize):
                sumA = 0
                for q_a in range(4):
                    state = (q_row, q_col)
                    sumA += qCalc[q_row][q_col][q_a] * policyHat[state][q_a]
                valueHat[q_row][q_col] = sumA
        # print('qAverage -- ', qAverage)
        print('valueHat -- ', valueHat)

        policy = np.empty([gridsize, gridsize], dtype=str)
        policy[gridsize-1][gridsize-1] = 'G'
        for row in range(gridsize):
            for col in range(gridsize):
                # if policyHat[row][col] == 'G' or policyHat[row][col] == 'o':
                # if policyHat[row][col] == 'G':
                #     continue
                state_index = (row, col)
                bestAction = actions[np.argmax(qCalc[state_index])]
                policy[row][col] = bestAction
        print('policy -- ', policy)
        gridsize *= 2

    plt.xlabel('Episodes')
    plt.ylabel('Number of Steps')
    plt.plot(x, y)
    plt.show()

    plt.xlabel('Episodes')
    plt.ylabel('Rewards over time')
    plt.plot(x1, y1)
    plt.show()


main()
# plotGraph()
