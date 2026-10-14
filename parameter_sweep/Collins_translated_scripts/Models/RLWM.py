# Python translation of original matlab script using ChatGPT

# Anne GE Collins, UC Berkeley
# annecollins@berkeley.edu, 2024
# Code for paper "RL or not RL? Parsing the processes that support human
# reward-based learning."

# This function performs one simulation of the model.

import numpy as np

def RLWM(theta, expe_data, M):
    """Simulate using the original 17-column data format. M is a dict; its mapping values remain one-based."""

    # This function performs one simulation of the model, and returns simulated
    # data in the same format as experimental data
    # input
    # theta: vector of parameters (size depends on model M)
    # expe_data: experimental data matrix that constrains the experiment on
    # which the model is simulated (e.g. sequence of blocks, stimuli, correct
    # actions, etc.)
    # M: structure containing information about model to be simulated.

    # Python input conversion (indices in the data and model mapping remain one-based)
    theta = np.asarray(theta, dtype=float).reshape(-1)
    thetamapping = np.asarray(M['thetamapping']).reshape(-1)
    pfixed = np.asarray(M['pfixed']).reshape(-1)
    fixedvalue = np.asarray(M['fixedvalue']).reshape(-1)

    # number of actions
    nA = 3
    beta = 25  # softmax noise
    expe_data = np.asarray(expe_data, dtype=float)

    # %% define model parameters
    K = theta[-1]  # capacity
    alpha = theta[int(thetamapping[0]) - 1]  # learning rate
    stick = theta[int(thetamapping[1]) - 1]  # motor perseveration
    rho = theta[int(thetamapping[2]) - 1]  # WM weight
    forget = theta[int(thetamapping[3]) - 1]  # WM decay
    epsilon = theta[int(thetamapping[4]) - 1]  # lapse rate

    # model dependent parameters: WM learning rate bias
    if pfixed[5]:
        biasWM = fixedvalue[5]
    else:
        biasWM = theta[int(thetamapping[5]) - 1]

    # RL learning rate bias
    if pfixed[6]:
        biasRL = fixedvalue[6]
    else:
        biasRL = theta[int(thetamapping[6]) - 1]

    # RL r0 value (r0=1 --> H model)
    if pfixed[7]:
        r0 = fixedvalue[7]
    else:
        r0 = theta[int(thetamapping[7]) - 1]

    # policy compression
    if len(pfixed) < 10:
        chunk = 0
    elif pfixed[8]:
        chunk = fixedvalue[8]
    else:
        chunk = theta[int(thetamapping[8]) - 1]

    # set up learning rates [alpha- alpha+]
    alphaRL = np.array([alpha * biasRL, alpha])
    alphaWM = np.array([biasWM, 1])

    # outcome values
    outcomes = np.array([r0, 1]) if M['value'] == 1 else np.array([1, 1])

    # %% set up the simulation experiment
    # extract relevant experimental data
    Allstimuli = expe_data[:, 4].copy()  # sequence of stimuli
    Allchoices = expe_data[:, 9].copy()  # sequence of choices
    Allrewards = expe_data[:, 11].copy()  # sequence of rewards
    Allsetsize = expe_data[:, 2].copy()  # sequence of set sizes
    Allblocks = expe_data[:, 1].copy()  # sequence of blocks
    Allrts = expe_data[:, 13].copy()  # sequence of reaction times
    AllaCor = expe_data[:, 8].copy()  # sequence of target correct action
    Alliter = expe_data[:, 7].copy()  # sequence of iteration number
    Alldelay = expe_data[:, 16].copy()  # sequence of delay
    Allpcor = expe_data[:, 15].copy()  # sequence of number of previous correct

    # create new simulation data
    sim_data = expe_data.copy()
    bRLs = np.empty((0, nA))
    bWMs = np.empty((0, nA))

    # identify number of blocks
    blocks = np.unique(Allblocks)

    # %% start the simulation
    # loop over blocks
    for bl in blocks:

        # simulate only for relevant trials (as in data analysis)
        Tb = np.flatnonzero((Allblocks == bl) & (Allrts > .15))

        # extract experiment information for current block
        stimuli = Allstimuli[Tb]
        choices = Allchoices[Tb]
        rewards = Allrewards[Tb]
        ns = int(Allsetsize[Tb[0]])
        acors = AllaCor[Tb]
        iters = Alliter[Tb]

        # WM weight
        w = rho * min(1, K / ns)
        wint = w if M['interact'] else 0

        # initialize RL/H weights, and WM weights.
        Q = (1 / nA) * np.ones((ns, nA))
        WM = (1 / nA) * np.ones((ns, nA))
        b = Q[0, :].copy()

        # loop over trials
        for k in range(1, len(choices)):
            s = int(stimuli[k - 1]) - 1
            if iters[k - 1] == 1:

                # make first choice same as participants
                choice = int(choices[k - 1])
            else:

                # make a choice
                choice = select(b)

                # store choice
                choices[k - 1] = choice

            # reward is deterministic if choice corresponds to target choice
            r = int(choice == acors[k - 1])

            # store reward
            rewards[k - 1] = r
            choice = choice - 1

            # model updates
            # WM decay
            WM = WM + forget * (1 / nA - WM)

            # compute RL/H RPE
            rpe = outcomes[r] - (wint * WM[s, choice] + (1 - wint) * Q[s, choice])

            # update RL/H
            Q[s, choice] = Q[s, choice] + alphaRL[r] * (outcomes[r] - (wint * WM[s, choice] + (1 - wint) * Q[s, choice]))

            # update WM
            WM[s, choice] = WM[s, choice] + alphaWM[r] * (r - WM[s, choice])

            # set up sticky choice
            side = np.zeros(nA)
            side[choice] = 1

            # compute policy compression
            if chunk > 0:
                if ns > bRLs.shape[0]:
                    extra = ns - bRLs.shape[0]
                    bRLs = np.vstack((bRLs, np.zeros((extra, nA))))
                    bWMs = np.vstack((bWMs, np.zeros((extra, nA))))
                for s in range(ns):
                    W = Q[int(stimuli[k]) - 1, :] + stick * side
                    bRLs[s, :] = np.exp(beta * W)
                    bRLs[s, :] = epsilon / nA + (1-epsilon) * bRLs[s, :] / np.sum(bRLs[s, :])
                    W = WM[int(stimuli[k]) - 1, :] + stick * side
                    bWMs[s, :] = np.exp(beta * W)
                    bWMs[s, :] = epsilon / nA + (1-epsilon) * bWMs[s, :] / np.sum(bWMs[s, :])
                # MATLAB mean uses the first non-singleton dimension; old rows persist.
                meanWM = np.mean(bWMs) if bWMs.shape[0] == 1 else np.mean(bWMs, axis=0)
                meanRL = np.mean(bRLs) if bRLs.shape[0] == 1 else np.mean(bRLs, axis=0)
                bs = w * meanWM + (1-w) * meanRL
            else:
                bs = 0

            # set up RL policy with sticky and compression
            W = Q[int(stimuli[k]) - 1, :] + stick * side + chunk * bs
            bRL = np.exp(beta * W)

            # include random lapses
            bRL = epsilon / nA + (1-epsilon) * bRL / np.sum(bRL)

            # set up WM policy with sticky and compression
            W = WM[int(stimuli[k]) - 1, :] + stick * side + chunk * bs
            bWM = np.exp(beta * W)

            # include random lapses
            bWM = epsilon / nA + (1-epsilon) * bWM / np.sum(bWM)

            # overall policy is the mixture
            b = w * bWM + (1-w) * bRL

        # store rewards and choices for this block
        Allrewards[Tb] = rewards
        Allchoices[Tb] = choices

        # compute and store derived variables delay and pcor
        Alldelay[Tb] = computedelay(stimuli, rewards)
        Allpcor[Tb] = computepcor(stimuli, rewards)

    # store simulation data
    sim_data[:, 9] = Allchoices
    sim_data[:, 11] = Allrewards
    sim_data[:, 12] = Allrewards
    sim_data[:, 15] = Allpcor
    sim_data[:, 16] = Alldelay
    return sim_data


# implement random choice selection based on a probability distribution
def select(pr):
    # MATLAB: find([0 cumsum(pr)] < rand, 1, 'last').
    indices = np.flatnonzero(np.r_[0, np.cumsum(pr)] < np.random.random())
    if indices.size == 0:
        raise IndexError('MATLAB select returned an empty index')
    return int(indices[-1]) + 1


# compute delay since last correct
def computedelay(stimuli, rewards):
    delay = np.full(stimuli.shape, np.nan)
    for s in np.unique(stimuli):
        T = np.flatnonzero(stimuli == s)
        for t in T[1:]:
            y = np.flatnonzero((stimuli[:t] == s) & (rewards[:t] == 1))
            if y.size:
                delay[t] = t - y[-1]
    return delay


# compute pcor
def computepcor(stimuli, rewards):
    pcor = 0 * stimuli
    for s in np.unique(stimuli):
        T = np.flatnonzero(stimuli == s)
        pcor[T] = np.cumsum(rewards[T]) - rewards[T]
    return pcor
