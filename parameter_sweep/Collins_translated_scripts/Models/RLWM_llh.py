# Python translation of original matlab script using ChatGPT

# Anne GE Collins, UC Berkeley
# annecollins@berkeley.edu, 2024
# Code for paper "RL or not RL? Parsing the processes that support human
# reward-based learning."

# This function computes the log-likelihood llh of the data under parameters
# (theta,K) for model M.

import numpy as np

def RLWM_llh(theta, K, data, M):
    """Return negative log-likelihood for the original six-column fitting data. M is a dict; its mapping values remain one-based."""

    # This function computes the likelihood of the model
    # theta: vector of parameters (size depends on model M)
    # K: discrete capacity value
    # data: data matrix
    # M: structure containing information about model to be simulated.

    # Python input conversion (indices in the data and model mapping remain one-based)
    theta = np.asarray(theta, dtype=float).reshape(-1)
    thetamapping = np.asarray(M['thetamapping']).reshape(-1)
    pfixed = np.asarray(M['pfixed']).reshape(-1)
    fixedvalue = np.asarray(M['fixedvalue']).reshape(-1)

    # number of actions
    nA = 3

    # %% define model parameters
    beta = 25  # softmax noise
    data = np.asarray(data)
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

    # %% extract relevant data
    Allstimuli = data[:, 0]  # sequence of stimuli
    Allchoices = data[:, 1]  # sequence of choices
    Allrewards = data[:, 2]  # sequence of rewards
    Allsetsize = data[:, 3]  # sequence of set sizes
    Allblocks = data[:, 4]  # sequence of blocks
    Allrts = data[:, 5]  # sequence of reaction times
    llh = 0
    ls = {}
    bRLs = np.empty((0, nA))
    bWMs = np.empty((0, nA))

    # %% run through the data
    blocks = np.unique(Allblocks)

    # loop over blocks
    for bl in blocks:

        # extract relevant trials for this block
        Tb = np.flatnonzero((Allblocks == bl) & (Allrts > .15))
        stimuli = Allstimuli[Tb]
        choices = Allchoices[Tb]
        rewards = Allrewards[Tb]
        ns = int(Allsetsize[Tb[0]])

        # WM weight
        w = rho * min(1, K / ns)
        wint = w if M['interact'] else 0

        # initialize RL/H weights, and WM weights.
        Q = (1 / nA) * np.ones((ns, nA))
        WM = (1 / nA) * np.ones((ns, nA))
        lt = np.log(1 / nA)
        llh = llh + lt

        # trial's stimulus, choice, and outcome
        for k in range(1, len(choices)):
            s = int(stimuli[k - 1]) - 1
            choice = int(choices[k - 1]) - 1
            r = int(rewards[k - 1])

            # update the model
            # WM decay
            WM = WM + forget * (1 / nA - WM)

            # compute RL/H RPE
            rpe = outcomes[r] - (wint * WM[s, choice] + (1 - wint) * Q[s, choice])

            # update RL/H
            Q[s, choice] = Q[s, choice] + alphaRL[r] * rpe

            # update WM
            WM[s, choice] = WM[s, choice] + alphaWM[r] * (r - WM[s, choice])

            # sticky
            side = np.zeros(nA)
            side[choice] = 1

            # policy compression
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

            # log-likelihood of next trial
            lt = np.log(b[int(choices[k]) - 1])

            # increment
            llh = llh + lt
            ls[k + 1] = lt

    # return -llh for minimization
    return -llh
