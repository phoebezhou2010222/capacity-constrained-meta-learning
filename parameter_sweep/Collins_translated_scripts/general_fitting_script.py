# Python Translation of original matlab script using ChatGPT

# Anne Collins, UC Berkeley
# annecollins@berkeley.edu, 2024
# Code for paper "RL or not RL? Parsing the processes that support human
# reward-based learning."

# This script performs model fitting and defines the models.
# This script is ***slow*** to run (if running all models and data sets
# without parallelization, it will take multiple days).
# run in demo mode for a demo.

# this script generated all the Fits in the Fits/ folder.

from copy import deepcopy
from pathlib import Path
import numpy as np
from scipy.io import loadmat, savemat
from scipy.optimize import minimize
from Models.RLWM_llh import RLWM_llh
from Models.RLWM_R1_llh import RLWM_R1_llh
from Models.RLWM_R2_llh import RLWM_R2_llh
from Models.RLWM_R3_llh import RLWM_R3_llh
from Models.RLWMH_llh import RLWMH_llh

# %% set up a demo
# runs 2 subjects on 2 models with fewer iterations
# turn to false to run the whole thing.
Demo = True

# define data-sets
Datasets = [1, 2, 3, 5, 12, 16]
Dataset = Datasets[0]
WL = 0


def define_models():
    # Ms[m-1] corresponds to MATLAB Ms{m}; sparse cells remain empty.
    Ms = [np.array([]) for _ in range(110)]
    curr_model = {}
    curr_model['name'] = 'RLWM'
    curr_model['pnames'] = ['alphaRL','stick','rho','forget','epsilon','biasWM','biasRL','r0','K']
    curr_model['pfixed'] = np.array([0, 0, 0, 0, 0, 1, 1, 1, 0])
    curr_model['fixedvalue'] = np.array([np.nan, np.nan, np.nan, np.nan, np.nan, 0, 0, 0, np.nan])
    curr_model['thetamapping'] = np.array([1, 2, 3, 4, 5, np.nan, np.nan, np.nan, 6])
    curr_model['pMin'] = np.array([0, -1, 0, 0, 0])
    curr_model['pMax'] = np.array([1, 1, 1, 1, 1])
    curr_model['interact'] = 0
    curr_model['value'] = 1
    curr_model['ID'] = 'WM0 RL0 0'
    Ms[0] = curr_model

    curr_model = {}
    curr_model['name'] = 'RLWM'
    curr_model['pnames'] = ['alphaRL','stick','rho','forget','epsilon','biasWM','biasRL','r0','K']
    curr_model['pfixed'] = np.array([0, 0, 0, 0, 0, 0, 1, 1, 0])
    curr_model['fixedvalue'] = np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0, 0, np.nan])
    curr_model['thetamapping'] = np.array([1, 2, 3, 4, 5, 6, np.nan, np.nan, 7])
    curr_model['pMin'] = np.array([0, -1, 0, 0, 0, 0])
    curr_model['pMax'] = np.array([1, 1, 1, 1, 1, 1])
    curr_model['interact'] = 0
    curr_model['value'] = 1
    curr_model['ID'] = 'WM RL0 0'
    Ms[1] = curr_model

    curr_model = {}
    curr_model['name'] = 'RLWM'
    curr_model['pnames'] = ['alphaRL','stick','rho','forget','epsilon','biasWM','biasRL','r0','K']
    curr_model['pfixed'] = np.array([0, 0, 0, 0, 0, 0, 1, 1, 0])
    curr_model['fixedvalue'] = np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 1, 0, np.nan])
    curr_model['thetamapping'] = np.array([1, 2, 3, 4, 5, 6, np.nan, np.nan, 7])
    curr_model['pMin'] = np.array([0, -1, 0, 0, 0, 0])
    curr_model['pMax'] = np.array([1, 1, 1, 1, 1, 1])
    curr_model['interact'] = 0
    curr_model['value'] = 1
    curr_model['ID'] = 'WM RL1 0'
    Ms[2] = curr_model

    curr_model = {}
    curr_model['name'] = 'RLWM'
    curr_model['pnames'] = ['alphaRL','stick','rho','forget','epsilon','biasWM','biasRL','r0','K']
    curr_model['pfixed'] = np.array([0, 0, 0, 0, 0, 1, 0, 1, 0])
    curr_model['fixedvalue'] = np.array([np.nan, np.nan, np.nan, np.nan, np.nan, 0, np.nan, 0, np.nan])
    curr_model['thetamapping'] = np.array([1, 2, 3, 4, 5, np.nan, 6, np.nan, 7])
    curr_model['pMin'] = np.array([0, -1, 0, 0, 0, 0])
    curr_model['pMax'] = np.array([1, 1, 1, 1, 1, 1])
    curr_model['interact'] = 0
    curr_model['value'] = 1
    curr_model['ID'] = 'WM0 RL 0'
    Ms[3] = curr_model

    curr_model = {}
    curr_model['name'] = 'RLWM'
    curr_model['pnames'] = ['alphaRL','stick','rho','forget','epsilon','biasWM','biasRL','r0','K']
    curr_model['pfixed'] = np.array([0, 0, 0, 0, 0, 1, 0, 1, 0])
    curr_model['fixedvalue'] = np.array([np.nan, np.nan, np.nan, np.nan, np.nan, 1, np.nan, 0, np.nan])
    curr_model['thetamapping'] = np.array([1, 2, 3, 4, 5, np.nan, 6, np.nan, 7])
    curr_model['pMin'] = np.array([0, -1, 0, 0, 0, 0])
    curr_model['pMax'] = np.array([1, 1, 1, 1, 1, 1])
    curr_model['interact'] = 0
    curr_model['value'] = 1
    curr_model['ID'] = 'WM1 RL 0'
    Ms[4] = curr_model

    curr_model = {}
    curr_model['name'] = 'RLWM'
    curr_model['pnames'] = ['alphaRL','stick','rho','forget','epsilon','biasWM','biasRL','r0','K']
    curr_model['pfixed'] = np.array([0, 0, 0, 0, 0, 0, 0, 1, 0])
    curr_model['fixedvalue'] = np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0, np.nan])
    curr_model['thetamapping'] = np.array([1, 2, 3, 4, 5, 6, 6, np.nan, 7])
    curr_model['pMin'] = np.array([0, -1, 0, 0, 0, 0])
    curr_model['pMax'] = np.array([1, 1, 1, 1, 1, 1])
    curr_model['interact'] = 0
    curr_model['value'] = 1
    curr_model['ID'] = 'WM=RL 0'
    Ms[5] = curr_model

    curr_model = {}
    curr_model['name'] = 'RLWM'
    curr_model['pnames'] = ['alphaRL','stick','rho','forget','epsilon','biasWM','biasRL','r0','K']
    curr_model['pfixed'] = np.array([0, 0, 0, 0, 0, 0, 0, 1, 0])
    curr_model['fixedvalue'] = np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0, np.nan])
    curr_model['thetamapping'] = np.array([1, 2, 3, 4, 5, 6, 7, np.nan, 8])
    curr_model['pMin'] = np.array([0, -1, 0, 0, 0, 0, 1])
    curr_model['pMax'] = np.array([1, 1, 1, 1, 1, 1, 1])
    curr_model['interact'] = 0
    curr_model['value'] = 1
    curr_model['ID'] = 'WM RL 0'
    Ms[6] = curr_model

    curr_model = {}
    curr_model['name'] = 'RLWM'
    curr_model['pnames'] = ['alphaRL','stick','rho','forget','epsilon','biasWM','biasRL','r0','K']
    curr_model['pfixed'] = np.array([0, 0, 0, 0, 0, 1, 1, 1, 0])
    curr_model['fixedvalue'] = np.array([np.nan, np.nan, np.nan, np.nan, np.nan, 1, 0, 0, np.nan])
    curr_model['thetamapping'] = np.array([1, 2, 3, 4, 5, np.nan, np.nan, np.nan, 6])
    curr_model['pMin'] = np.array([0, -1, 0, 0, 0])
    curr_model['pMax'] = np.array([1, 1, 1, 1, 1])
    curr_model['interact'] = 0
    curr_model['value'] = 1
    curr_model['ID'] = 'WM1 RL0 0'
    Ms[7] = curr_model

    curr_model = {}
    curr_model['name'] = 'RLWM'
    curr_model['pnames'] = ['alphaRL','stick','rho','forget','epsilon','biasWM','biasRL','r0','K']
    curr_model['pfixed'] = np.array([0, 0, 0, 0, 0, 1, 1, 1, 0])
    curr_model['fixedvalue'] = np.array([np.nan, np.nan, np.nan, np.nan, np.nan, 1, 1, 0, np.nan])
    curr_model['thetamapping'] = np.array([1, 2, 3, 4, 5, np.nan, np.nan, np.nan, 6])
    curr_model['pMin'] = np.array([0, -1, 0, 0, 0])
    curr_model['pMax'] = np.array([1, 1, 1, 1, 1])
    curr_model['interact'] = 0
    curr_model['value'] = 1
    curr_model['ID'] = 'WM1 RL1 0'
    Ms[8] = curr_model

    curr_model = {}
    curr_model['name'] = 'RLWM'
    curr_model['pnames'] = ['alphaRL','stick','rho','forget','epsilon','biasWM','biasRL','r0','K']
    curr_model['pfixed'] = np.array([0, 0, 0, 0, 0, 1, 1, 1, 0])
    curr_model['fixedvalue'] = np.array([np.nan, np.nan, np.nan, np.nan, np.nan, 0, 1, 0, np.nan])
    curr_model['thetamapping'] = np.array([1, 2, 3, 4, 5, np.nan, np.nan, np.nan, 6])
    curr_model['pMin'] = np.array([0, -1, 0, 0, 0])
    curr_model['pMax'] = np.array([1, 1, 1, 1, 1])
    curr_model['interact'] = 0
    curr_model['value'] = 1
    curr_model['ID'] = 'WM0 RL1 0'
    Ms[9] = curr_model

    curr_model = {}
    curr_model['name'] = 'RLWM'
    curr_model['pnames'] = ['alphaRL','stick','rho','forget','epsilon','biasWM','biasRL','r0','K']
    curr_model['pfixed'] = np.array([0, 0, 0, 0, 0, 1, 1, 0, 0])
    curr_model['fixedvalue'] = np.array([np.nan, np.nan, np.nan, np.nan, np.nan, 0, 1, np.nan, np.nan])
    curr_model['thetamapping'] = np.array([1, 2, 3, 4, 5, np.nan, np.nan, 6, 7])
    curr_model['pMin'] = np.array([0, -1, 0, 0, 0, 0])
    curr_model['pMax'] = np.array([1, 1, 1, 1, 1, 1])
    curr_model['interact'] = 0
    curr_model['value'] = 1
    curr_model['ID'] = 'WM0 RL1 r0'
    Ms[31] = curr_model

    curr_model = {}
    curr_model['name'] = 'RLWM'
    curr_model['pnames'] = ['alphaRL','stick','rho','forget','epsilon','biasWM','biasRL','r0','K']
    curr_model['pfixed'] = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0])
    curr_model['fixedvalue'] = np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 1, np.nan, np.nan])
    curr_model['thetamapping'] = np.array([1, 2, 3, 4, 5, 6, np.nan, 7, 8])
    curr_model['pMin'] = np.array([0, -1, 0, 0, 0, 0, 0, 0])
    curr_model['pMax'] = np.array([1, 1, 1, 1, 1, 1, 1, 1])
    curr_model['interact'] = 0
    curr_model['value'] = 1
    curr_model['ID'] = 'WM RL1 r0'
    Ms[34] = curr_model

    curr_model = {}
    curr_model['name'] = 'RLWM'
    curr_model['pnames'] = ['alphaRL','stick','rho','forget','epsilon','biasWM','biasRL','r0','K']
    curr_model['pfixed'] = np.array([0, 0, 0, 0, 0, 1, 1, 0, 0])
    curr_model['fixedvalue'] = np.array([np.nan, np.nan, np.nan, np.nan, np.nan, 1, 1, np.nan, np.nan])
    curr_model['thetamapping'] = np.array([1, 2, 3, 4, 5, np.nan, np.nan, 6, 7])
    curr_model['pMin'] = np.array([0, -1, 0, 0, 0, 0])
    curr_model['pMax'] = np.array([1, 1, 1, 1, 1, 1])
    curr_model['interact'] = 0
    curr_model['value'] = 1
    curr_model['ID'] = 'WM1 RL1 r0'
    Ms[37] = curr_model

    # 11-20: RL-WM interaction (the source leaves several cells empty).
    for i in [3, 4, 5, 6, 7, 9, 10]:
        curr_model = deepcopy(Ms[i-1])
        curr_model['interact'] = 1
        curr_model['ID'] = 'int ' + curr_model['ID']
        Ms[10+i-1] = curr_model
    # 21-30: WMH.
    for i in [3, 4, 5, 6, 7, 9, 10]:
        curr_model = deepcopy(Ms[i-1])
        curr_model['value'] = 0
        curr_model['ID'] = 'freq ' + curr_model['ID']
        Ms[i+20-1] = curr_model
    # 41-50: policy compression; retain the source mapping order.
    for i in range(1, 11):
        curr_model = deepcopy(Ms[i-1])
        curr_model['name'] = 'RLWM'
        curr_model['pnames'] = ['alphaRL','stick','rho','forget','epsilon','biasWM','biasRL','r0','chunk','K']
        curr_model['pfixed'] = np.r_[curr_model['pfixed'], 0]
        curr_model['fixedvalue'] = np.r_[curr_model['fixedvalue'], np.nan]
        curr_model['thetamapping'] = np.r_[curr_model['thetamapping'], 2+len(curr_model['pMin'])]
        curr_model['pMin'] = np.r_[curr_model['pMin'], 0]
        curr_model['pMax'] = np.r_[curr_model['pMax'], 1]
        curr_model['ID'] += '-PC'
        Ms[40+i-1] = curr_model
    # 51-60: WM learning rate.
    for i in range(1, 11):
        curr_model = deepcopy(Ms[i-1])
        curr_model['name'] = 'RLWM_R1'
        curr_model['pnames'] = ['alphaRL','stick','rho','forget','epsilon','biasWM','biasRL','r0','chunk','alphaWM1','forgetRL','K']
        curr_model['pfixed'] = np.r_[curr_model['pfixed'][:8], 1, 0, 1, 0]
        curr_model['fixedvalue'] = np.r_[curr_model['fixedvalue'][:8], 0, np.nan, 0]
        curr_model['pMin'] = np.r_[curr_model['pMin'], 0]
        curr_model['pMax'] = np.r_[curr_model['pMax'], 1]
        curr_model['thetamapping'] = np.r_[curr_model['thetamapping'][:-1], np.nan, len(curr_model['pMin']), np.nan, 1+len(curr_model['pMin'])]
        curr_model['ID'] += '-WMalpha'
        Ms[50+i-1] = curr_model
    # 61-70: RL forgetting. The source has only 11 pfixed entries here.
    for i in range(1, 11):
        curr_model = deepcopy(Ms[i-1])
        curr_model['name'] = 'RLWM_R1'
        curr_model['pnames'] = ['alphaRL','stick','rho','forget','epsilon','biasWM','biasRL','r0','chunk','alphaWM1','forgetRL','K']
        curr_model['pfixed'] = np.r_[curr_model['pfixed'][:8], 1, 1, 0]
        curr_model['fixedvalue'] = np.r_[curr_model['fixedvalue'][:8], 0, 0, np.nan]
        curr_model['pMin'] = np.r_[curr_model['pMin'], 0]
        curr_model['pMax'] = np.r_[curr_model['pMax'], 1]
        curr_model['thetamapping'] = np.r_[curr_model['thetamapping'][:-1], np.nan, len(curr_model['pMin']), np.nan, 1+len(curr_model['pMin'])]
        curr_model['ID'] += '-RLforget'
        Ms[60+i-1] = curr_model
    # 71-80: RL-H.
    for i in [3, 4, 5, 6, 7, 9, 10]:
        curr_model = deepcopy(Ms[i-1])
        curr_model['name'] = 'RLWM_R1'
        curr_model['pnames'] = ['alphaRL','stick','rho','forget','epsilon','biasWM','biasRL','r0','chunk','alphaWM1','forgetRL','K']
        curr_model['pfixed'] = np.r_[curr_model['pfixed'][:7], 1, 1, 0, 1]
        curr_model['fixedvalue'] = np.r_[curr_model['fixedvalue'][:7], 1, 0, np.nan, 0]
        curr_model['pMin'] = np.r_[curr_model['pMin'], 0]
        curr_model['pMax'] = np.r_[curr_model['pMax'], 1]
        curr_model['thetamapping'] = np.r_[curr_model['thetamapping'][:-1], np.nan, len(curr_model['pMin']), np.nan, 1+len(curr_model['pMin'])]
        curr_model['ID'] += '-RLH'
        Ms[70+i-1] = curr_model
    # 81: WMRLH, shared biases.
    curr_model = {}
    curr_model['name'] = 'RLWMH'
    curr_model['pnames'] = ['alphaRL','stick','rhoWM','forget','epsilon','biasWM','biasRL','r0','chunk','rhoH','alphaH','biasH','K']
    curr_model['pfixed'] = np.array([0,0,0,0,0,0,0,1,1,0,0,0,0])
    curr_model['fixedvalue'] = np.r_[np.full(7, np.nan), 1, 0, np.full(4, np.nan)]
    curr_model['pMin'] = np.r_[0, -1, np.zeros(6)]
    curr_model['pMax'] = np.ones(8)
    curr_model['thetamapping'] = np.array([1,2,3,4,5,6,6,np.nan,np.nan,7,8,6,np.nan])
    curr_model['interact'] = 0
    curr_model['value'] = 1
    curr_model['ID'] = 'RLWMH-shared bias'
    Ms[80] = curr_model
    # 82: no WM/H bias; free RL bias.
    curr_model = deepcopy(Ms[80])
    curr_model['pfixed'][[5,11]] = 1
    curr_model['fixedvalue'][[5,11]] = 1
    curr_model['thetamapping'] = np.array([1,2,3,4,5,np.nan,6,np.nan,np.nan,7,8,np.nan,np.nan])
    curr_model['ID'] = 'RLWMH-freeRL bias'
    Ms[81] = curr_model
    # 83: RLf+H; retain length-13 pfixed and NaN mappings from source.
    curr_model = {}
    curr_model['name'] = 'RLWMH'
    curr_model['pnames'] = ['alphaRL','stick','rhoWM','forget','epsilon','biasWM','biasRL','r0','chunk','rhoH','alphaH','biasH','forgetRL','K']
    curr_model['pfixed'] = np.array([0,0,1,1,0,1,0,1,1,0,0,1,0])
    curr_model['fixedvalue'] = np.full(13, np.nan)
    curr_model['fixedvalue'][[2,3,5]] = [0,0,0]
    curr_model['fixedvalue'][[7,8]] = [1,0]
    curr_model['fixedvalue'][11] = 1
    curr_model['pMin'] = np.r_[0, -1, np.zeros(5)]
    curr_model['pMax'] = np.ones(7)
    curr_model['thetamapping'] = np.array([1,2,np.nan,np.nan,3,np.nan,4,np.nan,np.nan,5,6,np.nan,7])
    curr_model['interact'] = 0
    curr_model['value'] = 1
    curr_model['ID'] = 'RLfH'
    Ms[82] = curr_model
    # 91-100: biased lapse policy.
    for i in range(1, 11):
        curr_model = deepcopy(Ms[i-1])
        curr_model['name'] = 'RLWM_R2'
        curr_model['pnames'] = ['alphaRL','stick','rho','forget','epsilon','biasWM','biasRL','A1','A2-A3','K']
        curr_model['pfixed'] = np.r_[curr_model['pfixed'][:7], 0,0,0]
        curr_model['fixedvalue'] = np.r_[curr_model['fixedvalue'][:7], np.nan,np.nan,np.nan]
        curr_model['pMin'] = np.r_[curr_model['pMin'], 0,0]
        curr_model['pMax'] = np.r_[curr_model['pMax'], 1,1]
        curr_model['thetamapping'] = np.r_[curr_model['thetamapping'][:-2], len(curr_model['pMin'])-1, len(curr_model['pMin']), np.nan]
        curr_model['ID'] += '-bias'
        Ms[90+i-1] = curr_model
    # 101-110: no WM over capacity.
    for i in range(1, 11):
        curr_model = deepcopy(Ms[i-1])
        curr_model['name'] = 'RLWM_R3'
        curr_model['ID'] += '- WMdrop'
        Ms[100+i-1] = curr_model
    return Ms


def main(Demo=Demo, Dataset=Dataset, models=None, niter=None, seed=None):
    root = Path(__file__).resolve().parent

    # %% load data set and subject list
    expe_data = loadmat(root / f'DataSets/Expe{Dataset}.mat', simplify_cells=True)['expe_data']
    all_subjects = np.unique(expe_data[:, 0])

    # already fitted some models; load previously fitted models
    previous = loadmat(root / f'NewFits/FitRLWM_dataset{Dataset}.mat', simplify_cells=True)
    All_Params = list(previous['All_Params'])
    All_fits = np.array(previous['All_fits'], copy=True)

    # %% set params
    Ms = define_models()

    # %% models to consider for fitting
    toconsider = list(range(101, 111)) if models is None else list(models)

    # %% set up fitting options
    niter = (2 if Demo else 10) if niter is None else niter
    subjects_list = all_subjects[:2] if Demo else all_subjects
    if Demo:
        toconsider = toconsider[:2]
    if niter < 1:
        raise ValueError('niter must be at least 1')
    if seed is not None:
        np.random.seed(seed)

    # %% fitting
    for m in toconsider:
        fit_model = Ms[m-1]
        if not isinstance(fit_model, dict):
            raise ValueError(f'Model {m} is undefined')
        pmin = np.asarray(fit_model['pMin'], dtype=float)
        pmax = np.asarray(fit_model['pMax'], dtype=float)
        fitmeasures = []
        fitparams = []

        for k, s in enumerate(subjects_list):
            print('Model#, Sub#, out of N subjects', m, k+1, len(subjects_list))
            # stim, action, reward, set size, block, RT
            data = expe_data[expe_data[:, 0] == s][:, [4, 9, 11, 2, 1, 13]]
            sofar = []
            # same starting points reused for every capacity value, as in MATLAB
            pars = pmin + np.random.random((niter, len(pmin))) * (pmax-pmin)

            for K in range(2, 6):
                myfitfun = lambda p: globals()[fit_model['name'] + '_llh'](p, K, data, fit_model)
                for it in range(niter):
                    par = pars[it, :]
                    output = minimize(myfitfun, par, method='SLSQP', bounds=list(zip(pmin, pmax)),
                                      options={'disp': False, 'maxiter': 1000, 'ftol': 1e-6})
                    sofar.append(np.r_[output.x, K, output.fun])
            sofar = np.asarray(sofar)

            # %% store information
            i = np.argmin(sofar[:, -1])
            llh = sofar[i, -1]
            param = sofar[i, :-1]
            ntrials = len(data)
            # Reproduce original MATLAB AIC/BIC expressions, including their extra +1.
            AIC = 2*llh + 2*(len(param)+1)
            BIC = 2*llh + np.log(ntrials)*(len(param)+1)
            AIC0 = -2*np.log(1/3)*ntrials
            psr2 = (AIC0-AIC)/AIC0
            fitmeasures.append([s, -llh, AIC, BIC, psr2, AIC0])
            fitparams.append(param)

        fitparams = np.asarray(fitparams)
        fitmeasures = np.asarray(fitmeasures)

        # Preserve all original participants' rows. In Demo, update only the fitted rows.
        # Never overwrite the supplied fitted results in demo mode.
        if All_fits.shape[0] != len(all_subjects) or All_fits.shape[1] != 6:
            raise ValueError('Saved All_fits dimensions do not match current dataset')
        if All_Params[m-1] is None or np.asarray(All_Params[m-1]).size == 0:
            full_params = np.full((len(all_subjects), fitparams.shape[1]), np.nan)
        else:
            existing = np.asarray(All_Params[m-1])
            full_params = existing.copy() if existing.shape == (len(all_subjects), fitparams.shape[1]) else np.full((len(all_subjects), fitparams.shape[1]), np.nan)
        for row, s in enumerate(subjects_list):
            pos = np.flatnonzero(all_subjects == s)
            if len(pos) != 1:
                raise ValueError(f'Unexpected subject index for {s}')
            full_params[pos[0], :] = fitparams[row]
            All_fits[pos[0], :, m-1] = fitmeasures[row]
        All_Params[m-1] = full_params

        cells = np.empty((1, len(Ms)), dtype=object)
        params_cells = np.empty((1, len(All_Params)), dtype=object)
        for i, model in enumerate(Ms):
            cells[0, i] = model
        for i, params in enumerate(All_Params):
            params_cells[0, i] = params
        output_dir = root / ('DemoFits' if Demo else 'NewFits')
        output_dir.mkdir(exist_ok=True)
        savemat(output_dir / f'FitRLWM_dataset{Dataset}.mat',
                {'Ms': cells, 'All_Params': params_cells, 'All_fits': All_fits,
                 'subjects_list': np.asarray(all_subjects)[None, :]}, oned_as='row')
    return Ms, All_Params, All_fits, subjects_list


if __name__ == '__main__':
    main()