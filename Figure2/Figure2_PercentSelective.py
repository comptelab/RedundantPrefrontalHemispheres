import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import pickle
from scipy.stats import *
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, StratifiedKFold
from circ_stats import *
import scipy.stats as sts
import matplotlib.pylab as pylab
from sklearn.model_selection import train_test_split
from sklearn.model_selection import LeaveOneOut
from filepaths import *


def Wimmer_FR(df = pd.DataFrame(), firing_type='n_delayFull'):
    df_grouped = df.groupby(['targ_angle'])
    # baseRF=np.mean(df[firing_type]) # mean FR for each neuron across targets
    targets = list(df_grouped.groups.keys())
    num_neurons = len(df[firing_type].values[0])
    receptive_fields = np.empty((len(targets), num_neurons), dtype=complex) * np.nan
    mean_count = np.empty((len(targets), num_neurons)) * np.nan
    for t, targ in enumerate(targets):  # for each target orientation
        # mean firing rate of each neuron for each target group, shape= (targets x neurons)
        receptive_fields[t, :] = np.mean(df_grouped.get_group(targ)[firing_type], axis=0)*np.exp(1j*targ)  # -baseRF)
        mean_count[t,:] = np.mean(df_grouped.get_group(targ)[firing_type], axis=0)

    rf_Wimmer = np.sum(receptive_fields, axis=0) / np.sum(mean_count, axis=0)
    theta_Wimmer = np.angle(rf_Wimmer)
    strength_Wimmer = np.abs(rf_Wimmer)
    return rf_Wimmer, theta_Wimmer, strength_Wimmer

def mov_avg(spike_matrix=[], w1=50, w2=250):
    m_err = []
    std_err = []
    x = np.arange(0, spike_matrix.shape[1], w1)
    for i, t in enumerate(x):
        idx = np.where((range(spike_matrix.shape[1])>(t-w2/2)) & (range(spike_matrix.shape[1])<(t+w2/2)))[0]
        # average over neurons, sum over time multiply to get in sp/s
        m_err.append(np.sum(np.mean(spike_matrix[:,idx], axis=0))*1000/w1)
        std_err.append(np.sum(np.mean(spike_matrix[:,idx], axis=0))*1000/w1)

    cut_defined = np.int(np.ceil(w1/w2))
    return (x[cut_defined:-cut_defined]), np.array(m_err)[cut_defined:-cut_defined], np.array(std_err)[cut_defined:-cut_defined]


def eval_receptiveFields(df = pd.DataFrame(), firing_type = 'n_delayFull', BOOTSTRAP=False):

    # COMPUTE TUNING CURVE FOR EACH NEURON

    # group by target position
    df_grouped = df.groupby(['targ_angle'])
    #baseRF=np.mean(df[firing_type']) # mean FR for each neuron across targets
    targets = list(df_grouped.groups.keys())
    num_neurons = len(df[firing_type].values[0])
    receptive_fields = np.empty((len(targets), num_neurons))*np.nan
    for t, targ in enumerate(targets):# for each target orientation
        # mean firing rate of each neuron for each target group, shape= (targets x neurons)
        group_activity = df_grouped.get_group(targ)[firing_type]
        receptive_fields[t, :] = np.mean(group_activity, axis=0)#-baseRF)

    # COMPUTE RECEPTIVE FIELD
    Wimmer_complex, Wimmer_rf, Wimmer_strength = Wimmer_FR(df = df, firing_type = firing_type)

    if BOOTSTRAP:
        df_shuffle = df.copy()
        shuffle_strength, shuffle_complex = [], []
        for _ in range(1000):
            df_shuffle['targ_angle'] = df_shuffle['targ_angle'].sample(frac=1).reset_index(drop=True)
            Shuf_complex, _, Shuf_strength = Wimmer_FR(df=df_shuffle, firing_type=firing_type)
            shuffle_complex.append(Shuf_complex)
            shuffle_strength.append(Shuf_strength)

        return receptive_fields, Wimmer_complex, Wimmer_rf, Wimmer_strength, shuffle_complex, shuffle_strength


    # fig, ax = plt.subplots(np.int(np.ceil(np.sqrt(num_neurons))), np.int(np.ceil(np.sqrt(num_neurons))), sharex=True, figsize=(15,15))
    # axes = np.concatenate(ax)

    return receptive_fields, Wimmer_complex, Wimmer_rf, Wimmer_strength




###############################################################
#                          LOAD DATA                          #
###############################################################

with open(DATA_SMITH + 'df_dat_correct.pickle', 'rb') as handle:
    df_dat_correct = pickle.load(handle)

with open(DATA_SMITH + 'leftRightIdx.pickle', 'rb') as handle:
    leftRightIdx = pickle.load(handle)

left_idx = {'Sa': [[] for i in range(len(leftRightIdx['left']['Sa']))],
            'Pe': [[] for i in range(len(leftRightIdx['left']['Pe']))],
            'Wa': [[] for i in range(len(leftRightIdx['left']['Wa']))]}
right_idx = {'Sa': [[] for i in range(len(leftRightIdx['left']['Sa']))],
             'Pe': [[] for i in range(len(leftRightIdx['left']['Pe']))],
             'Wa': [[] for i in range(len(leftRightIdx['left']['Wa']))]}
for m in ["Sa", "Pe", "Wa"]:
    for n in range(len(leftRightIdx['left'][m])):
        left_idx[m][n] = leftRightIdx['left'][m][n]
        right_idx[m][n] = leftRightIdx['right'][m][n]

df_dat_correct['hemifield'] = ['left' if (df_dat_correct['targ_angle'][i] < np.round(-np.pi / 2, 5)) | \
                                         (df_dat_correct['targ_angle'][i] > np.round(np.pi / 2, 5)) \
                                   else 'right' if (df_dat_correct['targ_angle'][i] > np.round(-np.pi / 2, 5)) & \
                                                   (df_dat_correct['targ_angle'][i] < np.round(np.pi / 2, 5)) \
                                  else 'border' \
                               for i in df_dat_correct.index]



monkeys = ['Sa', 'Pe', 'Wa']
color = ['darkblue', 'darkred']
label=['left', 'right']
df_all = pd.DataFrame()
for mono in monkeys:
    for sess in np.unique(df_dat_correct.loc[df_dat_correct['monkey']==mono].session):
        print(mono+str(sess))

        df_mono = df_dat_correct.loc[(df_dat_correct['monkey'] == mono) & (df_dat_correct['session'] == sess)].copy().reset_index(drop=True)
        df_mono['targ_angle'] = np.round(df_mono.targ_angle, 3)

        targets = np.unique(df_mono.targ_angle)

        num_neurons = len(left_idx[mono][sess][0])
        hemi = ['left' if left_idx[mono][sess][0][i] == 1 else "right" for i in range(num_neurons)]

        # PARAMETERS
        bins, steps = 200, 10
        min_stimstart = np.min(df_mono.targ_on)
        min_delay_len = int(np.min(df_mono.delay) + np.min(df_mono.targ_off - df_mono.targ_on))
        backdelay = 500

        # HEMISPHERE DEF.
        left = np.where(left_idx[mono][sess][0])[0]
        right = np.where(right_idx[mono][sess][0])[0]

        # delay average
        min_resp2end = int(np.min(df_mono.trial_end - df_mono.saccade))

        # moving average of spike counts from start to delay
        smoothed_spiking_del = np.squeeze([[np.sum(df_mono.loc[trial, 'sp_train'][:, t:t + bins], axis=1) * 1000 / bins \
                                            for t in np.arange(df_mono.loc[trial, 'targ_on'] - min_stimstart,\
                                                               df_mono.loc[
                                                                   trial, 'targ_on'] + min_delay_len - backdelay - bins,
                                                               steps)] \
                                           for trial in df_mono.index]).transpose(1, 0, 2)


        # COMPUTE BOOTSTRAPPED SELECTIVITY
        # only determine memory tuning curves on average delay firing
        delay_spiking = np.array([np.sum(
            df_mono.loc[n, 'sp_train'].toarray()[:, int(df_mono['targ_off'][n]):int(df_mono['go_cue'][n])], axis=1) * \
                                  1000 / (df_mono['go_cue'][n] - df_mono['targ_off'][n]) for n in df_mono.index])
        df_mono['n_delayFull'] = list(delay_spiking)  # all neurons during time of cue

        # compute RF on average delay, bootstrap through shuffling
        receptive_fields, Wimmer_complex, Wimmer_rf, Wimmer_strength,\
            Shuffle_complex, Shuffle_strength = eval_receptiveFields(df=df_mono,\
                                                                     firing_type='n_delayFull', BOOTSTRAP=True)

        # bootstrap RF strength to shuffle strength for selectivity
        Shuffle_strength = np.array(Shuffle_strength)
        n_shuffles = Shuffle_strength.shape[0]

        pvals = np.array([
            (np.sum(Shuffle_strength[:, i] >= Wimmer_strength[i]) + 1) / (n_shuffles + 1)
            for i in range(len(Wimmer_strength))
        ])

        print(str(len(np.where(np.array(pvals) < 0.05)[0])) + ' out of ' + str(
            len(pvals)) + ' selective neurons.')
        print(str(np.round(len(np.where(np.array(pvals) < 0.05)[0]) / len(pvals) * 100, 2)) + '% selective neurons')


        out = {'RF_complex': Wimmer_complex, 'RF_strength': Wimmer_strength,\
               'Shuffle_complex': list(np.array(Shuffle_complex).T), 'Shuffle_strength': list(np.array(Shuffle_strength).T),\
             'pvalue_selectivity':pvals, \
               'monkey':[mono for i in range(len(pvals))], 'session':[sess for i in range(len(pvals))],\
               'session_continuous': [mono+str(sess) for i in range(len(pvals))],\
               'num_timepoints': [500 / steps for i in range(len(pvals))]}

        df_out = pd.DataFrame(out)
        df_all = df_all.append(df_out)

        df_all.to_pickle('/home/melanie/Desktop/PhD/2_Smith/Results/FiringRates/NeuralSelectivity.pickle')