#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 14:54:17 2022

@author: melanie
"""

import numpy as np
import pandas as pd
import json
from joblib import Parallel, delayed
import multiprocessing
import statsmodels.api as sm
import pickle
import matplotlib.pyplot as plt
import h5py
import seaborn as sns
from scipy.stats import *
from scipy.signal import correlate
from matplotlib.ticker import FormatStrFormatter
# from statsmodels.stats.weightstats.CompareMeans import ztest_ind
from scipy.sparse import csr_matrix
from sklearn.model_selection import KFold, StratifiedKFold
from random import randint
from circ_stats import *
from statsmodels.tools.tools import add_constant
from sklearn.model_selection import train_test_split
from filepaths import *

# from sklearn.linear_model import PoissonRegressor

def decode_singletrial(dataframe, borders_full, neuron_type, mode):
    # targets

    intercept = False

    repeats = 20
    random_states = np.random.choice(100000, size=repeats)


    y = np.array([complex(dataframe['target_prev_xy'][i][0], dataframe['target_prev_xy'][i][1]) for i in dataframe.index])

    hemifield = dataframe.hemifield.values
    ipsicontra = ['ipsi' if hemifield[i] == neuron_type[:-1]\
                      else 'contra' if (hemifield[i] != neuron_type[:-1]) & (hemifield[i] != 'border') \
        else 'border' for i in range(len(hemifield))]
    # divide by mean per neuron to mean-center data on positive interval (mean across trials, times)

    # spiking = np.array([np.array(dataframe['bin_sp'][i]) for i in dataframe.index])
    # TODO! ZSCORE DATA
    # divide by mean per neuron to mean-center data on positive interval (mean across trials, times)
    spiking = np.array([np.array(dataframe[mode + '_sp_' + neuron_type][i]) for i in dataframe.index])
    mean_spikingpertime = np.mean(spiking, axis=0)
    std_spikingpertime = np.std(spiking, axis=0)
    std_spikingpertime[std_spikingpertime < 0.1] = 0.1
    # z-score to FR of each neuron @ each time (or divide by mean)
    # X_shuffle = spiking/mean_spiking
    X_shuffle = (spiking - mean_spikingpertime) / std_spikingpertime

    spiking_delay = np.array([np.array(dataframe[mode + '_delaysp_' + neuron_type][i]) for i in dataframe.index])
    mean_spikingdelay = np.mean(spiking_delay, axis=0)
    std_spikingdelay = np.std(spiking_delay, axis=0)
    std_spikingdelay[std_spikingdelay < 0.1] = 0.1
    # z-score to FR of each neuron @ each time (or divide by mean)
    X_delay = (spiking_delay - mean_spikingdelay) / std_spikingdelay

    # define number of neurons used (includes intercept
    num_neurons = X_shuffle.shape[2]
    if intercept == True:
        num_neurons = X_shuffle.shape[2] + 1
        X_delay = np.array([np.append(1, X_delay[i]) for i in range(len(X_delay))])  # add intercept

    #########################################################################################
    #                     GENERAL : calculate predictions for each timestep                     #
    #########################################################################################
    len_timing = borders_full[16]

    prediction = np.zeros((repeats, len(y), len_timing)).astype(complex)
    shuffleprediction = np.zeros((repeats, len(y), len_timing)).astype(complex)
    real = np.zeros((len(y))).astype(complex)
    weights = np.zeros((repeats, len(y), num_neurons)).astype(complex)

    # LOO crossvalidation to get single trial predictions

    for rep in range(repeats):
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_states[rep])
        for train_idx, test_idx in kf.split(X_delay, np.round(np.angle(y), 3).astype(str)):
            X_train = X_delay[train_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # train decoder
            weights_delay = np.linalg.pinv(X_train.T.dot(X_train)).dot(X_train.T).dot(y_train)
            weights[rep, test_idx, :] = weights_delay

            num_targs = len(np.unique(y_train))

            for t_id, delta_t_train in enumerate(range(len_timing)):  # train / test decoder in each time step
                # create training dataset: columns=neurons, rows=trials for previous/current trials
                X_test = X_shuffle[test_idx, delta_t_train]

                # keep real decoder predictions for correlations
                prediction[rep, test_idx, delta_t_train] = X_test.dot(weights_delay).astype(complex)
                real[test_idx] = y_test.astype(complex)

                # SHUFFLE
                # shuffle neural activity within targets with the same stimulus
                import copy
                X_test_shuffle = copy.deepcopy(X_test)
                # base shuffle on trials with same previous trial
                y_tosplit = copy.deepcopy(y_test)
                # shuffle those trials with the same cue (to keep autocorrelation)
                shuffle_prev = []
                for crossval_shuffle in range(50):
                    for targ_shuffle in np.unique(y_tosplit):
                        index = np.where(y_tosplit == targ_shuffle)[0]
                        X_test_shuffle[index, :] = X_test_shuffle[np.random.permutation(index), :]
                    shuffle_prev.append(X_test_shuffle.dot(weights_delay).astype(complex))

                shuffleprediction[rep, test_idx, delta_t_train] = np.mean(shuffle_prev, axis=0)

                assert np.all(np.histogram(X_test[:, 0])[0] == np.histogram(X_test_shuffle[:, 0])[0]), \
                    "Shuffle along wrong axis: Histograms for individual neurons must stay the same."
                assert np.any(X_test != X_test_shuffle), "No shuffled trials: Test shuffle."
    # average out repeats (predictions are complex numbers, use regular mean)
    prediction = np.mean(prediction, axis=0)
    std_prediction = np.std(prediction, axis=0)
    shuffleprediction = np.mean(shuffleprediction, axis=0)
    weights = np.mean(weights, axis=0)

    return list(prediction), list(std_prediction), list(shuffleprediction), list(real), list(weights), \
        ipsicontra, [num_targs for i in range(len(y))], [borders_full for i in range(len(y))],\
        [random_states for i in range(len(y))]

# plt.plot(np.mean([np.abs(circdist(np.angle(prediction[i]), np.angle(real[i]))) for i in range(len(real))], axis=0))
# plt.plot(np.mean([np.abs(circdist(np.angle(prediction[i]), np.angle(shuffleprediction[i]))) for i in range(len(real))], axis=0))
##################################################################################################
#                                               LOAD DATA                                        #
##################################################################################################

with open(DATA_SMITH + 'df_dat_correct_Sa0.pickle', 'rb') as handle:
    df_dat_corr = pickle.load(handle)

with open(DATA_SMITH + 'leftRightIdx_Sa0.pickle', 'rb') as handle:
    leftRightIdx = pickle.load(handle)

monkeys=['Sa']#['Sa','Pe', 'Wa']

left_idx = {m: [[] for i in range(len(leftRightIdx['left']['Sa']))] for m in monkeys}
right_idx = {m: [[] for i in range(len(leftRightIdx['left']['Sa']))] for m in monkeys}
for m in monkeys:
    for n in range(len(leftRightIdx['left'][m])):
        left_idx[m][n] = leftRightIdx['left'][m][n]
        right_idx[m][n] = leftRightIdx['right'][m][n]

# solely determine hemifield based on target
df_dat_corr['hemifield'] = ['left' if (df_dat_corr['targ_angle'][i] < np.round(-np.pi / 2, 5)) | \
                                      (df_dat_corr['targ_angle'][i] > np.round(np.pi / 2, 5)) \
                                else 'right' if (df_dat_corr['targ_angle'][i] > np.round(-np.pi / 2, 5)) & \
                                                (df_dat_corr['targ_angle'][i] < np.round(np.pi / 2, 5)) else 'border' \
                            for i in df_dat_corr.index]

##################################################################################################
#                                      SPLIT  DATA INTO SESSIONS                                 #
##################################################################################################

labels = ['prediction_left1', 'real', 'weights_left1', 'hemifield_left1', \
          'prediction_left2', 'weights_left2', 'hemifield_left2', \
          'prediction_right1', 'weights_right1', 'hemifield_right1', \
          'prediction_right2', 'weights_right2', 'hemifield_right2', \
          'borders_full', 'num_targs']
df_out = pd.DataFrame(columns=labels)
for mono in monkeys:#['Sa', 'Pe', 'Wa']:  # for each monkey #,'Pe','Wa'
    for sess in range(max(df_dat_corr.loc[df_dat_corr.monkey == mono].session) + 1):  # for each session

        # choose session
        df_Sa0_corr = df_dat_corr.loc[
            (df_dat_corr['monkey'] == mono) & (df_dat_corr['session'] == sess)].copy().reset_index(drop=True)

        left = np.where(left_idx[mono][sess] == 1)[1]  #
        right = np.where(right_idx[mono][sess] == 1)[1]

        # make spike trains into csr matrix for each trial
        mat = [csr_matrix(df_Sa0_corr.loc[n, 'sp_train']) for n in df_Sa0_corr['sp_train'].index]
        df_Sa0_corr.insert(5, 'n_mat', mat, True)

        # determine border points between different time periods, until beginning of delay
        bins = 200  # TODO!

        # determine border points INDIVID trials between different time periods, for end of delay
        timings2 = ['start', 'fix', 'targ_on', 'targ_off', 'go_cue', 'saccade', 'reward', 'trial_end']
        t_borders2 = ['start', 'start_front', 'start_back', 'fix', 'targ_on', 'targ_off', 'delay_front_end', \
                      'delay_start', 'delay_end', 'saccade_front', 'saccade', 'saccade_end', 'reward',
                      'trial_front_end', \
                      'end_start', 'trial_end']  # 'go_cue',
        borders = {'start': [], 'start_front': [], 'start_back': [], 'fix': [], 'targ_on': [], 'targ_off': [],
                   'delay_front_end': [], \
                   'delay_start': [], 'delay_end': [], 'saccade_front': [], 'saccade': [], 'saccade_end': [], \
                   'reward': [], 'trial_front_end': [], 'end_start': [], 'trial_end': []}  # 'go_cue': [],
        for i, m in enumerate(borders.keys()):
            if (m == 'start') | (m == 'fix') | (m == 'saccade') | (m == 'reward') | (m == 'trial_end'):
                borders[m] = ((df_Sa0_corr[m].values) / bins).astype(int)
            elif m == 'start_front':
                borders[m] = ((df_Sa0_corr['start'].values) / bins).astype(int) + (
                            (min(df_Sa0_corr['fix'].values - df_Sa0_corr['start'].values)) / bins).astype(int)
            elif m == 'start_back':
                borders[m] = ((df_Sa0_corr['fix'].values) / bins).astype(int) - (
                            (min(df_Sa0_corr['fix'].values - df_Sa0_corr['start'].values)) / bins).astype(int)
            elif (m == 'targ_on'):
                borders[m] = borders['fix'] + int(min(df_Sa0_corr['targ_on'] - df_Sa0_corr['fix']) / bins)
            elif (m == 'targ_off'):
                borders[m] = borders['targ_on'] + int(min(df_Sa0_corr['targ_off'] - df_Sa0_corr['targ_on']) / bins)
            elif (m == 'delay_front_end'):
                borders[m] = np.array(borders['targ_off']) + int(
                    min(df_Sa0_corr['go_cue'] - df_Sa0_corr['targ_off']) / bins)
            elif m == 'delay_start':
                # create shifted "start" of delay
                borders[m] = ((df_Sa0_corr['go_cue'].values) / bins).astype(int) - int(
                    min(df_Sa0_corr['go_cue'] - df_Sa0_corr['targ_off']) / bins)
            elif m == 'delay_end':
                # delay end
                borders[m] = ((df_Sa0_corr['go_cue'].values) / bins).astype(int)
            elif m == 'saccade_front':
                borders[m] = np.array(borders['delay_end']) + int(
                    min(df_Sa0_corr['saccade'] - df_Sa0_corr['go_cue']) / bins)
            elif m == 'saccade_end':
                borders[m] = np.array(borders['saccade']) + int(
                    min(df_Sa0_corr['reward'] - df_Sa0_corr['saccade']) / bins)
            elif m == 'trial_front_end':
                borders[m] = np.array(borders['reward']) + int(
                    min(df_Sa0_corr['trial_end'] - df_Sa0_corr['reward']) / bins)
            elif m == 'end_start':
                # shifted "start" of trial end : complete end of trial - minimum(trial_end-reward)
                borders[m] = [int(df_Sa0_corr.loc[n, 'trial_end'] / bins) - int(
                    min(df_Sa0_corr.loc[:, 'trial_end'] - df_Sa0_corr.loc[:, 'reward']) / bins) for n in
                              df_Sa0_corr.index]  #
            else:
                print('Error')

        # for first cut (different delay lengths)
        time_connections = [0, 2, 3, 4, 5, 7, 8, 10, 12, 14]
        bin_sp_trials = []
        period_spikes = []
        for idx, trial in enumerate(df_Sa0_corr.index):  # for all trials
            binned_spikes = []
            number_bins = []
            number_bins_len = []
            for period in time_connections:  # range(1,len(borders)-1):# from start_back for all time periods until trial_end
                # if period<5:
                number_bins.append(borders[t_borders2[period + 1]][0] - borders[t_borders2[period]][0])
                number_bins_len.append(borders[t_borders2[period + 1]][0])
                for t in range(borders[t_borders2[period + 1]][0] - borders[t_borders2[period]][
                    0]):  # for number of time bins in discrete timings:
                    # sum the matrix of neurons at timings in bin
                    binned_spikes.append(np.sum(df_Sa0_corr.loc[trial, 'n_mat'][:,
                                                borders[t_borders2[period]][idx] * bins + t * bins:
                                                borders[t_borders2[period]][idx] * bins + t * bins + bins].toarray(),
                                                axis=1))
            bin_sp_trials.append(binned_spikes)

        borders_full = [np.sum(abs(np.array(number_bins))[:i]) for i in range(len(number_bins) + 1)]
        borders_full = np.append(borders_full, borders_full + max(borders_full))

        # define delay activity
        bin_sp_delay = [np.sum(df_Sa0_corr.loc[n, 'sp_train'].toarray()[:, int(df_Sa0_corr['targ_off'][n]+100): \
                                                                int(df_Sa0_corr['go_cue'][n])],\
                                                    axis=1)*bins/(int(df_Sa0_corr['go_cue'][n])-int(df_Sa0_corr['targ_off'][n]+100))\
                                              for n in df_Sa0_corr.index]

        ###################################################################################
        #                                  SERIAL DEPENDENCE                              #
        ###################################################################################

        serial = {'target_prev': [], 'target_prev_xy': [], 'response_prev_xy': [], 'hemifield': [], \
                  'bin_sp_prev': [], 'bin_sp_curr': [], 'bin_sp': [], 'bin_delaysp': [], 'monkey': [], 'session': []}
        df_dat_corr_reset = df_Sa0_corr.reset_index()

        cut_off_time = 5
        for idx in df_Sa0_corr.index[:-1]:  # run through all correct trials (0,len)
            if df_Sa0_corr.loc[idx, 'trial_id'] + 1 == df_Sa0_corr.loc[
                idx + 1, 'trial_id']:  # only compare within one sesssion
                # print(df_dat.loc[df_dat_corr_reset.loc[idx,'index'], 'outcome'], df_dat.loc[df_dat_corr_reset.loc[idx+1,'index'], 'outcome'])
                serial['target_prev'].append(df_dat_corr_reset['targ_angle'][idx])
                serial['target_prev_xy'].append(df_dat_corr_reset['targ_xy'][idx])
                serial['response_prev_xy'].append(df_dat_corr_reset['saccade_xy'][idx])
                serial['hemifield'].append(df_dat_corr_reset['hemifield'][idx])
                serial['monkey'].append(df_dat_corr_reset['monkey'][idx])
                serial['session'].append(df_dat_corr_reset['session'][idx])
                serial['bin_sp_prev'].append(bin_sp_trials[idx])
                serial['bin_sp_curr'].append(bin_sp_trials[idx + 1])
                serial['bin_sp'].append(
                    np.append(np.array(bin_sp_trials)[idx], np.array(bin_sp_trials)[idx + 1], axis=0))
                serial['bin_delaysp'].append(np.array(bin_sp_delay)[idx])

        ##################################################################################################
        #                     GET RESIDUAL, NON-SHARED ACTIVITY OF NEURONS                               #
        ##################################################################################################

        ##################################################################################################
        #                             DECODING FROM REAL ACTIVITY                                        #
        ##################################################################################################

        modes = 'bin'
        # Crossvalidate across hemispheres
        out = {'prediction_left1': [], 'std_prediction_left1': [], 'meanprediction_left1': [],  'weights_left1': [], 'hemifield_left1': [], \
               'prediction_left2': [], 'std_prediction_left2': [], 'meanprediction_left2': [], 'weights_left2': [], 'hemifield_left2': [], \
               'prediction_right1': [], 'std_prediction_right1': [], 'meanprediction_right1': [], 'weights_right1': [], 'hemifield_right1': [], \
               'prediction_right2': [], 'std_prediction_right2': [], 'meanprediction_right2': [], 'weights_right2': [], 'hemifield_right2': [], \
              'randomstates_left1':[], 'randomstates_left2':[], 'randomstates_right1':[], 'randomstates_right2':[],\
               'real': [], 'borders_full': [], 'num_targs': []}
        for c in range(5):
            rand_state = np.random.randint(10000)
            left1, left2 = train_test_split(left, test_size=0.50, random_state=rand_state)
            right1, right2 = train_test_split(right, test_size=0.50, random_state=rand_state)

            for hemi in ['left1', 'left2', 'right1', 'right2']:
                # add sorted spiking by hemisphere to dataframe (random half of neurons left and right)
                serial['bin_sp_' + hemi] = [
                    [serial['bin_sp'][n][t][eval(hemi)] for t in range(len(serial['bin_sp'][n]))] for n in
                    range(len(serial['bin_sp']))]
                serial['bin_delaysp_' + hemi] = [serial['bin_delaysp'][n][eval(hemi)] for n in
                    range(len(serial['bin_delaysp']))]
                # update dataframe with new spiking information
                df_serial = pd.DataFrame(serial)

                # single trial decoder predictions for each specified (half) hemisphere
                predictions, std_prediction, shuffle, real, \
                    weight, ipsicontra, targs, borders, rand_states = decode_singletrial(df_serial, borders_full, hemi, modes)

                out['prediction_' + hemi].append(predictions)
                out['std_prediction_' + hemi].append(std_prediction)
                out['meanprediction_' + hemi].append(shuffle)
                out['weights_' + hemi].append(weight)
                out['randomstates_' + hemi].append(rand_states)
                out['hemifield_' + hemi].append(ipsicontra)
                # don't change across crossvals (but need to append for same size)
            out['real'].append(real)
            out['num_targs'].append(targs)
            out['borders_full'].append(borders)

        df_sess = pd.DataFrame(out)
        df_sess['monkey'] = [mono for i in df_sess.index]
        df_sess['session'] = [sess for i in df_sess.index]


        df_out = df_out.append(df_sess)
        df_out.reset_index(inplace=True, drop=True)
        with open('DelayDecoderCorrelations_bins200.pickle', 'wb') as handle:
            pickle.dump(df_out, handle)

        print('computed session ' + str(mono) + str(sess))
#
# df_out.reset_index(inplace=True)
# with open('SameTimeDecoderKFoldCorrelationsWithinHemi_bins100_allmonkeys.pickle', 'wb') as handle:
#     pickle.dump(df_out, handle)

# import matplotlib as mpl
# norm = mpl.colors.Normalize(vmin=0, vmax=360)
# cmap = plt.get_cmap('twilight')
# compare_neuron = 2
# #for targ in [45.0,90.0,135.0,180.0,225.0]:#np.unique(df_serial.target_prev.values):
# #for targ in [0.0,270.0,315.0]:#np.unique(df_serial.target_prev.values):
# for targ in np.unique(df_serial.target_prev.values):
#     plt.scatter(y[np.where(X[:,-1]==targ)[0]], X[:,compare_neuron][np.where(X[:,-1]==targ)[0]], c = cmap(norm(targ)), label=str(targ))
# plt.legend()
# plt.xlabel('target neuron')
# plt.ylabel('other neuron')