#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 14:23:15 2020

@author: melanie

calculate the prediction error (target) from a delay-decoder in a leave-1-out
cross validation during the reactivation period (borders_full[7]:borders_full[8]
of previous trial, borders_full[0]:borders_full[1] of current trial for all monkeys
"""

import numpy as np
import pandas as pd
from scipy.stats import *
from scipy.sparse import csr_matrix
from sklearn.model_selection import LeaveOneOut
from circ_stats import *
import h5py
import pickle
from matplotlib.pyplot import *
from filepaths import *
from sklearn.model_selection import StratifiedKFold
import copy
#from sklearn import preprocessing
#from numpy.linalg import inv

##################################################################################################
#                                               FUNCTIONS                                        #
##################################################################################################

# y is fixed only X changes based on time in trial
def decode_predictions(dataframe = pd.DataFrame(),borders_full=[], neuron_type='combined'):
    '''Only decode during reactivation to check strength of serial dependence for trials
    INPUT:      dataframe = serial dependence dataframe
                y = target/response position per trial in rad [-np.pi,np.pi]
                borders_full = borders of time periods 
    OUTPUT:     acc_curr = array of circular distance between prediction, target for time x trial
    '''

    # TODO! Use intercept or not
    intercept = False

    ## TODO! target decoding
    y = np.array([complex(dataframe['targ_prev_xy'][i][0],dataframe['targ_prev_xy'][i][1]) for i in dataframe.index])

    # TODO! ZSCORE DATA
    eye_x = np.array([np.array(dataframe.eye_x[i]) for i in dataframe.index])
    eye_y = np.array([np.array(dataframe.eye_y[i]) for i in dataframe.index])
    eye = np.array([eye_x, eye_y])
    eye = np.transpose(eye, (1,2,0))

    timing = range(borders_full[0], borders_full[16])

    # INITIALIZE OUTPUT ARRAY
    crossvalidate = 20
    number_perm=50
    random_states = np.random.choice(100000, size=crossvalidate)

    labels = ['mse_prev_'+neuron_type,'msacc_basecorrected_prev_'+neuron_type,\
              'num_targets', 'random_states_' + neuron_type]
    acc_crosscorr = {l: np.empty((len(timing)))*np.nan for l in labels}
    acc_crosscorr['num_targets'] = len(np.unique(y))*np.ones((len(y)))
    acc_crosscorr['random_states_' + neuron_type] = [random_states for i in range(len(y))]


    mse_prev= np.empty((crossvalidate, len(timing)))*np.nan
    basepred_prev = np.empty((crossvalidate, len(timing)))*np.nan
    basestdpred_prev = np.empty((crossvalidate, len(timing)))*np.nan
    strength_prev = np.empty((crossvalidate, len(timing)))*np.nan
    # LOO crossvalidation to get single trial predictions
    for cro in range(crossvalidate):
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state = random_states[cro])
        for t_id, delta_t_train in enumerate(timing):  # train / test decoder in each time step
            for train_idx, test_idx in kf.split(eye, np.round(np.angle(y), 3).astype(str)):
                X_train,X_test = eye[train_idx,delta_t_train], eye[test_idx,delta_t_train]
                y_train, y_test = y[train_idx], y[test_idx]

                # TRAIN REGULAR MODEL
                weights = np.linalg.pinv(X_train.T.dot(X_train)).dot(X_train.T).dot(y_train)

                mse_prev[cro, delta_t_train] = (np.mean((circdist(np.angle(X_test.dot(weights)), \
                                                                           np.angle(y_test))) ** 2))
                strength_prev[cro, delta_t_train] = (np.mean(np.abs(X_test.dot(weights))))

                #########################################################################################
                #                     BASELINE : calculate baseline for each shuffle                    #
                #########################################################################################
                y_test_forshuffle = copy.deepcopy(y_test)
                baseline = []
                for p in range(number_perm):
                    np.random.shuffle(y_test_forshuffle)
                    baseline.append(np.mean((circdist(np.angle(X_test.dot(weights)), \
                                                      np.angle(y_test_forshuffle))) ** 2))
                basepred_prev[cro, delta_t_train] = np.mean(baseline)
                basestdpred_prev[cro, delta_t_train] = np.std(baseline)

    print('Completed')
    # SAVE (round to 2 decimals to save space
    acc_crosscorr['mse_prev_' + neuron_type] = np.mean(mse_prev, axis=0)
    # z-score to baseline, then average out crossvalidations
    acc_crosscorr['msacc_basecorrected_prev_' + neuron_type] = np.mean((-1)*(mse_prev - basepred_prev)/ basestdpred_prev, axis=0)
    acc_crosscorr['strength_prev_' + neuron_type] = np.mean(strength_prev, axis=0)

    acc_crosscorr = {label: list([acc_crosscorr[label]]) for label in labels}
    return acc_crosscorr

##################################################################################################
#                                               LOAD DATA                                        #
##################################################################################################

with open(DATA_SMITH + 'df_dat_correct_eyetrackerSa0.pickle', 'rb') as handle:
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

##################################################################################################
#                                               SPLIT DATA                                        #
##################################################################################################

df_out = pd.DataFrame()
for mono in monkeys:#['Sa','Pe', 'Wa']:#, 'Pe']:#
    for sess in range(max(df_dat_corr['session'].loc[df_dat_corr['monkey']==mono])+1):
        print(sess)
        df_Sa0_corr = df_dat_corr.loc[(df_dat_corr['monkey']==mono) & (df_dat_corr['session']==sess)].copy().reset_index(drop=True)


        bins=200 # TODO!

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
        time_connections = [0,2,3,4,5,7,8,10,12,14]
        bin_eyex = []
        bin_eyey = []
        for idx, trial in enumerate(df_Sa0_corr.index):  # for all trials
            binned_eyex = []
            binned_eyey = []
            number_bins = []
            for period in time_connections:  # from start_back for all time periods until trial_end
                number_bins.append(borders[t_borders2[period + 1]][0] - borders[t_borders2[period]][0])
                for t in range(borders[t_borders2[period + 1]][0] - borders[t_borders2[period]][
                    0]):  # for number of time bins in discrete timings:
                    # BIN the eye position
                    binned_eyex.append(np.mean(df_Sa0_corr.loc[trial, 'eye_x'][
                                               borders[t_borders2[period]][idx] * bins + t * bins:
                                               borders[t_borders2[period]][idx] * bins + t * bins + bins]))
                    binned_eyey.append(np.mean(df_Sa0_corr.loc[trial, 'eye_y'][
                                               borders[t_borders2[period]][idx] * bins + t * bins:
                                               borders[t_borders2[period]][idx] * bins + t * bins + bins]))
            bin_eyex.append(binned_eyex)
            bin_eyey.append(binned_eyey)

        borders_full = [np.sum(abs(np.array(number_bins))[:i]) for i in range(len(number_bins)+1)]
        borders_prevcurr = np.append(borders_full, borders_full+max(borders_full))

        ###################################################################################
        #                                  SERIAL DEPENDENCE                              #              
        ###################################################################################
        
        serial = {'targ_prev_xy':[],'targ_curr_xy':[],\
                  'response_prev_xy':[],'response_curr_xy':[], 'error_prev':[], 'error_curr':[],\
                  'delay_prev':[], 'delay_curr':[], 'eye_x':[], 'eye_y':[],\
                   'index':[]}
        
        cut_off_time=5
        for idx in df_Sa0_corr.index[:-1]:# run through all correct trials (0,len)
            if df_Sa0_corr.loc[idx,'trial_id']+1 == df_Sa0_corr.loc[idx+1,'trial_id']: # only compare within one sesssion
                serial['targ_prev_xy'].append(df_Sa0_corr['targ_xy'][idx])
                serial['targ_curr_xy'].append(df_Sa0_corr['targ_xy'][idx+1])
                serial['response_prev_xy'].append(df_Sa0_corr['saccade_xy'][idx])
                serial['response_curr_xy'].append(df_Sa0_corr['saccade_xy'][idx+1])
                serial['error_prev'].append(df_Sa0_corr['error'][idx])
                serial['error_curr'].append(df_Sa0_corr['error'][idx + 1])
                serial['delay_prev'].append(df_Sa0_corr['go_cue'][idx] - df_Sa0_corr['targ_off'][idx])
                serial['delay_curr'].append(df_Sa0_corr['go_cue'][idx + 1] - df_Sa0_corr['targ_off'][idx + 1])
                serial['eye_x'].append(np.append(bin_eyex[idx], bin_eyex[idx + 1]))
                serial['eye_y'].append(np.append(bin_eyey[idx], bin_eyey[idx + 1]))
                serial['index'].append(df_Sa0_corr.index[idx])

        df = pd.DataFrame(serial)
        targ_prev = np.array([complex(df['targ_prev_xy'][i][0],df['targ_prev_xy'][i][1]) for i in df.index])
        targ_curr = np.array([complex(df['targ_curr_xy'][i][0],df['targ_curr_xy'][i][1]) for i in df.index])
        df['prev_curr'] = np.round(circdist(np.angle(targ_prev), np.angle(targ_curr)),3)

        ##################################################################################################
        #                                          DECODER                                               #
        ##################################################################################################

        # TRAIN COMBINED DECODER

        out_combined = decode_predictions(dataframe= df,borders_full= borders_prevcurr, neuron_type='combined')

        ##################################################################################################
        #                                          SAVE DATA                                             #
        ##################################################################################################

        df_sess = pd.DataFrame(out_combined)
        df_sess['monkey'] = [mono for i in df_sess.index]
        df_sess['session'] = [sess for i in df_sess.index]
        df_sess['delay_prev'] = [df.delay_prev]
        df_sess['delay_curr'] = [df.delay_curr]

        df_sess['borders_full'] = [list(borders_prevcurr) for i in df_sess.index]

        # append all monkeys, sessions into one dataframe
        df_out = df_out.append(df_sess)
        df_out.reset_index(inplace=True, drop=True)

        with open('SaccadeDecoder_200ms.pickle', 'wb') as handle:
            pickle.dump(df_out, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print('calculated monkey '+str(mono)+str(sess))
