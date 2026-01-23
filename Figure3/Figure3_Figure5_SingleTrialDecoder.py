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
    y_curr = np.array([complex(dataframe['targ_curr_xy'][i][0],dataframe['targ_curr_xy'][i][1]) for i in dataframe.index])

    # TODO! ZSCORE DATA
    # divide by mean per neuron to mean-center data on positive interval (mean across trials, times)
    spiking = np.array([np.array(dataframe.bin_sp[i]) for i in dataframe.index])
    mean_spiking = np.mean(spiking, axis=(0, 1))
    mean_spikingpertime = np.mean(spiking, axis=0)
    std_spikingpertime = np.std(spiking, axis=0)
    std_spikingpertime[std_spikingpertime < 0.1] = 0.1
    # z-score to FR of each neuron @ each time (or divide by mean)
    # X_shuffle = spiking/mean_spiking
    X_shuffle = (spiking - mean_spikingpertime) / std_spikingpertime

    # define number of neurons used (includes intercept
    num_neurons = X_shuffle.shape[2]
    if intercept == True:
        num_neurons = X_shuffle.shape[2] + 1

    X_ortho = np.array([np.array(dataframe.bin_sp_ortho[i]) for i in dataframe.index])
    # z-score to activity in this timestep
    mean_ortho = np.mean(X_ortho, axis=0)
    std_ortho = np.std(X_ortho, axis=0)
    std_ortho[std_ortho<0.1] = 0.1
    X_ortho = (X_ortho -  mean_ortho) / std_ortho

    # get hemifield of targets
    hemifield_prev = dataframe['hemifield_prev'].values
    hemifield_curr = dataframe['hemifield_curr'].values
    number_splits = X_shuffle.shape[0]

    timing = range(borders_full[0], borders_full[21])

    # INITIALIZE OUTPUT ARRAY
    crossvalidate = 20
    random_states = np.random.choice(100000, size=crossvalidate)

    labels = ['pred_complex_prev_'+neuron_type,'shufflepred_complex_prev_'+neuron_type,\
              'pred_complex_delay_' + neuron_type,'shufflepred_complex_delay_' + neuron_type,\
              'hemifield_prev_'+neuron_type, 'hemifield_curr_'+neuron_type,
              'weights_prev_'+neuron_type, 'weights_prev_delay_'+neuron_type, \
              'pred_complex_curr_' + neuron_type,'shufflepred_complex_curr_' + neuron_type,\
              'num_targets', 'random_states_' + neuron_type]
    acc_crosscorr = {l: np.zeros((len(y), len(timing)),dtype=np.complex_) for l in labels}
    acc_crosscorr['weights_prev_'+neuron_type] = np.zeros((len(y), len(timing), num_neurons))
    acc_crosscorr['weights_prev_delay_'+neuron_type] = np.zeros((len(y), len(timing), num_neurons))
    acc_crosscorr['num_targets'] = len(np.unique(y))*np.ones((len(y)))
    # acc_crosscorr['ortho_start'] = [start for i in range(len(y))]
    # acc_crosscorr['ortho_offset'] = [offset for i in range(len(y))]
    # acc_crosscorr['ortho_bins'] = (ortho_bins*bins)*np.ones((len(y)))
    # acc_crosscorr['ortho_steps'] = timesteps_ortho*np.ones((len(y)))
    acc_crosscorr['random_states_' + neuron_type] = [random_states for i in range(len(y))]
    # if we can't distinguish hemifield (use both hemispheres) give hemifield information about left/right
    if neuron_type=='combined':
        acc_crosscorr['hemifield_prev_'+neuron_type] = [-1 if hemifield_prev[i]=='left' else 1 if hemifield_prev[i]=='right' else 0\
                                                        for i in range(len(hemifield_prev))]
        acc_crosscorr['hemifield_curr_'+neuron_type] = [-1 if hemifield_curr[i]=='left' else 1 if hemifield_curr[i]=='right' else 0\
                                                        for i in range(len(hemifield_curr))]
    else: # give hemifield information about ipsi/contra instead of left/right
        acc_crosscorr['hemifield_prev_'+neuron_type] = [1 if hemifield_prev[i]==neuron_type\
                                                            else -1 if hemifield_prev[i]!='border'\
                                                        else 0 for i in range(len(hemifield_prev))]
        acc_crosscorr['hemifield_curr_' + neuron_type] = [1 if hemifield_curr[i] == neuron_type \
                                                              else -1 if hemifield_curr[i] != 'border' \
                                                         else 0 for i in range(len(hemifield_curr))]

    pred_prev = np.empty((crossvalidate, len(y), len(timing)),dtype=np.complex_)*np.nan
    shufflepred_prev = np.empty((crossvalidate, len(y), len(timing)),dtype=np.complex_)*np.nan
    pred_curr = np.empty((crossvalidate, len(y), len(timing)),dtype=np.complex_)*np.nan
    shufflepred_curr = np.empty((crossvalidate, len(y), len(timing)),dtype=np.complex_)*np.nan
    pred_ortho = np.empty((crossvalidate, len(y), len(timing)),dtype=np.complex_)*np.nan
    shufflepred_ortho = np.empty((crossvalidate, len(y), len(timing)),dtype=np.complex_)*np.nan
    for t_id, delta_t_train in enumerate(timing):# train / test decoder in each time step
        # create training dataset: columns=neurons, rows=trials for previous/current trials
        X = np.array([X_shuffle[n][delta_t_train] for n in dataframe['bin_sp'].index])
        # Remove/add intercept
        if intercept == True:
            X = np.array([np.append(1, X[i]) for i in range(len(X))]) #add intercept

        # LOO crossvalidation to get single trial predictions
        for cro in range(crossvalidate):
            kf = StratifiedKFold(n_splits=5, shuffle=True, random_state = random_states[cro])
            for train_idx, test_idx in kf.split(X, np.round(np.angle(y), 3).astype(str)):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test, y_train_curr, y_test_curr = y[train_idx], y[test_idx], y_curr[train_idx], y_curr[test_idx]

                # TRAIN REGULAR MODEL
                weights = np.linalg.pinv(X_train.T.dot(X_train)).dot(X_train.T).dot(y_train)
                weights_curr = np.linalg.pinv(X_train.T.dot(X_train)).dot(X_train.T).dot(y_train_curr)
                acc_crosscorr['weights_prev_' + neuron_type][test_idx, t_id] = np.round(np.angle(weights),2)

                ############################################
                #          ORTHOGONALIZE WEIGHTS           #
                ############################################

                # timesteps_ortho = 1

                # # initialize weights ortho as weights, later use it recursively
                # weights_orthogonalized = np.zeros((timesteps_ortho + 1, len(weights)), dtype=complex)
                # weights_orthogonalized[0] = weights
                # weights_orthogonalized_curr = np.zeros((timesteps_ortho + 1, len(weights_curr)), dtype=complex)
                # weights_orthogonalized_curr[0] = weights_curr

                # Delay alignment
                X_train_ortho = X_ortho[train_idx]
                # TRAIN WEIGHTS OF CODE WE WANT TO ORTHOGONALIZE TO: RESPONSE AT TIME X
                weights_delay = np.linalg.pinv(X_train_ortho.T.dot(X_train_ortho)).dot(X_train_ortho.T).dot(y_train)
                acc_crosscorr['weights_prev_delay_' + neuron_type][test_idx, t_id] = np.round(np.angle(weights_delay),2)

                pred_prev[cro, test_idx, delta_t_train] = np.round((weights.dot(X_test.T)).astype(np.complex), 2)
                pred_ortho[cro, test_idx, delta_t_train] = np.round((weights_delay.dot(X_test.T)).astype(np.complex), 2)
                pred_curr[cro, test_idx, delta_t_train] = np.round((weights_curr.dot(X_test.T)).astype(np.complex), 2)

                # SHUFFLE
                # shuffle neural activity within targets with the same stimulus
                X_test_shuffle = copy.deepcopy(X_test)
                # base shuffle on trials with same previous/current trial
                if delta_t_train < borders_full[14]:
                    y_tosplit = copy.deepcopy(y_test)
                else:
                    y_tosplit = copy.deepcopy(y_test_curr)
                # shuffle those trials with them same cue (to keep autocorrelation)
                shuffle_prev, shuffle_ortho=[], []
                shuffle_curr = []
                for crossval_shuffle in range(50):
                    # for shuffling within target groups
                    for targ_shuffle in np.unique(y_tosplit):
                        index = np.where(y_tosplit==targ_shuffle)[0]
                        X_test_shuffle[index,:] = X_test_shuffle[np.random.permutation(index), :]
                    shuffle_prev.append((weights.dot(X_test_shuffle.T)).astype(np.complex))
                    shuffle_ortho.append((weights_delay.dot(X_test_shuffle.T)).astype(np.complex))
                    shuffle_curr.append((weights_curr.dot(X_test_shuffle.T)).astype(np.complex))

                shuffle_prev, shuffle_prev_ortho = np.mean(shuffle_prev, axis=0), np.mean(shuffle_ortho, axis=0)
                shuffle_curr = np.mean(shuffle_curr, axis=0)

                assert np.all(np.histogram(X_test[:,0])[0] == np.histogram(X_test_shuffle[:,0])[0]),\
                    "Shuffle along wrong axis: Histograms for individual neurons must stay the same."
                assert np.any(X_test != X_test_shuffle), "No shuffled trials: Test shuffle."

                shufflepred_prev[cro, test_idx, delta_t_train] = np.round(shuffle_prev,2)
                shufflepred_ortho[cro, test_idx, delta_t_train] = np.round(shuffle_prev_ortho,2)
                shufflepred_curr[cro, test_idx, delta_t_train] = np.round(shuffle_curr, 2)

    print('Completed')
    # SAVE (round to 2 decimals to save space
    acc_crosscorr['pred_complex_prev_' + neuron_type] = np.mean(pred_prev, axis=0)
    acc_crosscorr['pred_complex_delay_' + neuron_type] = np.mean(pred_ortho, axis=0)
    acc_crosscorr['pred_complex_curr_' + neuron_type] = np.mean(pred_curr, axis=0)

    # predictions of shuffle
    acc_crosscorr['shufflepred_complex_prev_' + neuron_type] = np.mean(shufflepred_prev, axis=0)
    acc_crosscorr['shufflepred_complex_delay_' + neuron_type] = np.mean(shufflepred_ortho, axis=0)
    acc_crosscorr['shufflepred_complex_curr_' + neuron_type] = np.mean(shufflepred_curr, axis=0)

    acc_crosscorr = {label: list(acc_crosscorr[label]) for label in labels}
    return acc_crosscorr

# DETERMINE NECCESSARY NUMBER OF CROSSVALIDATIONS
# timings = range(borders_full[5], borders_full[6])
# # compute variances over crossvalidation
# # get prediction errors (shape = crossvals x trials x timing
# delay_err = np.array([[circdist(np.angle(pred_prev[:, trial, t]), np.angle(y)[trial]) for trial in range(len(y))] for t in timings]).T
# delay_pred = np.angle(pred_prev[:, :, borders_full[5]:borders_full[6]])
# variance_crossvals=[]
# for cr in range(crossvalidate-1):
#     # compute variance of prediction errors
#     method = 'sem'
#     variance_crossvals.append(np.mean(eval(method)(delay_err[:cr+1, :, :], axis=0)))
# plt.plot(variance_crossvals)
# plt.xlabel('# repeats')
# plt.ylabel(method+' of prediction errors')


# error = [circdist(np.angle(acc_crosscorr['pred_complex_prev_ortho_' + neuron_type][i]), np.angle(y[i]))\
#                       for i in range(len(y))]
# for i in range(5):
#     plt.plot(error[i])

# plt.fill_between([borders_full[13], borders_full[15]], [-np.pi, -np.pi], [np.pi, np.pi], alpha=0.2, color='grey')
# plt.axhline(0, color='k', dashes=[1,1])


# # # AVERAGE SINGLE TRIAL DECODING (NO SPLIT IPSI/CONTRA)
# squarederror = [circdist(np.angle(acc_crosscorr['pred_complex_prev_' + neuron_type][i]), np.angle(y[i])) ** 2 \
#                       for i in range(len(y))]
# squarederror_curr = [circdist(np.angle(acc_crosscorr['pred_complex_curr_' + neuron_type][i]), np.angle(y_curr[i])) ** 2 \
#                       for i in range(len(y_curr))]
# squarederror_ortho_curr = [circdist(np.angle(acc_crosscorr['pred_complex_curr_ortho_' + neuron_type][i]), np.angle(y_curr[i])) ** 2 \
#                       for i in range(len(y_curr))]
# squarederror_ortho = [circdist(np.angle(acc_crosscorr['pred_complex_prev_ortho_' + neuron_type][i]), np.angle(y[i]))**2\
#                        for i in range(len(y))]
# plt.plot(np.mean(squarederror, axis=0)*(-1), label='prev')
# plt.plot(np.mean(squarederror_curr, axis=0)*(-1), label='curr')
# plt.plot(np.mean(squarederror_ortho_curr, axis=0)*(-1), label='curr')
# plt.plot(np.mean(squarederror_ortho, axis=0)*(-1), label='ortho')
# plt.axvline(borders_full[8])
# plt.fill_between([borders_full[13], borders_full[15]], [-4, -4], [0, 0], alpha=0.2, color='grey')
# plt.legend()

# IPSI VS CONTRALATERAL SINGLE TRIAL AVERAGE DECODING
# for hemi in ['ipsi', 'contra']:
#     index = np.where(np.array(acc_crosscorr['hemifield_prev_'+neuron_type])==hemi)[0]
#     squarederror = [circdist(np.angle(acc_crosscorr['targ_prev_complex_' + neuron_type][i]),
#                                    np.angle(acc_crosscorr['pred_complex_' + neuron_type][i])) ** 2 \
#                           for i in index]
#     squarederror_ortho = [circdist(np.angle(acc_crosscorr['targ_prev_complex_' + neuron_type][i]),\
#                                    np.angle(acc_crosscorr['pred_complex_ortho_' + neuron_type][i]))**2\
#                            for i in index]
#     plt.plot(np.mean(squarederror, axis=0)*(-1), label=hemi)
#     #plt.plot(np.mean(squarederror_ortho, axis=0)*(-1), label='ortho, '+hemi)
# plt.axvline(borders_full[13])
# plt.axvline(borders_full[15])
# plt.legend()

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
df_dat_corr['hemifield'] = ['left' if (df_dat_corr['targ_angle'][i]<np.round(-np.pi/2,5)) |\
                               (df_dat_corr['targ_angle'][i]>np.round(np.pi/2,5)) \
                               else 'right' if (df_dat_corr['targ_angle'][i]>np.round(-np.pi/2,5)) &\
                               (df_dat_corr['targ_angle'][i]<np.round(np.pi/2,5)) else 'border' \
                               for i in df_dat_corr.index]

##################################################################################################
#                                               SPLIT DATA                                        #
##################################################################################################

df_out = pd.DataFrame()
for mono in monkeys:#, 'Pe']:#
    for sess in range(0,max(df_dat_corr['session'].loc[df_dat_corr['monkey']==mono])+1):

        print(sess)
        df_Sa0_corr = df_dat_corr.loc[(df_dat_corr['monkey']==mono) & (df_dat_corr['session']==sess)].copy().reset_index(drop=True)
               
        # make spike trains into csr matrix for each trial
        mat = [csr_matrix(df_Sa0_corr.loc[n,'sp_train']) for n in df_Sa0_corr['sp_train'].index]
        df_Sa0_corr.insert(5,'n_mat',mat,True)
        
        # determine border points between different time periods, until beginning of delay
        bins = 200 # TODO!
        
        # determine border points INDIVID trials between different time periods, for end of delay
        timings2 = ['start','fix','targ_on','targ_off','go_cue','saccade', 'reward', 'trial_end']
        t_borders2 = ['start', 'start_front', 'start_back', 'fix','targ_on','targ_off','delay_front_end',\
                  'delay_start', 'delay_end', 'saccade_front', 'saccade', 'saccade_end', 'reward', 'trial_front_end',\
                  'end_start', 'trial_end']#'go_cue',
        borders={'start': [],'start_front':[], 'start_back': [], 'fix': [],'targ_on': [],'targ_off': [],'delay_front_end':[],\
                  'delay_start': [], 'delay_end': [], 'saccade_front':[], 'saccade': [], 'saccade_end':[],\
                 'reward':[], 'trial_front_end':[],'end_start':[], 'trial_end':[]}#'go_cue': [],
        for i,m in enumerate(borders.keys()):
            if (m == 'start') | (m == 'fix') | (m == 'saccade') | (m == 'reward') | (m == 'trial_end'):
                borders[m] = ((df_Sa0_corr[m].values)/bins).astype(int)
            elif m == 'start_front':
                borders[m] = ((df_Sa0_corr['start'].values)/bins).astype(int)+((min(df_Sa0_corr['fix'].values-df_Sa0_corr['start'].values))/bins).astype(int)   
            elif m == 'start_back':
                borders[m] = ((df_Sa0_corr['fix'].values)/bins).astype(int)-((min(df_Sa0_corr['fix'].values-df_Sa0_corr['start'].values))/bins).astype(int)   
            elif (m == 'targ_on'):
                borders[m] = borders['fix']+int(min(df_Sa0_corr['targ_on']-df_Sa0_corr['fix'])/bins)
            elif (m == 'targ_off'):
                borders[m] = borders['targ_on']+int(min(df_Sa0_corr['targ_off']-df_Sa0_corr['targ_on'])/bins)
            elif (m == 'delay_front_end'):
                borders[m] = np.array(borders['targ_off']) + int(min(df_Sa0_corr['go_cue']-df_Sa0_corr['targ_off'])/bins)
            elif m == 'delay_start':
                # create shifted "start" of delay
                borders[m] = ((df_Sa0_corr['go_cue'].values)/bins).astype(int) - int(min(df_Sa0_corr['go_cue']-df_Sa0_corr['targ_off'])/bins)
            elif m == 'delay_end':
                # delay end
                borders[m] = ((df_Sa0_corr['go_cue'].values)/bins).astype(int)
            elif m == 'saccade_front':
                borders[m] = np.array(borders['delay_end']) + int(min(df_Sa0_corr['saccade']-df_Sa0_corr['go_cue'])/bins)
            elif m == 'saccade_end':
                borders[m] = np.array(borders['saccade']) + int(min(df_Sa0_corr['reward']-df_Sa0_corr['saccade'])/bins)
            elif m == 'trial_front_end':
                borders[m] = np.array(borders['reward']) + int(min(df_Sa0_corr['trial_end']-df_Sa0_corr['reward'])/bins)
            elif m =='end_start':
                # shifted "start" of trial end : complete end of trial - minimum(trial_end-reward)
                borders[m] = [int(df_Sa0_corr.loc[n,'trial_end']/bins)-int(min(df_Sa0_corr.loc[:,'trial_end']-df_Sa0_corr.loc[:,'reward'])/bins) for n in df_Sa0_corr.index]#
            else:
                print('Error')
                # create end delay, saccade start, reward start, trial_end through using minimum distance between periods, adding to delay_end, saccade_end,..
                #borders[m] = np.array(borders[t_borders2[i-1]]) + min([int((df_Sa0_corr.loc[n,timings2[i-1]]-df_Sa0_corr.loc[n,timings2[i-2]])/bins) for n in df_Sa0_corr.index])

        ## add shift between trial short end and trial long start
        #borders.append(borders[-1]+min(np.array(borders2['trial_end'])- np.array(borders2['delay_start'])))
        #print(borders)

        
       # for first cut (different delay lengths)
        time_connections = [0,2,3,4,5,7,8,10,12,14]
        bin_sp_trials=[]
        period_spikes=[]
        for idx, trial in enumerate(df_Sa0_corr.index):# for all trials
            binned_spikes = []
            number_bins=[]
            number_bins_len=[]
            for period in time_connections:# from start_back for all time periods until trial_end
                #if period<5:
                number_bins.append(borders[t_borders2[period+1]][0]-borders[t_borders2[period]][0])
                number_bins_len.append(borders[t_borders2[period+1]][0])
                for t in range(borders[t_borders2[period+1]][0]-borders[t_borders2[period]][0]): # for number of time bins in discrete timings:           
                    # sum the matrix of neurons at timings in bin
                    binned_spikes.append(np.sum(df_Sa0_corr.loc[trial, 'n_mat'][:,borders[t_borders2[period]][idx]*bins+t*bins:borders[t_borders2[period]][idx]*bins+t*bins+bins].toarray(), axis=1))
            bin_sp_trials.append(binned_spikes)
        
        
        borders_full = [np.sum(abs(np.array(number_bins))[:i]) for i in range(len(number_bins)+1)]
        borders_prevcurr = np.append(borders_full, borders_full+max(borders_full))

        # WHICH CODE TO RESPONSE ORTHOGONALIZE TO
        # convert to sum/bin (as all other timepoints)
        df_Sa0_corr['spiking_forAligning'] = [np.sum(df_Sa0_corr.loc[n, 'sp_train'].toarray()[:, int(df_Sa0_corr['targ_off'][n]+100): \
                                                                int(df_Sa0_corr['go_cue'][n])],\
                                                    axis=1)*bins/(int(df_Sa0_corr['go_cue'][n])-int(df_Sa0_corr['targ_off'][n]+100))\
                                              for n in df_Sa0_corr.index]

        ###################################################################################
        #                                  SERIAL DEPENDENCE                              #              
        ###################################################################################
        
        serial = {'targ_prev_xy':[],'targ_curr_xy':[],\
                  'response_prev_xy':[],'response_curr_xy':[], 'error_prev':[], 'error_curr':[],\
                  'delay_prev':[], 'delay_curr':[], 'hemifield_prev':[], 'hemifield_curr':[],\
                  'bin_sp':[], 'bin_sp_ortho':[], 'index':[]}
        
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
                serial['hemifield_prev'].append(df_Sa0_corr['hemifield'][idx])
                serial['hemifield_curr'].append(df_Sa0_corr['hemifield'][idx+1])
                serial['bin_sp'].append(np.append(bin_sp_trials[idx], bin_sp_trials[idx+1], axis=0))
                serial['bin_sp_ortho'].append(df_Sa0_corr['spiking_forAligning'][idx])
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

        # SPLIT into left/right neurons
        left, right = np.where(left_idx[mono][sess] == 1)[1], np.where(right_idx[mono][sess] == 1)[1]

        # TRAIN LEFT DECODER

        df_left = df.copy()
        df_left.drop(['bin_sp'], axis=1)
        # create dataframe with only left neurons
        df_left['bin_sp'] = [[df['bin_sp'][n][t][left] for t in range(len(df['bin_sp'][n]))] for n in range(len(df['bin_sp']))]
        df_left['bin_sp_ortho'] = [df['bin_sp_ortho'][n][left] for n in range(len(df['bin_sp_ortho']))]

        out_left = decode_predictions(dataframe = df_left, borders_full= borders_prevcurr, neuron_type='left')

        # TRAIN RIGHT DECODER

        # only right neurons
        df_right = df.copy()
        df_right.drop(['bin_sp'], axis=1)
        df_right['bin_sp'] = [[df['bin_sp'][n][t][right] for t in range(len(df['bin_sp'][n]))] for n in
                             range(len(df['bin_sp']))]
        df_right['bin_sp_ortho'] = [df['bin_sp_ortho'][n][right] for n in range(len(df['bin_sp_ortho']))]


        out_right = decode_predictions(dataframe= df_right,borders_full= borders_prevcurr, neuron_type='right')

        ##################################################################################################
        #                                          SAVE DATA                                             #
        ##################################################################################################

        leftright = {**out_combined, **out_left, **out_right}

        df_sess = pd.DataFrame(leftright)
        df_sess['monkey'] = [mono for i in df_sess.index]
        df_sess['session'] = [sess for i in df_sess.index]
        df_sess['targ_prev_xy'] = targ_prev
        df_sess['targ_curr_xy'] = targ_curr
        df_sess['prev_curr'] = df.prev_curr
        df_sess['behav_error_prev'] = df.error_prev
        df_sess['behav_error_curr'] = df.error_curr
        df_sess['delay_prev'] = df.delay_prev
        df_sess['delay_curr'] = df.delay_curr

        df_sess['borders_full'] = [list(borders_prevcurr) for i in df_sess.index]

        # append all monkeys, sessions into one dataframe
        df_out = df_out.append(df_sess)
        df_out.reset_index(inplace=True, drop=True)

        with open('Serial20times5FoldStratifiedCrossvalDelayDecodingPrevCurr_bins200_zscorednoIntercept_randomShuffleControlprevcurrcorrected_security.pickle', 'wb') as handle:
            pickle.dump(df_out, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open('Serial20times5FoldStratifiedCrossvalDelayDecodingPrevCurr_bins200_zscorednoIntercept_randomShuffleControlprevcurrcorrected_long.pickle', 'wb') as handle:
            pickle.dump(df_out, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print('calculated monkey '+str(mono)+str(sess))
