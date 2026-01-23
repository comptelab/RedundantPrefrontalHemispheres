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
    intercept = True

    ## TODO! target decoding
    y = np.array([complex(dataframe['targ_prev_xy'][i][0],dataframe['targ_prev_xy'][i][1]) for i in dataframe.index])
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

    X_delay = np.array([np.array(dataframe.bin_sp_delay[i]) for i in dataframe.index])
    # z-score to activity in this timestep
    mean_ortho = np.mean(X_delay, axis=0)
    std_ortho = np.std(X_delay, axis=0)
    std_ortho[std_ortho<0.1] = 0.1
    X_delay = (X_delay -  mean_ortho) / std_ortho

    # get hemifield of targets
    hemifield_prev = dataframe['hemifield_prev'].values

    timing = range(borders_full[0], borders_full[16])

    # INITIALIZE OUTPUT ARRAY
    crossvalidate = 10
    k_folds=5
    number_perm=50
    random_states = np.random.choice(100000, size=crossvalidate)

    if neuron_type == 'combined':
        labels = ['mse_prev_'+neuron_type,'msacc_basecorrected_prev_'+neuron_type, 'strength_prev_'+neuron_type,\
                  'hemifield_prev_'+neuron_type,\
                  'num_targets', 'random_states_' + neuron_type]
    else: # otherwise separate ipsi/ contra predictions
        labels = ['mse_prev_ipsi_' + neuron_type, 'msacc_basecorrected_prev_ipsi_' + neuron_type,  'strength_prev_ipsi_'+neuron_type,\
                  'mse_prev_contra_' + neuron_type, 'msacc_basecorrected_prev_contra_' + neuron_type,  'strength_prev_contra_'+neuron_type,\
                  'hemifield_prev_' + neuron_type,\
                  'num_targets', 'random_states_' + neuron_type]
    acc_crosscorr = {l: np.empty((len(timing)))*np.nan for l in labels}
    acc_crosscorr['num_targets'] = len(np.unique(y))*np.ones((len(y)))
    acc_crosscorr['random_states_' + neuron_type] = [random_states for i in range(len(y))]
    # if we can't distinguish hemifield (use both hemispheres) give hemifield information about left/right
    if neuron_type=='combined':
        acc_crosscorr['hemifield_prev_'+neuron_type] = ['left' if hemifield_prev[i]=='left' else 'right' if hemifield_prev[i]=='right' else 'border'\
                                                        for i in range(len(hemifield_prev))]
    else: # give hemifield information about ipsi/contra instead of left/right
        acc_crosscorr['hemifield_prev_'+neuron_type] = ['ipsi' if hemifield_prev[i]==neuron_type\
                                                            else 'contra' if hemifield_prev[i]!='border'\
                                                        else 'border' for i in range(len(hemifield_prev))]


    mse_prev= np.empty((crossvalidate, k_folds, len(timing)))*np.nan
    mse_prev_ipsi = np.empty((crossvalidate, k_folds,len(timing)))*np.nan
    mse_prev_contra = np.empty((crossvalidate,k_folds, len(timing)))*np.nan
    basepred_prev = np.empty((crossvalidate, k_folds,len(timing)))*np.nan
    basestdpred_prev = np.empty((crossvalidate, k_folds,len(timing)))*np.nan
    basepred_prev_ipsi = np.empty((crossvalidate, k_folds, len(timing)))*np.nan
    basestdpred_prev_ipsi = np.empty((crossvalidate, k_folds, len(timing)))*np.nan
    basepred_prev_contra = np.empty((crossvalidate, k_folds, len(timing)))*np.nan
    basestdpred_prev_contra = np.empty((crossvalidate, k_folds, len(timing)))*np.nan
    strength_prev= np.empty((crossvalidate, k_folds, len(timing)))*np.nan
    strength_prev_ipsi= np.empty((crossvalidate, k_folds, len(timing)))*np.nan
    strength_prev_contra= np.empty((crossvalidate, k_folds, len(timing)))*np.nan

    # LOO crossvalidation to get single trial predictions
    for cro in range(crossvalidate):
        kf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state = random_states[cro])
        for k, (train_idx, test_idx) in enumerate(kf.split(X_delay, np.round(np.angle(y), 3).astype(str))):
            X_train = X_delay[train_idx]
            if intercept == True:
                X_train = np.array([np.append(1, X_train[i]) for i in range(len(X_train))])  # add intercept
            y_train, y_test = y[train_idx], y[test_idx]
            hemifield_test = hemifield_prev[test_idx]

            # TRAIN REGULAR MODEL
            weights = np.linalg.pinv(X_train.T.dot(X_train)).dot(X_train.T).dot(y_train)
            for t_id, delta_t_train in enumerate(timing):  # train / test decoder in each time step
                # create training dataset: columns=neurons, rows=trials for previous/current trials
                X_test = np.array([X_shuffle[n][delta_t_train] for n in test_idx])
                # Remove/add intercept
                if intercept == True:
                    X_test = np.array([np.append(1, X_test[i]) for i in range(len(X_test))])  # add intercept

                if neuron_type == 'combined':
                    mse_prev[cro, k, delta_t_train] = (np.mean((circdist(np.angle(X_test.dot(weights)), \
                                                                               np.angle(y_test))) ** 2))
                    strength_prev[cro, k, delta_t_train] = (np.mean(np.abs(X_test.dot(weights))))
                else:
                    ipsi_idx = np.where(hemifield_test == neuron_type)[0]
                    contra_idx = np.where((hemifield_test != neuron_type) & (hemifield_test != 'border'))[0]
                    mse_prev_ipsi[cro, k, delta_t_train] = (np.mean((circdist(np.angle(X_test[ipsi_idx].dot(weights)), \
                                                           np.angle(y_test[ipsi_idx]))) ** 2))
                    strength_prev_ipsi[cro, k, delta_t_train] = (np.mean(np.abs(X_test[ipsi_idx].dot(weights))))
                    mse_prev_contra[cro, k, delta_t_train] = (np.mean((circdist(np.angle(X_test[contra_idx].dot(weights)), \
                                                                               np.angle(y_test[contra_idx]))) ** 2))
                    strength_prev_contra[cro, k, delta_t_train] = (np.mean(np.abs(X_test[contra_idx].dot(weights))))

                #########################################################################################
                #                     BASELINE : calculate baseline for each shuffle                    #
                #########################################################################################
                if neuron_type == "combined":
                    y_test_forshuffle = copy.deepcopy(y_test)
                    baseline = []
                    for p in range(number_perm):
                        np.random.shuffle(y_test_forshuffle)
                        baseline.append(np.mean((circdist(np.angle(X_test.dot(weights)), \
                                                          np.angle(y_test_forshuffle))) ** 2))
                    basepred_prev[cro, k, delta_t_train] = np.mean(baseline)
                    basestdpred_prev[cro, k, delta_t_train] = np.std(baseline)
                else:
                    y_test_forshuffle_ipsi = copy.deepcopy(y_test[ipsi_idx])
                    y_test_forshuffle_contra = copy.deepcopy(y_test[contra_idx])

                    baseline_ipsi = []
                    baseline_contra = []
                    # shuffle target labels for baseline
                    for p in range(number_perm):  # permute:
                        # random shuffle of test labels
                        np.random.shuffle(y_test_forshuffle_ipsi)
                        np.random.shuffle(y_test_forshuffle_contra)

                        # make predictions of models on shuffled labels
                        baseline_ipsi.append(np.mean((circdist(np.angle(X_test[ipsi_idx].dot(weights)), \
                                                               np.angle(y_test_forshuffle_ipsi))) ** 2))
                        baseline_contra.append(np.mean((circdist(np.angle(X_test[contra_idx].dot(weights)), \
                                                                 np.angle(y_test_forshuffle_contra))) ** 2))
                    basepred_prev_ipsi[cro, k, delta_t_train] = np.mean(baseline_ipsi)
                    basestdpred_prev_ipsi[cro, k, delta_t_train] = np.std(baseline_ipsi)
                    basepred_prev_contra[cro, k, delta_t_train] = np.mean(baseline_contra)
                    basestdpred_prev_contra[cro, k, delta_t_train] = np.std(baseline_contra)

    print('Completed')
    # SAVE (mean over k-folds, mean over repeats)
    if neuron_type == 'combined':
        acc_crosscorr['mse_prev_' + neuron_type] = np.mean(np.mean(mse_prev, axis=1), axis=0)
        # z-score to baseline, then average out crossvalidations
        acc_crosscorr['msacc_basecorrected_prev_' + neuron_type] = np.mean(np.mean((-1)*(mse_prev - basepred_prev)/\
                                                                                   basestdpred_prev, axis=1), axis=0)
        acc_crosscorr['strength_prev_' + neuron_type] = np.mean(np.mean(strength_prev, axis=1), axis=0)
    else:
        acc_crosscorr['mse_prev_ipsi_' + neuron_type] = np.mean(np.mean(mse_prev_ipsi, axis=1), axis=0)
        acc_crosscorr['mse_prev_contra_' + neuron_type] = np.mean(np.mean(mse_prev_contra, axis=1), axis=0)
        acc_crosscorr['msacc_basecorrected_prev_ipsi_' + neuron_type] = np.mean(np.mean((-1)*(mse_prev_ipsi - basepred_prev_ipsi)/ \
                                                                                 basestdpred_prev_ipsi, axis=1), axis=0)
        acc_crosscorr['msacc_basecorrected_prev_contra_' + neuron_type] = np.mean(np.mean((-1)*(mse_prev_contra - basepred_prev_contra)/\
                                                                                  basestdpred_prev_contra, axis=1), axis=0)
        acc_crosscorr['strength_prev_ipsi_' + neuron_type] = np.mean(np.mean(strength_prev_ipsi, axis=1), axis=0)
        acc_crosscorr['strength_prev_contra_' + neuron_type] = np.mean(np.mean(strength_prev_contra, axis=1), axis=0)

    acc_crosscorr = {label: list([acc_crosscorr[label]]) for label in labels}
    return acc_crosscorr

##################################################################################################
#                                               LOAD DATA                                        #
##################################################################################################

with open(DATA_SMITH + 'df_dat_correct_Sa0.pickle', 'rb') as handle:
    df_dat_corr = pickle.load(handle)

with open(DATA_SMITH + 'leftRightIdx_Sa0.pickle', 'rb') as handle:
    leftRightIdx = pickle.load(handle)

monkeys=['Sa']#['Sa','Pe', 'Wa']

left_idx = {m: [[] for i in range(len(leftRightIdx['left'][m]))] for m in monkeys}
right_idx = {m: [[] for i in range(len(leftRightIdx['left'][m]))] for m in monkeys}

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
    for sess in range(max(df_dat_corr['session'].loc[df_dat_corr['monkey']==mono])+1):
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
                  'bin_sp':[], 'bin_sp_delay':[], 'index':[]}
        
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
                serial['bin_sp_delay'].append(df_Sa0_corr['spiking_forAligning'][idx])
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
        df_left['bin_sp_delay'] = [df['bin_sp_delay'][n][left] for n in range(len(df['bin_sp_delay']))]

        out_left = decode_predictions(dataframe = df_left, borders_full= borders_prevcurr, neuron_type='left')

        # TRAIN RIGHT DECODER

        # only right neurons
        df_right = df.copy()
        df_right.drop(['bin_sp'], axis=1)
        df_right['bin_sp'] = [[df['bin_sp'][n][t][right] for t in range(len(df['bin_sp'][n]))] for n in
                             range(len(df['bin_sp']))]
        df_right['bin_sp_delay'] = [df['bin_sp_delay'][n][right] for n in range(len(df['bin_sp_delay']))]


        out_right = decode_predictions(dataframe= df_right,borders_full= borders_prevcurr, neuron_type='right')

        ##################################################################################################
        #                                          SAVE DATA                                             #
        ##################################################################################################
        leftright = out_combined.copy()
        leftright.update(out_left)
        leftright.update(out_right)
        #leftright = {**out_combined, **out_left, **out_right}

        df_sess = pd.DataFrame(leftright)
        df_sess['monkey'] = [mono for i in df_sess.index]
        df_sess['session'] = [sess for i in df_sess.index]
        df_sess['delay_prev'] = [df.delay_prev]
        df_sess['delay_curr'] = [df.delay_curr]

        df_sess['borders_full'] = [list(borders_prevcurr) for i in df_sess.index]

        # append all monkeys, sessions into one dataframe
        df_out = df_out.append(df_sess)
        df_out.reset_index(inplace=True, drop=True)

        with open('DelayDecoder_200ms.pickle', 'wb') as handle:
            pickle.dump(df_out, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print('calculated monkey '+str(mono)+str(sess))
