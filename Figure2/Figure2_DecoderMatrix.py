#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 10:21:03 2022

@author: melanie
"""

import numpy as np
import pandas as pd
from scipy.stats import *
from scipy.sparse import csr_matrix
from sklearn.model_selection import LeaveOneOut
from circ_stats import *
import pickle
from random import randint
from sklearn.model_selection import train_test_split
import copy
from filepaths import *
#from sklearn import preprocessing
#from numpy.linalg import inv

##################################################################################################
#                                               FUNCTIONS                                        #
##################################################################################################


# y is fixed only X changes based on time in trial
def delay_decoder(dataframe,borders_full, neuron_type):
    '''Only decode during reactivation to check strength of serial dependence for trials
    INPUT:      dataframe = serial dependence dataframe
                y = target/response position per trial in rad [-np.pi,np.pi]
                borders_full = borders of time periods 
    OUTPUT:     acc_curr = array of circular distance between prediction, target for time x trial
    '''
    # y is fixed only X changes based on time in trial
    
    ## TODO! target decoding
    y = np.array([complex(dataframe['target_prev_xy'][i][0],dataframe['target_prev_xy'][i][1]) for i in dataframe.index])

    # divide by mean per neuron to mean-center data on positive interval (mean across trials, times)
    spiking = np.append(np.array([np.array(dataframe['bin_sp_prev'][i]) for i in dataframe.index]),\
                        np.array([np.array(dataframe['bin_sp_curr'][i]) for i in dataframe.index]), axis=1)
    mean_spiking = np.mean(spiking, axis=(0,1))
    mean_spikingpertime = np.mean(spiking, axis=0)
    std_spikingpertime = np.std(spiking, axis=0)
    std_spikingpertime[std_spikingpertime < 0.1] = 0.1
    # z-score to FR of each neuron @ each time (or divide by mean)
    # X_shuffle = spiking/mean_spiking
    X_shuffle = (spiking - mean_spikingpertime) / std_spikingpertime

    # get hemifield of targets
    hemifield = dataframe['hemifield'].values
    num_neurons = X_shuffle.shape[2]
    
    timing = range(borders_full[0], borders_full[15]+1)
    
    number_perm = 50
    number_splits = 10
    
    # initialize output
    labels = ['mse_'+neuron_type, 'mse_ipsi_'+neuron_type, 'mse_contra_'+neuron_type, 'mse_border_'+neuron_type,\
              'base_'+neuron_type, 'base_ipsi_'+neuron_type, 'base_contra_'+neuron_type, 'base_border_'+neuron_type,\
              'base_std_'+neuron_type, 'base_std_ipsi_'+neuron_type, 'base_std_contra_'+neuron_type, 'base_std_border_'+neuron_type,\
                          'weights_'+neuron_type, 'num_targets', 'hemifield']
    acc_crosscorr = {label: np.zeros((number_splits, len(timing), len(timing))) for label in labels}
    acc_crosscorr['weights_'+neuron_type] = np.zeros((number_splits, len(timing), num_neurons), dtype=np.complex_)
    acc_crosscorr['num_targets'] = np.zeros((number_splits))
    acc_crosscorr['hemifield'] = [['ipsi' if neuron_type==hemi\
                                      else 'border' if hemi=='border'\
                                        else 'contra' for hemi in hemifield] for i in range(number_splits)]

    for crossval in range(number_splits): # crossvalidate results
    
        rand_state = randint(0,100000)# to get the same split for training, testing
        split=0.2
        
        for t_id, delta_t_train in enumerate(timing):# for trial start to target on, current trial
            # create training dataset: columns=neurons, rows=trials for previous/current trials
        
            X = np.array([X_shuffle[n][delta_t_train] for n in dataframe.index])
            # TODO remove/add intercept
            #X = np.array([np.append(1, X[i]) for i in range(len(X))]) #add intercept
            
            # get test set (same split as previously with same random state)
            y_train ,y_test , hemifield_train, hemifield_test = train_test_split(y, hemifield, test_size = split, random_state = rand_state)
            X_train, _  = train_test_split(X, test_size = split, random_state = rand_state)
            
            
        #########################################################################################
        #                                        TRAIN MODEL                                    #
        #########################################################################################
            # linear regression weights
            weights = np.linalg.pinv(X_train.T.dot(X_train)).dot(X_train.T).dot(y_train)     

            acc_crosscorr['weights_'+neuron_type][crossval,t_id,:] = weights 
            acc_crosscorr['num_targets'][crossval] = dataframe.num_targ.values[0]
        
        #########################################################################################
        #                     GENERAL : calculate decoder for each timestep                     #
        #########################################################################################
            for t_idtest, delta_t_test in enumerate(timing):# for trial start to target on, current trial
                # create training dataset: columns=neurons, rows=trials for previous/current trials
            
                # create testing in all other time steps
                X_testing = np.array([X_shuffle[n][delta_t_test] for n in dataframe.index])
                # TODO remove/add intercept
                #X_testing = np.array([np.append(1, X_testing[i]) for i in range(len(X_testing))]) #add intercept
                
                # get test set (same split as previously with same random state)
                _, X_test = train_test_split(X_testing, test_size = split, random_state = rand_state)
    
                # which trials are ipsilateral/contralateral to decoded hemisphere
                ipsi_idx = np.where(hemifield_test==neuron_type)[0]
                contra_idx = np.where((hemifield_test!=neuron_type) & (hemifield_test!='border'))[0]
                border_idx = np.where(hemifield_test=='border')[0]

                # get predictions (target decoder)
                acc_crosscorr['mse_' + neuron_type][crossval, t_id, t_idtest] = np.mean(circdist(np.angle(X_test.dot(weights)), \
                                                                              np.angle(y_test)) ** 2, axis=0)
                acc_crosscorr['mse_ipsi_'+neuron_type][crossval, t_id, t_idtest] = np.mean(circdist(np.angle(X_test[ipsi_idx].dot(weights)),\
                                                                              np.angle(y_test[ipsi_idx]))**2, axis=0)
                acc_crosscorr['mse_contra_'+neuron_type][crossval, t_id, t_idtest] = np.mean(circdist(np.angle(X_test[contra_idx].dot(weights)),\
                                                                              np.angle(y_test[contra_idx]))**2, axis=0)
                acc_crosscorr['mse_border_' + neuron_type][crossval, t_id, t_idtest] = np.mean(circdist(np.angle(X_test[border_idx].dot(weights)), \
                                                                              np.angle(y_test[border_idx])) ** 2, axis=0)

        
            #########################################################################################
            #                     BASELINE : calculate baseline for each shuffle                    #
            #########################################################################################
            
                y_test_forshuffle = copy.deepcopy(y_test)
                y_test_forshuffle_ipsi = copy.deepcopy(y_test[ipsi_idx])
                y_test_forshuffle_contra = copy.deepcopy(y_test[contra_idx])
                y_test_forshuffle_border = copy.deepcopy(y_test[border_idx])

                baseline=[]
                baseline_ipsi=[]
                baseline_contra=[]
                baseline_border=[]

                # shuffle target labels for baseline
                for p in range(number_perm): #permute:    
                
                    # random shuffle of test labels
                    np.random.shuffle(y_test_forshuffle)
                    np.random.shuffle(y_test_forshuffle_ipsi)
                    np.random.shuffle(y_test_forshuffle_contra)
                    np.random.shuffle(y_test_forshuffle_border)
                    
                    # make predictions of models on shuffled labels
                    baseline.append(np.mean((circdist(np.angle(X_test.dot(weights)), \
                                                           np.angle(y_test_forshuffle))) ** 2))
                    baseline_ipsi.append(np.mean((circdist(np.angle(X_test[ipsi_idx].dot(weights)),\
                                                           np.angle(y_test_forshuffle_ipsi)))**2))
                    baseline_contra.append(np.mean((circdist(np.angle(X_test[contra_idx].dot(weights)),\
                                                           np.angle(y_test_forshuffle_contra)))**2))
                    baseline_border.append(np.mean((circdist(np.angle(X_test[border_idx].dot(weights)), \
                                                             np.angle(y_test_forshuffle_border))) ** 2))

                
                #save mean/std for shuffle
                acc_crosscorr['base_'+neuron_type][crossval, t_id, t_idtest] = np.mean(baseline)
                acc_crosscorr['base_ipsi_'+neuron_type][crossval, t_id, t_idtest] = np.mean(baseline_ipsi)
                acc_crosscorr['base_contra_'+neuron_type][crossval, t_id, t_idtest] = np.mean(baseline_contra)
                acc_crosscorr['base_border_'+neuron_type][crossval, t_id, t_idtest] = np.mean(baseline_border)
                acc_crosscorr['base_std_'+neuron_type][crossval, t_id, t_idtest] = np.std(baseline)
                acc_crosscorr['base_std_ipsi_'+neuron_type][crossval, t_id, t_idtest] = np.std(baseline_ipsi)
                acc_crosscorr['base_std_contra_'+neuron_type][crossval, t_id, t_idtest] = np.std(baseline_contra)
                acc_crosscorr['base_std_border_'+neuron_type][crossval, t_id, t_idtest] = np.std(baseline_border)

            
    # make to list for later assignment to dataframe
    acc_crosscorr = {label: list(acc_crosscorr[label]) for label in labels}            
                
    return acc_crosscorr

# zscore_contra = (acc_crosscorr['mse_contra_left']-np.array(acc_crosscorr['base_contra_left']))/\
#     np.array(acc_crosscorr['base_std_contra_left'])
# zscore_ipsi = (acc_crosscorr['mse_ipsi_left']-np.array(acc_crosscorr['base_ipsi_left']))/\
#     np.array(acc_crosscorr['base_std_ipsi_left'])
# zscore_border = (acc_crosscorr['mse_border_left']-np.array(acc_crosscorr['base_border_left']))/\
#     np.array(acc_crosscorr['base_std_border_left'])
# plt.plot(np.nanmean(zscore_ipsi,axis=0)[15]*(-1), label='ipsi')
# plt.plot(np.nanmean(zscore_contra,axis=0)[15]*(-1), label='contra')
# plt.plot(np.nanmean(zscore_border,axis=0)[15]*(-1), label='border')
# plt.legend()

# plt.plot(np.mean(acc_crosscorr['base_std_ipsi_left'],axis=0))
# plt.plot(np.mean(acc_crosscorr['base_std_contra_left'],axis=0))
##################################################################################################
#                                               LOAD DATA                                        #
##################################################################################################
with open(DATA_SMITH + 'df_dat_correct_Sa0.pickle', 'rb') as handle:
    df_dat_correct = pickle.load(handle)

with open(DATA_SMITH + 'leftRightIdx_Sa0.pickle', 'rb') as handle:
    leftRightIdx = pickle.load(handle)

monkeys=['Sa']#['Sa','Pe', 'Wa']

left_idx = {m: [[] for i in range(len(leftRightIdx['left']['Sa']))] for m in monkeys}
right_idx = {m: [[] for i in range(len(leftRightIdx['left']['Sa']))] for m in monkeys}
for m in monkeys:
    for n in range(len(leftRightIdx['left'][m])):
        left_idx[m][n] = leftRightIdx['left'][m][n]
        right_idx[m][n] = leftRightIdx['right'][m][n]
    
df_dat_correct['hemifield'] = ['left' if (df_dat_correct['targ_angle'][i]<np.round(-np.pi/2,5)) |\
                               (df_dat_correct['targ_angle'][i]>np.round(np.pi/2,5)) \
                               else 'right' if (df_dat_correct['targ_angle'][i]>np.round(-np.pi/2,5)) &\
                               (df_dat_correct['targ_angle'][i]<np.round(np.pi/2,5))\
                                else 'border' \
                               for i in df_dat_correct.index]

print('loaded data')
##################################################################################################
#                                               SPLIT DATA                                        #
##################################################################################################
# labels = ['mse_left', 'mse_ipsi_left', 'mse_contra_left', 'base_left', 'base_ipsi_left', 'base_contra_left',\
#               'base_std_left', 'base_std_ipsi_left', 'base_std_contra_left',\
#                  'mse_response_left', 'mse_ipsi_response_left', 'mse_contra_response_left',\
#           'base_response_left', 'base_ipsi_response_left', 'base_contra_response_left',\
#                       'base_std_response_left','base_std_ipsi_response_left', 'base_std_contra_response_left',\
#                           'weights_left', 'weights_response_left',\
#         'mse_right', 'mse_ipsi_right', 'mse_contra_right', 'base_right','base_ipsi_right', 'base_contra_right',\
#               'base_std_right', 'base_std_ipsi_right', 'base_std_contra_right',\
#                  'mse_response_right', 'mse_ipsi_response_right', 'mse_contra_response_right',\
#           'base_response_right', 'base_ipsi_response_right', 'base_contra_response_right',\
#                       'base_std_response_right', 'base_std_ipsi_response_right', 'base_std_contra_response_right',\
#                           'weights_right', 'weights_response_right','num_targets']
    
labels = ['monkey','session']
df_out = pd.DataFrame(columns=labels)
for mono in monkeys:# for each monkey
    for sess in range(max(df_dat_correct.loc[df_dat_correct.monkey==mono].session)+1):#for each session
        print(mono+str(sess))
        # choose session
        df_Sa0_corr = df_dat_correct.loc[(df_dat_correct['monkey']==mono) & (df_dat_correct['session']==sess)]
        
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
            for period in time_connections:#range(1,len(borders)-1):# from start_back for all time periods until trial_end
                #if period<5:
                number_bins.append(borders[t_borders2[period+1]][0]-borders[t_borders2[period]][0])
                number_bins_len.append(borders[t_borders2[period+1]][0])
                for t in range(borders[t_borders2[period+1]][0]-borders[t_borders2[period]][0]): # for number of time bins in discrete timings:           
                    # sum the matrix of neurons at timings in bin
                    binned_spikes.append(np.sum(df_Sa0_corr.loc[trial, 'n_mat'][:,borders[t_borders2[period]][idx]*bins+t*bins:borders[t_borders2[period]][idx]*bins+t*bins+bins].toarray(), axis=1))
            bin_sp_trials.append(binned_spikes)
        
        borders_full = [np.sum(abs(np.array(number_bins))[:i]) for i in range(len(number_bins)+1)]
        borders_prevcurr = np.append(borders_full, borders_full+max(borders_full))

        ###################################################################################
        #                                  SERIAL DEPENDENCE                              #              
        ###################################################################################

        serial = {'trial_id':[], 'target_prev_xy':[],'response_prev_xy': [], 'bin_sp_prev':[],'bin_sp_curr':[],\
                  'hemifield':[], 'monkey': [], 'session':[], 'num_targ':[]}
        df_dat_corr_reset = df_Sa0_corr.reset_index()
        
        for idx in df_dat_corr_reset.index[:-1]:# run through all correct trials (0,len)
            if df_dat_corr_reset.loc[idx,'trial_id']+1 == df_dat_corr_reset.loc[idx+1,'trial_id']: # only compare within one sesssion
                #print(df_dat.loc[df_dat_corr_reset.loc[idx,'index'], 'outcome'], df_dat.loc[df_dat_corr_reset.loc[idx+1,'index'], 'outcome'])
                serial['trial_id'].append(df_dat_corr_reset.trial_id[idx])
                serial['target_prev_xy'].append(df_dat_corr_reset['targ_xy'][idx])
                serial['response_prev_xy'].append(df_dat_corr_reset['saccade_xy'][idx])
                serial['monkey'].append(df_dat_corr_reset['monkey'][idx])
                serial['session'].append(df_dat_corr_reset['session'][idx])
                serial['bin_sp_prev'].append(bin_sp_trials[idx])
                serial['bin_sp_curr'].append(bin_sp_trials[idx+1])
                serial['hemifield'].append(df_dat_corr_reset['hemifield'][idx])
                serial['num_targ'].append(len(np.unique(df_dat_corr_reset.loc[(df_dat_corr_reset.monkey == df_dat_corr_reset.monkey[idx]) & (df_dat_corr_reset.session == df_dat_corr_reset.session[idx])].targ_angle)))
                
        df_serial = pd.DataFrame(serial)

        # SPLIT into left/right neurons
        left = np.where(left_idx[mono][sess]==1)[1]#
        right = np.where(right_idx[mono][sess]==1)[1]
                
        # create dataframe with only left neurons
        df_serial_left = df_serial.copy()# ['bin_sp_prev'][0][0][left]
        df_serial_left.drop(['bin_sp_prev'], axis=1)
        df_serial_left['bin_sp_prev'] = [[df_serial['bin_sp_prev'][n][t][left] for t in range(len(df_serial['bin_sp_prev'][n]))] for n in range(len(df_serial['bin_sp_prev']))]
        df_serial_left.drop(['bin_sp_curr'], axis=1)
        df_serial_left['bin_sp_curr'] = [[df_serial['bin_sp_curr'][n][t][left] for t in range(len(df_serial['bin_sp_curr'][n]))] for n in range(len(df_serial['bin_sp_curr']))]
        
        # only right neurons
        df_serial_right = df_serial.copy()# ['bin_sp_prev'][0][0][left]
        df_serial_right.drop(['bin_sp_prev'], axis=1)
        df_serial_right['bin_sp_prev'] = [[df_serial['bin_sp_prev'][n][t][right] for t in range(len(df_serial['bin_sp_prev'][n]))] for n in range(len(df_serial['bin_sp_prev']))]
        df_serial_right.drop(['bin_sp_curr'], axis=1)
        df_serial_right['bin_sp_curr'] = [[df_serial['bin_sp_curr'][n][t][right] for t in range(len(df_serial['bin_sp_curr'][n]))] for n in range(len(df_serial['bin_sp_curr']))]

##################################################################################################
#                                            DECODER                                             #
##################################################################################################
        print('start decoder')

        out  = delay_decoder(df_serial_left, borders_prevcurr, 'left')
        print('left decoder complete')

        out2  = delay_decoder(df_serial_right, borders_prevcurr, 'right')
        print('right decoder complete')
        # append left hemisphere/right hemisphere results together
        leftright = {**out,**out2}
        
        df_sess = pd.DataFrame(leftright)
        df_sess['monkey'] = mono
        df_sess['session'] = sess
        df_sess['borders_full'] = [list(borders_prevcurr) for i in df_sess.index]
        
        # append all monkeys, sessions into one dataframe
        df_out = df_out.append(df_sess)

        df_out.reset_index(inplace=True, drop=True)   
        with open('SameTimeDecoderMatrix_IpsiContra_bins200_zscored_borders_SaPeWa.pickle', 'wb') as handle:
            pickle.dump(df_out, handle)

        print('computed session '+str(mono)+str(sess))
