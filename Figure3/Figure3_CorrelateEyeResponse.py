#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 14:54:17 2022

@author: melanie
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import *
from matplotlib.ticker import FormatStrFormatter
from scipy.sparse import csr_matrix
from sklearn.model_selection import KFold, StratifiedKFold
from circ_stats import *
from sklearn.model_selection import train_test_split
from filepaths import *
import copy
from pingouin import circ_corrcc



##################################################################################################
#                                               LOAD DATA                                        #
##################################################################################################
with open(DATA_SMITH + 'df_dat_correct_eyetrackerSa0.pickle', 'rb') as handle:
    df_dat_corr = pickle.load(handle)

monkeys=['Sa']#['Sa', 'Pe', 'Wa']
##################################################################################################
#                                      SPLIT  DATA INTO SESSIONS                                 #
##################################################################################################

df_out = pd.DataFrame()
for mono in monkeys:  # for each monkey #,'Pe','Wa'
    for sess in range(max(df_dat_corr.loc[df_dat_corr.monkey == mono].session) + 1):  # for each session
        print(mono+str(sess))

        # choose session
        df_Sa0_corr = df_dat_corr.loc[
            (df_dat_corr['monkey'] == mono) & (df_dat_corr['session'] == sess)].copy().reset_index(drop=True)

        if (mono=='Sa') & (sess==0):
            df_Sa0_corr = df_Sa0_corr.loc[25:].reset_index(drop=True)

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
        period_spikes = []
        bin_eyex = []
        bin_eyey = []
        for idx, trial in enumerate(df_Sa0_corr.index):  # for all trials
            binned_eyex = []
            binned_eyey = []
            number_bins = []
            number_bins_len = []
            for period in time_connections:  # range(1,len(borders)-1):# from start_back for all time periods until trial_end
                # if period<5:
                number_bins.append(borders[t_borders2[period + 1]][0] - borders[t_borders2[period]][0])
                number_bins_len.append(borders[t_borders2[period + 1]][0])
                for t in range(borders[t_borders2[period + 1]][0] - borders[t_borders2[period]][
                    0]):  # for number of time bins in discrete timings:
                    # sum the matrix of neurons at timings in bin
                    # BIN the eye position
                    binned_eyex.append(np.mean(df_Sa0_corr.loc[trial, 'eye_x'][
                                               borders[t_borders2[period]][idx] * bins + t * bins:
                                               borders[t_borders2[period]][
                                                   idx] * bins + t * bins + bins]))  # - fixavg_eyex[trial]
                    binned_eyey.append(np.mean(df_Sa0_corr.loc[trial, 'eye_y'][
                                               borders[t_borders2[period]][idx] * bins + t * bins:
                                               borders[t_borders2[period]][
                                                   idx] * bins + t * bins + bins]))  # - fixavg_eyey[trial]
            bin_eyex.append(binned_eyex)
            bin_eyey.append(binned_eyey)

        bin_eyex = np.array(bin_eyex)
        bin_eyey = np.array(bin_eyey)
        # remove mean in each time step (across trials to keep saccade)
        bin_eyex = bin_eyex - np.mean(bin_eyex, axis=0)
        bin_eyey = bin_eyey - np.mean(bin_eyey, axis=0)

        borders_full = [np.sum(abs(np.array(number_bins))[:i]) for i in range(len(number_bins) + 1)]
        borders_full = np.append(borders_full, borders_full + max(borders_full))

        df_Sa0_corr['eye_xy'] = [np.array([bin_eyex[i], bin_eyey[i]]).T for i in range(len(bin_eyex))]

        complex_eye_xy = np.array([np.array([complex(bin_eyex[trial][time], \
                                                     bin_eyey[trial][time]) \
                                             for time in range(len(bin_eyex[trial]))]) \
                                   for trial in range(len(bin_eyex))])

        complex_eye_full = np.append(complex_eye_xy, np.roll(complex_eye_xy,-1), axis=1)

        df_Sa0_corr['complex_eye_xy'] = list(complex_eye_full)

        # correlate angular response error with angular gaze error
        # for each timestep
        timing = range(borders_full[0], borders_full[15]+1)
        coef = np.empty((len(timing)))
        pval = np.empty((len(timing)))
        coefErr = np.empty((len(timing)))
        pvalErr = np.empty((len(timing)))
        for t in timing:
            gaze_angle = [np.angle(df_Sa0_corr['complex_eye_xy'][i][t]) for i in df_Sa0_corr.index]
            circtest = circ_corrcc(df_Sa0_corr['saccade_angle'].values, gaze_angle, correction_uniform=True)
            coef[t] = circtest[0]
            pval[t] = circtest[1]

            # error correlations
            gazeErr = circdist(gaze_angle, df_Sa0_corr['targ_angle'].values)
            responseErr = circdist(df_Sa0_corr['saccade_angle'].values, df_Sa0_corr['targ_angle'].values)
            circtestErr = circ_corrcc(responseErr, gazeErr, correction_uniform=True)
            coefErr[t] = circtestErr[0]
            pvalErr[t] = circtestErr[1]


        out = {'corr': list([coef]), 'pval': list([pval]), \
               'corrErr': list([coefErr]), 'pvalErr': list([pvalErr]), \
               'gaze': list([df_Sa0_corr['complex_eye_xy'].values]),
               'response': list([df_Sa0_corr['saccade_angle'].values]), \
                'real': list([df_Sa0_corr['targ_angle'].values]),
            'borders_full': list([borders_full])}
        #

        df_sess = pd.DataFrame(out)
        df_sess['monkey'] = [mono for i in df_sess.index]
        df_sess['session'] = [sess for i in df_sess.index]


        df_out = df_out.append(df_sess)
        df_out.reset_index(inplace=True, drop=True)
        with open('CorrelateEyeMovementsResponse_bins200.pickle', 'wb') as handle:
            pickle.dump(df_out, handle)

        print('computed session ' + str(mono) + str(sess))
