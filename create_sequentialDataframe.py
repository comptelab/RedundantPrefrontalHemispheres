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
import statsmodels.api as sm
from filepaths import *


##################################################################################################
#                                               LOAD DATA                                        #
##################################################################################################
def toComplex(radii, angles):
    return radii * exp(1j*angles)

with open(DATA_SMITH+'df_dat_correct.pickle', 'rb') as handle:
    df_dat = pickle.load(handle)
    
with open(DATA_SMITH+'leftRightIdx.pickle', 'rb') as handle:
    leftRightIdx = pickle.load(handle)

left_idx = {'Sa': [[] for i in range(len(leftRightIdx['left']['Sa']))], 'Pe':[[] for i in range(len(leftRightIdx['left']['Pe']))], 'Wa':[[] for i in range(len(leftRightIdx['left']['Wa']))]}
right_idx = {'Sa': [[] for i in range(len(leftRightIdx['left']['Sa']))], 'Pe':[[] for i in range(len(leftRightIdx['left']['Pe']))], 'Wa':[[] for i in range(len(leftRightIdx['left']['Wa']))]}
for m in ["Sa", "Pe", "Wa"]:
    for n in range(len(leftRightIdx['left'][m])):
        left_idx[m][n] = leftRightIdx['left'][m][n]
        right_idx[m][n] = leftRightIdx['right'][m][n]
        
df_dat['targ_angle'] = np.angle(np.array([complex(df_dat['targ_xy'][i][0],df_dat['targ_xy'][i][1]) for i in df_dat.index]))
df_dat['targ_angle'] = np.round(df_dat['targ_angle'],5)

df_dat['broke'] = [True if df_dat['outcome'][n]!='CORRECT' else False for n in df_dat.index]

df_dat['ITI'] = df_dat['trial_end']-df_dat['go_cue']
df_dat['ITI_prev'] = np.roll(df_dat['ITI'], 1)

df_dat_corr = df_dat.loc[df_dat['outcome']=='CORRECT']
df_dat_corr['saccade_angle'] = np.angle(np.array([complex(df_dat_corr['saccade_xy'][i][0],df_dat_corr['saccade_xy'][i][1]) for i in df_dat_corr.index]))


df_dat_correct = df_dat_corr.copy()


# determine how many target positions in each trial in varying sessions
numTargets = [[len(np.unique(df_dat_correct.loc[(df_dat_correct['monkey']==mono) \
                                             & (df_dat_correct['session']==sess)]['targ_angle']))\
            for sess in range(max(df_dat['session'].loc[df_dat['monkey']==mono])+1)]\
            for mono in ['Sa', 'Pe', 'Wa']]
numTargs = {'Sa': numTargets[0], 'Pe': numTargets[1], 'Wa':numTargets[2]}


# df_dat_correct['delay'] = np.round((df_dat_correct.go_cue.values-df_dat_correct.targ_off.values)/100)*100
# df_dat_correct.insert(7, 'error', circdist(df_dat_correct.saccade_angle.values, df_dat_correct.targ_angle.values))

# correct for quadrant errors
df_dat_corr_reset = df_dat_correct.copy().reset_index()
res_err = np.zeros((len(df_dat_corr_reset)))
z_all = np.zeros((len(df_dat_corr_reset)))
idx_sub=0
for mono_i,(m, df_mono) in enumerate(df_dat_corr_reset.groupby("monkey")): # each subject
    for sess_i,(sess, df_sess) in enumerate(df_mono.groupby("session")): # each subject   
        for dela_i,(dela, df_delay) in enumerate(df_sess.groupby("delay")): # each delay
            z_mean = [df_delay.groupby(['targ_angle']).error.mean()[df_delay.targ_angle[idx]] for idx in df_delay.index]
            for z_idx, idx in enumerate(df_delay.index):
                res_err[idx] = df_delay.error[idx] - z_mean[z_idx]
                z_all[idx] = z_mean[z_idx]

            df_delay.insert(8,'res_err',res_err[df_delay.index])
            df_delay.insert(5, 'res_response',circdist(df_delay.saccade_angle.values,z_mean))

df_dat_correct['error_nC'] = df_dat_correct.error
df_dat_correct['error'] = res_err
df_dat_correct['res_response'] = circdist(df_dat_correct.saccade_angle.values,z_all)

radii= np.ones(len(df_dat_correct))
df_dat_correct['complex_response'] = toComplex(radii, df_dat_correct['res_response'])

# solely determine hemifield based on target
df_dat_correct['hemifield'] = ['left' if (df_dat_correct['targ_angle'][i]<np.round(-np.pi/2,5)) | (df_dat_correct['targ_angle'][i]>np.round(np.pi/2,5)) else 'right' if (df_dat_correct['targ_angle'][i]>np.round(-np.pi/2,5)) & (df_dat_correct['targ_angle'][i]<np.round(np.pi/2,5)) else 'border' for i in df_dat_correct.index]
df_dat_correct['topdown'] = ['top' if (df_dat_correct['targ_angle'][i]<np.round(np.pi,5)) & (df_dat_correct['targ_angle'][i]>0) else 'down' if (df_dat_correct['targ_angle'][i]>np.round(-np.pi,5)) & (df_dat_correct['targ_angle'][i]<0) else 'border' for i in df_dat_correct.index]

for sub_i,(sub, subject) in enumerate(df_dat_correct.groupby("monkey")):
    cut_off = 1*np.std(subject.error)

    df_dat_correct.at[(abs(df_dat_correct.error)<cut_off) & (df_dat_correct.monkey==sub), 'hemifield'] = ['left' if (df_dat_correct['res_response'][i]<np.round(-np.pi/2,5)) | (df_dat_correct['res_response'][i]>np.round(np.pi/2,5)) else 'right' if (df_dat_correct['res_response'][i]>np.round(-np.pi/2,5)) & (df_dat_correct['res_response'][i]<np.round(np.pi/2,5)) else 'border' for i in df_dat_correct[(abs(df_dat_correct.error)<cut_off) & (df_dat_correct.monkey==sub)].index]

    df_dat_correct.at[(abs(df_dat_correct.error)<cut_off) & (df_dat_correct.monkey==sub), 'topdown'] = ['top' if (df_dat_correct['res_response'][i]<np.round(np.pi,5)) & (df_dat_correct['res_response'][i]>0) else 'down' if (df_dat_correct['res_response'][i]>-np.round(-np.pi,5)) & (df_dat_correct['res_response'][i]<0) else 'border' for i in df_dat_correct[(abs(df_dat_correct.error)<cut_off) & (df_dat_correct.monkey==sub)].index]

##################################################################################################
#                                               SPLIT DATA                                        #
##################################################################################################

df_serial_all = pd.DataFrame()
for mon in ['Sa', 'Pe', 'Wa']:#
    for sess in range(0,max(df_dat['session'].loc[df_dat['monkey']==mon])+1): 

        print(sess)
        #only use Sa, sess0
        #df_Sa0 = df_dat.loc[(df_dat['monkey']==mono) & (df_dat['session']==sess)]
        
        df_Sa0_corr = df_dat_correct.loc[(df_dat_correct['monkey']==mon) & (df_dat_correct['session']==sess)]
        
        ##################################################################################################
        #                          SERIAL DATAFRAME WITH CORRECTED ERRORS                                #
        ##################################################################################################
        serial = {'trial_id':[], 'target_xy_prevprev':[], 'target_prev': [], 'target_xy_prev':[],'response_prev': [],\
                  'response_prev_uncorrected': [],'err_prev':[],'err_prev_uncorrected':[],\
                  'response_xy_prev': [],'response_xy_prev_uncorrected': [], 'delay_prev': [],\
                  'target_curr': [],'target_xy_curr': [],\
                  'response_curr': [],'response_curr_uncorrected': [],\
                  'response_xy_curr':[], 'response_xy_curr_uncorrected':[], 'err':[], 'err_uncorrected':[],\
                  'rel_loc':[],\
                  'delay_curr': [], 'ITI':[],'ITI_prev':[], 'broke':[],\
                     'hemifield_prev':[],\
                          'hemifield_curr':[],'topdown_prev':[], 'topdown_curr':[],\
                              'monkey': [], 'session':[], 'num_targ':[]}
        df_dat_corr_reset = df_Sa0_corr.reset_index()

        for i, idx in enumerate(df_dat_corr_reset.index[:-1]):  # run through all correct trials (0,len)
            if df_dat_corr_reset.loc[idx, 'trial_id'] + 1 == df_dat_corr_reset.loc[idx + 1, 'trial_id']:                    #print(df_dat.loc[df_dat_corr_reset.loc[idx,'index'], 'outcome'], df_dat.loc[df_dat_corr_reset.loc[idx+1,'index'], 'outcome'])
                    serial['trial_id'].append(df_dat_corr_reset.trial_id[idx])
                    serial['target_prev'].append(df_dat_corr_reset['targ_angle'][idx])
                    serial['target_xy_prevprev'].append(df_dat_corr_reset['targ_xy'][df_dat_corr_reset.index[i-1]])
                    serial['target_xy_prev'].append(df_dat_corr_reset['targ_xy'][idx])
                    serial['response_prev'].append(df_dat_corr_reset['res_response'][idx])
                    serial['response_prev_uncorrected'].append(df_dat_corr_reset['saccade_angle'][idx])
                    serial['response_xy_prev'].append(df_dat_corr_reset['complex_response'][idx])
                    serial['response_xy_prev_uncorrected'].append(df_dat_corr_reset['saccade_xy'][idx])
                    serial['err_prev'].append(df_dat_corr_reset['error'][idx])
                    serial['err_prev_uncorrected'].append(df_dat_corr_reset['error_nC'][idx])
                    serial['delay_prev'].append(df_dat_corr_reset['go_cue'][idx]-df_dat_corr_reset['targ_off'][idx])
                    serial['target_curr'].append(df_dat_corr_reset['targ_angle'][idx+1])
                    serial['target_xy_curr'].append(df_dat_corr_reset['targ_xy'][idx+1])
                    serial['response_curr'].append(df_dat_corr_reset['res_response'][idx+1])
                    serial['response_curr_uncorrected'].append(df_dat_corr_reset['saccade_angle'][idx+1])
                    serial['response_xy_curr'].append(df_dat_corr_reset['complex_response'][idx+1])
                    serial['response_xy_curr_uncorrected'].append(df_dat_corr_reset['saccade_xy'][idx+1])
                    serial['err'].append(df_dat_corr_reset['error'][idx+1])
                    serial['err_uncorrected'].append(df_dat_corr_reset['error_nC'][idx+1])
                    serial['rel_loc'].append(circdist(df_dat_corr_reset['targ_angle'][idx],\
                                                      df_dat_corr_reset['targ_angle'][idx+1])[0])
                    serial['delay_curr'].append(df_dat_corr_reset['go_cue'][idx+1]-df_dat_corr_reset['targ_off'][idx+1])
                    serial['ITI'].append((df_dat_corr_reset['trial_end'][idx]-df_dat_corr_reset['go_cue'][idx])+np.sum(df_dat[df_dat_corr_reset.loc[idx,'index']+1:df_dat_corr_reset.loc[idx+1,'index']]['reward']) + (df_dat_corr_reset['targ_on'][idx+1]-df_dat_corr_reset['start'][idx+1]))# ITI time is time after reward + broken off fixations
                    serial['ITI_prev'].append(df_dat_corr_reset['ITI_prev'][idx])
                    serial['broke'].append(df_dat_corr_reset.loc[idx+1,'index']- (df_dat_corr_reset.loc[idx,'index']+1))# how many broken trials btwn 2 correct trials
                    serial['monkey'].append(df_dat_corr_reset['monkey'][idx])
                    serial['session'].append(df_dat_corr_reset['session'][idx])
                    serial['hemifield_prev'].append(df_dat_corr_reset['hemifield'][idx])
                    serial['hemifield_curr'].append(df_dat_corr_reset['hemifield'][idx+1])
                    serial['topdown_prev'].append(df_dat_corr_reset['topdown'][idx])
                    serial['topdown_curr'].append(df_dat_corr_reset['topdown'][idx+1])
                    serial['num_targ'].append(len(np.unique(df_dat_corr_reset.loc[(df_dat_corr_reset.monkey == df_dat_corr_reset.monkey[idx]) & (df_dat_corr_reset.session == df_dat_corr_reset.session[idx])].targ_angle)))

        df_serial = pd.DataFrame(serial)
        df_serial_all = df_serial_all.append(df_serial)

np.set_printoptions(threshold=sys.maxsize)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None) 
import pickle
with open('df_serial.pickle', 'wb') as handle:
    pickle.dump(df_serial_all, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
#df_serial_all.reset_index()        
#df_serial_all.to_csv('df_serial_all.csv')
