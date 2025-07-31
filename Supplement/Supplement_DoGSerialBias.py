#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  3 13:49:36 2021

@author: Heike Stein
"""
# MODEL SELECTION AND HYPERPAR. CROSSVALIDATION, BEHAVIOR

# author: heike stein
# last mod: 23/04/20

import numpy as np
import pandas as pd
import json
from joblib import Parallel, delayed
import multiprocessing
import statsmodels.api as sm
import statsmodels.formula.api as smf
from patsy import dmatrices
import helpers as hf
import pickle
from filepaths import *
import matplotlib.pyplot as plt
from circ_stats import *
import seaborn as sns
import matplotlib.ticker as ticker

numcores    = multiprocessing.cpu_count() # for parallel processing

plt.style.use("stylefile.mplstyle")

import seaborn as sns
#%matplotlib inline
#%config InlineBackend.figure_format = 'svg'

# ############################################################################
#                                LOAD DATA                                 #
############################################################################

with open(DATA_SMITH + 'df_dat_correct.pickle', 'rb') as handle:
    df_dat_corr = pickle.load(handle)

with open(DATA_SMITH + 'leftRightIdx.pickle', 'rb') as handle:
    leftRightIdx = pickle.load(handle)

left_idx = {'Sa': [[] for i in range(len(leftRightIdx['left']['Sa']))], 'Pe':[[] for i in range(len(leftRightIdx['left']['Pe']))], 'Wa':[[] for i in range(len(leftRightIdx['left']['Wa']))]}
right_idx = {'Sa': [[] for i in range(len(leftRightIdx['left']['Sa']))], 'Pe':[[] for i in range(len(leftRightIdx['left']['Pe']))], 'Wa':[[] for i in range(len(leftRightIdx['left']['Wa']))]}
for m in ["Sa", "Pe", "Wa"]:
    for n in range(len(leftRightIdx['left'][m])):
        left_idx[m][n] = leftRightIdx['left'][m][n]
        right_idx[m][n] = leftRightIdx['right'][m][n]
        

# solely determine hemifield based on target
df_dat_corr['hemifield'] = ['left' if (df_dat_corr['targ_angle'][i]<np.round(-np.pi/2,5)) |\
                               (df_dat_corr['targ_angle'][i]>np.round(np.pi/2,5)) \
                               else 'right' if (df_dat_corr['targ_angle'][i]>np.round(-np.pi/2,5)) &\
                               (df_dat_corr['targ_angle'][i]<np.round(np.pi/2,5)) else 'border' \
                               for i in df_dat_corr.index]
    


serial = {'trial_id':[], 'target_prev': [], 'response_prev': [], 'target_curr': [],\
          'response_curr': [],\
          'monkey': [], 'session':[]}
df_dat_corr_reset = df_dat_corr.reset_index()

cut_off_time=5
for idx in df_dat_corr_reset.index[:-1]:# run through all correct trials (0,len)
    if df_dat_corr_reset.loc[idx,'trial_id'] < df_dat_corr_reset.loc[idx+1,'trial_id']: # only compare within one sesssion
        if np.sum(df_dat_corr[df_dat_corr_reset.loc[idx,'index']+1:df_dat_corr_reset.loc[idx+1,'index']]['trial_end'])<cut_off_time: # only use trials with less than cut_off ms between 2 correct trials
            serial['trial_id'].append(df_dat_corr_reset.trial_id[idx])
            serial['target_prev'].append(df_dat_corr_reset['targ_angle'][idx])
            serial['response_prev'].append(df_dat_corr_reset['saccade_angle'][idx])
            serial['target_curr'].append(df_dat_corr_reset['targ_angle'][idx+1])
            serial['response_curr'].append(df_dat_corr_reset['saccade_angle'][idx+1])
            serial['monkey'].append(df_dat_corr_reset['monkey'][idx])
            serial['session'].append(df_dat_corr_reset['session'][idx])

rel_loc = (circdist(serial['target_prev'],serial['target_curr']))# relative location current prvious stimulus
err = (circdist(serial['response_curr'],serial['target_curr']))# error current trial

sb = {'prevcurr': rel_loc, 'error': err, 'target_prev': serial['target_prev'], 'target_curr': serial['target_curr'],\
      'monkey': serial['monkey'], 'session': serial['session'], 'trial_id':serial['trial_id']}
df_sb = pd.DataFrame(sb)


for mono in ['Sa', 'Pe', 'Wa']:
    dat = df_sb.loc[df_sb['monkey']==mono].copy()

    ############################################################################
    #                                  DOG FIT                                 #
    ############################################################################
    
    # dog(prevcurr) (1st derivative of Gaussian) model with different sigma hyperparameters
    bic_dog1 = []; mse_dog1 = []; s_dog1 = []; mixedmse_dog1=[]
    for s in np.arange(.3,3.0,.05):
        print('crossvalidating 1st derivative of Gaussian fits', s)
        # fit model and save AIC
        s_dog1.append(s)
        dat['DoGfit'] =  -hf.dog1(s, dat.prevcurr)

        y, X    = dmatrices('error ~ DoGfit',
                  data=dat, return_type = 'dataframe')
        glm     = sm.OLS(y, X).fit()
        bic_dog1.append(glm.bic)
        # cross-validate and save SSE
        # mse     = [hf.cross_validate(y, X, dat['session'].values) for i in range(1000)]#Parallel(n_jobs=numcores-2)(delayed(hf.cross_validate)(y, X) for i in range(1000))# TODO! 1000
        # mse_dog1.append(np.mean(mse))

    
    f, ax = plt.subplots(figsize=(2.4, 2))
    # ax.plot(s_dog1,bic_dog1, label= 'OLS model', color='darkblue')
    # ax.axvline(np.around(s_dog1[np.argmin(bic_dog1)],3), color='darkblue', dashes=[1,1])
    ax.ticklabel_format(axis='y', useOffset=False, style='sci')
    ax.set_xlabel(' behav. $\sigma$')
    # ax.set_ylabel('Crossval. DoG MSE', color='darkblue')
    # ax.tick_params(axis='y', colors='darkblue')
    # flippedax = ax.twinx()
    ax.plot(s_dog1,bic_dog1, label='BIC', color='darkorange')
    ax.axvline(np.around(s_dog1[np.argmin(bic_dog1)],3), color='darkorange', dashes=[1,1])
    ax.set_ylabel('BIC')
    ax.set_title('Optimal $\sigma$ '+mono+': '+str(np.around(s_dog1[np.argmin(bic_dog1)],3)))
    # flippedax.tick_params(axis='y', colors='darkorange')

    plt.tight_layout()
    plt.savefig('./DoG1FitSD_'+mono+'.svg')
    plt.show()
    
    # # # find out best model/hyperparameter from crossvalidated SSE and add line to dataframe
    # if sse_dog1[np.argmin(sse_dog1)] < sse_dog3[np.argmin(sse_dog3)]:
    #     print('1st derivative of Gaussian fits data best')
    # else:
    #     print('3rd derivative of Gaussian fits data best')
    
    sses = {'bic_dog1': dict(zip(np.around(s_dog1,3), bic_dog1))}#, 'dog3': dict(zip(np.around(s_dog3,3), mse_dog3))}

    # save stuff for supplementary figure 1
    df_sses = pd.DataFrame(sses)
    # save stuff for supplementary figure 1
    df_sses.to_csv('./DOG1FitSD_' + mono + '_LateDelay.csv')
