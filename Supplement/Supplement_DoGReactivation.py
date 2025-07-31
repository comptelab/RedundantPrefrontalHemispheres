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

numcores = multiprocessing.cpu_count()  # for parallel processing

import seaborn as sns
from scipy.stats import sem

# %matplotlib inline
# %config InlineBackend.figure_format = 'svg'

import matplotlib.pylab as pylab

plt.style.use("stylefile.mplstyle")

# ############################################################################
#                                LOAD DATA                                 #
############################################################################

# with open('/home/melanie/PycharmProjects/HemisphericWM/SingleTrialDecoding/Serial5FoldCrossvalSameTimeDecodingPrevCurr_ResponseOrthogonalized_bins300_zscorednoIntercept_randomShuffleControlprevcurrcorrected_SaPeWa_security.pickle', 'rb') as handle:
# repeated crossvalidation
with open(
        '/home/melanie/Desktop/PhD/2_Smith/Results/SingleTrialDecoding/Serial20times5FoldStratifiedCrossvalDelayDecodingPrevCurr_bins200_zscorednoIntercept_randomShuffleControlprevcurrcorrected_long.pickle',
        'rb') as handle:
    df_dat_corr = pickle.load(handle)

for mono in ['Sa', 'Pe', 'Wa']:  #
    dat = df_dat_corr.loc[df_dat_corr['monkey'] == mono].copy().reset_index(drop=True)

    timing = [range(dat.borders_full.values[i][17]-4, \
                       dat.borders_full.values[i][17]+1) for i in dat.index]
    mean_delay_err = [circmean(circdist(np.angle(dat.pred_complex_curr_combined[i][timing[i]]), \
                        np.angle(dat.shufflepred_complex_curr_combined[i][timing[i]])), low=-np.pi, high=np.pi) \
               for i in dat.index]

    # reativation loc
    Rtime = [range(dat.borders_full.values[i][13], dat.borders_full.values[i][14] + 1) for i in dat.index]
    reactivations = [np.angle(np.mean(dat['pred_complex_delay_combined'][i][Rtime[i]])) for i in dat.index]
    reactivationprev_curr = circdist(reactivations, np.angle(dat.targ_curr_xy))

    assert len(np.where(np.isnan(mean_delay_err))[0]) == 0, 'Not all trials have an error value.'
    dat['error'] = mean_delay_err
    dat['react_prev_curr'] = reactivationprev_curr

    ############################################################################
    #                                  DOG FIT                                 #
    ############################################################################

    # dog(prevcurr) (1st derivative of Gaussian) model with different sigma hyperparameters
    bic_dog1 = [];
    bicstd_dog1 = [];
    mse_dog1 = [];
    msestd_dog1 = [];
    s_dog1 = [];
    mixedmse_dog1 = []
    bic_dog1_session = []
    for s in np.arange(.3,3.0,.05):
        s = np.round(s, 2)
        print(mono + ', crossvalidating 1st derivative of Gaussian fits, sigma=', s)
        # fit model and save AIC
        dat['DoGfit'] = -hf.dog1(s, dat.react_prev_curr)

        # y, X = dmatrices('error ~ DoGfit',
        #                  data=dat, return_type='dataframe')
        # # cross-validate and save SSE (stratified by session), results = MSE, BIC
        # results = [hf.cross_validate(y, X, dat['session'].values) for i in range(
        #     1000)]  # Parallel(n_jobs=numcores-2)(delayed(hf.cross_validate)(y, X) for i in range(1000))# TODO! 1000
        #
        # mse_dog1.append(np.mean(results))
        # msestd_dog1.append(sem(results))

        s_dog1.append(s)
        # BIC (not crossvalidated)
        model = smf.ols('error ~ DoGfit', data=dat).fit()
        bic_dog1.append(model.bic)


    #msestd_dog1 = np.array(msestd_dog1)
    f, ax = plt.subplots(figsize=(2.4, 2.1))
    ax.set_xlabel('react. $\sigma$')
    ax.plot(s_dog1, bic_dog1, label='BIC per sess', color='darkorange')
    ax.axvline(np.around(s_dog1[np.argmin(bic_dog1)], 3), color='darkorange', dashes=[1, 1])
    ax.set_ylabel('BIC')
    ax.set_title('Optimal $\sigma$ '+mono+': '+str(np.around(s_dog1[np.argmin(bic_dog1)],3)))
    ax.ticklabel_format(axis='y', useOffset=False, style='sci')
    plt.tight_layout()
    plt.savefig('./REACTDoG1Fit_' + mono + '_Delay.svg')
    plt.show()

    sses = {'BICdog1': dict(zip(np.around(s_dog1, 3), bic_dog1))}  # , 'dog3': dict(zip(np.around(s_dog3,3), mse_dog3))}

    df_sses = pd.DataFrame(sses)
    # save stuff for supplementary figure 1
    df_sses.to_csv('./REACTDOG1Fit_' + mono + '_Delay.csv')

