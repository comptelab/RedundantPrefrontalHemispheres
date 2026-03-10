#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 14:29:10 2021

@author: melanie
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.stats import *
from scipy.sparse import csr_matrix
import pickle
from scipy.io import loadmat
import glob
from pymicro.view.vol_utils import compute_affine_transform
import math
import io
from circ_stats import *
import copy
from filepaths import *

######################################################################
#                              LOAD DATA                             #
######################################################################
monkeys = ['Sa', 'Pe', 'Wa']
data = {m: [] for m in monkeys}
data_eye = {m: [] for m in monkeys}
for m in monkeys:
    files = np.sort(glob.glob('../../Desktop/PhD/2_Smith/Data/final/%s*.mat' % m))
    files_eye = np.sort(glob.glob('../../Desktop/PhD/2_Smith/Data/final/eye_%s*.mat' % m))
    # files = np.sort(glob.glob('../Data/final/%s*.mat' %m))
    # files_eye = np.sort(glob.glob('../Data/final/eye_%s*.mat' %m))
    for f in files:
        data[m].append(loadmat(f))
        print(f)
    for fe in files_eye:
        data_eye[m].append(loadmat(fe))
        print(fe)

########### open dataset ############################################

dataset = []
monkey = []
sessions = []
left_id = []
right_id = []
for m in monkeys:
    for n in range(len(data[m])):
        for l, line in enumerate(data[m][n]['dat'][0]):
            line = list(line)
            if m == 'Wa':
                # for Wakko, eye data and spiking data is saved in diff. files
                line_eye = data_eye[m][n]['dat'][0][l][5]
                line.insert(6, line_eye)
            else:
                # Sa, Pe, have a weird "samples" variable that Wakko doesn't have
                del line[6]
            # [print(row) if any([row.shape==(0,0), not isinstance(row, np.ndarray)]) else print(row.flat[0]) if len(row[0])==1 else print(row[0]) for row in line]
            dataset.append([row if any([row.shape == (0, 0), not isinstance(row, np.ndarray)]) \
                                else row.flat[0] if len(row[0]) == 1 else row[0] if len(row) == 1 \
                else row for row in line])
            monkey.append(m)
            sessions.append(n)

            if m == 'Sa':
                left_id.append(data[m][n]['left_idx'])
                right_id.append(data[m][n]['right_idx'])
            else:
                # neuron array information is flipped on Pepe and Wakko (found by Megan McDonnal,\
                # left labeled array is right and vice versa): (FIX HERE!))
                left_id.append(data[m][n]['right_idx'])
                right_id.append(data[m][n]['left_idx'])

columns = ['trial_id', 'sp_train', 'outcome', 'timing', 'targ_xy', 'targ_angle', 'eyetracker', \
           'saccade_xy', 'saccade_angle']

df_dat = pd.DataFrame(dataset, columns=columns)
df_dat['monkey'] = monkey
df_dat['session'] = sessions
df_dat['left_id'] = left_id
df_dat['right_id'] = right_id
df_dat['eye_x'] = [df_dat.loc[i, 'eyetracker'][0] for i in df_dat.index]
df_dat['eye_y'] = [df_dat.loc[i, 'eyetracker'][1] for i in df_dat.index]
df_dat['eye_xy'] = [np.array([df_dat.loc[i, 'eye_x'], df_dat.loc[i, 'eye_x']]).T for i in df_dat.index]

df_dat = df_dat.drop(columns='eyetracker')

#####################################################################
#            CORRECT TRIALS, CHANGE SESSIONS PEPE                   #
#####################################################################


df_dat['broke'] = [True if df_dat['outcome'][n] != 'CORRECT' else False for n in df_dat.index]

df_dat_correct = df_dat.loc[df_dat['outcome'] == 'CORRECT'].copy()

#  CHANGE PEPE SESSIONS, REASSIGN SESSIONS WITH TARGET SWITCHES    #

# split_sessions = [0, 305, 240, 240, 240, 240, 240, 240, 240, 240, 240, 300, 240, 0, 0, 0]
# df = df_dat_correct.loc[df_dat_correct.monkey=='Pe']
#
# helper = 0
# new_session=[]
# for s, session in enumerate(np.unique(df.session)):
#     print(helper)
#     df_sess = df.loc[df.session == session].reset_index()
#     new_session.append([helper if (idx<split_sessions[s]) | (split_sessions[s]==0)\
#                    else helper+1 for idx in df_sess.index])
#     if split_sessions[s]==0:
#         helper+=1
#     else:
#         helper+=2
#
# df_dat_correct.loc[df_dat_correct.loc[df_dat_correct.monkey=='Pe'].index, 'session'] = np.concatenate(new_session)

df_dat_correct.reset_index(inplace=True)

#####################################################################
#          SAVE NEURON INDICES (HEMISPHERE ASSIGNMENT)              #
#####################################################################

#### load indices of neurons

left_idx = {'Sa': [[] for i in range(max(df_dat_correct.loc[df_dat_correct['monkey'] == 'Sa'].session) + 1)], \
            'Pe': [[] for i in range(max(df_dat_correct.loc[df_dat_correct['monkey'] == 'Pe'].session) + 1)], \
            'Wa': [[] for i in range(max(df_dat_correct.loc[df_dat_correct['monkey'] == 'Wa'].session) + 1)]}
#
right_idx = {'Sa': [[] for i in range(max(df_dat_correct.loc[df_dat_correct['monkey'] == 'Sa'].session) + 1)], \
             'Pe': [[] for i in range(max(df_dat_correct.loc[df_dat_correct['monkey'] == 'Pe'].session) + 1)], \
             'Wa': [[] for i in range(max(df_dat_correct.loc[df_dat_correct['monkey'] == 'Wa'].session) + 1)]}
#

for m in ["Sa", "Pe", "Wa"]:  #
    for n in range(max(df_dat_correct.loc[df_dat_correct['monkey'] == m].session) + 1):
        df_help = df_dat_correct.loc[(df_dat_correct.monkey == m) & (df_dat_correct.session == n)].copy().reset_index(
            drop=True)
        left_idx[m][n] = df_help.left_id[0]
        right_idx[m][n] = df_help.right_id[0]

#### save indices

idx = {'left': left_idx, 'right': right_idx}
with open(DATA_SMITH + 'leftRightIdx_eyetracker.pickle', 'wb') as handle:
    pickle.dump(idx, handle, protocol=pickle.HIGHEST_PROTOCOL)
handle.close()

######################################################################
#                         PROCESS TASK TIMING                        #
######################################################################

taskperiods = ['start', 'fix', 'targ_on', 'targ_off', 'go_cue', 'saccade', 'reward']

helper = [np.squeeze(np.concatenate(df_dat_correct['timing'][n])) \
              if len(df_dat_correct['timing'][n]) > 1 \
              else np.squeeze(df_dat_correct['timing'][n]) for n in df_dat_correct.index]
df_dat_correct['timing'] = [
    np.append(helper[n], np.ones(len(taskperiods) - len(df_dat_correct['timing'][n])) * max(helper[n])) \
        if len(df_dat_correct['timing'][n]) < len(taskperiods) \
        else helper[n] for n in df_dat_correct.index]
for i in range(len(taskperiods)):
    df_dat_correct[taskperiods[i]] = [line[i].flat[0] for line in df_dat_correct['timing']]
del df_dat_correct['timing']

df_dat_correct['trial_end'] = [df_dat_correct['sp_train'][n].shape[1] for n in df_dat_correct.index]

#####################################################################
#               CORRECT FOR EYETRACKER ERRORS                       #
#####################################################################


# determine mean of x/y position for each monkey, each session
ref_points = {'Sa': [], 'Pe':[], 'Wa':[]}
tsr_points = {'Sa': [], 'Pe':[], 'Wa':[]}
eyetsr_points = {'Sa': [], 'Pe':[], 'Wa':[]}
new_sac = np.empty((len(df_dat_correct), 2)) * np.nan
new_sactime = [[] for i in range(len(df_dat_correct))]
new_eye = []
for m in ['Sa', 'Pe', 'Wa']:  #
    transf = []
    ref_centr = []
    tsr_centr = []
    # plt.figure()
    for n in range(max(df_dat_correct.loc[df_dat_correct['monkey'] == m].session) + 1):
        df_dat_trial = df_dat_correct[(df_dat_correct['monkey'] == m) & (df_dat_correct['session'] == n)]

        # session 0 has some early trials that are completely different (compute transformation on rest of trials
        idx_used = df_dat_trial.index
        num_used = np.arange(len(idx_used))
        if (m == 'Sa') & (n == 0):
            idx_used = idx_used[25:]
            num_used = num_used[25:]

        ref_x = [df_dat_trial.loc[i, 'targ_xy'][0] for i in df_dat_trial.index]
        ref_y = [df_dat_trial.loc[i, 'targ_xy'][1] for i in df_dat_trial.index]

        # create TRANSFORM
        ref_points[m].append(np.array(list(zip(ref_x, ref_y))))
        tsr_points[m].append(np.array(list(zip([df_dat_trial.loc[i, 'saccade_xy'][0] for i in df_dat_trial.index], \
                                               [df_dat_trial.loc[i, 'saccade_xy'][1] for i in df_dat_trial.index]))))
        eyetsr_points[m].append([list(zip(df_dat_trial.loc[i, 'eye_x'], df_dat_trial.loc[i, 'eye_y'])) \
                                 for i in df_dat_trial.index])

        # get the affine transformations that map eye tracker data to reference points
        # compute transformation only on final saccade points
        translation, transformation = compute_affine_transform(ref_points[m][n][num_used], tsr_points[m][n][num_used])

        transf.append(transformation)  # save for transformation of data points later

        # now get the centroid in each condition for recentering
        ref_centroid = np.mean(ref_points[m][n][num_used], axis=0)
        tsr_centroid = np.mean(tsr_points[m][n][num_used], axis=0)
        eyetsr_centroid = np.mean([np.mean(eyetsr_points[m][n][num_used[i]][df_dat_trial.loc[idxs, 'fix']: \
                                                                            df_dat_trial.loc[idxs, 'targ_on']], axis=0) \
                                   for i, idxs in enumerate(idx_used)], axis=0)

        ref_centr.append(ref_centroid)  # save for transformation of data points later
        tsr_centr.append(tsr_centroid)  # save for transformation of data points later

        # remap the eye tracker data using the transformation
        new_points = np.empty_like(ref_points[m][n])
        new_eyepoints = []
        # for each trial
        for t, trial in enumerate(df_dat_trial.index):
            # add new location offset from targets, subtract old offset and transform based on saccade data
            new_points[t] = ref_centroid + np.dot(transformation, tsr_points[m][n][t] - tsr_centroid)
            new_sac[trial] = ref_centroid + np.dot(transformation, tsr_points[m][n][t] - tsr_centroid)
            # do this for each time point of the continuous eye data
            new_eyepoints.append(np.array([ref_centroid + \
                                           np.dot(transformation, eyetsr_points[m][n][t][time] - eyetsr_centroid) \
                                           for time in range(len(eyetsr_points[m][n][t]))]))
            new_sactime[trial] = np.array([ref_centroid + \
                                           np.dot(transformation, eyetsr_points[m][n][t][time] - eyetsr_centroid) \
                                           for time in range(len(eyetsr_points[m][n][t]))])

        # visualize the match
        print(len(new_points))
        # plt.figure()
        # plt.subplot(121)
        # plt.axhline(color='k')
        # plt.axvline(color='k')
        # plt.scatter(tsr_points[m][n][:,0],tsr_points[m][n][:,1], alpha=0.2)
        # plt.scatter(ref_points[m][n][:,0], ref_points[m][n][:,1], marker='x')
        # plt.title('orig')
        # plt.subplot(122)
        # plt.axhline(color='k')
        # plt.axvline(color='k')
        # plt.scatter(new_points[:,0], new_points[:,1], alpha=0.2)
        # plt.scatter(ref_points[m][n][:,0], ref_points[m][n][:,1], marker='x')
        # plt.title('corrected')
        # plt.show()
        # # #
        # # plt.figure()
        # plt.subplot(121)
        # plt.axhline(color='k')
        # plt.axvline(color='k')
        # trials=range(40,240)
        # for tria in trials:
        #     plt.scatter([eyetsr_points[m][n][tria][time][0] for time in range(len(eyetsr_points[m][n][tria]))],\
        #                 [eyetsr_points[m][n][tria][time][1] for time in range(len(eyetsr_points[m][n][tria]))], alpha=0.2)
        #     plt.scatter(ref_points[m][n][tria, 0], ref_points[m][n][tria, 1], marker='x', color='k')
        # plt.scatter(eyetsr_centroid[0], eyetsr_centroid[1], marker='x', color='k')
        # plt.scatter(tsr_centroid[0], tsr_centroid[1], marker='x', color='darkblue')
        # plt.xlim([-400, 400])
        # plt.ylim(-400, 400)
        # plt.title('orig')
        # plt.subplot(122)
        # plt.axhline(color='k')
        # plt.axvline(color='k')
        # for tria in trials:
        #     plt.scatter(new_eyepoints[tria][:, 0], new_eyepoints[tria][:, 1], alpha=0.2)
        #     plt.scatter(ref_points[m][n][tria, 0], ref_points[m][n][tria, 1], marker='x', color='k')
        # plt.xlim([-400, 400])
        # plt.ylim(-400, 400)
        # plt.title('corrected')
        # plt.show()

        del df_dat_trial

    ## FROM NOW ON ONLY CONTAINS MEAN CENTERED DATA
df_dat_correct['saccade_xy'] = list(new_sac)

df_dat_correct['eye_xy'] = new_sactime
df_dat_correct['eye_x'] = [new_sactime[i][:, 0] for i in range(len(new_sactime))]
df_dat_correct['eye_y'] = [new_sactime[i][:, 1] for i in range(len(new_sactime))]

df_dat_correct['targ_angle'] = np.round(
    [np.arctan2(df_dat_correct.loc[i, 'targ_xy'][1], df_dat_correct.loc[i, 'targ_xy'][0]) \
     for i in df_dat_correct.index], 5)

df_dat_correct['saccade_angle'] = np.round(
    [np.arctan2(df_dat_correct.loc[i, 'saccade_xy'][1], df_dat_correct.loc[i, 'saccade_xy'][0]) \
     for i in df_dat_correct.index], 5)

x_start = []
x_label = []
clockw = []

for i, n in enumerate(df_dat_correct.index):
    if df_dat_correct['saccade_angle'][n] == np.nan:
        clockw.append('skip')
    elif circdist(df_dat_correct['targ_angle'][n] * np.pi / 180, df_dat_correct['saccade_angle'][n] * np.pi / 180) <= 0:
        clockw.append('CW')
    else:
        clockw.append('CCW')

df_dat_correct['clockw'] = clockw
df_dat_correct.head()

# determine how many target positions in each trial in varying sessions
numTargets = [[len(np.unique(df_dat_correct.loc[(df_dat_correct['monkey'] == mono) \
                                                & (df_dat_correct['session'] == sess)]['targ_angle'])) \
               for sess in range(max(df_dat_correct['session'].loc[df_dat_correct['monkey'] == mono]) + 1)] \
              for mono in monkeys]
numTargs = {'Sa': numTargets[0], 'Pe': numTargets[1], 'Wa':numTargets[2]}

# insert delay rounded to 100ms
df_dat_correct.insert(8, 'delay', np.round((df_dat_correct.go_cue.values - df_dat_correct.targ_off.values) / 100) * 100)
df_dat_correct.insert(7, 'error', circdist(df_dat_correct.saccade_angle.values, df_dat_correct.targ_angle.values))

# drop trials with largest errors
# (if error is larger than half the distance between furthest targets we consider it a wrong response
# as it is closer to the other target)
cut = np.deg2rad(22.5)
df_dat_correct.drop(np.where(abs(df_dat_correct.error) > cut)[0], inplace=True)

# with open(DATA_SMITH+'df_dat_correct_eyetrackerSa0.pickle', 'wb') as handle:
#     pickle.dump(df_dat_correct, handle, protocol=pickle.HIGHEST_PROTOCOL)
# handle.close()

df_dat_correct.to_pickle(DATA_SMITH + 'df_dat_correct_eyetracker.pickle', protocol=pickle.HIGHEST_PROTOCOL)
