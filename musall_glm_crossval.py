# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 16:35:13 2021

@author: mmelin
"""

import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import ssm
from ssm.util import find_permutation
import scipy.io as sio
from musall_datainspection import formatSessions
import easygui as eg
import random

#%% important variables - Set the parameters of the GLM-HMM

num_states = 3        # number of discrete states
obs_dim = 1           # number of observed dimensions - choice
num_categories = 2    # number of categories for output - left or right
input_dim = 2         # input dimensions - stimulus and bias term
num_holdout = 4       # number of sessions to hold out for testing
N_iters = 1000 # maximum number of EM iterations. Fitting with stop earlier if increase in LL is below tolerance specified by tolerance parameter
datadir = 'C:\Data\churchland\musall_widefield_behavior_data'
print('\n\nNumber of holdout trials is ' + str(num_holdout))
#%% Get behavioral data - need to incoroporate stimulus and only correct trials

filelist = eg.fileopenbox(msg='choose files for analysis',default='C:\Data\churchland\musall_glm_fitting_data/',multiple=True)
num_sess = len(filelist)
tempind = random.sample(range(num_sess),num_holdout)
test_ind = np.zeros(num_sess,dtype = 'bool')
test_ind[tempind] = True
train_ind = ~test_ind
filelist = np.array(filelist) #convert to numpy array for logical indexing
train_filelist = filelist[train_ind]
test_filelist = filelist[test_ind]
allchoices, allstimsides, allnumtrials, inpts = formatSessions(train_filelist,input_dim) #takes file list and outputs formatted session data

#%% fit the GLM-HMM with MLE, use MAP for low trial number
mle_glmhmm = ssm.HMM(num_states, obs_dim, input_dim, observations="input_driven_obs", 
                   observation_kwargs=dict(C=num_categories), transitions="standard")

fit_ll_mle = mle_glmhmm.fit(allchoices, inputs=inpts, method="em", num_iters=N_iters, tolerance=10**-9) #maybe change params here

#now do maximum a priori estimation
prior_sigma = 2
prior_alpha = 2
map_glmhmm = ssm.HMM(num_states, obs_dim, input_dim, observations="input_driven_obs", 
             observation_kwargs=dict(C=num_categories,prior_sigma=prior_sigma),
             transitions="sticky", transition_kwargs=dict(alpha=prior_alpha,kappa=0))

fit_ll_map = map_glmhmm.fit(allchoices, inputs=inpts, method="em", num_iters=N_iters, tolerance=10**-4)
#%% Get the test trials and plot log liklihood

test_choices, test_allstimsides, test_allnumtrials, test_inpts = formatSessions(test_filelist,input_dim)
mle_test_ll = mle_glmhmm.log_likelihood(test_choices, inputs=test_inpts) 
map_test_ll = map_glmhmm.log_likelihood(test_choices, inputs=test_inpts) 

fig = plt.figure(figsize=(2, 2.5), dpi=80, facecolor='w', edgecolor='k')
loglikelihood_vals = [mle_test_ll, map_test_ll]
colors = ['Navy', 'Purple']
for z, occ in enumerate(loglikelihood_vals):
    plt.bar(z, occ, width = 0.8, color = colors[z])
plt.ylim((mle_test_ll-2, mle_test_ll+5))
plt.xticks([0, 1], ['mle', 'map'], fontsize = 10)
plt.xlabel('model', fontsize = 15)
plt.ylabel('loglikelihood - test trials', fontsize=15)