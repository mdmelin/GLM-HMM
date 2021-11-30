# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 16:48:28 2021

@author: mmelin
"""

import numpy as np
import matplotlib.pyplot as plt
import ssm
import scipy.io as sio
from musall_datainspection import formatSessions,formatSessions_chaoqun
import easygui as eg
import os
import pickle

#%% important variables - Set the parameters of the GLM-HMM

num_states = 3        # number of discrete states
obs_dim = 1           # number of observed dimensions - choice
num_categories = 2    # number of categories for output - left or right
input_dim = 2         # input dimensions - stimulus and bias term
tol = 10**-9 # tolerance, default value 10**-9
N_iters = 1000 # maximum number of EM iterations. Fitting with stop earlier if increase in LL is below tolerance specified by tolerance parameter
mthd = 'map' # training method - either 'em' or 'map'
saving = False
#%% Get behavioral data 
datapath = 'C:\\Data\\churchland\\musall_glm_fitting_data\\'
mouse = 'mSM63_chaoqun2'
filelist = eg.fileopenbox(msg='choose files for analysis',default=datapath,multiple=True)
allchoices, allstimsides, inpts = formatSessions_chaoqun(filelist,input_dim) #takes file list and outputs formatted session data

#%% Get previously trained GLM-HMM

modelpath = os.path.join(datapath,mouse,mouse+'.pickle') 
with open(modelpath, 'rb') as handle:
    glmhmm = pickle.load(handle)

#%% Get expected states

#reformat inpt and allchoices to proper format
inptslist = [inpts]
allchoices = np.array(np.expand_dims(allchoices,1))
allchoices = allchoices.astype('int32')
allchoiceslist = [allchoices]

posterior_probs = [glmhmm.expected_states(data=data, input=inpt)[0]
                for data, inpt
                in zip(allchoiceslist, inptslist)]

#%%
fig = plt.figure(figsize=(5, 2.5), dpi=80, facecolor='w', edgecolor='k')
cols = ['#ff7f00', '#4daf4a', '#377eb8', '#ff0000', '#ff7f00', '#ff7f20']
for k in range(num_states):
    plt.plot(posterior_probs[0][:, k], label="State " + str(k + 1), lw=2,
             color=cols[k])
plt.ylim((-0.01, 1.01))
plt.yticks([0, 0.5, 1], fontsize = 10)
plt.xlabel("trial #", fontsize = 15)
plt.ylabel("p(state)", fontsize = 15)

#%% save expected states
savedir = os.path.split(filelist[0])[0]
savedic = {'expected_states':posterior_probs[0]}
sio.savemat(os.path.join(datapath, mouse,'expected_states.mat'),savedic)

