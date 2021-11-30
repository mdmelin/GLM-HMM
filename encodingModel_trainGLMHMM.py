# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 15:15:40 2021

@author: mmelin

Trains the Ashwood GLM-HMM as an input to the Musall encoding model
"""
import numpy as np
import matplotlib.pyplot as plt
import ssm
import scipy.io as sio
import os
import pickle
import glob
from encodingModel_datainspection import formatSessions
#%% important variables - Set the parameters of the GLM-HMM

num_states = 3        # number of discrete states
obs_dim = 1           # number of observed dimensions - choice
num_categories = 2    # number of categories for output - left or right
input_dim = 2         # input dimensions - stimulus and bias term
tol = 10**-9 # tolerance, default value 10**-9
N_iters = 1000 # maximum number of EM iterations. Fitting with stop earlier if increase in LL is below tolerance specified by tolerance parameter
mthd = 'map' # training method - either 'em' or 'map'

saving = True
dpath = 'Y:/Widefield'
mouse = 'mSM64'
modality = 'audio'

#%% Get behavioral data - need to incoroporate stimulus and only correct trials
if saving:
        savename = input('Enter a name for saving data: ') # name for saved .mat file
print('\nUsing ONLY ' + modality + ' sessions')
datadir = os.path.join(dpath,mouse,'SpatialDisc')
session_dates = [name for name in os.listdir(datadir) if os.path.isdir(os.path.join(datadir, name))] #get directories that contain behavioral session data
numsess = len(session_dates)        
bhv_dir = [os.path.join(datadir,date)  for date in session_dates] #directories with behavior files
bhv_file_paths = [glob.glob(os.path.join(directory,'*_Session*.mat'))[0] for directory in bhv_dir] #list of full paths to behavior files
used_sessions, allchoices, allstimsides, allnumtrials, inpts, alltargstim, alldiststim = formatSessions(bhv_file_paths,input_dim,modality) #takes file list and outputs formatted session data
np_session_dates = np.asarray(session_dates) #convert to np array for logical indexing
session_dates = np_session_dates[used_sessions == 1]
session_dates = np.ndarray.tolist(session_dates) #convert back to list
#%% fit the GLM-HMM with MLE, use MAP for low trial number
if mthd == 'mle':
    glmhmm = ssm.HMM(num_states, obs_dim, input_dim, observations="input_driven_obs", 
                   observation_kwargs=dict(C=num_categories), transitions="standard")

    fit_ll = glmhmm.fit(allchoices, inputs=inpts, method='em', num_iters=N_iters, tolerance=tol) #maybe change params here

elif mthd == 'map':
    prior_sigma = 2
    prior_alpha = 2
    glmhmm = ssm.HMM(num_states, obs_dim, input_dim, observations="input_driven_obs", 
                 observation_kwargs=dict(C=num_categories,prior_sigma=prior_sigma),
                 transitions="sticky", transition_kwargs=dict(alpha=prior_alpha,kappa=0))
    
    fit_ll = glmhmm.fit(allchoices, inputs=inpts, method='em', num_iters=N_iters, tolerance=tol)

#%% Plotting of training iterations
fig = plt.figure(figsize=(4, 3), dpi=80, facecolor='w', edgecolor='k')
plt.plot(fit_ll, label=str(mthd))
plt.legend(loc="lower right")
plt.xlabel("EM Iteration")
plt.ylabel("Log Probability")
plt.show()

#%% Plot weights from generated models

fig = plt.figure(figsize=(4, 3), dpi=80, facecolor='w', edgecolor='k')
cols = ['#ff7f00', '#4daf4a', '#377eb8', '#ff0000', '#ff7f00', '#ff7f20']
recovered_weights = glmhmm.observations.params
for k in range(num_states):
    if k ==0:
        plt.plot(range(input_dim), recovered_weights[k][0], marker='o',
                 color=cols[k], linestyle='-',
                 lw=1.5, label="recovered from " + str(mthd))
        
    else:
        plt.plot(range(input_dim), recovered_weights[k][0], marker='o',
                 color=cols[k], linestyle='-',
                 lw=1.5, label="")
        plt.plot(range(input_dim), recovered_weights[k][0], color=cols[k],
                     lw=1.5,  label = '', linestyle = '--')
plt.yticks(fontsize=10)
plt.ylabel("GLM weight", fontsize=15)
plt.xlabel("covariate", fontsize=15)
plt.xticks([0, 1], ['stimulus', 'bias'], fontsize=12, rotation=45)
plt.axhline(y=0, color="k", alpha=0.5, ls="--")
plt.legend()
plt.title("Weight recovery", fontsize=15)


final_ll = glmhmm.log_likelihood(allchoices, inputs=inpts) 

#%% Plot transition matrices from generated models - could permute for prettier plotting

fig = plt.figure(figsize=(5, 2.5), dpi=80, facecolor='w', edgecolor='k')
recovered_trans_mat = np.exp(glmhmm.transitions.log_Ps)
plt.imshow(recovered_trans_mat, vmin=-0.8, vmax=1, cmap='bone')
for i in range(recovered_trans_mat.shape[0]):
    for j in range(recovered_trans_mat.shape[1]):
        text = plt.text(j, i, str(np.around(recovered_trans_mat[i, j], decimals=2)), ha="center", va="center",
                        color="k", fontsize=12)
plt.xlim(-0.5, num_states - 0.5)
plt.xticks(range(0, num_states), ('1', '2', '3'), fontsize=10)
plt.yticks(range(0, num_states), ('1', '2', '3'), fontsize=10)
plt.ylim(num_states - 0.5, -0.5)
plt.title(str(mthd), fontsize = 15)
plt.subplots_adjust(0, 0, 1, 1)

#%% Get expected states and do some plotting as a sanity check
#%%Importantly, the expected states skip over trials where a choice wasn't made. We can interpolate these values later in MATLAB. 

posterior_probs = [glmhmm.expected_states(data=data, input=inpt)[0]
                for data, inpt
                in zip(allchoices, inpts)]

for sess_id in range(len(session_dates)):
    fig = plt.figure(figsize=(5, 2.5), dpi=80, facecolor='w', edgecolor='k')
    print(session_dates[sess_id])
    
    for k in range(num_states):
        plt.plot(posterior_probs[sess_id][:, k], label="State " + str(k + 1), lw=2,
                 color=cols[k])
    plt.title('Session date: ' + session_dates[sess_id] + '. Index: ' + str(sess_id))
    plt.ylim((-0.01, 1.01))
    plt.yticks([0, 0.5, 1], fontsize = 10)
    plt.xlabel("trial #", fontsize = 15)
    plt.ylabel("p(state)", fontsize = 15)


#%% Saving
if saving:
    savedir = os.path.join(dpath, mouse, 'glm_hmm_models')
    savedic = {'glmhmm_params': glmhmm, 'method': mthd, 'model_training_sessions': session_dates, 'choices':allchoices, 'stimsides':allstimsides,'model_inputs':inpts, 'targ':alltargstim, 'dist': alldiststim, 'posterior_probs': posterior_probs}
    sio.savemat(os.path.join(savedir, savename + '.mat'),savedic)
    with open(os.path.join(savedir, savename + '.pickle'), 'wb') as handle:
        pickle.dump(glmhmm, handle, protocol=pickle.HIGHEST_PROTOCOL)