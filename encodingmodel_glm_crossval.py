# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 16:35:13 2021

@author: mmelin
"""
def main():

    import os
    os.chdir(r'C:\Data\churchland\encodingmodel_GLM-HMM')
    
    import numpy as np
    import numpy.random as npr
    import matplotlib.pyplot as plt
    import ssm
    from ssm.util import find_permutation
    import scipy.io as sio
    from encodingModel_datainspection import formatSessions
    import random
    from datetime import datetime
    import glob
    import multiprocessing as mp
    from multiprocessing import Pool
    from sklearn.model_selection import train_test_split
    
    #%% important variables - Set the parameters of the GLM-HMM
    
    obs_dim = 1           # number of observed dimensions - choice
    num_categories = 2    # number of categories for output - left or right
    input_dim = 2         # input dimensions - stimulus and bias term
    tol = 10**-9          # specify tolerance, default value 10**-9
    N_iters = 1000 # maximum number of EM iterations. default 200 Fitting with stop earlier if increase in LL is below tolerance specified by tolerance parameter
    
    percent_test = .2     # percent of sessions to hold out for testing
    max_states = 6        #iterate thru states from 1 to max states
    num_states = np.arange(1,max_states+1)
    saving = False
    subselect = True
    dpath = 'X:/Widefield'
    mouse = 'mSM63'
    modality = 'audio'
    
    subselectarray = np.arange(21,31) #mSM63 all discrimination audio sessions, arange doesnt include end point
    #subselectarray = np.arange(16,36) #mSM64 all discrimination audio sessions
    #subselectarray = np.arange(12,28) #msm65 all discrimination audio sessions
    #subselectarray = np.arange(13,29) #mSM66 all discrimination audio sessions

    #%% Get behavioral data - need to incoroporate stimulus and only correct trials
    if saving:
        savename = input('Enter a name for saving data: ') # name for saved .mat file
    
    print('\nUsing ONLY ' + modality + ' sessions')
    datadir = os.path.join(dpath,mouse,'SpatialDisc')
    session_dates = [name for name in os.listdir(datadir) if os.path.isdir(os.path.join(datadir, name))] #get directories that contain behavioral session data
    session_dates = sorted(session_dates,key=lambda date: datetime.strptime(date[:11], "%d-%b-%Y")) #parse and sort dates in order
    
    if subselect: #allows the user to select a subset of sessions to train from. Ideally the user will only select from the desired modality, but sessions from different modalities will be discarded anyways
        for i in range(len(session_dates)):
            print(str(i) + ': ' + session_dates[i])
        session_dates = [session_dates[x] for x in subselectarray]
        print('\nUser has selected ' + str(session_dates) + '\n')
    
    
    numsess = len(session_dates)        
    bhv_dir = [os.path.join(datadir,date)  for date in session_dates] #directories with behavior files
    bhv_file_paths = [glob.glob(os.path.join(directory,'*_Session*.mat'))[0] for directory in bhv_dir] #list of full paths to behavior files
    
    
    used_sessions, allchoices, allstimsides, allnumtrials, inpts, alltargstim, alldiststim = formatSessions(bhv_file_paths,input_dim,modality) #takes file list and outputs formatted session data
    
    np_session_dates = np.asarray(session_dates) #convert to np array for logical indexing
    session_dates = np_session_dates[used_sessions == 1] #the sessions in the desired modality, this has to be done AFTER subset selection because subset selection occurs over all modalities. otherwise indexing would be messed up. 
    session_dates = session_dates.tolist()
    
    #now split to training and testing datasets
    
    stateval = random.randint(1,10000) #pick a random state integer for consistent splitting across data
    
    training_sessions, testing_sessions = train_test_split(session_dates, test_size=percent_test, random_state=stateval)
    training_choices, testing_choices = train_test_split(allchoices, test_size=percent_test, random_state=stateval)
    training_stimsides, testing_stimsides = train_test_split(allstimsides, test_size=percent_test, random_state=stateval)
    training_numtrials, testing_numtrials = train_test_split(allnumtrials, test_size=percent_test, random_state=stateval)
    training_inpts, testing_inpts = train_test_split(inpts, test_size=percent_test, random_state=stateval)

    print('\nNumber of total sessions is ' + str(len(session_dates)) + '\n\n')
    print('\n\nNumber of holdout sessions is ' + str(len(testing_numtrials)))
    
    
    #%% fit the GLM-HMM with MLE and MAP - single threaded
    
    #num_states = np.array([1,2]) #for testing purposes only   
    
    prior_sigma = 2
    prior_alpha = 2
    
    mlemodels = []
    mlefits = []
    mapmodels = []
    mapfits = []
    
    for i in num_states: #ADD TRAINING CALCULATION HERE
        print('\nrunning for ' + str(i) + ' states.')
        #do ML estimation
        mle_glmhmm = ssm.HMM(i, obs_dim, input_dim, observations="input_driven_obs", 
                           observation_kwargs=dict(C=num_categories), transitions="standard")
        
        fit_ll_mle = mle_glmhmm.fit(training_choices, inputs=training_inpts, method="em", num_iters=N_iters, tolerance=tol) #maybe change params here
        
        #now do maximum a posteriori estimation
        map_glmhmm = ssm.HMM(i, obs_dim, input_dim, observations="input_driven_obs", 
                     observation_kwargs=dict(C=num_categories,prior_sigma=prior_sigma),
                     transitions="sticky", transition_kwargs=dict(alpha=prior_alpha,kappa=0))
        
        fit_ll_map = map_glmhmm.fit(training_choices, inputs=training_inpts, method="em", num_iters=N_iters, tolerance=tol)
        
        fig = plt.figure(figsize=(4, 3), dpi=80, facecolor='w', edgecolor='k')
        plt.plot(fit_ll_mle, label='MLE')
        plt.plot(fit_ll_map, label='MAP')
        plt.legend(loc="lower right")
        plt.title(str(i) + ' states')
        plt.xlabel("EM Iteration")
        plt.ylabel("Log Probability")
        plt.show()
        
        mlemodels.append(mle_glmhmm)
        mlefits.append(fit_ll_mle)
        mapmodels.append(map_glmhmm)
        mapfits.append(fit_ll_map)
    
    #%% fit the GLM-HMM with MLE and MAP - multiprocessing pool
    
# =============================================================================
#     
#     from ParallelModule import ptrain_models as worker_function
#     
#     
#     with mp.Pool(10) as p:
#         result = p.map(worker_function, num_states)
#             
#     
# =============================================================================
    
    #%%
    
    
    #%% Get the test trials and plot log liklihood
    mle_test_ll = []
    map_test_ll = []
    mle_train_ll = []
    map_train_ll = []
    
    for i in range(len(num_states)):
        mle_test_ll.append(mlemodels[i].log_likelihood(testing_choices, inputs=testing_inpts))
        map_test_ll.append(mapmodels[i].log_likelihood(testing_choices, inputs=testing_inpts))
        mle_train_ll.append(mlemodels[i].log_likelihood(training_choices, inputs=training_inpts))
        map_train_ll.append(mapmodels[i].log_likelihood(training_choices, inputs=training_inpts))
    
    fig = plt.figure(figsize=(2, 2.5), dpi=200, facecolor='w', edgecolor='k')
    
    plt.plot(num_states, mle_test_ll, '--b', label='MLE - test')
    plt.plot(num_states, map_test_ll, '--r', label='MAP - test')
    #plt.plot(num_states, mle_train_ll, 'b', label='MLE - train')
    #plt.plot(num_states, map_train_ll, 'r', label='MAP - train')
    
    plt.xticks(num_states, fontsize = 10)
    plt.xlabel('# States', fontsize = 15)
    plt.ylabel('Log-likelihood', fontsize=15)
    plt.legend()
    #plt.yaxis.set_visible(False)

if __name__ == '__main__': 
    main()