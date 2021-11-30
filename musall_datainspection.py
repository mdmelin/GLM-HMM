# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 15:04:26 2021

@author: mmelin
"""
import numpy as np
import scipy.io as sio
import os
import glob
import scipy.stats as spstats



def main():
    datadir = 'C:\Data\churchland\musall_glm_fitting_data\mSM63_all'
    os.chdir(datadir)
    session_names = glob.glob('./*.mat')
    numsess = len(session_names)
    file = open("SessionData.txt", 'w')

    for i in range(numsess):
        name = session_names[i]
        print(name)
        name = name[1:]
        name = datadir + name
        numtrials, numcorrect, stimtype, stimside, percentcorrect, correcttrials, choice, stimdict, targstim, diststim, singlespout, assisted, modality, rewarded, optotype = getSessionData(name)
        writestring = ["\n\n", name[49:],
                       "\nAssisted = ",str(np.mean(assisted)),
                       "\nSinglespout = ",str(np.mean(singlespout)),
                       "\nDiststim mean (discrimination) = ",str(np.mean(diststim)),
                       "\nModality = ",str(np.mean(modality)),
                       "\nNumber of trials = ", str(numtrials),
                       "\nNumber of correct trials = ", str(numcorrect),
                       "\nPercent trials correct = ", str(percentcorrect)]
        file.writelines(writestring)
    file.close()

def weirdDivision(x,y):
    if y == 0:
        return 0
    return x/y

def getSessionData(filepath): # get data from a single session, add modality, assisted, singlespout, etc. 
    importfile = sio.loadmat(filepath)
    choice = importfile['SessionData']['ResponseSide'][0, 0] - 1
    
    numtrials = int(importfile['SessionData']['nTrials'])
    stimside = importfile['SessionData']['CorrectSide'][0, 0] - 1
    stimtypes = importfile['SessionData']['StimType'][0, 0]
    stimstructs = importfile['SessionData']['stimEvents'][0,0]
    targstim = importfile['SessionData']['TargStim'][0,0]
    diststim = importfile['SessionData']['DistStim'][0,0]
    assisted = importfile['SessionData']['Assisted'][0,0]
    singlespout = importfile['SessionData']['SingleSpout'][0,0]
    rewarded = importfile['SessionData']['Rewarded'][0,0]
    try:
        optotype = importfile['SessionData']['optoType'][0,0]
    except ValueError:
        print("No opto in this session")
        optotype = np.empty((1,numtrials))
        optotype[:,:] = np.nan
    
    correcttrials = np.multiply(1, choice == stimside)
    percentcorrect = round(np.sum(correcttrials)/numtrials, 3)
    numcorrect = np.sum(correcttrials)
    modality = np.mean(stimtypes)
    
    allleftclicks = []
    allrightclicks = []
    allleftflashes = []
    allrightflashes = []
    
    for i in range(numtrials): 
        stimevent = stimstructs[0,i]
        numleftclicks = np.size(stimevent[0,0])
        numrightclicks = np.size(stimevent[0,1])
        numleftflashes = np.size(stimevent[0,2])
        numrightflashes = np.size(stimevent[0,3])
        
        
        allleftclicks.append(numleftclicks)
        allrightclicks.append(numrightclicks)
        allleftflashes.append(numleftflashes)
        allrightflashes.append(numrightflashes)
        
    #print('\n' + filepath[39:] + " left click rate: " + str(rateleftclicks))
    #print('\n' + filepath[39:] + " right click rate: " + str(raterightclicks))
    
    stimdict = {'laudio': allleftclicks, 'raudio': allrightclicks, 'lvis': allleftflashes, 'rvis': allrightflashes}
    
    
    return numtrials, numcorrect, stimtypes, stimside, percentcorrect, correcttrials, choice, stimdict, targstim, diststim, singlespout, assisted, modality, rewarded, optotype

def formatSessions(filelist,input_dim):
    allchoices = []
    allstimsides = []
    allnumtrials = []
    inpts = []
    alltargstim = []
    alldiststim = []
    num_sess = len(filelist)
    print('\n\nremoving trials with no choice on current trial and opto stim\n\n')
    for i in range(num_sess):
        numtrials, numcorrect, stimtype, stimside, percentcorrect, correcttrials, choice, stimdict, targstim, diststim, singlespout, assisted, modality, rewarded, optotype = getSessionData(filelist[i])
        targstim = np.int32(targstim)
        diststim = np.int32(diststim)
        
        nochoicetrials = np.isnan(choice) # remove trials with no choice and with opto
        nooptotrials = np.isnan(optotype)
        choice = choice[np.logical_and(~nochoicetrials,nooptotrials)]
        choice = np.int32(choice)
        choice = np.expand_dims(choice, axis=1)
        
        stimside = stimside[np.logical_and(~nochoicetrials,nooptotrials)]
        targstim = targstim[np.logical_and(~nochoicetrials,nooptotrials)]
        diststim = diststim[np.logical_and(~nochoicetrials,nooptotrials)]
        
        numtrials_withchoice = choice.size  #number of trials per session with choice
        
        allchoices.append(choice)
        allstimsides.append(stimside)
        allnumtrials.append(numtrials_withchoice)
        alltargstim.append(targstim)
        alldiststim.append(diststim)
        
        temp = np.negative(np.ones([numtrials_withchoice,input_dim]))
        #right now the following line is sign flipped
        coherence = np.divide(np.array(stimdict['laudio']), np.array(stimdict['laudio']) + np.array(stimdict['raudio']))#only audio trials right now
        coherence = coherence[np.squeeze(np.logical_and(~nochoicetrials,nooptotrials))] #remove trials without choice and opto
        coherence = np.subtract(coherence,.5) #zero center
        coherence = coherence / np.std(coherence) #normalize
        temp[:,0] = coherence 
        inpts.append(temp)
        
        
    return allchoices, allstimsides, allnumtrials, inpts, alltargstim, alldiststim

def formatSessions_chaoqun(filelist,input_dim):
    allchoices = np.empty(0)
    allstimsides = np.empty(0)
    allnumtrials = np.empty(0)
    allcoherences = np.empty(0)
    alloptotype = np.empty(0)
    num_sess = len(filelist)
    
    for i in range(num_sess):
        numtrials, numcorrect, stimtype, stimside, percentcorrect, correcttrials, choice, stimdict, targstim, diststim, singlespout, assisted, modality, rewarded, optotype = getSessionData(filelist[i])
        targstim = np.int32(targstim)
        diststim = np.int32(diststim)
        
        #right now the following line is sign flipped
        coherence = np.divide(np.array(stimdict['laudio']), np.array(stimdict['laudio']) + np.array(stimdict['raudio']))#only audio trials right now
        #coherence = np.array(stimdict['raudio']) - np.array(stimdict['laudio'])
        #coherence = spstats.zscore(coherence) #normalize and zero center
        coherence = np.subtract(coherence,.5)
        coherence = coherence / np.std(coherence)
       
        allcoherences = np.append(allcoherences,coherence)
        alloptotype = np.append(alloptotype,optotype)
        allchoices = np.append(allchoices,choice)
        allstimsides = np.append(allstimsides,stimside)
        allnumtrials = np.append(allnumtrials,numtrials)
        
    #trim sessions with chaoquns array here
    indfile = sio.loadmat('C:\\Data\\churchland\\CY_DLCproject\\mSM63_TrialIdx.mat')
    inds = indfile['TrialIdx'] - 1
    allchoices = allchoices[inds]
    allstimsides = allstimsides[inds]
    allcoherences = allcoherences[inds]
    alloptotype = alloptotype[inds]
    
    #trim sessions with no choice on current trial and opto stim
    print('\n\nremoving trials with no choice on current trial AND previous trial and opto stim\n\n')
    nooptotrials = np.isnan(alloptotype) # remove trials with opto
    nochoicetrials = np.isnan(allchoices) # remove trials with no choice 
    noprevchoicetrials = np.roll(nochoicetrials, 1)
    noprevchoicetrials[0,0] = 'True'
    
    inds2 = np.logical_and(~nochoicetrials,~noprevchoicetrials)
    inds3 = np.logical_and(inds2,nooptotrials)
    allchoices = allchoices[inds3]
    allstimsides = allstimsides[inds3]
    allcoherences = allcoherences[inds3]
    alloptotype = alloptotype[inds3]
    
    temp = np.ones([allchoices.size,2])
    temp[:,0] = allcoherences
    inpts = np.array(temp)
    return allchoices, allstimsides, inpts


if __name__ == '__main__':
    main()
