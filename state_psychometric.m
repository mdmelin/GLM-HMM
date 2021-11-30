clc;clear all;close all;
%This script takes latent state predictions from the GLM-HMM and computes
%the psychometric function for each latent state 
datafile = 'C:\Data\churchland\musall_glm_fitting_data\mSM80\map_allsessions.mat' %path to python output
%function [psychometric_functions] = state_psychometric(datafile)
load(datafile);
transmat = exp(model.transitions.log_Ps);
figure;
heatmap(transmat);
xlabel('state t+1')
ylabel('state t')
title('Probability of state transition')

[numsess,~] = size(files)
[~,numstates] = size(transmat);
choiceconc = [];
stateconc = [];
stimconc = [];

for i = 1:numsess
    stimside = stimsides{i};
    t = double(targ{i});
    d = double(dist{i});
    rstim = zeros(length(stimside),1);
    lstim = zeros(length(stimside),1);
    rstim(stimside == 1) = t(stimside == 1);
    rstim(stimside == 0) = d(stimside == 0);
    lstim(stimside == 0) = t(stimside == 0);
    lstim(stimside == 1) = d(stimside == 1);
    
    coh = rstim ./ (rstim + lstim);
    
    choiceconc = [choiceconc, choices{i}'];
    stateconc = [stateconc, latent_states{i}'];
    stimconc = [stimconc, coh'];
end

histogram(stimconc)
title('Histogram of coherence values')
ylabel('Trial number')
xlabel('Coherence')
numtrials = length(stimconc)

[~,likely_state] = max(stateconc,[],1); %get most likely state for a trial, as predicted from GLM-HMM 
stimvals = unique(stimconc);
statesep_choices = {};
statesep_stimulus = {};
pright = [];
for i = 1:numstates
    choices = choiceconc(likely_state == i);
    stimuli = stimconc(likely_state == i);
    for j = 1:length(stimvals) %iterate thru stimulus values and calculate p(right) for each
        choiceforstim = choices(stimuli == stimvals(j)); %cell dimensions are number of unique stimuli
        pright(i,j) = sum(choiceforstim) / length(choiceforstim); %probability of going right [state, stimulus value]
    end
end

for i = 1:length(stimvals) %now just find the curve over all states
        choiceforstim = choiceconc(stimconc == stimvals(i)); %cell dimensions are number of unique stimuli
        pright_allstates(i) = sum(choiceforstim) / length(choiceforstim); %probability of going right [state, stimulus value]
end

figure
subplot(2,1,1);
hold on;
ws = squeeze(model.observations.Wk);
title('Psychometric functions over states identified with GLM-HMM');
xlabel('Coherence');
ylabel('P(right)');
plot(stimvals, pright(1,:),'LineWidth',3)
plot(stimvals, pright(2,:),'LineWidth',3)
plot(stimvals, pright(3,:),'LineWidth',3)
plot(stimvals,pright_allstates,'LineWidth',3);
stimw = ws(:,1);
biasw = ws(:,2);
for i = 1:numstates
    labels{i} = sprintf('State weights: %.2f   %.2f',stimw(i),biasw(i));
end
labels{end+1} = 'All States';
legend(labels,'Location','southeast')

subplot(2,1,2);
title('Latent state weights');
hold on;
yline(0,'k--')
plot(ws','LineWidth',3);
xticks([1 2])
xticklabels({'Stimulus','Bias'});
ylabel('Weight');
%end