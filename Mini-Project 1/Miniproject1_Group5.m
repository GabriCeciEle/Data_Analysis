clear all
close all
clc

%% Data Import 
load('trainSet.mat');
load('testSet.mat');
load('trainLabels.mat');

%% Guidesheets 

[classA, classB, labelsA, labelsB] = GuidesheetI(trainData,trainLabels,testData);

% Final

GuidesheetII(trainData,trainLabels,testData, classA, classB, labelsA, labelsB);

GuidesheetIII(trainData,trainLabels,testData);

GuidesheetIV(trainData,trainLabels,testData);

% GuidesheetV();



