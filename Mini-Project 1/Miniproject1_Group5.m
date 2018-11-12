clear all
close all
clc

%% Data Import 
load('trainSet.mat');
load('testSet.mat');
load('trainLabels.mat');

%% Final 

Results = Final(trainData,trainLabels,testData);
ModelBuilding(trainData,trainLabels,testData,'linear',17);

%% Guidesheets 

[classA, classB, labelsA, labelsB] = GuidesheetI(trainData,trainLabels,testData);

GuidesheetII(trainData,trainLabels,testData, classA, classB, labelsA, labelsB);

GuidesheetIII(trainData,trainLabels,testData);

GuidesheetIV(trainData,trainLabels,testData);




