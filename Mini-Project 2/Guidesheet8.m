clear all
close all
clc

load ('Data.mat');

%% Data splitting 

k = 0.05;
rowsTrain = round(k*size(Data,1));
trainData = Data(1:rowsTrain,:);
testData = Data(rowsTrain+1:end,:);
