clear all
close all
clc

load ('Data.mat');

k = 0.7;
rowsTrain = round(k*size(Data,1));
trainData = Data(1:rowsTrain,:);
testData = Data(rowsTrain+1:end,:);

[trainDataNorm,mu,sigma] = zscore(trainData);
[coeff,score,~,~,explained] = pca(trainDataNorm);
testDataNorm = (testData - mu)./sigma;
testDatascore = testDataNorm*coeff;
cumulative = cumsum(explained)/sum(explained);

index = find(cumulative>0.9,1);
