clear all
close all
clc

load ('Data.mat');

k = 0.7;
rowsTrain = round(k*size(Data,1));
trainData = Data(1:rowsTrain,:);
testData = Data(rowsTrain+1:end,:);

[trainDataNorm,mu,sigma] = zscore(trainData);
[coeff,trainDatascore,~,~,explained] = pca(trainDataNorm);
testDataNorm = (testData - mu)./sigma;
testDatascore = testDataNorm*coeff;
cumulative = cumsum(explained)/sum(explained);

index = find(cumulative>0.9,1);

%% Regressor Training

chosenFeatures = 1:size(trainDatascore,2);
ytrain = PosX(1:rowsTrain);
ytest = PosX(rowsTrain+1:end);
Itrain = ones(size(ytrain,1),1);
Itest = ones(size(ytest,1),1);
FMtrain = trainData(:,chosenFeatures);
FMtest = testData(:,chosenFeatures);
Xtrain = [Itrain FMtrain];
Xtest = [Itest FMtest];

b = regress(ytrain,Xtrain);
trainErr = immse(ytrain,Xtrain*b);
testErr = immse(ytest,Xtest*b);
