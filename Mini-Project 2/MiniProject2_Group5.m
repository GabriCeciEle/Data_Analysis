clear all
close all
clc

load ('Data.mat');

%% Data splitting and PCA

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

chosenFeatures = 1:index;

xtrain = PosX(1:rowsTrain);
xtest = PosX(rowsTrain+1:end);
Ix_train = ones(size(xtrain,1),1);
Ix_test = ones(size(xtest,1),1);
FMx_train = trainDatascore(:,chosenFeatures);
FMx_test = testDatascore(:,chosenFeatures);
Xtrain = [Ix_train FMx_train];
Xtest = [Ix_test FMx_test];

bx = regress(xtrain,Xtrain);
trainErrX = immse(xtrain,Xtrain*bx);
testErrX = immse(xtest,Xtest*bx);

ytrain = PosY(1:rowsTrain);
ytest = PosY(rowsTrain+1:end);
Iy_train = ones(size(ytrain,1),1);
Iy_test = ones(size(ytest,1),1);
FMy_train = trainDatascore(:,chosenFeatures);
FMy_test = testDatascore(:,chosenFeatures);
Ytrain = [Iy_train FMy_train];
Ytest = [Iy_test FMy_test];

by = regress(ytrain,Ytrain);
trainErrY = immse(ytrain,Ytrain*by);
testErrY = immse(ytest,Ytest*by);


figure()
subplot(2,2,1)
plot(Xtrain*bx)
hold on 
plot(xtrain)
legend('Regressed','PosX')
title('Trainset')

subplot(2,2,2)
plot(Xtest*bx)
hold on 
plot(xtest)
legend('Regressed','PosX')
title('Testset')

subplot(2,2,3)
plot(Ytrain*by)
hold on 
plot(ytrain)
legend('Regressed','PosY')
title('Trainset')

subplot(2,2,4)
plot(Ytest*by)
hold on 
plot(ytest)
legend('Regressed','PosY')
title('Testset')


Xtrain_second = [Ix_train FMx_train FMx_train.^2];
Xtest_second = [Ix_test FMx_test FMx_test.^2];

Ytrain_second = [Iy_train FMy_train FMy_train.^2];
Ytest_second = [Iy_test FMy_test FMy_test.^2];

bx_second = regress(xtrain,Xtrain_second);
trainErrX_second = immse(xtrain,Xtrain_second*bx_second);
testErrX_second = immse(xtest,Xtest_second*bx_second);

by_second = regress(ytrain,Ytrain_second);
trainErrY_second = immse(ytrain,Ytrain_second*by_second);
testErrY_second = immse(ytest,Ytest_second*by_second);


%% Features

xaxis = 1:50:size(trainDatascore,2);

xtrain = PosX(1:rowsTrain);
xtest = PosX(rowsTrain+1:end);
Ix_train = ones(size(xtrain,1),1);
Ix_test = ones(size(xtest,1),1);

ytrain = PosY(1:rowsTrain);
ytest = PosY(rowsTrain+1:end);
Iy_train = ones(size(ytrain,1),1);
Iy_test = ones(size(ytest,1),1);

trainErrX = [];
trainErrY = [];
trainErrX_second = [];
trainErrY_second = [];

testErrX = [];
testErrY = [];
testErrX_second = [];
testErrY_second = [];


for ind = 1:50:size(trainDatascore,2)
    
    FMx_train = trainDatascore(:,1:ind);
    FMx_test = testDatascore(:,1:ind);
    Xtrain = [Ix_train FMx_train];
    Xtest = [Ix_test FMx_test];
    Xtrain_second = [Ix_train FMx_train FMx_train.^2];
    Xtest_second = [Ix_test FMx_test FMx_test.^2];

    FMy_train = trainDatascore(:,1:ind);
    FMy_test = testDatascore(:,1:ind);
    Ytrain = [Iy_train FMy_train];
    Ytest = [Iy_test FMy_test];
    Ytrain_second = [Iy_train FMy_train FMy_train.^2];
    Ytest_second = [Iy_test FMy_test FMy_test.^2];
    
    bx = regress(xtrain,Xtrain);
    by = regress(ytrain,Ytrain);
    bx_second = regress(xtrain,Xtrain_second);
    by_second = regress(ytrain,Ytrain_second);
    
    trainErrX = [trainErrX, immse(xtrain,Xtrain*bx)];
    testErrX = [testErrX, immse(xtest,Xtest*bx)];
    trainErrY = [trainErrY,immse(ytrain,Ytrain*by)];
    testErrY = [testErrY, immse(ytest,Ytest*by)];
    
    trainErrX_second = [trainErrX_second,immse(xtrain,Xtrain_second*bx_second)];
    testErrX_second = [testErrX_second,immse(xtest,Xtest_second*bx_second)];
    trainErrY_second = [trainErrY_second,immse(ytrain,Ytrain_second*by_second)];
    testErrY_second = [testErrY_second,immse(ytest,Ytest_second*by_second)];
    
end

figure()
subplot(2,2,1)
plot(xaxis,trainErrX)
hold on 
plot(xaxis,testErrX)
legend('Train','Test')
title('PosX first order')

subplot(2,2,2)
plot(xaxis,trainErrY)
hold on 
plot(xaxis,testErrY)
legend('Train','Test')
title('PosY first order')

subplot(2,2,3)
plot(xaxis,trainErrX_second)
hold on 
plot(xaxis,testErrX_second)
legend('Train','Test')
title('PosX second order')

subplot(2,2,4)
plot(xaxis,trainErrY_second)
hold on 
plot(xaxis,testErrY_second)
legend('Train','Test')
title('PosY second order')




