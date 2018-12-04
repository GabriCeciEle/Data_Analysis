clear all
close all
clc

load ('Data.mat');

%% Data splitting and normalization
train = 0.6; %0.05
validation = 0.2; %0.65

rowsTrain = round(train*size(Data,1));
rowsValidation = round(validation*size(Data,1));

xtrain = PosX(1:rowsTrain);
xvalidation = PosX(rowsTrain+1:rowsTrain+rowsValidation);
xtest = PosX(rowsTrain+rowsValidation+1:end);
xfulltrain = PosX(1:rowsTrain+rowsValidation);

ytrain = PosY(1:rowsTrain);
yvalidation = PosY(rowsTrain+1:rowsTrain+rowsValidation);
ytest = PosY(rowsTrain+rowsValidation+1:end);
yfulltrain = PosY(1:rowsTrain+rowsValidation);

% Splitting
trainData = Data(1:rowsTrain,:);
validationData = Data(rowsTrain+1:rowsTrain+rowsValidation,:);
testData = Data(rowsTrain+rowsValidation+1:end,:);
fullTrainData = Data(1:rowsTrain+rowsValidation,:);

% Normalization
[trainData, mu, sigma] = zscore(trainData);
validationData = (validationData - mu)./sigma;
[fullTrainData, mu, sigma] = zscore(fullTrainData);
testData = (testData - mu)./sigma;

%% Lambda and Alpha 
lambda = logspace(-10,0,15);
Alpha = 0.1:0.1:1;

%% Elastic Nets X
Results.X = optimization(Alpha, lambda, xtrain, xvalidation, trainData, validationData);
Results.Y = optimization(Alpha, lambda, ytrain, yvalidation, trainData, validationData);

%% Final Model Building X --------- to check absolutely, sleeping while doing it 

% If Elastic Nets
[B_final,FitInf_final] = ...
    lasso(fullTrainData,xfulltrain,'Lambda',Results.X.ElasticNets.optimalLambda,'Alpha',Results.X.ElasticNets.optimalAlpha);

Xpredicted_final = testData*B_final + FitInf_final.Intercept;

Results.X.ElasticNets.performance = immse(xtest,Xpredicted_final);

% If PCA + Regression 1st order
[coeff,fullTrainDatascore, ~, ~,explained] = pca(fullTrainData);
testDatascore = testData*coeff;

I_fullTrain = ones(size(xfulltrain,1),1);
I_test = ones(size(xtest,1),1);
FM_fulltrain = fullTrainDatascore(:,1:Results.X.PCAandRegression.numPCs_opt);
FM_test = testDatascore(:,1:Results.X.PCAandRegression.numPCs_opt);
fullTrain = [I_fullTrain FM_fulltrain];
Test = [I_test FM_test];

b = regress(xfulltrain,fullTrain);

Results.X.PCAandRegression.performance_1order = immse(xtest,Test*b);

FM_fulltrainSecond = fullTrainDatascore(:,1:Results.X.PCAandRegression.numPCs_optSecond);
FM_testSecond = testDatascore(:,1:Results.X.PCAandRegression.numPCs_optSecond);
fullTrainSecond = [I_fullTrain FM_fulltrainSecond FM_fulltrainSecond.^2];
TestSecond = [I_test FM_testSecond FM_testSecond.^2];

b_second = regress(xfulltrain,fullTrainSecond);

Results.X.PCAandRegression.performance_2order = immse(xtest,TestSecond*b_second);


%% Figures Elastic Nets
% figure('name','Non Zeros')
% semilogx(lambda,Results.ElasticNets.nonZeros)
% hold on
% semilogx(Results.ElasticNets.optimalLambda,Results.ElasticNets.nonZeros(a_opt,find(Results.ElasticNets.optimalLambda==lambda)),'*')
% grid on
% xlabel('Lambda')
% ylabel('#non zeros Beta Weights')
% legend('0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9','1','Chosen Model')
% 
% figure('name','Validation and Training')
% semilogx(lambda,Results.ElasticNets.validationErr(a_opt,:))
% hold on 
% semilogx(lambda, Results.ElasticNets.MSE_training(a_opt,:))
% grid on 
% xlabel('Lambda')
% ylabel('MSE')
% legend('Validation','Training')
% 
% figure('name','Validation')
% semilogx(lambda,Results.ElasticNets.validationErrX)
% hold on 
% semilogx(Results.ElasticNets.optimalLambda,Results.ElasticNets.validationErrX(a_opt,find(Results.ElasticNets.optimalLambda==lambda)),'*')
% grid on 
% xlabel('Lambda')
% ylabel('MSE')
% legend('0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9','1','Chosen Model')
% 
%% Figures PCA + Regression
% figure()
% subplot(2,1,1)
% plot(trainErrX)
% hold on 
% plot(validationErrX)
% legend('Train','Test')
% title('PosX first order')
% 
% subplot(2,1,2)
% plot(trainErrX_second)
% hold on 
% plot(validationErrX_second)
% legend('Train','Test')
% title('PosX second order')
% 

% figure with PosX and PosY in time and the obtained Xregressed and Y
% regressed


