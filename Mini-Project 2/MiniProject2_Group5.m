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

ytrain = PosY(1:rowsTrain);
yvalidation = PosY(rowsTrain+1:rowsTrain+rowsValidation);
ytest = PosY(rowsTrain+rowsValidation+1:end);

% Splitting
trainData = Data(1:rowsTrain,:);
validationData = Data(rowsTrain+1:rowsTrain+rowsValidation,:);
testData = Data(rowsTrain+rowsValidation+1:end,:);

% Normalization
[trainData, mu, sigma] = zscore(trainData);
validationData = (validationData - mu)./sigma;
testData = (testData - mu)./sigma;

%% Lambda and Alpha 
lambda = logspace(-10,0,15);
Alpha = logspace(-2,0,10); 

%% Elastic Nets X
Results.X = optimization(Alpha, lambda, xtrain, xvalidation, trainData, validationData);
Results.Y = optimization(Alpha, lambda, ytrain, yvalidation, trainData, validationData);

%% Final Model Building X

% If Elastic Nets
[B_final,FitInf_final] = ...
    lasso(trainData,xtrain,'Lambda',Results.X.ElasticNets.optimalLambda,'Alpha',Results.X.ElasticNets.optimalAlpha);

Xpredicted_final = testData*B_final + FitInf_final.Intercept;

Results.X.Finalperformance = immse(xtest,Xpredicted_final);


%% Final Model Y

[coeff,trainDatascore, ~, ~,explained] = pca(trainData);
testDatascore = testData*coeff;
cumulative = cumsum(explained)/sum(explained);
Results.Explainedindex = find(cumulative>=0.9,1);

I_train = ones(size(xtrain,1),1);
I_test = ones(size(xtest,1),1);
FM_trainSecond = trainDatascore(:,1:Results.Y.PCAandRegression.numPCs_optSecond);
FM_testSecond = testDatascore(:,1:Results.Y.PCAandRegression.numPCs_optSecond);
trainSecond = [I_train FM_trainSecond FM_trainSecond.^2];
testSecond = [I_test FM_testSecond FM_testSecond.^2];

b_second = regress(ytrain,trainSecond);
Ypredicted_final = testSecond*b_second;

Results.Y.Finalperformance_2order = immse(ytest,Ypredicted_final);

%% Figures

figure('name','different methods X')
methods = categorical({'Linear Regressor 1st order';'Linear Regressor 2nd order';'Elastic Net'});
subplot(1,2,1)
bar(methods,[Results.X.PCAandRegression.trainErr_opt Results.X.PCAandRegression.validationErr_opt;
    Results.X.PCAandRegression.trainErr_optSecond Results.X.PCAandRegression.validationErr_optSecond;
    Results.X.ElasticNets.trainErr Results.X.ElasticNets.validationErr]);
ylabel('MSE')
legend ('Train Error','Validation Error')
title('PosX')

subplot(1,2,2)
bar(methods,[Results.Y.PCAandRegression.trainErr_opt Results.Y.PCAandRegression.validationErr_opt;
    Results.Y.PCAandRegression.trainErr_optSecond Results.Y.PCAandRegression.validationErr_optSecond;
    Results.Y.ElasticNets.trainErr Results.Y.ElasticNets.validationErr]);
ylabel('MSE')
legend ('Train Error','Validation Error')
title('PosY')

figure('name','Train and Validation X')
subplot(1,2,1)
plot(Results.X.PCAandRegression.trainErr(1,:));
hold on 
plot(Results.X.PCAandRegression.validationErr(1,:))
legend('Train','Validation')
title('PosX first order')

subplot(1,2,2)
plot(Results.X.PCAandRegression.trainErr(2,:))
hold on 
plot(Results.X.PCAandRegression.validationErr(2,:))
legend('Train','Validation')
title('PosX second order')

figure('name','Train and Validation Y')
subplot(1,2,1)
plot(Results.Y.PCAandRegression.trainErr(1,:));
hold on 
plot(Results.Y.PCAandRegression.validationErr(1,:))
legend('Train','Validation')
title('PosY first order')

subplot(1,2,2)
plot(Results.Y.PCAandRegression.trainErr(2,:))
hold on 
plot(Results.Y.PCAandRegression.validationErr(2,:))
legend('Train','Validation')
title('PosY second order')

figure('name','Predicted Movements X')
plot(xtest)
hold on
plot(Xpredicted_final)
title('predicted X')
legend('true','predicted')

figure('name','Predicted Movements Y')
plot(ytest)
hold on
plot(Ypredicted_final)
title('predicted Y')
legend('true','predicted')

figure('name','Real trajectories')
plot(xtest(400:425),ytest(400:425))
hold on
plot(Xpredicted_final(400:425),Ypredicted_final(400:425))
title('Real Trajectories')
legend('true','predicted')

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


