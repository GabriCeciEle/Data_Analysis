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
Alpha = logspace(-2,0,10);
%% Lasso
[B,FitInf] = lasso(trainData,xtrain,'Lambda',lambda);

Xregressed = validationData*B + FitInf.Intercept;

validationErrX = [];
for lambda_= 1:size(lambda,2)
    
    validationErrX = [validationErrX, immse(xvalidation,Xregressed(:,lambda_))];
end

[M,I] = min(validationErrX);

optimalLambda = lambda(I);

[B_final,FitInf_final] = lasso(fullTrainData,xfulltrain,'Lambda',optimalLambda);

Xpredicted_final = testData*B_final + FitInf_final.Intercept;

performance_final = immse(xtest,Xpredicted_final);

%% Elastic nets
[B_en,FitInf_en] = lasso(trainData,xtrain,'Lambda',lambda,'Alpha',.5);

Xregressed_en = validationData*B_en + FitInf_en.Intercept;

validationErrX_en = [];
for lambda_= 1:size(lambda,2)
    validationErrX_en = [validationErrX_en, immse(xvalidation,Xregressed_en(:,lambda_))];
end

[M_en,I_en] = min(validationErrX_en);

optimalLambda_en = lambda(I_en);

[B_en_final,FitInf_en_final] = lasso(fullTrainData,xfulltrain,'Lambda',optimalLambda_en,'Alpha',.5);

Xpredicted_en_final = testData*B_en_final + FitInf_en_final.Intercept;

performance_en_final = immse(xtest,Xpredicted_en_final);

%% Optimization Lambda and Alpha
Results.ElasticNets.nonZeros = [];
Results.ElasticNets.MSE_training = [];

for alpha_= 1:length(Alpha)
    [B_opt,FitInf_opt] = lasso(trainData,xtrain,'Lambda',lambda,'Alpha',Alpha(alpha_));
    Xregressed_opt = validationData*B_opt + FitInf_opt.Intercept;
    
    for lambda_= 1:size(lambda,2)
        Results.ElasticNets.validationErrX(alpha_,lambda_) = immse(xvalidation,Xregressed_opt(:,lambda_));
    end
    
    Results.ElasticNets.nonZeros = [Results.ElasticNets.nonZeros;FitInf_opt.DF];
    Results.ElasticNets.MSE_training = [Results.ElasticNets.MSE_training;FitInf_opt.MSE];
    
    [M_opt(1,alpha_),I_opt] = min(Results.ElasticNets.validationErrX(alpha_,:));
    lambda_opt(1,alpha_) = lambda(I_opt);
end
    
[Results.ElasticNets.validationErrX_opt,a_opt] = min(M_opt);
Results.ElasticNets.optimalAlpha = Alpha(a_opt);
Results.ElasticNets.optimalLambda = lambda_opt(a_opt);

% Final
[B_opt_final,FitInf_opt_final] = ...
    lasso(fullTrainData,xfulltrain,'Lambda',Results.ElasticNets.optimalLambda,'Alpha',Results.ElasticNets.optimalAlpha);

Xpredicted_opt_final = testData*B_opt_final + FitInf_opt_final.Intercept;

Results.ElasticNets.performance = immse(xtest,Xpredicted_opt_final);

%% Figures

figure('name','Non Zeros')
semilogx(lambda,Results.ElasticNets.nonZeros)
hold on
semilogx(Results.ElasticNets.optimalLambda,Results.ElasticNets.nonZeros(a_opt,find(Results.ElasticNets.optimalLambda==lambda)),'*')
grid on
xlabel('Lambda')
ylabel('#non zeros Beta Weights')
legend('0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9','1','Chosen Model')

figure('name','Validation and Training')
semilogx(lambda,Results.ElasticNets.validationErrX(a_opt,:))
hold on 
semilogx(lambda, Results.ElasticNets.MSE_training(a_opt,:))
grid on 
xlabel('Lambda')
ylabel('MSE')
legend('Validation','Training')

figure('name','Validation')
semilogx(lambda,Results.ElasticNets.validationErrX)
hold on 
semilogx(Results.ElasticNets.optimalLambda,Results.ElasticNets.validationErrX(a_opt,find(Results.ElasticNets.optimalLambda==lambda)),'*')
grid on 
xlabel('Lambda')
ylabel('MSE')
legend('0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9','1','Chosen Model')

% figure with PosX and PosY in time and the obtained Xregressed and Y
% regressed

%% PCA + Regression

[coeff,trainDatascore, ~, ~,explained] = pca(trainData);
validationDatascore = validationData*coeff;

Results.PCAandRegression.cumulative = cumsum(explained)/sum(explained);
Results.PCAandRegression.index = find(Results.PCAandRegression.cumulative>0.9,1);

Ix_train = ones(size(xtrain,1),1);
Ix_validation = ones(size(xvalidation,1),1);

trainErrX = [];
trainErrX_second = [];
validationErrX = [];
validationErrX_second = [];

for ind = 1:150%size(trainDatascore,2)
    
    FMx_train = trainDatascore(:,1:ind);
    FMx_validation = validationDatascore(:,1:ind);
    Xtrain = [Ix_train FMx_train];
    Xvalidation = [Ix_validation FMx_validation];
    Xtrain_second = [Ix_train FMx_train FMx_train.^2];
    Xvalidation_second = [Ix_validation FMx_validation FMx_validation.^2];

    bx = regress(xtrain,Xtrain);
    bx_second = regress(xtrain,Xtrain_second);
    
    trainErrX = [trainErrX, immse(xtrain,Xtrain*bx)];
    trainErrX_second = [trainErrX_second,immse(xtrain,Xtrain_second*bx_second)];
    
    validationErrX = [validationErrX, immse(xvalidation,Xvalidation*bx)];
    validationErrX_second = [validationErrX_second,immse(xvalidation,Xvalidation_second*bx_second)];
    
end

Results.PCAandRegression.validationErrX = [validationErrX;validationErrX_second];
[Results.PCAandRegression.validationErrX_opt, Results.PCAandRegression.numPCs_opt]= min(validationErrX);
[Results.PCAandRegression.validationErrX_optSecond, Results.PCAandRegression.numPCs_optSecond]= min(validationErrX_second);


%% Figures

figure()
subplot(2,1,1)
plot(trainErrX)
hold on 
plot(validationErrX)
legend('Train','Test')
title('PosX first order')

subplot(2,1,2)
plot(trainErrX_second)
hold on 
plot(validationErrX_second)
legend('Train','Test')
title('PosX second order')


%% Final figures 
figure('name','Real and Regressed')
plot(PosX(700:1000),PosY(700:1000))






