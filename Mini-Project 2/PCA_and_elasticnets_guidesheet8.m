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

%% PCA

[coeff,trainDatascore, ~, ~,explained] = pca(trainData);
validationDatascore = validationData*coeff;

[coeff_full,fullTrainDatascore, ~, ~,explained_full] = pca(fullTrainData);
testDatascore = testData*coeff_full;

%% Lambda and Alpha 
lambda = logspace(-10,0,15);
%Alpha = logspace(-10,0,20);
Alpha = 0.1:0.1:1;

%% Lasso
[B,FitInf] = lasso(trainDatascore,xtrain,'Lambda',lambda);

Xregressed = validationDatascore*B + FitInf.Intercept;

validationErrX = [];
for lambda_= 1:size(lambda,2)
    
    validationErrX = [validationErrX, immse(xvalidation,Xregressed(:,lambda_))];
end

[M,I] = min(validationErrX);

optimalLambda = lambda(I);

[B_final,FitInf_final] = lasso(fullTrainDatascore,xfulltrain,'Lambda',optimalLambda);

Xpredicted_final = testData*B_final + FitInf_final.Intercept;

performance_final = immse(xtest,Xpredicted_final);

%% Elastic nets
[B_en,FitInf_en] = lasso(trainDatascore,xtrain,'Lambda',lambda,'Alpha',.5);

Xregressed_en = validationDatascore*B_en + FitInf_en.Intercept;

validationErrX_en = [];
for lambda_= 1:size(lambda,2)
    validationErrX_en = [validationErrX_en, immse(xvalidation,Xregressed_en(:,lambda_))];
end

[M_en,I_en] = min(validationErrX_en);

optimalLambda_en = lambda(I_en);

[B_en_final,FitInf_en_final] = lasso(fullTrainDatascore,xfulltrain,'Lambda',optimalLambda_en,'Alpha',.5);

Xpredicted_en_final = testDatascore*B_en_final + FitInf_en_final.Intercept;

performance_en_final = immse(xtest,Xpredicted_en_final);

%% Optimization Lambda and Alpha
Results.ElasticNets.nonZeros = [];
Results.ElasticNets.MSE_training = [];

for alpha_= 1:length(Alpha)
    [B_opt,FitInf_opt] = lasso(trainDatascore,xtrain,'Lambda',lambda,'Alpha',Alpha(alpha_));
    Xregressed_opt = validationDatascore*B_opt + FitInf_opt.Intercept;
    
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
    lasso(fullTrainDatascore,xfulltrain,'Lambda',Results.ElasticNets.optimalLambda,'Alpha',Results.ElasticNets.optimalAlpha);

Xpredicted_opt_final = testDatascore*B_opt_final + FitInf_opt_final.Intercept;

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