clear all
close all
clc

load ('Data.mat');

%% Data splitting
train = 0.6; %0.05
validation = 0.2; %0.65

rowsTrain = round(train*size(Data,1));
rowsValidation = round(validation*size(Data,1));

trainData = Data(1:rowsTrain,:);
[trainData, mu, sigma] = zscore(trainData);

validationData = Data(rowsTrain+1:rowsTrain+rowsValidation,:);
validationData = validationData-

testData = Data(rowsTrain+rowsValidation+1:end,:);
    
xtrain = PosX(1:rowsTrain);
xvalidation = PosX(rowsTrain+1:rowsTrain+rowsValidation);
xtest = PosX(rowsTrain+rowsValidation+1:end);

fullTrainData = Data(1:rowsTrain+rowsValidation,:);
xfulltrain = PosX(1:rowsTrain+rowsValidation);

%% Lambda and Alpha 
lambda = logspace(-10,0,15);
Alpha = 0.1:0.1:1;

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
nonZeros_opt = [];
MSE_opt_train = [];

for alpha_= 1:length(Alpha)
    [B_opt,FitInf_opt] = lasso(trainData,xtrain,'Lambda',lambda,'Alpha',Alpha(alpha_));
    Xregressed_opt = validationData*B_opt + FitInf_opt.Intercept;
    
    %validationErrX_opt = [];
    for lambda_= 1:size(lambda,2)
        validationErrX_opt(alpha_,lambda_) = immse(xvalidation,Xregressed_opt(:,lambda_));
        %validationErrX_opt = [validationErrX_opt, immse(xvalidation,Xregressed_opt(:,lambda_))];
    end
    
    nonZeros_opt = [nonZeros_opt;FitInf_opt.DF];
    MSE_opt_train = [MSE_opt_train;FitInf_opt.MSE];
    
    [M_opt(1,alpha_),I_opt] = min(validationErrX_opt(alpha_,:));
    lambda_opt(1,alpha_) = lambda(I_opt);
end
    
[M_opt_final,a_opt] = min(M_opt);
optimalAlpha = Alpha(a_opt);
optimalLambda_opt = lambda_opt(a_opt);

[B_opt_final,FitInf_opt_final] = lasso(fullTrainData,xfulltrain,'Lambda',optimalLambda_opt,'Alpha',optimalAlpha);

Xpredicted_opt_final = testData*B_opt_final + FitInf_opt_final.Intercept;

performance_opt_final = immse(xtest,Xpredicted_opt_final);

%% Figures

figure('name','Non Zeros')
semilogx(lambda,nonZeros_opt)
hold on
semilogx(optimalLambda_opt,nonZeros_opt(a_opt,find(optimalLambda_opt==lambda)),'*')
grid on
xlabel('Lambda')
ylabel('#non zeros Beta Weights')
legend('0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9','1','Chosen Model')

figure('name','Validation and Training')
semilogx(lambda,validationErrX_opt(a_opt,:))
hold on 
semilogx(lambda, MSE_opt_train(a_opt,:))
grid on 
xlabel('Lambda')
ylabel('MSE')
legend('Validation','Training')

figure('name','Validation')
semilogx(lambda,validationErrX_opt)
hold on 
semilogx(optimalLambda_opt,validationErrX_opt(a_opt,find(optimalLambda_opt==lambda)),'*')
grid on 
xlabel('Lambda')
ylabel('MSE')
legend('0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9','1','Chosen Model')

% figure with PosX and PosY in time and the obtained Xregressed and Y
% regressed








