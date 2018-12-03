clear all
close all
clc

load ('Data.mat');

%% Data splitting 

k = 0.05;
rowsTrain = round(k*size(Data,1));
trainData = Data(1:rowsTrain,:);
testData = Data(rowsTrain+1:end,:);

%chosenFeatures = 1:index;

xtrain = PosX(1:rowsTrain);
xtest = PosX(rowsTrain+1:end);
Ix_train = ones(size(xtrain,1),1);
Ix_test = ones(size(xtest,1),1);
FMx_train = trainData;
FMx_test = testData;
Xtrain = [Ix_train FMx_train];
Xtest = [Ix_test FMx_test];

bx = regress(xtrain,Xtrain);
trainErrX = immse(xtrain,Xtrain*bx);
testErrX = immse(xtest,Xtest*bx);

ytrain = PosY(1:rowsTrain);
ytest = PosY(rowsTrain+1:end);
Iy_train = ones(size(ytrain,1),1);
Iy_test = ones(size(ytest,1),1);
FMy_train = trainData;
FMy_test = testData;
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

%% LASSO
lambda = logspace(-10,0,15);

[B, FitInfo] = lasso(Xtrain,xtrain,'Lambda',lambda);

nonzer = [];
for i = 1:size(B,2)
    nonzer= [nonzer,size(find(abs(B(:,i))>0),1)];
end

%% LASSO Cross validation 
% not to concatenaten (I in the matrix)

lambda = logspace(-10,0,15);

for k = 1:1:9
    rowsTrain = round((k/10)*size(Data,1));
    rowsTest = round(((k/10)+0.1)*size(Data,1));
    trainData = Data(1:rowsTrain,:);
    testData = Data(rowsTrain+1:rowsTest,:);
    
    xtrain = PosX(1:rowsTrain);
    xtest = PosX(rowsTrain+1:rowsTest);
    Ix_train = ones(size(xtrain,1),1);
    Ix_test = ones(size(xtest,1),1);
    FMx_train = trainData;
    FMx_test = testData;
    Xtrain = [Ix_train FMx_train];
    Xtest = [Ix_test FMx_test];

    [B,FitInf] = lasso(Xtrain,xtrain,'Lambda',lambda);
    
    Xregressed = Xtest*B;
    
    for lambda_= 1:size(lambda,2)
        testErrX(k,lambda_) = immse(xtest,Xregressed(:,lambda_));
    end
    
end

semilogx(lambda,mean(testErrX,1))


%% LASSO easy

train = 0.6; %0.05
validation = 0.2; %0.65

rowsTrain = round(train*size(Data,1));
rowsValidation = round(validation*size(Data,1));

trainData = Data(1:rowsTrain,:);
validationData = Data(rowsTrain+1:rowsTrain+rowsValidation,:);
testData = Data(rowsTrain+rowsValidation+1:end,:);

xtrain = PosX(1:rowsTrain);
xvalidation = PosX(rowsTrain+1:rowsTrain+rowsValidation);
xtest = PosX(rowsTrain+rowsValidation+1:end);

lambda = logspace(-10,0,15);

[B,FitInf] = lasso(trainData,xtrain,'Lambda',lambda);

Xregressed = validationData*B + FitInf.Intercept;

validationErrX = [];
for lambda_= 1:size(lambda,2)
    validationErrX = [validationErrX, immse(xvalidation,Xregressed(:,lambda_))];
end

[M,I] = min(validationErrX);

optimalLambda = lambda(I);

fullTrainData = Data(1:rowsTrain+rowsValidation,:);
xfulltrain = PosX(1:rowsTrain+rowsValidation);

[B_final,FitInf_final] = lasso(fullTrainData,xfulltrain,'Lambda',optimalLambda);

Xpredicted_final = testData*B_final + FitInf_final.Intercept;

performance_final = immse(xtest,Xpredicted_final);

figure('name','Optimization')
semilogx(lambda,validationErrX)
hold on
semilogx(lambda,FitInf.MSE)
xlabel('Lambda')
ylabel('MSE')
legend('Validation','Training')

figure('name','Behavior')
plot(Xpredicted_final)
hold on 
plot(xtest)
legend('Regressed','PosX')


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

figure('name','Optimization elastic nets')
semilogx(lambda,validationErrX_en)
hold on
semilogx(lambda,FitInf_en.MSE)
xlabel('Lambda')
ylabel('MSE')

figure('name','Non zeros')
semilogx(lambda,FitInf.DF)
hold on
semilogx(lambda,FitInf_en.DF)
xlabel('Lambda')
ylabel('#non zero Beta Weights')
legend('Lasso','Elastic nets')

figure('name','Behavior')
plot(Xpredicted_en_final)
hold on 
plot(xtest)
legend('Regressed','PosX')

%% Optimization
Alpha = 0.1:0.1:1;
for alpha_= 1:length(Alpha)
    [B_opt,FitInf_opt] = lasso(trainData,xtrain,'Lambda',lambda,'Alpha',Alpha(alpha_));
    Xregressed_opt = validationData*B_opt + FitInf_opt.Intercept;
    
    validationErrX_opt = [];
    for lambda_= 1:size(lambda,2)
        validationErrX_opt = [validationErrX_opt, immse(xvalidation,Xregressed_opt(:,lambda_))];
    end

    [M_opt(1,alpha_),I_opt] = min(validationErrX_opt);
    lambda_opt(1,alpha_) = lambda(I_opt);
end
    
[M_opt_final,a_opt] = min(M_opt);
optimalAlpha = Alpha(a_opt);
optimalLambda_opt = lambda_opt(a_opt);

[B_opt_final,FitInf_opt_final] = lasso(fullTrainData,xfulltrain,'Lambda',optimalLambda_opt,'Alpha',optimalAlpha);

Xpredicted_opt_final = testData*B_opt_final + FitInf_opt_final.Intercept;

performance_opt_final = immse(xtest,Xpredicted_opt_final);


