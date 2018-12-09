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

%% Evaluation of different methods
Results.X = optimization(Alpha, lambda, xtrain, xvalidation, trainData, validationData);
Results.Y = optimization(Alpha, lambda, ytrain, yvalidation, trainData, validationData);

%% Final Model X

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

a1 = ['1st Order Regressor, ' num2str(Results.X.PCAandRegression.numPCs_opt) ' PCs'];
a2 = ['2nd Order Regressor, ' num2str(Results.X.PCAandRegression.numPCs_optSecond) ' PCs'];
a3 = ['Elastic Net, alpha = ' num2str(Results.X.ElasticNets.optimalAlpha) ', lambda = ' num2str(Results.X.ElasticNets.optimalLambda)];
methodsX = categorical({a1;a2;a3});
b1 = ['1st Order Regressor, ' num2str(Results.Y.PCAandRegression.numPCs_opt) ' PCs'];
b2 = ['2nd Order Regressor, ' num2str(Results.Y.PCAandRegression.numPCs_optSecond) ' PCs'];
b3 = ['Elastic Net, alpha = ' num2str(Results.Y.ElasticNets.optimalAlpha) ', lambda = ' num2str(Results.X.ElasticNets.optimalLambda)];
methodsY = categorical({b1;b2;b3});

methods = categorical({'1st Order Regressor';'2nd Order Regressor';'Elastic Net'});
figure('name','different methods X and Y')
subplot(1,2,1)
bar(methods,[Results.X.PCAandRegression.trainErr_opt Results.X.PCAandRegression.validationErr_opt;
    Results.X.PCAandRegression.trainErr_optSecond Results.X.PCAandRegression.validationErr_optSecond;
    Results.X.ElasticNets.trainErr Results.X.ElasticNets.validationErr]);
ylabel('MSE','fontsize',20)
legend ({'Train Error','Validation Error'}, 'fontsize',14)
title('PosX','fontsize',20)

subplot(1,2,2)
bar(methods,[Results.Y.PCAandRegression.trainErr_opt Results.Y.PCAandRegression.validationErr_opt;
    Results.Y.PCAandRegression.trainErr_optSecond Results.Y.PCAandRegression.validationErr_optSecond;
    Results.Y.ElasticNets.trainErr Results.Y.ElasticNets.validationErr]);
ylabel('MSE','fontsize',20)
legend ({'Train Error','Validation Error'}, 'fontsize',14)
title('PosY','fontsize',20)

figure('name','Non zeros beta weights, PosX')
semilogx(lambda,Results.X.ElasticNets.nonZeros)
hold on
semilogx(Results.X.ElasticNets.optimalLambda, Results.X.ElasticNets.nonZeros(find(Alpha==Results.X.ElasticNets.optimalAlpha),find(Results.X.ElasticNets.optimalLambda==lambda)),'*')
grid on
xlabel('Lambda','fontsize',20)
ylabel('Number of non zeros Beta Weights','fontsize',20)
title('Non zeros beta weights, PosX','fontsize',20)
legend({'alpha=0.01','alpha=0.0167','alpha=0.0278','alpha=0.0464','alpha=0.0774','alpha=0.1292','alpha=0.2154','alpha=0.3594','alpha=0.5995','alpha=1','Chosen Model'},'fontsize',14)

figure('name','Evolution of the MSE tot, PosX')
semilogx(lambda, Results.X.ElasticNets.MSE_training)
hold on
semilogx(lambda,Results.X.ElasticNets.validationErrtot)
grid on 
xlabel('Lambda')
ylabel('MSE')
title('Evolution of the MSE for all the alpha and lambda')
legend('Train error','Validation error')

figure('name','Evolution of the MSE, PosX')
semilogx(lambda, Results.X.ElasticNets.MSE_training(find(Alpha==Results.X.ElasticNets.optimalAlpha),:),'linewidth',2)
hold on
semilogx(lambda,Results.X.ElasticNets.validationErrtot(find(Alpha==Results.X.ElasticNets.optimalAlpha),:),'linewidth',2)
hold on
semilogx(Results.X.ElasticNets.optimalLambda,Results.X.ElasticNets.validationErr, 'k*','linewidth',1.5)
hold on
semilogx(Results.X.ElasticNets.optimalLambda,Results.X.ElasticNets.trainErr, 'k*','linewidth',1.5)
grid on 
xlabel('Lambda','fontsize',20)
ylabel('MSE','fontsize',20)
title('Evolution of the MSE for the optimal alpha found (alpha=0.0464), PosX','fontsize',20)
legend({'Train error','Validation error'},'fontsize',20)


figure('name','Evolution of the MSE, PosY')
plot(Results.Y.PCAandRegression.trainErr(2,:),'linewidth',2)
hold on 
plot(Results.Y.PCAandRegression.validationErr(2,:),'linewidth',2)
hold on
plot(26*ones(200),linspace(0,8e-4,200),'k--')
grid on
legend({'Train error','Validation error'},'fontsize',20)
xlabel('Number of PCs used','fontsize',20)
ylabel('MSE','fontsize',20)
title('Evolution of the MSE for the 2nd order regressor, PosY', 'fontsize', 20)

figure('name','Evolution of the MSE X and Y')
subplot(1,2,1)
semilogx(lambda, Results.X.ElasticNets.MSE_training(find(Alpha==Results.X.ElasticNets.optimalAlpha),:),'linewidth',2)
hold on
semilogx(lambda,Results.X.ElasticNets.validationErrtot(find(Alpha==Results.X.ElasticNets.optimalAlpha),:),'linewidth',2)
hold on
semilogx(Results.X.ElasticNets.optimalLambda,Results.X.ElasticNets.validationErr, 'k*','linewidth',1.5)
hold on
semilogx(Results.X.ElasticNets.optimalLambda,Results.X.ElasticNets.trainErr, 'k*','linewidth',1.5)
grid on 
xlabel('Lambda','fontsize',20)
ylabel('MSE','fontsize',20)
title('MSE for the optimal alpha, PosX','fontsize',20)
legend({'Train error','Validation error'},'fontsize',20)

subplot(1,2,2)
plot(Results.Y.PCAandRegression.trainErr(2,:),'linewidth',2)
hold on 
plot(Results.Y.PCAandRegression.validationErr(2,:),'linewidth',2)
hold on
plot(26*ones(200),linspace(0,8e-4,200),'k--')
grid on
legend({'Train error','Validation error'},'fontsize',20)
xlabel('Number of PCs used','fontsize',20)
ylabel('MSE','fontsize',20)
title('MSE 2nd order regressor, PosY', 'fontsize', 20)

figure('name','Predicted Movements X and Y')
subplot(2,2,1)
plot(xtest,'linewidth',2)
hold on
plot(Xpredicted_final,'linewidth',2)
grid on
axis ([1000 1500 -0.02 0.14])
title('Predicted X movements', 'fontsize',20)
xlabel('Time','fontsize',20)
ylabel('X coordinate','fontsize',20)
legend({'True','Predicted'},'fontsize',14)

subplot(2,2,2)
plot(ytest,'linewidth',2)
hold on
plot(Ypredicted_final,'linewidth',2)
grid on
axis ([1000 1500 0.14 0.32])
title('Predicted Y movements','fontsize',20)
xlabel('Time','fontsize',20)
ylabel('Y coordinate','fontsize',20)
legend({'True','Predicted'},'fontsize',14)

subplot(2,2,[3,4])
plot(xtest(1000:1025),ytest(1000:1025),'linewidth',2)
hold on
plot(Xpredicted_final(1000:1025),Ypredicted_final(1000:1025),'linewidth',2)
grid on
title('Real Trajectories samples 1000 to 1025','fontsize',20)
xlabel('X coordinate','fontsize',20)
ylabel('Y coordinate','fontsize',20)
legend({'True','Predicted'},'fontsize',14)
