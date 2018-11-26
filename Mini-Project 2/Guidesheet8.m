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

train = 0.05;
validation = 0.65;

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

figure('name','Optimization')
semilogx(lambda,validationErrX)
xlabel('Lambda')
ylabel('MSE')

[M,I] = min(validationErrX);

optimalLambda = lambda(I);

fullTrainData = Data(1:rowsTrain+rowsValidation,:);
xfulltrain = PosX(1:rowsTrain+rowsValidation);

[B,FitInf] = lasso(fullTrainData,xfulltrain,'Lambda',optimalLambda);

Xpredicted = testData*B + FitInf.Intercept;

performance = immse(xtest,Xpredicted);





