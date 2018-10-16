clear all
close all
clc

%% Data Import 
load('trainSet.mat');
load('testSet.mat');
load('trainLabels.mat');

%% %%%%%%%%%%%% GUIDESHEET I %%%%%%%%%%%%%%
%% Division in classes
classA = [];
classB = [];
labelsA = [];
labelsB = [];

for sample_ = 1:size(trainData,1)
    if trainLabels(sample_) == 0
        classA = [classA; trainData(sample_, :)];
        labelsA = [labelsA; trainLabels(sample_)];
    else
        classB = [classB; trainData(sample_, :)];
        labelsB = [labelsB; trainLabels(sample_)];

    end
end

%% Histogram
figure('name','Features distribution')
for subplotNumber = 1:10
    subplot(2,5,subplotNumber)
    histogram(classA(:,715+subplotNumber),[0:0.1:1])
    hold on
    histogram(classB(:,715+subplotNumber),[0:0.1:1])
    xlabel('Amplitude [\muV]')
    ylabel('Number of samples')
    title(int2str(715+subplotNumber))
end

figure('name','Features distribution')
for subplotNumber = 1:10
    subplot(2,5,subplotNumber)
    boxplot(classA(:,715+subplotNumber))
    hold on
    boxplot(classB(:,715+subplotNumber))
    title(int2str(715+subplotNumber))
end

%% Boxplot
featureDifferent = 1095;
featureSimilar = 723;

figure('name','2 Features')
subplot(1,2,1)
boxplot(classA(:,featureSimilar))
hold on
boxplot(classB(:,featureSimilar))
title('Similarity')
subplot(1,2,2)
boxplot(classA(:,featureDifferent))
hold on
boxplot(classB(:,featureDifferent))
title('Difference')

%% Boxplot with notch
figure('name','2 Features')
subplot(1,2,1)
boxplot(classA(:,featureSimilar),'Notch','on')
hold on
boxplot(classB(:,featureSimilar),'Notch','on')
title('Similarity')
subplot(1,2,2)
boxplot(classA(:,featureDifferent),'Notch','on')
hold on
boxplot(classB(:,featureDifferent),'Notch','on')
title('Difference')

% when we visualize the 95%, in the similar case the two means cannot be
% distinguished

%% t-test
% Checking if the populations are normal
classA_featureDifferent_normalized = (classA(:,featureDifferent)- mean(classA(:,featureDifferent)))/std(classA(:,featureDifferent));
classB_featureDifferent_normalized = (classB(:,featureDifferent)- mean(classB(:,featureDifferent)))/std(classB(:,featureDifferent));

[hA,pA] = kstest(classA_featureDifferent_normalized);
[hB,pB] = kstest(classB_featureDifferent_normalized);

if hA==0 && hB==0
    [hDifferent, pDifferent] = ttest2(classA(:,featureDifferent),classB(:,featureDifferent));
else
    Normality = ['One of the populations is not normal'];
end

classA_featureSimilar_normalized = (classA(:,featureSimilar)- mean(classA(:,featureSimilar)))/std(classA(:,featureSimilar));
classB_featureSimilar_normalized = (classB(:,featureSimilar)- mean(classB(:,featureSimilar)))/std(classB(:,featureSimilar));

[hA,pA] = kstest(classA_featureSimilar_normalized);
[hB,pB] = kstest(classB_featureSimilar_normalized);

if hA==0 && hB==0
    [hSimilar, pSimilar] = ttest2(classA(:,featureSimilar),classB(:,featureSimilar));
else
    Normality = ['One of the populations is not normal'];
end

% h = 0 for classA and classB with the similar feature, meaning that the
% difference in value of this feature is not statistically relevant to
% discriminate between one class and the other (high p value, order 10^-1)

% h = 1 for classA and classB with the different feature, meaning that the
% difference in value of this feature is statistically relevant to
% discriminate between one class and the other (low p value, order 10^-9)

% using t-test for everything is dangerous: not all the populations are
% normal

%% Normality + t-test for all features
pmin = 1;
for i=1:size(trainData,2)
    classA_normalized = (classA(:,i)- mean(classA(:,i)))/std(classA(:,i));
    classB_normalized = (classB(:,i)- mean(classB(:,i)))/std(classB(:,i));

    [hA,pA] = kstest(classA_normalized);
    [hB,pB] = kstest(classB_normalized);

    if hA==0 && hB==0
        [h,p] = ttest2(classA(:,i),classB(:,i));
        if p<pmin
            pmin=p;
            featureDifferent=i;
        end
    else
        [p,h] = ranksum(classA(:,i),classB(:,i));
        if p<pmin
            pmin=p;
            featureDifferent=i;
        end
    end
    
    
end
    
%% Feature thresholding

figure()
scatter(classA(:,featureDifferent), classA(:,featureSimilar),'r')
hold on
scatter(classB(:,featureDifferent), classB(:,featureSimilar),'b')
hold on 
line(ones(1,11)*0.6, 0:0.1:1)

sampleVector = trainData(:,featureDifferent);
tf = 0.6*ones(size(trainData,1),1);
labels = sampleVector > tf;
correct = 0;
for i = 1:size(trainData,1)
    if(labels(i)==trainLabels(i))
        correct = correct + 1;
    end
end
classificationAccuracy = correct/size(trainData,1);
classificationError = 1 - classificationAccuracy;

correctA = 0;
for i = 1:size(classA,1)
    if(labels(i)==trainLabels(i))
        correctA = correctA + 1;
    end
end

correctB = 0;
for i = size(classA,1)+1:size(trainData,1)
    if(labels(i)==trainLabels(i))
        correctB = correctB + 1;
    end
end

classError = 0.5*((size(classA,1)-correctA)/size(classA,1)) + 0.5*((size(classB,1)-correctB)/size(classB,1));

%% Evolution of the errors depending on the threshold

featureDifferent = 711;

classificationAccuracy = [];
classificationError = [];
classError = [];
threshold = 0:0.0001:1;

sampleVector = trainData(:,featureDifferent);

for tf_= threshold
    tf = tf_*ones(size(trainData,1),1);
    labels = sampleVector > tf_;
    correct = 0;
    for i = 1:size(trainData,1)
        if(labels(i)==trainLabels(i))
            correct = correct + 1;
        end
    end
    classificationAccuracy = [classificationAccuracy; correct/size(trainData,1)];
    classificationError = [classificationError; 1 - correct/size(trainData,1)];

    correctA = 0;
    correctB = 0;
    for i = 1:size(classA,1)
        if(labels(i)==trainLabels(i))
            correctA = correctA + 1;
        end
    end
    for i = size(classA,1)+1:size(trainData,1)
        if(labels(i)==trainLabels(i))
            correctB = correctB + 1;
        end
    end
    classError = [classError; 0.5*((size(classA,1)-correctA)/size(classA,1)) + 0.5*((size(classB,1)-correctB)/size(classB,1))];
end

figure()
plot(threshold,classificationAccuracy)
title('classificationAccuracy')

figure()
plot(threshold,classificationError)
title('classificationError')

figure()
plot(threshold,classError)
title('classError')

figure()
plot(threshold,classError.*classificationError)
title('Product classError classificationError')

min(classError)
% with 0.5, 0.5 -> threshold 0.567
% with 1/3, 2/3 -> threshold 0.4162

%% Test

sampletestVector = testData(:,featureDifferent);
thresholdTest = 0.499*ones(size(testData,1),1);
labelsTest = sampletestVector > thresholdTest;
labelToCSV(labelsTest,'labels.csv','csvlabels');

%% %%%%%%%%%%% GUIDESHEET II %%%%%%%%%%%%%%%%%%%
%% 
features = trainData(:,1:10:end);

[classifierLinear, yhatLinear, errLinear] = classification(features, trainLabels, 'linear');
[classifierDiaglinear, yhatDiaglinear, errDiaglinear] = classification(features, trainLabels, 'diaglinear');
[classifierDiagquadratic, yhatDiagquadratic, errDiaquadratic] = classification(features, trainLabels, 'diagquadratic');
[classifierQuadratic, yhatQuadratic, errQuadratic] = classification(features, trainLabels, 'pseudoquadratic');

c = categorical({'Linear';'Diaglinear'; 'PseudoQuadratic'; 'Diagquadratic'});
errors = [errLinear; errDiaglinear; errQuadratic; errDiaquadratic];

figure()
bar(c, errors)

%%
priorClassifier = fitcdiscr(features, trainLabels, 'discrimtype', 'pseudoquadratic', 'prior','uniform');
yhat_prior = predict(priorClassifier, features);
errQuadratic_prior = classificationError(trainLabels, yhat_prior); 


%%  Class Error without prior proba
%% Class Error with prior proba
%% barpolt
c = categorical({'ClassifError Quadratic';'Class Error Quadratic'; 'ClassifError Quadratic with prior'; 'Class Error Quadratic with prior'});
Errors = [errQuadratic, CE, errQuadratic_prior, CE_prior];

%% split data
set1 = zeros(298, 2048);
set2 = zeros(298, 2048);
set1_labels = zeros(298, 1);
set2_labels = zeros(298, 1);

mA=size(classA, 1);
mB=size(classB, 1);


for i = 1:(mA/2)
    set1(i,:) = classA(i,:);
    set1_labels(i, 1) = labelsA(i, 1);
end

for j = 1:(mA/2)
    set2(j, :) = classA(((mA/2)+j),:);
    set2_labels(i, 1) = labelsA(((mA/2)+j),:);
end
    
for i = 1:(mB/2)
    set1((mA/2)+i, :) = classB(i, :);
    set1_labels((mA/2)+i, 1) = labelsB(i, 1);

end

for j = 1:(mB/2)
    set2((mA/2)+j, :) = classB(((mB/2)+j),:);
    set2_labels((mA/2)+j, 1) = labelsB(((mB/2)+j),:);

end

%% WITHOUT USING THE FUNCTION ClassifErrors
% train a diaglinear classifier

[classifierDiaglinear_train, yhatDiaglinear_train, errDiaglinear_train] = classification(trainingSet, trainingSet_labels, 'diaglinear');
yhatDiaglinear_test = predict(classifierDiaglinear_train, testSet);
errDiaglinear_test = classificationError(testSet_labels,yhatDiaglinear_test);

% train a linear classifier

[classifierLinear_train, yhatLinear_train, errLinear_train] = classification(trainingSet, trainingSet_labels, 'linear');
yhatLinear_test = predict(classifierLinear_train, testSet);
[errLinear_test] = classificationError(testSet_labels,yhatLinear_test);

% train a diagquadratic classifier

[classifierDiagquadratic_train, yhatDiagquadratic_train, errDiagquadratic_train] = classification(trainingSet, trainingSet_labels, 'diagquadratic');
yhatDiagquadratic_test = predict(classifierDiagquadratic_train, testSet);
[errDiagquadratic_test] = classificationError(testSet_labels,yhatDiagquadratic_test);

% train a quadratic classifier

[classifierQuadratic_train, yhatQuadratic_train, errQuadratic_train] = classification(trainingSet, trainingSet_labels, 'pseudoquadratic');
yhatQuadratic_test = predict(classifierQuadratic_train, testSet);
[errQuadratic_test] = classificationError(testSet_labels,yhatQuadratic_test);

% barpolt

%c = categorical({'ClassifError diaglinear train'; 'ClassifError diaglinear test'; 'ClassifError linear train'; 'ClassifError linear test'; 'ClassifError diagquadratic train'; 'ClassifError diagquadratic test'; 'ClassifError quadratic train'; 'ClassifError quadratic test'});
name = categorical({'ClassifError diaglinear', 'ClassifError linear', 'ClassifError diagquadratic', 'ClassifError quadratic'});
ClassifErrors = [errDiaglinear_train, errDiaglinear_test; 
    errLinear_train, errLinear_test; 
    errDiagquadratic_train, errDiagquadratic_test; 
    errQuadratic_train, errQuadratic_test];
figure('name', 'Training error (set1) and Testing error (set 2) for 4 classifier')
bar(name, ClassifErrors)
legend('train', 'test');

%% USING THE FUNCTION ClassifErrors

ErrorsArray_train1_test2 = ClassifErrors( set1, set2, set1_labels, set2_labels);

name = categorical({'ClassifError diaglinear', 'ClassifError linear', 'ClassifError diagquadratic', 'ClassifError quadratic'});

figure('name', 'Training error (set1) and Testing error (set 2) for 4 classifier')
bar(name, ErrorsArray_train1_test2)
legend('train', 'test');

ErrorsArray_train2_test1 = ClassifErrors( set2, set1, set2_labels, set1_labels);

figure('name', 'Training error (set2) and Testing error (set 1) for 4 classifier')
bar(name, ErrorsArray_train2_test1)
legend('train', 'test');

%% using prior

ErrorsArray_train1_test2_prior = ClassifErrors_prior(set1, set2, set1_labels, set2_labels);

name = categorical({'ClassifError diaglinear', 'ClassifError linear', 'ClassifError diagquadratic', 'ClassifError quadratic'});

figure('name', 'Training error (set1) and Testing error (set 2) for 4 classifier (prior)')
bar(name, ErrorsArray_train1_test2_prior)
legend('train', 'test');

ErrorsArray_train2_test1_prior = ClassifErrors_prior(set2, set1, set2_labels, set1_labels);

figure('name', 'Training error (set2) and Testing error (set 1) for 4 classifier (prior)')
bar(name, ErrorsArray_train2_test1_prior)
legend('train', 'test');
% use classification_prior function

%% Partition

N = size(trainLabels, 1);
cpN = cvpartition(N,'kfold',10);
cpLabels = cvpartition(trainLabels,'kfold',10);