clear all
close all
clc

%AAAAAA

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
    Normality = 'One of the populations is not normal';
end

classA_featureSimilar_normalized = (classA(:,featureSimilar)- mean(classA(:,featureSimilar)))/std(classA(:,featureSimilar));
classB_featureSimilar_normalized = (classB(:,featureSimilar)- mean(classB(:,featureSimilar)))/std(classB(:,featureSimilar));

[hA,pA] = kstest(classA_featureSimilar_normalized);
[hB,pB] = kstest(classB_featureSimilar_normalized);

if hA==0 && hB==0
    [hSimilar, pSimilar] = ttest2(classA(:,featureSimilar),classB(:,featureSimilar));
else
    Normality = 'One of the populations is not normal';
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

%% Cleaning %%%%%%%%%%%%%%% line added for not having mismatches
clearvars -except trainData trainLabels testData classA classB labelsA labelsB
close all

%% %%%%%%%%%%% GUIDESHEET II %%%%%%%%%%%%%%%%%%%
%% LDA/QDA classifiers

% Linear, diaglinear, quadratic and diagquadratic classifiers
testFeatures = trainData(:,1:10:end);

[linclassifier, ~, classificErrLinear,~] = classification(testFeatures, trainLabels, 'linear', 'empirical');
[~, ~, classificErrDiaglinear,~] = classification(testFeatures, trainLabels, 'diaglinear', 'empirical');
[~, ~, classificErrDiaquadratic,~] = classification(testFeatures, trainLabels, 'diagquadratic', 'empirical');
[quadclassifier, yhatQuadratic, classificErrQuadratic,classErrQuadratic] = classification(testFeatures, trainLabels, 'pseudoquadratic', 'empirical');

names = categorical({'Linear';'Diaglinear'; 'PseudoQuadratic'; 'Diagquadratic'});
errors = [classificErrLinear; classificErrDiaglinear; classificErrQuadratic; classificErrDiaquadratic];

figure('name','Classification error')
bar(names, errors)
ylabel('Classification Error')
title('Classification error depending on the classifier')

% Addition of the prior probability

[priorClassifier,yhat_prior,classificErrQuadratic_prior,classErrQuadratic_prior] = classification(testFeatures,trainLabels,'pseudoquadratic','uniform');

names = categorical({'ClassifError Quadratic';'Class Error Quadratic'; 'ClassifError Quadratic with prior'; 'Class Error Quadratic with prior'});
Errors = [classificErrQuadratic, classErrQuadratic, classificErrQuadratic_prior, classErrQuadratic_prior];

figure('name','Class error')
bar(names, Errors)
ylabel('Error')
title('Class and classification error')

%% Training and testing error

% Split data
mA=size(classA, 1);
mB=size(classB, 1);

set1 = zeros(mA/2+(mB-1)/2, size(trainData,2));
set2 = zeros(mA/2+(mB-1)/2, size(trainData,2));
set1_labels = zeros(mA/2+(mB-1)/2, 1);
set2_labels = zeros(mA/2+(mB-1)/2, 1);

for i = 1:(mA/2)
    set1(i,:) = classA(i,:);
    set1_labels(i, 1) = labelsA(i, 1);
    
    set2(i, :) = classA(((mA/2)+i),:);
    set2_labels(i, 1) = labelsA(((mA/2)+i),:);
end
  
for i = 1:((mB-1)/2)
    set1((mA/2)+i, :) = classB(i, :);
    set1_labels((mA/2)+i, 1) = labelsB(i, 1);
    
    set2((mA/2)+i, :) = classB(((mB-1)/2+i),:);
    set2_labels((mA/2)+i, 1) = labelsB(((mB-1)/2+i),:);
end

%n = round(1/3*size(classB, 1));
%m = size(classB, 1)-n;

%tot = size(classB,1)+size(classA,1);
%tot = round(tot/2)-1; %%

%set1 = zeros(tot,2048);
%set2 = zeros(tot,2048);

%set1 = classB(1:n,:);
%set1_labels = labelsB(1:n);
%set1(n+1:tot,:) = classA(1:(tot-n),:);
%set1_labels(n+1:tot,:) = labelsA(1:(tot-n));

%set2 = classB(n+1:end,:);
%set2_labels = labelsB(n+1:end);
%set2(m:tot,:) = classA(tot-n:(2*tot-n-m),:);
%set2_labels(m:tot,:) = labelsA(tot-n:(2*tot-n-m));

set1 = set1(:,1:20:end);
set2 = set2(:,1:20:end);

% evaluating errors
[ErrorsArray_train1_test2_empirical, ErrorsArray_train1_test2_uniform] = arrayErrorsClassification(set1, set2, set1_labels, set2_labels);
[ErrorsArray_train2_test1_empirical, ErrorsArray_train2_test1_uniform] = arrayErrorsClassification(set2, set1, set2_labels, set1_labels);

name = categorical({'ClassifError diaglinear', 'ClassifError linear', 'ClassifError diagquadratic', 'ClassifError quadratic'});

figure('name', 'Training error and Testing error for 4 classifier')
subplot(2,2,1)
bar(name, ErrorsArray_train1_test2_empirical)
grid on
legend('train', 'test');
title('empirical prior probability, train=set1 and test=set2')
subplot(2,2,2)
bar(name, ErrorsArray_train1_test2_uniform)
grid on
legend('train', 'test');
title('uniform prior probability, train=set1 and test=set2')
subplot(2,2,3)
bar(name, ErrorsArray_train2_test1_empirical)
grid on
legend('train', 'test');
title('empirical prior probability, train=set2 and test=set1')
subplot(2,2,4)
bar(name, ErrorsArray_train2_test1_uniform)
grid on
legend('train', 'test');
title('uniform prior probability, train=set2 and test=set1')

% kaggle test
[classifierKaggle, ~, ~,~]=classification(set2, set2_labels,'pseudoquadratic','uniform');
yhat_kaggle = predict(classifierKaggle,testData(:,1:20:end));
labelToCSV(yhat_kaggle,'labels_2.csv','csvlabels');

%% Partition
N = size(trainLabels, 1);
cpN = cvpartition(N,'kfold',10);
cpLabels = cvpartition(trainLabels,'kfold',10);

%%% check how many test sample in each partition
for i=1:10
    N=sum(trainLabels(cpN.test(i))==1)
    L=sum(trainLabels(cpLabels.test(i))==1)
 end
%we have 14 sample from class B in test set of 59 to 60 samples 

%% compute the error for the 4 classifier for 10-fold partition

classification_error_matrix = zeros(8,10);

for k=1:10
    [ training_set, test_set, training_labels, test_labels ] = ...
        find_cvpartition(k, cpLabels, trainLabels, testFeatures);
    [ErrorsArray_cptrain1_cptest2_empirical, ErrorsArray_cptrain1_cptest2_uniform] = ...
        arrayErrorsClassification(training_set, test_set, training_labels, test_labels);
    for i=1:4
        classification_error_matrix(i,k)=ErrorsArray_cptrain1_cptest2_empirical(i,2);
    end
    for j=1:4
        classification_error_matrix(j+4,k)=ErrorsArray_cptrain1_cptest2_uniform(j,2);
    end
end

mean_classification_error_matrix = mean(classification_error_matrix, 2);
std_classification_error_matrix = std(classification_error_matrix, 0, 2);

% changing the partition using repartition(cp) each time.
classification_error_matrix_rep = zeros(8,10);

cpLabels_repartition = repartition(cpLabels);

for k=1:10
    [ training_set_rep, test_set_rep, training_labels_rep, test_labels_rep ] = ...
        find_cvpartition(k, cpLabels_repartition, trainLabels, testFeatures);
    [ErrorsArray_cp_rep_train1_cp_rep_test2_empirical, ErrorsArray_cp_rep_train1_cp_rep_test2_uniform] = ...
        arrayErrorsClassification(training_set_rep, test_set_rep, training_labels_rep, test_labels_rep);
    for i=1:4
        classification_error_matrix_rep(i,k)=ErrorsArray_cp_rep_train1_cp_rep_test2_empirical(i,2);
    end
    for j=1:4
        classification_error_matrix_rep(j+4,k)=ErrorsArray_cp_rep_train1_cp_rep_test2_uniform(j,2);
    end
end

mean_classification_error_matrix_rep = mean(classification_error_matrix_rep, 2);
std_classification_error_matrix_rep = std(classification_error_matrix_rep, 0, 2);

name = categorical({'1.Diaglinear empirical', '2.Linear empirical',...
    '3.Diagquadratic empirical', '4.Quadratic empirical', '5.Diaglinear uniform',...
    '6.Linear uniform', '7.Diagquadratic uniform', '8.Quadratic uniform'});

figure('name', 'Training error and Testing error for 4 classifiers and partitioning')
subplot(1,2,1)
bar(name, mean_classification_error_matrix)
hold on
errorbar(mean_classification_error_matrix,std_classification_error_matrix,'.')
grid on
title('Mean Classification error, 10-fold partition')
subplot(1,2,2)
bar(name, mean_classification_error_matrix_rep)
hold on
errorbar(mean_classification_error_matrix_rep,std_classification_error_matrix_rep,'.')
grid on
title('Mean Classification error, 10-fold partition and repartition')

%% %%%%%%%Guidesheet III %%%%%%%%%%
%%