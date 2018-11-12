function [] = GuidesheetII(trainData,trainLabels,testData, classA, classB, labelsA, labelsB)
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

%% Kaggle test
[classifierKaggle, ~, ~,~]=classification(trainData(:,1:20:end),trainLabels,'pseudoquadratic','uniform');
yhat_kaggle = predict(classifierKaggle,testData(:,1:20:end));
labelToCSV(yhat_kaggle,'labels_2.csv','csvlabels');

%% Partition

N = size(trainLabels, 1);
cpN = cvpartition(N,'kfold',10);
cpLabels = cvpartition(trainLabels,'kfold',10);
class_B_elem_N = [];
class_B_elem_group = [];

for i=1:10
    class_B_elem_N = [class_B_elem_N,sum(trainLabels(cpN.test(i))==1)];
    class_B_elem_group = [class_B_elem_group,sum(trainLabels(cpLabels.test(i))==1)];
end
 

%% Compute the error for the 4 classifier for 10-fold partition

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

end

