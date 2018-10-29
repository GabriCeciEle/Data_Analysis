function [] = GuidesheetIII(trainData,trainLabels,testData)

%% Cross-validation

numMaxFolds = 10;
numMaxFeatures = 200;

orderedInd = [];
orderedPower = [];

classif_error_train_Diaglin = [];
classif_error_test_Diaglin = [];
classif_error_train_LDA = [];
classif_error_test_LDA = [];

cpLabels = cvpartition(trainLabels,'kfold', numMaxFolds);

for f=1:numMaxFeatures    
    for k=1:numMaxFolds
        [ training_set_fs, test_set_fs, training_labels_fs, test_labels_fs ] = ...
            find_cvpartition(k, cpLabels, trainLabels, trainData);
        [orderedInd, orderedPower] = rankfeat(training_set_fs, training_labels_fs, 'fisher');
        [ErrorsArray_cp_fs_train1_cp_fs_test2_empirical, ~] = ...
            arrayErrorsClassification(training_set_fs(:, orderedInd(1:f)), test_set_fs(:, orderedInd(1:f)), training_labels_fs, test_labels_fs);
        
        classif_error_train_Diaglin(f, k) = ErrorsArray_cp_fs_train1_cp_fs_test2_empirical(1,1);
        classif_error_test_Diaglin(f, k) = ErrorsArray_cp_fs_train1_cp_fs_test2_empirical(1,2);
        classif_error_train_LDA(f, k) = ErrorsArray_cp_fs_train1_cp_fs_test2_empirical(2,1);
        classif_error_test_LDA(f, k) = ErrorsArray_cp_fs_train1_cp_fs_test2_empirical(2,2);
    end
end

figure('name','Cross-validation for hyperparameter optimization 1')
subplot(2,2,1)
plot(classif_error_train_Diaglin,'b')
grid on
hold on 
plot(mean(classif_error_train_Diaglin,2),'b','Linewidth',5)
xlabel('Number of features')
ylabel('Classification error')
title('Diaglinear classifier, train')

subplot(2,2,2)
plot(classif_error_test_Diaglin,'r')
grid on
hold on 
plot(mean(classif_error_test_Diaglin,2),'r','Linewidth',5)
xlabel('Number of features')
ylabel('Classification error')
title('Diaglinear classifier, test')

subplot(2,2,3)
plot(classif_error_train_LDA,'b')
grid on
hold on 
plot(mean(classif_error_train_LDA,2),'b','Linewidth',5)
xlabel('Number of features')
ylabel('Classification error')
title('LDA classifier, train')

subplot(2,2,4)
plot(classif_error_test_LDA,'r')
grid on
hold on 
plot(mean(classif_error_test_LDA,2),'r','Linewidth',5)
xlabel('Number of features')
ylabel('Classification error')
title('LDA classifier, test')

figure('name','Cross-validation for hyperparameter optimization 2')
subplot(2,1,1)
plot(mean(classif_error_train_Diaglin,2),'b','Linewidth',2)
grid on
hold on
plot(mean(classif_error_test_Diaglin,2),'r','Linewidth',2)
xlabel('Number of features')
ylabel('Classification error')
legend('Train','Test')
title('Diaglinear classifier')

subplot(2,1,2)
plot(mean(classif_error_train_LDA,2),'b','Linewidth',2)
grid on
hold on
plot(mean(classif_error_test_LDA,2),'r','Linewidth',2)
xlabel('Number of features')
ylabel('Classification error')
legend('Train','Test')
title('LDA classifier')




%% Random classifier

numMaxFolds = 4;
classif_error_train_random = [];
classif_error_test_random = [];
mean_classif_error_test_random = [];
cpLabels = cvpartition(trainLabels,'kfold', numMaxFolds);

for i = 1:1000    
    for k=1:numMaxFolds
        [ training_set_fs, test_set_fs, training_labels_fs, test_labels_fs ] = ...
            find_cvpartition(k, cpLabels, trainLabels, trainData);
        N = length(training_labels_fs);
        M = length(test_labels_fs);
        predicted_labels_train_random = randi([0,1],N,1);
        predicted_labels_test_random = randi([0,10],M,1);
        classif_error_train_random(k) = classificationError(training_labels_fs,predicted_labels_train_random);
        classif_error_test_random(k) = classificationError(test_labels_fs,predicted_labels_test_random);
    end
    mean_classif_error_test_random(i) = mean(classif_error_test_random);
end

figure('name','Test error across fold for random classifier')
plot(mean_classif_error_test_random,'b');
grid on
xlabel('Number of repetition');
ylabel('Classification Error');
title('Mean Test error random classifier');

%% Nested cross-validation

Inner = 5;
Outer = 10;
outerPartition = cvpartition(trainLabels,'kfold', Outer);
    
for k=1:Outer
    [outer_training, outer_test, outer_training_labels, outer_test_labels] = ...
        find_cvpartition(k,outerPartition,trainLabels,trainData);
    
    innerPartition = cvpartition(outer_training_labels,'kfold',Inner);
    
    for w=1:Inner
        [inner_training, inner_test, inner_training_labels, inner_test_labels] = ...
        find_cvpartition(w,innerPartition,outer_training_labels,outer_training);
        [orderedInd, orderedPower] = rankfeat(inner_training, inner_training_labels, 'fisher');
        
        for f=1:numMaxFeatures
            [InnerErrors_empirical, ~] = ...
            arrayErrorsClassification(inner_training(:, orderedInd(1:f)), inner_test(:, orderedInd(1:f)), inner_training_labels, inner_test_labels);
            classif_error_inner_train_diaglin(f,w) = InnerErrors_empirical(1,1);
            classif_error_inner_validation_diaglin(f,w) = InnerErrors_empirical(1,2);
            classif_error_inner_train_LDA(f,w) = InnerErrors_empirical(2,1);
            classif_error_inner_validation_LDA(f,w) = InnerErrors_empirical(2,2);
            classif_error_inner_train_diagquad(f,w) = InnerErrors_empirical(3,1);
            classif_error_inner_validation_diagquad(f,w) = InnerErrors_empirical(3,2);
            classif_error_inner_train_quad(f,w) = InnerErrors_empirical(4,1);
            classif_error_inner_validation_quad(f,w) = InnerErrors_empirical(4,2);
        end 
    end
    
    
    mean_Validation_error_diaglin = mean(classif_error_inner_validation_diaglin, 2);
    mean_Train_error_diaglin = mean(classif_error_inner_train_diaglin,2);
    mean_Validation_error_LDA = mean(classif_error_inner_validation_LDA, 2);
    mean_Train_error_LDA = mean(classif_error_inner_train_LDA,2);
    mean_Validation_error_diagquad = mean(classif_error_inner_validation_diagquad, 2);
    mean_Train_error_diagquad = mean(classif_error_inner_train_diagquad,2);
    mean_Validation_error_quad = mean(classif_error_inner_validation_quad, 2);
    mean_Train_error_quad = mean(classif_error_inner_train_quad,2);
    
    mean_Train_error = [mean_Train_error_diaglin,mean_Train_error_LDA,mean_Train_error_diagquad,mean_Train_error_quad];
    
        
    % diaglin, LDA, diagquad, quad
    [optimalValidationError(k,1),bestFeatureNumber(k,1)] = min(mean_Validation_error_diaglin);
    [optimalValidationError(k,2),bestFeatureNumber(k,2)] = min(mean_Validation_error_LDA);
    [optimalValidationError(k,3),bestFeatureNumber(k,3)] = min(mean_Validation_error_diagquad);
    [optimalValidationError(k,4),bestFeatureNumber(k,4)] = min(mean_Validation_error_quad);
    
    [final_optimalValidationError(k,1),model(k,1)] = min(optimalValidationError(k,:));
    bfn(k,1) = bestFeatureNumber(k,model(k,1));
    
    optimalTrainingError(k,1) = mean_Train_error(bfn(k,1),model(k,1));
    
    [outer_orderedInd, outer_orderedPower] = rankfeat(outer_training, outer_training_labels, 'fisher');
    [OuterErrors_empirical, ~] = ...
    arrayErrorsClassification(outer_training(:,outer_orderedInd(1:bfn(k,1))), outer_test(:,outer_orderedInd(1:bfn(k,1))),outer_training_labels, outer_test_labels);
    
    classif_error_outer_test(k,1) = OuterErrors_empirical(model(k,1),2); 
   
end

figure('name','Nested cross validation errors')
subplot(1,3,1)
boxplot(classif_error_outer_test)
title('Outer Errors')
subplot(1,3,2)
boxplot(final_optimalValidationError)
title('Validation Inner Errors')
subplot(1,3,3)
boxplot(optimalTrainingError)
title('Training Inner Errors')

figure('name','error obtained with simple cross validation')
subplot(1,2,1)
boxplot(mean(classif_error_test_Diaglin,2))
title('Diaglinear')
subplot(1,2,2)
boxplot(mean(classif_error_test_LDA,2))
title('LDA')


%% Cross validation for hyperparameters optimization

numMaxFolds = 10;
numMaxFeatures = 400;
cpLabels = cvpartition(trainLabels,'kfold', numMaxFolds);
classif_error_train_Diaglin = [];
classif_error_test_Diaglin = [];
classif_error_train_LDA = [];
classif_error_test_LDA = [];
classif_error_train_Diagquad = [];
classif_error_test_Diagquad = [];
classif_error_train_QDA = [];
classif_error_test_QDA = [];
for k=1:numMaxFolds    
    for f=1:numMaxFeatures
        [training_set_fs,test_set_fs,training_labels_fs,test_labels_fs] = ...
            find_cvpartition(k, cpLabels, trainLabels, trainData);
        [orderedInd, orderedPower] = rankfeat(training_set_fs, training_labels_fs, 'fisher');
        [ErrorsArray,~] = ...
            arrayErrorsClassification(training_set_fs(:, orderedInd(1:f)), test_set_fs(:, orderedInd(1:f)), training_labels_fs, test_labels_fs);
        
        classif_error_train_Diaglin(f, k) = ErrorsArray(1,1);
        classif_error_test_Diaglin(f, k) = ErrorsArray(1,2);
        classif_error_train_LDA(f, k) = ErrorsArray(2,1);
        classif_error_test_LDA(f, k) = ErrorsArray(2,2);
        classif_error_train_Diagquad(f, k) = ErrorsArray(3,1);
        classif_error_test_Diagquad(f, k) = ErrorsArray(3,2);
        classif_error_train_QDA(f, k) = ErrorsArray(4,1);
        classif_error_test_QDA(f, k) = ErrorsArray(4,2);
    end
end

TestErrorsCV = classif_error_test_Diaglin;
TestErrorsCV(:,:,2) = classif_error_test_LDA;
TestErrorsCV(:,:,3) = classif_error_test_Diagquad;
TestErrorsCV(:,:,4) = classif_error_test_QDA;

meanTestErrorsCV = mean(TestErrorsCV,2);

min = 1;

for f=1:size(meanTestErrorsCV,1)
    for m=1:size(meanTestErrorsCV,3)
        if meanTestErrorsCV(f,1,m)<= min
            Results.bestFeatureNumber = f;
            Results.classifierType = m;
            min=meanTestErrorsCV(f,1,m);
        end
    end
end

%% Model building

[orderedInd,orderedPower] = rankfeat(trainData, trainLabels, 'fisher');
[classifierKaggle, ~, ~,~] = classification(trainData(:,orderedInd(1:Results.bestFeatureNumber)),trainLabels,'pseudoquadratic','empirical');
yhat_kaggle = predict(classifierKaggle,testData(:,orderedInd(1:Results.bestFeatureNumber)));
labelToCSV(yhat_kaggle,'labels_3.csv','csvlabels');


end