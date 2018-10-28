function [] = GuidesheetIII(trainData,trainLabels,testData)

%% Cross-validation

numMaxFolds = 10;
numMaxFeatures = 20;

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

end