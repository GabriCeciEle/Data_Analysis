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
            
        end 
    end
    
    
    mean_Validation_error_diaglin = mean(classif_error_inner_validation_diaglin, 2);
    mean_Train_error_diaglin = mean(classif_error_inner_train_diaglin,2);
    
    [optimalValidationError(k,1),bestFeatureNumber(k,1)] = min(meanValidation_error);
    optimalTrainingError(k,1) = meanTrain_error(bestFeatureNumber(k,1));
    
    [outer_orderedInd, outer_orderedPower] = rankfeat(outer_training, outer_training_labels, 'fisher');
    [OuterErrors_empirical, ~] = ...
    arrayErrorsClassification(outer_training(:,outer_orderedInd(1:bestFeatureNumber(k,1))), outer_test(:,outer_orderedInd(1:bestFeatureNumber(k,1))),outer_training_labels, outer_test_labels);
    %classif_error_outer_train(1,k) = OuterErrors_empirical(2,1);
    classif_error_outer_test(1,k) = OuterErrors_empirical(2,2); 
   
end
