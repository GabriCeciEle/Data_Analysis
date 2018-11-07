function [] = Final(trainData,trainLabels,testData)
%% NCV

Inner = 5; %5
Outer = 10; %10
numMaxPCs = 20;

trainData = trainData(:,1:20:end);

outerPartition = cvpartition(trainLabels,'kfold', Outer);
    
for k=1:Outer
    [outer_training, outer_test, outer_training_labels, outer_test_labels] = ...
        find_cvpartition(k,outerPartition,trainLabels,trainData);
    
    innerPartition = cvpartition(outer_training_labels,'kfold',Inner);
    
    for w=1:Inner
        [inner_training, inner_test, inner_training_labels, inner_test_labels] = ...
            find_cvpartition(w,innerPartition,outer_training_labels,outer_training);
   
        for p=1:numMaxPCs

            norm_train = zscore(inner_training);
        
            [coeff,score,~,~,~] = pca(norm_train);
       
            norm_test = (inner_test - mean(inner_training,1))./std(inner_training,0,1);
            norm_score_test = norm_test*coeff;
            
            [orderedInd, ~] = rankfeat(score, inner_training_labels, 'fisher');
            
            ErrorsArray = ...
                arrayErrorsClass(score(:,orderedInd(1:p)), norm_score_test(:,orderedInd(1:p)), inner_training_labels, inner_test_labels);
        
            class_error_train_Diaglin(p,w) = ErrorsArray(1,1);
            class_error_test_Diaglin(p,w) = ErrorsArray(1,2);
            class_error_train_LDA(p,w) = ErrorsArray(2,1);
            class_error_test_LDA(p,w) = ErrorsArray(2,2);
            class_error_train_Diagquad(p,w) = ErrorsArray(3,1);
            class_error_test_Diagquad(p,w) = ErrorsArray(3,2);
            class_error_train_QDA(p,w) = ErrorsArray(4,1);
            class_error_test_QDA(p,w) = ErrorsArray(4,2);  
            
        end
       
    end

    mean_Validation_error_diaglin = mean(class_error_test_Diaglin, 2);
    mean_Train_error_diaglin = mean(class_error_train_Diaglin,2);
    
    mean_Validation_error_LDA = mean(class_error_test_LDA, 2);
    mean_Train_error_LDA = mean(class_error_train_LDA,2);
    
    mean_Validation_error_diagquad = mean(class_error_test_Diagquad, 2);
    mean_Train_error_diagquad = mean(class_error_train_Diagquad,2);
    
    mean_Validation_error_quad = mean(class_error_test_QDA, 2);
    mean_Train_error_quad = mean(class_error_train_QDA,2);
     
    mean_Train_error = [mean_Train_error_diaglin,mean_Train_error_LDA,mean_Train_error_diagquad,mean_Train_error_quad];
    
    % diaglin, LDA, diagquad, quad
    [optimalValidationError(k,1),bestPcNumber(k,1)] = min(mean_Validation_error_diaglin);
    [optimalValidationError(k,2),bestPcNumber(k,2)] = min(mean_Validation_error_LDA);
    [optimalValidationError(k,3),bestPcNumber(k,3)] = min(mean_Validation_error_diagquad);
    [optimalValidationError(k,4),bestPcNumber(k,4)] = min(mean_Validation_error_quad);
    
    [final_optimalValidationError(k,1),model(k,1)] = min(optimalValidationError(k,:));
    bPcNumb(k,1) = bestPcNumber(k,model(k,1));
    
    optimalTrainingError(k,1) = mean_Train_error(bPcNumb(k,1),model(k,1));
    
   
    norm_train = zscore(outer_training);
        
    [coeff,score,~,~,~] = pca(norm_train);
       
    norm_test = (outer_test - mean(outer_training,1))./std(outer_training,0,1);
    norm_score_test = norm_test*coeff;
            
    [orderedInd,~] = rankfeat(score, outer_training_labels, 'fisher');
            
    OuterErrors = ...
         arrayErrorsClass(score(orderedInd(:,1:bPcNumb(k,1))), norm_score_test(orderedInd(:,1:bPcNumb(k,1))), outer_training_labels, outer_test_labels);
        
    class_error_outer_test(k,1) = OuterErrors(model(k,1),2); 
   
end
        
mean_class_error_outer_test = mean(class_error_outer_test);
std_class_error_outer_test = std(class_error_outer_test);

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


%% CV for hyperparameters selection



end