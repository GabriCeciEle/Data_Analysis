function [] = Final(trainData,trainLabels,testData)
%% NCV

Inner = 5; 
Outer = 10; 
numMaxPCs = 100;
step = 10; %first try was 20

trainData_NCV = trainData(:,1:step:end);

outerPartition = cvpartition(trainLabels,'kfold', Outer);
    
for k=1:Outer
    [outer_training, outer_test, outer_training_labels, outer_test_labels] = ...
        find_cvpartition(k,outerPartition,trainLabels,trainData_NCV);
    
    innerPartition = cvpartition(outer_training_labels,'kfold',Inner);
    
    for w=1:Inner
        [inner_training, inner_test, inner_training_labels, inner_test_labels] = ...
            find_cvpartition(w,innerPartition,outer_training_labels,outer_training);
         
        norm_train = zscore(inner_training);
        [coeff,score,~,~,~] = pca(norm_train);
        norm_test = (inner_test - mean(inner_training,1))./std(inner_training,0,1);
        norm_score_test = norm_test*coeff;

        [orderedInd, ~] = rankfeat(score, inner_training_labels, 'fisher');

        for p=1:numMaxPCs
            
            ErrorsArray = ...
                arrayErrorsClass(score(:,orderedInd(1:p)), norm_score_test(:,orderedInd(1:p)), inner_training_labels, inner_test_labels);
        
            Results.NCV.class_error_train_Diaglin(p,w) = ErrorsArray(1,1);
            Results.NCV.class_error_test_Diaglin(p,w) = ErrorsArray(1,2);
            Results.NCV.class_error_train_LDA(p,w) = ErrorsArray(2,1);
            Results.NCV.class_error_test_LDA(p,w) = ErrorsArray(2,2);
            Results.NCV.class_error_train_Diagquad(p,w) = ErrorsArray(3,1);
            Results.NCV.class_error_test_Diagquad(p,w) = ErrorsArray(3,2);
            Results.NCV.class_error_train_QDA(p,w) = ErrorsArray(4,1);
            Results.NCV.class_error_test_QDA(p,w) = ErrorsArray(4,2);  
            
        end
       
    end

    Results.NCV.mean_Validation_error_diaglin = mean(Results.NCV.class_error_test_Diaglin, 2);
    Results.NCV.mean_Train_error_diaglin = mean(Results.NCV.class_error_train_Diaglin,2);
    
    Results.NCV.mean_Validation_error_LDA = mean(Results.NCV.class_error_test_LDA, 2);
    Results.NCV.mean_Train_error_LDA = mean(Results.NCV.class_error_train_LDA,2);
    
    Results.NCV.mean_Validation_error_diagquad = mean(Results.NCV.class_error_test_Diagquad, 2);
    Results.NCV.mean_Train_error_diagquad = mean(Results.NCV.class_error_train_Diagquad,2);
    
    Results.NCV.mean_Validation_error_quad = mean(Results.NCV.class_error_test_QDA, 2);
    Results.NCV.mean_Train_error_quad = mean(Results.NCV.class_error_train_QDA,2);
     
    %mean_Train_error = [mean_Train_error_diaglin,mean_Train_error_LDA,mean_Train_error_diagquad,mean_Train_error_quad];
    
    [Results.NCV.optimalValidationError(k,1),Results.NCV.bestPcNumber(k,1)] = min(Results.NCV.mean_Validation_error_diaglin);
    [Results.NCV.optimalValidationError(k,2),Results.NCV.bestPcNumber(k,2)] = min(Results.NCV.mean_Validation_error_LDA);
    [Results.NCV.optimalValidationError(k,3),Results.NCV.bestPcNumber(k,3)] = min(Results.NCV.mean_Validation_error_diagquad);
    [Results.NCV.optimalValidationError(k,4),Results.NCV.bestPcNumber(k,4)] = min(Results.NCV.mean_Validation_error_quad);
    
    % find the best number of PCs and the best classifier type
    [Results.NCV.final_optimalValidationError(k,1),Results.NCV.model(k,1)] = min(Results.NCV.optimalValidationError(k,:));
    Results.NCV.bPcNumb(k,1) = Results.NCV.bestPcNumber(k,Results.NCV.model(k,1));
    
    %optimalTrainingError(k,1) = mean_Train_error(bPcNumb(k,1),model(k,1));
    
    % model building for the outer fold
    norm_train = zscore(outer_training);       
    [coeff,score,~,~,~] = pca(norm_train);       
    norm_test = (outer_test - mean(outer_training,1))./std(outer_training,0,1);
    norm_score_test = norm_test*coeff;
            
    [orderedInd,~] = rankfeat(score, outer_training_labels, 'fisher');
            
    OuterErrors = ...
         arrayErrorsClass(score(:,orderedInd(1:Results.NCV.bPcNumb(k,1))), norm_score_test(:,orderedInd(1:Results.NCV.bPcNumb(k,1))), outer_training_labels, outer_test_labels);
        
    Results.NCV.class_error_outer_test(k,1) = OuterErrors(Results.NCV.model(k,1),2); 
   
end
        
Results.NCV.mean_class_error_outer_test = mean(Results.NCV.class_error_outer_test);
Results.NCV.std_class_error_outer_test = std(Results.NCV.class_error_outer_test);

figure('name', 'Performances')
subplot(2,1,1)
plot(Results.NCV.class_error_outer_test)
grid on
xlabel('Outer Fold')
ylabel('Class Error')
subplot(2,1,2)
bar(Results.NCV.mean_class_error_outer_test)
hold on
errorbar(Results.NCV.mean_class_error_outer_test,Results.NCV.std_class_error_outer_test,'.')
grid on
title('Class error, 10-fold partition')

%% Statistical significance

[h,p] = ttest(Results.NCV.class_error_outer_test,0.5);

%% CV for hyperparameters selection

numMaxFolds = 10; 
numMaxPCs = 100;

trainData_CV = trainData(:,1:step:end);

cpLabels = cvpartition(trainLabels,'kfold', numMaxFolds);

for k=1:numMaxFolds    
    [training_set,test_set,training_labels,test_labels] = ...
        find_cvpartition(k, cpLabels, trainLabels, trainData_CV);
    
    norm_train = zscore(training_set);  
    [coeff,score,~,~,~] = pca(norm_train);
    norm_test = (test_set - mean(training_set,1))./std(training_set,0,1);
    norm_score_test = norm_test*coeff;
    
    [orderedInd, ~] = rankfeat(score, training_labels, 'fisher');
    
    for p=1:numMaxPCs
        
        ErrorsArray = ...
        arrayErrorsClass(score(:,orderedInd(1:p)), norm_score_test(:,orderedInd(1:p)), training_labels, test_labels);
        
        class_error_train_Diaglin(p,k) = ErrorsArray(1,1);
        class_error_test_Diaglin(p,k) = ErrorsArray(1,2);
        class_error_train_LDA(p,k) = ErrorsArray(2,1);
        class_error_test_LDA(p,k) = ErrorsArray(2,2);
        class_error_train_Diagquad(p,k) = ErrorsArray(3,1);
        class_error_test_Diagquad(p,k) = ErrorsArray(3,2);
        class_error_train_QDA(p,k) = ErrorsArray(4,1);
        class_error_test_QDA(p,k) = ErrorsArray(4,2);
        
    end
end

mean_Validation_error_Diaglin = mean(class_error_test_Diaglin, 2);
mean_Validation_error_LDA = mean(class_error_test_LDA, 2);
mean_Validation_error_Diagquad = mean(class_error_test_Diagquad, 2);
mean_Validation_error_QDA = mean(class_error_test_QDA, 2);
mean_train_error_Diaglin = mean(class_error_train_Diaglin, 2);
mean_train_error_LDA = mean(class_error_train_LDA, 2);
mean_train_error_Diagquad = mean(class_error_train_Diagquad, 2);
mean_train_error_QDA = mean(class_error_train_QDA, 2);

[optimalValidationError_CV(1,1),bestPcNumber_CV(1,1)] = min(mean_Validation_error_Diaglin);
[optimalValidationError_CV(1,2),bestPcNumber_CV(1,2)] = min(mean_Validation_error_LDA);
[optimalValidationError_CV(1,3),bestPcNumber_CV(1,3)] = min(mean_Validation_error_Diagquad);
[optimalValidationError_CV(1,4),bestPcNumber_CV(1,4)] = min(mean_Validation_error_QDA);

% find the best number of PCs and the best classifier type
[Results.CV.final_optimalValidationError,Results.CV.model] = min(optimalValidationError_CV);
Results.CV.bPcNumb = bestPcNumber_CV(1,Results.CV.model);

%% Model building 

if Results.CV.model == 1
    classifiertype = 'diaglinear';
elseif Results.CV.model == 2
    classifiertype = 'linear';
elseif Results.CV.model == 3
    classifiertype = 'diagquadratic';
elseif Result.CV.model == 4
    classifiertype = 'pseudoquadratic';
end

trainData = trainData(:,1:step:end);   
final_norm_train = zscore(trainData);  
[coeff,score,~,~,~] = pca(final_norm_train);
final_norm_test = (testData(:,1:step:end) - mean(trainData,1))./std(trainData,0,1);
final_norm_score_test = final_norm_test*coeff;

[orderedInd, ~] = rankfeat(score, trainLabels, 'fisher');

%[classifierKaggle, ~, ~,~] = classification(score(:,1:10),trainLabels,'linear','uniform');
%[classifierKaggle, ~, ~,~] = classification(score(:,orderedInd(1:10)),trainLabels,'linear','uniform');
[classifierKaggle, ~, ~,~] = classification(score(:,orderedInd(1:Results.CV.bPcNumb)),trainLabels,classifiertype,'uniform');
yhat_kaggle = predict(classifierKaggle,final_norm_score_test(:,orderedInd(1:Results.CV.bPcNumb)));
labelToCSV(yhat_kaggle,'labels_final.csv','csvlabels');

end