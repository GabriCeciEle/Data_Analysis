function [] = Final(trainData,trainLabels,testData)
%% Parameters setting
Inner = 5; %NCV
Outer = 10; %NCV
numMaxFolds = 10; %CV
numMaxPCs = 20; 
stp=10;

trainData_NCV = trainData(:,700:stp:2000);
trainData_CV = trainData(:,700:stp:2000);

load('s.mat')

%% NCV

rng(s);
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
    
    [Results.NCV.optimalValidationError(k,1),Results.NCV.bestPcNumber(k,1)] = min(Results.NCV.mean_Validation_error_diaglin);
    [Results.NCV.optimalValidationError(k,2),Results.NCV.bestPcNumber(k,2)] = min(Results.NCV.mean_Validation_error_LDA);
    [Results.NCV.optimalValidationError(k,3),Results.NCV.bestPcNumber(k,3)] = min(Results.NCV.mean_Validation_error_diagquad);
    [Results.NCV.optimalValidationError(k,4),Results.NCV.bestPcNumber(k,4)] = min(Results.NCV.mean_Validation_error_quad);
    
    % find the best number of PCs and the best classifier type
    [Results.NCV.final_optimalValidationError(k,1),Results.NCV.model(k,1)] = min(Results.NCV.optimalValidationError(k,:));
    Results.NCV.bPcNumb(k,1) = Results.NCV.bestPcNumber(k,Results.NCV.model(k,1));
    
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

figure('name', 'Performances on Outer Folds')
plot(Results.NCV.class_error_outer_test)
grid on
xlabel('Outer Fold')
ylabel('Class Error')

figure('name', 'Performances')
bar(Results.NCV.mean_class_error_outer_test)
hold on
errorbar(Results.NCV.mean_class_error_outer_test,Results.NCV.std_class_error_outer_test,'.','linewidth',2)
grid on
title('Class error on the Validation Set')
ax=gca;
ax.TitleFontSizeMultiplier=2;
ylabel('Class error','fontsize',18)
xlabel('')

%% Statistical significance

[h,p] = ttest(Results.NCV.class_error_outer_test,0.5);

%% CV for hyperparameters selection

rng(s);
cpLabels = cvpartition(trainLabels,'kfold', numMaxFolds);

cumul_variance_onlyPCA = [];
cumul_variance_PCAandFisher = [];

for k=1:numMaxFolds    
    [training_set,test_set,training_labels,test_labels] = ...
        find_cvpartition(k, cpLabels, trainLabels, trainData_CV);
    
    norm_train = zscore(training_set);  
    [coeff,score,~,~,variance] = pca(norm_train);
    norm_test = (test_set - mean(training_set,1))./std(training_set,0,1);
    norm_score_test = norm_test*coeff;
    
    cumul_variance_onlyPCA = [cumul_variance_onlyPCA,cumsum(variance)/sum(variance)];
    
    [orderedInd, ~] = rankfeat(score, training_labels, 'fisher');
    
    cumul_variance_PCAandFisher = [cumul_variance_PCAandFisher,cumsum(variance(orderedInd(1:end)))/sum(variance(orderedInd(1:end)))];
    
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
std_Validation_error_Diaglin = std(class_error_test_Diaglin,0,2);

mean_Validation_error_LDA = mean(class_error_test_LDA, 2);
std_Validation_error_LDA = std(class_error_test_LDA,0,2);

mean_Validation_error_Diagquad = mean(class_error_test_Diagquad, 2);
std_Validation_error_Diagquad = std(class_error_test_Diagquad,0,2);

mean_Validation_error_QDA = mean(class_error_test_QDA, 2);
std_Validation_error_QDA = std(class_error_test_QDA,0,2);

mean_train_error_Diaglin = mean(class_error_train_Diaglin, 2);
std_train_error_Diaglin = std(class_error_train_Diaglin,0,2);

mean_train_error_LDA = mean(class_error_train_LDA, 2);
std_train_error_LDA = std(class_error_train_LDA,0, 2);

mean_train_error_Diagquad = mean(class_error_train_Diagquad, 2);
std_train_error_Diagquad = std(class_error_train_Diagquad,0, 2);

mean_train_error_QDA = mean(class_error_train_QDA, 2);
std_train_error_QDA = std(class_error_train_QDA,0, 2);

[optimalValidationError_CV(1,1),bestPcNumber_CV(1,1)] = min(mean_Validation_error_Diaglin);
[optimalValidationError_CV(1,2),bestPcNumber_CV(1,2)] = min(mean_Validation_error_LDA);
[optimalValidationError_CV(1,3),bestPcNumber_CV(1,3)] = min(mean_Validation_error_Diagquad);
[optimalValidationError_CV(1,4),bestPcNumber_CV(1,4)] = min(mean_Validation_error_QDA);

% find the best number of PCs and the best classifier type
[Results.CV.final_optimalValidationError,Results.CV.model] = min(optimalValidationError_CV);
Results.CV.bPcNumb = bestPcNumber_CV(1,Results.CV.model);

%% CV for hyperparameters selection Plots

% Cumulative variance
mean_variance_onlyPCA = mean(cumul_variance_onlyPCA,2);
std_variance_onlyPCA =std(cumul_variance_onlyPCA,0,2);

mean_variance_PCAandFisher = mean(cumul_variance_PCAandFisher,2);
std_variance_PCAandFisher =std(cumul_variance_PCAandFisher,0,2);

figure('name','Cumulated Explained Variance')
bar(mean_variance_onlyPCA*100)
hold on
bar(mean_variance_PCAandFisher*100)
hold on
errorbar(mean_variance_onlyPCA*100,std_variance_onlyPCA*100,'.','linewidth',2)
hold on
errorbar(mean_variance_PCAandFisher*100,std_variance_PCAandFisher*100,'.','linewidth',2)
hold on
plot([0:140],ones(1,141)*90,'linewidth',2)
grid on
ylim([0 100])
title({'Cumulated explained variance in';'function of the principal components'})
ax=gca;
ax.TitleFontSizeMultiplier=2;
al=legend('PCA','PCA+Fisher')
al.FontSize=18;
xlabel('#PCs','fontsize',18)
ylabel('Cumulated Explained Variance (%)','fontsize',18)


% Validation and Training Error
figure('name','Validation error and Training error Diaglinear')
errorbar(mean_Validation_error_Diaglin,std_Validation_error_Diaglin, 'linewidth',2)
hold on 
errorbar(mean_train_error_Diaglin,std_train_error_Diaglin, 'linewidth',2)
grid on
al=legend('Validation','Training')
al.FontSize=18;
title('Validation and Training Error Diaglinear Classifier')
ax=gca;
ax.TitleFontSizeMultiplier=2;
xlabel('#PCs','fontsize',18)
ylabel('Class Error','fontsize',18)

figure('name','Validation error and Training error LDA')
errorbar(mean_Validation_error_LDA,std_Validation_error_LDA, 'linewidth',2)
hold on 
errorbar(mean_train_error_LDA,std_train_error_LDA, 'linewidth',2)
grid on
al=legend('Validation','Training')
al.FontSize=18;
title('Validation and Training Error LDA Classifier')
ax=gca;
ax.TitleFontSizeMultiplier=2;
xlabel('#PCs','fontsize',18)
ylabel('Class Error','fontsize',18)

figure('name','Validation error and Training error Diagquadratic')
errorbar(mean_Validation_error_Diagquad,std_Validation_error_Diagquad, 'linewidth',2)
hold on 
errorbar(mean_train_error_Diagquad,std_train_error_Diagquad, 'linewidth',2)
grid on
al=legend('Validation','Training')
al.FontSize=18;
title('Validation and Training Error Diagquad Classifier')
ax=gca;
ax.TitleFontSizeMultiplier=2;
xlabel('#PCs','fontsize',18)
ylabel('Class Error','fontsize',18)

figure('name','Validation error and Training error QDA')
errorbar(mean_Validation_error_QDA,std_Validation_error_QDA, 'linewidth',2)
hold on 
errorbar(mean_train_error_QDA,std_train_error_QDA, 'linewidth',2)
grid on
al=legend('Validation','Training')
al.FontSize=18;
title('Validation and Training Error QDA Classifier')
ax=gca;
ax.TitleFontSizeMultiplier=2;
xlabel('#PCs','fontsize',18)
ylabel('Class Error','fontsize',18)

%% Model building 

if Results.CV.model == 1
    classifiertype = 'diaglinear';
elseif Results.CV.model == 2
    classifiertype = 'linear';
elseif Results.CV.model == 3
    classifiertype = 'diagquadratic';
elseif Results.CV.model == 4
    classifiertype = 'pseudoquadratic';
end

trainData_final = trainData(:,700:stp:2000);   
final_norm_train = zscore(trainData_final);  
[coeff,score,~,~,~] = pca(final_norm_train);
final_norm_test = (testData(:,700:stp:2000) - mean(trainData_final,1))./std(trainData_final,0,1);
final_norm_score_test = final_norm_test*coeff;

[orderedInd, ~] = rankfeat(score, trainLabels, 'fisher');

[classifierKaggle, ~, ~,~] = classification(score(:,orderedInd(1:Results.CV.bPcNumb)),trainLabels,classifiertype,'uniform');
yhat_kaggle = predict(classifierKaggle,final_norm_score_test(:,orderedInd(1:Results.CV.bPcNumb)));
labelToCSV(yhat_kaggle,'labels_final.csv','csvlabels');

% Covariance matrix Plot
cov_matrix_beforePCA = cov(zscore(trainData));
cov_matrix_PCA = cov(score);

figure
subplot(1,2,1);
imagesc(cov_matrix_beforePCA,[0,1])
xlabel('Feature','fontsize',18)
ylabel('Feature','fontsize',18)
colormap(gray)
colorbar
title('Before PCA')
ax=gca;
ax.TitleFontSizeMultiplier=2;

subplot(1,2,2);
imagesc(cov_matrix_PCA,[0,1])
xlabel('PC','fontsize',18)
ylabel('PC','fontsize',18)
colormap(gray)
colorbar
title('After PCA')
ax=gca;
ax.TitleFontSizeMultiplier=2;


end