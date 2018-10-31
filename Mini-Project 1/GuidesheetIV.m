function [] = GuidesheetIV(trainData,trainLabels,testData)

%UNTITLED Summary of this function goes here
%   Detailed explanation goes here


[orderedInd, orderedPower] = rankfeat(trainData, trainLabels, 'fisher');

%standardized (z-score) data
%norm_data = zscore(trainData);
[coeff,score,latent,tsquared,variance] = pca(trainData(:, orderedInd(1:20)));

%% covariance matrix

cov_matrix_data = cov(trainData(:, orderedInd(1:20)));
variance_data=diag(cov_matrix_data)';

figure;
subplot(1,2,1);
imagesc(cov_matrix_data)%'InitialMagnification','fit');
colorbar
title('Before PCA');

cov_matrix_PCA = cov(score);
variance_data_PCA=diag(cov_matrix_PCA)';
subplot(1,2,2);
imagesc(cov_matrix_PCA, [0 1])%,'InitialMagnification',2000);
colorbar
title('After PCA');


%% (cumulated) explained variance
% cumul_variance = zeros(length(variance),1);
% cumul_variance(1) = variance(1);
% for i=2:length(variance)
%     cumul_variance(i) = cumul_variance(i-1) + variance(i);
% end

cumul_variance = cumsum(variance)/sum(variance);
figure
bar(cumul_variance);
title({'Cumulated explained variance in';'function of the principal components'});
xlabel('Number of principal components');
ylabel('Cumulated explained variance (%)');
%ylim([0 105]);
grid on;

% Sorting and plotting the loadings of PCA1 in the descending order
figure
bar(coeff(:,1))
title({'Loadings of the 1st principal component (unsorted)'});
xlabel('Variables');
ylabel('Loadings');
%xticks(1:28);
%xticklabels(varNames);
%xtickangle(90);

[sorted_coef1, sorting_protocol1] = sort(abs(coeff(:,1)),'descend');
%sorted_var_names1 = varNames(sorting_protocol1);
figure
bar(sorted_coef1);
% xticks(1:28);
% xticklabels(sorted_var_names1);
% xtickangle(90);
title({'Loadings of the 1st principal component (sorted)'});
xlabel('Variables');
ylabel('Loadings');



%% Cross validation for hyperparameters optimization

numMaxFolds = 20;
numMaxPCs = 20;
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
    for f=1:numMaxPCs
        [training_set_fs,test_set_fs,training_labels_fs,test_labels_fs] = ...
            find_cvpartition(k, cpLabels, trainLabels, trainData);

        norm_train=zscore(training_set_fs);
        
        [coeff,score,latent,tsquared,variance] = pca(norm_train(:, 1:50:2048));
        %[sorted_coef1, sorting_protocol1] = sort(abs(coeff(:,1)),'descend');        
        %cumul_variance = cumsum(variance)/sum(variance);
       
        centered_test=zscore(testData(:, 1:50:2048));
        norm_score_test=centered_test*coeff;
        
        [ErrorsArray,~] = ...
            arrayErrorsClassification(score(:,1:numMaxPCs), norm_score_test(:,1:numMaxPCs), training_labels_fs, test_labels_fs);
        
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
        if meanTestErrorsCV(f,1,m)< min
            Results.bestPCsNumber = f;
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

